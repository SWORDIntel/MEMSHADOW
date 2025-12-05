#!/usr/bin/env python3
"""
Memory Tier Sync Protocol for DSMIL Brain Federation

Handles synchronization of memory across tiers (L1/L2/L3) using the
MEMSHADOW v2 32-byte binary protocol.

Features:
- Binary serialization of memory items
- Incremental sync (delta updates)
- Priority-based replication
- Conflict resolution
- Cross-node memory sharing
"""

import struct
import hashlib
import gzip
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timezone, timedelta
from enum import IntEnum, IntFlag
import sys
from pathlib import Path

from config.memshadow_config import get_memshadow_config
from ..metrics.memshadow_metrics import get_memshadow_metrics_registry

logger = logging.getLogger(__name__)
_MEMSHADOW_CONFIG = get_memshadow_config()
_MEMSHADOW_METRICS = get_memshadow_metrics_registry()

# Try to import MEMSHADOW protocol
try:
    protocol_path = Path(__file__).parent.parent.parent.parent / "libs" / "memshadow-protocol" / "python"
    if protocol_path.exists() and str(protocol_path) not in sys.path:
        sys.path.insert(0, str(protocol_path))

    from dsmil_protocol import (
        MemshadowHeader, MemshadowMessage, 
        MessageType, Priority, MessageFlags,
        HEADER_SIZE, should_route_p2p
    )
    # Aliases for backward compatibility
    MemshadowMessageType = MessageType
    MemshadowMessagePriority = Priority
    PROTOCOL_AVAILABLE = True
except ImportError:
    PROTOCOL_AVAILABLE = False
    MemshadowMessageType = None
    MemshadowMessagePriority = None
    logger.warning("MEMSHADOW protocol not available")


class MemoryTier(IntEnum):
    """Memory tier levels"""
    WORKING = 1     # L1 - Fast, volatile
    EPISODIC = 2    # L2 - Recent episodes
    SEMANTIC = 3    # L3 - Long-term knowledge


class SyncOperation(IntEnum):
    """Types of sync operations"""
    INSERT = 1
    UPDATE = 2
    DELETE = 3
    MERGE = 4      # Merge with existing
    REPLICATE = 5  # Full replication request


class SyncPriority(IntEnum):
    """Sync priority levels"""
    BACKGROUND = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class SyncFlags(IntFlag):
    """Flags for sync operations"""
    NONE = 0
    COMPRESSED = 0x01
    ENCRYPTED = 0x02
    DELTA_ONLY = 0x04
    REQUIRES_ACK = 0x08
    CONFLICT_RESOLUTION = 0x10
    CROSS_TIER = 0x20  # Sync involves multiple tiers


# Binary format for memory item header
# Total: 48 bytes per item header
ITEM_HEADER_FORMAT = "!32sQBBHH"  # item_id (32), timestamp_ns (8), tier (1), op (1), priority (2), payload_len (2)
ITEM_HEADER_SIZE = struct.calcsize(ITEM_HEADER_FORMAT)  # 46 bytes, padded to 48


@dataclass
class MemorySyncItem:
    """
    Single item in a memory sync operation
    """
    item_id: str                    # 32-char hex ID
    timestamp_ns: int               # Nanosecond timestamp
    tier: MemoryTier
    operation: SyncOperation
    priority: SyncPriority
    payload: bytes                  # Serialized content

    # Metadata
    source_node: str = ""
    version: int = 1
    checksum: str = ""

    def pack(self) -> bytes:
        """Pack item to binary format"""
        # Ensure item_id is 32 bytes
        item_id_bytes = self.item_id[:32].encode().ljust(32, b'\x00')

        # Pack header
        header = struct.pack(
            ITEM_HEADER_FORMAT,
            item_id_bytes,
            self.timestamp_ns,
            self.tier.value,
            self.operation.value,
            self.priority.value,
            len(self.payload)
        )

        # Pad to 48 bytes if needed
        header = header.ljust(48, b'\x00')

        return header + self.payload

    @classmethod
    def unpack(cls, data: bytes) -> Tuple['MemorySyncItem', int]:
        """Unpack item from binary format. Returns (item, bytes_consumed)"""
        if len(data) < 48:
            raise ValueError(f"Data too short for item header. Need 48, got {len(data)}")

        # Unpack header (first 46 bytes of ITEM_HEADER_FORMAT)
        item_id_bytes, timestamp_ns, tier_val, op_val, priority_val, payload_len = \
            struct.unpack(ITEM_HEADER_FORMAT, data[:46])

        # Extract item_id (strip null padding)
        item_id = item_id_bytes.rstrip(b'\x00').decode('utf-8', errors='replace')

        # Get payload (starts at byte 48)
        payload_start = 48
        payload_end = payload_start + payload_len

        if len(data) < payload_end:
            raise ValueError(f"Data too short for payload. Need {payload_end}, got {len(data)}")

        payload = data[payload_start:payload_end]

        return cls(
            item_id=item_id,
            timestamp_ns=timestamp_ns,
            tier=MemoryTier(tier_val),
            operation=SyncOperation(op_val),
            priority=SyncPriority(priority_val),
            payload=payload,
        ), payload_end


@dataclass
class MemorySyncBatch:
    """
    Batch of memory sync items
    """
    batch_id: str
    source_node: str
    target_node: str
    tier: MemoryTier
    items: List[MemorySyncItem] = field(default_factory=list)

    # Batch metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    flags: SyncFlags = SyncFlags.NONE

    # Tracking
    sync_vector: Dict[str, int] = field(default_factory=dict)  # node_id -> last_sync_timestamp

    def add_item(self, item: MemorySyncItem):
        """Add item to batch"""
        self.items.append(item)

    def pack(self) -> bytes:
        """Pack batch to binary format with MEMSHADOW header"""
        # Serialize items
        items_data = b''.join(item.pack() for item in self.items)

        # Compress if large
        if len(items_data) > _MEMSHADOW_CONFIG.compression_threshold_bytes:
            items_data = gzip.compress(items_data)
            self.flags |= SyncFlags.COMPRESSED

        # Create payload with batch metadata
        metadata = json.dumps({
            "batch_id": self.batch_id,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "tier": self.tier.value,
            "item_count": len(self.items),
            "flags": int(self.flags),
            "sync_vector": self.sync_vector,
        }).encode()

        # Combine: 4-byte metadata length + metadata + items
        payload = struct.pack("!I", len(metadata)) + metadata + items_data

        # Create MEMSHADOW header
        if PROTOCOL_AVAILABLE:
            header = MemshadowHeader(
                msg_type=MessageType.MEMORY_SYNC,
                priority=Priority.NORMAL,
                batch_count=len(self.items),
                payload_len=len(payload),
                timestamp_ns=int(self.created_at.timestamp() * 1e9),
            )
            return header.pack() + payload
        else:
            # Fallback: just return payload with simple header
            simple_header = struct.pack("!III", 0x4D53594E, len(self.items), len(payload))  # MSYN magic
            return simple_header + payload

    @classmethod
    def unpack(cls, data: bytes) -> 'MemorySyncBatch':
        """Unpack batch from binary format"""
        offset = 0

        # Parse MEMSHADOW header if present
        if PROTOCOL_AVAILABLE and len(data) >= HEADER_SIZE:
            try:
                header = MemshadowHeader.unpack(data[:HEADER_SIZE])
                offset = HEADER_SIZE
            except:
                # Try simple header fallback
                if len(data) >= 12:
                    magic, item_count, payload_len = struct.unpack("!III", data[:12])
                    if magic == 0x4D53594E:  # MSYN
                        offset = 12
        elif len(data) >= 12:
            magic, item_count, payload_len = struct.unpack("!III", data[:12])
            if magic == 0x4D53594E:
                offset = 12

        # Parse metadata
        metadata_len = struct.unpack("!I", data[offset:offset+4])[0]
        offset += 4

        metadata = json.loads(data[offset:offset+metadata_len].decode())
        offset += metadata_len

        # Get items data
        items_data = data[offset:]

        # Decompress if needed
        flags = SyncFlags(metadata.get("flags", 0))
        if flags & SyncFlags.COMPRESSED:
            items_data = gzip.decompress(items_data)

        # Parse items
        items = []
        item_offset = 0
        while item_offset < len(items_data):
            try:
                item, consumed = MemorySyncItem.unpack(items_data[item_offset:])
                items.append(item)
                item_offset += consumed
            except Exception as e:
                logger.warning(f"Error parsing sync item at offset {item_offset}: {e}")
                break

        return cls(
            batch_id=metadata.get("batch_id", ""),
            source_node=metadata.get("source_node", ""),
            target_node=metadata.get("target_node", ""),
            tier=MemoryTier(metadata.get("tier", 1)),
            items=items,
            flags=flags,
            sync_vector=metadata.get("sync_vector", {}),
        )


class MemorySyncManager:
    """
    Manages memory synchronization across nodes

    Tracks sync state, handles conflicts, and coordinates
    with the mesh network for data transfer.
    """

    def __init__(self, node_id: str, mesh=None):
        self.node_id = node_id
        self._mesh = mesh
        self._config = _MEMSHADOW_CONFIG
        self._metrics = _MEMSHADOW_METRICS

        # Sync state tracking
        self._sync_vectors: Dict[str, Dict[str, int]] = {}  # tier -> {node_id -> timestamp}
        self._pending_syncs: Dict[str, MemorySyncBatch] = {}
        self._sync_history: List[Dict] = []

        # Conflict tracking
        self._conflicts: List[Dict] = []

        # Memory tier references
        self._memory_tiers: Dict[MemoryTier, Any] = {}

        # Statistics
        self.stats = {
            "items_synced": 0,
            "batches_sent": 0,
            "batches_received": 0,
            "conflicts_resolved": 0,
            "bytes_transferred": 0,
        }

        logger.info(f"MemorySyncManager initialized for {node_id}")

    def register_memory_tier(self, tier: MemoryTier, memory_instance):
        """Register a memory tier for synchronization"""
        self._memory_tiers[tier] = memory_instance
        self._sync_vectors[tier.name] = {}
        logger.info(f"Registered {tier.name} memory tier")

    def get_sync_vector(self, tier: MemoryTier) -> Dict[str, int]:
        """Get current sync vector for a tier"""
        return self._sync_vectors.get(tier.name, {})

    def update_sync_vector(self, tier: MemoryTier, node_id: str, timestamp: int):
        """Update sync vector after successful sync"""
        if tier.name not in self._sync_vectors:
            self._sync_vectors[tier.name] = {}
        self._sync_vectors[tier.name][node_id] = timestamp

    def create_delta_batch(self, tier: MemoryTier, target_node: str,
                          since_timestamp: int = 0) -> Optional[MemorySyncBatch]:
        """
        Create a sync batch with changes since given timestamp
        """
        if tier not in self._memory_tiers:
            logger.warning(f"Memory tier {tier.name} not registered")
            return None

        memory = self._memory_tiers[tier]

        # Get items modified since timestamp
        # This would call memory tier's get_modified_since() method
        items = []

        try:
            # Get modified items from memory tier
            modified = getattr(memory, 'get_modified_since', lambda x: [])(since_timestamp)

            for item_data in modified:
                item = MemorySyncItem(
                    item_id=item_data.get("item_id", hashlib.sha256(str(item_data).encode()).hexdigest()[:32]),
                    timestamp_ns=item_data.get("timestamp_ns", int(time.time() * 1e9)),
                    tier=tier,
                    operation=SyncOperation(item_data.get("operation", SyncOperation.UPDATE)),
                    priority=SyncPriority(item_data.get("priority", SyncPriority.NORMAL)),
                    payload=json.dumps(item_data.get("content", item_data)).encode(),
                )
                items.append(item)
        except Exception as e:
            logger.error(f"Error getting modified items: {e}")

        if not items:
            return None

        if len(items) > self._config.max_batch_items:
            logger.debug(
                "Truncating delta batch from %d to %d items (config max)",
                len(items),
                self._config.max_batch_items,
            )
            items = items[: self._config.max_batch_items]

        batch = MemorySyncBatch(
            batch_id=hashlib.sha256(f"{self.node_id}-{time.time()}".encode()).hexdigest()[:16],
            source_node=self.node_id,
            target_node=target_node,
            tier=tier,
            items=items,
            flags=SyncFlags.DELTA_ONLY | SyncFlags.REQUIRES_ACK,
            sync_vector=self._sync_vectors.get(tier.name, {}),
        )

        return batch

    def apply_sync_batch(self, batch: MemorySyncBatch) -> Tuple[int, int]:
        """
        Apply a sync batch to local memory

        Returns: (items_applied, conflicts_detected)
        """
        start_time = time.time()
        if batch.tier not in self._memory_tiers:
            logger.warning(f"Memory tier {batch.tier.name} not registered")
            return 0, 0

        memory = self._memory_tiers[batch.tier]
        applied = 0
        conflicts = 0

        for item in batch.items:
            try:
                # Check for conflicts
                conflict = self._detect_conflict(item, memory)

                if conflict:
                    # Resolve conflict
                    resolved = self._resolve_conflict(item, conflict, memory)
                    if not resolved:
                        conflicts += 1
                        continue
                    self.stats["conflicts_resolved"] += 1

                # Apply item
                if item.operation == SyncOperation.INSERT:
                    self._apply_insert(item, memory)
                elif item.operation == SyncOperation.UPDATE:
                    self._apply_update(item, memory)
                elif item.operation == SyncOperation.DELETE:
                    self._apply_delete(item, memory)
                elif item.operation == SyncOperation.MERGE:
                    self._apply_merge(item, memory)

                applied += 1

            except Exception as e:
                logger.error(f"Error applying sync item {item.item_id}: {e}")

        # Update sync vector
        if batch.items:
            latest_ts = max(item.timestamp_ns for item in batch.items)
            self.update_sync_vector(batch.tier, batch.source_node, latest_ts)

        self.stats["items_synced"] += applied
        self.stats["batches_received"] += 1
        self._metrics.increment("memshadow_batches_received")
        if conflicts:
            self._metrics.increment("memshadow_conflicts_detected", conflicts)

        elapsed_ms = (time.time() - start_time) * 1000
        self._metrics.observe_latency(elapsed_ms)

        return applied, conflicts

    def _detect_conflict(self, item: MemorySyncItem, memory) -> Optional[Dict]:
        """Detect if item conflicts with existing data"""
        try:
            existing = getattr(memory, 'get_by_id', lambda x: None)(item.item_id)
            if existing:
                existing_ts = existing.get("timestamp_ns", 0)
                if existing_ts >= item.timestamp_ns:
                    return {
                        "type": "timestamp_conflict",
                        "existing": existing,
                        "incoming": item,
                    }
        except:
            pass
        return None

    def _resolve_conflict(self, item: MemorySyncItem, conflict: Dict, memory) -> bool:
        """
        Resolve a sync conflict

        Default strategy: Last-Write-Wins (LWW) with tie-breaker on node_id
        """
        existing = conflict["existing"]

        # LWW
        if item.timestamp_ns > existing.get("timestamp_ns", 0):
            return True  # Apply incoming
        elif item.timestamp_ns == existing.get("timestamp_ns", 0):
            # Tie-breaker: higher node_id wins
            if item.source_node > existing.get("source_node", ""):
                return True

        # Keep existing, log conflict
        self._conflicts.append({
            "item_id": item.item_id,
            "incoming_node": item.source_node,
            "existing_node": existing.get("source_node", ""),
            "resolved": "keep_existing",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return False

    def _apply_insert(self, item: MemorySyncItem, memory):
        """Apply insert operation"""
        content = json.loads(item.payload.decode())
        add_method = getattr(memory, 'add', None) or getattr(memory, 'store', None)
        if add_method:
            add_method(item.item_id, content)

    def _apply_update(self, item: MemorySyncItem, memory):
        """Apply update operation"""
        content = json.loads(item.payload.decode())
        update_method = getattr(memory, 'update', None)
        if update_method:
            update_method(item.item_id, content)
        else:
            # Fallback to add (overwrite)
            self._apply_insert(item, memory)

    def _apply_delete(self, item: MemorySyncItem, memory):
        """Apply delete operation"""
        delete_method = getattr(memory, 'delete', None) or getattr(memory, 'remove', None)
        if delete_method:
            delete_method(item.item_id)

    def _apply_merge(self, item: MemorySyncItem, memory):
        """Apply merge operation"""
        content = json.loads(item.payload.decode())
        merge_method = getattr(memory, 'merge', None)
        if merge_method:
            merge_method(item.item_id, content)
        else:
            # Fallback to update
            self._apply_update(item, memory)

    async def sync_with_peer(self, peer_id: str, tier: MemoryTier) -> Dict:
        """
        Perform full sync with a peer for given tier
        """
        start_time = time.time()
        # Get last sync timestamp for this peer
        last_sync = self._sync_vectors.get(tier.name, {}).get(peer_id, 0)

        # Create delta batch
        batch = self.create_delta_batch(tier, peer_id, last_sync)

        if not batch:
            return {"status": "up_to_date", "items_synced": 0}

        # Send via mesh
        if self._mesh:
            try:
                from messages import MessageTypes
                self._mesh.send(peer_id, MessageTypes.VECTOR_SYNC, batch.pack())
                self.stats["batches_sent"] += 1
                self.stats["bytes_transferred"] += len(batch.pack())
                self._metrics.increment("memshadow_batches_sent")
                self._metrics.observe_latency((time.time() - start_time) * 1000)

                return {"status": "sent", "items": len(batch.items)}
            except Exception as e:
                logger.error(f"Failed to send sync to {peer_id}: {e}")
                return {"status": "error", "error": str(e)}

        return {"status": "no_mesh", "items": len(batch.items)}

    def get_stats(self) -> Dict:
        """Get sync statistics"""
        return {
            **self.stats,
            "registered_tiers": list(self._memory_tiers.keys()),
            "pending_syncs": len(self._pending_syncs),
            "conflicts": len(self._conflicts),
        }


if __name__ == "__main__":
    # Self-test
    print("Memory Sync Protocol Test")
    print("=" * 50)

    # Create test item
    item = MemorySyncItem(
        item_id="abc123def456789012345678901234ab",
        timestamp_ns=int(time.time() * 1e9),
        tier=MemoryTier.WORKING,
        operation=SyncOperation.INSERT,
        priority=SyncPriority.NORMAL,
        payload=json.dumps({"key": "value", "number": 42}).encode(),
    )

    print(f"\n[1] Pack/Unpack Item")
    packed = item.pack()
    print(f"    Packed size: {len(packed)} bytes")

    unpacked, consumed = MemorySyncItem.unpack(packed)
    print(f"    Unpacked item_id: {unpacked.item_id}")
    print(f"    Tier: {unpacked.tier.name}")

    # Create batch
    print(f"\n[2] Pack/Unpack Batch")
    batch = MemorySyncBatch(
        batch_id="batch-001",
        source_node="node-a",
        target_node="node-b",
        tier=MemoryTier.EPISODIC,
        items=[item],
    )

    packed_batch = batch.pack()
    print(f"    Batch packed size: {len(packed_batch)} bytes")

    unpacked_batch = MemorySyncBatch.unpack(packed_batch)
    print(f"    Batch ID: {unpacked_batch.batch_id}")
    print(f"    Items: {len(unpacked_batch.items)}")

    # Test sync manager
    print(f"\n[3] Sync Manager")
    manager = MemorySyncManager("test-node")
    print(f"    Stats: {manager.get_stats()}")

    print("\n" + "=" * 50)
    print("Memory Sync Protocol test complete")

