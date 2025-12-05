"""
Memory Sync Protocol

Cross-node memory tier synchronization using MEMSHADOW Protocol v2.

Implements:
- MemorySyncItem: Single memory item for sync
- MemorySyncBatch: Batch of items with checksums
- MemorySyncManager: Delta computation, sync vectors, conflict resolution

Based on: HUB_DOCS/MEMSHADOW_INTEGRATION.md

Performance Targets:
- Batch size: 10-100 items
- Background sync: every ~30s
- Urgent sync: immediate
- Compression: for batches > 1KB
"""

import asyncio
import hashlib
import json
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

import structlog

# Import canonical protocol from libs/memshadow-protocol
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "libs" / "memshadow-protocol" / "python"))

from dsmil_protocol import (
    MemshadowHeader,
    MemshadowMessage,
    MessageType,
    Priority,
    MessageFlags,
    MemoryTier,
    SyncOperation,
    HEADER_SIZE,
    create_memory_sync_message,
)

logger = structlog.get_logger()


# ============================================================================
# Configuration (configurable, not hardcoded)
# ============================================================================

@dataclass
class SyncConfig:
    """Sync configuration parameters"""
    batch_size_min: int = 10
    batch_size_max: int = 100
    compression_threshold_bytes: int = 1024  # 1KB
    background_sync_interval_sec: float = 30.0
    urgent_sync_delay_sec: float = 0.1
    max_retries: int = 3
    conflict_resolution: str = "last_write_wins"  # or "first_write_wins", "merge"


DEFAULT_CONFIG = SyncConfig()


# ============================================================================
# Sync Priority (maps to MEMSHADOW Priority)
# ============================================================================

class SyncPriority(IntEnum):
    """Sync priority levels (map to MEMSHADOW Priority)"""
    BACKGROUND = 0  # LOW
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 4      # EMERGENCY
    
    def to_memshadow_priority(self) -> Priority:
        """Convert to MEMSHADOW Priority enum"""
        mapping = {0: Priority.LOW, 1: Priority.NORMAL, 2: Priority.HIGH, 4: Priority.EMERGENCY}
        return mapping.get(self.value, Priority.NORMAL)


# ============================================================================
# MemorySyncItem
# ============================================================================

@dataclass
class MemorySyncItem:
    """
    Single memory item for synchronization.
    
    Fields per MEMSHADOW_INTEGRATION.md:
    - item_id: UUID
    - timestamp_ns: nanosecond timestamp
    - tier: L1/L2/L3
    - operation: INSERT, UPDATE, DELETE, MERGE, REPLICATE
    - priority: sync priority
    - payload: compressed bytes
    """
    item_id: UUID
    timestamp_ns: int
    tier: MemoryTier
    operation: SyncOperation
    priority: SyncPriority
    payload: bytes
    
    # Additional metadata
    content_hash: str = ""
    version: int = 1
    source_node: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_compressed: bool = False
    
    def __post_init__(self):
        if isinstance(self.item_id, str):
            self.item_id = UUID(self.item_id)
        if not self.content_hash and self.payload:
            self.content_hash = hashlib.sha256(self.payload).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "item_id": str(self.item_id),
            "timestamp_ns": self.timestamp_ns,
            "tier": self.tier.value,
            "operation": self.operation.value,
            "priority": self.priority.value,
            "payload": self.payload.hex(),
            "content_hash": self.content_hash,
            "version": self.version,
            "source_node": self.source_node,
            "metadata": self.metadata,
            "is_compressed": self.is_compressed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemorySyncItem":
        """Deserialize from dictionary"""
        return cls(
            item_id=UUID(data["item_id"]),
            timestamp_ns=data["timestamp_ns"],
            tier=MemoryTier(data["tier"]),
            operation=SyncOperation(data["operation"]),
            priority=SyncPriority(data["priority"]),
            payload=bytes.fromhex(data["payload"]),
            content_hash=data.get("content_hash", ""),
            version=data.get("version", 1),
            source_node=data.get("source_node", ""),
            metadata=data.get("metadata", {}),
            is_compressed=data.get("is_compressed", False),
        )
    
    @classmethod
    def create(
        cls,
        payload: bytes,
        tier: MemoryTier = MemoryTier.WORKING,
        operation: SyncOperation = SyncOperation.INSERT,
        priority: SyncPriority = SyncPriority.NORMAL,
        source_node: str = "",
        compress: bool = True,
        compression_threshold: int = 1024,
    ) -> "MemorySyncItem":
        """Create a new sync item with optional compression"""
        is_compressed = False
        if compress and len(payload) > compression_threshold:
            compressed = zlib.compress(payload, level=6)
            if len(compressed) < len(payload):
                payload = compressed
                is_compressed = True
        
        return cls(
            item_id=uuid4(),
            timestamp_ns=int(time.time() * 1e9),
            tier=tier,
            operation=operation,
            priority=priority,
            payload=payload,
            source_node=source_node,
            is_compressed=is_compressed,
        )
    
    def decompress_payload(self) -> bytes:
        """Get decompressed payload"""
        if self.is_compressed:
            return zlib.decompress(self.payload)
        return self.payload


# ============================================================================
# MemorySyncBatch
# ============================================================================

@dataclass
class MemorySyncBatch:
    """
    Batch of sync items for efficient transmission.
    
    Fields per MEMSHADOW_INTEGRATION.md:
    - batch_id: UUID
    - source_node: originating node
    - target_node: destination ("*" for broadcast)
    - tier: memory tier
    - items: list of MemorySyncItem
    - checksum: SHA-256 of all item hashes
    - flags: MessageFlags (BATCHED, COMPRESSED, etc.)
    """
    batch_id: str = field(default_factory=lambda: str(uuid4()))
    source_node: str = ""
    target_node: str = "*"
    tier: MemoryTier = MemoryTier.WORKING
    items: List[MemorySyncItem] = field(default_factory=list)
    checksum: str = ""
    flags: int = 0
    priority: SyncPriority = SyncPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.checksum and self.items:
            self.checksum = self._compute_checksum()
        if self.items:
            self.flags |= int(MessageFlags.BATCHED)
    
    def _compute_checksum(self) -> str:
        """Compute batch checksum from all item hashes"""
        hasher = hashlib.sha256()
        for item in sorted(self.items, key=lambda x: str(x.item_id)):
            hasher.update(item.content_hash.encode())
        return hasher.hexdigest()[:32]
    
    def validate(self) -> bool:
        """Validate batch integrity"""
        if not self.items:
            return True
        return self._compute_checksum() == self.checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "batch_id": self.batch_id,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "tier": self.tier.value,
            "items": [item.to_dict() for item in self.items],
            "checksum": self.checksum,
            "flags": self.flags,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemorySyncBatch":
        """Deserialize from dictionary"""
        return cls(
            batch_id=data["batch_id"],
            source_node=data["source_node"],
            target_node=data.get("target_node", "*"),
            tier=MemoryTier(data["tier"]),
            items=[MemorySyncItem.from_dict(i) for i in data.get("items", [])],
            checksum=data.get("checksum", ""),
            flags=data.get("flags", 0),
            priority=SyncPriority(data.get("priority", 1)),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
        )
    
    def to_bytes(self) -> bytes:
        """Serialize batch to bytes for transmission"""
        data = json.dumps(self.to_dict()).encode("utf-8")
        
        # Compress if large
        if len(data) > DEFAULT_CONFIG.compression_threshold_bytes:
            compressed = zlib.compress(data, level=6)
            if len(compressed) < len(data):
                self.flags |= int(MessageFlags.COMPRESSED)
                return compressed
        
        return data
    
    @classmethod
    def from_bytes(cls, data: bytes, compressed: bool = False) -> "MemorySyncBatch":
        """Deserialize batch from bytes"""
        if compressed:
            data = zlib.decompress(data)
        return cls.from_dict(json.loads(data.decode("utf-8")))
    
    def to_memshadow_message(self) -> MemshadowMessage:
        """Wrap batch in MEMSHADOW message with header"""
        payload = self.to_bytes()
        compressed = bool(self.flags & int(MessageFlags.COMPRESSED))
        
        return create_memory_sync_message(
            payload=payload,
            priority=self.priority.to_memshadow_priority(),
            batch_count=len(self.items),
            compressed=compressed,
        )
    
    @classmethod
    def from_memshadow_message(cls, msg: MemshadowMessage) -> "MemorySyncBatch":
        """Parse batch from MEMSHADOW message"""
        compressed = bool(msg.header.flags & MessageFlags.COMPRESSED)
        return cls.from_bytes(msg.payload, compressed=compressed)
    
    @property
    def size_bytes(self) -> int:
        """Estimate batch size in bytes"""
        return sum(len(item.payload) for item in self.items)


# ============================================================================
# SyncVector
# ============================================================================

class SyncVector:
    """
    Per-peer per-tier sync vector tracking versions.
    
    Used for delta-only synchronization.
    """
    
    def __init__(self):
        self._versions: Dict[str, int] = {}  # item_id -> version
        self._timestamps: Dict[str, int] = {}  # item_id -> timestamp_ns
    
    def update(self, item_id: str, version: int, timestamp_ns: int = 0):
        """Update version for an item"""
        self._versions[item_id] = version
        if timestamp_ns:
            self._timestamps[item_id] = timestamp_ns
    
    def get_version(self, item_id: str) -> int:
        """Get current version for an item"""
        return self._versions.get(item_id, 0)
    
    def get_all(self) -> Dict[str, int]:
        """Get all version mappings"""
        return dict(self._versions)
    
    def compute_delta(self, remote_vector: Dict[str, int]) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Compute delta between local and remote vectors.
        
        Returns: (new_ids, updated_ids, deleted_ids)
        """
        local_ids = set(self._versions.keys())
        remote_ids = set(remote_vector.keys())
        
        new_ids = remote_ids - local_ids
        deleted_ids = local_ids - remote_ids
        updated_ids = {
            mid for mid in local_ids & remote_ids
            if remote_vector[mid] > self._versions[mid]
        }
        
        return new_ids, updated_ids, deleted_ids


# ============================================================================
# MemorySyncManager
# ============================================================================

class MemorySyncManager:
    """
    Central manager for memory synchronization.
    
    Responsibilities:
    - Register memory tiers
    - Create delta batches
    - Apply sync batches with conflict resolution
    - Sync with peers via mesh network
    - Track per-peer/per-tier sync vectors
    """
    
    def __init__(
        self,
        node_id: str,
        config: Optional[SyncConfig] = None,
        mesh_send_callback: Optional[Callable] = None,
    ):
        self.node_id = node_id
        self.config = config or DEFAULT_CONFIG
        self.mesh_send = mesh_send_callback
        
        # Registered memory tiers: tier -> instance
        self._tiers: Dict[MemoryTier, Any] = {}
        
        # Sync vectors: (peer_id, tier) -> SyncVector
        self._sync_vectors: Dict[Tuple[str, MemoryTier], SyncVector] = {}
        
        # Local item store: (tier, item_id) -> MemorySyncItem
        self._local_store: Dict[Tuple[MemoryTier, str], MemorySyncItem] = {}
        
        # Pending operations
        self._pending_batches: Dict[str, MemorySyncBatch] = {}
        
        # Stats
        self.stats = {
            "batches_created": 0,
            "batches_applied": 0,
            "items_synced": 0,
            "conflicts_resolved": 0,
            "bytes_transferred": 0,
        }
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        
        logger.info("MemorySyncManager initialized", node_id=node_id)
    
    def register_memory_tier(self, tier: MemoryTier, instance: Any):
        """
        Register a memory tier instance.
        
        Args:
            tier: MemoryTier.WORKING, EPISODIC, or SEMANTIC
            instance: Memory tier implementation (WorkingMemory, etc.)
        """
        self._tiers[tier] = instance
        logger.info("Memory tier registered", tier=tier.name, node=self.node_id)
    
    def get_sync_vector(self, peer_id: str, tier: MemoryTier) -> Dict[str, int]:
        """Get sync vector for a peer/tier combination"""
        key = (peer_id, tier)
        if key not in self._sync_vectors:
            self._sync_vectors[key] = SyncVector()
        return self._sync_vectors[key].get_all()
    
    async def store_local(self, item: MemorySyncItem):
        """Store an item in local state"""
        async with self._lock:
            key = (item.tier, str(item.item_id))
            self._local_store[key] = item
            
            # Update our own sync vector
            if (self.node_id, item.tier) not in self._sync_vectors:
                self._sync_vectors[(self.node_id, item.tier)] = SyncVector()
            self._sync_vectors[(self.node_id, item.tier)].update(
                str(item.item_id), item.version, item.timestamp_ns
            )
    
    async def create_delta_batch(
        self,
        peer_id: str,
        tier: MemoryTier,
        priority: SyncPriority = SyncPriority.NORMAL,
    ) -> MemorySyncBatch:
        """
        Create a delta batch containing only items changed since last sync with peer.
        
        Args:
            peer_id: Target peer node ID
            tier: Memory tier to sync
            priority: Sync priority level
        
        Returns:
            MemorySyncBatch with delta items
        """
        async with self._lock:
            # Get peer's sync vector
            remote_vector = self.get_sync_vector(peer_id, tier)
            
            # Get our current vector
            local_vector = self.get_sync_vector(self.node_id, tier)
            
            # Compute delta
            items_to_sync = []
            for (t, item_id), item in self._local_store.items():
                if t != tier:
                    continue
                local_ver = local_vector.get(item_id, 0)
                remote_ver = remote_vector.get(item_id, 0)
                if local_ver > remote_ver:
                    items_to_sync.append(item)
            
            # Respect batch size limits
            if len(items_to_sync) > self.config.batch_size_max:
                items_to_sync = items_to_sync[:self.config.batch_size_max]
        
        batch = MemorySyncBatch(
            source_node=self.node_id,
            target_node=peer_id,
            tier=tier,
            items=items_to_sync,
            priority=priority,
        )
        
        self._pending_batches[batch.batch_id] = batch
        self.stats["batches_created"] += 1
        
        logger.debug(
            "Delta batch created",
            batch_id=batch.batch_id,
            items=len(items_to_sync),
            tier=tier.name,
            target=peer_id,
        )
        
        return batch
    
    async def apply_sync_batch(
        self,
        batch: MemorySyncBatch,
    ) -> Dict[str, Any]:
        """
        Apply an incoming sync batch with conflict resolution.
        
        Uses last-write-wins (LWW) conflict resolution:
        - Higher timestamp wins
        - Tie-breaker: node_id lexicographic comparison
        
        Returns:
            Result dict with applied/skipped/conflict counts
        """
        start_time = time.time()
        
        result = {
            "success": True,
            "applied": 0,
            "skipped": 0,
            "conflicts": 0,
            "errors": [],
        }
        
        # Validate batch
        if not batch.validate():
            result["success"] = False
            result["errors"].append("Checksum validation failed")
            logger.warning("Batch checksum failed", batch_id=batch.batch_id)
            return result
        
        async with self._lock:
            for item in batch.items:
                try:
                    key = (item.tier, str(item.item_id))
                    existing = self._local_store.get(key)
                    
                    if existing:
                        # Conflict resolution
                        winner = self._resolve_conflict(existing, item)
                        if winner == existing:
                            result["skipped"] += 1
                            continue
                        result["conflicts"] += 1
                        self.stats["conflicts_resolved"] += 1
                    
                    # Apply item
                    if item.operation == SyncOperation.DELETE:
                        self._local_store.pop(key, None)
                    else:
                        self._local_store[key] = item
                    
                    # Update sync vector for source peer
                    peer_key = (batch.source_node, item.tier)
                    if peer_key not in self._sync_vectors:
                        self._sync_vectors[peer_key] = SyncVector()
                    self._sync_vectors[peer_key].update(
                        str(item.item_id), item.version, item.timestamp_ns
                    )
                    
                    result["applied"] += 1
                    self.stats["items_synced"] += 1
                    
                except Exception as e:
                    result["errors"].append(f"Item {item.item_id}: {str(e)}")
                    logger.error("Failed to apply sync item", item_id=str(item.item_id), error=str(e))
        
        self.stats["batches_applied"] += 1
        self.stats["bytes_transferred"] += batch.size_bytes
        
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(
            "Sync batch applied",
            batch_id=batch.batch_id,
            source=batch.source_node,
            applied=result["applied"],
            conflicts=result["conflicts"],
            latency_ms=f"{latency_ms:.2f}",
        )
        
        return result
    
    def _resolve_conflict(
        self,
        local: MemorySyncItem,
        remote: MemorySyncItem,
    ) -> MemorySyncItem:
        """
        Resolve conflict between local and remote items.
        
        Default: Last-write-wins (LWW) with node_id tie-breaker.
        """
        if self.config.conflict_resolution == "first_write_wins":
            if remote.timestamp_ns < local.timestamp_ns:
                return remote
            elif remote.timestamp_ns == local.timestamp_ns:
                return remote if remote.source_node < local.source_node else local
            return local
        
        else:  # last_write_wins (default)
            if remote.timestamp_ns > local.timestamp_ns:
                return remote
            elif remote.timestamp_ns == local.timestamp_ns:
                return remote if remote.source_node > local.source_node else local
            return local
    
    async def sync_with_peer(
        self,
        peer_id: str,
        tier: MemoryTier,
        priority: SyncPriority = SyncPriority.NORMAL,
    ) -> bool:
        """
        Initiate sync with a peer for a specific tier.
        
        Uses the mesh network to send delta batch.
        
        Returns:
            True if sync was initiated successfully
        """
        if not self.mesh_send:
            logger.warning("No mesh callback configured")
            return False
        
        batch = await self.create_delta_batch(peer_id, tier, priority)
        
        if not batch.items:
            logger.debug("No items to sync", peer=peer_id, tier=tier.name)
            return True
        
        try:
            msg = batch.to_memshadow_message()
            await self.mesh_send(peer_id, msg.pack())
            
            logger.info(
                "Sync initiated",
                peer=peer_id,
                tier=tier.name,
                items=len(batch.items),
                priority=priority.name,
            )
            return True
            
        except Exception as e:
            logger.error("Sync failed", peer=peer_id, error=str(e))
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sync manager statistics"""
        return {
            "node_id": self.node_id,
            "registered_tiers": [t.name for t in self._tiers.keys()],
            "local_items": len(self._local_store),
            "sync_vectors": len(self._sync_vectors),
            "pending_batches": len(self._pending_batches),
            **self.stats,
        }


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "SyncConfig",
    "DEFAULT_CONFIG",
    "SyncPriority",
    "MemorySyncItem",
    "MemorySyncBatch",
    "SyncVector",
    "MemorySyncManager",
]
