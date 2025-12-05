"""
Memory Sync Manager

Manages sync vectors, delta computation, and batch handling for MEMSHADOW
cross-node memory synchronization.

Based on: HUB_DOCS/MEMSHADOW_INTEGRATION.md
"""

import asyncio
import hashlib
import json
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

import numpy as np
import structlog

logger = structlog.get_logger()


class MemoryTier(IntEnum):
    """Memory tier levels for sync operations"""
    L1_WORKING = 1    # Hot working memory (RAMDISK, 256d)
    L2_EPISODIC = 2   # Episode/session memory (NVMe, full vectors)
    L3_SEMANTIC = 3   # Long-term semantic (Cold, compressed)
    
    @property
    def description(self) -> str:
        descs = {
            1: "Working Memory (Hot)",
            2: "Episodic Memory (Warm)",
            3: "Semantic Memory (Cold)",
        }
        return descs.get(self.value, "Unknown")
    
    @property
    def max_dimensions(self) -> int:
        """Maximum embedding dimensions for this tier"""
        dims = {
            1: 256,    # L1: Compressed for speed
            2: 2048,   # L2: Full resolution
            3: 4096,   # L3: Maximum fidelity
        }
        return dims.get(self.value, 256)


class ConflictResolution(IntEnum):
    """Conflict resolution strategies"""
    LAST_WRITE_WINS = 1
    FIRST_WRITE_WINS = 2
    HIGHEST_VERSION = 3
    MERGE = 4
    MANUAL = 5


@dataclass
class MemorySyncItem:
    """
    Represents a single memory item for synchronization.
    
    Contains the memory ID, embedding (optionally compressed),
    metadata, and versioning information.
    """
    memory_id: UUID
    content_hash: str
    tier: MemoryTier
    embedding: bytes                  # Compressed/serialized embedding
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    operation: str = "upsert"         # "upsert", "delete", "update_meta"
    
    # Embedding metadata
    embedding_dim: int = 256
    is_compressed: bool = False
    
    def __post_init__(self):
        if isinstance(self.memory_id, str):
            self.memory_id = UUID(self.memory_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "memory_id": str(self.memory_id),
            "content_hash": self.content_hash,
            "tier": self.tier.value,
            "embedding": self.embedding.hex() if self.embedding else "",
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "operation": self.operation,
            "embedding_dim": self.embedding_dim,
            "is_compressed": self.is_compressed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemorySyncItem":
        """Create from dictionary"""
        return cls(
            memory_id=UUID(data["memory_id"]),
            content_hash=data["content_hash"],
            tier=MemoryTier(data["tier"]),
            embedding=bytes.fromhex(data["embedding"]) if data.get("embedding") else b"",
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
            version=data.get("version", 1),
            operation=data.get("operation", "upsert"),
            embedding_dim=data.get("embedding_dim", 256),
            is_compressed=data.get("is_compressed", False),
        )
    
    @classmethod
    def from_embedding(
        cls,
        memory_id: UUID,
        embedding: np.ndarray,
        tier: MemoryTier,
        metadata: Optional[Dict[str, Any]] = None,
        compress: bool = True,
    ) -> "MemorySyncItem":
        """Create from numpy embedding array"""
        # Serialize embedding
        embedding_bytes = embedding.astype(np.float32).tobytes()
        
        # Compress if requested
        is_compressed = False
        if compress:
            compressed = zlib.compress(embedding_bytes, level=6)
            if len(compressed) < len(embedding_bytes):
                embedding_bytes = compressed
                is_compressed = True
        
        # Compute content hash from embedding
        content_hash = hashlib.sha256(embedding.tobytes()).hexdigest()
        
        return cls(
            memory_id=memory_id,
            content_hash=content_hash,
            tier=tier,
            embedding=embedding_bytes,
            metadata=metadata or {},
            embedding_dim=len(embedding),
            is_compressed=is_compressed,
        )
    
    def get_embedding(self) -> np.ndarray:
        """Extract numpy embedding from bytes"""
        data = self.embedding
        if self.is_compressed:
            data = zlib.decompress(data)
        return np.frombuffer(data, dtype=np.float32)


@dataclass
class MemorySyncBatch:
    """
    Batch of memory sync items for efficient transmission.
    
    Contains multiple MemorySyncItem objects along with
    routing and integrity information.
    """
    batch_id: str = field(default_factory=lambda: str(uuid4()))
    source_node: str = ""
    target_node: str = "*"            # "*" for broadcast
    tier: MemoryTier = MemoryTier.L1_WORKING
    items: List[MemorySyncItem] = field(default_factory=list)
    priority: int = 1                 # Maps to Priority enum
    flags: int = 0                    # MessageFlags
    timestamp: datetime = field(default_factory=datetime.utcnow)
    checksum: str = ""
    
    def __post_init__(self):
        if not self.checksum and self.items:
            self.checksum = self._compute_checksum()
    
    def _compute_checksum(self) -> str:
        """Compute batch checksum from items"""
        hasher = hashlib.sha256()
        for item in self.items:
            hasher.update(item.content_hash.encode())
        return hasher.hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "batch_id": self.batch_id,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "tier": self.tier.value,
            "items": [item.to_dict() for item in self.items],
            "priority": self.priority,
            "flags": self.flags,
            "timestamp": self.timestamp.isoformat(),
            "checksum": self.checksum,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemorySyncBatch":
        """Create from dictionary"""
        return cls(
            batch_id=data["batch_id"],
            source_node=data["source_node"],
            target_node=data.get("target_node", "*"),
            tier=MemoryTier(data["tier"]),
            items=[MemorySyncItem.from_dict(item) for item in data.get("items", [])],
            priority=data.get("priority", 1),
            flags=data.get("flags", 0),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
            checksum=data.get("checksum", ""),
        )
    
    def to_bytes(self) -> bytes:
        """Serialize batch to bytes for transmission"""
        return json.dumps(self.to_dict()).encode("utf-8")
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "MemorySyncBatch":
        """Deserialize batch from bytes"""
        return cls.from_dict(json.loads(data.decode("utf-8")))
    
    def validate(self) -> bool:
        """Validate batch integrity"""
        if not self.items:
            return True
        computed = self._compute_checksum()
        return computed == self.checksum
    
    @property
    def size_bytes(self) -> int:
        """Estimate batch size in bytes"""
        return sum(len(item.embedding) for item in self.items)


@dataclass
class SyncResult:
    """Result of a sync operation"""
    success: bool
    applied_count: int = 0
    skipped_count: int = 0
    conflict_count: int = 0
    errors: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "applied_count": self.applied_count,
            "skipped_count": self.skipped_count,
            "conflict_count": self.conflict_count,
            "errors": self.errors,
            "latency_ms": self.latency_ms,
        }


class SyncVector:
    """
    Sync vector tracking versions per memory ID.
    
    Used to compute deltas between local and remote state.
    """
    
    def __init__(self):
        self._versions: Dict[str, int] = {}  # memory_id -> version
        self._timestamps: Dict[str, datetime] = {}  # memory_id -> last_updated
    
    def update(self, memory_id: str, version: int):
        """Update version for a memory ID"""
        self._versions[memory_id] = version
        self._timestamps[memory_id] = datetime.utcnow()
    
    def get_version(self, memory_id: str) -> int:
        """Get current version for a memory ID"""
        return self._versions.get(memory_id, 0)
    
    def get_all(self) -> Dict[str, int]:
        """Get all version mappings"""
        return dict(self._versions)
    
    def compute_delta(self, remote_vector: Dict[str, int]) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Compute delta between local and remote vectors.
        
        Returns:
            Tuple of (new_ids, updated_ids, deleted_ids)
        """
        local_ids = set(self._versions.keys())
        remote_ids = set(remote_vector.keys())
        
        # New in remote
        new_ids = remote_ids - local_ids
        
        # Deleted from remote
        deleted_ids = local_ids - remote_ids
        
        # Updated (version mismatch)
        updated_ids = set()
        for mid in local_ids & remote_ids:
            if remote_vector[mid] > self._versions[mid]:
                updated_ids.add(mid)
        
        return new_ids, updated_ids, deleted_ids
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "versions": self._versions,
            "timestamps": {k: v.isoformat() for k, v in self._timestamps.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncVector":
        vec = cls()
        vec._versions = data.get("versions", {})
        vec._timestamps = {
            k: datetime.fromisoformat(v) 
            for k, v in data.get("timestamps", {}).items()
        }
        return vec


class MemorySyncManager:
    """
    Manages per-node sync vectors and delta computation.
    
    Responsibilities:
    - Track sync vectors per (node, tier)
    - Compute deltas for efficient sync
    - Create and apply sync batches
    - Handle conflict resolution
    """
    
    def __init__(
        self,
        node_id: str,
        conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS,
    ):
        self.node_id = node_id
        self.conflict_resolution = conflict_resolution
        
        # Sync vectors: (node_id, tier) -> SyncVector
        self._sync_vectors: Dict[Tuple[str, MemoryTier], SyncVector] = {}
        
        # Local memory store: (tier, memory_id) -> MemorySyncItem
        self._local_store: Dict[Tuple[MemoryTier, str], MemorySyncItem] = {}
        
        # Pending batches: batch_id -> MemorySyncBatch
        self._pending_batches: Dict[str, MemorySyncBatch] = {}
        
        # Statistics
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
    
    def get_sync_vector(self, node_id: str, tier: MemoryTier) -> Dict[str, int]:
        """Get sync vector for a node/tier pair"""
        key = (node_id, tier)
        if key not in self._sync_vectors:
            self._sync_vectors[key] = SyncVector()
        return self._sync_vectors[key].get_all()
    
    def update_sync_vector(
        self, 
        node_id: str, 
        tier: MemoryTier, 
        memory_id: str, 
        version: int
    ):
        """Update sync vector for a specific memory"""
        key = (node_id, tier)
        if key not in self._sync_vectors:
            self._sync_vectors[key] = SyncVector()
        self._sync_vectors[key].update(memory_id, version)
    
    def compute_delta(
        self,
        local_vector: Dict[str, int],
        remote_vector: Dict[str, int],
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Compute delta between local and remote sync vectors.
        
        Returns:
            Tuple of (new_ids, updated_ids, deleted_ids) representing
            changes needed to bring remote in sync with local.
        """
        local_ids = set(local_vector.keys())
        remote_ids = set(remote_vector.keys())
        
        # New in local (not in remote)
        new_ids = local_ids - remote_ids
        
        # Deleted from local (in remote but not local)
        deleted_ids = remote_ids - local_ids
        
        # Updated (version in local > version in remote)
        updated_ids = set()
        for mid in local_ids & remote_ids:
            if local_vector[mid] > remote_vector.get(mid, 0):
                updated_ids.add(mid)
        
        return new_ids, updated_ids, deleted_ids
    
    async def store_local(self, item: MemorySyncItem):
        """Store an item in the local store"""
        async with self._lock:
            key = (item.tier, str(item.memory_id))
            self._local_store[key] = item
            
            # Update local sync vector
            self.update_sync_vector(
                self.node_id, 
                item.tier, 
                str(item.memory_id), 
                item.version
            )
    
    async def get_local(
        self, 
        tier: MemoryTier, 
        memory_id: str
    ) -> Optional[MemorySyncItem]:
        """Get an item from local store"""
        key = (tier, memory_id)
        return self._local_store.get(key)
    
    async def create_batch(
        self,
        items: List[MemorySyncItem],
        target_node: str,
        tier: MemoryTier,
        priority: int = 1,
    ) -> MemorySyncBatch:
        """
        Create a sync batch from items.
        
        Args:
            items: Memory items to include
            target_node: Target node ID ("*" for broadcast)
            tier: Memory tier
            priority: Message priority
        
        Returns:
            MemorySyncBatch ready for transmission
        """
        batch = MemorySyncBatch(
            source_node=self.node_id,
            target_node=target_node,
            tier=tier,
            items=items,
            priority=priority,
        )
        
        self._pending_batches[batch.batch_id] = batch
        self.stats["batches_created"] += 1
        
        logger.info(
            "Sync batch created",
            batch_id=batch.batch_id,
            items=len(items),
            tier=tier.description,
            target=target_node,
        )
        
        return batch
    
    async def create_delta_batch(
        self,
        tier: MemoryTier,
        target_node: str,
        since_timestamp: Optional[datetime] = None,
        remote_vector: Optional[Dict[str, int]] = None,
    ) -> MemorySyncBatch:
        """
        Create a delta batch with only changed items.
        
        Args:
            tier: Memory tier to sync
            target_node: Target node ID
            since_timestamp: Only include items modified after this time
            remote_vector: Remote sync vector for delta computation
        
        Returns:
            MemorySyncBatch with delta items
        """
        items_to_sync = []
        
        async with self._lock:
            # Get local vector for this tier
            local_vector = self.get_sync_vector(self.node_id, tier)
            
            # Compute delta if remote vector provided
            if remote_vector:
                new_ids, updated_ids, _ = self.compute_delta(local_vector, remote_vector)
                ids_to_sync = new_ids | updated_ids
            else:
                ids_to_sync = set(local_vector.keys())
            
            # Collect items
            for memory_id in ids_to_sync:
                key = (tier, memory_id)
                item = self._local_store.get(key)
                if item:
                    # Filter by timestamp if provided
                    if since_timestamp and item.timestamp < since_timestamp:
                        continue
                    items_to_sync.append(item)
        
        return await self.create_batch(items_to_sync, target_node, tier)
    
    async def apply_batch(
        self,
        batch: MemorySyncBatch,
        local_tier_store: Optional[Any] = None,
    ) -> SyncResult:
        """
        Apply a sync batch to local state.
        
        Args:
            batch: Sync batch to apply
            local_tier_store: Optional local storage backend
        
        Returns:
            SyncResult with applied/skipped/conflict counts
        """
        import time
        start_time = time.time()
        
        result = SyncResult(success=True)
        
        # Validate batch
        if not batch.validate():
            result.success = False
            result.errors.append("Batch checksum validation failed")
            return result
        
        async with self._lock:
            for item in batch.items:
                try:
                    # Check for existing item
                    key = (item.tier, str(item.memory_id))
                    existing = self._local_store.get(key)
                    
                    if existing:
                        # Handle conflict
                        if existing.version >= item.version:
                            if existing.content_hash != item.content_hash:
                                # Real conflict
                                resolved = self._resolve_conflict(existing, item)
                                if resolved:
                                    self._local_store[key] = resolved
                                    result.applied_count += 1
                                    self.stats["conflicts_resolved"] += 1
                                else:
                                    result.conflict_count += 1
                            else:
                                result.skipped_count += 1
                            continue
                    
                    # Apply item
                    if item.operation == "delete":
                        if key in self._local_store:
                            del self._local_store[key]
                    else:
                        self._local_store[key] = item
                    
                    # Update sync vector
                    self.update_sync_vector(
                        batch.source_node,
                        item.tier,
                        str(item.memory_id),
                        item.version,
                    )
                    
                    result.applied_count += 1
                    self.stats["items_synced"] += 1
                    
                except Exception as e:
                    result.errors.append(f"Item {item.memory_id}: {str(e)}")
                    logger.error("Failed to apply sync item", 
                               memory_id=str(item.memory_id), error=str(e))
        
        # Update stats
        self.stats["batches_applied"] += 1
        self.stats["bytes_transferred"] += batch.size_bytes
        
        result.latency_ms = (time.time() - start_time) * 1000
        
        logger.info(
            "Sync batch applied",
            batch_id=batch.batch_id,
            applied=result.applied_count,
            skipped=result.skipped_count,
            conflicts=result.conflict_count,
            latency_ms=result.latency_ms,
        )
        
        return result
    
    def _resolve_conflict(
        self,
        local_item: MemorySyncItem,
        remote_item: MemorySyncItem,
    ) -> Optional[MemorySyncItem]:
        """
        Resolve conflict between local and remote items.
        
        Returns:
            Resolved item, or None if conflict cannot be resolved
        """
        if self.conflict_resolution == ConflictResolution.LAST_WRITE_WINS:
            if remote_item.timestamp > local_item.timestamp:
                return remote_item
            return local_item
        
        elif self.conflict_resolution == ConflictResolution.FIRST_WRITE_WINS:
            if remote_item.timestamp < local_item.timestamp:
                return remote_item
            return local_item
        
        elif self.conflict_resolution == ConflictResolution.HIGHEST_VERSION:
            if remote_item.version > local_item.version:
                return remote_item
            return local_item
        
        elif self.conflict_resolution == ConflictResolution.MERGE:
            # Merge metadata, prefer remote embedding if newer
            merged = MemorySyncItem(
                memory_id=local_item.memory_id,
                content_hash=remote_item.content_hash if remote_item.timestamp > local_item.timestamp else local_item.content_hash,
                tier=local_item.tier,
                embedding=remote_item.embedding if remote_item.timestamp > local_item.timestamp else local_item.embedding,
                metadata={**local_item.metadata, **remote_item.metadata},
                timestamp=max(local_item.timestamp, remote_item.timestamp),
                version=max(local_item.version, remote_item.version) + 1,
                operation="upsert",
                embedding_dim=local_item.embedding_dim,
                is_compressed=local_item.is_compressed,
            )
            return merged
        
        # MANUAL - cannot auto-resolve
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sync manager statistics"""
        return {
            **self.stats,
            "node_id": self.node_id,
            "local_items": len(self._local_store),
            "sync_vectors": len(self._sync_vectors),
            "pending_batches": len(self._pending_batches),
            "conflict_resolution": self.conflict_resolution.name,
        }
    
    def get_tier_stats(self, tier: MemoryTier) -> Dict[str, Any]:
        """Get statistics for a specific tier"""
        tier_items = [
            item for (t, _), item in self._local_store.items()
            if t == tier
        ]
        
        total_bytes = sum(len(item.embedding) for item in tier_items)
        
        return {
            "tier": tier.description,
            "item_count": len(tier_items),
            "total_bytes": total_bytes,
            "avg_embedding_size": total_bytes / len(tier_items) if tier_items else 0,
        }
