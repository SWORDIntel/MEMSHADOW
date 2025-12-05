"""
Spoke Memory Adapter

Spoke-side adapter for MEMSHADOW memory synchronization.
Wraps local memory tiers and provides sync capabilities.

Responsibilities:
- Local tier access (L1/L2/L3)
- Delta batch creation for hub sync
- Incoming batch application
- Self-improvement propagation via MEMSHADOW

Based on: HUB_DOCS/DSMIL Brain Federation.md
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import UUID

import numpy as np
import structlog

from .protocol import (
    MessageType,
    Priority,
    MessageFlags,
    MemshadowMessage,
    create_memory_sync_message,
    should_route_p2p,
)
from .sync_manager import (
    MemoryTier,
    MemorySyncBatch,
    MemorySyncItem,
    MemorySyncManager,
    SyncResult,
    SyncVector,
)

logger = structlog.get_logger()


@dataclass
class LocalTierConfig:
    """Configuration for a local memory tier"""
    tier: MemoryTier
    max_items: int = 10000
    embedding_dim: int = 256
    enable_compression: bool = True
    auto_migrate: bool = True
    

class LocalTierStore:
    """
    Simple in-memory store for a single memory tier.
    
    In production, this would interface with:
    - L1: RAMDisk/Redis for hot working memory
    - L2: NVMe vector store for episodic memory
    - L3: Cold storage for semantic memory
    """
    
    def __init__(self, config: LocalTierConfig):
        self.config = config
        self.tier = config.tier
        
        # In-memory storage: memory_id -> MemorySyncItem
        self._store: Dict[str, MemorySyncItem] = {}
        
        # Version tracking
        self._versions: Dict[str, int] = {}
        
        # Access tracking for temperature calculation
        self._access_times: Dict[str, datetime] = {}
        self._access_counts: Dict[str, int] = {}
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    async def store(
        self,
        memory_id: UUID,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemorySyncItem:
        """Store a memory item"""
        async with self._lock:
            mid = str(memory_id)
            version = self._versions.get(mid, 0) + 1
            
            item = MemorySyncItem.from_embedding(
                memory_id=memory_id,
                embedding=embedding,
                tier=self.tier,
                metadata=metadata,
                compress=self.config.enable_compression,
            )
            item.version = version
            
            self._store[mid] = item
            self._versions[mid] = version
            self._access_times[mid] = datetime.utcnow()
            self._access_counts[mid] = 0
            
            return item
    
    async def get(self, memory_id: UUID) -> Optional[MemorySyncItem]:
        """Get a memory item"""
        mid = str(memory_id)
        item = self._store.get(mid)
        
        if item:
            # Update access tracking
            self._access_times[mid] = datetime.utcnow()
            self._access_counts[mid] = self._access_counts.get(mid, 0) + 1
        
        return item
    
    async def delete(self, memory_id: UUID) -> bool:
        """Delete a memory item"""
        async with self._lock:
            mid = str(memory_id)
            if mid in self._store:
                del self._store[mid]
                self._versions.pop(mid, None)
                self._access_times.pop(mid, None)
                self._access_counts.pop(mid, None)
                return True
            return False
    
    async def get_all_ids(self) -> List[str]:
        """Get all memory IDs in this tier"""
        return list(self._store.keys())
    
    async def get_sync_vector(self) -> Dict[str, int]:
        """Get version vector for sync"""
        return dict(self._versions)
    
    async def get_items_since(
        self,
        timestamp: datetime,
    ) -> List[MemorySyncItem]:
        """Get all items modified since timestamp"""
        items = []
        for mid, item in self._store.items():
            if item.timestamp >= timestamp:
                items.append(item)
        return items
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get tier statistics"""
        total_bytes = sum(len(item.embedding) for item in self._store.values())
        
        return {
            "tier": self.tier.name,
            "item_count": len(self._store),
            "total_bytes": total_bytes,
            "max_items": self.config.max_items,
            "utilization": len(self._store) / self.config.max_items if self.config.max_items > 0 else 0,
        }


class SpokeMemoryAdapter:
    """
    Spoke-side adapter for MEMSHADOW memory synchronization.
    
    Wraps local L1/L2/L3 memory tiers and provides:
    - Unified interface for sync operations
    - Delta batch creation for efficient sync
    - Incoming batch application with conflict handling
    - Self-improvement propagation via MEMSHADOW sync
    
    Usage:
        adapter = SpokeMemoryAdapter(node_id="spoke-1")
        await adapter.start()
        
        # Store a memory
        await adapter.store_memory(
            memory_id=uuid4(),
            embedding=np.random.randn(256),
            tier=MemoryTier.L1_WORKING
        )
        
        # Create delta batch for hub
        batch = await adapter.create_delta_batch(
            tier=MemoryTier.L1_WORKING,
            target_node="hub",
            since_timestamp=last_sync_time
        )
        
        # Apply incoming batch from hub
        result = await adapter.apply_sync_batch(batch)
    """
    
    def __init__(
        self,
        node_id: str,
        hub_node_id: str = "hub",
        mesh_send_callback: Optional[Callable] = None,
        sync_interval_seconds: int = 60,
        enable_p2p: bool = True,
    ):
        """
        Initialize Spoke Memory Adapter.
        
        Args:
            node_id: Unique identifier for this spoke
            hub_node_id: Hub node identifier for routing
            mesh_send_callback: Callback to send messages via mesh
            sync_interval_seconds: Interval for automatic sync
            enable_p2p: Whether to allow P2P sync for critical updates
        """
        self.node_id = node_id
        self.hub_node_id = hub_node_id
        self.mesh_send = mesh_send_callback
        self.sync_interval_seconds = sync_interval_seconds
        self.enable_p2p = enable_p2p
        
        # Local tier stores
        self._tiers: Dict[MemoryTier, LocalTierStore] = {
            MemoryTier.L1_WORKING: LocalTierStore(LocalTierConfig(
                tier=MemoryTier.L1_WORKING,
                max_items=10000,
                embedding_dim=256,
                enable_compression=True,
            )),
            MemoryTier.L2_EPISODIC: LocalTierStore(LocalTierConfig(
                tier=MemoryTier.L2_EPISODIC,
                max_items=50000,
                embedding_dim=2048,
                enable_compression=True,
            )),
            MemoryTier.L3_SEMANTIC: LocalTierStore(LocalTierConfig(
                tier=MemoryTier.L3_SEMANTIC,
                max_items=100000,
                embedding_dim=4096,
                enable_compression=True,
            )),
        }
        
        # Sync manager for delta computation
        self._sync_manager = MemorySyncManager(node_id=node_id)
        
        # Track last sync time per tier
        self._last_sync: Dict[MemoryTier, datetime] = {}
        
        # Peer nodes for P2P sync
        self._peers: Set[str] = set()
        
        # Metrics
        self._metrics = {
            "stores": 0,
            "retrievals": 0,
            "batches_sent": 0,
            "batches_received": 0,
            "p2p_syncs": 0,
            "hub_syncs": 0,
            "conflicts_resolved": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
        }
        
        # Background tasks
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None
        
        logger.info(
            "SpokeMemoryAdapter initialized",
            node_id=node_id,
            hub_node_id=hub_node_id,
        )
    
    async def start(self):
        """Start the adapter background tasks"""
        logger.info("Starting SpokeMemoryAdapter", node_id=self.node_id)
        
        self._running = True
        
        # Start background sync task
        self._sync_task = asyncio.create_task(self._auto_sync_loop())
        
        logger.info("SpokeMemoryAdapter started")
    
    async def stop(self):
        """Stop the adapter"""
        logger.info("Stopping SpokeMemoryAdapter", node_id=self.node_id)
        
        self._running = False
        
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        logger.info("SpokeMemoryAdapter stopped")
    
    # ==================== Local Storage Operations ====================
    
    async def store_memory(
        self,
        memory_id: UUID,
        embedding: np.ndarray,
        tier: MemoryTier = MemoryTier.L1_WORKING,
        metadata: Optional[Dict[str, Any]] = None,
        propagate: bool = True,
        priority: Priority = Priority.NORMAL,
    ) -> MemorySyncItem:
        """
        Store a memory in the local tier.
        
        Args:
            memory_id: Unique memory identifier
            embedding: Memory embedding vector
            tier: Target memory tier
            metadata: Optional metadata
            propagate: Whether to propagate to hub
            priority: Sync priority for propagation
        
        Returns:
            Stored MemorySyncItem
        """
        if tier not in self._tiers:
            raise ValueError(f"Unsupported tier: {tier}")
        
        store = self._tiers[tier]
        item = await store.store(memory_id, embedding, metadata)
        
        # Also store in sync manager
        await self._sync_manager.store_local(item)
        
        self._metrics["stores"] += 1
        
        logger.debug(
            "Memory stored locally",
            memory_id=str(memory_id),
            tier=tier.name,
            propagate=propagate,
        )
        
        # Propagate if requested
        if propagate:
            await self._propagate_item(item, priority)
        
        return item
    
    async def get_memory(
        self,
        memory_id: UUID,
        tier: Optional[MemoryTier] = None,
    ) -> Optional[MemorySyncItem]:
        """
        Retrieve a memory from local tiers.
        
        Args:
            memory_id: Memory identifier
            tier: Specific tier to search (searches all if None)
        
        Returns:
            MemorySyncItem if found, None otherwise
        """
        tiers_to_search = [tier] if tier else list(self._tiers.keys())
        
        for t in tiers_to_search:
            if t in self._tiers:
                item = await self._tiers[t].get(memory_id)
                if item:
                    self._metrics["retrievals"] += 1
                    return item
        
        return None
    
    async def delete_memory(
        self,
        memory_id: UUID,
        tier: MemoryTier,
        propagate: bool = True,
    ) -> bool:
        """Delete a memory from local tier"""
        if tier not in self._tiers:
            return False
        
        deleted = await self._tiers[tier].delete(memory_id)
        
        if deleted and propagate:
            # Create delete sync item
            delete_item = MemorySyncItem(
                memory_id=memory_id,
                content_hash="",
                tier=tier,
                embedding=b"",
                operation="delete",
            )
            await self._propagate_item(delete_item, Priority.NORMAL)
        
        return deleted
    
    # ==================== Sync Operations ====================
    
    async def create_delta_batch(
        self,
        tier: MemoryTier,
        target_node: str,
        since_timestamp: Optional[datetime] = None,
        remote_vector: Optional[Dict[str, int]] = None,
    ) -> MemorySyncBatch:
        """
        Create a delta batch with changes since last sync.
        
        Args:
            tier: Memory tier to sync
            target_node: Target node ("hub" or peer ID)
            since_timestamp: Only include changes after this time
            remote_vector: Remote sync vector for delta computation
        
        Returns:
            MemorySyncBatch ready for transmission
        """
        if tier not in self._tiers:
            raise ValueError(f"Unsupported tier: {tier}")
        
        store = self._tiers[tier]
        
        # Get items to sync
        if since_timestamp:
            items = await store.get_items_since(since_timestamp)
        else:
            # Get all items if no timestamp
            items = [
                await store.get(UUID(mid))
                for mid in await store.get_all_ids()
            ]
            items = [i for i in items if i is not None]
        
        # Filter by remote vector if provided
        if remote_vector:
            local_vector = await store.get_sync_vector()
            new_ids, updated_ids, _ = self._sync_manager.compute_delta(
                local_vector, remote_vector
            )
            ids_to_include = new_ids | updated_ids
            items = [i for i in items if str(i.memory_id) in ids_to_include]
        
        # Create batch
        batch = MemorySyncBatch(
            source_node=self.node_id,
            target_node=target_node,
            tier=tier,
            items=items,
            priority=Priority.NORMAL.value,
        )
        
        logger.debug(
            "Delta batch created",
            tier=tier.name,
            target=target_node,
            items=len(items),
        )
        
        return batch
    
    async def apply_sync_batch(
        self,
        batch: MemorySyncBatch,
    ) -> SyncResult:
        """
        Apply a sync batch to local tiers.
        
        Args:
            batch: Sync batch to apply
        
        Returns:
            SyncResult with applied/conflict counts
        """
        start_time = time.time()
        
        if batch.tier not in self._tiers:
            return SyncResult(
                success=False,
                errors=[f"Unsupported tier: {batch.tier}"],
            )
        
        store = self._tiers[batch.tier]
        result = SyncResult(success=True)
        
        # Validate batch
        if not batch.validate():
            result.success = False
            result.errors.append("Batch checksum validation failed")
            return result
        
        for item in batch.items:
            try:
                if item.operation == "delete":
                    await store.delete(item.memory_id)
                    result.applied_count += 1
                else:
                    # Check for conflict
                    existing = await store.get(item.memory_id)
                    if existing and existing.version >= item.version:
                        if existing.content_hash != item.content_hash:
                            # Conflict - use last-write-wins
                            if item.timestamp > existing.timestamp:
                                embedding = item.get_embedding()
                                await store.store(
                                    item.memory_id, embedding, item.metadata
                                )
                                result.applied_count += 1
                            else:
                                result.skipped_count += 1
                            result.conflict_count += 1
                            self._metrics["conflicts_resolved"] += 1
                        else:
                            result.skipped_count += 1
                        continue
                    
                    # Apply item
                    embedding = item.get_embedding()
                    await store.store(item.memory_id, embedding, item.metadata)
                    result.applied_count += 1
                    
            except Exception as e:
                result.errors.append(f"Item {item.memory_id}: {str(e)}")
                logger.error(
                    "Failed to apply sync item",
                    memory_id=str(item.memory_id),
                    error=str(e),
                )
        
        # Update metrics
        self._metrics["batches_received"] += 1
        self._metrics["bytes_received"] += batch.size_bytes
        
        result.latency_ms = (time.time() - start_time) * 1000
        
        # Update last sync time
        self._last_sync[batch.tier] = datetime.utcnow()
        
        logger.info(
            "Sync batch applied",
            source=batch.source_node,
            tier=batch.tier.name,
            applied=result.applied_count,
            skipped=result.skipped_count,
            conflicts=result.conflict_count,
            latency_ms=result.latency_ms,
        )
        
        return result
    
    async def _propagate_item(
        self,
        item: MemorySyncItem,
        priority: Priority,
    ):
        """Propagate a single item to hub (and peers for critical)"""
        batch = MemorySyncBatch(
            source_node=self.node_id,
            target_node=self.hub_node_id,
            tier=item.tier,
            items=[item],
            priority=priority.value,
        )
        
        await self._send_batch(batch, priority)
    
    async def _send_batch(
        self,
        batch: MemorySyncBatch,
        priority: Priority,
    ):
        """Send a sync batch via mesh"""
        if not self.mesh_send:
            logger.debug("No mesh callback configured, skipping send")
            return
        
        try:
            msg = create_memory_sync_message(
                payload=batch.to_bytes(),
                priority=priority,
                batch_count=len(batch.items),
                compressed=True,
            )
            
            # Determine routing
            if should_route_p2p(priority) and self.enable_p2p:
                # P2P: Send to peers directly + hub notification
                for peer_id in self._peers:
                    if peer_id != batch.source_node:
                        await self.mesh_send(peer_id, msg.pack())
                        self._metrics["p2p_syncs"] += 1
                
                # Also notify hub
                await self.mesh_send(self.hub_node_id, msg.pack())
            else:
                # Hub-relayed: Send to hub only
                await self.mesh_send(self.hub_node_id, msg.pack())
                self._metrics["hub_syncs"] += 1
            
            self._metrics["batches_sent"] += 1
            self._metrics["bytes_sent"] += batch.size_bytes
            
            logger.debug(
                "Batch sent",
                target=batch.target_node,
                tier=batch.tier.name,
                items=len(batch.items),
                priority=priority.name,
            )
            
        except Exception as e:
            logger.error(
                "Failed to send batch",
                target=batch.target_node,
                error=str(e),
            )
    
    # ==================== Message Handling ====================
    
    async def handle_sync_request(
        self,
        data: bytes,
        peer_id: str,
    ):
        """
        Handle incoming MEMORY_SYNC message from hub or peer.
        
        This is the main entry point for incoming sync requests.
        """
        try:
            message = MemshadowMessage.unpack(data)
            
            if message.header.msg_type != MessageType.MEMORY_SYNC:
                logger.warning(
                    "Unexpected message type",
                    msg_type=message.header.msg_type.name,
                    peer=peer_id,
                )
                return
            
            # Parse batch
            batch = MemorySyncBatch.from_bytes(message.payload)
            
            # Check if this is for us
            if batch.target_node != "*" and batch.target_node != self.node_id:
                logger.debug(
                    "Batch not for this node, ignoring",
                    target=batch.target_node,
                    our_id=self.node_id,
                )
                return
            
            # Apply batch
            result = await self.apply_sync_batch(batch)
            
            # Send ACK if required
            if message.header.flags & MessageFlags.REQUIRES_ACK:
                await self._send_ack(peer_id, batch.batch_id, result)
            
            logger.info(
                "MEMORY_SYNC handled",
                peer=peer_id,
                batch_id=batch.batch_id,
                applied=result.applied_count,
            )
            
        except Exception as e:
            logger.error(
                "Failed to handle MEMORY_SYNC",
                peer=peer_id,
                error=str(e),
            )
    
    async def _send_ack(
        self,
        node_id: str,
        batch_id: str,
        result: SyncResult,
    ):
        """Send acknowledgment for a sync batch"""
        if not self.mesh_send:
            return
        
        ack_payload = json.dumps({
            "batch_id": batch_id,
            "result": result.to_dict(),
        }).encode()
        
        msg = MemshadowMessage.create(
            msg_type=MessageType.ACK,
            payload=ack_payload,
            priority=Priority.HIGH,
        )
        
        try:
            await self.mesh_send(node_id, msg.pack())
        except Exception as e:
            logger.error("Failed to send ACK", node_id=node_id, error=str(e))
    
    # ==================== Self-Improvement Propagation ====================
    
    async def propagate_improvement(
        self,
        improvement_type: str,
        data: Dict[str, Any],
        priority: Priority = Priority.HIGH,
    ):
        """
        Propagate self-improvement updates via MEMSHADOW sync.
        
        Used for:
        - Pattern cache updates
        - Model weight deltas
        - Learned correlations
        
        Args:
            improvement_type: Type of improvement ("pattern", "weights", "config")
            data: Improvement data
            priority: Propagation priority
        """
        # Create a synthetic memory item for the improvement
        import hashlib
        from uuid import uuid4
        
        # Serialize improvement data
        improvement_bytes = json.dumps({
            "type": improvement_type,
            "data": data,
            "source": self.node_id,
            "timestamp": datetime.utcnow().isoformat(),
        }).encode()
        
        # Create embedding from improvement (hash-based for dedup)
        hash_bytes = hashlib.sha256(improvement_bytes).digest()
        embedding = np.frombuffer(hash_bytes, dtype=np.float32)
        # Pad to min dimension
        if len(embedding) < 8:
            embedding = np.pad(embedding, (0, 8 - len(embedding)))
        
        item = MemorySyncItem(
            memory_id=uuid4(),
            content_hash=hashlib.sha256(improvement_bytes).hexdigest(),
            tier=MemoryTier.L1_WORKING,  # Improvements are hot data
            embedding=improvement_bytes,  # Store raw data as "embedding"
            metadata={
                "is_improvement": True,
                "improvement_type": improvement_type,
            },
            is_compressed=False,
        )
        
        # Propagate with requested priority
        await self._propagate_item(item, priority)
        
        logger.info(
            "Improvement propagated via MEMSHADOW",
            improvement_type=improvement_type,
            priority=priority.name,
        )
    
    # ==================== Peer Management ====================
    
    def register_peer(self, peer_id: str):
        """Register a peer node for P2P sync"""
        self._peers.add(peer_id)
        logger.debug("Peer registered", peer_id=peer_id)
    
    def deregister_peer(self, peer_id: str):
        """Deregister a peer node"""
        self._peers.discard(peer_id)
        logger.debug("Peer deregistered", peer_id=peer_id)
    
    # ==================== Background Tasks ====================
    
    async def _auto_sync_loop(self):
        """Background loop for automatic hub sync"""
        while self._running:
            try:
                await asyncio.sleep(self.sync_interval_seconds)
                
                # Sync each tier
                for tier in self._tiers.keys():
                    since = self._last_sync.get(tier)
                    batch = await self.create_delta_batch(
                        tier=tier,
                        target_node=self.hub_node_id,
                        since_timestamp=since,
                    )
                    
                    if batch.items:
                        await self._send_batch(batch, Priority.NORMAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Auto-sync loop error", error=str(e))
    
    # ==================== Statistics ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics"""
        tier_stats = {}
        for tier, store in self._tiers.items():
            # Run sync to get stats
            tier_stats[tier.name] = {
                "item_count": len(store._store),
                "last_sync": self._last_sync.get(tier, datetime.min).isoformat() if self._last_sync.get(tier) else None,
            }
        
        return {
            "node_id": self.node_id,
            "hub_node_id": self.hub_node_id,
            "metrics": self._metrics,
            "tiers": tier_stats,
            "peer_count": len(self._peers),
            "sync_manager_stats": self._sync_manager.get_stats(),
        }
    
    async def get_tier_stats(self, tier: MemoryTier) -> Dict[str, Any]:
        """Get statistics for a specific tier"""
        if tier not in self._tiers:
            return {}
        
        return await self._tiers[tier].get_stats()
