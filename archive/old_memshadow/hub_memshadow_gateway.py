"""
Hub MEMSHADOW Gateway

Hub-side gateway for coordinating MEMSHADOW memory synchronization
across all nodes in the DSMIL Brain Federation.

Responsibilities:
- Node registration with memory tier capabilities
- Sync scheduling (priority-aware)
- Batch routing (hub-relay vs P2P)
- Cross-node memory coordination
- Observability and metrics

Based on: HUB_DOCS/DSMIL Brain Federation.md
"""

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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


class NodeSyncState(Enum):
    """State of a node's sync status"""
    UNKNOWN = "unknown"
    SYNCED = "synced"
    SYNCING = "syncing"
    STALE = "stale"
    OFFLINE = "offline"


@dataclass
class NodeMemoryCapabilities:
    """Memory tier capabilities for a registered node"""
    node_id: str
    supported_tiers: Set[MemoryTier] = field(default_factory=set)
    max_batch_size: int = 100
    supports_compression: bool = True
    supports_p2p: bool = True
    registered_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "supported_tiers": [t.value for t in self.supported_tiers],
            "max_batch_size": self.max_batch_size,
            "supports_compression": self.supports_compression,
            "supports_p2p": self.supports_p2p,
            "registered_at": self.registered_at.isoformat(),
        }


@dataclass
class NodeSyncInfo:
    """Sync state tracking for a node"""
    node_id: str
    state: NodeSyncState = NodeSyncState.UNKNOWN
    last_sync: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    sync_vectors: Dict[MemoryTier, SyncVector] = field(default_factory=dict)
    pending_batches: List[str] = field(default_factory=list)
    
    # Metrics
    total_syncs: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    conflicts_resolved: int = 0
    bytes_transferred: int = 0
    avg_latency_ms: float = 0.0
    
    def update_latency(self, latency_ms: float):
        """Update running average latency"""
        if self.total_syncs == 0:
            self.avg_latency_ms = latency_ms
        else:
            # Exponential moving average
            self.avg_latency_ms = 0.9 * self.avg_latency_ms + 0.1 * latency_ms
    
    def is_healthy(self) -> bool:
        """Check if node is in healthy sync state"""
        if self.state in (NodeSyncState.OFFLINE, NodeSyncState.UNKNOWN):
            return False
        if self.last_heartbeat:
            # Consider stale if no heartbeat in 5 minutes
            if datetime.utcnow() - self.last_heartbeat > timedelta(minutes=5):
                return False
        return True


@dataclass
class SyncScheduleEntry:
    """Entry in the sync schedule queue"""
    node_id: str
    tier: MemoryTier
    priority: Priority
    scheduled_at: datetime = field(default_factory=datetime.utcnow)
    batch_id: Optional[str] = None
    
    def __lt__(self, other: "SyncScheduleEntry"):
        # Higher priority first, then earlier scheduled
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.scheduled_at < other.scheduled_at


class HubMemshadowGateway:
    """
    Hub-side gateway for MEMSHADOW memory synchronization.
    
    Acts as the canonical coordination point for cross-node memory sync:
    - Registers nodes and their memory capabilities
    - Manages sync vectors per (node, tier)
    - Routes MEMORY_SYNC messages based on priority
    - Schedules and executes sync operations
    
    Usage:
        gateway = HubMemshadowGateway(hub_id="central-hub")
        await gateway.start()
        
        # Register a spoke node
        gateway.register_memory_node(
            "spoke-1",
            capabilities=NodeMemoryCapabilities(
                node_id="spoke-1",
                supported_tiers={MemoryTier.L1_WORKING, MemoryTier.L2_EPISODIC}
            )
        )
        
        # Schedule sync
        await gateway.schedule_sync("spoke-1", MemoryTier.L1_WORKING, Priority.NORMAL)
    """
    
    def __init__(
        self,
        hub_id: str = "hub",
        mesh_send_callback: Optional[Callable] = None,
        sync_interval_seconds: int = 30,
        max_batch_size: int = 100,
    ):
        """
        Initialize Hub MEMSHADOW Gateway.
        
        Args:
            hub_id: Unique identifier for this hub
            mesh_send_callback: Callback to send messages via mesh network
            sync_interval_seconds: Interval for scheduled sync operations
            max_batch_size: Maximum items per sync batch
        """
        self.hub_id = hub_id
        self.mesh_send = mesh_send_callback
        self.sync_interval_seconds = sync_interval_seconds
        self.max_batch_size = max_batch_size
        
        # Node registry: node_id -> capabilities
        self._nodes: Dict[str, NodeMemoryCapabilities] = {}
        
        # Sync state per node
        self._sync_info: Dict[str, NodeSyncInfo] = {}
        
        # Hub's own sync manager (for aggregated state)
        self._sync_manager = MemorySyncManager(node_id=hub_id)
        
        # Sync schedule queue (priority queue simulation)
        self._sync_queue: List[SyncScheduleEntry] = []
        self._queue_lock = asyncio.Lock()
        
        # Metrics
        self._metrics = {
            "total_nodes": 0,
            "syncs_scheduled": 0,
            "syncs_completed": 0,
            "syncs_failed": 0,
            "batches_routed": 0,
            "p2p_syncs": 0,
            "hub_relayed_syncs": 0,
            "conflicts_total": 0,
            "bytes_total": 0,
        }
        
        # Background tasks
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("HubMemshadowGateway initialized", hub_id=hub_id)
    
    async def start(self):
        """Start the gateway background tasks"""
        logger.info("Starting HubMemshadowGateway", hub_id=self.hub_id)
        
        self._running = True
        
        # Start background sync processor
        self._sync_task = asyncio.create_task(self._sync_loop())
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("HubMemshadowGateway started")
    
    async def stop(self):
        """Stop the gateway"""
        logger.info("Stopping HubMemshadowGateway", hub_id=self.hub_id)
        
        self._running = False
        
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("HubMemshadowGateway stopped")
    
    # ==================== Node Registration ====================
    
    def register_memory_node(
        self,
        node_id: str,
        capabilities: Optional[NodeMemoryCapabilities] = None,
    ):
        """
        Register a node with its memory tier capabilities.
        
        Args:
            node_id: Unique node identifier
            capabilities: Memory capabilities (default all tiers)
        """
        if capabilities is None:
            capabilities = NodeMemoryCapabilities(
                node_id=node_id,
                supported_tiers={MemoryTier.L1_WORKING, MemoryTier.L2_EPISODIC, MemoryTier.L3_SEMANTIC},
            )
        
        self._nodes[node_id] = capabilities
        
        # Initialize sync info
        self._sync_info[node_id] = NodeSyncInfo(
            node_id=node_id,
            sync_vectors={tier: SyncVector() for tier in capabilities.supported_tiers},
        )
        
        self._metrics["total_nodes"] = len(self._nodes)
        
        logger.info(
            "Memory node registered",
            node_id=node_id,
            tiers=[t.name for t in capabilities.supported_tiers],
        )
    
    def deregister_memory_node(self, node_id: str):
        """Deregister a node"""
        if node_id in self._nodes:
            del self._nodes[node_id]
            
            if node_id in self._sync_info:
                del self._sync_info[node_id]
            
            self._metrics["total_nodes"] = len(self._nodes)
            
            logger.info("Memory node deregistered", node_id=node_id)
    
    def update_node_heartbeat(self, node_id: str):
        """Update heartbeat timestamp for a node"""
        if node_id in self._sync_info:
            self._sync_info[node_id].last_heartbeat = datetime.utcnow()
            if self._sync_info[node_id].state == NodeSyncState.OFFLINE:
                self._sync_info[node_id].state = NodeSyncState.STALE
    
    # ==================== Sync Scheduling ====================
    
    async def schedule_sync(
        self,
        node_id: str,
        tier: MemoryTier,
        priority: Priority = Priority.NORMAL,
    ) -> str:
        """
        Schedule a sync operation for a node/tier.
        
        Args:
            node_id: Target node
            tier: Memory tier to sync
            priority: Sync priority
        
        Returns:
            Schedule entry ID
        """
        if node_id not in self._nodes:
            raise ValueError(f"Unknown node: {node_id}")
        
        capabilities = self._nodes[node_id]
        if tier not in capabilities.supported_tiers:
            raise ValueError(f"Node {node_id} does not support tier {tier.name}")
        
        entry = SyncScheduleEntry(
            node_id=node_id,
            tier=tier,
            priority=priority,
        )
        
        async with self._queue_lock:
            self._sync_queue.append(entry)
            self._sync_queue.sort()
        
        self._metrics["syncs_scheduled"] += 1
        
        logger.debug(
            "Sync scheduled",
            node_id=node_id,
            tier=tier.name,
            priority=priority.name,
        )
        
        return f"{node_id}:{tier.name}:{entry.scheduled_at.isoformat()}"
    
    async def schedule_broadcast_sync(
        self,
        tier: MemoryTier,
        priority: Priority = Priority.NORMAL,
    ) -> int:
        """
        Schedule sync for all nodes supporting a tier.
        
        Returns:
            Number of syncs scheduled
        """
        count = 0
        for node_id, caps in self._nodes.items():
            if tier in caps.supported_tiers:
                await self.schedule_sync(node_id, tier, priority)
                count += 1
        
        logger.info(
            "Broadcast sync scheduled",
            tier=tier.name,
            priority=priority.name,
            nodes=count,
        )
        
        return count
    
    # ==================== Batch Handling ====================
    
    async def apply_remote_batch(
        self,
        node_id: str,
        batch: MemorySyncBatch,
    ) -> SyncResult:
        """
        Apply a sync batch received from a remote node.
        
        Args:
            node_id: Source node ID
            batch: Sync batch to apply
        
        Returns:
            SyncResult with applied/conflict counts
        """
        start_time = time.time()
        
        if node_id not in self._sync_info:
            # Auto-register unknown node
            self.register_memory_node(node_id)
        
        sync_info = self._sync_info[node_id]
        sync_info.state = NodeSyncState.SYNCING
        
        # Apply to hub's sync manager
        result = await self._sync_manager.apply_batch(batch)
        
        # Update sync state
        sync_info.total_syncs += 1
        if result.success:
            sync_info.successful_syncs += 1
            sync_info.state = NodeSyncState.SYNCED
        else:
            sync_info.failed_syncs += 1
            sync_info.state = NodeSyncState.STALE
        
        sync_info.conflicts_resolved += result.conflict_count
        sync_info.bytes_transferred += batch.size_bytes
        sync_info.last_sync = datetime.utcnow()
        sync_info.update_latency(result.latency_ms)
        
        # Update sync vector
        for item in batch.items:
            if batch.tier in sync_info.sync_vectors:
                sync_info.sync_vectors[batch.tier].update(
                    str(item.memory_id), item.version
                )
        
        # Update metrics
        self._metrics["batches_routed"] += 1
        self._metrics["conflicts_total"] += result.conflict_count
        self._metrics["bytes_total"] += batch.size_bytes
        
        logger.info(
            "Remote batch applied",
            source_node=node_id,
            batch_id=batch.batch_id,
            items=len(batch.items),
            applied=result.applied_count,
            conflicts=result.conflict_count,
            latency_ms=result.latency_ms,
        )
        
        # Route to other nodes if broadcast
        if batch.target_node == "*":
            await self._route_to_other_nodes(node_id, batch)
        
        return result
    
    async def _route_to_other_nodes(
        self,
        source_node: str,
        batch: MemorySyncBatch,
    ):
        """Route a batch to other nodes (hub relay)"""
        for node_id, caps in self._nodes.items():
            if node_id == source_node:
                continue
            
            if batch.tier not in caps.supported_tiers:
                continue
            
            # Check if node is healthy
            sync_info = self._sync_info.get(node_id)
            if not sync_info or not sync_info.is_healthy():
                continue
            
            # Send via mesh
            if self.mesh_send:
                try:
                    msg = create_memory_sync_message(
                        payload=batch.to_bytes(),
                        priority=Priority(batch.priority),
                        batch_count=len(batch.items),
                        compressed=batch.flags & MessageFlags.COMPRESSED,
                    )
                    await self.mesh_send(node_id, msg.pack())
                    self._metrics["hub_relayed_syncs"] += 1
                except Exception as e:
                    logger.error(
                        "Failed to route batch",
                        target_node=node_id,
                        error=str(e),
                    )
    
    # ==================== Message Handling ====================
    
    async def handle_memory_sync(
        self,
        data: bytes,
        peer_id: str,
    ):
        """
        Handle incoming MEMORY_SYNC message.
        
        This is the main entry point for MEMSHADOW sync messages.
        """
        try:
            # Parse message
            message = MemshadowMessage.unpack(data)
            
            if message.header.msg_type != MessageType.MEMORY_SYNC:
                logger.warning(
                    "Unexpected message type in memory sync handler",
                    msg_type=message.header.msg_type.name,
                    peer=peer_id,
                )
                return
            
            # Parse batch from payload
            batch = MemorySyncBatch.from_bytes(message.payload)
            
            # Determine routing
            priority = message.header.priority
            
            if should_route_p2p(priority):
                # P2P message - hub still processes for state tracking
                self._metrics["p2p_syncs"] += 1
                logger.debug(
                    "P2P sync received (hub notification)",
                    batch_id=batch.batch_id,
                    priority=priority.name,
                )
            
            # Apply batch
            result = await self.apply_remote_batch(peer_id, batch)
            
            # Send ACK if required
            if message.header.flags & MessageFlags.REQUIRES_ACK:
                await self._send_ack(peer_id, batch.batch_id, result)
            
            logger.info(
                "MEMORY_SYNC handled",
                peer=peer_id,
                batch_id=batch.batch_id,
                priority=priority.name,
                applied=result.applied_count,
            )
            
        except Exception as e:
            logger.error(
                "Failed to handle MEMORY_SYNC",
                peer=peer_id,
                error=str(e),
            )
            self._metrics["syncs_failed"] += 1
    
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
    
    # ==================== Background Tasks ====================
    
    async def _sync_loop(self):
        """Background loop for processing scheduled syncs"""
        while self._running:
            try:
                await asyncio.sleep(self.sync_interval_seconds)
                
                # Process scheduled syncs
                async with self._queue_lock:
                    # Process up to 10 entries per iteration
                    to_process = self._sync_queue[:10]
                    self._sync_queue = self._sync_queue[10:]
                
                for entry in to_process:
                    await self._execute_scheduled_sync(entry)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Sync loop error", error=str(e))
    
    async def _execute_scheduled_sync(self, entry: SyncScheduleEntry):
        """Execute a scheduled sync operation"""
        try:
            sync_info = self._sync_info.get(entry.node_id)
            if not sync_info or not sync_info.is_healthy():
                logger.debug(
                    "Skipping sync for unhealthy node",
                    node_id=entry.node_id,
                )
                return
            
            # Create delta batch from hub state to node
            remote_vector = sync_info.sync_vectors.get(entry.tier)
            if remote_vector:
                batch = await self._sync_manager.create_delta_batch(
                    tier=entry.tier,
                    target_node=entry.node_id,
                    remote_vector=remote_vector.get_all(),
                )
                
                if batch.items:
                    # Send to node
                    if self.mesh_send:
                        msg = create_memory_sync_message(
                            payload=batch.to_bytes(),
                            priority=entry.priority,
                            batch_count=len(batch.items),
                        )
                        await self.mesh_send(entry.node_id, msg.pack())
                        
                        self._metrics["syncs_completed"] += 1
                        logger.debug(
                            "Scheduled sync executed",
                            node_id=entry.node_id,
                            tier=entry.tier.name,
                            items=len(batch.items),
                        )
            
        except Exception as e:
            logger.error(
                "Failed to execute scheduled sync",
                node_id=entry.node_id,
                tier=entry.tier.name,
                error=str(e),
            )
            self._metrics["syncs_failed"] += 1
    
    async def _cleanup_loop(self):
        """Background loop for cleanup and health checks"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Every minute
                
                # Check node health
                for node_id, sync_info in self._sync_info.items():
                    if sync_info.last_heartbeat:
                        age = datetime.utcnow() - sync_info.last_heartbeat
                        if age > timedelta(minutes=5):
                            sync_info.state = NodeSyncState.OFFLINE
                            logger.info(
                                "Node marked offline",
                                node_id=node_id,
                                last_heartbeat=sync_info.last_heartbeat.isoformat(),
                            )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup loop error", error=str(e))
    
    # ==================== Statistics ====================
    
    def get_hub_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive hub memory statistics.
        
        Returns:
            Dictionary with per-node/per-tier stats
        """
        node_stats = {}
        for node_id, sync_info in self._sync_info.items():
            caps = self._nodes.get(node_id)
            node_stats[node_id] = {
                "state": sync_info.state.value,
                "is_healthy": sync_info.is_healthy(),
                "last_sync": sync_info.last_sync.isoformat() if sync_info.last_sync else None,
                "last_heartbeat": sync_info.last_heartbeat.isoformat() if sync_info.last_heartbeat else None,
                "total_syncs": sync_info.total_syncs,
                "successful_syncs": sync_info.successful_syncs,
                "failed_syncs": sync_info.failed_syncs,
                "conflicts_resolved": sync_info.conflicts_resolved,
                "bytes_transferred": sync_info.bytes_transferred,
                "avg_latency_ms": sync_info.avg_latency_ms,
                "supported_tiers": [t.name for t in caps.supported_tiers] if caps else [],
                "tier_sync_vectors": {
                    tier.name: len(vec.get_all())
                    for tier, vec in sync_info.sync_vectors.items()
                },
            }
        
        # Calculate ratios
        total_syncs = self._metrics["p2p_syncs"] + self._metrics["hub_relayed_syncs"]
        p2p_ratio = self._metrics["p2p_syncs"] / total_syncs if total_syncs > 0 else 0
        
        return {
            "hub_id": self.hub_id,
            "metrics": {
                **self._metrics,
                "p2p_ratio": p2p_ratio,
                "sync_queue_size": len(self._sync_queue),
            },
            "nodes": node_stats,
            "sync_manager_stats": self._sync_manager.get_stats(),
            "tier_stats": {
                tier.name: self._sync_manager.get_tier_stats(tier)
                for tier in MemoryTier
            },
        }
    
    def get_node_sync_status(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get sync status for a specific node"""
        if node_id not in self._sync_info:
            return None
        
        sync_info = self._sync_info[node_id]
        return {
            "node_id": node_id,
            "state": sync_info.state.value,
            "is_healthy": sync_info.is_healthy(),
            "last_sync": sync_info.last_sync.isoformat() if sync_info.last_sync else None,
            "pending_batches": len(sync_info.pending_batches),
            "metrics": {
                "total_syncs": sync_info.total_syncs,
                "successful_syncs": sync_info.successful_syncs,
                "failed_syncs": sync_info.failed_syncs,
                "avg_latency_ms": sync_info.avg_latency_ms,
            },
        }
