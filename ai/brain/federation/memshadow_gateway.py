#!/usr/bin/env python3
"""
MEMSHADOW Gateway for DSMIL Brain Federation

Hub-side gateway for coordinating MEMSHADOW memory synchronization across all nodes.

Features:
- Node registration with memory tier capabilities
- Sync scheduling (priority-aware queue)
- Batch routing (hub-relay vs P2P based on priority)
- Per-node/per-tier sync vectors
- Metrics and observability

This module integrates with HubOrchestrator to provide MEMSHADOW protocol support.
"""

import asyncio
import logging
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from datetime import datetime, timezone
from collections import defaultdict
import sys
from pathlib import Path

from config.memshadow_config import get_memshadow_config

logger = logging.getLogger(__name__)
_MEMSHADOW_CONFIG = get_memshadow_config()
_MEMSHADOW_METRICS = get_memshadow_metrics_registry()

# Import MEMSHADOW protocol
try:
    protocol_path = Path(__file__).parent.parent.parent.parent / "libs" / "memshadow-protocol" / "python"
    if protocol_path.exists() and str(protocol_path) not in sys.path:
        sys.path.insert(0, str(protocol_path))
    
    from dsmil_protocol import (
        MemshadowHeader, MemshadowMessage, MessageType, Priority, MessageFlags,
        HEADER_SIZE, should_route_p2p
    )
    PROTOCOL_AVAILABLE = True
except ImportError:
    PROTOCOL_AVAILABLE = False
    logger.warning("MEMSHADOW protocol not available")

# Import memory sync protocol
try:
    from ..memory.memory_sync_protocol import (
        MemorySyncItem, MemorySyncBatch, MemorySyncManager,
        MemoryTier, SyncOperation, SyncPriority, SyncFlags
    )
    SYNC_PROTOCOL_AVAILABLE = True
except ImportError:
    try:
        # Direct import fallback
        sync_path = Path(__file__).parent.parent / "memory"
        if str(sync_path) not in sys.path:
            sys.path.insert(0, str(sync_path))
        from memory_sync_protocol import (
            MemorySyncItem, MemorySyncBatch, MemorySyncManager,
            MemoryTier, SyncOperation, SyncPriority, SyncFlags
        )
        SYNC_PROTOCOL_AVAILABLE = True
    except ImportError:
        SYNC_PROTOCOL_AVAILABLE = False
        logger.warning("Memory sync protocol not available")

from ..metrics.memshadow_metrics import get_memshadow_metrics_registry


@dataclass
class NodeMemoryCapabilities:
    """Memory capabilities of a registered node"""
    node_id: str
    tiers: Set[MemoryTier] = field(default_factory=set)
    storage_bytes: Dict[MemoryTier, int] = field(default_factory=dict)
    sync_enabled: bool = True
    last_sync: Optional[datetime] = None
    pending_items: int = 0


@dataclass
class SyncScheduleEntry:
    """Entry in the sync schedule queue"""
    node_id: str
    tier: MemoryTier
    priority: Priority
    scheduled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = "scheduled"


class HubMemshadowGateway:
    """
    Hub-side MEMSHADOW Gateway
    
    Coordinates memory synchronization across all registered nodes.
    Handles routing decisions based on message priority.
    
    Usage:
        gateway = HubMemshadowGateway(hub_id="central-hub")
        
        # Register node with memory capabilities
        gateway.register_node("node-1", endpoint, {
            MemoryTier.WORKING: 1024*1024,
            MemoryTier.EPISODIC: 10*1024*1024,
        })
        
        # Schedule sync
        gateway.schedule_sync("node-1", MemoryTier.WORKING, Priority.HIGH)
        
        # Route incoming batch
        await gateway.route_batch(batch)
    """
    
    def __init__(self, hub_id: str, mesh_send: Optional[Callable] = None):
        """
        Initialize hub gateway.
        
        Args:
            hub_id: Unique hub identifier
            mesh_send: Function to send messages via mesh (node_id, data)
        """
        self.hub_id = hub_id
        self.mesh_send = mesh_send
        self._config = _MEMSHADOW_CONFIG
        self._metrics = _MEMSHADOW_METRICS
        
        # Node registry
        self._nodes: Dict[str, NodeMemoryCapabilities] = {}
        
        # Sync vectors per (node, tier)
        self._sync_vectors: Dict[Tuple[str, MemoryTier], Dict[str, int]] = defaultdict(dict)
        
        # Sync schedule queue
        self._sync_queue: List[SyncScheduleEntry] = []
        
        # Pending batches awaiting ACK
        self._pending_batches: Dict[str, Tuple[MemorySyncBatch, datetime]] = {}
        
        # Metrics
        self._metrics = {
            "batches_routed": 0,
            "p2p_routes": 0,
            "hub_routes": 0,
            "conflicts_resolved": 0,
            "bytes_synced": 0,
            "nodes_synced": 0,
        }
        
        # Sync manager for hub's own operations
        if SYNC_PROTOCOL_AVAILABLE:
            self._sync_manager = MemorySyncManager(hub_id)
        else:
            self._sync_manager = None
        
        logger.info(f"HubMemshadowGateway initialized: {hub_id}")
    
    def register_node(self, node_id: str, endpoint: str,
                     memory_tiers: Dict[MemoryTier, int],
                     capabilities: Optional[Dict] = None) -> NodeMemoryCapabilities:
        """
        Register a node with its memory tier capabilities.
        
        Args:
            node_id: Unique node identifier
            endpoint: Network endpoint
            memory_tiers: Dict of tier -> storage bytes
            capabilities: Additional capabilities
            
        Returns:
            NodeMemoryCapabilities for the registered node
        """
        node_caps = NodeMemoryCapabilities(
            node_id=node_id,
            tiers=set(memory_tiers.keys()),
            storage_bytes=memory_tiers,
            sync_enabled=True,
        )
        
        self._nodes[node_id] = node_caps
        
        # Initialize sync vectors for each tier
        for tier in memory_tiers.keys():
            self._sync_vectors[(node_id, tier)] = {}
        
        logger.info(f"Registered node {node_id} with tiers: {[t.name for t in memory_tiers.keys()]}")
        return node_caps

    def register_node_memshadow_caps(self, node_id: str, tiers: Dict[MemoryTier, int], endpoint: str = "", capabilities: Optional[Dict] = None) -> NodeMemoryCapabilities:
        """
        Public wrapper matching spec terminology for registering MEMSHADOW capabilities.
        """
        return self.register_node(node_id, endpoint, tiers, capabilities)
    
    def unregister_node(self, node_id: str):
        """Remove a node from the registry"""
        if node_id in self._nodes:
            del self._nodes[node_id]
            # Clean up sync vectors
            keys_to_remove = [k for k in self._sync_vectors if k[0] == node_id]
            for key in keys_to_remove:
                del self._sync_vectors[key]
            logger.info(f"Unregistered node: {node_id}")
    
    def get_sync_vector(self, node_id: str, tier: MemoryTier) -> Dict[str, int]:
        """Get sync vector for a node/tier combination"""
        return dict(self._sync_vectors.get((node_id, tier), {}))
    
    def update_sync_vector(self, node_id: str, tier: MemoryTier,
                          source_node: str, timestamp: int):
        """Update sync vector after successful sync"""
        self._sync_vectors[(node_id, tier)][source_node] = timestamp
    
    def schedule_sync(self, node_id: str, tier: MemoryTier,
                     priority: Priority = Priority.NORMAL,
                     reason: str = "scheduled") -> bool:
        """
        Schedule a sync operation for a node/tier.
        
        Args:
            node_id: Target node
            tier: Memory tier to sync
            priority: Sync priority
            reason: Reason for sync
            
        Returns:
            True if scheduled successfully
        """
        if node_id not in self._nodes:
            logger.warning(f"Cannot schedule sync for unknown node: {node_id}")
            return False
        
        if tier not in self._nodes[node_id].tiers:
            logger.warning(f"Node {node_id} doesn't have tier {tier.name}")
            return False
        
        entry = SyncScheduleEntry(
            node_id=node_id,
            tier=tier,
            priority=priority,
            reason=reason,
        )
        
        # Insert by priority (higher priority first)
        insert_idx = 0
        for i, e in enumerate(self._sync_queue):
            if priority.value > e.priority.value:
                insert_idx = i
                break
            insert_idx = i + 1
        
        self._sync_queue.insert(insert_idx, entry)
        logger.debug(f"Scheduled sync: {node_id}/{tier.name} priority={priority.name}")
        return True
    
    def get_pending_syncs(self) -> List[SyncScheduleEntry]:
        """Get pending sync operations"""
        return list(self._sync_queue)
    
    async def route_batch(self, batch: MemorySyncBatch,
                         priority: Optional[Priority] = None) -> Dict[str, bool]:
        """
        Route a sync batch to appropriate destinations.
        
        Routing rules:
        - CRITICAL/EMERGENCY: Direct P2P + hub notification
        - HIGH: Hub-relayed with priority queue
        - NORMAL/LOW: Standard hub routing
        
        Args:
            batch: The sync batch to route
            priority: Override priority (otherwise uses batch default)
            
        Returns:
            Dict of node_id -> success status
        """
        if priority is None:
            priority = Priority.NORMAL  # Default
        
        results: Dict[str, bool] = {}
        self._metrics["batches_routed"] += 1
        
        # Determine routing mode
        allow_p2p = self._config.enable_p2p_for_critical
        use_p2p = allow_p2p and (should_route_p2p(priority) if PROTOCOL_AVAILABLE else priority.value >= Priority.CRITICAL.value)
        
        # Pack batch if needed
        batch_data = batch.pack()
        self._metrics["bytes_synced"] += len(batch_data)
        
        if use_p2p:
            # P2P routing: send directly to target + notify hub
            self._metrics["p2p_routes"] += 1
            
            target = batch.target_node
            if target and target in self._nodes:
                try:
                    send_start = time.time()
                    if self.mesh_send:
                        await self.mesh_send(target, batch_data)
                    results[target] = True
                    logger.debug(f"P2P routed batch to {target}")
                    _MEMSHADOW_METRICS.increment("memshadow_batches_sent")
                    _MEMSHADOW_METRICS.observe_latency((time.time() - send_start) * 1000)
                except Exception as e:
                    logger.error(f"P2P route failed to {target}: {e}")
                    results[target] = False
            
            # Also store locally for consistency
            self._pending_batches[batch.batch_id] = (batch, datetime.now(timezone.utc))
            
        else:
            # Hub routing: send to all relevant nodes
            self._metrics["hub_routes"] += 1
            
            target_nodes = self._get_target_nodes(batch)
            
            for node_id in target_nodes:
                try:
                    send_start = time.time()
                    if self.mesh_send:
                        await self.mesh_send(node_id, batch_data)
                    results[node_id] = True
                    logger.debug(f"Hub routed batch to {node_id}")
                    _MEMSHADOW_METRICS.increment("memshadow_batches_sent")
                    _MEMSHADOW_METRICS.observe_latency((time.time() - send_start) * 1000)
                except Exception as e:
                    logger.error(f"Hub route failed to {node_id}: {e}")
                    results[node_id] = False
        
        self._metrics["nodes_synced"] += sum(1 for v in results.values() if v)
        return results
    
    def _get_target_nodes(self, batch: MemorySyncBatch) -> List[str]:
        """Determine target nodes for a batch"""
        if batch.target_node and batch.target_node != "*":
            return [batch.target_node] if batch.target_node in self._nodes else []
        
        # Broadcast to all nodes with this tier
        return [
            node_id for node_id, caps in self._nodes.items()
            if batch.tier in caps.tiers and node_id != batch.source_node
        ]

    async def handle_memshadow_message(self, header: MemshadowHeader, payload: bytes, source_node: str) -> Dict[str, Any]:
        """
        Handle arbitrary MEMSHADOW messages routed to the hub.
        """
        if header.msg_type == MessageType.MEMORY_SYNC:
            data = header.pack() + payload
            return await self.handle_incoming_batch(data, source_node)

        if header.msg_type == MessageType.MEMORY_STORE:
            # Store requests trigger a follow-up sync to propagate to peers
            self.schedule_sync(source_node, MemoryTier.WORKING, header.priority, reason="memory_store")
            return {"status": "scheduled", "action": "memory_store"}

        if header.msg_type == MessageType.MEMORY_QUERY:
            self.schedule_sync(source_node, MemoryTier.WORKING, header.priority, reason="memory_query_response")
            return {"status": "scheduled", "action": "memory_query"}

        if header.msg_type in (
            MessageType.IMPROVEMENT_ANNOUNCE,
            MessageType.IMPROVEMENT_PAYLOAD,
            MessageType.IMPROVEMENT_REQUEST,
        ):
            logger.debug("Improvement message routed via gateway: %s", header.msg_type.name)
            return {"status": "ack", "action": "improvement_passthrough"}

        return {"status": "ignored", "reason": f"unsupported_msg_type:{header.msg_type.name}"}
    
    async def handle_incoming_batch(self, data: bytes, source_node: str) -> Dict:
        """
        Handle an incoming sync batch from a node.
        
        Args:
            data: Raw batch data
            source_node: Source node ID
            
        Returns:
            Result dict with status and any errors
        """
        try:
            # Parse batch
            batch = MemorySyncBatch.unpack(data)
            
            # Update sync vector
            if batch.items:
                latest_ts = max(item.timestamp_ns for item in batch.items)
                self.update_sync_vector(source_node, batch.tier, batch.source_node, latest_ts)
            
            # Update node status
            if source_node in self._nodes:
                self._nodes[source_node].last_sync = datetime.now(timezone.utc)
                self._nodes[source_node].pending_items = 0
            
            # Route to other nodes if needed
            if batch.target_node == "*" or batch.target_node == self.hub_id:
                # Broadcast to other nodes
                batch.source_node = source_node  # Preserve origin
                results = await self.route_batch(batch)
                return {"status": "routed", "targets": results}
            
            return {"status": "received", "items": len(batch.items)}
            
        except Exception as e:
            logger.error(f"Failed to handle incoming batch from {source_node}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def process_sync_queue(self, max_items: int = 10) -> int:
        """
        Process pending sync operations.
        
        Args:
            max_items: Maximum items to process
            
        Returns:
            Number of items processed
        """
        processed = 0
        
        while self._sync_queue and processed < max_items:
            entry = self._sync_queue.pop(0)
            
            try:
                # Create sync request
                if self._sync_manager and entry.node_id in self._nodes:
                    # Request delta from node
                    if self.mesh_send:
                        sync_request = {
                            "type": "sync_request",
                            "tier": entry.tier.value,
                            "priority": entry.priority.value,
                            "since": self._sync_vectors.get((entry.node_id, entry.tier), {}).get(entry.node_id, 0),
                        }
                        import json
                        await self.mesh_send(entry.node_id, json.dumps(sync_request).encode())
                    processed += 1
                    
            except Exception as e:
                logger.error(f"Failed to process sync for {entry.node_id}: {e}")
        
        return processed
    
    def get_node_stats(self, node_id: str) -> Optional[Dict]:
        """Get memory sync stats for a node"""
        if node_id not in self._nodes:
            return None
        
        caps = self._nodes[node_id]
        return {
            "node_id": node_id,
            "tiers": [t.name for t in caps.tiers],
            "storage": caps.storage_bytes,
            "sync_enabled": caps.sync_enabled,
            "last_sync": caps.last_sync.isoformat() if caps.last_sync else None,
            "pending_items": caps.pending_items,
        }
    
    def get_all_stats(self) -> Dict:
        """Get overall gateway stats"""
        return {
            "hub_id": self.hub_id,
            "registered_nodes": len(self._nodes),
            "pending_syncs": len(self._sync_queue),
            "pending_batches": len(self._pending_batches),
            "metrics": dict(self._metrics),
            "nodes": [self.get_node_stats(n) for n in self._nodes],
        }


class SpokeMemoryAdapter:
    """
    Spoke-side MEMSHADOW memory adapter.
    
    Wraps local memory tiers and handles sync operations.
    
    Usage:
        adapter = SpokeMemoryAdapter(node_id="node-1", hub_node_id="hub")
        
        # Register local memory tiers
        adapter.register_tier(MemoryTier.WORKING, working_memory_instance)
        
        # Create delta batch for sync
        batch = adapter.create_delta_batch(MemoryTier.WORKING, target_node, since_ts)
        
        # Apply incoming batch
        applied, conflicts = adapter.apply_sync_batch(batch)
    """
    
    def __init__(self, node_id: str, hub_node_id: str = "hub",
                 mesh_send: Optional[Callable] = None):
        """
        Initialize spoke adapter.
        
        Args:
            node_id: This node's ID
            hub_node_id: Hub node ID for routing
            mesh_send: Function to send messages
        """
        self.node_id = node_id
        self.hub_node_id = hub_node_id
        self.mesh_send = mesh_send
        
        # Local memory tiers
        self._tiers: Dict[MemoryTier, Any] = {}
        
        # Sync manager
        if SYNC_PROTOCOL_AVAILABLE:
            self._sync_manager = MemorySyncManager(node_id)
        else:
            self._sync_manager = None
        
        # Known peers for P2P
        self._peers: Set[str] = set()
        
        # Pending batches
        self._pending_batches: Dict[str, MemorySyncBatch] = {}
        
        # Metrics
        self._metrics = {
            "batches_sent": 0,
            "batches_received": 0,
            "items_synced": 0,
            "conflicts": 0,
            "p2p_syncs": 0,
            "hub_syncs": 0,
        }
        
        logger.info(f"SpokeMemoryAdapter initialized: {node_id}")
    
    def register_tier(self, tier: MemoryTier, memory_instance: Any) -> bool:
        """
        Register a local memory tier.
        
        Args:
            tier: Memory tier type
            memory_instance: Memory tier instance
            
        Returns:
            True if registered successfully
        """
        self._tiers[tier] = memory_instance
        
        if self._sync_manager:
            self._sync_manager.register_memory_tier(tier, memory_instance)
        
        logger.info(f"Registered {tier.name} memory tier")
        return True
    
    def add_peer(self, peer_id: str):
        """Add a known peer for P2P sync"""
        self._peers.add(peer_id)
    
    def remove_peer(self, peer_id: str):
        """Remove a peer"""
        self._peers.discard(peer_id)
    
    def create_delta_batch(self, tier: MemoryTier, target_node: str,
                          since_timestamp: int = 0) -> Optional[MemorySyncBatch]:
        """
        Create a delta sync batch for a tier.
        
        Args:
            tier: Memory tier
            target_node: Target node for sync
            since_timestamp: Only items modified after this timestamp
            
        Returns:
            MemorySyncBatch if there are items to sync
        """
        if not self._sync_manager:
            logger.warning("Sync manager not available")
            return None
        
        if tier not in self._tiers:
            logger.warning(f"Tier {tier.name} not registered")
            return None
        
        return self._sync_manager.create_delta_batch(tier, target_node, since_timestamp)
    
    def apply_sync_batch(self, batch: MemorySyncBatch) -> Tuple[int, int]:
        """
        Apply an incoming sync batch.
        
        Args:
            batch: The batch to apply
            
        Returns:
            Tuple of (items_applied, conflicts_detected)
        """
        if not self._sync_manager:
            logger.warning("Sync manager not available")
            return 0, 0
        
        applied, conflicts = self._sync_manager.apply_sync_batch(batch)
        
        self._metrics["batches_received"] += 1
        self._metrics["items_synced"] += applied
        self._metrics["conflicts"] += conflicts
        
        return applied, conflicts
    
    async def sync_tier(self, tier: MemoryTier, priority: Priority = Priority.NORMAL) -> Dict:
        """
        Initiate sync for a tier.
        
        Routing:
        - CRITICAL/EMERGENCY: Send P2P to all peers + notify hub
        - Others: Send to hub for distribution
        
        Args:
            tier: Tier to sync
            priority: Sync priority
            
        Returns:
            Sync result
        """
        # Get last sync timestamp from sync vector
        sync_vector = self._sync_manager.get_sync_vector(tier) if self._sync_manager else {}
        last_sync = max(sync_vector.values()) if sync_vector else 0
        
        # Create delta batch
        batch = self.create_delta_batch(tier, "*", last_sync)
        
        if not batch:
            return {"status": "up_to_date", "items": 0}
        
        # Route based on priority
        success = await self._send_batch(batch, priority)
        
        self._metrics["batches_sent"] += 1
        
        return {
            "status": "sent" if success else "failed",
            "items": len(batch.items),
            "priority": priority.name,
        }
    
    async def _send_batch(self, batch: MemorySyncBatch, priority: Priority) -> bool:
        """
        Send batch using appropriate routing.
        
        Priority routing rules:
        - CRITICAL/EMERGENCY: Direct P2P + hub notification
        - HIGH: Hub-relayed with priority
        - NORMAL/LOW: Standard hub routing
        """
        if not self.mesh_send:
            return False
        
        packed = batch.pack()
        
        try:
            if PROTOCOL_AVAILABLE and should_route_p2p(priority):
                # P2P: Send to all peers directly
                self._metrics["p2p_syncs"] += 1
                for peer_id in self._peers:
                    if peer_id != batch.source_node:
                        try:
                            await self.mesh_send(peer_id, packed)
                        except Exception as e:
                            logger.warning(f"P2P send failed to {peer_id}: {e}")
                
                # Also notify hub
                await self.mesh_send(self.hub_node_id, packed)
            else:
                # Hub relay
                self._metrics["hub_syncs"] += 1
                await self.mesh_send(self.hub_node_id, packed)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send batch: {e}")
            self._pending_batches[batch.batch_id] = batch
            return False
    
    async def propagate_improvement(self, improvement_data: Dict,
                                   priority: Priority = Priority.HIGH) -> bool:
        """
        Propagate a local improvement via MEMSHADOW sync.
        
        This is used when local self-improvement produces patterns or weights
        that should be shared across the federation.
        
        Args:
            improvement_data: Improvement package data
            priority: Propagation priority
            
        Returns:
            True if propagated successfully
        """
        if not SYNC_PROTOCOL_AVAILABLE:
            return False
        
        # Create a sync item for the improvement
        import json
        item = MemorySyncItem(
            item_id=improvement_data.get("improvement_id", hashlib.sha256(str(time.time()).encode()).hexdigest()[:32]),
            timestamp_ns=int(time.time() * 1e9),
            tier=MemoryTier.SEMANTIC,  # Improvements go to L3
            operation=SyncOperation.UPDATE,
            priority=SyncPriority(min(priority.value, 4)),
            payload=json.dumps(improvement_data).encode(),
        )
        
        # Create batch
        batch = MemorySyncBatch(
            batch_id=hashlib.sha256(f"{self.node_id}-improvement-{time.time()}".encode()).hexdigest()[:16],
            source_node=self.node_id,
            target_node="*",  # Broadcast
            tier=MemoryTier.SEMANTIC,
            items=[item],
            flags=SyncFlags.REQUIRES_ACK,
        )
        
        return await self._send_batch(batch, priority)
    
    def get_stats(self) -> Dict:
        """Get adapter statistics"""
        return {
            "node_id": self.node_id,
            "registered_tiers": [t.name for t in self._tiers.keys()],
            "peer_count": len(self._peers),
            "pending_batches": len(self._pending_batches),
            "metrics": dict(self._metrics),
        }


if __name__ == "__main__":
    print("MEMSHADOW Gateway Self-Test")
    print("=" * 50)
    
    # Test Hub Gateway
    print("\n[1] Hub Gateway")
    gateway = HubMemshadowGateway("test-hub")
    
    # Register nodes
    gateway.register_node("node-1", "localhost:8001", {
        MemoryTier.WORKING: 1024*1024,
        MemoryTier.EPISODIC: 10*1024*1024,
    })
    gateway.register_node("node-2", "localhost:8002", {
        MemoryTier.WORKING: 1024*1024,
        MemoryTier.SEMANTIC: 100*1024*1024,
    })
    print(f"    Registered 2 nodes")
    
    # Schedule sync
    gateway.schedule_sync("node-1", MemoryTier.WORKING, Priority.HIGH)
    gateway.schedule_sync("node-2", MemoryTier.SEMANTIC, Priority.NORMAL)
    print(f"    Scheduled 2 syncs")
    
    # Get stats
    stats = gateway.get_all_stats()
    print(f"    Nodes: {stats['registered_nodes']}")
    print(f"    Pending syncs: {stats['pending_syncs']}")
    
    # Test Spoke Adapter
    print("\n[2] Spoke Adapter")
    adapter = SpokeMemoryAdapter("node-1", "hub")
    adapter.add_peer("node-2")
    print(f"    Created adapter with 1 peer")
    
    # Get stats
    spoke_stats = adapter.get_stats()
    print(f"    Tiers: {spoke_stats['registered_tiers']}")
    print(f"    Peers: {spoke_stats['peer_count']}")
    
    print("\n" + "=" * 50)
    print("MEMSHADOW Gateway test complete")
