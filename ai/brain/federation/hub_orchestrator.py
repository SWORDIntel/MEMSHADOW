"""
Hub Orchestrator with MEMSHADOW Gateway Integration

Central coordination point for the DSMIL Brain Federation distributed network.

Responsibilities:
- Node registration and health monitoring
- Query distribution and response aggregation
- Intelligence propagation
- PSYCH message handling (from SHRINK)
- Self-improvement announcement relay
- MEMSHADOW memory tier synchronization (hub-first routing)

Based on: HUB_DOCS/DSMIL Brain Federation.md
"""

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

import structlog

# Import canonical protocol
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
    should_route_p2p,
    get_routing_mode,
    HEADER_SIZE,
)

# Import memory sync protocol
sys.path.insert(0, str(Path(__file__).parent.parent))
from memory.memory_sync_protocol import (
    MemorySyncBatch,
    MemorySyncManager,
    SyncPriority,
    SyncConfig,
)

logger = structlog.get_logger()


# ============================================================================
# Node and Query Types
# ============================================================================

class NodeCapability(Enum):
    """Spoke node capabilities"""
    SEARCH = "search"
    CORRELATE = "correlate"
    ANALYZE = "analyze"
    SHRINK = "shrink"  # SHRINK-enabled node
    MEMORY_SYNC = "memory_sync"


class QueryPriority(IntEnum):
    """Query priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class NodeMemoryCapabilities:
    """Memory tier capabilities for a node"""
    supports_l1: bool = True
    supports_l2: bool = True
    supports_l3: bool = True
    max_batch_size: int = 100
    compression_supported: bool = True


@dataclass
class RegisteredNode:
    """Registered spoke node information"""
    node_id: str
    endpoint: str
    capabilities: Set[NodeCapability]
    data_domains: Set[str]
    memory_capabilities: NodeMemoryCapabilities
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    is_healthy: bool = True
    sync_priority: int = 0  # Higher = sync first


@dataclass
class AggregatedResponse:
    """Aggregated response from distributed query"""
    query_id: str
    results: List[Dict[str, Any]]
    responding_nodes: List[str]
    latency_ms: float
    errors: List[str] = field(default_factory=list)


# ============================================================================
# Hub MEMSHADOW Gateway
# ============================================================================

class HubMemshadowGateway:
    """
    Hub-side MEMSHADOW gateway for memory synchronization.
    
    Responsibilities:
    - Register nodes with memory tier capabilities
    - Schedule sync operations (priority-aware queue)
    - Route batches (hub-relay vs P2P based on priority)
    - Track per-node/per-tier sync vectors
    - Collect metrics and observability
    """
    
    def __init__(
        self,
        hub_id: str,
        mesh_send_callback: Optional[Callable] = None,
        config: Optional[SyncConfig] = None,
    ):
        self.hub_id = hub_id
        self.mesh_send = mesh_send_callback
        self.config = config or SyncConfig()
        
        # Node memory capabilities: node_id -> NodeMemoryCapabilities
        self._node_capabilities: Dict[str, NodeMemoryCapabilities] = {}
        
        # Sync vectors: (node_id, tier) -> version dict
        self._sync_vectors: Dict[tuple, Dict[str, int]] = defaultdict(dict)
        
        # Pending syncs: priority queue
        self._pending_syncs: List[tuple] = []  # (priority, timestamp, node_id, tier)
        
        # Per-node sync managers
        self._node_sync_managers: Dict[str, MemorySyncManager] = {}
        
        # Metrics
        self._metrics = {
            "nodes_registered": 0,
            "syncs_scheduled": 0,
            "batches_routed": 0,
            "p2p_batches": 0,
            "hub_relayed_batches": 0,
            "bytes_transferred": 0,
        }
        
        # Background sync task
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("HubMemshadowGateway initialized", hub_id=hub_id)
    
    def register_node(
        self,
        node_id: str,
        endpoint: str,
        memory_caps: Optional[NodeMemoryCapabilities] = None,
    ):
        """
        Register a node with its memory tier capabilities.
        
        Called during node registration to record which tiers the node supports.
        """
        if memory_caps is None:
            memory_caps = NodeMemoryCapabilities()
        
        self._node_capabilities[node_id] = memory_caps
        self._metrics["nodes_registered"] = len(self._node_capabilities)
        
        # Initialize sync vectors for this node
        for tier in [MemoryTier.WORKING, MemoryTier.EPISODIC, MemoryTier.SEMANTIC]:
            if self._tier_supported(node_id, tier):
                self._sync_vectors[(node_id, tier)] = {}
        
        logger.info(
            "Node memory capabilities registered",
            node_id=node_id,
            l1=memory_caps.supports_l1,
            l2=memory_caps.supports_l2,
            l3=memory_caps.supports_l3,
        )
    
    def _tier_supported(self, node_id: str, tier: MemoryTier) -> bool:
        """Check if node supports a memory tier"""
        caps = self._node_capabilities.get(node_id)
        if not caps:
            return False
        if tier == MemoryTier.WORKING:
            return caps.supports_l1
        elif tier == MemoryTier.EPISODIC:
            return caps.supports_l2
        elif tier == MemoryTier.SEMANTIC:
            return caps.supports_l3
        return False
    
    def schedule_sync(
        self,
        node_id: str,
        tier: MemoryTier,
        priority: SyncPriority = SyncPriority.NORMAL,
    ):
        """
        Schedule a sync operation with a node.
        
        Args:
            node_id: Target node
            tier: Memory tier to sync
            priority: Sync priority (affects queue order)
        """
        if not self._tier_supported(node_id, tier):
            logger.warning("Node does not support tier", node_id=node_id, tier=tier.name)
            return
        
        # Add to priority queue (negative priority for max-heap behavior)
        entry = (-priority.value, time.time(), node_id, tier)
        self._pending_syncs.append(entry)
        self._pending_syncs.sort()  # Sort by priority then time
        
        self._metrics["syncs_scheduled"] += 1
        
        logger.debug(
            "Sync scheduled",
            node_id=node_id,
            tier=tier.name,
            priority=priority.name,
            queue_depth=len(self._pending_syncs),
        )
    
    async def route_batch(
        self,
        batch: MemorySyncBatch,
    ) -> Dict[str, Any]:
        """
        Route a sync batch to appropriate destination(s).
        
        Routing rules based on priority:
        - CRITICAL/EMERGENCY (Priority.CRITICAL+): P2P + hub notification
        - HIGH (Priority.HIGH): Hub-relayed with priority queue
        - NORMAL/LOW: Standard hub routing
        """
        memshadow_priority = batch.priority.to_memshadow_priority()
        routing_mode = get_routing_mode(memshadow_priority)
        
        result = {
            "batch_id": batch.batch_id,
            "routing_mode": routing_mode,
            "targets_sent": [],
            "errors": [],
        }
        
        if not self.mesh_send:
            result["errors"].append("No mesh send callback configured")
            return result
        
        msg = batch.to_memshadow_message()
        packed = msg.pack()
        
        # Determine targets
        if batch.target_node == "*":
            # Broadcast to all nodes that support this tier
            targets = [
                nid for nid in self._node_capabilities.keys()
                if self._tier_supported(nid, batch.tier)
            ]
        else:
            targets = [batch.target_node]
        
        # Route based on priority
        if should_route_p2p(memshadow_priority):
            # P2P routing: Send directly to targets (via hub as relay)
            self._metrics["p2p_batches"] += 1
            for target in targets:
                try:
                    await self.mesh_send(target, packed)
                    result["targets_sent"].append(target)
                except Exception as e:
                    result["errors"].append(f"{target}: {str(e)}")
        else:
            # Hub-relayed: Hub decides propagation
            self._metrics["hub_relayed_batches"] += 1
            for target in targets:
                try:
                    await self.mesh_send(target, packed)
                    result["targets_sent"].append(target)
                except Exception as e:
                    result["errors"].append(f"{target}: {str(e)}")
        
        self._metrics["batches_routed"] += 1
        self._metrics["bytes_transferred"] += len(packed) * len(result["targets_sent"])
        
        logger.info(
            "Batch routed",
            batch_id=batch.batch_id,
            routing_mode=routing_mode,
            targets=len(result["targets_sent"]),
        )
        
        return result
    
    async def handle_incoming_batch(
        self,
        data: bytes,
        source_node: str,
    ) -> Dict[str, Any]:
        """
        Handle an incoming sync batch from a spoke.
        
        May forward to other nodes based on routing rules.
        """
        try:
            msg = MemshadowMessage.unpack(data)
            batch = MemorySyncBatch.from_memshadow_message(msg)
            
            # Update sync vector for source
            for item in batch.items:
                key = (source_node, batch.tier)
                self._sync_vectors[key][str(item.item_id)] = item.version
            
            # Decide if we should forward
            if batch.target_node == "*" or batch.target_node != self.hub_id:
                return await self.route_batch(batch)
            
            return {"batch_id": batch.batch_id, "status": "received_at_hub"}
            
        except Exception as e:
            logger.error("Failed to handle incoming batch", error=str(e))
            return {"error": str(e)}
    
    def get_sync_vector(self, node_id: str, tier: MemoryTier) -> Dict[str, int]:
        """Get sync vector for a node/tier combination"""
        return dict(self._sync_vectors.get((node_id, tier), {}))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get gateway statistics"""
        return {
            "hub_id": self.hub_id,
            "pending_syncs": len(self._pending_syncs),
            **self._metrics,
        }


# ============================================================================
# Hub Orchestrator
# ============================================================================

class HubOrchestrator:
    """
    Central coordination point for the DSMIL Brain Federation.
    
    Integrates HubMemshadowGateway for memory synchronization.
    """
    
    def __init__(
        self,
        hub_id: str,
        mesh_port: int = 8889,
        use_mesh: bool = True,
        sync_config: Optional[SyncConfig] = None,
    ):
        self.hub_id = hub_id
        self.mesh_port = mesh_port
        self.use_mesh = use_mesh
        
        # Registered nodes
        self._nodes: Dict[str, RegisteredNode] = {}
        
        # Mesh send callback (set by mesh integration)
        self._mesh_send: Optional[Callable] = None
        
        # Message handlers: MessageType -> handler
        self._handlers: Dict[MessageType, Callable] = {}
        
        # MEMSHADOW Gateway (hub-side memory sync)
        self._memshadow_gateway = HubMemshadowGateway(
            hub_id=hub_id,
            mesh_send_callback=self._mesh_send_wrapper,
            config=sync_config,
        )
        
        # Setup message handlers
        self._setup_handlers()
        
        # Stats
        self._stats = {
            "queries_distributed": 0,
            "intel_propagated": 0,
            "improvements_relayed": 0,
            "psych_messages_processed": 0,
        }
        
        logger.info("HubOrchestrator initialized", hub_id=hub_id, mesh_port=mesh_port)
    
    def _setup_handlers(self):
        """Register message handlers for MEMSHADOW message types"""
        # PSYCH handlers
        self._handlers[MessageType.PSYCH_ASSESSMENT] = self._handle_psych_intel
        self._handlers[MessageType.PSYCH_THREAT_ALERT] = self._handle_psych_threat
        self._handlers[MessageType.PSYCH_ANOMALY] = self._handle_psych_intel
        
        # Memory handlers
        self._handlers[MessageType.MEMORY_SYNC] = self._handle_memory_sync
        self._handlers[MessageType.MEMORY_STORE] = self._handle_memory_store
        self._handlers[MessageType.MEMORY_QUERY] = self._handle_memory_query
        
        # Improvement handlers
        self._handlers[MessageType.IMPROVEMENT_ANNOUNCE] = self._handle_improvement_announce
        self._handlers[MessageType.IMPROVEMENT_REQUEST] = self._handle_improvement_request
        self._handlers[MessageType.IMPROVEMENT_PAYLOAD] = self._handle_improvement_payload
        
        # Federation handlers
        self._handlers[MessageType.NODE_REGISTER] = self._handle_node_register
        self._handlers[MessageType.INTEL_PROPAGATE] = self._handle_intel_propagate
    
    async def _mesh_send_wrapper(self, peer_id: str, data: bytes):
        """Wrapper for mesh send that uses configured callback"""
        if self._mesh_send:
            await self._mesh_send(peer_id, data)
        else:
            logger.warning("No mesh send callback configured", peer=peer_id)
    
    def set_mesh_callback(self, callback: Callable):
        """Set the mesh send callback"""
        self._mesh_send = callback
        self._memshadow_gateway.mesh_send = self._mesh_send_wrapper
    
    # ========================================================================
    # Node Registration
    # ========================================================================
    
    def register_node(
        self,
        node_id: str,
        endpoint: str,
        capabilities: Set[NodeCapability],
        data_domains: Optional[Set[str]] = None,
        memory_capabilities: Optional[NodeMemoryCapabilities] = None,
    ) -> RegisteredNode:
        """
        Register a spoke node.
        
        Args:
            node_id: Unique node identifier
            endpoint: Node endpoint URL
            capabilities: Set of NodeCapability values
            data_domains: Data domains the node handles
            memory_capabilities: Memory tier capabilities (for MEMSHADOW sync)
        
        Returns:
            RegisteredNode instance
        """
        if memory_capabilities is None:
            memory_capabilities = NodeMemoryCapabilities()
        
        # Add MEMORY_SYNC capability if node supports any memory tier
        if (memory_capabilities.supports_l1 or 
            memory_capabilities.supports_l2 or 
            memory_capabilities.supports_l3):
            capabilities = capabilities | {NodeCapability.MEMORY_SYNC}
        
        node = RegisteredNode(
            node_id=node_id,
            endpoint=endpoint,
            capabilities=capabilities,
            data_domains=data_domains or set(),
            memory_capabilities=memory_capabilities,
        )
        
        self._nodes[node_id] = node
        
        # Register with MEMSHADOW gateway
        self._memshadow_gateway.register_node(
            node_id=node_id,
            endpoint=endpoint,
            memory_caps=memory_capabilities,
        )
        
        logger.info(
            "Node registered",
            node_id=node_id,
            capabilities=[c.value for c in capabilities],
        )
        
        return node
    
    def get_node(self, node_id: str) -> Optional[RegisteredNode]:
        """Get a registered node by ID"""
        return self._nodes.get(node_id)
    
    def get_healthy_nodes(self) -> List[RegisteredNode]:
        """Get all healthy registered nodes"""
        return [n for n in self._nodes.values() if n.is_healthy]
    
    def update_node_health(self, node_id: str, is_healthy: bool):
        """Update node health status"""
        if node_id in self._nodes:
            self._nodes[node_id].is_healthy = is_healthy
            self._nodes[node_id].last_heartbeat = datetime.utcnow()
    
    # ========================================================================
    # Query Distribution
    # ========================================================================
    
    async def query(
        self,
        query_text: str,
        priority: QueryPriority = QueryPriority.NORMAL,
        target_nodes: Optional[List[str]] = None,
        timeout_sec: float = 30.0,
    ) -> AggregatedResponse:
        """
        Distribute query to nodes and aggregate responses.
        
        Args:
            query_text: Query string
            priority: Query priority
            target_nodes: Specific nodes to query (or None for all healthy nodes)
            timeout_sec: Query timeout
        
        Returns:
            AggregatedResponse with results from all responding nodes
        """
        query_id = str(uuid4())
        start_time = time.time()
        
        # Select target nodes
        if target_nodes:
            nodes = [self._nodes[nid] for nid in target_nodes if nid in self._nodes]
        else:
            nodes = [n for n in self._nodes.values() if n.is_healthy]
        
        # Create MEMSHADOW query message
        payload = json.dumps({
            "query_id": query_id,
            "query_text": query_text,
            "timestamp": int(time.time() * 1e9),
        }).encode()
        
        msg = MemshadowMessage.create(
            msg_type=MessageType.QUERY_DISTRIBUTE,
            payload=payload,
            priority=Priority(priority.value),
        )
        
        # Send to nodes
        results = []
        errors = []
        responding_nodes = []
        
        if self._mesh_send:
            for node in nodes:
                try:
                    await self._mesh_send(node.node_id, msg.pack())
                    responding_nodes.append(node.node_id)
                except Exception as e:
                    errors.append(f"{node.node_id}: {str(e)}")
        
        self._stats["queries_distributed"] += 1
        
        latency_ms = (time.time() - start_time) * 1000
        
        return AggregatedResponse(
            query_id=query_id,
            results=results,
            responding_nodes=responding_nodes,
            latency_ms=latency_ms,
            errors=errors,
        )
    
    # ========================================================================
    # Intelligence Propagation
    # ========================================================================
    
    def propagate_intel(
        self,
        intel_report: Dict[str, Any],
        target_nodes: Optional[List[str]] = None,
        priority: Priority = Priority.NORMAL,
    ):
        """
        Broadcast intelligence to network nodes.
        
        Args:
            intel_report: Intelligence report data
            target_nodes: Specific nodes (or None for broadcast)
            priority: Message priority
        """
        payload = json.dumps(intel_report).encode()
        
        msg = MemshadowMessage.create(
            msg_type=MessageType.INTEL_PROPAGATE,
            payload=payload,
            priority=priority,
        )
        
        if target_nodes:
            nodes = [self._nodes[nid] for nid in target_nodes if nid in self._nodes]
        else:
            nodes = list(self._nodes.values())
        
        for node in nodes:
            if self._mesh_send:
                asyncio.create_task(self._mesh_send(node.node_id, msg.pack()))
        
        self._stats["intel_propagated"] += 1
        
        logger.info(
            "Intel propagated",
            targets=len(nodes),
            priority=priority.name,
        )
    
    # ========================================================================
    # Message Handlers
    # ========================================================================
    
    async def _handle_psych_intel(self, data: bytes, peer_id: str):
        """
        Process SHRINK psychological data.
        
        Steps:
        1. Parse MEMSHADOW message
        2. Store in appropriate memory tier
        3. Evaluate significance
        4. Broadcast significant updates
        """
        try:
            msg = MemshadowMessage.unpack(data)
            
            # Log receipt with rate limiting for safety
            logger.info(
                "PSYCH intel received",
                msg_type=msg.header.msg_type.name,
                priority=msg.header.priority.name,
                source=peer_id,
                payload_len=msg.header.payload_len,
            )
            
            # Evaluate significance (example thresholds)
            # OT-Safety: This is observation/analysis only, no direct actuation
            significance = self._evaluate_psych_significance(msg.payload)
            
            if significance >= 0.7:
                # High significance: broadcast to network
                self.propagate_intel(
                    {"type": "psych_update", "source": peer_id, "significance": significance},
                    priority=Priority.HIGH if significance >= 0.9 else Priority.NORMAL,
                )
            
            self._stats["psych_messages_processed"] += 1
            
        except Exception as e:
            logger.error("Failed to handle PSYCH intel", error=str(e))
    
    async def _handle_psych_threat(self, data: bytes, peer_id: str):
        """
        Handle PSYCH threat alert (high priority).
        
        OT-Safety: Logs and routes alert but does NOT take direct action.
        Any OT-affecting actions must go through separate control services.
        """
        try:
            msg = MemshadowMessage.unpack(data)
            
            logger.warning(
                "PSYCH THREAT ALERT",
                source=peer_id,
                priority=msg.header.priority.name,
            )
            
            # Route with CRITICAL priority, but do not auto-actuate
            # OT systems must have their own safety checks
            self.propagate_intel(
                {
                    "type": "threat_alert",
                    "source": peer_id,
                    "requires_human_review": True,  # Safety flag
                },
                priority=Priority.CRITICAL,
            )
            
        except Exception as e:
            logger.error("Failed to handle PSYCH threat", error=str(e))
    
    async def _handle_memory_sync(self, data: bytes, peer_id: str):
        """
        Handle MEMORY_SYNC message via MEMSHADOW gateway.
        
        Routes the batch appropriately based on priority.
        """
        result = await self._memshadow_gateway.handle_incoming_batch(data, peer_id)
        logger.debug("Memory sync handled", source=peer_id, result=result)
    
    async def _handle_memory_store(self, data: bytes, peer_id: str):
        """Handle MEMORY_STORE message"""
        logger.debug("Memory store received", source=peer_id)
    
    async def _handle_memory_query(self, data: bytes, peer_id: str):
        """Handle MEMORY_QUERY message"""
        logger.debug("Memory query received", source=peer_id)
    
    async def _handle_improvement_announce(self, data: bytes, peer_id: str):
        """
        Relay improvement announcements.
        
        Evaluates improvement and broadcasts to interested nodes.
        """
        try:
            msg = MemshadowMessage.unpack(data)
            improvement = json.loads(msg.payload.decode())
            
            gain_percent = improvement.get("gain_percent", 0)
            
            # Determine propagation strategy
            if gain_percent > 20:
                priority = Priority.CRITICAL  # P2P + hub
            elif gain_percent > 10:
                priority = Priority.HIGH      # Hub with priority
            else:
                priority = Priority.LOW       # Background
            
            # Rebroadcast
            relay_msg = MemshadowMessage.create(
                msg_type=MessageType.IMPROVEMENT_ANNOUNCE,
                payload=msg.payload,
                priority=priority,
            )
            
            for node_id, node in self._nodes.items():
                if node_id != peer_id and node.is_healthy:
                    if self._mesh_send:
                        await self._mesh_send(node_id, relay_msg.pack())
            
            self._stats["improvements_relayed"] += 1
            
            logger.info(
                "Improvement relayed",
                source=peer_id,
                gain_percent=gain_percent,
                priority=priority.name,
            )
            
        except Exception as e:
            logger.error("Failed to relay improvement", error=str(e))
    
    async def _handle_improvement_request(self, data: bytes, peer_id: str):
        """Forward improvement request to source node"""
        logger.debug("Improvement request received", source=peer_id)
    
    async def _handle_improvement_payload(self, data: bytes, peer_id: str):
        """Forward improvement payload to requesting node"""
        logger.debug("Improvement payload received", source=peer_id)
    
    async def _handle_node_register(self, data: bytes, peer_id: str):
        """Handle node registration message"""
        try:
            msg = MemshadowMessage.unpack(data)
            reg_data = json.loads(msg.payload.decode())
            
            self.register_node(
                node_id=reg_data["node_id"],
                endpoint=reg_data.get("endpoint", ""),
                capabilities={NodeCapability(c) for c in reg_data.get("capabilities", [])},
            )
            
        except Exception as e:
            logger.error("Failed to handle node registration", error=str(e))
    
    async def _handle_intel_propagate(self, data: bytes, peer_id: str):
        """Handle intel propagation from other nodes"""
        logger.debug("Intel propagation received", source=peer_id)
    
    def _evaluate_psych_significance(self, payload: bytes) -> float:
        """
        Evaluate psychological data significance.
        
        Returns a score 0.0-1.0 indicating significance.
        """
        # Placeholder: real implementation would analyze the psych event
        return 0.5
    
    # ========================================================================
    # MEMSHADOW Gateway Access
    # ========================================================================
    
    @property
    def memshadow_gateway(self) -> HubMemshadowGateway:
        """Access the MEMSHADOW gateway"""
        return self._memshadow_gateway
    
    def schedule_memory_sync(
        self,
        node_id: str,
        tier: MemoryTier,
        priority: SyncPriority = SyncPriority.NORMAL,
    ):
        """Schedule a memory sync with a node"""
        self._memshadow_gateway.schedule_sync(node_id, tier, priority)
    
    # ========================================================================
    # Dispatch
    # ========================================================================
    
    async def dispatch_message(self, data: bytes, peer_id: str) -> bool:
        """
        Dispatch incoming MEMSHADOW message to appropriate handler.
        
        Returns:
            True if message was handled successfully
        """
        try:
            header = MemshadowHeader.unpack(data[:HEADER_SIZE])
            
            if not header.validate():
                logger.warning("Invalid MEMSHADOW header", peer=peer_id)
                return False
            
            handler = self._handlers.get(header.msg_type)
            if handler:
                await handler(data, peer_id)
                return True
            else:
                logger.debug("No handler for message type", msg_type=header.msg_type.name)
                return False
                
        except Exception as e:
            logger.error("Message dispatch error", error=str(e), peer=peer_id)
            return False
    
    # ========================================================================
    # Stats
    # ========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            "hub_id": self.hub_id,
            "registered_nodes": len(self._nodes),
            "healthy_nodes": len(self.get_healthy_nodes()),
            "memshadow_gateway": self._memshadow_gateway.get_stats(),
            **self._stats,
        }


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "NodeCapability",
    "QueryPriority",
    "NodeMemoryCapabilities",
    "RegisteredNode",
    "AggregatedResponse",
    "HubMemshadowGateway",
    "HubOrchestrator",
]
