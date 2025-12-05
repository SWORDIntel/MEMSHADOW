"""
Federation Hub Orchestrator

Integrates HubMemshadowGateway with the DSMIL Brain Federation hub
to provide unified memory orchestration across all nodes.

This module serves as the canonical coordination point for:
- Cross-node L1/L2/L3 memory sync
- MEMSHADOW Protocol v2 message handling
- Hub routing rules (priority-aware hub vs P2P)
- Self-improvement propagation

Based on: HUB_DOCS/DSMIL Brain Federation.md
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID

import structlog

from .hub_memshadow_gateway import (
    HubMemshadowGateway,
    NodeMemoryCapabilities,
    NodeSyncState,
)
from .protocol import (
    MessageType,
    Priority,
    MessageFlags,
    MemshadowMessage,
    MessageRouter,
)
from .sync_manager import (
    MemoryTier,
    MemorySyncBatch,
    SyncResult,
)

logger = structlog.get_logger()


class NodeCapability(Enum):
    """Node capability flags"""
    SEARCH = "search"
    CORRELATE = "correlate"
    MEMORY_STORAGE = "memory_storage"
    THREAT_INTEL = "threat_intel"
    SEMANTIC_SEARCH = "semantic_search"
    NEURAL_STORAGE = "neural_storage"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    PSYCH_INTEL = "psych_intel"


@dataclass
class RegisteredNode:
    """A registered spoke node in the federation"""
    node_id: str
    endpoint: str
    capabilities: Set[NodeCapability] = field(default_factory=set)
    data_domains: Set[str] = field(default_factory=set)
    memory_tiers: Set[MemoryTier] = field(default_factory=set)
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "endpoint": self.endpoint,
            "capabilities": [c.value for c in self.capabilities],
            "data_domains": list(self.data_domains),
            "memory_tiers": [t.name for t in self.memory_tiers],
            "registered_at": self.registered_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "is_active": self.is_active,
        }


@dataclass
class QueryPriority(Enum):
    """Query priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class FederationHubOrchestrator:
    """
    Federation Hub Orchestrator
    
    Central coordination point for the DSMIL Brain Federation network,
    integrating MEMSHADOW memory sync with existing hub functionality.
    
    Responsibilities:
    - Node registration and health monitoring
    - Query distribution and response aggregation
    - Intelligence propagation
    - MEMSHADOW memory sync coordination
    - PSYCH message handling (from SHRINK)
    - Self-improvement announcement relay
    
    Usage:
        hub = FederationHubOrchestrator(hub_id="dsmil-central")
        await hub.start()
        
        # Register a spoke node
        await hub.register_node(
            node_id="spoke-1",
            endpoint="spoke1.local:8889",
            capabilities={NodeCapability.SEARCH, NodeCapability.CORRELATE}
        )
        
        # Distribute a query
        results = await hub.query("search for threat actors", priority=QueryPriority.HIGH)
    """
    
    def __init__(
        self,
        hub_id: str = "dsmil-central",
        mesh_port: int = 8889,
        use_mesh: bool = True,
        sync_interval_seconds: int = 30,
    ):
        """
        Initialize Federation Hub Orchestrator.
        
        Args:
            hub_id: Unique hub identifier
            mesh_port: Port for mesh communication
            use_mesh: Whether to enable mesh networking
            sync_interval_seconds: Interval for memory sync operations
        """
        self.hub_id = hub_id
        self.mesh_port = mesh_port
        self.use_mesh = use_mesh
        
        # Registered nodes
        self._nodes: Dict[str, RegisteredNode] = {}
        
        # MEMSHADOW Gateway for memory sync
        self.memshadow_gateway = HubMemshadowGateway(
            hub_id=hub_id,
            mesh_send_callback=self._mesh_send,
            sync_interval_seconds=sync_interval_seconds,
        )
        
        # Message router for MEMSHADOW protocol
        self._router = MessageRouter(node_id=hub_id, is_hub=True)
        self._setup_message_handlers()
        
        # Mesh network (optional)
        self._mesh: Optional[Any] = None  # QuantumMesh instance when available
        
        # Query tracking
        self._pending_queries: Dict[str, Dict] = {}
        
        # Statistics
        self._stats = {
            "nodes_registered": 0,
            "queries_distributed": 0,
            "intel_propagated": 0,
            "psych_messages_handled": 0,
            "improvements_relayed": 0,
        }
        
        # Background tasks
        self._running = False
        self._health_task: Optional[asyncio.Task] = None
        
        logger.info(
            "FederationHubOrchestrator initialized",
            hub_id=hub_id,
            mesh_port=mesh_port,
        )
    
    def _setup_message_handlers(self):
        """Register MEMSHADOW protocol message handlers"""
        # Memory operations
        self._router.register_handler(
            MessageType.MEMORY_SYNC,
            self._handle_memory_sync
        )
        self._router.register_handler(
            MessageType.MEMORY_STORE,
            self._handle_memory_store
        )
        self._router.register_handler(
            MessageType.MEMORY_QUERY,
            self._handle_memory_query
        )
        
        # Federation operations
        self._router.register_handler(
            MessageType.NODE_REGISTER,
            self._handle_node_register
        )
        self._router.register_handler(
            MessageType.HEARTBEAT,
            self._handle_heartbeat
        )
        
        # PSYCH intelligence
        self._router.register_handler(
            MessageType.PSYCH_ASSESSMENT,
            self._handle_psych_intel
        )
        self._router.register_handler(
            MessageType.PSYCH_THREAT_ALERT,
            self._handle_psych_threat
        )
        
        # Self-improvement
        self._router.register_handler(
            MessageType.IMPROVEMENT_ANNOUNCE,
            self._handle_improvement_announce
        )
    
    async def start(self):
        """Start the hub orchestrator"""
        logger.info("Starting FederationHubOrchestrator", hub_id=self.hub_id)
        
        self._running = True
        
        # Start MEMSHADOW gateway
        await self.memshadow_gateway.start()
        
        # Start health monitoring
        self._health_task = asyncio.create_task(self._health_monitor_loop())
        
        # Initialize mesh if enabled
        if self.use_mesh:
            await self._init_mesh()
        
        logger.info("FederationHubOrchestrator started")
    
    async def stop(self):
        """Stop the hub orchestrator"""
        logger.info("Stopping FederationHubOrchestrator", hub_id=self.hub_id)
        
        self._running = False
        
        # Stop health monitoring
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        
        # Stop MEMSHADOW gateway
        await self.memshadow_gateway.stop()
        
        # Cleanup mesh
        if self._mesh:
            self._mesh.stop()
        
        logger.info("FederationHubOrchestrator stopped")
    
    async def _init_mesh(self):
        """Initialize mesh networking"""
        try:
            # Try to import mesh library
            from app.services.mesh_client import MESH_AVAILABLE
            
            if MESH_AVAILABLE:
                logger.info("Mesh networking available")
                # Mesh initialization would happen here
            else:
                logger.info("Mesh networking not available, running in standalone mode")
                
        except ImportError:
            logger.info("Mesh client not available")
    
    async def _mesh_send(self, node_id: str, data: bytes):
        """Send data to a node via mesh network"""
        if self._mesh:
            try:
                self._mesh.send(node_id, MessageType.MEMORY_SYNC.value, data)
            except Exception as e:
                logger.error("Mesh send failed", node_id=node_id, error=str(e))
        else:
            logger.debug("No mesh available, queueing message", node_id=node_id)
    
    # ==================== Node Management ====================
    
    async def register_node(
        self,
        node_id: str,
        endpoint: str,
        capabilities: Optional[Set[NodeCapability]] = None,
        data_domains: Optional[Set[str]] = None,
        memory_tiers: Optional[Set[MemoryTier]] = None,
    ) -> RegisteredNode:
        """
        Register a spoke node with the hub.
        
        Args:
            node_id: Unique node identifier
            endpoint: Node network endpoint (host:port)
            capabilities: Node capabilities
            data_domains: Data domains the node handles
            memory_tiers: Memory tiers the node supports
        
        Returns:
            RegisteredNode object
        """
        # Default capabilities
        if capabilities is None:
            capabilities = {NodeCapability.SEARCH, NodeCapability.MEMORY_STORAGE}
        
        if data_domains is None:
            data_domains = {"memories"}
        
        if memory_tiers is None:
            memory_tiers = {MemoryTier.L1_WORKING, MemoryTier.L2_EPISODIC}
        
        node = RegisteredNode(
            node_id=node_id,
            endpoint=endpoint,
            capabilities=capabilities,
            data_domains=data_domains,
            memory_tiers=memory_tiers,
        )
        
        self._nodes[node_id] = node
        self._stats["nodes_registered"] = len(self._nodes)
        
        # Register with MEMSHADOW gateway
        self.memshadow_gateway.register_memory_node(
            node_id,
            NodeMemoryCapabilities(
                node_id=node_id,
                supported_tiers=memory_tiers,
            )
        )
        
        logger.info(
            "Node registered",
            node_id=node_id,
            endpoint=endpoint,
            capabilities=[c.value for c in capabilities],
        )
        
        return node
    
    def deregister_node(self, node_id: str):
        """Deregister a node from the hub"""
        if node_id in self._nodes:
            del self._nodes[node_id]
            self.memshadow_gateway.deregister_memory_node(node_id)
            self._stats["nodes_registered"] = len(self._nodes)
            logger.info("Node deregistered", node_id=node_id)
    
    def get_node(self, node_id: str) -> Optional[RegisteredNode]:
        """Get a registered node by ID"""
        return self._nodes.get(node_id)
    
    def get_nodes_by_capability(
        self, 
        capability: NodeCapability
    ) -> List[RegisteredNode]:
        """Get all nodes with a specific capability"""
        return [
            node for node in self._nodes.values()
            if capability in node.capabilities and node.is_active
        ]
    
    # ==================== Query Distribution ====================
    
    async def query(
        self,
        query_text: str,
        priority: QueryPriority = QueryPriority.NORMAL,
        target_domains: Optional[Set[str]] = None,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Distribute a query to relevant spoke nodes.
        
        Args:
            query_text: Query string
            priority: Query priority
            target_domains: Specific data domains to query
            timeout_seconds: Query timeout
        
        Returns:
            Aggregated response from all responding nodes
        """
        from uuid import uuid4
        
        query_id = str(uuid4())
        
        # Find relevant nodes
        target_nodes = []
        for node in self._nodes.values():
            if not node.is_active:
                continue
            if target_domains and not (node.data_domains & target_domains):
                continue
            target_nodes.append(node)
        
        if not target_nodes:
            return {
                "query_id": query_id,
                "success": False,
                "error": "No available nodes for query",
                "results": [],
            }
        
        # Track pending responses
        self._pending_queries[query_id] = {
            "query_text": query_text,
            "pending": set(n.node_id for n in target_nodes),
            "responses": [],
            "started_at": datetime.utcnow(),
        }
        
        # Create query message
        query_payload = json.dumps({
            "query_id": query_id,
            "query_text": query_text,
            "priority": priority.value,
        }).encode()
        
        msg = MemshadowMessage.create(
            msg_type=MessageType.QUERY_DISTRIBUTE,
            payload=query_payload,
            priority=Priority(priority.value),
        )
        
        # Send to all target nodes
        for node in target_nodes:
            await self._mesh_send(node.node_id, msg.pack())
        
        self._stats["queries_distributed"] += 1
        
        # Wait for responses with timeout
        try:
            await asyncio.wait_for(
                self._wait_for_query_responses(query_id),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.warning("Query timed out", query_id=query_id)
        
        # Aggregate results
        query_state = self._pending_queries.pop(query_id, {})
        
        return {
            "query_id": query_id,
            "success": True,
            "query_text": query_text,
            "results": query_state.get("responses", []),
            "nodes_queried": len(target_nodes),
            "nodes_responded": len(query_state.get("responses", [])),
        }
    
    async def _wait_for_query_responses(self, query_id: str):
        """Wait for all query responses"""
        while query_id in self._pending_queries:
            state = self._pending_queries[query_id]
            if not state["pending"]:
                break
            await asyncio.sleep(0.1)
    
    # ==================== Intelligence Propagation ====================
    
    async def propagate_intel(
        self,
        intel_report: Dict[str, Any],
        target_nodes: Optional[List[str]] = None,
        priority: Priority = Priority.NORMAL,
    ):
        """
        Propagate intelligence report to spoke nodes.
        
        Args:
            intel_report: Intelligence data to propagate
            target_nodes: Specific nodes (None for all)
            priority: Propagation priority
        """
        # Determine targets
        if target_nodes:
            targets = [self._nodes[n] for n in target_nodes if n in self._nodes]
        else:
            targets = list(self._nodes.values())
        
        # Create message
        payload = json.dumps(intel_report).encode()
        msg = MemshadowMessage.create(
            msg_type=MessageType.INTEL_PROPAGATE,
            payload=payload,
            priority=priority,
        )
        
        # Send to targets
        for node in targets:
            if node.is_active:
                await self._mesh_send(node.node_id, msg.pack())
        
        self._stats["intel_propagated"] += 1
        
        logger.info(
            "Intel propagated",
            targets=len(targets),
            priority=priority.name,
        )
    
    # ==================== Message Handlers ====================
    
    def _handle_memory_sync(self, message: MemshadowMessage, peer_id: str):
        """Handle MEMORY_SYNC message"""
        asyncio.create_task(
            self.memshadow_gateway.handle_memory_sync(message.pack(), peer_id)
        )
    
    def _handle_memory_store(self, message: MemshadowMessage, peer_id: str):
        """Handle MEMORY_STORE message"""
        logger.debug("MEMORY_STORE received", peer=peer_id)
        # Forward to MEMSHADOW gateway for processing
    
    def _handle_memory_query(self, message: MemshadowMessage, peer_id: str):
        """Handle MEMORY_QUERY message"""
        logger.debug("MEMORY_QUERY received", peer=peer_id)
        # Process memory query
    
    def _handle_node_register(self, message: MemshadowMessage, peer_id: str):
        """Handle NODE_REGISTER message"""
        try:
            data = json.loads(message.payload.decode())
            asyncio.create_task(
                self.register_node(
                    node_id=data.get("node_id", peer_id),
                    endpoint=data.get("endpoint", ""),
                    capabilities={NodeCapability(c) for c in data.get("capabilities", [])},
                    data_domains=set(data.get("data_domains", [])),
                    memory_tiers={MemoryTier(t) for t in data.get("memory_tiers", [1, 2])},
                )
            )
        except Exception as e:
            logger.error("Failed to handle NODE_REGISTER", error=str(e))
    
    def _handle_heartbeat(self, message: MemshadowMessage, peer_id: str):
        """Handle HEARTBEAT message"""
        if peer_id in self._nodes:
            self._nodes[peer_id].last_seen = datetime.utcnow()
            self.memshadow_gateway.update_node_heartbeat(peer_id)
    
    def _handle_psych_intel(self, message: MemshadowMessage, peer_id: str):
        """Handle PSYCH_ASSESSMENT message from SHRINK nodes"""
        try:
            # Store in memory tiers
            logger.info("PSYCH intel received", peer=peer_id)
            self._stats["psych_messages_handled"] += 1
            
            # Evaluate significance and broadcast if needed
            # (Implementation would check risk scores, dark triad, etc.)
            
        except Exception as e:
            logger.error("Failed to handle PSYCH intel", error=str(e))
    
    def _handle_psych_threat(self, message: MemshadowMessage, peer_id: str):
        """Handle PSYCH_THREAT_ALERT with CRITICAL priority"""
        logger.warning(
            "PSYCH threat alert received",
            peer=peer_id,
            priority="CRITICAL",
        )
        # Route with CRITICAL priority to all nodes
        asyncio.create_task(
            self.propagate_intel(
                {"type": "psych_threat", "source": peer_id, "payload": message.payload.hex()},
                priority=Priority.CRITICAL,
            )
        )
    
    def _handle_improvement_announce(self, message: MemshadowMessage, peer_id: str):
        """Handle IMPROVEMENT_ANNOUNCE message"""
        try:
            data = json.loads(message.payload.decode())
            logger.info(
                "Improvement announced",
                peer=peer_id,
                improvement_type=data.get("type"),
            )
            self._stats["improvements_relayed"] += 1
            
            # Relay to interested nodes
            asyncio.create_task(
                self.propagate_intel(
                    {"type": "improvement_announce", "source": peer_id, "data": data},
                    priority=Priority.HIGH,
                )
            )
            
        except Exception as e:
            logger.error("Failed to handle IMPROVEMENT_ANNOUNCE", error=str(e))
    
    # ==================== Background Tasks ====================
    
    async def _health_monitor_loop(self):
        """Monitor node health"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Every minute
                
                now = datetime.utcnow()
                for node in self._nodes.values():
                    # Check if node is stale
                    age = (now - node.last_seen).total_seconds()
                    if age > 300:  # 5 minutes
                        if node.is_active:
                            node.is_active = False
                            logger.warning(
                                "Node marked inactive",
                                node_id=node.node_id,
                                last_seen=node.last_seen.isoformat(),
                            )
                    else:
                        if not node.is_active:
                            node.is_active = True
                            logger.info(
                                "Node marked active",
                                node_id=node.node_id,
                            )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitor error", error=str(e))
    
    # ==================== Statistics ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hub orchestrator statistics"""
        active_nodes = sum(1 for n in self._nodes.values() if n.is_active)
        
        return {
            "hub_id": self.hub_id,
            "stats": self._stats,
            "nodes": {
                "total": len(self._nodes),
                "active": active_nodes,
                "inactive": len(self._nodes) - active_nodes,
            },
            "memshadow_stats": self.memshadow_gateway.get_hub_memory_stats(),
            "pending_queries": len(self._pending_queries),
        }
    
    def get_node_list(self) -> List[Dict[str, Any]]:
        """Get list of all registered nodes"""
        return [node.to_dict() for node in self._nodes.values()]


# Singleton instance
_hub_instance: Optional[FederationHubOrchestrator] = None


def get_federation_hub() -> FederationHubOrchestrator:
    """Get or create the global federation hub"""
    global _hub_instance
    if _hub_instance is None:
        _hub_instance = FederationHubOrchestrator()
    return _hub_instance


async def init_federation_hub(
    hub_id: str = "dsmil-central",
    mesh_port: int = 8889,
) -> FederationHubOrchestrator:
    """Initialize and start the federation hub"""
    global _hub_instance
    _hub_instance = FederationHubOrchestrator(
        hub_id=hub_id,
        mesh_port=mesh_port,
    )
    await _hub_instance.start()
    return _hub_instance


async def shutdown_federation_hub():
    """Shutdown the federation hub"""
    global _hub_instance
    if _hub_instance:
        await _hub_instance.stop()
        _hub_instance = None
