#!/usr/bin/env python3
"""
Hub Orchestrator for DSMIL Brain Federation

Central hub that:
- Originates all NL queries
- Distributes queries to relevant nodes
- Aggregates and synthesizes responses
- Manages node registry and health
- Propagates intelligence updates

This is the "brain's brain" - the central coordination point
for the distributed intelligence network.
"""

import asyncio
import hashlib
import threading
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Set, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from collections import defaultdict
from pathlib import Path
import json
import time

from ..plugins.ingest.memshadow_ingest import ingest_memshadow_binary

logger = logging.getLogger(__name__)

# Try to import dsmil-mesh library
try:
    # Add libs path if needed
    libs_path = Path(__file__).parent.parent.parent.parent / "libs" / "dsmil-mesh" / "python"
    if libs_path.exists() and str(libs_path) not in sys.path:
        sys.path.insert(0, str(libs_path))

    from mesh import QuantumMesh, Peer, MeshConfig
    from messages import MessageTypes, MessagePriority
    MESH_AVAILABLE = True
except ImportError:
    MESH_AVAILABLE = False
    QuantumMesh = None
    MessageTypes = None
    MessagePriority = None
    logger.warning("dsmil-mesh not available - using simulated mode")

# Try to import improvement types for self-improvement propagation
try:
    from .improvement_types import (
        ImprovementPackage, ImprovementAnnouncement, ImprovementAck,
        ImprovementReject, ImprovementPriority
    )
    from .improvement_tracker import ImprovementTracker
    IMPROVEMENT_AVAILABLE = True
except ImportError:
    IMPROVEMENT_AVAILABLE = False
    logger.debug("Improvement types not available")

# Try to import MEMSHADOW protocol for binary encoding
try:
    protocol_path = Path(__file__).parent.parent.parent.parent / "libs" / "memshadow-protocol" / "python"
    if protocol_path.exists() and str(protocol_path) not in sys.path:
        sys.path.insert(0, str(protocol_path))

    from dsmil_protocol import (
        MemshadowHeader, MemshadowMessage, MessageType, Priority,
        MessageFlags, HEADER_SIZE, should_route_p2p, PSYCH_EVENT_SIZE
    )
    PROTOCOL_AVAILABLE = True
except ImportError:
    PROTOCOL_AVAILABLE = False
    MessageType = None
    Priority = None
    PSYCH_EVENT_SIZE = 64
    logger.debug("MEMSHADOW protocol not available")


class NodeStatus(Enum):
    """Status of a registered node"""
    ONLINE = auto()
    OFFLINE = auto()
    DEGRADED = auto()      # Partially functional
    SYNCING = auto()       # Currently syncing
    SUSPICIOUS = auto()    # Security concern
    TERMINATED = auto()    # Self-destructed or removed


class QueryPriority(Enum):
    """Priority levels for queries"""
    BACKGROUND = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class QueryType(Enum):
    """Types of queries the hub can distribute"""
    SEARCH = auto()        # Search for information
    CORRELATE = auto()     # Find correlations
    ANALYZE = auto()       # Deep analysis
    PREDICT = auto()       # Predictive query
    VERIFY = auto()        # Verify information
    ALERT = auto()         # Alert/notification
    SYNC = auto()          # Synchronization query
    HEALTH = auto()        # Health check


@dataclass
class NodeCapability:
    """Capabilities of a registered node"""
    node_id: str
    capabilities: Set[str] = field(default_factory=set)

    # Resource levels
    compute_power: float = 1.0  # Relative compute power
    memory_available: int = 0   # Bytes available
    storage_available: int = 0  # Bytes available

    # Specializations
    specializations: Set[str] = field(default_factory=set)
    # e.g., {"malware_analysis", "network_traffic", "osint"}

    # Data holdings
    data_domains: Set[str] = field(default_factory=set)
    # e.g., {"threat_intel", "network_logs", "user_behavior"}

    # Hardware
    has_gpu: bool = False
    has_tpm: bool = False

    def can_handle(self, query_type: QueryType, domain: Optional[str] = None) -> bool:
        """Check if node can handle a specific query type"""
        if domain and domain not in self.data_domains:
            return False
        return True


@dataclass
class RegisteredNode:
    """A node registered with the hub"""
    node_id: str
    status: NodeStatus
    capabilities: NodeCapability

    # Network info
    endpoint: str  # How to reach this node
    public_key: bytes = b""

    # Health tracking
    last_heartbeat: Optional[datetime] = None
    failed_heartbeats: int = 0
    latency_ms: float = 0.0

    # Performance
    queries_processed: int = 0
    avg_response_time_ms: float = 0.0
    success_rate: float = 1.0

    # Trust
    trust_score: float = 1.0

    # Registration
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Sync state
    last_sync: Optional[datetime] = None
    sync_version: int = 0


@dataclass
class DistributedQuery:
    """A query distributed to nodes"""
    query_id: str
    query_type: QueryType
    priority: QueryPriority

    # Query content
    natural_language: Optional[str] = None  # Original NL query
    structured_query: Optional[Dict] = None  # Parsed structured query

    # Targeting
    target_nodes: List[str] = field(default_factory=list)  # Empty = all applicable
    required_capabilities: Set[str] = field(default_factory=set)
    required_domains: Set[str] = field(default_factory=set)

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    timeout_seconds: float = 30.0

    # Aggregation
    require_consensus: bool = False
    min_responses: int = 1

    # Context
    context: Dict[str, Any] = field(default_factory=dict)

    # Tracking
    responses_received: int = 0
    nodes_queried: List[str] = field(default_factory=list)


@dataclass
class NodeResponse:
    """Response from a single node"""
    node_id: str
    query_id: str

    # Result
    success: bool
    result: Any = None
    error: Optional[str] = None

    # Confidence
    confidence: float = 0.5

    # Timing
    response_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Supporting evidence
    evidence: List[Dict] = field(default_factory=list)


@dataclass
class AggregatedResponse:
    """Aggregated response from multiple nodes"""
    query_id: str

    # Results
    synthesized_result: Any = None
    individual_responses: List[NodeResponse] = field(default_factory=list)

    # Statistics
    nodes_queried: int = 0
    nodes_responded: int = 0
    consensus_reached: bool = False
    consensus_confidence: float = 0.0

    # Timing
    total_time_ms: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class QueryRouter:
    """
    Routes queries to appropriate nodes based on capabilities and load
    """

    def __init__(self, nodes: Dict[str, RegisteredNode]):
        self._nodes = nodes

    def route(self, query: DistributedQuery) -> List[str]:
        """
        Determine which nodes should receive this query

        Returns:
            List of node IDs to query
        """
        if query.target_nodes:
            # Explicit targeting
            return [
                nid for nid in query.target_nodes
                if nid in self._nodes and self._nodes[nid].status == NodeStatus.ONLINE
            ]

        candidates = []

        for node_id, node in self._nodes.items():
            if node.status != NodeStatus.ONLINE:
                continue

            # Check capabilities
            if query.required_capabilities:
                if not query.required_capabilities.issubset(node.capabilities.capabilities):
                    continue

            # Check domains
            if query.required_domains:
                if not query.required_domains.issubset(node.capabilities.data_domains):
                    continue

            # Score the node
            score = self._score_node(node, query)
            candidates.append((score, node_id))

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Return top candidates
        if query.require_consensus:
            # Need at least min_responses nodes
            return [nid for _, nid in candidates[:max(query.min_responses, 3)]]
        else:
            # For non-consensus, top 1-3 based on priority
            count = 1 if query.priority.value < QueryPriority.HIGH.value else 3
            return [nid for _, nid in candidates[:count]]

    def _score_node(self, node: RegisteredNode, query: DistributedQuery) -> float:
        """Score a node for routing priority"""
        score = 0.0

        # Trust score
        score += node.trust_score * 0.3

        # Performance (inverse of response time, capped)
        if node.avg_response_time_ms > 0:
            perf_score = min(1.0, 100 / node.avg_response_time_ms)
        else:
            perf_score = 0.5
        score += perf_score * 0.2

        # Success rate
        score += node.success_rate * 0.2

        # Recency of sync
        if node.last_sync:
            sync_age = (datetime.now(timezone.utc) - node.last_sync).total_seconds()
            sync_score = max(0, 1.0 - sync_age / 86400)  # Decay over 24h
        else:
            sync_score = 0.5
        score += sync_score * 0.15

        # Latency
        latency_score = max(0, 1.0 - node.latency_ms / 1000)  # 1s max
        score += latency_score * 0.15

        return score


class ResponseAggregator:
    """
    Aggregates responses from multiple nodes
    """

    def aggregate(self, query: DistributedQuery,
                  responses: List[NodeResponse]) -> AggregatedResponse:
        """
        Aggregate multiple node responses into a single response
        """
        if not responses:
            return AggregatedResponse(
                query_id=query.query_id,
                nodes_queried=len(query.nodes_queried),
                nodes_responded=0,
            )

        # Filter successful responses
        successful = [r for r in responses if r.success]

        # Calculate consensus
        consensus_reached = False
        consensus_confidence = 0.0
        synthesized = None

        if successful:
            # Simple consensus: majority agreement
            results = [r.result for r in successful]

            if query.require_consensus:
                # Check for agreement
                result_counts = defaultdict(list)
                for r in successful:
                    result_key = json.dumps(r.result, sort_keys=True, default=str)
                    result_counts[result_key].append(r)

                # Find majority
                majority = max(result_counts.values(), key=len)
                if len(majority) >= query.min_responses:
                    consensus_reached = True
                    consensus_confidence = sum(r.confidence for r in majority) / len(majority)
                    synthesized = majority[0].result
            else:
                # Take highest confidence result
                best = max(successful, key=lambda r: r.confidence)
                synthesized = best.result
                consensus_confidence = best.confidence

        # Calculate total time
        total_time = max(r.response_time_ms for r in responses) if responses else 0

        return AggregatedResponse(
            query_id=query.query_id,
            synthesized_result=synthesized,
            individual_responses=responses,
            nodes_queried=len(query.nodes_queried),
            nodes_responded=len(responses),
            consensus_reached=consensus_reached,
            consensus_confidence=consensus_confidence,
            total_time_ms=total_time,
        )


class HubOrchestrator:
    """
    Central Hub Orchestrator

    The master coordination point for the distributed brain network.
    All intelligence queries originate here and are distributed to nodes.

    Usage:
        hub = HubOrchestrator(hub_id="dsmil-central")

        # Register nodes
        hub.register_node(node_id, endpoint, capabilities)

        # Query the network
        response = await hub.query("What threats target our infrastructure?")

        # Propagate intelligence
        hub.propagate_intel(intel_report)
    """

    def __init__(self, hub_id: str = "dsmil-central", mesh_port: int = 8889,
                 use_mesh: bool = True, brain_interface: Any = None):
        """
        Initialize hub orchestrator

        Args:
            hub_id: Unique identifier for this hub
            mesh_port: Port for mesh network communication
            use_mesh: Whether to use dsmil-mesh for communication
        """
        self.hub_id = hub_id
        self.mesh_port = mesh_port
        self.use_mesh = use_mesh and MESH_AVAILABLE
        self._brain_interface = brain_interface

        # Node registry
        self._nodes: Dict[str, RegisteredNode] = {}
        self._node_lock = threading.RLock()

        # Query management
        self._active_queries: Dict[str, DistributedQuery] = {}
        self._query_responses: Dict[str, List[NodeResponse]] = defaultdict(list)
        self._response_events: Dict[str, asyncio.Event] = {}

        # Components
        self._router = QueryRouter(self._nodes)
        self._aggregator = ResponseAggregator()

        # Mesh network (if available)
        self._mesh: Optional[QuantumMesh] = None
        if self.use_mesh:
            self._init_mesh()

        # MEMSHADOW gateway
        from .memshadow_gateway import HubMemshadowGateway  # Local import to avoid cycles
        self._memshadow_gateway = HubMemshadowGateway(self.hub_id)
        if self._mesh:
            self._memshadow_gateway.mesh_send = self._mesh.send

        # Callbacks
        self.on_node_registered: Optional[Callable[[str], None]] = None
        self.on_node_offline: Optional[Callable[[str], None]] = None
        self.on_intel_received: Optional[Callable[[Dict], None]] = None

        # Background tasks
        self._running = False
        self._health_thread: Optional[threading.Thread] = None

        # Offline reconciliation queue
        self._reconciliation_queue: Dict[str, List[Dict]] = defaultdict(list)

        # Statistics
        self.stats = {
            "queries_processed": 0,
            "nodes_registered": 0,
            "intel_propagated": 0,
            "mesh_enabled": self.use_mesh,
        }

        logger.info(f"HubOrchestrator initialized: {hub_id} (mesh={'enabled' if self.use_mesh else 'disabled'})")

    @property
    def memshadow_gateway(self):
        return self._memshadow_gateway

    def _init_mesh(self):
        """Initialize mesh network"""
        if not MESH_AVAILABLE:
            logger.warning("Cannot initialize mesh - dsmil-mesh not available")
            return

        try:
            self._mesh = QuantumMesh(
                node_id=self.hub_id,
                port=self.mesh_port,
                config=MeshConfig(
                    node_id=self.hub_id,
                    port=self.mesh_port,
                    cluster_name="dsmil-brain",
                )
            )

            # Register message handlers
            self._mesh.on_message(MessageTypes.QUERY_RESPONSE, self._handle_mesh_response)
            self._mesh.on_message(MessageTypes.INTEL_REPORT, self._handle_mesh_intel)
            self._mesh.on_message(MessageTypes.HEARTBEAT, self._handle_mesh_heartbeat)

            # SHRINK psychological intelligence handlers
            self._mesh.on_message(MessageTypes.PSYCH_ASSESSMENT, self._handle_psych_intel)
            self._mesh.on_message(MessageTypes.DARK_TRIAD_UPDATE, self._handle_psych_intel)
            self._mesh.on_message(MessageTypes.RISK_UPDATE, self._handle_psych_intel)
            self._mesh.on_message(MessageTypes.NEURO_UPDATE, self._handle_psych_intel)
            self._mesh.on_message(MessageTypes.TMI_UPDATE, self._handle_psych_intel)
            self._mesh.on_message(MessageTypes.COGNITIVE_UPDATE, self._handle_psych_intel)
            self._mesh.on_message(MessageTypes.PSYCH_THREAT_ALERT, self._handle_psych_threat)
            self._mesh.on_message(MessageTypes.PSYCH_ANOMALY, self._handle_psych_threat)
            self._mesh.on_message(MessageTypes.PSYCH_RISK_THRESHOLD, self._handle_psych_threat)

            # Self-improvement propagation handlers
            self._mesh.on_message(MessageTypes.IMPROVEMENT_ANNOUNCE, self._handle_improvement_announce)
            self._mesh.on_message(MessageTypes.IMPROVEMENT_REQUEST, self._handle_improvement_request)
            self._mesh.on_message(MessageTypes.IMPROVEMENT_PAYLOAD, self._handle_improvement_payload)
            self._mesh.on_message(MessageTypes.IMPROVEMENT_ACK, self._handle_improvement_ack)
            self._mesh.on_message(MessageTypes.IMPROVEMENT_METRICS, self._handle_improvement_metrics)

            logger.info(f"Mesh network initialized on port {self.mesh_port}")

        except Exception as e:
            logger.error(f"Failed to initialize mesh: {e}")
            self._mesh = None
            self.use_mesh = False

    def _handle_mesh_response(self, data: bytes, peer_id: str):
        """Handle query response from mesh peer"""
        try:
            response_data = json.loads(data.decode())
            query_id = response_data.get("query_id")

            if query_id in self._active_queries:
                response = NodeResponse(
                    node_id=peer_id,
                    query_id=query_id,
                    success=response_data.get("success", False),
                    result=response_data.get("result"),
                    confidence=response_data.get("confidence", 0.5),
                    response_time_ms=response_data.get("response_time_ms", 0),
                )

                self._query_responses[query_id].append(response)

                # Signal that response received
                if query_id in self._response_events:
                    # Check if we have enough responses
                    query = self._active_queries[query_id]
                    if len(self._query_responses[query_id]) >= query.min_responses:
                        self._response_events[query_id].set()

        except Exception as e:
            logger.error(f"Error handling mesh response: {e}")

    def _handle_mesh_intel(self, data: bytes, peer_id: str):
        """Handle intel report from mesh peer"""
        try:
            intel_data = json.loads(data.decode())

            logger.info(f"Received intel from {peer_id}")

            if self.on_intel_received:
                self.on_intel_received(intel_data)

        except Exception as e:
            logger.error(f"Error handling mesh intel: {e}")

    def _handle_mesh_heartbeat(self, data: bytes, peer_id: str):
        """Handle heartbeat from mesh peer"""
        try:
            heartbeat = json.loads(data.decode())
            self.record_heartbeat(peer_id, heartbeat.get("latency_ms", 0))
        except Exception as e:
            logger.debug(f"Error handling heartbeat: {e}")

    def _handle_psych_intel(self, data: bytes, peer_id: str):
        """
        Handle psychological intelligence from SHRINK nodes

        Routes psych assessments to memory tiers and broadcasts to network.
        """
        try:
            handled_binary = False
            if PROTOCOL_AVAILABLE and len(data) >= HEADER_SIZE:
                ingest_result = ingest_memshadow_binary(data, brain_interface=getattr(self, "_brain_interface", None))
                if ingest_result.success and ingest_result.data:
                    handled_binary = True
                    for event in ingest_result.data:
                        logger.info("Received MEMSHADOW psych intel from %s (session=%s)", peer_id, event.get("session_id"))
                        self._store_psych_intel(event, peer_id)
                        if self._is_significant_psych_update(event):
                            self._broadcast_psych_intel(event, exclude_node=peer_id)
                    if self.on_intel_received:
                        self.on_intel_received({
                            "type": "psych_intel",
                            "data": ingest_result.data,
                            "source": peer_id,
                        })
            if not handled_binary:
                psych_data = json.loads(data.decode())
                logger.info(f"Received psych intel from {peer_id}: {psych_data.get('type', 'unknown')}")
                self._store_psych_intel(psych_data, peer_id)
                if self._is_significant_psych_update(psych_data):
                    self._broadcast_psych_intel(psych_data, exclude_node=peer_id)
                if self.on_intel_received:
                    self.on_intel_received({"type": "psych_intel", "data": psych_data, "source": peer_id})

        except Exception as e:
            logger.error(f"Error handling psych intel: {e}")

    def _handle_psych_threat(self, data: bytes, peer_id: str):
        """
        Handle psychological threat alerts (high priority)

        These are critical alerts requiring immediate attention.
        """
        try:
            # Parse as JSON (threat alerts are typically JSON for rapid parsing)
            threat_data = json.loads(data.decode())

            logger.warning(f"PSYCH THREAT from {peer_id}: {threat_data.get('threat_type', 'unknown')}")

            # Store with high priority
            self._store_psych_intel(threat_data, peer_id, priority="CRITICAL")

            # Immediate broadcast to all nodes
            self._broadcast_psych_intel(threat_data, priority=MessagePriority.CRITICAL if MessagePriority else None)

            # Trigger callback
            if self.on_intel_received:
                self.on_intel_received({"type": "psych_threat", "data": threat_data, "source": peer_id})

        except Exception as e:
            logger.error(f"Error handling psych threat: {e}")

    def _handle_improvement_announce(self, data: bytes, peer_id: str):
        """Handle improvement announcement from peer"""
        try:
            if IMPROVEMENT_AVAILABLE:
                announcement = ImprovementAnnouncement.unpack(data)
                logger.info(f"Improvement announced by {peer_id}: {announcement.improvement_id} "
                           f"({announcement.improvement_percentage:.1f}% improvement)")

                # Store announcement for nodes that may want to request it
                self._pending_improvements = getattr(self, '_pending_improvements', {})
                self._pending_improvements[announcement.improvement_id] = {
                    "announcement": announcement,
                    "source_node": peer_id,
                    "timestamp": datetime.now(timezone.utc),
                }

                # Relay to all nodes (except source)
                if self._mesh:
                    for node_id in self._nodes:
                        if node_id != peer_id:
                            try:
                                self._mesh.send(node_id, MessageTypes.IMPROVEMENT_ANNOUNCE, data)
                            except:
                                pass
            else:
                # Fallback to JSON
                announcement = json.loads(data.decode())
                logger.info(f"Improvement announced by {peer_id}: {announcement}")

        except Exception as e:
            logger.error(f"Error handling improvement announce: {e}")

    def _handle_improvement_request(self, data: bytes, peer_id: str):
        """Handle request for improvement data"""
        try:
            request = json.loads(data.decode())
            improvement_id = request.get("improvement_id")

            pending = getattr(self, '_pending_improvements', {})
            if improvement_id in pending:
                source_node = pending[improvement_id]["source_node"]

                # Forward request to source node (for P2P transfer)
                if self._mesh and source_node in self._nodes:
                    self._mesh.send(source_node, MessageTypes.IMPROVEMENT_REQUEST,
                                   json.dumps({"improvement_id": improvement_id, "requester": peer_id}).encode())

            logger.debug(f"Improvement request from {peer_id}: {improvement_id}")

        except Exception as e:
            logger.error(f"Error handling improvement request: {e}")

    def _handle_improvement_payload(self, data: bytes, peer_id: str):
        """Handle improvement payload (actual improvement data)"""
        try:
            if IMPROVEMENT_AVAILABLE:
                package = ImprovementPackage.unpack(data)
                logger.info(f"Improvement payload from {peer_id}: {package.improvement_id}")

                # Store for application
                self._received_improvements = getattr(self, '_received_improvements', {})
                self._received_improvements[package.improvement_id] = {
                    "package": package,
                    "source": peer_id,
                    "received_at": datetime.now(timezone.utc),
                }
            else:
                logger.debug(f"Improvement payload from {peer_id} (no handler)")

        except Exception as e:
            logger.error(f"Error handling improvement payload: {e}")

    def _handle_improvement_ack(self, data: bytes, peer_id: str):
        """Handle acknowledgment of improvement application"""
        try:
            if IMPROVEMENT_AVAILABLE:
                ack = ImprovementAck.unpack(data)
                logger.info(f"Improvement ACK from {peer_id}: {ack.improvement_id} (success={ack.success})")
            else:
                ack = json.loads(data.decode())
                logger.info(f"Improvement ACK from {peer_id}: {ack}")

        except Exception as e:
            logger.error(f"Error handling improvement ack: {e}")

    def _handle_improvement_metrics(self, data: bytes, peer_id: str):
        """Handle performance metrics share from peer"""
        try:
            metrics = json.loads(data.decode())
            logger.debug(f"Improvement metrics from {peer_id}: {metrics.get('improvement_id', 'unknown')}")

        except Exception as e:
            logger.error(f"Error handling improvement metrics: {e}")

    def _parse_psych_binary(self, data: bytes, header: 'MemshadowHeader') -> Dict:
        """Parse binary MEMSHADOW protocol psych event"""
        # Extract payload after header
        payload = data[HEADER_SIZE:HEADER_SIZE + header.payload_len]

        # For now, return minimal structure
        # Full parsing would use PsychEvent from dsmil_protocol
        return {
            "type": "psych_event",
            "message_type": header.message_type,
            "timestamp_ns": header.timestamp_ns,
            "batch_count": header.batch_count,
            "raw_payload_size": len(payload),
        }

    def _store_psych_intel(self, psych_data: Dict, source_node: str, priority: str = "NORMAL"):
        """Store psychological intelligence in memory tiers"""
        # This would integrate with the Brain's memory system
        # For now, just track in stats
        self._psych_intel_received = getattr(self, '_psych_intel_received', 0) + 1

        logger.debug(f"Stored psych intel from {source_node} (priority={priority})")

    def _is_significant_psych_update(self, psych_data: Dict) -> bool:
        """Check if psych update is significant enough to broadcast"""
        # Significant if: high risk, dark triad detected, or anomaly
        scores = psych_data.get("scores", {})
        if scores.get("espionage_exposure", 0) > 0.7:
            return True
        if scores.get("dark_triad_average", 0) > 0.7:
            return True
        if psych_data.get("anomaly_detected", False):
            return True
        return False

    def _broadcast_psych_intel(self, psych_data: Dict, exclude_node: str = None,
                               priority: Optional[int] = None):
        """Broadcast psychological intelligence to network"""
        if not self._mesh:
            return

        msg_data = json.dumps(psych_data).encode()

        for node_id in self._nodes:
            if node_id == exclude_node:
                continue
            try:
                self._mesh.send(node_id, MessageTypes.PSYCH_ASSESSMENT, msg_data)
            except Exception as e:
                logger.debug(f"Failed to broadcast psych intel to {node_id}: {e}")

    def start_mesh(self):
        """Start the mesh network"""
        if self._mesh:
            self._mesh.start()
            logger.info("Mesh network started")

    def stop_mesh(self):
        """Stop the mesh network"""
        if self._mesh:
            self._mesh.stop()
            logger.info("Mesh network stopped")

    def register_node(self, node_id: str, endpoint: str,
                     capabilities: NodeCapability,
                     public_key: bytes = b"") -> RegisteredNode:
        """
        Register a new node with the hub

        Args:
            node_id: Unique node identifier
            endpoint: Network endpoint to reach node
            capabilities: Node capabilities
            public_key: Node's public key for auth

        Returns:
            RegisteredNode instance
        """
        with self._node_lock:
            node = RegisteredNode(
                node_id=node_id,
                status=NodeStatus.ONLINE,
                capabilities=capabilities,
                endpoint=endpoint,
                public_key=public_key,
                last_heartbeat=datetime.now(timezone.utc),
            )

            self._nodes[node_id] = node
            self.stats["nodes_registered"] += 1

            # Check for pending reconciliation
            if node_id in self._reconciliation_queue:
                pending = self._reconciliation_queue.pop(node_id)
                logger.info(f"Node {node_id} has {len(pending)} pending sync items")

            if self.on_node_registered:
                self.on_node_registered(node_id)

            logger.info(f"Node registered: {node_id} at {endpoint}")
            return node

    def unregister_node(self, node_id: str, reason: str = "manual"):
        """Unregister a node from the hub"""
        with self._node_lock:
            if node_id in self._nodes:
                self._nodes[node_id].status = NodeStatus.TERMINATED
                del self._nodes[node_id]
                logger.info(f"Node unregistered: {node_id} ({reason})")

    def record_heartbeat(self, node_id: str, latency_ms: float = 0):
        """Record heartbeat from a node"""
        with self._node_lock:
            if node_id in self._nodes:
                node = self._nodes[node_id]
                node.last_heartbeat = datetime.now(timezone.utc)
                node.latency_ms = latency_ms
                node.failed_heartbeats = 0

                if node.status == NodeStatus.OFFLINE:
                    node.status = NodeStatus.ONLINE
                    logger.info(f"Node {node_id} came back online")

    def parse_query(self, natural_language: str) -> Dict:
        """
        Parse natural language query into structured format

        This would use NLP in a full implementation.
        For now, simple keyword-based parsing.
        """
        query_lower = natural_language.lower()

        structured = {
            "original": natural_language,
            "keywords": [],
            "intent": "search",
            "entities": [],
            "domains": set(),
        }

        # Detect intent
        if any(w in query_lower for w in ["predict", "forecast", "will", "future"]):
            structured["intent"] = "predict"
        elif any(w in query_lower for w in ["correlate", "relate", "connection", "link"]):
            structured["intent"] = "correlate"
        elif any(w in query_lower for w in ["analyze", "examine", "investigate"]):
            structured["intent"] = "analyze"
        elif any(w in query_lower for w in ["verify", "confirm", "check"]):
            structured["intent"] = "verify"

        # Extract keywords (simplified)
        words = natural_language.split()
        structured["keywords"] = [w for w in words if len(w) > 3]

        # Detect domains
        if any(w in query_lower for w in ["threat", "attack", "malware", "apt"]):
            structured["domains"].add("threat_intel")
        if any(w in query_lower for w in ["network", "traffic", "packet", "flow"]):
            structured["domains"].add("network_logs")
        if any(w in query_lower for w in ["user", "behavior", "activity"]):
            structured["domains"].add("user_behavior")

        return structured

    async def query(self, natural_language: str,
                   priority: QueryPriority = QueryPriority.NORMAL,
                   timeout: float = 30.0,
                   require_consensus: bool = False,
                   min_responses: int = 1,
                   context: Optional[Dict] = None) -> AggregatedResponse:
        """
        Distribute a query to the network

        Args:
            natural_language: The query in natural language
            priority: Query priority
            timeout: Timeout in seconds
            require_consensus: Require consensus among nodes
            min_responses: Minimum number of responses required
            context: Additional context

        Returns:
            AggregatedResponse with synthesized result
        """
        # Generate query ID
        query_id = hashlib.sha256(
            f"{natural_language}:{time.time()}".encode()
        ).hexdigest()[:16]

        # Parse query
        structured = self.parse_query(natural_language)

        # Map intent to query type
        intent_to_type = {
            "search": QueryType.SEARCH,
            "predict": QueryType.PREDICT,
            "correlate": QueryType.CORRELATE,
            "analyze": QueryType.ANALYZE,
            "verify": QueryType.VERIFY,
        }
        query_type = intent_to_type.get(structured["intent"], QueryType.SEARCH)

        # Create distributed query
        query = DistributedQuery(
            query_id=query_id,
            query_type=query_type,
            priority=priority,
            natural_language=natural_language,
            structured_query=structured,
            timeout_seconds=timeout,
            require_consensus=require_consensus,
            min_responses=min_responses,
            required_domains=structured.get("domains", set()),
            context=context or {},
        )

        # Route query
        target_nodes = self._router.route(query)
        query.nodes_queried = target_nodes

        if not target_nodes:
            logger.warning(f"No nodes available for query: {query_id}")
            return AggregatedResponse(
                query_id=query_id,
                nodes_queried=0,
                nodes_responded=0,
            )

        # Track query
        self._active_queries[query_id] = query

        # Dispatch to nodes (would be async network calls)
        responses = await self._dispatch_query(query, target_nodes)

        # Aggregate responses
        aggregated = self._aggregator.aggregate(query, responses)

        # Update statistics
        self.stats["queries_processed"] += 1

        # Update node performance metrics
        for response in responses:
            if response.node_id in self._nodes:
                node = self._nodes[response.node_id]
                node.queries_processed += 1
                # Update rolling average response time
                node.avg_response_time_ms = (
                    node.avg_response_time_ms * 0.9 + response.response_time_ms * 0.1
                )

        # Cleanup
        del self._active_queries[query_id]

        return aggregated

    async def _dispatch_query(self, query: DistributedQuery,
                              target_nodes: List[str]) -> List[NodeResponse]:
        """
        Dispatch query to target nodes via mesh network
        """
        # Use mesh network if available
        if self._mesh and self.use_mesh:
            return await self._dispatch_via_mesh(query, target_nodes)

        # Fallback to simulated mode
        return await self._dispatch_simulated(query, target_nodes)

    async def _dispatch_via_mesh(self, query: DistributedQuery,
                                  target_nodes: List[str]) -> List[NodeResponse]:
        """Dispatch query via mesh network"""
        # Prepare query message
        query_msg = json.dumps({
            "query_id": query.query_id,
            "type": query.query_type.name,
            "priority": query.priority.name,
            "natural_language": query.natural_language,
            "structured": query.structured_query,
            "timeout": query.timeout_seconds,
            "context": query.context,
        }).encode()

        # Create event for response collection
        self._response_events[query.query_id] = asyncio.Event()
        self._query_responses[query.query_id] = []

        # Send to target nodes (or broadcast if targeting all)
        if len(target_nodes) == len(self._nodes):
            # Broadcast to all
            self._mesh.broadcast(MessageTypes.BRAIN_QUERY, query_msg)
        else:
            # Send to specific nodes
            for node_id in target_nodes:
                try:
                    self._mesh.send(node_id, MessageTypes.BRAIN_QUERY, query_msg)
                except Exception as e:
                    logger.warning(f"Failed to send query to {node_id}: {e}")

        # Wait for responses with timeout
        try:
            await asyncio.wait_for(
                self._response_events[query.query_id].wait(),
                timeout=query.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.warning(f"Query {query.query_id} timed out")

        # Collect responses
        responses = self._query_responses.pop(query.query_id, [])
        self._response_events.pop(query.query_id, None)

        return responses

    async def _dispatch_simulated(self, query: DistributedQuery,
                                   target_nodes: List[str]) -> List[NodeResponse]:
        """Dispatch query in simulated mode (no mesh)"""
        responses = []

        for node_id in target_nodes:
            if node_id not in self._nodes:
                continue

            node = self._nodes[node_id]
            start_time = time.time()

            try:
                # Simulate a response
                result = {
                    "node_id": node_id,
                    "query_keywords": query.structured_query.get("keywords", []),
                    "data": f"Simulated response from {node_id}",
                    "mode": "simulated",
                }

                response_time = (time.time() - start_time) * 1000

                responses.append(NodeResponse(
                    node_id=node_id,
                    query_id=query.query_id,
                    success=True,
                    result=result,
                    confidence=0.7,
                    response_time_ms=response_time,
                ))

            except Exception as e:
                responses.append(NodeResponse(
                    node_id=node_id,
                    query_id=query.query_id,
                    success=False,
                    error=str(e),
                    response_time_ms=(time.time() - start_time) * 1000,
                ))

        return responses

    def propagate_intel(self, intel_report: Dict,
                       target_nodes: Optional[List[str]] = None,
                       priority: QueryPriority = QueryPriority.NORMAL):
        """
        Propagate intelligence to nodes via mesh network

        Args:
            intel_report: Intelligence report to propagate
            target_nodes: Specific nodes (None = all)
            priority: Propagation priority
        """
        # Prepare intel message
        intel_msg = json.dumps({
            "type": "intel_propagation",
            "priority": priority.name,
            "report": intel_report,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": self.hub_id,
        }).encode()

        # Use mesh if available
        if self._mesh and self.use_mesh:
            if target_nodes:
                for node_id in target_nodes:
                    try:
                        self._mesh.send(node_id, MessageTypes.KNOWLEDGE_UPDATE, intel_msg)
                    except Exception as e:
                        logger.warning(f"Failed to send intel to {node_id}: {e}")
                        self._queue_for_reconciliation(node_id, intel_report)
            else:
                # Broadcast to all
                self._mesh.broadcast(MessageTypes.KNOWLEDGE_UPDATE, intel_msg)
        else:
            # Queue for later or log
            with self._node_lock:
                targets = target_nodes or list(self._nodes.keys())
                for node_id in targets:
                    if node_id not in self._nodes:
                        continue
                    node = self._nodes[node_id]
                    if node.status != NodeStatus.ONLINE:
                        self._queue_for_reconciliation(node_id, intel_report)
                    else:
                        logger.debug(f"Propagating intel to {node_id} (simulated)")

        self.stats["intel_propagated"] += 1

    def _queue_for_reconciliation(self, node_id: str, intel_report: Dict):
        """Queue intel for reconciliation when node comes back online"""
        self._reconciliation_queue[node_id].append({
            "type": "intel",
            "data": intel_report,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_node_status(self, node_id: str) -> Optional[Dict]:
        """Get status of a specific node"""
        with self._node_lock:
            if node_id not in self._nodes:
                return None

            node = self._nodes[node_id]
            return {
                "node_id": node.node_id,
                "status": node.status.name,
                "last_heartbeat": node.last_heartbeat.isoformat() if node.last_heartbeat else None,
                "latency_ms": node.latency_ms,
                "trust_score": node.trust_score,
                "queries_processed": node.queries_processed,
                "success_rate": node.success_rate,
                "capabilities": list(node.capabilities.capabilities),
                "domains": list(node.capabilities.data_domains),
            }

    def get_all_nodes_status(self) -> List[Dict]:
        """Get status of all nodes"""
        with self._node_lock:
            return [
                self.get_node_status(nid)
                for nid in self._nodes
            ]

    def start_health_monitoring(self, interval: float = 10.0):
        """Start background health monitoring"""
        if self._running:
            return

        self._running = True

        def health_loop():
            while self._running:
                try:
                    self._check_node_health()
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                time.sleep(interval)

        self._health_thread = threading.Thread(target=health_loop, daemon=True)
        self._health_thread.start()
        logger.info("Health monitoring started")

    def stop_health_monitoring(self):
        """Stop health monitoring"""
        self._running = False
        if self._health_thread:
            self._health_thread.join(timeout=5.0)

    def _check_node_health(self):
        """Check health of all nodes"""
        with self._node_lock:
            now = datetime.now(timezone.utc)

            for node_id, node in self._nodes.items():
                if node.status == NodeStatus.TERMINATED:
                    continue

                if node.last_heartbeat:
                    elapsed = (now - node.last_heartbeat).total_seconds()

                    if elapsed > 60:  # 60 seconds without heartbeat
                        if node.status == NodeStatus.ONLINE:
                            node.status = NodeStatus.OFFLINE
                            node.failed_heartbeats += 1
                            logger.warning(f"Node {node_id} went offline")

                            if self.on_node_offline:
                                self.on_node_offline(node_id)

    def get_hub_stats(self) -> Dict:
        """Get hub statistics"""
        with self._node_lock:
            online_count = sum(
                1 for n in self._nodes.values()
                if n.status == NodeStatus.ONLINE
            )

            return {
                **self.stats,
                "hub_id": self.hub_id,
                "total_nodes": len(self._nodes),
                "online_nodes": online_count,
                "active_queries": len(self._active_queries),
                "pending_reconciliation": sum(
                    len(q) for q in self._reconciliation_queue.values()
                ),
            }

    async def handle_memshadow_message(self, header: MemshadowHeader, payload: bytes, source_node: str) -> Dict[str, Any]:
        """
        Delegate MEMSHADOW control-plane messages to the gateway.
        """
        return await self._memshadow_gateway.handle_memshadow_message(header, payload, source_node)


if __name__ == "__main__":
    print("Hub Orchestrator Self-Test")
    print("=" * 50)

    import asyncio

    hub = HubOrchestrator("test-hub")

    print(f"\n[1] Register Nodes")
    for i in range(3):
        cap = NodeCapability(
            node_id=f"node-{i}",
            capabilities={"search", "correlate"},
            data_domains={"threat_intel", "network_logs"},
            compute_power=1.0,
            has_gpu=(i == 0),
        )
        hub.register_node(f"node-{i}", f"localhost:800{i}", cap)

    status = hub.get_all_nodes_status()
    print(f"    Registered {len(status)} nodes")

    print(f"\n[2] Query Network")
    async def test_query():
        response = await hub.query(
            "What are the current threat indicators targeting our network?",
            priority=QueryPriority.HIGH,
            require_consensus=False,
        )
        return response

    response = asyncio.run(test_query())
    print(f"    Query ID: {response.query_id}")
    print(f"    Nodes queried: {response.nodes_queried}")
    print(f"    Nodes responded: {response.nodes_responded}")
    print(f"    Result: {response.synthesized_result}")

    print(f"\n[3] Propagate Intel")
    hub.propagate_intel({
        "type": "threat_indicator",
        "ioc": "192.168.1.100",
        "confidence": 0.85,
    })

    print(f"\n[4] Hub Stats")
    stats = hub.get_hub_stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 50)
    print("Hub Orchestrator test complete")

