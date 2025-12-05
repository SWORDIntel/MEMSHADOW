#!/usr/bin/env python3
"""
Spoke Client for DSMIL Brain Federation

Node client that:
- NO local NL interface (queries from hub only)
- Receives queries from hub
- Performs local correlation and analysis
- Returns results to hub
- Operates autonomously when hub offline
- Coordinates with peers when needed
"""

import asyncio
import hashlib
import threading
import logging
import time
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Set
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
import json

from config.memshadow_config import get_memshadow_config

logger = logging.getLogger(__name__)

# Try to import dsmil-mesh library
try:
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

# Try to import improvement types
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

try:
    protocol_path = Path(__file__).parent.parent.parent / "libs" / "memshadow-protocol" / "python"
    if protocol_path.exists() and str(protocol_path) not in sys.path:
        sys.path.insert(0, str(protocol_path))

    from dsmil_protocol import Priority as MemshadowPriority
    MEMSHADOW_PROTOCOL_AVAILABLE = True
except ImportError:
    MEMSHADOW_PROTOCOL_AVAILABLE = False
    MemshadowPriority = None  # type: ignore

try:
    from .memshadow_gateway import SpokeMemoryAdapter
    from ..memory.memory_sync_protocol import MemoryTier
    MEMSHADOW_ADAPTER_AVAILABLE = True
except ImportError:
    MEMSHADOW_ADAPTER_AVAILABLE = False
    logger.debug("SpokeMemoryAdapter not available")


class SpokeState(Enum):
    """State of the spoke node"""
    INITIALIZING = auto()
    CONNECTED = auto()      # Connected to hub
    DISCONNECTED = auto()   # Disconnected from hub
    AUTONOMOUS = auto()     # Operating independently
    PEER_MODE = auto()      # Coordinating with peers
    SYNCING = auto()        # Syncing with hub
    COMPROMISED = auto()    # Security issue detected


class QuerySource(Enum):
    """Source of a query"""
    HUB = auto()           # From central hub
    PEER = auto()          # From peer node
    INTERNAL = auto()      # Internal system query
    # NO USER - queries don't come from local users


@dataclass
class LocalCorrelation:
    """Result of local correlation"""
    correlation_id: str
    query_id: str

    # Results
    matches: List[Dict] = field(default_factory=list)
    relationships: List[Dict] = field(default_factory=list)
    patterns: List[Dict] = field(default_factory=list)

    # Confidence
    confidence: float = 0.5
    evidence_count: int = 0

    # Timing
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class QueryResponse:
    """Response to a query"""
    query_id: str
    node_id: str

    # Result
    success: bool
    correlation: Optional[LocalCorrelation] = None
    error: Optional[str] = None

    # Source tracking
    data_sources: List[str] = field(default_factory=list)

    # Confidence
    confidence: float = 0.5

    # Timing
    response_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "query_id": self.query_id,
            "node_id": self.node_id,
            "success": self.success,
            "confidence": self.confidence,
            "response_time_ms": self.response_time_ms,
            "correlation": {
                "matches": self.correlation.matches if self.correlation else [],
                "patterns": self.correlation.patterns if self.correlation else [],
            } if self.correlation else None,
            "error": self.error,
        }


class LocalCorrelator:
    """
    Performs local correlation on node's data

    This is where the actual intelligence analysis happens
    on each node's local data.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._local_data: Dict[str, List[Dict]] = {}
        self._indices: Dict[str, Dict] = {}

    def add_data(self, domain: str, data: List[Dict]):
        """Add data to local store"""
        if domain not in self._local_data:
            self._local_data[domain] = []
        self._local_data[domain].extend(data)
        self._build_index(domain)

    def _build_index(self, domain: str):
        """Build search index for domain"""
        self._indices[domain] = {}
        for i, item in enumerate(self._local_data.get(domain, [])):
            for key, value in item.items():
                if isinstance(value, str):
                    key_index = self._indices[domain].setdefault(key, {})
                    value_lower = value.lower()
                    if value_lower not in key_index:
                        key_index[value_lower] = []
                    key_index[value_lower].append(i)

    def correlate(self, query: Dict) -> LocalCorrelation:
        """
        Perform local correlation based on query

        Args:
            query: Structured query from hub

        Returns:
            LocalCorrelation result
        """
        start_time = time.time()

        correlation_id = hashlib.sha256(
            f"{self.node_id}:{query.get('query_id', '')}:{time.time()}".encode()
        ).hexdigest()[:16]

        matches = []
        relationships = []
        patterns = []

        # Search keywords in local data
        keywords = query.get("keywords", [])
        domains = query.get("domains", set()) or set(self._local_data.keys())

        for domain in domains:
            if domain not in self._local_data:
                continue

            domain_data = self._local_data[domain]

            # Search for keyword matches
            for keyword in keywords:
                keyword_lower = keyword.lower()

                for idx, item in enumerate(domain_data):
                    for key, value in item.items():
                        if isinstance(value, str) and keyword_lower in value.lower():
                            matches.append({
                                "domain": domain,
                                "index": idx,
                                "item": item,
                                "matched_field": key,
                                "matched_keyword": keyword,
                            })

        # Find relationships between matches
        if len(matches) > 1:
            for i, m1 in enumerate(matches[:-1]):
                for m2 in matches[i+1:]:
                    # Check for common fields
                    common = set(m1["item"].keys()) & set(m2["item"].keys())
                    for field in common:
                        if m1["item"][field] == m2["item"][field]:
                            relationships.append({
                                "type": "shared_value",
                                "field": field,
                                "value": m1["item"][field],
                                "items": [m1["index"], m2["index"]],
                            })

        # Detect patterns
        if matches:
            # Simple pattern: repeated values
            value_counts = {}
            for match in matches:
                for key, value in match["item"].items():
                    if isinstance(value, str):
                        value_counts[(key, value)] = value_counts.get((key, value), 0) + 1

            for (key, value), count in value_counts.items():
                if count >= 2:
                    patterns.append({
                        "type": "repeated_value",
                        "field": key,
                        "value": value,
                        "count": count,
                    })

        processing_time = (time.time() - start_time) * 1000

        # Calculate confidence based on evidence
        confidence = min(0.95, 0.3 + (len(matches) * 0.05) + (len(relationships) * 0.1))

        return LocalCorrelation(
            correlation_id=correlation_id,
            query_id=query.get("query_id", ""),
            matches=matches,
            relationships=relationships,
            patterns=patterns,
            confidence=confidence,
            evidence_count=len(matches),
            processing_time_ms=processing_time,
        )


class SpokeClient:
    """
    Spoke Node Client

    A node in the distributed brain network that:
    - Receives queries from the hub (no local NL interface)
    - Performs local correlation and analysis
    - Returns results to hub
    - Can operate autonomously or with peers when hub offline

    Usage:
        spoke = SpokeClient(node_id="node-001", hub_endpoint="hub.local:8000")

        # Connect to hub
        await spoke.connect()

        # Process incoming query (called by hub)
        response = await spoke.process_query(query)

        # Start autonomous mode if hub offline
        spoke.enter_autonomous_mode()
    """

    def __init__(self, node_id: str, hub_endpoint: str = "",
                 capabilities: Optional[Set[str]] = None,
                 data_domains: Optional[Set[str]] = None,
                 mesh_port: int = 8889,
                 use_mesh: bool = True):
        """
        Initialize spoke client

        Args:
            node_id: Unique identifier for this node
            hub_endpoint: Network endpoint of hub (legacy, use mesh instead)
            capabilities: Node capabilities
            data_domains: Data domains this node has
            mesh_port: Port for mesh network
            use_mesh: Whether to use dsmil-mesh for communication
        """
        self.node_id = node_id
        self.hub_endpoint = hub_endpoint
        self.capabilities = capabilities or {"search", "correlate"}
        self.data_domains = data_domains or set()
        self.mesh_port = mesh_port
        self.use_mesh = use_mesh and MESH_AVAILABLE

        # State
        self.state = SpokeState.INITIALIZING

        # Local correlator
        self._correlator = LocalCorrelator(node_id)

        # Hub connection
        self._hub_connected = False
        self._last_hub_contact: Optional[datetime] = None

        # Peer connections
        self._peers: Dict[str, Dict] = {}

        # Offline queue (queries/intel to sync when reconnected)
        self._offline_queue: List[Dict] = []

        # Mesh network
        self._mesh: Optional[QuantumMesh] = None
        if self.use_mesh:
            self._init_mesh()

        # Heartbeat
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False
        self._heartbeat_interval = 5.0  # seconds

        # Callbacks
        self.on_state_change: Optional[Callable[[SpokeState], None]] = None
        self.on_hub_reconnect: Optional[Callable[[], None]] = None

        # Statistics
        self.stats = {
            "queries_processed": 0,
            "correlations_performed": 0,
            "intel_received": 0,
            "autonomous_operations": 0,
            "mesh_enabled": self.use_mesh,
        }

        self._memshadow_config = get_memshadow_config()
        self._memshadow_enabled = MEMSHADOW_ADAPTER_AVAILABLE and self._memshadow_config.enable_shrink_ingest
        self._memshadow_adapter: Optional[SpokeMemoryAdapter] = None
        self._memshadow_tiers: Dict[MemoryTier, Any] = {}
        if MEMSHADOW_ADAPTER_AVAILABLE:
            hub_id = self.hub_endpoint or "hub"
            self._memshadow_adapter = SpokeMemoryAdapter(self.node_id, hub_node_id=hub_id)

        logger.info(f"SpokeClient initialized: {node_id} (mesh={'enabled' if self.use_mesh else 'disabled'})")

    def _init_mesh(self):
        """Initialize mesh network"""
        if not MESH_AVAILABLE:
            logger.warning("Cannot initialize mesh - dsmil-mesh not available")
            return

        try:
            self._mesh = QuantumMesh(
                node_id=self.node_id,
                port=self.mesh_port,
                config=MeshConfig(
                    node_id=self.node_id,
                    port=self.mesh_port,
                    cluster_name="dsmil-brain",
                )
            )

            # Register message handlers
            self._mesh.on_message(MessageTypes.BRAIN_QUERY, self._handle_mesh_query)
            self._mesh.on_message(MessageTypes.KNOWLEDGE_UPDATE, self._handle_mesh_intel)
            self._mesh.on_message(MessageTypes.VECTOR_SYNC, self._handle_mesh_sync)

            # P2P improvement handlers
            self._mesh.on_message(MessageTypes.IMPROVEMENT_ANNOUNCE, self._handle_improvement_announce)
            self._mesh.on_message(MessageTypes.IMPROVEMENT_REQUEST, self._handle_improvement_request)
            self._mesh.on_message(MessageTypes.IMPROVEMENT_PAYLOAD, self._handle_improvement_payload)
            self._mesh.on_message(MessageTypes.IMPROVEMENT_ACK, self._handle_improvement_ack)
            self._mesh.on_message(MessageTypes.IMPROVEMENT_METRICS, self._handle_improvement_metrics)

            # SHRINK psychological intel handlers
            self._mesh.on_message(MessageTypes.PSYCH_ASSESSMENT, self._handle_psych_intel)
            self._mesh.on_message(MessageTypes.DARK_TRIAD_UPDATE, self._handle_psych_intel)
            self._mesh.on_message(MessageTypes.RISK_UPDATE, self._handle_psych_intel)
            self._mesh.on_message(MessageTypes.PSYCH_THREAT_ALERT, self._handle_psych_threat)

            logger.info(f"Mesh network initialized on port {self.mesh_port}")

        except Exception as e:
            logger.error(f"Failed to initialize mesh: {e}")
            self._mesh = None
            self.use_mesh = False
        else:
            if self._memshadow_adapter and self._mesh:
                self._memshadow_adapter.mesh_send = self._mesh.send

    def _handle_mesh_query(self, data: bytes, peer_id: str):
        """Handle incoming query from mesh (from hub)"""
        try:
            query_data = json.loads(data.decode())
            logger.info(f"Received query from hub via mesh: {query_data.get('query_id', 'unknown')}")

            # Process query asynchronously
            asyncio.create_task(self._process_mesh_query(query_data, peer_id))

        except Exception as e:
            logger.error(f"Error handling mesh query: {e}")

    async def _process_mesh_query(self, query_data: Dict, source_peer: str):
        """Process a query received via mesh"""
        start_time = time.time()

        try:
            # Perform local correlation
            correlation = self._correlator.correlate(query_data)

            # Build response
            response = QueryResponse(
                query_id=query_data.get("query_id", ""),
                node_id=self.node_id,
                success=True,
                correlation=correlation,
                confidence=correlation.confidence,
                response_time_ms=(time.time() - start_time) * 1000,
            )

            # Send response back via mesh
            if self._mesh:
                response_msg = json.dumps({
                    "query_id": response.query_id,
                    "node_id": response.node_id,
                    "success": response.success,
                    "result": response.to_dict(),
                    "confidence": response.confidence,
                    "response_time_ms": response.response_time_ms,
                }).encode()

                self._mesh.send(source_peer, MessageTypes.QUERY_RESPONSE, response_msg)

            self.stats["queries_processed"] += 1
            self.stats["correlations_performed"] += 1

        except Exception as e:
            logger.error(f"Error processing mesh query: {e}")

            # Send error response
            if self._mesh:
                error_msg = json.dumps({
                    "query_id": query_data.get("query_id", ""),
                    "node_id": self.node_id,
                    "success": False,
                    "error": str(e),
                }).encode()
                self._mesh.send(source_peer, MessageTypes.QUERY_RESPONSE, error_msg)

    def _handle_mesh_intel(self, data: bytes, peer_id: str):
        """Handle intel update from hub via mesh"""
        try:
            intel_data = json.loads(data.decode())
            logger.info(f"Received intel update from {peer_id}")

            # Store intel locally
            report = intel_data.get("report", {})
            # Would update local knowledge base here

            self.stats["intel_received"] += 1
            self._last_hub_contact = datetime.now(timezone.utc)

        except Exception as e:
            logger.error(f"Error handling mesh intel: {e}")

    def _handle_mesh_sync(self, data: bytes, peer_id: str):
        """Handle vector sync from mesh"""
        try:
            sync_data = json.loads(data.decode())
            logger.info(f"Received sync request from {peer_id}")
            # Handle vector synchronization
        except Exception as e:
            logger.error(f"Error handling mesh sync: {e}")

    def _handle_improvement_announce(self, data: bytes, peer_id: str):
        """
        Handle improvement announcement from peer (P2P)

        Evaluates if improvement is compatible and wanted, then requests it.
        """
        try:
            if IMPROVEMENT_AVAILABLE:
                announcement = ImprovementAnnouncement.unpack(data)
                logger.info(f"Improvement announced by peer {peer_id}: {announcement.improvement_id} "
                           f"({announcement.improvement_percentage:.1f}% improvement)")

                # Store for potential request
                self._pending_improvements = getattr(self, '_pending_improvements', {})
                self._pending_improvements[announcement.improvement_id] = {
                    "announcement": announcement,
                    "source_peer": peer_id,
                    "timestamp": datetime.now(timezone.utc),
                }

                # Auto-request if significant improvement and compatible
                if announcement.improvement_percentage >= 10.0:  # 10%+ improvement
                    self._request_improvement(announcement.improvement_id, peer_id)
            else:
                announcement = json.loads(data.decode())
                logger.info(f"Improvement announced by peer {peer_id}: {announcement}")

        except Exception as e:
            logger.error(f"Error handling improvement announce from {peer_id}: {e}")

    def _handle_improvement_request(self, data: bytes, peer_id: str):
        """Handle request for improvement we announced"""
        try:
            request = json.loads(data.decode())
            improvement_id = request.get("improvement_id")

            # If we have this improvement, send it
            local_improvements = getattr(self, '_local_improvements', {})
            if improvement_id in local_improvements:
                package = local_improvements[improvement_id]
                if self._mesh:
                    self._mesh.send(peer_id, MessageTypes.IMPROVEMENT_PAYLOAD, package.pack())
                    logger.info(f"Sent improvement {improvement_id} to {peer_id}")

        except Exception as e:
            logger.error(f"Error handling improvement request from {peer_id}: {e}")

    def _handle_improvement_payload(self, data: bytes, peer_id: str):
        """Handle improvement payload (actual data)"""
        try:
            if IMPROVEMENT_AVAILABLE:
                package = ImprovementPackage.unpack(data)
                logger.info(f"Received improvement payload from {peer_id}: {package.improvement_id}")

                # Apply improvement
                success = self._apply_improvement(package)

                # Send ACK
                if self._mesh:
                    ack = ImprovementAck(
                        improvement_id=package.improvement_id,
                        success=success,
                        applied_at=datetime.now(timezone.utc),
                        new_metrics=self._get_current_metrics() if success else None,
                    )
                    self._mesh.send(peer_id, MessageTypes.IMPROVEMENT_ACK, ack.pack())

        except Exception as e:
            logger.error(f"Error handling improvement payload from {peer_id}: {e}")

    def _handle_improvement_ack(self, data: bytes, peer_id: str):
        """Handle acknowledgment from peer that applied our improvement"""
        try:
            if IMPROVEMENT_AVAILABLE:
                ack = ImprovementAck.unpack(data)
                logger.info(f"Improvement ACK from {peer_id}: {ack.improvement_id} (success={ack.success})")

                # Track adoption
                self._improvement_adoptions = getattr(self, '_improvement_adoptions', {})
                self._improvement_adoptions[ack.improvement_id] = \
                    self._improvement_adoptions.get(ack.improvement_id, 0) + (1 if ack.success else 0)

        except Exception as e:
            logger.error(f"Error handling improvement ack from {peer_id}: {e}")

    def _handle_improvement_metrics(self, data: bytes, peer_id: str):
        """Handle performance metrics share from peer"""
        try:
            metrics = json.loads(data.decode())
            logger.debug(f"Metrics from peer {peer_id}: {metrics.get('improvement_id', 'unknown')}")
        except Exception as e:
            logger.error(f"Error handling improvement metrics from {peer_id}: {e}")

    def _handle_psych_intel(self, data: bytes, peer_id: str):
        """Handle psychological intelligence from network"""
        try:
            psych_data = json.loads(data.decode())
            logger.debug(f"Received psych intel from {peer_id}")

            # Store in local knowledge base
            self._correlator.add_data("psych_intel", [psych_data])
            self.stats["intel_received"] += 1

        except Exception as e:
            logger.error(f"Error handling psych intel from {peer_id}: {e}")

    def _handle_psych_threat(self, data: bytes, peer_id: str):
        """Handle psychological threat alert (high priority)"""
        try:
            threat_data = json.loads(data.decode())
            logger.warning(f"PSYCH THREAT from {peer_id}: {threat_data.get('threat_type', 'unknown')}")

            # Store as high-priority intel
            threat_data["_priority"] = "CRITICAL"
            self._correlator.add_data("psych_threats", [threat_data])

        except Exception as e:
            logger.error(f"Error handling psych threat from {peer_id}: {e}")

    def _request_improvement(self, improvement_id: str, peer_id: str):
        """Request an improvement from a peer"""
        if self._mesh:
            request = json.dumps({"improvement_id": improvement_id}).encode()
            self._mesh.send(peer_id, MessageTypes.IMPROVEMENT_REQUEST, request)
            logger.info(f"Requested improvement {improvement_id} from {peer_id}")

    def _apply_improvement(self, package: 'ImprovementPackage') -> bool:
        """Apply an improvement package locally"""
        try:
            # TODO: Implement actual improvement application
            # This would update model weights, config values, or learned patterns
            logger.info(f"Applied improvement {package.improvement_id} ({package.improvement_type.name})")
            return True
        except Exception as e:
            logger.error(f"Failed to apply improvement: {e}")
            return False

    def _get_current_metrics(self) -> Dict:
        """Get current performance metrics"""
        return {
            "queries_processed": self.stats.get("queries_processed", 0),
            "correlations_performed": self.stats.get("correlations_performed", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def announce_improvement(self, improvement: 'ImprovementPackage'):
        """
        Announce a local improvement to all peers (P2P broadcast)

        Called when local improvement tracker detects significant improvement.
        """
        if not IMPROVEMENT_AVAILABLE:
            return

        # Store locally
        self._local_improvements = getattr(self, '_local_improvements', {})
        self._local_improvements[improvement.improvement_id] = improvement

        # Create announcement
        announcement = ImprovementAnnouncement(
            improvement_id=improvement.improvement_id,
            improvement_type=improvement.improvement_type,
            improvement_percentage=improvement.improvement_percentage,
            compatibility_level=improvement.compatibility_level,
            size_bytes=len(improvement.pack()),
            source_node=self.node_id,
        )

        # Broadcast to all peers
        if self._mesh:
            for peer_id in self._peers:
                try:
                    self._mesh.send(peer_id, MessageTypes.IMPROVEMENT_ANNOUNCE, announcement.pack())
                except:
                    pass

            logger.info(f"Announced improvement {improvement.improvement_id} to {len(self._peers)} peers")

    def start_mesh(self):
        """Start the mesh network"""
        if self._mesh:
            self._mesh.start()
            self._hub_connected = True
            self._set_state(SpokeState.CONNECTED)
            logger.info("Mesh network started")

    def stop_mesh(self):
        """Stop the mesh network"""
        if self._mesh:
            self._mesh.stop()
            self._hub_connected = False
            logger.info("Mesh network stopped")

    def _set_state(self, new_state: SpokeState):
        """Update state and trigger callback"""
        old_state = self.state
        self.state = new_state

        if self.on_state_change:
            self.on_state_change(new_state)

        logger.info(f"State change: {old_state.name} -> {new_state.name}")

    async def connect(self) -> bool:
        """
        Connect to the central hub via mesh network

        Returns:
            True if connected successfully
        """
        try:
            # Use mesh network if available
            if self._mesh and self.use_mesh:
                self._mesh.start()
                self._hub_connected = True
                self._last_hub_contact = datetime.now(timezone.utc)
                self._set_state(SpokeState.CONNECTED)

                # Start heartbeat
                self._start_heartbeat()

                # Sync offline queue if any
                if self._offline_queue:
                    await self._sync_offline_queue()

                logger.info(f"Connected to hub via mesh network")
                return True
            else:
                # Legacy endpoint connection (simulated)
                self._hub_connected = True
                self._last_hub_contact = datetime.now(timezone.utc)
                self._set_state(SpokeState.CONNECTED)

                # Start heartbeat
                self._start_heartbeat()

                # Sync offline queue if any
                if self._offline_queue:
                    await self._sync_offline_queue()

                logger.info(f"Connected to hub at {self.hub_endpoint} (simulated)")
                return True

        except Exception as e:
            logger.error(f"Failed to connect to hub: {e}")
            self._hub_connected = False
            self._set_state(SpokeState.DISCONNECTED)
            return False

    async def disconnect(self):
        """Disconnect from hub"""
        self._hub_connected = False
        self._stop_heartbeat()
        self._set_state(SpokeState.DISCONNECTED)
        logger.info("Disconnected from hub")

    async def process_query(self, query: Dict, source: QuerySource = QuerySource.HUB) -> QueryResponse:
        """
        Process an incoming query

        Args:
            query: Structured query from hub or peer
            source: Source of the query

        Returns:
            QueryResponse with results
        """
        start_time = time.time()

        # Validate source - we don't accept user queries
        if source not in (QuerySource.HUB, QuerySource.PEER, QuerySource.INTERNAL):
            return QueryResponse(
                query_id=query.get("query_id", ""),
                node_id=self.node_id,
                success=False,
                error="Invalid query source",
            )

        try:
            # Perform local correlation
            correlation = self._correlator.correlate(query)

            response_time = (time.time() - start_time) * 1000

            self.stats["queries_processed"] += 1
            self.stats["correlations_performed"] += 1

            return QueryResponse(
                query_id=query.get("query_id", ""),
                node_id=self.node_id,
                success=True,
                correlation=correlation,
                confidence=correlation.confidence,
                response_time_ms=response_time,
                data_sources=list(self.data_domains),
            )

        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return QueryResponse(
                query_id=query.get("query_id", ""),
                node_id=self.node_id,
                success=False,
                error=str(e),
                response_time_ms=(time.time() - start_time) * 1000,
            )

    async def receive_intel(self, intel_report: Dict) -> bool:
        """
        Receive intelligence from hub

        Args:
            intel_report: Intelligence report to ingest

        Returns:
            True if processed successfully
        """
        try:
            # Add to local data
            domain = intel_report.get("domain", "threat_intel")
            data = intel_report.get("data", intel_report)

            self._correlator.add_data(domain, [data])
            self.stats["intel_received"] += 1

            logger.debug(f"Received intel: {intel_report.get('type', 'unknown')}")
            return True

        except Exception as e:
            logger.error(f"Intel processing error: {e}")
            return False

    def add_local_data(self, domain: str, data: List[Dict]):
        """Add data to local storage for correlation"""
        self._correlator.add_data(domain, data)
        if domain not in self.data_domains:
            self.data_domains.add(domain)

    def enable_memshadow_sync(self, enable: bool):
        """Toggle participation in MEMSHADOW sync."""
        self._memshadow_enabled = enable and MEMSHADOW_ADAPTER_AVAILABLE

    def register_memshadow_tier(self, tier: Any, memory_instance: Any) -> bool:
        """Register a local memory tier for MEMSHADOW sync."""
        if not self._memshadow_adapter:
            return False
        if isinstance(tier, MemoryTier):
            resolved_tier = tier
        else:
            try:
                resolved_tier = MemoryTier[tier]
            except Exception:
                logger.warning("Unknown MemoryTier %s for MEMSHADOW registration", tier)
                return False
        self._memshadow_tiers[resolved_tier] = memory_instance
        return self._memshadow_adapter.register_tier(resolved_tier, memory_instance)

    async def sync_memshadow_tier(self, tier: MemoryTier, priority: Optional[Any] = None) -> Dict[str, Any]:
        """Trigger a MEMSHADOW sync for a given tier."""
        if not self._memshadow_enabled or not self._memshadow_adapter:
            return {"status": "disabled"}
        if not MEMSHADOW_PROTOCOL_AVAILABLE:
            return {"status": "protocol_unavailable"}
        priority = priority or (MemshadowPriority.NORMAL if MemshadowPriority else None)
        if priority is None:
            return {"status": "priority_unavailable"}
        return await self._memshadow_adapter.sync_tier(tier, priority)

    def enter_autonomous_mode(self):
        """
        Enter autonomous operation mode

        Called when hub is unreachable and node needs to
        operate independently.
        """
        self._set_state(SpokeState.AUTONOMOUS)
        self.stats["autonomous_operations"] += 1
        logger.warning("Entered autonomous mode - hub unreachable")

    def enter_peer_mode(self, peers: List[Dict]):
        """
        Enter peer coordination mode

        Args:
            peers: List of peer nodes to coordinate with
        """
        for peer in peers:
            self._peers[peer["node_id"]] = peer

        self._set_state(SpokeState.PEER_MODE)
        logger.info(f"Entered peer mode with {len(peers)} peers")

    async def query_peers(self, query: Dict) -> List[QueryResponse]:
        """
        Query peer nodes when in peer mode

        Args:
            query: Query to send to peers

        Returns:
            List of responses from peers
        """
        if self.state != SpokeState.PEER_MODE:
            return []

        responses = []
        for peer_id, peer_info in self._peers.items():
            try:
                # Would do actual network call here
                # For now, simulate
                responses.append(QueryResponse(
                    query_id=query.get("query_id", ""),
                    node_id=peer_id,
                    success=True,
                    confidence=0.5,
                ))
            except Exception as e:
                logger.error(f"Peer query failed for {peer_id}: {e}")

        return responses

    async def _sync_offline_queue(self):
        """Sync queued items when reconnecting to hub"""
        if not self._offline_queue:
            return

        self._set_state(SpokeState.SYNCING)

        logger.info(f"Syncing {len(self._offline_queue)} queued items")

        synced = []
        for item in self._offline_queue:
            try:
                # Would send to hub here
                synced.append(item)
            except Exception as e:
                logger.error(f"Sync failed for item: {e}")

        # Remove synced items
        for item in synced:
            self._offline_queue.remove(item)

        self._set_state(SpokeState.CONNECTED)

    def _start_heartbeat(self):
        """Start heartbeat thread"""
        if self._running:
            return

        self._running = True

        def heartbeat_loop():
            while self._running:
                try:
                    self._send_heartbeat()
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    self._check_hub_status()

                time.sleep(self._heartbeat_interval)

        self._heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat(self):
        """Stop heartbeat thread"""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5.0)

    def _send_heartbeat(self):
        """Send heartbeat to hub"""
        if not self._hub_connected:
            return

        # Would do actual network call here
        # Update last contact time on success
        self._last_hub_contact = datetime.now(timezone.utc)

    def _check_hub_status(self):
        """Check if hub is still reachable"""
        if self._last_hub_contact:
            elapsed = (datetime.now(timezone.utc) - self._last_hub_contact).total_seconds()

            if elapsed > 30:  # 30 seconds without contact
                self._hub_connected = False

                if self.state == SpokeState.CONNECTED:
                    # Enter autonomous or peer mode
                    if self._peers:
                        self.enter_peer_mode(list(self._peers.values()))
                    else:
                        self.enter_autonomous_mode()

    def get_status(self) -> Dict:
        """Get current status"""
        memshadow_status = {
            "enabled": self._memshadow_enabled,
            "registered_tiers": [t.name for t in self._memshadow_tiers.keys()],
        }
        if self._memshadow_adapter:
            memshadow_status["adapter"] = self._memshadow_adapter.get_stats()

        return {
            "node_id": self.node_id,
            "state": self.state.name,
            "hub_connected": self._hub_connected,
            "last_hub_contact": self._last_hub_contact.isoformat() if self._last_hub_contact else None,
            "capabilities": list(self.capabilities),
            "data_domains": list(self.data_domains),
            "peer_count": len(self._peers),
            "offline_queue_size": len(self._offline_queue),
            "stats": self.stats,
            "memshadow": memshadow_status,
        }

    def get_capabilities_report(self) -> Dict:
        """Get capabilities for registration with hub"""
        return {
            "node_id": self.node_id,
            "capabilities": list(self.capabilities),
            "data_domains": list(self.data_domains),
            "has_gpu": False,  # Would detect
            "has_tpm": False,  # Would detect
        }


if __name__ == "__main__":
    print("Spoke Client Self-Test")
    print("=" * 50)

    import asyncio

    spoke = SpokeClient(
        node_id="test-spoke-001",
        hub_endpoint="localhost:8000",
        capabilities={"search", "correlate", "analyze"},
        data_domains={"threat_intel", "network_logs"},
    )

    print(f"\n[1] Add Local Data")
    spoke.add_local_data("threat_intel", [
        {"ioc": "192.168.1.100", "type": "ip", "threat": "malware"},
        {"ioc": "evil.com", "type": "domain", "threat": "c2"},
        {"ioc": "192.168.1.100", "type": "ip", "threat": "botnet"},
    ])
    spoke.add_local_data("network_logs", [
        {"src": "10.0.0.1", "dst": "192.168.1.100", "action": "blocked"},
        {"src": "10.0.0.2", "dst": "evil.com", "action": "allowed"},
    ])
    print(f"    Added data to 2 domains")

    print(f"\n[2] Process Query")
    async def test_query():
        response = await spoke.process_query({
            "query_id": "test-001",
            "keywords": ["192.168.1.100", "malware"],
            "domains": {"threat_intel", "network_logs"},
        })
        return response

    response = asyncio.run(test_query())
    print(f"    Success: {response.success}")
    print(f"    Confidence: {response.confidence:.2f}")
    if response.correlation:
        print(f"    Matches: {len(response.correlation.matches)}")
        print(f"    Patterns: {len(response.correlation.patterns)}")
        print(f"    Relationships: {len(response.correlation.relationships)}")

    print(f"\n[3] Receive Intel")
    asyncio.run(spoke.receive_intel({
        "type": "threat_indicator",
        "ioc": "newbad.com",
        "threat": "phishing",
    }))

    print(f"\n[4] Autonomous Mode")
    spoke.enter_autonomous_mode()
    print(f"    State: {spoke.state.name}")

    print(f"\n[5] Status")
    status = spoke.get_status()
    for key, value in status.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 50)
    print("Spoke Client test complete")

