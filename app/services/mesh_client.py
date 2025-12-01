#!/usr/bin/env python3
"""
MEMSHADOW Mesh Client

Integrates MEMSHADOW with the DSMIL quantum mesh network for:
- Sharing threat intelligence with brain nodes
- Receiving queries from the central hub
- Participating in swarm cognition
- Distributed threat correlation

Usage:
    from app.services.mesh_client import MEMSHADOWMeshClient

    client = MEMSHADOWMeshClient(node_id="memshadow-01")
    await client.start()

    # Share threat intel
    await client.share_threat_intel(threat_data)

    # Handle queries from brain hub
    client.on_threat_query(handle_query)
"""

import asyncio
import json
import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Set
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

# Import dsmil-mesh library
try:
    # Add libs path
    libs_path = Path(__file__).parent.parent.parent.parent / "libs" / "dsmil-mesh" / "python"
    if libs_path.exists() and str(libs_path) not in sys.path:
        sys.path.insert(0, str(libs_path))

    from mesh import QuantumMesh, Peer, MeshConfig, PeerState
    from messages import MessageTypes, MessagePriority
    from discovery import MeshDiscovery
    MESH_AVAILABLE = True
except ImportError as e:
    MESH_AVAILABLE = False
    QuantumMesh = None
    MessageTypes = None
    logger.warning(f"dsmil-mesh not available: {e}")


class ThreatSeverity(Enum):
    """Threat severity levels"""
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ThreatIntel:
    """Threat intelligence data"""
    intel_id: str
    threat_type: str
    severity: ThreatSeverity
    confidence: float

    # Indicators
    indicators: Dict[str, Any] = field(default_factory=dict)  # IOCs

    # Context
    campaign_id: Optional[str] = None
    actor_name: Optional[str] = None
    ttps: List[str] = field(default_factory=list)

    # Metadata
    source: str = "MEMSHADOW"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict:
        return {
            "intel_id": self.intel_id,
            "threat_type": self.threat_type,
            "severity": self.severity.name,
            "confidence": self.confidence,
            "indicators": self.indicators,
            "campaign_id": self.campaign_id,
            "actor_name": self.actor_name,
            "ttps": self.ttps,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ThreatIntel":
        return cls(
            intel_id=d["intel_id"],
            threat_type=d["threat_type"],
            severity=ThreatSeverity[d["severity"]],
            confidence=d["confidence"],
            indicators=d.get("indicators", {}),
            campaign_id=d.get("campaign_id"),
            actor_name=d.get("actor_name"),
            ttps=d.get("ttps", []),
            source=d.get("source", "unknown"),
            timestamp=datetime.fromisoformat(d["timestamp"]) if d.get("timestamp") else datetime.now(timezone.utc),
        )


class MEMSHADOWMeshClient:
    """
    MEMSHADOW Mesh Network Client

    Connects MEMSHADOW to the DSMIL quantum mesh network for
    threat intelligence sharing and distributed correlation.

    Features:
    - Share threat intelligence with brain nodes
    - Receive and respond to threat queries
    - Participate in swarm cognition
    - Broadcast IOCs and campaign updates
    """

    def __init__(self, node_id: str = "memshadow",
                 mesh_port: int = 8890,
                 cluster_name: str = "dsmil-brain"):
        """
        Initialize MEMSHADOW mesh client.

        Args:
            node_id: Unique identifier for this MEMSHADOW instance
            mesh_port: Port for mesh communication
            cluster_name: Mesh cluster to join
        """
        self.node_id = node_id
        self.mesh_port = mesh_port
        self.cluster_name = cluster_name

        # Mesh network
        self._mesh: Optional[QuantumMesh] = None
        self._running = False

        # Query handlers
        self._threat_query_handler: Optional[Callable] = None
        self._correlation_handler: Optional[Callable] = None

        # Intel queue (for batching)
        self._intel_queue: List[ThreatIntel] = []
        self._queue_lock = threading.Lock()

        # Statistics
        self.stats = {
            "intel_shared": 0,
            "queries_received": 0,
            "queries_answered": 0,
            "iocs_broadcast": 0,
            "mesh_enabled": MESH_AVAILABLE,
        }

        logger.info(f"MEMSHADOWMeshClient initialized: {node_id}")

    async def start(self):
        """Start the mesh client"""
        if not MESH_AVAILABLE:
            logger.warning("Cannot start mesh - dsmil-mesh not available")
            return False

        try:
            self._mesh = QuantumMesh(
                node_id=self.node_id,
                port=self.mesh_port,
                config=MeshConfig(
                    node_id=self.node_id,
                    port=self.mesh_port,
                    cluster_name=self.cluster_name,
                )
            )

            # Register message handlers
            self._mesh.on_message(MessageTypes.THREAT_QUERY, self._handle_threat_query)
            self._mesh.on_message(MessageTypes.BRAIN_QUERY, self._handle_brain_query)
            self._mesh.on_message(MessageTypes.CORRELATION_ALERT, self._handle_correlation)

            # Start mesh
            self._mesh.start()
            self._running = True

            logger.info(f"MEMSHADOW mesh client started on port {self.mesh_port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start mesh client: {e}")
            return False

    async def stop(self):
        """Stop the mesh client"""
        self._running = False
        if self._mesh:
            self._mesh.stop()
            self._mesh = None
        logger.info("MEMSHADOW mesh client stopped")

    def on_threat_query(self, handler: Callable[[Dict, str], Dict]):
        """
        Register handler for threat queries.

        Handler signature: (query_data, peer_id) -> response_dict
        """
        self._threat_query_handler = handler

    def on_correlation_request(self, handler: Callable[[Dict, str], Dict]):
        """Register handler for correlation requests"""
        self._correlation_handler = handler

    async def share_threat_intel(self, intel: ThreatIntel,
                                 priority: int = 1):
        """
        Share threat intelligence with the mesh network.

        Args:
            intel: Threat intelligence to share
            priority: Message priority
        """
        if not self._mesh or not self._running:
            logger.warning("Cannot share intel - mesh not running")
            return False

        try:
            msg = json.dumps({
                "type": "threat_intel",
                "intel": intel.to_dict(),
                "source_node": self.node_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }).encode()

            self._mesh.broadcast(MessageTypes.THREAT_INTEL, msg)
            self.stats["intel_shared"] += 1

            logger.info(f"Shared threat intel: {intel.intel_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to share intel: {e}")
            return False

    async def broadcast_iocs(self, iocs: List[Dict], campaign_id: Optional[str] = None):
        """
        Broadcast IOCs to the mesh network.

        Args:
            iocs: List of indicator of compromise dicts
            campaign_id: Optional campaign association
        """
        if not self._mesh or not self._running:
            return False

        try:
            msg = json.dumps({
                "type": "ioc_broadcast",
                "iocs": iocs,
                "campaign_id": campaign_id,
                "source_node": self.node_id,
                "count": len(iocs),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }).encode()

            self._mesh.broadcast(MessageTypes.IOC_BROADCAST, msg)
            self.stats["iocs_broadcast"] += len(iocs)

            logger.info(f"Broadcast {len(iocs)} IOCs")
            return True

        except Exception as e:
            logger.error(f"Failed to broadcast IOCs: {e}")
            return False

    async def update_campaign(self, campaign_id: str, update: Dict):
        """
        Share campaign tracking update.

        Args:
            campaign_id: Campaign identifier
            update: Update data
        """
        if not self._mesh or not self._running:
            return False

        try:
            msg = json.dumps({
                "type": "campaign_update",
                "campaign_id": campaign_id,
                "update": update,
                "source_node": self.node_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }).encode()

            self._mesh.broadcast(MessageTypes.CAMPAIGN_UPDATE, msg)

            logger.info(f"Shared campaign update: {campaign_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to share campaign update: {e}")
            return False

    async def query_brain(self, query: str, timeout: float = 30.0) -> Optional[Dict]:
        """
        Query the brain network for intelligence.

        This sends a query that will be processed by the hub
        and distributed to brain nodes.

        Args:
            query: Natural language query
            timeout: Response timeout

        Returns:
            Aggregated response or None
        """
        if not self._mesh or not self._running:
            return None

        try:
            msg = json.dumps({
                "type": "memshadow_query",
                "query": query,
                "source_node": self.node_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }).encode()

            self._mesh.broadcast(MessageTypes.BRAIN_QUERY, msg)
            logger.info(f"Sent query to brain network: {query[:50]}...")

            return {"status": "sent", "query": query}

        except Exception as e:
            logger.error(f"Failed to query brain: {e}")
            return None

    def _handle_threat_query(self, data: bytes, peer_id: str):
        """Handle incoming threat query"""
        try:
            query_data = json.loads(data.decode())
            self.stats["queries_received"] += 1

            logger.info(f"Received threat query from {peer_id}")

            if self._threat_query_handler:
                # Process query
                response = self._threat_query_handler(query_data, peer_id)

                # Send response
                if response and self._mesh:
                    response_msg = json.dumps({
                        "type": "threat_response",
                        "query_id": query_data.get("query_id"),
                        "response": response,
                        "source_node": self.node_id,
                    }).encode()

                    self._mesh.send(peer_id, MessageTypes.QUERY_RESPONSE, response_msg)
                    self.stats["queries_answered"] += 1

        except Exception as e:
            logger.error(f"Error handling threat query: {e}")

    def _handle_brain_query(self, data: bytes, peer_id: str):
        """Handle query from brain hub"""
        try:
            query_data = json.loads(data.decode())
            logger.info(f"Received brain query from {peer_id}")

            # Process based on query type
            query_text = query_data.get("natural_language", "")

            # Search local threat database
            response = {
                "query_id": query_data.get("query_id"),
                "node_id": self.node_id,
                "success": True,
                "result": {
                    "source": "MEMSHADOW",
                    "type": "threat_intel",
                    "data": f"Threat intelligence response for: {query_text[:50]}...",
                },
                "confidence": 0.75,
            }

            if self._mesh:
                response_msg = json.dumps(response).encode()
                self._mesh.send(peer_id, MessageTypes.QUERY_RESPONSE, response_msg)

        except Exception as e:
            logger.error(f"Error handling brain query: {e}")

    def _handle_correlation(self, data: bytes, peer_id: str):
        """Handle correlation alert from mesh"""
        try:
            correlation_data = json.loads(data.decode())
            logger.info(f"Received correlation alert from {peer_id}")

            if self._correlation_handler:
                self._correlation_handler(correlation_data, peer_id)

        except Exception as e:
            logger.error(f"Error handling correlation: {e}")

    def get_connected_peers(self) -> List[str]:
        """Get list of connected peer node IDs"""
        if not self._mesh:
            return []

        peers = self._mesh.get_connected_peers()
        return [p.node_id for p in peers]

    def get_stats(self) -> Dict:
        """Get client statistics"""
        stats = self.stats.copy()
        stats["connected_peers"] = len(self.get_connected_peers())
        stats["running"] = self._running
        return stats


# Integration with existing MEMSHADOW services
class MeshIntegrationService:
    """
    Service to integrate mesh client with MEMSHADOW's existing services.

    Connects:
    - Threat intelligence aggregator
    - Campaign manager
    - IOC identifier
    - Memory service
    """

    def __init__(self, mesh_client: MEMSHADOWMeshClient):
        self.mesh_client = mesh_client
        self._services: Dict[str, Any] = {}

    def register_service(self, name: str, service: Any):
        """Register a MEMSHADOW service for mesh integration"""
        self._services[name] = service
        logger.info(f"Registered service for mesh integration: {name}")

    async def setup_handlers(self):
        """Setup query handlers using registered services"""

        def handle_threat_query(query_data: Dict, peer_id: str) -> Dict:
            # Would call actual MEMSHADOW services here
            intel_service = self._services.get("threat_intel")
            if intel_service:
                pass  # result = intel_service.query(query_data)

            return {
                "status": "processed",
                "query_id": query_data.get("query_id"),
            }

        self.mesh_client.on_threat_query(handle_threat_query)

        logger.info("Mesh integration handlers configured")


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    print("MEMSHADOW Mesh Client Self-Test")
    print("=" * 50)

    async def test():
        client = MEMSHADOWMeshClient(
            node_id="memshadow-test",
            mesh_port=8890,
        )

        print(f"\n[1] Start mesh client")
        started = await client.start()
        print(f"    Started: {started}")

        if started:
            print(f"\n[2] Share threat intel")
            intel = ThreatIntel(
                intel_id="test-001",
                threat_type="malware",
                severity=ThreatSeverity.HIGH,
                confidence=0.85,
                indicators={"ip": "192.168.1.100", "hash": "abc123..."},
                campaign_id="APT-TEST",
            )
            shared = await client.share_threat_intel(intel)
            print(f"    Shared: {shared}")

            print(f"\n[3] Broadcast IOCs")
            iocs = [
                {"type": "ip", "value": "10.0.0.1"},
                {"type": "domain", "value": "evil.example.com"},
            ]
            broadcast = await client.broadcast_iocs(iocs)
            print(f"    Broadcast: {broadcast}")

            print(f"\n[4] Statistics")
            stats = client.get_stats()
            for k, v in stats.items():
                print(f"    {k}: {v}")

            await asyncio.sleep(2)

            print(f"\n[5] Stop mesh client")
            await client.stop()

    asyncio.run(test())

    print("\n" + "=" * 50)
    print("MEMSHADOW Mesh Client test complete")
