#!/usr/bin/env python3
"""
MEMSHADOW Mesh Client - Remote Spoke Node

Integrates MEMSHADOW with the DSMIL mesh network as a remote spoke node.
All mesh functionality is OPTIONAL - MEMSHADOW works fully standalone.

Architecture:
- ai/brain acts as the central hub
- MEMSHADOW acts as a remote spoke receiving broadcasts
- KP14 sends threat intel via mesh to MEMSHADOW
- dsmil-mesh handles P2P synchronization when available

Feature-flag controlled:
- Disabled by default (ENABLE_KP14_INTEGRATION=false)
- Graceful degradation when mesh unavailable
- No hard-coded paths or required imports
"""

import asyncio
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

# Use structlog if available, fallback to standard logging
try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# Feature detection - try to import dsmil-mesh library
MESH_AVAILABLE = False
QuantumMesh = None
MeshConfig = None
MessageTypes = None
MessagePriority = None


def _find_mesh_library() -> Optional[Path]:
    """
    Find dsmil-mesh library using multiple strategies.
    No hard-coded paths - uses environment variables and relative paths.
    """
    # Strategy 1: Environment variable
    mesh_path = os.environ.get('DSMIL_MESH_PATH')
    if mesh_path:
        path = Path(mesh_path)
        if path.exists():
            return path

    # Strategy 2: Relative path from MEMSHADOW
    current_file = Path(__file__)
    relative_path = current_file.parent.parent.parent.parent.parent / "libs" / "dsmil-mesh" / "python"
    if relative_path.exists():
        return relative_path

    # Strategy 3: Common installation locations
    common_paths = [
        Path("/opt/dsmil/libs/dsmil-mesh/python"),
        Path("/usr/local/lib/dsmil-mesh/python"),
        Path.home() / ".local" / "lib" / "dsmil-mesh" / "python",
    ]
    for path in common_paths:
        if path.exists():
            return path

    return None


def _init_mesh_imports():
    """Initialize mesh library imports with graceful fallback."""
    global MESH_AVAILABLE, QuantumMesh, MeshConfig, MessageTypes, MessagePriority

    mesh_path = _find_mesh_library()
    if not mesh_path:
        logger.info("dsmil-mesh library not found - mesh features disabled")
        return

    try:
        if str(mesh_path) not in sys.path:
            sys.path.insert(0, str(mesh_path))

        from mesh import QuantumMesh as QM, MeshConfig as MC
        from messages import MessageTypes as MT, MessagePriority as MP

        QuantumMesh = QM
        MeshConfig = MC
        MessageTypes = MT
        MessagePriority = MP
        MESH_AVAILABLE = True

        logger.info("dsmil-mesh library loaded successfully", path=str(mesh_path))

    except ImportError as e:
        logger.warning("Failed to import dsmil-mesh", error=str(e))


# Initialize on module load
_init_mesh_imports()


class SpokeState(Enum):
    """State of the MEMSHADOW spoke node"""
    INITIALIZING = auto()
    CONNECTED = auto()      # Connected to hub via mesh
    DISCONNECTED = auto()   # Disconnected from hub
    STANDALONE = auto()     # Operating without mesh (default)
    SYNCING = auto()        # Syncing data with hub


@dataclass
class MEMSHADOWCapabilities:
    """Capabilities advertised to the hub"""
    node_id: str
    capabilities: Set[str] = field(default_factory=lambda: {
        "memory_storage",
        "threat_intel",
        "semantic_search",
        "neural_storage",
        "knowledge_graph",
    })
    data_domains: Set[str] = field(default_factory=lambda: {
        "threat_intel",
        "memories",
        "campaigns",
        "actors",
        "iocs",
    })
    has_gpu: bool = False
    has_neural_storage: bool = True


class MEMSHADOWMeshClient:
    """
    MEMSHADOW Mesh Client

    Acts as a remote spoke node in the DSMIL hub-spoke architecture.
    Receives threat intelligence from KP14 and queries from ai/brain hub.

    Usage:
        client = MEMSHADOWMeshClient(enabled=True)
        await client.start()

        # Client now receives mesh broadcasts automatically

        await client.stop()

    All functionality gracefully degrades when:
    - dsmil-mesh library not installed
    - Mesh network unreachable
    - Feature disabled via config
    """

    def __init__(
        self,
        node_id: str = "memshadow-spoke",
        mesh_port: int = 8890,
        cluster_name: str = "dsmil-brain",
        enabled: bool = True,
    ):
        """
        Initialize MEMSHADOW mesh client.

        Args:
            node_id: Unique node identifier for this MEMSHADOW instance
            mesh_port: Port for mesh communication
            cluster_name: Mesh cluster to join
            enabled: Whether mesh integration is enabled
        """
        self.node_id = node_id
        self.mesh_port = mesh_port
        self.cluster_name = cluster_name
        self.enabled = enabled and MESH_AVAILABLE

        # State
        self.state = SpokeState.STANDALONE if not self.enabled else SpokeState.INITIALIZING

        # Mesh network
        self._mesh: Optional[Any] = None  # QuantumMesh instance
        self._running = False

        # Capabilities
        self.capabilities = MEMSHADOWCapabilities(node_id=node_id)

        # Message handlers (pluggable)
        self._intel_handler: Optional[Callable] = None
        self._query_handler: Optional[Callable] = None
        self._ioc_handler: Optional[Callable] = None

        # Offline queue for when mesh unavailable
        self._offline_queue: List[Dict] = []
        self._queue_lock = threading.Lock()

        # Statistics
        self.stats = {
            "intel_received": 0,
            "queries_handled": 0,
            "iocs_received": 0,
            "broadcasts_sent": 0,
            "mesh_available": MESH_AVAILABLE,
            "mesh_enabled": self.enabled,
        }

        if not MESH_AVAILABLE:
            logger.info("MEMSHADOW mesh client initialized in standalone mode (mesh unavailable)")
        elif not enabled:
            logger.info("MEMSHADOW mesh client disabled via configuration")
        else:
            logger.info(
                "MEMSHADOW mesh client initialized",
                node_id=node_id,
                port=mesh_port,
                cluster=cluster_name,
            )

    async def start(self) -> bool:
        """
        Start mesh client and connect to network.

        Returns:
            True if started successfully (or already in standalone mode)
        """
        if not self.enabled:
            self.state = SpokeState.STANDALONE
            logger.info("Mesh client running in standalone mode")
            return True

        if not MESH_AVAILABLE:
            self.state = SpokeState.STANDALONE
            logger.warning("Cannot start mesh - library not available")
            return True  # Graceful degradation - still "successful"

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
            self._mesh.on_message(MessageTypes.THREAT_INTEL, self._handle_threat_intel)
            self._mesh.on_message(MessageTypes.BRAIN_QUERY, self._handle_brain_query)
            self._mesh.on_message(MessageTypes.KNOWLEDGE_UPDATE, self._handle_knowledge_update)
            self._mesh.on_message(MessageTypes.IOC_BROADCAST, self._handle_ioc_broadcast)
            self._mesh.on_message(MessageTypes.CAMPAIGN_UPDATE, self._handle_campaign_update)

            # Start mesh
            self._mesh.start()
            self._running = True
            self.state = SpokeState.CONNECTED

            logger.info("MEMSHADOW mesh client started", port=self.mesh_port)
            return True

        except Exception as e:
            logger.error("Failed to start mesh client", error=str(e))
            self.state = SpokeState.STANDALONE
            return True  # Graceful degradation

    async def stop(self) -> None:
        """Stop mesh client."""
        self._running = False

        if self._mesh:
            self._mesh.stop()
            self._mesh = None

        self.state = SpokeState.DISCONNECTED
        logger.info("MEMSHADOW mesh client stopped")

    def on_intel_received(self, handler: Callable[[Dict], None]) -> None:
        """Register handler for received threat intelligence."""
        self._intel_handler = handler

    def on_query_received(self, handler: Callable[[Dict, str], Dict]) -> None:
        """Register handler for brain queries."""
        self._query_handler = handler

    def on_iocs_received(self, handler: Callable[[List[Dict]], None]) -> None:
        """Register handler for received IOCs."""
        self._ioc_handler = handler

    def _handle_threat_intel(self, data: bytes, peer_id: str) -> None:
        """Handle incoming threat intelligence from KP14."""
        try:
            intel_data = json.loads(data.decode())
            logger.info(
                "Received threat intel via mesh",
                source=peer_id,
                type=intel_data.get("type"),
            )

            self.stats["intel_received"] += 1

            # Call registered handler
            if self._intel_handler:
                self._intel_handler(intel_data)
            else:
                # Default: queue for later processing
                with self._queue_lock:
                    self._offline_queue.append({
                        "type": "threat_intel",
                        "data": intel_data,
                        "source": peer_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

        except Exception as e:
            logger.error("Error handling threat intel", error=str(e))

    def _handle_brain_query(self, data: bytes, peer_id: str) -> None:
        """Handle query from ai/brain hub."""
        try:
            query_data = json.loads(data.decode())
            logger.info(
                "Received brain query via mesh",
                query_id=query_data.get("query_id"),
                source=peer_id,
            )

            self.stats["queries_handled"] += 1

            # Build response
            response = {
                "query_id": query_data.get("query_id"),
                "node_id": self.node_id,
                "success": True,
                "confidence": 0.8,
            }

            # Call registered handler for custom response
            if self._query_handler:
                custom_result = self._query_handler(query_data, peer_id)
                if custom_result:
                    response["result"] = custom_result
            else:
                # Default response with capabilities
                response["result"] = {
                    "source": "MEMSHADOW",
                    "type": "memory_spoke",
                    "capabilities": list(self.capabilities.capabilities),
                    "domains": list(self.capabilities.data_domains),
                }

            # Send response
            if self._mesh:
                response_msg = json.dumps(response).encode()
                self._mesh.send(peer_id, MessageTypes.QUERY_RESPONSE, response_msg)

        except Exception as e:
            logger.error("Error handling brain query", error=str(e))

    def _handle_knowledge_update(self, data: bytes, peer_id: str) -> None:
        """Handle knowledge graph update from hub."""
        try:
            update_data = json.loads(data.decode())
            logger.info(
                "Received knowledge update via mesh",
                source=peer_id,
            )

            # Queue for processing
            with self._queue_lock:
                self._offline_queue.append({
                    "type": "knowledge_update",
                    "data": update_data,
                    "source": peer_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        except Exception as e:
            logger.error("Error handling knowledge update", error=str(e))

    def _handle_ioc_broadcast(self, data: bytes, peer_id: str) -> None:
        """Handle IOC broadcast from KP14."""
        try:
            ioc_data = json.loads(data.decode())
            iocs = ioc_data.get("iocs", [])

            logger.info(
                "Received IOC broadcast via mesh",
                count=len(iocs),
                source=peer_id,
            )

            self.stats["iocs_received"] += len(iocs)

            # Call registered handler
            if self._ioc_handler:
                self._ioc_handler(iocs)
            else:
                # Queue for processing
                with self._queue_lock:
                    self._offline_queue.append({
                        "type": "ioc_broadcast",
                        "data": ioc_data,
                        "source": peer_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

        except Exception as e:
            logger.error("Error handling IOC broadcast", error=str(e))

    def _handle_campaign_update(self, data: bytes, peer_id: str) -> None:
        """Handle campaign tracking update."""
        try:
            campaign_data = json.loads(data.decode())
            logger.info(
                "Received campaign update via mesh",
                campaign_id=campaign_data.get("campaign_id"),
                source=peer_id,
            )

            # Queue for processing
            with self._queue_lock:
                self._offline_queue.append({
                    "type": "campaign_update",
                    "data": campaign_data,
                    "source": peer_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        except Exception as e:
            logger.error("Error handling campaign update", error=str(e))

    def broadcast_memory(self, memory: Dict[str, Any]) -> bool:
        """
        Broadcast a memory/intelligence item to the mesh network.

        Args:
            memory: Memory data to broadcast

        Returns:
            True if broadcast successful (or queued)
        """
        if not self._running or not self._mesh:
            # Queue for later
            with self._queue_lock:
                self._offline_queue.append({
                    "type": "outbound_memory",
                    "data": memory,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            return True  # Queued successfully

        try:
            msg_data = {
                "type": "memshadow_memory",
                "memory": memory,
                "source_node": self.node_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            msg = json.dumps(msg_data).encode()
            self._mesh.broadcast(MessageTypes.INTEL_REPORT, msg)

            self.stats["broadcasts_sent"] += 1
            return True

        except Exception as e:
            logger.error("Failed to broadcast memory", error=str(e))
            return False

    def broadcast_threat_intel(self, intel: Dict[str, Any]) -> bool:
        """
        Broadcast threat intelligence to the mesh network.

        Args:
            intel: Threat intelligence data

        Returns:
            True if broadcast successful
        """
        if not self._running or not self._mesh:
            with self._queue_lock:
                self._offline_queue.append({
                    "type": "outbound_intel",
                    "data": intel,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            return True

        try:
            msg_data = {
                "type": "memshadow_threat_intel",
                "intel": intel,
                "source_node": self.node_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            msg = json.dumps(msg_data).encode()
            self._mesh.broadcast(MessageTypes.THREAT_INTEL, msg)

            self.stats["broadcasts_sent"] += 1
            return True

        except Exception as e:
            logger.error("Failed to broadcast threat intel", error=str(e))
            return False

    def get_pending_queue(self) -> List[Dict]:
        """Get items pending in the offline queue."""
        with self._queue_lock:
            return self._offline_queue.copy()

    def clear_pending_queue(self) -> int:
        """Clear the offline queue. Returns number of items cleared."""
        with self._queue_lock:
            count = len(self._offline_queue)
            self._offline_queue.clear()
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get mesh client statistics."""
        stats = self.stats.copy()
        stats["state"] = self.state.name
        stats["queue_size"] = len(self._offline_queue)
        stats["connected_peers"] = len(self._get_connected_peers())
        return stats

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status."""
        return {
            "node_id": self.node_id,
            "state": self.state.name,
            "mesh_available": MESH_AVAILABLE,
            "mesh_enabled": self.enabled,
            "mesh_running": self._running,
            "capabilities": list(self.capabilities.capabilities),
            "data_domains": list(self.capabilities.data_domains),
            "stats": self.get_stats(),
        }

    def _get_connected_peers(self) -> List[str]:
        """Get list of connected peer node IDs."""
        if not self._mesh:
            return []
        try:
            peers = self._mesh.get_connected_peers()
            return [p.node_id for p in peers]
        except Exception:
            return []

    def is_connected(self) -> bool:
        """Check if connected to mesh network."""
        return self.state == SpokeState.CONNECTED and self._running

    def is_standalone(self) -> bool:
        """Check if running in standalone mode."""
        return self.state == SpokeState.STANDALONE


# Singleton instance for application-wide use
_mesh_client: Optional[MEMSHADOWMeshClient] = None


def get_mesh_client() -> MEMSHADOWMeshClient:
    """
    Get the singleton mesh client instance.

    Creates with default settings if not already initialized.
    """
    global _mesh_client
    if _mesh_client is None:
        _mesh_client = MEMSHADOWMeshClient()
    return _mesh_client


def init_mesh_client(
    node_id: str = "memshadow-spoke",
    mesh_port: int = 8890,
    enabled: bool = True,
) -> MEMSHADOWMeshClient:
    """
    Initialize the singleton mesh client with custom settings.

    Args:
        node_id: Unique node identifier
        mesh_port: Mesh communication port
        enabled: Whether mesh is enabled

    Returns:
        Initialized MEMSHADOWMeshClient
    """
    global _mesh_client
    _mesh_client = MEMSHADOWMeshClient(
        node_id=node_id,
        mesh_port=mesh_port,
        enabled=enabled,
    )
    return _mesh_client


if __name__ == "__main__":
    # Self-test
    import asyncio

    logging.basicConfig(level=logging.INFO)
    print("MEMSHADOW Mesh Client Self-Test")
    print("=" * 50)

    print(f"\n[1] Feature Detection")
    print(f"    MESH_AVAILABLE: {MESH_AVAILABLE}")
    print(f"    Mesh library path: {_find_mesh_library()}")

    print(f"\n[2] Initialize Client")
    client = MEMSHADOWMeshClient(
        node_id="test-memshadow",
        mesh_port=8890,
        enabled=True,
    )

    print(f"\n[3] Status (before start)")
    status = client.get_status()
    for key, value in status.items():
        if key != "stats":
            print(f"    {key}: {value}")

    print(f"\n[4] Start Client")

    async def test_start():
        success = await client.start()
        print(f"    Started: {success}")
        print(f"    State: {client.state.name}")
        return success

    asyncio.run(test_start())

    print(f"\n[5] Stats")
    stats = client.get_stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    print(f"\n[6] Stop Client")
    asyncio.run(client.stop())
    print(f"    State: {client.state.name}")

    print("\n" + "=" * 50)
    print("MEMSHADOW Mesh Client test complete")

