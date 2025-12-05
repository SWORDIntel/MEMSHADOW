"""
DSMIL Brain Federation

Distributed intelligence network with hub-spoke architecture and P2P capabilities.

Components:
- HubOrchestrator: Central coordination with MEMSHADOW gateway
- SpokeClient: Node client with SpokeMemoryAdapter
- ImprovementTracker: Detects and packages improvements
- ImprovementTypes: Improvement package definitions

Usage:
    from ai.brain.federation import HubOrchestrator, SpokeClient
    
    hub = HubOrchestrator(hub_id="dsmil-central")
    spoke = SpokeClient(node_id="node-001", hub_endpoint="hub.local:8000")
"""

from .hub_orchestrator import (
    HubOrchestrator,
    HubMemshadowGateway,
    NodeCapability,
    NodeMemoryCapabilities,
    RegisteredNode,
    QueryPriority,
    AggregatedResponse,
)

from .spoke_client import (
    SpokeClient,
    SpokeMemoryAdapter,
    LocalTierInterface,
    InMemoryTier,
    QuerySource,
    QueryResponse,
)

from .improvement_types import (
    ImprovementType,
    ImprovementPriority,
    ImprovementPackage,
    ImprovementMetrics,
)

from .improvement_tracker import (
    ImprovementTracker,
    MetricWindow,
)


__all__ = [
    # Hub
    "HubOrchestrator",
    "HubMemshadowGateway",
    "NodeCapability",
    "NodeMemoryCapabilities",
    "RegisteredNode",
    "QueryPriority",
    "AggregatedResponse",
    # Spoke
    "SpokeClient",
    "SpokeMemoryAdapter",
    "LocalTierInterface",
    "InMemoryTier",
    "QuerySource",
    "QueryResponse",
    # Improvements
    "ImprovementType",
    "ImprovementPriority",
    "ImprovementPackage",
    "ImprovementMetrics",
    "ImprovementTracker",
    "MetricWindow",
]
