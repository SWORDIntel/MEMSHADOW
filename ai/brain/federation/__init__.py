#!/usr/bin/env python3
"""
DSMIL Brain Federation Layer

Hub-spoke architecture for distributed intelligence:
- DSMIL as central hub (query originator, aggregator)
- Spoke nodes (local correlation, no NL interface)
- Offline/peer coordination mode
- Intel propagation and sync

Central Hub Query Model:
- All NL queries originate from hub
- Hub distributes to relevant nodes
- Nodes correlate locally, return results
- Hub aggregates and synthesizes
"""

from .hub_orchestrator import (
    HubOrchestrator,
    DistributedQuery,
    AggregatedResponse,
    NodeCapability,
)

from .spoke_client import (
    SpokeClient,
    LocalCorrelation,
    QueryResponse,
    SpokeState,
)

from .offline_coordinator import (
    OfflineCoordinator,
    PeerNetwork,
    ConsensusProtocol,
)

from .intel_propagator import (
    IntelPropagator,
    IntelReport,
    PropagationPriority,
)

from .sync_protocol import (
    SyncProtocol,
    DeltaSync,
    SyncState,
    ConflictResolution,
)

__all__ = [
    # Hub
    "HubOrchestrator",
    "DistributedQuery",
    "AggregatedResponse",
    "NodeCapability",
    # Spoke
    "SpokeClient",
    "LocalCorrelation",
    "QueryResponse",
    "SpokeState",
    # Offline
    "OfflineCoordinator",
    "PeerNetwork",
    "ConsensusProtocol",
    # Intel
    "IntelPropagator",
    "IntelReport",
    "PropagationPriority",
    # Sync
    "SyncProtocol",
    "DeltaSync",
    "SyncState",
    "ConflictResolution",
]

