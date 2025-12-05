"""
MEMSHADOW Memory Synchronization Module

Hub-first MEMSHADOW integration for the DSMIL Brain Federation.
Provides cross-node memory tier synchronization using MEMSHADOW Protocol v2.

Components:
- protocol: MEMSHADOW Protocol v2 header/message handling
- sync_manager: MemorySyncManager for delta computation and batch handling
- hub_memshadow_gateway: Hub-side coordination for cross-node sync
- spoke_memory_adapter: Spoke-side local tier access and sync
- federation_hub: Unified hub orchestrator with MEMSHADOW integration

See HUB_MEMSHADOW_DESIGN.md for architecture details.
"""

from .protocol import (
    MemshadowHeader,
    MemshadowMessage,
    MessageType,
    Priority,
    MessageFlags,
    MessageRouter,
    MEMSHADOW_MAGIC,
    MEMSHADOW_VERSION,
    HEADER_SIZE,
    create_memory_sync_message,
    create_ack_message,
    should_route_p2p,
)
from .sync_manager import (
    MemoryTier,
    MemorySyncItem,
    MemorySyncBatch,
    MemorySyncManager,
    SyncResult,
    SyncVector,
    ConflictResolution,
)
from .hub_memshadow_gateway import (
    HubMemshadowGateway,
    NodeMemoryCapabilities,
    NodeSyncState,
    NodeSyncInfo,
)
from .spoke_memory_adapter import (
    SpokeMemoryAdapter,
    LocalTierStore,
    LocalTierConfig,
)
from .federation_hub import (
    FederationHubOrchestrator,
    RegisteredNode,
    NodeCapability,
    QueryPriority,
    get_federation_hub,
    init_federation_hub,
    shutdown_federation_hub,
)

__all__ = [
    # Protocol
    "MemshadowHeader",
    "MemshadowMessage",
    "MessageType",
    "Priority",
    "MessageFlags",
    "MessageRouter",
    "MEMSHADOW_MAGIC",
    "MEMSHADOW_VERSION",
    "HEADER_SIZE",
    "create_memory_sync_message",
    "create_ack_message",
    "should_route_p2p",
    # Sync Manager
    "MemoryTier",
    "MemorySyncItem",
    "MemorySyncBatch",
    "MemorySyncManager",
    "SyncResult",
    "SyncVector",
    "ConflictResolution",
    # Hub Gateway
    "HubMemshadowGateway",
    "NodeMemoryCapabilities",
    "NodeSyncState",
    "NodeSyncInfo",
    # Spoke Adapter
    "SpokeMemoryAdapter",
    "LocalTierStore",
    "LocalTierConfig",
    # Federation Hub
    "FederationHubOrchestrator",
    "RegisteredNode",
    "NodeCapability",
    "QueryPriority",
    "get_federation_hub",
    "init_federation_hub",
    "shutdown_federation_hub",
]
