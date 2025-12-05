# DSMIL Brain Federation

Distributed intelligence network with hub-spoke architecture and P2P capabilities.

## Architecture

```
                    ┌─────────────┐
                    │  Hub Node   │
                    │(Orchestrator)│
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐        ┌────▼────┐        ┌────▼────┐
   │ Spoke 1 │        │ Spoke 2 │        │ Spoke 3 │
   │ (SHRINK)│        │ (Brain) │        │ (Brain) │
   └────┬────┘        └────┬────┘        └────┬────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    P2P Direct Links
```

## Components

### HubOrchestrator

Central coordination point with integrated MEMSHADOW gateway.

```python
from ai.brain.federation import HubOrchestrator, NodeCapability

hub = HubOrchestrator(hub_id="dsmil-central", mesh_port=8889)

# Register nodes
node = hub.register_node(
    node_id="node-001",
    endpoint="localhost:8001",
    capabilities={NodeCapability.SEARCH, NodeCapability.CORRELATE},
)

# Distribute query
response = await hub.query("search term", priority=QueryPriority.HIGH)

# Propagate intel
hub.propagate_intel({"type": "threat", "data": {...}}, priority=Priority.CRITICAL)
```

### HubMemshadowGateway

Hub-side memory synchronization gateway (integrated into HubOrchestrator).

Features:
- Node memory tier registration
- Sync scheduling (priority-aware queue)
- Batch routing (hub-relay vs P2P)
- Per-node/per-tier sync vectors
- Metrics and observability

```python
# Access via orchestrator
gateway = hub.memshadow_gateway

# Schedule sync
hub.schedule_memory_sync(
    node_id="node-001",
    tier=MemoryTier.WORKING,
    priority=SyncPriority.HIGH,
)
```

### SpokeClient

Node client with integrated SpokeMemoryAdapter.

```python
from ai.brain.federation import SpokeClient

spoke = SpokeClient(
    node_id="node-001",
    hub_endpoint="hub.local:8000",
    capabilities={"search", "correlate"},
)

# Connect to hub
await spoke.connect()

# Access memory adapter
adapter = spoke.memory_adapter

# Store and sync data
item_id = await adapter.store(
    tier=MemoryTier.WORKING,
    data=b"data",
)
await adapter.sync_to_hub(MemoryTier.WORKING)
```

### SpokeMemoryAdapter

Local memory tier access with sync capabilities.

Features:
- L1/L2/L3 tier storage
- Delta batch creation
- Incoming batch application
- Conflict resolution
- P2P sync for critical updates
- Self-improvement propagation

```python
from ai.brain.federation import SpokeMemoryAdapter, InMemoryTier

adapter = SpokeMemoryAdapter(node_id="node-001")

# Register tiers
adapter.register_tier(MemoryTier.WORKING, InMemoryTier(MemoryTier.WORKING))

# Create delta batch for sync
batch = await adapter.create_delta_batch(
    tier=MemoryTier.WORKING,
    target_node="hub",
)

# Apply incoming batch
result = await adapter.apply_sync_batch(batch)
```

## Message Flow

### SHRINK → Brain → Mesh

```
SHRINK Kernel Module
    │ (Netlink, MEMSHADOW protocol)
    ▼
Userspace Receiver
    │ (HTTP POST /api/v1/ingest/shrink)
    ▼
Brain API Endpoint
    │ (MEMSHADOW ingest plugin)
    ▼
Brain Memory Tiers (L1/L2/L3)
    │ (Significant updates only)
    ▼
Hub Orchestrator
    │ (Mesh broadcast)
    ▼
Other Spoke Nodes
```

### Self-Improvement Propagation

```
Node A detects improvement
    │
    ▼
Improvement Tracker
    │ (Packages improvement)
    ▼
Spoke Client
    │ (IMPROVEMENT_ANNOUNCE)
    ├─► Hub (normal priority)
    └─► Peers directly (critical priority)
    │
    ▼
Interested peers request (IMPROVEMENT_REQUEST)
    │
    ▼
Node A sends payload (IMPROVEMENT_PAYLOAD)
    │
    ▼
Peers apply and ACK (IMPROVEMENT_ACK)
```

## Priority Routing

| Priority | Routing | Use Case |
|----------|---------|----------|
| LOW/NORMAL | Hub-relayed | Standard operations |
| HIGH | Hub with priority queue | Important updates |
| CRITICAL | P2P + hub notification | Urgent alerts |
| EMERGENCY | Immediate P2P | Critical safety |

## Improvement Types

1. **Model Weight Deltas** - Compressed neural network updates
2. **Config Tuning** - Threshold and parameter changes
3. **Learned Patterns** - Discovered correlations and patterns

## Testing

```bash
python3 test_memshadow_integration.py
```

## Related Documentation

- [MEMSHADOW Protocol](../../../libs/memshadow-protocol/README.md)
- [Memory Sync Protocol](../memory/MEMSHADOW_INTEGRATION.md)
- [Hub Integration](../../../HUB_DOCS/DSMIL%20Brain%20Federation.md)
