# DSMIL Brain Federation

Distributed intelligence network with hub-spoke architecture and P2P capabilities.

## Architecture

### Hub-Spoke Model

```
                    ┌─────────────┐
                    │  Hub Node   │
                    │ (Orchestrator)│
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

### Communication Modes

1. **Hub-Relayed:** Normal operations go through hub
2. **Direct P2P:** Critical updates bypass hub for speed
3. **Hybrid:** Automatic routing based on priority

## Components

### Hub Orchestrator (`hub_orchestrator.py`)

Central coordination point for the distributed brain network.

**Responsibilities:**
- Node registration and health monitoring
- Query distribution and response aggregation
- Intelligence propagation
- PSYCH message handling (from SHRINK)
- Self-improvement announcement relay

**Key Methods:**
- `register_node()` - Register a spoke node
- `query()` - Distribute query to nodes
- `propagate_intel()` - Broadcast intelligence
- `_handle_psych_intel()` - Process SHRINK psychological data
- `_handle_improvement_announce()` - Relay improvement announcements

### Spoke Client (`spoke_client.py`)

Node client that receives queries and performs local correlation.

**Responsibilities:**
- Receive queries from hub
- Perform local correlation and analysis
- Return results to hub
- Operate autonomously when hub offline
- P2P improvement propagation

**Key Methods:**
- `connect()` - Connect to hub
- `process_query()` - Process incoming query
- `announce_improvement()` - Broadcast local improvements
- `_handle_improvement_payload()` - Apply received improvements

## Message Flow

### SHRINK → Brain → Mesh

```
SHRINK Kernel Module
    │ (Netlink, MEMSHADOW protocol)
    ▼
Userspace Receiver (kernel_receiver.py)
    │ (HTTP POST /api/v1/ingest/shrink)
    ▼
Brain API Endpoint (shrink_endpoint.py)
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
Improvement Tracker (improvement_tracker.py)
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

## Self-Improvement System

### Improvement Types (`improvement_types.py`)

1. **Model Weight Deltas** - Compressed neural network updates
2. **Config Tuning** - Threshold and parameter changes
3. **Learned Patterns** - Discovered correlations and patterns

### Improvement Tracker (`improvement_tracker.py`)

Tracks local performance metrics and detects statistically significant improvements.

**Features:**
- Performance metric tracking (accuracy, latency, confidence)
- Statistical significance detection
- Improvement packaging
- Compatibility checking
- Effectiveness measurement

### Propagation Strategy

- **Critical improvements** (>20% gain): Direct P2P to all peers
- **Normal improvements** (10-20% gain): Via hub relay
- **Minor improvements** (<10% gain): Background propagation

## Memory Tier Integration

All memory tiers (L1/L2/L3) support MEMSHADOW protocol synchronization:

- **Working Memory (L1):** Fast sync for active data
- **Episodic Memory (L2):** Episode synchronization
- **Semantic Memory (L3):** Knowledge graph sync

See [Memory Sync Protocol](../memory/memory_sync_protocol.py) for details.

## PSYCH Message Handling

The hub orchestrator handles SHRINK psychological intelligence:

1. **Receives** PSYCH_* messages from SHRINK nodes
2. **Stores** in memory tiers (L1/L2/L3)
3. **Evaluates** significance (high risk, dark triad, anomalies)
4. **Broadcasts** significant updates to network
5. **Routes** threat alerts with CRITICAL priority

## Configuration

### Hub Configuration

```python
hub = HubOrchestrator(
    hub_id="dsmil-central",
    mesh_port=8889,
    use_mesh=True
)
```

### Spoke Configuration

```python
spoke = SpokeClient(
    node_id="node-001",
    hub_endpoint="hub.local:8000",
    capabilities={"search", "correlate"},
    data_domains={"threat_intel", "psych_intel"},
    mesh_port=8889,
    use_mesh=True
)
```

## Testing

Integration tests verify:
- Message handler registration
- PSYCH intel routing
- Improvement propagation
- P2P communication
- Memory tier sync

Run tests:
```bash
python3 test_memshadow_integration.py
```

## API Reference

### HubOrchestrator

```python
class HubOrchestrator:
    def register_node(self, node_id: str, endpoint: str, 
                     capabilities: NodeCapability) -> RegisteredNode
    async def query(self, query_text: str, 
                   priority: QueryPriority = QueryPriority.NORMAL) -> AggregatedResponse
    def propagate_intel(self, intel_report: Dict, 
                       target_nodes: Optional[List[str]] = None)
    def _handle_psych_intel(self, data: bytes, peer_id: str)
    def _handle_improvement_announce(self, data: bytes, peer_id: str)
```

### SpokeClient

```python
class SpokeClient:
    async def connect(self) -> bool
    async def process_query(self, query: Dict, 
                           source: QuerySource) -> QueryResponse
    def announce_improvement(self, improvement: ImprovementPackage)
    def _handle_improvement_payload(self, data: bytes, peer_id: str)
```

## Related Documentation

- [MEMSHADOW Protocol](../../../libs/memshadow-protocol/README.md)
- [Memory Sync Protocol](../memory/memory_sync_protocol.py)
- [Improvement Types](improvement_types.py)
- [SHRINK Integration](../../../external/intel/shrink/README.md)
