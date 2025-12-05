# Hub-First MEMSHADOW Integration Design

## Overview

This document describes the integration of MEMSHADOW Protocol v2 into the DSMIL Brain Federation architecture, establishing the Hub Orchestrator as the canonical coordination point for MEMSHADOW memory synchronization across all nodes.

## Current State

### Hub-Spoke Architecture (from DSMIL Brain Federation)

```
                    ┌─────────────────────┐
                    │   Hub Orchestrator   │
                    │  (Central Control)   │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   ┌────▼────┐            ┌────▼────┐            ┌────▼────┐
   │ Spoke 1 │            │ Spoke 2 │            │ Spoke 3 │
   │(MEMSHADOW)│          │ (Brain) │            │ (Brain) │
   └─────────┘            └─────────┘            └─────────┘
```

### Existing Components

1. **Hub Orchestrator** (`app/services/neural_storage/orchestrator.py`)
   - Manages tiered database (L0-L4 tiers)
   - Neural connection discovery
   - Memory migration and deduplication
   - No cross-node sync capabilities

2. **Spoke/Mesh Client** (`app/services/mesh_client.py`)
   - MEMSHADOWMeshClient for mesh integration
   - Handles threat intel, brain queries, IOCs
   - No memory tier synchronization

3. **Memory Services**
   - `TieredDatabaseManager`: Multi-dimensional storage (4096d→256d)
   - `MemoryObject`: Core abstraction with embeddings and links
   - `DifferentialSyncProtocol`: Delta-based sync protocol
   - `FederatedCoordinator`: Privacy-preserving federation

4. **MEMSHADOW Protocol v2** (from docs)
   - 32-byte header format
   - Message types: MEMORY_STORE (0x0301), MEMORY_QUERY (0x0302), MEMORY_RESPONSE (0x0303), MEMORY_SYNC (0x0304)
   - Priority levels: LOW (0) → EMERGENCY (4)
   - Flags: ENCRYPTED, COMPRESSED, BATCHED, REQUIRES_ACK, etc.

### Memory Tiers

| Tier | Name | Dimensions | Purpose |
|------|------|------------|---------|
| L0 | T0_CACHE | 256d | Process-local KV cache |
| L1 | T1_RAMDISK | up to 4096d | Hot working set |
| L2 | T2_NVME | Full vectors | NVMe vector/graph store |
| L3 | T3_COLD | Compressed | Cold object store |

---

## Target State

### Hub-Centric MEMSHADOW Coordination

```
                    ┌─────────────────────────────┐
                    │      Hub Orchestrator        │
                    │  ┌───────────────────────┐  │
                    │  │ HubMemshadowGateway   │  │
                    │  │ - Tier registration    │  │
                    │  │ - Sync scheduling      │  │
                    │  │ - Batch routing        │  │
                    │  │ - Conflict resolution  │  │
                    │  └───────────────────────┘  │
                    └──────────────┬──────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           │ MEMSHADOW Protocol v2 │                       │
           │                       │                       │
    ┌──────▼──────┐         ┌──────▼──────┐         ┌──────▼──────┐
    │   Spoke 1   │         │   Spoke 2   │         │   Spoke 3   │
    │┌───────────┐│◄───────►│┌───────────┐│◄───────►│┌───────────┐│
    ││  Memory   ││   P2P   ││  Memory   ││   P2P   ││  Memory   ││
    ││  Adapter  ││(URGENT) ││  Adapter  ││(URGENT) ││  Adapter  ││
    │├───────────┤│         │├───────────┤│         │├───────────┤│
    ││ L1 | L2   ││         ││ L1 | L2   ││         ││ L1 | L2   ││
    ││ L3       ││         ││ L3       ││         ││ L3       ││
    │└───────────┘│         │└───────────┘│         │└───────────┘│
    └─────────────┘         └─────────────┘         └─────────────┘
```

### Communication Modes

1. **Hub-Relayed (NORMAL/LOW priority)**
   - All standard memory sync goes through hub
   - Hub decides downstream propagation
   - Hub maintains global sync state

2. **Direct P2P (CRITICAL/URGENT priority)**
   - Critical updates bypass hub for speed
   - Hub receives notification copy
   - Used for emergency memory propagation

---

## Component Specifications

### 1. MemoryTier Enum

```python
class MemoryTier(IntEnum):
    L1_WORKING = 1    # Hot working memory (RAMDISK)
    L2_EPISODIC = 2   # Episode/session memory (NVMe)
    L3_SEMANTIC = 3   # Long-term semantic (Cold)
```

### 2. MemorySyncItem

```python
@dataclass
class MemorySyncItem:
    memory_id: UUID
    content_hash: str
    tier: MemoryTier
    embedding: bytes              # Compressed embedding
    metadata: Dict[str, Any]
    timestamp: datetime
    version: int
    operation: str                # "upsert", "delete", "update_meta"
```

### 3. MemorySyncBatch

```python
@dataclass
class MemorySyncBatch:
    batch_id: str
    source_node: str
    target_node: str              # "*" for broadcast
    tier: MemoryTier
    items: List[MemorySyncItem]
    priority: Priority
    flags: int                    # BATCHED, COMPRESSED, etc.
    timestamp: datetime
    checksum: str
```

### 4. MemorySyncManager

Manages per-node sync vectors and delta computation:

```python
class MemorySyncManager:
    def get_sync_vector(node_id: str, tier: MemoryTier) -> Dict[str, int]
    def compute_delta(local_vector, remote_vector) -> List[str]
    def create_batch(items, target_node, tier, priority) -> MemorySyncBatch
    def apply_batch(batch: MemorySyncBatch) -> SyncResult
    def resolve_conflict(local_item, remote_item) -> MemorySyncItem
```

---

## Hub-Side: HubMemshadowGateway

### Responsibilities

1. **Node Registration**
   - Record memory tier capabilities per node
   - Initialize sync vectors for each (node, tier) pair
   - Track node health and sync status

2. **Sync Scheduling**
   - `schedule_sync(node_id, tier, priority)`: Queue sync operation
   - Priority-aware scheduling (CRITICAL immediate, LOW batched)
   - Rate limiting per node

3. **Batch Routing**
   - `apply_remote_batch(node_id, batch)`: Apply incoming batch
   - Route batches to appropriate nodes based on tier and priority
   - Fan-out broadcasts for global updates

4. **Statistics**
   - `get_hub_memory_stats()`: Per-node/per-tier sync metrics
   - Conflict counts and resolution outcomes
   - P2P vs hub-relayed ratios

### Integration with Hub Orchestrator

```python
class HubOrchestrator:
    def __init__(self):
        self.memshadow_gateway = HubMemshadowGateway(self)
        # Register MEMORY_SYNC handler
        self.mesh.on_message(MessageTypes.MEMORY_SYNC, 
                            self.memshadow_gateway.handle_memory_sync)
    
    def register_node(self, node_id, endpoint, capabilities):
        # Existing registration
        node = super().register_node(...)
        # Record memory capabilities
        self.memshadow_gateway.register_memory_node(
            node_id, 
            capabilities.memory_tiers
        )
```

---

## Spoke-Side: SpokeMemoryAdapter

### Responsibilities

1. **Local Tier Access**
   - Wrap L1/L2/L3 local tiers
   - Provide unified interface for sync operations

2. **Delta Batch Creation**
   - `create_delta_batch(tier, target_node, since_timestamp)`: Generate sync batch
   - Compress embeddings for transmission
   - Compute checksums

3. **Batch Application**
   - `apply_sync_batch(batch)`: Apply incoming batch to local tiers
   - Handle conflicts per federation rules
   - Update local sync vectors

4. **Self-Improvement Propagation**
   - Pattern cache updates → MEMSHADOW sync path
   - Model weight deltas → MEMSHADOW sync path

### Integration with Spoke Client

```python
class SpokeClient:
    def __init__(self):
        self.memory_adapter = SpokeMemoryAdapter(self.local_tiers)
        # Register MEMORY_SYNC handler
        self.mesh.on_message(MessageTypes.MEMORY_SYNC,
                            self.memory_adapter.handle_sync_request)
```

---

## Protocol Framing

### MEMORY_SYNC Message

```
┌────────────────────────────────────────────────┐
│ MEMSHADOW Header (32 bytes)                    │
├────────────────────────────────────────────────┤
│ magic     : 0x4D534857 (8 bytes)               │
│ version   : 2 (2 bytes)                        │
│ priority  : 0-4 (2 bytes)                      │
│ msg_type  : 0x0304 MEMORY_SYNC (2 bytes)       │
│ flags     : BATCHED|COMPRESSED|... (2 bytes)   │
│ batch_count: N items (2 bytes)                 │
│ payload_len: size (8 bytes)                    │
│ timestamp_ns: nanoseconds (8 bytes)            │
│ reserved  : (8 bytes)                          │
└────────────────────────────────────────────────┘
│ Payload (variable length)                      │
├────────────────────────────────────────────────┤
│ batch_id  : UUID (16 bytes)                    │
│ source_node: string (var)                      │
│ target_node: string (var)                      │
│ tier      : L1/L2/L3 (1 byte)                  │
│ items[]   : MemorySyncItem array               │
│ checksum  : SHA256 (32 bytes)                  │
└────────────────────────────────────────────────┘
```

### Routing Rules

| Priority | Routing | Description |
|----------|---------|-------------|
| EMERGENCY (4) | Direct P2P + Hub notify | Immediate action required |
| CRITICAL (3) | Direct P2P + Hub notify | Urgent alerts |
| HIGH (2) | Hub-relayed, priority queue | Important updates |
| NORMAL (1) | Hub-relayed, standard queue | Standard sync |
| LOW (0) | Hub-relayed, background | Background operations |

---

## Observability

### Metrics

1. **Per-Node Metrics**
   - `memshadow_sync_frequency{node_id, tier}`: Syncs per minute
   - `memshadow_batch_size{node_id, tier}`: Average items per batch
   - `memshadow_sync_latency_ms{node_id, tier}`: Sync round-trip time

2. **Global Metrics**
   - `memshadow_conflicts_total{resolution}`: Conflict counts
   - `memshadow_p2p_ratio`: P2P vs hub-relayed percentage
   - `memshadow_sync_backlog`: Pending sync operations

### Logs

```python
logger.info("MEMORY_SYNC received",
    source=node_id,
    tier=tier,
    batch_id=batch.batch_id,
    items=len(batch.items),
    priority=priority.name)

logger.info("MEMORY_SYNC applied",
    batch_id=batch.batch_id,
    applied=applied_count,
    conflicts=conflict_count,
    latency_ms=latency)
```

---

## Test Plan

### Integration Tests (`test_hub_memshadow_sync.py`)

1. **Basic Sync Flow**
   - Hub + 2 spokes with in-memory tiers
   - Write on Spoke 1 → verify propagation to Hub → verify propagation to Spoke 2

2. **Priority Routing**
   - NORMAL priority → verify hub relay
   - CRITICAL priority → verify P2P + hub notification

3. **Conflict Resolution**
   - Concurrent writes on Spoke 1 and Spoke 2
   - Verify conflict detection and resolution

4. **Tier-Specific Sync**
   - L1 (hot) sync with low latency
   - L3 (cold) sync with batch compression

5. **Failure Handling**
   - Hub offline → P2P continues for CRITICAL
   - Spoke offline → queued sync on reconnect

---

## Migration Path

1. **Phase 1**: Deploy `HubMemshadowGateway` without breaking changes
2. **Phase 2**: Deploy `SpokeMemoryAdapter` to spoke nodes
3. **Phase 3**: Enable MEMORY_SYNC message handlers
4. **Phase 4**: Migrate self-improvement to use MEMSHADOW sync
5. **Phase 5**: Deprecate legacy sync protocols

---

## File Structure

```
app/services/
├── memshadow/
│   ├── __init__.py
│   ├── protocol.py           # MEMSHADOW Protocol v2 implementation
│   ├── hub_memshadow_gateway.py  # Hub-side gateway
│   ├── spoke_memory_adapter.py   # Spoke-side adapter
│   └── sync_manager.py       # MemorySyncManager
└── neural_storage/
    └── orchestrator.py       # Updated with gateway integration

tests/
└── test_hub_memshadow_sync.py    # Integration tests
```

---

## References

- `HUB_DOCS/DSMIL Brain Federation.md`
- `HUB_DOCS/MEMSHADOW PROTOCOL.md`
- `HUB_DOCS/MEMSHADOW_INTEGRATION.md`
