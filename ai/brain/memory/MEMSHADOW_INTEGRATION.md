# MEMSHADOW Integration for Brain Memory Tiers

Complete guide for integrating MEMSHADOW Protocol v2 with Brain memory tiers.

## Memory Tiers

### L1 - Working Memory
- **Location:** `working_memory.py`
- **Storage:** RAMDISK (or in-memory)
- **Capacity:** Limited (default 1000 items)
- **Embedding Dim:** 256 (compressed for speed)
- **Eviction:** LRU
- **Sync Priority:** NORMAL/HIGH

### L2 - Episodic Memory
- **Location:** `episodic_memory.py`
- **Storage:** NVMe
- **Capacity:** 10,000 episodes
- **Embedding Dim:** 2048 (full resolution)
- **Organization:** Session/episode-based
- **Sync Priority:** NORMAL

### L3 - Semantic Memory
- **Location:** `semantic_memory.py`
- **Storage:** Cold storage
- **Capacity:** 100,000 concepts
- **Embedding Dim:** 4096 (maximum fidelity)
- **Organization:** Concept/relationship graph
- **Compression:** Enabled by default
- **Sync Priority:** LOW/BACKGROUND

## Sync Protocol

### MemorySyncItem

Individual memory item for synchronization:

```python
from ai.brain.memory import MemorySyncItem, SyncPriority

item = MemorySyncItem.create(
    payload=b"embedding data",
    tier=MemoryTier.WORKING,
    operation=SyncOperation.INSERT,
    priority=SyncPriority.NORMAL,
    source_node="node-001",
)
```

Fields:
- `item_id`: UUID
- `timestamp_ns`: Nanosecond timestamp
- `tier`: L1/L2/L3
- `operation`: INSERT, UPDATE, DELETE, MERGE, REPLICATE
- `priority`: BACKGROUND, LOW, NORMAL, HIGH, URGENT
- `payload`: Compressed bytes

### MemorySyncBatch

Batch of items for efficient transmission:

```python
from ai.brain.memory import MemorySyncBatch

batch = MemorySyncBatch(
    source_node="node-001",
    target_node="hub",
    tier=MemoryTier.WORKING,
    items=[item1, item2, item3],
)

# Convert to MEMSHADOW message
msg = batch.to_memshadow_message()
packed = msg.pack()
```

### MemorySyncManager

Central manager for sync operations:

```python
from ai.brain.memory import MemorySyncManager

manager = MemorySyncManager(node_id="node-001")

# Register memory tiers
manager.register_memory_tier(MemoryTier.WORKING, working_memory)
manager.register_memory_tier(MemoryTier.EPISODIC, episodic_memory)

# Create delta batch
batch = await manager.create_delta_batch(
    peer_id="node-002",
    tier=MemoryTier.WORKING,
)

# Apply incoming batch with conflict resolution
result = await manager.apply_sync_batch(batch)
```

## Conflict Resolution

Default: Last-Write-Wins (LWW)

1. Compare timestamps
2. If equal, compare node_id lexicographically
3. Higher timestamp (or node_id) wins

Alternative strategies:
- `first_write_wins`
- `merge` (application-specific)
- `manual` (requires human review)

## Priority Routing

| Priority | Value | Routing | Use Case |
|----------|-------|---------|----------|
| BACKGROUND | 0 | Hub, batched | Bulk sync |
| LOW | 0 | Hub routing | Non-critical |
| NORMAL | 1 | Hub routing | Standard ops |
| HIGH | 2 | Hub priority queue | Important |
| URGENT | 4 | P2P + hub | Critical updates |

## Performance Targets

- **Batch Size:** 10-100 items (configurable)
- **Background Sync:** Every 30 seconds
- **Urgent Sync:** Immediate (<100ms)
- **Compression:** For batches > 1KB

## Configuration

```python
from ai.brain.memory import SyncConfig

config = SyncConfig(
    batch_size_min=10,
    batch_size_max=100,
    compression_threshold_bytes=1024,
    background_sync_interval_sec=30.0,
    urgent_sync_delay_sec=0.1,
    max_retries=3,
    conflict_resolution="last_write_wins",
)
```

## Integration with Federation

The `SpokeMemoryAdapter` integrates with `SpokeClient`:

```python
from ai.brain.federation import SpokeClient, InMemoryTier

spoke = SpokeClient(node_id="node-001", hub_endpoint="hub:8000")

# Register tiers
l1 = InMemoryTier(MemoryTier.WORKING)
spoke.memory_adapter.register_tier(MemoryTier.WORKING, l1)

# Store data (automatically tracked for sync)
item_id = await spoke.memory_adapter.store(
    tier=MemoryTier.WORKING,
    data=b"important data",
)

# Sync to hub
await spoke.memory_adapter.sync_to_hub(MemoryTier.WORKING)
```

## Testing

Run integration tests:

```bash
python3 test_memshadow_integration.py
```

## Related Documentation

- [MEMSHADOW Protocol](../../../libs/memshadow-protocol/README.md)
- [Brain Federation](../federation/README.md)
- [Hub Integration](../../../HUB_DOCS/MEMSHADOW_INTEGRATION.md)
