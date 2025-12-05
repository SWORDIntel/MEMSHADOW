# Memory Tier MEMSHADOW Integration

All memory tiers (L1/L2/L3) support MEMSHADOW protocol v2 for distributed synchronization.

## Overview

Memory synchronization enables:
- Cross-node memory sharing
- Incremental sync (delta updates)
- Conflict resolution
- Priority-based replication
- Automatic tier promotion

## Sync Protocol

### MemorySyncItem

Single item in a sync operation (48-byte header + payload):

```python
item = MemorySyncItem(
    item_id="32-char-hex-id",
    timestamp_ns=int(time.time() * 1e9),
    tier=MemoryTier.WORKING,
    operation=SyncOperation.INSERT,
    priority=SyncPriority.NORMAL,
    payload=json.dumps({"data": "value"}).encode()
)
```

### MemorySyncBatch

Batch of sync items with MEMSHADOW header:

```python
batch = MemorySyncBatch(
    batch_id="batch-001",
    source_node="node-a",
    target_node="node-b",
    tier=MemoryTier.WORKING,
    items=[item1, item2, ...]
)
```

### Sync Operations

- `INSERT` - New item
- `UPDATE` - Modify existing item
- `DELETE` - Remove item
- `MERGE` - Merge with existing item
- `REPLICATE` - Full replication request

### Sync Priorities

- `BACKGROUND` - Low priority sync
- `LOW` - Standard sync
- `NORMAL` - Normal priority
- `HIGH` - Important updates
- `URGENT` - Critical sync

## Memory Tier Methods

All tiers implement sync protocol methods:

### Working Memory (L1)

```python
wm = WorkingMemory()

# Sync protocol methods
modified = wm.get_modified_since(timestamp_ns)
item_dict = wm.get_by_id(item_id)
wm.add(item_id, content_dict)
wm.update(item_id, content_dict)
wm.delete(item_id)
wm.merge(item_id, content_dict)
```

### Episodic Memory (L2)

Same interface as L1, optimized for episode data.

### Semantic Memory (L3)

Same interface as L1, optimized for knowledge graph nodes.

## Sync Manager

`MemorySyncManager` coordinates synchronization:

```python
sync_manager = MemorySyncManager(node_id="node-001", mesh=mesh_instance)

# Register memory tiers
sync_manager.register_memory_tier(MemoryTier.WORKING, working_memory)
sync_manager.register_memory_tier(MemoryTier.EPISODIC, episodic_memory)
sync_manager.register_memory_tier(MemoryTier.SEMANTIC, semantic_memory)

# Create delta batch
batch = sync_manager.create_delta_batch(
    tier=MemoryTier.WORKING,
    target_node="node-002",
    since_timestamp=last_sync_timestamp
)

# Apply received batch
applied, conflicts = sync_manager.apply_sync_batch(batch)

# Sync with peer
result = await sync_manager.sync_with_peer("node-002", MemoryTier.WORKING)
```

## Conflict Resolution

Default strategy: **Last-Write-Wins (LWW)**

1. Compare timestamps
2. If equal, use node_id as tie-breaker
3. Log conflicts for analysis
4. Apply winning version

## Delta Sync

Only modified items since last sync are transmitted:

```python
# Get items modified since timestamp
modified = memory_tier.get_modified_since(last_sync_timestamp_ns)

# Create batch with only changes
batch = sync_manager.create_delta_batch(
    tier=MemoryTier.WORKING,
    target_node="peer-node",
    since_timestamp=last_sync_timestamp_ns
)
```

## Sync Vectors

Track last sync timestamp per peer per tier:

```python
# Get sync vector for tier
sync_vector = sync_manager.get_sync_vector(MemoryTier.WORKING)
# Returns: {"node-002": 1234567890, "node-003": 1234567800}

# Update after successful sync
sync_manager.update_sync_vector(
    tier=MemoryTier.WORKING,
    node_id="node-002",
    timestamp=1234567890
)
```

## Compression

Large batches are automatically compressed:

```python
# Batch > 1KB is compressed with gzip
if len(items_data) > 1024:
    items_data = gzip.compress(items_data)
    batch.flags |= SyncFlags.COMPRESSED
```

## Integration with Mesh Network

Sync batches are sent via DSMIL-Mesh:

```python
from messages import MessageTypes

# Send sync batch
mesh.send(peer_id, MessageTypes.VECTOR_SYNC, batch.pack())

# Receive sync batch
def handle_sync(data: bytes, peer_id: str):
    batch = MemorySyncBatch.unpack(data)
    applied, conflicts = sync_manager.apply_sync_batch(batch)
```

## Testing

Test memory sync:

```python
from ai.brain.memory.memory_sync_protocol import (
    MemorySyncItem, MemorySyncBatch, MemoryTier, SyncOperation
)

# Create test item
item = MemorySyncItem(...)
packed = item.pack()
unpacked, _ = MemorySyncItem.unpack(packed)

# Create batch
batch = MemorySyncBatch(...)
packed_batch = batch.pack()
unpacked_batch = MemorySyncBatch.unpack(packed_batch)
```

## Performance Considerations

- **Batch Size:** Optimal 10-100 items per batch
- **Sync Frequency:** Background sync every 30s, urgent sync immediately
- **Compression:** Enabled for batches > 1KB
- **Delta Only:** Only sync modified items since last sync

## Related Documentation

- [MEMSHADOW Protocol](../../../libs/memshadow-protocol/README.md)
- [Working Memory](working_memory.py)
- [Episodic Memory](episodic_memory.py)
- [Semantic Memory](semantic_memory.py)
