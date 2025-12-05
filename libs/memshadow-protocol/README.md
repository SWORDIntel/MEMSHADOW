# MEMSHADOW Protocol Library

Unified binary wire format for intra-node communications within the DSMIL ecosystem.

## Protocol Version

- **Version:** 2.0
- **Header Size:** 32 bytes
- **Magic Number:** `0x4D534857` ("MSHW" in ASCII)
- **Byte Order:** Network (big-endian)

## Header Structure

```c
struct dsmil_msg_header_t {
    __u64 magic;          // 8 bytes - Protocol magic (0x4D534857)
    __u16 version;        // 2 bytes - Protocol version (2)
    __u16 priority;       // 2 bytes - Message priority (0-4)
    __u16 msg_type;       // 2 bytes - Message type
    __u16 flags_batch;    // 2 bytes - Flags (low byte) + batch_count (high byte)
    __u64 payload_len;    // 8 bytes - Payload length in bytes
    __u64 timestamp_ns;   // 8 bytes - Nanosecond timestamp
};  // Total: 32 bytes
```

## Message Types

### System/Control (0x00xx)
- `HEARTBEAT` (0x0001)
- `ACK` (0x0002)
- `ERROR` (0x0003)
- `HANDSHAKE` (0x0004)
- `DISCONNECT` (0x0005)

### SHRINK Psychological Intelligence (0x01xx)
- `PSYCH_ASSESSMENT` (0x0100)
- `DARK_TRIAD_UPDATE` (0x0101)
- `RISK_UPDATE` (0x0102)
- `NEURO_UPDATE` (0x0103)
- `TMI_UPDATE` (0x0104)
- `COGARCH_UPDATE` (0x0105)
- `PSYCH_THREAT_ALERT` (0x0110)
- `PSYCH_ANOMALY` (0x0111)
- `PSYCH_RISK_THRESHOLD` (0x0112)

### Threat Intelligence (0x02xx)
- `THREAT_REPORT` (0x0201)
- `INTEL_REPORT` (0x0202)
- `KNOWLEDGE_UPDATE` (0x0203)

### Memory Operations (0x03xx)
- `MEMORY_STORE` (0x0301)
- `MEMORY_QUERY` (0x0302)
- `MEMORY_RESPONSE` (0x0303)
- `MEMORY_SYNC` (0x0304)

### Federation/Mesh (0x04xx)
- `NODE_REGISTER` (0x0401)
- `NODE_DEREGISTER` (0x0402)
- `QUERY_DISTRIBUTE` (0x0403)
- `QUERY_RESPONSE` (0x0404)
- `INTEL_PROPAGATE` (0x0405)

### Self-Improvement (0x05xx)
- `IMPROVEMENT_ANNOUNCE` (0x0501)
- `IMPROVEMENT_REQUEST` (0x0502)
- `IMPROVEMENT_PAYLOAD` (0x0503)
- `IMPROVEMENT_ACK` (0x0504)
- `IMPROVEMENT_REJECT` (0x0505)
- `IMPROVEMENT_METRICS` (0x0506)

## Priority Levels

| Priority | Value | Routing |
|----------|-------|---------|
| LOW | 0 | Background, hub routing |
| NORMAL | 1 | Standard hub routing |
| HIGH | 2 | Hub-relayed with priority queue |
| CRITICAL | 3 | Direct P2P + hub notification |
| EMERGENCY | 4 | Immediate P2P action |

## Message Flags

| Flag | Value | Description |
|------|-------|-------------|
| ENCRYPTED | 0x0001 | Payload is encrypted |
| COMPRESSED | 0x0002 | Payload is compressed |
| BATCHED | 0x0004 | Contains multiple items |
| REQUIRES_ACK | 0x0008 | Requires acknowledgment |
| FRAGMENTED | 0x0010 | Message is fragmented |
| LAST_FRAGMENT | 0x0020 | Last fragment in sequence |
| FROM_KERNEL | 0x0040 | Originated from kernel module |
| HIGH_CONFIDENCE | 0x0080 | High confidence in data |
| PQC_SIGNED | 0x0100 | Post-quantum cryptography signed |

## Python Usage

```python
from dsmil_protocol import (
    MemshadowHeader,
    MemshadowMessage,
    MessageType,
    Priority,
    MessageFlags,
    should_route_p2p,
)

# Create a message
header = MemshadowHeader(
    priority=Priority.NORMAL,
    msg_type=MessageType.MEMORY_SYNC,
    flags=MessageFlags.BATCHED,
    payload_len=1024,
)

# Pack to binary
packed = header.pack()
assert len(packed) == 32

# Unpack from binary
unpacked = MemshadowHeader.unpack(packed)
assert unpacked.msg_type == MessageType.MEMORY_SYNC

# Create complete message
msg = MemshadowMessage.create(
    msg_type=MessageType.MEMORY_SYNC,
    payload=b"sync data here",
    priority=Priority.HIGH,
)

# Check routing
if should_route_p2p(msg.header.priority):
    # Use P2P routing
    pass
else:
    # Use hub relay
    pass
```

## Testing

```bash
python3 -m pytest python/tests/
```

## Files

- `python/dsmil_protocol.py` - Python implementation
- `c/include/dsmil_protocol.h` - C header (for kernel modules)

## Related Documentation

- [DSMIL Brain Federation](../../ai/brain/federation/README.md)
- [Memory Sync Protocol](../../ai/brain/memory/MEMSHADOW_INTEGRATION.md)
- [HUB Integration](../../HUB_DOCS/MEMSHADOW_INTEGRATION.md)
