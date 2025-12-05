# MEMSHADOW Protocol

Unified binary wire format for intra-node communications within the DSMIL ecosystem.

## Overview

The MEMSHADOW Protocol (v2) provides a standardized 32-byte header format for all messages exchanged between:
- Kernel modules ↔ Userspace processes
- Mesh network nodes (hub ↔ spokes, spoke ↔ spoke)
- Memory tier synchronization
- Self-improvement propagation

## Protocol Version

**Current Version:** 2.0  
**Header Size:** 32 bytes  
**Magic Number:** `0x4D534857` ("MSHW" in ASCII)

## Header Structure

```c
struct dsmil_msg_header_t {
    __u64 magic;          // 8 bytes - Protocol magic (0x4D534857)
    __u16 version;        // 2 bytes - Protocol version (2)
    __u16 priority;       // 2 bytes - Message priority (0-4)
    __u16 msg_type;       // 2 bytes - Message type (see MessageType enum)
    __u16 flags;          // 2 bytes - Message flags (encrypted, compressed, etc.)
    __u16 batch_count;    // 2 bytes - Number of items in batch (0=single)
    __u64 payload_len;    // 8 bytes - Payload length in bytes
    __u64 timestamp_ns;   // 8 bytes - Nanosecond timestamp
    __u64 reserved;       // 8 bytes - Reserved for future expansion
};  // Total: 32 bytes
```

### Network Byte Order

All multi-byte fields are in **network byte order (big-endian)** for cross-platform compatibility.

## Message Types

### System/Control (0x00xx)
- `HEARTBEAT` (0x0001)
- `ACK` (0x0002)
- `ERROR` (0x0003)
- `HANDSHAKE` (0x0004)
- `DISCONNECT` (0x0005)

### SHRINK Psychological Intelligence (0x01xx)
- `PSYCH_ASSESSMENT` (0x0100) - Full psychological assessment
- `DARK_TRIAD_UPDATE` (0x0101) - Dark triad scores update
- `RISK_UPDATE` (0x0102) - Risk scores update
- `NEURO_UPDATE` (0x0103) - Neuro indicators update
- `TMI_UPDATE` (0x0104) - TMI scores update
- `COGARCH_UPDATE` (0x0105) - Cognitive architecture update
- `PSYCH_THREAT_ALERT` (0x0110) - Psychological threat indicator
- `PSYCH_ANOMALY` (0x0111) - Behavioral anomaly detected
- `PSYCH_RISK_THRESHOLD` (0x0112) - Risk threshold exceeded

### Threat Intelligence (0x02xx)
- `THREAT_REPORT` (0x0201)
- `INTEL_REPORT` (0x0202)
- `KNOWLEDGE_UPDATE` (0x0203)

### Memory Operations (0x03xx)
- `MEMORY_STORE` (0x0301)
- `MEMORY_QUERY` (0x0302)
- `MEMORY_RESPONSE` (0x0303)
- `MEMORY_SYNC` (0x0304) - Memory tier synchronization

### Federation/Mesh (0x04xx)
- `NODE_REGISTER` (0x0401)
- `NODE_DEREGISTER` (0x0402)
- `QUERY_DISTRIBUTE` (0x0403)
- `QUERY_RESPONSE` (0x0404)
- `INTEL_PROPAGATE` (0x0405)

### Self-Improvement (0x05xx)
- `IMPROVEMENT_ANNOUNCE` (0x0501) - Broadcast improvement availability
- `IMPROVEMENT_REQUEST` (0x0502) - Request specific improvement
- `IMPROVEMENT_PAYLOAD` (0x0503) - Actual improvement data
- `IMPROVEMENT_ACK` (0x0504) - Confirm receipt and application
- `IMPROVEMENT_REJECT` (0x0505) - Reject incompatible improvement
- `IMPROVEMENT_METRICS` (0x0506) - Performance metrics share

## Priority Levels

- `LOW` (0) - Background operations
- `NORMAL` (1) - Standard operations
- `HIGH` (2) - Important updates
- `CRITICAL` (3) - Urgent alerts
- `EMERGENCY` (4) - Immediate action required

## Message Flags

- `ENCRYPTED` (0x0001) - Payload is encrypted
- `COMPRESSED` (0x0002) - Payload is compressed
- `BATCHED` (0x0004) - Contains multiple items
- `REQUIRES_ACK` (0x0008) - Requires acknowledgment
- `FRAGMENTED` (0x0010) - Message is fragmented
- `LAST_FRAGMENT` (0x0020) - Last fragment in sequence
- `FROM_KERNEL` (0x0040) - Originated from kernel module
- `HIGH_CONFIDENCE` (0x0080) - High confidence in data
- `PQC_SIGNED` (0x0100) - Post-quantum cryptography signed

## Psychological Event Structure

SHRINK kernel module emits 64-byte psychological events:

```c
struct dsmil_psych_event_t {
    __u64 session_id;           // 8 bytes - Session identifier
    __u32 timestamp_offset_us;  // 4 bytes - Microsecond offset from session start
    __u8  event_type;            // 1 byte  - Event type (KEYPRESS, SCORE_UPDATE, etc.)
    __u8  flags;                // 1 byte  - Event flags
    __u16 window_size;           // 2 bytes - Analysis window size
    __u64 context_hash;          // 8 bytes - Context hash
    float acute_stress;          // 4 bytes - Acute stress score
    float machiavellianism;      // 4 bytes - Machiavellianism score
    float narcissism;            // 4 bytes - Narcissism score
    float psychopathy;           // 4 bytes - Psychopathy score
    float burnout_probability;   // 4 bytes - Burnout probability
    float espionage_exposure;    // 4 bytes - Espionage exposure risk
    float confidence;            // 4 bytes - Confidence in scores
    char  reserved[12];          // 12 bytes - Reserved for expansion
};  // Total: 64 bytes
```

## Usage Examples

### Python

```python
from dsmil_protocol import MemshadowHeader, PsychEvent, MessageType, Priority

# Create header
header = MemshadowHeader(
    magic=0x4D534857,
    version=2,
    priority=Priority.NORMAL,
    msg_type=MessageType.PSYCH_ASSESSMENT,
    flags=0,
    batch_count=1,
    payload_len=64,
    timestamp_ns=int(time.time() * 1e9)
)

# Pack to binary
packed = header.pack()

# Unpack from binary
unpacked = MemshadowHeader.unpack(packed)
```

### C

```c
#include "dsmil_protocol.h"

struct dsmil_msg_header_t header;
dsmil_header_init(&header, MSG_TYPE_PSYCH_ASSESSMENT, PRIORITY_NORMAL, 0);

// Pack to buffer
uint8_t buffer[32];
memcpy(buffer, &header, sizeof(header));
```

## Implementation Files

- **C Header:** `c/include/dsmil_protocol.h`
- **Python Module:** `python/dsmil_protocol.py`
- **Kernel Module:** Used by `external/intel/shrink/kernel_module/shrink_monitor.c`

## Testing

Run the integration test suite:

```bash
python3 test_memshadow_integration.py
```

## References

- [DSMIL Brain Federation Architecture](../../ai/brain/federation/README.md)
- [Memory Sync Protocol](../../ai/brain/memory/MEMSHADOW_INTEGRATION.md)
- [Self-Improvement System](../../ai/brain/federation/improvement_types.py)
