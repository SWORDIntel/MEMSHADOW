# MEMSHADOW Protocol Integration Guide

Complete integration guide for the MEMSHADOW protocol v2 across the DSMIL ecosystem.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Protocol Specification](#protocol-specification)
4. [Integration Points](#integration-points)
5. [Implementation Guide](#implementation-guide)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)

## Overview

The MEMSHADOW Protocol v2 is a unified binary wire format for intra-node communications within the DSMIL ecosystem. It replaces multiple ad-hoc protocols with a single, standardized format.

### Key Features

- **32-byte header** with magic number, version, priority, timestamps
- **Network byte order** for cross-platform compatibility
- **Batching support** for efficient bulk transfers
- **Priority levels** for routing decisions
- **Extensible** with reserved fields for future expansion

### Use Cases

1. **Kernel ↔ Userspace:** SHRINK kernel module to userspace receiver
2. **Mesh Network:** Hub-spoke and P2P node communication
3. **Memory Sync:** Cross-node memory tier synchronization
4. **Self-Improvement:** Propagation of model weights, configs, patterns

## Architecture

### Protocol Stack

```
┌─────────────────────────────────────┐
│   Application Layer                 │
│   (SHRINK, Brain, Memory Tiers)     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   MEMSHADOW Protocol v2              │
│   (32-byte header + payload)         │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   Transport Layer                    │
│   (Netlink, Mesh Network, HTTP)      │
└─────────────────────────────────────┘
```

### Data Flow

```
SHRINK Kernel Module
    │ (Netlink socket, MEMSHADOW binary)
    ▼
Userspace Receiver (kernel_receiver.py)
    │ (HTTP POST, JSON or binary)
    ▼
Brain API Endpoint (/api/v1/ingest/shrink)
    │ (MEMSHADOW ingest plugin)
    ▼
Brain Memory Tiers (L1/L2/L3)
    │ (Significant updates only)
    ▼
Hub Orchestrator
    │ (Mesh network, MEMSHADOW binary)
    ├─► Other Spoke Nodes (hub-relayed)
    └─► Direct P2P (critical updates)
```

## Protocol Specification

### Header Structure

See [MEMSHADOW Protocol README](../libs/memshadow-protocol/README.md) for complete specification.

### Message Types

- **0x00xx:** System/Control
- **0x01xx:** SHRINK Psychological Intelligence
- **0x02xx:** Threat Intelligence
- **0x03xx:** Memory Operations
- **0x04xx:** Federation/Mesh
- **0x05xx:** Self-Improvement

### Priority Routing

- **CRITICAL/EMERGENCY:** Direct P2P bypassing hub
- **HIGH:** Hub-relayed with priority queue
- **NORMAL:** Standard hub routing
- **LOW:** Background processing

## Integration Points

### 1. SHRINK Kernel Module

**File:** `external/intel/shrink/kernel_module/shrink_monitor.c`

**Changes:**
- Emits MEMSHADOW v2 header (32 bytes)
- Batches psych events (64 bytes each)
- Sends via Netlink socket

**Example:**
```c
struct dsmil_msg_header_t header;
dsmil_header_init_kernel(&header, MSG_TYPE_PSYCH_ASSESSMENT, 
                         PRIORITY_NORMAL, 0);
// ... populate header ...
netlink_send(&header, sizeof(header));
```

### 2. Userspace Receiver

**File:** `external/intel/shrink/shrink/kernel_receiver.py`

**Purpose:**
- Receives Netlink messages from kernel
- Parses MEMSHADOW protocol
- Forwards to Brain API

### 3. Brain API Endpoint

**File:** `ai/brain/api/shrink_endpoint.py`

**Endpoint:** `POST /api/v1/ingest/shrink`

**Supports:**
- JSON format (legacy)
- Binary MEMSHADOW format (preferred)

### 4. MEMSHADOW Ingest Plugin

**File:** `ai/brain/plugins/ingest/memshadow_ingest.py`

**Purpose:**
- Parses MEMSHADOW binary messages
- Extracts psych events
- Converts to Brain ingest format

### 5. Hub Orchestrator

**File:** `ai/brain/federation/hub_orchestrator.py`

**Handlers:**
- `_handle_psych_intel()` - Process SHRINK data
- `_handle_psych_threat()` - High-priority alerts
- `_handle_improvement_announce()` - Self-improvement relay

### 6. Spoke Client

**File:** `ai/brain/federation/spoke_client.py`

**Features:**
- P2P improvement propagation
- Direct peer communication
- Psych intel handlers

### 7. Memory Sync Protocol

**File:** `ai/brain/memory/memory_sync_protocol.py`

**Purpose:**
- Synchronize memory tiers across nodes
- Delta sync (only changes)
- Conflict resolution

## Implementation Guide

### Step 1: Install Dependencies

```bash
# Python protocol library
cd libs/memshadow-protocol/python
pip install -e .

# C header (for kernel modules)
# Include: libs/memshadow-protocol/c/include/dsmil_protocol.h
```

### Step 2: Create MEMSHADOW Message

**Python:**
```python
from dsmil_protocol import MemshadowHeader, PsychEvent, MessageType, Priority

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

psych_event = PsychEvent(
    session_id=12345,
    timestamp_offset_us=0,
    event_type=PsychEventType.SCORE_UPDATE,
    flags=0,
    window_size=100,
    context_hash=0,
    acute_stress=0.5,
    machiavellianism=0.3,
    narcissism=0.4,
    psychopathy=0.2,
    burnout_probability=0.1,
    espionage_exposure=0.05,
    confidence=0.85
)

# Pack to binary
message = header.pack() + psych_event.pack()
```

**C:**
```c
#include "dsmil_protocol.h"

struct dsmil_msg_header_t header;
struct dsmil_psych_event_t event;

dsmil_header_init_kernel(&header, MSG_TYPE_PSYCH_ASSESSMENT, 
                         PRIORITY_NORMAL, 0);
// ... populate event ...
```

### Step 3: Send via Transport

**Netlink (Kernel → Userspace):**
```c
netlink_send(&header, sizeof(header));
netlink_send(&event, sizeof(event));
```

**HTTP (Userspace → Brain):**
```python
import requests

response = requests.post(
    'http://brain.local:8000/api/v1/ingest/shrink',
    data=message,
    headers={'Content-Type': 'application/octet-stream'}
)
```

**Mesh Network (Node → Node):**
```python
from messages import MessageTypes

mesh.send(peer_id, MessageTypes.PSYCH_ASSESSMENT, message)
```

### Step 4: Receive and Parse

**Python:**
```python
from dsmil_protocol import MemshadowHeader, PsychEvent

# Parse header
header = MemshadowHeader.unpack(data[:32])

# Parse payload
if header.msg_type == MessageType.PSYCH_ASSESSMENT:
    event = PsychEvent.unpack(data[32:96])
```

## Testing

### Unit Tests

```bash
# Test protocol packing/unpacking
python3 -m pytest libs/memshadow-protocol/python/tests/

# Test integration
python3 test_memshadow_integration.py
```

### Integration Test Results

All 6 tests passed:
- ✓ MEMSHADOW Protocol
- ✓ Hub Orchestrator
- ✓ Memory Sync Protocol
- ✓ Working Memory Integration
- ✓ Improvement Types
- ✓ Spoke Client P2P

## Troubleshooting

### Common Issues

1. **Invalid Magic Number**
   - Check: Header magic should be `0x4D534857`
   - Fix: Ensure network byte order

2. **Size Mismatch**
   - Check: Header must be exactly 32 bytes
   - Fix: Use `struct.pack()` with correct format string

3. **Import Errors**
   - Check: Python path includes protocol library
   - Fix: Add to `sys.path` or install package

4. **Parse Errors**
   - Check: Payload length matches header
   - Fix: Verify `payload_len` field

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Related Documentation

- [MEMSHADOW Protocol README](../libs/memshadow-protocol/README.md)
- [Brain Federation README](../ai/brain/federation/README.md)
- [Memory Sync Protocol](../ai/brain/memory/MEMSHADOW_INTEGRATION.md)
- [Brain API README](../ai/brain/api/README.md)

## Version History

- **v2.0** (Current): 32-byte header, nanosecond timestamps, reserved fields
- **v1.0** (Legacy): 16-byte header, deprecated

## Support

For issues or questions:
1. Check this documentation
2. Review test suite for examples
3. Examine source code comments
4. Check `dmesg` for kernel module errors
