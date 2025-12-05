"""
DSMIL MEMSHADOW Protocol v2 - Canonical Implementation

Unified binary wire format for intra-node communications within the DSMIL ecosystem.

This is the canonical protocol library referenced by:
- ai/brain/federation/hub_orchestrator.py
- ai/brain/federation/spoke_client.py
- ai/brain/memory/memory_sync_protocol.py
- external/intel/shrink/shrink/kernel_receiver.py

Protocol Version: 2.0
Header Size: 32 bytes
Magic Number: 0x4D534857 ("MSHW" in ASCII)

Usage:
    from dsmil_protocol import MemshadowHeader, MessageType, Priority, MessageFlags
    
    header = MemshadowHeader(
        priority=Priority.NORMAL,
        msg_type=MessageType.MEMORY_SYNC,
        payload_len=1024
    )
    packed = header.pack()
    unpacked = MemshadowHeader.unpack(packed)
"""

import struct
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, IntFlag
from typing import Any, Dict, Optional, Tuple

# ============================================================================
# Protocol Constants
# ============================================================================

MEMSHADOW_MAGIC = 0x4D534857  # "MSHW" in ASCII
MEMSHADOW_VERSION = 2
HEADER_SIZE = 32


# ============================================================================
# Enumerations (match HUB_DOCS/MEMSHADOW PROTOCOL.md exactly)
# ============================================================================

class MessageType(IntEnum):
    """MEMSHADOW Protocol v2 Message Types"""
    
    # System/Control (0x00xx)
    HEARTBEAT = 0x0001
    ACK = 0x0002
    ERROR = 0x0003
    HANDSHAKE = 0x0004
    DISCONNECT = 0x0005
    
    # SHRINK Psychological Intelligence (0x01xx)
    PSYCH_ASSESSMENT = 0x0100
    DARK_TRIAD_UPDATE = 0x0101
    RISK_UPDATE = 0x0102
    NEURO_UPDATE = 0x0103
    TMI_UPDATE = 0x0104
    COGARCH_UPDATE = 0x0105
    PSYCH_THREAT_ALERT = 0x0110
    PSYCH_ANOMALY = 0x0111
    PSYCH_RISK_THRESHOLD = 0x0112
    
    # Threat Intelligence (0x02xx)
    THREAT_REPORT = 0x0201
    INTEL_REPORT = 0x0202
    KNOWLEDGE_UPDATE = 0x0203
    
    # Memory Operations (0x03xx)
    MEMORY_STORE = 0x0301
    MEMORY_QUERY = 0x0302
    MEMORY_RESPONSE = 0x0303
    MEMORY_SYNC = 0x0304
    
    # Federation/Mesh (0x04xx)
    NODE_REGISTER = 0x0401
    NODE_DEREGISTER = 0x0402
    QUERY_DISTRIBUTE = 0x0403
    QUERY_RESPONSE = 0x0404
    INTEL_PROPAGATE = 0x0405
    
    # Self-Improvement (0x05xx)
    IMPROVEMENT_ANNOUNCE = 0x0501
    IMPROVEMENT_REQUEST = 0x0502
    IMPROVEMENT_PAYLOAD = 0x0503
    IMPROVEMENT_ACK = 0x0504
    IMPROVEMENT_REJECT = 0x0505
    IMPROVEMENT_METRICS = 0x0506


class Priority(IntEnum):
    """
    Message priority levels for routing decisions.
    
    Routing rules:
    - LOW (0): Background processing
    - NORMAL (1): Standard hub routing
    - HIGH (2): Hub-relayed with priority queue
    - CRITICAL (3): Direct P2P + hub notification
    - EMERGENCY (4): Immediate P2P action required
    """
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 4
    
    # Aliases for sync operations (map to standard levels)
    BACKGROUND = 0
    URGENT = 4
    
    def should_use_p2p(self) -> bool:
        """Check if this priority should use direct P2P routing"""
        return self >= Priority.CRITICAL
    
    def should_require_ack(self) -> bool:
        """Check if this priority should require acknowledgment"""
        return self >= Priority.HIGH
    
    def is_hub_relayed(self) -> bool:
        """Check if this priority uses hub relay (not P2P)"""
        return self < Priority.CRITICAL


class MessageFlags(IntFlag):
    """Message flags for payload handling"""
    NONE = 0x0000
    ENCRYPTED = 0x0001       # Payload is encrypted
    COMPRESSED = 0x0002      # Payload is compressed
    BATCHED = 0x0004         # Contains multiple items
    REQUIRES_ACK = 0x0008    # Requires acknowledgment
    FRAGMENTED = 0x0010      # Message is fragmented
    LAST_FRAGMENT = 0x0020   # Last fragment in sequence
    FROM_KERNEL = 0x0040     # Originated from kernel module
    HIGH_CONFIDENCE = 0x0080 # High confidence in data
    PQC_SIGNED = 0x0100      # Post-quantum cryptography signed


class SyncOperation(IntEnum):
    """Memory sync operations"""
    INSERT = 1
    UPDATE = 2
    DELETE = 3
    MERGE = 4
    REPLICATE = 5


class MemoryTier(IntEnum):
    """Memory tier levels"""
    WORKING = 1    # L1 - Hot working memory
    EPISODIC = 2   # L2 - Episodic/session memory
    SEMANTIC = 3   # L3 - Long-term semantic memory
    
    # Aliases
    L1 = 1
    L2 = 2
    L3 = 3


# ============================================================================
# Header Structure
# ============================================================================

@dataclass
class MemshadowHeader:
    """
    MEMSHADOW Protocol v2 Header (32 bytes)
    
    Wire format (big-endian):
        magic         : 8 bytes - Protocol magic (0x4D534857)
        version       : 2 bytes - Protocol version (2)
        priority      : 2 bytes - Message priority (0-4)
        msg_type      : 2 bytes - Message type
        flags         : 2 bytes - Message flags
        batch_count   : 2 bytes - Number of items in batch (0=single)
        payload_len   : 8 bytes - Payload length in bytes
        timestamp_ns  : 8 bytes - Nanosecond timestamp
    
    Total: 32 bytes exactly
    """
    magic: int = MEMSHADOW_MAGIC
    version: int = MEMSHADOW_VERSION
    priority: Priority = Priority.NORMAL
    msg_type: MessageType = MessageType.HEARTBEAT
    flags: MessageFlags = MessageFlags.NONE
    batch_count: int = 0
    payload_len: int = 0
    timestamp_ns: int = field(default_factory=lambda: int(time.time() * 1e9))
    
    # Format: Q(8) + H(2)*4 + Q(8) + Q(8) = 32 bytes
    # magic(8) + version(2) + priority(2) + msg_type(2) + flags_batch(2) + payload_len(8) + timestamp_ns(8)
    _FORMAT = ">QHHHHQQ"
    
    def pack(self) -> bytes:
        """Pack header to binary (32 bytes, network byte order)"""
        # Combine flags (low byte) and batch_count (high byte) into single uint16
        flags_batch = (int(self.flags) & 0xFF) | ((self.batch_count & 0xFF) << 8)
        
        return struct.pack(
            self._FORMAT,
            self.magic,
            self.version,
            int(self.priority),
            int(self.msg_type),
            flags_batch,
            self.payload_len,
            self.timestamp_ns,
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> "MemshadowHeader":
        """Unpack header from binary (32 bytes)"""
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Header too short: {len(data)} bytes, expected {HEADER_SIZE}")
        
        (
            magic,
            version,
            priority,
            msg_type,
            flags_batch,
            payload_len,
            timestamp_ns,
        ) = struct.unpack(cls._FORMAT, data[:HEADER_SIZE])
        
        if magic != MEMSHADOW_MAGIC:
            raise ValueError(f"Invalid magic: 0x{magic:016X}, expected 0x{MEMSHADOW_MAGIC:016X}")
        
        flags = MessageFlags(flags_batch & 0xFF)
        batch_count = (flags_batch >> 8) & 0xFF
        
        return cls(
            magic=magic,
            version=version,
            priority=Priority(priority),
            msg_type=MessageType(msg_type),
            flags=flags,
            batch_count=batch_count,
            payload_len=payload_len,
            timestamp_ns=timestamp_ns,
        )
    
    @property
    def timestamp(self) -> datetime:
        """Get timestamp as datetime"""
        return datetime.fromtimestamp(self.timestamp_ns / 1e9)
    
    def validate(self) -> bool:
        """Validate header fields"""
        return self.magic == MEMSHADOW_MAGIC and self.version == MEMSHADOW_VERSION


# ============================================================================
# Message Container
# ============================================================================

@dataclass
class MemshadowMessage:
    """Complete MEMSHADOW message with header and payload"""
    header: MemshadowHeader
    payload: bytes = b""
    
    def pack(self) -> bytes:
        """Pack complete message to binary"""
        self.header.payload_len = len(self.payload)
        return self.header.pack() + self.payload
    
    @classmethod
    def unpack(cls, data: bytes) -> "MemshadowMessage":
        """Unpack complete message from binary"""
        header = MemshadowHeader.unpack(data[:HEADER_SIZE])
        payload = data[HEADER_SIZE:HEADER_SIZE + header.payload_len]
        return cls(header=header, payload=payload)
    
    @classmethod
    def create(
        cls,
        msg_type: MessageType,
        payload: bytes,
        priority: Priority = Priority.NORMAL,
        flags: MessageFlags = MessageFlags.NONE,
        batch_count: int = 0,
    ) -> "MemshadowMessage":
        """Create a new MEMSHADOW message"""
        header = MemshadowHeader(
            msg_type=msg_type,
            priority=priority,
            flags=flags,
            batch_count=batch_count,
            payload_len=len(payload),
        )
        return cls(header=header, payload=payload)


# ============================================================================
# SHRINK Psychological Event Structure (64 bytes)
# ============================================================================

@dataclass
class PsychEvent:
    """
    SHRINK psychological event (64 bytes).
    
    Wire format matches kernel module struct dsmil_psych_event_t.
    """
    session_id: int = 0
    timestamp_offset_us: int = 0
    event_type: int = 0
    flags: int = 0
    window_size: int = 0
    context_hash: int = 0
    acute_stress: float = 0.0
    machiavellianism: float = 0.0
    narcissism: float = 0.0
    psychopathy: float = 0.0
    burnout_probability: float = 0.0
    espionage_exposure: float = 0.0
    confidence: float = 0.0
    
    # Format: Q(8) + I(4) + B(1) + B(1) + H(2) + Q(8) + 7*f(28) + 12x(12) = 64 bytes
    _FORMAT = ">QIBBHQfffffff12x"
    
    def pack(self) -> bytes:
        """Pack to binary (64 bytes)"""
        return struct.pack(
            self._FORMAT,
            self.session_id,
            self.timestamp_offset_us,
            self.event_type,
            self.flags,
            self.window_size,
            self.context_hash,
            self.acute_stress,
            self.machiavellianism,
            self.narcissism,
            self.psychopathy,
            self.burnout_probability,
            self.espionage_exposure,
            self.confidence,
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> "PsychEvent":
        """Unpack from binary (64 bytes)"""
        if len(data) < 64:
            raise ValueError(f"PsychEvent too short: {len(data)} bytes")
        
        (
            session_id,
            timestamp_offset_us,
            event_type,
            flags,
            window_size,
            context_hash,
            acute_stress,
            machiavellianism,
            narcissism,
            psychopathy,
            burnout_probability,
            espionage_exposure,
            confidence,
        ) = struct.unpack(cls._FORMAT, data[:64])
        
        return cls(
            session_id=session_id,
            timestamp_offset_us=timestamp_offset_us,
            event_type=event_type,
            flags=flags,
            window_size=window_size,
            context_hash=context_hash,
            acute_stress=acute_stress,
            machiavellianism=machiavellianism,
            narcissism=narcissism,
            psychopathy=psychopathy,
            burnout_probability=burnout_probability,
            espionage_exposure=espionage_exposure,
            confidence=confidence,
        )


class PsychEventType(IntEnum):
    """SHRINK psychological event types"""
    KEYPRESS = 1
    MOUSE_MOVE = 2
    SCORE_UPDATE = 3
    WINDOW_CHANGE = 4
    SESSION_START = 5
    SESSION_END = 6


# ============================================================================
# Routing Helpers
# ============================================================================

def should_route_p2p(priority: Priority) -> bool:
    """
    Determine if message should be routed via P2P.
    
    CRITICAL/EMERGENCY: Direct P2P + hub notification
    HIGH/NORMAL/LOW: Hub-relayed
    """
    return priority >= Priority.CRITICAL


def get_routing_mode(priority: Priority) -> str:
    """Get human-readable routing mode for a priority level"""
    if priority >= Priority.CRITICAL:
        return "p2p+hub"
    elif priority >= Priority.HIGH:
        return "hub-priority"
    else:
        return "hub-normal"


# ============================================================================
# Convenience Constructors
# ============================================================================

def create_memory_sync_message(
    payload: bytes,
    priority: Priority = Priority.NORMAL,
    batch_count: int = 1,
    compressed: bool = False,
) -> MemshadowMessage:
    """Create a MEMORY_SYNC message"""
    flags = MessageFlags.BATCHED
    if compressed:
        flags |= MessageFlags.COMPRESSED
    if priority.should_require_ack():
        flags |= MessageFlags.REQUIRES_ACK
    
    return MemshadowMessage.create(
        msg_type=MessageType.MEMORY_SYNC,
        payload=payload,
        priority=priority,
        flags=flags,
        batch_count=batch_count,
    )


def create_psych_message(
    events: list,
    priority: Priority = Priority.NORMAL,
) -> MemshadowMessage:
    """Create a PSYCH_ASSESSMENT message with batched events"""
    payload = b"".join(e.pack() for e in events)
    
    return MemshadowMessage.create(
        msg_type=MessageType.PSYCH_ASSESSMENT,
        payload=payload,
        priority=priority,
        flags=MessageFlags.BATCHED,
        batch_count=len(events),
    )


def create_improvement_announce(
    improvement_id: str,
    improvement_type: str,
    gain_percent: float,
    priority: Priority = Priority.NORMAL,
) -> MemshadowMessage:
    """Create an IMPROVEMENT_ANNOUNCE message"""
    import json
    payload = json.dumps({
        "improvement_id": improvement_id,
        "type": improvement_type,
        "gain_percent": gain_percent,
    }).encode()
    
    return MemshadowMessage.create(
        msg_type=MessageType.IMPROVEMENT_ANNOUNCE,
        payload=payload,
        priority=priority,
    )


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    # Constants
    "MEMSHADOW_MAGIC",
    "MEMSHADOW_VERSION",
    "HEADER_SIZE",
    # Enums
    "MessageType",
    "Priority",
    "MessageFlags",
    "SyncOperation",
    "MemoryTier",
    "PsychEventType",
    # Data structures
    "MemshadowHeader",
    "MemshadowMessage",
    "PsychEvent",
    # Helpers
    "should_route_p2p",
    "get_routing_mode",
    "create_memory_sync_message",
    "create_psych_message",
    "create_improvement_announce",
]
