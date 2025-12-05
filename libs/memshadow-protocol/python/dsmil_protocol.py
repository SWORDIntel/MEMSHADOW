"""
DSMIL MEMSHADOW Protocol v2 - Canonical Implementation

Unified binary wire format for intra-node communications within the DSMIL ecosystem.

This is the canonical protocol library referenced by:
- ai/brain/federation/hub_orchestrator.py
- ai/brain/federation/spoke_client.py
- ai/brain/memory/memory_sync_protocol.py
- ai/brain/plugins/ingest/memshadow_ingest.py
- external/intel/shrink/shrink/kernel_receiver.py

Protocol Version: 2.0
Header Size: 32 bytes
Magic Number: 0x4D534857 ("MSHW" in ASCII)
"""

import struct
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, IntFlag
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# Protocol Constants
# ============================================================================

MEMSHADOW_MAGIC = 0x4D534857  # "MSHW" in ASCII
MEMSHADOW_VERSION = 2
HEADER_SIZE = 32
PSYCH_EVENT_SIZE = 64


# ============================================================================
# Enumerations
# ============================================================================

class MessageType(IntEnum):
    """MEMSHADOW Protocol v2 Message Types"""
    
    # System/Control (0x00xx)
    HEARTBEAT = 0x0001
    ACK = 0x0002
    NACK = 0x0003
    ERROR = 0x0003  # Alias for NACK
    HANDSHAKE = 0x0004
    DISCONNECT = 0x0005
    
    # SHRINK Psychological Intelligence (0x01xx)
    PSYCH_ASSESSMENT = 0x0100
    DARK_TRIAD_UPDATE = 0x0101
    RISK_UPDATE = 0x0102
    NEURO_UPDATE = 0x0103
    TMI_UPDATE = 0x0104
    COGARCH_UPDATE = 0x0105
    COGNITIVE_UPDATE = 0x0105  # Alias
    FULL_PSYCH = 0x0106
    PSYCH_THREAT_ALERT = 0x0110
    PSYCH_ANOMALY = 0x0111
    PSYCH_RISK_THRESHOLD = 0x0112
    
    # Threat Intelligence (0x02xx)
    THREAT_REPORT = 0x0201
    INTEL_REPORT = 0x0202
    KNOWLEDGE_UPDATE = 0x0203
    BRAIN_INTEL_REPORT = 0x0204
    INTEL_PROPAGATE = 0x0205
    
    # Memory Operations (0x03xx)
    MEMORY_STORE = 0x0301
    MEMORY_QUERY = 0x0302
    MEMORY_RESPONSE = 0x0303
    MEMORY_SYNC = 0x0304
    VECTOR_SYNC = 0x0305
    
    # Federation/Mesh (0x04xx)
    NODE_REGISTER = 0x0401
    NODE_DEREGISTER = 0x0402
    QUERY_DISTRIBUTE = 0x0403
    BRAIN_QUERY = 0x0403  # Alias
    QUERY_RESPONSE = 0x0404
    
    # Self-Improvement (0x05xx)
    IMPROVEMENT_ANNOUNCE = 0x0501
    IMPROVEMENT_REQUEST = 0x0502
    IMPROVEMENT_PAYLOAD = 0x0503
    IMPROVEMENT_ACK = 0x0504
    IMPROVEMENT_REJECT = 0x0505
    IMPROVEMENT_METRICS = 0x0506


# Alias for backward compatibility
MemshadowMessageType = MessageType


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
    
    # Aliases for sync operations
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


# Alias for backward compatibility
MemshadowMessagePriority = Priority


class MessageFlags(IntFlag):
    """Message flags for payload handling"""
    NONE = 0x0000
    ENCRYPTED = 0x0001
    COMPRESSED = 0x0002
    BATCHED = 0x0004
    REQUIRES_ACK = 0x0008
    FRAGMENTED = 0x0010
    LAST_FRAGMENT = 0x0020
    FROM_KERNEL = 0x0040
    HIGH_CONFIDENCE = 0x0080
    PQC_SIGNED = 0x0100


class SyncOperation(IntEnum):
    """Memory sync operations"""
    INSERT = 1
    UPDATE = 2
    DELETE = 3
    MERGE = 4
    REPLICATE = 5


class MemoryTier(IntEnum):
    """Memory tier levels"""
    WORKING = 1
    EPISODIC = 2
    SEMANTIC = 3
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
        flags_batch   : 2 bytes - Flags (low byte) + batch_count (high byte)
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
    sequence_num: int = 0  # Optional sequence number (embedded in flags_batch)
    
    _FORMAT = ">QHHHHQQ"
    
    # Aliases for backward compatibility
    @property
    def message_type(self) -> MessageType:
        return self.msg_type
    
    @message_type.setter
    def message_type(self, value: MessageType):
        self.msg_type = value
    
    def pack(self) -> bytes:
        """Pack header to binary (32 bytes, network byte order)"""
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
    
    _FORMAT = ">QIBBHQfffffff12x"
    
    @property
    def dark_triad_average(self) -> float:
        """Calculate average dark triad score"""
        return (self.machiavellianism + self.narcissism + self.psychopathy) / 3.0
    
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
        if len(data) < PSYCH_EVENT_SIZE:
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
        ) = struct.unpack(cls._FORMAT, data[:PSYCH_EVENT_SIZE])
        
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
# Message Container
# ============================================================================

@dataclass
class MemshadowMessage:
    """Complete MEMSHADOW message with header and payload"""
    header: MemshadowHeader
    payload: bytes = b""
    events: List[PsychEvent] = field(default_factory=list)
    
    @property
    def raw_payload(self) -> bytes:
        """Alias for payload"""
        return self.payload
    
    def pack(self) -> bytes:
        """Pack complete message to binary"""
        self.header.payload_len = len(self.payload)
        return self.header.pack() + self.payload
    
    @classmethod
    def unpack(cls, data: bytes) -> "MemshadowMessage":
        """Unpack complete message from binary"""
        header = MemshadowHeader.unpack(data[:HEADER_SIZE])
        payload = data[HEADER_SIZE:HEADER_SIZE + header.payload_len]
        
        # Parse psych events if this is a psych message type
        events = []
        if header.msg_type in (
            MessageType.PSYCH_ASSESSMENT,
            MessageType.DARK_TRIAD_UPDATE,
            MessageType.RISK_UPDATE,
            MessageType.NEURO_UPDATE,
            MessageType.TMI_UPDATE,
            MessageType.COGNITIVE_UPDATE,
            MessageType.FULL_PSYCH,
        ):
            offset = 0
            while offset + PSYCH_EVENT_SIZE <= len(payload):
                try:
                    event = PsychEvent.unpack(payload[offset:])
                    events.append(event)
                    offset += PSYCH_EVENT_SIZE
                except:
                    break
        
        return cls(header=header, payload=payload, events=events)
    
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
# Protocol Detection and Helpers
# ============================================================================

def detect_protocol_version(data: bytes) -> int:
    """
    Detect MEMSHADOW protocol version from raw data.
    
    Returns:
        Protocol version (1 or 2), or 0 if not MEMSHADOW protocol
    """
    if len(data) < 8:
        return 0
    
    # Check for v2 magic (8 bytes)
    magic = struct.unpack(">Q", data[:8])[0]
    if magic == MEMSHADOW_MAGIC:
        if len(data) >= 10:
            version = struct.unpack(">H", data[8:10])[0]
            return version
        return 2
    
    # Check for v1 magic (4 bytes)
    magic4 = struct.unpack(">I", data[:4])[0]
    if magic4 == 0x4D534857:  # MSHW in 4 bytes
        return 1
    
    return 0


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
    events: List[PsychEvent],
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
    "PSYCH_EVENT_SIZE",
    # Enums
    "MessageType",
    "MemshadowMessageType",  # Alias
    "Priority",
    "MemshadowMessagePriority",  # Alias
    "MessageFlags",
    "SyncOperation",
    "MemoryTier",
    "PsychEventType",
    # Data structures
    "MemshadowHeader",
    "MemshadowMessage",
    "PsychEvent",
    # Helpers
    "detect_protocol_version",
    "should_route_p2p",
    "get_routing_mode",
    "create_memory_sync_message",
    "create_psych_message",
    "create_improvement_announce",
]
