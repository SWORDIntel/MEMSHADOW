"""
MEMSHADOW Protocol v2 Implementation

Binary wire format for intra-node communications within the DSMIL ecosystem.
Provides 32-byte header format for all messages exchanged between:
- Mesh network nodes (hub ↔ spokes, spoke ↔ spoke)
- Memory tier synchronization
- Self-improvement propagation

Based on: HUB_DOCS/MEMSHADOW PROTOCOL.md
"""

import struct
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, IntFlag
from typing import Any, Dict, Optional, Tuple
import hashlib
import json

import structlog

logger = structlog.get_logger()

# Protocol constants
MEMSHADOW_MAGIC = 0x4D534857  # "MSHW" in ASCII
MEMSHADOW_VERSION = 2
HEADER_SIZE = 32


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
    """Message priority levels for routing decisions"""
    LOW = 0        # Background operations
    NORMAL = 1     # Standard operations
    HIGH = 2       # Important updates
    CRITICAL = 3   # Urgent alerts
    EMERGENCY = 4  # Immediate action required
    
    def should_use_p2p(self) -> bool:
        """Check if this priority should use direct P2P routing"""
        return self >= Priority.CRITICAL
    
    def should_require_ack(self) -> bool:
        """Check if this priority should require acknowledgment"""
        return self >= Priority.HIGH


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


@dataclass
class MemshadowHeader:
    """
    MEMSHADOW Protocol v2 Header (32 bytes)
    
    Structure:
        magic         : 8 bytes - Protocol magic (0x4D534857)
        version       : 2 bytes - Protocol version (2)
        priority      : 2 bytes - Message priority (0-4)
        msg_type      : 2 bytes - Message type
        flags         : 2 bytes - Message flags
        batch_count   : 2 bytes - Number of items in batch
        payload_len   : 8 bytes - Payload length in bytes
        timestamp_ns  : 8 bytes - Nanosecond timestamp
        reserved      : 8 bytes - Reserved for future expansion (not in 32-byte header)
    
    Note: Header is 32 bytes total, network byte order (big-endian)
    """
    magic: int = MEMSHADOW_MAGIC
    version: int = MEMSHADOW_VERSION
    priority: Priority = Priority.NORMAL
    msg_type: MessageType = MessageType.HEARTBEAT
    flags: MessageFlags = MessageFlags.NONE
    batch_count: int = 0
    payload_len: int = 0
    timestamp_ns: int = field(default_factory=lambda: int(time.time() * 1e9))
    
    # Header format: big-endian, all fields
    # Q=uint64, H=uint16
    # Total: 8 + 2 + 2 + 2 + 2 + 2 + 8 + 8 = 34 bytes, but we use reserved to pad
    # Adjusted: Q + 6*H + Q = 8 + 12 + 8 = 28 bytes, need 4 more for 32
    # Using: QHHHHHxHQ where x is padding
    # Actually: magic(8) + version(2) + priority(2) + msg_type(2) + flags(2) + 
    #           batch_count(2) + payload_len(8) + timestamp_ns(8) = 34
    # We'll use a compact 32-byte format
    _HEADER_FORMAT = ">QHHHHHHQ"  # 8 + 2*6 + 8 = 28 bytes, pad with reserved 4 bytes
    _HEADER_FORMAT_FULL = ">QHHHHHQI"  # Add 4-byte reserved at end = 32 bytes
    
    def pack(self) -> bytes:
        """Pack header to binary (32 bytes, network byte order)"""
        # Using format: Q(magic) H(version) H(priority) H(msg_type) H(flags) 
        #               H(batch_count_hi) H(batch_count_lo/reserved) Q(payload_len) 
        # Simplified: pack to 32 bytes with reserved
        header_data = struct.pack(
            ">QHHHHHHQ",
            self.magic,
            self.version,
            int(self.priority),
            int(self.msg_type),
            int(self.flags),
            self.batch_count,
            0,  # reserved
            self.payload_len,
        )
        # Add timestamp (8 bytes) - total will be 36, truncate or adjust
        # Actually let's recompute for exactly 32 bytes
        # Q(8) + H(2)*5 + I(4) + Q(8) = 8 + 10 + 4 + 8 = 30 bytes - need 2 more
        # Q(8) + H(2)*6 + Q(8) = 8 + 12 + 8 = 28 bytes - need 4 more
        # Let's use: Q(8) + H(2)*4 + Q(8) + Q(8) = 8 + 8 + 8 + 8 = 32 bytes
        return struct.pack(
            ">QHHHHQQ",
            self.magic,
            self.version,
            int(self.priority),
            int(self.msg_type),
            int(self.flags) | (self.batch_count << 8),  # Combine flags and batch_count
            self.payload_len,
            self.timestamp_ns,
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> "MemshadowHeader":
        """Unpack header from binary (32 bytes)"""
        if len(data) < 32:
            raise ValueError(f"Header too short: {len(data)} bytes, expected 32")
        
        (
            magic,
            version,
            priority,
            msg_type,
            flags_batch,
            payload_len,
            timestamp_ns,
        ) = struct.unpack(">QHHHHQQ", data[:32])
        
        if magic != MEMSHADOW_MAGIC:
            raise ValueError(f"Invalid magic number: 0x{magic:016X}, expected 0x{MEMSHADOW_MAGIC:016X}")
        
        flags = flags_batch & 0xFF
        batch_count = (flags_batch >> 8) & 0xFF
        
        return cls(
            magic=magic,
            version=version,
            priority=Priority(priority),
            msg_type=MessageType(msg_type),
            flags=MessageFlags(flags),
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
        if self.magic != MEMSHADOW_MAGIC:
            return False
        if self.version != MEMSHADOW_VERSION:
            return False
        return True


@dataclass
class MemshadowMessage:
    """
    Complete MEMSHADOW message with header and payload.
    """
    header: MemshadowHeader
    payload: bytes = b""
    
    def pack(self) -> bytes:
        """Pack complete message to binary"""
        # Update header with payload length
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


def compute_checksum(data: bytes) -> str:
    """Compute SHA-256 checksum for data"""
    return hashlib.sha256(data).hexdigest()


def should_route_p2p(priority: Priority) -> bool:
    """
    Determine if a message should be routed via P2P.
    
    Based on MEMSHADOW Protocol routing rules:
    - CRITICAL/EMERGENCY: Direct P2P + hub notification
    - HIGH: Hub-relayed with priority queue
    - NORMAL/LOW: Standard hub routing
    """
    return priority >= Priority.CRITICAL


def should_hub_relay(priority: Priority) -> bool:
    """
    Determine if a message should be relayed through the hub.
    
    All messages except EMERGENCY-only should go through hub.
    """
    return True  # Hub always gets a copy for coordination


class MessageRouter:
    """
    Routes MEMSHADOW messages based on priority and type.
    
    Implements the routing rules from MEMSHADOW Protocol:
    - URGENT/CRITICAL: P2P + hub notification
    - NORMAL/LOW: Hub-relayed; hub decides propagation
    """
    
    def __init__(self, node_id: str, is_hub: bool = False):
        self.node_id = node_id
        self.is_hub = is_hub
        self._handlers: Dict[MessageType, callable] = {}
        
    def register_handler(self, msg_type: MessageType, handler: callable):
        """Register a handler for a message type"""
        self._handlers[msg_type] = handler
        logger.debug("Handler registered", msg_type=msg_type.name, node=self.node_id)
    
    def route(self, message: MemshadowMessage, peer_id: str) -> Tuple[bool, bool]:
        """
        Route a message and determine delivery targets.
        
        Returns:
            Tuple of (should_deliver_locally, should_forward_to_hub)
        """
        priority = message.header.priority
        msg_type = message.header.msg_type
        
        should_deliver = True
        should_forward_hub = not self.is_hub  # Spokes forward to hub
        
        # P2P routing for critical messages
        if should_route_p2p(priority):
            # Critical: deliver locally and notify hub
            should_deliver = True
            should_forward_hub = not self.is_hub
        
        # Handle message if we have a handler
        if should_deliver and msg_type in self._handlers:
            try:
                self._handlers[msg_type](message, peer_id)
            except Exception as e:
                logger.error("Handler error", msg_type=msg_type.name, error=str(e))
        
        return should_deliver, should_forward_hub
    
    def dispatch(self, data: bytes, peer_id: str) -> bool:
        """
        Dispatch incoming binary message to appropriate handler.
        
        Returns:
            True if message was handled successfully
        """
        try:
            message = MemshadowMessage.unpack(data)
            
            if not message.header.validate():
                logger.warning("Invalid message header", peer=peer_id)
                return False
            
            logger.debug(
                "MEMSHADOW message received",
                msg_type=message.header.msg_type.name,
                priority=message.header.priority.name,
                payload_len=message.header.payload_len,
                peer=peer_id,
            )
            
            delivered, _ = self.route(message, peer_id)
            return delivered
            
        except Exception as e:
            logger.error("Message dispatch error", error=str(e), peer=peer_id)
            return False


# Convenience functions for creating common messages

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


def create_ack_message(original_msg: MemshadowMessage) -> MemshadowMessage:
    """Create an ACK response for a message"""
    return MemshadowMessage.create(
        msg_type=MessageType.ACK,
        payload=struct.pack(">Q", original_msg.header.timestamp_ns),
        priority=Priority.HIGH,
    )


def create_error_message(error_code: int, error_msg: str) -> MemshadowMessage:
    """Create an ERROR message"""
    payload = struct.pack(">I", error_code) + error_msg.encode("utf-8")
    return MemshadowMessage.create(
        msg_type=MessageType.ERROR,
        payload=payload,
        priority=Priority.HIGH,
    )
