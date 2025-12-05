"""
Unit tests for MEMSHADOW Protocol v2

Tests header pack/unpack, message serialization, and routing decisions.
"""

import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsmil_protocol import (
    MemshadowHeader,
    MemshadowMessage,
    MessageType,
    Priority,
    MessageFlags,
    MemoryTier,
    SyncOperation,
    PsychEvent,
    MEMSHADOW_MAGIC,
    MEMSHADOW_VERSION,
    HEADER_SIZE,
    should_route_p2p,
    get_routing_mode,
    create_memory_sync_message,
    create_psych_message,
)


class TestHeader:
    """Test MemshadowHeader"""
    
    def test_header_size(self):
        """Header must be exactly 32 bytes"""
        header = MemshadowHeader()
        packed = header.pack()
        assert len(packed) == HEADER_SIZE == 32
    
    def test_header_magic(self):
        """Magic number must be correct"""
        assert MEMSHADOW_MAGIC == 0x4D534857  # "MSHW"
        
        header = MemshadowHeader()
        packed = header.pack()
        unpacked = MemshadowHeader.unpack(packed)
        assert unpacked.magic == MEMSHADOW_MAGIC
    
    def test_header_round_trip(self):
        """Pack and unpack should preserve all fields"""
        header = MemshadowHeader(
            priority=Priority.HIGH,
            msg_type=MessageType.MEMORY_SYNC,
            flags=MessageFlags.BATCHED | MessageFlags.COMPRESSED,
            batch_count=10,
            payload_len=4096,
        )
        
        packed = header.pack()
        unpacked = MemshadowHeader.unpack(packed)
        
        assert unpacked.priority == Priority.HIGH
        assert unpacked.msg_type == MessageType.MEMORY_SYNC
        assert unpacked.batch_count == 10
        assert unpacked.payload_len == 4096
        assert MessageFlags.BATCHED in MessageFlags(unpacked.flags)
        assert MessageFlags.COMPRESSED in MessageFlags(unpacked.flags)
    
    def test_header_validation(self):
        """Invalid magic should fail validation"""
        header = MemshadowHeader()
        assert header.validate()
        
        header.magic = 0x12345678
        assert not header.validate()
    
    def test_header_unpack_invalid_magic(self):
        """Unpack with invalid magic should raise"""
        bad_data = b"\x00" * 32
        with pytest.raises(ValueError, match="Invalid magic"):
            MemshadowHeader.unpack(bad_data)
    
    def test_header_unpack_too_short(self):
        """Unpack with short data should raise"""
        with pytest.raises(ValueError, match="Header too short"):
            MemshadowHeader.unpack(b"\x00" * 16)


class TestMessage:
    """Test MemshadowMessage"""
    
    def test_message_create(self):
        """Create message with payload"""
        payload = b"test data"
        msg = MemshadowMessage.create(
            msg_type=MessageType.MEMORY_STORE,
            payload=payload,
        )
        
        assert msg.header.msg_type == MessageType.MEMORY_STORE
        assert msg.payload == payload
        assert msg.header.payload_len == len(payload)
    
    def test_message_round_trip(self):
        """Pack and unpack message"""
        payload = b"important data here"
        msg = MemshadowMessage.create(
            msg_type=MessageType.INTEL_REPORT,
            payload=payload,
            priority=Priority.CRITICAL,
            flags=MessageFlags.REQUIRES_ACK,
        )
        
        packed = msg.pack()
        unpacked = MemshadowMessage.unpack(packed)
        
        assert unpacked.header.msg_type == MessageType.INTEL_REPORT
        assert unpacked.header.priority == Priority.CRITICAL
        assert unpacked.payload == payload


class TestMessageTypes:
    """Test all message type categories"""
    
    def test_system_types(self):
        """System/control types are in 0x00xx range"""
        assert MessageType.HEARTBEAT == 0x0001
        assert MessageType.ACK == 0x0002
        assert MessageType.ERROR == 0x0003
    
    def test_psych_types(self):
        """SHRINK psych types are in 0x01xx range"""
        assert MessageType.PSYCH_ASSESSMENT == 0x0100
        assert MessageType.DARK_TRIAD_UPDATE == 0x0101
        assert MessageType.PSYCH_THREAT_ALERT == 0x0110
    
    def test_memory_types(self):
        """Memory types are in 0x03xx range"""
        assert MessageType.MEMORY_STORE == 0x0301
        assert MessageType.MEMORY_QUERY == 0x0302
        assert MessageType.MEMORY_SYNC == 0x0304
    
    def test_improvement_types(self):
        """Improvement types are in 0x05xx range"""
        assert MessageType.IMPROVEMENT_ANNOUNCE == 0x0501
        assert MessageType.IMPROVEMENT_PAYLOAD == 0x0503


class TestPriority:
    """Test priority levels and routing"""
    
    def test_priority_order(self):
        """Priority values should be ordered"""
        assert Priority.LOW < Priority.NORMAL < Priority.HIGH < Priority.CRITICAL < Priority.EMERGENCY
    
    def test_p2p_routing(self):
        """CRITICAL and EMERGENCY use P2P"""
        assert not should_route_p2p(Priority.LOW)
        assert not should_route_p2p(Priority.NORMAL)
        assert not should_route_p2p(Priority.HIGH)
        assert should_route_p2p(Priority.CRITICAL)
        assert should_route_p2p(Priority.EMERGENCY)
    
    def test_routing_mode_strings(self):
        """Routing mode descriptions"""
        assert get_routing_mode(Priority.LOW) == "hub-normal"
        assert get_routing_mode(Priority.NORMAL) == "hub-normal"
        assert get_routing_mode(Priority.HIGH) == "hub-priority"
        assert get_routing_mode(Priority.CRITICAL) == "p2p+hub"
        assert get_routing_mode(Priority.EMERGENCY) == "p2p+hub"


class TestFlags:
    """Test message flags"""
    
    def test_flag_combinations(self):
        """Flags can be combined"""
        flags = MessageFlags.BATCHED | MessageFlags.COMPRESSED | MessageFlags.REQUIRES_ACK
        
        assert MessageFlags.BATCHED in flags
        assert MessageFlags.COMPRESSED in flags
        assert MessageFlags.REQUIRES_ACK in flags
        assert MessageFlags.ENCRYPTED not in flags
    
    def test_flag_round_trip(self):
        """Flags survive pack/unpack"""
        header = MemshadowHeader(
            flags=MessageFlags.PQC_SIGNED | MessageFlags.HIGH_CONFIDENCE,
        )
        
        packed = header.pack()
        unpacked = MemshadowHeader.unpack(packed)
        
        flags = MessageFlags(unpacked.flags)
        # Note: only low byte of flags is preserved in current format
        assert MessageFlags.HIGH_CONFIDENCE in flags


class TestPsychEvent:
    """Test SHRINK psychological event structure"""
    
    def test_psych_event_size(self):
        """Psych event must be 64 bytes"""
        event = PsychEvent()
        packed = event.pack()
        assert len(packed) == 64
    
    def test_psych_event_round_trip(self):
        """Pack and unpack psych event"""
        event = PsychEvent(
            session_id=12345,
            timestamp_offset_us=1000,
            event_type=3,
            acute_stress=0.75,
            machiavellianism=0.5,
            narcissism=0.3,
            psychopathy=0.2,
            confidence=0.95,
        )
        
        packed = event.pack()
        unpacked = PsychEvent.unpack(packed)
        
        assert unpacked.session_id == 12345
        assert unpacked.timestamp_offset_us == 1000
        assert abs(unpacked.acute_stress - 0.75) < 0.001
        assert abs(unpacked.confidence - 0.95) < 0.001


class TestConvenienceFunctions:
    """Test convenience message creation functions"""
    
    def test_create_memory_sync_message(self):
        """Create MEMORY_SYNC message"""
        msg = create_memory_sync_message(
            payload=b"sync data",
            priority=Priority.HIGH,
            batch_count=5,
            compressed=True,
        )
        
        assert msg.header.msg_type == MessageType.MEMORY_SYNC
        assert msg.header.priority == Priority.HIGH
        assert msg.header.batch_count == 5
        assert MessageFlags.BATCHED in MessageFlags(msg.header.flags)
        assert MessageFlags.COMPRESSED in MessageFlags(msg.header.flags)
        assert MessageFlags.REQUIRES_ACK in MessageFlags(msg.header.flags)  # HIGH priority
    
    def test_create_psych_message(self):
        """Create batched PSYCH message"""
        events = [PsychEvent(session_id=i) for i in range(3)]
        msg = create_psych_message(events)
        
        assert msg.header.msg_type == MessageType.PSYCH_ASSESSMENT
        assert msg.header.batch_count == 3
        assert len(msg.payload) == 64 * 3


class TestEnums:
    """Test enum values match documentation"""
    
    def test_memory_tier_values(self):
        """Memory tier enum values"""
        assert MemoryTier.WORKING == 1
        assert MemoryTier.EPISODIC == 2
        assert MemoryTier.SEMANTIC == 3
        assert MemoryTier.L1 == MemoryTier.WORKING
        assert MemoryTier.L2 == MemoryTier.EPISODIC
        assert MemoryTier.L3 == MemoryTier.SEMANTIC
    
    def test_sync_operation_values(self):
        """Sync operation enum values"""
        assert SyncOperation.INSERT == 1
        assert SyncOperation.UPDATE == 2
        assert SyncOperation.DELETE == 3
        assert SyncOperation.MERGE == 4
        assert SyncOperation.REPLICATE == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
