#!/usr/bin/env python3
"""
MEMSHADOW Integration Tests for DSMIL Brain Federation

Tests the complete MEMSHADOW protocol stack:
- Protocol header pack/unpack (32-byte format)
- Message creation and serialization
- Memory sync manager operations
- Hub gateway routing decisions  
- Spoke adapter sync
- Priority-based routing (P2P vs hub)

Run: python3 test_memshadow_integration.py
"""

import asyncio
import sys
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "libs" / "memshadow-protocol" / "python"))
sys.path.insert(0, str(Path(__file__).parent / "ai" / "brain" / "memory"))
sys.path.insert(0, str(Path(__file__).parent / "ai" / "brain" / "federation"))

# Import protocol
from dsmil_protocol import (
    MemshadowHeader, MemshadowMessage, MessageType, Priority, MessageFlags,
    PsychEvent, PsychEventType, HEADER_SIZE, PSYCH_EVENT_SIZE,
    should_route_p2p, detect_protocol_version
)

# Import memory sync
from memory_sync_protocol import (
    MemorySyncItem, MemorySyncBatch, MemorySyncManager,
    MemoryTier, SyncOperation, SyncPriority, SyncFlags
)

# Import gateway
from memshadow_gateway import HubMemshadowGateway, SpokeMemoryAdapter, NodeMemoryCapabilities


class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []
    
    def ok(self, name: str):
        self.passed += 1
        print(f"  ✓ {name}")
    
    def fail(self, name: str, error: str):
        self.failed += 1
        self.errors.append(f"{name}: {error}")
        print(f"  ✗ {name}: {error}")
    
    def summary(self) -> str:
        total = self.passed + self.failed
        return f"Passed: {self.passed}/{total}"


class MockMeshNetwork:
    """Simulated mesh network for testing"""
    
    def __init__(self):
        self.messages: Dict[str, List[Tuple[str, bytes]]] = {}
        self.handlers: Dict[str, callable] = {}
    
    def register_node(self, node_id: str):
        self.messages[node_id] = []
    
    async def send(self, target: str, data: bytes):
        if target not in self.messages:
            self.messages[target] = []
        self.messages[target].append((target, data))
    
    def get_messages(self, node_id: str) -> List[Tuple[str, bytes]]:
        return self.messages.get(node_id, [])
    
    def clear(self):
        for node in self.messages:
            self.messages[node] = []


def test_protocol_header(results: TestResults):
    """Test MEMSHADOW protocol header pack/unpack"""
    print("\n[1] Protocol Header Tests")
    
    # Test basic header creation
    header = MemshadowHeader(
        priority=Priority.NORMAL,
        msg_type=MessageType.MEMORY_SYNC,
        payload_len=1024
    )
    
    if header.msg_type == MessageType.MEMORY_SYNC:
        results.ok("Header creation")
    else:
        results.fail("Header creation", f"Wrong msg_type: {header.msg_type}")
    
    # Test pack size
    packed = header.pack()
    if len(packed) == HEADER_SIZE:
        results.ok(f"Header pack size ({HEADER_SIZE} bytes)")
    else:
        results.fail("Header pack size", f"Got {len(packed)}, expected {HEADER_SIZE}")
    
    # Test unpack
    unpacked = MemshadowHeader.unpack(packed)
    if unpacked.msg_type == MessageType.MEMORY_SYNC:
        results.ok("Header unpack msg_type")
    else:
        results.fail("Header unpack msg_type", f"Got {unpacked.msg_type}")
    
    if unpacked.priority == Priority.NORMAL:
        results.ok("Header unpack priority")
    else:
        results.fail("Header unpack priority", f"Got {unpacked.priority}")
    
    if unpacked.payload_len == 1024:
        results.ok("Header unpack payload_len")
    else:
        results.fail("Header unpack payload_len", f"Got {unpacked.payload_len}")
    
    # Test flags
    header_with_flags = MemshadowHeader(
        msg_type=MessageType.MEMORY_SYNC,
        flags=MessageFlags.BATCHED | MessageFlags.COMPRESSED,
        batch_count=10
    )
    packed_flags = header_with_flags.pack()
    unpacked_flags = MemshadowHeader.unpack(packed_flags)
    
    if MessageFlags.BATCHED in unpacked_flags.flags:
        results.ok("Header flags BATCHED")
    else:
        results.fail("Header flags BATCHED", "Flag not preserved")
    
    if unpacked_flags.batch_count == 10:
        results.ok("Header batch_count")
    else:
        results.fail("Header batch_count", f"Got {unpacked_flags.batch_count}")


def test_message_types(results: TestResults):
    """Test message type ranges and values"""
    print("\n[2] Message Type Tests")
    
    # Test SHRINK psych types are in 0x01xx range
    psych_types = [MessageType.PSYCH_ASSESSMENT, MessageType.DARK_TRIAD_UPDATE, MessageType.RISK_UPDATE]
    for mt in psych_types:
        if 0x0100 <= mt.value <= 0x01FF:
            results.ok(f"{mt.name} in SHRINK range (0x01xx)")
        else:
            results.fail(f"{mt.name} range", f"0x{mt.value:04X} not in 0x01xx")
    
    # Test Memory types are in 0x03xx range
    memory_types = [MessageType.MEMORY_STORE, MessageType.MEMORY_QUERY, MessageType.MEMORY_SYNC]
    for mt in memory_types:
        if 0x0300 <= mt.value <= 0x03FF:
            results.ok(f"{mt.name} in Memory range (0x03xx)")
        else:
            results.fail(f"{mt.name} range", f"0x{mt.value:04X} not in 0x03xx")
    
    # Test Improvement types are in 0x05xx range
    improvement_types = [MessageType.IMPROVEMENT_ANNOUNCE, MessageType.IMPROVEMENT_PAYLOAD]
    for mt in improvement_types:
        if 0x0500 <= mt.value <= 0x05FF:
            results.ok(f"{mt.name} in Improvement range (0x05xx)")
        else:
            results.fail(f"{mt.name} range", f"0x{mt.value:04X} not in 0x05xx")


def test_priority_routing(results: TestResults):
    """Test priority-based routing decisions"""
    print("\n[3] Priority Routing Tests")
    
    # LOW/NORMAL/HIGH should use hub
    for p in [Priority.LOW, Priority.NORMAL, Priority.HIGH]:
        if not should_route_p2p(p):
            results.ok(f"{p.name} routes via hub")
        else:
            results.fail(f"{p.name} routing", "Should not use P2P")
    
    # CRITICAL/EMERGENCY should use P2P
    for p in [Priority.CRITICAL, Priority.EMERGENCY]:
        if should_route_p2p(p):
            results.ok(f"{p.name} routes via P2P")
        else:
            results.fail(f"{p.name} routing", "Should use P2P")


def test_psych_event(results: TestResults):
    """Test SHRINK psychological event pack/unpack"""
    print("\n[4] SHRINK Psych Event Tests")
    
    event = PsychEvent(
        session_id=12345,
        timestamp_offset_us=100000,
        event_type=PsychEventType.SCORE_UPDATE,
        acute_stress=0.75,
        machiavellianism=0.3,
        narcissism=0.4,
        psychopathy=0.2,
        espionage_exposure=0.85,
        confidence=0.9
    )
    
    # Test pack size
    packed = event.pack()
    if len(packed) == PSYCH_EVENT_SIZE:
        results.ok(f"PsychEvent pack size ({PSYCH_EVENT_SIZE} bytes)")
    else:
        results.fail("PsychEvent pack size", f"Got {len(packed)}, expected {PSYCH_EVENT_SIZE}")
    
    # Test unpack
    unpacked = PsychEvent.unpack(packed)
    if unpacked.session_id == 12345:
        results.ok("PsychEvent session_id")
    else:
        results.fail("PsychEvent session_id", f"Got {unpacked.session_id}")
    
    if abs(unpacked.acute_stress - 0.75) < 0.001:
        results.ok("PsychEvent acute_stress")
    else:
        results.fail("PsychEvent acute_stress", f"Got {unpacked.acute_stress}")
    
    if abs(unpacked.espionage_exposure - 0.85) < 0.001:
        results.ok("PsychEvent espionage_exposure")
    else:
        results.fail("PsychEvent espionage_exposure", f"Got {unpacked.espionage_exposure}")
    
    # Test dark triad average
    expected_avg = (0.3 + 0.4 + 0.2) / 3.0
    if abs(unpacked.dark_triad_average - expected_avg) < 0.001:
        results.ok("PsychEvent dark_triad_average")
    else:
        results.fail("PsychEvent dark_triad_average", f"Got {unpacked.dark_triad_average}")


def test_memory_sync_item(results: TestResults):
    """Test MemorySyncItem pack/unpack"""
    print("\n[5] Memory Sync Item Tests")
    
    item = MemorySyncItem(
        item_id="test-item-001234567890123456",
        timestamp_ns=int(time.time() * 1e9),
        tier=MemoryTier.WORKING,
        operation=SyncOperation.INSERT,
        priority=SyncPriority.NORMAL,
        payload=json.dumps({"key": "value", "count": 42}).encode()
    )
    
    # Test pack
    packed = item.pack()
    if len(packed) >= 48:  # 48-byte header + payload
        results.ok("MemorySyncItem pack")
    else:
        results.fail("MemorySyncItem pack", f"Too short: {len(packed)}")
    
    # Test unpack
    unpacked, consumed = MemorySyncItem.unpack(packed)
    if "test-item" in unpacked.item_id:
        results.ok("MemorySyncItem item_id")
    else:
        results.fail("MemorySyncItem item_id", f"Got {unpacked.item_id}")
    
    if unpacked.tier == MemoryTier.WORKING:
        results.ok("MemorySyncItem tier")
    else:
        results.fail("MemorySyncItem tier", f"Got {unpacked.tier}")
    
    if unpacked.operation == SyncOperation.INSERT:
        results.ok("MemorySyncItem operation")
    else:
        results.fail("MemorySyncItem operation", f"Got {unpacked.operation}")


def test_memory_sync_batch(results: TestResults):
    """Test MemorySyncBatch pack/unpack"""
    print("\n[6] Memory Sync Batch Tests")
    
    items = [
        MemorySyncItem(
            item_id=f"item-{i:032d}",
            timestamp_ns=int(time.time() * 1e9) + i,
            tier=MemoryTier.EPISODIC,
            operation=SyncOperation.UPDATE,
            priority=SyncPriority.NORMAL,
            payload=json.dumps({"index": i}).encode()
        )
        for i in range(3)
    ]
    
    batch = MemorySyncBatch(
        batch_id="batch-001",
        source_node="node-a",
        target_node="node-b",
        tier=MemoryTier.EPISODIC,
        items=items
    )
    
    # Test pack
    packed = batch.pack()
    if len(packed) > 0:
        results.ok(f"MemorySyncBatch pack ({len(packed)} bytes)")
    else:
        results.fail("MemorySyncBatch pack", "Empty result")
    
    # Test unpack
    unpacked = MemorySyncBatch.unpack(packed)
    if unpacked.batch_id == "batch-001":
        results.ok("MemorySyncBatch batch_id")
    else:
        results.fail("MemorySyncBatch batch_id", f"Got {unpacked.batch_id}")
    
    if len(unpacked.items) == 3:
        results.ok("MemorySyncBatch item count")
    else:
        results.fail("MemorySyncBatch item count", f"Got {len(unpacked.items)}")
    
    if unpacked.tier == MemoryTier.EPISODIC:
        results.ok("MemorySyncBatch tier")
    else:
        results.fail("MemorySyncBatch tier", f"Got {unpacked.tier}")


def test_sync_manager(results: TestResults):
    """Test MemorySyncManager operations"""
    print("\n[7] Sync Manager Tests")
    
    # Create a mock memory tier
    class MockMemoryTier:
        def __init__(self):
            self._items = {}
            self._timestamps = {}
        
        def store(self, item_id, content):
            self._items[item_id] = content
            self._timestamps[item_id] = int(time.time() * 1e9)
        
        def get_by_id(self, item_id):
            if item_id in self._items:
                return {
                    "content": self._items[item_id],
                    "timestamp_ns": self._timestamps[item_id]
                }
            return None
        
        def get_modified_since(self, timestamp):
            return [
                {"item_id": k, "content": v, "timestamp_ns": self._timestamps[k]}
                for k, v in self._items.items()
                if self._timestamps[k] > timestamp
            ]
    
    manager = MemorySyncManager("test-node")
    working = MockMemoryTier()
    
    # Register tier
    manager.register_memory_tier(MemoryTier.WORKING, working)
    if MemoryTier.WORKING in manager._memory_tiers:
        results.ok("Register memory tier")
    else:
        results.fail("Register memory tier", "Tier not registered")
    
    # Add items
    working.store("item-1", {"data": "test1"})
    working.store("item-2", {"data": "test2"})
    
    # Create delta batch
    batch = manager.create_delta_batch(MemoryTier.WORKING, "target-node", 0)
    if batch and len(batch.items) >= 2:
        results.ok(f"Create delta batch ({len(batch.items)} items)")
    else:
        results.fail("Create delta batch", f"Got {len(batch.items) if batch else 0} items")
    
    # Get sync vector
    vector = manager.get_sync_vector(MemoryTier.WORKING)
    results.ok("Get sync vector")
    
    # Get stats
    stats = manager.get_stats()
    if "registered_tiers" in stats:
        results.ok("Get sync stats")
    else:
        results.fail("Get sync stats", "Missing registered_tiers")


def test_hub_gateway(results: TestResults):
    """Test HubMemshadowGateway operations"""
    print("\n[8] Hub Gateway Tests")
    
    mesh = MockMeshNetwork()
    gateway = HubMemshadowGateway("test-hub", mesh_send=mesh.send)
    
    # Register nodes
    gateway.register_node("node-1", "localhost:8001", {
        MemoryTier.WORKING: 1024*1024,
        MemoryTier.EPISODIC: 10*1024*1024,
    })
    gateway.register_node("node-2", "localhost:8002", {
        MemoryTier.WORKING: 2048*1024,
        MemoryTier.SEMANTIC: 100*1024*1024,
    })
    
    stats = gateway.get_all_stats()
    if stats["registered_nodes"] == 2:
        results.ok("Register nodes")
    else:
        results.fail("Register nodes", f"Got {stats['registered_nodes']}")
    
    # Schedule sync
    gateway.schedule_sync("node-1", MemoryTier.WORKING, Priority.HIGH)
    pending = gateway.get_pending_syncs()
    if len(pending) == 1 and pending[0].priority == Priority.HIGH:
        results.ok("Schedule sync with priority")
    else:
        results.fail("Schedule sync", f"Got {len(pending)} pending")
    
    # Get node stats
    node_stats = gateway.get_node_stats("node-1")
    if node_stats and MemoryTier.WORKING.name in node_stats["tiers"]:
        results.ok("Get node stats")
    else:
        results.fail("Get node stats", "Missing tier info")
    
    # Test unregister
    gateway.unregister_node("node-2")
    stats = gateway.get_all_stats()
    if stats["registered_nodes"] == 1:
        results.ok("Unregister node")
    else:
        results.fail("Unregister node", f"Still have {stats['registered_nodes']}")


def test_spoke_adapter(results: TestResults):
    """Test SpokeMemoryAdapter operations"""
    print("\n[9] Spoke Adapter Tests")
    
    mesh = MockMeshNetwork()
    mesh.register_node("hub")
    mesh.register_node("node-1")
    mesh.register_node("node-2")
    
    adapter = SpokeMemoryAdapter("node-1", "hub", mesh_send=mesh.send)
    
    # Add peer
    adapter.add_peer("node-2")
    stats = adapter.get_stats()
    if stats["peer_count"] == 1:
        results.ok("Add peer")
    else:
        results.fail("Add peer", f"Got {stats['peer_count']}")
    
    # Test metrics
    if "batches_sent" in stats["metrics"]:
        results.ok("Adapter metrics")
    else:
        results.fail("Adapter metrics", "Missing batches_sent")
    
    # Remove peer
    adapter.remove_peer("node-2")
    stats = adapter.get_stats()
    if stats["peer_count"] == 0:
        results.ok("Remove peer")
    else:
        results.fail("Remove peer", f"Got {stats['peer_count']}")


async def test_end_to_end_sync(results: TestResults):
    """Test end-to-end sync between hub and spokes"""
    print("\n[10] End-to-End Sync Tests")
    
    # Create mesh
    mesh = MockMeshNetwork()
    mesh.register_node("hub")
    mesh.register_node("spoke-1")
    mesh.register_node("spoke-2")
    
    # Create hub gateway
    hub = HubMemshadowGateway("hub", mesh_send=mesh.send)
    hub.register_node("spoke-1", "localhost:8001", {MemoryTier.WORKING: 1024*1024})
    hub.register_node("spoke-2", "localhost:8002", {MemoryTier.WORKING: 1024*1024})
    
    # Create spoke adapters
    spoke1 = SpokeMemoryAdapter("spoke-1", "hub", mesh_send=mesh.send)
    spoke1.add_peer("spoke-2")
    spoke2 = SpokeMemoryAdapter("spoke-2", "hub", mesh_send=mesh.send)
    spoke2.add_peer("spoke-1")
    
    # Create a batch on spoke-1
    batch = MemorySyncBatch(
        batch_id="sync-batch-001",
        source_node="spoke-1",
        target_node="*",
        tier=MemoryTier.WORKING,
        items=[
            MemorySyncItem(
                item_id="data-item-001234567890123456",
                timestamp_ns=int(time.time() * 1e9),
                tier=MemoryTier.WORKING,
                operation=SyncOperation.INSERT,
                priority=SyncPriority.NORMAL,
                payload=b'{"test": "data"}'
            )
        ]
    )
    
    # Route with NORMAL priority (should go via hub)
    result = await hub.route_batch(batch, Priority.NORMAL)
    if hub._metrics["hub_routes"] >= 1:
        results.ok("Hub routing for NORMAL priority")
    else:
        results.fail("Hub routing", "No hub routes recorded")
    
    # Route with CRITICAL priority (should use P2P)
    result = await hub.route_batch(batch, Priority.CRITICAL)
    if hub._metrics["p2p_routes"] >= 1:
        results.ok("P2P routing for CRITICAL priority")
    else:
        results.fail("P2P routing", "No P2P routes recorded")
    
    # Handle incoming batch
    packed = batch.pack()
    handle_result = await hub.handle_incoming_batch(packed, "spoke-1")
    if handle_result.get("status") in ["received", "routed"]:
        results.ok("Handle incoming batch")
    else:
        results.fail("Handle incoming batch", f"Status: {handle_result.get('status')}")


def test_protocol_detection(results: TestResults):
    """Test protocol version detection"""
    print("\n[11] Protocol Detection Tests")
    
    # Create v2 header
    header = MemshadowHeader(
        msg_type=MessageType.HEARTBEAT,
        priority=Priority.NORMAL
    )
    packed = header.pack()
    
    version = detect_protocol_version(packed)
    if version == 2:
        results.ok("Detect v2 protocol")
    else:
        results.fail("Detect v2 protocol", f"Got version {version}")
    
    # Test invalid data
    version = detect_protocol_version(b"\x00\x00\x00\x00")
    if version == 0:
        results.ok("Detect invalid protocol")
    else:
        results.fail("Detect invalid protocol", f"Got version {version}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("MEMSHADOW Integration Tests")
    print("=" * 60)
    
    results = TestResults()
    
    # Run synchronous tests
    test_protocol_header(results)
    test_message_types(results)
    test_priority_routing(results)
    test_psych_event(results)
    test_memory_sync_item(results)
    test_memory_sync_batch(results)
    test_sync_manager(results)
    test_hub_gateway(results)
    test_spoke_adapter(results)
    test_protocol_detection(results)
    
    # Run async tests
    asyncio.run(test_end_to_end_sync(results))
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {results.summary()}")
    
    if results.failed > 0:
        print("\nFailed tests:")
        for error in results.errors:
            print(f"  - {error}")
        print()
        return 1
    
    print("\n✓ All tests passed!")
    return 0


if __name__ == "__main__":
    exit(main())
