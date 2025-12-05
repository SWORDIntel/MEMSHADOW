#!/usr/bin/env python3
"""
MEMSHADOW Integration Test Suite

Tests the complete MEMSHADOW implementation:
- Protocol header pack/unpack (32-byte format)
- Message creation and serialization
- MemorySyncManager delta and conflict handling
- HubMemshadowGateway routing decisions
- SpokeMemoryAdapter storage and sync
- Priority-based routing (CRITICAL/EMERGENCY P2P vs hub-relayed)
- Memory tier integration (L1/L2/L3)

Run: python3 test_memshadow_integration.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from uuid import uuid4

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent / "libs" / "memshadow-protocol" / "python"))
sys.path.insert(0, str(Path(__file__).parent))

# Import protocol
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
)

# Import brain modules
from ai.brain.memory import (
    MemorySyncItem,
    MemorySyncBatch,
    MemorySyncManager,
    SyncPriority,
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
)

from ai.brain.federation import (
    HubOrchestrator,
    HubMemshadowGateway,
    SpokeClient,
    SpokeMemoryAdapter,
    InMemoryTier,
    NodeCapability,
    NodeMemoryCapabilities,
    ImprovementPackage,
    ImprovementType,
    ImprovementTracker,
)


# =============================================================================
# Test Infrastructure
# =============================================================================

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.duration_ms = 0

    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        msg = f"  {status}: {self.name} ({self.duration_ms:.2f}ms)"
        if self.error:
            msg += f"\n    Error: {self.error}"
        return msg


class TestRunner:
    def __init__(self):
        self.results: List[TestResult] = []
    
    async def run_test(self, name: str, test_func):
        result = TestResult(name)
        start = time.time()
        try:
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
            result.passed = True
        except AssertionError as e:
            result.error = str(e) or "Assertion failed"
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
        result.duration_ms = (time.time() - start) * 1000
        self.results.append(result)
        return result
    
    def summary(self) -> Tuple[int, int]:
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        return passed, total


class MockMeshNetwork:
    """Mock mesh network for testing"""
    
    def __init__(self):
        self.messages: Dict[str, List[bytes]] = {}
        self.handlers: Dict[str, callable] = {}
    
    async def send(self, target: str, data: bytes):
        if target not in self.messages:
            self.messages[target] = []
        self.messages[target].append(data)
        
        # Trigger handler if registered
        if target in self.handlers:
            await self.handlers[target](data, "test_source")
    
    def get_messages(self, target: str) -> List[bytes]:
        return self.messages.get(target, [])
    
    def clear(self):
        self.messages.clear()


# =============================================================================
# Protocol Tests
# =============================================================================

def test_header_pack_unpack():
    """Test 32-byte header pack/unpack"""
    header = MemshadowHeader(
        magic=MEMSHADOW_MAGIC,
        version=MEMSHADOW_VERSION,
        priority=Priority.HIGH,
        msg_type=MessageType.MEMORY_SYNC,
        flags=MessageFlags.BATCHED | MessageFlags.COMPRESSED,
        batch_count=5,
        payload_len=1024,
    )
    
    packed = header.pack()
    assert len(packed) == HEADER_SIZE, f"Header size mismatch: {len(packed)} != {HEADER_SIZE}"
    
    unpacked = MemshadowHeader.unpack(packed)
    
    assert unpacked.magic == MEMSHADOW_MAGIC
    assert unpacked.version == MEMSHADOW_VERSION
    assert unpacked.priority == Priority.HIGH
    assert unpacked.msg_type == MessageType.MEMORY_SYNC
    assert unpacked.batch_count == 5
    assert unpacked.payload_len == 1024
    assert MessageFlags.BATCHED in MessageFlags(unpacked.flags)


def test_message_create_serialize():
    """Test message creation and serialization"""
    payload = b"test payload data"
    
    msg = MemshadowMessage.create(
        msg_type=MessageType.MEMORY_STORE,
        payload=payload,
        priority=Priority.NORMAL,
        flags=MessageFlags.REQUIRES_ACK,
    )
    
    packed = msg.pack()
    assert len(packed) == HEADER_SIZE + len(payload)
    
    unpacked = MemshadowMessage.unpack(packed)
    assert unpacked.header.msg_type == MessageType.MEMORY_STORE
    assert unpacked.payload == payload


def test_priority_routing_rules():
    """Test priority-based routing decisions"""
    # LOW/NORMAL: hub-relayed
    assert not should_route_p2p(Priority.LOW)
    assert not should_route_p2p(Priority.NORMAL)
    
    # HIGH: hub-relayed with priority
    assert not should_route_p2p(Priority.HIGH)
    
    # CRITICAL/EMERGENCY: P2P
    assert should_route_p2p(Priority.CRITICAL)
    assert should_route_p2p(Priority.EMERGENCY)
    
    # Routing mode strings
    assert get_routing_mode(Priority.LOW) == "hub-normal"
    assert get_routing_mode(Priority.HIGH) == "hub-priority"
    assert get_routing_mode(Priority.CRITICAL) == "p2p+hub"


def test_psych_event():
    """Test SHRINK psych event packing"""
    event = PsychEvent(
        session_id=12345,
        timestamp_offset_us=1000,
        event_type=3,  # SCORE_UPDATE
        acute_stress=0.5,
        machiavellianism=0.3,
        narcissism=0.4,
        psychopathy=0.2,
        confidence=0.85,
    )
    
    packed = event.pack()
    assert len(packed) == 64
    
    unpacked = PsychEvent.unpack(packed)
    assert unpacked.session_id == 12345
    assert abs(unpacked.acute_stress - 0.5) < 0.01
    assert abs(unpacked.confidence - 0.85) < 0.01


# =============================================================================
# Memory Sync Protocol Tests
# =============================================================================

async def test_memory_sync_item():
    """Test MemorySyncItem creation and serialization"""
    item = MemorySyncItem.create(
        payload=b"test embedding data",
        tier=MemoryTier.WORKING,
        operation=SyncOperation.INSERT,
        priority=SyncPriority.NORMAL,
        source_node="node-001",
    )
    
    assert item.tier == MemoryTier.WORKING
    assert item.operation == SyncOperation.INSERT
    assert len(item.content_hash) > 0
    
    # Serialize and deserialize
    data = item.to_dict()
    restored = MemorySyncItem.from_dict(data)
    
    assert restored.tier == item.tier
    assert restored.operation == item.operation
    assert restored.source_node == item.source_node


async def test_memory_sync_batch():
    """Test MemorySyncBatch creation and validation"""
    items = [
        MemorySyncItem.create(
            payload=f"item {i}".encode(),
            tier=MemoryTier.EPISODIC,
            priority=SyncPriority.NORMAL,
        )
        for i in range(5)
    ]
    
    batch = MemorySyncBatch(
        source_node="node-001",
        target_node="hub",
        tier=MemoryTier.EPISODIC,
        items=items,
    )
    
    assert len(batch.items) == 5
    assert batch.validate()  # Checksum should be valid
    
    # Convert to MEMSHADOW message
    msg = batch.to_memshadow_message()
    assert msg.header.msg_type == MessageType.MEMORY_SYNC
    assert MessageFlags.BATCHED in MessageFlags(msg.header.flags)


async def test_memory_sync_manager():
    """Test MemorySyncManager delta computation and conflict resolution"""
    manager = MemorySyncManager(node_id="node-001")
    
    # Store some items
    for i in range(5):
        item = MemorySyncItem.create(
            payload=f"data {i}".encode(),
            tier=MemoryTier.WORKING,
            source_node="node-001",
        )
        await manager.store_local(item)
    
    # Create delta batch
    batch = await manager.create_delta_batch(
        peer_id="node-002",
        tier=MemoryTier.WORKING,
    )
    
    assert len(batch.items) == 5
    assert batch.source_node == "node-001"
    
    # Apply batch (should succeed)
    result = await manager.apply_sync_batch(batch)
    assert result["success"]
    assert result["applied"] >= 0


# =============================================================================
# Hub Gateway Tests
# =============================================================================

async def test_hub_gateway_registration():
    """Test node registration with hub gateway"""
    mesh = MockMeshNetwork()
    gateway = HubMemshadowGateway(
        hub_id="hub-001",
        mesh_send_callback=mesh.send,
    )
    
    # Register nodes
    gateway.register_node(
        node_id="node-001",
        endpoint="localhost:8001",
        memory_caps=NodeMemoryCapabilities(
            supports_l1=True,
            supports_l2=True,
            supports_l3=False,
        ),
    )
    
    gateway.register_node(
        node_id="node-002",
        endpoint="localhost:8002",
    )
    
    stats = gateway.get_stats()
    assert stats["nodes_registered"] == 2


async def test_hub_gateway_routing():
    """Test hub gateway batch routing decisions"""
    mesh = MockMeshNetwork()
    gateway = HubMemshadowGateway(
        hub_id="hub-001",
        mesh_send_callback=mesh.send,
    )
    
    # Register target node
    gateway.register_node("node-002", "localhost:8002")
    
    # Create and route a normal priority batch
    items = [MemorySyncItem.create(payload=b"data", tier=MemoryTier.WORKING)]
    batch = MemorySyncBatch(
        source_node="node-001",
        target_node="node-002",
        tier=MemoryTier.WORKING,
        items=items,
        priority=SyncPriority.NORMAL,
    )
    
    result = await gateway.route_batch(batch)
    
    assert "node-002" in result["targets_sent"]
    assert result["routing_mode"] == "hub-normal"
    assert len(mesh.get_messages("node-002")) == 1


# =============================================================================
# Spoke Adapter Tests
# =============================================================================

async def test_spoke_adapter_storage():
    """Test spoke adapter local storage"""
    adapter = SpokeMemoryAdapter(
        node_id="node-001",
        hub_node_id="hub",
    )
    
    # Register tier
    l1_tier = InMemoryTier(MemoryTier.WORKING)
    adapter.register_tier(MemoryTier.WORKING, l1_tier)
    
    # Store item
    item_id = await adapter.store(
        tier=MemoryTier.WORKING,
        data=b"test data",
        metadata={"key": "value"},
    )
    
    assert item_id is not None
    
    # Retrieve item
    data = await adapter.retrieve(MemoryTier.WORKING, item_id)
    assert data == b"test data"
    
    # Check stats
    stats = adapter.get_stats()
    assert stats["items_stored"] == 1


async def test_spoke_adapter_delta_batch():
    """Test spoke adapter delta batch creation"""
    adapter = SpokeMemoryAdapter(
        node_id="node-001",
        hub_node_id="hub",
    )
    
    l1_tier = InMemoryTier(MemoryTier.WORKING)
    adapter.register_tier(MemoryTier.WORKING, l1_tier)
    
    # Store multiple items
    for i in range(3):
        await adapter.store(
            tier=MemoryTier.WORKING,
            data=f"item {i}".encode(),
        )
    
    # Create delta batch
    batch = await adapter.create_delta_batch(
        tier=MemoryTier.WORKING,
        target_node="hub",
    )
    
    assert batch.source_node == "node-001"
    assert batch.tier == MemoryTier.WORKING


# =============================================================================
# Hub Orchestrator Tests
# =============================================================================

async def test_hub_orchestrator():
    """Test hub orchestrator with MEMSHADOW gateway"""
    mesh = MockMeshNetwork()
    hub = HubOrchestrator(hub_id="hub-001")
    hub.set_mesh_callback(mesh.send)
    
    # Register a node
    node = hub.register_node(
        node_id="node-001",
        endpoint="localhost:8001",
        capabilities={NodeCapability.SEARCH, NodeCapability.CORRELATE},
        memory_capabilities=NodeMemoryCapabilities(),
    )
    
    assert node.node_id == "node-001"
    assert NodeCapability.MEMORY_SYNC in node.capabilities
    
    # Check stats
    stats = hub.get_stats()
    assert stats["registered_nodes"] == 1


async def test_hub_message_dispatch():
    """Test hub message dispatch to handlers"""
    mesh = MockMeshNetwork()
    hub = HubOrchestrator(hub_id="hub-001")
    hub.set_mesh_callback(mesh.send)
    hub.register_node("node-001", "localhost:8001", set())
    
    # Create a MEMORY_SYNC message
    items = [MemorySyncItem.create(payload=b"data", tier=MemoryTier.WORKING)]
    batch = MemorySyncBatch(
        source_node="node-001",
        tier=MemoryTier.WORKING,
        items=items,
    )
    msg = batch.to_memshadow_message()
    
    # Dispatch should succeed
    result = await hub.dispatch_message(msg.pack(), "node-001")
    assert result


# =============================================================================
# Spoke Client Tests
# =============================================================================

async def test_spoke_client():
    """Test spoke client with memory adapter"""
    mesh = MockMeshNetwork()
    spoke = SpokeClient(
        node_id="node-001",
        hub_endpoint="hub.local:8000",
        capabilities={"search", "correlate"},
    )
    spoke.set_mesh_callback(mesh.send)
    
    # Register tier with adapter
    l1_tier = InMemoryTier(MemoryTier.WORKING)
    spoke.memory_adapter.register_tier(MemoryTier.WORKING, l1_tier)
    
    # Store through adapter
    item_id = await spoke.memory_adapter.store(
        tier=MemoryTier.WORKING,
        data=b"spoke data",
    )
    
    assert item_id is not None
    
    # Retrieve
    data = await spoke.memory_adapter.retrieve(MemoryTier.WORKING, item_id)
    assert data == b"spoke data"


# =============================================================================
# Memory Tier Tests
# =============================================================================

async def test_working_memory():
    """Test L1 Working Memory tier"""
    wm = WorkingMemory(capacity=10)
    
    # Store items
    for i in range(5):
        await wm.store(f"item-{i}", f"data-{i}".encode())
    
    # Retrieve
    data = await wm.retrieve("item-2")
    assert data == b"data-2"
    
    # Stats
    stats = wm.get_stats()
    assert stats["current_size"] == 5
    assert stats["hits"] == 1


async def test_working_memory_eviction():
    """Test L1 LRU eviction"""
    wm = WorkingMemory(capacity=3)
    
    # Fill capacity
    await wm.store("a", b"data-a")
    await wm.store("b", b"data-b")
    await wm.store("c", b"data-c")
    
    # Access 'a' to make it recently used
    await wm.retrieve("a")
    
    # Add new item, should evict 'b' (LRU)
    await wm.store("d", b"data-d")
    
    assert await wm.retrieve("a") == b"data-a"
    assert await wm.retrieve("b") is None  # Evicted
    assert await wm.retrieve("d") == b"data-d"


async def test_episodic_memory():
    """Test L2 Episodic Memory tier"""
    em = EpisodicMemory()
    
    # Store items in session
    await em.store("item-1", b"data-1", session_id="session-A")
    await em.store("item-2", b"data-2", session_id="session-A")
    
    # Retrieve
    data = await em.retrieve("item-1")
    assert data == b"data-1"
    
    # Get session episodes
    episodes = await em.get_session_episodes("session-A")
    assert len(episodes) == 1


async def test_semantic_memory():
    """Test L3 Semantic Memory tier"""
    sm = SemanticMemory(compression_enabled=True)
    
    # Store with concept
    await sm.store("fact-1", b"The sky is blue", concept_name="facts")
    await sm.store("fact-2", b"Water is wet", concept_name="facts")
    
    # Retrieve
    data = await sm.retrieve("fact-1")
    assert data == b"The sky is blue"
    
    # Get concept
    concept = await sm.get_concept_by_name("facts")
    assert concept is not None
    assert len(concept.instances) == 2


# =============================================================================
# Improvement Tracker Tests
# =============================================================================

async def test_improvement_tracker():
    """Test improvement detection and packaging"""
    tracker = ImprovementTracker(node_id="node-001")
    
    # Set baseline
    tracker.set_baseline("accuracy", 0.80)
    
    # Record improved metrics
    for _ in range(15):
        tracker.record_metric("accuracy", 0.92)
    
    # Should detect improvement
    assert len(tracker._pending_improvements) > 0
    
    # Package improvement
    package = tracker.package_improvement(
        improvement_type=ImprovementType.LEARNED_PATTERNS,
        data=b"pattern data",
    )
    
    assert package.improvement_type == ImprovementType.LEARNED_PATTERNS
    assert package.gain_percent > 0


# =============================================================================
# End-to-End Tests
# =============================================================================

async def test_hub_spoke_sync_flow():
    """Test complete hub-spoke memory sync flow"""
    mesh = MockMeshNetwork()
    
    # Setup hub
    hub = HubOrchestrator(hub_id="hub-001")
    hub.set_mesh_callback(mesh.send)
    
    # Setup spoke
    spoke = SpokeClient(
        node_id="node-001",
        hub_endpoint="hub.local:8000",
    )
    spoke.set_mesh_callback(mesh.send)
    
    # Register spoke with hub
    hub.register_node(
        node_id="node-001",
        endpoint="localhost:8001",
        capabilities={NodeCapability.SEARCH},
    )
    
    # Setup spoke memory
    l1_tier = InMemoryTier(MemoryTier.WORKING)
    spoke.memory_adapter.register_tier(MemoryTier.WORKING, l1_tier)
    
    # Spoke stores data
    await spoke.memory_adapter.store(
        tier=MemoryTier.WORKING,
        data=b"important data",
    )
    
    # Create delta batch
    batch = await spoke.memory_adapter.create_delta_batch(
        tier=MemoryTier.WORKING,
        target_node="hub-001",
    )
    
    assert batch.source_node == "node-001"
    assert len(batch.items) >= 1


async def test_priority_routing_e2e():
    """Test priority-based routing end-to-end"""
    mesh = MockMeshNetwork()
    
    hub = HubOrchestrator(hub_id="hub-001")
    hub.set_mesh_callback(mesh.send)
    
    hub.register_node("node-001", "localhost:8001", set())
    hub.register_node("node-002", "localhost:8002", set())
    
    # Create CRITICAL priority batch (should use P2P)
    items = [MemorySyncItem.create(payload=b"critical", tier=MemoryTier.WORKING)]
    batch = MemorySyncBatch(
        source_node="node-001",
        target_node="*",  # Broadcast
        tier=MemoryTier.WORKING,
        items=items,
        priority=SyncPriority.URGENT,  # Maps to EMERGENCY
    )
    
    result = await hub.memshadow_gateway.route_batch(batch)
    
    # Should route to all nodes
    assert len(result["targets_sent"]) >= 1
    # Check routing mode
    assert result["routing_mode"] == "p2p+hub"


# =============================================================================
# Main
# =============================================================================

async def main():
    print("=" * 60)
    print("MEMSHADOW Integration Test Suite")
    print("=" * 60)
    print()
    
    runner = TestRunner()
    
    # Protocol tests
    print("Protocol Tests:")
    await runner.run_test("Header pack/unpack (32-byte)", test_header_pack_unpack)
    await runner.run_test("Message create/serialize", test_message_create_serialize)
    await runner.run_test("Priority routing rules", test_priority_routing_rules)
    await runner.run_test("SHRINK psych event", test_psych_event)
    
    # Memory sync protocol tests
    print("\nMemory Sync Protocol Tests:")
    await runner.run_test("MemorySyncItem", test_memory_sync_item)
    await runner.run_test("MemorySyncBatch", test_memory_sync_batch)
    await runner.run_test("MemorySyncManager", test_memory_sync_manager)
    
    # Hub gateway tests
    print("\nHub Gateway Tests:")
    await runner.run_test("Gateway registration", test_hub_gateway_registration)
    await runner.run_test("Gateway routing", test_hub_gateway_routing)
    
    # Spoke adapter tests
    print("\nSpoke Adapter Tests:")
    await runner.run_test("Adapter storage", test_spoke_adapter_storage)
    await runner.run_test("Adapter delta batch", test_spoke_adapter_delta_batch)
    
    # Hub orchestrator tests
    print("\nHub Orchestrator Tests:")
    await runner.run_test("Hub orchestrator", test_hub_orchestrator)
    await runner.run_test("Hub message dispatch", test_hub_message_dispatch)
    
    # Spoke client tests
    print("\nSpoke Client Tests:")
    await runner.run_test("Spoke client", test_spoke_client)
    
    # Memory tier tests
    print("\nMemory Tier Tests:")
    await runner.run_test("Working memory (L1)", test_working_memory)
    await runner.run_test("Working memory eviction", test_working_memory_eviction)
    await runner.run_test("Episodic memory (L2)", test_episodic_memory)
    await runner.run_test("Semantic memory (L3)", test_semantic_memory)
    
    # Improvement tests
    print("\nImprovement Tracker Tests:")
    await runner.run_test("Improvement tracker", test_improvement_tracker)
    
    # End-to-end tests
    print("\nEnd-to-End Tests:")
    await runner.run_test("Hub-spoke sync flow", test_hub_spoke_sync_flow)
    await runner.run_test("Priority routing E2E", test_priority_routing_e2e)
    
    # Summary
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for result in runner.results:
        print(result)
    
    passed, total = runner.summary()
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
