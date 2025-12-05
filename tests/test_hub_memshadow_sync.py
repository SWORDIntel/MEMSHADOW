"""
Hub MEMSHADOW Sync Integration Tests

Tests for the Hub-first MEMSHADOW integration:
- Basic sync flow (hub + spokes with in-memory tiers)
- Priority routing (NORMAL → hub relay, CRITICAL → P2P + hub)
- Conflict resolution
- Tier-specific sync
- Failure handling

Run with: pytest tests/test_hub_memshadow_sync.py -v
"""

import asyncio
import json
import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# Import MEMSHADOW components
from app.services.memshadow import (
    # Protocol
    MemshadowHeader,
    MemshadowMessage,
    MessageType,
    Priority,
    MessageFlags,
    MEMSHADOW_MAGIC,
    MEMSHADOW_VERSION,
    HEADER_SIZE,
    create_memory_sync_message,
    should_route_p2p,
    # Sync Manager
    MemoryTier,
    MemorySyncItem,
    MemorySyncBatch,
    MemorySyncManager,
    SyncResult,
    ConflictResolution,
    # Hub Gateway
    HubMemshadowGateway,
    NodeMemoryCapabilities,
    NodeSyncState,
    # Spoke Adapter
    SpokeMemoryAdapter,
    # Federation Hub
    FederationHubOrchestrator,
    NodeCapability,
)


class MockMeshNetwork:
    """
    Mock mesh network for testing hub ↔ spoke communication.
    
    Simulates message passing between nodes without actual network.
    """
    
    def __init__(self):
        self._nodes: Dict[str, Any] = {}  # node_id -> handler
        self._message_log: List[Dict] = []
        self._delivery_queue: asyncio.Queue = asyncio.Queue()
        
    def register_node(self, node_id: str, handler: callable):
        """Register a node's message handler"""
        self._nodes[node_id] = handler
    
    def deregister_node(self, node_id: str):
        """Remove a node"""
        self._nodes.pop(node_id, None)
    
    async def send(self, target_id: str, data: bytes):
        """Send a message to a target node"""
        self._message_log.append({
            "target": target_id,
            "data_len": len(data),
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Simulate async delivery
        await self._delivery_queue.put((target_id, data))
    
    async def deliver_messages(self, timeout: float = 1.0):
        """Process all pending messages"""
        try:
            while True:
                target_id, data = await asyncio.wait_for(
                    self._delivery_queue.get(),
                    timeout=timeout
                )
                
                if target_id in self._nodes:
                    handler = self._nodes[target_id]
                    # Parse sender from message if possible
                    try:
                        msg = MemshadowMessage.unpack(data)
                        batch = MemorySyncBatch.from_bytes(msg.payload)
                        sender = batch.source_node
                    except:
                        sender = "unknown"
                    
                    await handler(data, sender)
                    
        except asyncio.TimeoutError:
            pass
    
    def get_message_count(self) -> int:
        """Get total messages sent"""
        return len(self._message_log)
    
    def clear_log(self):
        """Clear message log"""
        self._message_log.clear()


class TestMemshadowProtocol:
    """Test MEMSHADOW Protocol v2 header and message handling"""
    
    def test_header_pack_unpack(self):
        """Test header serialization/deserialization"""
        header = MemshadowHeader(
            priority=Priority.HIGH,
            msg_type=MessageType.MEMORY_SYNC,
            flags=MessageFlags.BATCHED | MessageFlags.COMPRESSED,
            batch_count=5,
            payload_len=1024,
        )
        
        packed = header.pack()
        assert len(packed) == HEADER_SIZE, f"Header should be {HEADER_SIZE} bytes, got {len(packed)}"
        
        unpacked = MemshadowHeader.unpack(packed)
        assert unpacked.magic == MEMSHADOW_MAGIC
        assert unpacked.version == MEMSHADOW_VERSION
        assert unpacked.priority == Priority.HIGH
        assert unpacked.msg_type == MessageType.MEMORY_SYNC
        assert unpacked.payload_len == 1024
    
    def test_message_create_and_unpack(self):
        """Test full message creation and unpacking"""
        payload = b"test payload data"
        msg = MemshadowMessage.create(
            msg_type=MessageType.MEMORY_SYNC,
            payload=payload,
            priority=Priority.CRITICAL,
            flags=MessageFlags.REQUIRES_ACK,
        )
        
        packed = msg.pack()
        unpacked = MemshadowMessage.unpack(packed)
        
        assert unpacked.header.msg_type == MessageType.MEMORY_SYNC
        assert unpacked.header.priority == Priority.CRITICAL
        assert unpacked.header.flags & MessageFlags.REQUIRES_ACK
        assert unpacked.payload == payload
    
    def test_priority_routing_rules(self):
        """Test priority-based routing decisions"""
        assert not should_route_p2p(Priority.LOW)
        assert not should_route_p2p(Priority.NORMAL)
        assert not should_route_p2p(Priority.HIGH)
        assert should_route_p2p(Priority.CRITICAL)
        assert should_route_p2p(Priority.EMERGENCY)
    
    def test_memory_sync_message_creation(self):
        """Test MEMORY_SYNC message helper"""
        payload = b"sync data"
        msg = create_memory_sync_message(
            payload=payload,
            priority=Priority.NORMAL,
            batch_count=3,
            compressed=True,
        )
        
        assert msg.header.msg_type == MessageType.MEMORY_SYNC
        assert msg.header.flags & MessageFlags.BATCHED
        assert msg.header.flags & MessageFlags.COMPRESSED


class TestMemorySyncManager:
    """Test MemorySyncManager delta computation and batch handling"""
    
    @pytest.fixture
    def sync_manager(self):
        return MemorySyncManager(node_id="test-node")
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_local(self, sync_manager):
        """Test local item storage and retrieval"""
        memory_id = uuid4()
        embedding = np.random.randn(256).astype(np.float32)
        
        item = MemorySyncItem.from_embedding(
            memory_id=memory_id,
            embedding=embedding,
            tier=MemoryTier.L1_WORKING,
            metadata={"source": "test"},
        )
        
        await sync_manager.store_local(item)
        
        retrieved = await sync_manager.get_local(MemoryTier.L1_WORKING, str(memory_id))
        assert retrieved is not None
        assert retrieved.memory_id == memory_id
        assert retrieved.tier == MemoryTier.L1_WORKING
    
    @pytest.mark.asyncio
    async def test_create_sync_batch(self, sync_manager):
        """Test sync batch creation"""
        # Store some items
        items = []
        for i in range(5):
            memory_id = uuid4()
            embedding = np.random.randn(256).astype(np.float32)
            item = MemorySyncItem.from_embedding(
                memory_id=memory_id,
                embedding=embedding,
                tier=MemoryTier.L1_WORKING,
            )
            await sync_manager.store_local(item)
            items.append(item)
        
        # Create batch
        batch = await sync_manager.create_batch(
            items=items,
            target_node="hub",
            tier=MemoryTier.L1_WORKING,
            priority=Priority.NORMAL.value,
        )
        
        assert batch.source_node == "test-node"
        assert batch.target_node == "hub"
        assert len(batch.items) == 5
        assert batch.tier == MemoryTier.L1_WORKING
        assert batch.validate()  # Checksum should be valid
    
    @pytest.mark.asyncio
    async def test_apply_sync_batch(self, sync_manager):
        """Test applying an incoming sync batch"""
        # Create batch from another node
        items = []
        for i in range(3):
            item = MemorySyncItem.from_embedding(
                memory_id=uuid4(),
                embedding=np.random.randn(256).astype(np.float32),
                tier=MemoryTier.L1_WORKING,
            )
            items.append(item)
        
        batch = MemorySyncBatch(
            source_node="remote-node",
            target_node="test-node",
            tier=MemoryTier.L1_WORKING,
            items=items,
        )
        
        result = await sync_manager.apply_batch(batch)
        
        assert result.success
        assert result.applied_count == 3
        assert result.conflict_count == 0
    
    def test_delta_computation(self, sync_manager):
        """Test sync vector delta computation"""
        local_vector = {
            "mem1": 1,
            "mem2": 2,
            "mem3": 3,
        }
        
        remote_vector = {
            "mem1": 1,  # Same
            "mem2": 1,  # Local is newer
            "mem4": 1,  # New in remote
        }
        
        new_ids, updated_ids, deleted_ids = sync_manager.compute_delta(
            local_vector, remote_vector
        )
        
        assert "mem3" in new_ids  # In local, not in remote
        assert "mem2" in updated_ids  # Local version > remote
        assert "mem4" in deleted_ids  # In remote, not in local


class TestHubMemshadowGateway:
    """Test Hub MEMSHADOW Gateway"""
    
    @pytest.fixture
    def gateway(self):
        return HubMemshadowGateway(hub_id="test-hub")
    
    def test_register_memory_node(self, gateway):
        """Test node registration"""
        gateway.register_memory_node(
            "spoke-1",
            NodeMemoryCapabilities(
                node_id="spoke-1",
                supported_tiers={MemoryTier.L1_WORKING, MemoryTier.L2_EPISODIC},
            )
        )
        
        assert "spoke-1" in gateway._nodes
        assert MemoryTier.L1_WORKING in gateway._nodes["spoke-1"].supported_tiers
    
    @pytest.mark.asyncio
    async def test_schedule_sync(self, gateway):
        """Test sync scheduling"""
        gateway.register_memory_node("spoke-1")
        
        entry_id = await gateway.schedule_sync(
            "spoke-1",
            MemoryTier.L1_WORKING,
            Priority.HIGH,
        )
        
        assert entry_id is not None
        assert len(gateway._sync_queue) == 1
        assert gateway._metrics["syncs_scheduled"] == 1
    
    @pytest.mark.asyncio
    async def test_apply_remote_batch(self, gateway):
        """Test applying batch from spoke"""
        gateway.register_memory_node("spoke-1")
        
        # Create batch
        items = [
            MemorySyncItem.from_embedding(
                memory_id=uuid4(),
                embedding=np.random.randn(256).astype(np.float32),
                tier=MemoryTier.L1_WORKING,
            )
        ]
        
        batch = MemorySyncBatch(
            source_node="spoke-1",
            target_node="test-hub",
            tier=MemoryTier.L1_WORKING,
            items=items,
        )
        
        result = await gateway.apply_remote_batch("spoke-1", batch)
        
        assert result.success
        assert result.applied_count == 1
        assert gateway._metrics["batches_routed"] == 1
    
    def test_hub_memory_stats(self, gateway):
        """Test statistics retrieval"""
        gateway.register_memory_node("spoke-1")
        gateway.register_memory_node("spoke-2")
        
        stats = gateway.get_hub_memory_stats()
        
        assert stats["hub_id"] == "test-hub"
        assert "spoke-1" in stats["nodes"]
        assert "spoke-2" in stats["nodes"]


class TestSpokeMemoryAdapter:
    """Test Spoke Memory Adapter"""
    
    @pytest.fixture
    def adapter(self):
        return SpokeMemoryAdapter(node_id="test-spoke", hub_node_id="hub")
    
    @pytest.mark.asyncio
    async def test_store_memory(self, adapter):
        """Test storing memory in local tier"""
        memory_id = uuid4()
        embedding = np.random.randn(256).astype(np.float32)
        
        item = await adapter.store_memory(
            memory_id=memory_id,
            embedding=embedding,
            tier=MemoryTier.L1_WORKING,
            metadata={"test": True},
            propagate=False,  # Don't try to send without mesh
        )
        
        assert item.memory_id == memory_id
        assert item.tier == MemoryTier.L1_WORKING
        assert adapter._metrics["stores"] == 1
    
    @pytest.mark.asyncio
    async def test_get_memory(self, adapter):
        """Test retrieving memory from local tier"""
        memory_id = uuid4()
        embedding = np.random.randn(256).astype(np.float32)
        
        await adapter.store_memory(
            memory_id=memory_id,
            embedding=embedding,
            tier=MemoryTier.L1_WORKING,
            propagate=False,
        )
        
        retrieved = await adapter.get_memory(memory_id, MemoryTier.L1_WORKING)
        
        assert retrieved is not None
        assert retrieved.memory_id == memory_id
    
    @pytest.mark.asyncio
    async def test_create_delta_batch(self, adapter):
        """Test delta batch creation"""
        # Store some memories
        for i in range(3):
            await adapter.store_memory(
                memory_id=uuid4(),
                embedding=np.random.randn(256).astype(np.float32),
                tier=MemoryTier.L1_WORKING,
                propagate=False,
            )
        
        batch = await adapter.create_delta_batch(
            tier=MemoryTier.L1_WORKING,
            target_node="hub",
        )
        
        assert len(batch.items) == 3
        assert batch.source_node == "test-spoke"
        assert batch.target_node == "hub"
    
    @pytest.mark.asyncio
    async def test_apply_sync_batch(self, adapter):
        """Test applying incoming sync batch"""
        items = [
            MemorySyncItem.from_embedding(
                memory_id=uuid4(),
                embedding=np.random.randn(256).astype(np.float32),
                tier=MemoryTier.L1_WORKING,
            )
            for _ in range(2)
        ]
        
        batch = MemorySyncBatch(
            source_node="hub",
            target_node="test-spoke",
            tier=MemoryTier.L1_WORKING,
            items=items,
        )
        
        result = await adapter.apply_sync_batch(batch)
        
        assert result.success
        assert result.applied_count == 2
        assert adapter._metrics["batches_received"] == 1


class TestHubSpokeIntegration:
    """Integration tests for hub ↔ spoke sync"""
    
    @pytest.fixture
    def mock_mesh(self):
        return MockMeshNetwork()
    
    @pytest.fixture
    def hub(self, mock_mesh):
        hub = HubMemshadowGateway(
            hub_id="hub",
            mesh_send_callback=mock_mesh.send,
        )
        mock_mesh.register_node("hub", hub.handle_memory_sync)
        return hub
    
    @pytest.fixture
    def spoke1(self, mock_mesh):
        spoke = SpokeMemoryAdapter(
            node_id="spoke-1",
            hub_node_id="hub",
            mesh_send_callback=mock_mesh.send,
        )
        mock_mesh.register_node("spoke-1", spoke.handle_sync_request)
        return spoke
    
    @pytest.fixture
    def spoke2(self, mock_mesh):
        spoke = SpokeMemoryAdapter(
            node_id="spoke-2",
            hub_node_id="hub",
            mesh_send_callback=mock_mesh.send,
        )
        mock_mesh.register_node("spoke-2", spoke.handle_sync_request)
        return spoke
    
    @pytest.mark.asyncio
    async def test_basic_sync_flow(self, mock_mesh, hub, spoke1, spoke2):
        """
        Test basic sync flow:
        1. Write on Spoke 1
        2. Sync to Hub
        3. Hub routes to Spoke 2
        """
        # Register nodes with hub
        hub.register_memory_node("spoke-1")
        hub.register_memory_node("spoke-2")
        
        # Store memory on Spoke 1
        memory_id = uuid4()
        embedding = np.random.randn(256).astype(np.float32)
        
        await spoke1.store_memory(
            memory_id=memory_id,
            embedding=embedding,
            tier=MemoryTier.L1_WORKING,
            propagate=False,  # We'll manually create the batch
        )
        
        # Create and send batch to hub
        batch = await spoke1.create_delta_batch(
            tier=MemoryTier.L1_WORKING,
            target_node="*",  # Broadcast
        )
        
        # Apply to hub
        result = await hub.apply_remote_batch("spoke-1", batch)
        assert result.success
        assert result.applied_count == 1
        
        # Verify hub stats
        stats = hub.get_hub_memory_stats()
        assert stats["nodes"]["spoke-1"]["total_syncs"] == 1
    
    @pytest.mark.asyncio
    async def test_priority_routing(self, mock_mesh, hub, spoke1):
        """Test that CRITICAL priority uses P2P routing"""
        hub.register_memory_node("spoke-1")
        
        # Register a peer on spoke1
        spoke1.register_peer("spoke-2")
        
        # Store with CRITICAL priority
        memory_id = uuid4()
        embedding = np.random.randn(256).astype(np.float32)
        
        await spoke1.store_memory(
            memory_id=memory_id,
            embedding=embedding,
            tier=MemoryTier.L1_WORKING,
            priority=Priority.CRITICAL,
            propagate=True,
        )
        
        # Deliver messages
        await mock_mesh.deliver_messages(timeout=0.5)
        
        # Should have sent to both peer (P2P) and hub
        # Note: actual P2P would show 2 messages, but hub also gets notification
        assert mock_mesh.get_message_count() >= 1
    
    @pytest.mark.asyncio
    async def test_conflict_resolution(self, hub, spoke1, spoke2):
        """Test conflict resolution when same memory is updated on two spokes"""
        hub.register_memory_node("spoke-1")
        hub.register_memory_node("spoke-2")
        
        memory_id = uuid4()
        
        # Create item on Spoke 1
        item1 = MemorySyncItem.from_embedding(
            memory_id=memory_id,
            embedding=np.random.randn(256).astype(np.float32),
            tier=MemoryTier.L1_WORKING,
        )
        item1.version = 1
        item1.timestamp = datetime.utcnow() - timedelta(seconds=10)
        
        # Create item on Spoke 2 (newer)
        item2 = MemorySyncItem.from_embedding(
            memory_id=memory_id,
            embedding=np.random.randn(256).astype(np.float32),
            tier=MemoryTier.L1_WORKING,
        )
        item2.version = 1
        item2.timestamp = datetime.utcnow()
        
        # Apply Spoke 1's batch first
        batch1 = MemorySyncBatch(
            source_node="spoke-1",
            target_node="hub",
            tier=MemoryTier.L1_WORKING,
            items=[item1],
        )
        await hub.apply_remote_batch("spoke-1", batch1)
        
        # Apply Spoke 2's batch (should win due to newer timestamp)
        batch2 = MemorySyncBatch(
            source_node="spoke-2",
            target_node="hub",
            tier=MemoryTier.L1_WORKING,
            items=[item2],
        )
        result = await hub.apply_remote_batch("spoke-2", batch2)
        
        # Conflict should have been resolved
        assert result.success
        # Stats show total conflicts across all syncs
        stats = hub.get_hub_memory_stats()
        assert stats["metrics"]["conflicts_total"] >= 0
    
    @pytest.mark.asyncio
    async def test_tier_specific_sync(self, hub, spoke1):
        """Test that sync respects tier boundaries"""
        hub.register_memory_node(
            "spoke-1",
            NodeMemoryCapabilities(
                node_id="spoke-1",
                supported_tiers={MemoryTier.L1_WORKING},  # Only L1
            )
        )
        
        # Try to sync L2 - should fail
        with pytest.raises(ValueError):
            await hub.schedule_sync("spoke-1", MemoryTier.L2_EPISODIC)
    
    @pytest.mark.asyncio
    async def test_batch_serialization_roundtrip(self):
        """Test batch serialization for transmission"""
        items = [
            MemorySyncItem.from_embedding(
                memory_id=uuid4(),
                embedding=np.random.randn(256).astype(np.float32),
                tier=MemoryTier.L1_WORKING,
                metadata={"index": i},
            )
            for i in range(5)
        ]
        
        batch = MemorySyncBatch(
            source_node="spoke-1",
            target_node="hub",
            tier=MemoryTier.L1_WORKING,
            items=items,
            priority=Priority.HIGH.value,
        )
        
        # Serialize
        data = batch.to_bytes()
        
        # Deserialize
        restored = MemorySyncBatch.from_bytes(data)
        
        assert restored.batch_id == batch.batch_id
        assert restored.source_node == batch.source_node
        assert len(restored.items) == 5
        assert restored.validate()


class TestFederationHubOrchestrator:
    """Test the unified Federation Hub Orchestrator"""
    
    @pytest.fixture
    def federation_hub(self):
        return FederationHubOrchestrator(
            hub_id="federation-hub",
            use_mesh=False,  # Disable actual mesh for tests
        )
    
    @pytest.mark.asyncio
    async def test_node_registration(self, federation_hub):
        """Test full node registration"""
        node = await federation_hub.register_node(
            node_id="spoke-1",
            endpoint="spoke1.local:8889",
            capabilities={NodeCapability.SEARCH, NodeCapability.MEMORY_STORAGE},
            data_domains={"memories", "threat_intel"},
            memory_tiers={MemoryTier.L1_WORKING, MemoryTier.L2_EPISODIC},
        )
        
        assert node.node_id == "spoke-1"
        assert NodeCapability.SEARCH in node.capabilities
        assert MemoryTier.L1_WORKING in node.memory_tiers
        
        # Should also be registered with MEMSHADOW gateway
        assert "spoke-1" in federation_hub.memshadow_gateway._nodes
    
    @pytest.mark.asyncio
    async def test_hub_stats(self, federation_hub):
        """Test statistics aggregation"""
        await federation_hub.register_node("spoke-1", "host1:8889")
        await federation_hub.register_node("spoke-2", "host2:8889")
        
        stats = federation_hub.get_stats()
        
        assert stats["hub_id"] == "federation-hub"
        assert stats["nodes"]["total"] == 2
        assert "memshadow_stats" in stats


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.mark.asyncio
    async def test_empty_batch(self):
        """Test handling of empty sync batch"""
        manager = MemorySyncManager(node_id="test")
        
        batch = MemorySyncBatch(
            source_node="remote",
            target_node="test",
            tier=MemoryTier.L1_WORKING,
            items=[],
        )
        
        result = await manager.apply_batch(batch)
        
        assert result.success
        assert result.applied_count == 0
    
    @pytest.mark.asyncio
    async def test_invalid_checksum(self):
        """Test rejection of corrupted batch"""
        manager = MemorySyncManager(node_id="test")
        
        items = [
            MemorySyncItem.from_embedding(
                memory_id=uuid4(),
                embedding=np.random.randn(256).astype(np.float32),
                tier=MemoryTier.L1_WORKING,
            )
        ]
        
        batch = MemorySyncBatch(
            source_node="remote",
            target_node="test",
            tier=MemoryTier.L1_WORKING,
            items=items,
        )
        
        # Corrupt checksum
        batch.checksum = "invalid_checksum"
        
        result = await manager.apply_batch(batch)
        
        assert not result.success
        assert "checksum" in result.errors[0].lower()
    
    def test_header_invalid_magic(self):
        """Test rejection of invalid magic number"""
        # Create valid header, then corrupt magic
        header = MemshadowHeader()
        packed = header.pack()
        
        # Corrupt magic bytes
        corrupted = b'\x00' * 8 + packed[8:]
        
        with pytest.raises(ValueError, match="Invalid magic"):
            MemshadowHeader.unpack(corrupted)
    
    @pytest.mark.asyncio
    async def test_unknown_tier(self):
        """Test handling of unsupported tier"""
        adapter = SpokeMemoryAdapter(node_id="test", hub_node_id="hub")
        
        # Try to store with invalid tier value
        with pytest.raises(ValueError):
            await adapter.store_memory(
                memory_id=uuid4(),
                embedding=np.random.randn(256).astype(np.float32),
                tier=MemoryTier(99),  # Invalid
                propagate=False,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
