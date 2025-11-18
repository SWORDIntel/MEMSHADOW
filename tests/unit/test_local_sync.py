"""
Tests for Local Sync Agent and Differential Protocol
Phase 4: Distributed Architecture
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from app.services.sync.local_agent import (
    LocalSyncAgent,
    SyncStatus,
    SyncDirection
)
from app.services.sync.differential_protocol import (
    DifferentialSyncProtocol,
    SyncDelta,
    SyncBatch
)


# Local Sync Agent Tests

@pytest.mark.asyncio
async def test_local_sync_agent_initialization(tmp_path):
    """Test local sync agent initialization"""
    cache_path = tmp_path / "memshadow_cache"

    agent = LocalSyncAgent(
        local_cache_path=str(cache_path),
        sync_interval=300
    )

    assert agent.local_cache_path == cache_path
    assert agent.l1_cache.exists()
    assert agent.l2_cache.exists()
    assert agent.status == SyncStatus.IDLE
    assert agent.last_sync is None


@pytest.mark.asyncio
async def test_cache_item_l1(tmp_path):
    """Test caching item in L1"""
    agent = LocalSyncAgent(local_cache_path=str(tmp_path))

    test_data = {
        "id": "mem_123",
        "content": "Test memory",
        "created_at": datetime.utcnow().isoformat()
    }

    await agent.cache_item("mem_123", test_data, tier="l1")

    # Verify file exists
    cached_file = agent.l1_cache / "mem_123.json"
    assert cached_file.exists()

    # Verify content
    cached_data = json.loads(cached_file.read_text())
    assert cached_data["id"] == "mem_123"
    assert cached_data["content"] == "Test memory"


@pytest.mark.asyncio
async def test_cache_item_l2(tmp_path):
    """Test caching item in L2"""
    agent = LocalSyncAgent(local_cache_path=str(tmp_path))

    test_data = {
        "id": "mem_456",
        "content": "Warm data"
    }

    await agent.cache_item("mem_456", test_data, tier="l2")

    cached_file = agent.l2_cache / "mem_456.json"
    assert cached_file.exists()


@pytest.mark.asyncio
async def test_get_cached_item_l1_hit(tmp_path):
    """Test cache hit in L1"""
    agent = LocalSyncAgent(local_cache_path=str(tmp_path))

    test_data = {"id": "mem_123", "content": "Test"}
    await agent.cache_item("mem_123", test_data, tier="l1")

    # Retrieve from cache
    cached = await agent.get_cached_item("mem_123")

    assert cached is not None
    assert cached["id"] == "mem_123"
    assert cached["content"] == "Test"


@pytest.mark.asyncio
async def test_get_cached_item_l2_promotion(tmp_path):
    """Test L2 cache hit with L1 promotion"""
    agent = LocalSyncAgent(local_cache_path=str(tmp_path))

    test_data = {"id": "mem_456", "content": "Warm data"}
    await agent.cache_item("mem_456", test_data, tier="l2")

    # Item should be in L2
    assert (agent.l2_cache / "mem_456.json").exists()
    assert not (agent.l1_cache / "mem_456.json").exists()

    # Retrieve from cache (should promote to L1)
    cached = await agent.get_cached_item("mem_456")

    assert cached is not None
    assert cached["content"] == "Warm data"

    # Should now be in L1
    assert (agent.l1_cache / "mem_456.json").exists()


@pytest.mark.asyncio
async def test_cache_miss(tmp_path):
    """Test cache miss"""
    agent = LocalSyncAgent(local_cache_path=str(tmp_path))

    cached = await agent.get_cached_item("nonexistent")
    assert cached is None


@pytest.mark.asyncio
async def test_get_cache_stats(tmp_path):
    """Test cache statistics"""
    agent = LocalSyncAgent(local_cache_path=str(tmp_path))

    # Add some items
    await agent.cache_item("mem_1", {"id": "mem_1"}, tier="l1")
    await agent.cache_item("mem_2", {"id": "mem_2"}, tier="l1")
    await agent.cache_item("mem_3", {"id": "mem_3"}, tier="l2")

    stats = await agent.get_cache_stats()

    assert stats["l1"]["count"] == 2
    assert stats["l2"]["count"] == 1
    assert stats["l1"]["size_bytes"] > 0
    assert stats["l2"]["size_bytes"] > 0
    assert stats["status"] == SyncStatus.IDLE


@pytest.mark.asyncio
async def test_sync_upload_direction(tmp_path):
    """Test sync with upload direction"""
    agent = LocalSyncAgent(local_cache_path=str(tmp_path))

    # Add some local changes
    await agent.cache_item("mem_1", {"id": "mem_1", "content": "Test"}, tier="l1")

    result = await agent.sync(direction=SyncDirection.UPLOAD)

    assert result["direction"] == SyncDirection.UPLOAD
    assert "uploaded" in result
    assert "duration_seconds" in result


@pytest.mark.asyncio
async def test_sync_download_direction(tmp_path):
    """Test sync with download direction"""
    agent = LocalSyncAgent(local_cache_path=str(tmp_path))

    result = await agent.sync(direction=SyncDirection.DOWNLOAD)

    assert result["direction"] == SyncDirection.DOWNLOAD
    assert "downloaded" in result


@pytest.mark.asyncio
async def test_sync_bidirectional(tmp_path):
    """Test bidirectional sync"""
    agent = LocalSyncAgent(local_cache_path=str(tmp_path))

    result = await agent.sync(direction=SyncDirection.BIDIRECTIONAL)

    assert result["direction"] == SyncDirection.BIDIRECTIONAL
    assert "uploaded" in result
    assert "downloaded" in result
    assert "conflicts" in result


# Differential Sync Protocol Tests

@pytest.mark.asyncio
async def test_differential_protocol_initialization():
    """Test differential protocol initialization"""
    protocol = DifferentialSyncProtocol(user_id="test_user")

    assert protocol.user_id == "test_user"
    assert len(protocol.change_log) == 0
    assert len(protocol.checkpoints) == 0


@pytest.mark.asyncio
async def test_create_delta():
    """Test creating sync delta"""
    protocol = DifferentialSyncProtocol(user_id="test_user")

    delta = await protocol.create_delta(
        operation="create",
        resource_type="memory",
        resource_id="mem_123",
        data={"content": "Test memory"}
    )

    assert delta.operation == "create"
    assert delta.resource_type == "memory"
    assert delta.resource_id == "mem_123"
    assert delta.data["content"] == "Test memory"
    assert delta.checksum is not None
    assert delta.version == 1


@pytest.mark.asyncio
async def test_create_delta_update():
    """Test creating update delta"""
    protocol = DifferentialSyncProtocol(user_id="test_user")

    old_data = {"content": "Original"}
    new_data = {"content": "Updated"}

    delta = await protocol.create_delta(
        operation="update",
        resource_type="memory",
        resource_id="mem_123",
        data=new_data,
        previous_data=old_data
    )

    assert delta.operation == "update"
    assert delta.data == new_data
    assert delta.previous_data == old_data


@pytest.mark.asyncio
async def test_create_delta_delete():
    """Test creating delete delta"""
    protocol = DifferentialSyncProtocol(user_id="test_user")

    delta = await protocol.create_delta(
        operation="delete",
        resource_type="memory",
        resource_id="mem_123"
    )

    assert delta.operation == "delete"
    assert delta.resource_id == "mem_123"
    assert delta.data is None


@pytest.mark.asyncio
async def test_compute_diff_new_items():
    """Test computing diff for new items"""
    protocol = DifferentialSyncProtocol(user_id="test_user")

    local_state = {
        "mem_1": {"content": "Memory 1"},
        "mem_2": {"content": "Memory 2"}
    }

    remote_state = {
        "mem_1": {"content": "Memory 1"}
        # mem_2 is new locally
    }

    deltas = await protocol.compute_diff(local_state, remote_state)

    # Should have one create delta for mem_2
    create_deltas = [d for d in deltas if d.operation == "create"]
    assert len(create_deltas) == 1
    assert create_deltas[0].resource_id == "mem_2"


@pytest.mark.asyncio
async def test_compute_diff_deleted_items():
    """Test computing diff for deleted items"""
    protocol = DifferentialSyncProtocol(user_id="test_user")

    local_state = {
        "mem_1": {"content": "Memory 1"}
    }

    remote_state = {
        "mem_1": {"content": "Memory 1"},
        "mem_2": {"content": "Memory 2"}
        # mem_2 was deleted locally
    }

    deltas = await protocol.compute_diff(local_state, remote_state)

    delete_deltas = [d for d in deltas if d.operation == "delete"]
    assert len(delete_deltas) == 1
    assert delete_deltas[0].resource_id == "mem_2"


@pytest.mark.asyncio
async def test_compute_diff_updated_items():
    """Test computing diff for updated items"""
    protocol = DifferentialSyncProtocol(user_id="test_user")

    local_state = {
        "mem_1": {"content": "Updated locally"}
    }

    remote_state = {
        "mem_1": {"content": "Original content"}
    }

    deltas = await protocol.compute_diff(local_state, remote_state)

    update_deltas = [d for d in deltas if d.operation == "update"]
    assert len(update_deltas) == 1
    assert update_deltas[0].resource_id == "mem_1"


@pytest.mark.asyncio
async def test_create_batch():
    """Test creating sync batch"""
    protocol = DifferentialSyncProtocol(user_id="test_user")

    delta1 = await protocol.create_delta("create", "memory", "mem_1", {"content": "A"})
    delta2 = await protocol.create_delta("create", "memory", "mem_2", {"content": "B"})

    batch = await protocol.create_batch(
        deltas=[delta1, delta2],
        device_id="laptop-001"
    )

    assert len(batch.deltas) == 2
    assert batch.device_id == "laptop-001"
    assert batch.compression_enabled is True
    assert batch.total_size > 0


@pytest.mark.asyncio
async def test_apply_batch():
    """Test applying sync batch"""
    protocol = DifferentialSyncProtocol(user_id="test_user")

    # Create batch with create operations
    delta1 = await protocol.create_delta("create", "memory", "mem_1", {"content": "A"})
    delta2 = await protocol.create_delta("create", "memory", "mem_2", {"content": "B"})

    batch = await protocol.create_batch([delta1, delta2], "laptop-001")

    current_state = {}
    new_state = await protocol.apply_batch(batch, current_state)

    assert "mem_1" in new_state
    assert "mem_2" in new_state
    assert new_state["mem_1"]["content"] == "A"
    assert new_state["mem_2"]["content"] == "B"


@pytest.mark.asyncio
async def test_create_checkpoint():
    """Test creating sync checkpoint"""
    protocol = DifferentialSyncProtocol(user_id="test_user")

    checkpoint = await protocol.create_checkpoint(
        device_id="laptop-001",
        sync_state={"mem_1": {"content": "Test"}}
    )

    assert checkpoint.device_id == "laptop-001"
    assert checkpoint.sync_state["mem_1"]["content"] == "Test"
    assert "laptop-001" in protocol.checkpoints


@pytest.mark.asyncio
async def test_get_checkpoint():
    """Test retrieving checkpoint"""
    protocol = DifferentialSyncProtocol(user_id="test_user")

    # Create checkpoint
    await protocol.create_checkpoint(
        "laptop-001",
        {"mem_1": {"content": "Test"}}
    )

    # Retrieve it
    checkpoint = await protocol.get_checkpoint("laptop-001")

    assert checkpoint is not None
    assert checkpoint.device_id == "laptop-001"
    assert checkpoint.sync_state["mem_1"]["content"] == "Test"


@pytest.mark.asyncio
async def test_estimate_bandwidth():
    """Test bandwidth estimation"""
    protocol = DifferentialSyncProtocol(user_id="test_user")

    delta1 = await protocol.create_delta("create", "memory", "mem_1", {"content": "Test" * 100})
    delta2 = await protocol.create_delta("create", "memory", "mem_2", {"content": "Test" * 100})

    bandwidth = await protocol.estimate_bandwidth([delta1, delta2])

    assert bandwidth > 0
    assert isinstance(bandwidth, int)
