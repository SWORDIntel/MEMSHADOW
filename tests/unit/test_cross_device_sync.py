"""
Tests for Cross-Device Synchronization
Phase 4: Distributed Architecture
"""

import pytest
from datetime import datetime, timedelta
from app.services.distributed.cross_device_sync import (
    CrossDeviceSyncOrchestrator,
    DeviceType,
    DeviceStatus,
    SyncPriority
)


@pytest.mark.asyncio
async def test_orchestrator_initialization():
    """Test orchestrator initialization"""
    orchestrator = CrossDeviceSyncOrchestrator(user_id="test_user")

    assert orchestrator.user_id == "test_user"
    assert orchestrator.max_devices == 10
    assert len(orchestrator.devices) == 0
    assert len(orchestrator.sync_history) == 0


@pytest.mark.asyncio
async def test_register_device():
    """Test device registration"""
    orchestrator = CrossDeviceSyncOrchestrator(user_id="test_user")

    device = await orchestrator.register_device(
        device_id="laptop-001",
        device_type=DeviceType.LAPTOP,
        device_name="MacBook Pro",
        capabilities={"storage_gb": 100, "compute": "high"}
    )

    assert device.device_id == "laptop-001"
    assert device.device_type == DeviceType.LAPTOP
    assert device.device_name == "MacBook Pro"
    assert device.status == DeviceStatus.ONLINE
    assert device.capabilities["storage_gb"] == 100
    assert "laptop-001" in orchestrator.devices


@pytest.mark.asyncio
async def test_register_multiple_devices():
    """Test registering multiple devices"""
    orchestrator = CrossDeviceSyncOrchestrator(user_id="test_user")

    await orchestrator.register_device(
        "laptop-001", DeviceType.LAPTOP, "MacBook"
    )
    await orchestrator.register_device(
        "mobile-001", DeviceType.MOBILE, "iPhone"
    )
    await orchestrator.register_device(
        "tablet-001", DeviceType.TABLET, "iPad"
    )

    assert len(orchestrator.devices) == 3
    assert "laptop-001" in orchestrator.devices
    assert "mobile-001" in orchestrator.devices
    assert "tablet-001" in orchestrator.devices


@pytest.mark.asyncio
async def test_max_devices_limit():
    """Test maximum device limit"""
    orchestrator = CrossDeviceSyncOrchestrator(
        user_id="test_user",
        max_devices=2
    )

    await orchestrator.register_device(
        "device-001", DeviceType.LAPTOP, "Device 1"
    )
    await orchestrator.register_device(
        "device-002", DeviceType.MOBILE, "Device 2"
    )

    # Should fail - exceeds max
    with pytest.raises(ValueError, match="Maximum device limit"):
        await orchestrator.register_device(
            "device-003", DeviceType.TABLET, "Device 3"
        )


@pytest.mark.asyncio
async def test_sync_memory():
    """Test memory synchronization"""
    orchestrator = CrossDeviceSyncOrchestrator(user_id="test_user")

    # Register devices
    await orchestrator.register_device(
        "laptop-001", DeviceType.LAPTOP, "Laptop"
    )
    await orchestrator.register_device(
        "mobile-001", DeviceType.MOBILE, "Mobile"
    )

    # Sync memory from laptop to mobile
    operation = await orchestrator.sync_memory(
        memory_id="mem_123",
        from_device="laptop-001",
        target_devices=["mobile-001"],
        priority=SyncPriority.HIGH
    )

    assert operation.source_device == "laptop-001"
    assert "mobile-001" in operation.target_devices
    assert operation.resource_type == "memory"
    assert operation.resource_id == "mem_123"
    assert operation.priority == SyncPriority.HIGH


@pytest.mark.asyncio
async def test_sync_to_all_devices():
    """Test syncing to all devices"""
    orchestrator = CrossDeviceSyncOrchestrator(user_id="test_user")

    # Register multiple devices
    await orchestrator.register_device(
        "laptop-001", DeviceType.LAPTOP, "Laptop"
    )
    await orchestrator.register_device(
        "mobile-001", DeviceType.MOBILE, "Mobile"
    )
    await orchestrator.register_device(
        "tablet-001", DeviceType.TABLET, "Tablet"
    )

    # Sync to all devices (target_devices=None)
    operation = await orchestrator.sync_memory(
        memory_id="mem_456",
        from_device="laptop-001",
        target_devices=None,  # All devices
        priority=SyncPriority.NORMAL
    )

    # Should target mobile and tablet (all except source)
    assert len(operation.target_devices) == 2
    assert "mobile-001" in operation.target_devices
    assert "tablet-001" in operation.target_devices
    assert "laptop-001" not in operation.target_devices


@pytest.mark.asyncio
async def test_sync_priorities():
    """Test sync priority queue"""
    orchestrator = CrossDeviceSyncOrchestrator(user_id="test_user")

    await orchestrator.register_device(
        "laptop-001", DeviceType.LAPTOP, "Laptop"
    )
    await orchestrator.register_device(
        "mobile-001", DeviceType.MOBILE, "Mobile"
    )

    # Queue operations with different priorities
    await orchestrator.sync_memory(
        "mem_1", "laptop-001", priority=SyncPriority.LOW
    )
    await orchestrator.sync_memory(
        "mem_2", "laptop-001", priority=SyncPriority.CRITICAL
    )
    await orchestrator.sync_memory(
        "mem_3", "laptop-001", priority=SyncPriority.HIGH
    )

    # Check queue sizes
    assert len(orchestrator.sync_queue[SyncPriority.LOW]) >= 1
    # CRITICAL is processed immediately, so queue may be empty
    assert len(orchestrator.sync_queue[SyncPriority.HIGH]) >= 1


@pytest.mark.asyncio
async def test_get_device_status():
    """Test getting device status"""
    orchestrator = CrossDeviceSyncOrchestrator(user_id="test_user")

    await orchestrator.register_device(
        "laptop-001",
        DeviceType.LAPTOP,
        "MacBook Pro",
        capabilities={"storage_gb": 100}
    )

    status = await orchestrator.get_device_status("laptop-001")

    assert status["device_id"] == "laptop-001"
    assert status["device_name"] == "MacBook Pro"
    assert status["device_type"] == DeviceType.LAPTOP
    assert status["status"] == DeviceStatus.ONLINE
    assert "last_seen" in status
    assert "storage_used_gb" in status
    assert "storage_quota_gb" in status


@pytest.mark.asyncio
async def test_get_sync_statistics():
    """Test sync statistics"""
    orchestrator = CrossDeviceSyncOrchestrator(user_id="test_user")

    await orchestrator.register_device(
        "laptop-001", DeviceType.LAPTOP, "Laptop"
    )
    await orchestrator.register_device(
        "mobile-001", DeviceType.MOBILE, "Mobile"
    )

    # Perform a sync
    operation = await orchestrator.sync_memory(
        "mem_123", "laptop-001", priority=SyncPriority.CRITICAL
    )

    stats = await orchestrator.get_sync_statistics()

    assert stats["total_devices"] == 2
    assert stats["online_devices"] == 2
    assert stats["offline_devices"] == 0
    assert "total_syncs" in stats
    assert "success_rate" in stats


@pytest.mark.asyncio
async def test_conflict_resolution_last_write_wins():
    """Test conflict resolution with last-write-wins strategy"""
    orchestrator = CrossDeviceSyncOrchestrator(user_id="test_user")
    orchestrator.conflict_strategy = "last_write_wins"

    device_versions = {
        "laptop-001": {
            "content": "Version from laptop",
            "updated_at": "2025-01-01T10:00:00"
        },
        "mobile-001": {
            "content": "Version from mobile",
            "updated_at": "2025-01-01T12:00:00"  # More recent
        }
    }

    resolution = await orchestrator.resolve_conflict(
        resource_id="mem_conflict",
        device_versions=device_versions
    )

    assert resolution["resolution_strategy"] == "last_write_wins"
    assert resolution["winning_device"] == "mobile-001"
    assert "resolved_at" in resolution


@pytest.mark.asyncio
async def test_conflict_resolution_manual():
    """Test conflict resolution with manual strategy"""
    orchestrator = CrossDeviceSyncOrchestrator(user_id="test_user")
    orchestrator.conflict_strategy = "manual"

    device_versions = {
        "laptop-001": {"content": "Version 1"},
        "mobile-001": {"content": "Version 2"}
    }

    resolution = await orchestrator.resolve_conflict(
        "mem_conflict", device_versions
    )

    assert resolution["resolution_strategy"] == "manual"
    assert resolution["requires_user_action"] is True
    assert "versions" in resolution


@pytest.mark.asyncio
async def test_device_offline_timeout():
    """Test device offline timeout detection"""
    orchestrator = CrossDeviceSyncOrchestrator(user_id="test_user")

    device = await orchestrator.register_device(
        "laptop-001", DeviceType.LAPTOP, "Laptop"
    )

    # Manually set last_seen to old time
    device.last_seen = datetime.utcnow() - timedelta(minutes=15)

    # Update device statuses
    await orchestrator._update_device_statuses()

    # Device should be marked offline
    assert orchestrator.devices["laptop-001"].status == DeviceStatus.OFFLINE
