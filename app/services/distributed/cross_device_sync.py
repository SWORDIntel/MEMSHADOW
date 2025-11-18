"""
Cross-Device Synchronization Orchestrator
Phase 4: Distributed Architecture - Multi-device memory synchronization

Coordinates memory sync across:
- Laptops
- Mobile devices
- Tablets
- Edge devices
- Cloud backend
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import uuid
import structlog
from dataclasses import dataclass, field
import asyncio

logger = structlog.get_logger()


class DeviceType(str, Enum):
    """Device types"""
    LAPTOP = "laptop"
    DESKTOP = "desktop"
    MOBILE = "mobile"
    TABLET = "tablet"
    EDGE = "edge"
    CLOUD = "cloud"


class DeviceStatus(str, Enum):
    """Device connection status"""
    ONLINE = "online"
    OFFLINE = "offline"
    SYNCING = "syncing"
    ERROR = "error"


class SyncPriority(str, Enum):
    """Sync priority levels"""
    CRITICAL = "critical"  # Immediate sync
    HIGH = "high"         # Within 1 minute
    NORMAL = "normal"     # Within 5 minutes
    LOW = "low"          # Within 30 minutes


@dataclass
class Device:
    """Device information"""
    device_id: str
    device_type: DeviceType
    device_name: str
    status: DeviceStatus = DeviceStatus.OFFLINE
    last_seen: Optional[datetime] = None
    last_sync: Optional[datetime] = None
    sync_version: int = 0
    capabilities: Dict[str, Any] = field(default_factory=dict)
    storage_quota_gb: float = 10.0
    storage_used_gb: float = 0.0


@dataclass
class SyncOperation:
    """Sync operation metadata"""
    operation_id: str
    source_device: str
    target_devices: List[str]
    priority: SyncPriority
    resource_type: str
    resource_id: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "pending"
    error: Optional[str] = None


class CrossDeviceSyncOrchestrator:
    """
    Orchestrates memory synchronization across multiple devices.

    Features:
    - Device discovery and registration
    - Priority-based sync scheduling
    - Conflict resolution across devices
    - Bandwidth optimization
    - Offline support
    - Device capability awareness

    Example:
        orchestrator = CrossDeviceSyncOrchestrator()
        await orchestrator.register_device(
            device_id="laptop-001",
            device_type=DeviceType.LAPTOP,
            device_name="MacBook Pro"
        )
        await orchestrator.sync_memory(
            memory_id="mem_123",
            from_device="laptop-001",
            priority=SyncPriority.HIGH
        )
    """

    def __init__(
        self,
        user_id: str,
        max_devices: int = 10,
        sync_interval_seconds: int = 300,  # 5 minutes
        cloud_endpoint: str = "https://api.memshadow.cloud"
    ):
        self.user_id = user_id
        self.max_devices = max_devices
        self.sync_interval_seconds = sync_interval_seconds
        self.cloud_endpoint = cloud_endpoint

        # Device registry
        self.devices: Dict[str, Device] = {}

        # Sync queue (priority-based)
        self.sync_queue: Dict[SyncPriority, List[SyncOperation]] = {
            SyncPriority.CRITICAL: [],
            SyncPriority.HIGH: [],
            SyncPriority.NORMAL: [],
            SyncPriority.LOW: []
        }

        # Sync history
        self.sync_history: List[SyncOperation] = []

        # Active sync operations
        self.active_syncs: Set[str] = set()

        # Conflict resolution strategy
        self.conflict_strategy = "last_write_wins"  # Options: last_write_wins, manual, merge

        logger.info(
            "Cross-device sync orchestrator initialized",
            user_id=user_id,
            max_devices=max_devices,
            sync_interval=sync_interval_seconds
        )

    async def register_device(
        self,
        device_id: str,
        device_type: DeviceType,
        device_name: str,
        capabilities: Optional[Dict[str, Any]] = None
    ) -> Device:
        """
        Register a new device for synchronization.

        Args:
            device_id: Unique device identifier
            device_type: Type of device
            device_name: Human-readable device name
            capabilities: Device capabilities (storage, compute, etc.)

        Returns:
            Registered device object
        """
        if len(self.devices) >= self.max_devices:
            raise ValueError(f"Maximum device limit ({self.max_devices}) reached")

        if device_id in self.devices:
            logger.info("Device already registered, updating", device_id=device_id)
            device = self.devices[device_id]
            device.status = DeviceStatus.ONLINE
            device.last_seen = datetime.utcnow()
            return device

        device = Device(
            device_id=device_id,
            device_type=device_type,
            device_name=device_name,
            status=DeviceStatus.ONLINE,
            last_seen=datetime.utcnow(),
            capabilities=capabilities or {}
        )

        self.devices[device_id] = device

        logger.info(
            "Device registered",
            device_id=device_id,
            device_type=device_type,
            device_name=device_name,
            total_devices=len(self.devices)
        )

        # Trigger initial sync for new device
        await self._trigger_initial_sync(device_id)

        return device

    async def unregister_device(self, device_id: str):
        """Unregister a device"""
        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not found")

        device = self.devices[device_id]
        device.status = DeviceStatus.OFFLINE

        logger.info("Device unregistered", device_id=device_id)

        # Optionally remove from registry after grace period
        # For now, just mark as offline

    async def sync_memory(
        self,
        memory_id: str,
        from_device: str,
        target_devices: Optional[List[str]] = None,
        priority: SyncPriority = SyncPriority.NORMAL
    ) -> SyncOperation:
        """
        Sync a memory across devices.

        Args:
            memory_id: Memory to sync
            from_device: Source device
            target_devices: Target devices (None = all devices)
            priority: Sync priority

        Returns:
            Sync operation
        """
        if from_device not in self.devices:
            raise ValueError(f"Source device {from_device} not registered")

        # Determine target devices
        if target_devices is None:
            # Sync to all online devices except source
            target_devices = [
                device_id for device_id, device in self.devices.items()
                if device.status == DeviceStatus.ONLINE and device_id != from_device
            ]

        # Create sync operation
        operation = SyncOperation(
            operation_id=str(uuid.uuid4()),
            source_device=from_device,
            target_devices=target_devices,
            priority=priority,
            resource_type="memory",
            resource_id=memory_id,
            created_at=datetime.utcnow()
        )

        # Add to queue
        self.sync_queue[priority].append(operation)

        logger.info(
            "Memory sync queued",
            memory_id=memory_id,
            from_device=from_device,
            target_devices=target_devices,
            priority=priority,
            operation_id=operation.operation_id
        )

        # Trigger immediate processing for critical priority
        if priority == SyncPriority.CRITICAL:
            await self._process_sync_operation(operation)

        return operation

    async def _process_sync_operation(self, operation: SyncOperation):
        """Process a single sync operation"""
        if operation.operation_id in self.active_syncs:
            logger.warning(
                "Sync operation already active",
                operation_id=operation.operation_id
            )
            return

        self.active_syncs.add(operation.operation_id)
        operation.status = "processing"

        logger.info(
            "Processing sync operation",
            operation_id=operation.operation_id,
            resource_type=operation.resource_type,
            resource_id=operation.resource_id,
            target_count=len(operation.target_devices)
        )

        try:
            # Update source device status
            if operation.source_device in self.devices:
                self.devices[operation.source_device].status = DeviceStatus.SYNCING

            # Sync to each target device
            for target_device_id in operation.target_devices:
                if target_device_id not in self.devices:
                    logger.warning(
                        "Target device not found",
                        device_id=target_device_id
                    )
                    continue

                target_device = self.devices[target_device_id]

                if target_device.status != DeviceStatus.ONLINE:
                    logger.info(
                        "Skipping offline device",
                        device_id=target_device_id,
                        status=target_device.status
                    )
                    continue

                # Perform actual sync
                await self._sync_to_device(
                    operation.resource_id,
                    target_device_id,
                    operation.resource_type
                )

            operation.status = "completed"
            operation.completed_at = datetime.utcnow()

            # Update source device
            if operation.source_device in self.devices:
                self.devices[operation.source_device].status = DeviceStatus.ONLINE
                self.devices[operation.source_device].last_sync = datetime.utcnow()
                self.devices[operation.source_device].sync_version += 1

            logger.info(
                "Sync operation completed",
                operation_id=operation.operation_id,
                duration_ms=(
                    operation.completed_at - operation.created_at
                ).total_seconds() * 1000
            )

        except Exception as e:
            operation.status = "error"
            operation.error = str(e)
            logger.error(
                "Sync operation failed",
                operation_id=operation.operation_id,
                error=str(e),
                exc_info=True
            )

        finally:
            self.active_syncs.remove(operation.operation_id)
            self.sync_history.append(operation)

    async def _sync_to_device(
        self,
        resource_id: str,
        target_device_id: str,
        resource_type: str
    ):
        """Sync resource to specific device"""
        target_device = self.devices[target_device_id]
        target_device.status = DeviceStatus.SYNCING

        logger.debug(
            "Syncing to device",
            resource_id=resource_id,
            device_id=target_device_id,
            resource_type=resource_type
        )

        # In production, would make actual API call to device or cloud
        # Example:
        # if target_device.device_type == DeviceType.CLOUD:
        #     await self._sync_to_cloud(resource_id)
        # else:
        #     await self._sync_to_local_device(resource_id, target_device_id)

        # Simulate sync delay
        await asyncio.sleep(0.1)

        # Update target device
        target_device.status = DeviceStatus.ONLINE
        target_device.last_sync = datetime.utcnow()
        target_device.sync_version += 1

        logger.debug("Sync to device completed", device_id=target_device_id)

    async def _trigger_initial_sync(self, device_id: str):
        """Trigger initial full sync for newly registered device"""
        logger.info("Triggering initial sync for device", device_id=device_id)

        # In production, would fetch all user memories and sync to device
        # For now, just log the event

        device = self.devices[device_id]
        device.last_sync = datetime.utcnow()

    async def process_sync_queue(self):
        """Process pending sync operations from queue"""
        logger.debug("Processing sync queue")

        # Process in priority order
        for priority in [
            SyncPriority.CRITICAL,
            SyncPriority.HIGH,
            SyncPriority.NORMAL,
            SyncPriority.LOW
        ]:
            queue = self.sync_queue[priority]

            while queue:
                operation = queue.pop(0)
                await self._process_sync_operation(operation)

    async def start_sync_loop(self):
        """Main sync loop"""
        logger.info("Starting sync loop", interval=self.sync_interval_seconds)

        while True:
            try:
                # Update device statuses
                await self._update_device_statuses()

                # Process sync queue
                await self.process_sync_queue()

                # Wait for next interval
                await asyncio.sleep(self.sync_interval_seconds)

            except Exception as e:
                logger.error("Sync loop error", error=str(e), exc_info=True)
                await asyncio.sleep(60)  # Wait before retry

    async def _update_device_statuses(self):
        """Update device online/offline statuses"""
        timeout_threshold = datetime.utcnow() - timedelta(minutes=10)

        for device_id, device in self.devices.items():
            if device.last_seen and device.last_seen < timeout_threshold:
                if device.status == DeviceStatus.ONLINE:
                    device.status = DeviceStatus.OFFLINE
                    logger.info("Device marked offline (timeout)", device_id=device_id)

    async def get_device_status(self, device_id: str) -> Dict[str, Any]:
        """Get detailed device status"""
        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not found")

        device = self.devices[device_id]

        return {
            "device_id": device.device_id,
            "device_name": device.device_name,
            "device_type": device.device_type,
            "status": device.status,
            "last_seen": device.last_seen.isoformat() if device.last_seen else None,
            "last_sync": device.last_sync.isoformat() if device.last_sync else None,
            "sync_version": device.sync_version,
            "storage_used_gb": device.storage_used_gb,
            "storage_quota_gb": device.storage_quota_gb,
            "storage_percent_used": (
                device.storage_used_gb / device.storage_quota_gb * 100
                if device.storage_quota_gb > 0 else 0
            ),
            "capabilities": device.capabilities
        }

    async def get_sync_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics"""
        total_syncs = len(self.sync_history)
        successful_syncs = sum(
            1 for op in self.sync_history if op.status == "completed"
        )
        failed_syncs = sum(
            1 for op in self.sync_history if op.status == "error"
        )

        # Calculate average sync time
        completed_ops = [
            op for op in self.sync_history
            if op.status == "completed" and op.completed_at
        ]

        avg_sync_time_ms = 0
        if completed_ops:
            total_time = sum(
                (op.completed_at - op.created_at).total_seconds() * 1000
                for op in completed_ops
            )
            avg_sync_time_ms = total_time / len(completed_ops)

        return {
            "total_devices": len(self.devices),
            "online_devices": sum(
                1 for d in self.devices.values() if d.status == DeviceStatus.ONLINE
            ),
            "offline_devices": sum(
                1 for d in self.devices.values() if d.status == DeviceStatus.OFFLINE
            ),
            "total_syncs": total_syncs,
            "successful_syncs": successful_syncs,
            "failed_syncs": failed_syncs,
            "success_rate": (
                successful_syncs / total_syncs * 100 if total_syncs > 0 else 0
            ),
            "avg_sync_time_ms": avg_sync_time_ms,
            "pending_syncs": sum(len(q) for q in self.sync_queue.values()),
            "active_syncs": len(self.active_syncs)
        }

    async def resolve_conflict(
        self,
        resource_id: str,
        device_versions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve sync conflict between device versions.

        Args:
            resource_id: Resource with conflict
            device_versions: Dict of device_id -> version data

        Returns:
            Resolved version
        """
        logger.info(
            "Resolving sync conflict",
            resource_id=resource_id,
            num_versions=len(device_versions),
            strategy=self.conflict_strategy
        )

        if self.conflict_strategy == "last_write_wins":
            # Find most recent version
            latest_device = max(
                device_versions.items(),
                key=lambda x: x[1].get("updated_at", "1970-01-01")
            )

            logger.info(
                "Conflict resolved (last_write_wins)",
                resource_id=resource_id,
                winning_device=latest_device[0]
            )

            return {
                "resolution_strategy": "last_write_wins",
                "winning_device": latest_device[0],
                "winning_version": latest_device[1],
                "resolved_at": datetime.utcnow().isoformat()
            }

        elif self.conflict_strategy == "manual":
            # Queue for manual resolution
            logger.info(
                "Conflict queued for manual resolution",
                resource_id=resource_id
            )

            return {
                "resolution_strategy": "manual",
                "requires_user_action": True,
                "versions": device_versions
            }

        else:
            raise ValueError(f"Unknown conflict strategy: {self.conflict_strategy}")


# Example usage
async def example_usage():
    """Example of cross-device sync"""
    orchestrator = CrossDeviceSyncOrchestrator(user_id="user_123")

    # Register devices
    laptop = await orchestrator.register_device(
        device_id="laptop-001",
        device_type=DeviceType.LAPTOP,
        device_name="MacBook Pro",
        capabilities={"storage_gb": 100, "compute": "high"}
    )

    mobile = await orchestrator.register_device(
        device_id="mobile-001",
        device_type=DeviceType.MOBILE,
        device_name="iPhone",
        capabilities={"storage_gb": 20, "compute": "medium"}
    )

    # Sync a memory from laptop to all devices
    sync_op = await orchestrator.sync_memory(
        memory_id="mem_123",
        from_device="laptop-001",
        priority=SyncPriority.HIGH
    )

    # Get statistics
    stats = await orchestrator.get_sync_statistics()
    print(f"Sync stats: {stats}")
