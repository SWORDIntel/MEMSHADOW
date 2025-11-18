"""
MEMSHADOW Distributed Architecture Services
Phase 4: NPU Acceleration, Cross-Device Sync, Edge Deployment

This package provides services for distributed memory management:
- NPU/GPU/CPU hardware acceleration for local inference
- Cross-device synchronization orchestration
- Edge deployment for resource-constrained devices
"""

from app.services.distributed.npu_accelerator import (
    NPUAccelerator,
    AcceleratorType,
    QuantizationType,
    npu_accelerator
)

from app.services.distributed.cross_device_sync import (
    CrossDeviceSyncOrchestrator,
    Device,
    DeviceType,
    DeviceStatus,
    SyncPriority,
    SyncOperation
)

from app.services.distributed.edge_deployment import (
    EdgeDeploymentService,
    EdgeProfile,
    ComputeMode,
    EdgeConfiguration
)

__all__ = [
    # NPU Accelerator
    "NPUAccelerator",
    "AcceleratorType",
    "QuantizationType",
    "npu_accelerator",

    # Cross-Device Sync
    "CrossDeviceSyncOrchestrator",
    "Device",
    "DeviceType",
    "DeviceStatus",
    "SyncPriority",
    "SyncOperation",

    # Edge Deployment
    "EdgeDeploymentService",
    "EdgeProfile",
    "ComputeMode",
    "EdgeConfiguration",
]
