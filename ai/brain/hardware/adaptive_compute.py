#!/usr/bin/env python3
"""
Adaptive Compute for DSMIL Brain

Hardware detection and capability assessment:
- CPU/GPU/TPU detection
- Memory capacity detection
- Network bandwidth assessment
- Dynamic capability updates
"""

import os
import platform
import subprocess
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# DSMIL Accelerator Integration
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../hardware'))
    from dsmil_accelerator_interface import get_accelerator_interface
    HAS_DSMIL_ACCEL = True
except ImportError:
    HAS_DSMIL_ACCEL = False


@dataclass
class ComputeResource:
    """A compute resource"""
    resource_type: str  # "cpu", "gpu", "tpu", "npu"
    name: str

    # Capacity
    cores: int = 0
    memory_mb: int = 0
    compute_units: int = 0

    # Utilization
    current_load: float = 0.0
    memory_used_mb: int = 0

    # Availability
    available: bool = True


@dataclass
class HardwareCapabilities:
    """Hardware capabilities of a node"""
    node_id: str

    # System
    platform: str = ""
    architecture: str = ""

    # CPU
    cpu_cores: int = 0
    cpu_threads: int = 0
    cpu_freq_mhz: float = 0.0

    # Memory
    ram_total_mb: int = 0
    ram_available_mb: int = 0

    # GPU (CUDA/Discrete)
    gpu_available: bool = False
    gpu_count: int = 0
    gpu_memory_mb: int = 0

    # Intel XPU (Arc Graphics)
    xpu_available: bool = False
    xpu_tops: float = 0.0
    xpu_xe_cores: int = 0

    # Intel NPU (AI Boost)
    npu_available: bool = False
    npu_tops: float = 0.0

    # Total AI Compute
    total_tops: float = 0.0

    # Storage
    storage_total_gb: float = 0.0
    storage_available_gb: float = 0.0

    # Network
    network_bandwidth_mbps: float = 0.0

    # TPM
    tpm_available: bool = False
    tpm_version: str = ""

    # DSMIL Kernel Driver
    dsmil_driver_loaded: bool = False

    # Compute resources
    resources: List[ComputeResource] = field(default_factory=list)

    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AdaptiveCompute:
    """
    Adaptive Compute System

    Detects and monitors hardware capabilities.

    Usage:
        compute = AdaptiveCompute()

        # Detect hardware
        capabilities = compute.detect_hardware()

        # Monitor utilization
        compute.start_monitoring()

        # Get current state
        state = compute.get_current_state()
    """

    def __init__(self, node_id: str = ""):
        self.node_id = node_id or platform.node()

        self._capabilities: Optional[HardwareCapabilities] = None
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

        self._lock = threading.RLock()

        logger.info(f"AdaptiveCompute initialized (node={self.node_id})")

    def detect_hardware(self) -> HardwareCapabilities:
        """Detect hardware capabilities"""
        with self._lock:
            capabilities = HardwareCapabilities(node_id=self.node_id)

            # Platform
            capabilities.platform = platform.system()
            capabilities.architecture = platform.machine()

            # CPU
            capabilities.cpu_cores = os.cpu_count() or 1
            capabilities.cpu_threads = capabilities.cpu_cores  # Simplified

            # RAM
            try:
                import psutil
                mem = psutil.virtual_memory()
                capabilities.ram_total_mb = mem.total // (1024 * 1024)
                capabilities.ram_available_mb = mem.available // (1024 * 1024)
            except ImportError:
                # Fallback
                capabilities.ram_total_mb = 8192  # Assume 8GB
                capabilities.ram_available_mb = 4096

            # GPU detection (CUDA)
            capabilities.gpu_available = self._detect_gpu()
            if capabilities.gpu_available:
                capabilities.gpu_count = 1
                capabilities.gpu_memory_mb = 8192  # Assume

            # Intel XPU (Arc Graphics) detection
            xpu_available, xpu_tops, xe_cores = self._detect_intel_xpu()
            capabilities.xpu_available = xpu_available
            capabilities.xpu_tops = xpu_tops
            capabilities.xpu_xe_cores = xe_cores

            # Intel NPU (AI Boost) detection
            npu_available, npu_tops = self._detect_intel_npu()
            capabilities.npu_available = npu_available
            capabilities.npu_tops = npu_tops

            # Calculate total TOPS
            capabilities.total_tops = 0.0
            if capabilities.xpu_available:
                capabilities.total_tops += capabilities.xpu_tops
            if capabilities.npu_available:
                capabilities.total_tops += capabilities.npu_tops

            # DSMIL kernel driver
            capabilities.dsmil_driver_loaded = self._detect_dsmil_driver()

            # Storage
            try:
                import shutil
                total, used, free = shutil.disk_usage("/")
                capabilities.storage_total_gb = total / (1024 ** 3)
                capabilities.storage_available_gb = free / (1024 ** 3)
            except:
                capabilities.storage_total_gb = 100
                capabilities.storage_available_gb = 50

            # TPM detection
            capabilities.tpm_available = self._detect_tpm()

            # Build resource list
            capabilities.resources = [
                ComputeResource(
                    resource_type="cpu",
                    name=platform.processor() or "CPU",
                    cores=capabilities.cpu_cores,
                    memory_mb=capabilities.ram_total_mb,
                )
            ]

            if capabilities.gpu_available:
                capabilities.resources.append(
                    ComputeResource(
                        resource_type="gpu",
                        name="CUDA GPU",
                        memory_mb=capabilities.gpu_memory_mb,
                    )
                )

            if capabilities.xpu_available:
                capabilities.resources.append(
                    ComputeResource(
                        resource_type="xpu",
                        name=f"Intel Arc Graphics ({capabilities.xpu_tops:.0f} TOPS)",
                        cores=capabilities.xpu_xe_cores,
                        compute_units=int(capabilities.xpu_tops),
                        memory_mb=2048,  # Shared memory
                    )
                )

            if capabilities.npu_available:
                capabilities.resources.append(
                    ComputeResource(
                        resource_type="npu",
                        name=f"Intel NPU ({capabilities.npu_tops:.0f} TOPS)",
                        compute_units=int(capabilities.npu_tops),
                        memory_mb=16,  # On-die memory
                    )
                )

            self._capabilities = capabilities
            return capabilities

    def _detect_gpu(self) -> bool:
        """Detect CUDA GPU availability"""
        # Try CUDA
        try:
            import torch
            return torch.cuda.is_available()
        except:
            pass

        # Try checking for nvidia-smi
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            return result.returncode == 0
        except:
            pass

        return False

    def _detect_intel_xpu(self) -> tuple:
        """
        Detect Intel XPU (Arc Graphics) availability.

        Returns:
            Tuple of (available, tops, xe_cores)
        """
        # Check DSMIL accelerator interface first
        if HAS_DSMIL_ACCEL:
            try:
                accel_if = get_accelerator_interface()
                status = accel_if.get_arc_status()
                if status.get("available", False):
                    return True, status.get("tops", 40), status.get("xe_cores", 8)
            except Exception as e:
                logger.debug(f"DSMIL XPU check: {e}")

        # Check via Intel Extension for PyTorch
        try:
            import intel_extension_for_pytorch as ipex
            if hasattr(ipex, 'xpu') and ipex.xpu.is_available():
                return True, 40.0, 8  # Meteor Lake defaults
        except ImportError:
            pass

        # Check for Intel i915/xe driver via sysfs
        try:
            if os.path.exists("/sys/class/drm"):
                for card in os.listdir("/sys/class/drm"):
                    if card.startswith("card"):
                        driver_path = f"/sys/class/drm/{card}/device/driver"
                        if os.path.exists(driver_path):
                            driver = os.path.basename(os.readlink(driver_path))
                            if driver in ("i915", "xe"):
                                return True, 40.0, 8
        except Exception:
            pass

        # Check via lspci for Intel Arc
        try:
            result = subprocess.run(
                ["lspci", "-n", "-d", "8086:7d55"],
                capture_output=True, text=True
            )
            if result.stdout.strip():
                return True, 40.0, 8
        except Exception:
            pass

        return False, 0.0, 0

    def _detect_intel_npu(self) -> tuple:
        """
        Detect Intel NPU (AI Boost) availability.

        Returns:
            Tuple of (available, tops)
        """
        # Check DSMIL accelerator interface first
        if HAS_DSMIL_ACCEL:
            try:
                accel_if = get_accelerator_interface()
                status = accel_if.get_npu_status()
                if status.get("available", False):
                    return True, status.get("tops", 30)
            except Exception as e:
                logger.debug(f"DSMIL NPU check: {e}")

        # Check for NPU device
        npu_paths = [
            "/dev/dsmil_npu",
            "/dev/accel/accel0",  # Linux accel subsystem
            "/sys/class/accel/accel0",
        ]

        for path in npu_paths:
            if os.path.exists(path):
                return True, 30.0  # Meteor Lake NPU default

        # Check via lspci for Intel NPU
        try:
            result = subprocess.run(
                ["lspci", "-n", "-d", "8086:7d1d"],
                capture_output=True, text=True
            )
            if result.stdout.strip():
                return True, 30.0
        except Exception:
            pass

        return False, 0.0

    def _detect_tpm(self) -> bool:
        """Detect TPM availability"""
        # Linux TPM
        if os.path.exists("/dev/tpm0"):
            return True

        # TPM resource manager
        if os.path.exists("/dev/tpmrm0"):
            return True

        # Windows would check via WMI

        return False

    def _detect_dsmil_driver(self) -> bool:
        """Detect if DSMIL kernel driver is loaded."""
        # Check for device
        if os.path.exists("/dev/dsmil0"):
            return True

        # Check loaded modules
        try:
            with open("/proc/modules", "r") as f:
                modules = f.read()
                if "dsmil" in modules:
                    return True
        except Exception:
            pass

        return False

    def get_current_state(self) -> Dict:
        """Get current hardware state"""
        with self._lock:
            if not self._capabilities:
                self.detect_hardware()

            state = {
                "node_id": self.node_id,
                "cpu_cores": self._capabilities.cpu_cores,
                "ram_total_mb": self._capabilities.ram_total_mb,
                "ram_available_mb": self._capabilities.ram_available_mb,
                "gpu_available": self._capabilities.gpu_available,
                "tpm_available": self._capabilities.tpm_available,
            }

            # Update utilization
            try:
                import psutil
                state["cpu_load"] = psutil.cpu_percent()
                mem = psutil.virtual_memory()
                state["ram_available_mb"] = mem.available // (1024 * 1024)
            except:
                pass

            return state

    def get_recommended_workers(self) -> int:
        """Get recommended number of parallel workers"""
        with self._lock:
            if not self._capabilities:
                self.detect_hardware()

            # Base on CPU cores and available RAM
            cpu_workers = max(1, self._capabilities.cpu_cores - 1)
            ram_workers = max(1, self._capabilities.ram_available_mb // 512)  # 512MB per worker

            return min(cpu_workers, ram_workers)

    def get_optimal_batch_size(self, item_size_mb: float = 1.0) -> int:
        """Get optimal batch size for processing"""
        with self._lock:
            if not self._capabilities:
                self.detect_hardware()

            available_mb = self._capabilities.ram_available_mb * 0.5  # Use 50% of available
            return max(1, int(available_mb / item_size_mb))

    def get_stats(self) -> Dict:
        """Get hardware statistics"""
        with self._lock:
            if not self._capabilities:
                return {"status": "not_detected"}

            return {
                "platform": self._capabilities.platform,
                "architecture": self._capabilities.architecture,
                "cpu_cores": self._capabilities.cpu_cores,
                "ram_total_mb": self._capabilities.ram_total_mb,
                "gpu_available": self._capabilities.gpu_available,
                "gpu_count": self._capabilities.gpu_count,
                "xpu_available": self._capabilities.xpu_available,
                "xpu_tops": self._capabilities.xpu_tops,
                "npu_available": self._capabilities.npu_available,
                "npu_tops": self._capabilities.npu_tops,
                "total_tops": self._capabilities.total_tops,
                "tpm_available": self._capabilities.tpm_available,
                "dsmil_driver": self._capabilities.dsmil_driver_loaded,
                "resources": len(self._capabilities.resources),
            }


if __name__ == "__main__":
    print("=" * 60)
    print("  Adaptive Compute Self-Test (Intel Accelerator Support)")
    print("=" * 60)

    compute = AdaptiveCompute()

    print("\n[1] Detect Hardware")
    capabilities = compute.detect_hardware()
    print(f"    Platform: {capabilities.platform}")
    print(f"    Architecture: {capabilities.architecture}")
    print(f"    CPU Cores: {capabilities.cpu_cores}")
    print(f"    RAM Total: {capabilities.ram_total_mb} MB")
    print(f"    RAM Available: {capabilities.ram_available_mb} MB")
    print(f"    CUDA GPU Available: {capabilities.gpu_available}")
    print(f"    TPM Available: {capabilities.tpm_available}")
    print(f"    DSMIL Driver: {capabilities.dsmil_driver_loaded}")

    print("\n[2] Intel Accelerators")
    print(f"    XPU (Arc GPU): {capabilities.xpu_available}")
    if capabilities.xpu_available:
        print(f"      TOPS: {capabilities.xpu_tops}")
        print(f"      Xe-Cores: {capabilities.xpu_xe_cores}")
    print(f"    NPU (AI Boost): {capabilities.npu_available}")
    if capabilities.npu_available:
        print(f"      TOPS: {capabilities.npu_tops}")
    print(f"    Total TOPS: {capabilities.total_tops}")

    print("\n[3] Compute Resources")
    for res in capabilities.resources:
        info = f"{res.resource_type}: {res.name}"
        if res.compute_units:
            info += f" ({res.compute_units} compute units)"
        print(f"    {info}")

    print("\n[4] Current State")
    state = compute.get_current_state()
    for k, v in state.items():
        print(f"    {k}: {v}")

    print("\n[5] Recommendations")
    print(f"    Workers: {compute.get_recommended_workers()}")
    print(f"    Batch size (1MB items): {compute.get_optimal_batch_size(1.0)}")

    print("\n[6] Statistics")
    stats = compute.get_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.1f}")
        else:
            print(f"    {k}: {v}")

    print("\n" + "=" * 60)
    print("Adaptive Compute test complete")

