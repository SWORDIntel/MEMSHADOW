"""
Hardware Device Detection
Detects available acceleration hardware (NPU, GPU, NCS2, AVX-512)
"""

import subprocess
import platform
from typing import Dict, List, Any
import structlog

logger = structlog.get_logger()


class HardwareDetector:
    """
    Detect and enumerate available hardware acceleration devices
    """

    def __init__(self):
        self.detected_devices = {}
        self._detect_all()

    def _detect_all(self):
        """Detect all available hardware"""
        self.detected_devices = {
            "cpu": self._detect_cpu(),
            "avx512": self._detect_avx512(),
            "gpu": self._detect_gpu(),
            "npu": self._detect_npu(),
            "ncs2": self._detect_ncs2()
        }

        logger.info("Hardware detection complete", devices=self.detected_devices)

    def _detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU information"""
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()

                # Extract CPU model
                for line in cpuinfo.split("\n"):
                    if "model name" in line:
                        model = line.split(":")[1].strip()
                        break
                else:
                    model = "Unknown"

                # Count cores
                cores = cpuinfo.count("processor")

                return {
                    "available": True,
                    "model": model,
                    "cores": cores,
                    "architecture": platform.machine()
                }
            else:
                return {
                    "available": True,
                    "model": platform.processor(),
                    "cores": "unknown",
                    "architecture": platform.machine()
                }

        except Exception as e:
            logger.error("CPU detection error", error=str(e))
            return {"available": False}

    def _detect_avx512(self) -> Dict[str, Any]:
        """Detect AVX-512 support"""
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()

                has_avx512 = "avx512" in cpuinfo.lower()

                return {
                    "available": has_avx512,
                    "instructions": ["AVX512F", "AVX512DQ", "AVX512BW"] if has_avx512 else []
                }
            else:
                return {"available": False}

        except Exception as e:
            logger.error("AVX-512 detection error", error=str(e))
            return {"available": False}

    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU (Intel ARC, NVIDIA, AMD)"""
        try:
            # Try Intel
            result = subprocess.run(
                ["lspci"],
                capture_output=True,
                text=True,
                timeout=5
            )

            gpu_lines = [line for line in result.stdout.split("\n") if "VGA" in line or "3D" in line]

            if gpu_lines:
                gpu_info = gpu_lines[0]

                return {
                    "available": True,
                    "device": gpu_info,
                    "vendor": self._extract_vendor(gpu_info)
                }

            return {"available": False}

        except Exception as e:
            logger.error("GPU detection error", error=str(e))
            return {"available": False}

    def _detect_npu(self) -> Dict[str, Any]:
        """Detect Intel NPU (AI Boost)"""
        try:
            # Check for Intel NPU device
            result = subprocess.run(
                ["lspci", "-nn"],
                capture_output=True,
                text=True,
                timeout=5
            )

            # Intel NPU typically shows as "Processing accelerators"
            npu_lines = [line for line in result.stdout.split("\n")
                        if "processing accelerators" in line.lower() and "intel" in line.lower()]

            if npu_lines:
                return {
                    "available": True,
                    "device": npu_lines[0],
                    "type": "Intel AI Boost NPU"
                }

            return {"available": False}

        except Exception as e:
            logger.error("NPU detection error", error=str(e))
            return {"available": False}

    def _detect_ncs2(self) -> Dict[str, Any]:
        """Detect Intel Neural Compute Stick 2"""
        try:
            # Check USB devices
            result = subprocess.run(
                ["lsusb"],
                capture_output=True,
                text=True,
                timeout=5
            )

            # NCS2 shows as "Movidius MyriadX"
            ncs2_lines = [line for line in result.stdout.split("\n")
                         if "movidius" in line.lower() or "myriad" in line.lower()]

            if ncs2_lines:
                return {
                    "available": True,
                    "device": ncs2_lines[0],
                    "type": "Intel NCS2"
                }

            return {"available": False}

        except Exception as e:
            logger.error("NCS2 detection error", error=str(e))
            return {"available": False}

    def _extract_vendor(self, gpu_info: str) -> str:
        """Extract GPU vendor from lspci output"""
        if "intel" in gpu_info.lower():
            return "Intel"
        elif "nvidia" in gpu_info.lower():
            return "NVIDIA"
        elif "amd" in gpu_info.lower():
            return "AMD"
        else:
            return "Unknown"

    def get_available_devices(self) -> List[str]:
        """Get list of available acceleration devices"""
        available = []

        for device_name, device_info in self.detected_devices.items():
            if device_info.get("available", False):
                available.append(device_name)

        return available

    def get_fastest_device(self) -> str:
        """Get fastest available device for computation"""
        # Priority order: AVX-512 > GPU > NPU > NCS2 > CPU
        priority = ["avx512", "gpu", "npu", "ncs2", "cpu"]

        for device in priority:
            if self.detected_devices.get(device, {}).get("available", False):
                return device

        return "cpu"

    def get_device_info(self, device_name: str) -> Dict[str, Any]:
        """Get detailed information about a device"""
        return self.detected_devices.get(device_name, {"available": False})

    def print_summary(self):
        """Print hardware detection summary"""
        print("\n" + "=" * 60)
        print("HARDWARE DETECTION SUMMARY")
        print("=" * 60)

        for device_name, device_info in self.detected_devices.items():
            status = "✓ AVAILABLE" if device_info.get("available") else "✗ NOT FOUND"
            print(f"\n{device_name.upper()}: {status}")

            if device_info.get("available"):
                for key, value in device_info.items():
                    if key != "available":
                        print(f"  {key}: {value}")

        print("\n" + "=" * 60)
        print(f"Recommended device: {self.get_fastest_device().upper()}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test detection
    detector = HardwareDetector()
    detector.print_summary()
