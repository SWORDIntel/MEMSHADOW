"""
AVX-512 WiFi Cracker Wrapper
Python wrapper for native AVX-512 optimized WiFi cracking

Expects compiled C module at: app/services/hardware/native/avx512_wpa
Performance: 200,000-500,000 H/s on modern Intel P-cores
"""

import subprocess
import os
from pathlib import Path
from typing import Optional, Dict, Any
import structlog

logger = structlog.get_logger()


class AVX512Cracker:
    """
    Wrapper for AVX-512 optimized WiFi password cracking
    """

    def __init__(self):
        self.native_binary = Path(__file__).parent / "native" / "avx512_wpa"
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if AVX-512 cracker is available"""
        # Check if CPU supports AVX-512
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()

            has_avx512 = "avx512" in cpuinfo.lower()

            if not has_avx512:
                logger.warning("CPU does not support AVX-512")
                return False

            # Check if native binary exists
            if not self.native_binary.exists():
                logger.warning(
                    "AVX-512 native binary not found",
                    path=str(self.native_binary)
                )
                return False

            # Check if binary is executable
            if not os.access(self.native_binary, os.X_OK):
                logger.warning("AVX-512 binary not executable")
                return False

            logger.info("AVX-512 cracker available")
            return True

        except Exception as e:
            logger.error("AVX-512 availability check error", error=str(e))
            return False

    def crack_handshake(
        self,
        capture_file: str,
        wordlist_path: str,
        bssid: str,
        essid: str = ""
    ) -> Dict[str, Any]:
        """
        Crack WiFi handshake using AVX-512 acceleration

        Args:
            capture_file: Path to .cap file with handshake
            wordlist_path: Path to wordlist
            bssid: Target BSSID
            essid: Target ESSID (optional)

        Returns:
            Dict with cracking results
        """
        if not self.available:
            return {
                "status": "error",
                "message": "AVX-512 cracker not available"
            }

        logger.info(
            "Starting AVX-512 cracking",
            capture_file=capture_file,
            wordlist=wordlist_path,
            bssid=bssid
        )

        try:
            # Build command
            cmd = [
                str(self.native_binary),
                "--capture", capture_file,
                "--wordlist", wordlist_path,
                "--bssid", bssid
            ]

            if essid:
                cmd.extend(["--essid", essid])

            # Execute cracker
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            # Parse output
            output = result.stdout

            if "PASSWORD FOUND" in output:
                # Extract password from output
                for line in output.split("\n"):
                    if "PASSWORD:" in line:
                        password = line.split("PASSWORD:")[1].strip()
                        return {
                            "status": "success",
                            "password_found": True,
                            "password": password,
                            "bssid": bssid,
                            "essid": essid,
                            "method": "AVX-512"
                        }

            return {
                "status": "completed",
                "password_found": False,
                "message": "Password not found in wordlist",
                "bssid": bssid
            }

        except subprocess.TimeoutExpired:
            logger.warning("AVX-512 cracking timed out")
            return {
                "status": "timeout",
                "message": "Cracking operation timed out (1 hour)",
                "bssid": bssid
            }

        except Exception as e:
            logger.error("AVX-512 cracking error", error=str(e))
            return {
                "status": "error",
                "message": str(e),
                "bssid": bssid
            }

    def benchmark(self, iterations: int = 100000) -> Dict[str, Any]:
        """
        Benchmark AVX-512 cracking performance

        Args:
            iterations: Number of hash calculations

        Returns:
            Benchmark results
        """
        if not self.available:
            return {
                "status": "error",
                "message": "AVX-512 cracker not available"
            }

        logger.info("Running AVX-512 benchmark", iterations=iterations)

        try:
            cmd = [
                str(self.native_binary),
                "--benchmark",
                "--iterations", str(iterations)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            output = result.stdout

            # Parse benchmark results
            for line in output.split("\n"):
                if "H/s" in line or "hashes/sec" in line:
                    # Extract hash rate
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.replace(",", "").replace(".", "").isdigit():
                            hash_rate = float(part.replace(",", ""))
                            return {
                                "status": "success",
                                "hash_rate": hash_rate,
                                "unit": "H/s",
                                "method": "AVX-512",
                                "iterations": iterations
                            }

            return {
                "status": "completed",
                "message": "Benchmark completed but could not parse hash rate",
                "raw_output": output
            }

        except Exception as e:
            logger.error("Benchmark error", error=str(e))
            return {
                "status": "error",
                "message": str(e)
            }

    def compile_native(self) -> Dict[str, Any]:
        """
        Compile native AVX-512 binary from source

        Returns:
            Compilation results
        """
        source_dir = Path(__file__).parent / "native"
        makefile = source_dir / "Makefile"

        if not makefile.exists():
            return {
                "status": "error",
                "message": "Makefile not found",
                "path": str(makefile)
            }

        logger.info("Compiling AVX-512 native binary")

        try:
            # Run make
            result = subprocess.run(
                ["make", "-C", str(source_dir)],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                # Check if binary was created
                if self.native_binary.exists():
                    # Make executable
                    os.chmod(self.native_binary, 0o755)

                    return {
                        "status": "success",
                        "message": "AVX-512 binary compiled successfully",
                        "binary_path": str(self.native_binary)
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Compilation succeeded but binary not found"
                    }
            else:
                return {
                    "status": "error",
                    "message": "Compilation failed",
                    "stderr": result.stderr
                }

        except Exception as e:
            logger.error("Compilation error", error=str(e))
            return {
                "status": "error",
                "message": str(e)
            }


if __name__ == "__main__":
    # Test AVX-512 cracker
    cracker = AVX512Cracker()

    print("\n" + "=" * 60)
    print("AVX-512 WiFi Cracker")
    print("=" * 60)
    print(f"\nAvailable: {cracker.available}")

    if cracker.available:
        print("\nRunning benchmark...")
        benchmark = cracker.benchmark(iterations=10000)
        print(f"Status: {benchmark.get('status')}")
        if benchmark.get("status") == "success":
            print(f"Hash Rate: {benchmark.get('hash_rate'):,.0f} H/s")
    else:
        print("\n[!] AVX-512 not available")
        print("[!] Compile native binary with: make -C app/services/hardware/native/")

    print("\n" + "=" * 60 + "\n")
