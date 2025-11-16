"""
C2 Session Management
Manages active C2 sessions with deployed agents
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import structlog

logger = structlog.get_logger()


class C2Session:
    """
    Represents an active C2 session with a deployed agent
    """

    def __init__(
        self,
        agent_id: str,
        agent_hostname: str,
        agent_ip: str,
        agent_os: str,
        agent_user: str,
        encryption_key: str
    ):
        self.session_id = str(uuid.uuid4())
        self.agent_id = agent_id
        self.agent_hostname = agent_hostname
        self.agent_ip = agent_ip
        self.agent_os = agent_os
        self.agent_user = agent_user
        self.encryption_key = encryption_key

        self.established_at = datetime.now(timezone.utc)
        self.last_seen = datetime.now(timezone.utc)
        self.status = "active"

        self.tasks_sent = 0
        self.tasks_completed = 0
        self.data_exfiltrated = 0  # bytes

        self.metadata = {}

        logger.info(
            "C2 session established",
            session_id=self.session_id,
            agent_id=agent_id,
            hostname=agent_hostname
        )

    def update_last_seen(self):
        """Update last seen timestamp"""
        self.last_seen = datetime.now(timezone.utc)

    def get_uptime_seconds(self) -> int:
        """Get session uptime in seconds"""
        delta = datetime.now(timezone.utc) - self.established_at
        return int(delta.total_seconds())

    def get_idle_seconds(self) -> int:
        """Get idle time in seconds"""
        delta = datetime.now(timezone.utc) - self.last_seen
        return int(delta.total_seconds())

    def is_alive(self, timeout_seconds: int = 300) -> bool:
        """Check if session is alive based on last seen"""
        return self.get_idle_seconds() < timeout_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "agent_hostname": self.agent_hostname,
            "agent_ip": self.agent_ip,
            "agent_os": self.agent_os,
            "agent_user": self.agent_user,
            "established_at": self.established_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "uptime_seconds": self.get_uptime_seconds(),
            "idle_seconds": self.get_idle_seconds(),
            "status": self.status,
            "is_alive": self.is_alive(),
            "tasks_sent": self.tasks_sent,
            "tasks_completed": self.tasks_completed,
            "data_exfiltrated": self.data_exfiltrated,
            "metadata": self.metadata
        }


class SessionManager:
    """
    Manages multiple C2 sessions
    """

    def __init__(self):
        self.sessions: Dict[str, C2Session] = {}

    def add_session(self, session: C2Session):
        """Add a new session"""
        self.sessions[session.session_id] = session
        logger.info("Session added", session_id=session.session_id, total_sessions=len(self.sessions))

    def get_session(self, session_id: str) -> Optional[C2Session]:
        """Get session by ID"""
        return self.sessions.get(session_id)

    def get_session_by_agent_id(self, agent_id: str) -> Optional[C2Session]:
        """Get session by agent ID"""
        for session in self.sessions.values():
            if session.agent_id == agent_id:
                return session
        return None

    def remove_session(self, session_id: str):
        """Remove a session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.status = "terminated"
            del self.sessions[session_id]
            logger.info("Session removed", session_id=session_id)

    def get_all_sessions(self) -> List[C2Session]:
        """Get all sessions"""
        return list(self.sessions.values())

    def get_active_sessions(self) -> List[C2Session]:
        """Get only active (alive) sessions"""
        return [s for s in self.sessions.values() if s.is_alive()]

    def cleanup_dead_sessions(self, timeout_seconds: int = 300):
        """Remove dead sessions"""
        dead_sessions = [
            sid for sid, session in self.sessions.items()
            if not session.is_alive(timeout_seconds)
        ]

        for session_id in dead_sessions:
            self.remove_session(session_id)

        if dead_sessions:
            logger.info("Dead sessions cleaned up", count=len(dead_sessions))

    def get_session_count(self) -> int:
        """Get total session count"""
        return len(self.sessions)

    def get_active_count(self) -> int:
        """Get active session count"""
        return len(self.get_active_sessions())
