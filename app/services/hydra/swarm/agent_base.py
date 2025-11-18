"""
HYDRA SWARM Base Agent
Phase 7: Base class for all MVS (Multi-Vector Swarm) agents

All security testing agents inherit from BaseAgent
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import asyncio
import structlog

logger = structlog.get_logger()


class AgentStatus(str, Enum):
    """Agent operational status"""
    IDLE = "idle"
    SCANNING = "scanning"
    TESTING = "testing"
    EXPLOITING = "exploiting"
    REPORTING = "reporting"
    ERROR = "error"
    TERMINATED = "terminated"


class AgentCapability(str, Enum):
    """Agent capabilities"""
    RECONNAISSANCE = "reconnaissance"
    API_MAPPING = "api_mapping"
    AUTH_TESTING = "auth_testing"
    INJECTION_TESTING = "injection_testing"
    EXPLOITATION = "exploitation"
    PERSISTENCE = "persistence"
    REPORTING = "reporting"


class Finding:
    """Security finding discovered by agent"""
    def __init__(
        self,
        severity: str,  # CRITICAL, HIGH, MEDIUM, LOW, INFO
        title: str,
        description: str,
        evidence: Dict[str, Any],
        remediation: Optional[str] = None
    ):
        self.finding_id = str(uuid.uuid4())
        self.severity = severity
        self.title = title
        self.description = description
        self.evidence = evidence
        self.remediation = remediation
        self.discovered_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "remediation": self.remediation,
            "discovered_at": self.discovered_at.isoformat()
        }


class BaseAgent(ABC):
    """
    Base class for all HYDRA SWARM agents.

    All MVS agents inherit from this class and implement:
    - execute(): Main agent logic
    - get_capabilities(): List of agent capabilities
    - cleanup(): Resource cleanup

    Example:
        class ReconAgent(BaseAgent):
            async def execute(self, target, blackboard):
                # Perform reconnaissance
                findings = []
                # ... agent logic ...
                return findings

            def get_capabilities(self):
                return [AgentCapability.RECONNAISSANCE]
    """

    def __init__(
        self,
        agent_name: str,
        agent_type: str,
        coordinator_id: str
    ):
        self.agent_id = str(uuid.uuid4())
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.coordinator_id = coordinator_id

        self.status = AgentStatus.IDLE
        self.findings: List[Finding] = []
        self.metadata: Dict[str, Any] = {}

        self.created_at = datetime.utcnow()
        self.last_heartbeat = datetime.utcnow()
        self.is_running = False

        logger.info(
            "Agent initialized",
            agent_id=self.agent_id,
            agent_name=agent_name,
            agent_type=agent_type
        )

    @abstractmethod
    async def execute(
        self,
        target: str,
        blackboard: Any,
        mission_params: Optional[Dict[str, Any]] = None
    ) -> List[Finding]:
        """
        Execute agent's primary function.

        Args:
            target: Target URL/system to test
            blackboard: Redis blackboard for agent communication
            mission_params: Mission-specific parameters

        Returns:
            List of security findings
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """
        Get agent capabilities.

        Returns:
            List of capabilities this agent possesses
        """
        pass

    async def start(
        self,
        target: str,
        blackboard: Any,
        mission_params: Optional[Dict[str, Any]] = None
    ):
        """
        Start agent execution.

        Args:
            target: Target system
            blackboard: Communication blackboard
            mission_params: Mission parameters
        """
        if self.is_running:
            logger.warning("Agent already running", agent_id=self.agent_id)
            return

        self.is_running = True
        self.status = AgentStatus.SCANNING

        logger.info(
            "Agent starting",
            agent_id=self.agent_id,
            target=target,
            capabilities=self.get_capabilities()
        )

        try:
            # Execute agent logic
            findings = await self.execute(target, blackboard, mission_params)

            # Store findings
            self.findings.extend(findings)

            # Report to coordinator
            await self._report_findings(blackboard, findings)

            self.status = AgentStatus.IDLE

            logger.info(
                "Agent execution complete",
                agent_id=self.agent_id,
                findings_count=len(findings)
            )

        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(
                "Agent execution failed",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            raise

        finally:
            self.is_running = False

    async def _report_findings(
        self,
        blackboard: Any,
        findings: List[Finding]
    ):
        """Report findings to coordinator via blackboard"""
        for finding in findings:
            await blackboard.publish(
                channel=f"findings:{self.coordinator_id}",
                message={
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                    "finding": finding.to_dict()
                }
            )

    async def heartbeat(self, blackboard: Any):
        """Send heartbeat to coordinator"""
        self.last_heartbeat = datetime.utcnow()

        await blackboard.set(
            key=f"agent:{self.agent_id}:heartbeat",
            value={
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "status": self.status,
                "timestamp": self.last_heartbeat.isoformat()
            },
            ttl=60  # 1 minute TTL
        )

    async def get_shared_intelligence(
        self,
        blackboard: Any,
        intelligence_type: str
    ) -> List[Dict[str, Any]]:
        """
        Get shared intelligence from blackboard.

        Args:
            blackboard: Redis blackboard
            intelligence_type: Type of intelligence (endpoints, credentials, etc.)

        Returns:
            List of intelligence items
        """
        data = await blackboard.get(f"intel:{intelligence_type}")
        return data if data else []

    async def share_intelligence(
        self,
        blackboard: Any,
        intelligence_type: str,
        data: Dict[str, Any]
    ):
        """
        Share intelligence with other agents.

        Args:
            blackboard: Redis blackboard
            intelligence_type: Type of intelligence
            data: Intelligence data
        """
        key = f"intel:{intelligence_type}"
        existing = await blackboard.get(key) or []
        existing.append({
            **data,
            "source_agent": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        await blackboard.set(key, existing)

        logger.debug(
            "Intelligence shared",
            agent_id=self.agent_id,
            type=intelligence_type
        )

    def add_finding(
        self,
        severity: str,
        title: str,
        description: str,
        evidence: Dict[str, Any],
        remediation: Optional[str] = None
    ) -> Finding:
        """
        Create and add a finding.

        Args:
            severity: CRITICAL, HIGH, MEDIUM, LOW, INFO
            title: Finding title
            description: Detailed description
            evidence: Evidence data
            remediation: Remediation advice

        Returns:
            Created finding
        """
        finding = Finding(
            severity=severity,
            title=title,
            description=description,
            evidence=evidence,
            remediation=remediation
        )

        self.findings.append(finding)

        logger.info(
            "Finding added",
            agent_id=self.agent_id,
            severity=severity,
            title=title
        )

        return finding

    async def cleanup(self):
        """Cleanup agent resources"""
        self.is_running = False
        self.status = AgentStatus.TERMINATED

        logger.info("Agent cleanup complete", agent_id=self.agent_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        findings_by_severity = {}
        for finding in self.findings:
            sev = finding.severity
            findings_by_severity[sev] = findings_by_severity.get(sev, 0) + 1

        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "status": self.status,
            "total_findings": len(self.findings),
            "findings_by_severity": findings_by_severity,
            "capabilities": self.get_capabilities(),
            "uptime_seconds": (datetime.utcnow() - self.created_at).total_seconds(),
            "is_running": self.is_running
        }
