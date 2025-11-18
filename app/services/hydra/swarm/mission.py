"""
HYDRA SWARM Mission Orchestration
Phase 7: High-level mission planning, execution, and reporting

The mission system provides:
- Mission templates for common scenarios
- Automated agent selection and deployment
- Real-time progress monitoring
- Finding aggregation and reporting
- Mission replay and analysis
"""

from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import asyncio
import uuid
from datetime import datetime, timedelta
import structlog

from .coordinator import (
    SwarmCoordinator,
    Mission,
    MissionStatus,
    SwarmMode
)
from .blackboard import RedisBlackboard
from .agent_base import (
    AgentCapability,
    Finding,
    FindingSeverity
)
from .agent_recon import ReconAgent
from .agent_apimapper import APIMapperAgent
from .agent_authtest import AuthTestAgent

logger = structlog.get_logger()


class MissionTemplate(Enum):
    """Pre-defined mission templates"""
    QUICK_SCAN = "quick_scan"  # Fast reconnaissance
    FULL_ASSESSMENT = "full_assessment"  # Comprehensive security assessment
    API_AUDIT = "api_audit"  # API-specific testing
    AUTH_AUDIT = "auth_audit"  # Authentication/authorization focus
    CONTINUOUS_MONITORING = "continuous_monitoring"  # Ongoing surveillance


class MissionReport:
    """
    Comprehensive mission report.

    Contains all findings, statistics, and recommendations.
    """
    def __init__(self, mission_id: str, mission: Mission):
        self.mission_id = mission_id
        self.mission = mission
        self.generated_at = datetime.utcnow()

        # Statistics
        self.total_findings = 0
        self.findings_by_severity: Dict[FindingSeverity, int] = {}
        self.findings_by_agent: Dict[str, int] = {}

        # Findings
        self.critical_findings: List[Finding] = []
        self.high_findings: List[Finding] = []
        self.medium_findings: List[Finding] = []
        self.low_findings: List[Finding] = []
        self.info_findings: List[Finding] = []

        # Metrics
        self.duration_seconds: Optional[float] = None
        self.agents_deployed = 0
        self.endpoints_discovered = 0
        self.technologies_identified: List[str] = []

    def add_finding(self, finding: Finding, agent_id: str):
        """Add finding to report"""
        self.total_findings += 1

        # By severity
        self.findings_by_severity[finding.severity] = \
            self.findings_by_severity.get(finding.severity, 0) + 1

        # By agent
        self.findings_by_agent[agent_id] = \
            self.findings_by_agent.get(agent_id, 0) + 1

        # Categorize
        if finding.severity == FindingSeverity.CRITICAL:
            self.critical_findings.append(finding)
        elif finding.severity == FindingSeverity.HIGH:
            self.high_findings.append(finding)
        elif finding.severity == FindingSeverity.MEDIUM:
            self.medium_findings.append(finding)
        elif finding.severity == FindingSeverity.LOW:
            self.low_findings.append(finding)
        else:
            self.info_findings.append(finding)

    def calculate_risk_score(self) -> float:
        """
        Calculate overall risk score (0-100).

        Based on finding severity and count.
        """
        weights = {
            FindingSeverity.CRITICAL: 10.0,
            FindingSeverity.HIGH: 5.0,
            FindingSeverity.MEDIUM: 2.0,
            FindingSeverity.LOW: 0.5,
            FindingSeverity.INFO: 0.0
        }

        score = sum(
            self.findings_by_severity.get(severity, 0) * weight
            for severity, weight in weights.items()
        )

        # Normalize to 0-100
        return min(score, 100.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            "mission_id": self.mission_id,
            "target": self.mission.target,
            "mode": self.mission.mode.value,
            "status": self.mission.status.value,
            "generated_at": self.generated_at.isoformat(),

            # Statistics
            "total_findings": self.total_findings,
            "findings_by_severity": {
                severity.value: count
                for severity, count in self.findings_by_severity.items()
            },
            "findings_by_agent": self.findings_by_agent,

            # Risk
            "risk_score": self.calculate_risk_score(),

            # Findings summary
            "critical_count": len(self.critical_findings),
            "high_count": len(self.high_findings),
            "medium_count": len(self.medium_findings),
            "low_count": len(self.low_findings),
            "info_count": len(self.info_findings),

            # Metrics
            "duration_seconds": self.duration_seconds,
            "agents_deployed": self.agents_deployed,
            "endpoints_discovered": self.endpoints_discovered,
            "technologies_identified": self.technologies_identified,

            # Mission details
            "started_at": self.mission.started_at.isoformat() if self.mission.started_at else None,
            "completed_at": self.mission.completed_at.isoformat() if self.mission.completed_at else None
        }


class MissionOrchestrator:
    """
    Mission Orchestrator for HYDRA SWARM.

    Provides high-level mission management, automated agent deployment,
    and comprehensive reporting.

    Example:
        orchestrator = MissionOrchestrator()
        await orchestrator.start()

        # Run a mission
        report = await orchestrator.run_mission(
            template=MissionTemplate.FULL_ASSESSMENT,
            target="https://api.example.com"
        )

        print(f"Risk Score: {report.risk_score}")
        print(f"Critical Findings: {report.critical_count}")
    """

    def __init__(
        self,
        coordinator: Optional[SwarmCoordinator] = None,
        blackboard: Optional[RedisBlackboard] = None
    ):
        """
        Initialize mission orchestrator.

        Args:
            coordinator: Swarm coordinator
            blackboard: Redis blackboard
        """
        self.orchestrator_id = f"orchestrator_{uuid.uuid4().hex[:8]}"
        self.coordinator = coordinator or SwarmCoordinator()
        self.blackboard = blackboard or RedisBlackboard()

        # Mission tracking
        self.active_missions: Dict[str, Mission] = {}
        self.completed_missions: Dict[str, MissionReport] = {}

        # Agent pool
        self.agent_pool: Dict[str, Any] = {}

        logger.info("Mission orchestrator initialized", orchestrator_id=self.orchestrator_id)

    async def start(self):
        """Start the orchestrator"""
        logger.info("Starting mission orchestrator", orchestrator_id=self.orchestrator_id)

        # Start coordinator
        await self.coordinator.start()

        # Initialize agent pool
        await self._initialize_agent_pool()

    async def stop(self):
        """Stop the orchestrator"""
        logger.info("Stopping mission orchestrator", orchestrator_id=self.orchestrator_id)

        # Stop coordinator
        await self.coordinator.stop()

    async def _initialize_agent_pool(self):
        """Initialize pool of available agents"""
        # Create agent instances
        recon_agent = ReconAgent(agent_id="recon_001")
        api_agent = APIMapperAgent(agent_id="api_001")
        auth_agent = AuthTestAgent(agent_id="auth_001")

        # Register with coordinator
        await self.coordinator.register_agent(recon_agent)
        await self.coordinator.register_agent(api_agent)
        await self.coordinator.register_agent(auth_agent)

        # Store in pool
        self.agent_pool[recon_agent.agent_id] = recon_agent
        self.agent_pool[api_agent.agent_id] = api_agent
        self.agent_pool[auth_agent.agent_id] = auth_agent

        logger.info(
            "Agent pool initialized",
            agent_count=len(self.agent_pool),
            agents=list(self.agent_pool.keys())
        )

    async def run_mission(
        self,
        template: MissionTemplate,
        target: str,
        custom_params: Optional[Dict] = None
    ) -> MissionReport:
        """
        Run a mission from template.

        Args:
            template: Mission template to use
            target: Target URL
            custom_params: Custom mission parameters

        Returns:
            Mission report with findings and statistics
        """
        logger.info(
            "Running mission",
            template=template.value,
            target=target,
            orchestrator_id=self.orchestrator_id
        )

        # Create mission from template
        mission = self._create_mission_from_template(template, target, custom_params)

        # Assign mission to coordinator
        success = await self.coordinator.assign_mission(mission)

        if not success:
            logger.error("Failed to assign mission - no available agents")
            raise RuntimeError("No available agents for mission")

        # Track mission
        self.active_missions[mission.mission_id] = mission

        # Execute mission with agents
        await self._execute_mission(mission)

        # Generate report
        report = await self._generate_mission_report(mission)

        # Mark as completed
        self.completed_missions[mission.mission_id] = report
        del self.active_missions[mission.mission_id]

        logger.info(
            "Mission completed",
            mission_id=mission.mission_id,
            findings=report.total_findings,
            risk_score=report.calculate_risk_score()
        )

        return report

    def _create_mission_from_template(
        self,
        template: MissionTemplate,
        target: str,
        custom_params: Optional[Dict]
    ) -> Mission:
        """Create mission configuration from template"""
        params = custom_params or {}

        if template == MissionTemplate.QUICK_SCAN:
            return Mission(
                target=target,
                mode=SwarmMode.RECONNAISSANCE,
                objectives=[
                    "Discover endpoints",
                    "Identify technologies",
                    "Check security headers"
                ],
                required_capabilities=[AgentCapability.RECONNAISSANCE],
                max_agents=2,
                timeout_minutes=10
            )

        elif template == MissionTemplate.FULL_ASSESSMENT:
            return Mission(
                target=target,
                mode=SwarmMode.ASSESSMENT,
                objectives=[
                    "Complete reconnaissance",
                    "Map API endpoints",
                    "Test authentication",
                    "Identify vulnerabilities"
                ],
                required_capabilities=[
                    AgentCapability.RECONNAISSANCE,
                    AgentCapability.API_MAPPING,
                    AgentCapability.AUTH_TESTING
                ],
                max_agents=5,
                timeout_minutes=30
            )

        elif template == MissionTemplate.API_AUDIT:
            return Mission(
                target=target,
                mode=SwarmMode.ASSESSMENT,
                objectives=[
                    "Discover all API endpoints",
                    "Document API structure",
                    "Test rate limiting",
                    "Analyze response schemas"
                ],
                required_capabilities=[
                    AgentCapability.API_MAPPING,
                    AgentCapability.RECONNAISSANCE
                ],
                max_agents=3,
                timeout_minutes=20
            )

        elif template == MissionTemplate.AUTH_AUDIT:
            return Mission(
                target=target,
                mode=SwarmMode.ASSESSMENT,
                objectives=[
                    "Test authentication mechanisms",
                    "Analyze JWT security",
                    "Check session management",
                    "Test access controls"
                ],
                required_capabilities=[AgentCapability.AUTH_TESTING],
                max_agents=2,
                timeout_minutes=15
            )

        else:  # CONTINUOUS_MONITORING
            return Mission(
                target=target,
                mode=SwarmMode.RECONNAISSANCE,
                objectives=["Monitor for changes", "Track new endpoints"],
                required_capabilities=[AgentCapability.RECONNAISSANCE],
                max_agents=1,
                timeout_minutes=60
            )

    async def _execute_mission(self, mission: Mission):
        """Execute mission with assigned agents"""
        # Get assigned agents
        agent_tasks = []

        for agent_id in mission.assigned_agents:
            if agent_id in self.agent_pool:
                agent = self.agent_pool[agent_id]

                # Create task for agent execution
                task = asyncio.create_task(
                    self._run_agent_mission(agent, mission)
                )
                agent_tasks.append(task)

        # Wait for all agents to complete
        if agent_tasks:
            await asyncio.gather(*agent_tasks, return_exceptions=True)

        # Mark mission complete
        await self.coordinator.complete_mission(mission.mission_id)

    async def _run_agent_mission(self, agent: Any, mission: Mission):
        """Run mission for a single agent"""
        try:
            # Execute agent
            findings = await agent.execute(
                target=mission.target,
                blackboard=self.blackboard,
                mission_params={"mission_id": mission.mission_id}
            )

            # Report findings to coordinator
            for finding in findings:
                await self.coordinator.report_finding(
                    agent_id=agent.agent_id,
                    mission_id=mission.mission_id,
                    finding=finding
                )

        except Exception as e:
            logger.error(
                "Agent mission execution error",
                agent_id=agent.agent_id,
                mission_id=mission.mission_id,
                error=str(e)
            )

    async def _generate_mission_report(self, mission: Mission) -> MissionReport:
        """Generate comprehensive mission report"""
        report = MissionReport(mission.mission_id, mission)

        # Calculate duration
        if mission.started_at and mission.completed_at:
            duration = mission.completed_at - mission.started_at
            report.duration_seconds = duration.total_seconds()

        # Add findings
        for finding in mission.findings:
            # Determine which agent reported it (stored in blackboard)
            agent_id = "unknown"  # Would retrieve from blackboard
            report.add_finding(finding, agent_id)

        # Get shared intelligence
        intelligence = await self._gather_intelligence()

        # Extract metrics
        report.agents_deployed = len(mission.assigned_agents)
        report.endpoints_discovered = intelligence.get("endpoint_count", 0)
        report.technologies_identified = intelligence.get("technologies", [])

        return report

    async def _gather_intelligence(self) -> Dict[str, Any]:
        """Gather shared intelligence from blackboard"""
        intelligence = {}

        # Get endpoint intelligence
        api_intel = await self.blackboard.get("intel:api_endpoints") or {}
        intelligence["endpoint_count"] = api_intel.get("endpoint_count", 0)

        # Get technology intelligence
        tech_intel = await self.blackboard.get("intel:technologies") or {}
        intelligence["technologies"] = tech_intel.get("technologies", [])

        # Get security posture
        security_intel = await self.blackboard.get("intel:security_posture") or {}
        intelligence["security_posture"] = security_intel

        return intelligence

    async def get_mission_status(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a mission"""
        return await self.coordinator.get_mission_status(mission_id)

    async def list_active_missions(self) -> List[Dict[str, Any]]:
        """List all active missions"""
        return [
            {
                "mission_id": mission.mission_id,
                "target": mission.target,
                "mode": mission.mode.value,
                "status": mission.status.value,
                "assigned_agents": len(mission.assigned_agents),
                "findings": len(mission.findings)
            }
            for mission in self.active_missions.values()
        ]

    async def get_swarm_health(self) -> Dict[str, Any]:
        """Get overall swarm health status"""
        status = await self.coordinator.get_swarm_status()

        return {
            "orchestrator_id": self.orchestrator_id,
            "coordinator_status": status,
            "active_missions": len(self.active_missions),
            "completed_missions": len(self.completed_missions),
            "total_findings": sum(
                report.total_findings
                for report in self.completed_missions.values()
            )
        }


# Global orchestrator instance
orchestrator = MissionOrchestrator()
