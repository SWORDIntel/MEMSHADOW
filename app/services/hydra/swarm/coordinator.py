"""
HYDRA SWARM Coordinator Node
Phase 7: Multi-Agent Security Testing C2 Server

The coordinator is the command and control (C2) server that:
- Registers and manages agent fleet
- Assigns missions to available agents
- Monitors agent health and status
- Aggregates findings from all agents
- Coordinates multi-agent activities
- Provides real-time swarm intelligence
"""

from typing import Dict, List, Optional, Any
from enum import Enum
import uuid
import asyncio
from datetime import datetime, timedelta
import structlog

from .agent_base import (
    BaseAgent,
    AgentStatus,
    AgentCapability,
    Finding
)
from .blackboard import RedisBlackboard

logger = structlog.get_logger()


class MissionStatus(Enum):
    """Mission lifecycle states"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SwarmMode(Enum):
    """Swarm operational modes"""
    RECONNAISSANCE = "reconnaissance"  # Gather intelligence
    ASSESSMENT = "assessment"  # Evaluate security posture
    PENETRATION = "penetration"  # Active exploitation
    COORDINATED = "coordinated"  # Multi-vector attack


class Mission:
    """
    Represents a security testing mission.

    A mission is assigned to one or more agents and has specific objectives.
    """
    def __init__(
        self,
        target: str,
        mode: SwarmMode,
        objectives: List[str],
        required_capabilities: Optional[List[AgentCapability]] = None,
        max_agents: int = 5,
        timeout_minutes: int = 30
    ):
        self.mission_id = str(uuid.uuid4())
        self.target = target
        self.mode = mode
        self.objectives = objectives
        self.required_capabilities = required_capabilities or []
        self.max_agents = max_agents
        self.timeout_minutes = timeout_minutes
        self.status = MissionStatus.PENDING
        self.assigned_agents: List[str] = []
        self.findings: List[Finding] = []
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None


class AgentRegistration:
    """Tracks registered agent metadata"""
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[AgentCapability]
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.status = AgentStatus.IDLE
        self.current_mission: Optional[str] = None
        self.last_heartbeat = datetime.utcnow()
        self.registered_at = datetime.utcnow()
        self.findings_count = 0


class SwarmCoordinator:
    """
    HYDRA SWARM Coordinator - C2 Server for Multi-Agent Security Testing.

    The coordinator manages the entire agent fleet, assigns missions,
    monitors health, and aggregates intelligence.

    Example:
        coordinator = SwarmCoordinator()
        await coordinator.start()

        # Create mission
        mission = Mission(
            target="https://api.example.com",
            mode=SwarmMode.ASSESSMENT,
            objectives=["Map API endpoints", "Test authentication"]
        )

        # Deploy agents
        await coordinator.assign_mission(mission)

        # Monitor progress
        status = await coordinator.get_mission_status(mission.mission_id)
        findings = await coordinator.get_mission_findings(mission.mission_id)
    """

    def __init__(self, blackboard: Optional[RedisBlackboard] = None):
        """
        Initialize coordinator.

        Args:
            blackboard: Redis blackboard for agent communication
        """
        self.coordinator_id = f"coordinator_{uuid.uuid4().hex[:8]}"
        self.blackboard = blackboard or RedisBlackboard()

        # Agent fleet tracking
        self.agents: Dict[str, AgentRegistration] = {}

        # Mission tracking
        self.missions: Dict[str, Mission] = {}

        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Configuration
        self.heartbeat_timeout_seconds = 30
        self.heartbeat_check_interval = 10

        logger.info("Swarm coordinator initialized", coordinator_id=self.coordinator_id)

    async def start(self):
        """Start coordinator background tasks"""
        logger.info("Starting swarm coordinator", coordinator_id=self.coordinator_id)

        # Start monitoring tasks
        self._monitor_task = asyncio.create_task(self._monitor_agents())
        self._heartbeat_task = asyncio.create_task(self._check_heartbeats())

        # Announce coordinator presence
        await self.blackboard.publish("swarm:coordinator:online", {
            "coordinator_id": self.coordinator_id,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def stop(self):
        """Stop coordinator and cleanup"""
        logger.info("Stopping swarm coordinator", coordinator_id=self.coordinator_id)

        # Cancel background tasks
        if self._monitor_task:
            self._monitor_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        # Notify agents
        await self.blackboard.publish("swarm:coordinator:offline", {
            "coordinator_id": self.coordinator_id,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def register_agent(
        self,
        agent: BaseAgent
    ) -> bool:
        """
        Register an agent with the swarm.

        Args:
            agent: Agent to register

        Returns:
            True if registered successfully
        """
        registration = AgentRegistration(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            capabilities=agent.capabilities
        )

        self.agents[agent.agent_id] = registration

        # Store in blackboard
        await self.blackboard.set(
            f"agent:{agent.agent_id}:registration",
            {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "capabilities": [cap.value for cap in agent.capabilities],
                "registered_at": registration.registered_at.isoformat()
            }
        )

        logger.info(
            "Agent registered",
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            capabilities=len(agent.capabilities)
        )

        # Publish registration event
        await self.blackboard.publish("swarm:agent:registered", {
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type
        })

        return True

    async def unregister_agent(self, agent_id: str):
        """Unregister an agent from the swarm"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            await self.blackboard.delete(f"agent:{agent_id}:registration")
            await self.blackboard.delete(f"agent:{agent_id}:heartbeat")

            logger.info("Agent unregistered", agent_id=agent_id)

    async def assign_mission(self, mission: Mission) -> bool:
        """
        Assign a mission to available agents.

        Args:
            mission: Mission to assign

        Returns:
            True if mission assigned successfully
        """
        # Find suitable agents
        available_agents = await self._find_available_agents(
            mission.required_capabilities,
            mission.max_agents
        )

        if not available_agents:
            logger.warning(
                "No available agents for mission",
                mission_id=mission.mission_id,
                required_capabilities=[cap.value for cap in mission.required_capabilities]
            )
            return False

        # Store mission
        self.missions[mission.mission_id] = mission
        mission.status = MissionStatus.ASSIGNED
        mission.assigned_agents = [agent.agent_id for agent in available_agents]
        mission.started_at = datetime.utcnow()

        # Store in blackboard
        await self.blackboard.set(
            f"mission:{mission.mission_id}:config",
            {
                "mission_id": mission.mission_id,
                "target": mission.target,
                "mode": mission.mode.value,
                "objectives": mission.objectives,
                "assigned_agents": mission.assigned_agents,
                "status": mission.status.value,
                "started_at": mission.started_at.isoformat()
            }
        )

        # Assign to agents
        for agent_reg in available_agents:
            agent_reg.status = AgentStatus.SCANNING
            agent_reg.current_mission = mission.mission_id

            # Publish mission assignment
            await self.blackboard.publish(
                f"agent:{agent_reg.agent_id}:mission",
                {
                    "mission_id": mission.mission_id,
                    "target": mission.target,
                    "mode": mission.mode.value,
                    "objectives": mission.objectives
                }
            )

        logger.info(
            "Mission assigned",
            mission_id=mission.mission_id,
            target=mission.target,
            agents=len(available_agents)
        )

        mission.status = MissionStatus.IN_PROGRESS
        return True

    async def report_finding(
        self,
        agent_id: str,
        mission_id: str,
        finding: Finding
    ):
        """
        Agent reports a security finding.

        Args:
            agent_id: Reporting agent
            mission_id: Associated mission
            finding: Security finding
        """
        # Update mission findings
        if mission_id in self.missions:
            self.missions[mission_id].findings.append(finding)

        # Update agent stats
        if agent_id in self.agents:
            self.agents[agent_id].findings_count += 1

        # Store in blackboard
        await self.blackboard.append(
            f"mission:{mission_id}:findings",
            {
                "finding_id": finding.finding_id,
                "agent_id": agent_id,
                "severity": finding.severity.value,
                "title": finding.title,
                "description": finding.description,
                "timestamp": finding.timestamp.isoformat()
            }
        )

        # Increment findings counter
        await self.blackboard.increment(f"findings:count:{mission_id}")

        logger.info(
            "Finding reported",
            agent_id=agent_id,
            mission_id=mission_id,
            severity=finding.severity.value,
            title=finding.title
        )

        # Publish finding event
        await self.blackboard.publish("swarm:finding:reported", {
            "mission_id": mission_id,
            "agent_id": agent_id,
            "severity": finding.severity.value,
            "title": finding.title
        })

    async def update_agent_heartbeat(self, agent_id: str, status: AgentStatus):
        """Update agent heartbeat and status"""
        if agent_id in self.agents:
            self.agents[agent_id].last_heartbeat = datetime.utcnow()
            self.agents[agent_id].status = status

            # Update blackboard
            await self.blackboard.set(
                f"agent:{agent_id}:heartbeat",
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": status.value
                },
                ttl=self.heartbeat_timeout_seconds
            )

    async def complete_mission(self, mission_id: str, success: bool = True):
        """Mark mission as completed"""
        if mission_id in self.missions:
            mission = self.missions[mission_id]
            mission.status = MissionStatus.COMPLETED if success else MissionStatus.FAILED
            mission.completed_at = datetime.utcnow()

            # Free up agents
            for agent_id in mission.assigned_agents:
                if agent_id in self.agents:
                    self.agents[agent_id].status = AgentStatus.IDLE
                    self.agents[agent_id].current_mission = None

            # Update blackboard
            await self.blackboard.set(
                f"mission:{mission_id}:status",
                {
                    "status": mission.status.value,
                    "completed_at": mission.completed_at.isoformat(),
                    "findings_count": len(mission.findings)
                }
            )

            logger.info(
                "Mission completed",
                mission_id=mission_id,
                status=mission.status.value,
                findings=len(mission.findings),
                duration_minutes=(mission.completed_at - mission.started_at).total_seconds() / 60
            )

    async def get_mission_status(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """Get current mission status"""
        if mission_id not in self.missions:
            return None

        mission = self.missions[mission_id]

        return {
            "mission_id": mission.mission_id,
            "target": mission.target,
            "mode": mission.mode.value,
            "status": mission.status.value,
            "assigned_agents": len(mission.assigned_agents),
            "findings_count": len(mission.findings),
            "created_at": mission.created_at.isoformat(),
            "started_at": mission.started_at.isoformat() if mission.started_at else None,
            "completed_at": mission.completed_at.isoformat() if mission.completed_at else None
        }

    async def get_mission_findings(self, mission_id: str) -> List[Finding]:
        """Get all findings for a mission"""
        if mission_id in self.missions:
            return self.missions[mission_id].findings
        return []

    async def get_swarm_status(self) -> Dict[str, Any]:
        """Get overall swarm status"""
        total_agents = len(self.agents)
        active_agents = sum(1 for a in self.agents.values() if a.status != AgentStatus.IDLE)
        active_missions = sum(1 for m in self.missions.values() if m.status == MissionStatus.IN_PROGRESS)
        total_findings = sum(len(m.findings) for m in self.missions.values())

        return {
            "coordinator_id": self.coordinator_id,
            "total_agents": total_agents,
            "active_agents": active_agents,
            "idle_agents": total_agents - active_agents,
            "active_missions": active_missions,
            "total_missions": len(self.missions),
            "total_findings": total_findings,
            "agent_capabilities": self._get_capability_distribution()
        }

    async def _find_available_agents(
        self,
        required_capabilities: List[AgentCapability],
        max_agents: int
    ) -> List[AgentRegistration]:
        """Find available agents matching required capabilities"""
        available = []

        for agent in self.agents.values():
            # Check if idle
            if agent.status != AgentStatus.IDLE:
                continue

            # Check if has required capabilities
            if required_capabilities:
                has_all = all(
                    cap in agent.capabilities
                    for cap in required_capabilities
                )
                if not has_all:
                    continue

            available.append(agent)

            if len(available) >= max_agents:
                break

        return available

    async def _monitor_agents(self):
        """Background task to monitor agent activities"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_check_interval)

                # Check for stale missions
                now = datetime.utcnow()
                for mission in self.missions.values():
                    if mission.status == MissionStatus.IN_PROGRESS:
                        if mission.started_at:
                            elapsed = (now - mission.started_at).total_seconds() / 60
                            if elapsed > mission.timeout_minutes:
                                logger.warning(
                                    "Mission timeout",
                                    mission_id=mission.mission_id,
                                    elapsed_minutes=elapsed
                                )
                                await self.complete_mission(mission.mission_id, success=False)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Agent monitoring error", error=str(e))

    async def _check_heartbeats(self):
        """Background task to check agent heartbeats"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_check_interval)

                now = datetime.utcnow()
                timeout_threshold = timedelta(seconds=self.heartbeat_timeout_seconds)

                for agent in self.agents.values():
                    time_since_heartbeat = now - agent.last_heartbeat

                    if time_since_heartbeat > timeout_threshold:
                        logger.warning(
                            "Agent heartbeat timeout",
                            agent_id=agent.agent_id,
                            seconds_since_heartbeat=time_since_heartbeat.total_seconds()
                        )

                        # Mark agent as offline
                        agent.status = AgentStatus.IDLE

                        # Reassign mission if agent was working on one
                        if agent.current_mission and agent.current_mission in self.missions:
                            mission = self.missions[agent.current_mission]
                            mission.assigned_agents.remove(agent.agent_id)
                            agent.current_mission = None

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat check error", error=str(e))

    def _get_capability_distribution(self) -> Dict[str, int]:
        """Get distribution of capabilities across agent fleet"""
        distribution: Dict[str, int] = {}

        for agent in self.agents.values():
            for capability in agent.capabilities:
                cap_name = capability.value
                distribution[cap_name] = distribution.get(cap_name, 0) + 1

        return distribution


# Global coordinator instance
coordinator = SwarmCoordinator()
