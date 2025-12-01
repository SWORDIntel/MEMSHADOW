"""
HYDRA SWARM - Multi-Vector Swarm Security Testing
Phase 7: Autonomous multi-agent security testing framework

Components:
- Coordinator Node: C2 server managing agent swarm
- MVS Agents: Autonomous security testing agents
- Redis Blackboard: Agent communication and state
- Mission System: Task orchestration and reporting
- The Arena: Isolated testing environment
"""

from app.services.hydra.swarm.coordinator import (
    SwarmCoordinator,
    coordinator,
    Mission,
    MissionStatus,
    SwarmMode
)
from app.services.hydra.swarm.agent_base import (
    BaseAgent,
    AgentStatus,
    AgentCapability,
    Finding,
    FindingSeverity
)
from app.services.hydra.swarm.blackboard import RedisBlackboard, blackboard
from app.services.hydra.swarm.mission import (
    MissionOrchestrator,
    orchestrator,
    MissionTemplate,
    MissionReport
)
from app.services.hydra.swarm.agent_recon import ReconAgent
from app.services.hydra.swarm.agent_apimapper import APIMapperAgent
from app.services.hydra.swarm.agent_authtest import AuthTestAgent

__all__ = [
    # Coordinator
    "SwarmCoordinator",
    "coordinator",
    "Mission",
    "MissionStatus",
    "SwarmMode",

    # Agents
    "BaseAgent",
    "AgentStatus",
    "AgentCapability",
    "Finding",
    "FindingSeverity",
    "ReconAgent",
    "APIMapperAgent",
    "AuthTestAgent",

    # Blackboard
    "RedisBlackboard",
    "blackboard",

    # Mission Orchestration
    "MissionOrchestrator",
    "orchestrator",
    "MissionTemplate",
    "MissionReport",
]
