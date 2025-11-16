"""
SWARM - Autonomous Red Team Agent Swarm for MEMSHADOW

This module implements HYDRA Phase 3: The SWARM Project
An autonomous agent swarm for distributed security testing and vulnerability analysis.
"""

from .coordinator import SwarmCoordinator
from .blackboard import Blackboard
from .mission import Mission, MissionStage, MissionLoader

__all__ = [
    'SwarmCoordinator',
    'Blackboard',
    'Mission',
    'MissionStage',
    'MissionLoader'
]
