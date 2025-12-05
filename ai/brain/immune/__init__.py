#!/usr/bin/env python3
"""
DSMIL Brain Digital Immune System

Biological-inspired defense and adaptation:
- Antibodies: Known threat signatures
- T-Cell Agents: Autonomous anomaly hunters
- Memory B-Cells: Rapid threat re-recognition
- Cytokine Alerts: Network-wide threat propagation
- Neural Plasticity: Self-restructuring under stress
- Genetic Evolution: Strategy evolution
"""

from .antibodies import (
    AntibodyLibrary,
    ThreatSignature,
    SignatureMatch,
)

from .tcell_agents import (
    TCellAgent,
    TCellSwarm,
    AnomalyHunt,
)

from .memory_bcells import (
    MemoryBCell,
    ThreatMemoryBank,
    RapidResponse,
)

from .cytokine_alerts import (
    CytokineSystem,
    ThreatAlert,
    AlertCascade,
)

from .neural_plasticity import (
    NeuralPlasticity,
    PathwayStrength,
    DamageRecovery,
)

from .genetic_evolution import (
    GeneticEvolution,
    AnalysisStrategy,
    StrategyPopulation,
)

__all__ = [
    # Antibodies
    "AntibodyLibrary",
    "ThreatSignature",
    "SignatureMatch",
    # T-Cells
    "TCellAgent",
    "TCellSwarm",
    "AnomalyHunt",
    # Memory B-Cells
    "MemoryBCell",
    "ThreatMemoryBank",
    "RapidResponse",
    # Cytokines
    "CytokineSystem",
    "ThreatAlert",
    "AlertCascade",
    # Plasticity
    "NeuralPlasticity",
    "PathwayStrength",
    "DamageRecovery",
    # Evolution
    "GeneticEvolution",
    "AnalysisStrategy",
    "StrategyPopulation",
]

