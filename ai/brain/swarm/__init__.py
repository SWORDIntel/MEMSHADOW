#!/usr/bin/env python3
"""
DSMIL Brain Swarm Cognition Layer

Collective distributed intelligence:
- Emergence: Collective reasoning > individual
- Stigmergy: Indirect coordination via knowledge markers
- Adversarial Improvement: Self-attacking for hardening
"""

from .emergence import (
    EmergentIntelligence,
    CollectiveReasoning,
    NetworkPattern,
)

from .stigmergy import (
    StigmergicKnowledge,
    PheromoneMarker,
    KnowledgeHighway,
)

from .adversarial_improvement import (
    AdversarialImprover,
    ProbeResult,
    VulnerabilityPatch,
)

__all__ = [
    "EmergentIntelligence", "CollectiveReasoning", "NetworkPattern",
    "StigmergicKnowledge", "PheromoneMarker", "KnowledgeHighway",
    "AdversarialImprover", "ProbeResult", "VulnerabilityPatch",
]

