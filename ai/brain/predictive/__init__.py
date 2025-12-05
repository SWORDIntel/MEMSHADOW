#!/usr/bin/env python3
"""
DSMIL Brain Predictive Intelligence Layer

Predictive and precognitive intelligence capabilities:
- Threat Precognition: Bayesian attack graph modeling, adversary trajectory analysis
- Causal Inference: Counterfactual reasoning, root cause analysis, do-calculus
- Pattern-of-Life: Behavioral baselines, deviation scoring, activity prediction
"""

from .threat_precognition import (
    ThreatPrecognition,
    ThreatForecast,
    AttackGraph,
    AdversaryTrajectory,
)

from .causal_inference import (
    CausalInferenceEngine,
    CausalGraph,
    CausalQuery,
    Intervention,
    Counterfactual,
)

from .pattern_of_life import (
    PatternOfLifeEngine,
    EntityProfile,
    BehavioralBaseline,
    DeviationAlert,
    ActivityPrediction,
)

__all__ = [
    # Threat Precognition
    "ThreatPrecognition",
    "ThreatForecast",
    "AttackGraph",
    "AdversaryTrajectory",
    # Causal Inference
    "CausalInferenceEngine",
    "CausalGraph",
    "CausalQuery",
    "Intervention",
    "Counterfactual",
    # Pattern of Life
    "PatternOfLifeEngine",
    "EntityProfile",
    "BehavioralBaseline",
    "DeviationAlert",
    "ActivityPrediction",
]

