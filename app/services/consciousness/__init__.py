"""
Consciousness-Inspired Architecture
Phase 8.3: Global workspace, attention, and metacognition

Implements cognitive architecture patterns:
- Global Workspace Theory (GWT)
- Multi-head attention mechanisms
- Metacognitive monitoring
- Context-aware focus shifting

Based on:
- Baars, B. J. (1988). A Cognitive Theory of Consciousness
- Dehaene, S. et al. (2017). What is consciousness, and could machines have it?
- Vaswani et al. (2017). Attention is All You Need
"""

from app.services.consciousness.global_workspace import (
    GlobalWorkspace,
    WorkspaceItem,
    WorkspaceState,
    BroadcastResult
)

from app.services.consciousness.attention_director import (
    AttentionDirector,
    AttentionHead,
    AttentionResult,
    FocusStrategy
)

from app.services.consciousness.metacognition import (
    MetacognitiveMonitor,
    ConfidenceEstimator,
    MetacognitiveState,
    UncertaintySource
)

from app.services.consciousness.consciousness_integrator import (
    ConsciousnessIntegrator,
    ProcessingMode,
    ConsciousDecision
)

__all__ = [
    # Global Workspace
    "GlobalWorkspace",
    "WorkspaceItem",
    "WorkspaceState",
    "BroadcastResult",

    # Attention
    "AttentionDirector",
    "AttentionHead",
    "AttentionResult",
    "FocusStrategy",

    # Metacognition
    "MetacognitiveMonitor",
    "ConfidenceEstimator",
    "MetacognitiveState",
    "UncertaintySource",

    # Integration
    "ConsciousnessIntegrator",
    "ProcessingMode",
    "ConsciousDecision",
]
