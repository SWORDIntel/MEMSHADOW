"""
Meta-Learning Memory System
Phase 8.2: Memory system that learns how to learn

Components:
- MAML Trainer: Model-agnostic meta-learning for few-shot adaptation
- Memory Adapter: Rapidly adapt to new memory domains
- Performance Tracker: Track and baseline performance metrics
- Improvement Engine: Propose and implement optimizations
- Continual Learner: Learn without catastrophic forgetting
"""

from app.services.meta_learning.maml_memory import MAMLMemoryAdapter, MemoryTask
from app.services.meta_learning.performance_tracker import (
    PerformanceTracker,
    PerformanceMetric,
    Baseline
)
from app.services.meta_learning.improvement_engine import (
    ImprovementEngine,
    ImprovementProposal,
    ProposalCategory,
    RiskLevel
)
from app.services.meta_learning.continual_learner import (
    ContinualLearner,
    EWC,
    ProgressiveNN
)

__all__ = [
    "MAMLMemoryAdapter",
    "MemoryTask",
    "PerformanceTracker",
    "PerformanceMetric",
    "Baseline",
    "ImprovementEngine",
    "ImprovementProposal",
    "ProposalCategory",
    "RiskLevel",
    "ContinualLearner",
    "EWC",
    "ProgressiveNN",
]
