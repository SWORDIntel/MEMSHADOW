#!/usr/bin/env python3
"""
DSMIL Brain Autonomous Operations Layer

Self-directed intelligence operations:
- Collection Tasking: Gap analysis and autonomous collection
- Asset Identification: Automated source identification
- Self-Healing: Infrastructure auto-recovery
"""

from .collection_tasking import (
    AutonomousCollector,
    IntelligenceGap,
    CollectionTask,
    CollectionOutcome,
)

from .asset_identification import (
    AssetIdentifier,
    PotentialAsset,
    ApproachStrategy,
    VulnerabilityIndicator,
)

from .self_healing import (
    SelfHealingInfrastructure,
    NodeHealth,
    FailoverEvent,
    RecoveryAction,
)

__all__ = [
    "AutonomousCollector", "IntelligenceGap", "CollectionTask", "CollectionOutcome",
    "AssetIdentifier", "PotentialAsset", "ApproachStrategy", "VulnerabilityIndicator",
    "SelfHealingInfrastructure", "NodeHealth", "FailoverEvent", "RecoveryAction",
]

