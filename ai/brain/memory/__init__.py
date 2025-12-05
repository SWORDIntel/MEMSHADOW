#!/usr/bin/env python3
"""
DSMIL Brain Memory Fabric

Hierarchical memory system with three tiers:
- L1 Working Memory: High-speed, hardware-adaptive scratchpad
- L2 Episodic Memory: Session-based experiences with temporal indexing
- L3 Semantic Memory: Long-term knowledge graph with concept relationships

Features:
- Automatic hardware detection and memory sizing
- Cross-tier consolidation and promotion
- Distributed sync for multi-node operation
- Forgetting curves with importance weighting
"""

from .working_memory import (
    WorkingMemory,
    WorkingMemoryItem,
    AttentionWeight,
    MemoryPressure,
)

from .episodic_memory import (
    EpisodicMemory,
    Episode,
    Event,
    ExperienceReplay,
)

from .semantic_memory import (
    SemanticMemory,
    Concept,
    Relationship,
    KnowledgeGraph,
)

from .memory_consolidator import (
    MemoryConsolidator,
    ConsolidationPolicy,
    PromotionCriteria,
)

from .memory_sync_protocol import (
    MemorySyncManager,
    MemorySyncBatch,
    MemorySyncItem,
    MemoryTier,
    SyncOperation,
    SyncPriority,
    SyncFlags,
)

__all__ = [
    # Working Memory
    "WorkingMemory",
    "WorkingMemoryItem",
    "AttentionWeight",
    "MemoryPressure",
    # Episodic Memory
    "EpisodicMemory",
    "Episode",
    "Event",
    "ExperienceReplay",
    # Semantic Memory
    "SemanticMemory",
    "Concept",
    "Relationship",
    "KnowledgeGraph",
    # Consolidator
    "MemoryConsolidator",
    "ConsolidationPolicy",
    "PromotionCriteria",
    # Sync Protocol (MEMSHADOW v2)
    "MemorySyncManager",
    "MemorySyncBatch",
    "MemorySyncItem",
    "MemoryTier",
    "SyncOperation",
    "SyncPriority",
    "SyncFlags",
]

