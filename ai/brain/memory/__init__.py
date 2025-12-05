"""
DSMIL Brain Memory Module

Memory tier implementations with MEMSHADOW synchronization.

Tiers:
- L1 Working Memory: Fast, limited capacity, RAMDISK
- L2 Episodic Memory: Session/episode storage, NVMe
- L3 Semantic Memory: Long-term knowledge, compressed

Components:
- MemorySyncProtocol: Sync items, batches, and manager
- WorkingMemory: L1 tier implementation
- EpisodicMemory: L2 tier implementation
- SemanticMemory: L3 tier implementation
"""

from .memory_sync_protocol import (
    SyncConfig,
    DEFAULT_CONFIG,
    SyncPriority,
    MemorySyncItem,
    MemorySyncBatch,
    SyncVector,
    MemorySyncManager,
)

from .working_memory import WorkingMemory
from .episodic_memory import EpisodicMemory
from .semantic_memory import SemanticMemory


__all__ = [
    # Sync protocol
    "SyncConfig",
    "DEFAULT_CONFIG",
    "SyncPriority",
    "MemorySyncItem",
    "MemorySyncBatch",
    "SyncVector",
    "MemorySyncManager",
    # Memory tiers
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
]
