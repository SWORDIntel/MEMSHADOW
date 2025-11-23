"""
Neural Storage System - Brain-like Multi-Tiered Memory Management

This module implements a sophisticated hierarchical storage system that mimics
human brain memory organization with:

- Multi-tiered dimensional databases (high → low dimensional → RAMDISK)
- Neural connection discovery and creation
- Dynamic CPU allocation for database operations
- Hot/cold storage migration
- Cross-tier deduplication

Architecture:
    Tier 0: Ultra-High Dimensional (4096d) - Maximum fidelity, archival
    Tier 1: High Dimensional (2048d) - Standard long-term storage
    Tier 2: Medium Dimensional (1024d) - Warm storage
    Tier 3: Low Dimensional (512d) - Hot compressed
    Tier 4: RAMDISK (256d) - Ultra-fast working memory

Usage:
    from app.services.neural_storage import get_neural_storage

    # Get the orchestrator
    orchestrator = await get_neural_storage()

    # Store a memory
    record = await orchestrator.store_memory(
        memory_id=uuid,
        content_hash=hash,
        embedding=embedding
    )

    # Search with brain-like associations
    results = await orchestrator.search(query_embedding, top_k=10)

    # Get associated memories via spreading activation
    related = await orchestrator.get_associated_memories(memory_id)
"""

from .tiered_database import TieredDatabaseManager, StorageTier
from .neural_connection_engine import NeuralConnectionEngine
from .dynamic_cpu_manager import DynamicCPUManager
from .ramdisk_storage import RAMDiskStorage
from .memory_migration import MemoryMigrationManager
from .deduplication import CrossTierDeduplicator
from .orchestrator import (
    NeuralStorageOrchestrator,
    NeuralStorageConfig,
    get_neural_storage,
    shutdown_neural_storage,
)
from .integration import NeuralStorageIntegration, create_neural_storage_config

__all__ = [
    # Core components
    "TieredDatabaseManager",
    "StorageTier",
    "NeuralConnectionEngine",
    "DynamicCPUManager",
    "RAMDiskStorage",
    "MemoryMigrationManager",
    "CrossTierDeduplicator",
    # Orchestrator
    "NeuralStorageOrchestrator",
    "NeuralStorageConfig",
    "get_neural_storage",
    "shutdown_neural_storage",
    # Integration
    "NeuralStorageIntegration",
    "create_neural_storage_config",
]
