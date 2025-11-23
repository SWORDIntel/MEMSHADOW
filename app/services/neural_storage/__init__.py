"""
Neural Storage System - Brain-Inspired Memory Fabric for Multi-AI

A sophisticated hierarchical storage system that provides:

- Multi-tiered dimensional databases (T0 cache → T1 RAMDISK → T2 NVMe → T3 Cold)
- Neural connection discovery and creation (Hebbian learning)
- Dynamic CPU allocation for database operations
- Hot/cold storage migration based on temperature
- Cross-tier semantic deduplication
- Multi-AI access control with policy engine
- Spreading activation for associative retrieval
- Pattern completion for partial cues
- Self-optimizing telemetry and auto-tuning

Storage Tiers:
    T0: Process-local KV cache (shared memory/mmap, 256d only)
    T1: RAMDisk hot working set (full vectors up to 4096d)
    T2: NVMe vector store + graph store
    T3: Cold object store (compressed embeddings)

Usage:
    # Basic orchestrator (single AI)
    from app.services.neural_storage import get_neural_storage
    orchestrator = await get_neural_storage()

    # Multi-AI memory fabric
    from app.services.neural_storage import get_memory_fabric
    fabric = await get_memory_fabric()

    # Store memory
    memory = await fabric.put_memory(
        content="Important fact",
        embedding=embedding_vector,
        agent_id="claude-1"
    )

    # Recall context for an AI
    memories, edges = await fabric.recall_context(
        query="related query",
        query_embedding=query_vec,
        agent_id="claude-1"
    )
"""

# Core abstractions
from .core_abstractions import (
    MemoryObject,
    MemoryGraph,
    MemoryGraphView,
    MemoryLink,
    RelationType,
    ShareScope,
    StorageTierLevel,
    EmbeddingProjection,
    PolicyMetadata,
    TierMetadata,
)

# Storage components
from .tiered_database import TieredDatabaseManager, StorageTier
from .ramdisk_storage import RAMDiskStorage
from .memory_migration import MemoryMigrationManager
from .deduplication import CrossTierDeduplicator
from .dynamic_cpu_manager import DynamicCPUManager, WorkloadIntensity

# Neural components
from .neural_connection_engine import NeuralConnectionEngine, SynapticConnection
from .embedding_adapter_hub import EmbeddingAdapterHub, ModelSpec
from .spreading_activation import (
    SpreadingActivationKernel,
    ActivationConstraints,
    ActivationConfig,
    ActivationResult,
)
from .pattern_completion import PatternCompletionEngine, CompletionResult

# Policy and telemetry
from .policy_engine import PolicyEngine, AgentProfile, AccessLevel, Compartment
from .telemetry_tuner import TelemetryAutoTuner, TelemetryCollector, AutoTuner

# Orchestrators
from .orchestrator import (
    NeuralStorageOrchestrator,
    NeuralStorageConfig,
    get_neural_storage,
    shutdown_neural_storage,
)
from .memory_fabric import (
    MemoryFabric,
    FabricConfig,
    get_memory_fabric,
    shutdown_memory_fabric,
)
from .integration import NeuralStorageIntegration, create_neural_storage_config

__all__ = [
    # Core abstractions
    "MemoryObject",
    "MemoryGraph",
    "MemoryGraphView",
    "MemoryLink",
    "RelationType",
    "ShareScope",
    "StorageTierLevel",
    "EmbeddingProjection",
    "PolicyMetadata",
    "TierMetadata",

    # Storage components
    "TieredDatabaseManager",
    "StorageTier",
    "RAMDiskStorage",
    "MemoryMigrationManager",
    "CrossTierDeduplicator",
    "DynamicCPUManager",
    "WorkloadIntensity",

    # Neural components
    "NeuralConnectionEngine",
    "SynapticConnection",
    "EmbeddingAdapterHub",
    "ModelSpec",
    "SpreadingActivationKernel",
    "ActivationConstraints",
    "ActivationConfig",
    "ActivationResult",
    "PatternCompletionEngine",
    "CompletionResult",

    # Policy and telemetry
    "PolicyEngine",
    "AgentProfile",
    "AccessLevel",
    "Compartment",
    "TelemetryAutoTuner",
    "TelemetryCollector",
    "AutoTuner",

    # Basic orchestrator
    "NeuralStorageOrchestrator",
    "NeuralStorageConfig",
    "get_neural_storage",
    "shutdown_neural_storage",

    # Multi-AI memory fabric
    "MemoryFabric",
    "FabricConfig",
    "get_memory_fabric",
    "shutdown_memory_fabric",

    # Integration
    "NeuralStorageIntegration",
    "create_neural_storage_config",
]
