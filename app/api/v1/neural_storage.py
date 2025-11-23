"""
Neural Storage API Endpoints

REST API for the brain-like neural storage system providing:
- Statistics and monitoring
- Manual optimization triggers
- Connection discovery
- Tier management
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from uuid import UUID

from app.services.neural_storage.orchestrator import (
    get_neural_storage,
    NeuralStorageOrchestrator
)
from app.services.neural_storage.tiered_database import StorageTier

router = APIRouter(prefix="/neural-storage", tags=["Neural Storage"])


# ==================== Request/Response Models ====================

class NeuralStorageStats(BaseModel):
    """Statistics for the neural storage system"""
    orchestrator: Dict[str, Any]
    tiered_database: Dict[str, Any]
    cpu_manager: Dict[str, Any]
    ramdisk: Dict[str, Any]
    connections: Dict[str, Any]
    migration: Dict[str, Any]
    deduplication: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health status response"""
    status: str
    components: Dict[str, str]
    metrics: Dict[str, float]


class OptimizationResult(BaseModel):
    """Result of optimization operation"""
    deduplication: Dict[str, int]
    cold_cleaned: int
    orphans_removed: int
    connections_decayed: int


class AssociatedMemory(BaseModel):
    """Associated memory response"""
    memory_id: str
    activation_score: float


class ConnectionSuggestion(BaseModel):
    """Suggested connection response"""
    memory_id: str
    confidence: float
    reason: str


class TierDistribution(BaseModel):
    """Tier distribution response"""
    distribution: Dict[str, int]


# ==================== Endpoints ====================

@router.get("/stats", response_model=NeuralStorageStats)
async def get_stats():
    """
    Get comprehensive statistics for the neural storage system.

    Returns statistics for all components including:
    - Tiered database metrics
    - CPU manager resource allocation
    - RAMDISK utilization
    - Connection graph statistics
    - Migration statistics
    - Deduplication statistics
    """
    orchestrator = await get_neural_storage()
    stats = await orchestrator.get_stats()
    return stats


@router.get("/health", response_model=HealthResponse)
async def get_health():
    """
    Get health status of all neural storage components.

    Returns the health status of:
    - Tiered database
    - RAMDISK
    - CPU manager
    - Connection engine
    - Migration manager
    - Deduplicator
    """
    orchestrator = await get_neural_storage()
    health = await orchestrator.get_health()
    return health


@router.post("/optimize", response_model=OptimizationResult)
async def run_optimization():
    """
    Trigger manual optimization of the neural storage system.

    This runs:
    - Cross-tier deduplication
    - Cold memory cleanup
    - Orphaned connection cleanup
    - Connection decay
    """
    orchestrator = await get_neural_storage()
    result = await orchestrator.optimize()
    return result


@router.post("/rebalance")
async def rebalance_tiers():
    """
    Rebalance memory distribution across storage tiers.

    Promotes hot memories to faster tiers and demotes cold memories
    to slower tiers based on access patterns.
    """
    orchestrator = await get_neural_storage()
    distribution = await orchestrator.rebalance_tiers()
    return {"distribution": distribution}


@router.post("/discover-connections")
async def discover_connections(
    batch_size: int = Query(default=100, ge=1, le=1000)
):
    """
    Run connection discovery on memories.

    This is the brain-like "making connections" operation that:
    - Finds semantic similarities between memories
    - Creates synaptic-like connections
    - Detects memory clusters
    """
    orchestrator = await get_neural_storage()
    discovered = await orchestrator.discover_all_connections(batch_size=batch_size)
    return {
        "connections_discovered": discovered,
        "batch_size": batch_size
    }


@router.get("/associated/{memory_id}", response_model=List[AssociatedMemory])
async def get_associated_memories(
    memory_id: UUID,
    max_depth: int = Query(default=2, ge=1, le=5),
    top_k: int = Query(default=10, ge=1, le=100)
):
    """
    Get memories associated with a given memory using spreading activation.

    This mimics how the brain retrieves related memories through
    neural pathway activation.
    """
    orchestrator = await get_neural_storage()
    results = await orchestrator.get_associated_memories(
        memory_id, max_depth=max_depth, top_k=top_k
    )
    return [
        AssociatedMemory(memory_id=str(mid), activation_score=score)
        for mid, score in results
    ]


@router.get("/complete-pattern", response_model=List[AssociatedMemory])
async def complete_pattern(
    memory_ids: List[UUID] = Query(...),
    top_k: int = Query(default=5, ge=1, le=50)
):
    """
    Complete a partial memory pattern by finding likely associated memories.

    Given a set of memories, finds other memories that commonly
    appear together with them.
    """
    orchestrator = await get_neural_storage()
    results = await orchestrator.complete_pattern(memory_ids, top_k=top_k)
    return [
        AssociatedMemory(memory_id=str(mid), activation_score=score)
        for mid, score in results
    ]


@router.get("/bridge/{memory_a}/{memory_b}")
async def find_bridge(memory_a: UUID, memory_b: UUID):
    """
    Find memories that bridge (connect) two seemingly unrelated memories.

    Discovers hidden relationships that might have been missed.
    """
    orchestrator = await get_neural_storage()
    paths = await orchestrator.find_bridge(memory_a, memory_b)
    return {
        "memory_a": str(memory_a),
        "memory_b": str(memory_b),
        "bridging_paths": [
            [str(mid) for mid in path]
            for path in paths
        ]
    }


@router.get("/suggest-connections/{memory_id}", response_model=List[ConnectionSuggestion])
async def suggest_connections(
    memory_id: UUID,
    top_k: int = Query(default=5, ge=1, le=20)
):
    """
    Suggest potential connections for a memory.

    Returns suggested memories with confidence scores and reasons.
    """
    orchestrator = await get_neural_storage()
    suggestions = await orchestrator.suggest_connections(memory_id, top_k=top_k)
    return [
        ConnectionSuggestion(memory_id=str(mid), confidence=conf, reason=reason)
        for mid, conf, reason in suggestions
    ]


@router.get("/tier-distribution", response_model=TierDistribution)
async def get_tier_distribution():
    """
    Get the current distribution of memories across storage tiers.
    """
    orchestrator = await get_neural_storage()
    distribution = await orchestrator.migration_manager.get_tier_distribution()
    return TierDistribution(distribution=distribution)


@router.post("/promote/{memory_id}")
async def promote_memory(
    memory_id: UUID,
    target_tier: str = Query(default="RAMDISK")
):
    """
    Force promote a memory to a faster storage tier.
    """
    try:
        tier = StorageTier[target_tier.upper()]
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tier: {target_tier}. Valid tiers: {[t.name for t in StorageTier]}"
        )

    orchestrator = await get_neural_storage()
    success = await orchestrator.migration_manager.force_promote(memory_id, tier)

    if not success:
        raise HTTPException(status_code=404, detail="Memory not found or already at target tier")

    return {"memory_id": str(memory_id), "promoted_to": tier.name}


@router.post("/demote/{memory_id}")
async def demote_memory(
    memory_id: UUID,
    target_tier: str = Query(default="ULTRA_HIGH")
):
    """
    Force demote a memory to a slower storage tier.
    """
    try:
        tier = StorageTier[target_tier.upper()]
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tier: {target_tier}. Valid tiers: {[t.name for t in StorageTier]}"
        )

    orchestrator = await get_neural_storage()
    success = await orchestrator.migration_manager.force_demote(memory_id, tier)

    if not success:
        raise HTTPException(status_code=404, detail="Memory not found or already at target tier")

    return {"memory_id": str(memory_id), "demoted_to": tier.name}


@router.get("/ramdisk/hottest")
async def get_hottest_memories(
    top_k: int = Query(default=10, ge=1, le=100)
):
    """
    Get the hottest (most active) memories in RAMDISK.
    """
    orchestrator = await get_neural_storage()
    hottest = await orchestrator.ramdisk.get_hottest_memories(top_k=top_k)
    return {
        "hottest_memories": [
            {"memory_id": str(mid), "temperature": temp}
            for mid, temp in hottest
        ]
    }


@router.get("/ramdisk/coldest")
async def get_coldest_memories(
    top_k: int = Query(default=10, ge=1, le=100)
):
    """
    Get the coldest (least active) memories in RAMDISK.
    """
    orchestrator = await get_neural_storage()
    coldest = await orchestrator.ramdisk.get_coldest_memories(top_k=top_k)
    return {
        "coldest_memories": [
            {"memory_id": str(mid), "temperature": temp}
            for mid, temp in coldest
        ]
    }


@router.post("/cpu/burst")
async def request_cpu_burst(
    duration_seconds: float = Query(default=10.0, ge=1.0, le=60.0)
):
    """
    Request temporary burst CPU capacity for intensive operations.
    """
    orchestrator = await get_neural_storage()
    await orchestrator.cpu_manager.request_burst(duration_seconds)
    return {
        "burst_requested": True,
        "duration_seconds": duration_seconds
    }


@router.post("/cpu/priority")
async def set_cpu_priority(
    priority: int = Query(ge=1, le=10)
):
    """
    Set processing priority (1-10, higher = more resources).
    """
    orchestrator = await get_neural_storage()
    orchestrator.cpu_manager.set_priority(priority)
    return {"priority_set": priority}
