"""
Memory Fabric API - Unified API Surface

Data Plane (fast path):
- PutMemoryObject(mem_object, scope)
- QueryByEmbedding(embedding, filters, k)
- QueryGraph(seed_ids, depth, budget)
- RecallContext(query, agent_id, constraints) → ranked MemoryObjects + edges

Control / Monitoring Plane:
- /stats/tiers – hit/miss, sizes, temps
- /stats/agents – per-AI usage, latency
- /graph/introspect/{id} – connections, activation paths
- /policy/* – define compartments, sharing rules
- /orchestrator/jobs – migration/dedup/index tasks
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Set
from uuid import UUID
import numpy as np

from app.services.neural_storage.memory_fabric import get_memory_fabric, MemoryFabric
from app.services.neural_storage.core_abstractions import ShareScope, RelationType
from app.services.neural_storage.policy_engine import AccessLevel
from app.services.neural_storage.spreading_activation import ActivationConstraints

router = APIRouter(tags=["Memory Fabric"])


# ==================== Request/Response Models ====================

class PutMemoryRequest(BaseModel):
    """Request to store a memory"""
    content: str
    embedding: List[float]
    model_id: str = "default"
    agent_id: Optional[str] = None
    share_scope: str = "global"
    metadata: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class PutMemoryResponse(BaseModel):
    """Response from storing a memory"""
    memory_id: str
    content_hash: str
    canonical_dimension: int


class QueryEmbeddingRequest(BaseModel):
    """Request to query by embedding"""
    embedding: List[float]
    agent_id: Optional[str] = None
    k: int = 10
    filters: Optional[Dict[str, Any]] = None


class QueryResult(BaseModel):
    """A single query result"""
    memory_id: str
    score: float
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryGraphRequest(BaseModel):
    """Request for graph query"""
    seed_ids: List[str]
    agent_id: Optional[str] = None
    depth: int = 2
    budget: int = 50


class GraphQueryResult(BaseModel):
    """Result of graph query"""
    activations: Dict[str, float]
    edges: List[Dict[str, Any]]
    stats: Dict[str, Any]


class RecallContextRequest(BaseModel):
    """Request for context recall"""
    query: str
    embedding: List[float]
    agent_id: str
    context_memory_ids: Optional[List[str]] = None
    max_results: int = 20


class RecallContextResponse(BaseModel):
    """Response from context recall"""
    memories: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    confidence: float


class RegisterAgentRequest(BaseModel):
    """Request to register an agent"""
    agent_id: str
    name: str
    groups: Optional[List[str]] = None
    clearance: str = "INTERNAL"


class AddConnectionRequest(BaseModel):
    """Request to add a connection"""
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    agent_id: Optional[str] = None


# ==================== Data Plane Endpoints ====================

@router.post("/data/memory", response_model=PutMemoryResponse)
async def put_memory(request: PutMemoryRequest):
    """
    Store a memory object (Data Plane).

    PutMemoryObject(mem_object, scope)
    """
    fabric = await get_memory_fabric()

    # Parse share scope
    try:
        scope = ShareScope(request.share_scope)
    except ValueError:
        scope = ShareScope.GLOBAL

    memory = await fabric.put_memory(
        content=request.content,
        embedding=np.array(request.embedding, dtype=np.float32),
        model_id=request.model_id,
        agent_id=request.agent_id,
        share_scope=scope,
        metadata=request.metadata,
        session_id=request.session_id
    )

    return PutMemoryResponse(
        memory_id=str(memory.id),
        content_hash=memory.content_hash,
        canonical_dimension=memory.canonical_dimension
    )


@router.post("/data/query/embedding", response_model=List[QueryResult])
async def query_by_embedding(request: QueryEmbeddingRequest):
    """
    Query memories by embedding similarity (Data Plane).

    QueryByEmbedding(embedding, filters, k)
    """
    fabric = await get_memory_fabric()

    results = await fabric.query_by_embedding(
        embedding=np.array(request.embedding, dtype=np.float32),
        agent_id=request.agent_id,
        k=request.k,
        filters=request.filters
    )

    return [
        QueryResult(
            memory_id=str(mem.id),
            score=score,
            content=mem.content[:200] if mem.content else None,
            metadata=mem.metadata
        )
        for mem, score in results
    ]


@router.post("/data/query/graph", response_model=GraphQueryResult)
async def query_graph(request: QueryGraphRequest):
    """
    Query by graph traversal (Data Plane).

    QueryGraph(seed_ids, depth, budget)
    """
    fabric = await get_memory_fabric()

    result = await fabric.query_graph(
        seed_ids=[UUID(sid) for sid in request.seed_ids],
        agent_id=request.agent_id,
        depth=request.depth,
        budget=request.budget
    )

    return GraphQueryResult(
        activations={str(k): v for k, v in result.activations.items()},
        edges=[
            {"source": str(s), "target": str(t), "weight": w}
            for s, t, w in result.subgraph_edges
        ],
        stats=result.stats
    )


@router.post("/data/recall", response_model=RecallContextResponse)
async def recall_context(request: RecallContextRequest):
    """
    Recall context for an AI agent (Data Plane).

    RecallContext(query, agent_id, constraints) → ranked MemoryObjects + edges
    """
    fabric = await get_memory_fabric()

    context_ids = None
    if request.context_memory_ids:
        context_ids = [UUID(mid) for mid in request.context_memory_ids]

    memories, edges = await fabric.recall_context(
        query=request.query,
        query_embedding=np.array(request.embedding, dtype=np.float32),
        agent_id=request.agent_id,
        context_memories=context_ids,
        max_results=request.max_results
    )

    return RecallContextResponse(
        memories=[
            {
                "memory_id": str(m.id),
                "content": m.content[:500] if m.content else None,
                "metadata": m.metadata
            }
            for m in memories
        ],
        edges=[
            {"source": str(s), "target": str(t), "weight": w}
            for s, t, w in edges
        ],
        confidence=0.8 if memories else 0.0
    )


# ==================== Control Plane Endpoints ====================

@router.get("/stats/tiers")
async def get_tier_stats():
    """
    Get tier statistics (Control Plane).

    /stats/tiers – hit/miss, sizes, temps
    """
    fabric = await get_memory_fabric()
    stats = await fabric.get_stats()

    telemetry = stats.get("telemetry", {}).get("telemetry", {})

    return {
        "tier_stats": telemetry.get("tier_stats", {}),
        "overall_hit_rate": telemetry.get("overall_hit_rate", 0),
        "overall_latency_ms": telemetry.get("overall_avg_latency_ms", 0),
        "migration_churn": telemetry.get("migration_churn_5min", 0),
    }


@router.get("/stats/agents")
async def get_agent_stats():
    """
    Get per-agent statistics (Control Plane).

    /stats/agents – per-AI usage, latency
    """
    fabric = await get_memory_fabric()
    stats = await fabric.get_stats()

    telemetry = stats.get("telemetry", {}).get("telemetry", {})

    return {
        "agent_stats": telemetry.get("agent_stats", {}),
        "registered_agents": stats.get("policy", {}).get("agents", {}) if stats.get("policy") else {}
    }


@router.get("/graph/introspect/{memory_id}")
async def introspect_memory(memory_id: str):
    """
    Introspect a memory node (Control Plane).

    /graph/introspect/{id} – connections, activation paths
    """
    fabric = await get_memory_fabric()
    mem_uuid = UUID(memory_id)

    memory = fabric.graph.nodes.get(mem_uuid)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    neighbors = fabric.graph.get_neighbors(mem_uuid)

    return {
        "memory_id": str(memory.id),
        "content_hash": memory.content_hash,
        "tier": memory.tier_meta.current_tier.name,
        "temperature": memory.tier_meta.temperature,
        "access_count": memory.counters.access_count,
        "connections": [
            {
                "target_id": str(tid),
                "relation": edge.relation_type.value,
                "weight": edge.weight
            }
            for tid, edge in neighbors
        ],
        "embeddings": list(memory.embeddings.keys()),
        "policy": {
            "share_scope": memory.policy.share_scope.value,
            "owner": memory.policy.owner_agent_id,
            "clearance": memory.policy.clearance_level
        }
    }


@router.post("/policy/agent")
async def register_agent(request: RegisterAgentRequest):
    """
    Register an AI agent (Control Plane).

    /policy/agent – register new agent
    """
    fabric = await get_memory_fabric()

    try:
        clearance = AccessLevel[request.clearance.upper()]
    except KeyError:
        clearance = AccessLevel.INTERNAL

    profile = await fabric.register_agent(
        agent_id=request.agent_id,
        name=request.name,
        groups=set(request.groups) if request.groups else None,
        clearance=clearance
    )

    if not profile:
        raise HTTPException(status_code=400, detail="Policy engine not enabled")

    return {
        "agent_id": profile.agent_id,
        "name": profile.name,
        "clearance": profile.clearance_level.name,
        "groups": list(profile.groups)
    }


@router.get("/policy/export")
async def export_policy():
    """
    Export policy configuration (Control Plane).

    /policy/export – full policy export
    """
    fabric = await get_memory_fabric()

    if not fabric.policy_engine:
        return {"enabled": False}

    return fabric.policy_engine.export_policy()


@router.post("/graph/connection")
async def add_connection(request: AddConnectionRequest):
    """
    Add a connection between memories (Control Plane).

    /graph/connection – create edge
    """
    fabric = await get_memory_fabric()

    try:
        relation = RelationType(request.relation_type)
    except ValueError:
        relation = RelationType.ASSOCIATED

    success = await fabric.add_connection(
        source_id=UUID(request.source_id),
        target_id=UUID(request.target_id),
        relation_type=relation,
        weight=request.weight,
        agent_id=request.agent_id
    )

    if not success:
        raise HTTPException(status_code=400, detail="Failed to create connection")

    return {"success": True}


@router.get("/orchestrator/jobs")
async def get_orchestrator_jobs():
    """
    Get orchestrator job status (Control Plane).

    /orchestrator/jobs – migration/dedup/index tasks
    """
    fabric = await get_memory_fabric()

    return {
        "migration": {
            "pending": len(fabric.migration_manager.pending_migrations),
            "in_progress": len(fabric.migration_manager.in_progress),
            "stats": fabric.migration_manager.stats
        },
        "deduplication": await fabric.deduplicator.get_deduplication_stats(),
        "cpu_manager": await fabric.cpu_manager.get_resource_stats(),
    }


@router.post("/orchestrator/optimize")
async def trigger_optimization():
    """
    Trigger optimization cycle (Control Plane).

    /orchestrator/optimize – run maintenance
    """
    fabric = await get_memory_fabric()

    # Run dedup
    dedup_result = await fabric.deduplicator.auto_deduplicate(batch_size=100)

    # Decay connections
    await fabric.connection_engine.decay_connections()

    return {
        "deduplication": dedup_result,
        "maintenance": "completed"
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    fabric = await get_memory_fabric()
    return await fabric.get_health()


@router.get("/stats")
async def get_all_stats():
    """Get comprehensive statistics"""
    fabric = await get_memory_fabric()
    return await fabric.get_stats()


# ==================== Tuner Endpoints ====================

@router.get("/tuner/stats")
async def get_tuner_stats():
    """Get auto-tuner statistics"""
    fabric = await get_memory_fabric()
    return fabric.telemetry.get_combined_stats()


@router.post("/tuner/parameter")
async def set_tuning_parameter(
    name: str = Query(...),
    value: float = Query(...)
):
    """Set a tuning parameter manually"""
    fabric = await get_memory_fabric()
    fabric.telemetry.tuner.set_parameter(name, value, reason="api_override")
    return {"parameter": name, "value": value}


@router.post("/tuner/reset")
async def reset_tuner():
    """Reset tuner to defaults"""
    fabric = await get_memory_fabric()
    fabric.telemetry.tuner.reset_to_defaults()
    return {"status": "reset"}
