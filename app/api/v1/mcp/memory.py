"""
MCP Endpoints for Memory Operations

MCP-compatible endpoints with advanced NLP querying:
- Fuzzy matching
- 2048-dimensional vector search
- Cross-system intelligence correlation
- IoC extraction and analysis
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List

from app.api.dependencies import get_current_active_user, get_db
from app.models.user import User
from app.services.memory_service import MemoryService
from app.schemas.memory import MemoryCreate, MemorySearch
from app.services.advanced_nlp_service import advanced_nlp_service
from app.services.fuzzy_vector_intel import fuzzy_vector_intel
from app.services.vanta_blackwidow.ioc_identifier import ioc_identifier

router = APIRouter()


@router.post("/store")
async def mcp_store_memory(
    *,
    content: str,
    metadata: Dict[str, Any] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    MCP Tool: Store a memory

    Args:
        content: Memory content to store
        metadata: Optional metadata

    Returns:
        Memory ID and confirmation
    """
    memory_service = MemoryService(db)

    memory_in = MemoryCreate(
        content=content,
        extra_data=metadata or {}
    )

    try:
        memory = await memory_service.create_memory(
            user_id=current_user.id,
            content=memory_in.content,
            extra_data=memory_in.extra_data
        )

        # Queue embedding generation
        from app.workers.tasks import generate_embedding_task
        generate_embedding_task.delay(memory_id=str(memory.id), content=content)

        return {
            "tool": "store_memory",
            "status": "success",
            "memory_id": str(memory.id),
            "content_length": len(content),
            "message": "Memory stored successfully"
        }

    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/search")
async def mcp_search_memories(
    *,
    query: str,
    limit: int = 10,
    filters: Dict[str, Any] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    MCP Tool: Search memories

    Args:
        query: Search query
        limit: Maximum number of results
        filters: Optional filters

    Returns:
        List of matching memories
    """
    memory_service = MemoryService(db)

    memories = await memory_service.search_memories(
        user_id=current_user.id,
        query=query,
        filters=filters,
        limit=limit
    )

    return {
        "tool": "search_memories",
        "status": "success",
        "query": query,
        "num_results": len(memories),
        "memories": [
            {
                "id": str(m.id),
                "content": m.content,
                "metadata": m.extra_data,
                "created_at": m.created_at.isoformat()
            }
            for m in memories
        ]
    }


@router.post("/search/advanced")
async def mcp_advanced_search(
    *,
    query: str,
    use_fuzzy: bool = True,
    expand_query: bool = True,
    extract_iocs: bool = True,
    limit: int = 10,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    MCP Tool: Advanced NLP search with 2048-dim vectors & fuzzy matching

    Args:
        query: Search query
        use_fuzzy: Enable fuzzy matching
        expand_query: Enable query expansion
        extract_iocs: Extract IoCs from query
        limit: Maximum results

    Returns:
        Advanced search results
    """
    memory_service = MemoryService(db)

    # Classify query intent
    intent_classification = await advanced_nlp_service.classify_security_intent(query)

    # Query expansion
    queries = [query]
    if expand_query:
        queries = await advanced_nlp_service.expand_query(query)

    # Extract IoCs
    iocs = []
    if extract_iocs:
        iocs = ioc_identifier.extract_iocs(query)

    # Search with all query variants
    all_results = []
    for q in queries:
        memories = await memory_service.search_memories(
            user_id=current_user.id,
            query=q,
            limit=limit * 2
        )
        all_results.extend(memories)

    # Deduplicate
    seen_ids = set()
    unique_results = []
    for m in all_results:
        if m.id not in seen_ids:
            seen_ids.add(m.id)
            unique_results.append({
                'id': str(m.id),
                'content': m.content,
                'metadata': m.extra_data,
                'created_at': m.created_at.isoformat()
            })

    # Semantic reranking
    reranked = await advanced_nlp_service.semantic_rerank(query, unique_results, top_k=limit)

    return {
        "tool": "advanced_search",
        "status": "success",
        "query": {"original": query, "expanded": queries, "intent": intent_classification},
        "iocs_found": [{'type': ioc.type, 'value': ioc.value, 'threat_level': ioc.threat_level} for ioc in iocs],
        "num_results": len(reranked),
        "results": reranked,
        "metadata": {"vector_dim": 2048, "fuzzy_matching": use_fuzzy, "reranked": True}
    }


@router.post("/intelligence/analyze")
async def mcp_intelligence_analysis(
    *,
    text: str,
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    MCP Tool: Comprehensive intelligence analysis (IoC extraction, vectorization, threat assessment)
    """
    analysis = await fuzzy_vector_intel.comprehensive_intelligence_analysis(text)
    return {"tool": "intelligence_analysis", "status": "success", "analysis": analysis}


@router.post("/fuzzy/match")
async def mcp_fuzzy_match(
    *,
    query: str,
    candidates: List[str],
    threshold: float = 0.7,
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    MCP Tool: Fuzzy matching with hybrid string + semantic (2048-dim) scoring
    """
    matches = await fuzzy_vector_intel.fuzzy_match_text(query, candidates, threshold)
    return {"tool": "fuzzy_match", "status": "success", "matches": matches, "vector_dim": 2048}
