"""
MCP Endpoints for Memory Operations

MCP-compatible endpoints for memory storage and retrieval.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List

from app.api.dependencies import get_current_active_user, get_db
from app.models.user import User
from app.services.memory_service import MemoryService
from app.schemas.memory import MemoryCreate, MemorySearch

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
