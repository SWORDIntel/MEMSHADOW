"""
DSMILSYSTEM Canonical Memory API

Clean, minimal API surface aligned with DSMILSYSTEM architecture:
- store_memory(layer, device, payload, tags, ttl, clearance, context)
- search_memory(layer, device, query, k, filters, clearance)
- delete_memory(memory_id, layer, device, clearance)
- compact_memory(layer, device, clearance)
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field, validator
from uuid import UUID
import structlog

from app.api.dependencies import get_db
from app.services.memory_service_dsmil import MemoryServiceDSMIL

router = APIRouter(prefix="/dsmil/memory", tags=["DSMILSYSTEM Memory"])
logger = structlog.get_logger()


class StoreMemoryRequest(BaseModel):
    """Request to store memory"""
    layer_id: int = Field(..., ge=2, le=9, description="Layer ID (2-9)")
    device_id: int = Field(..., ge=0, le=103, description="Device ID (0-103)")
    payload: Dict[str, Any] = Field(..., description="Memory payload (must contain 'content' and 'user_id')")
    tags: List[str] = Field(default_factory=list, description="List of tags")
    ttl: Optional[int] = Field(None, ge=1, description="Time-to-live in seconds")
    clearance: str = Field(..., description="Clearance token")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context (correlation_id, etc.)")
    
    @validator("payload")
    def validate_payload(cls, v):
        if "content" not in v:
            raise ValueError("Payload must contain 'content'")
        if "user_id" not in v:
            raise ValueError("Payload must contain 'user_id'")
        return v


class StoreMemoryResponse(BaseModel):
    """Response from storing memory"""
    memory_id: UUID
    layer_id: int
    device_id: int
    tier: str
    correlation_id: str


class SearchMemoryRequest(BaseModel):
    """Request to search memory"""
    layer_id: int = Field(..., ge=2, le=9, description="Layer ID (2-9)")
    device_id: int = Field(..., ge=0, le=103, description="Device ID (0-103)")
    query: str = Field(..., description="Search query")
    k: int = Field(10, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")
    clearance: str = Field(..., description="Clearance token")


class MemoryResult(BaseModel):
    """Memory search result"""
    id: str
    layer_id: int
    device_id: int
    clearance_token: str
    user_id: str
    content: str
    content_hash: str
    tags: List[str]
    extra_data: Dict[str, Any]
    roe_metadata: Dict[str, Any]
    correlation_id: Optional[str]
    tier: str
    similarity: Optional[float] = None
    created_at: Optional[str]
    updated_at: Optional[str]
    accessed_at: Optional[str]


class SearchMemoryResponse(BaseModel):
    """Response from searching memory"""
    results: List[MemoryResult]
    total: int
    layer_id: int
    device_id: int


class DeleteMemoryRequest(BaseModel):
    """Request to delete memory"""
    layer_id: int = Field(..., ge=2, le=9, description="Layer ID (2-9)")
    device_id: int = Field(..., ge=0, le=103, description="Device ID (0-103)")
    clearance: str = Field(..., description="Clearance token")


class CompactMemoryRequest(BaseModel):
    """Request to compact memory"""
    layer_id: int = Field(..., ge=2, le=9, description="Layer ID (2-9)")
    device_id: int = Field(..., ge=0, le=103, description="Device ID (0-103)")
    clearance: str = Field(..., description="Clearance token")


class CompactMemoryResponse(BaseModel):
    """Response from compacting memory"""
    promoted_hot_to_warm: int
    promoted_warm_to_cold: int
    removed_expired: int


@router.post("/store", response_model=StoreMemoryResponse, status_code=status.HTTP_201_CREATED)
async def store_memory(
    request: StoreMemoryRequest,
    db: AsyncSession = Depends(get_db)
) -> StoreMemoryResponse:
    """
    Store memory with DSMILSYSTEM semantics.
    
    All memories are tagged with:
    - layer_id: Layer 2-9
    - device_id: Device 0-103
    - clearance_token: Access control token
    
    Memories are stored in hot tier (Redis) initially, then promoted to warm/cold based on access patterns.
    """
    memory_service = MemoryServiceDSMIL(db)
    
    try:
        memory_id = await memory_service.store_memory(
            layer_id=request.layer_id,
            device_id=request.device_id,
            payload=request.payload,
            tags=request.tags,
            ttl=request.ttl,
            clearance=request.clearance,
            context=request.context
        )
        
        correlation_id = request.context.get("correlation_id") if request.context else str(memory_id)
        
        logger.info(
            "Memory stored via DSMILSYSTEM API",
            memory_id=str(memory_id),
            layer_id=request.layer_id,
            device_id=request.device_id
        )
        
        return StoreMemoryResponse(
            memory_id=memory_id,
            layer_id=request.layer_id,
            device_id=request.device_id,
            tier="hot",
            correlation_id=correlation_id
        )
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to store memory", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/search", response_model=SearchMemoryResponse)
async def search_memory(
    request: SearchMemoryRequest,
    db: AsyncSession = Depends(get_db)
) -> SearchMemoryResponse:
    """
    Search memories with DSMILSYSTEM semantics.
    
    Searches across all tiers (hot -> warm -> cold) and enforces:
    - Clearance token validation
    - Upward-only flow rules
    - ROE (Rules of Engagement) enforcement
    """
    memory_service = MemoryServiceDSMIL(db)
    
    try:
        results = await memory_service.search_memory(
            layer_id=request.layer_id,
            device_id=request.device_id,
            query=request.query,
            k=request.k,
            filters=request.filters,
            clearance=request.clearance
        )
        
        logger.info(
            "Memory search via DSMILSYSTEM API",
            layer_id=request.layer_id,
            device_id=request.device_id,
            query=request.query,
            results_count=len(results)
        )
        
        # Convert to response models
        memory_results = [
            MemoryResult(**result)
            for result in results
        ]
        
        return SearchMemoryResponse(
            results=memory_results,
            total=len(memory_results),
            layer_id=request.layer_id,
            device_id=request.device_id
        )
        
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error("Failed to search memory", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: UUID,
    request: DeleteMemoryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete memory with clearance check.
    
    Requires:
    - Valid clearance token
    - Proper layer/device context
    - Access permission
    """
    memory_service = MemoryServiceDSMIL(db)
    
    try:
        success = await memory_service.delete_memory(
            memory_id=memory_id,
            layer_id=request.layer_id,
            device_id=request.device_id,
            clearance=request.clearance
        )
        
        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")
        
        logger.info(
            "Memory deleted via DSMILSYSTEM API",
            memory_id=str(memory_id),
            layer_id=request.layer_id,
            device_id=request.device_id
        )
        
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error("Failed to delete memory", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/compact", response_model=CompactMemoryResponse)
async def compact_memory(
    request: CompactMemoryRequest,
    db: AsyncSession = Depends(get_db)
) -> CompactMemoryResponse:
    """
    Compact memory (promote hot->warm->cold, remove expired entries).
    
    Requires clearance token for authorization.
    """
    memory_service = MemoryServiceDSMIL(db)
    
    try:
        result = await memory_service.compact_memory(
            layer_id=request.layer_id,
            device_id=request.device_id,
            clearance=request.clearance
        )
        
        logger.info(
            "Memory compaction via DSMILSYSTEM API",
            layer_id=request.layer_id,
            device_id=request.device_id,
            result=result
        )
        
        return CompactMemoryResponse(**result)
        
    except Exception as e:
        logger.error("Failed to compact memory", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
