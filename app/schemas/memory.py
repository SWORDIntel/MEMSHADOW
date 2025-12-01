from pydantic import BaseModel, UUID4, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

# Shared properties
class MemoryBase(BaseModel):
    content: str
    extra_data: Optional[Dict[str, Any]] = {}

# Properties to receive via API on creation
class MemoryCreate(MemoryBase):
    pass

# Properties to receive via API on update
class MemoryUpdate(BaseModel):
    content: Optional[str] = None
    extra_data: Optional[Dict[str, Any]] = None

# Properties for search
class MemorySearch(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None

# Confidence metadata for retrieval results
class ConfidenceMetadata(BaseModel):
    """Metacognitive confidence estimate for a memory retrieval result (Phase 8.3 v2)"""
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score (0-1): P(results sufficient to answer query)")
    meta_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence about the confidence estimate itself")
    similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Semantic similarity to query")
    uncertainty_sources: List[str] = Field(default_factory=list, description="Specific sources of uncertainty (e.g., 'low_similarity', 'stale_result')")
    should_review: bool = Field(False, description="Whether human review is recommended")

    class Config:
        json_schema_extra = {
            "example": {
                "confidence": 0.85,
                "meta_confidence": 0.92,
                "similarity_score": 0.92,
                "uncertainty_sources": [],
                "should_review": False
            }
        }

class MemoryInDBBase(MemoryBase):
    id: UUID4
    user_id: UUID4
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Additional properties to return via API
class MemoryResponse(MemoryInDBBase):
    confidence_metadata: Optional[ConfidenceMetadata] = Field(
        None,
        description="Metacognitive confidence estimate (if enabled)"
    )

# Additional properties stored in DB
class MemoryInDB(MemoryInDBBase):
    content_hash: str
    embedding: Optional[List[float]] = None