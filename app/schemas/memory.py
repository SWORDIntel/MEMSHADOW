from pydantic import BaseModel, UUID4
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

class MemoryInDBBase(MemoryBase):
    id: UUID4
    user_id: UUID4
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Additional properties to return via API
class MemoryResponse(MemoryInDBBase):
    pass

# Additional properties stored in DB
class MemoryInDB(MemoryInDBBase):
    content_hash: str
    embedding: Optional[List[float]] = None