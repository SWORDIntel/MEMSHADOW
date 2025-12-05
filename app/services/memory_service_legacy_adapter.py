"""
Legacy Memory Service Adapter

Thin adapter layer that wraps DSMILSYSTEM memory service to maintain
backward compatibility with existing MEMSHADOW APIs.

Provides default layer/device mapping and clearance token handling.
"""
import hashlib
from typing import List, Dict, Any, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.services.memory_service_dsmil import MemoryServiceDSMIL
from app.models.memory import Memory as LegacyMemory
from app.schemas.memory import MemoryResponse

logger = structlog.get_logger()


class LegacyMemoryAdapter:
    """
    Adapter for legacy MEMSHADOW API.
    
    Maps legacy user-scoped operations to DSMILSYSTEM layer/device semantics:
    - Default layer: 6 (Application layer)
    - Default device: 0
    - Default clearance: "UNCLASSIFIED"
    """
    
    DEFAULT_LAYER = 6  # Application layer
    DEFAULT_DEVICE = 0
    DEFAULT_CLEARANCE = "UNCLASSIFIED"
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.dsmil_service = MemoryServiceDSMIL(db)
    
    async def create_memory(
        self,
        user_id: UUID,
        content: str,
        extra_data: Dict[str, Any]
    ) -> LegacyMemory:
        """
        Create memory using legacy API (backward compatible).
        
        Maps to DSMILSYSTEM store_memory with defaults.
        """
        # Map to DSMILSYSTEM
        payload = {
            "content": content,
            "user_id": str(user_id),
            "extra_data": extra_data
        }
        
        memory_id = await self.dsmil_service.store_memory(
            layer_id=self.DEFAULT_LAYER,
            device_id=self.DEFAULT_DEVICE,
            payload=payload,
            tags=extra_data.get("tags", []),
            ttl=None,
            clearance=self.DEFAULT_CLEARANCE,
            context={"legacy_api": True}
        )
        
        # Return legacy memory object (read from DB)
        memory = await self.db.get(LegacyMemory, memory_id)
        if not memory:
            # Fallback: create legacy memory record
            memory = LegacyMemory(
                id=memory_id,
                user_id=user_id,
                content=content,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                extra_data=extra_data
            )
            self.db.add(memory)
            await self.db.commit()
            await self.db.refresh(memory)
        
        logger.info(
            "Legacy memory created (mapped to DSMILSYSTEM)",
            memory_id=str(memory_id),
            user_id=str(user_id),
            layer_id=self.DEFAULT_LAYER,
            device_id=self.DEFAULT_DEVICE
        )
        
        return memory
    
    async def search_memories(
        self,
        user_id: UUID,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> tuple[List[LegacyMemory], Optional[List]]:
        """
        Search memories using legacy API (backward compatible).
        
        Maps to DSMILSYSTEM search_memory with defaults.
        """
        # Map to DSMILSYSTEM
        results = await self.dsmil_service.search_memory(
            layer_id=self.DEFAULT_LAYER,
            device_id=self.DEFAULT_DEVICE,
            query=query,
            k=limit + offset,  # Get more to handle offset
            filters=filters,
            clearance=self.DEFAULT_CLEARANCE
        )
        
        # Filter by user_id and apply offset
        user_results = [
            r for r in results
            if r.get("user_id") == str(user_id)
        ][offset:offset + limit]
        
        # Convert to legacy memory objects
        memories = []
        for result in user_results:
            # Try to get from DB
            memory = await self.db.get(LegacyMemory, UUID(result["id"]))
            if memory:
                memories.append(memory)
            else:
                # Create legacy memory object from result
                memory = LegacyMemory(
                    id=UUID(result["id"]),
                    user_id=UUID(result["user_id"]),
                    content=result["content"],
                    content_hash=result["content_hash"],
                    extra_data=result.get("extra_data", {}),
                    embedding=result.get("embedding")
                )
                memories.append(memory)
        
        logger.info(
            "Legacy memory search (mapped to DSMILSYSTEM)",
            user_id=str(user_id),
            query=query,
            results_count=len(memories)
        )
        
        return memories, None  # No confidence metadata for legacy API
    
    async def get_memory(
        self,
        memory_id: UUID,
        user_id: UUID
    ) -> Optional[LegacyMemory]:
        """Get memory using legacy API"""
        memory = await self.db.get(LegacyMemory, memory_id)
        if memory and memory.user_id == user_id:
            return memory
        return None
    
    async def update_memory(
        self,
        memory_id: UUID,
        user_id: UUID,
        updates: Dict[str, Any]
    ) -> Optional[LegacyMemory]:
        """Update memory using legacy API"""
        memory = await self.db.get(LegacyMemory, memory_id)
        if not memory or memory.user_id != user_id:
            return None
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(memory, key):
                setattr(memory, key, value)
        
        await self.db.commit()
        await self.db.refresh(memory)
        
        return memory
    
    async def delete_memory(
        self,
        memory_id: UUID,
        user_id: UUID
    ) -> bool:
        """Delete memory using legacy API"""
        memory = await self.db.get(LegacyMemory, memory_id)
        if not memory or memory.user_id != user_id:
            return False
        
        # Delete via DSMILSYSTEM service
        await self.dsmil_service.delete_memory(
            memory_id=memory_id,
            layer_id=self.DEFAULT_LAYER,
            device_id=self.DEFAULT_DEVICE,
            clearance=self.DEFAULT_CLEARANCE
        )
        
        return True
