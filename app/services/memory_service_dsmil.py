"""
DSMILSYSTEM-aligned Memory Service

Refactored memory service with:
- Layer/device semantics
- Clearance token enforcement
- Multi-tier storage (hot/warm/cold)
- Event bus integration
- Upward-only flow enforcement
"""
import hashlib
import uuid
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
import structlog
import numpy as np
import json

from app.models.memory_dsmil import Memory, MemoryTier
from app.db.chromadb import chroma_client
from app.db.redis import redis_client
from app.services.embedding_service import EmbeddingService
from app.services.dsmil import (
    event_bus,
    clearance_validator,
    warm_tier,
    AccessDecision
)
from app.core.config import settings

logger = structlog.get_logger()


class MemoryServiceDSMIL:
    """
    DSMILSYSTEM-aligned memory service.
    
    All operations require:
    - layer_id: Layer 2-9
    - device_id: Device 0-103
    - clearance_token: Access control token
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedding_service = EmbeddingService()
    
    def _validate_layer(self, layer_id: int) -> bool:
        """Validate layer ID"""
        if not (2 <= layer_id <= 9):
            raise ValueError(f"Invalid layer_id: {layer_id}. Must be between 2 and 9.")
        return True
    
    def _validate_device(self, device_id: int) -> bool:
        """Validate device ID"""
        if not (0 <= device_id <= 103):
            raise ValueError(f"Invalid device_id: {device_id}. Must be between 0 and 103.")
        return True
    
    async def store_memory(
        self,
        layer_id: int,
        device_id: int,
        payload: Dict[str, Any],
        tags: List[str],
        ttl: Optional[int],
        clearance: str,
        context: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """
        Store memory with DSMILSYSTEM semantics.
        
        Args:
            layer_id: Layer ID (2-9)
            device_id: Device ID (0-103)
            payload: Memory payload (must contain 'content' and 'user_id')
            tags: List of tags
            ttl: Time-to-live in seconds (optional)
            clearance: Clearance token
            context: Additional context (correlation_id, etc.)
            
        Returns:
            Memory ID
        """
        # Validate inputs
        self._validate_layer(layer_id)
        self._validate_device(device_id)
        
        content = payload.get("content")
        user_id = payload.get("user_id")
        
        if not content:
            raise ValueError("Payload must contain 'content'")
        if not user_id:
            raise ValueError("Payload must contain 'user_id'")
        
        if isinstance(user_id, str):
            user_id = UUID(user_id)
        
        # Generate memory ID and hash
        memory_id = uuid.uuid4()
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Check for duplicates
        existing = await self.db.execute(
            select(Memory).where(
                and_(
                    Memory.layer_id == layer_id,
                    Memory.device_id == device_id,
                    Memory.content_hash == content_hash,
                    Memory.user_id == user_id
                )
            )
        )
        if existing.scalar_one_or_none():
            raise ValueError("Duplicate memory content")
        
        # Generate embedding
        embedding_vector = await self.embedding_service.generate_embedding(content)
        
        # Extract context
        correlation_id = context.get("correlation_id") if context else str(uuid.uuid4())
        roe_metadata = payload.get("roe_metadata", {})
        
        # Determine initial tier (hot for new memories)
        tier = MemoryTier.HOT
        
        # Create memory record
        memory = Memory(
            id=memory_id,
            layer_id=layer_id,
            device_id=device_id,
            clearance_token=clearance,
            user_id=user_id,
            content=content,
            content_hash=content_hash,
            embedding=embedding_vector.tolist() if isinstance(embedding_vector, np.ndarray) else embedding_vector,
            tags=tags,
            extra_data=payload.get("extra_data", {}),
            roe_metadata=roe_metadata,
            correlation_id=correlation_id,
            tier=tier
        )
        
        self.db.add(memory)
        await self.db.commit()
        await self.db.refresh(memory)
        
        # Store in hot tier (Redis)
        await self._store_in_hot_tier(memory, ttl)
        
        # Emit event
        await event_bus.emit_memory_event(
            event_type="store",
            layer_id=layer_id,
            device_id=device_id,
            memory_id=str(memory_id),
            correlation_id=correlation_id,
            metadata={"tier": tier.value}
        )
        
        logger.info(
            "Memory stored",
            memory_id=str(memory_id),
            layer_id=layer_id,
            device_id=device_id,
            clearance=clearance,
            tier=tier.value
        )
        
        return memory_id
    
    async def _store_in_hot_tier(self, memory: Memory, ttl: Optional[int]):
        """Store memory in hot tier (Redis)"""
        try:
            redis = await redis_client.get_client()
            
            key = f"memory:hot:{memory.layer_id}:{memory.device_id}:{memory.id}"
            value = {
                "id": str(memory.id),
                "layer_id": memory.layer_id,
                "device_id": memory.device_id,
                "clearance_token": memory.clearance_token,
                "user_id": str(memory.user_id),
                "content": memory.content,
                "content_hash": memory.content_hash,
                "tags": memory.tags,
                "extra_data": memory.extra_data,
                "roe_metadata": memory.roe_metadata,
                "correlation_id": memory.correlation_id,
                "created_at": memory.created_at.isoformat() if memory.created_at else None
            }
            
            await redis.setex(
                key,
                ttl or 3600,  # Default 1 hour TTL
                json.dumps(value)
            )
            
        except Exception as e:
            logger.warning("Failed to store in hot tier", error=str(e))
    
    async def search_memory(
        self,
        layer_id: int,
        device_id: int,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        clearance: str
    ) -> List[Dict[str, Any]]:
        """
        Search memories with DSMILSYSTEM semantics.
        
        Args:
            layer_id: Layer ID (2-9)
            device_id: Device ID (0-103)
            query: Search query
            k: Number of results
            filters: Additional filters
            clearance: Clearance token
            
        Returns:
            List of memory dicts
        """
        # Validate inputs
        self._validate_layer(layer_id)
        self._validate_device(device_id)
        
        # Search across tiers (hot -> warm -> cold)
        results = []
        
        # 1. Search hot tier (Redis)
        hot_results = await self._search_hot_tier(layer_id, device_id, query, k, clearance)
        results.extend(hot_results)
        
        # 2. Search warm tier (SQLite)
        if len(results) < k:
            warm_results = await self._search_warm_tier(layer_id, device_id, query, k - len(results), clearance)
            results.extend(warm_results)
        
        # 3. Search cold tier (PostgreSQL + ChromaDB)
        if len(results) < k:
            cold_results = await self._search_cold_tier(layer_id, device_id, query, k - len(results), filters, clearance)
            results.extend(cold_results)
        
        # Emit event
        await event_bus.emit_memory_event(
            event_type="search",
            layer_id=layer_id,
            device_id=device_id,
            memory_id="",
            correlation_id=str(uuid.uuid4()),
            metadata={"query": query, "results_count": len(results)}
        )
        
        logger.info(
            "Memory search completed",
            layer_id=layer_id,
            device_id=device_id,
            query=query,
            results_count=len(results)
        )
        
        return results[:k]
    
    async def _search_hot_tier(
        self,
        layer_id: int,
        device_id: int,
        query: str,
        k: int,
        clearance: str
    ) -> List[Dict[str, Any]]:
        """Search hot tier (Redis)"""
        try:
            redis = await redis_client.get_client()
            
            # Simple text search in hot tier (limited functionality)
            pattern = f"memory:hot:{layer_id}:{device_id}:*"
            keys = []
            cursor = 0
            
            # Scan for keys (limited to avoid performance issues)
            while len(keys) < 100:  # Limit scan
                cursor, batch = await redis.scan(cursor, match=pattern, count=10)
                keys.extend(batch)
                if cursor == 0:
                    break
            
            results = []
            for key in keys[:k]:
                value = await redis.get(key)
                if value:
                    memory_data = json.loads(value)
                    # Check clearance
                    decision, reason = clearance_validator.validate_access(
                        clearance,
                        memory_data["layer_id"],
                        layer_id,
                        memory_data["clearance_token"],
                        memory_data.get("roe_metadata", {}),
                        "read"
                    )
                    if decision == AccessDecision.ALLOWED:
                        results.append(memory_data)
            
            return results
            
        except Exception as e:
            logger.warning("Hot tier search failed", error=str(e))
            return []
    
    async def _search_warm_tier(
        self,
        layer_id: int,
        device_id: int,
        query: str,
        k: int,
        clearance: str
    ) -> List[Dict[str, Any]]:
        """Search warm tier (SQLite)"""
        try:
            memories = await warm_tier.search(
                layer_id=layer_id,
                device_id=device_id,
                clearance_token=None,  # Check clearance after retrieval
                limit=k * 2  # Get more to filter by clearance
            )
            
            results = []
            for memory in memories:
                # Check clearance
                decision, reason = clearance_validator.validate_access(
                    clearance,
                    memory["layer_id"],
                    layer_id,
                    memory["clearance_token"],
                    memory.get("roe_metadata", {}),
                    "read"
                )
                if decision == AccessDecision.ALLOWED:
                    results.append(memory)
                    if len(results) >= k:
                        break
            
            return results
            
        except Exception as e:
            logger.warning("Warm tier search failed", error=str(e))
            return []
    
    async def _search_cold_tier(
        self,
        layer_id: int,
        device_id: int,
        query: str,
        k: int,
        filters: Optional[Dict[str, Any]],
        clearance: str
    ) -> List[Dict[str, Any]]:
        """Search cold tier (PostgreSQL + ChromaDB)"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(query)
            
            # Build query
            query_stmt = select(Memory).where(
                and_(
                    Memory.layer_id == layer_id,
                    Memory.device_id == device_id
                )
            )
            
            # Apply filters
            if filters:
                if "tags" in filters:
                    # PostgreSQL JSONB contains check
                    query_stmt = query_stmt.where(
                        Memory.tags.contains(filters["tags"])
                    )
            
            # Execute query
            result = await self.db.execute(query_stmt.limit(k * 2))
            memories = result.scalars().all()
            
            # Filter by clearance and compute similarity
            scored_memories = []
            for memory in memories:
                # Check clearance
                decision, reason = clearance_validator.validate_access(
                    clearance,
                    memory.layer_id,
                    layer_id,
                    memory.clearance_token,
                    memory.roe_metadata,
                    "read"
                )
                
                if decision == AccessDecision.ALLOWED:
                    # Compute similarity if embedding exists
                    similarity = 0.0
                    if memory.embedding and query_embedding is not None:
                        # Cosine similarity
                        mem_vec = np.array(memory.embedding)
                        query_vec = np.array(query_embedding)
                        similarity = np.dot(mem_vec, query_vec) / (
                            np.linalg.norm(mem_vec) * np.linalg.norm(query_vec)
                        )
                    
                    scored_memories.append({
                        "memory": memory,
                        "similarity": similarity
                    })
            
            # Sort by similarity
            scored_memories.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Convert to dicts
            results = []
            for item in scored_memories[:k]:
                memory = item["memory"]
                results.append({
                    "id": str(memory.id),
                    "layer_id": memory.layer_id,
                    "device_id": memory.device_id,
                    "clearance_token": memory.clearance_token,
                    "user_id": str(memory.user_id),
                    "content": memory.content,
                    "content_hash": memory.content_hash,
                    "tags": memory.tags,
                    "extra_data": memory.extra_data,
                    "roe_metadata": memory.roe_metadata,
                    "correlation_id": memory.correlation_id,
                    "tier": memory.tier.value,
                    "similarity": item["similarity"],
                    "created_at": memory.created_at.isoformat() if memory.created_at else None,
                    "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                    "accessed_at": memory.accessed_at.isoformat() if memory.accessed_at else None,
                })
            
            return results
            
        except Exception as e:
            logger.error("Cold tier search failed", error=str(e))
            return []
    
    async def delete_memory(
        self,
        memory_id: UUID,
        layer_id: int,
        device_id: int,
        clearance: str
    ) -> bool:
        """
        Delete memory with clearance check.
        
        Args:
            memory_id: Memory ID
            layer_id: Layer ID
            device_id: Device ID
            clearance: Clearance token
            
        Returns:
            True if deleted successfully
        """
        # Validate inputs
        self._validate_layer(layer_id)
        self._validate_device(device_id)
        
        # Get memory
        memory = await self.db.get(Memory, memory_id)
        if not memory:
            return False
        
        # Check clearance
        decision, reason = clearance_validator.validate_access(
            clearance,
            memory.layer_id,
            layer_id,
            memory.clearance_token,
            memory.roe_metadata,
            "delete"
        )
        
        if decision != AccessDecision.ALLOWED:
            logger.warning(
                "Memory delete denied",
                memory_id=str(memory_id),
                reason=reason,
                layer_id=layer_id,
                device_id=device_id
            )
            raise PermissionError(f"Access denied: {reason}")
        
        # Delete from all tiers
        await self._delete_from_hot_tier(memory)
        await warm_tier.delete(memory.layer_id, str(memory_id))
        
        # Delete from database
        await self.db.delete(memory)
        await self.db.commit()
        
        # Emit event
        await event_bus.emit_memory_event(
            event_type="delete",
            layer_id=layer_id,
            device_id=device_id,
            memory_id=str(memory_id),
            correlation_id=str(uuid.uuid4())
        )
        
        logger.info(
            "Memory deleted",
            memory_id=str(memory_id),
            layer_id=layer_id,
            device_id=device_id
        )
        
        return True
    
    async def _delete_from_hot_tier(self, memory: Memory):
        """Delete memory from hot tier"""
        try:
            redis = await redis_client.get_client()
            key = f"memory:hot:{memory.layer_id}:{memory.device_id}:{memory.id}"
            await redis.delete(key)
        except Exception as e:
            logger.warning("Failed to delete from hot tier", error=str(e))
    
    async def compact_memory(
        self,
        layer_id: int,
        device_id: int,
        clearance: str
    ) -> Dict[str, Any]:
        """
        Compact memory (promote hot->warm->cold, remove old entries).
        
        Args:
            layer_id: Layer ID
            device_id: Device ID
            clearance: Clearance token
            
        Returns:
            Compaction results
        """
        # Validate inputs
        self._validate_layer(layer_id)
        self._validate_device(device_id)
        
        # TODO: Implement compaction logic
        # - Promote frequently accessed hot memories to warm
        # - Promote warm memories to cold
        # - Remove expired entries
        
        logger.info(
            "Memory compaction",
            layer_id=layer_id,
            device_id=device_id
        )
        
        return {
            "promoted_hot_to_warm": 0,
            "promoted_warm_to_cold": 0,
            "removed_expired": 0
        }
