"""
Semantic Memory (L3 Tier)

Long-term semantic knowledge storage with maximum fidelity embeddings.
Compressed and stored on cold storage for high capacity.

Based on: HUB_DOCS/MEMSHADOW_INTEGRATION.md
"""

import hashlib
import json
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import structlog

logger = structlog.get_logger()


@dataclass
class SemanticConcept:
    """A semantic concept with relationships"""
    concept_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    data: bytes = b""
    embedding: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Relationships
    related_to: Set[str] = field(default_factory=set)  # concept_ids
    instances: Set[str] = field(default_factory=set)   # item_ids
    
    # Item storage within concept
    item_data: Dict[str, bytes] = field(default_factory=dict)  # item_id -> data
    item_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Versioning
    version: int = 1
    created_at: int = field(default_factory=lambda: int(time.time() * 1e9))
    updated_at: int = field(default_factory=lambda: int(time.time() * 1e9))
    
    # Compression
    is_compressed: bool = False
    original_size: int = 0
    
    def compress(self):
        """Compress the data"""
        if self.is_compressed:
            return
        self.original_size = len(self.data)
        compressed = zlib.compress(self.data, level=9)
        if len(compressed) < len(self.data):
            self.data = compressed
            self.is_compressed = True
    
    def decompress(self) -> bytes:
        """Get decompressed data"""
        if self.is_compressed:
            return zlib.decompress(self.data)
        return self.data
    
    def add_relationship(self, other_concept_id: str):
        """Add a relationship to another concept"""
        self.related_to.add(other_concept_id)
        self.updated_at = int(time.time() * 1e9)
        self.version += 1
    
    def store_item(self, item_id: str, data: bytes, metadata: Optional[Dict[str, Any]] = None):
        """Store an item within this concept"""
        self.instances.add(item_id)
        self.item_data[item_id] = data
        if metadata:
            self.item_metadata[item_id] = metadata
        self.updated_at = int(time.time() * 1e9)
    
    def get_item(self, item_id: str) -> Optional[bytes]:
        """Get item data from concept"""
        return self.item_data.get(item_id)


class SemanticMemory:
    """
    L3 Semantic Memory Tier
    
    Characteristics:
    - Long-term knowledge storage
    - Maximum 4096-dimension embeddings
    - Compressed storage
    - Relationship/graph structure
    - Cold storage backing
    - Consolidation from L2
    
    Implements MEMSHADOW sync interface.
    """
    
    TIER_NAME = "L3_SEMANTIC"
    MAX_DIMENSION = 4096
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        compression_enabled: bool = True,
        max_concepts: int = 100000,
    ):
        self.storage_path = Path(storage_path) if storage_path else None
        self.compression_enabled = compression_enabled
        self.max_concepts = max_concepts
        
        # Concepts: concept_id -> SemanticConcept
        self._concepts: Dict[str, SemanticConcept] = {}
        
        # Item to concept mapping
        self._item_to_concept: Dict[str, str] = {}
        
        # Name index
        self._name_index: Dict[str, str] = {}  # name -> concept_id
        
        # Stats
        self._stats = {
            "concepts_created": 0,
            "items_stored": 0,
            "items_retrieved": 0,
            "relationships_created": 0,
            "bytes_saved": 0,
        }
        
        logger.info(
            "SemanticMemory initialized",
            storage=storage_path,
            compression=compression_enabled,
        )
    
    async def store(
        self,
        item_id: str,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        concept_name: Optional[str] = None,
        embedding: Optional[bytes] = None,
    ) -> bool:
        """
        Store item in semantic memory.
        
        Items can be associated with concepts for organization.
        """
        if concept_name:
            # Get or create concept and store item within it
            concept = await self._get_or_create_concept(concept_name)
            concept.store_item(item_id, data, metadata)
            if embedding:
                concept.embedding = embedding
            self._item_to_concept[item_id] = concept.concept_id
        else:
            # Create standalone concept for item
            concept = SemanticConcept(
                concept_id=item_id,
                data=data,
                embedding=embedding,
                metadata=metadata or {},
            )
            concept.instances.add(item_id)
            concept.item_data[item_id] = data
            if self.compression_enabled:
                concept.compress()
            self._concepts[item_id] = concept
            self._item_to_concept[item_id] = item_id
        
        self._stats["items_stored"] += 1
        
        logger.debug(
            "Stored in L3",
            item_id=item_id,
            concept=concept_name,
        )
        
        return True
    
    async def retrieve(self, item_id: str) -> Optional[bytes]:
        """Retrieve item from semantic memory"""
        concept_id = self._item_to_concept.get(item_id)
        if not concept_id:
            return None
        
        concept = self._concepts.get(concept_id)
        if not concept:
            return None
        
        # Get item-specific data if stored within concept
        item_data = concept.get_item(item_id)
        if item_data:
            self._stats["items_retrieved"] += 1
            return item_data
        
        # Fall back to concept data (for standalone items)
        self._stats["items_retrieved"] += 1
        return concept.decompress()
    
    async def delete(self, item_id: str) -> bool:
        """Delete item from semantic memory"""
        concept_id = self._item_to_concept.pop(item_id, None)
        if not concept_id:
            return False
        
        # Remove from concept instances
        concept = self._concepts.get(concept_id)
        if concept:
            concept.instances.discard(item_id)
            # If concept has no instances and is standalone, remove it
            if not concept.instances and concept_id == item_id:
                del self._concepts[concept_id]
        
        return True
    
    async def list_items(self, since_timestamp: Optional[int] = None) -> List[str]:
        """List all item IDs"""
        if since_timestamp is None:
            return list(self._item_to_concept.keys())
        
        result = []
        for item_id, concept_id in self._item_to_concept.items():
            concept = self._concepts.get(concept_id)
            if concept and concept.created_at >= since_timestamp:
                result.append(item_id)
        return result
    
    async def get_metadata(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for an item"""
        concept_id = self._item_to_concept.get(item_id)
        if not concept_id:
            return None
        
        concept = self._concepts.get(concept_id)
        return concept.metadata if concept else None
    
    async def _get_or_create_concept(self, name: str) -> SemanticConcept:
        """Get existing concept by name or create new one"""
        if name in self._name_index:
            concept_id = self._name_index[name]
            return self._concepts[concept_id]
        
        # Create new concept
        concept = SemanticConcept(name=name)
        self._concepts[concept.concept_id] = concept
        self._name_index[name] = concept.concept_id
        
        self._stats["concepts_created"] += 1
        
        logger.debug("Concept created", name=name, id=concept.concept_id)
        return concept
    
    async def get_concept(self, concept_id: str) -> Optional[SemanticConcept]:
        """Get a concept by ID"""
        return self._concepts.get(concept_id)
    
    async def get_concept_by_name(self, name: str) -> Optional[SemanticConcept]:
        """Get a concept by name"""
        concept_id = self._name_index.get(name)
        return self._concepts.get(concept_id) if concept_id else None
    
    async def add_relationship(
        self,
        concept_id: str,
        related_concept_id: str,
        bidirectional: bool = True,
    ):
        """Add a relationship between concepts"""
        concept = self._concepts.get(concept_id)
        if not concept:
            return
        
        concept.add_relationship(related_concept_id)
        
        if bidirectional:
            related = self._concepts.get(related_concept_id)
            if related:
                related.add_relationship(concept_id)
        
        self._stats["relationships_created"] += 1
    
    async def get_related_concepts(self, concept_id: str) -> List[SemanticConcept]:
        """Get concepts related to a given concept"""
        concept = self._concepts.get(concept_id)
        if not concept:
            return []
        
        return [
            self._concepts[rid]
            for rid in concept.related_to
            if rid in self._concepts
        ]
    
    async def consolidate_from_episodic(
        self,
        episode_data: Dict[str, bytes],
        concept_name: str,
    ):
        """
        Consolidate episodic memories into semantic knowledge.
        
        This is the L2 -> L3 transition process.
        """
        concept = await self._get_or_create_concept(concept_name)
        
        for item_id, data in episode_data.items():
            concept.instances.add(item_id)
            self._item_to_concept[item_id] = concept.concept_id
        
        # Merge data (simplistic: just concatenate)
        merged_data = b"".join(episode_data.values())
        concept.data = merged_data
        
        if self.compression_enabled:
            original_size = len(concept.data)
            concept.compress()
            self._stats["bytes_saved"] += original_size - len(concept.data)
        
        logger.info(
            "Consolidated to L3",
            concept=concept_name,
            items=len(episode_data),
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory tier statistics"""
        total_size = sum(len(c.data) for c in self._concepts.values())
        compressed_count = sum(1 for c in self._concepts.values() if c.is_compressed)
        
        return {
            "tier": self.TIER_NAME,
            "max_concepts": self.max_concepts,
            "current_concepts": len(self._concepts),
            "total_items": len(self._item_to_concept),
            "compressed_concepts": compressed_count,
            "total_size_bytes": total_size,
            **self._stats,
        }
    
    def clear(self):
        """Clear all semantic memory"""
        self._concepts.clear()
        self._item_to_concept.clear()
        self._name_index.clear()
        logger.info("Semantic memory cleared")


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "SemanticConcept",
    "SemanticMemory",
]
