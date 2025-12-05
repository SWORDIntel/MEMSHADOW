#!/usr/bin/env python3
"""
ChromaDB Persistent Backend for DSMIL Brain Vector Database

Provides persistent vector storage with:
- Disk persistence (survives restarts)
- Distributed collection support
- Native embedding functions
- Metadata filtering
- Efficient similarity search at scale

Usage:
    from ai.brain.vectors.chromadb_backend import ChromaDBBackend

    backend = ChromaDBBackend(persist_path="/data/brain_vectors")
    backend.add_vectors(vectors, metadata)
    results = backend.search(query_vector, k=10)
"""

import os
import json
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Tuple
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import chromadb
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available. Install with: pip install chromadb")


@dataclass
class VectorMetadata:
    """Metadata for stored vectors"""
    source: str = ""
    entity_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created: str = ""
    vector_type: str = "generic"
    confidence: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for ChromaDB storage"""
        return {
            "source": self.source,
            "entity_ids": json.dumps(self.entity_ids),
            "tags": json.dumps(self.tags),
            "created": self.created or datetime.now(timezone.utc).isoformat(),
            "vector_type": self.vector_type,
            "confidence": self.confidence,
            "extra": json.dumps(self.extra),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VectorMetadata":
        """Create from ChromaDB stored dict"""
        return cls(
            source=d.get("source", ""),
            entity_ids=json.loads(d.get("entity_ids", "[]")),
            tags=json.loads(d.get("tags", "[]")),
            created=d.get("created", ""),
            vector_type=d.get("vector_type", "generic"),
            confidence=d.get("confidence", 1.0),
            extra=json.loads(d.get("extra", "{}")),
        )


class ChromaDBBackend:
    """
    ChromaDB-backed persistent vector storage.

    Features:
    - Persistent storage to disk
    - Multiple collections for different vector types
    - Metadata filtering
    - Efficient ANN search
    - Automatic embedding support

    Args:
        persist_path: Directory for persistent storage (None for in-memory)
        collection_name: Name of the default collection
        embedding_function: Optional custom embedding function
    """

    # Collection names for different vector types
    COLLECTIONS = {
        "default": "brain_vectors",
        "temporal": "temporal_vectors",
        "behavioral": "behavioral_vectors",
        "multimodal": "multimodal_vectors",
        "knowledge": "knowledge_graph_vectors",
        "threat": "threat_vectors",
    }

    def __init__(self, persist_path: Optional[str] = None,
                 collection_name: str = "brain_vectors",
                 embedding_function: Any = None):
        """Initialize ChromaDB backend"""

        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not installed. Run: pip install chromadb")

        self.persist_path = persist_path
        self.collection_name = collection_name

        # Initialize ChromaDB client
        if persist_path:
            Path(persist_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=persist_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            logger.info(f"ChromaDB initialized with persistence at: {persist_path}")
        else:
            self._client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info("ChromaDB initialized in-memory mode")

        # Get or create default collection
        self._embedding_function = embedding_function
        self._collections: Dict[str, Any] = {}

        self._default_collection = self._get_or_create_collection(collection_name)

    def _get_or_create_collection(self, name: str) -> Any:
        """Get or create a ChromaDB collection"""
        if name in self._collections:
            return self._collections[name]

        collection = self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        self._collections[name] = collection
        return collection

    def add_vector(self, vector_id: str, vector: List[float],
                  metadata: Optional[VectorMetadata] = None,
                  collection: str = None) -> bool:
        """
        Add a single vector to the database.

        Args:
            vector_id: Unique identifier
            vector: Vector embedding
            metadata: Optional metadata
            collection: Collection name (default: brain_vectors)

        Returns:
            True if successful
        """
        coll = self._get_or_create_collection(collection or self.collection_name)

        try:
            meta_dict = metadata.to_dict() if metadata else {}

            coll.add(
                ids=[vector_id],
                embeddings=[vector],
                metadatas=[meta_dict],
            )

            return True

        except Exception as e:
            logger.error(f"Failed to add vector {vector_id}: {e}")
            return False

    def add_vectors(self, vector_ids: List[str], vectors: List[List[float]],
                   metadatas: Optional[List[VectorMetadata]] = None,
                   collection: str = None) -> int:
        """
        Add multiple vectors in batch.

        Args:
            vector_ids: List of unique identifiers
            vectors: List of vector embeddings
            metadatas: Optional list of metadata
            collection: Collection name

        Returns:
            Number of vectors added
        """
        coll = self._get_or_create_collection(collection or self.collection_name)

        try:
            meta_dicts = [m.to_dict() for m in metadatas] if metadatas else None

            coll.add(
                ids=vector_ids,
                embeddings=vectors,
                metadatas=meta_dicts,
            )

            return len(vector_ids)

        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            return 0

    def search(self, query_vector: List[float], k: int = 10,
              where: Optional[Dict] = None,
              collection: str = None) -> List[Tuple[str, float, VectorMetadata]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding
            k: Number of results
            where: Optional metadata filter
            collection: Collection name

        Returns:
            List of (vector_id, distance, metadata) tuples
        """
        coll = self._get_or_create_collection(collection or self.collection_name)

        try:
            results = coll.query(
                query_embeddings=[query_vector],
                n_results=k,
                where=where,
                include=["distances", "metadatas"],
            )

            output = []
            if results["ids"] and results["ids"][0]:
                ids = results["ids"][0]
                distances = results["distances"][0] if results["distances"] else [0] * len(ids)
                metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)

                for vid, dist, meta in zip(ids, distances, metadatas):
                    # Convert distance to similarity (cosine: 1 - distance)
                    similarity = 1.0 - dist
                    metadata = VectorMetadata.from_dict(meta)
                    output.append((vid, similarity, metadata))

            return output

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_by_metadata(self, where: Dict, k: int = 100,
                          collection: str = None) -> List[Tuple[str, VectorMetadata]]:
        """
        Search by metadata only (no vector similarity).

        Args:
            where: Metadata filter
            k: Maximum results
            collection: Collection name

        Returns:
            List of (vector_id, metadata) tuples
        """
        coll = self._get_or_create_collection(collection or self.collection_name)

        try:
            results = coll.get(
                where=where,
                limit=k,
                include=["metadatas"],
            )

            output = []
            if results["ids"]:
                for vid, meta in zip(results["ids"], results["metadatas"] or [{}] * len(results["ids"])):
                    metadata = VectorMetadata.from_dict(meta)
                    output.append((vid, metadata))

            return output

        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []

    def get_vector(self, vector_id: str, collection: str = None) -> Optional[Tuple[List[float], VectorMetadata]]:
        """
        Get a specific vector by ID.

        Args:
            vector_id: Vector identifier
            collection: Collection name

        Returns:
            (vector, metadata) tuple or None
        """
        coll = self._get_or_create_collection(collection or self.collection_name)

        try:
            results = coll.get(
                ids=[vector_id],
                include=["embeddings", "metadatas"],
            )

            if results["ids"]:
                vector = results["embeddings"][0] if results["embeddings"] else []
                meta = results["metadatas"][0] if results["metadatas"] else {}
                return (vector, VectorMetadata.from_dict(meta))

            return None

        except Exception as e:
            logger.error(f"Get vector failed: {e}")
            return None

    def update_metadata(self, vector_id: str, metadata: VectorMetadata,
                       collection: str = None) -> bool:
        """
        Update metadata for a vector.

        Args:
            vector_id: Vector identifier
            metadata: New metadata
            collection: Collection name

        Returns:
            True if successful
        """
        coll = self._get_or_create_collection(collection or self.collection_name)

        try:
            coll.update(
                ids=[vector_id],
                metadatas=[metadata.to_dict()],
            )
            return True

        except Exception as e:
            logger.error(f"Update metadata failed: {e}")
            return False

    def delete_vector(self, vector_id: str, collection: str = None) -> bool:
        """
        Delete a vector.

        Args:
            vector_id: Vector identifier
            collection: Collection name

        Returns:
            True if successful
        """
        coll = self._get_or_create_collection(collection or self.collection_name)

        try:
            coll.delete(ids=[vector_id])
            return True

        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def delete_by_metadata(self, where: Dict, collection: str = None) -> int:
        """
        Delete vectors matching metadata filter.

        Args:
            where: Metadata filter
            collection: Collection name

        Returns:
            Number of vectors deleted
        """
        coll = self._get_or_create_collection(collection or self.collection_name)

        try:
            # Get matching IDs first
            results = coll.get(where=where, limit=10000)

            if results["ids"]:
                coll.delete(ids=results["ids"])
                return len(results["ids"])

            return 0

        except Exception as e:
            logger.error(f"Delete by metadata failed: {e}")
            return 0

    def count(self, collection: str = None) -> int:
        """Get total vector count in collection"""
        coll = self._get_or_create_collection(collection or self.collection_name)
        return coll.count()

    def list_collections(self) -> List[str]:
        """List all collections"""
        return [c.name for c in self._client.list_collections()]

    def get_collection_stats(self, collection: str = None) -> Dict[str, Any]:
        """Get statistics for a collection"""
        coll = self._get_or_create_collection(collection or self.collection_name)

        return {
            "name": coll.name,
            "count": coll.count(),
            "metadata": coll.metadata,
        }

    def clear_collection(self, collection: str = None):
        """Clear all vectors from a collection"""
        name = collection or self.collection_name

        try:
            self._client.delete_collection(name)
            # Recreate empty
            self._collections.pop(name, None)
            self._get_or_create_collection(name)

            logger.info(f"Cleared collection: {name}")

        except Exception as e:
            logger.error(f"Clear collection failed: {e}")

    def persist(self):
        """Force persistence to disk (only needed for some configurations)"""
        # ChromaDB PersistentClient auto-persists
        pass


class ChromaDBVectorStore:
    """
    Higher-level vector store that integrates with ai/brain.

    Provides collections for different vector types:
    - default: General purpose vectors
    - temporal: Time-series vectors
    - behavioral: Behavior pattern vectors
    - multimodal: Cross-modal vectors
    - knowledge: Knowledge graph embeddings
    - threat: Threat intelligence vectors
    """

    def __init__(self, persist_path: str = None):
        """Initialize the vector store"""

        if persist_path is None:
            # Default to project data directory
            persist_path = str(Path(__file__).parent.parent.parent.parent / "data" / "vectors")

        self._backend = ChromaDBBackend(persist_path=persist_path)

        # Pre-create standard collections
        for coll_type, coll_name in ChromaDBBackend.COLLECTIONS.items():
            self._backend._get_or_create_collection(coll_name)

        logger.info(f"ChromaDBVectorStore initialized with {len(self._backend.list_collections())} collections")

    @property
    def default(self) -> ChromaDBBackend:
        """Access default collection backend"""
        return self._backend

    def temporal(self) -> ChromaDBBackend:
        """Get temporal vectors collection"""
        self._backend.collection_name = ChromaDBBackend.COLLECTIONS["temporal"]
        return self._backend

    def behavioral(self) -> ChromaDBBackend:
        """Get behavioral vectors collection"""
        self._backend.collection_name = ChromaDBBackend.COLLECTIONS["behavioral"]
        return self._backend

    def multimodal(self) -> ChromaDBBackend:
        """Get multimodal vectors collection"""
        self._backend.collection_name = ChromaDBBackend.COLLECTIONS["multimodal"]
        return self._backend

    def knowledge(self) -> ChromaDBBackend:
        """Get knowledge graph vectors collection"""
        self._backend.collection_name = ChromaDBBackend.COLLECTIONS["knowledge"]
        return self._backend

    def threat(self) -> ChromaDBBackend:
        """Get threat intelligence vectors collection"""
        self._backend.collection_name = ChromaDBBackend.COLLECTIONS["threat"]
        return self._backend

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get stats for all collections"""
        stats = {}
        for coll_name in self._backend.list_collections():
            stats[coll_name] = self._backend.get_collection_stats(coll_name)
        return stats


if __name__ == "__main__":
    print("ChromaDB Backend Self-Test")
    print("=" * 50)

    import tempfile
    import random

    # Test with temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n[1] Initialize with persistence at: {tmpdir}")
        backend = ChromaDBBackend(persist_path=tmpdir)

        print("\n[2] Add vectors")
        for i in range(10):
            vector = [random.gauss(0, 1) for _ in range(128)]
            metadata = VectorMetadata(
                source=f"test-source-{i}",
                entity_ids=[f"entity-{i % 3}"],
                tags=["threat" if i % 2 == 0 else "normal"],
                vector_type="test",
            )
            backend.add_vector(f"vec-{i}", vector, metadata)

        print(f"    Added 10 vectors")
        print(f"    Count: {backend.count()}")

        print("\n[3] Search")
        query = [random.gauss(0, 1) for _ in range(128)]
        results = backend.search(query, k=5)
        print(f"    Found {len(results)} results")
        for vid, sim, meta in results:
            print(f"      {vid}: similarity={sim:.3f}, tags={meta.tags}")

        print("\n[4] Metadata filter search")
        results = backend.search_by_metadata({"vector_type": "test"}, k=5)
        print(f"    Found {len(results)} with type='test'")

        print("\n[5] Get specific vector")
        result = backend.get_vector("vec-0")
        if result:
            vec, meta = result
            print(f"    vec-0: dim={len(vec)}, source={meta.source}")

        print("\n[6] Collection stats")
        stats = backend.get_collection_stats()
        print(f"    {stats}")

        print("\n[7] Test ChromaDBVectorStore")
        store = ChromaDBVectorStore(persist_path=tmpdir)
        all_stats = store.get_all_stats()
        print(f"    Collections: {list(all_stats.keys())}")

    print("\n" + "=" * 50)
    print("ChromaDB Backend test complete")

