import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import structlog
from functools import lru_cache

from app.core.config import settings, MemoryOperationMode

logger = structlog.get_logger()

class ChromaDBClient:
    def __init__(self):
        self.client = None
        self.collection = None

    async def init_client(self):
        """
        Initialize ChromaDB client with support for configurable embedding dimensions

        ChromaDB automatically handles variable-dimension vectors, so we don't need
        to explicitly configure the dimension. The collection metadata tracks the
        expected dimension for validation purposes.
        """
        try:
            self.client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )

            # Collection metadata includes embedding configuration
            collection_metadata = {
                "hnsw:space": "cosine",  # Cosine similarity for semantic search
                "embedding_dimension": settings.EMBEDDING_DIMENSION,
                "embedding_model": settings.EMBEDDING_MODEL,
                "embedding_backend": settings.EMBEDDING_BACKEND,
                "projection_enabled": settings.EMBEDDING_USE_PROJECTION
            }

            # Get or create collection with updated metadata
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION,
                metadata=collection_metadata
            )

            # Verify collection is ready
            collection_count = self.collection.count()

            logger.info(
                "ChromaDB client initialized",
                collection=settings.CHROMA_COLLECTION,
                embedding_dimension=settings.EMBEDDING_DIMENSION,
                embedding_model=settings.EMBEDDING_MODEL,
                backend=settings.EMBEDDING_BACKEND,
                existing_vectors=collection_count,
                distance_metric="cosine"
            )

        except Exception as e:
            logger.error("ChromaDB initialization failed", error=str(e))
            raise

    async def close_client(self):
        """Close ChromaDB client"""
        # ChromaDB HTTP client doesn't need explicit closing
        logger.info("ChromaDB client closed")

    async def add_embedding(
        self,
        memory_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ):
        """
        Add embedding to ChromaDB with dimension validation

        Args:
            memory_id: Unique identifier for the memory
            embedding: Embedding vector
            metadata: Metadata to store with the embedding
        """
        try:
            # Validate embedding dimension
            if len(embedding) != settings.EMBEDDING_DIMENSION:
                logger.error(
                    "Embedding dimension mismatch",
                    memory_id=memory_id,
                    expected=settings.EMBEDDING_DIMENSION,
                    actual=len(embedding)
                )
                raise ValueError(
                    f"Embedding dimension mismatch: expected {settings.EMBEDDING_DIMENSION}, "
                    f"got {len(embedding)}"
                )

            self.collection.add(
                embeddings=[embedding],
                ids=[memory_id],
                metadatas=[metadata]
            )
            logger.debug(
                "Embedding added",
                memory_id=memory_id,
                dimension=len(embedding)
            )
        except Exception as e:
            logger.error("Failed to add embedding",
                        memory_id=memory_id,
                        error=str(e))
            raise

    async def add_embeddings_batch(
        self,
        memory_ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ):
        """
        Add multiple embeddings to ChromaDB in a single batch operation.
        Much more efficient than adding one at a time.

        Args:
            memory_ids: List of unique identifiers
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts
        """
        try:
            # Validate all lists have the same length
            if not (len(memory_ids) == len(embeddings) == len(metadatas)):
                raise ValueError(
                    f"Length mismatch: memory_ids={len(memory_ids)}, "
                    f"embeddings={len(embeddings)}, metadatas={len(metadatas)}"
                )

            # Validate all embedding dimensions
            for i, embedding in enumerate(embeddings):
                if len(embedding) != settings.EMBEDDING_DIMENSION:
                    raise ValueError(
                        f"Embedding {i} dimension mismatch: expected {settings.EMBEDDING_DIMENSION}, "
                        f"got {len(embedding)}"
                    )

            self.collection.add(
                embeddings=embeddings,
                ids=memory_ids,
                metadatas=metadatas
            )

            logger.info(
                "Batch embeddings added",
                count=len(memory_ids),
                dimension=settings.EMBEDDING_DIMENSION
            )

        except Exception as e:
            logger.error("Failed to add batch embeddings", error=str(e))
            raise

    async def search_similar(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for similar embeddings"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )

            logger.debug("Similarity search completed",
                        n_results=len(results['ids'][0]))

            return results
        except Exception as e:
            logger.error("Similarity search failed", error=str(e))
            raise

    async def delete_embedding(self, memory_id: str):
        """Delete embedding from ChromaDB"""
        try:
            self.collection.delete(ids=[memory_id])
            logger.debug("Embedding deleted", memory_id=memory_id)
        except Exception as e:
            logger.error("Failed to delete embedding",
                        memory_id=memory_id,
                        error=str(e))
            raise

# Global client instance
chroma_client = ChromaDBClient()

async def init_client():
    await chroma_client.init_client()

async def close_client():
    await chroma_client.close_client()