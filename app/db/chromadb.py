import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import structlog

from app.core.config import settings

logger = structlog.get_logger()

class ChromaDBClient:
    def __init__(self):
        self.client = None
        self.collection = None

    async def init_client(self):
        """Initialize ChromaDB client"""
        try:
            self.client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )

            logger.info("ChromaDB client initialized",
                       collection=settings.CHROMA_COLLECTION)
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
        """Add embedding to ChromaDB"""
        try:
            self.collection.add(
                embeddings=[embedding],
                ids=[memory_id],
                metadatas=[metadata]
            )
            logger.debug("Embedding added", memory_id=memory_id)
        except Exception as e:
            logger.error("Failed to add embedding",
                        memory_id=memory_id,
                        error=str(e))
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