from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import structlog
import hashlib

from app.core.config import settings
from app.db.redis import redis_client

logger = structlog.get_logger()

class EmbeddingService:
    _model = None
    _device = None

    @classmethod
    def _initialize_model(cls):
        """Initialize the embedding model"""
        if cls._model is None:
            try:
                # Detect available device
                cls._device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )

                # Load model
                cls._model = SentenceTransformer(
                    settings.EMBEDDING_MODEL,
                    device=cls._device
                )

                # Warm up model
                _ = cls._model.encode("warmup", convert_to_numpy=True)

                logger.info("Embedding model initialized",
                           model=settings.EMBEDDING_MODEL,
                           device=str(cls._device))

            except Exception as e:
                logger.error("Failed to initialize embedding model",
                            error=str(e))
                raise

    def __init__(self):
        self._initialize_model()

    async def generate_embedding(
        self,
        text: str,
        use_cache: bool = True
    ) -> List[float]:
        """Generate embedding for text"""
        # Check cache
        if use_cache:
            cache_key = f"embedding:text:{hashlib.md5(text.encode()).hexdigest()}"
            cached = await redis_client.cache_get(cache_key)
            if cached:
                logger.debug("Embedding cache hit")
                return cached

        try:
            # Generate embedding
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            # Convert to list
            embedding_list = embedding.tolist()

            # Cache result
            if use_cache:
                await redis_client.cache_set(
                    cache_key,
                    embedding_list,
                    ttl=settings.EMBEDDING_CACHE_TTL
                )

            return embedding_list

        except Exception as e:
            logger.error("Embedding generation failed",
                        error=str(e),
                        text_length=len(text))
            raise

    async def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            return embeddings.tolist()

        except Exception as e:
            logger.error("Batch embedding generation failed",
                        error=str(e),
                        batch_size=len(texts))
            raise