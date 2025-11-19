from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import structlog
import hashlib

from app.core.config import settings
from app.db.redis import redis_client

logger = structlog.get_logger()


class ProjectionLayer(nn.Module):
    """
    Neural projection layer to expand embeddings to higher dimensions
    Uses learnable linear transformation for dimensionality expansion
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        # Initialize with Xavier uniform for better convergence
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x):
        return self.projection(x)


class EmbeddingService:
    _model = None
    _device = None
    _projection_layer = None
    _model_dimension = None

    @classmethod
    def _initialize_model(cls):
        """Initialize the embedding model with optional projection layer"""
        if cls._model is None:
            try:
                # Detect available device
                cls._device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )

                backend = settings.EMBEDDING_BACKEND.lower()

                if backend == "sentence-transformers":
                    # Load sentence-transformer model
                    cls._model = SentenceTransformer(
                        settings.EMBEDDING_MODEL,
                        device=cls._device
                    )

                    # Get actual model dimension
                    warmup_embedding = cls._model.encode("warmup", convert_to_numpy=True)
                    cls._model_dimension = len(warmup_embedding)

                    # Initialize projection layer if dimensions don't match
                    if settings.EMBEDDING_USE_PROJECTION and cls._model_dimension != settings.EMBEDDING_DIMENSION:
                        logger.info(
                            "Creating projection layer",
                            from_dim=cls._model_dimension,
                            to_dim=settings.EMBEDDING_DIMENSION
                        )
                        cls._projection_layer = ProjectionLayer(
                            cls._model_dimension,
                            settings.EMBEDDING_DIMENSION
                        ).to(cls._device)
                        cls._projection_layer.eval()  # Set to evaluation mode
                    else:
                        cls._projection_layer = None

                    logger.info(
                        "Embedding model initialized",
                        backend="sentence-transformers",
                        model=settings.EMBEDDING_MODEL,
                        model_dimension=cls._model_dimension,
                        target_dimension=settings.EMBEDDING_DIMENSION,
                        projection_enabled=cls._projection_layer is not None,
                        device=str(cls._device)
                    )

                elif backend == "openai":
                    # OpenAI embeddings will be handled in the encode method
                    logger.info(
                        "OpenAI embedding backend selected",
                        model=settings.OPENAI_EMBEDDING_MODEL,
                        dimension=settings.EMBEDDING_DIMENSION
                    )
                    cls._model = "openai"  # Marker
                    cls._model_dimension = settings.EMBEDDING_DIMENSION

                else:
                    raise ValueError(f"Unsupported embedding backend: {backend}")

            except Exception as e:
                logger.error("Failed to initialize embedding model",
                            error=str(e))
                raise

    def __init__(self):
        self._initialize_model()

    def _apply_projection(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply projection layer to embeddings if configured"""
        if self._projection_layer is None:
            return embeddings

        with torch.no_grad():
            # Convert to tensor
            tensor_embeddings = torch.from_numpy(embeddings).float().to(self._device)

            # Apply projection
            projected = self._projection_layer(tensor_embeddings)

            # Normalize the projected embeddings
            projected = torch.nn.functional.normalize(projected, p=2, dim=-1)

            # Convert back to numpy
            return projected.cpu().numpy()

    async def generate_embedding(
        self,
        text: str,
        use_cache: bool = True
    ) -> List[float]:
        """
        Generate embedding for text

        Args:
            text: Input text to embed
            use_cache: Whether to use Redis cache

        Returns:
            List of floats representing the embedding vector
        """
        # Check cache (include dimension in key for compatibility)
        cache_key = f"embedding:v2:{settings.EMBEDDING_DIMENSION}:{hashlib.md5(text.encode()).hexdigest()}"
        if use_cache:
            cached = await redis_client.cache_get(cache_key)
            if cached:
                logger.debug("Embedding cache hit", dimension=len(cached))
                return cached

        try:
            backend = settings.EMBEDDING_BACKEND.lower()

            if backend == "sentence-transformers":
                # Generate base embedding
                embedding = self._model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

                # Apply projection if needed
                if self._projection_layer is not None:
                    embedding = self._apply_projection(embedding)

                # Convert to list
                embedding_list = embedding.tolist()

            elif backend == "openai":
                # Use OpenAI API for embeddings
                embedding_list = await self._generate_openai_embedding(text)

            else:
                raise ValueError(f"Unsupported backend: {backend}")

            # Validate dimension
            if len(embedding_list) != settings.EMBEDDING_DIMENSION:
                logger.warning(
                    "Embedding dimension mismatch",
                    expected=settings.EMBEDDING_DIMENSION,
                    actual=len(embedding_list)
                )

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
                        text_length=len(text),
                        backend=settings.EMBEDDING_BACKEND)
            raise

    async def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        try:
            import openai
            openai.api_key = settings.OPENAI_API_KEY

            response = await openai.Embedding.acreate(
                input=text,
                model=settings.OPENAI_EMBEDDING_MODEL,
                dimensions=settings.EMBEDDING_DIMENSION  # OpenAI supports configurable dims
            )

            return response['data'][0]['embedding']

        except ImportError:
            logger.error("OpenAI library not installed. Install with: pip install openai")
            raise
        except Exception as e:
            logger.error("OpenAI embedding generation failed", error=str(e))
            raise

    async def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        try:
            backend = settings.EMBEDDING_BACKEND.lower()

            if backend == "sentence-transformers":
                # Generate base embeddings
                embeddings = self._model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

                # Apply projection if needed
                if self._projection_layer is not None:
                    embeddings = self._apply_projection(embeddings)

                return embeddings.tolist()

            elif backend == "openai":
                # Process in batches for OpenAI
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    for text in batch:
                        embedding = await self._generate_openai_embedding(text)
                        all_embeddings.append(embedding)

                return all_embeddings

            else:
                raise ValueError(f"Unsupported backend: {backend}")

        except Exception as e:
            logger.error("Batch embedding generation failed",
                        error=str(e),
                        batch_size=len(texts),
                        backend=settings.EMBEDDING_BACKEND)
            raise

    def get_model_info(self) -> dict:
        """Get information about the current embedding model"""
        return {
            "backend": settings.EMBEDDING_BACKEND,
            "model": settings.EMBEDDING_MODEL if settings.EMBEDDING_BACKEND == "sentence-transformers" else settings.OPENAI_EMBEDDING_MODEL,
            "model_dimension": self._model_dimension,
            "target_dimension": settings.EMBEDDING_DIMENSION,
            "projection_enabled": self._projection_layer is not None,
            "device": str(self._device) if self._device else "N/A"
        }