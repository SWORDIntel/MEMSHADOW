"""
EmbeddingAdapterHub - Multi-Model Embedding Normalization

Normalizes and projects embeddings from multiple models to:
- Canonical "fabric space" (256d/512d) for fast approximate search
- Per-model projections for high-fidelity retrieval
- Compressed representations for deep storage tiers

Supports PCA, PQ (Product Quantization), and FAISS-type compression.
"""

import asyncio
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import structlog

logger = structlog.get_logger()


@dataclass
class ModelSpec:
    """Specification for an embedding model"""
    model_id: str
    native_dimension: int
    model_family: str  # "openai", "sentence-transformers", "cohere", etc.
    normalize_output: bool = True
    max_sequence_length: int = 512


@dataclass
class ProjectionMatrix:
    """Learned projection matrix between dimensions"""
    source_dim: int
    target_dim: int
    matrix: np.ndarray
    inverse_matrix: Optional[np.ndarray] = None
    method: str = "random_orthogonal"  # "pca", "random_orthogonal", "learned"
    quality_score: float = 0.0  # Cosine similarity preservation score


@dataclass
class QuantizationCodebook:
    """Product quantization codebook"""
    num_subvectors: int
    bits_per_subvector: int
    centroids: np.ndarray  # Shape: (num_subvectors, 2^bits, subvector_dim)


class EmbeddingAdapterHub:
    """
    Hub for normalizing and projecting embeddings from multiple AI models.

    Provides:
    - Canonical fabric space (256d/512d) for unified search
    - Per-model projections with quality preservation
    - Compression for deep storage tiers
    """

    # Standard dimensions
    FABRIC_256 = 256
    FABRIC_512 = 512
    STANDARD_DIMS = [256, 512, 1024, 2048, 3072, 4096]

    def __init__(
        self,
        canonical_dimension: int = 256,
        enable_pq_compression: bool = True,
        pq_bits: int = 8
    ):
        self.canonical_dimension = canonical_dimension
        self.enable_pq = enable_pq_compression
        self.pq_bits = pq_bits

        # Registered models
        self.models: Dict[str, ModelSpec] = {}

        # Projection matrices: (source_dim, target_dim) → ProjectionMatrix
        self.projections: Dict[Tuple[int, int], ProjectionMatrix] = {}

        # PQ codebooks per dimension
        self.codebooks: Dict[int, QuantizationCodebook] = {}

        # Normalization stats per model
        self.model_stats: Dict[str, Dict[str, float]] = {}

        # Register common models
        self._register_default_models()

        logger.info("EmbeddingAdapterHub initialized",
                   canonical_dim=canonical_dimension)

    def _register_default_models(self):
        """Register common embedding models"""
        default_models = [
            ModelSpec("openai-text-embedding-3-large", 3072, "openai"),
            ModelSpec("openai-text-embedding-3-small", 1536, "openai"),
            ModelSpec("openai-ada-002", 1536, "openai"),
            ModelSpec("bge-large-en-v1.5", 1024, "sentence-transformers"),
            ModelSpec("bge-base-en-v1.5", 768, "sentence-transformers"),
            ModelSpec("all-mpnet-base-v2", 768, "sentence-transformers"),
            ModelSpec("gte-large", 1024, "sentence-transformers"),
            ModelSpec("cohere-embed-multilingual-v3.0", 1024, "cohere"),
            ModelSpec("cohere-embed-english-v3.0", 1024, "cohere"),
            ModelSpec("claude-embeddings", 2048, "anthropic"),
        ]

        for model in default_models:
            self.register_model(model)

    def register_model(self, spec: ModelSpec):
        """Register a new embedding model"""
        self.models[spec.model_id] = spec

        # Pre-compute projection to canonical dimension
        if spec.native_dimension != self.canonical_dimension:
            self._get_or_create_projection(spec.native_dimension, self.canonical_dimension)

        logger.debug("Model registered", model_id=spec.model_id,
                    native_dim=spec.native_dimension)

    def _get_or_create_projection(
        self,
        source_dim: int,
        target_dim: int,
        method: str = "random_orthogonal"
    ) -> ProjectionMatrix:
        """Get or create a projection matrix"""
        key = (source_dim, target_dim)

        if key not in self.projections:
            if method == "random_orthogonal":
                matrix = self._create_orthogonal_projection(source_dim, target_dim)
            elif method == "pca":
                # PCA would require training data
                matrix = self._create_orthogonal_projection(source_dim, target_dim)
            else:
                matrix = self._create_orthogonal_projection(source_dim, target_dim)

            # Create inverse if dimensions allow
            inverse = None
            if source_dim > target_dim:
                inverse = np.linalg.pinv(matrix)

            self.projections[key] = ProjectionMatrix(
                source_dim=source_dim,
                target_dim=target_dim,
                matrix=matrix,
                inverse_matrix=inverse,
                method=method
            )

        return self.projections[key]

    def _create_orthogonal_projection(
        self,
        source_dim: int,
        target_dim: int
    ) -> np.ndarray:
        """Create a quality-preserving random orthogonal projection"""
        if source_dim >= target_dim:
            # Reduce dimension
            random_matrix = np.random.randn(target_dim, source_dim)
            q, _ = np.linalg.qr(random_matrix.T)
            return q.T[:target_dim, :]
        else:
            # Expand dimension (pad with zeros initially)
            expansion = np.zeros((target_dim, source_dim))
            expansion[:source_dim, :] = np.eye(source_dim)
            # Add orthogonal noise to remaining dims
            remaining = target_dim - source_dim
            noise = np.random.randn(remaining, source_dim) * 0.01
            expansion[source_dim:, :] = noise
            return expansion

    def normalize(
        self,
        embedding: np.ndarray,
        model_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Normalize an embedding to unit length.

        If model_id is provided, uses model-specific normalization stats.
        """
        embedding = np.asarray(embedding, dtype=np.float32)

        # Model-specific normalization
        if model_id and model_id in self.model_stats:
            stats = self.model_stats[model_id]
            if "mean" in stats and "std" in stats:
                embedding = (embedding - stats["mean"]) / (stats["std"] + 1e-8)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def project_to_canonical(
        self,
        embedding: np.ndarray,
        source_model_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Project embedding to canonical fabric space.

        Returns a normalized embedding in the canonical dimension.
        """
        embedding = np.asarray(embedding, dtype=np.float32)
        source_dim = len(embedding)

        # Normalize first
        embedding = self.normalize(embedding, source_model_id)

        if source_dim == self.canonical_dimension:
            return embedding

        # Get projection matrix
        proj = self._get_or_create_projection(source_dim, self.canonical_dimension)

        # Project
        canonical = np.dot(proj.matrix, embedding)

        # Re-normalize
        norm = np.linalg.norm(canonical)
        if norm > 0:
            canonical = canonical / norm

        return canonical.astype(np.float32)

    def project_between(
        self,
        embedding: np.ndarray,
        target_dim: int,
        source_model_id: Optional[str] = None
    ) -> np.ndarray:
        """Project embedding to a specific target dimension"""
        embedding = np.asarray(embedding, dtype=np.float32)
        source_dim = len(embedding)

        if source_dim == target_dim:
            return self.normalize(embedding, source_model_id)

        proj = self._get_or_create_projection(source_dim, target_dim)
        projected = np.dot(proj.matrix, embedding)

        # Normalize
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected = projected / norm

        return projected.astype(np.float32)

    def create_tier_projections(
        self,
        embedding: np.ndarray,
        source_model_id: Optional[str] = None
    ) -> Dict[int, np.ndarray]:
        """
        Create projections for all standard tier dimensions.

        Returns dict: dimension → projected embedding
        """
        embedding = self.normalize(embedding, source_model_id)
        source_dim = len(embedding)

        projections = {source_dim: embedding}

        for target_dim in self.STANDARD_DIMS:
            if target_dim != source_dim and target_dim <= source_dim:
                projections[target_dim] = self.project_between(
                    embedding, target_dim, source_model_id
                )

        return projections

    def compress_pq(
        self,
        embedding: np.ndarray,
        num_subvectors: int = 8
    ) -> Tuple[np.ndarray, int]:
        """
        Compress embedding using Product Quantization.

        Returns (codes, codebook_id)
        """
        dimension = len(embedding)
        codebook_key = (dimension, num_subvectors)

        if codebook_key not in self.codebooks:
            self._create_pq_codebook(dimension, num_subvectors)

        codebook = self.codebooks[codebook_key]
        subvector_dim = dimension // num_subvectors

        # Encode each subvector
        codes = np.zeros(num_subvectors, dtype=np.uint8)

        for i in range(num_subvectors):
            start = i * subvector_dim
            end = start + subvector_dim
            subvector = embedding[start:end]

            # Find nearest centroid
            centroids = codebook.centroids[i]
            distances = np.linalg.norm(centroids - subvector, axis=1)
            codes[i] = np.argmin(distances)

        return codes, dimension

    def decompress_pq(
        self,
        codes: np.ndarray,
        dimension: int,
        num_subvectors: int = 8
    ) -> np.ndarray:
        """Decompress PQ codes back to embedding"""
        codebook_key = (dimension, num_subvectors)

        if codebook_key not in self.codebooks:
            raise ValueError(f"No codebook for {codebook_key}")

        codebook = self.codebooks[codebook_key]
        subvector_dim = dimension // num_subvectors

        embedding = np.zeros(dimension, dtype=np.float32)

        for i in range(num_subvectors):
            start = i * subvector_dim
            end = start + subvector_dim
            embedding[start:end] = codebook.centroids[i, codes[i]]

        return embedding

    def _create_pq_codebook(
        self,
        dimension: int,
        num_subvectors: int
    ):
        """Create a random PQ codebook (would be trained in production)"""
        num_centroids = 2 ** self.pq_bits
        subvector_dim = dimension // num_subvectors

        # Random centroids (in production, train with K-means)
        centroids = np.random.randn(
            num_subvectors, num_centroids, subvector_dim
        ).astype(np.float32)

        # Normalize
        for i in range(num_subvectors):
            for j in range(num_centroids):
                norm = np.linalg.norm(centroids[i, j])
                if norm > 0:
                    centroids[i, j] /= norm

        self.codebooks[(dimension, num_subvectors)] = QuantizationCodebook(
            num_subvectors=num_subvectors,
            bits_per_subvector=self.pq_bits,
            centroids=centroids
        )

    def compute_similarity(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings.

        Handles dimension mismatch by projecting to common space.
        """
        a = np.asarray(embedding_a, dtype=np.float32)
        b = np.asarray(embedding_b, dtype=np.float32)

        # Handle dimension mismatch
        if len(a) != len(b):
            # Project both to smaller dimension
            target_dim = min(len(a), len(b))
            if len(a) > target_dim:
                a = self.project_between(a, target_dim)
            if len(b) > target_dim:
                b = self.project_between(b, target_dim)

        # Normalize
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)

        if a_norm > 0:
            a = a / a_norm
        if b_norm > 0:
            b = b / b_norm

        if metric == "cosine":
            return float(np.dot(a, b))
        elif metric == "euclidean":
            return float(1.0 / (1.0 + np.linalg.norm(a - b)))
        else:
            return float(np.dot(a, b))

    def batch_project_to_canonical(
        self,
        embeddings: np.ndarray,
        source_model_id: Optional[str] = None
    ) -> np.ndarray:
        """Batch project multiple embeddings to canonical space"""
        if len(embeddings.shape) == 1:
            return self.project_to_canonical(embeddings, source_model_id)

        source_dim = embeddings.shape[1]

        if source_dim == self.canonical_dimension:
            # Just normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / np.maximum(norms, 1e-8)

        proj = self._get_or_create_projection(source_dim, self.canonical_dimension)

        # Batch matrix multiply
        projected = np.dot(embeddings, proj.matrix.T)

        # Normalize
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        projected = projected / np.maximum(norms, 1e-8)

        return projected.astype(np.float32)

    def estimate_projection_quality(
        self,
        source_dim: int,
        target_dim: int,
        num_samples: int = 1000
    ) -> float:
        """
        Estimate projection quality by measuring cosine similarity preservation.

        Returns average cosine similarity preservation score (0.0 to 1.0).
        """
        # Generate random sample pairs
        samples = np.random.randn(num_samples, source_dim).astype(np.float32)
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        samples = samples / np.maximum(norms, 1e-8)

        # Compute original similarities
        orig_sims = np.dot(samples, samples.T)

        # Project
        proj = self._get_or_create_projection(source_dim, target_dim)
        projected = np.dot(samples, proj.matrix.T)
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        projected = projected / np.maximum(norms, 1e-8)

        # Compute projected similarities
        proj_sims = np.dot(projected, projected.T)

        # Correlation between similarity matrices (excluding diagonal)
        mask = ~np.eye(num_samples, dtype=bool)
        correlation = np.corrcoef(
            orig_sims[mask].flatten(),
            proj_sims[mask].flatten()
        )[0, 1]

        # Update projection quality score
        key = (source_dim, target_dim)
        if key in self.projections:
            self.projections[key].quality_score = float(correlation)

        return float(correlation)

    def get_stats(self) -> Dict[str, Any]:
        """Get hub statistics"""
        return {
            "canonical_dimension": self.canonical_dimension,
            "registered_models": len(self.models),
            "projection_matrices": len(self.projections),
            "pq_codebooks": len(self.codebooks),
            "models": list(self.models.keys()),
        }
