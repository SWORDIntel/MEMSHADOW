"""
Multi-Modal Embedding Service
Phase 3: Intelligence Layer - Support for diverse content types
"""

from typing import Dict, List, Any, Optional, Literal
from enum import Enum
import structlog
from app.services.embedding_service import EmbeddingService

logger = structlog.get_logger()

class ContentType(str, Enum):
    """Supported content types for multi-modal embedding"""
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"

class MultiModalEmbeddingService:
    """
    Service for generating embeddings across different content modalities.
    Extends the base embedding service with specialized handlers.
    """
    
    def __init__(self):
        self.text_embedder = EmbeddingService()
        self.modality_handlers = {
            ContentType.TEXT: self._handle_text,
            ContentType.CODE: self._handle_code,
            ContentType.IMAGE: self._handle_image,
            ContentType.DOCUMENT: self._handle_document,
        }
    
    async def generate_embedding(
        self,
        content: Any,
        content_type: ContentType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate embedding for content based on its type.
        
        Args:
            content: The content to embed
            content_type: Type of content
            metadata: Additional metadata about the content
        
        Returns:
            Dictionary with embedding and metadata
        """
        logger.info("Generating multi-modal embedding", content_type=content_type)
        
        handler = self.modality_handlers.get(content_type)
        if not handler:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        result = await handler(content, metadata or {})
        result["content_type"] = content_type
        
        return result
    
    async def _handle_text(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle text embedding"""
        embedding = await self.text_embedder.generate_embedding(content)
        
        return {
            "embedding": embedding,
            "dimension": len(embedding),
            "method": "sentence-transformers",
            "metadata": metadata
        }
    
    async def _handle_code(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle code embedding with programming language awareness.
        
        Uses specialized code embeddings or enhanced preprocessing.
        """
        language = metadata.get("language", "unknown")
        
        # Preprocess code: add language markers
        preprocessed = f"[{language}] {content}"
        
        # In production, use specialized code embedding models like:
        # - CodeBERT
        # - GraphCodeBERT
        # - UniXcoder
        embedding = await self.text_embedder.generate_embedding(preprocessed)
        
        return {
            "embedding": embedding,
            "dimension": len(embedding),
            "method": "code-aware-embedding",
            "language": language,
            "metadata": metadata
        }
    
    async def _handle_image(
        self,
        content: Any,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle image embedding.
        
        In production, would use:
        - CLIP for text-image joint embeddings
        - ResNet/ViT for visual features
        - OCR + text embedding for text-heavy images
        """
        logger.info("Processing image embedding")
        
        # Mock structure - in production:
        # from PIL import Image
        # import torch
        # from transformers import CLIPProcessor, CLIPModel
        
        # For now, return a mock embedding
        embedding = [0.0] * 512  # CLIP produces 512-dim embeddings
        
        return {
            "embedding": embedding,
            "dimension": len(embedding),
            "method": "CLIP",
            "metadata": metadata
        }
    
    async def _handle_document(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle document embedding with structure awareness.
        
        Considers document structure, headings, sections, etc.
        """
        doc_type = metadata.get("doc_type", "unknown")
        
        # For structured documents, might want to:
        # 1. Extract sections
        # 2. Embed each section separately
        # 3. Combine with weighted average
        
        # For now, use text embedding with document markers
        preprocessed = f"[{doc_type} document] {content}"
        embedding = await self.text_embedder.generate_embedding(preprocessed)
        
        return {
            "embedding": embedding,
            "dimension": len(embedding),
            "method": "document-aware-embedding",
            "doc_type": doc_type,
            "metadata": metadata
        }
    
    async def generate_batch_embeddings(
        self,
        contents: List[tuple[Any, ContentType]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for multiple contents of potentially different types.
        
        Args:
            contents: List of (content, content_type) tuples
            metadata: Optional list of metadata dicts
        
        Returns:
            List of embedding results
        """
        if metadata is None:
            metadata = [{}] * len(contents)
        
        results = []
        for (content, content_type), meta in zip(contents, metadata):
            result = await self.generate_embedding(content, content_type, meta)
            results.append(result)
        
        logger.info("Batch embeddings generated", count=len(results))
        return results
    
    async def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
        method: Literal["cosine", "euclidean", "dot"] = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            method: Similarity metric to use
        
        Returns:
            Similarity score
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have same dimension")
        
        if method == "cosine":
            # Cosine similarity
            import math
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            norm1 = math.sqrt(sum(a * a for a in embedding1))
            norm2 = math.sqrt(sum(b * b for b in embedding2))
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        elif method == "dot":
            # Dot product
            return sum(a * b for a, b in zip(embedding1, embedding2))
        
        elif method == "euclidean":
            # Negative euclidean distance (higher is more similar)
            import math
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(embedding1, embedding2)))
            return -distance
        
        else:
            raise ValueError(f"Unsupported similarity method: {method}")
    
    async def find_cross_modal_matches(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find best matches across different modalities.
        
        Useful for queries like "find images related to this text"
        """
        results = []
        
        for candidate in candidate_embeddings:
            similarity = await self.compute_similarity(
                query_embedding,
                candidate["embedding"]
            )
            
            results.append({
                "content_type": candidate["content_type"],
                "similarity": similarity,
                "metadata": candidate.get("metadata", {})
            })
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]


# Global instance
multimodal_embeddings = MultiModalEmbeddingService()
