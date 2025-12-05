#!/usr/bin/env python3
"""
Multimodal Vector Engine for DSMIL Brain

Unified representations across modalities:
- Text, audio, image, video
- Cross-modal retrieval
- Modality-agnostic search
"""

import hashlib
import threading
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Tuple
from datetime import datetime, timezone
from enum import Enum, auto

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Types of modalities"""
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    STRUCTURED = auto()  # JSON, etc.
    BINARY = auto()


@dataclass
class MultimodalVector:
    """A vector with modality information"""
    vector_id: str
    vector: List[float]

    # Modality
    modality: ModalityType
    original_format: str = ""  # e.g., "jpeg", "mp3", "txt"

    # Content reference
    content_hash: str = ""
    content_size: int = 0

    # Metadata
    source: str = ""
    tags: Set[str] = field(default_factory=set)

    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MultimodalVectorEngine:
    """
    Multimodal Vector Engine

    Manages vectors from multiple modalities in unified space.

    Usage:
        engine = MultimodalVectorEngine()

        # Add vectors from different modalities
        engine.add_vector("text1", text_vector, ModalityType.TEXT)
        engine.add_vector("img1", image_vector, ModalityType.IMAGE)

        # Cross-modal search
        results = engine.search(query_vector, target_modalities={ModalityType.IMAGE})
    """

    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions

        self._vectors: Dict[str, MultimodalVector] = {}
        self._by_modality: Dict[ModalityType, Set[str]] = {m: set() for m in ModalityType}

        self._lock = threading.RLock()

        logger.info(f"MultimodalVectorEngine initialized (dimensions={dimensions})")

    def add_vector(self, vector_id: str, vector: List[float],
                  modality: ModalityType,
                  original_format: str = "",
                  content_hash: str = "",
                  source: str = "",
                  tags: Optional[Set[str]] = None) -> MultimodalVector:
        """Add a multimodal vector"""
        with self._lock:
            # Normalize dimensions
            if len(vector) != self.dimensions:
                if len(vector) < self.dimensions:
                    vector = vector + [0.0] * (self.dimensions - len(vector))
                else:
                    vector = vector[:self.dimensions]

            mv = MultimodalVector(
                vector_id=vector_id,
                vector=vector,
                modality=modality,
                original_format=original_format,
                content_hash=content_hash,
                source=source,
                tags=tags or set(),
            )

            self._vectors[vector_id] = mv
            self._by_modality[modality].add(vector_id)

            return mv

    def search(self, query_vector: List[float],
              k: int = 10,
              source_modality: Optional[ModalityType] = None,
              target_modalities: Optional[Set[ModalityType]] = None) -> List[Tuple[str, float, ModalityType]]:
        """
        Search for similar vectors

        Args:
            query_vector: Query embedding
            k: Number of results
            source_modality: Modality of query (for logging)
            target_modalities: Filter to specific modalities

        Returns:
            List of (vector_id, similarity, modality) tuples
        """
        with self._lock:
            # Get candidates
            if target_modalities:
                candidates = set()
                for modality in target_modalities:
                    candidates.update(self._by_modality.get(modality, set()))
            else:
                candidates = set(self._vectors.keys())

            # Calculate similarities
            results = []
            for vid in candidates:
                mv = self._vectors[vid]
                similarity = self._cosine_similarity(query_vector, mv.vector)
                results.append((vid, similarity, mv.modality))

            # Sort and return
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]

    def cross_modal_search(self, query_vector: List[float],
                          query_modality: ModalityType,
                          k: int = 10) -> Dict[ModalityType, List[Tuple[str, float]]]:
        """
        Search across all modalities, returning results grouped by modality
        """
        with self._lock:
            results_by_modality = {}

            for modality in ModalityType:
                if modality == query_modality:
                    continue  # Skip same modality

                candidates = self._by_modality.get(modality, set())

                results = []
                for vid in candidates:
                    mv = self._vectors[vid]
                    similarity = self._cosine_similarity(query_vector, mv.vector)
                    results.append((vid, similarity))

                results.sort(key=lambda x: x[1], reverse=True)
                results_by_modality[modality] = results[:k]

            return results_by_modality

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity"""
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def get_by_modality(self, modality: ModalityType,
                       limit: int = 100) -> List[MultimodalVector]:
        """Get vectors of specific modality"""
        with self._lock:
            vector_ids = list(self._by_modality.get(modality, set()))[:limit]
            return [self._vectors[vid] for vid in vector_ids]

    def get_modality_stats(self) -> Dict[str, int]:
        """Get count by modality"""
        with self._lock:
            return {
                m.name: len(ids) for m, ids in self._by_modality.items()
            }

    def get_stats(self) -> Dict:
        """Get engine statistics"""
        with self._lock:
            return {
                "total_vectors": len(self._vectors),
                "by_modality": self.get_modality_stats(),
            }


if __name__ == "__main__":
    print("Multimodal Vector Engine Self-Test")
    print("=" * 50)

    import random

    engine = MultimodalVectorEngine(dimensions=128)

    print("\n[1] Add Multimodal Vectors")

    # Add text vectors
    for i in range(5):
        vector = [random.gauss(0, 1) for _ in range(128)]
        engine.add_vector(f"text-{i}", vector, ModalityType.TEXT, original_format="txt")

    # Add image vectors (similar space but slightly different distribution)
    for i in range(5):
        vector = [random.gauss(0.2, 1) for _ in range(128)]
        engine.add_vector(f"image-{i}", vector, ModalityType.IMAGE, original_format="jpeg")

    # Add audio vectors
    for i in range(3):
        vector = [random.gauss(-0.1, 1) for _ in range(128)]
        engine.add_vector(f"audio-{i}", vector, ModalityType.AUDIO, original_format="mp3")

    print("    Added vectors: 5 text, 5 image, 3 audio")

    print("\n[2] Modality Statistics")
    stats = engine.get_modality_stats()
    for modality, count in stats.items():
        print(f"    {modality}: {count}")

    print("\n[3] Search All Modalities")
    query = [random.gauss(0.1, 1) for _ in range(128)]
    results = engine.search(query, k=5)
    print("    Top 5 results:")
    for vid, sim, modality in results:
        print(f"      {vid} ({modality.name}): {sim:.3f}")

    print("\n[4] Search Specific Modality")
    results = engine.search(query, k=3, target_modalities={ModalityType.IMAGE})
    print("    Image-only results:")
    for vid, sim, modality in results:
        print(f"      {vid}: {sim:.3f}")

    print("\n[5] Cross-Modal Search")
    cross_results = engine.cross_modal_search(query, ModalityType.TEXT, k=2)
    for modality, results in cross_results.items():
        if results:
            print(f"    {modality.name}:")
            for vid, sim in results:
                print(f"      {vid}: {sim:.3f}")

    print("\n[6] Overall Statistics")
    stats = engine.get_stats()
    print(f"    Total vectors: {stats['total_vectors']}")

    print("\n" + "=" * 50)
    print("Multimodal Vector Engine test complete")

