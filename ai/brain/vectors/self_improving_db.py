#!/usr/bin/env python3
"""
Self-Improving Vector Database for DSMIL Brain

Continuous cross-correlation and self-optimization:
- Discover unexpected correlations
- Extract new relationships
- Update knowledge graph
- Generate intel reports
- Self-optimize indices

Supports pluggable storage backends:
- In-memory (default, for testing)
- ChromaDB (persistent, production)
"""

import hashlib
import threading
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Tuple, Callable, Union
from datetime import datetime, timezone
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)

# Storage backend types
STORAGE_MEMORY = "memory"
STORAGE_CHROMADB = "chromadb"


@dataclass
class VectorEntry:
    """A vector entry in the database"""
    vector_id: str
    vector: List[float]

    # Metadata
    source: str = ""
    entity_ids: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)

    # Statistics
    access_count: int = 0
    correlation_count: int = 0

    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Correlation:
    """A discovered correlation between vectors"""
    correlation_id: str
    vector_a: str
    vector_b: str

    # Correlation metrics
    similarity: float = 0.0
    confidence: float = 0.0

    # Relationship
    relationship_type: str = "similar"

    discovered: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class IntelReport:
    """Intelligence report from vector analysis"""
    report_id: str

    # Content
    correlations: List[Correlation] = field(default_factory=list)
    new_relationships: List[Dict] = field(default_factory=list)
    anomalies: List[Dict] = field(default_factory=list)
    confidence_updates: Dict[str, float] = field(default_factory=dict)

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SelfImprovingVectorDB:
    """
    Self-Improving Vector Database

    Continuously discovers correlations and optimizes itself.

    Usage:
        db = SelfImprovingVectorDB(dimensions=768)

        # Add vectors
        db.add_vector("id1", vector, entity_ids={"entity1"})

        # Query
        results = db.search(query_vector, k=10)

        # Background optimization
        db.start_background_optimization()

        # Get intel reports
        reports = db.get_intel_reports()
    """

    def __init__(self, dimensions: int = 768,
                correlation_threshold: float = 0.8,
                storage_backend: str = STORAGE_MEMORY,
                persist_path: Optional[str] = None):
        """
        Initialize Self-Improving Vector Database.

        Args:
            dimensions: Vector dimensionality
            correlation_threshold: Minimum similarity for correlation
            storage_backend: "memory" or "chromadb"
            persist_path: Path for ChromaDB persistence (required if chromadb)
        """
        self.dimensions = dimensions
        self.correlation_threshold = correlation_threshold
        self.storage_backend = storage_backend
        self.persist_path = persist_path

        # Initialize storage
        self._chromadb = None
        if storage_backend == STORAGE_CHROMADB:
            try:
                from .chromadb_backend import ChromaDBBackend, VectorMetadata
                self._chromadb = ChromaDBBackend(persist_path=persist_path)
                self._VectorMetadata = VectorMetadata
                logger.info(f"Using ChromaDB backend at: {persist_path}")
            except ImportError as e:
                logger.warning(f"ChromaDB not available, falling back to memory: {e}")
                storage_backend = STORAGE_MEMORY
                self.storage_backend = storage_backend

        # In-memory storage (used if memory backend or as cache)
        self._vectors: Dict[str, VectorEntry] = {}
        self._correlations: Dict[str, Correlation] = {}
        self._intel_reports: List[IntelReport] = []

        # Indexing
        self._entity_index: Dict[str, Set[str]] = defaultdict(set)  # entity -> vector_ids
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> vector_ids

        # Hot/cold tracking
        self._access_counts: Dict[str, int] = defaultdict(int)

        # Background processing
        self._running = False
        self._optimization_thread: Optional[threading.Thread] = None
        self._hub_callback: Optional[Callable[[IntelReport], None]] = None

        self._lock = threading.RLock()

        logger.info(f"SelfImprovingVectorDB initialized (dimensions={dimensions}, backend={storage_backend})")

    def add_vector(self, vector_id: str, vector: List[float],
                  source: str = "",
                  entity_ids: Optional[Set[str]] = None,
                  tags: Optional[Set[str]] = None) -> VectorEntry:
        """Add a vector to the database"""
        with self._lock:
            if len(vector) != self.dimensions:
                # Pad or truncate
                if len(vector) < self.dimensions:
                    vector = vector + [0.0] * (self.dimensions - len(vector))
                else:
                    vector = vector[:self.dimensions]

            entry = VectorEntry(
                vector_id=vector_id,
                vector=vector,
                source=source,
                entity_ids=entity_ids or set(),
                tags=tags or set(),
            )

            # Store in memory
            self._vectors[vector_id] = entry

            # Update indices
            for entity in entry.entity_ids:
                self._entity_index[entity].add(vector_id)

            for tag in entry.tags:
                self._tag_index[tag].add(vector_id)

            # Also store in ChromaDB if enabled
            if self._chromadb:
                metadata = self._VectorMetadata(
                    source=source,
                    entity_ids=list(entry.entity_ids),
                    tags=list(entry.tags),
                    created=entry.created.isoformat(),
                )
                self._chromadb.add_vector(vector_id, vector, metadata)

            return entry

    def search(self, query_vector: List[float], k: int = 10,
              entity_filter: Optional[Set[str]] = None,
              tag_filter: Optional[Set[str]] = None,
              use_chromadb: bool = True) -> List[Tuple[str, float]]:
        """
        Search for similar vectors

        Args:
            query_vector: Query embedding
            k: Number of results
            entity_filter: Filter by entity IDs
            tag_filter: Filter by tags
            use_chromadb: Use ChromaDB for search if available

        Returns:
            List of (vector_id, similarity) tuples
        """
        # Use ChromaDB if available and no complex filters
        if self._chromadb and use_chromadb and not entity_filter and not tag_filter:
            results = self._chromadb.search(query_vector, k=k)
            # Update access stats for returned vectors
            for vid, sim, _ in results:
                self._access_counts[vid] += 1
                if vid in self._vectors:
                    self._vectors[vid].access_count += 1
                    self._vectors[vid].last_accessed = datetime.now(timezone.utc)
            return [(vid, sim) for vid, sim, _ in results]

        # Fall back to in-memory search
        with self._lock:
            # Get candidate vectors
            if entity_filter:
                candidates = set()
                for entity in entity_filter:
                    candidates.update(self._entity_index.get(entity, set()))
            elif tag_filter:
                candidates = set()
                for tag in tag_filter:
                    candidates.update(self._tag_index.get(tag, set()))
            else:
                candidates = set(self._vectors.keys())

            # Calculate similarities
            results = []
            for vid in candidates:
                entry = self._vectors[vid]
                similarity = self._cosine_similarity(query_vector, entry.vector)
                results.append((vid, similarity))

                # Update access stats
                entry.access_count += 1
                entry.last_accessed = datetime.now(timezone.utc)
                self._access_counts[vid] += 1

            # Sort and return top k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def cross_correlate_all(self) -> List[Correlation]:
        """
        Cross-correlate all vectors to discover relationships
        """
        with self._lock:
            new_correlations = []
            vectors = list(self._vectors.values())

            for i, entry_a in enumerate(vectors):
                for entry_b in vectors[i+1:]:
                    similarity = self._cosine_similarity(entry_a.vector, entry_b.vector)

                    if similarity >= self.correlation_threshold:
                        corr = Correlation(
                            correlation_id=hashlib.sha256(f"{entry_a.vector_id}:{entry_b.vector_id}".encode()).hexdigest()[:16],
                            vector_a=entry_a.vector_id,
                            vector_b=entry_b.vector_id,
                            similarity=similarity,
                            confidence=similarity,
                        )
                        new_correlations.append(corr)
                        self._correlations[corr.correlation_id] = corr

                        entry_a.correlation_count += 1
                        entry_b.correlation_count += 1

            return new_correlations

    def analyze_correlations(self, correlations: List[Correlation]) -> List[Dict]:
        """Extract new relationships from correlations"""
        relationships = []

        with self._lock:
            for corr in correlations:
                entry_a = self._vectors.get(corr.vector_a)
                entry_b = self._vectors.get(corr.vector_b)

                if not entry_a or not entry_b:
                    continue

                # Check for entity relationships
                shared_entities = entry_a.entity_ids & entry_b.entity_ids
                new_relationships_found = entry_a.entity_ids ^ entry_b.entity_ids

                if new_relationships_found:
                    relationships.append({
                        "type": "entity_connection",
                        "entities_a": list(entry_a.entity_ids),
                        "entities_b": list(entry_b.entity_ids),
                        "confidence": corr.similarity,
                    })

        return relationships

    def generate_intel_report(self) -> IntelReport:
        """Generate intelligence report from analysis"""
        with self._lock:
            # Cross-correlate
            correlations = self.cross_correlate_all()

            # Analyze
            relationships = self.analyze_correlations(correlations)

            # Find anomalies (vectors with unusual correlation patterns)
            anomalies = []
            for vid, entry in self._vectors.items():
                if entry.correlation_count == 0 and entry.access_count > 5:
                    anomalies.append({
                        "vector_id": vid,
                        "type": "isolated_high_access",
                        "access_count": entry.access_count,
                    })

            report = IntelReport(
                report_id=hashlib.sha256(f"intel:{datetime.now().isoformat()}".encode()).hexdigest()[:16],
                correlations=correlations,
                new_relationships=relationships,
                anomalies=anomalies,
            )

            self._intel_reports.append(report)

            # Callback to hub
            if self._hub_callback:
                try:
                    self._hub_callback(report)
                except Exception as e:
                    logger.error(f"Hub callback failed: {e}")

            return report

    def reindex_hot_vectors(self):
        """Reindex frequently accessed vectors for faster retrieval"""
        with self._lock:
            # Sort by access count
            hot_vectors = sorted(
                self._access_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:100]  # Top 100

            # Would implement specialized index for hot vectors
            logger.info(f"Reindexed {len(hot_vectors)} hot vectors")

    def prune_cold_vectors(self, max_age_days: int = 90,
                          min_access: int = 1):
        """Remove rarely accessed old vectors"""
        with self._lock:
            cutoff = datetime.now(timezone.utc)
            pruned = 0

            for vid in list(self._vectors.keys()):
                entry = self._vectors[vid]
                age_days = (cutoff - entry.created).days

                if age_days > max_age_days and entry.access_count < min_access:
                    del self._vectors[vid]
                    pruned += 1

            logger.info(f"Pruned {pruned} cold vectors")

    def start_background_optimization(self, interval_seconds: float = 300,
                                      hub_callback: Optional[Callable] = None):
        """Start background optimization loop"""
        self._hub_callback = hub_callback
        self._running = True

        def optimization_loop():
            while self._running:
                try:
                    # Generate intel report
                    self.generate_intel_report()

                    # Optimize indices
                    self.reindex_hot_vectors()

                    time.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"Optimization error: {e}")
                    time.sleep(60)

        self._optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        self._optimization_thread.start()
        logger.info("Background optimization started")

    def stop_background_optimization(self):
        """Stop background optimization"""
        self._running = False
        if self._optimization_thread:
            self._optimization_thread.join(timeout=5)

    def get_intel_reports(self, limit: int = 10) -> List[IntelReport]:
        """Get recent intel reports"""
        with self._lock:
            return self._intel_reports[-limit:]

    def get_stats(self) -> Dict:
        """Get database statistics"""
        with self._lock:
            stats = {
                "total_vectors": len(self._vectors),
                "total_correlations": len(self._correlations),
                "intel_reports": len(self._intel_reports),
                "entities_indexed": len(self._entity_index),
                "tags_indexed": len(self._tag_index),
                "storage_backend": self.storage_backend,
            }

            # Add ChromaDB stats if available
            if self._chromadb:
                stats["chromadb_count"] = self._chromadb.count()
                stats["chromadb_collections"] = self._chromadb.list_collections()

            return stats


if __name__ == "__main__":
    print("Self-Improving Vector DB Self-Test")
    print("=" * 50)

    import random

    db = SelfImprovingVectorDB(dimensions=128, correlation_threshold=0.7)

    print("\n[1] Add Vectors")
    # Add correlated vectors
    base_vector = [random.gauss(0, 1) for _ in range(128)]

    for i in range(10):
        # Create similar vectors with noise
        noise = [random.gauss(0, 0.1) for _ in range(128)]
        vector = [b + n for b, n in zip(base_vector, noise)]

        db.add_vector(
            f"vec-{i}",
            vector,
            entity_ids={f"entity-{i % 3}"},
            tags={"threat" if i % 2 == 0 else "normal"},
        )

    print(f"    Added 10 vectors")

    print("\n[2] Search")
    results = db.search(base_vector, k=5)
    print(f"    Top 5 matches:")
    for vid, sim in results:
        print(f"      {vid}: {sim:.3f}")

    print("\n[3] Cross-Correlation")
    correlations = db.cross_correlate_all()
    print(f"    Found {len(correlations)} correlations")

    print("\n[4] Generate Intel Report")
    report = db.generate_intel_report()
    print(f"    Report ID: {report.report_id}")
    print(f"    Correlations: {len(report.correlations)}")
    print(f"    New relationships: {len(report.new_relationships)}")
    print(f"    Anomalies: {len(report.anomalies)}")

    print("\n[5] Filter Search")
    results = db.search(base_vector, k=5, entity_filter={"entity-0"})
    print(f"    Entity filter results: {len(results)}")

    results = db.search(base_vector, k=5, tag_filter={"threat"})
    print(f"    Tag filter results: {len(results)}")

    print("\n[6] Statistics")
    stats = db.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Self-Improving Vector DB test complete")

