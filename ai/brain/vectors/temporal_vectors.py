#!/usr/bin/env python3
"""
Temporal Vector Engine for DSMIL Brain

Time-aware vector representations:
- Time-series embeddings
- Sequence embeddings
- Temporal query support
"""

import hashlib
import threading
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesEmbedding:
    """Embedding for a time series"""
    embedding_id: str
    values: List[float]  # Time series values
    timestamps: List[datetime]

    # Computed embedding
    vector: List[float] = field(default_factory=list)

    # Metadata
    source: str = ""
    entity_id: str = ""


@dataclass
class TemporalVector:
    """A vector with temporal metadata"""
    vector_id: str
    vector: List[float]

    # Temporal info
    valid_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_to: Optional[datetime] = None  # None = still valid

    # Version
    version: int = 1
    previous_version: Optional[str] = None


class TemporalVectorEngine:
    """
    Temporal Vector Engine

    Manages time-aware vectors and embeddings.

    Usage:
        engine = TemporalVectorEngine()

        # Add temporal vector
        engine.add_vector("id", vector, valid_from=datetime(...))

        # Query at specific time
        results = engine.search_at_time(query, timestamp)

        # Get time series embedding
        embedding = engine.embed_time_series(values, timestamps)
    """

    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions

        self._vectors: Dict[str, List[TemporalVector]] = defaultdict(list)  # id -> versions
        self._time_index: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)

        self._lock = threading.RLock()

        logger.info(f"TemporalVectorEngine initialized (dimensions={dimensions})")

    def add_vector(self, vector_id: str, vector: List[float],
                  valid_from: Optional[datetime] = None,
                  valid_to: Optional[datetime] = None) -> TemporalVector:
        """Add a temporal vector"""
        with self._lock:
            versions = self._vectors[vector_id]

            # Check if updating existing
            version = len(versions) + 1
            previous = versions[-1].vector_id if versions else None

            # Expire previous version
            if versions and versions[-1].valid_to is None:
                versions[-1].valid_to = valid_from or datetime.now(timezone.utc)

            tv = TemporalVector(
                vector_id=f"{vector_id}_v{version}",
                vector=vector,
                valid_from=valid_from or datetime.now(timezone.utc),
                valid_to=valid_to,
                version=version,
                previous_version=previous,
            )

            versions.append(tv)

            # Update time index
            self._time_index[vector_id].append((tv.valid_from, tv.vector_id))

            return tv

    def get_vector_at_time(self, vector_id: str,
                          timestamp: datetime) -> Optional[TemporalVector]:
        """Get vector valid at specific time"""
        with self._lock:
            versions = self._vectors.get(vector_id, [])

            for tv in reversed(versions):
                if tv.valid_from <= timestamp:
                    if tv.valid_to is None or tv.valid_to > timestamp:
                        return tv

            return None

    def search_at_time(self, query_vector: List[float],
                      timestamp: datetime,
                      k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar vectors at specific time"""
        with self._lock:
            results = []

            for base_id, versions in self._vectors.items():
                # Get vector valid at timestamp
                tv = None
                for v in reversed(versions):
                    if v.valid_from <= timestamp:
                        if v.valid_to is None or v.valid_to > timestamp:
                            tv = v
                            break

                if tv:
                    similarity = self._cosine_similarity(query_vector, tv.vector)
                    results.append((base_id, similarity))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]

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

    def embed_time_series(self, values: List[float],
                         timestamps: List[datetime],
                         source: str = "",
                         entity_id: str = "") -> TimeSeriesEmbedding:
        """
        Create embedding from time series data
        """
        with self._lock:
            # Simple time series embedding
            # In production, would use more sophisticated methods

            n = len(values)
            if n == 0:
                return TimeSeriesEmbedding(
                    embedding_id=hashlib.sha256(f"ts:{datetime.now().isoformat()}".encode()).hexdigest()[:16],
                    values=[],
                    timestamps=[],
                )

            # Statistical features
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n
            std = math.sqrt(variance) if variance > 0 else 0

            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val

            # Trend features
            if n > 1:
                trend = (values[-1] - values[0]) / n
            else:
                trend = 0

            # Autocorrelation (lag 1)
            if n > 1:
                autocorr = sum((values[i] - mean) * (values[i-1] - mean) for i in range(1, n)) / (n * variance) if variance > 0 else 0
            else:
                autocorr = 0

            # Create feature vector
            features = [
                mean, std, min_val, max_val, range_val, trend, autocorr,
                values[0] if values else 0,  # First value
                values[-1] if values else 0,  # Last value
                float(n),  # Length
            ]

            # Pad to dimensions
            vector = features + [0.0] * (self.dimensions - len(features))
            vector = vector[:self.dimensions]

            return TimeSeriesEmbedding(
                embedding_id=hashlib.sha256(f"ts:{source}:{entity_id}:{datetime.now().isoformat()}".encode()).hexdigest()[:16],
                values=values,
                timestamps=timestamps,
                vector=vector,
                source=source,
                entity_id=entity_id,
            )

    def get_vector_history(self, vector_id: str) -> List[TemporalVector]:
        """Get all versions of a vector"""
        with self._lock:
            return list(self._vectors.get(vector_id, []))

    def get_stats(self) -> Dict:
        """Get engine statistics"""
        with self._lock:
            total_versions = sum(len(v) for v in self._vectors.values())
            return {
                "unique_vectors": len(self._vectors),
                "total_versions": total_versions,
                "avg_versions": total_versions / max(len(self._vectors), 1),
            }


if __name__ == "__main__":
    print("Temporal Vector Engine Self-Test")
    print("=" * 50)

    import random

    engine = TemporalVectorEngine(dimensions=64)

    print("\n[1] Add Temporal Vectors")
    base_time = datetime.now(timezone.utc) - timedelta(days=30)

    for i in range(5):
        vector = [random.gauss(0, 1) for _ in range(64)]
        valid_from = base_time + timedelta(days=i * 5)
        engine.add_vector(f"entity-{i}", vector, valid_from=valid_from)
    print("    Added 5 temporal vectors")

    print("\n[2] Update Vector (new version)")
    new_vector = [random.gauss(0, 1) for _ in range(64)]
    engine.add_vector("entity-0", new_vector, valid_from=datetime.now(timezone.utc))
    print("    Updated entity-0 with new version")

    print("\n[3] Get Vector History")
    history = engine.get_vector_history("entity-0")
    print(f"    entity-0 has {len(history)} versions")
    for h in history:
        print(f"      v{h.version}: valid from {h.valid_from.date()}")

    print("\n[4] Query at Historical Time")
    query_time = base_time + timedelta(days=10)
    query_vector = [random.gauss(0, 1) for _ in range(64)]
    results = engine.search_at_time(query_vector, query_time, k=3)
    print(f"    Query at {query_time.date()}")
    for vid, sim in results:
        print(f"      {vid}: {sim:.3f}")

    print("\n[5] Time Series Embedding")
    ts_values = [10 + i * 0.5 + random.gauss(0, 1) for i in range(20)]
    ts_timestamps = [base_time + timedelta(hours=i) for i in range(20)]

    embedding = engine.embed_time_series(ts_values, ts_timestamps, source="sensor", entity_id="temp-1")
    print(f"    Embedding ID: {embedding.embedding_id}")
    print(f"    Vector length: {len(embedding.vector)}")
    print(f"    First 5 features: {[f'{v:.3f}' for v in embedding.vector[:5]]}")

    print("\n[6] Statistics")
    stats = engine.get_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.2f}")
        else:
            print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Temporal Vector Engine test complete")

