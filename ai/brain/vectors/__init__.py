#!/usr/bin/env python3
"""
DSMIL Brain Vector Engine Layer

Ultra-high dimensional vector storage and processing:
- Self-Improving DB: Continuous cross-correlation and optimization
- ChromaDB Backend: Persistent vector storage
- Temporal Vectors: Time-series and sequence embeddings
- Multimodal Vectors: Unified text/audio/image/video representations
- Behavioral Vectors: Action, decision, and outcome embeddings
"""

from .self_improving_db import (
    SelfImprovingVectorDB,
    VectorEntry,
    Correlation,
    IntelReport,
    STORAGE_MEMORY,
    STORAGE_CHROMADB,
)

from .temporal_vectors import (
    TemporalVectorEngine,
    TemporalVector,
    TimeSeriesEmbedding,
)

from .multimodal_vectors import (
    MultimodalVectorEngine,
    MultimodalVector,
    ModalityType,
)

from .behavioral_vectors import (
    BehavioralVectorEngine,
    BehavioralVector,
    ActionPattern,
)

# Optional ChromaDB backend
try:
    from .chromadb_backend import (
        ChromaDBBackend,
        ChromaDBVectorStore,
        VectorMetadata,
    )
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    ChromaDBBackend = None
    ChromaDBVectorStore = None
    VectorMetadata = None

__all__ = [
    # Core
    "SelfImprovingVectorDB", "VectorEntry", "Correlation", "IntelReport",
    "STORAGE_MEMORY", "STORAGE_CHROMADB",
    # ChromaDB
    "ChromaDBBackend", "ChromaDBVectorStore", "VectorMetadata", "CHROMADB_AVAILABLE",
    # Temporal
    "TemporalVectorEngine", "TemporalVector", "TimeSeriesEmbedding",
    # Multimodal
    "MultimodalVectorEngine", "MultimodalVector", "ModalityType",
    # Behavioral
    "BehavioralVectorEngine", "BehavioralVector", "ActionPattern",
]

