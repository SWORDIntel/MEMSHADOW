"""
Fuzzy Matching & 2048-Dimensional Vector Intelligence Integration

Comprehensive end-to-end intelligence analysis system that:
- Performs fuzzy matching across multiple data sources
- Integrates 2048-dimensional vector embeddings
- Connects with external vector systems
- Provides similarity scoring and clustering
- Enables cross-system intelligence correlation
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import structlog

from app.core.config import settings
from app.services.vanta_blackwidow.ioc_identifier import ioc_identifier

logger = structlog.get_logger()


class FuzzyVectorIntelligence:
    """
    Advanced fuzzy matching and vector-based intelligence analysis system

    Integrates:
    - 2048-dimensional semantic vectors
    - Fuzzy string matching
    - Cross-system vector alignment
    - IoC correlation
    - Threat intelligence clustering
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.vector_dimension = settings.EMBEDDING_DIMENSION
        self._initialize_model()

        # Vector storage for cross-system integration
        self.external_vector_systems = {}

        logger.info(
            "Fuzzy Vector Intelligence initialized",
            vector_dim=self.vector_dimension,
            device=str(self.device)
        )

    def _initialize_model(self):
        """Initialize the 2048-dim vector model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device=self.device)

    async def fuzzy_match_text(
        self,
        query: str,
        candidates: List[str],
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform fuzzy text matching with similarity scoring

        Args:
            query: Query string
            candidates: List of candidate strings
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List of matches with scores
        """
        matches = []

        for candidate in candidates:
            # String-based fuzzy matching
            string_similarity = SequenceMatcher(None, query.lower(), candidate.lower()).ratio()

            # Vector-based semantic similarity
            query_vec = self.model.encode(query, convert_to_tensor=True)
            candidate_vec = self.model.encode(candidate, convert_to_tensor=True)
            semantic_similarity = util.cos_sim(query_vec, candidate_vec).item()

            # Hybrid score (weighted combination)
            hybrid_score = (0.4 * string_similarity) + (0.6 * semantic_similarity)

            if hybrid_score >= threshold:
                matches.append({
                    'candidate': candidate,
                    'string_similarity': float(string_similarity),
                    'semantic_similarity': float(semantic_similarity),
                    'hybrid_score': float(hybrid_score),
                    'match_type': 'fuzzy'
                })

        # Sort by hybrid score
        matches.sort(key=lambda x: x['hybrid_score'], reverse=True)

        logger.info(f"Fuzzy matching found {len(matches)} matches above threshold {threshold}")

        return matches

    async def vectorize_intelligence(
        self,
        data: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Convert intelligence data to 2048-dimensional vectors

        Args:
            data: List of text data
            batch_size: Batch size for encoding

        Returns:
            Numpy array of shape (n_samples, 2048)
        """
        vectors = self.model.encode(
            data,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        logger.info(f"Vectorized {len(data)} intelligence items to {self.vector_dimension}D space")

        # Pad or project to exactly 2048 dimensions if needed
        if vectors.shape[1] != 2048:
            vectors = self._project_to_2048d(vectors)

        return vectors

    def _project_to_2048d(self, vectors: np.ndarray) -> np.ndarray:
        """
        Project vectors to exactly 2048 dimensions

        Args:
            vectors: Input vectors

        Returns:
            2048-dimensional vectors
        """
        current_dim = vectors.shape[1]

        if current_dim == 2048:
            return vectors

        # If smaller, pad with zeros
        if current_dim < 2048:
            padding = np.zeros((vectors.shape[0], 2048 - current_dim))
            return np.hstack([vectors, padding])

        # If larger, use PCA-like projection (simplified)
        # In production, use proper dimensionality reduction
        return vectors[:, :2048]

    async def find_similar_intelligence(
        self,
        query_vector: np.ndarray,
        vector_database: np.ndarray,
        metadata: List[Dict[str, Any]],
        top_k: int = 10,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find similar intelligence items using vector similarity

        Args:
            query_vector: Query vector (2048-dim)
            vector_database: Database of vectors (n_samples, 2048)
            metadata: Metadata for each vector
            top_k: Number of top results to return
            threshold: Minimum similarity threshold

        Returns:
            List of similar items with scores
        """
        # Ensure query is 2D
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, vector_database)[0]

        # Get top-k indices above threshold
        indices = np.argsort(similarities)[::-1]
        filtered_indices = [i for i in indices if similarities[i] >= threshold][:top_k]

        results = []
        for idx in filtered_indices:
            result = metadata[idx].copy() if idx < len(metadata) else {}
            result['similarity_score'] = float(similarities[idx])
            result['vector_index'] = int(idx)
            results.append(result)

        logger.info(f"Found {len(results)} similar intelligence items")

        return results

    async def cluster_intelligence(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
        eps: float = 0.3,
        min_samples: int = 2
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Cluster intelligence items using DBSCAN

        Args:
            vectors: 2048-dimensional vectors
            metadata: Metadata for each vector
            eps: DBSCAN epsilon parameter
            min_samples: Minimum samples per cluster

        Returns:
            Dictionary mapping cluster IDs to items
        """
        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = clustering.fit_predict(vectors)

        # Organize by cluster
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            cluster_id = f"cluster_{label}" if label != -1 else "outliers"

            if cluster_id not in clusters:
                clusters[cluster_id] = []

            item = metadata[idx].copy() if idx < len(metadata) else {}
            item['cluster_label'] = int(label)
            item['vector_index'] = idx

            clusters[cluster_id].append(item)

        logger.info(
            f"Clustered intelligence into {len(clusters)} groups",
            num_outliers=len(clusters.get('outliers', []))
        )

        return clusters

    async def correlate_iocs(
        self,
        iocs: List[Dict[str, Any]],
        threshold: float = 0.75
    ) -> List[Dict[str, Any]]:
        """
        Correlate IoCs using vector similarity

        Args:
            iocs: List of IoC dictionaries with 'value' and 'context' fields
            threshold: Correlation threshold

        Returns:
            List of correlated IoC groups
        """
        if not iocs:
            return []

        # Extract contexts for vectorization
        contexts = [ioc.get('context', ioc.get('value', '')) for ioc in iocs]

        # Vectorize
        vectors = await self.vectorize_intelligence(contexts)

        # Find correlations
        similarity_matrix = cosine_similarity(vectors)

        correlations = []
        for i in range(len(iocs)):
            for j in range(i + 1, len(iocs)):
                if similarity_matrix[i][j] >= threshold:
                    correlations.append({
                        'ioc_1': iocs[i],
                        'ioc_2': iocs[j],
                        'correlation_score': float(similarity_matrix[i][j]),
                        'relationship': 'semantically_related'
                    })

        logger.info(f"Found {len(correlations)} IoC correlations above threshold {threshold}")

        return correlations

    async def integrate_external_vectors(
        self,
        system_name: str,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]]
    ):
        """
        Integrate vectors from an external system

        Args:
            system_name: Name of the external system
            vectors: Vectors from external system
            metadata: Metadata for each vector
        """
        # Project to 2048D if needed
        if vectors.shape[1] != 2048:
            vectors = self._project_to_2048d(vectors)

        self.external_vector_systems[system_name] = {
            'vectors': vectors,
            'metadata': metadata,
            'integrated_at': np.datetime64('now')
        }

        logger.info(
            f"Integrated {len(vectors)} vectors from external system '{system_name}'",
            dimension=vectors.shape[1]
        )

    async def cross_system_search(
        self,
        query: str,
        systems: List[str] = None,
        top_k_per_system: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across multiple integrated vector systems

        Args:
            query: Search query
            systems: List of system names to search (None for all)
            top_k_per_system: Number of results per system

        Returns:
            Dictionary mapping system names to results
        """
        # Vectorize query
        query_vec = self.model.encode(query, convert_to_numpy=True)
        query_vec = self._project_to_2048d(query_vec.reshape(1, -1))

        if systems is None:
            systems = list(self.external_vector_systems.keys())

        results = {}

        for system_name in systems:
            if system_name not in self.external_vector_systems:
                logger.warning(f"System '{system_name}' not found in integrated systems")
                continue

            system_data = self.external_vector_systems[system_name]

            # Search in this system
            system_results = await self.find_similar_intelligence(
                query_vec,
                system_data['vectors'],
                system_data['metadata'],
                top_k=top_k_per_system,
                threshold=settings.SEMANTIC_SIMILARITY_THRESHOLD
            )

            results[system_name] = system_results

        logger.info(f"Cross-system search completed across {len(results)} systems")

        return results

    async def comprehensive_intelligence_analysis(
        self,
        text_data: str
    ) -> Dict[str, Any]:
        """
        Perform comprehensive end-to-end intelligence analysis

        Args:
            text_data: Input text data

        Returns:
            Comprehensive analysis report
        """
        logger.info("Starting comprehensive intelligence analysis")

        # 1. Extract IoCs
        iocs = ioc_identifier.extract_iocs(text_data)

        # 2. Vectorize the input
        vector = await self.vectorize_intelligence([text_data])

        # 3. Find similar intelligence across systems
        cross_system_results = await self.cross_system_search(
            text_data,
            top_k_per_system=10
        )

        # 4. Correlate extracted IoCs
        ioc_correlations = await self.correlate_iocs([
            {'value': ioc.value, 'context': ioc.context, 'type': ioc.type}
            for ioc in iocs
        ])

        # 5. Fuzzy match against known threats (example)
        # In production, this would query threat databases

        report = {
            'analysis_type': 'comprehensive_intelligence',
            'vector_dimension': 2048,
            'extracted_iocs': {
                'total': len(iocs),
                'by_type': {},
                'high_threat': [ioc.value for ioc in iocs if ioc.threat_level in ['high', 'critical']]
            },
            'vector_representation': vector.tolist(),
            'cross_system_matches': cross_system_results,
            'ioc_correlations': {
                'total_correlations': len(ioc_correlations),
                'correlations': ioc_correlations[:10]  # Top 10
            },
            'threat_assessment': self._assess_threat_level(iocs, ioc_correlations)
        }

        logger.info("Comprehensive intelligence analysis completed")

        return report

    def _assess_threat_level(
        self,
        iocs: List,
        correlations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Assess overall threat level based on analysis

        Args:
            iocs: Extracted IoCs
            correlations: IoC correlations

        Returns:
            Threat assessment
        """
        critical_count = sum(1 for ioc in iocs if ioc.threat_level == 'critical')
        high_count = sum(1 for ioc in iocs if ioc.threat_level == 'high')

        if critical_count > 0:
            level = 'CRITICAL'
        elif high_count > 2:
            level = 'HIGH'
        elif high_count > 0 or len(correlations) > 5:
            level = 'MEDIUM'
        else:
            level = 'LOW'

        return {
            'threat_level': level,
            'critical_indicators': critical_count,
            'high_indicators': high_count,
            'correlation_density': len(correlations),
            'confidence': 0.8 if critical_count > 0 else 0.6
        }


# Global instance
fuzzy_vector_intel = FuzzyVectorIntelligence()
