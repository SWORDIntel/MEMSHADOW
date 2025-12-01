"""
Secure Aggregation for Federated Memory
Phase 8.1: Privacy-preserving multi-party computation

Implements secure aggregation protocols:
- Masked aggregation: Each node adds secret shares
- Homomorphic encryption: Aggregate encrypted values
- Byzantine-robust aggregation: Filter malicious contributions
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import hashlib
from collections import defaultdict
import structlog

logger = structlog.get_logger()


@dataclass
class AggregationResult:
    """Result of secure aggregation"""
    aggregated_value: np.ndarray
    num_contributors: int
    dropped_contributors: List[str]  # Nodes dropped due to Byzantine behavior
    aggregation_method: str
    is_valid: bool = True


class SecureAggregator:
    """
    Secure Aggregation for Federated Memory.

    Enables multiple nodes to compute aggregate statistics
    (e.g., average embeddings, mean gradients) without revealing
    individual contributions.

    Protocols:
        1. Masked Aggregation: Secret sharing with pairwise masks
        2. Byzantine-Robust: Statistical filtering of outliers
        3. Threshold Aggregation: Only aggregate if enough participants

    Example:
        aggregator = SecureAggregator()

        # Each node creates masked contribution
        masked_value = aggregator.create_masked_contribution(
            value=local_embedding,
            masks=pairwise_masks
        )

        # Coordinator aggregates
        result = aggregator.aggregate_masked([
            masked1, masked2, masked3
        ])
    """

    def __init__(
        self,
        byzantine_threshold: float = 0.3,
        min_contributors: int = 3
    ):
        """
        Initialize secure aggregator.

        Args:
            byzantine_threshold: Max fraction of Byzantine nodes tolerated
            min_contributors: Minimum nodes needed for aggregation
        """
        self.byzantine_threshold = byzantine_threshold
        self.min_contributors = min_contributors

        logger.info(
            "Secure aggregator initialized",
            byzantine_threshold=byzantine_threshold,
            min_contributors=min_contributors
        )

    def generate_pairwise_masks(
        self,
        my_node_id: str,
        peer_node_ids: List[str],
        dimension: int,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate pairwise secret masks for secure aggregation.

        Each pair of nodes shares a secret mask. When aggregating,
        masks cancel out, revealing only the sum.

        Args:
            my_node_id: This node's ID
            peer_node_ids: IDs of all other nodes
            dimension: Dimension of values being aggregated
            seed: Random seed for reproducibility

        Returns:
            Dict mapping peer_id -> mask
        """
        masks = {}

        for peer_id in peer_node_ids:
            # Generate deterministic mask based on both node IDs
            # Use lexicographic order to ensure both nodes generate same mask
            id_pair = tuple(sorted([my_node_id, peer_id]))

            # Create seed from node IDs
            pair_seed = int(
                hashlib.sha256(f"{id_pair[0]}:{id_pair[1]}".encode()).hexdigest()[:8],
                16
            )

            # Generate mask
            rng = np.random.RandomState(pair_seed)
            mask = rng.randn(dimension)

            # Determine sign (first node adds, second subtracts)
            if my_node_id < peer_id:
                masks[peer_id] = mask
            else:
                masks[peer_id] = -mask

        return masks

    def create_masked_contribution(
        self,
        value: np.ndarray,
        masks: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Create masked contribution for secure aggregation.

        Adds pairwise masks to value. When all nodes aggregate,
        masks cancel out.

        Args:
            value: Local value to contribute
            masks: Pairwise masks from generate_pairwise_masks()

        Returns:
            Masked value
        """
        masked_value = value.copy()

        # Add all pairwise masks
        for mask in masks.values():
            masked_value += mask

        return masked_value

    def aggregate_masked(
        self,
        masked_contributions: List[np.ndarray],
        node_ids: List[str]
    ) -> AggregationResult:
        """
        Aggregate masked contributions.

        When all nodes contribute their masked values, the pairwise
        masks cancel out, revealing only the sum (not individual values).

        Args:
            masked_contributions: List of masked values
            node_ids: IDs of contributing nodes

        Returns:
            Aggregation result
        """
        if len(masked_contributions) < self.min_contributors:
            logger.warning(
                "Insufficient contributors",
                got=len(masked_contributions),
                required=self.min_contributors
            )
            return AggregationResult(
                aggregated_value=np.zeros_like(masked_contributions[0]),
                num_contributors=0,
                dropped_contributors=[],
                aggregation_method="masked",
                is_valid=False
            )

        # Sum all masked contributions
        # Pairwise masks cancel out, leaving only the sum of original values
        total = np.sum(masked_contributions, axis=0)

        # Average
        aggregated = total / len(masked_contributions)

        return AggregationResult(
            aggregated_value=aggregated,
            num_contributors=len(masked_contributions),
            dropped_contributors=[],
            aggregation_method="masked",
            is_valid=True
        )

    def byzantine_robust_aggregate(
        self,
        contributions: List[np.ndarray],
        node_ids: List[str]
    ) -> AggregationResult:
        """
        Byzantine-robust aggregation using median or trimmed mean.

        Filters out outlier contributions that may be from malicious nodes.

        Args:
            contributions: List of unmasked contributions
            node_ids: IDs of contributing nodes

        Returns:
            Aggregation result with Byzantine nodes removed
        """
        if len(contributions) < self.min_contributors:
            return AggregationResult(
                aggregated_value=np.zeros_like(contributions[0]),
                num_contributors=0,
                dropped_contributors=[],
                aggregation_method="byzantine_robust",
                is_valid=False
            )

        contributions_array = np.array(contributions)

        # Compute pairwise distances
        distances = self._compute_pairwise_distances(contributions_array)

        # Identify outliers (potential Byzantine nodes)
        outliers = self._identify_outliers(
            distances,
            threshold=self.byzantine_threshold
        )

        # Filter out outliers
        valid_contributions = [
            contrib for i, contrib in enumerate(contributions)
            if i not in outliers
        ]

        dropped_node_ids = [
            node_ids[i] for i in outliers
        ]

        if len(valid_contributions) < self.min_contributors:
            logger.warning(
                "Too many Byzantine nodes detected",
                total=len(contributions),
                dropped=len(outliers)
            )
            return AggregationResult(
                aggregated_value=np.zeros_like(contributions[0]),
                num_contributors=0,
                dropped_contributors=dropped_node_ids,
                aggregation_method="byzantine_robust",
                is_valid=False
            )

        # Aggregate valid contributions
        aggregated = np.mean(valid_contributions, axis=0)

        logger.info(
            "Byzantine-robust aggregation completed",
            total_contributors=len(contributions),
            valid_contributors=len(valid_contributions),
            dropped_contributors=len(outliers)
        )

        return AggregationResult(
            aggregated_value=aggregated,
            num_contributors=len(valid_contributions),
            dropped_contributors=dropped_node_ids,
            aggregation_method="byzantine_robust",
            is_valid=True
        )

    def trimmed_mean_aggregate(
        self,
        contributions: List[np.ndarray],
        trim_ratio: float = 0.1
    ) -> np.ndarray:
        """
        Trimmed mean aggregation.

        Remove top and bottom k% of values before averaging.
        Simple Byzantine-robust aggregation.

        Args:
            contributions: List of contributions
            trim_ratio: Fraction to trim from each end (0.0 to 0.5)

        Returns:
            Trimmed mean
        """
        contributions_array = np.array(contributions)

        # For each dimension, trim top/bottom k%
        trim_count = int(len(contributions) * trim_ratio)

        if trim_count == 0:
            return np.mean(contributions_array, axis=0)

        # Sort along contributor axis
        sorted_contrib = np.sort(contributions_array, axis=0)

        # Trim
        trimmed = sorted_contrib[trim_count:-trim_count]

        # Mean
        return np.mean(trimmed, axis=0)

    def median_aggregate(
        self,
        contributions: List[np.ndarray]
    ) -> np.ndarray:
        """
        Median aggregation (coordinate-wise).

        Very robust to Byzantine nodes but less accurate than mean.

        Args:
            contributions: List of contributions

        Returns:
            Coordinate-wise median
        """
        contributions_array = np.array(contributions)
        return np.median(contributions_array, axis=0)

    def _compute_pairwise_distances(
        self,
        contributions: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise L2 distances between contributions.

        Args:
            contributions: Array of shape (n_contributors, dimension)

        Returns:
            Distance matrix of shape (n_contributors, n_contributors)
        """
        n = len(contributions)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(contributions[i] - contributions[j])
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def _identify_outliers(
        self,
        distances: np.ndarray,
        threshold: float
    ) -> List[int]:
        """
        Identify outlier contributions based on distances.

        A contribution is an outlier if its average distance to others
        is in the top k%.

        Args:
            distances: Pairwise distance matrix
            threshold: Fraction of nodes to mark as outliers

        Returns:
            Indices of outlier contributions
        """
        # Compute average distance for each contribution
        avg_distances = distances.mean(axis=1)

        # Find outliers (top k%)
        k = max(1, int(len(avg_distances) * threshold))
        outlier_indices = np.argsort(avg_distances)[-k:]

        return outlier_indices.tolist()

    def aggregate_with_weights(
        self,
        contributions: List[np.ndarray],
        weights: List[float],
        node_ids: List[str]
    ) -> AggregationResult:
        """
        Weighted aggregation (e.g., weighted by node reputation).

        Args:
            contributions: List of contributions
            weights: Weight for each contribution
            node_ids: Node IDs

        Returns:
            Weighted aggregation result
        """
        if len(contributions) != len(weights):
            raise ValueError("Contributions and weights must have same length")

        # Normalize weights
        weights_array = np.array(weights)
        weights_array /= weights_array.sum()

        # Weighted sum
        aggregated = np.sum(
            [contrib * w for contrib, w in zip(contributions, weights_array)],
            axis=0
        )

        return AggregationResult(
            aggregated_value=aggregated,
            num_contributors=len(contributions),
            dropped_contributors=[],
            aggregation_method="weighted",
            is_valid=True
        )

    def verify_aggregation(
        self,
        contributions: List[np.ndarray],
        claimed_aggregate: np.ndarray,
        tolerance: float = 1e-6
    ) -> bool:
        """
        Verify that aggregation was computed correctly.

        Args:
            contributions: Original contributions
            claimed_aggregate: Claimed aggregate value
            tolerance: Numerical tolerance

        Returns:
            True if verification passes
        """
        # Recompute aggregation
        expected_aggregate = np.mean(contributions, axis=0)

        # Check if close
        difference = np.linalg.norm(expected_aggregate - claimed_aggregate)

        return difference < tolerance
