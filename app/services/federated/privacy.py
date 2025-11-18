"""
Differential Privacy for Federated Memory
Phase 8.1: Privacy-preserving memory sharing

Implements ε-differential privacy mechanisms:
- Laplace mechanism for numeric data
- Exponential mechanism for categorical data
- Gaussian mechanism for deep learning gradients
- Privacy budget accounting
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime
import structlog

logger = structlog.get_logger()


class PrivacyMechanism(Enum):
    """Privacy-preserving mechanisms"""
    LAPLACE = "laplace"  # For numeric queries
    GAUSSIAN = "gaussian"  # For gradients (zero-concentrated DP)
    EXPONENTIAL = "exponential"  # For categorical selection
    RANDOMIZED_RESPONSE = "randomized_response"  # For boolean data


@dataclass
class PrivacyBudget:
    """
    Differential privacy budget tracker.

    Tracks ε (epsilon) and δ (delta) parameters across queries.
    """
    epsilon: float  # Privacy loss parameter
    delta: float = 1e-5  # Failure probability

    # Spent budget
    epsilon_spent: float = 0.0
    queries_made: int = 0

    # Composition tracking
    query_history: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.query_history is None:
            self.query_history = []

    @property
    def epsilon_remaining(self) -> float:
        """Remaining privacy budget"""
        return max(0.0, self.epsilon - self.epsilon_spent)

    @property
    def budget_used_percent(self) -> float:
        """Percentage of budget used"""
        return (self.epsilon_spent / self.epsilon) * 100 if self.epsilon > 0 else 100.0

    def can_afford(self, epsilon_cost: float) -> bool:
        """Check if we can afford a query"""
        return self.epsilon_remaining >= epsilon_cost

    def spend(self, epsilon_cost: float, mechanism: PrivacyMechanism, query_type: str):
        """Spend privacy budget"""
        if not self.can_afford(epsilon_cost):
            raise ValueError(
                f"Insufficient privacy budget. "
                f"Remaining: {self.epsilon_remaining:.3f}, "
                f"Requested: {epsilon_cost:.3f}"
            )

        self.epsilon_spent += epsilon_cost
        self.queries_made += 1

        # Record query
        self.query_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "epsilon": epsilon_cost,
            "mechanism": mechanism.value,
            "query_type": query_type,
            "cumulative_epsilon": self.epsilon_spent
        })

        logger.debug(
            "Privacy budget spent",
            epsilon_cost=epsilon_cost,
            remaining=self.epsilon_remaining,
            percent_used=self.budget_used_percent
        )


class DifferentialPrivacy:
    """
    Differential Privacy implementation for federated memory.

    Provides privacy-preserving mechanisms for sharing memory updates
    across a federation without revealing individual user data.

    Mechanisms:
        - Laplace: Add noise to numeric data
        - Gaussian: Add noise to gradients
        - Exponential: Private selection from set
        - Randomized Response: Private boolean data

    Example:
        dp = DifferentialPrivacy(epsilon=1.0)

        # Add noise to embedding
        noisy_embedding = dp.laplace_mechanism(
            embedding,
            sensitivity=0.1,
            epsilon=0.1
        )

        # Private gradient sharing
        noisy_gradient = dp.gaussian_mechanism(
            gradient,
            sensitivity=1.0,
            epsilon=0.2,
            delta=1e-5
        )
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize differential privacy.

        Args:
            epsilon: Total privacy budget
            delta: Failure probability
        """
        self.budget = PrivacyBudget(epsilon=epsilon, delta=delta)

        logger.info(
            "Differential privacy initialized",
            epsilon=epsilon,
            delta=delta
        )

    def laplace_mechanism(
        self,
        data: Union[float, np.ndarray],
        sensitivity: float,
        epsilon: float
    ) -> Union[float, np.ndarray]:
        """
        Laplace mechanism for numeric data.

        Adds Laplace noise calibrated to sensitivity and epsilon.

        Args:
            data: Numeric data to privatize
            sensitivity: L1 sensitivity of the query
            epsilon: Privacy budget for this query

        Returns:
            Privatized data with Laplace noise
        """
        # Check budget
        if not self.budget.can_afford(epsilon):
            raise ValueError("Insufficient privacy budget")

        # Calculate noise scale
        scale = sensitivity / epsilon

        # Add Laplace noise
        if isinstance(data, np.ndarray):
            noise = np.random.laplace(0, scale, data.shape)
            noisy_data = data + noise
        else:
            noise = np.random.laplace(0, scale)
            noisy_data = data + noise

        # Spend budget
        self.budget.spend(epsilon, PrivacyMechanism.LAPLACE, "numeric_query")

        return noisy_data

    def gaussian_mechanism(
        self,
        data: np.ndarray,
        sensitivity: float,
        epsilon: float,
        delta: Optional[float] = None
    ) -> np.ndarray:
        """
        Gaussian mechanism for gradient sharing.

        Adds Gaussian noise for (ε, δ)-differential privacy.
        Preferred for deep learning gradients.

        Args:
            data: Data to privatize (typically gradients)
            sensitivity: L2 sensitivity
            epsilon: Privacy budget
            delta: Failure probability (defaults to budget delta)

        Returns:
            Privatized data with Gaussian noise
        """
        delta = delta or self.budget.delta

        # Check budget
        if not self.budget.can_afford(epsilon):
            raise ValueError("Insufficient privacy budget")

        # Calculate noise scale (for (ε, δ)-DP)
        sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon

        # Add Gaussian noise
        noise = np.random.normal(0, sigma, data.shape)
        noisy_data = data + noise

        # Spend budget (Gaussian has delta cost too)
        self.budget.spend(epsilon, PrivacyMechanism.GAUSSIAN, "gradient_sharing")

        return noisy_data

    def exponential_mechanism(
        self,
        candidates: List[Any],
        quality_scores: List[float],
        sensitivity: float,
        epsilon: float
    ) -> Any:
        """
        Exponential mechanism for private selection.

        Selects from a set of candidates with probability proportional
        to quality score, while preserving privacy.

        Args:
            candidates: List of candidate items
            quality_scores: Quality score for each candidate
            sensitivity: Sensitivity of quality function
            epsilon: Privacy budget

        Returns:
            Selected candidate
        """
        if len(candidates) != len(quality_scores):
            raise ValueError("Candidates and scores must have same length")

        # Check budget
        if not self.budget.can_afford(epsilon):
            raise ValueError("Insufficient privacy budget")

        # Calculate selection probabilities
        scores = np.array(quality_scores)
        probabilities = np.exp((epsilon * scores) / (2 * sensitivity))
        probabilities /= probabilities.sum()

        # Sample candidate
        selected_idx = np.random.choice(len(candidates), p=probabilities)
        selected = candidates[selected_idx]

        # Spend budget
        self.budget.spend(epsilon, PrivacyMechanism.EXPONENTIAL, "selection")

        return selected

    def randomized_response(
        self,
        value: bool,
        epsilon: float
    ) -> bool:
        """
        Randomized response for boolean data.

        Simple local differential privacy for boolean values.

        Args:
            value: True boolean value
            epsilon: Privacy budget

        Returns:
            Privatized boolean value
        """
        # Check budget
        if not self.budget.can_afford(epsilon):
            raise ValueError("Insufficient privacy budget")

        # Calculate flip probability
        p_truth = np.exp(epsilon) / (np.exp(epsilon) + 1)

        # Flip coin
        privatized = value if np.random.random() < p_truth else not value

        # Spend budget
        self.budget.spend(epsilon, PrivacyMechanism.RANDOMIZED_RESPONSE, "boolean")

        return privatized

    def clip_gradients(
        self,
        gradients: np.ndarray,
        max_norm: float
    ) -> np.ndarray:
        """
        Clip gradients to bound sensitivity.

        Essential preprocessing for DP-SGD (differentially private SGD).

        Args:
            gradients: Gradient tensor
            max_norm: Maximum L2 norm

        Returns:
            Clipped gradients
        """
        norm = np.linalg.norm(gradients)

        if norm > max_norm:
            gradients = gradients * (max_norm / norm)

        return gradients

    def aggregate_with_privacy(
        self,
        updates: List[np.ndarray],
        epsilon: float,
        clip_norm: float = 1.0
    ) -> np.ndarray:
        """
        Aggregate updates from multiple nodes with differential privacy.

        Implements DP-FedAvg (Differentially Private Federated Averaging).

        Args:
            updates: List of updates from different nodes
            epsilon: Privacy budget for aggregation
            clip_norm: Clipping threshold for updates

        Returns:
            Aggregated update with privacy guarantee
        """
        # Clip each update
        clipped_updates = [
            self.clip_gradients(update, clip_norm)
            for update in updates
        ]

        # Average
        avg_update = np.mean(clipped_updates, axis=0)

        # Add noise for privacy
        sensitivity = 2 * clip_norm / len(updates)  # L2 sensitivity
        noisy_avg = self.gaussian_mechanism(
            avg_update,
            sensitivity=sensitivity,
            epsilon=epsilon
        )

        return noisy_avg

    def get_budget_status(self) -> Dict[str, Any]:
        """Get privacy budget status"""
        return {
            "epsilon_total": self.budget.epsilon,
            "epsilon_spent": self.budget.epsilon_spent,
            "epsilon_remaining": self.budget.epsilon_remaining,
            "budget_used_percent": self.budget.budget_used_percent,
            "queries_made": self.budget.queries_made,
            "delta": self.budget.delta
        }

    def reset_budget(self):
        """Reset privacy budget (use with extreme caution!)"""
        logger.warning("Resetting privacy budget - this should be rare!")

        self.budget.epsilon_spent = 0.0
        self.budget.queries_made = 0
        self.budget.query_history = []


def calculate_composition_epsilon(
    epsilons: List[float],
    composition_type: str = "advanced"
) -> float:
    """
    Calculate composed privacy budget.

    When making multiple queries, privacy degrades. This calculates
    the total privacy loss under different composition theorems.

    Args:
        epsilons: List of epsilon values from individual queries
        composition_type: "basic", "advanced", or "zcdp"

    Returns:
        Composed epsilon
    """
    if composition_type == "basic":
        # Basic composition: sum all epsilons
        return sum(epsilons)

    elif composition_type == "advanced":
        # Advanced composition: sqrt(2k ln(1/δ)) * ε
        # Tighter bound for many queries
        k = len(epsilons)
        max_epsilon = max(epsilons) if epsilons else 0

        return max_epsilon * np.sqrt(2 * k * np.log(1 / 1e-5))

    elif composition_type == "zcdp":
        # Zero-concentrated differential privacy
        # ρ-zCDP: sum of ε²
        return np.sqrt(sum(eps ** 2 for eps in epsilons))

    else:
        raise ValueError(f"Unknown composition type: {composition_type}")


def estimate_noise_impact(
    epsilon: float,
    sensitivity: float,
    data_range: float
) -> Dict[str, float]:
    """
    Estimate impact of differential privacy noise.

    Helps understand the privacy-utility tradeoff.

    Args:
        epsilon: Privacy parameter
        sensitivity: Query sensitivity
        data_range: Range of data values

    Returns:
        Noise statistics
    """
    # Laplace noise scale
    scale = sensitivity / epsilon

    # Noise statistics
    noise_std = scale * np.sqrt(2)
    noise_mean = 0.0

    # Signal-to-noise ratio
    snr = data_range / noise_std if noise_std > 0 else float('inf')

    # Expected relative error
    relative_error = noise_std / data_range if data_range > 0 else float('inf')

    return {
        "noise_scale": scale,
        "noise_std": noise_std,
        "noise_mean": noise_mean,
        "signal_to_noise_ratio": snr,
        "expected_relative_error": relative_error,
        "epsilon": epsilon
    }
