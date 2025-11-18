"""
Attention Director
Phase 8.3: Multi-head attention and focus management

Implements attention mechanisms for selective information processing:
- Multi-head attention (à la Transformer architecture)
- Top-down (goal-driven) attention
- Bottom-up (stimulus-driven) attention
- Context-aware focus shifting
- Attention resource allocation

Based on:
- Vaswani et al. (2017). Attention is All You Need
- Posner & Petersen (1990). The Attention System of the Human Brain
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np
import structlog

logger = structlog.get_logger()

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class FocusStrategy(Enum):
    """Attention focus strategies"""
    TOP_DOWN = "top_down"          # Goal-driven (executive control)
    BOTTOM_UP = "bottom_up"        # Stimulus-driven (salient events)
    BALANCED = "balanced"          # Mix of both
    EXPLORATORY = "exploratory"    # Curiosity-driven
    HABITUAL = "habitual"          # Pattern-driven (autopilot)


@dataclass
class AttentionHead:
    """
    Single attention head.

    In multi-head attention, each head learns different patterns.
    """
    head_id: str
    head_index: int
    dimension: int

    # What this head focuses on
    specialization: str  # e.g., "temporal", "spatial", "semantic"

    # Learned parameters (in production: actual weight matrices)
    query_weight: Optional[np.ndarray] = None
    key_weight: Optional[np.ndarray] = None
    value_weight: Optional[np.ndarray] = None

    # Statistics
    activations_count: int = 0
    avg_attention_entropy: float = 0.0  # Higher = more diffuse attention


@dataclass
class AttentionResult:
    """Result of attention computation"""
    attended_items: List[str]  # Item IDs in order of attention
    attention_weights: Dict[str, float]  # item_id -> weight
    focus_strategy: FocusStrategy
    confidence: float  # How confident in attention allocation

    # Head-level results
    head_results: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Context
    query_context: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MultiHeadAttention:
    """
    Multi-head attention mechanism.

    Splits attention into multiple heads, each learning different patterns.
    Based on Transformer architecture (Vaswani et al., 2017).
    """

    def __init__(
        self,
        num_heads: int = 8,
        model_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention.

        Args:
            num_heads: Number of attention heads
            model_dim: Model dimensionality
            dropout: Dropout rate
        """
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads
        self.dropout = dropout

        # Create heads
        self.heads = [
            AttentionHead(
                head_id=f"head_{i}",
                head_index=i,
                dimension=self.head_dim,
                specialization=self._assign_specialization(i)
            )
            for i in range(num_heads)
        ]

        # PyTorch layers (if available)
        if TORCH_AVAILABLE:
            self.attention = nn.MultiheadAttention(
                embed_dim=model_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )

        logger.info(
            "Multi-head attention initialized",
            num_heads=num_heads,
            model_dim=model_dim
        )

    def _assign_specialization(self, head_index: int) -> str:
        """Assign specialization to attention head"""
        specializations = [
            "temporal",    # Time-based patterns
            "spatial",     # Spatial relationships
            "semantic",    # Meaning-based
            "episodic",    # Event sequences
            "associative", # Related concepts
            "causal",      # Cause-effect
            "hierarchical",# Part-whole
            "analogical"   # Similarities
        ]
        return specializations[head_index % len(specializations)]

    async def attend(
        self,
        query: np.ndarray,
        keys: List[np.ndarray],
        values: List[np.ndarray],
        key_ids: List[str],
        mask: Optional[np.ndarray] = None
    ) -> AttentionResult:
        """
        Compute multi-head attention.

        Args:
            query: Query vector (what we're looking for)
            keys: Key vectors (what items offer)
            values: Value vectors (actual item representations)
            key_ids: IDs for each key
            mask: Optional attention mask

        Returns:
            Attention result with weights for each item
        """
        if TORCH_AVAILABLE and hasattr(self, 'attention'):
            return await self._attend_torch(query, keys, values, key_ids, mask)
        else:
            return await self._attend_numpy(query, keys, values, key_ids, mask)

    async def _attend_torch(
        self,
        query: np.ndarray,
        keys: List[np.ndarray],
        values: List[np.ndarray],
        key_ids: List[str],
        mask: Optional[np.ndarray]
    ) -> AttentionResult:
        """PyTorch-based attention computation"""
        # Convert to tensors
        query_tensor = torch.from_numpy(query).unsqueeze(0).unsqueeze(0).float()
        key_tensor = torch.from_numpy(np.array(keys)).unsqueeze(0).float()
        value_tensor = torch.from_numpy(np.array(values)).unsqueeze(0).float()

        # Compute attention
        with torch.no_grad():
            attn_output, attn_weights = self.attention(
                query_tensor, key_tensor, value_tensor
            )

        # Extract weights
        weights = attn_weights.squeeze(0).squeeze(0).numpy()

        # Build result
        attention_dict = {
            key_id: float(weight)
            for key_id, weight in zip(key_ids, weights)
        }

        # Sort by weight
        sorted_items = sorted(
            attention_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return AttentionResult(
            attended_items=[item_id for item_id, _ in sorted_items],
            attention_weights=attention_dict,
            focus_strategy=FocusStrategy.BALANCED,
            confidence=self._compute_confidence(weights)
        )

    async def _attend_numpy(
        self,
        query: np.ndarray,
        keys: List[np.ndarray],
        values: List[np.ndarray],
        key_ids: List[str],
        mask: Optional[np.ndarray]
    ) -> AttentionResult:
        """NumPy-based attention computation (scaled dot-product)"""
        keys_array = np.array(keys)

        # Compute attention scores: Q·K^T / sqrt(d)
        scores = np.dot(query, keys_array.T) / np.sqrt(query.shape[0])

        # Apply mask if provided
        if mask is not None:
            scores = scores + mask

        # Softmax to get weights
        exp_scores = np.exp(scores - np.max(scores))
        weights = exp_scores / exp_scores.sum()

        # Build result
        attention_dict = {
            key_id: float(weight)
            for key_id, weight in zip(key_ids, weights)
        }

        # Sort by weight
        sorted_items = sorted(
            attention_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return AttentionResult(
            attended_items=[item_id for item_id, _ in sorted_items],
            attention_weights=attention_dict,
            focus_strategy=FocusStrategy.BALANCED,
            confidence=self._compute_confidence(weights)
        )

    def _compute_confidence(self, weights: np.ndarray) -> float:
        """
        Compute confidence in attention allocation.

        Higher confidence = more focused attention (low entropy).
        Lower confidence = diffuse attention (high entropy).
        """
        # Compute entropy
        weights_safe = weights + 1e-10  # Avoid log(0)
        entropy = -np.sum(weights_safe * np.log(weights_safe))

        # Normalize to [0, 1]
        max_entropy = np.log(len(weights))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Confidence is inverse of entropy
        confidence = 1.0 - normalized_entropy

        return float(confidence)


class AttentionDirector:
    """
    Attention Director for MEMSHADOW.

    Manages attention allocation across different information sources,
    implements focus strategies, and coordinates multi-head attention.

    Combines:
        - Multi-head attention (Transformer-style)
        - Top-down goal-driven attention
        - Bottom-up stimulus-driven attention
        - Context-aware focus shifting

    Example:
        director = AttentionDirector(num_heads=8)

        # Attend to items based on query
        result = await director.attend(
            query_context={"goal": "find security vulnerabilities"},
            items=memory_items,
            strategy=FocusStrategy.TOP_DOWN
        )

        # Top items get attention
        for item_id in result.attended_items[:3]:
            process_item(item_id)
    """

    def __init__(
        self,
        num_heads: int = 8,
        model_dim: int = 512,
        enable_top_down: bool = True,
        enable_bottom_up: bool = True
    ):
        """
        Initialize attention director.

        Args:
            num_heads: Number of attention heads
            model_dim: Model dimensionality
            enable_top_down: Enable goal-driven attention
            enable_bottom_up: Enable stimulus-driven attention
        """
        self.num_heads = num_heads
        self.model_dim = model_dim

        # Multi-head attention
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            model_dim=model_dim
        )

        # Attention modes
        self.enable_top_down = enable_top_down
        self.enable_bottom_up = enable_bottom_up

        # Current focus
        self.current_strategy = FocusStrategy.BALANCED
        self.focus_history: List[FocusStrategy] = []

        # Attention budget (simulates limited attention resources)
        self.attention_budget = 1.0  # 100%
        self.attention_allocated = 0.0

        logger.info(
            "Attention director initialized",
            num_heads=num_heads,
            top_down=enable_top_down,
            bottom_up=enable_bottom_up
        )

    async def attend(
        self,
        query_context: Dict[str, Any],
        items: List[Dict[str, Any]],
        strategy: Optional[FocusStrategy] = None
    ) -> AttentionResult:
        """
        Attend to items based on query context.

        Args:
            query_context: Context describing what to attend to
            items: Items to consider (must have 'id' and 'embedding')
            strategy: Focus strategy (defaults to current strategy)

        Returns:
            Attention result with ranked items
        """
        strategy = strategy or self.current_strategy

        # Extract embeddings
        item_ids = [item['id'] for item in items]
        embeddings = [
            self._get_embedding(item) for item in items
        ]

        # Create query vector based on context and strategy
        query_vector = self._create_query_vector(query_context, strategy)

        # Compute attention
        result = await self.mha.attend(
            query=query_vector,
            keys=embeddings,
            values=embeddings,
            key_ids=item_ids
        )

        # Apply strategy-specific modifications
        result = await self._apply_strategy(result, strategy, query_context)

        # Update focus history
        self.focus_history.append(strategy)
        if len(self.focus_history) > 100:
            self.focus_history.pop(0)

        logger.debug(
            "Attention computed",
            strategy=strategy.value,
            items_count=len(items),
            top_item=result.attended_items[0] if result.attended_items else None,
            confidence=result.confidence
        )

        return result

    async def shift_focus(
        self,
        new_strategy: FocusStrategy,
        reason: str
    ):
        """
        Shift attention focus to new strategy.

        Args:
            new_strategy: New focus strategy
            reason: Why the shift occurred
        """
        old_strategy = self.current_strategy
        self.current_strategy = new_strategy

        logger.info(
            "Focus shifted",
            from_strategy=old_strategy.value,
            to_strategy=new_strategy.value,
            reason=reason
        )

    async def allocate_attention(
        self,
        task_name: str,
        amount: float
    ) -> bool:
        """
        Allocate attention budget to a task.

        Args:
            task_name: Task requesting attention
            amount: Amount of attention needed (0.0 to 1.0)

        Returns:
            True if allocation succeeded
        """
        if self.attention_allocated + amount > self.attention_budget:
            logger.warning(
                "Insufficient attention budget",
                task=task_name,
                requested=amount,
                available=self.attention_budget - self.attention_allocated
            )
            return False

        self.attention_allocated += amount

        logger.debug(
            "Attention allocated",
            task=task_name,
            amount=amount,
            remaining=self.attention_budget - self.attention_allocated
        )

        return True

    async def release_attention(self, amount: float):
        """Release attention budget"""
        self.attention_allocated = max(0, self.attention_allocated - amount)

    async def get_focus_stats(self) -> Dict[str, Any]:
        """Get attention focus statistics"""
        # Count strategy usage
        strategy_counts = {}
        for strategy in self.focus_history:
            strategy_counts[strategy.value] = strategy_counts.get(strategy.value, 0) + 1

        return {
            "current_strategy": self.current_strategy.value,
            "attention_budget": self.attention_budget,
            "attention_allocated": self.attention_allocated,
            "attention_available": self.attention_budget - self.attention_allocated,
            "focus_history_size": len(self.focus_history),
            "strategy_distribution": strategy_counts
        }

    # Private methods

    def _get_embedding(self, item: Dict[str, Any]) -> np.ndarray:
        """Extract or create embedding for item"""
        if 'embedding' in item:
            return np.array(item['embedding'])

        # Create mock embedding
        return np.random.randn(self.model_dim)

    def _create_query_vector(
        self,
        query_context: Dict[str, Any],
        strategy: FocusStrategy
    ) -> np.ndarray:
        """
        Create query vector based on context and strategy.

        In production: would use learned embeddings of goals/contexts.
        """
        # Mock implementation
        query = np.random.randn(self.model_dim)

        # Modify based on strategy
        if strategy == FocusStrategy.TOP_DOWN:
            # Focus on goal-relevant features
            query = query * 1.5

        elif strategy == FocusStrategy.BOTTOM_UP:
            # More uniform attention
            query = query * 0.8

        elif strategy == FocusStrategy.EXPLORATORY:
            # Add noise for exploration
            query = query + np.random.randn(self.model_dim) * 0.3

        return query

    async def _apply_strategy(
        self,
        result: AttentionResult,
        strategy: FocusStrategy,
        context: Dict[str, Any]
    ) -> AttentionResult:
        """Apply strategy-specific modifications to attention result"""

        if strategy == FocusStrategy.BOTTOM_UP:
            # Boost salient items
            result = self._boost_salient(result, context)

        elif strategy == FocusStrategy.TOP_DOWN:
            # Boost goal-relevant items
            result = self._boost_relevant(result, context)

        elif strategy == FocusStrategy.EXPLORATORY:
            # Add randomness to encourage exploration
            result = self._add_exploration_noise(result)

        result.focus_strategy = strategy
        result.query_context = context

        return result

    def _boost_salient(
        self,
        result: AttentionResult,
        context: Dict[str, Any]
    ) -> AttentionResult:
        """Boost attention to salient (attention-grabbing) items"""
        # In production: would check item salience scores
        return result

    def _boost_relevant(
        self,
        result: AttentionResult,
        context: Dict[str, Any]
    ) -> AttentionResult:
        """Boost attention to goal-relevant items"""
        # In production: would check relevance to current goals
        return result

    def _add_exploration_noise(self, result: AttentionResult) -> AttentionResult:
        """Add noise to encourage exploration"""
        # Add small random perturbations to weights
        for item_id in result.attention_weights:
            noise = np.random.uniform(-0.05, 0.05)
            result.attention_weights[item_id] = max(
                0, result.attention_weights[item_id] + noise
            )

        # Re-normalize
        total = sum(result.attention_weights.values())
        if total > 0:
            for item_id in result.attention_weights:
                result.attention_weights[item_id] /= total

        return result
