"""
PatternCompletionEngine - Memory Pattern Completion

For partial or noisy cues:
1. ANN search in embedding space (T0-T2)
2. Blend with graph neighborhood from SpreadingActivation
3. Optional re-ranking via small cross-encoder / reranker

Supports "I've seen something like this before" across many AIs.
"""

import asyncio
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from uuid import UUID
import structlog

from .core_abstractions import MemoryGraph, MemoryGraphView, MemoryObject, StorageTierLevel
from .spreading_activation import SpreadingActivationKernel, ActivationConstraints, ActivationResult

logger = structlog.get_logger()


@dataclass
class CompletionCandidate:
    """A candidate memory for pattern completion"""
    memory_id: UUID
    score: float
    source: str  # "ann", "graph", "rerank"
    embedding_similarity: float = 0.0
    graph_activation: float = 0.0
    rerank_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResult:
    """Result of pattern completion"""
    candidates: List[CompletionCandidate]
    query_coverage: float  # How well the results cover the query
    confidence: float  # Overall confidence in completion
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ANNIndex:
    """Simple ANN index for a tier (would use FAISS/ScaNN in production)"""
    tier: StorageTierLevel
    embeddings: np.ndarray  # (n, dim) matrix
    memory_ids: List[UUID]
    dimension: int

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[UUID, float]]:
        """Brute force search (replace with actual ANN in production)"""
        if len(self.embeddings) == 0:
            return []

        # Handle dimension mismatch
        query = np.asarray(query, dtype=np.float32)
        if len(query) != self.dimension:
            # Simple projection/truncation
            if len(query) > self.dimension:
                query = query[:self.dimension]
            else:
                query = np.pad(query, (0, self.dimension - len(query)))

        # Normalize
        q_norm = np.linalg.norm(query)
        if q_norm > 0:
            query = query / q_norm

        # Compute similarities
        similarities = np.dot(self.embeddings, query)

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim >= threshold:
                results.append((self.memory_ids[idx], sim))

        return results


class PatternCompletionEngine:
    """
    Engine for completing partial memory patterns.

    Combines:
    1. ANN search in embedding space for semantic matches
    2. Graph-based spreading activation for contextual matches
    3. Re-ranking for final ordering

    This enables "I've seen something like this before" across many AIs.
    """

    def __init__(
        self,
        graph: MemoryGraph,
        activation_kernel: SpreadingActivationKernel,
        embedding_hub: Any = None,
        ann_weight: float = 0.4,
        graph_weight: float = 0.4,
        rerank_weight: float = 0.2
    ):
        self.graph = graph
        self.activation_kernel = activation_kernel
        self.embedding_hub = embedding_hub

        self.ann_weight = ann_weight
        self.graph_weight = graph_weight
        self.rerank_weight = rerank_weight

        # ANN indexes per tier
        self.ann_indexes: Dict[StorageTierLevel, ANNIndex] = {}

        # Reranker (could be a small cross-encoder)
        self.reranker: Optional[Callable] = None

        # Statistics
        self.stats = {
            "completions": 0,
            "avg_candidates": 0,
            "avg_confidence": 0,
        }

        logger.info("PatternCompletionEngine initialized",
                   ann_weight=ann_weight,
                   graph_weight=graph_weight)

    def build_ann_index(
        self,
        tier: StorageTierLevel,
        memories: List[MemoryObject],
        dimension: int
    ):
        """Build ANN index for a tier"""
        if not memories:
            return

        embeddings = []
        memory_ids = []

        for mem in memories:
            # Get canonical embedding or project
            if mem.canonical_embedding is not None:
                emb = mem.canonical_embedding
                if len(emb) != dimension:
                    if self.embedding_hub:
                        emb = self.embedding_hub.project_between(emb, dimension)
                    else:
                        # Simple truncation/padding
                        if len(emb) > dimension:
                            emb = emb[:dimension]
                        else:
                            emb = np.pad(emb, (0, dimension - len(emb)))

                # Normalize
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

                embeddings.append(emb)
                memory_ids.append(mem.id)

        if embeddings:
            self.ann_indexes[tier] = ANNIndex(
                tier=tier,
                embeddings=np.array(embeddings, dtype=np.float32),
                memory_ids=memory_ids,
                dimension=dimension
            )

            logger.info("ANN index built",
                       tier=tier.name,
                       num_vectors=len(embeddings),
                       dimension=dimension)

    async def complete(
        self,
        query_embedding: np.ndarray,
        partial_cues: Optional[List[UUID]] = None,
        constraints: Optional[ActivationConstraints] = None,
        top_k: int = 20,
        use_reranker: bool = True
    ) -> CompletionResult:
        """
        Complete a partial memory pattern.

        Args:
            query_embedding: The query to complete
            partial_cues: Optional seed memory IDs as cues
            constraints: Policy constraints
            top_k: Number of results to return
            use_reranker: Whether to apply reranking

        Returns:
            CompletionResult with ranked candidates
        """
        self.stats["completions"] += 1

        candidates: Dict[UUID, CompletionCandidate] = {}

        # 1. ANN search across tiers
        ann_results = await self._ann_search(
            query_embedding,
            top_k=top_k * 2,
            constraints=constraints
        )

        for memory_id, similarity in ann_results:
            if memory_id not in candidates:
                candidates[memory_id] = CompletionCandidate(
                    memory_id=memory_id,
                    score=0.0,
                    source="ann"
                )
            candidates[memory_id].embedding_similarity = similarity

        # 2. Graph-based activation
        if partial_cues or ann_results:
            seed_ids = partial_cues or [mid for mid, _ in ann_results[:5]]
            activation_result = await self.activation_kernel.activate(
                seed_ids=seed_ids,
                query_embedding=query_embedding,
                constraints=constraints
            )

            for memory_id, activation in activation_result.activations.items():
                if memory_id not in candidates:
                    candidates[memory_id] = CompletionCandidate(
                        memory_id=memory_id,
                        score=0.0,
                        source="graph"
                    )
                candidates[memory_id].graph_activation = activation

        # 3. Combine scores
        for cand in candidates.values():
            cand.score = (
                self.ann_weight * cand.embedding_similarity +
                self.graph_weight * cand.graph_activation
            )

        # 4. Optional reranking
        if use_reranker and self.reranker:
            candidates = await self._rerank(
                query_embedding, candidates, constraints
            )

        # Sort and select top-k
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda c: c.score,
            reverse=True
        )[:top_k]

        # Calculate confidence
        confidence = self._calculate_confidence(sorted_candidates)

        return CompletionResult(
            candidates=sorted_candidates,
            query_coverage=self._calculate_coverage(sorted_candidates, query_embedding),
            confidence=confidence,
            stats={
                "ann_candidates": len(ann_results),
                "graph_candidates": len(activation_result.activations) if partial_cues or ann_results else 0,
                "total_candidates": len(candidates),
            }
        )

    async def _ann_search(
        self,
        query: np.ndarray,
        top_k: int,
        constraints: Optional[ActivationConstraints]
    ) -> List[Tuple[UUID, float]]:
        """Search across all ANN indexes"""
        all_results = []

        for tier, index in self.ann_indexes.items():
            tier_results = index.search(query, k=top_k, threshold=0.3)
            all_results.extend(tier_results)

        # Deduplicate and sort
        seen = set()
        unique_results = []
        for memory_id, score in sorted(all_results, key=lambda x: x[1], reverse=True):
            if memory_id not in seen:
                # Apply policy filter if constraints provided
                if constraints and constraints.agent_id:
                    node = self.graph.nodes.get(memory_id)
                    if node and not node.is_accessible_by(
                        constraints.agent_id, constraints.agent_groups
                    ):
                        continue
                seen.add(memory_id)
                unique_results.append((memory_id, score))

        return unique_results[:top_k]

    async def _rerank(
        self,
        query: np.ndarray,
        candidates: Dict[UUID, CompletionCandidate],
        constraints: Optional[ActivationConstraints]
    ) -> Dict[UUID, CompletionCandidate]:
        """Apply reranking to candidates"""
        if not self.reranker:
            return candidates

        # Get memory content for reranking
        for memory_id, cand in candidates.items():
            node = self.graph.nodes.get(memory_id)
            if node:
                # Call reranker (would be cross-encoder in production)
                try:
                    rerank_score = await self.reranker(query, node)
                    cand.rerank_score = rerank_score
                    cand.score = (
                        (1 - self.rerank_weight) * cand.score +
                        self.rerank_weight * rerank_score
                    )
                except Exception as e:
                    logger.warning("Rerank failed", memory_id=str(memory_id), error=str(e))

        return candidates

    def _calculate_confidence(self, candidates: List[CompletionCandidate]) -> float:
        """Calculate overall completion confidence"""
        if not candidates:
            return 0.0

        # Confidence based on score distribution
        scores = [c.score for c in candidates]
        max_score = max(scores)
        avg_score = np.mean(scores)

        # High max with good spread = high confidence
        score_spread = np.std(scores) if len(scores) > 1 else 0

        confidence = (
            0.5 * max_score +
            0.3 * avg_score +
            0.2 * min(1.0, score_spread * 2)
        )

        return min(1.0, confidence)

    def _calculate_coverage(
        self,
        candidates: List[CompletionCandidate],
        query: np.ndarray
    ) -> float:
        """Calculate how well results cover the query semantics"""
        if not candidates:
            return 0.0

        # Simple coverage: average of top scores
        top_scores = [c.embedding_similarity for c in candidates[:5]]
        return float(np.mean(top_scores)) if top_scores else 0.0

    async def find_similar_patterns(
        self,
        pattern_memories: List[UUID],
        agent_id: Optional[str] = None,
        top_k: int = 10
    ) -> List[Tuple[List[UUID], float]]:
        """
        Find similar patterns in the memory fabric.

        Given a set of memories that form a pattern, finds other
        memory sets that form similar patterns.
        """
        # Get embeddings for pattern
        pattern_embeddings = []
        for mem_id in pattern_memories:
            node = self.graph.nodes.get(mem_id)
            if node and node.canonical_embedding is not None:
                pattern_embeddings.append(node.canonical_embedding)

        if not pattern_embeddings:
            return []

        # Average pattern embedding
        avg_embedding = np.mean(pattern_embeddings, axis=0)

        # Search for similar
        result = await self.complete(
            query_embedding=avg_embedding,
            partial_cues=pattern_memories[:3],
            constraints=ActivationConstraints(agent_id=agent_id) if agent_id else None,
            top_k=top_k * 3
        )

        # Cluster results into patterns
        # (simplified - would use actual clustering in production)
        patterns = []
        used = set(pattern_memories)

        for cand in result.candidates:
            if cand.memory_id not in used:
                # Find neighbors that are also in results
                node = self.graph.nodes.get(cand.memory_id)
                if node:
                    pattern = [cand.memory_id]
                    for link in node.links:
                        if link.target_id in {c.memory_id for c in result.candidates}:
                            if link.target_id not in used:
                                pattern.append(link.target_id)

                    if len(pattern) >= 2:
                        patterns.append((pattern, cand.score))
                        used.update(pattern)

        return patterns[:top_k]

    def set_reranker(self, reranker: Callable):
        """Set custom reranker function"""
        self.reranker = reranker

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            "completions": self.stats["completions"],
            "indexes": {
                tier.name: len(idx.memory_ids)
                for tier, idx in self.ann_indexes.items()
            },
            "weights": {
                "ann": self.ann_weight,
                "graph": self.graph_weight,
                "rerank": self.rerank_weight,
            },
            "has_reranker": self.reranker is not None,
        }
