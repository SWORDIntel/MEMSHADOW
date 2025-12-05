#!/usr/bin/env python3
"""
Memory Consolidator for DSMIL Brain

Manages cross-tier memory consolidation:
- Promotes important items from working → episodic → semantic
- Applies forgetting curves and importance scoring
- Handles distributed synchronization
- Extracts patterns and knowledge from experiences
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
import hashlib

from .working_memory import WorkingMemory, WorkingMemoryItem, MemoryPressure
from .episodic_memory import EpisodicMemory, Episode, Event, EventType
from .semantic_memory import SemanticMemory, Concept, ConceptType, RelationType

logger = logging.getLogger(__name__)


class ConsolidationPolicy(Enum):
    """Policies for memory consolidation"""
    AGGRESSIVE = auto()   # Promote aggressively, keep less in lower tiers
    BALANCED = auto()     # Balanced promotion and retention
    CONSERVATIVE = auto() # Keep more in lower tiers, promote cautiously


@dataclass
class PromotionCriteria:
    """Criteria for promoting memory items between tiers"""
    # Working → Episodic
    working_min_access_count: int = 3
    working_min_score: float = 0.5

    # Episodic → Semantic
    episodic_min_importance: float = 0.6
    episodic_min_access_count: int = 2
    episodic_min_events: int = 3

    # Semantic reinforcement
    semantic_min_evidence: int = 2
    semantic_min_confidence: float = 0.5


@dataclass
class ConsolidationResult:
    """Results from a consolidation run"""
    timestamp: datetime
    working_to_episodic: int = 0
    episodic_to_semantic: int = 0
    patterns_extracted: int = 0
    items_forgotten: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


class PatternExtractor:
    """
    Extracts patterns and knowledge from episodic memories

    Looks for:
    - Recurring themes across episodes
    - Causal patterns
    - Entity relationships
    - Behavioral patterns
    """

    def __init__(self, semantic_memory: SemanticMemory):
        self.semantic = semantic_memory

    def extract_from_episode(self, episode: Episode) -> List[Dict]:
        """
        Extract knowledge patterns from an episode

        Args:
            episode: Episode to analyze

        Returns:
            List of extracted patterns
        """
        patterns = []

        # Extract entity relationships from events
        entities = set()
        actions = []

        for event in episode.events:
            # Extract entities from content
            if isinstance(event.content, dict):
                for key, value in event.content.items():
                    if isinstance(value, str) and len(value) < 100:
                        entities.add((key, value))
            elif isinstance(event.content, str):
                # Simple entity extraction (would use NER in production)
                words = event.content.split()
                for word in words:
                    if word[0].isupper() and len(word) > 2:
                        entities.add(("entity", word))

            # Track actions
            if event.event_type == EventType.ACTION:
                actions.append(event)

        # Create entity concepts
        for entity_type, entity_name in entities:
            patterns.append({
                "type": "entity",
                "name": entity_name,
                "entity_type": entity_type,
                "source_episode": episode.episode_id,
            })

        # Extract causal chains
        for event in episode.events:
            if event.caused_by:
                causing_event = next(
                    (e for e in episode.events if e.event_id == event.caused_by),
                    None
                )
                if causing_event:
                    patterns.append({
                        "type": "causal",
                        "cause": str(causing_event.content)[:100],
                        "effect": str(event.content)[:100],
                        "source_episode": episode.episode_id,
                    })

        # Extract insights from episode
        for insight in episode.key_insights:
            patterns.append({
                "type": "insight",
                "content": insight,
                "source_episode": episode.episode_id,
            })

        return patterns

    def apply_patterns_to_semantic(self, patterns: List[Dict]) -> int:
        """
        Apply extracted patterns to semantic memory

        Returns:
            Number of facts added
        """
        added = 0

        for pattern in patterns:
            try:
                if pattern["type"] == "entity":
                    self.semantic.add_concept(
                        name=pattern["name"],
                        concept_type=ConceptType.ENTITY,
                        source_episode=pattern.get("source_episode"),
                    )
                    added += 1

                elif pattern["type"] == "causal":
                    self.semantic.add_fact(
                        subject=pattern["cause"][:50],
                        predicate="CAUSES",
                        obj=pattern["effect"][:50],
                        source_episode=pattern.get("source_episode"),
                    )
                    added += 1

                elif pattern["type"] == "insight":
                    self.semantic.add_concept(
                        name=pattern["content"][:100],
                        concept_type=ConceptType.PATTERN,
                        source_episode=pattern.get("source_episode"),
                        importance=0.7,
                    )
                    added += 1

            except Exception as e:
                logger.warning(f"Failed to apply pattern: {e}")

        return added


class MemoryConsolidator:
    """
    Orchestrates memory consolidation across all tiers

    Features:
    - Automatic promotion based on criteria
    - Pattern extraction from episodes
    - Forgetting curve application
    - Distributed sync coordination

    Usage:
        consolidator = MemoryConsolidator(
            working_memory=wm,
            episodic_memory=em,
            semantic_memory=sm
        )

        # Run consolidation
        result = consolidator.consolidate()

        # Start background consolidation
        consolidator.start_background_consolidation(interval=300)
    """

    def __init__(self, working_memory: WorkingMemory,
                 episodic_memory: EpisodicMemory,
                 semantic_memory: SemanticMemory,
                 policy: ConsolidationPolicy = ConsolidationPolicy.BALANCED,
                 criteria: Optional[PromotionCriteria] = None):
        """
        Initialize consolidator

        Args:
            working_memory: L1 working memory
            episodic_memory: L2 episodic memory
            semantic_memory: L3 semantic memory
            policy: Consolidation policy
            criteria: Promotion criteria
        """
        self.working = working_memory
        self.episodic = episodic_memory
        self.semantic = semantic_memory

        self.policy = policy
        self.criteria = criteria or self._get_criteria_for_policy(policy)

        # Pattern extractor
        self.pattern_extractor = PatternExtractor(semantic_memory)

        # Background thread
        self._consolidation_thread: Optional[threading.Thread] = None
        self._running = False

        # History
        self._consolidation_history: List[ConsolidationResult] = []

        # Callbacks
        self.on_consolidation_complete: Optional[Callable[[ConsolidationResult], None]] = None
        self.on_pattern_extracted: Optional[Callable[[Dict], None]] = None

        logger.info(f"MemoryConsolidator initialized with {policy.name} policy")

    def _get_criteria_for_policy(self, policy: ConsolidationPolicy) -> PromotionCriteria:
        """Get criteria based on policy"""
        if policy == ConsolidationPolicy.AGGRESSIVE:
            return PromotionCriteria(
                working_min_access_count=2,
                working_min_score=0.3,
                episodic_min_importance=0.4,
                episodic_min_access_count=1,
                episodic_min_events=2,
            )
        elif policy == ConsolidationPolicy.CONSERVATIVE:
            return PromotionCriteria(
                working_min_access_count=5,
                working_min_score=0.7,
                episodic_min_importance=0.8,
                episodic_min_access_count=3,
                episodic_min_events=5,
            )
        else:  # BALANCED
            return PromotionCriteria()

    def consolidate(self) -> ConsolidationResult:
        """
        Run a full consolidation cycle

        Returns:
            ConsolidationResult with statistics
        """
        start_time = time.time()
        result = ConsolidationResult(timestamp=datetime.now(timezone.utc))

        try:
            # Phase 1: Working → Episodic
            result.working_to_episodic = self._consolidate_working_to_episodic()

            # Phase 2: Episodic → Semantic
            result.episodic_to_semantic, result.patterns_extracted = \
                self._consolidate_episodic_to_semantic()

            # Phase 3: Apply forgetting
            result.items_forgotten = self._apply_forgetting()

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Consolidation error: {e}")

        result.duration_seconds = time.time() - start_time
        self._consolidation_history.append(result)

        # Trigger callback
        if self.on_consolidation_complete:
            self.on_consolidation_complete(result)

        logger.info(f"Consolidation complete: W→E={result.working_to_episodic}, "
                   f"E→S={result.episodic_to_semantic}, patterns={result.patterns_extracted}, "
                   f"forgotten={result.items_forgotten}")

        return result

    def _consolidate_working_to_episodic(self) -> int:
        """Promote items from working memory to episodic memory"""
        promoted = 0

        # Get candidates from working memory
        candidates = self.working.get_items_for_consolidation(
            min_access_count=self.criteria.working_min_access_count,
            min_score=self.criteria.working_min_score,
        )

        for item in candidates:
            try:
                # Record as event in current or new episode
                event = self.episodic.record_event(
                    event_type=EventType.OBSERVATION,
                    content=item.content,
                    context=item.metadata,
                    importance=item.attention.compute_score(),
                    metadata={
                        "source": "working_memory",
                        "item_id": item.item_id,
                        "access_count": item.access_count,
                    }
                )

                promoted += 1

                # Optionally remove from working memory if pressure is high
                if self.working.get_pressure() in (MemoryPressure.HIGH, MemoryPressure.CRITICAL):
                    self.working.delete(item.item_id)

            except Exception as e:
                logger.warning(f"Failed to promote item {item.item_id}: {e}")

        return promoted

    def _consolidate_episodic_to_semantic(self) -> tuple[int, int]:
        """
        Promote episodes to semantic memory

        Returns:
            (episodes_processed, patterns_extracted)
        """
        episodes_processed = 0
        patterns_extracted = 0

        # Get consolidation candidates
        candidates = self.episodic.get_for_consolidation(
            min_importance=self.criteria.episodic_min_importance,
            min_access_count=self.criteria.episodic_min_access_count,
        )

        for episode in candidates:
            if len(episode.events) < self.criteria.episodic_min_events:
                continue

            try:
                # Extract patterns
                patterns = self.pattern_extractor.extract_from_episode(episode)

                # Apply to semantic memory
                added = self.pattern_extractor.apply_patterns_to_semantic(patterns)

                patterns_extracted += len(patterns)
                episodes_processed += 1

                # Trigger callback
                if self.on_pattern_extracted:
                    for pattern in patterns:
                        self.on_pattern_extracted(pattern)

            except Exception as e:
                logger.warning(f"Failed to consolidate episode {episode.episode_id}: {e}")

        return episodes_processed, patterns_extracted

    def _apply_forgetting(self) -> int:
        """Apply forgetting curves across all memory tiers"""
        forgotten = 0

        # Apply to semantic memory
        initial_count = len(self.semantic.graph._concepts)
        self.semantic.apply_forgetting()
        forgotten += initial_count - len(self.semantic.graph._concepts)

        return forgotten

    def start_background_consolidation(self, interval: float = 300.0):
        """
        Start background consolidation thread

        Args:
            interval: Seconds between consolidation runs
        """
        if self._running:
            return

        self._running = True

        def consolidation_loop():
            while self._running:
                try:
                    # Check if consolidation is needed
                    pressure = self.working.get_pressure()

                    if pressure in (MemoryPressure.HIGH, MemoryPressure.CRITICAL):
                        # More frequent consolidation under pressure
                        self.consolidate()
                        time.sleep(interval / 4)
                    else:
                        self.consolidate()
                        time.sleep(interval)

                except Exception as e:
                    logger.error(f"Background consolidation error: {e}")
                    time.sleep(interval)

        self._consolidation_thread = threading.Thread(
            target=consolidation_loop, daemon=True
        )
        self._consolidation_thread.start()
        logger.info(f"Background consolidation started (interval={interval}s)")

    def stop_background_consolidation(self):
        """Stop background consolidation"""
        self._running = False
        if self._consolidation_thread:
            self._consolidation_thread.join(timeout=10.0)
        logger.info("Background consolidation stopped")

    def force_consolidation(self):
        """Force immediate consolidation regardless of criteria"""
        original_criteria = self.criteria
        self.criteria = PromotionCriteria(
            working_min_access_count=1,
            working_min_score=0.1,
            episodic_min_importance=0.1,
            episodic_min_access_count=0,
            episodic_min_events=1,
        )

        result = self.consolidate()

        self.criteria = original_criteria
        return result

    def get_consolidation_status(self) -> Dict:
        """Get current consolidation status"""
        return {
            "policy": self.policy.name,
            "criteria": {
                "working_min_access_count": self.criteria.working_min_access_count,
                "working_min_score": self.criteria.working_min_score,
                "episodic_min_importance": self.criteria.episodic_min_importance,
                "episodic_min_access_count": self.criteria.episodic_min_access_count,
            },
            "is_running": self._running,
            "history_count": len(self._consolidation_history),
            "last_run": self._consolidation_history[-1].timestamp.isoformat()
                       if self._consolidation_history else None,
        }

    def get_memory_overview(self) -> Dict:
        """Get overview of all memory tiers"""
        return {
            "working_memory": self.working.get_stats(),
            "episodic_memory": self.episodic.get_stats(),
            "semantic_memory": self.semantic.get_stats(),
            "consolidation": self.get_consolidation_status(),
        }

    def export_all_for_sync(self) -> Dict:
        """Export all memory tiers for synchronization"""
        return {
            "working": self.working.export_for_sync(),
            "episodic": self.episodic.export_for_sync(),
            "semantic": self.semantic.export_for_sync(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


if __name__ == "__main__":
    print("Memory Consolidator Self-Test")
    print("=" * 50)

    # Initialize all memory tiers
    wm = WorkingMemory()
    em = EpisodicMemory()
    sm = SemanticMemory()

    # Initialize consolidator
    consolidator = MemoryConsolidator(wm, em, sm)

    print(f"\n[1] Populate Working Memory")
    for i in range(20):
        item_id = wm.store(
            f"fact-{i}",
            {"data": f"Important fact #{i}", "value": i * 10},
            metadata={"category": "test"}
        )
        # Access some items multiple times
        if i < 5:
            for _ in range(5):
                wm.retrieve(item_id)
    print(f"    Stored 20 items, accessed 5 frequently")

    print(f"\n[2] Populate Episodic Memory")
    ep = em.start_episode("Test Episode", tags={"test"})
    for i in range(10):
        em.record_event(
            EventType.OBSERVATION,
            f"Event {i} content",
            importance=0.5 + (i * 0.05)
        )
    em.close_episode(
        summary="Test episode with multiple events",
        insights=["Key pattern detected", "Important correlation found"]
    )
    # Access episode
    em.recall_episode(ep.episode_id)
    em.recall_episode(ep.episode_id)
    print(f"    Created episode with 10 events")

    print(f"\n[3] Run Consolidation")
    result = consolidator.consolidate()
    print(f"    Working → Episodic: {result.working_to_episodic}")
    print(f"    Episodic → Semantic: {result.episodic_to_semantic}")
    print(f"    Patterns extracted: {result.patterns_extracted}")
    print(f"    Items forgotten: {result.items_forgotten}")
    print(f"    Duration: {result.duration_seconds:.3f}s")

    print(f"\n[4] Memory Overview")
    overview = consolidator.get_memory_overview()
    print(f"    Working: {overview['working_memory']['item_count']} items")
    print(f"    Episodic: {overview['episodic_memory']['total_episodes']} episodes")
    print(f"    Semantic: {overview['semantic_memory']['concept_count']} concepts")

    print(f"\n[5] Consolidation Status")
    status = consolidator.get_consolidation_status()
    for key, value in status.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 50)
    print("Memory Consolidator test complete")

