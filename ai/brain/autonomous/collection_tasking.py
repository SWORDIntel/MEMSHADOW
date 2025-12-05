#!/usr/bin/env python3
"""
Autonomous Collection Tasking for DSMIL Brain

Self-directed intelligence gap analysis and collection:
- Identify what we don't know
- Priority score unknowns
- Generate collection tasks
- Learn from outcomes
"""

import hashlib
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set
from datetime import datetime, timezone
from enum import Enum, auto
from collections import defaultdict

logger = logging.getLogger(__name__)


class GapPriority(Enum):
    """Intelligence gap priority"""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1


class CollectionMethod(Enum):
    """Collection methods"""
    PASSIVE = auto()    # Monitor existing sources
    ACTIVE = auto()     # Actively query sources
    TARGETED = auto()   # Specific target collection
    SWEEP = auto()      # Broad collection sweep


@dataclass
class IntelligenceGap:
    """An identified intelligence gap"""
    gap_id: str
    topic: str
    description: str
    priority: GapPriority = GapPriority.MEDIUM

    # Context
    related_entities: Set[str] = field(default_factory=set)
    related_gaps: Set[str] = field(default_factory=set)

    # Assessment
    impact_if_unfilled: float = 0.5  # 0-1
    fillability: float = 0.5  # How likely we can fill it

    # Status
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_assessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_filled: bool = False


@dataclass
class CollectionTask:
    """A collection task to fill a gap"""
    task_id: str
    gap_id: str
    method: CollectionMethod

    # Instructions
    target: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Resource requirements
    estimated_time: float = 0.0  # hours
    risk_level: float = 0.0  # 0-1

    # Status
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_complete: bool = False


@dataclass
class CollectionOutcome:
    """Outcome of a collection task"""
    outcome_id: str
    task_id: str

    # Results
    success: bool = False
    data_collected: bool = False
    gap_filled: bool = False

    # Learning
    effectiveness: float = 0.0  # How well did method work
    notes: str = ""


class AutonomousCollector:
    """
    Autonomous Collection Tasking System

    Analyzes intelligence gaps and generates collection tasks.

    Usage:
        collector = AutonomousCollector()

        # Analyze gaps
        gaps = collector.analyze_gaps(knowledge_graph)

        # Generate tasks
        tasks = collector.generate_collection_tasks(gaps)

        # After execution, report outcome
        collector.report_outcome(outcome)

        # System learns and improves
        collector.evolve_strategy(outcomes)
    """

    def __init__(self):
        self._gaps: Dict[str, IntelligenceGap] = {}
        self._tasks: Dict[str, CollectionTask] = {}
        self._outcomes: Dict[str, CollectionOutcome] = {}

        # Learning: track method effectiveness per topic
        self._method_scores: Dict[str, Dict[CollectionMethod, float]] = defaultdict(
            lambda: {m: 0.5 for m in CollectionMethod}
        )

        self._lock = threading.RLock()

        logger.info("AutonomousCollector initialized")

    def analyze_gaps(self, knowledge_base: Optional[Dict] = None) -> List[IntelligenceGap]:
        """
        Analyze knowledge base to identify gaps

        What do we NOT know that we should?
        """
        with self._lock:
            gaps = []

            # Would integrate with actual knowledge graph
            # For now, demonstrate gap identification logic

            # Check for entities with low confidence
            if knowledge_base:
                for entity_id, entity in knowledge_base.get("entities", {}).items():
                    confidence = entity.get("confidence", 0)
                    if confidence < 0.5:
                        gap = IntelligenceGap(
                            gap_id=hashlib.sha256(f"conf:{entity_id}".encode()).hexdigest()[:16],
                            topic=f"entity:{entity_id}",
                            description=f"Low confidence on entity {entity_id}",
                            priority=GapPriority.MEDIUM,
                            related_entities={entity_id},
                            impact_if_unfilled=0.7,
                            fillability=0.6,
                        )
                        gaps.append(gap)
                        self._gaps[gap.gap_id] = gap

                # Check for missing relationships
                for rel in knowledge_base.get("expected_relationships", []):
                    if rel not in knowledge_base.get("relationships", {}):
                        gap = IntelligenceGap(
                            gap_id=hashlib.sha256(f"rel:{rel}".encode()).hexdigest()[:16],
                            topic=f"relationship:{rel}",
                            description=f"Missing expected relationship: {rel}",
                            priority=GapPriority.HIGH,
                            impact_if_unfilled=0.8,
                            fillability=0.5,
                        )
                        gaps.append(gap)
                        self._gaps[gap.gap_id] = gap

            return gaps

    def identify_gap(self, topic: str, description: str,
                    priority: GapPriority = GapPriority.MEDIUM,
                    entities: Optional[Set[str]] = None) -> IntelligenceGap:
        """Manually identify a gap"""
        with self._lock:
            gap = IntelligenceGap(
                gap_id=hashlib.sha256(f"{topic}:{datetime.now().isoformat()}".encode()).hexdigest()[:16],
                topic=topic,
                description=description,
                priority=priority,
                related_entities=entities or set(),
            )
            self._gaps[gap.gap_id] = gap
            return gap

    def generate_collection_tasks(self, gaps: List[IntelligenceGap]) -> List[CollectionTask]:
        """
        Generate collection tasks for gaps

        Optimizes resource allocation across gaps.
        """
        with self._lock:
            tasks = []

            # Sort by priority and impact
            sorted_gaps = sorted(
                gaps,
                key=lambda g: (g.priority.value, g.impact_if_unfilled),
                reverse=True
            )

            for gap in sorted_gaps:
                if gap.is_filled:
                    continue

                # Select best method based on learned effectiveness
                topic_category = gap.topic.split(":")[0]
                method_scores = self._method_scores[topic_category]
                best_method = max(method_scores.items(), key=lambda x: x[1])[0]

                task = CollectionTask(
                    task_id=hashlib.sha256(f"task:{gap.gap_id}:{datetime.now().isoformat()}".encode()).hexdigest()[:16],
                    gap_id=gap.gap_id,
                    method=best_method,
                    target=gap.topic,
                    parameters={
                        "entities": list(gap.related_entities),
                        "description": gap.description,
                    },
                    estimated_time=self._estimate_time(best_method),
                    risk_level=self._assess_risk(best_method),
                )

                tasks.append(task)
                self._tasks[task.task_id] = task

            return tasks

    def _estimate_time(self, method: CollectionMethod) -> float:
        """Estimate collection time in hours"""
        estimates = {
            CollectionMethod.PASSIVE: 24.0,
            CollectionMethod.ACTIVE: 1.0,
            CollectionMethod.TARGETED: 4.0,
            CollectionMethod.SWEEP: 8.0,
        }
        return estimates.get(method, 2.0)

    def _assess_risk(self, method: CollectionMethod) -> float:
        """Assess risk level of collection method"""
        risks = {
            CollectionMethod.PASSIVE: 0.1,
            CollectionMethod.ACTIVE: 0.3,
            CollectionMethod.TARGETED: 0.6,
            CollectionMethod.SWEEP: 0.4,
        }
        return risks.get(method, 0.5)

    def report_outcome(self, outcome: CollectionOutcome):
        """Report collection outcome for learning"""
        with self._lock:
            self._outcomes[outcome.outcome_id] = outcome

            # Update gap status
            task = self._tasks.get(outcome.task_id)
            if task:
                task.is_complete = True

                gap = self._gaps.get(task.gap_id)
                if gap and outcome.gap_filled:
                    gap.is_filled = True

    def evolve_strategy(self, outcomes: Optional[List[CollectionOutcome]] = None):
        """
        Learn from outcomes to improve collection strategy
        """
        with self._lock:
            outcomes_to_process = outcomes or list(self._outcomes.values())

            for outcome in outcomes_to_process:
                task = self._tasks.get(outcome.task_id)
                if not task:
                    continue

                gap = self._gaps.get(task.gap_id)
                if not gap:
                    continue

                # Update method effectiveness for this topic
                topic_category = gap.topic.split(":")[0]

                # Weighted update
                current_score = self._method_scores[topic_category][task.method]
                new_score = current_score * 0.9 + outcome.effectiveness * 0.1
                self._method_scores[topic_category][task.method] = new_score

    def get_unfilled_gaps(self) -> List[IntelligenceGap]:
        """Get all unfilled gaps"""
        with self._lock:
            return [g for g in self._gaps.values() if not g.is_filled]

    def get_stats(self) -> Dict:
        """Get collector statistics"""
        with self._lock:
            return {
                "total_gaps": len(self._gaps),
                "unfilled_gaps": len([g for g in self._gaps.values() if not g.is_filled]),
                "total_tasks": len(self._tasks),
                "completed_tasks": len([t for t in self._tasks.values() if t.is_complete]),
                "outcomes_recorded": len(self._outcomes),
            }


if __name__ == "__main__":
    print("Autonomous Collection Self-Test")
    print("=" * 50)

    collector = AutonomousCollector()

    print("\n[1] Analyze Gaps")
    knowledge = {
        "entities": {
            "APT29": {"confidence": 0.3},
            "target_network": {"confidence": 0.8},
        },
        "expected_relationships": ["APT29->target_network"],
        "relationships": {},
    }
    gaps = collector.analyze_gaps(knowledge)
    print(f"    Found {len(gaps)} gaps")
    for gap in gaps:
        print(f"      - {gap.topic}: {gap.description}")

    print("\n[2] Manual Gap Identification")
    manual_gap = collector.identify_gap(
        topic="actor:unknown_threat",
        description="Unknown threat actor in network",
        priority=GapPriority.CRITICAL,
        entities={"unknown_threat"},
    )
    print(f"    Added gap: {manual_gap.gap_id}")

    print("\n[3] Generate Collection Tasks")
    all_gaps = collector.get_unfilled_gaps()
    tasks = collector.generate_collection_tasks(all_gaps)
    print(f"    Generated {len(tasks)} tasks")
    for task in tasks:
        print(f"      - {task.task_id}: {task.method.name} on {task.target}")

    print("\n[4] Report Outcome")
    if tasks:
        outcome = CollectionOutcome(
            outcome_id=hashlib.sha256(b"test_outcome").hexdigest()[:16],
            task_id=tasks[0].task_id,
            success=True,
            data_collected=True,
            gap_filled=True,
            effectiveness=0.8,
        )
        collector.report_outcome(outcome)
        print(f"    Reported outcome: success={outcome.success}")

    print("\n[5] Evolve Strategy")
    collector.evolve_strategy()
    print("    Strategy evolved based on outcomes")

    print("\n[6] Statistics")
    stats = collector.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Autonomous Collection test complete")

