#!/usr/bin/env python3
"""
Temporal Knowledge Graph for DSMIL Brain

Facts with time validity:
- Validity windows [T_start, T_end]
- Historical state reconstruction
- Temporal queries: "What was true at time T?"
"""

import hashlib
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set
from datetime import datetime, timezone
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TemporalFact:
    """A fact with temporal validity"""
    fact_id: str
    subject: str
    predicate: str
    object: str

    # Validity window
    valid_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_to: Optional[datetime] = None  # None = still valid

    # Confidence
    confidence: float = 1.0

    # Source
    source: str = ""


@dataclass
class TemporalQuery:
    """Query parameters for temporal search"""
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None

    # Time point or range
    at_time: Optional[datetime] = None
    from_time: Optional[datetime] = None
    to_time: Optional[datetime] = None


class TemporalKnowledgeGraph:
    """
    Temporal Knowledge Graph

    Stores facts with time validity windows.

    Usage:
        graph = TemporalKnowledgeGraph()

        # Add fact
        graph.add_fact("Alice", "worksAt", "Acme", valid_from=start_date)

        # Update (old fact ends, new begins)
        graph.update_fact("Alice", "worksAt", "NewCorp")

        # Query at time
        facts = graph.query_at_time(datetime(...))

        # Reconstruct historical state
        state = graph.reconstruct_state(datetime(...))
    """

    def __init__(self):
        self._facts: Dict[str, TemporalFact] = {}
        self._by_subject: Dict[str, Set[str]] = defaultdict(set)
        self._by_predicate: Dict[str, Set[str]] = defaultdict(set)
        self._by_object: Dict[str, Set[str]] = defaultdict(set)

        self._lock = threading.RLock()

        logger.info("TemporalKnowledgeGraph initialized")

    def add_fact(self, subject: str, predicate: str, object: str,
                valid_from: Optional[datetime] = None,
                valid_to: Optional[datetime] = None,
                confidence: float = 1.0,
                source: str = "") -> TemporalFact:
        """Add a temporal fact"""
        with self._lock:
            fact_id = hashlib.sha256(
                f"{subject}:{predicate}:{object}:{valid_from}".encode()
            ).hexdigest()[:16]

            fact = TemporalFact(
                fact_id=fact_id,
                subject=subject,
                predicate=predicate,
                object=object,
                valid_from=valid_from or datetime.now(timezone.utc),
                valid_to=valid_to,
                confidence=confidence,
                source=source,
            )

            self._facts[fact_id] = fact
            self._by_subject[subject].add(fact_id)
            self._by_predicate[predicate].add(fact_id)
            self._by_object[object].add(fact_id)

            return fact

    def update_fact(self, subject: str, predicate: str, new_object: str,
                   end_time: Optional[datetime] = None) -> TemporalFact:
        """
        Update a fact (ends old fact, creates new one)
        """
        with self._lock:
            now = end_time or datetime.now(timezone.utc)

            # Find and end current fact
            for fact_id in self._by_subject.get(subject, set()):
                fact = self._facts.get(fact_id)
                if fact and fact.predicate == predicate and fact.valid_to is None:
                    fact.valid_to = now

            # Create new fact
            return self.add_fact(subject, predicate, new_object, valid_from=now)

    def query(self, query: TemporalQuery) -> List[TemporalFact]:
        """Query facts with temporal constraints"""
        with self._lock:
            # Get candidate fact IDs
            candidates = set(self._facts.keys())

            if query.subject:
                candidates &= self._by_subject.get(query.subject, set())

            if query.predicate:
                candidates &= self._by_predicate.get(query.predicate, set())

            if query.object:
                candidates &= self._by_object.get(query.object, set())

            # Filter by time
            results = []
            for fact_id in candidates:
                fact = self._facts[fact_id]

                if query.at_time:
                    if not self._is_valid_at(fact, query.at_time):
                        continue

                if query.from_time:
                    if fact.valid_to and fact.valid_to < query.from_time:
                        continue

                if query.to_time:
                    if fact.valid_from > query.to_time:
                        continue

                results.append(fact)

            return results

    def _is_valid_at(self, fact: TemporalFact, timestamp: datetime) -> bool:
        """Check if fact was valid at timestamp"""
        if fact.valid_from > timestamp:
            return False
        if fact.valid_to and fact.valid_to <= timestamp:
            return False
        return True

    def query_at_time(self, timestamp: datetime,
                     subject: Optional[str] = None,
                     predicate: Optional[str] = None) -> List[TemporalFact]:
        """Query all facts valid at specific time"""
        query = TemporalQuery(
            subject=subject,
            predicate=predicate,
            at_time=timestamp,
        )
        return self.query(query)

    def reconstruct_state(self, timestamp: datetime) -> Dict[str, Dict[str, str]]:
        """
        Reconstruct knowledge graph state at specific time

        Returns dict of {subject: {predicate: object}}
        """
        with self._lock:
            state = defaultdict(dict)

            for fact in self._facts.values():
                if self._is_valid_at(fact, timestamp):
                    state[fact.subject][fact.predicate] = fact.object

            return dict(state)

    def get_history(self, subject: str, predicate: str) -> List[TemporalFact]:
        """Get history of a specific fact"""
        with self._lock:
            facts = []

            for fact_id in self._by_subject.get(subject, set()):
                fact = self._facts[fact_id]
                if fact.predicate == predicate:
                    facts.append(fact)

            return sorted(facts, key=lambda f: f.valid_from)

    def get_changes_in_range(self, from_time: datetime,
                            to_time: datetime) -> List[TemporalFact]:
        """Get all facts that changed in time range"""
        with self._lock:
            changes = []

            for fact in self._facts.values():
                # Started in range
                if from_time <= fact.valid_from <= to_time:
                    changes.append(fact)
                # Ended in range
                elif fact.valid_to and from_time <= fact.valid_to <= to_time:
                    changes.append(fact)

            return sorted(changes, key=lambda f: f.valid_from)

    def get_stats(self) -> Dict:
        """Get graph statistics"""
        with self._lock:
            current_facts = sum(1 for f in self._facts.values() if f.valid_to is None)
            historical_facts = len(self._facts) - current_facts

            return {
                "total_facts": len(self._facts),
                "current_facts": current_facts,
                "historical_facts": historical_facts,
                "subjects": len(self._by_subject),
                "predicates": len(self._by_predicate),
            }


if __name__ == "__main__":
    print("Temporal Knowledge Graph Self-Test")
    print("=" * 50)

    from datetime import timedelta

    graph = TemporalKnowledgeGraph()

    print("\n[1] Add Facts")
    base_time = datetime.now(timezone.utc) - timedelta(days=365)

    # Alice's job history
    graph.add_fact("Alice", "worksAt", "StartupA", valid_from=base_time)

    # After 6 months, she changes jobs
    change_time = base_time + timedelta(days=180)
    graph.update_fact("Alice", "worksAt", "BigCorp", end_time=change_time)

    # Bob works at BigCorp
    graph.add_fact("Bob", "worksAt", "BigCorp", valid_from=base_time)

    print("    Added job history for Alice and Bob")

    print("\n[2] Query at Historical Time")
    query_time = base_time + timedelta(days=90)
    facts = graph.query_at_time(query_time)
    print(f"    Facts at {query_time.date()}:")
    for f in facts:
        print(f"      {f.subject} {f.predicate} {f.object}")

    print("\n[3] Query Current State")
    current_facts = graph.query_at_time(datetime.now(timezone.utc))
    print("    Current facts:")
    for f in current_facts:
        print(f"      {f.subject} {f.predicate} {f.object}")

    print("\n[4] Get History")
    history = graph.get_history("Alice", "worksAt")
    print(f"    Alice's work history ({len(history)} entries):")
    for f in history:
        end = f.valid_to.date() if f.valid_to else "present"
        print(f"      {f.object}: {f.valid_from.date()} to {end}")

    print("\n[5] Reconstruct State")
    state = graph.reconstruct_state(query_time)
    print(f"    State at {query_time.date()}:")
    for subj, preds in state.items():
        for pred, obj in preds.items():
            print(f"      {subj} {pred} {obj}")

    print("\n[6] Statistics")
    stats = graph.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Temporal Knowledge Graph test complete")

