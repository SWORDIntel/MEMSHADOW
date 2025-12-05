#!/usr/bin/env python3
"""
Episodic Memory (L2) for DSMIL Brain

Session-based experience storage:
- Temporal indexing of events
- Causal relationship tracking
- Auto-consolidation from working memory
- Experience replay for learning
- Episode-based organization
"""

import time
import hashlib
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Tuple, Set
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from collections import defaultdict
import json
import heapq

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events that can be recorded"""
    QUERY = auto()          # User/system query
    RESPONSE = auto()       # AI response
    OBSERVATION = auto()    # Observed data point
    ACTION = auto()         # Action taken
    DECISION = auto()       # Decision made
    ERROR = auto()          # Error occurred
    CORRELATION = auto()    # Correlation discovered
    INSIGHT = auto()        # Generated insight
    ALERT = auto()          # Alert/warning
    STATE_CHANGE = auto()   # System state change


class CausalRelation(Enum):
    """Types of causal relationships"""
    CAUSES = auto()         # A causes B
    ENABLES = auto()        # A enables B
    PREVENTS = auto()       # A prevents B
    CORRELATES = auto()     # A correlates with B
    FOLLOWS = auto()        # B follows A temporally
    CONTRADICTS = auto()    # A contradicts B


@dataclass
class Event:
    """
    Single event in episodic memory
    """
    event_id: str
    event_type: EventType
    content: Any
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Context at time of event
    context: Dict[str, Any] = field(default_factory=dict)

    # Relationships
    caused_by: Optional[str] = None  # ID of causing event
    causes: List[str] = field(default_factory=list)  # IDs of caused events
    related_events: Set[str] = field(default_factory=set)

    # Importance
    importance: float = 0.5  # 0-1 scale
    emotional_valence: float = 0.0  # -1 to 1 (negative to positive)

    # Source
    source_node: Optional[str] = None
    source_episode: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "importance": self.importance,
            "metadata": self.metadata,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Event":
        return cls(
            event_id=data["event_id"],
            event_type=EventType[data["event_type"]],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            context=data.get("context", {}),
            importance=data.get("importance", 0.5),
            metadata=data.get("metadata", {}),
            tags=set(data.get("tags", [])),
        )


@dataclass
class CausalLink:
    """Link between events with causal information"""
    source_event_id: str
    target_event_id: str
    relation: CausalRelation
    strength: float = 0.5  # 0-1 confidence in relationship
    evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Episode:
    """
    Collection of related events forming an episode

    An episode represents a coherent sequence of events,
    like a task execution, conversation, or investigation.
    """
    episode_id: str
    name: str
    events: List[Event] = field(default_factory=list)

    # Timing
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    is_active: bool = True

    # Summary
    summary: Optional[str] = None
    key_insights: List[str] = field(default_factory=list)

    # Importance
    importance: float = 0.5
    access_count: int = 0

    # Relationships
    parent_episode: Optional[str] = None
    child_episodes: List[str] = field(default_factory=list)
    related_episodes: Set[str] = field(default_factory=set)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    def add_event(self, event: Event):
        """Add event to episode"""
        event.source_episode = self.episode_id
        self.events.append(event)

        # Update importance based on event importance
        self.importance = max(self.importance, event.importance)

    def close(self, summary: Optional[str] = None):
        """Close the episode"""
        self.is_active = False
        self.end_time = datetime.now(timezone.utc)
        if summary:
            self.summary = summary

    def get_duration(self) -> Optional[timedelta]:
        """Get episode duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return datetime.now(timezone.utc) - self.start_time

    def to_dict(self) -> Dict:
        return {
            "episode_id": self.episode_id,
            "name": self.name,
            "events": [e.to_dict() for e in self.events],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "is_active": self.is_active,
            "summary": self.summary,
            "importance": self.importance,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }


class ExperienceReplay:
    """
    Experience replay buffer for learning from past episodes

    Implements prioritized experience replay:
    - High-importance episodes replayed more frequently
    - Recent episodes prioritized
    - Diverse sampling to avoid overfitting
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        """
        Initialize replay buffer

        Args:
            capacity: Maximum number of episodes to store
            alpha: Priority exponent (0 = uniform, 1 = full priority)
        """
        self.capacity = capacity
        self.alpha = alpha

        self._episodes: Dict[str, Episode] = {}
        self._priorities: Dict[str, float] = {}
        self._lock = threading.Lock()

    def add(self, episode: Episode, priority: Optional[float] = None):
        """Add episode to replay buffer"""
        with self._lock:
            if priority is None:
                priority = episode.importance

            self._episodes[episode.episode_id] = episode
            self._priorities[episode.episode_id] = priority ** self.alpha

            # Evict oldest if over capacity
            if len(self._episodes) > self.capacity:
                # Remove lowest priority episode
                min_id = min(self._priorities, key=self._priorities.get)
                del self._episodes[min_id]
                del self._priorities[min_id]

    def sample(self, batch_size: int = 32) -> List[Episode]:
        """
        Sample episodes weighted by priority

        Args:
            batch_size: Number of episodes to sample

        Returns:
            List of sampled episodes
        """
        import random

        with self._lock:
            if not self._episodes:
                return []

            # Compute sampling probabilities
            total_priority = sum(self._priorities.values())
            probs = {k: v / total_priority for k, v in self._priorities.items()}

            # Sample
            episode_ids = list(self._episodes.keys())
            weights = [probs[eid] for eid in episode_ids]

            sampled_ids = random.choices(
                episode_ids,
                weights=weights,
                k=min(batch_size, len(episode_ids))
            )

            return [self._episodes[eid] for eid in sampled_ids]

    def update_priority(self, episode_id: str, priority: float):
        """Update priority of an episode"""
        with self._lock:
            if episode_id in self._priorities:
                self._priorities[episode_id] = priority ** self.alpha


class EpisodicMemory:
    """
    L2 Episodic Memory - Session-based experience storage

    Stores events organized into episodes with:
    - Temporal indexing
    - Causal relationship tracking
    - Experience replay for learning

    Usage:
        em = EpisodicMemory()

        # Start a new episode
        episode = em.start_episode("Investigation Task")

        # Record events
        em.record_event(EventType.QUERY, "What is the threat level?")
        em.record_event(EventType.RESPONSE, "Threat level is HIGH")

        # Close episode
        em.close_episode(summary="Investigated and identified high threat")

        # Recall similar episodes
        similar = em.recall_similar(current_context, top_k=5)
    """

    def __init__(self, max_episodes: int = 10000,
                 max_events_per_episode: int = 1000,
                 enable_replay: bool = True):
        """
        Initialize episodic memory

        Args:
            max_episodes: Maximum episodes to store
            max_events_per_episode: Max events per episode
            enable_replay: Enable experience replay buffer
        """
        self.max_episodes = max_episodes
        self.max_events_per_episode = max_events_per_episode

        # Storage
        self._episodes: Dict[str, Episode] = {}
        self._events: Dict[str, Event] = {}  # All events indexed by ID
        self._causal_links: List[CausalLink] = []

        # Indices
        self._temporal_index: Dict[str, List[str]] = defaultdict(list)  # date -> episode_ids
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> episode_ids
        self._type_index: Dict[EventType, List[str]] = defaultdict(list)  # type -> event_ids

        # Current active episode
        self._current_episode: Optional[Episode] = None

        # Experience replay
        self.enable_replay = enable_replay
        self.replay_buffer = ExperienceReplay() if enable_replay else None

        # Lock
        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            "episodes_created": 0,
            "events_recorded": 0,
            "causal_links": 0,
            "recalls": 0,
        }

        logger.info(f"EpisodicMemory initialized: max {max_episodes} episodes")

    def start_episode(self, name: str,
                      parent_id: Optional[str] = None,
                      metadata: Optional[Dict] = None,
                      tags: Optional[Set[str]] = None) -> Episode:
        """
        Start a new episode

        Args:
            name: Episode name/description
            parent_id: Parent episode ID if nested
            metadata: Additional metadata
            tags: Tags for indexing

        Returns:
            New Episode instance
        """
        with self._lock:
            # Close current episode if active
            if self._current_episode and self._current_episode.is_active:
                self.close_episode()

            # Generate ID
            episode_id = hashlib.sha256(
                f"{name}:{time.time()}".encode()
            ).hexdigest()[:16]

            episode = Episode(
                episode_id=episode_id,
                name=name,
                parent_episode=parent_id,
                metadata=metadata or {},
                tags=tags or set(),
            )

            # Update parent
            if parent_id and parent_id in self._episodes:
                self._episodes[parent_id].child_episodes.append(episode_id)

            # Store
            self._episodes[episode_id] = episode
            self._current_episode = episode

            # Index by date
            date_key = episode.start_time.date().isoformat()
            self._temporal_index[date_key].append(episode_id)

            # Index by tags
            for tag in episode.tags:
                self._tag_index[tag].add(episode_id)

            self.stats["episodes_created"] += 1

            logger.debug(f"Episode started: {episode_id} - {name}")
            return episode

    def close_episode(self, summary: Optional[str] = None,
                     insights: Optional[List[str]] = None) -> Optional[Episode]:
        """
        Close the current episode

        Args:
            summary: Episode summary
            insights: Key insights from episode

        Returns:
            Closed Episode or None
        """
        with self._lock:
            if not self._current_episode:
                return None

            episode = self._current_episode
            episode.close(summary)

            if insights:
                episode.key_insights.extend(insights)

            # Add to replay buffer
            if self.enable_replay and self.replay_buffer:
                self.replay_buffer.add(episode)

            self._current_episode = None

            logger.debug(f"Episode closed: {episode.episode_id}")
            return episode

    def record_event(self, event_type: EventType, content: Any,
                    context: Optional[Dict] = None,
                    importance: float = 0.5,
                    caused_by: Optional[str] = None,
                    tags: Optional[Set[str]] = None,
                    metadata: Optional[Dict] = None) -> Event:
        """
        Record an event

        Args:
            event_type: Type of event
            content: Event content
            context: Context at time of event
            importance: Event importance (0-1)
            caused_by: ID of causing event
            tags: Tags for indexing
            metadata: Additional metadata

        Returns:
            Recorded Event
        """
        with self._lock:
            # Generate ID
            event_id = hashlib.sha256(
                f"{event_type.name}:{time.time()}:{id(content)}".encode()
            ).hexdigest()[:16]

            event = Event(
                event_id=event_id,
                event_type=event_type,
                content=content,
                context=context or {},
                importance=importance,
                caused_by=caused_by,
                tags=tags or set(),
                metadata=metadata or {},
            )

            # Store event
            self._events[event_id] = event

            # Add to current episode
            if self._current_episode:
                if len(self._current_episode.events) < self.max_events_per_episode:
                    self._current_episode.add_event(event)
            else:
                # Auto-start episode if none active
                self.start_episode(f"Auto-episode ({event_type.name})")
                self._current_episode.add_event(event)

            # Update causal relationships
            if caused_by and caused_by in self._events:
                self._events[caused_by].causes.append(event_id)
                self._add_causal_link(caused_by, event_id, CausalRelation.CAUSES)

            # Index by type
            self._type_index[event_type].append(event_id)

            self.stats["events_recorded"] += 1

            return event

    def _add_causal_link(self, source_id: str, target_id: str,
                         relation: CausalRelation, strength: float = 0.5):
        """Add causal link between events"""
        link = CausalLink(
            source_event_id=source_id,
            target_event_id=target_id,
            relation=relation,
            strength=strength,
        )
        self._causal_links.append(link)
        self.stats["causal_links"] += 1

    def recall_episode(self, episode_id: str) -> Optional[Episode]:
        """
        Recall specific episode by ID

        Args:
            episode_id: Episode identifier

        Returns:
            Episode or None
        """
        with self._lock:
            episode = self._episodes.get(episode_id)
            if episode:
                episode.access_count += 1
                self.stats["recalls"] += 1
            return episode

    def recall_by_time(self, start: datetime, end: datetime) -> List[Episode]:
        """
        Recall episodes within time range

        Args:
            start: Start time
            end: End time

        Returns:
            List of episodes
        """
        with self._lock:
            results = []

            for episode in self._episodes.values():
                if start <= episode.start_time <= end:
                    results.append(episode)
                    episode.access_count += 1

            self.stats["recalls"] += len(results)
            return sorted(results, key=lambda e: e.start_time)

    def recall_by_tags(self, tags: Set[str], match_all: bool = False) -> List[Episode]:
        """
        Recall episodes by tags

        Args:
            tags: Tags to search for
            match_all: Require all tags (AND) vs any tag (OR)

        Returns:
            List of matching episodes
        """
        with self._lock:
            if match_all:
                # Intersection of all tag sets
                episode_ids = None
                for tag in tags:
                    tag_episodes = self._tag_index.get(tag, set())
                    if episode_ids is None:
                        episode_ids = tag_episodes.copy()
                    else:
                        episode_ids &= tag_episodes
                episode_ids = episode_ids or set()
            else:
                # Union of all tag sets
                episode_ids = set()
                for tag in tags:
                    episode_ids |= self._tag_index.get(tag, set())

            results = [self._episodes[eid] for eid in episode_ids if eid in self._episodes]

            for episode in results:
                episode.access_count += 1

            self.stats["recalls"] += len(results)
            return results

    def recall_similar(self, context: Dict[str, Any],
                       top_k: int = 5) -> List[Episode]:
        """
        Recall episodes similar to current context

        Args:
            context: Current context to match
            top_k: Number of results

        Returns:
            List of similar episodes
        """
        with self._lock:
            scored = []

            for episode in self._episodes.values():
                score = self._compute_similarity(episode, context)
                scored.append((score, episode))

            # Sort by score descending
            scored.sort(key=lambda x: x[0], reverse=True)

            results = [ep for score, ep in scored[:top_k]]

            for episode in results:
                episode.access_count += 1

            self.stats["recalls"] += len(results)
            return results

    def _compute_similarity(self, episode: Episode, context: Dict) -> float:
        """Compute similarity between episode and context"""
        score = 0.0

        # Tag overlap
        if context.get("tags"):
            context_tags = set(context["tags"])
            tag_overlap = len(episode.tags & context_tags) / max(len(context_tags), 1)
            score += tag_overlap * 0.3

        # Metadata match
        for key, value in context.items():
            if key in episode.metadata:
                if episode.metadata[key] == value:
                    score += 0.2

        # Recency bonus
        age = (datetime.now(timezone.utc) - episode.start_time).total_seconds()
        recency = 1.0 / (1.0 + age / 86400)  # Decay over days
        score += recency * 0.2

        # Importance bonus
        score += episode.importance * 0.3

        return score

    def get_causal_chain(self, event_id: str,
                         direction: str = "both",
                         max_depth: int = 5) -> List[Event]:
        """
        Get causal chain for an event

        Args:
            event_id: Starting event ID
            direction: "causes", "caused_by", or "both"
            max_depth: Maximum chain depth

        Returns:
            List of related events in causal order
        """
        with self._lock:
            if event_id not in self._events:
                return []

            chain = [self._events[event_id]]
            visited = {event_id}

            def follow_chain(current_id: str, depth: int):
                if depth >= max_depth:
                    return

                event = self._events.get(current_id)
                if not event:
                    return

                # Follow causes
                if direction in ("causes", "both"):
                    for caused_id in event.causes:
                        if caused_id not in visited:
                            visited.add(caused_id)
                            chain.append(self._events[caused_id])
                            follow_chain(caused_id, depth + 1)

                # Follow caused_by
                if direction in ("caused_by", "both"):
                    if event.caused_by and event.caused_by not in visited:
                        visited.add(event.caused_by)
                        chain.append(self._events[event.caused_by])
                        follow_chain(event.caused_by, depth + 1)

            follow_chain(event_id, 0)

            # Sort by timestamp
            chain.sort(key=lambda e: e.timestamp)

            return chain

    def sample_for_replay(self, batch_size: int = 32) -> List[Episode]:
        """Sample episodes for experience replay"""
        if self.replay_buffer:
            return self.replay_buffer.sample(batch_size)
        return []

    def get_for_consolidation(self, min_importance: float = 0.6,
                              min_access_count: int = 2) -> List[Episode]:
        """
        Get episodes ready for consolidation to semantic memory

        Returns episodes that are:
        - Important enough
        - Accessed multiple times (validated)
        - Closed (complete)
        """
        with self._lock:
            candidates = []

            for episode in self._episodes.values():
                if (not episode.is_active and
                    episode.importance >= min_importance and
                    episode.access_count >= min_access_count):
                    candidates.append(episode)

            return candidates

    def get_stats(self) -> Dict:
        """Get memory statistics"""
        with self._lock:
            active_episodes = sum(1 for e in self._episodes.values() if e.is_active)

            return {
                **self.stats,
                "total_episodes": len(self._episodes),
                "active_episodes": active_episodes,
                "total_events": len(self._events),
                "current_episode": self._current_episode.episode_id if self._current_episode else None,
            }

    def export_for_sync(self) -> Dict:
        """Export state for synchronization"""
        with self._lock:
            return {
                "episodes": {
                    eid: ep.to_dict()
                    for eid, ep in self._episodes.items()
                },
                "stats": self.stats,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


if __name__ == "__main__":
    print("Episodic Memory Self-Test")
    print("=" * 50)

    em = EpisodicMemory()

    print(f"\n[1] Start Episode")
    episode = em.start_episode(
        "Test Investigation",
        tags={"test", "investigation"},
        metadata={"priority": "high"}
    )
    print(f"    Episode ID: {episode.episode_id}")

    print(f"\n[2] Record Events")
    e1 = em.record_event(
        EventType.QUERY,
        "What is the current threat level?",
        importance=0.7
    )
    print(f"    Event 1: {e1.event_id}")

    e2 = em.record_event(
        EventType.RESPONSE,
        "Threat level is HIGH based on recent indicators",
        importance=0.8,
        caused_by=e1.event_id
    )
    print(f"    Event 2: {e2.event_id} (caused by E1)")

    e3 = em.record_event(
        EventType.ACTION,
        "Initiating defensive measures",
        importance=0.9,
        caused_by=e2.event_id
    )
    print(f"    Event 3: {e3.event_id} (caused by E2)")

    print(f"\n[3] Close Episode")
    em.close_episode(
        summary="Identified high threat and initiated defenses",
        insights=["Threat correlation confirmed", "Response time: 2 seconds"]
    )

    print(f"\n[4] Causal Chain")
    chain = em.get_causal_chain(e1.event_id, direction="causes")
    print(f"    Chain length: {len(chain)}")
    for event in chain:
        print(f"      - {event.event_type.name}: {str(event.content)[:50]}")

    print(f"\n[5] Start New Episode and Recall")
    em.start_episode("Related Task", tags={"test"})
    similar = em.recall_similar({"tags": ["investigation"]}, top_k=3)
    print(f"    Similar episodes: {len(similar)}")

    print(f"\n[6] Consolidation Candidates")
    # Access episode multiple times
    em.recall_episode(episode.episode_id)
    em.recall_episode(episode.episode_id)

    candidates = em.get_for_consolidation()
    print(f"    Candidates: {len(candidates)}")

    print(f"\n[7] Statistics")
    stats = em.get_stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 50)
    print("Episodic Memory test complete")

