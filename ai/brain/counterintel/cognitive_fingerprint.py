#!/usr/bin/env python3
"""
Cognitive Fingerprinting for DSMIL Brain

Identifies and attributes actors based on behavioral patterns:
- Linguistic analysis (writing style)
- Tactical patterns (attack methodology)
- Temporal patterns (activity schedule)
- Tool preferences
- Decision patterns
"""

import hashlib
import threading
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class FingerprintType(Enum):
    """Types of fingerprint components"""
    LINGUISTIC = auto()
    TACTICAL = auto()
    TEMPORAL = auto()
    TOOLING = auto()
    DECISION = auto()
    NETWORK = auto()


@dataclass
class FingerPrintComponent:
    """A component of a cognitive fingerprint"""
    component_type: FingerprintType
    features: Dict[str, float]
    confidence: float
    sample_count: int
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def similarity(self, other: "FingerPrintComponent") -> float:
        """Calculate similarity with another component"""
        if self.component_type != other.component_type:
            return 0.0

        # Cosine similarity of feature vectors
        common_keys = set(self.features.keys()) & set(other.features.keys())
        if not common_keys:
            return 0.0

        dot_product = sum(self.features[k] * other.features[k] for k in common_keys)

        mag1 = math.sqrt(sum(v ** 2 for v in self.features.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in other.features.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)


@dataclass
class CognitiveFingerprint:
    """Complete cognitive fingerprint for an entity"""
    entity_id: str
    entity_type: str  # "threat_actor", "user", "bot", etc.

    # Components
    linguistic: Optional[FingerPrintComponent] = None
    tactical: Optional[FingerPrintComponent] = None
    temporal: Optional[FingerPrintComponent] = None
    tooling: Optional[FingerPrintComponent] = None
    decision: Optional[FingerPrintComponent] = None

    # Metadata
    aliases: Set[str] = field(default_factory=set)
    confidence: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def get_components(self) -> List[FingerPrintComponent]:
        """Get all non-None components"""
        components = []
        for comp in [self.linguistic, self.tactical, self.temporal, self.tooling, self.decision]:
            if comp:
                components.append(comp)
        return components

    def overall_similarity(self, other: "CognitiveFingerprint") -> float:
        """Calculate overall similarity with another fingerprint"""
        similarities = []
        weights = {
            FingerprintType.LINGUISTIC: 0.25,
            FingerprintType.TACTICAL: 0.30,
            FingerprintType.TEMPORAL: 0.15,
            FingerprintType.TOOLING: 0.20,
            FingerprintType.DECISION: 0.10,
        }

        component_pairs = [
            (self.linguistic, other.linguistic),
            (self.tactical, other.tactical),
            (self.temporal, other.temporal),
            (self.tooling, other.tooling),
            (self.decision, other.decision),
        ]

        total_weight = 0
        weighted_sim = 0

        for c1, c2 in component_pairs:
            if c1 and c2:
                sim = c1.similarity(c2)
                weight = weights.get(c1.component_type, 0.1)
                weighted_sim += sim * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sim / total_weight


@dataclass
class AttributionMatch:
    """Result of fingerprint matching"""
    match_id: str
    query_fingerprint_id: str
    matched_entity_id: str

    # Similarity scores
    overall_similarity: float
    component_similarities: Dict[str, float] = field(default_factory=dict)

    # Confidence
    confidence: float = 0.0

    # Evidence
    matching_features: List[str] = field(default_factory=list)

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class LinguisticAnalyzer:
    """Analyzes linguistic patterns in text"""

    def analyze(self, texts: List[str]) -> FingerPrintComponent:
        """Analyze linguistic features from text samples"""
        if not texts:
            return FingerPrintComponent(
                component_type=FingerprintType.LINGUISTIC,
                features={},
                confidence=0.0,
                sample_count=0,
            )

        features = {}

        # Aggregate features from all texts
        all_words = []
        all_sentences = []

        for text in texts:
            words = text.lower().split()
            all_words.extend(words)
            sentences = re.split(r'[.!?]+', text)
            all_sentences.extend([s for s in sentences if s.strip()])

        # Average word length
        if all_words:
            features["avg_word_length"] = sum(len(w) for w in all_words) / len(all_words)

        # Average sentence length
        if all_sentences:
            features["avg_sentence_length"] = sum(len(s.split()) for s in all_sentences) / len(all_sentences)

        # Vocabulary richness (unique words / total words)
        if all_words:
            features["vocabulary_richness"] = len(set(all_words)) / len(all_words)

        # Punctuation frequency
        all_text = " ".join(texts)
        if all_text:
            features["exclamation_freq"] = all_text.count("!") / len(all_text) * 100
            features["question_freq"] = all_text.count("?") / len(all_text) * 100
            features["comma_freq"] = all_text.count(",") / len(all_text) * 100

        # Common word patterns (function words)
        function_words = ["the", "a", "an", "is", "are", "was", "were", "be", "been",
                         "have", "has", "had", "do", "does", "did", "will", "would",
                         "could", "should", "may", "might", "must", "shall"]

        word_counts = Counter(all_words)
        total_words = len(all_words)

        for fw in function_words:
            features[f"fw_{fw}"] = word_counts.get(fw, 0) / max(1, total_words)

        # Confidence based on sample size
        confidence = min(0.95, len(texts) / 20)

        return FingerPrintComponent(
            component_type=FingerprintType.LINGUISTIC,
            features=features,
            confidence=confidence,
            sample_count=len(texts),
        )


class TacticalAnalyzer:
    """Analyzes tactical patterns in attacks"""

    def analyze(self, attack_patterns: List[Dict]) -> FingerPrintComponent:
        """Analyze tactical features from attack patterns"""
        if not attack_patterns:
            return FingerPrintComponent(
                component_type=FingerprintType.TACTICAL,
                features={},
                confidence=0.0,
                sample_count=0,
            )

        features = {}

        # Count techniques used
        technique_counts = Counter()
        target_counts = Counter()
        method_counts = Counter()

        for pattern in attack_patterns:
            techniques = pattern.get("techniques", [])
            for tech in techniques:
                technique_counts[tech] += 1

            target = pattern.get("target_type")
            if target:
                target_counts[target] += 1

            method = pattern.get("method")
            if method:
                method_counts[method] += 1

        # Normalize technique frequencies
        total_patterns = len(attack_patterns)
        for tech, count in technique_counts.most_common(20):
            features[f"tech_{tech}"] = count / total_patterns

        for target, count in target_counts.most_common(10):
            features[f"target_{target}"] = count / total_patterns

        for method, count in method_counts.most_common(10):
            features[f"method_{method}"] = count / total_patterns

        # Tactical characteristics
        features["technique_diversity"] = len(technique_counts) / max(1, total_patterns)

        confidence = min(0.95, len(attack_patterns) / 10)

        return FingerPrintComponent(
            component_type=FingerprintType.TACTICAL,
            features=features,
            confidence=confidence,
            sample_count=len(attack_patterns),
        )


class TemporalAnalyzer:
    """Analyzes temporal patterns in activity"""

    def analyze(self, timestamps: List[datetime]) -> FingerPrintComponent:
        """Analyze temporal patterns from activity timestamps"""
        if not timestamps:
            return FingerPrintComponent(
                component_type=FingerprintType.TEMPORAL,
                features={},
                confidence=0.0,
                sample_count=0,
            )

        features = {}

        # Hour distribution
        hour_counts = Counter(t.hour for t in timestamps)
        for hour in range(24):
            features[f"hour_{hour}"] = hour_counts.get(hour, 0) / len(timestamps)

        # Day of week distribution
        day_counts = Counter(t.weekday() for t in timestamps)
        for day in range(7):
            features[f"day_{day}"] = day_counts.get(day, 0) / len(timestamps)

        # Peak hours
        peak_hours = [h for h, _ in hour_counts.most_common(3)]
        features["peak_hour_1"] = peak_hours[0] if len(peak_hours) > 0 else 12
        features["peak_hour_2"] = peak_hours[1] if len(peak_hours) > 1 else 12

        # Activity consistency
        if len(timestamps) > 1:
            sorted_ts = sorted(timestamps)
            intervals = [(sorted_ts[i+1] - sorted_ts[i]).total_seconds() / 3600
                        for i in range(len(sorted_ts) - 1)]
            if intervals:
                features["avg_interval_hours"] = sum(intervals) / len(intervals)
                features["interval_variance"] = sum((i - features["avg_interval_hours"]) ** 2
                                                   for i in intervals) / len(intervals)

        confidence = min(0.95, len(timestamps) / 50)

        return FingerPrintComponent(
            component_type=FingerprintType.TEMPORAL,
            features=features,
            confidence=confidence,
            sample_count=len(timestamps),
        )


class CognitiveFingerprinter:
    """
    Cognitive Fingerprinting Engine

    Creates and matches behavioral fingerprints for entity attribution.

    Usage:
        fingerprinter = CognitiveFingerprinter()

        # Build fingerprint from observations
        fp = fingerprinter.build_fingerprint(
            entity_id="unknown-actor",
            texts=["sample text 1", "sample text 2"],
            attack_patterns=[{"techniques": ["T1566"], "target_type": "email"}],
            timestamps=[datetime.now()]
        )

        # Match against known fingerprints
        matches = fingerprinter.match_fingerprint(fp)
    """

    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize fingerprinter

        Args:
            similarity_threshold: Minimum similarity for a match
        """
        self.similarity_threshold = similarity_threshold

        self._fingerprints: Dict[str, CognitiveFingerprint] = {}

        # Analyzers
        self._linguistic = LinguisticAnalyzer()
        self._tactical = TacticalAnalyzer()
        self._temporal = TemporalAnalyzer()

        self._lock = threading.RLock()

        logger.info("CognitiveFingerprinter initialized")

    def build_fingerprint(self, entity_id: str,
                         entity_type: str = "unknown",
                         texts: Optional[List[str]] = None,
                         attack_patterns: Optional[List[Dict]] = None,
                         timestamps: Optional[List[datetime]] = None,
                         tools: Optional[List[str]] = None) -> CognitiveFingerprint:
        """
        Build cognitive fingerprint from observations

        Args:
            entity_id: Entity identifier
            entity_type: Type of entity
            texts: Text samples for linguistic analysis
            attack_patterns: Attack pattern observations
            timestamps: Activity timestamps
            tools: Tools used

        Returns:
            CognitiveFingerprint
        """
        fingerprint = CognitiveFingerprint(
            entity_id=entity_id,
            entity_type=entity_type,
        )

        # Linguistic component
        if texts:
            fingerprint.linguistic = self._linguistic.analyze(texts)

        # Tactical component
        if attack_patterns:
            fingerprint.tactical = self._tactical.analyze(attack_patterns)

        # Temporal component
        if timestamps:
            fingerprint.temporal = self._temporal.analyze(timestamps)

        # Tooling component
        if tools:
            tool_counts = Counter(tools)
            total = len(tools)
            features = {f"tool_{t}": c / total for t, c in tool_counts.items()}
            fingerprint.tooling = FingerPrintComponent(
                component_type=FingerprintType.TOOLING,
                features=features,
                confidence=min(0.95, len(tools) / 10),
                sample_count=len(tools),
            )

        # Calculate overall confidence
        components = fingerprint.get_components()
        if components:
            fingerprint.confidence = sum(c.confidence for c in components) / len(components)

        return fingerprint

    def register_fingerprint(self, fingerprint: CognitiveFingerprint):
        """Register a fingerprint for matching"""
        with self._lock:
            self._fingerprints[fingerprint.entity_id] = fingerprint

    def update_fingerprint(self, entity_id: str,
                          texts: Optional[List[str]] = None,
                          attack_patterns: Optional[List[Dict]] = None,
                          timestamps: Optional[List[datetime]] = None):
        """Update an existing fingerprint with new observations"""
        with self._lock:
            if entity_id not in self._fingerprints:
                return

            existing = self._fingerprints[entity_id]

            # Merge new observations
            if texts and existing.linguistic:
                new_ling = self._linguistic.analyze(texts)
                # Average features
                for key, value in new_ling.features.items():
                    if key in existing.linguistic.features:
                        existing.linguistic.features[key] = (
                            existing.linguistic.features[key] + value
                        ) / 2
                    else:
                        existing.linguistic.features[key] = value
                existing.linguistic.sample_count += new_ling.sample_count

            existing.last_updated = datetime.now(timezone.utc)

    def match_fingerprint(self, query: CognitiveFingerprint,
                         top_k: int = 5) -> List[AttributionMatch]:
        """
        Match a fingerprint against known fingerprints

        Args:
            query: Fingerprint to match
            top_k: Number of top matches to return

        Returns:
            List of AttributionMatch results
        """
        matches = []

        with self._lock:
            for entity_id, known in self._fingerprints.items():
                if entity_id == query.entity_id:
                    continue

                similarity = query.overall_similarity(known)

                if similarity >= self.similarity_threshold:
                    # Calculate component similarities
                    component_sims = {}
                    if query.linguistic and known.linguistic:
                        component_sims["linguistic"] = query.linguistic.similarity(known.linguistic)
                    if query.tactical and known.tactical:
                        component_sims["tactical"] = query.tactical.similarity(known.tactical)
                    if query.temporal and known.temporal:
                        component_sims["temporal"] = query.temporal.similarity(known.temporal)
                    if query.tooling and known.tooling:
                        component_sims["tooling"] = query.tooling.similarity(known.tooling)

                    match = AttributionMatch(
                        match_id=hashlib.sha256(
                            f"{query.entity_id}:{entity_id}".encode()
                        ).hexdigest()[:16],
                        query_fingerprint_id=query.entity_id,
                        matched_entity_id=entity_id,
                        overall_similarity=similarity,
                        component_similarities=component_sims,
                        confidence=similarity * query.confidence * known.confidence,
                    )
                    matches.append(match)

        # Sort by similarity
        matches.sort(key=lambda m: m.overall_similarity, reverse=True)

        return matches[:top_k]

    def get_fingerprint(self, entity_id: str) -> Optional[CognitiveFingerprint]:
        """Get fingerprint by entity ID"""
        return self._fingerprints.get(entity_id)

    def get_stats(self) -> Dict:
        """Get fingerprinter statistics"""
        return {
            "registered_fingerprints": len(self._fingerprints),
            "similarity_threshold": self.similarity_threshold,
        }


if __name__ == "__main__":
    print("Cognitive Fingerprinting Self-Test")
    print("=" * 50)

    fingerprinter = CognitiveFingerprinter(similarity_threshold=0.5)

    print("\n[1] Build Known Fingerprints")

    # APT29 fingerprint
    apt29_fp = fingerprinter.build_fingerprint(
        entity_id="APT29",
        entity_type="threat_actor",
        texts=[
            "We have successfully compromised the target network.",
            "The payload has been delivered via spearphishing email.",
            "Exfiltration will proceed through DNS tunneling.",
        ],
        attack_patterns=[
            {"techniques": ["T1566.001", "T1071.004"], "target_type": "government", "method": "spearphishing"},
            {"techniques": ["T1566.001", "T1059.001"], "target_type": "government", "method": "spearphishing"},
        ],
        timestamps=[
            datetime(2024, 1, 15, 14, 30),
            datetime(2024, 1, 15, 15, 45),
            datetime(2024, 1, 16, 14, 20),
        ],
        tools=["Cobalt Strike", "Mimikatz", "PowerShell"],
    )
    fingerprinter.register_fingerprint(apt29_fp)
    print(f"    APT29: confidence={apt29_fp.confidence:.2f}")

    # APT28 fingerprint
    apt28_fp = fingerprinter.build_fingerprint(
        entity_id="APT28",
        entity_type="threat_actor",
        texts=[
            "Target acquired. Beginning exploitation phase.",
            "Credential harvesting in progress via phishing site.",
            "C2 established. Moving to lateral movement.",
        ],
        attack_patterns=[
            {"techniques": ["T1189", "T1203"], "target_type": "defense", "method": "watering_hole"},
            {"techniques": ["T1566.002", "T1078"], "target_type": "defense", "method": "credential_theft"},
        ],
        timestamps=[
            datetime(2024, 1, 15, 9, 30),
            datetime(2024, 1, 15, 10, 15),
            datetime(2024, 1, 16, 9, 45),
        ],
        tools=["X-Agent", "Responder", "Impacket"],
    )
    fingerprinter.register_fingerprint(apt28_fp)
    print(f"    APT28: confidence={apt28_fp.confidence:.2f}")

    print("\n[2] Build Unknown Actor Fingerprint")
    unknown_fp = fingerprinter.build_fingerprint(
        entity_id="unknown-001",
        entity_type="unknown",
        texts=[
            "Successfully compromised target infrastructure.",
            "Payload delivered via email attachment.",
            "Data exfiltration via DNS in progress.",
        ],
        attack_patterns=[
            {"techniques": ["T1566.001", "T1071.004"], "target_type": "government", "method": "spearphishing"},
        ],
        timestamps=[
            datetime(2024, 2, 1, 14, 15),
            datetime(2024, 2, 1, 15, 30),
        ],
        tools=["Cobalt Strike", "PowerShell"],
    )
    print(f"    Unknown: confidence={unknown_fp.confidence:.2f}")

    print("\n[3] Match Fingerprint")
    matches = fingerprinter.match_fingerprint(unknown_fp)
    print(f"    Found {len(matches)} matches:")
    for match in matches:
        print(f"      - {match.matched_entity_id}: {match.overall_similarity:.2%} similarity")
        for comp, sim in match.component_similarities.items():
            print(f"          {comp}: {sim:.2%}")

    print("\n[4] Statistics")
    stats = fingerprinter.get_stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 50)
    print("Cognitive Fingerprinting test complete")

