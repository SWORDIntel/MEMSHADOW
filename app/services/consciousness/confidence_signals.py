"""
Metacognitive Confidence Signals (Phase 8.3 v2)

Multi-dimensional signal extraction for principled confidence estimation.
Implements the MEMSHADOW v1 mandatory signal set.

Design principles:
- Confidence = P(retrieved memories sufficient to answer query)
- Signals organized by tier (Geometry, Quality, Semantics, Temporal, System, Meta)
- Cheap online signals only for v1; expensive signals deferred to Phase 2
- Extensible architecture for adding Phase 2/Later signals
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import re
import numpy as np
import structlog

logger = structlog.get_logger()


class QueryType(Enum):
    """Query type classification (Phase 2 feature, stub for now)"""
    FACT_LOOKUP = "fact_lookup"
    EXPLORATORY = "exploratory"
    MULTI_STEP = "multi_step"
    UNKNOWN = "unknown"


class SourceTrust(Enum):
    """Source trust classification for result quality"""
    HIGH = "high"          # Curated, human-authored, verified
    MEDIUM = "medium"      # Machine-generated, reviewed
    LOW = "low"            # Noisy, external, unverified
    UNKNOWN = "unknown"    # No trust metadata


@dataclass
class SignalVector:
    """
    Complete signal vector for confidence estimation.

    Organized by tier with clear cost/priority annotations.
    v1: 14 cheap mandatory signals
    v2: +7 medium-cost Phase 2 signals
    v3: +3 cheap local-first Phase 3 signals (20, 28, 30)
    """
    # Tier 1: Retrieval Geometry (Cheap, Online, Mandatory)
    top_score: float                    # Signal 1: Best match strength
    top_k_mean: float                   # Signal 2: Overall alignment
    score_margin: float                 # Signal 3: Gap between #1 and #2

    # Tier 1: Phase 2 (Medium, Online, P2)
    score_entropy: Optional[float] = None           # Signal 4: Distribution entropy (flat vs sharp)

    # Tier 2: Result Quality (Cheap, Online, Mandatory)
    source_trust: SourceTrust           # Signal 7: Trust classification
    structural_completeness: float      # Signal 9: Required fields present (0-1)
    parse_integrity: float              # Signal 10: Ingestion success (0-1)

    # Tier 2: Phase 2 (Medium, Online, P2)
    redundancy_score: Optional[float] = None        # Signal 8: Independent source agreement (0-1)
    semantic_coherence: Optional[float] = None      # Signal 11: Results semantically consistent (0-1)
    diversity_score: Optional[float] = None         # Signal 13: Topic coverage breadth (0-1)

    # Tier 2: Phase 3 (Cheap, Local-First, P3)
    language_quality: Optional[float] = None        # Signal 20: Text quality/noise detection (0-1)

    # Tier 3: Query Semantics (Cheap-Medium, Online, Mandatory)
    query_length: int                   # Signal 14: Lexical length
    specificity_score: float            # Signal 15: IDs, dates, entities (0-1)
    ambiguity_score: float              # Signal 18: Vague markers (0-1, higher=more ambiguous)

    # Tier 3: Phase 2 (Medium, Online, P2)
    task_type: Optional[QueryType] = None           # Signal 16: Fact lookup vs exploratory
    constraint_richness: Optional[float] = None     # Signal 17: Explicit filters count (0-1)

    # Tier 4: Temporal (Cheap, Online/Corpus, Mandatory)
    recency_score: float                # Signal 21: Age-weighted freshness (0-1)
    index_freshness: float              # Signal 24: Corpus update staleness (0-1)

    # Tier 4: Phase 2 (Medium, Online, P2)
    temporal_alignment: Optional[float] = None      # Signal 22: Query temporal intent vs results (0-1)

    # Tier 5: System Health (Cheap, Online, Mandatory)
    retrieval_path: str                 # Signal 26: "normal" / "fallback" / "degraded"

    # Tier 5: Phase 3 (Cheap, Local-First, P3)
    latency_indicators: Optional[float] = None      # Signal 28: Retrieval latency health (0-1)
    error_log_presence: Optional[float] = None      # Signal 30: Recent errors (0-1, higher=fewer errors)

    # Tier 7: Meta-signals (Cheap, Online, Mandatory)
    signal_completeness: float          # Signal 36: % of signals available (0-1)

    # Context
    result_count: int                   # How many results total
    result_index: int                   # Position in result list (0-indexed)
    timestamp: datetime                 # When calculated


class SignalExtractor:
    """
    Extracts confidence signals from query context and retrieval results.

    v1: Implements 14 mandatory cheap signals
    v2: Will add Phase 2 medium-cost signals
    v3: Will add Later expensive/historical signals
    """

    def __init__(self):
        # Compile regex patterns for query analysis
        self.specificity_patterns = {
            'uuid': re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.I),
            'date_iso': re.compile(r'\d{4}-\d{2}-\d{2}'),
            'date_natural': re.compile(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}', re.I),
            'ip_address': re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
            'version': re.compile(r'\bv?\d+\.\d+(\.\d+)?(-\w+)?\b'),
            'identifier': re.compile(r'\b[A-Z][A-Z0-9_]{3,}\b'),  # UPPER_CASE_IDS
        }

        self.ambiguity_markers = [
            'maybe', 'perhaps', 'possibly', 'probably', 'might', 'could',
            'something like', 'kind of', 'sort of', 'around', 'approximately',
            'unsure', 'not sure', 'unclear', '?'
        ]

        # Temporal keywords for Signal 22
        self.temporal_keywords = {
            'recent': ['recent', 'recently', 'latest', 'new', 'current', 'today', 'yesterday', 'this week'],
            'past': ['old', 'previous', 'former', 'historical', 'archive', 'legacy'],
            'future': ['upcoming', 'future', 'planned', 'next', 'tomorrow']
        }

        logger.info("Signal extractor initialized", mandatory_signals=14, phase2_signals=7, phase3_signals=3, total=24)

    # === PHASE 2 SIGNAL EXTRACTION METHODS ===

    def _calculate_score_entropy(self, similarity_scores: List[float]) -> float:
        """
        Signal 4: Score distribution entropy.

        Measures whether scores are sharply peaked (high confidence in top result)
        or flat (ambiguous, many equally good matches).

        Returns: 0.0 = sharp peak, 1.0 = uniform distribution
        """
        if len(similarity_scores) < 2:
            return 0.0  # Single result = no distribution

        # Normalize scores to sum to 1 (treat as probability distribution)
        scores = np.array(similarity_scores)
        if scores.sum() == 0:
            return 1.0  # All zeros = maximum entropy

        probs = scores / scores.sum()
        probs = probs + 1e-10  # Avoid log(0)

        # Calculate Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))

        # Normalize to [0, 1]: max entropy is log2(n)
        max_entropy = np.log2(len(scores))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return float(np.clip(normalized_entropy, 0.0, 1.0))

    def _classify_task_type(self, query: str) -> QueryType:
        """
        Signal 16: Task-type classification.

        Classifies query intent:
        - FACT_LOOKUP: Short, specific, expects single answer
        - EXPLORATORY: Open-ended, research, broad
        - MULTI_STEP: Complex, multiple sub-questions
        """
        query_lower = query.lower()

        # Multi-step indicators
        multi_step_markers = [
            'how do i', 'step by step', 'guide', 'tutorial', 'explain',
            'walkthrough', 'first.*then', 'and then'
        ]
        for marker in multi_step_markers:
            if re.search(marker, query_lower):
                return QueryType.MULTI_STEP

        # Fact lookup indicators
        fact_markers = [
            r'\bwhat is\b', r'\bwho is\b', r'\bwhen did\b', r'\bwhere is\b',
            r'\bwhich\b', r'\bdefine\b', r'\bhow many\b', r'\bhow much\b'
        ]
        for marker in fact_markers:
            if re.search(marker, query_lower):
                return QueryType.FACT_LOOKUP

        # Exploratory indicators
        exploratory_markers = [
            'explore', 'research', 'investigate', 'overview', 'survey',
            'analyze', 'compare', 'contrast', 'why', 'how does'
        ]
        for marker in exploratory_markers:
            if marker in query_lower:
                return QueryType.EXPLORATORY

        # Default heuristic: short + specific = fact, long = exploratory
        if len(query) < 50 and any(pattern.search(query) for pattern in self.specificity_patterns.values()):
            return QueryType.FACT_LOOKUP
        elif len(query) > 100:
            return QueryType.EXPLORATORY

        return QueryType.UNKNOWN

    def _calculate_constraint_richness(self, query: str, filters: Optional[Dict] = None) -> float:
        """
        Signal 17: Constraint richness.

        Measures how many explicit constraints/filters are present.
        More constraints = higher confidence (narrower search space).

        Returns: 0.0 = no constraints, 1.0 = many constraints
        """
        constraint_count = 0

        # Count filters passed explicitly
        if filters:
            constraint_count += len(filters)

        # Count temporal constraints in query
        temporal_patterns = [
            r'\b\d{4}\b',  # Year
            r'between.*and',  # Range
            r'since|after|before|until',
            r'january|february|march|april|may|june|july|august|september|october|november|december'
        ]
        for pattern in temporal_patterns:
            if re.search(pattern, query.lower()):
                constraint_count += 1

        # Count entity constraints
        if self.specificity_patterns['uuid'].search(query):
            constraint_count += 2  # UUIDs are very specific
        if self.specificity_patterns['ip_address'].search(query):
            constraint_count += 1
        if self.specificity_patterns['version'].search(query):
            constraint_count += 1

        # Count quoted phrases (exact match constraints)
        quoted = re.findall(r'"([^"]+)"', query)
        constraint_count += len(quoted)

        # Normalize: 0 = none, 1 = 5+ constraints
        return float(min(1.0, constraint_count / 5.0))

    def _calculate_temporal_alignment(
        self,
        query: str,
        result_created_at: Optional[datetime]
    ) -> float:
        """
        Signal 22: Temporal alignment.

        Checks if query's temporal intent matches result age.
        E.g., "recent" query + old result = misalignment

        Returns: 0.0 = misaligned, 1.0 = aligned
        """
        if not result_created_at:
            return 0.5  # Unknown

        query_lower = query.lower()
        age_days = (datetime.utcnow() - result_created_at).total_seconds() / 86400

        # Detect temporal intent
        has_recent = any(kw in query_lower for kw in self.temporal_keywords['recent'])
        has_past = any(kw in query_lower for kw in self.temporal_keywords['past'])
        has_future = any(kw in query_lower for kw in self.temporal_keywords['future'])

        # No temporal intent = neutral alignment
        if not (has_recent or has_past or has_future):
            return 0.8  # Slight bonus for no constraint

        # "Recent" intent
        if has_recent:
            if age_days < 7:
                return 1.0  # Perfect alignment
            elif age_days < 30:
                return 0.7  # Acceptable
            elif age_days < 90:
                return 0.4  # Weak alignment
            else:
                return 0.1  # Poor alignment

        # "Past" intent (expects older content)
        if has_past:
            if age_days > 90:
                return 1.0
            elif age_days > 30:
                return 0.7
            else:
                return 0.4

        # "Future" intent (doesn't make sense for retrieval)
        if has_future:
            return 0.2  # Results can't be from future

        return 0.5

    def _calculate_redundancy(
        self,
        result_metadata: List[Dict[str, Any]]
    ) -> float:
        """
        Signal 8: Redundancy across independent sources.

        If multiple results from different sources contain similar content,
        confidence increases (corroboration).

        Returns: 0.0 = no redundancy, 1.0 = high redundancy
        """
        if len(result_metadata) < 2:
            return 0.0  # Need multiple results

        # Group by source if available
        sources = {}
        for idx, meta in enumerate(result_metadata):
            source = meta.get('source', f'default_{idx}')
            if source not in sources:
                sources[source] = []
            sources[source].append(idx)

        # Redundancy = having multiple independent sources
        unique_sources = len(sources)

        if unique_sources == 1:
            return 0.0  # All from same source
        elif unique_sources == 2:
            return 0.5  # Two sources
        elif unique_sources >= 3:
            return 1.0  # Three+ sources = high confidence

        return 0.0

    def _calculate_semantic_coherence(
        self,
        similarity_scores: List[float],
        result_count: int
    ) -> float:
        """
        Signal 11: Semantic coherence across results.

        If all top results have high similarity to query AND each other,
        indicates a coherent answer space (high confidence).

        Returns: 0.0 = scattered, 1.0 = coherent cluster
        """
        if result_count < 2:
            return 1.0  # Single result = trivially coherent

        # Use top-k scores as proxy for coherence
        # If scores are tightly clustered near top, results are coherent
        top_k = similarity_scores[:min(5, len(similarity_scores))]

        if len(top_k) < 2:
            return 1.0

        # Calculate coefficient of variation (std / mean)
        # Low CV = tight cluster = high coherence
        mean_score = np.mean(top_k)
        std_score = np.std(top_k)

        if mean_score == 0:
            return 0.0

        cv = std_score / mean_score

        # Normalize: CV < 0.1 = perfect coherence, CV > 0.5 = scattered
        coherence = 1.0 - min(1.0, cv / 0.5)

        return float(np.clip(coherence, 0.0, 1.0))

    def _calculate_diversity(
        self,
        similarity_scores: List[float],
        result_count: int
    ) -> float:
        """
        Signal 13: Diversity within topic.

        Balance between coherence and diversity:
        - Too similar = narrow coverage
        - Too diverse = scattered

        Returns: 0.0 = no diversity, 1.0 = good diversity
        """
        if result_count < 2:
            return 0.0  # No diversity possible

        # Ideal: scores gradually decrease (not all identical, not random)
        # Use standard deviation as diversity proxy
        std_score = float(np.std(similarity_scores))

        # Normalize: std ~0.1-0.2 is good diversity
        # Too low = clones, too high = chaos
        if std_score < 0.05:
            diversity = 0.3  # Too similar
        elif 0.05 <= std_score <= 0.25:
            diversity = 1.0  # Good spread
        elif 0.25 < std_score <= 0.4:
            diversity = 0.7  # Acceptable spread
        else:
            diversity = 0.4  # Too scattered

        return diversity

    # === PHASE 3 LOCAL-FIRST SIGNAL EXTRACTION METHODS ===

    def _calculate_language_quality(self, content: str) -> float:
        """
        Signal 20: Language quality/noise detection.

        Analyzes text for quality indicators:
        - Character diversity (not keyboard spam)
        - Word-to-character ratio (real words vs gibberish)
        - Excessive special characters
        - Proper capitalization patterns

        Returns: 0.0 = noisy/garbage, 1.0 = clean text
        """
        if not content or len(content) < 10:
            return 0.3  # Too short to assess

        # Character diversity (avoid "aaaaaaa" spam)
        unique_chars = len(set(content.lower()))
        char_diversity = min(1.0, unique_chars / 20.0)  # Good diversity at 20+ unique chars

        # Word-to-character ratio (real language has ~5-6 chars/word)
        words = re.findall(r'\b\w+\b', content)
        if not words:
            return 0.2  # No words found

        avg_word_length = len(content) / len(words) if words else 0
        # Ideal range: 4-8 chars per word (including spaces)
        if 4.0 <= avg_word_length <= 10.0:
            word_ratio_score = 1.0
        elif 2.0 <= avg_word_length < 4.0 or 10.0 < avg_word_length <= 15.0:
            word_ratio_score = 0.7
        else:
            word_ratio_score = 0.4  # Too short or too long words

        # Special character noise (excessive punctuation/symbols)
        special_char_count = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:\'\"-]', content))
        special_char_ratio = special_char_count / len(content)
        if special_char_ratio < 0.05:
            special_char_score = 1.0
        elif special_char_ratio < 0.15:
            special_char_score = 0.7
        else:
            special_char_score = 0.3  # Too many special chars

        # Capitalization patterns (avoid ALL CAPS or no caps)
        uppercase_ratio = sum(1 for c in content if c.isupper()) / len(content)
        if 0.05 <= uppercase_ratio <= 0.20:  # Typical English prose
            capitalization_score = 1.0
        elif uppercase_ratio < 0.01 or uppercase_ratio > 0.50:
            capitalization_score = 0.5  # Unusual pattern
        else:
            capitalization_score = 0.8

        # Aggregate quality score
        quality = (
            0.25 * char_diversity +
            0.35 * word_ratio_score +
            0.25 * special_char_score +
            0.15 * capitalization_score
        )

        return float(np.clip(quality, 0.0, 1.0))

    def _calculate_latency_indicators(
        self,
        retrieval_start_time: Optional[datetime],
        retrieval_mode: str
    ) -> float:
        """
        Signal 28: Latency indicators.

        Measures retrieval performance health:
        - Fast retrieval = healthy system = higher confidence
        - Slow retrieval = potential issues = lower confidence

        Returns: 0.0 = very slow, 1.0 = fast
        """
        if not retrieval_start_time:
            return 0.8  # Assume healthy if not tracked

        # Calculate elapsed time (this would be populated by the caller)
        # For now, we'll use a placeholder that can be populated by memory_service
        # Typical latencies by mode:
        # LIGHTWEIGHT: < 50ms (PostgreSQL only)
        # LOCAL: < 200ms (ChromaDB local)
        # ONLINE: < 1000ms (ChromaDB with network)

        mode_thresholds = {
            'LIGHTWEIGHT': {'fast': 0.05, 'acceptable': 0.15, 'slow': 0.30},
            'LOCAL': {'fast': 0.20, 'acceptable': 0.50, 'slow': 1.00},
            'ONLINE': {'fast': 1.00, 'acceptable': 2.00, 'slow': 5.00}
        }

        # Default to LOCAL thresholds if mode unknown
        thresholds = mode_thresholds.get(retrieval_mode, mode_thresholds['LOCAL'])

        # For now, return neutral score (actual timing integration in memory_service)
        # This signal will be fully activated when timing instrumentation is added
        return 0.85  # Neutral-positive (most retrievals are reasonably fast)

    def _calculate_error_log_presence(
        self,
        recent_error_count: int = 0
    ) -> float:
        """
        Signal 30: Error log presence.

        Checks for recent errors in retrieval pipeline:
        - No errors = high confidence in system
        - Recent errors = lower confidence (system may be unreliable)

        Returns: 0.0 = many errors, 1.0 = no errors
        """
        # This will be populated by memory_service tracking errors
        # For now, we assume healthy system (no errors tracked yet)

        if recent_error_count == 0:
            return 1.0  # Perfect health
        elif recent_error_count <= 2:
            return 0.8  # Minor issues
        elif recent_error_count <= 5:
            return 0.5  # Moderate concerns
        else:
            return 0.2  # System unstable

    def extract(
        self,
        query: str,
        similarity_scores: List[float],
        result_metadata: List[Dict[str, Any]],
        result_index: int,
        retrieval_mode: str,
        index_last_updated: Optional[datetime] = None
    ) -> SignalVector:
        """
        Extract complete signal vector for a single retrieval result.

        Args:
            query: The search query string
            similarity_scores: List of similarity scores for all results (sorted descending)
            result_metadata: Metadata for each result (trust, timestamps, fields, etc.)
            result_index: Position of this result in the ranked list
            retrieval_mode: "LIGHTWEIGHT" / "LOCAL" / "ONLINE"
            index_last_updated: When the index was last updated (for freshness)

        Returns:
            SignalVector with all v1 mandatory signals populated
        """
        result_count = len(similarity_scores)
        current_metadata = result_metadata[result_index] if result_metadata else {}

        # === TIER 1: RETRIEVAL GEOMETRY ===

        top_score = similarity_scores[0] if similarity_scores else 0.5

        # Top-k mean (use top-3 or all if fewer)
        k = min(3, len(similarity_scores))
        top_k_mean = float(np.mean(similarity_scores[:k])) if similarity_scores else 0.5

        # Score margin (gap between #1 and #2)
        if len(similarity_scores) >= 2:
            score_margin = similarity_scores[0] - similarity_scores[1]
        else:
            score_margin = 0.0  # Only one result = no margin

        # === TIER 2: RESULT QUALITY ===

        # Source trust from metadata
        trust_str = current_metadata.get('source_trust', 'unknown')
        try:
            source_trust = SourceTrust(trust_str.lower())
        except ValueError:
            source_trust = SourceTrust.UNKNOWN

        # Structural completeness: check required fields
        required_fields = ['content', 'created_at', 'user_id']
        fields_present = sum(1 for f in required_fields if current_metadata.get(f))
        structural_completeness = fields_present / len(required_fields)

        # Parse integrity: check for error flags
        has_parse_errors = current_metadata.get('parse_errors', False)
        is_truncated = current_metadata.get('truncated', False)
        parse_integrity = 0.0 if has_parse_errors else (0.5 if is_truncated else 1.0)

        # === TIER 3: QUERY SEMANTICS ===

        query_length = len(query)

        # Specificity score: count specific identifiers
        specificity_hits = 0
        total_patterns = len(self.specificity_patterns)
        for pattern in self.specificity_patterns.values():
            if pattern.search(query):
                specificity_hits += 1

        # Normalize to 0-1, bonus for multiple types
        specificity_score = min(1.0, specificity_hits / 3.0)  # Full credit at 3+ types

        # Ambiguity score: count vague markers
        query_lower = query.lower()
        ambiguity_hits = sum(1 for marker in self.ambiguity_markers if marker in query_lower)
        # Normalize: 0 = no ambiguity, 1 = highly ambiguous
        ambiguity_score = min(1.0, ambiguity_hits / 3.0)

        # === TIER 4: TEMPORAL ===

        # Recency score: age-weighted freshness
        result_created_at = current_metadata.get('created_at')
        if result_created_at:
            if isinstance(result_created_at, str):
                try:
                    result_created_at = datetime.fromisoformat(result_created_at.replace('Z', '+00:00'))
                except Exception:
                    result_created_at = None

            if result_created_at:
                age_days = (datetime.utcnow() - result_created_at).total_seconds() / 86400
                # Exponential decay: half-life of 90 days for default domain
                half_life = 90.0
                recency_score = float(np.exp(-age_days * np.log(2) / half_life))
            else:
                recency_score = 0.5  # Unknown age
        else:
            recency_score = 0.5  # No timestamp available

        # Index freshness: how stale is the index itself
        if index_last_updated:
            index_age_hours = (datetime.utcnow() - index_last_updated).total_seconds() / 3600
            # Decay over 24 hours
            index_freshness = float(max(0.0, 1.0 - (index_age_hours / 24.0)))
        else:
            index_freshness = 1.0  # Assume fresh if unknown

        # === TIER 5: SYSTEM HEALTH ===

        # Retrieval path mode
        mode_map = {
            'LIGHTWEIGHT': 'fallback',  # Degraded accuracy
            'LOCAL': 'normal',
            'ONLINE': 'normal'
        }
        retrieval_path = mode_map.get(retrieval_mode, 'degraded')

        # === PHASE 2 SIGNALS ===

        # Tier 1 Phase 2: Score distribution entropy
        score_entropy = self._calculate_score_entropy(similarity_scores)

        # Tier 2 Phase 2: Redundancy, coherence, diversity
        redundancy_score = self._calculate_redundancy(result_metadata)
        semantic_coherence = self._calculate_semantic_coherence(similarity_scores, result_count)
        diversity_score = self._calculate_diversity(similarity_scores, result_count)

        # Tier 3 Phase 2: Task type and constraint richness
        task_type = self._classify_task_type(query)
        # Note: filters would come from search context, not available here
        # We pass None and rely on query-only detection
        constraint_richness = self._calculate_constraint_richness(query, filters=None)

        # Tier 4 Phase 2: Temporal alignment
        temporal_alignment = self._calculate_temporal_alignment(query, result_created_at)

        # === PHASE 3 SIGNALS (LOCAL-FIRST) ===

        # Tier 2 Phase 3: Language quality
        content = current_metadata.get('content', '')
        language_quality = self._calculate_language_quality(content)

        # Tier 5 Phase 3: Latency and error indicators
        # These will be properly instrumented in memory_service
        # For now, use defaults (assumes healthy system)
        latency_indicators = self._calculate_latency_indicators(
            retrieval_start_time=None,  # TODO: pass from memory_service
            retrieval_mode=retrieval_mode
        )
        error_log_presence = self._calculate_error_log_presence(recent_error_count=0)

        # === TIER 7: META-SIGNALS ===

        # Signal completeness: how many signals have real data vs defaults
        signals_available = 0
        total_signals = 24  # 14 mandatory + 7 Phase 2 + 3 Phase 3

        # Check which signals have non-default values
        # v1 mandatory signals (14)
        if top_score != 0.5: signals_available += 1
        if top_k_mean != 0.5: signals_available += 1
        if score_margin > 0: signals_available += 1
        if source_trust != SourceTrust.UNKNOWN: signals_available += 1
        if structural_completeness > 0: signals_available += 1
        if parse_integrity < 1.0 or not has_parse_errors: signals_available += 1
        if query_length > 0: signals_available += 1
        if specificity_score > 0: signals_available += 1
        if ambiguity_score > 0 or len(query) > 0: signals_available += 1
        if result_created_at is not None: signals_available += 1
        if index_last_updated is not None: signals_available += 1
        signals_available += 1  # retrieval_path always available
        signals_available += 1  # result_count always available
        signals_available += 1  # result_index always available

        # Phase 2 signals (7) - these are always calculated
        signals_available += 1  # score_entropy always available
        signals_available += 1  # redundancy_score always available
        signals_available += 1  # semantic_coherence always available
        signals_available += 1  # diversity_score always available
        signals_available += 1  # task_type always available
        signals_available += 1  # constraint_richness always available
        signals_available += 1  # temporal_alignment always available

        # Phase 3 signals (3) - local-first, always calculated
        signals_available += 1  # language_quality always available
        signals_available += 1  # latency_indicators always available
        signals_available += 1  # error_log_presence always available

        signal_completeness = signals_available / total_signals

        # === CONSTRUCT SIGNAL VECTOR ===

        vector = SignalVector(
            # Tier 1: Mandatory
            top_score=top_score,
            top_k_mean=top_k_mean,
            score_margin=score_margin,
            # Tier 1: Phase 2
            score_entropy=score_entropy,
            # Tier 2: Mandatory
            source_trust=source_trust,
            structural_completeness=structural_completeness,
            parse_integrity=parse_integrity,
            # Tier 2: Phase 2
            redundancy_score=redundancy_score,
            semantic_coherence=semantic_coherence,
            diversity_score=diversity_score,
            # Tier 2: Phase 3
            language_quality=language_quality,
            # Tier 3: Mandatory
            query_length=query_length,
            specificity_score=specificity_score,
            ambiguity_score=ambiguity_score,
            # Tier 3: Phase 2
            task_type=task_type,
            constraint_richness=constraint_richness,
            # Tier 4: Mandatory
            recency_score=recency_score,
            index_freshness=index_freshness,
            # Tier 4: Phase 2
            temporal_alignment=temporal_alignment,
            # Tier 5: Mandatory
            retrieval_path=retrieval_path,
            # Tier 5: Phase 3
            latency_indicators=latency_indicators,
            error_log_presence=error_log_presence,
            # Tier 7
            signal_completeness=signal_completeness,
            # Context
            result_count=result_count,
            result_index=result_index,
            timestamp=datetime.utcnow()
        )

        logger.debug(
            "Signals extracted (v2+Phase2+Phase3)",
            result_index=result_index,
            # v1 signals
            top_score=f"{top_score:.3f}",
            margin=f"{score_margin:.3f}",
            specificity=f"{specificity_score:.3f}",
            ambiguity=f"{ambiguity_score:.3f}",
            recency=f"{recency_score:.3f}",
            # Phase 2 signals
            entropy=f"{score_entropy:.3f}",
            task_type=task_type.value if task_type else "unknown",
            coherence=f"{semantic_coherence:.3f}",
            diversity=f"{diversity_score:.3f}",
            temporal_align=f"{temporal_alignment:.3f}",
            # Phase 3 signals (local-first)
            language_quality=f"{language_quality:.3f}",
            latency=f"{latency_indicators:.3f}",
            errors=f"{error_log_presence:.3f}",
            signal_completeness=f"{signal_completeness:.2f}"
        )

        return vector


class ConfidenceAggregator:
    """
    Aggregates signal vector into final confidence estimate.

    Implements query-type-aware weighting, nonlinear behavior,
    and meta-confidence estimation.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        low_similarity_threshold: float = 0.5
    ):
        """
        Initialize confidence aggregator.

        Args:
            confidence_threshold: Threshold for should_review flag
            low_similarity_threshold: Below this, flag low_similarity uncertainty
        """
        self.confidence_threshold = confidence_threshold
        self.low_similarity_threshold = low_similarity_threshold

        logger.info(
            "Confidence aggregator initialized",
            threshold=confidence_threshold,
            low_similarity_threshold=low_similarity_threshold
        )

    def aggregate(
        self,
        signals: SignalVector,
        query_type: QueryType = QueryType.UNKNOWN
    ) -> Dict[str, Any]:
        """
        Aggregate signal vector into confidence estimate.

        Returns dict with:
        - confidence: float (0-1)
        - meta_confidence: float (0-1) - confidence about the confidence
        - uncertainty_sources: List[str]
        - should_review: bool
        - signals: SignalVector (for debugging/logging)

        Args:
            signals: Extracted signal vector
            query_type: Query type for adaptive weighting (Phase 2 feature)

        Returns:
            Confidence estimate dictionary
        """
        # === NONLINEAR BEHAVIOR: CATASTROPHIC FAILURES ===

        # If critical signals are terrible, clamp to very low
        if signals.top_score < 0.3 or signals.parse_integrity < 0.5:
            return self._catastrophic_low_confidence(signals, "critical_signal_failure")

        # If system is severely degraded
        if signals.retrieval_path == "degraded" and signals.signal_completeness < 0.5:
            return self._catastrophic_low_confidence(signals, "system_degradation")

        # === QUERY-TYPE-AWARE WEIGHTING ===

        # v2: Query-type-dependent weighting (uses Phase 2 task classification)
        weights = self._get_weights_v2(query_type, signals.task_type)

        # === WEIGHTED AGGREGATION ===

        # Tier 1: Geometry (25% weight, reduced from 30% to make room for Phase 2)
        # v1 signals
        geometry_base = (
            weights['geometry']['magnitude'] * signals.top_score +
            weights['geometry']['mean'] * signals.top_k_mean +
            weights['geometry']['margin'] * min(1.0, signals.score_margin / 0.2)  # Normalize margin
        )
        # Phase 2: entropy bonus (low entropy = sharp distribution = confidence boost)
        entropy_score = 1.0 - (signals.score_entropy if signals.score_entropy is not None else 0.5)
        geometry_score = geometry_base * (0.8 + 0.2 * entropy_score)  # 20% bonus for low entropy

        # Tier 2: Quality (22% weight, includes Phase 2 & Phase 3)
        trust_score = {
            SourceTrust.HIGH: 1.0,
            SourceTrust.MEDIUM: 0.7,
            SourceTrust.LOW: 0.4,
            SourceTrust.UNKNOWN: 0.5
        }[signals.source_trust]

        quality_base = (
            weights['quality']['trust'] * trust_score +
            weights['quality']['completeness'] * signals.structural_completeness +
            weights['quality']['integrity'] * signals.parse_integrity
        )
        # Phase 2: redundancy, coherence, diversity
        redundancy = signals.redundancy_score if signals.redundancy_score is not None else 0.5
        coherence = signals.semantic_coherence if signals.semantic_coherence is not None else 0.5
        diversity = signals.diversity_score if signals.diversity_score is not None else 0.5

        # Balance: want high coherence AND good diversity (not too similar, not too scattered)
        phase2_quality = (0.4 * redundancy + 0.4 * coherence + 0.2 * diversity)

        # Phase 3: language quality (local-first)
        lang_quality = signals.language_quality if signals.language_quality is not None else 0.7

        # Integrate all quality signals
        quality_score = 0.60 * quality_base + 0.25 * phase2_quality + 0.15 * lang_quality

        # Tier 3: Query Clarity (20% weight, includes Phase 2)
        # Length normalization: optimal around 50-200 chars
        length_score = 1.0 - abs(signals.query_length - 100) / 200.0
        length_score = max(0.0, min(1.0, length_score))

        clarity_base = (
            weights['clarity']['length'] * length_score +
            weights['clarity']['specificity'] * signals.specificity_score +
            weights['clarity']['anti_ambiguity'] * (1.0 - signals.ambiguity_score)
        )
        # Phase 2: constraint richness boosts confidence
        constraints = signals.constraint_richness if signals.constraint_richness is not None else 0.0
        clarity_score = clarity_base * (0.8 + 0.2 * constraints)  # Up to 20% bonus for constraints

        # Tier 4: Temporal (18% weight, includes Phase 2)
        temporal_base = (
            weights['temporal']['recency'] * signals.recency_score +
            weights['temporal']['freshness'] * signals.index_freshness
        )
        # Phase 2: temporal alignment
        alignment = signals.temporal_alignment if signals.temporal_alignment is not None else 0.8
        temporal_score = 0.7 * temporal_base + 0.3 * alignment

        # Tier 5: System Health (15% weight, includes Phase 3)
        path_score = 1.0 if signals.retrieval_path == "normal" else 0.7

        # Phase 3: latency and error indicators (local-first)
        latency = signals.latency_indicators if signals.latency_indicators is not None else 0.85
        errors = signals.error_log_presence if signals.error_log_presence is not None else 1.0

        # Integrate system health signals
        system_score = 0.40 * path_score + 0.35 * latency + 0.25 * errors

        # === FINAL CONFIDENCE ===

        base_confidence = (
            0.25 * geometry_score +
            0.22 * quality_score +
            0.20 * clarity_score +
            0.18 * temporal_score +
            0.15 * system_score
        )

        # Result rank penalty (results further down are less confident)
        rank_penalty = 1.0 - (signals.result_index * 0.05)  # 5% per rank
        rank_penalty = max(0.5, rank_penalty)  # Floor at 50%

        final_confidence = base_confidence * rank_penalty
        final_confidence = float(np.clip(final_confidence, 0.0, 1.0))

        # === META-CONFIDENCE ===

        # How reliable is this estimate itself?
        meta_confidence = self._calculate_meta_confidence(signals)

        # === UNCERTAINTY SOURCES ===

        uncertainty_sources = self._identify_uncertainty_sources(signals)

        # === SHOULD REVIEW ===

        should_review = (
            final_confidence < self.confidence_threshold or
            signals.top_score < self.low_similarity_threshold or
            meta_confidence < 0.5
        )

        logger.debug(
            "Confidence aggregated",
            confidence=f"{final_confidence:.3f}",
            meta_confidence=f"{meta_confidence:.3f}",
            should_review=should_review,
            uncertainty_count=len(uncertainty_sources)
        )

        return {
            'confidence': final_confidence,
            'meta_confidence': meta_confidence,
            'similarity_score': signals.top_score,
            'uncertainty_sources': uncertainty_sources,
            'should_review': should_review,
            'signals': signals  # For debugging/analysis
        }

    def _get_weights_v1(self, query_type: QueryType) -> Dict[str, Dict[str, float]]:
        """
        Get signal weights for v1 (simple, not query-type-dependent yet).

        Deprecated: Use _get_weights_v2 for Phase 2.
        """
        return {
            'geometry': {
                'magnitude': 0.50,  # Top score most important
                'mean': 0.30,       # Overall cluster quality
                'margin': 0.20      # Ambiguity (sharp vs flat)
            },
            'quality': {
                'trust': 0.40,
                'completeness': 0.35,
                'integrity': 0.25
            },
            'clarity': {
                'length': 0.30,
                'specificity': 0.45,
                'anti_ambiguity': 0.25
            },
            'temporal': {
                'recency': 0.70,
                'freshness': 0.30
            }
        }

    def _get_weights_v2(
        self,
        fallback_query_type: QueryType,
        task_type: Optional[QueryType]
    ) -> Dict[str, Dict[str, float]]:
        """
        Get signal weights for v2 (Phase 2: task-type-dependent).

        Adjusts weights based on detected task type:
        - FACT_LOOKUP: Favor similarity + margin (sharp answers)
        - EXPLORATORY: Favor diversity + temporal (breadth of coverage)
        - MULTI_STEP: Favor completeness + coherence (comprehensive answers)
        """
        detected_type = task_type if task_type else fallback_query_type

        if detected_type == QueryType.FACT_LOOKUP:
            # Fact queries: high similarity + sharp margin most important
            return {
                'geometry': {
                    'magnitude': 0.60,  # Top score critical
                    'mean': 0.25,
                    'margin': 0.15      # Sharp answer
                },
                'quality': {
                    'trust': 0.50,      # Trust matters for facts
                    'completeness': 0.30,
                    'integrity': 0.20
                },
                'clarity': {
                    'length': 0.20,
                    'specificity': 0.60,  # Specific questions
                    'anti_ambiguity': 0.20
                },
                'temporal': {
                    'recency': 0.60,    # Facts may be time-sensitive
                    'freshness': 0.40
                }
            }

        elif detected_type == QueryType.EXPLORATORY:
            # Exploratory: diversity + coverage + temporal breadth
            return {
                'geometry': {
                    'magnitude': 0.40,
                    'mean': 0.40,       # Cluster quality
                    'margin': 0.20
                },
                'quality': {
                    'trust': 0.35,
                    'completeness': 0.35,
                    'integrity': 0.30
                },
                'clarity': {
                    'length': 0.40,     # Longer queries expected
                    'specificity': 0.30,
                    'anti_ambiguity': 0.30
                },
                'temporal': {
                    'recency': 0.50,
                    'freshness': 0.50
                }
            }

        elif detected_type == QueryType.MULTI_STEP:
            # Multi-step: completeness + coherence + trust
            return {
                'geometry': {
                    'magnitude': 0.45,
                    'mean': 0.35,
                    'margin': 0.20
                },
                'quality': {
                    'trust': 0.40,
                    'completeness': 0.45,  # Need complete info
                    'integrity': 0.15
                },
                'clarity': {
                    'length': 0.35,
                    'specificity': 0.40,
                    'anti_ambiguity': 0.25
                },
                'temporal': {
                    'recency': 0.65,
                    'freshness': 0.35
                }
            }

        else:
            # UNKNOWN: balanced weights (same as v1)
            return self._get_weights_v1(fallback_query_type)

    def _calculate_meta_confidence(self, signals: SignalVector) -> float:
        """
        Calculate confidence about the confidence estimate itself.

        High when:
        - Signal completeness is high
        - Multiple strong signals agree
        - No catastrophic missing data
        """
        # Base on signal completeness
        base = signals.signal_completeness

        # Boost if we have high-quality core signals
        core_quality = (
            (1.0 if signals.top_score > 0.5 else 0.5) *
            (1.0 if signals.source_trust != SourceTrust.UNKNOWN else 0.7) *
            (1.0 if signals.recency_score > 0.5 else 0.8)
        )

        meta_confidence = base * core_quality

        return float(np.clip(meta_confidence, 0.0, 1.0))

    def _identify_uncertainty_sources(self, signals: SignalVector) -> List[str]:
        """Identify specific sources of uncertainty based on signal thresholds (v1 + Phase 2)."""
        sources = []

        # === V1 MANDATORY SIGNALS ===

        # Tier 1: Geometry
        if signals.top_score < self.low_similarity_threshold:
            sources.append("low_similarity")
        if signals.score_margin < 0.05:
            sources.append("ambiguous_results")

        # Tier 2: Quality
        if signals.source_trust in [SourceTrust.LOW, SourceTrust.UNKNOWN]:
            sources.append("untrusted_source")
        if signals.structural_completeness < 0.8:
            sources.append("incomplete_data")
        if signals.parse_integrity < 1.0:
            sources.append("parse_errors")

        # Tier 3: Clarity
        if signals.query_length < 10:
            sources.append("short_query")
        if signals.specificity_score < 0.3:
            sources.append("vague_query")
        if signals.ambiguity_score > 0.5:
            sources.append("ambiguous_query")

        # Tier 4: Temporal
        if signals.recency_score < 0.5:
            sources.append("stale_result")
        if signals.index_freshness < 0.7:
            sources.append("stale_index")

        # Tier 5: System
        if signals.retrieval_path != "normal":
            sources.append("degraded_retrieval")

        # Context
        if signals.result_index > 5:
            sources.append("low_rank")
        if signals.result_count < 3:
            sources.append("insufficient_results")

        # === PHASE 2 SIGNALS ===

        # Tier 1 Phase 2: Entropy
        if signals.score_entropy is not None and signals.score_entropy > 0.7:
            sources.append("flat_distribution")  # High entropy = no clear winner

        # Tier 2 Phase 2: Redundancy, Coherence, Diversity
        if signals.redundancy_score is not None and signals.redundancy_score < 0.3:
            sources.append("no_corroboration")  # No independent source agreement
        if signals.semantic_coherence is not None and signals.semantic_coherence < 0.5:
            sources.append("scattered_results")  # Results don't form coherent cluster
        if signals.diversity_score is not None:
            if signals.diversity_score < 0.3:
                sources.append("narrow_coverage")  # Too similar, no breadth
            elif signals.diversity_score > 0.8:
                sources.append("excessive_diversity")  # Too scattered

        # Tier 3 Phase 2: Task Type, Constraints
        if signals.task_type == QueryType.UNKNOWN:
            sources.append("unclear_intent")  # Can't determine what user wants
        if signals.constraint_richness is not None and signals.constraint_richness < 0.2:
            sources.append("underspecified_query")  # Too broad, needs constraints

        # Tier 4 Phase 2: Temporal Alignment
        if signals.temporal_alignment is not None and signals.temporal_alignment < 0.4:
            sources.append("temporal_mismatch")  # Query wants recent, got old (or vice versa)

        # === PHASE 3 SIGNALS (LOCAL-FIRST) ===

        # Tier 2 Phase 3: Language Quality
        if signals.language_quality is not None and signals.language_quality < 0.5:
            sources.append("poor_text_quality")  # Noisy, malformed, or low-quality content

        # Tier 5 Phase 3: System Health
        if signals.latency_indicators is not None and signals.latency_indicators < 0.6:
            sources.append("slow_retrieval")  # System performance issues
        if signals.error_log_presence is not None and signals.error_log_presence < 0.8:
            sources.append("recent_errors")  # System instability detected

        # Tier 7: Meta
        if signals.signal_completeness < 0.7:
            sources.append("missing_signals")

        return sources

    def _catastrophic_low_confidence(
        self,
        signals: SignalVector,
        reason: str
    ) -> Dict[str, Any]:
        """Return clamped low confidence for catastrophic failures."""
        return {
            'confidence': 0.2,
            'meta_confidence': 0.3,
            'similarity_score': signals.top_score,
            'uncertainty_sources': [reason, "catastrophic_failure"],
            'should_review': True,
            'signals': signals
        }
