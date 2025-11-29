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
    """
    # Tier 1: Retrieval Geometry (Cheap, Online, Mandatory)
    top_score: float                    # Signal 1: Best match strength
    top_k_mean: float                   # Signal 2: Overall alignment
    score_margin: float                 # Signal 3: Gap between #1 and #2

    # Tier 2: Result Quality (Cheap, Online, Mandatory)
    source_trust: SourceTrust           # Signal 7: Trust classification
    structural_completeness: float      # Signal 9: Required fields present (0-1)
    parse_integrity: float              # Signal 10: Ingestion success (0-1)

    # Tier 3: Query Semantics (Cheap-Medium, Online, Mandatory)
    query_length: int                   # Signal 14: Lexical length
    specificity_score: float            # Signal 15: IDs, dates, entities (0-1)
    ambiguity_score: float              # Signal 18: Vague markers (0-1, higher=more ambiguous)

    # Tier 4: Temporal (Cheap, Online/Corpus, Mandatory)
    recency_score: float                # Signal 21: Age-weighted freshness (0-1)
    index_freshness: float              # Signal 24: Corpus update staleness (0-1)

    # Tier 5: System Health (Cheap, Online, Mandatory)
    retrieval_path: str                 # Signal 26: "normal" / "fallback" / "degraded"

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

        logger.info("Signal extractor initialized", mandatory_signals=14)

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

        # === TIER 7: META-SIGNALS ===

        # Signal completeness: how many signals have real data vs defaults
        signals_available = 0
        total_signals = 14

        # Check which signals have non-default values
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

        signal_completeness = signals_available / total_signals

        # === CONSTRUCT SIGNAL VECTOR ===

        vector = SignalVector(
            # Tier 1
            top_score=top_score,
            top_k_mean=top_k_mean,
            score_margin=score_margin,
            # Tier 2
            source_trust=source_trust,
            structural_completeness=structural_completeness,
            parse_integrity=parse_integrity,
            # Tier 3
            query_length=query_length,
            specificity_score=specificity_score,
            ambiguity_score=ambiguity_score,
            # Tier 4
            recency_score=recency_score,
            index_freshness=index_freshness,
            # Tier 5
            retrieval_path=retrieval_path,
            # Tier 7
            signal_completeness=signal_completeness,
            # Context
            result_count=result_count,
            result_index=result_index,
            timestamp=datetime.utcnow()
        )

        logger.debug(
            "Signals extracted",
            result_index=result_index,
            top_score=f"{top_score:.3f}",
            margin=f"{score_margin:.3f}",
            specificity=f"{specificity_score:.3f}",
            ambiguity=f"{ambiguity_score:.3f}",
            recency=f"{recency_score:.3f}",
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

        # v1: Simple weighting (Phase 2 will add query-type branching)
        weights = self._get_weights_v1(query_type)

        # === WEIGHTED AGGREGATION ===

        # Tier 1: Geometry (30% weight)
        geometry_score = (
            weights['geometry']['magnitude'] * signals.top_score +
            weights['geometry']['mean'] * signals.top_k_mean +
            weights['geometry']['margin'] * min(1.0, signals.score_margin / 0.2)  # Normalize margin
        )

        # Tier 2: Quality (25% weight)
        trust_score = {
            SourceTrust.HIGH: 1.0,
            SourceTrust.MEDIUM: 0.7,
            SourceTrust.LOW: 0.4,
            SourceTrust.UNKNOWN: 0.5
        }[signals.source_trust]

        quality_score = (
            weights['quality']['trust'] * trust_score +
            weights['quality']['completeness'] * signals.structural_completeness +
            weights['quality']['integrity'] * signals.parse_integrity
        )

        # Tier 3: Query Clarity (20% weight)
        # Length normalization: optimal around 50-200 chars
        length_score = 1.0 - abs(signals.query_length - 100) / 200.0
        length_score = max(0.0, min(1.0, length_score))

        clarity_score = (
            weights['clarity']['length'] * length_score +
            weights['clarity']['specificity'] * signals.specificity_score +
            weights['clarity']['anti_ambiguity'] * (1.0 - signals.ambiguity_score)
        )

        # Tier 4: Temporal (15% weight)
        temporal_score = (
            weights['temporal']['recency'] * signals.recency_score +
            weights['temporal']['freshness'] * signals.index_freshness
        )

        # Tier 5: System Health (10% weight)
        path_score = 1.0 if signals.retrieval_path == "normal" else 0.7
        system_score = path_score

        # === FINAL CONFIDENCE ===

        base_confidence = (
            0.30 * geometry_score +
            0.25 * quality_score +
            0.20 * clarity_score +
            0.15 * temporal_score +
            0.10 * system_score
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

        Phase 2 will branch on query_type.
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
        """Identify specific sources of uncertainty based on signal thresholds."""
        sources = []

        # Geometry
        if signals.top_score < self.low_similarity_threshold:
            sources.append("low_similarity")
        if signals.score_margin < 0.05:
            sources.append("ambiguous_results")

        # Quality
        if signals.source_trust in [SourceTrust.LOW, SourceTrust.UNKNOWN]:
            sources.append("untrusted_source")
        if signals.structural_completeness < 0.8:
            sources.append("incomplete_data")
        if signals.parse_integrity < 1.0:
            sources.append("parse_errors")

        # Clarity
        if signals.query_length < 10:
            sources.append("short_query")
        if signals.specificity_score < 0.3:
            sources.append("vague_query")
        if signals.ambiguity_score > 0.5:
            sources.append("ambiguous_query")

        # Temporal
        if signals.recency_score < 0.5:
            sources.append("stale_result")
        if signals.index_freshness < 0.7:
            sources.append("stale_index")

        # System
        if signals.retrieval_path != "normal":
            sources.append("degraded_retrieval")

        # Result position
        if signals.result_index > 5:
            sources.append("low_rank")

        # Few results
        if signals.result_count < 3:
            sources.append("insufficient_results")

        # Meta
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
