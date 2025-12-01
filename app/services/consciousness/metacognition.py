"""
Metacognition
Phase 8.3: Self-monitoring and confidence estimation

Implements metacognitive processes - "thinking about thinking":
- Confidence estimation
- Uncertainty quantification
- Self-monitoring
- Error detection
- Competence assessment

Based on:
- Flavell, J. H. (1979). Metacognition and cognitive monitoring
- Yeung & Summerfield (2012). Metacognition in human decision-making
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import structlog

logger = structlog.get_logger()


class UncertaintySource(Enum):
    """Sources of uncertainty in cognition"""
    AMBIGUOUS_INPUT = "ambiguous_input"      # Input is unclear
    INSUFFICIENT_DATA = "insufficient_data"  # Not enough information
    CONFLICTING_INFO = "conflicting_info"    # Contradictory signals
    NOVEL_SITUATION = "novel_situation"      # Never seen before
    LOW_CONFIDENCE = "low_confidence"        # Model unsure
    DISTRIBUTION_SHIFT = "distribution_shift" # Out of distribution


class MetacognitiveState(Enum):
    """Current metacognitive state"""
    CONFIDENT = "confident"          # High confidence in processing
    UNCERTAIN = "uncertain"          # Unsure about decisions
    MONITORING = "monitoring"        # Actively checking performance
    ERROR_DETECTED = "error_detected" # Detected a mistake
    LEARNING = "learning"            # Acquiring new knowledge


@dataclass
class ConfidenceEstimate:
    """
    Confidence estimate for a decision/prediction.

    Represents how certain the system is about its own outputs.
    """
    decision_id: str
    confidence: float  # 0.0 to 1.0

    # Uncertainty breakdown
    aleatoric_uncertainty: float  # Irreducible (data noise)
    epistemic_uncertainty: float  # Reducible (model uncertainty)

    # Sources
    uncertainty_sources: List[UncertaintySource]

    # Evidence
    supporting_evidence_count: int
    contradicting_evidence_count: int

    # Context
    timestamp: datetime = field(default_factory=datetime.utcnow)
    decision_context: Optional[Dict[str, Any]] = None

    @property
    def total_uncertainty(self) -> float:
        """Combined uncertainty"""
        return self.aleatoric_uncertainty + self.epistemic_uncertainty

    @property
    def should_defer(self) -> bool:
        """Should defer decision to human/oracle?"""
        return self.confidence < 0.5 or self.total_uncertainty > 0.7


@dataclass
class PerformanceMonitor:
    """Monitors performance on recent decisions"""
    window_size: int = 100
    recent_decisions: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_decision(self, decision_id: str, correct: bool, confidence: float):
        """Record a decision outcome"""
        self.recent_decisions.append({
            'id': decision_id,
            'correct': correct,
            'confidence': confidence,
            'timestamp': datetime.utcnow()
        })

    @property
    def accuracy(self) -> float:
        """Recent accuracy"""
        if not self.recent_decisions:
            return 0.0

        correct_count = sum(1 for d in self.recent_decisions if d['correct'])
        return correct_count / len(self.recent_decisions)

    @property
    def calibration_error(self) -> float:
        """
        Calibration error (confidence vs accuracy alignment).

        Lower is better. 0 = perfectly calibrated.
        """
        if not self.recent_decisions:
            return 0.0

        # Expected calibration error (ECE)
        # Compare confidence to actual accuracy in bins
        bins = 10
        bin_edges = np.linspace(0, 1, bins + 1)

        total_error = 0.0
        total_samples = 0

        for i in range(bins):
            # Get decisions in this confidence bin
            bin_decisions = [
                d for d in self.recent_decisions
                if bin_edges[i] <= d['confidence'] < bin_edges[i + 1]
            ]

            if not bin_decisions:
                continue

            # Average confidence in bin
            avg_confidence = sum(d['confidence'] for d in bin_decisions) / len(bin_decisions)

            # Accuracy in bin
            bin_accuracy = sum(1 for d in bin_decisions if d['correct']) / len(bin_decisions)

            # Calibration error for this bin
            total_error += len(bin_decisions) * abs(avg_confidence - bin_accuracy)
            total_samples += len(bin_decisions)

        return total_error / total_samples if total_samples > 0 else 0.0


class ConfidenceEstimator:
    """
    Estimates confidence in decisions and predictions.

    Uses multiple signals to assess confidence:
    - Model entropy/variance
    - Agreement between ensemble members
    - Distance from training distribution
    - Historical accuracy on similar inputs
    """

    def __init__(
        self,
        calibration_temperature: float = 1.0
    ):
        """
        Initialize confidence estimator.

        Args:
            calibration_temperature: Temperature for confidence calibration
        """
        self.calibration_temperature = calibration_temperature

        # Historical data for calibration
        self.decision_history: List[ConfidenceEstimate] = []

        logger.info("Confidence estimator initialized")

    def estimate_confidence(
        self,
        decision_id: str,
        model_output: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ConfidenceEstimate:
        """
        Estimate confidence for a decision.

        Args:
            decision_id: Decision identifier
            model_output: Output from the model
            context: Decision context

        Returns:
            Confidence estimate
        """
        # Extract signals
        entropy = self._compute_entropy(model_output)
        variance = self._compute_variance(model_output)
        agreement = self._compute_agreement(model_output)

        # Compute base confidence (inverse of entropy)
        base_confidence = 1.0 / (1.0 + entropy)

        # Apply temperature calibration
        calibrated_confidence = self._apply_calibration(base_confidence)

        # Decompose uncertainty
        aleatoric = variance / (1.0 + variance)  # Data uncertainty
        epistemic = entropy / (1.0 + entropy)    # Model uncertainty

        # Identify uncertainty sources
        sources = self._identify_uncertainty_sources(
            entropy, variance, agreement, context
        )

        # Count evidence
        supporting = model_output.get('supporting_evidence', 0)
        contradicting = model_output.get('contradicting_evidence', 0)

        estimate = ConfidenceEstimate(
            decision_id=decision_id,
            confidence=calibrated_confidence,
            aleatoric_uncertainty=aleatoric,
            epistemic_uncertainty=epistemic,
            uncertainty_sources=sources,
            supporting_evidence_count=supporting,
            contradicting_evidence_count=contradicting,
            decision_context=context
        )

        # Store for calibration
        self.decision_history.append(estimate)
        if len(self.decision_history) > 1000:
            self.decision_history.pop(0)

        logger.debug(
            "Confidence estimated",
            decision_id=decision_id,
            confidence=calibrated_confidence,
            uncertainty_sources=[s.value for s in sources]
        )

        return estimate

    def _compute_entropy(self, model_output: Dict[str, Any]) -> float:
        """Compute output entropy (uncertainty)"""
        # If model provides probabilities
        if 'probabilities' in model_output:
            probs = np.array(model_output['probabilities'])
            probs = probs + 1e-10  # Avoid log(0)
            entropy = -np.sum(probs * np.log(probs))
            return float(entropy)

        # Default: moderate entropy
        return 0.5

    def _compute_variance(self, model_output: Dict[str, Any]) -> float:
        """Compute output variance"""
        if 'variance' in model_output:
            return float(model_output['variance'])

        # Default
        return 0.3

    def _compute_agreement(self, model_output: Dict[str, Any]) -> float:
        """Compute agreement between ensemble members"""
        if 'ensemble_agreement' in model_output:
            return float(model_output['ensemble_agreement'])

        # Default: moderate agreement
        return 0.7

    def _apply_calibration(self, confidence: float) -> float:
        """Apply temperature calibration to confidence"""
        # Temperature scaling
        adjusted = confidence ** (1.0 / self.calibration_temperature)

        # Clip to [0, 1]
        return np.clip(adjusted, 0.0, 1.0)

    def _identify_uncertainty_sources(
        self,
        entropy: float,
        variance: float,
        agreement: float,
        context: Optional[Dict[str, Any]]
    ) -> List[UncertaintySource]:
        """Identify sources of uncertainty"""
        sources = []

        if entropy > 0.7:
            sources.append(UncertaintySource.LOW_CONFIDENCE)

        if variance > 0.6:
            sources.append(UncertaintySource.AMBIGUOUS_INPUT)

        if agreement < 0.5:
            sources.append(UncertaintySource.CONFLICTING_INFO)

        if context and context.get('is_novel', False):
            sources.append(UncertaintySource.NOVEL_SITUATION)

        if context and context.get('data_size', float('inf')) < 10:
            sources.append(UncertaintySource.INSUFFICIENT_DATA)

        return sources


class MetacognitiveMonitor:
    """
    Metacognitive Monitor for MEMSHADOW.

    Monitors the system's own cognitive processes, detects errors,
    estimates confidence, and triggers corrective actions when needed.

    Implements:
        - Confidence estimation
        - Performance monitoring
        - Error detection
        - Competence assessment
        - Self-correction triggers

    Example:
        monitor = MetacognitiveMonitor()

        # Make a decision
        decision = model.predict(input_data)

        # Estimate confidence
        confidence = await monitor.estimate_confidence(
            decision_id="dec_001",
            decision_output=decision,
            context={"domain": "security"}
        )

        # Check if we should defer
        if confidence.should_defer:
            request_human_review(decision)

        # Record outcome (for calibration)
        await monitor.record_outcome("dec_001", correct=True)
    """

    def __init__(
        self,
        enable_self_correction: bool = True,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize metacognitive monitor.

        Args:
            enable_self_correction: Trigger corrections on low confidence
            confidence_threshold: Minimum confidence for autonomous action
        """
        self.enable_self_correction = enable_self_correction
        self.confidence_threshold = confidence_threshold

        # Components
        self.confidence_estimator = ConfidenceEstimator()
        self.performance_monitor = PerformanceMonitor()

        # State
        self.current_state = MetacognitiveState.CONFIDENT
        self.state_history: List[Tuple[MetacognitiveState, datetime]] = []

        # Error detection
        self.errors_detected = 0
        self.corrections_triggered = 0

        logger.info(
            "Metacognitive monitor initialized",
            self_correction=enable_self_correction,
            confidence_threshold=confidence_threshold
        )

    async def estimate_confidence(
        self,
        decision_id: str,
        decision_output: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ConfidenceEstimate:
        """
        Estimate confidence for a decision.

        Args:
            decision_id: Decision identifier
            decision_output: Model output
            context: Decision context

        Returns:
            Confidence estimate
        """
        estimate = self.confidence_estimator.estimate_confidence(
            decision_id, decision_output, context
        )

        # Update metacognitive state based on confidence
        if estimate.confidence < 0.3:
            await self._transition_state(
                MetacognitiveState.ERROR_DETECTED,
                reason="Very low confidence"
            )

        elif estimate.confidence < self.confidence_threshold:
            await self._transition_state(
                MetacognitiveState.UNCERTAIN,
                reason="Below confidence threshold"
            )

        else:
            await self._transition_state(
                MetacognitiveState.CONFIDENT,
                reason="High confidence"
            )

        # Trigger self-correction if needed
        if self.enable_self_correction and estimate.should_defer:
            await self._trigger_correction(decision_id, estimate)

        return estimate

    async def record_outcome(
        self,
        decision_id: str,
        correct: bool,
        actual_outcome: Optional[Any] = None
    ):
        """
        Record decision outcome for calibration.

        Args:
            decision_id: Decision identifier
            correct: Whether decision was correct
            actual_outcome: The actual outcome (optional)
        """
        # Find corresponding confidence estimate
        estimate = next(
            (e for e in self.confidence_estimator.decision_history
             if e.decision_id == decision_id),
            None
        )

        if estimate:
            # Record in performance monitor
            self.performance_monitor.add_decision(
                decision_id, correct, estimate.confidence
            )

            logger.debug(
                "Outcome recorded",
                decision_id=decision_id,
                correct=correct,
                confidence=estimate.confidence
            )

            # Check for error
            if not correct and estimate.confidence > 0.7:
                # High confidence but wrong = error detection
                self.errors_detected += 1

                await self._transition_state(
                    MetacognitiveState.ERROR_DETECTED,
                    reason="High confidence error"
                )

    async def assess_competence(
        self,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assess competence in a domain.

        Args:
            domain: Optional domain to assess (None = overall)

        Returns:
            Competence assessment
        """
        # Get performance metrics
        accuracy = self.performance_monitor.accuracy
        calibration = self.performance_monitor.calibration_error

        # Determine competence level
        if accuracy > 0.9 and calibration < 0.1:
            competence_level = "expert"
        elif accuracy > 0.7 and calibration < 0.2:
            competence_level = "proficient"
        elif accuracy > 0.5:
            competence_level = "intermediate"
        else:
            competence_level = "novice"

        return {
            "domain": domain or "general",
            "competence_level": competence_level,
            "accuracy": accuracy,
            "calibration_error": calibration,
            "sample_size": len(self.performance_monitor.recent_decisions),
            "errors_detected": self.errors_detected,
            "corrections_triggered": self.corrections_triggered
        }

    async def get_state(self) -> Dict[str, Any]:
        """Get current metacognitive state"""
        return {
            "current_state": self.current_state.value,
            "confidence_threshold": self.confidence_threshold,
            "errors_detected": self.errors_detected,
            "corrections_triggered": self.corrections_triggered,
            "decisions_made": len(self.performance_monitor.recent_decisions),
            "recent_accuracy": self.performance_monitor.accuracy,
            "calibration_error": self.performance_monitor.calibration_error
        }

    # Private methods

    async def _transition_state(
        self,
        new_state: MetacognitiveState,
        reason: str
    ):
        """Transition to new metacognitive state"""
        if new_state != self.current_state:
            old_state = self.current_state
            self.current_state = new_state

            # Record transition
            self.state_history.append((new_state, datetime.utcnow()))
            if len(self.state_history) > 1000:
                self.state_history.pop(0)

            logger.info(
                "Metacognitive state transition",
                from_state=old_state.value,
                to_state=new_state.value,
                reason=reason
            )

    async def _trigger_correction(
        self,
        decision_id: str,
        estimate: ConfidenceEstimate
    ):
        """Trigger self-correction for low-confidence decision"""
        self.corrections_triggered += 1

        logger.warning(
            "Self-correction triggered",
            decision_id=decision_id,
            confidence=estimate.confidence,
            uncertainty_sources=[s.value for s in estimate.uncertainty_sources]
        )

        # In production: trigger specific corrective actions
        # - Request more data
        # - Consult alternative models
        # - Defer to human
        # - Gather more evidence
