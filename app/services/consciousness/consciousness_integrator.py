"""
Consciousness Integrator
Phase 8.3: Unified consciousness-inspired processing

Integrates all consciousness components:
- Global Workspace (limited capacity awareness)
- Attention Director (selective focus)
- Metacognitive Monitor (self-monitoring)

Creates a unified "conscious" processing pipeline that combines
these cognitive architecture patterns.

Based on:
- Global Workspace Theory (Baars, 1988)
- Integrated Information Theory (Tononi, 2004)
- Attention Schema Theory (Graziano, 2013)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import asyncio
import structlog

from app.services.consciousness.global_workspace import (
    GlobalWorkspace,
    WorkspaceItem,
    ItemPriority
)
from app.services.consciousness.attention_director import (
    AttentionDirector,
    FocusStrategy
)
from app.services.consciousness.metacognition import (
    MetacognitiveMonitor,
    MetacognitiveState
)

logger = structlog.get_logger()


class ProcessingMode(Enum):
    """Processing modes for consciousness integrator"""
    AUTOMATIC = "automatic"      # Fast, unconscious, habitual
    CONTROLLED = "controlled"    # Slow, conscious, deliberate
    HYBRID = "hybrid"           # Mix of automatic and controlled


@dataclass
class ConsciousDecision:
    """
    A decision made through conscious processing.

    Represents the output of the full consciousness pipeline:
    workspace → attention → metacognition → decision.
    """
    decision_id: str
    decision_type: str

    # The actual decision/action
    action: str
    parameters: Dict[str, Any]

    # Confidence and metacognition
    confidence: float
    should_defer: bool  # Defer to human/oracle?

    # Processing details
    processing_mode: ProcessingMode
    workspace_items_considered: List[str]
    attended_items: List[str]
    metacognitive_state: str

    # Timing
    processing_time_ms: float
    timestamp: datetime


class ConsciousnessIntegrator:
    """
    Consciousness Integrator for MEMSHADOW.

    Integrates multiple consciousness-inspired components into a
    unified cognitive architecture:

        1. Global Workspace: Limited-capacity conscious awareness
        2. Attention Director: Selective focus on relevant information
        3. Metacognitive Monitor: Self-monitoring and confidence

    Processing Pipeline:
        Input → Workspace Competition → Attention Selection →
        Metacognitive Monitoring → Conscious Decision

    Enables:
        - Focused processing on most relevant information
        - Self-awareness of confidence and competence
        - Automatic vs controlled processing modes
        - Human-in-the-loop for uncertain decisions

    Example:
        integrator = ConsciousnessIntegrator()
        await integrator.start()

        # Process information consciously
        decision = await integrator.process_consciously(
            input_items=[item1, item2, item3],
            goal_context={"task": "detect vulnerabilities"},
            mode=ProcessingMode.CONTROLLED
        )

        # Check if we're confident
        if decision.should_defer:
            # Low confidence - get human input
            human_decision = await request_human_review(decision)
        else:
            # High confidence - proceed autonomously
            execute_decision(decision)
    """

    def __init__(
        self,
        workspace_capacity: int = 7,
        num_attention_heads: int = 8,
        enable_metacognition: bool = True,
        default_mode: ProcessingMode = ProcessingMode.HYBRID
    ):
        """
        Initialize consciousness integrator.

        Args:
            workspace_capacity: Global workspace capacity (7±2)
            num_attention_heads: Number of attention heads
            enable_metacognition: Enable metacognitive monitoring
            default_mode: Default processing mode
        """
        self.workspace_capacity = workspace_capacity
        self.num_attention_heads = num_attention_heads
        self.enable_metacognition = enable_metacognition
        self.default_mode = default_mode

        # Components
        self.workspace = GlobalWorkspace(capacity=workspace_capacity)
        self.attention = AttentionDirector(num_heads=num_attention_heads)
        self.metacognition = MetacognitiveMonitor() if enable_metacognition else None

        # Processing statistics
        self.decisions_made = 0
        self.autonomous_decisions = 0
        self.deferred_decisions = 0
        self.avg_processing_time_ms = 0.0

        logger.info(
            "Consciousness integrator initialized",
            workspace_capacity=workspace_capacity,
            attention_heads=num_attention_heads,
            metacognition=enable_metacognition
        )

    async def start(self):
        """Start all consciousness components"""
        logger.info("Starting consciousness integrator")

        await self.workspace.start()

        # Subscribe attention director to workspace broadcasts
        self.workspace.subscribe_module("attention_director")
        self.workspace.subscribe_module("metacognition")

    async def stop(self):
        """Stop all components"""
        logger.info("Stopping consciousness integrator")

        await self.workspace.stop()

    async def process_consciously(
        self,
        input_items: List[Dict[str, Any]],
        goal_context: Dict[str, Any],
        mode: Optional[ProcessingMode] = None
    ) -> ConsciousDecision:
        """
        Process information through full consciousness pipeline.

        Args:
            input_items: Items to process (must have 'id', 'content', 'priority')
            goal_context: Context about current goals/task
            mode: Processing mode (default to configured default)

        Returns:
            Conscious decision with confidence estimate
        """
        start_time = datetime.utcnow()
        mode = mode or self.default_mode

        logger.debug(
            "Conscious processing started",
            items_count=len(input_items),
            mode=mode.value
        )

        # Step 1: Compete for workspace access
        workspace_items = await self._compete_for_workspace(input_items)

        # Step 2: Apply attention to workspace contents
        attention_result = await self.attention.attend(
            query_context=goal_context,
            items=workspace_items,
            strategy=self._select_attention_strategy(mode)
        )

        # Step 3: Make decision based on attended items
        decision_output = await self._make_decision(
            attended_items=attention_result.attended_items,
            workspace_items=workspace_items,
            goal_context=goal_context,
            mode=mode
        )

        # Step 4: Metacognitive monitoring
        confidence_estimate = None
        if self.metacognition:
            confidence_estimate = await self.metacognition.estimate_confidence(
                decision_id=decision_output['id'],
                decision_output=decision_output,
                context=goal_context
            )

        # Create conscious decision
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        decision = ConsciousDecision(
            decision_id=decision_output['id'],
            decision_type=decision_output.get('type', 'unknown'),
            action=decision_output.get('action', 'none'),
            parameters=decision_output.get('parameters', {}),
            confidence=confidence_estimate.confidence if confidence_estimate else 0.8,
            should_defer=confidence_estimate.should_defer if confidence_estimate else False,
            processing_mode=mode,
            workspace_items_considered=[i['id'] for i in workspace_items],
            attended_items=attention_result.attended_items,
            metacognitive_state=self.metacognition.current_state.value if self.metacognition else "n/a",
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow()
        )

        # Update statistics
        self.decisions_made += 1
        if decision.should_defer:
            self.deferred_decisions += 1
        else:
            self.autonomous_decisions += 1

        # Update avg processing time
        self.avg_processing_time_ms = (
            (self.avg_processing_time_ms * (self.decisions_made - 1) + processing_time)
            / self.decisions_made
        )

        logger.info(
            "Conscious decision made",
            decision_id=decision.decision_id,
            action=decision.action,
            confidence=decision.confidence,
            should_defer=decision.should_defer,
            processing_time_ms=processing_time
        )

        return decision

    async def process_automatically(
        self,
        input_item: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process information automatically (fast, unconscious).

        Bypasses conscious processing for routine/habitual actions.

        Args:
            input_item: Item to process

        Returns:
            Decision output (simplified)
        """
        # Fast path - no workspace/attention/metacognition
        decision = {
            'id': f"auto_{self.decisions_made}",
            'type': 'automatic',
            'action': 'process',
            'parameters': input_item,
            'confidence': 0.9  # High confidence for automatic processing
        }

        self.decisions_made += 1
        self.autonomous_decisions += 1

        return decision

    async def adapt_processing_mode(
        self,
        current_performance: Dict[str, float]
    ):
        """
        Adapt processing mode based on performance.

        Switches between automatic and controlled based on success rate.

        Args:
            current_performance: Dict with 'accuracy', 'speed', etc.
        """
        accuracy = current_performance.get('accuracy', 0.5)
        speed = current_performance.get('speed', 0.5)

        # High accuracy + high speed → automatic
        if accuracy > 0.9 and speed > 0.8:
            new_mode = ProcessingMode.AUTOMATIC
            reason = "High performance - switching to automatic"

        # Low accuracy → controlled
        elif accuracy < 0.6:
            new_mode = ProcessingMode.CONTROLLED
            reason = "Low accuracy - switching to controlled"

        # Medium → hybrid
        else:
            new_mode = ProcessingMode.HYBRID
            reason = "Medium performance - using hybrid"

        if new_mode != self.default_mode:
            old_mode = self.default_mode
            self.default_mode = new_mode

            logger.info(
                "Processing mode adapted",
                from_mode=old_mode.value,
                to_mode=new_mode.value,
                reason=reason
            )

    async def get_consciousness_state(self) -> Dict[str, Any]:
        """Get current state of consciousness system"""
        workspace_state = await self.workspace.get_state()
        attention_stats = await self.attention.get_focus_stats()
        metacog_state = await self.metacognition.get_state() if self.metacognition else {}

        return {
            "processing_mode": self.default_mode.value,
            "decisions_made": self.decisions_made,
            "autonomous_decisions": self.autonomous_decisions,
            "deferred_decisions": self.deferred_decisions,
            "defer_rate": self.deferred_decisions / max(1, self.decisions_made),
            "avg_processing_time_ms": self.avg_processing_time_ms,

            # Workspace
            "workspace": {
                "capacity": workspace_state.capacity,
                "current_items": workspace_state.current_items,
                "utilization_percent": workspace_state.utilization_percent,
                "total_broadcasts": workspace_state.total_broadcasts
            },

            # Attention
            "attention": attention_stats,

            # Metacognition
            "metacognition": metacog_state
        }

    # Private methods

    async def _compete_for_workspace(
        self,
        input_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Items compete for limited workspace capacity.

        Args:
            input_items: Candidate items

        Returns:
            Items that won competition (now in workspace)
        """
        # Convert to WorkspaceItems
        workspace_items = []
        for item in input_items:
            workspace_item = WorkspaceItem(
                item_id=item['id'],
                content=item.get('content', {}),
                source_module=item.get('source', 'unknown'),
                salience=item.get('salience', 0.5),
                relevance=item.get('relevance', 0.5),
                novelty=item.get('novelty', 0.5),
                priority=self._map_priority(item.get('priority', 'normal'))
            )

            # Try to add to workspace
            added = await self.workspace.add_item(workspace_item)

            if added:
                workspace_items.append(item)

        return workspace_items

    def _map_priority(self, priority_str: str) -> ItemPriority:
        """Map string priority to ItemPriority enum"""
        mapping = {
            'critical': ItemPriority.CRITICAL,
            'high': ItemPriority.HIGH,
            'normal': ItemPriority.NORMAL,
            'low': ItemPriority.LOW,
            'minimal': ItemPriority.MINIMAL
        }
        return mapping.get(priority_str.lower(), ItemPriority.NORMAL)

    def _select_attention_strategy(self, mode: ProcessingMode) -> FocusStrategy:
        """Select attention strategy based on processing mode"""
        if mode == ProcessingMode.AUTOMATIC:
            return FocusStrategy.HABITUAL

        elif mode == ProcessingMode.CONTROLLED:
            return FocusStrategy.TOP_DOWN

        else:  # HYBRID
            return FocusStrategy.BALANCED

    async def _make_decision(
        self,
        attended_items: List[str],
        workspace_items: List[Dict[str, Any]],
        goal_context: Dict[str, Any],
        mode: ProcessingMode
    ) -> Dict[str, Any]:
        """
        Make decision based on attended items.

        Args:
            attended_items: Item IDs that got attention
            workspace_items: All items in workspace
            goal_context: Current goals
            mode: Processing mode

        Returns:
            Decision output
        """
        # Get top attended item
        if not attended_items:
            return {
                'id': f"decision_{self.decisions_made}",
                'type': 'no_decision',
                'action': 'none',
                'parameters': {},
                'reason': 'No items attended'
            }

        top_item_id = attended_items[0]

        # Find corresponding workspace item
        top_item = next(
            (item for item in workspace_items if item['id'] == top_item_id),
            None
        )

        if not top_item:
            return {
                'id': f"decision_{self.decisions_made}",
                'type': 'error',
                'action': 'none',
                'parameters': {}
            }

        # Make decision based on item content and goals
        # In production: would use actual decision-making logic
        decision = {
            'id': f"decision_{self.decisions_made}",
            'type': 'action',
            'action': self._determine_action(top_item, goal_context),
            'parameters': {
                'item_id': top_item_id,
                'item_content': top_item.get('content', {}),
                'mode': mode.value
            },
            'probabilities': [0.7, 0.2, 0.1],  # Mock probabilities for confidence estimation
            'supporting_evidence': len(attended_items),
            'contradicting_evidence': 0
        }

        return decision

    def _determine_action(
        self,
        item: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Determine action based on item and context"""
        # In production: sophisticated action selection
        # For now: simple mock
        item_type = item.get('content', {}).get('type', 'unknown')

        if 'security' in context.get('task', '').lower():
            return f"analyze_{item_type}_for_threats"
        else:
            return f"process_{item_type}"


# Global integrator instance
consciousness = ConsciousnessIntegrator()
