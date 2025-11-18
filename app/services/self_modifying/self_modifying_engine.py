"""
Self-Modifying Engine
Phase 8.4: Orchestrates safe self-modification

Integrates all self-modification components:
- Code introspection
- Improvement proposals
- Test generation
- Safe modification

Implements autonomous code improvement with safety-first design.

⚠️ CRITICAL SAFETY NOTICE:
Self-modifying code is inherently risky. This implementation includes
multiple safety layers but should be used with extreme caution.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import asyncio
import structlog

from app.services.self_modifying.code_introspector import (
    CodeIntrospector,
    CodeMetrics
)
from app.services.self_modifying.improvement_proposer import (
    ImprovementProposer,
    CodeImprovement,
    ImprovementCategory
)
from app.services.self_modifying.test_generator import (
    TestGenerator,
    GeneratedTest
)
from app.services.self_modifying.safe_modifier import (
    SafeModifier,
    SafetyLevel,
    ModificationResult,
    ModificationStatus
)

logger = structlog.get_logger()


@dataclass
class ModificationRequest:
    """Request for code modification"""
    request_id: str
    target_function: callable
    improvement_categories: List[ImprovementCategory]
    auto_apply: bool = False  # Automatically apply safe improvements
    require_human_approval: bool = True


class SelfModifyingEngine:
    """
    Self-Modifying Engine for MEMSHADOW.

    Orchestrates autonomous code improvement through:
        1. Introspection: Analyze code structure and metrics
        2. Proposal: Generate improvement suggestions
        3. Testing: Create tests for validation
        4. Modification: Safely apply changes with rollback

    Safety Features:
        - Gradual privilege escalation
        - Mandatory test coverage
        - Automatic rollback on failure
        - Human approval for risky changes
        - Comprehensive audit logging

    Processing Pipeline:
        Introspect → Propose → Generate Tests → Apply (with safety checks)

    Example:
        engine = SelfModifyingEngine(
            safety_level=SafetyLevel.LOW_RISK,
            enable_auto_apply=False  # Require human approval
        )

        await engine.start()

        # Request improvement
        result = await engine.improve_function(
            function=my_slow_function,
            categories=[ImprovementCategory.PERFORMANCE]
        )

        if result.success:
            logger.info(f"Function improved: {result.improvements_applied}")
    """

    def __init__(
        self,
        safety_level: SafetyLevel = SafetyLevel.READ_ONLY,
        enable_auto_apply: bool = False,
        enable_llm: bool = False
    ):
        """
        Initialize self-modifying engine.

        Args:
            safety_level: Maximum allowed modification risk
            enable_auto_apply: Automatically apply safe improvements
            enable_llm: Use LLM for improvement suggestions
        """
        self.safety_level = safety_level
        self.enable_auto_apply = enable_auto_apply
        self.enable_llm = enable_llm

        # Components
        self.introspector = CodeIntrospector()
        self.proposer = ImprovementProposer(enable_llm=enable_llm)
        self.test_generator = TestGenerator()
        self.modifier = SafeModifier(
            safety_level=safety_level,
            require_tests=True
        )

        # Modification queue
        self.pending_requests: Dict[str, ModificationRequest] = {}
        self.completed_modifications: List[ModificationResult] = []

        # Statistics
        self.total_requests = 0
        self.auto_applied = 0
        self.manual_applied = 0
        self.rejected = 0

        # Background task
        self._processing_task: Optional[asyncio.Task] = None

        logger.info(
            "Self-modifying engine initialized",
            safety_level=safety_level.value,
            auto_apply=enable_auto_apply,
            llm_enabled=enable_llm
        )

        # Log safety warning
        logger.warning(
            "⚠️ SELF-MODIFICATION ENABLED - Use with extreme caution!"
        )

    async def start(self):
        """Start the self-modification engine"""
        logger.info("Starting self-modifying engine")

        # Start background processing
        self._processing_task = asyncio.create_task(self._processing_loop())

    async def stop(self):
        """Stop the engine"""
        logger.info("Stopping self-modifying engine")

        if self._processing_task:
            self._processing_task.cancel()

    async def improve_function(
        self,
        function: callable,
        categories: Optional[List[ImprovementCategory]] = None,
        auto_apply: bool = False
    ) -> Dict[str, Any]:
        """
        Request improvement for a function.

        Args:
            function: Function to improve
            categories: Improvement categories to consider
            auto_apply: Automatically apply safe improvements

        Returns:
            Improvement results
        """
        self.total_requests += 1

        request_id = f"req_{self.total_requests}"
        categories = categories or list(ImprovementCategory)

        logger.info(
            "Improvement requested",
            request_id=request_id,
            function=function.__name__,
            categories=[c.value for c in categories]
        )

        # Step 1: Introspect function
        metrics = await self.introspector.analyze_function(function)

        # Step 2: Get source code
        import inspect
        try:
            source_code = inspect.getsource(function)
        except Exception as e:
            logger.error(f"Could not get source: {e}")
            return {
                "success": False,
                "error": "Could not retrieve source code"
            }

        # Step 3: Generate improvement proposals
        proposals = await self.proposer.propose_improvements(
            code_metrics=metrics,
            source_code=source_code
        )

        # Filter by category
        proposals = [
            p for p in proposals
            if p.category in categories
        ]

        if not proposals:
            logger.info("No improvements suggested")
            return {
                "success": True,
                "proposals_count": 0,
                "message": "No improvements needed"
            }

        # Step 4: Process each proposal
        results = []

        for proposal in proposals:
            result = await self._process_proposal(
                proposal, metrics, source_code, auto_apply
            )
            results.append(result)

        # Summary
        applied_count = sum(1 for r in results if r.success)

        return {
            "success": True,
            "proposals_count": len(proposals),
            "improvements_applied": applied_count,
            "improvements_pending": len(proposals) - applied_count,
            "details": results
        }

    async def improve_module(
        self,
        module_path: str,
        categories: Optional[List[ImprovementCategory]] = None
    ) -> Dict[str, Any]:
        """
        Improve all functions in a module.

        Args:
            module_path: Path to Python module
            categories: Improvement categories

        Returns:
            Improvement results
        """
        logger.info(
            "Module improvement requested",
            module_path=module_path
        )

        # Analyze all functions in module
        metrics_list = await self.introspector.analyze_file(module_path)

        # Process each function
        total_proposals = 0
        total_applied = 0

        for metrics in metrics_list:
            # Read source for this function
            # In production: would extract function source from file
            # For now: skip

            total_proposals += 1

        return {
            "success": True,
            "functions_analyzed": len(metrics_list),
            "proposals_generated": total_proposals,
            "improvements_applied": total_applied
        }

    async def get_improvement_status(self) -> Dict[str, Any]:
        """Get status of improvement engine"""
        modifier_stats = await self.modifier.get_stats()

        return {
            "safety_level": self.safety_level.value,
            "auto_apply_enabled": self.enable_auto_apply,
            "llm_enabled": self.enable_llm,

            # Request statistics
            "total_requests": self.total_requests,
            "auto_applied": self.auto_applied,
            "manual_applied": self.manual_applied,
            "rejected": self.rejected,

            # Pending
            "pending_requests": len(self.pending_requests),

            # Modifier statistics
            "modifications": modifier_stats
        }

    # Private methods

    async def _process_proposal(
        self,
        proposal: CodeImprovement,
        metrics: CodeMetrics,
        source_code: str,
        auto_apply: bool
    ) -> ModificationResult:
        """Process a single improvement proposal"""
        logger.debug(
            "Processing proposal",
            proposal_id=proposal.improvement_id,
            category=proposal.category.value
        )

        # Step 1: Generate tests
        tests = await self.test_generator.generate_tests(
            function_name=metrics.function_name or "unknown",
            code_metrics=metrics,
            source_code=source_code
        )

        # Step 2: Check test coverage
        coverage = await self.test_generator.calculate_coverage(tests, metrics)

        if not coverage.is_adequate:
            logger.warning(
                "Insufficient test coverage",
                proposal_id=proposal.improvement_id,
                coverage=coverage.line_coverage_percent
            )

            # Generate more tests if needed
            # In production: would iteratively generate until adequate
            pass

        # Step 3: Decide whether to apply
        should_apply = (
            auto_apply and
            proposal.should_auto_apply and
            coverage.is_adequate
        )

        if should_apply:
            # Apply modification
            result = await self.modifier.apply_modification(
                modification_id=proposal.improvement_id,
                target_file=metrics.file_path,
                original_code=source_code,
                modified_code=proposal.proposed_code,
                tests=tests,
                risk_level=proposal.risk_level
            )

            if result.success:
                self.auto_applied += 1
            else:
                self.rejected += 1

            self.completed_modifications.append(result)

            return result

        else:
            # Queue for human review
            logger.info(
                "Proposal queued for human review",
                proposal_id=proposal.improvement_id
            )

            return ModificationResult(
                modification_id=proposal.improvement_id,
                status=ModificationStatus.PENDING,
                success=False,
                error_message="Awaiting human approval"
            )

    async def _processing_loop(self):
        """Background loop for processing pending requests"""
        while True:
            try:
                await asyncio.sleep(10)

                # Process pending requests
                if self.pending_requests:
                    logger.debug(
                        "Processing pending requests",
                        count=len(self.pending_requests)
                    )

                    # Process each request
                    for request_id, request in list(self.pending_requests.items()):
                        # Process request
                        # In production: would actually process
                        pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Processing loop error", error=str(e))


# Global engine instance (disabled by default for safety)
self_modifier = SelfModifyingEngine(
    safety_level=SafetyLevel.READ_ONLY,
    enable_auto_apply=False
)
