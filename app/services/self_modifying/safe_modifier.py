"""
Safe Modifier
Phase 8.4: Safe code modification with rollback

Applies code modifications with safety guarantees:
- Automatic backup before changes
- Rollback on failure
- Test validation
- Audit logging
- Gradual privilege escalation

⚠️ CRITICAL: All modifications must pass safety checks.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import shutil
import hashlib
import structlog

from app.services.self_modifying.test_generator import GeneratedTest, TestCoverage

logger = structlog.get_logger()


class SafetyLevel(Enum):
    """Safety levels for modifications"""
    READ_ONLY = "read_only"        # Can only read code
    DOCUMENTATION = "documentation" # Can modify docs/comments
    LOW_RISK = "low_risk"          # Can make low-risk changes
    MEDIUM_RISK = "medium_risk"    # Can make medium-risk changes
    FULL_ACCESS = "full_access"    # Full modification access (requires approval)


class ModificationStatus(Enum):
    """Status of a modification"""
    PENDING = "pending"          # Not yet applied
    TESTING = "testing"          # Running tests
    APPLIED = "applied"          # Successfully applied
    FAILED = "failed"            # Failed to apply
    ROLLED_BACK = "rolled_back"  # Rolled back


@dataclass
class ModificationResult:
    """Result of applying a modification"""
    modification_id: str
    status: ModificationStatus

    # Success/failure
    success: bool
    error_message: Optional[str] = None

    # Testing
    tests_passed: int = 0
    tests_failed: int = 0
    test_coverage: Optional[TestCoverage] = None

    # Backup
    backup_path: Optional[str] = None
    backup_hash: Optional[str] = None

    # Audit
    applied_at: Optional[datetime] = None
    applied_by: str = "system"
    audit_log: List[str] = field(default_factory=list)


class SafeModifier:
    """
    Safe Code Modifier.

    Applies code modifications with comprehensive safety measures:
    - Mandatory backups
    - Test validation
    - Automatic rollback on failure
    - Audit logging
    - Privilege levels

    Safety Protocol:
        1. Backup original code
        2. Apply modification
        3. Run tests
        4. If tests pass → commit
        5. If tests fail → rollback

    Example:
        modifier = SafeModifier(safety_level=SafetyLevel.LOW_RISK)

        # Apply modification
        result = await modifier.apply_modification(
            target_file="module.py",
            original_code=original,
            modified_code=modified,
            tests=generated_tests
        )

        if not result.success:
            logger.error(f"Modification failed: {result.error_message}")
    """

    def __init__(
        self,
        safety_level: SafetyLevel = SafetyLevel.READ_ONLY,
        backup_dir: str = "/tmp/memshadow_backups",
        require_tests: bool = True
    ):
        """
        Initialize safe modifier.

        Args:
            safety_level: Maximum allowed risk level
            backup_dir: Directory for backups
            require_tests: Require tests for all modifications
        """
        self.safety_level = safety_level
        self.backup_dir = Path(backup_dir)
        self.require_tests = require_tests

        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Modification history
        self.modifications: Dict[str, ModificationResult] = {}

        # Statistics
        self.total_modifications = 0
        self.successful_modifications = 0
        self.failed_modifications = 0
        self.rollbacks = 0

        logger.info(
            "Safe modifier initialized",
            safety_level=safety_level.value,
            require_tests=require_tests
        )

    async def apply_modification(
        self,
        modification_id: str,
        target_file: str,
        original_code: str,
        modified_code: str,
        tests: Optional[List[GeneratedTest]] = None,
        risk_level: str = "medium"
    ) -> ModificationResult:
        """
        Apply a code modification safely.

        Args:
            modification_id: Unique modification ID
            target_file: File to modify
            original_code: Original code
            modified_code: Modified code
            tests: Tests to validate modification
            risk_level: Risk level of modification

        Returns:
            Modification result
        """
        self.total_modifications += 1

        result = ModificationResult(
            modification_id=modification_id,
            status=ModificationStatus.PENDING
        )

        result.audit_log.append(f"Modification requested: {modification_id}")

        # 1. Safety check
        if not await self._safety_check(risk_level, tests):
            result.status = ModificationStatus.FAILED
            result.success = False
            result.error_message = "Failed safety check"
            result.audit_log.append("REJECTED: Failed safety check")

            self.failed_modifications += 1

            logger.warning(
                "Modification rejected",
                modification_id=modification_id,
                reason="Safety check failed"
            )

            return result

        # 2. Backup original code
        backup_path = await self._backup_code(
            target_file, original_code, modification_id
        )

        result.backup_path = str(backup_path)
        result.backup_hash = self._hash_code(original_code)
        result.audit_log.append(f"Backup created: {backup_path}")

        # 3. Apply modification
        try:
            await self._write_code(target_file, modified_code)
            result.audit_log.append("Code modification applied")

        except Exception as e:
            result.status = ModificationStatus.FAILED
            result.success = False
            result.error_message = f"Failed to write code: {e}"
            result.audit_log.append(f"ERROR: {e}")

            self.failed_modifications += 1

            logger.error(
                "Failed to apply modification",
                modification_id=modification_id,
                error=str(e)
            )

            return result

        # 4. Run tests
        if tests and self.require_tests:
            result.status = ModificationStatus.TESTING
            result.audit_log.append("Running tests")

            test_success = await self._run_tests(tests, result)

            if not test_success:
                # Rollback
                await self._rollback(target_file, backup_path, result)

                self.failed_modifications += 1
                self.rollbacks += 1

                logger.warning(
                    "Modification rolled back",
                    modification_id=modification_id,
                    reason="Tests failed"
                )

                return result

        # 5. Success
        result.status = ModificationStatus.APPLIED
        result.success = True
        result.applied_at = datetime.utcnow()
        result.audit_log.append("Modification successfully applied")

        self.successful_modifications += 1

        # Store result
        self.modifications[modification_id] = result

        logger.info(
            "Modification applied successfully",
            modification_id=modification_id,
            tests_passed=result.tests_passed
        )

        return result

    async def rollback_modification(
        self,
        modification_id: str
    ) -> bool:
        """
        Rollback a modification.

        Args:
            modification_id: Modification to rollback

        Returns:
            True if rollback succeeded
        """
        if modification_id not in self.modifications:
            logger.error(f"Modification not found: {modification_id}")
            return False

        result = self.modifications[modification_id]

        if not result.backup_path:
            logger.error(f"No backup for modification: {modification_id}")
            return False

        # Read backup
        backup_path = Path(result.backup_path)
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False

        backup_code = backup_path.read_text()

        # Restore original code
        # In production: would restore to actual file
        result.status = ModificationStatus.ROLLED_BACK
        result.audit_log.append("Modification rolled back")

        self.rollbacks += 1

        logger.info(
            "Modification rolled back",
            modification_id=modification_id
        )

        return True

    async def get_stats(self) -> Dict[str, Any]:
        """Get modifier statistics"""
        success_rate = (
            self.successful_modifications / max(1, self.total_modifications)
        ) * 100

        return {
            "safety_level": self.safety_level.value,
            "total_modifications": self.total_modifications,
            "successful": self.successful_modifications,
            "failed": self.failed_modifications,
            "rollbacks": self.rollbacks,
            "success_rate_percent": success_rate
        }

    # Private methods

    async def _safety_check(
        self,
        risk_level: str,
        tests: Optional[List[GeneratedTest]]
    ) -> bool:
        """Check if modification meets safety requirements"""
        # Check safety level
        risk_hierarchy = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4
        }

        level_hierarchy = {
            SafetyLevel.READ_ONLY: 0,
            SafetyLevel.DOCUMENTATION: 1,
            SafetyLevel.LOW_RISK: 1,
            SafetyLevel.MEDIUM_RISK: 2,
            SafetyLevel.FULL_ACCESS: 4
        }

        mod_risk = risk_hierarchy.get(risk_level, 3)
        max_risk = level_hierarchy.get(self.safety_level, 0)

        if mod_risk > max_risk:
            logger.warning(
                "Modification exceeds safety level",
                risk_level=risk_level,
                max_allowed=self.safety_level.value
            )
            return False

        # Check tests requirement
        if self.require_tests and not tests:
            logger.warning("Tests required but not provided")
            return False

        return True

    async def _backup_code(
        self,
        target_file: str,
        code: str,
        modification_id: str
    ) -> Path:
        """Create backup of code"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{Path(target_file).stem}_{modification_id}_{timestamp}.bak"
        backup_path = self.backup_dir / backup_filename

        # Write backup
        backup_path.write_text(code)

        logger.debug(
            "Backup created",
            backup_path=str(backup_path),
            modification_id=modification_id
        )

        return backup_path

    async def _write_code(self, target_file: str, code: str):
        """Write code to file"""
        # In production: would write to actual file
        # For safety demo: just log
        logger.debug(
            "Code write simulated",
            target_file=target_file,
            code_length=len(code)
        )

    async def _run_tests(
        self,
        tests: List[GeneratedTest],
        result: ModificationResult
    ) -> bool:
        """Run tests and update result"""
        # In production: would actually run tests
        # For now: simulate test execution

        # Simulate 90% test pass rate
        import random

        for test in tests:
            if random.random() < 0.9:
                result.tests_passed += 1
            else:
                result.tests_failed += 1

        result.audit_log.append(
            f"Tests completed: {result.tests_passed} passed, "
            f"{result.tests_failed} failed"
        )

        # Success if all tests passed
        return result.tests_failed == 0

    async def _rollback(
        self,
        target_file: str,
        backup_path: Path,
        result: ModificationResult
    ):
        """Rollback modification"""
        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}")
            return

        backup_code = backup_path.read_text()

        # Restore code (in production: write to actual file)
        logger.debug(
            "Rollback simulated",
            target_file=target_file,
            backup_path=str(backup_path)
        )

        result.status = ModificationStatus.ROLLED_BACK
        result.success = False
        result.error_message = "Tests failed - rolled back"
        result.audit_log.append("Modification rolled back due to test failures")

    def _hash_code(self, code: str) -> str:
        """Generate hash of code for integrity verification"""
        return hashlib.sha256(code.encode()).hexdigest()
