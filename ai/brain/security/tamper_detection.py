#!/usr/bin/env python3
"""
Tamper Detection System for DSMIL Brain

Monitors for signs of tampering or compromise:
- Memory integrity canaries
- Code signature verification
- Timing anomaly detection
- Process spawn monitoring
- Network pattern anomalies

When tampering is detected, triggers self-destruct protocol.
"""

import os
import sys
import time
import ctypes
import hashlib
import secrets
import threading
import logging
import inspect
import struct
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, List, Set, Any, Tuple
from datetime import datetime, timezone
from enum import Enum, auto
from collections import deque
import psutil

logger = logging.getLogger(__name__)


class TamperType(Enum):
    """Types of tampering detected"""
    MEMORY_CORRUPTION = auto()
    CODE_MODIFICATION = auto()
    TIMING_ANOMALY = auto()
    UNAUTHORIZED_PROCESS = auto()
    NETWORK_ANOMALY = auto()
    DEBUGGER_ATTACHED = auto()
    PTRACE_DETECTED = auto()
    ENVIRONMENT_TAMPERING = auto()
    CANARY_VIOLATION = auto()
    INTEGRITY_FAILURE = auto()


class TamperSeverity(Enum):
    """Severity levels for tampering events"""
    LOW = auto()      # Suspicious but could be benign
    MEDIUM = auto()   # Likely tampering, investigate
    HIGH = auto()     # Definite tampering, prepare response
    CRITICAL = auto() # Active attack, immediate action required


@dataclass
class TamperEvidence:
    """Evidence of tampering for reporting"""
    tamper_type: TamperType
    severity: TamperSeverity
    timestamp: datetime
    description: str
    details: Dict[str, Any]
    stack_trace: Optional[str] = None
    memory_dump: Optional[bytes] = None

    def to_dict(self) -> Dict:
        return {
            "type": self.tamper_type.name,
            "severity": self.severity.name,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "details": self.details,
            "has_stack_trace": self.stack_trace is not None,
            "has_memory_dump": self.memory_dump is not None,
        }


@dataclass
class IntegrityCanary:
    """Memory canary for integrity checking"""
    canary_id: str
    location: str
    expected_value: bytes
    check_interval: float = 1.0
    last_check: Optional[datetime] = None

    def generate_value(self) -> bytes:
        """Generate a new canary value"""
        return secrets.token_bytes(32)


class TamperDetector:
    """
    Comprehensive tamper detection system

    Monitors multiple vectors for signs of tampering:
    - Memory integrity via canaries
    - Code hash verification
    - Timing analysis
    - Process monitoring
    - Debugger detection

    Usage:
        detector = TamperDetector()
        detector.start_monitoring()

        # Set callback for tamper events
        detector.on_tamper = handle_tamper

        # Manual check
        if detector.check_all():
            print("System integrity verified")
    """

    # Known debugger process names
    DEBUGGER_PROCESSES = {
        "gdb", "lldb", "strace", "ltrace", "ida", "ida64", "idaw",
        "x64dbg", "x32dbg", "ollydbg", "windbg", "radare2", "r2",
        "ghidra", "frida", "frida-server", "peda", "gef", "pwndbg"
    }

    def __init__(self):
        """Initialize tamper detector"""
        self._canaries: Dict[str, IntegrityCanary] = {}
        self._code_hashes: Dict[str, bytes] = {}
        self._timing_baseline: Dict[str, float] = {}
        self._allowed_processes: Set[int] = set()

        # Event tracking
        self._events: deque = deque(maxlen=1000)
        self._event_lock = threading.Lock()

        # Callbacks
        self.on_tamper: Optional[Callable[[TamperEvidence], None]] = None
        self.on_critical: Optional[Callable[[TamperEvidence], None]] = None

        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._check_interval = 0.5  # seconds

        # Initialize baseline
        self._initialize_baseline()

    def _initialize_baseline(self):
        """Establish baseline measurements"""
        # Record current process as allowed
        self._allowed_processes.add(os.getpid())
        if hasattr(os, 'getppid'):
            self._allowed_processes.add(os.getppid())

        # Hash critical code sections
        self._hash_code_sections()

        # Establish timing baseline
        self._establish_timing_baseline()

        logger.info("Tamper detection baseline established")

    def _hash_code_sections(self):
        """Hash critical code for integrity checking"""
        # Hash this module
        try:
            with open(__file__, 'rb') as f:
                self._code_hashes[__file__] = hashlib.sha384(f.read()).digest()
        except Exception:
            pass

        # Hash imported modules
        for name, module in sys.modules.items():
            if module and hasattr(module, '__file__') and module.__file__:
                try:
                    if 'brain' in module.__file__:
                        with open(module.__file__, 'rb') as f:
                            self._code_hashes[module.__file__] = hashlib.sha384(f.read()).digest()
                except Exception:
                    pass

    def _establish_timing_baseline(self):
        """Establish baseline for timing-sensitive operations"""
        # Measure typical hash computation time
        test_data = b"x" * 1000
        times = []
        for _ in range(100):
            start = time.perf_counter()
            hashlib.sha256(test_data).digest()
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        self._timing_baseline["sha256_1kb"] = avg_time
        self._timing_baseline["sha256_1kb_max"] = max(times) * 2

    # ==================== Canary System ====================

    def create_canary(self, location: str, check_interval: float = 1.0) -> IntegrityCanary:
        """
        Create a memory canary at a logical location

        Args:
            location: Descriptive location name
            check_interval: How often to check (seconds)

        Returns:
            IntegrityCanary instance
        """
        canary_id = secrets.token_hex(8)
        value = secrets.token_bytes(32)

        canary = IntegrityCanary(
            canary_id=canary_id,
            location=location,
            expected_value=value,
            check_interval=check_interval,
        )

        self._canaries[canary_id] = canary
        logger.debug(f"Canary created at {location}: {canary_id}")

        return canary

    def check_canary(self, canary: IntegrityCanary) -> bool:
        """
        Check if a canary has been modified

        Note: In a real implementation, this would check actual memory.
        This is a simplified version that tracks logical canaries.
        """
        if canary.canary_id not in self._canaries:
            return False

        stored = self._canaries[canary.canary_id]
        canary.last_check = datetime.now(timezone.utc)

        # In real implementation, would read memory at canary location
        # and compare with expected value
        return stored.expected_value == canary.expected_value

    def check_all_canaries(self) -> List[IntegrityCanary]:
        """Check all canaries, return list of violated ones"""
        violated = []
        for canary in self._canaries.values():
            if not self.check_canary(canary):
                violated.append(canary)
                self._record_event(TamperEvidence(
                    tamper_type=TamperType.CANARY_VIOLATION,
                    severity=TamperSeverity.CRITICAL,
                    timestamp=datetime.now(timezone.utc),
                    description=f"Canary violated at {canary.location}",
                    details={"canary_id": canary.canary_id, "location": canary.location},
                ))
        return violated

    # ==================== Code Integrity ====================

    def verify_code_integrity(self) -> Tuple[bool, List[str]]:
        """
        Verify integrity of critical code sections

        Returns:
            (all_valid, list_of_modified_files)
        """
        modified = []

        for filepath, expected_hash in self._code_hashes.items():
            try:
                with open(filepath, 'rb') as f:
                    current_hash = hashlib.sha384(f.read()).digest()

                if current_hash != expected_hash:
                    modified.append(filepath)
                    self._record_event(TamperEvidence(
                        tamper_type=TamperType.CODE_MODIFICATION,
                        severity=TamperSeverity.CRITICAL,
                        timestamp=datetime.now(timezone.utc),
                        description=f"Code modified: {filepath}",
                        details={
                            "file": filepath,
                            "expected": expected_hash.hex()[:32],
                            "actual": current_hash.hex()[:32],
                        },
                    ))
            except Exception as e:
                logger.warning(f"Could not verify {filepath}: {e}")

        return len(modified) == 0, modified

    # ==================== Debugger Detection ====================

    def detect_debugger(self) -> bool:
        """
        Detect if a debugger is attached

        Returns:
            True if debugger detected
        """
        # Check TracerPid on Linux
        if sys.platform.startswith('linux'):
            try:
                with open('/proc/self/status', 'r') as f:
                    for line in f:
                        if line.startswith('TracerPid:'):
                            tracer_pid = int(line.split(':')[1].strip())
                            if tracer_pid != 0:
                                self._record_event(TamperEvidence(
                                    tamper_type=TamperType.DEBUGGER_ATTACHED,
                                    severity=TamperSeverity.CRITICAL,
                                    timestamp=datetime.now(timezone.utc),
                                    description=f"Debugger attached (PID: {tracer_pid})",
                                    details={"tracer_pid": tracer_pid},
                                ))
                                return True
            except Exception:
                pass

        # Check for debugger processes
        try:
            for proc in psutil.process_iter(['name', 'pid']):
                if proc.info['name'].lower() in self.DEBUGGER_PROCESSES:
                    self._record_event(TamperEvidence(
                        tamper_type=TamperType.DEBUGGER_ATTACHED,
                        severity=TamperSeverity.HIGH,
                        timestamp=datetime.now(timezone.utc),
                        description=f"Debugger process detected: {proc.info['name']}",
                        details={"process": proc.info['name'], "pid": proc.info['pid']},
                    ))
                    return True
        except Exception:
            pass

        # Timing-based detection (debuggers slow execution)
        test_data = b"x" * 1000
        start = time.perf_counter()
        for _ in range(100):
            hashlib.sha256(test_data).digest()
        elapsed = time.perf_counter() - start

        # If significantly slower than baseline, might be debugged
        baseline = self._timing_baseline.get("sha256_1kb_max", 0.001) * 100
        if elapsed > baseline * 3:
            self._record_event(TamperEvidence(
                tamper_type=TamperType.TIMING_ANOMALY,
                severity=TamperSeverity.MEDIUM,
                timestamp=datetime.now(timezone.utc),
                description="Execution timing anomaly (possible debugger)",
                details={"expected_max": baseline, "actual": elapsed},
            ))
            return True

        return False

    # ==================== Process Monitoring ====================

    def check_process_tree(self) -> List[Dict]:
        """
        Check for suspicious processes in our tree

        Returns:
            List of suspicious process info dicts
        """
        suspicious = []
        our_pid = os.getpid()

        try:
            our_proc = psutil.Process(our_pid)
            parent = our_proc.parent()

            # Check parent chain
            while parent:
                if parent.name().lower() in self.DEBUGGER_PROCESSES:
                    suspicious.append({
                        "pid": parent.pid,
                        "name": parent.name(),
                        "relationship": "parent",
                    })
                parent = parent.parent()

            # Check children
            for child in our_proc.children(recursive=True):
                if child.pid not in self._allowed_processes:
                    info = {
                        "pid": child.pid,
                        "name": child.name(),
                        "relationship": "child",
                    }

                    if child.name().lower() in self.DEBUGGER_PROCESSES:
                        info["suspicious"] = True
                        suspicious.append(info)

        except Exception as e:
            logger.warning(f"Process tree check failed: {e}")

        if suspicious:
            self._record_event(TamperEvidence(
                tamper_type=TamperType.UNAUTHORIZED_PROCESS,
                severity=TamperSeverity.HIGH,
                timestamp=datetime.now(timezone.utc),
                description="Suspicious processes in tree",
                details={"processes": suspicious},
            ))

        return suspicious

    # ==================== Environment Checks ====================

    def check_environment(self) -> List[str]:
        """
        Check for suspicious environment variables

        Returns:
            List of suspicious env vars
        """
        suspicious_vars = []

        # Vars that indicate debugging/tracing
        debug_vars = [
            "LD_PRELOAD", "LD_LIBRARY_PATH",  # Library injection
            "MALLOC_CHECK_", "MALLOC_PERTURB_",  # Memory debugging
            "PYTHONMALLOC", "PYTHONFAULTHANDLER",  # Python debugging
            "GDB_PYTHON_PATH", "DISPLAY",  # Debugger-related
        ]

        for var in debug_vars:
            if var in os.environ:
                suspicious_vars.append(var)

        if suspicious_vars:
            self._record_event(TamperEvidence(
                tamper_type=TamperType.ENVIRONMENT_TAMPERING,
                severity=TamperSeverity.MEDIUM,
                timestamp=datetime.now(timezone.utc),
                description="Suspicious environment variables",
                details={"variables": suspicious_vars},
            ))

        return suspicious_vars

    # ==================== Comprehensive Check ====================

    def check_all(self) -> Tuple[bool, List[TamperEvidence]]:
        """
        Perform all tamper checks

        Returns:
            (is_clean, list_of_evidence)
        """
        evidence = []

        # Check canaries
        violated = self.check_all_canaries()
        if violated:
            evidence.extend([TamperEvidence(
                tamper_type=TamperType.CANARY_VIOLATION,
                severity=TamperSeverity.CRITICAL,
                timestamp=datetime.now(timezone.utc),
                description=f"Canary violated at {c.location}",
                details={"canary_id": c.canary_id},
            ) for c in violated])

        # Check code integrity
        code_ok, modified = self.verify_code_integrity()
        if not code_ok:
            evidence.extend([TamperEvidence(
                tamper_type=TamperType.CODE_MODIFICATION,
                severity=TamperSeverity.CRITICAL,
                timestamp=datetime.now(timezone.utc),
                description=f"Code modified: {f}",
                details={"file": f},
            ) for f in modified])

        # Check for debugger
        if self.detect_debugger():
            evidence.append(TamperEvidence(
                tamper_type=TamperType.DEBUGGER_ATTACHED,
                severity=TamperSeverity.CRITICAL,
                timestamp=datetime.now(timezone.utc),
                description="Debugger detected",
                details={},
            ))

        # Check process tree
        suspicious_procs = self.check_process_tree()
        if suspicious_procs:
            evidence.append(TamperEvidence(
                tamper_type=TamperType.UNAUTHORIZED_PROCESS,
                severity=TamperSeverity.HIGH,
                timestamp=datetime.now(timezone.utc),
                description="Suspicious processes detected",
                details={"processes": suspicious_procs},
            ))

        # Check environment
        suspicious_env = self.check_environment()
        if suspicious_env:
            evidence.append(TamperEvidence(
                tamper_type=TamperType.ENVIRONMENT_TAMPERING,
                severity=TamperSeverity.MEDIUM,
                timestamp=datetime.now(timezone.utc),
                description="Suspicious environment",
                details={"variables": suspicious_env},
            ))

        is_clean = len(evidence) == 0

        # Trigger callbacks for critical events
        for ev in evidence:
            if ev.severity == TamperSeverity.CRITICAL and self.on_critical:
                self.on_critical(ev)
            elif self.on_tamper:
                self.on_tamper(ev)

        return is_clean, evidence

    # ==================== Background Monitoring ====================

    def start_monitoring(self, interval: float = 0.5):
        """Start background monitoring thread"""
        if self._monitoring:
            return

        self._monitoring = True
        self._check_interval = interval

        def monitor_loop():
            while self._monitoring:
                try:
                    is_clean, evidence = self.check_all()
                    if not is_clean:
                        logger.warning(f"Tamper check found {len(evidence)} issues")
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")

                time.sleep(self._check_interval)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Tamper monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Tamper monitoring stopped")

    def _record_event(self, evidence: TamperEvidence):
        """Record a tamper event"""
        with self._event_lock:
            self._events.append(evidence)

        # Trigger callback
        if evidence.severity == TamperSeverity.CRITICAL and self.on_critical:
            self.on_critical(evidence)
        elif self.on_tamper:
            self.on_tamper(evidence)

    def get_recent_events(self, limit: int = 100) -> List[TamperEvidence]:
        """Get recent tamper events"""
        with self._event_lock:
            return list(self._events)[-limit:]

    def get_status(self) -> Dict:
        """Get current detector status"""
        return {
            "monitoring": self._monitoring,
            "check_interval": self._check_interval,
            "canary_count": len(self._canaries),
            "code_hashes_count": len(self._code_hashes),
            "recent_events": len(self._events),
            "allowed_processes": len(self._allowed_processes),
        }


if __name__ == "__main__":
    print("Tamper Detection Self-Test")
    print("=" * 50)

    detector = TamperDetector()

    # Set up callback
    def on_tamper(evidence: TamperEvidence):
        print(f"    TAMPER: {evidence.tamper_type.name} - {evidence.description}")

    detector.on_tamper = on_tamper

    print("\n[1] Creating canaries")
    canary1 = detector.create_canary("memory_pool_header")
    canary2 = detector.create_canary("key_storage_guard")
    print(f"    Created {len(detector._canaries)} canaries")

    print("\n[2] Verifying code integrity")
    code_ok, modified = detector.verify_code_integrity()
    print(f"    Code integrity: {'✓ OK' if code_ok else '✗ Modified'}")

    print("\n[3] Checking for debugger")
    debugger = detector.detect_debugger()
    print(f"    Debugger: {'✗ Detected' if debugger else '✓ Not detected'}")

    print("\n[4] Checking process tree")
    suspicious = detector.check_process_tree()
    print(f"    Suspicious processes: {len(suspicious)}")

    print("\n[5] Checking environment")
    env_issues = detector.check_environment()
    print(f"    Suspicious env vars: {len(env_issues)}")

    print("\n[6] Comprehensive check")
    is_clean, evidence = detector.check_all()
    print(f"    System clean: {'✓ Yes' if is_clean else '✗ No'}")
    print(f"    Evidence items: {len(evidence)}")

    print("\n[7] Status")
    status = detector.get_status()
    for key, value in status.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 50)
    print("Tamper detection test complete")

