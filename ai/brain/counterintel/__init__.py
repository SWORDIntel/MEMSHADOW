#!/usr/bin/env python3
"""
DSMIL Brain Counter-Intelligence Suite

Counter-intelligence and security capabilities:
- Cognitive Fingerprinting: Actor attribution via behavioral patterns
- Canary System: Honeypot knowledge for leak detection
- Counter-Surveillance: Detect reconnaissance and probing
- Adversarial Memory: Anti-extraction defenses
"""

from .cognitive_fingerprint import (
    CognitiveFingerprinter,
    CognitiveFingerprint,
    AttributionMatch,
    FingerPrintComponent,
)

from .canary_system import (
    CanarySystem,
    CanaryToken,
    HoneypotKnowledge,
    LeakDetection,
)

from .counter_surveillance import (
    CounterSurveillance,
    ProbeDetection,
    ReconPattern,
    ExtractionAttempt,
)

from .adversarial_memory import (
    AdversarialMemory,
    DistributedSecret,
    ExtractionDefense,
    PlausibleDecoy,
)

__all__ = [
    # Cognitive Fingerprint
    "CognitiveFingerprinter",
    "CognitiveFingerprint",
    "AttributionMatch",
    "FingerPrintComponent",
    # Canary System
    "CanarySystem",
    "CanaryToken",
    "HoneypotKnowledge",
    "LeakDetection",
    # Counter-Surveillance
    "CounterSurveillance",
    "ProbeDetection",
    "ReconPattern",
    "ExtractionAttempt",
    # Adversarial Memory
    "AdversarialMemory",
    "DistributedSecret",
    "ExtractionDefense",
    "PlausibleDecoy",
]

