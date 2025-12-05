#!/usr/bin/env python3
"""
Antibody System for DSMIL Brain Digital Immune System

Known threat signatures and rapid pattern matching:
- Signature storage and indexing
- Rapid matching algorithms
- Signature evolution and updates
- Cross-node signature sharing
"""

import hashlib
import re
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Pattern
from datetime import datetime, timezone, timedelta
from enum import Enum, auto

logger = logging.getLogger(__name__)


class SignatureType(Enum):
    """Types of threat signatures"""
    HASH = auto()           # File/content hash
    PATTERN = auto()        # Regex pattern
    BEHAVIORAL = auto()     # Behavioral pattern
    NETWORK = auto()        # Network indicator
    STRUCTURAL = auto()     # Code structure
    SEMANTIC = auto()       # Semantic meaning


class SignatureStatus(Enum):
    """Status of a signature"""
    ACTIVE = auto()
    DEPRECATED = auto()
    EXPERIMENTAL = auto()
    DISABLED = auto()


@dataclass
class ThreatSignature:
    """A threat signature for detection"""
    signature_id: str
    name: str
    signature_type: SignatureType

    # Detection
    pattern: str  # Hash, regex, or structural pattern
    compiled_pattern: Optional[Pattern] = None

    # Classification
    threat_family: str = ""
    severity: int = 5  # 1-10
    confidence: float = 0.9

    # Metadata
    description: str = ""
    source: str = ""
    tags: Set[str] = field(default_factory=set)

    # Tracking
    status: SignatureStatus = SignatureStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Statistics
    match_count: int = 0
    false_positive_count: int = 0
    last_match: Optional[datetime] = None

    def __post_init__(self):
        if self.signature_type == SignatureType.PATTERN and not self.compiled_pattern:
            try:
                self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE)
            except re.error:
                logger.warning(f"Invalid regex pattern: {self.pattern}")

    def match(self, data: str) -> bool:
        """Check if data matches this signature"""
        if self.status != SignatureStatus.ACTIVE:
            return False

        if self.signature_type == SignatureType.HASH:
            # Hash comparison
            data_hash = hashlib.sha256(data.encode()).hexdigest()
            return data_hash == self.pattern or self.pattern in data

        elif self.signature_type == SignatureType.PATTERN:
            # Regex matching
            if self.compiled_pattern:
                return bool(self.compiled_pattern.search(data))
            return self.pattern.lower() in data.lower()

        else:
            # String containment
            return self.pattern.lower() in data.lower()

    def record_match(self):
        """Record a signature match"""
        self.match_count += 1
        self.last_match = datetime.now(timezone.utc)

    def record_false_positive(self):
        """Record a false positive"""
        self.false_positive_count += 1
        # Auto-decrease confidence
        self.confidence = max(0.1, self.confidence - 0.05)

    def effectiveness_score(self) -> float:
        """Calculate effectiveness score"""
        if self.match_count == 0:
            return 0.5

        fp_rate = self.false_positive_count / max(1, self.match_count + self.false_positive_count)
        return self.confidence * (1 - fp_rate)


@dataclass
class SignatureMatch:
    """Result of a signature match"""
    signature_id: str
    signature_name: str

    # Match details
    matched_data: str
    match_location: int = 0
    match_length: int = 0

    # Classification
    threat_family: str = ""
    severity: int = 5
    confidence: float = 0.9

    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AntibodyLibrary:
    """
    Antibody (Signature) Library

    Stores and matches threat signatures for rapid detection.

    Usage:
        library = AntibodyLibrary()

        # Add signature
        library.add_signature(ThreatSignature(
            signature_id="mal-001",
            name="Malware.GenericTrojan",
            signature_type=SignatureType.PATTERN,
            pattern=r"eval\s*\(\s*base64_decode",
            severity=8
        ))

        # Scan content
        matches = library.scan(suspicious_content)
    """

    def __init__(self):
        self._signatures: Dict[str, ThreatSignature] = {}

        # Indices for fast lookup
        self._by_type: Dict[SignatureType, Set[str]] = {}
        self._by_family: Dict[str, Set[str]] = {}
        self._by_hash: Dict[str, str] = {}  # For fast hash lookup

        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            "total_signatures": 0,
            "total_scans": 0,
            "total_matches": 0,
        }

        logger.info("AntibodyLibrary initialized")

    def add_signature(self, signature: ThreatSignature):
        """Add a signature to the library"""
        with self._lock:
            self._signatures[signature.signature_id] = signature

            # Index by type
            if signature.signature_type not in self._by_type:
                self._by_type[signature.signature_type] = set()
            self._by_type[signature.signature_type].add(signature.signature_id)

            # Index by family
            if signature.threat_family:
                if signature.threat_family not in self._by_family:
                    self._by_family[signature.threat_family] = set()
                self._by_family[signature.threat_family].add(signature.signature_id)

            # Index hashes for fast lookup
            if signature.signature_type == SignatureType.HASH:
                self._by_hash[signature.pattern] = signature.signature_id

            self.stats["total_signatures"] = len(self._signatures)

    def remove_signature(self, signature_id: str):
        """Remove a signature from the library"""
        with self._lock:
            if signature_id not in self._signatures:
                return

            sig = self._signatures[signature_id]

            # Remove from indices
            if sig.signature_type in self._by_type:
                self._by_type[sig.signature_type].discard(signature_id)

            if sig.threat_family and sig.threat_family in self._by_family:
                self._by_family[sig.threat_family].discard(signature_id)

            if sig.signature_type == SignatureType.HASH:
                self._by_hash.pop(sig.pattern, None)

            del self._signatures[signature_id]
            self.stats["total_signatures"] = len(self._signatures)

    def scan(self, data: str,
            signature_types: Optional[Set[SignatureType]] = None,
            families: Optional[Set[str]] = None,
            min_severity: int = 1) -> List[SignatureMatch]:
        """
        Scan data against signatures

        Args:
            data: Data to scan
            signature_types: Filter by signature types
            families: Filter by threat families
            min_severity: Minimum severity to report

        Returns:
            List of SignatureMatch objects
        """
        matches = []

        with self._lock:
            self.stats["total_scans"] += 1

            # Determine which signatures to check
            sig_ids = set(self._signatures.keys())

            if signature_types:
                type_ids = set()
                for sig_type in signature_types:
                    type_ids |= self._by_type.get(sig_type, set())
                sig_ids &= type_ids

            if families:
                family_ids = set()
                for family in families:
                    family_ids |= self._by_family.get(family, set())
                sig_ids &= family_ids

            # Fast hash check first
            data_hash = hashlib.sha256(data.encode()).hexdigest()
            if data_hash in self._by_hash:
                hash_sig_id = self._by_hash[data_hash]
                if hash_sig_id in sig_ids:
                    sig = self._signatures[hash_sig_id]
                    if sig.severity >= min_severity:
                        sig.record_match()
                        matches.append(SignatureMatch(
                            signature_id=sig.signature_id,
                            signature_name=sig.name,
                            matched_data=data_hash,
                            threat_family=sig.threat_family,
                            severity=sig.severity,
                            confidence=sig.confidence,
                        ))

            # Pattern matching
            for sig_id in sig_ids:
                sig = self._signatures[sig_id]

                if sig.signature_type == SignatureType.HASH:
                    continue  # Already checked

                if sig.severity < min_severity:
                    continue

                if sig.match(data):
                    sig.record_match()

                    # Find match location for patterns
                    location = 0
                    length = 0
                    if sig.compiled_pattern:
                        m = sig.compiled_pattern.search(data)
                        if m:
                            location = m.start()
                            length = m.end() - m.start()

                    matches.append(SignatureMatch(
                        signature_id=sig.signature_id,
                        signature_name=sig.name,
                        matched_data=data[location:location+min(length, 100)],
                        match_location=location,
                        match_length=length,
                        threat_family=sig.threat_family,
                        severity=sig.severity,
                        confidence=sig.confidence,
                    ))

            self.stats["total_matches"] += len(matches)

        # Sort by severity descending
        matches.sort(key=lambda m: m.severity, reverse=True)

        return matches

    def scan_hash(self, hash_value: str) -> Optional[SignatureMatch]:
        """Fast hash lookup"""
        with self._lock:
            if hash_value in self._by_hash:
                sig_id = self._by_hash[hash_value]
                sig = self._signatures[sig_id]
                sig.record_match()

                return SignatureMatch(
                    signature_id=sig.signature_id,
                    signature_name=sig.name,
                    matched_data=hash_value,
                    threat_family=sig.threat_family,
                    severity=sig.severity,
                    confidence=sig.confidence,
                )
        return None

    def update_signature(self, signature_id: str, **updates):
        """Update signature fields"""
        with self._lock:
            if signature_id in self._signatures:
                sig = self._signatures[signature_id]
                for key, value in updates.items():
                    if hasattr(sig, key):
                        setattr(sig, key, value)
                sig.last_updated = datetime.now(timezone.utc)

    def get_signature(self, signature_id: str) -> Optional[ThreatSignature]:
        """Get signature by ID"""
        return self._signatures.get(signature_id)

    def get_signatures_by_family(self, family: str) -> List[ThreatSignature]:
        """Get all signatures in a family"""
        with self._lock:
            sig_ids = self._by_family.get(family, set())
            return [self._signatures[sid] for sid in sig_ids if sid in self._signatures]

    def export_signatures(self) -> List[Dict]:
        """Export all signatures"""
        with self._lock:
            return [
                {
                    "signature_id": sig.signature_id,
                    "name": sig.name,
                    "signature_type": sig.signature_type.name,
                    "pattern": sig.pattern,
                    "threat_family": sig.threat_family,
                    "severity": sig.severity,
                    "confidence": sig.confidence,
                    "tags": list(sig.tags),
                }
                for sig in self._signatures.values()
            ]

    def import_signatures(self, signatures: List[Dict]):
        """Import signatures from export"""
        for sig_data in signatures:
            sig = ThreatSignature(
                signature_id=sig_data["signature_id"],
                name=sig_data["name"],
                signature_type=SignatureType[sig_data["signature_type"]],
                pattern=sig_data["pattern"],
                threat_family=sig_data.get("threat_family", ""),
                severity=sig_data.get("severity", 5),
                confidence=sig_data.get("confidence", 0.9),
                tags=set(sig_data.get("tags", [])),
            )
            self.add_signature(sig)

    def get_stats(self) -> Dict:
        """Get library statistics"""
        with self._lock:
            type_counts = {
                t.name: len(ids)
                for t, ids in self._by_type.items()
            }

            return {
                **self.stats,
                "signature_types": type_counts,
                "families": len(self._by_family),
            }


if __name__ == "__main__":
    print("Antibody Library Self-Test")
    print("=" * 50)

    library = AntibodyLibrary()

    print("\n[1] Add Signatures")
    signatures = [
        ThreatSignature(
            signature_id="mal-001",
            name="Malware.EvalBase64",
            signature_type=SignatureType.PATTERN,
            pattern=r"eval\s*\(\s*base64_decode",
            threat_family="Obfuscated.PHP",
            severity=8,
            tags={"malware", "php", "obfuscation"}
        ),
        ThreatSignature(
            signature_id="mal-002",
            name="Malware.ShellExec",
            signature_type=SignatureType.PATTERN,
            pattern=r"shell_exec|system\s*\(|exec\s*\(",
            threat_family="WebShell",
            severity=9,
            tags={"malware", "webshell"}
        ),
        ThreatSignature(
            signature_id="mal-003",
            name="Malware.KnownHash",
            signature_type=SignatureType.HASH,
            pattern="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            threat_family="Trojan.Generic",
            severity=10,
        ),
        ThreatSignature(
            signature_id="mal-004",
            name="Indicator.C2Domain",
            signature_type=SignatureType.NETWORK,
            pattern="malware-c2.evil.com",
            threat_family="Network.C2",
            severity=7,
        ),
    ]

    for sig in signatures:
        library.add_signature(sig)

    print(f"    Added {len(signatures)} signatures")

    print("\n[2] Scan Malicious Content")
    malicious_code = """
    <?php
    $cmd = base64_decode($_GET['c']);
    eval($cmd);
    system('whoami');
    ?>
    """

    matches = library.scan(malicious_code)
    print(f"    Matches found: {len(matches)}")
    for match in matches:
        print(f"      - {match.signature_name}: severity={match.severity}, confidence={match.confidence:.2f}")

    print("\n[3] Scan Clean Content")
    clean_code = """
    function hello() {
        console.log("Hello, World!");
    }
    """

    matches = library.scan(clean_code)
    print(f"    Matches found: {len(matches)}")

    print("\n[4] Scan with Filters")
    matches = library.scan(malicious_code,
                          signature_types={SignatureType.PATTERN},
                          min_severity=8)
    print(f"    Pattern matches (severity>=8): {len(matches)}")

    print("\n[5] Statistics")
    stats = library.get_stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 50)
    print("Antibody Library test complete")

