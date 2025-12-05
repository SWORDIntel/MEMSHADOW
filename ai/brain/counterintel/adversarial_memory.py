#!/usr/bin/env python3
"""
Adversarial Memory Fortress for DSMIL Brain

Anti-extraction defenses:
- Computation-bound recall (anti-bulk-extraction)
- Distributed secret sharing (no single node complete)
- Plausible deniability decoys
- Extraction detection and response
"""

import hashlib
import secrets
import threading
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Tuple
from datetime import datetime, timezone
from enum import Enum, auto

logger = logging.getLogger(__name__)


class SecretType(Enum):
    """Types of protected secrets"""
    CRITICAL = auto()      # Most sensitive, requires quorum
    SENSITIVE = auto()     # Sensitive, computation-bound
    INTERNAL = auto()      # Internal only, basic protection
    DECOY = auto()         # Plausible deniability decoy


@dataclass
class DistributedSecret:
    """A secret distributed across multiple shares"""
    secret_id: str
    secret_type: SecretType

    # Shares
    total_shares: int
    threshold: int  # Minimum shares to reconstruct
    share_holders: Dict[str, bytes] = field(default_factory=dict)  # node_id -> share

    # Metadata (no actual secret here)
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Access tracking
    reconstruction_attempts: int = 0
    last_access: Optional[datetime] = None


@dataclass
class PlausibleDecoy:
    """A decoy that provides plausible deniability"""
    decoy_id: str

    # Content
    decoy_content: Any
    real_content_hint: str = ""  # Hint to real content (for authorized access)

    # Plausibility
    plausibility_score: float = 0.8

    # Trigger
    trigger_key: bytes = b""  # Key that reveals it's a decoy

    # Tracking
    access_count: int = 0
    decoy_served_count: int = 0


@dataclass
class ExtractionDefense:
    """Configuration for extraction defense"""
    defense_id: str

    # Rate limiting
    max_queries_per_minute: int = 10
    max_data_per_hour_mb: float = 1.0

    # Computation binding
    min_computation_ms: int = 100  # Minimum time per query
    computation_factor: float = 1.5  # Multiplier for sensitive data

    # Alerts
    alert_threshold: int = 5
    block_threshold: int = 10

    # Status
    is_active: bool = True


class SecretSharing:
    """
    Shamir's Secret Sharing implementation

    Splits secrets into shares where k of n shares
    are needed to reconstruct.
    """

    PRIME = 2**127 - 1  # Large prime for finite field

    @classmethod
    def split(cls, secret: bytes, n: int, k: int) -> List[Tuple[int, int]]:
        """
        Split secret into n shares with threshold k

        Args:
            secret: Secret to split
            n: Number of shares
            k: Threshold for reconstruction

        Returns:
            List of (x, y) share tuples
        """
        # Convert secret to integer
        secret_int = int.from_bytes(secret, 'big') % cls.PRIME

        # Generate random polynomial coefficients
        coefficients = [secret_int] + [secrets.randbelow(cls.PRIME) for _ in range(k - 1)]

        # Generate shares
        shares = []
        for x in range(1, n + 1):
            y = sum(c * pow(x, i, cls.PRIME) for i, c in enumerate(coefficients)) % cls.PRIME
            shares.append((x, y))

        return shares

    @classmethod
    def reconstruct(cls, shares: List[Tuple[int, int]], secret_len: int) -> bytes:
        """
        Reconstruct secret from shares using Lagrange interpolation

        Args:
            shares: List of (x, y) shares
            secret_len: Expected length of secret in bytes

        Returns:
            Reconstructed secret
        """
        k = len(shares)

        def lagrange_basis(i: int, x: int) -> int:
            xi, _ = shares[i]
            result = 1
            for j in range(k):
                if i != j:
                    xj, _ = shares[j]
                    result = result * (x - xj) * pow(xi - xj, cls.PRIME - 2, cls.PRIME) % cls.PRIME
            return result

        # Interpolate at x=0 to get secret
        secret_int = sum(
            yi * lagrange_basis(i, 0)
            for i, (_, yi) in enumerate(shares)
        ) % cls.PRIME

        return secret_int.to_bytes(secret_len, 'big')


class ComputationBound:
    """Implements computation-bound memory access"""

    def __init__(self, base_iterations: int = 10000):
        self.base_iterations = base_iterations

    def compute_proof(self, challenge: bytes, difficulty: float = 1.0) -> bytes:
        """
        Compute proof of work before accessing data

        Args:
            challenge: Challenge bytes
            difficulty: Difficulty multiplier

        Returns:
            Proof bytes
        """
        iterations = int(self.base_iterations * difficulty)

        result = challenge
        for _ in range(iterations):
            result = hashlib.sha256(result).digest()

        return result

    def verify_proof(self, challenge: bytes, proof: bytes, difficulty: float = 1.0) -> bool:
        """Verify a proof of work"""
        expected = self.compute_proof(challenge, difficulty)
        return secrets.compare_digest(expected, proof)


class AdversarialMemory:
    """
    Adversarial Memory Protection System

    Protects sensitive data from bulk extraction:
    - Rate limiting
    - Computation-bound access
    - Distributed secret sharing
    - Plausible deniability

    Usage:
        am = AdversarialMemory()

        # Store sensitive data
        secret_id = am.store_distributed("my_secret", secret_bytes, threshold=3, shares=5)

        # Retrieve (requires proof of work)
        data = am.retrieve_protected("key", accessor_id)

        # Add decoy
        am.add_decoy("fake_key", fake_data, real_data)
    """

    def __init__(self, computation_iterations: int = 10000):
        self._secrets: Dict[str, DistributedSecret] = {}
        self._decoys: Dict[str, PlausibleDecoy] = {}
        self._protected_data: Dict[str, Tuple[Any, SecretType]] = {}

        # Defense configuration
        self._defense = ExtractionDefense(defense_id="default")

        # Computation bound
        self._computation = ComputationBound(computation_iterations)

        # Rate limiting
        self._access_history: Dict[str, List[datetime]] = {}
        self._data_extracted: Dict[str, float] = {}  # accessor -> MB extracted

        # Blocked accessors
        self._blocked: Set[str] = set()

        self._lock = threading.RLock()

        # Callbacks
        self.on_extraction_attempt: Optional[callable] = None
        self.on_threshold_exceeded: Optional[callable] = None

        logger.info("AdversarialMemory initialized")

    def store_distributed(self, secret_id: str, secret: bytes,
                         threshold: int, shares: int,
                         node_ids: Optional[List[str]] = None) -> DistributedSecret:
        """
        Store a secret using distributed sharing

        Args:
            secret_id: Identifier for the secret
            secret: Secret bytes to store
            threshold: Minimum shares needed
            shares: Total number of shares
            node_ids: Optional list of node IDs to assign shares

        Returns:
            DistributedSecret metadata (no actual secret)
        """
        if threshold > shares:
            raise ValueError("Threshold cannot exceed number of shares")

        # Generate shares
        share_tuples = SecretSharing.split(secret, shares, threshold)

        # Assign to nodes
        if node_ids is None:
            node_ids = [f"node-{i}" for i in range(shares)]

        share_holders = {}
        for i, (x, y) in enumerate(share_tuples):
            node_id = node_ids[i % len(node_ids)]
            # Encode share as bytes
            share_bytes = x.to_bytes(16, 'big') + y.to_bytes(16, 'big')
            share_holders[node_id] = share_bytes

        distributed = DistributedSecret(
            secret_id=secret_id,
            secret_type=SecretType.CRITICAL,
            total_shares=shares,
            threshold=threshold,
            share_holders=share_holders,
            description=f"Distributed secret with {threshold}-of-{shares} threshold",
        )

        with self._lock:
            self._secrets[secret_id] = distributed

        logger.info(f"Stored distributed secret: {secret_id} ({threshold}/{shares})")
        return distributed

    def reconstruct_distributed(self, secret_id: str,
                               shares: Dict[str, bytes],
                               secret_len: int = 32) -> Optional[bytes]:
        """
        Reconstruct a distributed secret from shares

        Args:
            secret_id: Secret identifier
            shares: Dict of node_id -> share_bytes
            secret_len: Expected secret length

        Returns:
            Reconstructed secret or None
        """
        with self._lock:
            if secret_id not in self._secrets:
                return None

            secret_meta = self._secrets[secret_id]
            secret_meta.reconstruction_attempts += 1
            secret_meta.last_access = datetime.now(timezone.utc)

            if len(shares) < secret_meta.threshold:
                logger.warning(f"Insufficient shares for {secret_id}: {len(shares)}/{secret_meta.threshold}")
                return None

            # Decode shares
            share_tuples = []
            for node_id, share_bytes in shares.items():
                x = int.from_bytes(share_bytes[:16], 'big')
                y = int.from_bytes(share_bytes[16:], 'big')
                share_tuples.append((x, y))

            # Reconstruct
            try:
                return SecretSharing.reconstruct(share_tuples[:secret_meta.threshold], secret_len)
            except Exception as e:
                logger.error(f"Failed to reconstruct {secret_id}: {e}")
                return None

    def store_protected(self, key: str, data: Any,
                       secret_type: SecretType = SecretType.SENSITIVE):
        """Store data with protection"""
        with self._lock:
            self._protected_data[key] = (data, secret_type)

    def retrieve_protected(self, key: str, accessor_id: str,
                          proof: Optional[bytes] = None) -> Tuple[Optional[Any], bool]:
        """
        Retrieve protected data

        Args:
            key: Data key
            accessor_id: Who is accessing
            proof: Proof of work for computation-bound access

        Returns:
            (data, is_decoy) tuple
        """
        # Check if blocked
        if accessor_id in self._blocked:
            logger.warning(f"Blocked accessor attempted access: {accessor_id}")
            return None, False

        # Rate limiting
        if not self._check_rate_limit(accessor_id):
            if self.on_threshold_exceeded:
                self.on_threshold_exceeded(accessor_id, "rate_limit")
            return None, False

        with self._lock:
            # Check for decoy first
            if key in self._decoys:
                decoy = self._decoys[key]
                decoy.access_count += 1

                # Check if accessor has real key
                if proof and decoy.trigger_key:
                    if secrets.compare_digest(proof, decoy.trigger_key):
                        # Authorized - return real content hint
                        return decoy.real_content_hint, False

                # Return decoy
                decoy.decoy_served_count += 1
                return decoy.decoy_content, True

            # Check protected data
            if key not in self._protected_data:
                return None, False

            data, secret_type = self._protected_data[key]

            # Require computation proof for sensitive data
            if secret_type in (SecretType.SENSITIVE, SecretType.CRITICAL):
                if proof is None:
                    # Generate challenge
                    challenge = hashlib.sha256(f"{key}:{accessor_id}:{time.time()}".encode()).digest()

                    # Return challenge instead of data
                    logger.info(f"Computation proof required for {key}")
                    return {"challenge": challenge.hex(), "difficulty": 1.5}, False

                # Verify proof
                difficulty = 2.0 if secret_type == SecretType.CRITICAL else 1.5
                challenge = bytes.fromhex(proof.get("challenge", ""))
                if not self._computation.verify_proof(challenge, bytes.fromhex(proof.get("proof", "")), difficulty):
                    logger.warning(f"Invalid computation proof from {accessor_id}")
                    return None, False

            # Track extraction
            self._track_extraction(accessor_id, key, data)

            return data, False

    def _check_rate_limit(self, accessor_id: str) -> bool:
        """Check if accessor is within rate limits"""
        now = datetime.now(timezone.utc)

        with self._lock:
            if accessor_id not in self._access_history:
                self._access_history[accessor_id] = []

            # Clean old entries
            cutoff = now - timedelta(minutes=1)
            self._access_history[accessor_id] = [
                t for t in self._access_history[accessor_id]
                if t > cutoff
            ]

            # Check limit
            if len(self._access_history[accessor_id]) >= self._defense.max_queries_per_minute:
                return False

            # Record access
            self._access_history[accessor_id].append(now)
            return True

    def _track_extraction(self, accessor_id: str, key: str, data: Any):
        """Track data extraction for anomaly detection"""
        # Estimate data size
        import sys
        try:
            size_bytes = sys.getsizeof(data)
        except:
            size_bytes = len(str(data))

        size_mb = size_bytes / (1024 * 1024)

        with self._lock:
            self._data_extracted[accessor_id] = self._data_extracted.get(accessor_id, 0) + size_mb

            # Check threshold
            if self._data_extracted[accessor_id] > self._defense.max_data_per_hour_mb:
                logger.warning(f"Extraction threshold exceeded for {accessor_id}")

                if self.on_extraction_attempt:
                    self.on_extraction_attempt(accessor_id, self._data_extracted[accessor_id])

                # Block after threshold
                if self._data_extracted[accessor_id] > self._defense.max_data_per_hour_mb * 2:
                    self._blocked.add(accessor_id)

    def add_decoy(self, key: str, decoy_content: Any,
                 real_content_hint: str = "",
                 trigger_key: Optional[bytes] = None) -> PlausibleDecoy:
        """
        Add a plausible deniability decoy

        Args:
            key: Key that triggers decoy
            decoy_content: Fake content to serve
            real_content_hint: Hint for authorized access
            trigger_key: Key that bypasses decoy

        Returns:
            PlausibleDecoy object
        """
        decoy = PlausibleDecoy(
            decoy_id=f"decoy-{secrets.token_hex(4)}",
            decoy_content=decoy_content,
            real_content_hint=real_content_hint,
            trigger_key=trigger_key or secrets.token_bytes(32),
        )

        with self._lock:
            self._decoys[key] = decoy

        return decoy

    def block_accessor(self, accessor_id: str, reason: str = "manual"):
        """Block an accessor"""
        with self._lock:
            self._blocked.add(accessor_id)
            logger.warning(f"Blocked accessor {accessor_id}: {reason}")

    def unblock_accessor(self, accessor_id: str):
        """Unblock an accessor"""
        with self._lock:
            self._blocked.discard(accessor_id)

    def get_stats(self) -> Dict:
        """Get memory protection statistics"""
        with self._lock:
            return {
                "distributed_secrets": len(self._secrets),
                "protected_data": len(self._protected_data),
                "decoys": len(self._decoys),
                "blocked_accessors": len(self._blocked),
                "total_extraction_mb": sum(self._data_extracted.values()),
            }


if __name__ == "__main__":
    print("Adversarial Memory Self-Test")
    print("=" * 50)

    am = AdversarialMemory(computation_iterations=1000)  # Lower for testing

    print("\n[1] Distributed Secret Sharing")
    secret = b"This is a top secret message!"
    dist_secret = am.store_distributed(
        "secret-001",
        secret,
        threshold=3,
        shares=5,
        node_ids=["node-1", "node-2", "node-3", "node-4", "node-5"]
    )
    print(f"    Created distributed secret: {dist_secret.secret_id}")
    print(f"    Threshold: {dist_secret.threshold}/{dist_secret.total_shares}")

    # Reconstruct with enough shares
    shares = dict(list(dist_secret.share_holders.items())[:3])
    reconstructed = am.reconstruct_distributed("secret-001", shares, len(secret))
    print(f"    Reconstructed: {reconstructed == secret}")

    # Try with insufficient shares
    insufficient = dict(list(dist_secret.share_holders.items())[:2])
    failed = am.reconstruct_distributed("secret-001", insufficient, len(secret))
    print(f"    Insufficient shares blocked: {failed is None}")

    print("\n[2] Protected Data with Computation Bound")
    am.store_protected("sensitive-key", {"data": "sensitive data"}, SecretType.SENSITIVE)

    # Try without proof
    result, is_decoy = am.retrieve_protected("sensitive-key", "user-001")
    print(f"    Without proof: challenge returned = {'challenge' in (result or {})}")

    print("\n[3] Plausible Deniability Decoy")
    trigger = secrets.token_bytes(32)
    decoy = am.add_decoy(
        "secret-plans",
        decoy_content={"plans": "We're planning a surprise birthday party"},
        real_content_hint="Use decryption key stored in HSM",
        trigger_key=trigger
    )

    # Access without trigger (gets decoy)
    result, is_decoy_flag = am.retrieve_protected("secret-plans", "adversary-001")
    print(f"    Adversary gets decoy: {is_decoy_flag}")
    print(f"    Decoy content: {result}")

    # Access with trigger (gets hint)
    result, is_decoy_flag = am.retrieve_protected("secret-plans", "authorized-001", proof=trigger)
    print(f"    Authorized gets hint: {is_decoy_flag == False}")

    print("\n[4] Rate Limiting")
    for i in range(12):
        result, _ = am.retrieve_protected("sensitive-key", "spammer-001")

    result, _ = am.retrieve_protected("sensitive-key", "spammer-001")
    print(f"    Rate limited after many requests: {result is None}")

    print("\n[5] Statistics")
    stats = am.get_stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 50)
    print("Adversarial Memory test complete")

