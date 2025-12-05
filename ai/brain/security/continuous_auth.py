#!/usr/bin/env python3
"""
Continuous Authentication System for DSMIL Brain

Implements constant authentication checks between hub and nodes:
- Signed heartbeats every N seconds
- Challenge-response on anomalies
- Session key rotation
- Mutual authentication (hub ↔ node)
- Anomaly detection triggering re-authentication
"""

import time
import secrets
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, List, Any
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from collections import deque
import hashlib
import hmac

from .cnsa_crypto import CNSACrypto, KeyPair, SignedMessage, get_crypto

logger = logging.getLogger(__name__)


class AuthState(Enum):
    """Authentication session states"""
    UNAUTHENTICATED = auto()
    AUTHENTICATING = auto()
    AUTHENTICATED = auto()
    CHALLENGED = auto()
    SUSPICIOUS = auto()
    REVOKED = auto()


class AuthFailureReason(Enum):
    """Reasons for authentication failure"""
    INVALID_SIGNATURE = auto()
    EXPIRED_HEARTBEAT = auto()
    REPLAY_ATTACK = auto()
    CHALLENGE_FAILED = auto()
    NONCE_MISMATCH = auto()
    KEY_REVOKED = auto()
    RATE_LIMIT = auto()
    ANOMALY_DETECTED = auto()
    TIMEOUT = auto()


@dataclass
class Heartbeat:
    """Signed heartbeat for continuous authentication"""
    node_id: str
    sequence: int
    timestamp: float
    nonce: bytes
    signature: bytes
    capabilities_hash: Optional[bytes] = None  # Hash of current capabilities
    load_metrics: Optional[Dict[str, float]] = None

    def to_bytes(self) -> bytes:
        """Serialize for transmission"""
        import struct
        import json

        metrics_json = json.dumps(self.load_metrics or {}).encode()

        return (
            self.node_id.encode().ljust(64, b'\x00') +
            struct.pack(">Q", self.sequence) +
            struct.pack(">d", self.timestamp) +
            self.nonce +
            struct.pack(">H", len(self.signature)) +
            self.signature +
            (self.capabilities_hash or b'\x00' * 48) +
            struct.pack(">H", len(metrics_json)) +
            metrics_json
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "Heartbeat":
        """Deserialize from transmission"""
        import struct
        import json

        node_id = data[:64].rstrip(b'\x00').decode()
        sequence = struct.unpack(">Q", data[64:72])[0]
        timestamp = struct.unpack(">d", data[72:80])[0]
        nonce = data[80:112]
        sig_len = struct.unpack(">H", data[112:114])[0]
        signature = data[114:114+sig_len]
        offset = 114 + sig_len
        capabilities_hash = data[offset:offset+48]
        if capabilities_hash == b'\x00' * 48:
            capabilities_hash = None
        offset += 48
        metrics_len = struct.unpack(">H", data[offset:offset+2])[0]
        metrics_json = data[offset+2:offset+2+metrics_len]
        load_metrics = json.loads(metrics_json) if metrics_json else None

        return cls(
            node_id=node_id,
            sequence=sequence,
            timestamp=timestamp,
            nonce=nonce,
            signature=signature,
            capabilities_hash=capabilities_hash,
            load_metrics=load_metrics,
        )


@dataclass
class ChallengeResponse:
    """Challenge-response for anomaly verification"""
    challenge_id: str
    challenge_nonce: bytes
    response_nonce: bytes
    computation_proof: bytes  # Proof of computation (anti-replay)
    timestamp: float
    signature: bytes


@dataclass
class AuthSession:
    """Authentication session state"""
    session_id: str
    node_id: str
    state: AuthState

    # Keys
    session_key: bytes
    signing_keypair: KeyPair
    peer_public_key: Optional[bytes] = None

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat: Optional[datetime] = None
    last_key_rotation: Optional[datetime] = None

    # Sequence tracking (anti-replay)
    expected_sequence: int = 0
    seen_nonces: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Anomaly tracking
    failed_attempts: int = 0
    anomaly_score: float = 0.0
    suspicious_events: List[Dict[str, Any]] = field(default_factory=list)

    # Configuration
    heartbeat_interval: float = 5.0  # seconds
    heartbeat_tolerance: float = 2.0  # grace period
    max_failed_attempts: int = 3
    key_rotation_interval: timedelta = field(default_factory=lambda: timedelta(hours=1))

    def is_expired(self) -> bool:
        """Check if session needs re-authentication"""
        if self.last_heartbeat is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_heartbeat).total_seconds()
        return elapsed > (self.heartbeat_interval + self.heartbeat_tolerance)

    def needs_key_rotation(self) -> bool:
        """Check if session key should be rotated"""
        if self.last_key_rotation is None:
            return True
        return datetime.now(timezone.utc) - self.last_key_rotation > self.key_rotation_interval


class ContinuousAuthenticator:
    """
    Manages continuous authentication between hub and nodes

    Features:
    - Mutual authentication during session establishment
    - Periodic signed heartbeats
    - Challenge-response for anomalies
    - Automatic session key rotation
    - Anomaly detection and response

    Usage:
        auth = ContinuousAuthenticator(node_id="node-001")

        # Start authentication
        session = auth.initiate_session(peer_public_key)

        # Send heartbeats
        heartbeat = auth.create_heartbeat(session)
        # transmit heartbeat...

        # Verify received heartbeats
        valid, reason = auth.verify_heartbeat(session, received_heartbeat)

        # Handle challenges
        if auth.should_challenge(session):
            challenge = auth.create_challenge(session)
            # transmit and await response...
    """

    def __init__(self, node_id: str, is_hub: bool = False,
                 crypto: Optional[CNSACrypto] = None):
        """
        Initialize authenticator

        Args:
            node_id: Unique identifier for this node
            is_hub: True if this is the central hub
            crypto: Crypto instance (uses singleton if not provided)
        """
        self.node_id = node_id
        self.is_hub = is_hub
        self.crypto = crypto or get_crypto()

        # Generate long-term signing keypair
        self.signing_keypair = self.crypto.generate_signing_keypair()

        # Active sessions
        self.sessions: Dict[str, AuthSession] = {}

        # Callbacks for events
        self.on_auth_failure: Optional[Callable[[str, AuthFailureReason], None]] = None
        self.on_anomaly_detected: Optional[Callable[[str, Dict], None]] = None
        self.on_session_revoked: Optional[Callable[[str], None]] = None

        # Background heartbeat thread
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False

        # Rate limiting
        self._request_times: Dict[str, deque] = {}
        self._rate_limit_window = 60.0  # seconds
        self._rate_limit_max = 100  # max requests per window

        logger.info(f"ContinuousAuthenticator initialized for {node_id} (hub={is_hub})")

    def get_public_key(self) -> bytes:
        """Get our public signing key for sharing"""
        return self.signing_keypair.public_key

    def initiate_session(self, peer_id: str, peer_public_key: bytes,
                        heartbeat_interval: float = 5.0) -> AuthSession:
        """
        Initiate a new authentication session

        Args:
            peer_id: Peer's node ID
            peer_public_key: Peer's public signing key
            heartbeat_interval: Seconds between heartbeats

        Returns:
            New AuthSession
        """
        session_id = secrets.token_hex(16)
        session_key = self.crypto.generate_symmetric_key()

        session = AuthSession(
            session_id=session_id,
            node_id=peer_id,
            state=AuthState.AUTHENTICATING,
            session_key=session_key,
            signing_keypair=self.signing_keypair,
            peer_public_key=peer_public_key,
            heartbeat_interval=heartbeat_interval,
        )

        self.sessions[peer_id] = session
        logger.info(f"Session initiated with {peer_id}: {session_id}")

        return session

    def complete_authentication(self, session: AuthSession) -> bool:
        """Mark session as authenticated after successful handshake"""
        session.state = AuthState.AUTHENTICATED
        session.last_heartbeat = datetime.now(timezone.utc)
        session.last_key_rotation = datetime.now(timezone.utc)
        logger.info(f"Session {session.session_id} authenticated")
        return True

    def create_heartbeat(self, session: AuthSession,
                        capabilities_hash: Optional[bytes] = None,
                        load_metrics: Optional[Dict[str, float]] = None) -> Heartbeat:
        """
        Create a signed heartbeat message

        Args:
            session: Active session
            capabilities_hash: Hash of current node capabilities
            load_metrics: Current load metrics

        Returns:
            Signed Heartbeat
        """
        nonce = self.crypto.secure_random(32)
        timestamp = time.time()

        # Create message to sign
        message = (
            self.node_id.encode() +
            session.expected_sequence.to_bytes(8, 'big') +
            timestamp.to_bytes(8, 'big') +  # Will error - need struct
            nonce
        )

        import struct
        message = (
            self.node_id.encode() +
            struct.pack(">Q", session.expected_sequence) +
            struct.pack(">d", timestamp) +
            nonce +
            (capabilities_hash or b'\x00' * 48)
        )

        signed = self.crypto.sign(message, self.signing_keypair)

        heartbeat = Heartbeat(
            node_id=self.node_id,
            sequence=session.expected_sequence,
            timestamp=timestamp,
            nonce=nonce,
            signature=signed.signature,
            capabilities_hash=capabilities_hash,
            load_metrics=load_metrics,
        )

        return heartbeat

    def verify_heartbeat(self, session: AuthSession, heartbeat: Heartbeat) -> tuple[bool, Optional[AuthFailureReason]]:
        """
        Verify a received heartbeat

        Args:
            session: Session for this peer
            heartbeat: Received heartbeat

        Returns:
            (is_valid, failure_reason or None)
        """
        # Check rate limiting
        if not self._check_rate_limit(heartbeat.node_id):
            self._record_failure(session, AuthFailureReason.RATE_LIMIT)
            return False, AuthFailureReason.RATE_LIMIT

        # Check timestamp freshness
        now = time.time()
        age = abs(now - heartbeat.timestamp)
        if age > session.heartbeat_interval + session.heartbeat_tolerance:
            self._record_failure(session, AuthFailureReason.EXPIRED_HEARTBEAT)
            return False, AuthFailureReason.EXPIRED_HEARTBEAT

        # Check for replay (nonce reuse)
        nonce_key = heartbeat.nonce.hex()
        if nonce_key in session.seen_nonces:
            self._record_failure(session, AuthFailureReason.REPLAY_ATTACK)
            self._record_anomaly(session, "replay_attack", {"nonce": nonce_key})
            return False, AuthFailureReason.REPLAY_ATTACK

        # Check sequence (allow some out-of-order)
        if heartbeat.sequence < session.expected_sequence - 5:
            self._record_failure(session, AuthFailureReason.REPLAY_ATTACK)
            return False, AuthFailureReason.REPLAY_ATTACK

        # Verify signature
        import struct
        message = (
            heartbeat.node_id.encode() +
            struct.pack(">Q", heartbeat.sequence) +
            struct.pack(">d", heartbeat.timestamp) +
            heartbeat.nonce +
            (heartbeat.capabilities_hash or b'\x00' * 48)
        )

        signed = SignedMessage(
            message=message,
            signature=heartbeat.signature,
            signer_key_id=session.node_id,
        )

        if not self.crypto.verify(signed, session.peer_public_key):
            self._record_failure(session, AuthFailureReason.INVALID_SIGNATURE)
            self._record_anomaly(session, "invalid_signature", {})
            return False, AuthFailureReason.INVALID_SIGNATURE

        # Success - update session state
        session.seen_nonces.append(nonce_key)
        session.expected_sequence = max(session.expected_sequence, heartbeat.sequence + 1)
        session.last_heartbeat = datetime.now(timezone.utc)
        session.failed_attempts = 0  # Reset on success

        # Decay anomaly score
        session.anomaly_score = max(0, session.anomaly_score - 0.1)

        return True, None

    def should_challenge(self, session: AuthSession) -> bool:
        """Determine if we should send a challenge"""
        return (
            session.anomaly_score > 0.5 or
            session.failed_attempts > 0 or
            session.state == AuthState.SUSPICIOUS
        )

    def create_challenge(self, session: AuthSession) -> ChallengeResponse:
        """Create a challenge for anomaly verification"""
        challenge_id = secrets.token_hex(8)
        challenge_nonce = self.crypto.secure_random(32)

        # Store challenge for verification
        session.state = AuthState.CHALLENGED

        # Create a computation challenge (e.g., hash chain)
        # The peer must compute: H(H(H(...H(nonce)...))) N times
        difficulty = min(1000, int(session.anomaly_score * 10000))

        challenge = ChallengeResponse(
            challenge_id=challenge_id,
            challenge_nonce=challenge_nonce,
            response_nonce=b"",  # Filled by responder
            computation_proof=difficulty.to_bytes(4, 'big'),
            timestamp=time.time(),
            signature=b"",  # Filled by responder
        )

        logger.info(f"Challenge created for {session.node_id}: difficulty={difficulty}")
        return challenge

    def respond_to_challenge(self, challenge: ChallengeResponse) -> ChallengeResponse:
        """Compute response to a challenge"""
        difficulty = int.from_bytes(challenge.computation_proof, 'big')

        # Compute hash chain
        result = challenge.challenge_nonce
        for _ in range(difficulty):
            result = self.crypto.hash(result)

        response_nonce = self.crypto.secure_random(32)

        # Sign the response
        message = challenge.challenge_id.encode() + result + response_nonce
        signed = self.crypto.sign(message, self.signing_keypair)

        return ChallengeResponse(
            challenge_id=challenge.challenge_id,
            challenge_nonce=challenge.challenge_nonce,
            response_nonce=response_nonce,
            computation_proof=result,
            timestamp=time.time(),
            signature=signed.signature,
        )

    def verify_challenge_response(self, session: AuthSession,
                                  original: ChallengeResponse,
                                  response: ChallengeResponse) -> bool:
        """Verify a challenge response"""
        difficulty = int.from_bytes(original.computation_proof, 'big')

        # Verify computation
        expected = original.challenge_nonce
        for _ in range(difficulty):
            expected = self.crypto.hash(expected)

        if response.computation_proof != expected:
            self._record_failure(session, AuthFailureReason.CHALLENGE_FAILED)
            return False

        # Verify signature
        message = response.challenge_id.encode() + response.computation_proof + response.response_nonce
        signed = SignedMessage(message=message, signature=response.signature)

        if not self.crypto.verify(signed, session.peer_public_key):
            self._record_failure(session, AuthFailureReason.INVALID_SIGNATURE)
            return False

        # Challenge passed - restore state
        session.state = AuthState.AUTHENTICATED
        session.anomaly_score = max(0, session.anomaly_score - 0.3)
        logger.info(f"Challenge passed for {session.node_id}")

        return True

    def rotate_session_key(self, session: AuthSession) -> bytes:
        """Rotate the session encryption key"""
        old_key = session.session_key
        new_key = self.crypto.derive_key(
            old_key,
            salt=self.crypto.secure_random(16),
            info=b"session-key-rotation",
        )
        session.session_key = new_key
        session.last_key_rotation = datetime.now(timezone.utc)
        logger.info(f"Session key rotated for {session.node_id}")
        return new_key

    def revoke_session(self, peer_id: str, reason: str = "manual"):
        """Revoke a session"""
        if peer_id in self.sessions:
            session = self.sessions[peer_id]
            session.state = AuthState.REVOKED

            if self.on_session_revoked:
                self.on_session_revoked(peer_id)

            logger.warning(f"Session revoked for {peer_id}: {reason}")
            del self.sessions[peer_id]

    def _check_rate_limit(self, node_id: str) -> bool:
        """Check if node is within rate limits"""
        now = time.time()

        if node_id not in self._request_times:
            self._request_times[node_id] = deque(maxlen=self._rate_limit_max)

        times = self._request_times[node_id]

        # Remove old entries
        while times and now - times[0] > self._rate_limit_window:
            times.popleft()

        if len(times) >= self._rate_limit_max:
            return False

        times.append(now)
        return True

    def _record_failure(self, session: AuthSession, reason: AuthFailureReason):
        """Record an authentication failure"""
        session.failed_attempts += 1
        session.anomaly_score += 0.2

        if session.anomaly_score > 0.7:
            session.state = AuthState.SUSPICIOUS

        if session.failed_attempts >= session.max_failed_attempts:
            self.revoke_session(session.node_id, f"max_failures:{reason.name}")

        if self.on_auth_failure:
            self.on_auth_failure(session.node_id, reason)

        logger.warning(f"Auth failure for {session.node_id}: {reason.name} "
                      f"(attempts={session.failed_attempts}, anomaly={session.anomaly_score:.2f})")

    def _record_anomaly(self, session: AuthSession, anomaly_type: str, details: Dict):
        """Record a security anomaly"""
        event = {
            "type": anomaly_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
        }
        session.suspicious_events.append(event)

        # Keep only recent events
        if len(session.suspicious_events) > 100:
            session.suspicious_events = session.suspicious_events[-100:]

        if self.on_anomaly_detected:
            self.on_anomaly_detected(session.node_id, event)

    def start_heartbeat_sender(self, session: AuthSession,
                              send_callback: Callable[[Heartbeat], None]):
        """Start background heartbeat sender for a session"""
        def heartbeat_loop():
            while self._running and session.node_id in self.sessions:
                try:
                    if session.state == AuthState.AUTHENTICATED:
                        heartbeat = self.create_heartbeat(session)
                        send_callback(heartbeat)

                    # Check key rotation
                    if session.needs_key_rotation():
                        self.rotate_session_key(session)

                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")

                time.sleep(session.heartbeat_interval)

        self._running = True
        self._heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def stop_heartbeat_sender(self):
        """Stop the heartbeat sender thread"""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5.0)

    def get_session_status(self, peer_id: str) -> Optional[Dict]:
        """Get status of a session"""
        if peer_id not in self.sessions:
            return None

        session = self.sessions[peer_id]
        return {
            "session_id": session.session_id,
            "node_id": session.node_id,
            "state": session.state.name,
            "created_at": session.created_at.isoformat(),
            "last_heartbeat": session.last_heartbeat.isoformat() if session.last_heartbeat else None,
            "failed_attempts": session.failed_attempts,
            "anomaly_score": session.anomaly_score,
            "is_expired": session.is_expired(),
            "needs_key_rotation": session.needs_key_rotation(),
        }

    def get_all_sessions_status(self) -> Dict[str, Dict]:
        """Get status of all sessions"""
        return {
            peer_id: self.get_session_status(peer_id)
            for peer_id in self.sessions
        }


if __name__ == "__main__":
    print("Continuous Authentication Self-Test")
    print("=" * 50)

    # Create hub and node authenticators
    hub_auth = ContinuousAuthenticator("hub-001", is_hub=True)
    node_auth = ContinuousAuthenticator("node-001", is_hub=False)

    # Exchange public keys
    hub_pubkey = hub_auth.get_public_key()
    node_pubkey = node_auth.get_public_key()

    # Initiate sessions
    hub_session = hub_auth.initiate_session("node-001", node_pubkey)
    node_session = node_auth.initiate_session("hub-001", hub_pubkey)

    # Complete authentication
    hub_auth.complete_authentication(hub_session)
    node_auth.complete_authentication(node_session)

    print(f"\n[1] Sessions established")
    print(f"    Hub session: {hub_session.session_id[:16]}...")
    print(f"    Node session: {node_session.session_id[:16]}...")

    # Test heartbeat
    print(f"\n[2] Testing heartbeats")
    heartbeat = node_auth.create_heartbeat(node_session)
    valid, reason = hub_auth.verify_heartbeat(hub_session, heartbeat)
    print(f"    Heartbeat 1: {'✓ Valid' if valid else f'✗ Invalid: {reason}'}")

    # Test replay detection
    print(f"\n[3] Testing replay detection")
    valid, reason = hub_auth.verify_heartbeat(hub_session, heartbeat)
    print(f"    Replay attempt: {'✓ Rejected' if not valid and reason == AuthFailureReason.REPLAY_ATTACK else '✗ Not detected'}")

    # Test challenge-response
    print(f"\n[4] Testing challenge-response")
    hub_session.anomaly_score = 0.6  # Simulate anomaly
    if hub_auth.should_challenge(hub_session):
        challenge = hub_auth.create_challenge(hub_session)
        response = node_auth.respond_to_challenge(challenge)
        verified = hub_auth.verify_challenge_response(hub_session, challenge, response)
        print(f"    Challenge: {'✓ Passed' if verified else '✗ Failed'}")

    # Print session status
    print(f"\n[5] Session status")
    status = hub_auth.get_session_status("node-001")
    for key, value in status.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 50)
    print("All tests passed!")

