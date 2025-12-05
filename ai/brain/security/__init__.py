#!/usr/bin/env python3
"""
DSMIL Brain Security Layer

CNSA 2.0 compliant military-grade security:
- AES-256-GCM symmetric encryption
- P-384 ECDSA signatures
- SHA-384 hashing
- X25519/Kyber768 hybrid key exchange (PQC-ready)
- HKDF-SHA384 key derivation
- Continuous authentication
- Tamper detection with self-destruct
"""

from .cnsa_crypto import (
    CNSACrypto,
    KeyPair,
    EncryptedPayload,
    SignedMessage,
    HybridKeyExchange,
)

from .continuous_auth import (
    ContinuousAuthenticator,
    AuthSession,
    Heartbeat,
    ChallengeResponse,
)

from .tamper_detection import (
    TamperDetector,
    TamperEvidence,
    IntegrityCanary,
    TamperType,
)

from .self_destruct import (
    SelfDestructProtocol,
    EmergencyIntelCapture,
    SecureWipe,
)

from .tpm_optional_hsm import (
    SecureKeyStore,
    TPMBackend,
    SoftwareHSMBackend,
)

__all__ = [
    # Crypto
    "CNSACrypto",
    "KeyPair",
    "EncryptedPayload",
    "SignedMessage",
    "HybridKeyExchange",
    # Auth
    "ContinuousAuthenticator",
    "AuthSession",
    "Heartbeat",
    "ChallengeResponse",
    # Tamper
    "TamperDetector",
    "TamperEvidence",
    "IntegrityCanary",
    "TamperType",
    # Self-destruct
    "SelfDestructProtocol",
    "EmergencyIntelCapture",
    "SecureWipe",
    # Key storage
    "SecureKeyStore",
    "TPMBackend",
    "SoftwareHSMBackend",
]

