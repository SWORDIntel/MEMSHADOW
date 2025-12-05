#!/usr/bin/env python3
"""
DSMIL Brain Homomorphic Intelligence Layer

Privacy-preserving computation:
- Encrypted Compute: Process data without decryption
- Zero-Knowledge Proofs: Prove facts without revealing data
"""

from .encrypted_compute import (
    HomomorphicIntelligence,
    EncryptedVector,
    EncryptedQuery,
    EncryptedResult,
)

from .zkp import (
    ZeroKnowledgeProver,
    ZKProof,
    ProofStatement,
)

__all__ = [
    "HomomorphicIntelligence", "EncryptedVector", "EncryptedQuery", "EncryptedResult",
    "ZeroKnowledgeProver", "ZKProof", "ProofStatement",
]

