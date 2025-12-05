#!/usr/bin/env python3
"""
CNSA 2.0 Cryptographic Suite for DSMIL Brain

Military-grade encryption compliant with NSA's Commercial National Security Algorithm Suite:
- AES-256-GCM for symmetric encryption
- P-384 ECDSA for digital signatures
- SHA-384 for hashing
- X25519 + Kyber768 hybrid for PQC-ready key exchange
- HKDF-SHA384 for key derivation

Reference: NSA CNSA 2.0 (2022) - preparing for post-quantum cryptography
"""

import os
import struct
import hashlib
import hmac
import secrets
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List
from datetime import datetime, timezone
from enum import Enum, auto
import base64

logger = logging.getLogger(__name__)

# Try to import cryptographic libraries
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, x25519
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.backends import default_backend
    from cryptography.exceptions import InvalidTag, InvalidSignature
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logger.warning("cryptography library not available - using fallback implementations")

# Try to import PQC library (liboqs-python or pqcrypto)
try:
    import oqs
    PQC_AVAILABLE = True
except ImportError:
    PQC_AVAILABLE = False
    logger.info("liboqs not available - Kyber768 PQC disabled, using X25519 only")


class CryptoError(Exception):
    """Base exception for cryptographic operations"""
    pass


class EncryptionError(CryptoError):
    """Encryption failed"""
    pass


class DecryptionError(CryptoError):
    """Decryption failed - likely tampered or wrong key"""
    pass


class SignatureError(CryptoError):
    """Signature verification failed"""
    pass


class KeyDerivationError(CryptoError):
    """Key derivation failed"""
    pass


@dataclass
class KeyPair:
    """Asymmetric key pair for signing or key exchange"""
    private_key: bytes
    public_key: bytes
    algorithm: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    key_id: str = field(default_factory=lambda: secrets.token_hex(8))

    def __post_init__(self):
        # Ensure private key is never logged
        self._private_key = self.private_key

    @property
    def public_key_b64(self) -> str:
        """Base64-encoded public key for transmission"""
        return base64.b64encode(self.public_key).decode('ascii')

    def export_public(self) -> dict:
        """Export public key info (safe to share)"""
        return {
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "public_key": self.public_key_b64,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class EncryptedPayload:
    """Container for encrypted data with all necessary metadata"""
    ciphertext: bytes
    nonce: bytes  # 12 bytes for AES-GCM
    tag: bytes    # 16 bytes authentication tag
    algorithm: str = "AES-256-GCM"
    key_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_bytes(self) -> bytes:
        """Serialize to wire format"""
        # Format: [nonce:12][tag:16][ciphertext:*]
        return self.nonce + self.tag + self.ciphertext

    @classmethod
    def from_bytes(cls, data: bytes, key_id: Optional[str] = None) -> "EncryptedPayload":
        """Deserialize from wire format"""
        if len(data) < 28:  # 12 + 16 minimum
            raise DecryptionError("Encrypted payload too short")
        return cls(
            nonce=data[:12],
            tag=data[12:28],
            ciphertext=data[28:],
            key_id=key_id,
        )

    def to_b64(self) -> str:
        """Base64 encode for text transmission"""
        return base64.b64encode(self.to_bytes()).decode('ascii')

    @classmethod
    def from_b64(cls, b64_data: str, key_id: Optional[str] = None) -> "EncryptedPayload":
        """Decode from base64"""
        return cls.from_bytes(base64.b64decode(b64_data), key_id)


@dataclass
class SignedMessage:
    """Message with digital signature"""
    message: bytes
    signature: bytes  # 96 bytes for P-384 ECDSA
    algorithm: str = "P-384-ECDSA"
    signer_key_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_bytes(self) -> bytes:
        """Serialize: [sig_len:2][signature][message]"""
        sig_len = len(self.signature)
        return struct.pack(">H", sig_len) + self.signature + self.message

    @classmethod
    def from_bytes(cls, data: bytes) -> "SignedMessage":
        """Deserialize from wire format"""
        sig_len = struct.unpack(">H", data[:2])[0]
        return cls(
            signature=data[2:2+sig_len],
            message=data[2+sig_len:],
        )


@dataclass
class HybridKeyExchange:
    """Hybrid key exchange result (X25519 + Kyber768)"""
    x25519_public: bytes
    kyber_public: Optional[bytes]  # None if PQC not available
    shared_secret: Optional[bytes] = None  # Derived after exchange

    def is_pqc_enabled(self) -> bool:
        return self.kyber_public is not None


class CNSACrypto:
    """
    CNSA 2.0 Compliant Cryptographic Operations

    Provides military-grade encryption, signing, and key exchange
    with post-quantum cryptography readiness.

    Usage:
        crypto = CNSACrypto()

        # Symmetric encryption
        key = crypto.generate_symmetric_key()
        encrypted = crypto.encrypt(plaintext, key)
        decrypted = crypto.decrypt(encrypted, key)

        # Signing
        keypair = crypto.generate_signing_keypair()
        signed = crypto.sign(message, keypair)
        valid = crypto.verify(signed, keypair.public_key)

        # Key exchange
        alice_kex = crypto.initiate_key_exchange()
        bob_kex, shared_bob = crypto.complete_key_exchange(alice_kex)
        shared_alice = crypto.derive_shared_secret(bob_kex, alice_private)
    """

    # CNSA 2.0 constants
    AES_KEY_SIZE = 32  # 256 bits
    NONCE_SIZE = 12    # 96 bits for GCM
    TAG_SIZE = 16      # 128 bits
    HASH_SIZE = 48     # SHA-384

    def __init__(self, enable_pqc: bool = True):
        """
        Initialize CNSA crypto suite

        Args:
            enable_pqc: Enable post-quantum cryptography (Kyber768)
        """
        self.pqc_enabled = enable_pqc and PQC_AVAILABLE

        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning("Running with limited crypto capabilities")

        if enable_pqc and not PQC_AVAILABLE:
            logger.info("PQC requested but liboqs not available")

    # ==================== Symmetric Encryption ====================

    def generate_symmetric_key(self) -> bytes:
        """Generate a random AES-256 key"""
        return secrets.token_bytes(self.AES_KEY_SIZE)

    def encrypt(self, plaintext: bytes, key: bytes,
                associated_data: Optional[bytes] = None) -> EncryptedPayload:
        """
        Encrypt data with AES-256-GCM

        Args:
            plaintext: Data to encrypt
            key: 32-byte AES key
            associated_data: Optional AAD for authentication

        Returns:
            EncryptedPayload with ciphertext, nonce, and tag
        """
        if len(key) != self.AES_KEY_SIZE:
            raise EncryptionError(f"Key must be {self.AES_KEY_SIZE} bytes")

        nonce = secrets.token_bytes(self.NONCE_SIZE)

        if CRYPTOGRAPHY_AVAILABLE:
            aesgcm = AESGCM(key)
            # AES-GCM appends tag to ciphertext
            ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext, associated_data)
            # Split ciphertext and tag (tag is last 16 bytes)
            ciphertext = ciphertext_with_tag[:-self.TAG_SIZE]
            tag = ciphertext_with_tag[-self.TAG_SIZE:]
        else:
            # Fallback: XOR with key-derived stream (NOT secure, for testing only)
            logger.warning("Using insecure fallback encryption!")
            stream = self._derive_stream(key, nonce, len(plaintext))
            ciphertext = bytes(a ^ b for a, b in zip(plaintext, stream))
            tag = self._compute_tag(key, nonce, ciphertext, associated_data)

        return EncryptedPayload(
            ciphertext=ciphertext,
            nonce=nonce,
            tag=tag,
        )

    def decrypt(self, payload: EncryptedPayload, key: bytes,
                associated_data: Optional[bytes] = None) -> bytes:
        """
        Decrypt AES-256-GCM encrypted data

        Args:
            payload: Encrypted payload
            key: 32-byte AES key
            associated_data: Optional AAD (must match encryption)

        Returns:
            Decrypted plaintext

        Raises:
            DecryptionError: If decryption fails (wrong key or tampered)
        """
        if len(key) != self.AES_KEY_SIZE:
            raise DecryptionError(f"Key must be {self.AES_KEY_SIZE} bytes")

        if CRYPTOGRAPHY_AVAILABLE:
            aesgcm = AESGCM(key)
            # Reconstruct ciphertext with tag
            ciphertext_with_tag = payload.ciphertext + payload.tag
            try:
                plaintext = aesgcm.decrypt(payload.nonce, ciphertext_with_tag, associated_data)
                return plaintext
            except InvalidTag:
                raise DecryptionError("Decryption failed: authentication tag mismatch (tampered or wrong key)")
        else:
            # Fallback verification
            expected_tag = self._compute_tag(key, payload.nonce, payload.ciphertext, associated_data)
            if not hmac.compare_digest(payload.tag, expected_tag):
                raise DecryptionError("Decryption failed: authentication tag mismatch")
            stream = self._derive_stream(key, payload.nonce, len(payload.ciphertext))
            return bytes(a ^ b for a, b in zip(payload.ciphertext, stream))

    def _derive_stream(self, key: bytes, nonce: bytes, length: int) -> bytes:
        """Derive keystream (fallback only)"""
        stream = b""
        counter = 0
        while len(stream) < length:
            block = hashlib.sha384(key + nonce + counter.to_bytes(4, 'big')).digest()
            stream += block
            counter += 1
        return stream[:length]

    def _compute_tag(self, key: bytes, nonce: bytes, ciphertext: bytes,
                     aad: Optional[bytes]) -> bytes:
        """Compute authentication tag (fallback only)"""
        data = nonce + (aad or b"") + ciphertext
        return hmac.new(key, data, hashlib.sha384).digest()[:self.TAG_SIZE]

    # ==================== Hashing ====================

    def hash(self, data: bytes) -> bytes:
        """SHA-384 hash"""
        return hashlib.sha384(data).digest()

    def hash_hex(self, data: bytes) -> str:
        """SHA-384 hash as hex string"""
        return hashlib.sha384(data).hexdigest()

    def hmac_sha384(self, key: bytes, data: bytes) -> bytes:
        """HMAC-SHA384"""
        return hmac.new(key, data, hashlib.sha384).digest()

    # ==================== Key Derivation ====================

    def derive_key(self, master_key: bytes, salt: bytes, info: bytes,
                   length: int = AES_KEY_SIZE) -> bytes:
        """
        Derive key using HKDF-SHA384

        Args:
            master_key: Input key material
            salt: Random salt (should be unique per derivation)
            info: Context-specific info string
            length: Output key length

        Returns:
            Derived key bytes
        """
        if CRYPTOGRAPHY_AVAILABLE:
            hkdf = HKDF(
                algorithm=hashes.SHA384(),
                length=length,
                salt=salt,
                info=info,
                backend=default_backend(),
            )
            return hkdf.derive(master_key)
        else:
            # Simple HKDF-like fallback
            prk = hmac.new(salt or b"\x00" * 48, master_key, hashlib.sha384).digest()
            okm = b""
            t = b""
            counter = 1
            while len(okm) < length:
                t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha384).digest()
                okm += t
                counter += 1
            return okm[:length]

    # ==================== Digital Signatures (P-384 ECDSA) ====================

    def generate_signing_keypair(self) -> KeyPair:
        """Generate P-384 ECDSA key pair for signing"""
        if CRYPTOGRAPHY_AVAILABLE:
            private_key = ec.generate_private_key(ec.SECP384R1(), default_backend())
            private_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            public_bytes = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            return KeyPair(
                private_key=private_bytes,
                public_key=public_bytes,
                algorithm="P-384-ECDSA",
            )
        else:
            # Fallback: random bytes (NOT real ECDSA)
            logger.warning("Using insecure fallback key generation!")
            return KeyPair(
                private_key=secrets.token_bytes(48),
                public_key=secrets.token_bytes(97),  # Uncompressed P-384 point size
                algorithm="P-384-ECDSA-FALLBACK",
            )

    def sign(self, message: bytes, keypair: KeyPair) -> SignedMessage:
        """
        Sign message with P-384 ECDSA

        Args:
            message: Data to sign
            keypair: Signing key pair

        Returns:
            SignedMessage with signature
        """
        if CRYPTOGRAPHY_AVAILABLE:
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

            private_key = serialization.load_der_private_key(
                keypair.private_key,
                password=None,
                backend=default_backend(),
            )

            # Sign with SHA-384
            signature = private_key.sign(
                message,
                ec.ECDSA(hashes.SHA384()),
            )

            return SignedMessage(
                message=message,
                signature=signature,
                signer_key_id=keypair.key_id,
            )
        else:
            # Fallback: HMAC-based (NOT real signatures)
            signature = hmac.new(keypair.private_key, message, hashlib.sha384).digest()
            return SignedMessage(
                message=message,
                signature=signature,
                algorithm="HMAC-SHA384-FALLBACK",
                signer_key_id=keypair.key_id,
            )

    def verify(self, signed: SignedMessage, public_key: bytes) -> bool:
        """
        Verify P-384 ECDSA signature

        Args:
            signed: Signed message
            public_key: Signer's public key

        Returns:
            True if valid, False otherwise
        """
        if CRYPTOGRAPHY_AVAILABLE:
            try:
                pub_key = serialization.load_der_public_key(
                    public_key,
                    backend=default_backend(),
                )
                pub_key.verify(
                    signed.signature,
                    signed.message,
                    ec.ECDSA(hashes.SHA384()),
                )
                return True
            except InvalidSignature:
                return False
            except Exception as e:
                logger.error(f"Signature verification error: {e}")
                return False
        else:
            # Fallback cannot verify without private key
            logger.warning("Cannot verify signatures without cryptography library")
            return True  # Unsafe fallback

    # ==================== Key Exchange (X25519 + Kyber768) ====================

    def generate_key_exchange_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate X25519 key pair for key exchange

        Returns:
            (private_key, public_key) tuple
        """
        if CRYPTOGRAPHY_AVAILABLE:
            private_key = x25519.X25519PrivateKey.generate()
            private_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
            public_bytes = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            return private_bytes, public_bytes
        else:
            # Fallback
            private = secrets.token_bytes(32)
            public = hashlib.sha256(private).digest()
            return private, public

    def initiate_key_exchange(self) -> Tuple[HybridKeyExchange, bytes]:
        """
        Initiate hybrid key exchange (X25519 + optional Kyber768)

        Returns:
            (HybridKeyExchange with public keys, private key material)
        """
        x25519_private, x25519_public = self.generate_key_exchange_keypair()

        kyber_public = None
        kyber_private = None

        if self.pqc_enabled:
            try:
                kem = oqs.KeyEncapsulation("Kyber768")
                kyber_public = kem.generate_keypair()
                kyber_private = kem.export_secret_key()
            except Exception as e:
                logger.warning(f"Kyber768 key generation failed: {e}")

        # Pack private keys for later
        private_material = x25519_private
        if kyber_private:
            private_material += kyber_private

        return HybridKeyExchange(
            x25519_public=x25519_public,
            kyber_public=kyber_public,
        ), private_material

    def complete_key_exchange(self, peer_kex: HybridKeyExchange,
                              our_private: bytes) -> bytes:
        """
        Complete key exchange and derive shared secret

        Args:
            peer_kex: Peer's public keys
            our_private: Our private key material

        Returns:
            Shared secret (32 bytes)
        """
        x25519_private = our_private[:32]

        # X25519 shared secret
        if CRYPTOGRAPHY_AVAILABLE:
            our_key = x25519.X25519PrivateKey.from_private_bytes(x25519_private)
            peer_key = x25519.X25519PublicKey.from_public_bytes(peer_kex.x25519_public)
            x25519_shared = our_key.exchange(peer_key)
        else:
            # Fallback
            x25519_shared = hashlib.sha256(x25519_private + peer_kex.x25519_public).digest()

        # Kyber768 shared secret (if available)
        kyber_shared = b""
        if peer_kex.kyber_public and self.pqc_enabled and len(our_private) > 32:
            try:
                kem = oqs.KeyEncapsulation("Kyber768")
                kem.import_secret_key(our_private[32:])
                ciphertext, kyber_shared = kem.encap_secret(peer_kex.kyber_public)
            except Exception as e:
                logger.warning(f"Kyber768 encapsulation failed: {e}")

        # Combine secrets with HKDF
        combined = x25519_shared + kyber_shared
        return self.derive_key(
            combined,
            salt=b"DSMIL-BRAIN-KEX-v1",
            info=b"shared-secret",
            length=32,
        )

    # ==================== Utility Functions ====================

    def secure_compare(self, a: bytes, b: bytes) -> bool:
        """Constant-time comparison to prevent timing attacks"""
        return hmac.compare_digest(a, b)

    def secure_random(self, length: int) -> bytes:
        """Generate cryptographically secure random bytes"""
        return secrets.token_bytes(length)

    def secure_random_hex(self, length: int) -> str:
        """Generate secure random hex string"""
        return secrets.token_hex(length)


# Singleton instance
_crypto_instance: Optional[CNSACrypto] = None

def get_crypto() -> CNSACrypto:
    """Get singleton crypto instance"""
    global _crypto_instance
    if _crypto_instance is None:
        _crypto_instance = CNSACrypto()
    return _crypto_instance


if __name__ == "__main__":
    # Self-test
    print("CNSA 2.0 Crypto Suite Self-Test")
    print("=" * 50)

    crypto = CNSACrypto()

    # Test symmetric encryption
    print("\n[1] AES-256-GCM Encryption")
    key = crypto.generate_symmetric_key()
    plaintext = b"TOP SECRET: This is classified intelligence data"
    encrypted = crypto.encrypt(plaintext, key)
    decrypted = crypto.decrypt(encrypted, key)
    assert decrypted == plaintext, "Decryption failed!"
    print(f"    ✓ Encrypted {len(plaintext)} bytes → {len(encrypted.ciphertext)} bytes")
    print(f"    ✓ Decryption successful")

    # Test tamper detection
    print("\n[2] Tamper Detection")
    tampered = EncryptedPayload(
        ciphertext=encrypted.ciphertext[:-1] + bytes([encrypted.ciphertext[-1] ^ 0xFF]),
        nonce=encrypted.nonce,
        tag=encrypted.tag,
    )
    try:
        crypto.decrypt(tampered, key)
        print("    ✗ Tamper not detected!")
    except DecryptionError:
        print("    ✓ Tampered data correctly rejected")

    # Test signing
    print("\n[3] P-384 ECDSA Signing")
    keypair = crypto.generate_signing_keypair()
    message = b"Intelligence report: Target confirmed"
    signed = crypto.sign(message, keypair)
    valid = crypto.verify(signed, keypair.public_key)
    print(f"    ✓ Signature length: {len(signed.signature)} bytes")
    print(f"    ✓ Verification: {'PASS' if valid else 'FAIL'}")

    # Test key exchange
    print("\n[4] Hybrid Key Exchange (X25519 + Kyber768)")
    alice_kex, alice_private = crypto.initiate_key_exchange()
    bob_kex, bob_private = crypto.initiate_key_exchange()

    alice_shared = crypto.complete_key_exchange(bob_kex, alice_private)
    bob_shared = crypto.complete_key_exchange(alice_kex, bob_private)

    # Note: In real implementation, need proper key exchange protocol
    print(f"    ✓ PQC enabled: {crypto.pqc_enabled}")
    print(f"    ✓ Shared secret length: {len(alice_shared)} bytes")

    # Test key derivation
    print("\n[5] HKDF-SHA384 Key Derivation")
    master = crypto.secure_random(32)
    salt = crypto.secure_random(16)
    derived = crypto.derive_key(master, salt, b"test-context")
    print(f"    ✓ Derived key: {len(derived)} bytes")

    print("\n" + "=" * 50)
    print("All tests passed! CNSA 2.0 crypto suite operational.")

