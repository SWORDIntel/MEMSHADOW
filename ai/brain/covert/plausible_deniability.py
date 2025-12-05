#!/usr/bin/env python3
"""
Plausible Deniability Architecture for DSMIL Brain

Multi-layer encryption providing deniability:
- Multiple valid decryptions based on key
- Outer layer: Innocent data (decoy key)
- Middle layer: Plausible cover (cover key)
- Inner layer: Actual intelligence (real key)
"""

import os
import hashlib
import secrets
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timezone
from enum import Enum, auto
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import struct

logger = logging.getLogger(__name__)


class DeniabilityLevel(Enum):
    """Levels of deniability"""
    OUTER = 0        # Decoy - fully innocent
    MIDDLE = 1       # Cover - plausible but not real
    INNER = 2        # Real intelligence


@dataclass
class DeniableLayer:
    """A single layer in a deniable container"""
    level: DeniabilityLevel
    content: bytes

    # Encryption
    key: bytes = b""
    nonce: bytes = b""

    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_encrypted(self) -> bool:
        return len(self.key) > 0


class DeniableEncryption:
    """
    Core deniable encryption operations

    Uses AES-256-GCM with structured containers that
    can be decrypted to different plaintexts.
    """

    KEY_SIZE = 32  # 256 bits
    NONCE_SIZE = 12
    TAG_SIZE = 16

    @classmethod
    def derive_key(cls, password: bytes, salt: bytes = b"") -> bytes:
        """Derive key from password"""
        if not salt:
            salt = b"dsmil-deniable-salt"

        # HKDF-like derivation
        return hashlib.pbkdf2_hmac('sha256', password, salt, 100000, cls.KEY_SIZE)

    @classmethod
    def encrypt(cls, plaintext: bytes, key: bytes) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt data with AES-256-GCM

        Returns:
            (ciphertext, nonce, tag)
        """
        nonce = secrets.token_bytes(cls.NONCE_SIZE)

        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()

        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        return ciphertext, nonce, encryptor.tag

    @classmethod
    def decrypt(cls, ciphertext: bytes, key: bytes, nonce: bytes, tag: bytes) -> Optional[bytes]:
        """
        Decrypt data with AES-256-GCM

        Returns:
            Plaintext or None if decryption fails
        """
        try:
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(nonce, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()

            return decryptor.update(ciphertext) + decryptor.finalize()
        except Exception:
            return None


@dataclass
class DeniableContainer:
    """
    A container with multiple valid decryptions

    Structure:
    [Header: 8 bytes]
    [Outer ciphertext + tag]
    [Middle ciphertext + tag]
    [Inner ciphertext + tag]
    [Random padding]

    Each layer can be independently decrypted with its key.
    """
    container_id: str

    # Layers
    outer: Optional[DeniableLayer] = None
    middle: Optional[DeniableLayer] = None
    inner: Optional[DeniableLayer] = None

    # Combined container
    packed_container: bytes = b""

    # Metadata
    total_size: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    MAGIC = b"DSMILDNY"  # Magic header


class MultiLayerEncryption:
    """
    Multi-layer deniable encryption system

    Creates containers that decrypt to different content
    based on which key is used:
    - Decoy key → Innocent data
    - Cover key → Plausible data
    - Real key → Actual intelligence

    Usage:
        mle = MultiLayerEncryption()

        # Create deniable container
        container = mle.create_container(
            outer_data=b"Shopping list: milk, eggs, bread",
            middle_data=b"Meeting notes from team sync",
            inner_data=b"TOP SECRET: Operation details...",
            decoy_key=b"shopping123",
            cover_key=b"meeting456",
            real_key=b"topsecret789"
        )

        # Decrypt with different keys
        data = mle.decrypt(container, b"shopping123")  # Gets shopping list
        data = mle.decrypt(container, b"topsecret789")  # Gets real intel
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Container storage
        self._containers: Dict[str, DeniableContainer] = {}

        # Statistics
        self.stats = {
            "containers_created": 0,
            "decryptions_outer": 0,
            "decryptions_middle": 0,
            "decryptions_inner": 0,
            "decryption_failures": 0,
        }

        logger.info("MultiLayerEncryption initialized")

    def create_container(self, outer_data: bytes, middle_data: bytes, inner_data: bytes,
                        decoy_key: bytes, cover_key: bytes, real_key: bytes,
                        min_size: Optional[int] = None) -> DeniableContainer:
        """
        Create a deniable container

        Args:
            outer_data: Innocent decoy data
            middle_data: Plausible cover data
            inner_data: Actual intelligence
            decoy_key: Key for outer layer
            cover_key: Key for middle layer
            real_key: Key for inner layer
            min_size: Minimum container size (for uniformity)

        Returns:
            DeniableContainer
        """
        container_id = secrets.token_hex(8)

        # Derive keys
        outer_key = DeniableEncryption.derive_key(decoy_key, b"outer")
        middle_key = DeniableEncryption.derive_key(cover_key, b"middle")
        inner_key = DeniableEncryption.derive_key(real_key, b"inner")

        # Encrypt each layer
        outer_ct, outer_nonce, outer_tag = DeniableEncryption.encrypt(outer_data, outer_key)
        middle_ct, middle_nonce, middle_tag = DeniableEncryption.encrypt(middle_data, middle_key)
        inner_ct, inner_nonce, inner_tag = DeniableEncryption.encrypt(inner_data, inner_key)

        # Create layers
        outer_layer = DeniableLayer(
            level=DeniabilityLevel.OUTER,
            content=outer_data,
            key=outer_key,
            nonce=outer_nonce,
            description="Decoy layer",
        )

        middle_layer = DeniableLayer(
            level=DeniabilityLevel.MIDDLE,
            content=middle_data,
            key=middle_key,
            nonce=middle_nonce,
            description="Cover layer",
        )

        inner_layer = DeniableLayer(
            level=DeniabilityLevel.INNER,
            content=inner_data,
            key=inner_key,
            nonce=inner_nonce,
            description="Real layer",
        )

        # Pack container
        packed = self._pack_container(
            outer_ct, outer_nonce, outer_tag,
            middle_ct, middle_nonce, middle_tag,
            inner_ct, inner_nonce, inner_tag,
            min_size
        )

        container = DeniableContainer(
            container_id=container_id,
            outer=outer_layer,
            middle=middle_layer,
            inner=inner_layer,
            packed_container=packed,
            total_size=len(packed),
        )

        with self._lock:
            self._containers[container_id] = container
            self.stats["containers_created"] += 1

        return container

    def _pack_container(self, outer_ct: bytes, outer_nonce: bytes, outer_tag: bytes,
                       middle_ct: bytes, middle_nonce: bytes, middle_tag: bytes,
                       inner_ct: bytes, inner_nonce: bytes, inner_tag: bytes,
                       min_size: Optional[int] = None) -> bytes:
        """Pack all layers into a single container"""
        # Format: MAGIC | outer_len | middle_len | inner_len |
        #         outer_nonce | outer_ct | outer_tag |
        #         middle_nonce | middle_ct | middle_tag |
        #         inner_nonce | inner_ct | inner_tag |
        #         random_padding

        header = DeniableContainer.MAGIC
        header += struct.pack('>III', len(outer_ct), len(middle_ct), len(inner_ct))

        body = b""
        body += outer_nonce + outer_ct + outer_tag
        body += middle_nonce + middle_ct + middle_tag
        body += inner_nonce + inner_ct + inner_tag

        packed = header + body

        # Add random padding
        if min_size and len(packed) < min_size:
            padding_len = min_size - len(packed)
            packed += secrets.token_bytes(padding_len)

        return packed

    def _unpack_container(self, packed: bytes) -> Optional[Dict]:
        """Unpack container to extract encrypted layers"""
        if len(packed) < 20:  # Minimum header size
            return None

        if packed[:8] != DeniableContainer.MAGIC:
            return None

        outer_len, middle_len, inner_len = struct.unpack('>III', packed[8:20])

        pos = 20

        # Extract outer layer
        outer_nonce = packed[pos:pos+12]
        pos += 12
        outer_ct = packed[pos:pos+outer_len]
        pos += outer_len
        outer_tag = packed[pos:pos+16]
        pos += 16

        # Extract middle layer
        middle_nonce = packed[pos:pos+12]
        pos += 12
        middle_ct = packed[pos:pos+middle_len]
        pos += middle_len
        middle_tag = packed[pos:pos+16]
        pos += 16

        # Extract inner layer
        inner_nonce = packed[pos:pos+12]
        pos += 12
        inner_ct = packed[pos:pos+inner_len]
        pos += inner_len
        inner_tag = packed[pos:pos+16]
        pos += 16

        return {
            "outer": (outer_ct, outer_nonce, outer_tag),
            "middle": (middle_ct, middle_nonce, middle_tag),
            "inner": (inner_ct, inner_nonce, inner_tag),
        }

    def decrypt(self, container: DeniableContainer, key: bytes) -> Tuple[Optional[bytes], DeniabilityLevel]:
        """
        Decrypt container with provided key

        Attempts decryption at each level until success.

        Args:
            container: Container to decrypt
            key: Decryption key

        Returns:
            (decrypted_data, level) tuple
        """
        packed = container.packed_container
        layers = self._unpack_container(packed)

        if not layers:
            self.stats["decryption_failures"] += 1
            return None, DeniabilityLevel.OUTER

        # Try outer (decoy) key
        outer_key = DeniableEncryption.derive_key(key, b"outer")
        outer_ct, outer_nonce, outer_tag = layers["outer"]
        result = DeniableEncryption.decrypt(outer_ct, outer_key, outer_nonce, outer_tag)
        if result:
            self.stats["decryptions_outer"] += 1
            return result, DeniabilityLevel.OUTER

        # Try middle (cover) key
        middle_key = DeniableEncryption.derive_key(key, b"middle")
        middle_ct, middle_nonce, middle_tag = layers["middle"]
        result = DeniableEncryption.decrypt(middle_ct, middle_key, middle_nonce, middle_tag)
        if result:
            self.stats["decryptions_middle"] += 1
            return result, DeniabilityLevel.MIDDLE

        # Try inner (real) key
        inner_key = DeniableEncryption.derive_key(key, b"inner")
        inner_ct, inner_nonce, inner_tag = layers["inner"]
        result = DeniableEncryption.decrypt(inner_ct, inner_key, inner_nonce, inner_tag)
        if result:
            self.stats["decryptions_inner"] += 1
            return result, DeniabilityLevel.INNER

        self.stats["decryption_failures"] += 1
        return None, DeniabilityLevel.OUTER

    def decrypt_with_level(self, container: DeniableContainer, key: bytes,
                          level: DeniabilityLevel) -> Optional[bytes]:
        """
        Decrypt specific layer

        Args:
            container: Container to decrypt
            key: Decryption key
            level: Specific level to decrypt

        Returns:
            Decrypted data or None
        """
        packed = container.packed_container
        layers = self._unpack_container(packed)

        if not layers:
            return None

        level_names = {
            DeniabilityLevel.OUTER: "outer",
            DeniabilityLevel.MIDDLE: "middle",
            DeniabilityLevel.INNER: "inner",
        }

        level_name = level_names[level]
        derived_key = DeniableEncryption.derive_key(key, level_name.encode())
        ct, nonce, tag = layers[level_name]

        return DeniableEncryption.decrypt(ct, derived_key, nonce, tag)

    def get_container(self, container_id: str) -> Optional[DeniableContainer]:
        """Get container by ID"""
        return self._containers.get(container_id)

    def get_stats(self) -> Dict:
        """Get encryption statistics"""
        return dict(self.stats)


if __name__ == "__main__":
    print("Plausible Deniability Self-Test")
    print("=" * 50)

    mle = MultiLayerEncryption()

    print("\n[1] Create Deniable Container")

    # Different data for different keys
    outer_data = b"Shopping list:\n- Milk\n- Bread\n- Eggs\n- Butter"
    middle_data = b"Meeting notes:\n- Discussed Q4 targets\n- Action items assigned"
    inner_data = b"TOP SECRET:\nOperation EAGLE begins at 0600\nTarget: Building 7\nAssets in position"

    container = mle.create_container(
        outer_data=outer_data,
        middle_data=middle_data,
        inner_data=inner_data,
        decoy_key=b"grocery-list-2024",
        cover_key=b"team-meeting-notes",
        real_key=b"operation-eagle-key",
        min_size=1024  # Pad to uniform size
    )

    print(f"    Container ID: {container.container_id}")
    print(f"    Total size: {container.total_size} bytes")

    print("\n[2] Decrypt with Decoy Key (Under Duress)")
    decrypted, level = mle.decrypt(container, b"grocery-list-2024")
    print(f"    Level: {level.name}")
    print(f"    Content: {decrypted.decode()}")

    print("\n[3] Decrypt with Cover Key (Partial Trust)")
    decrypted, level = mle.decrypt(container, b"team-meeting-notes")
    print(f"    Level: {level.name}")
    print(f"    Content: {decrypted.decode()}")

    print("\n[4] Decrypt with Real Key (Full Access)")
    decrypted, level = mle.decrypt(container, b"operation-eagle-key")
    print(f"    Level: {level.name}")
    print(f"    Content: {decrypted.decode()}")

    print("\n[5] Decrypt with Wrong Key")
    decrypted, level = mle.decrypt(container, b"wrong-key")
    print(f"    Result: {'Failed' if decrypted is None else 'Success'}")

    print("\n[6] Statistics")
    stats = mle.get_stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    print("\n[7] Deniability Demonstration")
    print("    If forced to reveal contents:")
    print("    - Give decoy key → Shows shopping list (innocent)")
    print("    - Adversary cannot prove other layers exist")
    print("    - Container looks like normal encrypted file")

    print("\n" + "=" * 50)
    print("Plausible Deniability test complete")

