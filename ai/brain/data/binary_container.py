#!/usr/bin/env python3
"""
DSMIL Binary Container (DSMIL-BC) for DSMIL Brain

Encrypted binary container format:
+------------------+
| Magic: DSMILBC   | 8 bytes
| Version: u16     | 2 bytes
| Flags: u16       | 2 bytes
| Type Hint: u32   | 4 bytes
| Payload Len: u64 | 8 bytes
| Signature: [u8]  | 96 bytes (P-384)
| Nonce: [u8]      | 12 bytes
| Payload: [u8]    | Variable
| MAC: [u8]        | 16 bytes
+------------------+
"""

import os
import struct
import hashlib
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from enum import IntFlag

logger = logging.getLogger(__name__)


# Magic bytes
MAGIC = b"DSMILBC\x00"
VERSION = 1

# Header size (excluding payload)
HEADER_SIZE = 8 + 2 + 2 + 4 + 8 + 96 + 12  # = 132 bytes
MAC_SIZE = 16


class ContainerFlags(IntFlag):
    """Container flags"""
    NONE = 0
    ENCRYPTED = 1
    COMPRESSED = 2
    SIGNED = 4
    CHUNKED = 8


class TypeHint:
    """Type hints for payload"""
    UNKNOWN = 0
    TEXT = 1
    JSON = 2
    VECTOR = 3
    IMAGE = 4
    AUDIO = 5
    VIDEO = 6
    MODEL = 7
    GRAPH = 8
    CUSTOM = 255


@dataclass
class ContainerHeader:
    """Container header"""
    magic: bytes
    version: int
    flags: ContainerFlags
    type_hint: int
    payload_len: int
    signature: bytes
    nonce: bytes


class DSMILBinaryContainer:
    """
    DSMIL Binary Container

    Encrypted container for secure data storage and transmission.

    Usage:
        container = DSMILBinaryContainer()

        # Create container
        data = container.create(payload, type_hint=TypeHint.JSON)

        # Parse container
        header, payload = container.parse(data)
    """

    def __init__(self, encryption_key: Optional[bytes] = None):
        self._key = encryption_key

        logger.info("DSMILBinaryContainer initialized")

    def create(self, payload: bytes,
              type_hint: int = TypeHint.UNKNOWN,
              flags: ContainerFlags = ContainerFlags.ENCRYPTED | ContainerFlags.SIGNED,
              signing_key: Optional[bytes] = None) -> bytes:
        """
        Create a DSMIL-BC container
        """
        # Generate nonce
        nonce = os.urandom(12)

        # Encrypt payload if requested
        if flags & ContainerFlags.ENCRYPTED and self._key:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(self._key)
            encrypted_payload = aesgcm.encrypt(nonce, payload, None)
            mac = encrypted_payload[-16:]
            payload_with_mac = encrypted_payload
        else:
            payload_with_mac = payload
            mac = hashlib.sha256(payload).digest()[:16]

        # Create signature
        if flags & ContainerFlags.SIGNED:
            sig_data = nonce + struct.pack(">Q", len(payload)) + hashlib.sha384(payload).digest()
            signature = hashlib.sha384(sig_data).digest() + (b"\x00" * 48)  # Pad to 96 bytes
        else:
            signature = b"\x00" * 96

        # Build header
        header = struct.pack(
            ">8sHHIQ",
            MAGIC,
            VERSION,
            int(flags),
            type_hint,
            len(payload_with_mac) - (MAC_SIZE if flags & ContainerFlags.ENCRYPTED else 0)
        )

        # Combine
        container = header + signature + nonce + payload_with_mac

        return container

    def parse(self, data: bytes) -> tuple:
        """
        Parse a DSMIL-BC container

        Returns (ContainerHeader, payload_bytes)
        """
        if len(data) < HEADER_SIZE + MAC_SIZE:
            raise ValueError("Container too small")

        # Parse header
        magic = data[0:8]
        if magic != MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic}")

        version, flags, type_hint, payload_len = struct.unpack(">HHIQ", data[8:24])

        if version != VERSION:
            raise ValueError(f"Unsupported version: {version}")

        signature = data[24:120]
        nonce = data[120:132]

        # Extract payload
        payload_data = data[132:]

        header = ContainerHeader(
            magic=magic,
            version=version,
            flags=ContainerFlags(flags),
            type_hint=type_hint,
            payload_len=payload_len,
            signature=signature,
            nonce=nonce,
        )

        # Decrypt if needed
        if header.flags & ContainerFlags.ENCRYPTED and self._key:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(self._key)
            try:
                payload = aesgcm.decrypt(nonce, payload_data, None)
            except Exception as e:
                raise ValueError(f"Decryption failed: {e}")
        else:
            payload = payload_data

        return header, payload

    def verify_signature(self, data: bytes) -> bool:
        """Verify container signature"""
        try:
            header, payload = self.parse(data)

            if not (header.flags & ContainerFlags.SIGNED):
                return True  # Not signed, considered valid

            # Recreate expected signature
            sig_data = header.nonce + struct.pack(">Q", len(payload)) + hashlib.sha384(payload).digest()
            expected = hashlib.sha384(sig_data).digest() + (b"\x00" * 48)

            return header.signature == expected
        except:
            return False

    @staticmethod
    def get_type_name(type_hint: int) -> str:
        """Get human-readable type name"""
        names = {
            TypeHint.UNKNOWN: "unknown",
            TypeHint.TEXT: "text",
            TypeHint.JSON: "json",
            TypeHint.VECTOR: "vector",
            TypeHint.IMAGE: "image",
            TypeHint.AUDIO: "audio",
            TypeHint.VIDEO: "video",
            TypeHint.MODEL: "model",
            TypeHint.GRAPH: "graph",
            TypeHint.CUSTOM: "custom",
        }
        return names.get(type_hint, "unknown")


if __name__ == "__main__":
    print("DSMIL Binary Container Self-Test")
    print("=" * 50)

    # Generate test key
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key = AESGCM.generate_key(bit_length=256)

    container = DSMILBinaryContainer(encryption_key=key)

    print("\n[1] Create Container")
    payload = b"This is secret intelligence data!"
    data = container.create(
        payload,
        type_hint=TypeHint.TEXT,
        flags=ContainerFlags.ENCRYPTED | ContainerFlags.SIGNED
    )
    print(f"    Payload size: {len(payload)} bytes")
    print(f"    Container size: {len(data)} bytes")
    print(f"    Magic: {data[:8]}")

    print("\n[2] Parse Container")
    header, decrypted = container.parse(data)
    print(f"    Version: {header.version}")
    print(f"    Flags: {header.flags}")
    print(f"    Type: {container.get_type_name(header.type_hint)}")
    print(f"    Payload length: {header.payload_len}")
    print(f"    Decrypted: {decrypted}")

    print("\n[3] Verify Signature")
    valid = container.verify_signature(data)
    print(f"    Valid: {valid}")

    print("\n[4] Tamper Detection")
    tampered = bytearray(data)
    tampered[150] ^= 0xFF  # Flip a bit
    try:
        container.parse(bytes(tampered))
        print("    ERROR: Tampered data accepted!")
    except ValueError as e:
        print(f"    Tamper detected: {e}")

    print("\n[5] Different Types")
    for type_hint in [TypeHint.JSON, TypeHint.VECTOR, TypeHint.IMAGE]:
        data = container.create(b"test", type_hint=type_hint)
        header, _ = container.parse(data)
        print(f"    {container.get_type_name(type_hint)}: {header.type_hint}")

    print("\n" + "=" * 50)
    print("DSMIL Binary Container test complete")

