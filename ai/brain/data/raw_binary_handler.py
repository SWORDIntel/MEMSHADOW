#!/usr/bin/env python3
"""
Raw Binary Handler for DSMIL Brain

Handles raw binary data with type hints and chunking support.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Iterator
from datetime import datetime, timezone
from enum import Enum, auto

logger = logging.getLogger(__name__)


class BinaryTypeHint(Enum):
    """Type hints for binary data"""
    UNKNOWN = auto()
    EXECUTABLE = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    ARCHIVE = auto()
    DOCUMENT = auto()
    DATABASE = auto()
    FIRMWARE = auto()
    NETWORK_CAPTURE = auto()
    MEMORY_DUMP = auto()
    MALWARE_SAMPLE = auto()


# Magic bytes for common formats
MAGIC_SIGNATURES = {
    b"\x7fELF": BinaryTypeHint.EXECUTABLE,
    b"MZ": BinaryTypeHint.EXECUTABLE,
    b"\x89PNG": BinaryTypeHint.IMAGE,
    b"\xff\xd8\xff": BinaryTypeHint.IMAGE,  # JPEG
    b"GIF8": BinaryTypeHint.IMAGE,
    b"RIFF": BinaryTypeHint.AUDIO,  # WAV
    b"ID3": BinaryTypeHint.AUDIO,  # MP3
    b"\x00\x00\x00\x18ftypmp4": BinaryTypeHint.VIDEO,
    b"\x1a\x45\xdf\xa3": BinaryTypeHint.VIDEO,  # MKV
    b"PK\x03\x04": BinaryTypeHint.ARCHIVE,  # ZIP
    b"\x1f\x8b": BinaryTypeHint.ARCHIVE,  # GZIP
    b"%PDF": BinaryTypeHint.DOCUMENT,
    b"\xd0\xcf\x11\xe0": BinaryTypeHint.DOCUMENT,  # MS Office
    b"SQLite format": BinaryTypeHint.DATABASE,
    b"\xd4\xc3\xb2\xa1": BinaryTypeHint.NETWORK_CAPTURE,  # PCAP
}


@dataclass
class BinaryChunk:
    """A chunk of binary data"""
    chunk_id: str
    sequence: int
    data: bytes
    checksum: str

    # Context
    total_chunks: int = 0
    parent_id: str = ""


@dataclass
class BinaryMetadata:
    """Metadata for binary data"""
    size: int
    checksum: str
    type_hint: BinaryTypeHint
    entropy: float = 0.0  # 0-8, high = encrypted/compressed
    magic_bytes: bytes = b""

    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RawBinaryHandler:
    """
    Raw Binary Handler

    Processes raw binary data with type detection and chunking.

    Usage:
        handler = RawBinaryHandler()

        # Analyze binary
        metadata = handler.analyze(data)

        # Chunk for transmission
        chunks = handler.chunk(data, chunk_size=65536)

        # Reassemble
        data = handler.reassemble(chunks)
    """

    def __init__(self, default_chunk_size: int = 65536):
        self.default_chunk_size = default_chunk_size

        logger.info("RawBinaryHandler initialized")

    def analyze(self, data: bytes) -> BinaryMetadata:
        """Analyze binary data"""
        # Checksum
        checksum = hashlib.sha256(data).hexdigest()

        # Type detection
        type_hint = self._detect_type(data)

        # Entropy calculation
        entropy = self._calculate_entropy(data)

        # Magic bytes
        magic = data[:16] if len(data) >= 16 else data

        return BinaryMetadata(
            size=len(data),
            checksum=checksum,
            type_hint=type_hint,
            entropy=entropy,
            magic_bytes=magic,
        )

    def _detect_type(self, data: bytes) -> BinaryTypeHint:
        """Detect binary type from magic bytes"""
        if len(data) < 4:
            return BinaryTypeHint.UNKNOWN

        for magic, hint in MAGIC_SIGNATURES.items():
            if data.startswith(magic):
                return hint

        return BinaryTypeHint.UNKNOWN

    def _calculate_entropy(self, data: bytes, sample_size: int = 65536) -> float:
        """Calculate Shannon entropy (0-8, higher = more random)"""
        import math

        if not data:
            return 0.0

        # Sample if too large
        sample = data[:sample_size]

        # Count byte frequencies
        freq = [0] * 256
        for byte in sample:
            freq[byte] += 1

        # Calculate entropy
        entropy = 0.0
        length = len(sample)

        for count in freq:
            if count > 0:
                p = count / length
                entropy -= p * math.log2(p)

        return entropy

    def chunk(self, data: bytes, chunk_size: Optional[int] = None,
             parent_id: str = "") -> List[BinaryChunk]:
        """Split binary into chunks"""
        size = chunk_size or self.default_chunk_size
        chunks = []

        total = (len(data) + size - 1) // size

        for i, offset in enumerate(range(0, len(data), size)):
            chunk_data = data[offset:offset + size]
            chunk = BinaryChunk(
                chunk_id=hashlib.sha256(f"{parent_id}:{i}".encode()).hexdigest()[:16],
                sequence=i,
                data=chunk_data,
                checksum=hashlib.sha256(chunk_data).hexdigest(),
                total_chunks=total,
                parent_id=parent_id,
            )
            chunks.append(chunk)

        return chunks

    def reassemble(self, chunks: List[BinaryChunk]) -> bytes:
        """Reassemble chunks into original binary"""
        # Sort by sequence
        sorted_chunks = sorted(chunks, key=lambda c: c.sequence)

        # Verify completeness
        expected = sorted_chunks[0].total_chunks if sorted_chunks else 0
        if len(sorted_chunks) != expected:
            raise ValueError(f"Missing chunks: have {len(sorted_chunks)}, expected {expected}")

        # Verify checksums and reassemble
        data = b""
        for chunk in sorted_chunks:
            if hashlib.sha256(chunk.data).hexdigest() != chunk.checksum:
                raise ValueError(f"Checksum mismatch for chunk {chunk.sequence}")
            data += chunk.data

        return data

    def iter_chunks(self, data: bytes, chunk_size: Optional[int] = None) -> Iterator[BinaryChunk]:
        """Iterate over chunks (memory efficient)"""
        size = chunk_size or self.default_chunk_size
        total = (len(data) + size - 1) // size
        parent_id = hashlib.sha256(data[:1024]).hexdigest()[:8]

        for i, offset in enumerate(range(0, len(data), size)):
            chunk_data = data[offset:offset + size]
            yield BinaryChunk(
                chunk_id=hashlib.sha256(f"{parent_id}:{i}".encode()).hexdigest()[:16],
                sequence=i,
                data=chunk_data,
                checksum=hashlib.sha256(chunk_data).hexdigest(),
                total_chunks=total,
                parent_id=parent_id,
            )

    def is_likely_encrypted(self, data: bytes, threshold: float = 7.5) -> bool:
        """Check if data appears encrypted (high entropy)"""
        entropy = self._calculate_entropy(data)
        return entropy >= threshold

    def is_likely_compressed(self, data: bytes) -> bool:
        """Check if data appears compressed"""
        entropy = self._calculate_entropy(data)
        type_hint = self._detect_type(data)

        return entropy > 7.0 or type_hint == BinaryTypeHint.ARCHIVE


if __name__ == "__main__":
    print("Raw Binary Handler Self-Test")
    print("=" * 50)

    handler = RawBinaryHandler()

    print("\n[1] Analyze Text-like Binary")
    text_data = b"This is some plain text data that has low entropy"
    meta = handler.analyze(text_data)
    print(f"    Size: {meta.size}")
    print(f"    Type: {meta.type_hint.name}")
    print(f"    Entropy: {meta.entropy:.2f}")

    print("\n[2] Analyze Random Binary")
    import os
    random_data = os.urandom(10000)
    meta = handler.analyze(random_data)
    print(f"    Size: {meta.size}")
    print(f"    Type: {meta.type_hint.name}")
    print(f"    Entropy: {meta.entropy:.2f}")
    print(f"    Likely encrypted: {handler.is_likely_encrypted(random_data)}")

    print("\n[3] Magic Byte Detection")
    test_cases = [
        (b"\x7fELF\x00\x00\x00\x00", "ELF"),
        (b"MZ\x00\x00\x00\x00", "PE"),
        (b"\x89PNG\r\n\x1a\n", "PNG"),
        (b"PK\x03\x04\x00\x00", "ZIP"),
        (b"%PDF-1.4", "PDF"),
    ]
    for data, expected in test_cases:
        padded = data + b"\x00" * 100
        meta = handler.analyze(padded)
        print(f"    {expected}: {meta.type_hint.name}")

    print("\n[4] Chunking")
    large_data = b"X" * 100000
    chunks = handler.chunk(large_data, chunk_size=32768)
    print(f"    Original: {len(large_data)} bytes")
    print(f"    Chunks: {len(chunks)}")
    print(f"    First chunk size: {len(chunks[0].data)}")

    print("\n[5] Reassembly")
    reassembled = handler.reassemble(chunks)
    print(f"    Reassembled: {len(reassembled)} bytes")
    print(f"    Match: {reassembled == large_data}")

    print("\n[6] Chunk Iterator")
    chunks_iter = list(handler.iter_chunks(large_data, chunk_size=32768))
    print(f"    Chunks via iterator: {len(chunks_iter)}")

    print("\n" + "=" * 50)
    print("Raw Binary Handler test complete")

