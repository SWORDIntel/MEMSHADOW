#!/usr/bin/env python3
"""
Steganographic Channels for DSMIL Brain

Hidden communication methods:
- Intelligence embedded in normal traffic
- Covert timing channels
- Ordering-based encoding
- Emergency exfiltration paths
- Deniable dead drops
"""

import hashlib
import secrets
import threading
import logging
import time
import struct
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timezone
from enum import Enum, auto
import base64

logger = logging.getLogger(__name__)


class ChannelType(Enum):
    """Types of covert channels"""
    TIMING = auto()          # Encode in inter-packet timing
    ORDERING = auto()        # Encode in message ordering
    CONTENT = auto()         # Encode in content modifications
    PROTOCOL = auto()        # Encode in protocol fields
    SIZE = auto()            # Encode in packet sizes
    FREQUENCY = auto()       # Encode in transmission frequency


class StegoMethod(Enum):
    """Steganography methods"""
    LSB = auto()             # Least significant bit
    SPREAD_SPECTRUM = auto() # Spread across carrier
    WHITESPACE = auto()      # Hide in whitespace
    UNICODE = auto()         # Use unicode lookalikes
    TIMING = auto()          # Timing-based encoding
    METADATA = auto()        # Hide in metadata


@dataclass
class StegoMessage:
    """A steganographic message"""
    message_id: str
    content: bytes

    # Encoding
    method: StegoMethod
    carrier_type: str  # Type of carrier (text, image, timing, etc.)

    # Result
    encoded_carrier: Any = None

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def capacity_used(self) -> float:
        """Return percentage of carrier capacity used"""
        if not self.encoded_carrier:
            return 0.0
        return len(self.content) / (len(str(self.encoded_carrier)) * 0.1)


@dataclass
class CovertTransport:
    """Configuration for a covert transport channel"""
    transport_id: str
    channel_type: ChannelType

    # Configuration
    bandwidth_bps: float = 1.0  # Covert bandwidth in bits per second
    reliability: float = 0.9
    detectability: float = 0.1  # Probability of detection

    # State
    is_active: bool = True
    messages_sent: int = 0
    bytes_transmitted: int = 0


class TimingChannel:
    """
    Covert timing channel

    Encodes information in inter-message timing delays.
    """

    def __init__(self, base_delay_ms: float = 100,
                 bit_delay_ms: float = 50):
        """
        Initialize timing channel

        Args:
            base_delay_ms: Base delay between messages
            bit_delay_ms: Additional delay per '1' bit
        """
        self.base_delay_ms = base_delay_ms
        self.bit_delay_ms = bit_delay_ms

        self._encoding_in_progress = False
        self._decoding_buffer: List[float] = []

    def encode_byte(self, byte_value: int) -> List[float]:
        """
        Encode a byte as timing delays

        Args:
            byte_value: Byte to encode (0-255)

        Returns:
            List of delays in milliseconds
        """
        delays = []

        for bit_pos in range(8):
            bit = (byte_value >> (7 - bit_pos)) & 1
            delay = self.base_delay_ms + (bit * self.bit_delay_ms)
            delays.append(delay)

        return delays

    def encode_message(self, message: bytes) -> List[float]:
        """
        Encode a message as timing sequence

        Args:
            message: Message bytes

        Returns:
            List of timing delays
        """
        delays = []

        # Add length header (4 bytes)
        length = len(message)
        length_bytes = struct.pack('>I', length)

        for byte in length_bytes + message:
            delays.extend(self.encode_byte(byte))

        return delays

    def decode_delays(self, delays: List[float]) -> bytes:
        """
        Decode timing delays to message

        Args:
            delays: List of observed delays

        Returns:
            Decoded message bytes
        """
        if len(delays) < 32:  # Need at least header
            return b""

        threshold = self.base_delay_ms + (self.bit_delay_ms / 2)

        # Convert delays to bits
        bits = []
        for delay in delays:
            bits.append(1 if delay >= threshold else 0)

        # Convert bits to bytes
        result = bytearray()
        for i in range(0, len(bits) - 7, 8):
            byte_bits = bits[i:i+8]
            byte_value = sum(b << (7 - j) for j, b in enumerate(byte_bits))
            result.append(byte_value)

        if len(result) < 4:
            return b""

        # Extract length
        length = struct.unpack('>I', bytes(result[:4]))[0]

        # Extract message
        if len(result) >= 4 + length:
            return bytes(result[4:4+length])

        return b""


class OrderingChannel:
    """
    Covert ordering channel

    Encodes information in the ordering of messages/packets.
    """

    def __init__(self, chunk_size: int = 4):
        """
        Initialize ordering channel

        Args:
            chunk_size: Number of items per encoded chunk
        """
        self.chunk_size = chunk_size
        # With 4 items, we can encode log2(4!) = ~4.58 bits per chunk
        self.bits_per_chunk = 4  # Conservative estimate

    def _permutation_to_bits(self, perm: List[int]) -> int:
        """Convert permutation to bit value using Lehmer code"""
        n = len(perm)
        if n <= 1:
            return 0

        # Calculate Lehmer code
        lehmer = 0
        factorial = 1
        for i in range(n - 1, -1, -1):
            count = sum(1 for j in range(i + 1, n) if perm[j] < perm[i])
            lehmer += count * factorial
            factorial *= (n - i)

        return lehmer

    def _bits_to_permutation(self, bits: int, n: int) -> List[int]:
        """Convert bit value to permutation"""
        items = list(range(n))
        result = []

        for i in range(n, 0, -1):
            factorial = 1
            for j in range(1, i):
                factorial *= j

            index = bits // factorial
            bits %= factorial

            if index >= len(items):
                index = len(items) - 1

            result.append(items.pop(index))

        return result

    def encode(self, data: bytes, carriers: List[Any]) -> List[Any]:
        """
        Encode data in carrier ordering

        Args:
            data: Data to encode
            carriers: Carrier items to reorder

        Returns:
            Reordered carriers
        """
        if len(carriers) < self.chunk_size:
            return carriers

        # Pad data to byte boundary
        bits_needed = (len(carriers) // self.chunk_size) * self.bits_per_chunk
        bytes_to_encode = (bits_needed + 7) // 8

        if len(data) < bytes_to_encode:
            data = data + bytes(bytes_to_encode - len(data))

        # Convert data to bit stream
        bits = int.from_bytes(data[:bytes_to_encode], 'big')

        result = []
        chunk_count = len(carriers) // self.chunk_size

        for i in range(chunk_count):
            chunk = carriers[i * self.chunk_size:(i + 1) * self.chunk_size]

            # Extract bits for this chunk
            chunk_bits = (bits >> ((chunk_count - 1 - i) * self.bits_per_chunk)) & ((1 << self.bits_per_chunk) - 1)

            # Get permutation for these bits
            perm = self._bits_to_permutation(chunk_bits, self.chunk_size)

            # Apply permutation
            reordered = [chunk[p] for p in perm]
            result.extend(reordered)

        # Add remaining carriers unchanged
        result.extend(carriers[chunk_count * self.chunk_size:])

        return result

    def decode(self, carriers: List[Any], original_order: List[Any]) -> bytes:
        """
        Decode data from carrier ordering

        Args:
            carriers: Received carriers in encoded order
            original_order: Original/expected order

        Returns:
            Decoded data
        """
        if len(carriers) < self.chunk_size:
            return b""

        chunk_count = len(carriers) // self.chunk_size
        bits = 0

        for i in range(chunk_count):
            chunk = carriers[i * self.chunk_size:(i + 1) * self.chunk_size]
            original_chunk = original_order[i * self.chunk_size:(i + 1) * self.chunk_size]

            # Determine permutation
            perm = []
            for item in chunk:
                try:
                    perm.append(original_chunk.index(item))
                except ValueError:
                    perm.append(0)

            # Convert permutation to bits
            chunk_bits = self._permutation_to_bits(perm)

            bits = (bits << self.bits_per_chunk) | chunk_bits

        # Convert bits to bytes
        byte_count = (chunk_count * self.bits_per_chunk + 7) // 8
        return bits.to_bytes(byte_count, 'big')


class SteganographicChannel:
    """
    Main steganographic channel manager

    Provides multiple methods for hiding information:
    - Whitespace encoding
    - Unicode lookalikes
    - Timing channels
    - Ordering channels

    Usage:
        channel = SteganographicChannel()

        # Hide message in text
        stego_text = channel.hide_in_text("secret message", "normal cover text...")

        # Extract message
        message = channel.extract_from_text(stego_text)

        # Use timing channel
        delays = channel.encode_timing(secret_bytes)
    """

    def __init__(self):
        self._timing_channel = TimingChannel()
        self._ordering_channel = OrderingChannel()

        # Unicode lookalikes for text encoding
        self._unicode_map = {
            'a': 'а',  # Cyrillic
            'e': 'е',
            'o': 'о',
            'p': 'р',
            'c': 'с',
            'x': 'х',
            'y': 'у',
            'A': 'А',
            'E': 'Е',
            'O': 'О',
            'P': 'Р',
            'C': 'С',
        }
        self._reverse_unicode_map = {v: k for k, v in self._unicode_map.items()}

        self._lock = threading.Lock()

        # Statistics
        self.stats = {
            "messages_encoded": 0,
            "messages_decoded": 0,
            "bytes_hidden": 0,
        }

        logger.info("SteganographicChannel initialized")

    def hide_in_whitespace(self, message: bytes, cover_text: str) -> str:
        """
        Hide message in whitespace at end of lines

        Uses tabs and spaces to encode bits.
        """
        lines = cover_text.split('\n')

        # Convert message to bits
        bits = bin(int.from_bytes(message, 'big'))[2:]
        # Pad to multiple of 8
        bits = bits.zfill((len(bits) + 7) // 8 * 8)

        # Prepend length
        length_bits = bin(len(message))[2:].zfill(16)
        all_bits = length_bits + bits

        # Encode bits in line endings
        result_lines = []
        bit_index = 0

        for line in lines:
            line = line.rstrip()  # Remove existing trailing whitespace

            # Add encoded whitespace
            trailing = ""
            for _ in range(8):  # 8 bits per line
                if bit_index < len(all_bits):
                    if all_bits[bit_index] == '1':
                        trailing += '\t'
                    else:
                        trailing += ' '
                    bit_index += 1

            result_lines.append(line + trailing)

        self.stats["messages_encoded"] += 1
        self.stats["bytes_hidden"] += len(message)

        return '\n'.join(result_lines)

    def extract_from_whitespace(self, stego_text: str) -> bytes:
        """Extract message hidden in whitespace"""
        lines = stego_text.split('\n')

        bits = ""
        for line in lines:
            # Extract trailing whitespace
            trailing = ""
            for char in reversed(line):
                if char in ' \t':
                    trailing = char + trailing
                else:
                    break

            # Decode whitespace to bits
            for char in trailing:
                if char == '\t':
                    bits += '1'
                else:
                    bits += '0'

        if len(bits) < 16:
            return b""

        # Extract length
        length = int(bits[:16], 2)

        # Extract message
        message_bits = bits[16:16 + length * 8]
        if len(message_bits) < length * 8:
            return b""

        message_int = int(message_bits, 2) if message_bits else 0

        self.stats["messages_decoded"] += 1

        return message_int.to_bytes(length, 'big')

    def hide_in_unicode(self, message: bytes, cover_text: str) -> str:
        """
        Hide message using unicode lookalikes

        Replaces ASCII characters with visually similar unicode.
        """
        # Convert message to bits
        bits = bin(int.from_bytes(message, 'big'))[2:].zfill(len(message) * 8)

        # Prepend length
        length_bits = bin(len(message))[2:].zfill(16)
        all_bits = length_bits + bits

        result = []
        bit_index = 0

        for char in cover_text:
            if bit_index < len(all_bits) and char in self._unicode_map:
                if all_bits[bit_index] == '1':
                    result.append(self._unicode_map[char])
                else:
                    result.append(char)
                bit_index += 1
            else:
                result.append(char)

        self.stats["messages_encoded"] += 1
        self.stats["bytes_hidden"] += len(message)

        return ''.join(result)

    def extract_from_unicode(self, stego_text: str) -> bytes:
        """Extract message hidden in unicode lookalikes"""
        bits = ""

        for char in stego_text:
            if char in self._reverse_unicode_map:
                bits += '1'
            elif char in self._unicode_map:
                bits += '0'

        if len(bits) < 16:
            return b""

        # Extract length
        length = int(bits[:16], 2)

        # Extract message
        message_bits = bits[16:16 + length * 8]
        if len(message_bits) < length * 8:
            return b""

        message_int = int(message_bits, 2) if message_bits else 0

        self.stats["messages_decoded"] += 1

        return message_int.to_bytes(length, 'big')

    def encode_timing(self, message: bytes) -> List[float]:
        """Encode message as timing delays"""
        delays = self._timing_channel.encode_message(message)
        self.stats["messages_encoded"] += 1
        self.stats["bytes_hidden"] += len(message)
        return delays

    def decode_timing(self, delays: List[float]) -> bytes:
        """Decode message from timing delays"""
        message = self._timing_channel.decode_delays(delays)
        if message:
            self.stats["messages_decoded"] += 1
        return message

    def encode_ordering(self, message: bytes, items: List[Any]) -> List[Any]:
        """Encode message in item ordering"""
        result = self._ordering_channel.encode(message, items)
        self.stats["messages_encoded"] += 1
        self.stats["bytes_hidden"] += len(message)
        return result

    def decode_ordering(self, items: List[Any], original_order: List[Any]) -> bytes:
        """Decode message from item ordering"""
        message = self._ordering_channel.decode(items, original_order)
        if message:
            self.stats["messages_decoded"] += 1
        return message

    def get_stats(self) -> Dict:
        """Get channel statistics"""
        return dict(self.stats)


if __name__ == "__main__":
    print("Steganographic Channels Self-Test")
    print("=" * 50)

    channel = SteganographicChannel()

    print("\n[1] Whitespace Encoding")
    cover = """This is a normal document.
It contains multiple lines.
Nothing suspicious here.
Just regular text content.
End of document."""

    secret = b"Hello, World!"
    stego = channel.hide_in_whitespace(secret, cover)

    print(f"    Original cover: {len(cover)} chars")
    print(f"    Stego text: {len(stego)} chars")
    print(f"    Hidden message: {len(secret)} bytes")

    extracted = channel.extract_from_whitespace(stego)
    print(f"    Extracted: {extracted}")
    print(f"    Match: {extracted == secret}")

    print("\n[2] Unicode Encoding")
    cover2 = "This is a perfectly normal and innocent piece of text."
    secret2 = b"spy"

    stego2 = channel.hide_in_unicode(secret2, cover2)
    print(f"    Cover: {cover2}")
    print(f"    Stego: {stego2}")
    print(f"    Visually identical: {cover2.lower() != stego2.lower()}")

    extracted2 = channel.extract_from_unicode(stego2)
    print(f"    Extracted: {extracted2}")
    print(f"    Match: {extracted2 == secret2}")

    print("\n[3] Timing Channel")
    secret3 = b"TOP SECRET"
    delays = channel.encode_timing(secret3)
    print(f"    Message: {secret3}")
    print(f"    Delays: {len(delays)} timing values")
    print(f"    Sample delays: {delays[:8]}")

    extracted3 = channel.decode_timing(delays)
    print(f"    Extracted: {extracted3}")
    print(f"    Match: {extracted3 == secret3}")

    print("\n[4] Ordering Channel")
    items = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    secret4 = b"\x05"  # Simple byte

    encoded_items = channel.encode_ordering(secret4, items)
    print(f"    Original order: {items}")
    print(f"    Encoded order: {encoded_items}")

    extracted4 = channel.decode_ordering(encoded_items, items)
    print(f"    Extracted: {extracted4.hex()}")

    print("\n[5] Statistics")
    stats = channel.get_stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 50)
    print("Steganographic Channels test complete")

