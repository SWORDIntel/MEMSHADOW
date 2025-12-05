#!/usr/bin/env python3
"""
DSMIL Brain Covert Operations Layer

Covert communication and deniability capabilities:
- Steganographic Channels: Hidden communication in normal traffic
- Plausible Deniability: Multi-layer encryption with multiple valid decryptions
"""

from .stego_channels import (
    SteganographicChannel,
    StegoMessage,
    TimingChannel,
    OrderingChannel,
    CovertTransport,
)

from .plausible_deniability import (
    DeniableContainer,
    DeniableLayer,
    MultiLayerEncryption,
    DeniabilityLevel,
)

__all__ = [
    # Steganography
    "SteganographicChannel",
    "StegoMessage",
    "TimingChannel",
    "OrderingChannel",
    "CovertTransport",
    # Deniability
    "DeniableContainer",
    "DeniableLayer",
    "MultiLayerEncryption",
    "DeniabilityLevel",
]

