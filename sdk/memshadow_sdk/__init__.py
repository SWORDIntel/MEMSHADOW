"""
MEMSHADOW Python SDK
A memlayer-inspired client library for MEMSHADOW memory persistence
"""

__version__ = "0.1.0"

from .client import MemshadowClient
from .wrappers.openai import OpenAI
from .wrappers.anthropic import Anthropic

__all__ = [
    "MemshadowClient",
    "OpenAI",
    "Anthropic",
]
