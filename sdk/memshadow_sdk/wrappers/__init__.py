"""
LLM Wrappers for MEMSHADOW
Provides drop-in replacements for popular LLM clients with automatic memory persistence
"""

from .openai import OpenAI
from .anthropic import Anthropic

__all__ = ["OpenAI", "Anthropic"]
