#!/usr/bin/env python3
"""
DSMIL Brain Plugin System

Extensible plugin architecture for data ingestion:
- Auto-discovery of plugins
- Hot-reload support
- Schema validation
- Built-in ingest plugins
"""

from .ingest_framework import (
    IngestPlugin,
    PluginManager,
    IngestResult,
    PluginConfig,
)

__all__ = [
    "IngestPlugin", "PluginManager", "IngestResult", "PluginConfig",
]

