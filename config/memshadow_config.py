#!/usr/bin/env python3
"""
Centralized MEMSHADOW configuration loader.

This module provides a single place to fetch tunable values for the entire
MEMSHADOW subsystem (SHRINK ingest, memory sync, federation gateway, etc.).
Configuration is resolved in the following order (lowest -> highest priority):

1. Built-in defaults defined in :class:`MemshadowConfig`
2. Optional YAML/JSON file (config/memshadow.yaml or path specified via
   MEMSHADOW_CONFIG_PATH)
3. Environment variables (MEMSHADOW_*)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional
import json
import os

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


DEFAULT_CONFIG_PATHS = (
    Path(__file__).parent / "memshadow.yaml",
    Path(__file__).parent / "memshadow.yml",
)


@dataclass(frozen=True)
class MemshadowConfig:
    """Resolved runtime configuration for MEMSHADOW subsystems."""

    background_sync_interval_seconds: int = 30
    max_batch_items: int = 100
    compression_threshold_bytes: int = 1024
    enable_p2p_for_critical: bool = True
    enable_shrink_ingest: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "background_sync_interval_seconds": self.background_sync_interval_seconds,
            "max_batch_items": self.max_batch_items,
            "compression_threshold_bytes": self.compression_threshold_bytes,
            "enable_p2p_for_critical": self.enable_p2p_for_critical,
            "enable_shrink_ingest": self.enable_shrink_ingest,
        }


def _load_file_settings(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        raw = path.read_text().strip()
        if not raw:
            return {}

        if path.suffix.lower() in {".yaml", ".yml"} and yaml:
            return yaml.safe_load(raw) or {}

        return json.loads(raw)
    except Exception:
        # Fall back to empty config on parse failure to keep defaults safe
        return {}


def _apply_env_overrides(config: MemshadowConfig) -> MemshadowConfig:
    overrides: Dict[str, Any] = {}

    env_mapping = {
        "MEMSHADOW_BACKGROUND_SYNC_INTERVAL_SECONDS": "background_sync_interval_seconds",
        "MEMSHADOW_MAX_BATCH_ITEMS": "max_batch_items",
        "MEMSHADOW_COMPRESSION_THRESHOLD_BYTES": "compression_threshold_bytes",
        "MEMSHADOW_ENABLE_P2P_FOR_CRITICAL": "enable_p2p_for_critical",
        "MEMSHADOW_ENABLE_SHRINK_INGEST": "enable_shrink_ingest",
    }

    for env_key, field_name in env_mapping.items():
        if env_key not in os.environ:
            continue
        value = os.environ[env_key]
        if field_name in {
            "background_sync_interval_seconds",
            "max_batch_items",
            "compression_threshold_bytes",
        }:
            try:
                overrides[field_name] = int(value)
            except ValueError:
                continue
        else:
            overrides[field_name] = value.lower() in {"1", "true", "yes", "on"}

    return replace(config, **overrides) if overrides else config


def load_memshadow_config(config_path: Optional[str] = None) -> MemshadowConfig:
    """
    Load MEMSHADOW configuration from defaults, file, and environment variables.
    """
    config = MemshadowConfig()

    path_candidates = list(DEFAULT_CONFIG_PATHS)
    if config_path:
        path_candidates.insert(0, Path(config_path))
    elif os.environ.get("MEMSHADOW_CONFIG_PATH"):
        path_candidates.insert(0, Path(os.environ["MEMSHADOW_CONFIG_PATH"]))

    for candidate in path_candidates:
        file_settings = _load_file_settings(candidate)
        if not file_settings:
            continue
        config = replace(
            config,
            **{
                key: file_settings.get(key, getattr(config, key))
                for key in config.to_dict().keys()
            },
        )
        break

    return _apply_env_overrides(config)


@lru_cache(maxsize=1)
def get_memshadow_config(config_path: Optional[str] = None) -> MemshadowConfig:
    """
    Cached accessor for shared configuration consumers.
    """
    return load_memshadow_config(config_path)

