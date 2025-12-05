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

Extended for layer-aware bandwidth governance (per 03_MEMORY_BANDWIDTH_OPTIMIZATION):
- Per-layer bandwidth budgets
- Degradation policy thresholds
- Priority-based cutoffs
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple
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


class DSMILLayerID(IntEnum):
    """DSMIL Layer identifiers for config (matches memshadow_layer_mapping.DSMILLayer)."""
    PHYSICAL_SECURITY = 2
    NETWORK_INFRASTRUCTURE = 3
    DATA_PROCESSING = 4
    ANALYTICS_CORRELATION = 5
    FEDERATION_MESH = 6
    PRIMARY_AI_MEMORY = 7
    SECURITY_ANALYTICS = 8
    DECISION_SUPPORT = 9


class MemshadowCategoryID(IntEnum):
    """MEMSHADOW category identifiers for config."""
    PSYCH = 1
    THREAT = 2
    MEMORY = 3
    FEDERATION = 4
    IMPROVEMENT = 5


@dataclass(frozen=True)
class LayerBudgetConfig:
    """Per-layer bandwidth budget configuration."""
    layer_id: int
    max_bytes_per_sec: int = 0  # 0 = use calculated default
    max_messages_per_sec: int = 0  # 0 = no limit
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "max_bytes_per_sec": self.max_bytes_per_sec,
            "max_messages_per_sec": self.max_messages_per_sec,
        }


@dataclass(frozen=True)
class CategoryBudgetConfig:
    """Per-category bandwidth budget configuration."""
    category_id: int
    max_bytes_per_sec: int = 0  # 0 = use calculated default
    max_messages_per_sec: int = 0  # 0 = no limit
    allows_degradation: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category_id": self.category_id,
            "max_bytes_per_sec": self.max_bytes_per_sec,
            "max_messages_per_sec": self.max_messages_per_sec,
            "allows_degradation": self.allows_degradation,
        }


@dataclass(frozen=True)
class DegradationPolicyConfig:
    """Degradation policy configuration."""
    
    # Thresholds (as percentage of budget, 0.0 to 1.0)
    degrade_threshold_percent: float = 0.80  # Start degrading at 80%
    drop_threshold_percent: float = 0.95     # Start dropping at 95%
    recovery_threshold_percent: float = 0.50  # Exit degradation below 50%
    
    # Priority cutoffs (Priority enum values: LOW=0, NORMAL=1, HIGH=2, CRITICAL=3, EMERGENCY=4)
    never_degrade_priority: int = 3  # CRITICAL and above never degraded
    never_drop_priority: int = 4     # EMERGENCY never dropped
    
    # Degradation order for cutbacks (category IDs, first to last)
    # Default: LOW -> NORMAL priority categories first, IMPROVEMENT -> FEDERATION -> MEMORY -> PSYCH -> THREAT
    cutback_order: Tuple[int, ...] = (5, 4, 3, 1, 2)  # IMPROVEMENT, FEDERATION, MEMORY, PSYCH, THREAT
    
    # Sync mode adjustments
    normal_sync_interval_ms: int = 1000
    degraded_sync_interval_ms: int = 5000
    critical_only_sync_interval_ms: int = 10000
    
    # Batch size adjustments
    normal_batch_size: int = 100
    degraded_batch_size: int = 500
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "degrade_threshold_percent": self.degrade_threshold_percent,
            "drop_threshold_percent": self.drop_threshold_percent,
            "recovery_threshold_percent": self.recovery_threshold_percent,
            "never_degrade_priority": self.never_degrade_priority,
            "never_drop_priority": self.never_drop_priority,
            "cutback_order": list(self.cutback_order),
            "normal_sync_interval_ms": self.normal_sync_interval_ms,
            "degraded_sync_interval_ms": self.degraded_sync_interval_ms,
            "critical_only_sync_interval_ms": self.critical_only_sync_interval_ms,
            "normal_batch_size": self.normal_batch_size,
            "degraded_batch_size": self.degraded_batch_size,
        }


@dataclass(frozen=True)
class BandwidthGovernorConfig:
    """Bandwidth governor configuration (from 03_MEMORY_BANDWIDTH_OPTIMIZATION)."""
    
    # Overall system constraints
    total_system_bandwidth_gbps: float = 64.0  # Total 64 GB/s
    memshadow_bandwidth_percent: float = 0.15  # MEMSHADOW gets 15% of total
    
    # Per-layer budgets (empty = use calculated defaults)
    layer_budgets: Tuple[LayerBudgetConfig, ...] = ()
    
    # Per-category budgets (empty = use calculated defaults)
    category_budgets: Tuple[CategoryBudgetConfig, ...] = ()
    
    # Degradation policy
    degradation_policy: DegradationPolicyConfig = field(default_factory=DegradationPolicyConfig)
    
    # Operating modes
    monitor_only: bool = False  # If True, observe but don't enforce
    enable_layer8_hooks: bool = True  # Enable Layer 8 security hooks
    
    # Statistics
    stats_window_seconds: float = 60.0  # Rolling window for rate calculation
    
    def get_memshadow_budget_bytes_per_sec(self) -> int:
        """Calculate total MEMSHADOW bandwidth budget in bytes/sec."""
        return int(self.total_system_bandwidth_gbps * 1e9 * self.memshadow_bandwidth_percent)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_system_bandwidth_gbps": self.total_system_bandwidth_gbps,
            "memshadow_bandwidth_percent": self.memshadow_bandwidth_percent,
            "memshadow_budget_bytes_per_sec": self.get_memshadow_budget_bytes_per_sec(),
            "layer_budgets": [lb.to_dict() for lb in self.layer_budgets],
            "category_budgets": [cb.to_dict() for cb in self.category_budgets],
            "degradation_policy": self.degradation_policy.to_dict(),
            "monitor_only": self.monitor_only,
            "enable_layer8_hooks": self.enable_layer8_hooks,
            "stats_window_seconds": self.stats_window_seconds,
        }


@dataclass(frozen=True)
class MemshadowConfig:
    """Resolved runtime configuration for MEMSHADOW subsystems."""

    # Original settings
    background_sync_interval_seconds: int = 30
    max_batch_items: int = 100
    compression_threshold_bytes: int = 1024
    enable_p2p_for_critical: bool = True
    enable_shrink_ingest: bool = True
    enable_psych_ingest: bool = True
    enable_threat_ingest: bool = True
    enable_memory_ingest: bool = True
    enable_federation_ingest: bool = True
    enable_improvement_ingest: bool = True
    
    # NEW: Bandwidth governor configuration
    bandwidth_governor: BandwidthGovernorConfig = field(default_factory=BandwidthGovernorConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "background_sync_interval_seconds": self.background_sync_interval_seconds,
            "max_batch_items": self.max_batch_items,
            "compression_threshold_bytes": self.compression_threshold_bytes,
            "enable_p2p_for_critical": self.enable_p2p_for_critical,
            "enable_shrink_ingest": self.enable_shrink_ingest,
            "enable_psych_ingest": self.enable_psych_ingest,
            "enable_threat_ingest": self.enable_threat_ingest,
            "enable_memory_ingest": self.enable_memory_ingest,
            "enable_federation_ingest": self.enable_federation_ingest,
            "enable_improvement_ingest": self.enable_improvement_ingest,
            "bandwidth_governor": self.bandwidth_governor.to_dict(),
        }
    
    # Convenience accessors for governor settings
    @property
    def degrade_threshold_percent(self) -> float:
        return self.bandwidth_governor.degradation_policy.degrade_threshold_percent
    
    @property
    def drop_threshold_percent(self) -> float:
        return self.bandwidth_governor.degradation_policy.drop_threshold_percent
    
    @property
    def monitor_only(self) -> bool:
        return self.bandwidth_governor.monitor_only
    
    @property
    def enable_layer8_hooks(self) -> bool:
        return self.bandwidth_governor.enable_layer8_hooks


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


def _parse_bandwidth_governor_config(settings: Dict[str, Any]) -> BandwidthGovernorConfig:
    """Parse bandwidth governor config from file settings."""
    bg = settings.get("bandwidth_governor", {})
    if not bg:
        return BandwidthGovernorConfig()
    
    # Parse degradation policy
    dp_dict = bg.get("degradation_policy", {})
    degradation_policy = DegradationPolicyConfig(
        degrade_threshold_percent=dp_dict.get("degrade_threshold_percent", 0.80),
        drop_threshold_percent=dp_dict.get("drop_threshold_percent", 0.95),
        recovery_threshold_percent=dp_dict.get("recovery_threshold_percent", 0.50),
        never_degrade_priority=dp_dict.get("never_degrade_priority", 3),
        never_drop_priority=dp_dict.get("never_drop_priority", 4),
        cutback_order=tuple(dp_dict.get("cutback_order", [5, 4, 3, 1, 2])),
        normal_sync_interval_ms=dp_dict.get("normal_sync_interval_ms", 1000),
        degraded_sync_interval_ms=dp_dict.get("degraded_sync_interval_ms", 5000),
        critical_only_sync_interval_ms=dp_dict.get("critical_only_sync_interval_ms", 10000),
        normal_batch_size=dp_dict.get("normal_batch_size", 100),
        degraded_batch_size=dp_dict.get("degraded_batch_size", 500),
    )
    
    # Parse layer budgets
    layer_budgets = tuple(
        LayerBudgetConfig(
            layer_id=lb.get("layer_id", 7),
            max_bytes_per_sec=lb.get("max_bytes_per_sec", 0),
            max_messages_per_sec=lb.get("max_messages_per_sec", 0),
        )
        for lb in bg.get("layer_budgets", [])
    )
    
    # Parse category budgets
    category_budgets = tuple(
        CategoryBudgetConfig(
            category_id=cb.get("category_id", 1),
            max_bytes_per_sec=cb.get("max_bytes_per_sec", 0),
            max_messages_per_sec=cb.get("max_messages_per_sec", 0),
            allows_degradation=cb.get("allows_degradation", True),
        )
        for cb in bg.get("category_budgets", [])
    )
    
    return BandwidthGovernorConfig(
        total_system_bandwidth_gbps=bg.get("total_system_bandwidth_gbps", 64.0),
        memshadow_bandwidth_percent=bg.get("memshadow_bandwidth_percent", 0.15),
        layer_budgets=layer_budgets,
        category_budgets=category_budgets,
        degradation_policy=degradation_policy,
        monitor_only=bg.get("monitor_only", False),
        enable_layer8_hooks=bg.get("enable_layer8_hooks", True),
        stats_window_seconds=bg.get("stats_window_seconds", 60.0),
    )


def _apply_env_overrides(config: MemshadowConfig) -> MemshadowConfig:
    overrides: Dict[str, Any] = {}

    # Original env mappings
    env_mapping = {
        "MEMSHADOW_BACKGROUND_SYNC_INTERVAL_SECONDS": "background_sync_interval_seconds",
        "MEMSHADOW_MAX_BATCH_ITEMS": "max_batch_items",
        "MEMSHADOW_COMPRESSION_THRESHOLD_BYTES": "compression_threshold_bytes",
        "MEMSHADOW_ENABLE_P2P_FOR_CRITICAL": "enable_p2p_for_critical",
        "MEMSHADOW_ENABLE_SHRINK_INGEST": "enable_shrink_ingest",
        "MEMSHADOW_ENABLE_PSYCH_INGEST": "enable_psych_ingest",
        "MEMSHADOW_ENABLE_THREAT_INGEST": "enable_threat_ingest",
        "MEMSHADOW_ENABLE_MEMORY_INGEST": "enable_memory_ingest",
        "MEMSHADOW_ENABLE_FEDERATION_INGEST": "enable_federation_ingest",
        "MEMSHADOW_ENABLE_IMPROVEMENT_INGEST": "enable_improvement_ingest",
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

    # NEW: Bandwidth governor env overrides
    bg_overrides = {}
    dp_overrides = {}
    
    bg_env_mapping = {
        "MEMSHADOW_BANDWIDTH_PERCENT": ("memshadow_bandwidth_percent", float),
        "MEMSHADOW_MONITOR_ONLY": ("monitor_only", bool),
        "MEMSHADOW_ENABLE_LAYER8_HOOKS": ("enable_layer8_hooks", bool),
        "MEMSHADOW_STATS_WINDOW_SECONDS": ("stats_window_seconds", float),
    }
    
    dp_env_mapping = {
        "MEMSHADOW_DEGRADE_THRESHOLD_PERCENT": ("degrade_threshold_percent", float),
        "MEMSHADOW_DROP_THRESHOLD_PERCENT": ("drop_threshold_percent", float),
        "MEMSHADOW_NEVER_DEGRADE_PRIORITY": ("never_degrade_priority", int),
        "MEMSHADOW_NEVER_DROP_PRIORITY": ("never_drop_priority", int),
    }
    
    for env_key, (field_name, field_type) in bg_env_mapping.items():
        if env_key not in os.environ:
            continue
        value = os.environ[env_key]
        try:
            if field_type == bool:
                bg_overrides[field_name] = value.lower() in {"1", "true", "yes", "on"}
            else:
                bg_overrides[field_name] = field_type(value)
        except (ValueError, TypeError):
            continue
    
    for env_key, (field_name, field_type) in dp_env_mapping.items():
        if env_key not in os.environ:
            continue
        value = os.environ[env_key]
        try:
            dp_overrides[field_name] = field_type(value)
        except (ValueError, TypeError):
            continue
    
    # Apply governor overrides if any
    if bg_overrides or dp_overrides:
        current_bg = config.bandwidth_governor
        current_dp = current_bg.degradation_policy
        
        if dp_overrides:
            new_dp = replace(current_dp, **dp_overrides)
            bg_overrides["degradation_policy"] = new_dp
        
        if bg_overrides:
            new_bg = replace(current_bg, **bg_overrides)
            overrides["bandwidth_governor"] = new_bg

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
        
        # Parse basic settings
        basic_overrides = {}
        basic_keys = {
            "background_sync_interval_seconds",
            "max_batch_items",
            "compression_threshold_bytes",
            "enable_p2p_for_critical",
            "enable_shrink_ingest",
            "enable_psych_ingest",
            "enable_threat_ingest",
            "enable_memory_ingest",
            "enable_federation_ingest",
            "enable_improvement_ingest",
        }
        for key in basic_keys:
            if key in file_settings:
                basic_overrides[key] = file_settings[key]
        
        # Parse bandwidth governor config
        if "bandwidth_governor" in file_settings:
            basic_overrides["bandwidth_governor"] = _parse_bandwidth_governor_config(file_settings)
        
        if basic_overrides:
            config = replace(config, **basic_overrides)
        break

    return _apply_env_overrides(config)


@lru_cache(maxsize=1)
def get_memshadow_config(config_path: Optional[str] = None) -> MemshadowConfig:
    """
    Cached accessor for shared configuration consumers.
    """
    return load_memshadow_config(config_path)


def clear_memshadow_config_cache() -> None:
    """Clear the cached config (useful for testing or config reload)."""
    get_memshadow_config.cache_clear()

