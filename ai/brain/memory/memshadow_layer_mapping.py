#!/usr/bin/env python3
"""
MEMSHADOW Layer & Device Mapping for DSMIL HIL Integration

Maps MEMSHADOW message categories to DSMIL hardware integration layers (2-9)
and device IDs as specified in the Master Plan and Advanced Layers guide.

Layer Architecture (from 00_MASTER_PLAN_OVERVIEW):
  - Layer 2: Physical Security (sensors, cameras)
  - Layer 3: Network Infrastructure (switches, routers)
  - Layer 4: Data Processing (edge compute, gateways)
  - Layer 5: Analytics & Correlation
  - Layer 6: Federation & Mesh (inter-node communication)
  - Layer 7: Primary AI / Memory Fabric (main DSMIL brain)
  - Layer 8: Security Analytics / Threat Intel / Behavioral Biometrics
  - Layer 9: Decision Support / Advisory / Human-in-the-Loop

Memory Bandwidth Reference (from 03_MEMORY_BANDWIDTH_OPTIMIZATION):
  - Total system: 64 GB/s aggregate bandwidth
  - Layer 7 allocation: ~62 GB available
  - MEMSHADOW traffic target: configurable % of layer bandwidth
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

__all__ = [
    "DSMILLayer",
    "MemshadowCategory",
    "LayerDeviceInfo",
    "LayerMapping",
    "get_target_layers_for_category",
    "get_devices_for_category",
    "get_layer_budget_bytes_per_sec",
    "get_category_layer_mapping",
    "get_all_layer_mappings",
    "LAYER_MEMORY_BUDGETS_GB",
    "LAYER_BANDWIDTH_BUDGETS_GBPS",
]


class DSMILLayer(IntEnum):
    """
    DSMIL Hardware Integration Layers.
    
    These correspond to the layered architecture defined in the Master Plan.
    """
    PHYSICAL_SECURITY = 2
    NETWORK_INFRASTRUCTURE = 3
    DATA_PROCESSING = 4
    ANALYTICS_CORRELATION = 5
    FEDERATION_MESH = 6
    PRIMARY_AI_MEMORY = 7
    SECURITY_ANALYTICS = 8
    DECISION_SUPPORT = 9


class MemshadowCategory(IntEnum):
    """
    MEMSHADOW message categories derived from MessageType ranges.
    
    0x01xx -> PSYCH
    0x02xx -> THREAT
    0x03xx -> MEMORY
    0x04xx -> FEDERATION
    0x05xx -> IMPROVEMENT
    """
    PSYCH = 1
    THREAT = 2
    MEMORY = 3
    FEDERATION = 4
    IMPROVEMENT = 5
    UNKNOWN = 0

    @classmethod
    def from_msg_type_value(cls, msg_type_value: int) -> "MemshadowCategory":
        """Derive category from a MEMSHADOW MessageType numeric value."""
        if 0x0100 <= msg_type_value <= 0x01FF:
            return cls.PSYCH
        if 0x0200 <= msg_type_value <= 0x02FF:
            return cls.THREAT
        if 0x0300 <= msg_type_value <= 0x03FF:
            return cls.MEMORY
        if 0x0400 <= msg_type_value <= 0x04FF:
            return cls.FEDERATION
        if 0x0500 <= msg_type_value <= 0x05FF:
            return cls.IMPROVEMENT
        return cls.UNKNOWN

    @classmethod
    def from_string(cls, name: str) -> "MemshadowCategory":
        """Convert category name string to enum."""
        mapping = {
            "psych": cls.PSYCH,
            "threat": cls.THREAT,
            "memory": cls.MEMORY,
            "federation": cls.FEDERATION,
            "improvement": cls.IMPROVEMENT,
        }
        return mapping.get(name.lower(), cls.UNKNOWN)


@dataclass(frozen=True)
class LayerDeviceInfo:
    """
    Information about devices in a specific layer.
    
    Attributes:
        layer: The DSMIL layer this device belongs to
        device_id: Unique device identifier (e.g., "L7-AI-CORE-01")
        device_type: Type of device (e.g., "ai_processor", "memory_controller")
        memory_allocation_bytes: Memory allocated to this device
        bandwidth_allocation_bps: Bandwidth allocated (bytes per second)
        is_primary: Whether this is a primary device for the category
    """
    layer: DSMILLayer
    device_id: str
    device_type: str
    memory_allocation_bytes: int = 0
    bandwidth_allocation_bps: int = 0
    is_primary: bool = False


@dataclass
class LayerMapping:
    """
    Complete mapping for a MEMSHADOW category to layers and devices.
    
    Attributes:
        category: The MEMSHADOW category
        primary_layers: Primary destination layers (always receive data)
        secondary_layers: Secondary layers (receive on demand or overflow)
        devices: List of devices that handle this category
        priority_weight: Relative priority for bandwidth allocation (1-10)
        allows_degradation: Whether this category can be degraded under load
        min_priority_for_layer8_hook: Minimum priority level for Layer 8 hooks
    """
    category: MemshadowCategory
    primary_layers: FrozenSet[DSMILLayer]
    secondary_layers: FrozenSet[DSMILLayer] = field(default_factory=frozenset)
    devices: Tuple[LayerDeviceInfo, ...] = field(default_factory=tuple)
    priority_weight: int = 5
    allows_degradation: bool = True
    min_priority_for_layer8_hook: int = 2  # Priority.HIGH and above

    @property
    def all_layers(self) -> FrozenSet[DSMILLayer]:
        """All layers (primary + secondary)."""
        return self.primary_layers | self.secondary_layers


# =============================================================================
# Layer Memory & Bandwidth Budgets (from Master Plan docs)
# =============================================================================

# Memory budgets per layer (in GB)
LAYER_MEMORY_BUDGETS_GB: Dict[DSMILLayer, float] = {
    DSMILLayer.PHYSICAL_SECURITY: 2.0,
    DSMILLayer.NETWORK_INFRASTRUCTURE: 4.0,
    DSMILLayer.DATA_PROCESSING: 8.0,
    DSMILLayer.ANALYTICS_CORRELATION: 12.0,
    DSMILLayer.FEDERATION_MESH: 6.0,
    DSMILLayer.PRIMARY_AI_MEMORY: 20.0,  # Largest allocation for AI core
    DSMILLayer.SECURITY_ANALYTICS: 8.0,
    DSMILLayer.DECISION_SUPPORT: 2.0,
}

# Bandwidth budgets per layer (in GB/s) - total ~64 GB/s
LAYER_BANDWIDTH_BUDGETS_GBPS: Dict[DSMILLayer, float] = {
    DSMILLayer.PHYSICAL_SECURITY: 2.0,
    DSMILLayer.NETWORK_INFRASTRUCTURE: 8.0,
    DSMILLayer.DATA_PROCESSING: 10.0,
    DSMILLayer.ANALYTICS_CORRELATION: 12.0,
    DSMILLayer.FEDERATION_MESH: 6.0,
    DSMILLayer.PRIMARY_AI_MEMORY: 20.0,
    DSMILLayer.SECURITY_ANALYTICS: 4.0,
    DSMILLayer.DECISION_SUPPORT: 2.0,
}


def get_layer_budget_bytes_per_sec(layer: DSMILLayer) -> int:
    """
    Get the bandwidth budget for a layer in bytes per second.
    
    Args:
        layer: The DSMIL layer
        
    Returns:
        Bandwidth budget in bytes/sec
    """
    gbps = LAYER_BANDWIDTH_BUDGETS_GBPS.get(layer, 1.0)
    return int(gbps * 1_000_000_000)  # Convert GB/s to B/s


# =============================================================================
# Category -> Layer Mappings
# =============================================================================

# Pre-defined device configurations
_PSYCH_DEVICES = (
    LayerDeviceInfo(
        layer=DSMILLayer.PRIMARY_AI_MEMORY,
        device_id="L7-SHRINK-PROC-01",
        device_type="psych_processor",
        memory_allocation_bytes=2 * 1024**3,  # 2 GB
        bandwidth_allocation_bps=int(2 * 1e9),  # 2 GB/s
        is_primary=True,
    ),
    LayerDeviceInfo(
        layer=DSMILLayer.SECURITY_ANALYTICS,
        device_id="L8-BEHAVIORAL-01",
        device_type="behavioral_biometrics",
        memory_allocation_bytes=1 * 1024**3,  # 1 GB
        bandwidth_allocation_bps=int(1 * 1e9),  # 1 GB/s
        is_primary=False,
    ),
)

_THREAT_DEVICES = (
    LayerDeviceInfo(
        layer=DSMILLayer.SECURITY_ANALYTICS,
        device_id="L8-THREAT-INTEL-01",
        device_type="threat_intel_engine",
        memory_allocation_bytes=4 * 1024**3,  # 4 GB
        bandwidth_allocation_bps=int(3 * 1e9),  # 3 GB/s
        is_primary=True,
    ),
    LayerDeviceInfo(
        layer=DSMILLayer.ANALYTICS_CORRELATION,
        device_id="L5-CORRELATOR-01",
        device_type="event_correlator",
        memory_allocation_bytes=2 * 1024**3,
        bandwidth_allocation_bps=int(2 * 1e9),
        is_primary=False,
    ),
)

_MEMORY_DEVICES = (
    LayerDeviceInfo(
        layer=DSMILLayer.PRIMARY_AI_MEMORY,
        device_id="L7-MEMFABRIC-01",
        device_type="memory_fabric_controller",
        memory_allocation_bytes=16 * 1024**3,  # 16 GB
        bandwidth_allocation_bps=int(15 * 1e9),  # 15 GB/s
        is_primary=True,
    ),
)

_FEDERATION_DEVICES = (
    LayerDeviceInfo(
        layer=DSMILLayer.FEDERATION_MESH,
        device_id="L6-MESH-HUB-01",
        device_type="mesh_hub",
        memory_allocation_bytes=4 * 1024**3,
        bandwidth_allocation_bps=int(5 * 1e9),
        is_primary=True,
    ),
    LayerDeviceInfo(
        layer=DSMILLayer.PRIMARY_AI_MEMORY,
        device_id="L7-FED-CACHE-01",
        device_type="federation_cache",
        memory_allocation_bytes=2 * 1024**3,
        bandwidth_allocation_bps=int(2 * 1e9),
        is_primary=False,
    ),
    LayerDeviceInfo(
        layer=DSMILLayer.SECURITY_ANALYTICS,
        device_id="L8-FED-SECURITY-01",
        device_type="federation_security",
        memory_allocation_bytes=1 * 1024**3,
        bandwidth_allocation_bps=int(1 * 1e9),
        is_primary=False,
    ),
)

_IMPROVEMENT_DEVICES = (
    LayerDeviceInfo(
        layer=DSMILLayer.PRIMARY_AI_MEMORY,
        device_id="L7-IMPROVE-CORE-01",
        device_type="improvement_engine",
        memory_allocation_bytes=4 * 1024**3,
        bandwidth_allocation_bps=int(3 * 1e9),
        is_primary=True,
    ),
    LayerDeviceInfo(
        layer=DSMILLayer.SECURITY_ANALYTICS,
        device_id="L8-IMPROVE-VALIDATOR-01",
        device_type="improvement_validator",
        memory_allocation_bytes=1 * 1024**3,
        bandwidth_allocation_bps=int(500 * 1e6),  # 500 MB/s
        is_primary=False,
    ),
    LayerDeviceInfo(
        layer=DSMILLayer.DECISION_SUPPORT,
        device_id="L9-IMPROVE-ADVISOR-01",
        device_type="improvement_advisor",
        memory_allocation_bytes=512 * 1024**2,  # 512 MB
        bandwidth_allocation_bps=int(100 * 1e6),  # 100 MB/s
        is_primary=False,
    ),
)


# Complete category mappings
_CATEGORY_MAPPINGS: Dict[MemshadowCategory, LayerMapping] = {
    # PSYCH: Behavioral biometrics from SHRINK
    # Primary: Layer 7 (AI core) + Layer 8 (Behavioral Biometrics)
    MemshadowCategory.PSYCH: LayerMapping(
        category=MemshadowCategory.PSYCH,
        primary_layers=frozenset({DSMILLayer.PRIMARY_AI_MEMORY, DSMILLayer.SECURITY_ANALYTICS}),
        secondary_layers=frozenset({DSMILLayer.DECISION_SUPPORT}),
        devices=_PSYCH_DEVICES,
        priority_weight=7,
        allows_degradation=True,
        min_priority_for_layer8_hook=2,  # HIGH and above
    ),
    
    # THREAT: Threat intelligence
    # Primary: Layer 8 (Threat Intel, Network Security AI)
    MemshadowCategory.THREAT: LayerMapping(
        category=MemshadowCategory.THREAT,
        primary_layers=frozenset({DSMILLayer.SECURITY_ANALYTICS}),
        secondary_layers=frozenset({DSMILLayer.ANALYTICS_CORRELATION, DSMILLayer.DECISION_SUPPORT}),
        devices=_THREAT_DEVICES,
        priority_weight=9,  # High priority for threat intel
        allows_degradation=False,  # Never degrade threat intel
        min_priority_for_layer8_hook=1,  # NORMAL and above
    ),
    
    # MEMORY: L1/L2/L3 sync operations
    # Primary: Layer 7 (primary AI / memory fabric)
    MemshadowCategory.MEMORY: LayerMapping(
        category=MemshadowCategory.MEMORY,
        primary_layers=frozenset({DSMILLayer.PRIMARY_AI_MEMORY}),
        secondary_layers=frozenset({DSMILLayer.FEDERATION_MESH}),
        devices=_MEMORY_DEVICES,
        priority_weight=8,
        allows_degradation=True,
        min_priority_for_layer8_hook=3,  # CRITICAL only
    ),
    
    # FEDERATION/MESH: Inter-node communication
    # Primary: Layer 6-8 federation/mesh
    MemshadowCategory.FEDERATION: LayerMapping(
        category=MemshadowCategory.FEDERATION,
        primary_layers=frozenset({DSMILLayer.FEDERATION_MESH, DSMILLayer.PRIMARY_AI_MEMORY}),
        secondary_layers=frozenset({DSMILLayer.SECURITY_ANALYTICS}),
        devices=_FEDERATION_DEVICES,
        priority_weight=6,
        allows_degradation=True,
        min_priority_for_layer8_hook=2,
    ),
    
    # IMPROVEMENT: Self-improvement / advisory flows
    # Primary: Layer 7 + Layer 8/9 advisory
    MemshadowCategory.IMPROVEMENT: LayerMapping(
        category=MemshadowCategory.IMPROVEMENT,
        primary_layers=frozenset({DSMILLayer.PRIMARY_AI_MEMORY}),
        secondary_layers=frozenset({DSMILLayer.SECURITY_ANALYTICS, DSMILLayer.DECISION_SUPPORT}),
        devices=_IMPROVEMENT_DEVICES,
        priority_weight=5,
        allows_degradation=True,
        min_priority_for_layer8_hook=2,
    ),
    
    # UNKNOWN: Default fallback
    MemshadowCategory.UNKNOWN: LayerMapping(
        category=MemshadowCategory.UNKNOWN,
        primary_layers=frozenset({DSMILLayer.PRIMARY_AI_MEMORY}),
        secondary_layers=frozenset(),
        devices=(),
        priority_weight=1,
        allows_degradation=True,
        min_priority_for_layer8_hook=3,
    ),
}


# =============================================================================
# Public API Functions
# =============================================================================

def get_target_layers_for_category(
    category: MemshadowCategory | str | int,
    include_secondary: bool = True
) -> FrozenSet[DSMILLayer]:
    """
    Get the target layers for a MEMSHADOW category.
    
    Args:
        category: Category as enum, string name, or MessageType value
        include_secondary: Whether to include secondary (overflow) layers
        
    Returns:
        Set of DSMIL layers that handle this category
        
    Example:
        >>> layers = get_target_layers_for_category(MemshadowCategory.PSYCH)
        >>> DSMILLayer.PRIMARY_AI_MEMORY in layers
        True
    """
    cat = _normalize_category(category)
    mapping = _CATEGORY_MAPPINGS.get(cat, _CATEGORY_MAPPINGS[MemshadowCategory.UNKNOWN])
    
    if include_secondary:
        return mapping.all_layers
    return mapping.primary_layers


def get_devices_for_category(
    category: MemshadowCategory | str | int,
    layer: Optional[DSMILLayer] = None,
    primary_only: bool = False
) -> Tuple[LayerDeviceInfo, ...]:
    """
    Get the devices that handle a specific MEMSHADOW category.
    
    Args:
        category: Category as enum, string name, or MessageType value
        layer: Optional filter to specific layer
        primary_only: Only return primary devices
        
    Returns:
        Tuple of LayerDeviceInfo for matching devices
        
    Example:
        >>> devices = get_devices_for_category("psych", layer=DSMILLayer.PRIMARY_AI_MEMORY)
        >>> len(devices) >= 1
        True
    """
    cat = _normalize_category(category)
    mapping = _CATEGORY_MAPPINGS.get(cat, _CATEGORY_MAPPINGS[MemshadowCategory.UNKNOWN])
    
    devices = mapping.devices
    
    if layer is not None:
        devices = tuple(d for d in devices if d.layer == layer)
    
    if primary_only:
        devices = tuple(d for d in devices if d.is_primary)
    
    return devices


def get_category_layer_mapping(category: MemshadowCategory | str | int) -> LayerMapping:
    """
    Get the complete layer mapping for a category.
    
    Args:
        category: Category as enum, string name, or MessageType value
        
    Returns:
        LayerMapping with all routing information
    """
    cat = _normalize_category(category)
    return _CATEGORY_MAPPINGS.get(cat, _CATEGORY_MAPPINGS[MemshadowCategory.UNKNOWN])


def get_all_layer_mappings() -> Dict[MemshadowCategory, LayerMapping]:
    """
    Get all category -> layer mappings.
    
    Returns:
        Dictionary of all mappings
    """
    return dict(_CATEGORY_MAPPINGS)


def should_trigger_layer8_hook(
    category: MemshadowCategory | str | int,
    priority: int
) -> bool:
    """
    Check if a message should trigger Layer 8 security hooks.
    
    Args:
        category: The MEMSHADOW category
        priority: The message priority level (0-4)
        
    Returns:
        True if Layer 8 hooks should be triggered
    """
    cat = _normalize_category(category)
    mapping = _CATEGORY_MAPPINGS.get(cat, _CATEGORY_MAPPINGS[MemshadowCategory.UNKNOWN])
    
    # Layer 8 must be in target layers
    if DSMILLayer.SECURITY_ANALYTICS not in mapping.all_layers:
        return False
    
    return priority >= mapping.min_priority_for_layer8_hook


def get_category_priority_weight(category: MemshadowCategory | str | int) -> int:
    """
    Get the priority weight for bandwidth allocation.
    
    Higher weight = higher priority for bandwidth allocation.
    
    Args:
        category: The MEMSHADOW category
        
    Returns:
        Priority weight (1-10)
    """
    cat = _normalize_category(category)
    mapping = _CATEGORY_MAPPINGS.get(cat, _CATEGORY_MAPPINGS[MemshadowCategory.UNKNOWN])
    return mapping.priority_weight


def can_degrade_category(category: MemshadowCategory | str | int) -> bool:
    """
    Check if a category can be degraded under load.
    
    Args:
        category: The MEMSHADOW category
        
    Returns:
        True if degradation is allowed
    """
    cat = _normalize_category(category)
    mapping = _CATEGORY_MAPPINGS.get(cat, _CATEGORY_MAPPINGS[MemshadowCategory.UNKNOWN])
    return mapping.allows_degradation


# =============================================================================
# Internal Helpers
# =============================================================================

def _normalize_category(category: MemshadowCategory | str | int) -> MemshadowCategory:
    """Normalize category input to MemshadowCategory enum."""
    if isinstance(category, MemshadowCategory):
        return category
    if isinstance(category, str):
        return MemshadowCategory.from_string(category)
    if isinstance(category, int):
        # Could be MessageType value or category value
        if category <= 5:
            return MemshadowCategory(category)
        return MemshadowCategory.from_msg_type_value(category)
    return MemshadowCategory.UNKNOWN


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("MEMSHADOW Layer Mapping Self-Test")
    print("=" * 60)
    
    print("\n[1] Layer Memory Budgets:")
    for layer, gb in sorted(LAYER_MEMORY_BUDGETS_GB.items(), key=lambda x: x[0].value):
        print(f"    {layer.name}: {gb:.1f} GB")
    
    print("\n[2] Layer Bandwidth Budgets:")
    total_bw = 0.0
    for layer, gbps in sorted(LAYER_BANDWIDTH_BUDGETS_GBPS.items(), key=lambda x: x[0].value):
        print(f"    {layer.name}: {gbps:.1f} GB/s")
        total_bw += gbps
    print(f"    TOTAL: {total_bw:.1f} GB/s")
    
    print("\n[3] Category -> Layer Mappings:")
    for cat in MemshadowCategory:
        if cat == MemshadowCategory.UNKNOWN:
            continue
        layers = get_target_layers_for_category(cat)
        devices = get_devices_for_category(cat)
        weight = get_category_priority_weight(cat)
        degradable = can_degrade_category(cat)
        print(f"    {cat.name}:")
        print(f"        Layers: {[l.name for l in sorted(layers, key=lambda x: x.value)]}")
        print(f"        Devices: {len(devices)}")
        print(f"        Priority Weight: {weight}/10")
        print(f"        Degradable: {degradable}")
    
    print("\n[4] Layer 8 Hook Triggers:")
    for cat in [MemshadowCategory.PSYCH, MemshadowCategory.THREAT]:
        for priority in range(5):
            triggers = should_trigger_layer8_hook(cat, priority)
            print(f"    {cat.name} @ priority {priority}: {'YES' if triggers else 'no'}")
    
    print("\n" + "=" * 60)
    print("MEMSHADOW Layer Mapping test complete")
