#!/usr/bin/env python3
"""
DSMIL Brain Hardware Layer

Hardware detection and adaptive compute routing:
- Adaptive Compute: Hardware capability detection
- Compute Router: Optimal task routing based on resources
"""

from .adaptive_compute import (
    AdaptiveCompute,
    HardwareCapabilities,
    ComputeResource,
)

from .compute_router import (
    ComputeRouter,
    TaskRequirements,
    RoutingDecision,
)

__all__ = [
    "AdaptiveCompute", "HardwareCapabilities", "ComputeResource",
    "ComputeRouter", "TaskRequirements", "RoutingDecision",
]

