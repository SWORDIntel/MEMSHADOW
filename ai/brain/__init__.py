#!/usr/bin/env python3
"""
DSMIL Second Brain: Distributed Intelligence Architecture

"NSA in a Program" - Distributed AI memory across many points with constant
intelligence ingestion, analysis, cross-correlation, and autonomous improvement.

Central Hub Query Model:
- All intelligence queries originate from the DSMIL central hub
- Nodes receive queries, correlate locally, return results
- Hub aggregates and synthesizes distributed responses

Architecture Layers:
- Layer 0: Federated Node Network
- Layer 1: CNSA Security Core
- Layer 2: Distributed Memory Fabric
- Layer 3: Self-Improving Vector Database
- Layer 4: Predictive Intelligence Engine
- Layer 5: Counter-Intelligence Suite
- Layer 6: Temporal & 4D Intelligence
- Layer 7: Covert Operations
- Layer 8: Digital Immune System
- Layer 9: Intelligence Fusion
- Layer 10: Homomorphic Intelligence
- Layer 11: Autonomous Operations
- Layer 12: Meta-Intelligence
- Layer 13: Data Formats & Ingestion
- Layer 14: Swarm Cognition
"""

__version__ = "1.0.0"
__codename__ = "CEREBRUM"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .brain_interface import DSMILBrain

# Lazy imports to avoid circular dependencies
def get_brain() -> "DSMILBrain":
    """Get the singleton brain instance"""
    from .brain_interface import DSMILBrain
    return DSMILBrain.get_instance()

__all__ = [
    "get_brain",
    "__version__",
    "__codename__",
]

