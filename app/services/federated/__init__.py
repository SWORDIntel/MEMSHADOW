"""
Federated & Swarm Memory System
Phase 8.1: Distributed memory with privacy preservation

Components:
- Federated Coordinator: Manages distributed memory nodes
- Differential Privacy: Privacy-preserving aggregation
- Gossip Protocol: Peer-to-peer knowledge sharing
- Secure Aggregation: Encrypted gradient aggregation
- CRDT: Conflict-free replicated data types
"""

from app.services.federated.coordinator import FederatedCoordinator, coordinator
from app.services.federated.privacy import DifferentialPrivacy, PrivacyBudget
from app.services.federated.gossip import GossipProtocol, GossipMessage
from app.services.federated.aggregation import SecureAggregator
from app.services.federated.crdt import MemoryCRDT

__all__ = [
    "FederatedCoordinator",
    "coordinator",
    "DifferentialPrivacy",
    "PrivacyBudget",
    "GossipProtocol",
    "GossipMessage",
    "SecureAggregator",
    "MemoryCRDT",
]
