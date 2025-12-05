#!/usr/bin/env python3
"""
Offline Coordinator for DSMIL Brain Federation

Manages peer coordination when hub is offline:
- Peer discovery and connection
- Consensus protocols for queries
- Work distribution among peers
- Conflict resolution
- Eventual consistency
"""

import asyncio
import hashlib
import threading
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Set, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from collections import defaultdict
import json
import random

logger = logging.getLogger(__name__)


class ConsensusType(Enum):
    """Types of consensus protocols"""
    SIMPLE_MAJORITY = auto()  # >50% agreement
    SUPER_MAJORITY = auto()   # >66% agreement
    UNANIMOUS = auto()        # 100% agreement
    WEIGHTED = auto()         # Weighted by trust scores


class PeerState(Enum):
    """State of a peer connection"""
    UNKNOWN = auto()
    DISCOVERED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    AUTHENTICATED = auto()
    SYNCED = auto()
    DISCONNECTED = auto()
    UNTRUSTED = auto()


@dataclass
class PeerNode:
    """Information about a peer node"""
    node_id: str
    endpoint: str
    state: PeerState = PeerState.UNKNOWN

    # Authentication
    public_key: bytes = b""
    is_authenticated: bool = False

    # Trust
    trust_score: float = 0.5

    # Connection quality
    latency_ms: float = 0.0
    last_contact: Optional[datetime] = None
    failed_attempts: int = 0

    # Capabilities
    capabilities: Set[str] = field(default_factory=set)
    data_domains: Set[str] = field(default_factory=set)

    # Sync state
    sync_version: int = 0
    last_sync: Optional[datetime] = None


@dataclass
class ConsensusProposal:
    """A proposal for consensus"""
    proposal_id: str
    proposer_id: str
    proposal_type: str  # "query", "intel", "config"
    content: Dict

    # Voting
    votes_for: Set[str] = field(default_factory=set)
    votes_against: Set[str] = field(default_factory=set)

    # Status
    is_decided: bool = False
    decision: Optional[bool] = None

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deadline: Optional[datetime] = None


class ConsensusProtocol:
    """
    Implements consensus among peer nodes

    Used when hub is offline and nodes need to agree
    on queries, intel, or actions.
    """

    def __init__(self, node_id: str,
                 consensus_type: ConsensusType = ConsensusType.SIMPLE_MAJORITY):
        self.node_id = node_id
        self.consensus_type = consensus_type

        self._proposals: Dict[str, ConsensusProposal] = {}
        self._peer_trust: Dict[str, float] = {}
        self._lock = threading.RLock()

    def set_peer_trust(self, peer_id: str, trust: float):
        """Set trust score for a peer"""
        self._peer_trust[peer_id] = trust

    def propose(self, proposal_type: str, content: Dict,
                deadline_seconds: float = 30.0) -> ConsensusProposal:
        """
        Create a new consensus proposal

        Args:
            proposal_type: Type of proposal
            content: Proposal content
            deadline_seconds: Voting deadline

        Returns:
            ConsensusProposal instance
        """
        proposal_id = hashlib.sha256(
            f"{self.node_id}:{proposal_type}:{time.time()}".encode()
        ).hexdigest()[:16]

        deadline = datetime.now(timezone.utc) + timedelta(seconds=deadline_seconds)

        proposal = ConsensusProposal(
            proposal_id=proposal_id,
            proposer_id=self.node_id,
            proposal_type=proposal_type,
            content=content,
            deadline=deadline,
        )

        # Proposer votes for their own proposal
        proposal.votes_for.add(self.node_id)

        with self._lock:
            self._proposals[proposal_id] = proposal

        return proposal

    def vote(self, proposal_id: str, voter_id: str, vote_for: bool) -> bool:
        """
        Vote on a proposal

        Args:
            proposal_id: Proposal to vote on
            voter_id: ID of voting node
            vote_for: True = for, False = against

        Returns:
            True if vote recorded
        """
        with self._lock:
            if proposal_id not in self._proposals:
                return False

            proposal = self._proposals[proposal_id]

            if proposal.is_decided:
                return False

            if vote_for:
                proposal.votes_for.add(voter_id)
                proposal.votes_against.discard(voter_id)
            else:
                proposal.votes_against.add(voter_id)
                proposal.votes_for.discard(voter_id)

            return True

    def check_consensus(self, proposal_id: str,
                       total_voters: int) -> Tuple[bool, Optional[bool]]:
        """
        Check if consensus has been reached

        Args:
            proposal_id: Proposal to check
            total_voters: Total number of potential voters

        Returns:
            (is_decided, decision) - decision is None if not decided
        """
        with self._lock:
            if proposal_id not in self._proposals:
                return False, None

            proposal = self._proposals[proposal_id]

            if proposal.is_decided:
                return True, proposal.decision

            votes_for = len(proposal.votes_for)
            votes_against = len(proposal.votes_against)
            total_votes = votes_for + votes_against

            # Check deadline
            if proposal.deadline and datetime.now(timezone.utc) > proposal.deadline:
                # Deadline passed - decide based on current votes
                if total_votes == 0:
                    proposal.is_decided = True
                    proposal.decision = False
                    return True, False

            # Check consensus based on type
            if self.consensus_type == ConsensusType.SIMPLE_MAJORITY:
                threshold = total_voters / 2
                if votes_for > threshold:
                    proposal.is_decided = True
                    proposal.decision = True
                elif votes_against >= (total_voters - threshold):
                    proposal.is_decided = True
                    proposal.decision = False

            elif self.consensus_type == ConsensusType.SUPER_MAJORITY:
                threshold = total_voters * 2 / 3
                if votes_for > threshold:
                    proposal.is_decided = True
                    proposal.decision = True
                elif votes_against > (total_voters - threshold):
                    proposal.is_decided = True
                    proposal.decision = False

            elif self.consensus_type == ConsensusType.UNANIMOUS:
                if votes_for == total_voters:
                    proposal.is_decided = True
                    proposal.decision = True
                elif votes_against > 0:
                    proposal.is_decided = True
                    proposal.decision = False

            elif self.consensus_type == ConsensusType.WEIGHTED:
                # Weighted by trust scores
                weight_for = sum(
                    self._peer_trust.get(v, 0.5)
                    for v in proposal.votes_for
                )
                weight_against = sum(
                    self._peer_trust.get(v, 0.5)
                    for v in proposal.votes_against
                )
                total_weight = sum(
                    self._peer_trust.get(p, 0.5)
                    for p in range(total_voters)  # Simplified
                )

                if weight_for > total_weight / 2:
                    proposal.is_decided = True
                    proposal.decision = True
                elif weight_against >= total_weight / 2:
                    proposal.is_decided = True
                    proposal.decision = False

            return proposal.is_decided, proposal.decision if proposal.is_decided else None

    def get_proposal(self, proposal_id: str) -> Optional[ConsensusProposal]:
        """Get proposal by ID"""
        return self._proposals.get(proposal_id)


class PeerNetwork:
    """
    Manages peer-to-peer network for offline operation
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._peers: Dict[str, PeerNode] = {}
        self._lock = threading.RLock()

    def add_peer(self, node_id: str, endpoint: str,
                 public_key: bytes = b"") -> PeerNode:
        """Add a peer to the network"""
        with self._lock:
            peer = PeerNode(
                node_id=node_id,
                endpoint=endpoint,
                public_key=public_key,
                state=PeerState.DISCOVERED,
            )
            self._peers[node_id] = peer
            return peer

    def remove_peer(self, node_id: str):
        """Remove a peer from the network"""
        with self._lock:
            if node_id in self._peers:
                del self._peers[node_id]

    def get_peer(self, node_id: str) -> Optional[PeerNode]:
        """Get peer by ID"""
        return self._peers.get(node_id)

    def get_connected_peers(self) -> List[PeerNode]:
        """Get all connected peers"""
        with self._lock:
            return [
                p for p in self._peers.values()
                if p.state in (PeerState.CONNECTED, PeerState.AUTHENTICATED, PeerState.SYNCED)
            ]

    def get_peer_count(self) -> int:
        """Get total peer count"""
        return len(self._peers)

    def update_peer_state(self, node_id: str, state: PeerState):
        """Update peer state"""
        with self._lock:
            if node_id in self._peers:
                self._peers[node_id].state = state

    def record_contact(self, node_id: str, latency_ms: float):
        """Record successful contact with peer"""
        with self._lock:
            if node_id in self._peers:
                peer = self._peers[node_id]
                peer.last_contact = datetime.now(timezone.utc)
                peer.latency_ms = latency_ms
                peer.failed_attempts = 0


class OfflineCoordinator:
    """
    Coordinates node operation when hub is offline

    Features:
    - Peer discovery and management
    - Consensus-based decision making
    - Work distribution among peers
    - State synchronization

    Usage:
        coordinator = OfflineCoordinator(node_id="node-001")

        # Discover and connect to peers
        await coordinator.discover_peers()

        # Propose a query for consensus
        result = await coordinator.consensus_query(query)

        # Distribute work
        await coordinator.distribute_work(tasks)
    """

    def __init__(self, node_id: str,
                 consensus_type: ConsensusType = ConsensusType.SIMPLE_MAJORITY):
        """
        Initialize offline coordinator

        Args:
            node_id: This node's ID
            consensus_type: Type of consensus to use
        """
        self.node_id = node_id

        # Components
        self.peer_network = PeerNetwork(node_id)
        self.consensus = ConsensusProtocol(node_id, consensus_type)

        # Discovery
        self._discovery_endpoints: List[str] = []
        self._known_peers: Dict[str, str] = {}  # node_id -> last known endpoint

        # Work distribution
        self._pending_work: Dict[str, Dict] = {}
        self._work_assignments: Dict[str, str] = {}  # work_id -> assigned_node

        # Sync state
        self._local_version = 0
        self._peer_versions: Dict[str, int] = {}

        # Background tasks
        self._running = False
        self._coordination_thread: Optional[threading.Thread] = None

        # Statistics
        self.stats = {
            "peers_discovered": 0,
            "consensus_proposals": 0,
            "work_distributed": 0,
            "syncs_completed": 0,
        }

        logger.info(f"OfflineCoordinator initialized for {node_id}")

    def add_discovery_endpoint(self, endpoint: str):
        """Add an endpoint for peer discovery"""
        if endpoint not in self._discovery_endpoints:
            self._discovery_endpoints.append(endpoint)

    def remember_peer(self, node_id: str, endpoint: str):
        """Remember a peer's endpoint for reconnection"""
        self._known_peers[node_id] = endpoint

    async def discover_peers(self) -> List[PeerNode]:
        """
        Discover available peers

        Returns:
            List of discovered peers
        """
        discovered = []

        # Try known peers first
        for node_id, endpoint in self._known_peers.items():
            try:
                # Would do actual network probe
                peer = self.peer_network.add_peer(node_id, endpoint)
                discovered.append(peer)
                self.stats["peers_discovered"] += 1
            except Exception as e:
                logger.debug(f"Known peer {node_id} unreachable: {e}")

        # Try discovery endpoints
        for endpoint in self._discovery_endpoints:
            try:
                # Would do actual discovery protocol
                pass
            except Exception as e:
                logger.debug(f"Discovery endpoint {endpoint} failed: {e}")

        return discovered

    async def connect_to_peer(self, peer: PeerNode) -> bool:
        """
        Establish connection with a peer

        Args:
            peer: Peer to connect to

        Returns:
            True if connected
        """
        try:
            self.peer_network.update_peer_state(peer.node_id, PeerState.CONNECTING)

            # Would do actual connection
            # Simulate success

            self.peer_network.update_peer_state(peer.node_id, PeerState.CONNECTED)
            self.peer_network.record_contact(peer.node_id, random.uniform(10, 100))

            return True

        except Exception as e:
            logger.error(f"Failed to connect to peer {peer.node_id}: {e}")
            self.peer_network.update_peer_state(peer.node_id, PeerState.DISCONNECTED)
            return False

    async def consensus_query(self, query: Dict,
                             timeout: float = 30.0) -> Tuple[bool, Optional[Dict]]:
        """
        Submit query for peer consensus

        Args:
            query: Query to submit
            timeout: Timeout for consensus

        Returns:
            (consensus_reached, aggregated_result)
        """
        # Create proposal
        proposal = self.consensus.propose("query", query, timeout)
        self.stats["consensus_proposals"] += 1

        # Broadcast to peers
        peers = self.peer_network.get_connected_peers()

        if not peers:
            logger.warning("No connected peers for consensus")
            return False, None

        # Collect votes (would be async network calls)
        for peer in peers:
            # Simulate peer voting
            vote = random.choice([True, True, True, False])  # Bias toward approval
            self.consensus.vote(proposal.proposal_id, peer.node_id, vote)

        # Check consensus
        is_decided, decision = self.consensus.check_consensus(
            proposal.proposal_id, len(peers) + 1  # +1 for ourselves
        )

        if is_decided and decision:
            # Consensus reached - aggregate results
            # Would actually execute query on peers and aggregate
            return True, {"consensus": True, "query": query}

        return is_decided, None if not decision else {"consensus": False}

    async def distribute_work(self, tasks: List[Dict]) -> Dict[str, str]:
        """
        Distribute work among peers

        Args:
            tasks: List of tasks to distribute

        Returns:
            Dict of task_id -> assigned_node_id
        """
        assignments = {}
        peers = self.peer_network.get_connected_peers()

        if not peers:
            logger.warning("No peers available for work distribution")
            return assignments

        # Simple round-robin distribution
        for i, task in enumerate(tasks):
            task_id = task.get("task_id", f"task-{i}")
            peer = peers[i % len(peers)]

            assignments[task_id] = peer.node_id
            self._work_assignments[task_id] = peer.node_id
            self._pending_work[task_id] = task

            self.stats["work_distributed"] += 1

        return assignments

    async def sync_with_peer(self, peer: PeerNode) -> bool:
        """
        Synchronize state with a peer

        Args:
            peer: Peer to sync with

        Returns:
            True if sync successful
        """
        try:
            # Get peer's version
            peer_version = self._peer_versions.get(peer.node_id, 0)

            if peer_version < self._local_version:
                # We're ahead - send updates to peer
                # Would send delta
                pass
            elif peer_version > self._local_version:
                # Peer is ahead - receive updates
                # Would receive delta
                self._local_version = peer_version

            peer.last_sync = datetime.now(timezone.utc)
            peer.sync_version = self._local_version

            self.stats["syncs_completed"] += 1
            return True

        except Exception as e:
            logger.error(f"Sync with {peer.node_id} failed: {e}")
            return False

    async def sync_all_peers(self):
        """Sync with all connected peers"""
        peers = self.peer_network.get_connected_peers()

        for peer in peers:
            await self.sync_with_peer(peer)

    def start_coordination(self, interval: float = 30.0):
        """Start background coordination"""
        if self._running:
            return

        self._running = True

        async def coordination_loop():
            while self._running:
                try:
                    # Discover new peers
                    await self.discover_peers()

                    # Connect to unconnected peers
                    for peer in self.peer_network._peers.values():
                        if peer.state == PeerState.DISCOVERED:
                            await self.connect_to_peer(peer)

                    # Sync with connected peers
                    await self.sync_all_peers()

                except Exception as e:
                    logger.error(f"Coordination error: {e}")

                await asyncio.sleep(interval)

        def run_loop():
            asyncio.run(coordination_loop())

        self._coordination_thread = threading.Thread(target=run_loop, daemon=True)
        self._coordination_thread.start()
        logger.info("Coordination started")

    def stop_coordination(self):
        """Stop background coordination"""
        self._running = False
        if self._coordination_thread:
            self._coordination_thread.join(timeout=5.0)

    def get_network_status(self) -> Dict:
        """Get network status"""
        peers = self.peer_network.get_connected_peers()

        return {
            "node_id": self.node_id,
            "local_version": self._local_version,
            "total_peers": self.peer_network.get_peer_count(),
            "connected_peers": len(peers),
            "pending_work": len(self._pending_work),
            "stats": self.stats,
        }


if __name__ == "__main__":
    print("Offline Coordinator Self-Test")
    print("=" * 50)

    import asyncio

    coordinator = OfflineCoordinator(
        node_id="test-node-001",
        consensus_type=ConsensusType.SIMPLE_MAJORITY
    )

    print(f"\n[1] Add Known Peers")
    coordinator.remember_peer("node-002", "localhost:8002")
    coordinator.remember_peer("node-003", "localhost:8003")
    coordinator.remember_peer("node-004", "localhost:8004")

    async def test_discovery():
        peers = await coordinator.discover_peers()
        return peers

    peers = asyncio.run(test_discovery())
    print(f"    Discovered {len(peers)} peers")

    print(f"\n[2] Connect to Peers")
    async def test_connect():
        for peer in peers:
            success = await coordinator.connect_to_peer(peer)
            print(f"    {peer.node_id}: {'Connected' if success else 'Failed'}")

    asyncio.run(test_connect())

    print(f"\n[3] Consensus Query")
    async def test_consensus():
        reached, result = await coordinator.consensus_query({
            "query": "What is the threat level?",
            "priority": "high"
        })
        return reached, result

    reached, result = asyncio.run(test_consensus())
    print(f"    Consensus reached: {reached}")
    print(f"    Result: {result}")

    print(f"\n[4] Distribute Work")
    async def test_distribute():
        tasks = [
            {"task_id": "t1", "action": "analyze"},
            {"task_id": "t2", "action": "correlate"},
            {"task_id": "t3", "action": "search"},
        ]
        assignments = await coordinator.distribute_work(tasks)
        return assignments

    assignments = asyncio.run(test_distribute())
    print(f"    Assignments:")
    for task_id, node_id in assignments.items():
        print(f"      {task_id} -> {node_id}")

    print(f"\n[5] Network Status")
    status = coordinator.get_network_status()
    for key, value in status.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 50)
    print("Offline Coordinator test complete")

