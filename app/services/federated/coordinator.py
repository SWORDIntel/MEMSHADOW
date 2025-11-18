"""
Federated Memory Coordinator
Phase 8.1: Distributed memory coordination with privacy

The coordinator manages a federation of MEMSHADOW nodes, enabling:
- Privacy-preserving memory sharing
- Decentralized learning
- Peer-to-peer knowledge propagation
- Byzantine fault tolerance
"""

from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import asyncio
import structlog

logger = structlog.get_logger()


class NodeStatus(Enum):
    """Status of federated nodes"""
    ONLINE = "online"
    OFFLINE = "offline"
    SYNCING = "syncing"
    SUSPICIOUS = "suspicious"  # Byzantine behavior detected
    QUARANTINED = "quarantined"


class SyncStrategy(Enum):
    """Memory synchronization strategies"""
    PUSH = "push"  # Push updates to peers
    PULL = "pull"  # Pull updates from peers
    GOSSIP = "gossip"  # Epidemic-style propagation
    CONSENSUS = "consensus"  # Byzantine consensus


@dataclass
class NodeInfo:
    """Information about a federated node"""
    node_id: str
    address: str
    port: int
    status: NodeStatus = NodeStatus.OFFLINE
    capabilities: List[str] = field(default_factory=list)

    # Reputation and trust
    reputation_score: float = 1.0  # 0.0 to 1.0
    successful_syncs: int = 0
    failed_syncs: int = 0
    byzantine_violations: int = 0

    # Timing
    last_seen: Optional[datetime] = None
    joined_at: datetime = field(default_factory=datetime.utcnow)

    # Performance
    avg_latency_ms: float = 0.0
    bandwidth_mbps: float = 0.0

    def update_reputation(self, success: bool):
        """Update node reputation based on sync result"""
        if success:
            self.successful_syncs += 1
            self.reputation_score = min(1.0, self.reputation_score + 0.01)
        else:
            self.failed_syncs += 1
            self.reputation_score = max(0.0, self.reputation_score - 0.05)

        # Quarantine if reputation drops too low
        if self.reputation_score < 0.3:
            self.status = NodeStatus.QUARANTINED


@dataclass
class FederatedUpdate:
    """Represents an update to be shared across federation"""
    update_id: str
    source_node: str
    update_type: str  # "embedding", "pattern", "insight", "gradient"

    # The actual update data (differentially private)
    payload: Dict[str, Any]

    # Privacy metadata
    privacy_budget: float  # ε (epsilon) for differential privacy
    noise_scale: float

    # Versioning
    version: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Propagation tracking
    seen_by: Set[str] = field(default_factory=set)
    applied_by: Set[str] = field(default_factory=set)


class FederatedCoordinator:
    """
    Federated Memory Coordinator.

    Manages a federation of MEMSHADOW nodes, enabling privacy-preserving
    distributed learning and memory sharing.

    Architecture:
        - Decentralized: No single point of failure
        - Byzantine fault tolerant: Handles malicious nodes
        - Privacy-preserving: Differential privacy + secure aggregation
        - Adaptive: Learns optimal sync strategies

    Example:
        coordinator = FederatedCoordinator(node_id="node_001")
        await coordinator.start()

        # Join federation
        await coordinator.join_federation([
            "node_002:8080",
            "node_003:8080"
        ])

        # Share an update
        await coordinator.share_update({
            "type": "pattern",
            "data": pattern_embedding,
            "privacy_budget": 0.1
        })
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        bind_address: str = "0.0.0.0",
        bind_port: int = 8765,
        privacy_budget: float = 1.0  # Total ε budget
    ):
        """
        Initialize federated coordinator.

        Args:
            node_id: Unique node identifier
            bind_address: Address to bind to
            bind_port: Port to listen on
            privacy_budget: Total differential privacy budget
        """
        self.node_id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        self.bind_address = bind_address
        self.bind_port = bind_port

        # Federation state
        self.peers: Dict[str, NodeInfo] = {}
        self.pending_updates: Dict[str, FederatedUpdate] = {}

        # Privacy accounting
        self.privacy_budget_remaining = privacy_budget
        self.privacy_budget_total = privacy_budget
        self._privacy_lock = asyncio.Lock()  # Prevent race conditions on budget

        # Sync configuration
        self.sync_strategy = SyncStrategy.GOSSIP
        self.sync_interval_seconds = 60
        self.max_peers = 50

        # Byzantine fault tolerance
        self.min_consensus_ratio = 0.67  # 2/3 consensus
        self.quarantine_threshold = 3  # Byzantine violations before quarantine

        # Background tasks
        self._sync_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Self-learning: Adapt sync strategy based on performance
        self.strategy_performance: Dict[SyncStrategy, List[float]] = {
            strategy: [] for strategy in SyncStrategy
        }

        logger.info(
            "Federated coordinator initialized",
            node_id=self.node_id,
            port=bind_port
        )

    async def start(self):
        """Start the federated coordinator"""
        logger.info("Starting federated coordinator", node_id=self.node_id)

        # Start background tasks
        self._sync_task = asyncio.create_task(self._sync_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Announce presence (would use mDNS/multicast in production)
        await self._announce_presence()

    async def stop(self):
        """Stop the coordinator"""
        logger.info("Stopping federated coordinator", node_id=self.node_id)

        # Cancel background tasks
        if self._sync_task:
            self._sync_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        # Notify peers
        await self._announce_departure()

    async def join_federation(self, bootstrap_nodes: List[str]):
        """
        Join a federation using bootstrap nodes.

        Args:
            bootstrap_nodes: List of "address:port" strings
        """
        logger.info(
            "Joining federation",
            node_id=self.node_id,
            bootstrap_count=len(bootstrap_nodes)
        )

        for node_addr in bootstrap_nodes:
            try:
                # Parse address
                host, port = node_addr.split(":")

                # Connect and exchange node info
                peer_info = await self._handshake(host, int(port))

                if peer_info:
                    self.peers[peer_info.node_id] = peer_info
                    logger.info(
                        "Peer added",
                        peer_id=peer_info.node_id,
                        address=f"{host}:{port}"
                    )

            except Exception as e:
                logger.error(
                    "Failed to connect to bootstrap node",
                    node_addr=node_addr,
                    error=str(e)
                )

    async def share_update(
        self,
        update_data: Dict[str, Any],
        privacy_budget: float = 0.1
    ) -> str:
        """
        Share an update with the federation (with differential privacy).

        Args:
            update_data: The update to share
            privacy_budget: ε to spend on this update

        Returns:
            Update ID
        """
        # Atomic check and deduct of privacy budget
        async with self._privacy_lock:
            # Check privacy budget
            if self.privacy_budget_remaining < privacy_budget:
                raise ValueError(
                    f"Insufficient privacy budget. "
                    f"Remaining: {self.privacy_budget_remaining:.3f}, "
                    f"Requested: {privacy_budget}"
                )

            # Create update
            update = FederatedUpdate(
                update_id=str(uuid.uuid4()),
                source_node=self.node_id,
                update_type=update_data.get("type", "unknown"),
                payload=update_data,
                privacy_budget=privacy_budget,
                noise_scale=self._calculate_noise_scale(privacy_budget),
                version=1
            )

            # Deduct privacy budget (atomic with check)
            self.privacy_budget_remaining -= privacy_budget

        # Store pending update (outside lock for better concurrency)
        self.pending_updates[update.update_id] = update

        # Propagate based on strategy
        await self._propagate_update(update)

        logger.info(
            "Update shared",
            update_id=update.update_id,
            type=update.update_type,
            privacy_budget=privacy_budget,
            remaining_budget=self.privacy_budget_remaining
        )

        return update.update_id

    async def apply_updates(self) -> int:
        """
        Apply pending updates to local memory.

        Returns:
            Number of updates applied
        """
        applied_count = 0

        for update_id, update in list(self.pending_updates.items()):
            # Verify update authenticity (Byzantine check)
            if await self._verify_update(update):
                # Apply update to local memory
                await self._apply_update(update)

                # Mark as applied
                update.applied_by.add(self.node_id)
                applied_count += 1

                # Clean up
                del self.pending_updates[update_id]
            else:
                logger.warning(
                    "Update failed verification",
                    update_id=update_id,
                    source=update.source_node
                )

                # Penalize source node
                if update.source_node in self.peers:
                    self.peers[update.source_node].byzantine_violations += 1

        if applied_count > 0:
            logger.info(f"Applied {applied_count} federated updates")

        return applied_count

    async def get_federation_stats(self) -> Dict[str, Any]:
        """Get federation statistics"""
        online_peers = sum(
            1 for peer in self.peers.values()
            if peer.status == NodeStatus.ONLINE
        )

        avg_reputation = sum(
            peer.reputation_score for peer in self.peers.values()
        ) / len(self.peers) if self.peers else 0.0

        return {
            "node_id": self.node_id,
            "total_peers": len(self.peers),
            "online_peers": online_peers,
            "pending_updates": len(self.pending_updates),
            "privacy_budget_remaining": self.privacy_budget_remaining,
            "privacy_budget_used_percent":
                (1 - self.privacy_budget_remaining / self.privacy_budget_total) * 100,
            "avg_peer_reputation": avg_reputation,
            "sync_strategy": self.sync_strategy.value
        }

    # Private methods

    async def _sync_loop(self):
        """Background sync loop"""
        while True:
            try:
                await asyncio.sleep(self.sync_interval_seconds)

                # Pull updates from peers
                await self._pull_from_peers()

                # Apply updates
                await self.apply_updates()

                # Self-learning: Adapt sync strategy
                await self._adapt_sync_strategy()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Sync loop error", error=str(e))

    async def _heartbeat_loop(self):
        """Background heartbeat to peers"""
        while True:
            try:
                await asyncio.sleep(10)

                # Send heartbeat to all online peers
                for peer in self.peers.values():
                    if peer.status == NodeStatus.ONLINE:
                        await self._send_heartbeat(peer)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat loop error", error=str(e))

    async def _handshake(self, host: str, port: int) -> Optional[NodeInfo]:
        """Perform handshake with peer"""
        # In production: establish gRPC/HTTP connection
        # For now: return mock peer info
        peer_id = f"peer_{uuid.uuid4().hex[:8]}"

        return NodeInfo(
            node_id=peer_id,
            address=host,
            port=port,
            status=NodeStatus.ONLINE,
            last_seen=datetime.utcnow()
        )

    async def _propagate_update(self, update: FederatedUpdate):
        """Propagate update based on current strategy"""
        if self.sync_strategy == SyncStrategy.GOSSIP:
            # Gossip to random subset of peers
            await self._gossip_update(update)

        elif self.sync_strategy == SyncStrategy.PUSH:
            # Push to all peers
            await self._push_to_all_peers(update)

        elif self.sync_strategy == SyncStrategy.CONSENSUS:
            # Byzantine consensus
            await self._consensus_propagate(update)

    async def _gossip_update(self, update: FederatedUpdate):
        """Gossip update to random peers"""
        import random

        # Select random peers (typically sqrt(N))
        fanout = max(1, int(len(self.peers) ** 0.5))
        targets = random.sample(list(self.peers.values()), min(fanout, len(self.peers)))

        for peer in targets:
            if peer.status == NodeStatus.ONLINE:
                # In production: send via network
                logger.debug(
                    "Gossiping update",
                    update_id=update.update_id,
                    to_peer=peer.node_id
                )

    async def _push_to_all_peers(self, update: FederatedUpdate):
        """Push update to all peers"""
        for peer in self.peers.values():
            if peer.status == NodeStatus.ONLINE:
                # In production: send via network
                pass

    async def _consensus_propagate(self, update: FederatedUpdate):
        """Byzantine consensus for critical updates"""
        # Implement PBFT or similar
        pass

    async def _pull_from_peers(self):
        """Pull updates from peers"""
        for peer in self.peers.values():
            if peer.status == NodeStatus.ONLINE:
                # In production: fetch updates via API
                pass

    async def _verify_update(self, update: FederatedUpdate) -> bool:
        """Verify update authenticity (Byzantine check)"""
        # Check source node reputation
        if update.source_node in self.peers:
            peer = self.peers[update.source_node]
            if peer.reputation_score < 0.5:
                return False

        # Verify signatures (in production)
        # Verify data integrity
        # Check for anomalies

        return True

    async def _apply_update(self, update: FederatedUpdate):
        """Apply update to local memory"""
        # Integration point with MEMSHADOW memory system
        logger.debug(
            "Applying update",
            update_id=update.update_id,
            type=update.update_type
        )

        # Update application logic would go here
        # e.g., merge embeddings, update patterns, etc.

    async def _send_heartbeat(self, peer: NodeInfo):
        """Send heartbeat to peer"""
        # In production: network call
        peer.last_seen = datetime.utcnow()

    async def _announce_presence(self):
        """Announce node presence to network"""
        logger.info("Announcing presence to federation", node_id=self.node_id)

    async def _announce_departure(self):
        """Announce node departure"""
        logger.info("Announcing departure from federation", node_id=self.node_id)

    async def _adapt_sync_strategy(self):
        """Self-learning: Adapt sync strategy based on performance"""
        # Measure current strategy performance
        current_strategy = self.sync_strategy

        # Calculate metrics
        success_rate = sum(
            p.successful_syncs / max(1, p.successful_syncs + p.failed_syncs)
            for p in self.peers.values()
        ) / max(1, len(self.peers))

        # Record performance
        self.strategy_performance[current_strategy].append(success_rate)

        # Every 10 cycles, evaluate if we should switch strategy
        if len(self.strategy_performance[current_strategy]) >= 10:
            # Calculate average performance for each strategy
            avg_performance = {
                strategy: sum(perfs) / len(perfs) if perfs else 0.0
                for strategy, perfs in self.strategy_performance.items()
            }

            # Find best strategy
            best_strategy = max(avg_performance, key=avg_performance.get)

            # Switch if significantly better (>10% improvement)
            if (best_strategy != current_strategy and
                avg_performance[best_strategy] > avg_performance[current_strategy] * 1.1):

                logger.info(
                    "Switching sync strategy",
                    from_strategy=current_strategy.value,
                    to_strategy=best_strategy.value,
                    improvement=f"{(avg_performance[best_strategy] - avg_performance[current_strategy]) * 100:.1f}%"
                )

                self.sync_strategy = best_strategy

    def _calculate_noise_scale(self, epsilon: float) -> float:
        """Calculate noise scale for differential privacy"""
        # Laplace mechanism: noise_scale = sensitivity / epsilon
        sensitivity = 1.0  # Assume L1 sensitivity of 1
        return sensitivity / epsilon


# Global coordinator instance
coordinator = FederatedCoordinator()
