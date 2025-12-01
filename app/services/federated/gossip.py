"""
Gossip Protocol for Federated Memory
Phase 8.1: Epidemic-style knowledge propagation

Implements gossip-based dissemination of memory updates:
- Push gossip: Send updates to random peers
- Pull gossip: Request updates from random peers
- Push-pull: Hybrid approach
- Anti-entropy: Ensure eventual consistency
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import uuid
import random
import asyncio
import structlog

logger = structlog.get_logger()


class GossipMode(Enum):
    """Gossip dissemination modes"""
    PUSH = "push"  # Push updates to peers
    PULL = "pull"  # Pull updates from peers
    PUSH_PULL = "push_pull"  # Hybrid


class MessageType(Enum):
    """Gossip message types"""
    UPDATE = "update"  # Memory update
    DIGEST = "digest"  # Summary of held updates
    REQUEST = "request"  # Request specific updates
    RESPONSE = "response"  # Response to request
    HEARTBEAT = "heartbeat"  # Liveness check


@dataclass
class GossipMessage:
    """
    Gossip protocol message.

    Messages are exchanged between federated nodes to propagate
    memory updates in an epidemic fashion.
    """
    message_id: str
    message_type: MessageType
    sender_id: str

    # Payload
    payload: Dict[str, Any]

    # Routing
    recipient_id: Optional[str] = None
    hop_count: int = 0
    max_hops: int = 10

    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: int = 300  # 5 minutes

    @property
    def is_expired(self) -> bool:
        """Check if message has expired"""
        age = datetime.utcnow() - self.timestamp
        return age.total_seconds() > self.ttl_seconds

    @property
    def can_forward(self) -> bool:
        """Check if message can be forwarded"""
        return self.hop_count < self.max_hops and not self.is_expired


@dataclass
class UpdateDigest:
    """
    Summary of updates held by a node.

    Used in anti-entropy protocol to identify missing updates.
    """
    node_id: str
    update_ids: Set[str]
    version_vector: Dict[str, int]  # node_id -> max_version
    timestamp: datetime = field(default_factory=datetime.utcnow)


class GossipProtocol:
    """
    Gossip Protocol for Federated Memory.

    Implements epidemic-style propagation of memory updates across
    a federation of nodes. Ensures eventual consistency without
    centralized coordination.

    Algorithm:
        1. Every T seconds, select random peer
        2. Send digest of local updates
        3. Peer compares and requests missing updates
        4. Exchange missing updates
        5. Repeat

    Properties:
        - Eventual consistency
        - Fault tolerance
        - Logarithmic convergence time
        - Low latency propagation

    Example:
        gossip = GossipProtocol(node_id="node_001")
        await gossip.start()

        # Send update
        await gossip.broadcast_update({
            "type": "pattern",
            "data": pattern_data
        })
    """

    def __init__(
        self,
        node_id: str,
        mode: GossipMode = GossipMode.PUSH_PULL,
        fanout: int = 3,
        gossip_interval_seconds: float = 1.0
    ):
        """
        Initialize gossip protocol.

        Args:
            node_id: This node's identifier
            mode: Gossip mode (push, pull, or push-pull)
            fanout: Number of peers to gossip to each round
            gossip_interval_seconds: Time between gossip rounds
        """
        self.node_id = node_id
        self.mode = mode
        self.fanout = fanout
        self.gossip_interval = gossip_interval_seconds

        # State
        self.peers: Dict[str, str] = {}  # node_id -> address
        self.updates: Dict[str, Any] = {}  # update_id -> update_data
        # Use bounded deque to prevent unbounded memory growth
        self.seen_messages: deque = deque(maxlen=10000)  # Keep last 10k messages

        # Version vector for causality tracking
        self.version_vector: Dict[str, int] = {node_id: 0}

        # Metrics
        self.messages_sent = 0
        self.messages_received = 0
        self.updates_propagated = 0

        # Background tasks
        self._gossip_task: Optional[asyncio.Task] = None
        self._anti_entropy_task: Optional[asyncio.Task] = None

        logger.info(
            "Gossip protocol initialized",
            node_id=node_id,
            mode=mode.value,
            fanout=fanout
        )

    async def start(self):
        """Start gossip protocol"""
        logger.info("Starting gossip protocol", node_id=self.node_id)

        # Start background tasks
        self._gossip_task = asyncio.create_task(self._gossip_loop())
        self._anti_entropy_task = asyncio.create_task(self._anti_entropy_loop())

    async def stop(self):
        """Stop gossip protocol"""
        logger.info("Stopping gossip protocol", node_id=self.node_id)

        if self._gossip_task:
            self._gossip_task.cancel()
        if self._anti_entropy_task:
            self._anti_entropy_task.cancel()

    def add_peer(self, peer_id: str, address: str):
        """Add a peer to gossip with"""
        self.peers[peer_id] = address
        logger.debug("Peer added", peer_id=peer_id, address=address)

    def remove_peer(self, peer_id: str):
        """Remove a peer"""
        if peer_id in self.peers:
            del self.peers[peer_id]
            logger.debug("Peer removed", peer_id=peer_id)

    async def broadcast_update(self, update_data: Dict[str, Any]) -> str:
        """
        Broadcast an update to the federation.

        Args:
            update_data: Update to broadcast

        Returns:
            Update ID
        """
        # Generate update ID
        update_id = str(uuid.uuid4())

        # Increment version vector
        self.version_vector[self.node_id] = \
            self.version_vector.get(self.node_id, 0) + 1

        # Store update
        self.updates[update_id] = {
            "id": update_id,
            "data": update_data,
            "source": self.node_id,
            "version": self.version_vector[self.node_id],
            "timestamp": datetime.utcnow().isoformat()
        }

        # Create gossip message
        message = GossipMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.UPDATE,
            sender_id=self.node_id,
            payload=self.updates[update_id]
        )

        # Gossip to peers
        await self._gossip_message(message)

        self.updates_propagated += 1

        logger.info(
            "Update broadcast initiated",
            update_id=update_id,
            peers_count=len(self.peers)
        )

        return update_id

    async def handle_message(self, message: GossipMessage):
        """
        Handle incoming gossip message.

        Args:
            message: Received message
        """
        # Check if already seen
        if message.message_id in self.seen_messages:
            return

        # Mark as seen
        self.seen_messages.add(message.message_id)
        self.messages_received += 1

        # Handle based on type
        if message.message_type == MessageType.UPDATE:
            await self._handle_update(message)

        elif message.message_type == MessageType.DIGEST:
            await self._handle_digest(message)

        elif message.message_type == MessageType.REQUEST:
            await self._handle_request(message)

        elif message.message_type == MessageType.RESPONSE:
            await self._handle_response(message)

        # Forward if applicable (for PUSH mode)
        if self.mode == GossipMode.PUSH and message.can_forward:
            await self._forward_message(message)

    async def get_digest(self) -> UpdateDigest:
        """
        Get digest of local updates.

        Returns:
            Digest summarizing local state
        """
        return UpdateDigest(
            node_id=self.node_id,
            update_ids=set(self.updates.keys()),
            version_vector=self.version_vector.copy()
        )

    async def get_stats(self) -> Dict[str, Any]:
        """Get gossip protocol statistics"""
        return {
            "node_id": self.node_id,
            "mode": self.mode.value,
            "peers_count": len(self.peers),
            "updates_held": len(self.updates),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "updates_propagated": self.updates_propagated,
            "version_vector": self.version_vector
        }

    # Private methods

    async def _gossip_loop(self):
        """Main gossip loop"""
        while True:
            try:
                await asyncio.sleep(self.gossip_interval)

                if not self.peers:
                    continue

                # Select random peers
                peers_to_gossip = self._select_gossip_targets()

                # Gossip based on mode
                if self.mode == GossipMode.PUSH:
                    await self._push_gossip(peers_to_gossip)

                elif self.mode == GossipMode.PULL:
                    await self._pull_gossip(peers_to_gossip)

                elif self.mode == GossipMode.PUSH_PULL:
                    await self._push_pull_gossip(peers_to_gossip)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Gossip loop error", error=str(e))

    async def _anti_entropy_loop(self):
        """
        Anti-entropy loop for eventual consistency.

        Periodically exchanges digests with random peer to detect
        and repair missing updates.
        """
        while True:
            try:
                # Run less frequently than regular gossip
                await asyncio.sleep(10)

                if not self.peers:
                    continue

                # Select random peer
                peer_id = random.choice(list(self.peers.keys()))

                # Exchange digests
                await self._anti_entropy_exchange(peer_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Anti-entropy loop error", error=str(e))

    def _select_gossip_targets(self) -> List[str]:
        """Select random peers for gossip"""
        if len(self.peers) <= self.fanout:
            return list(self.peers.keys())

        return random.sample(list(self.peers.keys()), self.fanout)

    async def _push_gossip(self, targets: List[str]):
        """Push updates to targets"""
        # Send random subset of updates to each target
        for peer_id in targets:
            # Select updates to send
            updates_to_send = random.sample(
                list(self.updates.values()),
                min(10, len(self.updates))
            )

            for update in updates_to_send:
                message = GossipMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.UPDATE,
                    sender_id=self.node_id,
                    recipient_id=peer_id,
                    payload=update
                )

                await self._send_message(peer_id, message)

    async def _pull_gossip(self, targets: List[str]):
        """Pull updates from targets"""
        for peer_id in targets:
            # Request digest
            message = GossipMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.REQUEST,
                sender_id=self.node_id,
                recipient_id=peer_id,
                payload={"request_type": "digest"}
            )

            await self._send_message(peer_id, message)

    async def _push_pull_gossip(self, targets: List[str]):
        """Hybrid push-pull gossip"""
        for peer_id in targets:
            # Send digest
            digest = await self.get_digest()

            message = GossipMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.DIGEST,
                sender_id=self.node_id,
                recipient_id=peer_id,
                payload={
                    "update_ids": list(digest.update_ids),
                    "version_vector": digest.version_vector
                }
            )

            await self._send_message(peer_id, message)

    async def _anti_entropy_exchange(self, peer_id: str):
        """Exchange digests with peer for anti-entropy"""
        local_digest = await self.get_digest()

        # Send digest
        message = GossipMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.DIGEST,
            sender_id=self.node_id,
            recipient_id=peer_id,
            payload={
                "update_ids": list(local_digest.update_ids),
                "version_vector": local_digest.version_vector,
                "anti_entropy": True
            }
        )

        await self._send_message(peer_id, message)

    async def _gossip_message(self, message: GossipMessage):
        """Gossip a message to random peers"""
        targets = self._select_gossip_targets()

        for peer_id in targets:
            await self._send_message(peer_id, message)

    async def _send_message(self, peer_id: str, message: GossipMessage):
        """Send message to specific peer"""
        if peer_id not in self.peers:
            return

        # In production: send via network (gRPC, HTTP, etc.)
        self.messages_sent += 1

        logger.debug(
            "Message sent",
            to_peer=peer_id,
            message_type=message.message_type.value
        )

    async def _forward_message(self, message: GossipMessage):
        """Forward message to random peers"""
        message.hop_count += 1

        targets = self._select_gossip_targets()

        for peer_id in targets:
            if peer_id != message.sender_id:  # Don't send back to sender
                await self._send_message(peer_id, message)

    async def _handle_update(self, message: GossipMessage):
        """Handle UPDATE message"""
        update_id = message.payload.get("id")

        if update_id and update_id not in self.updates:
            # New update - store it
            self.updates[update_id] = message.payload

            # Update version vector
            source = message.payload.get("source")
            version = message.payload.get("version", 0)

            if source:
                self.version_vector[source] = max(
                    self.version_vector.get(source, 0),
                    version
                )

            logger.debug("Update received", update_id=update_id, source=source)

    async def _handle_digest(self, message: GossipMessage):
        """Handle DIGEST message"""
        peer_update_ids = set(message.payload.get("update_ids", []))
        peer_version_vector = message.payload.get("version_vector", {})

        # Find updates we have that peer doesn't
        missing_at_peer = set(self.updates.keys()) - peer_update_ids

        # Find updates peer has that we don't
        missing_at_us = peer_update_ids - set(self.updates.keys())

        # If this is anti-entropy, request missing updates
        if message.payload.get("anti_entropy") and missing_at_us:
            request = GossipMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.REQUEST,
                sender_id=self.node_id,
                recipient_id=message.sender_id,
                payload={"update_ids": list(missing_at_us)}
            )

            await self._send_message(message.sender_id, request)

        # Send updates peer is missing
        if missing_at_peer:
            for update_id in list(missing_at_peer)[:10]:  # Limit to 10
                update = self.updates[update_id]

                response = GossipMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.UPDATE,
                    sender_id=self.node_id,
                    recipient_id=message.sender_id,
                    payload=update
                )

                await self._send_message(message.sender_id, response)

    async def _handle_request(self, message: GossipMessage):
        """Handle REQUEST message"""
        requested_ids = message.payload.get("update_ids", [])

        # Send requested updates
        for update_id in requested_ids:
            if update_id in self.updates:
                response = GossipMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.RESPONSE,
                    sender_id=self.node_id,
                    recipient_id=message.sender_id,
                    payload=self.updates[update_id]
                )

                await self._send_message(message.sender_id, response)

    async def _handle_response(self, message: GossipMessage):
        """Handle RESPONSE message"""
        # Same as UPDATE
        await self._handle_update(message)
