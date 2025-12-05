"""
Spoke Client with MEMSHADOW Memory Adapter

Node client for the DSMIL Brain Federation that:
- Receives queries from hub and performs local correlation
- Operates autonomously when hub is offline
- Participates in P2P improvement propagation
- Exposes local memory tiers via SpokeMemoryAdapter

Based on: HUB_DOCS/DSMIL Brain Federation.md
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

import structlog

# Import canonical protocol
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "libs" / "memshadow-protocol" / "python"))

from dsmil_protocol import (
    MemshadowHeader,
    MemshadowMessage,
    MessageType,
    Priority,
    MessageFlags,
    MemoryTier,
    should_route_p2p,
    HEADER_SIZE,
)

# Import memory sync protocol
sys.path.insert(0, str(Path(__file__).parent.parent))
from memory.memory_sync_protocol import (
    MemorySyncItem,
    MemorySyncBatch,
    MemorySyncManager,
    SyncPriority,
    SyncConfig,
    SyncOperation,
)

logger = structlog.get_logger()


# ============================================================================
# Query Types
# ============================================================================

class QuerySource(Enum):
    """Source of incoming query"""
    HUB = "hub"
    PEER = "peer"
    LOCAL = "local"


@dataclass
class QueryResponse:
    """Response to a query"""
    query_id: str
    results: List[Dict[str, Any]]
    confidence: float
    processing_time_ms: float
    source: QuerySource


# ============================================================================
# Local Memory Tier Interface
# ============================================================================

class LocalTierInterface:
    """Interface for local memory tier implementations"""
    
    async def store(self, item_id: str, data: bytes, metadata: Dict[str, Any]) -> bool:
        raise NotImplementedError
    
    async def retrieve(self, item_id: str) -> Optional[bytes]:
        raise NotImplementedError
    
    async def delete(self, item_id: str) -> bool:
        raise NotImplementedError
    
    async def list_items(self, since_timestamp: Optional[int] = None) -> List[str]:
        raise NotImplementedError
    
    async def get_metadata(self, item_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class InMemoryTier(LocalTierInterface):
    """Simple in-memory tier implementation for testing"""
    
    def __init__(self, tier: MemoryTier):
        self.tier = tier
        self._store: Dict[str, bytes] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    async def store(self, item_id: str, data: bytes, metadata: Dict[str, Any]) -> bool:
        self._store[item_id] = data
        self._metadata[item_id] = {**metadata, "stored_at": int(time.time() * 1e9)}
        return True
    
    async def retrieve(self, item_id: str) -> Optional[bytes]:
        return self._store.get(item_id)
    
    async def delete(self, item_id: str) -> bool:
        self._store.pop(item_id, None)
        self._metadata.pop(item_id, None)
        return True
    
    async def list_items(self, since_timestamp: Optional[int] = None) -> List[str]:
        if since_timestamp is None:
            return list(self._store.keys())
        return [
            k for k, m in self._metadata.items()
            if m.get("stored_at", 0) >= since_timestamp
        ]
    
    async def get_metadata(self, item_id: str) -> Optional[Dict[str, Any]]:
        return self._metadata.get(item_id)


# ============================================================================
# Spoke Memory Adapter
# ============================================================================

class SpokeMemoryAdapter:
    """
    Spoke-side adapter for local memory tiers.
    
    Responsibilities:
    - Wrap access to local L1/L2/L3 tiers
    - Create delta batches for sync to hub/peers
    - Apply incoming sync batches with conflict resolution
    - Expose P2P sync entrypoints
    - Propagate self-improvement updates via MEMSHADOW
    """
    
    def __init__(
        self,
        node_id: str,
        hub_node_id: str = "hub",
        mesh_send_callback: Optional[Callable] = None,
        config: Optional[SyncConfig] = None,
    ):
        self.node_id = node_id
        self.hub_node_id = hub_node_id
        self.mesh_send = mesh_send_callback
        self.config = config or SyncConfig()
        
        # Local memory tiers
        self._tiers: Dict[MemoryTier, LocalTierInterface] = {}
        
        # Sync manager for delta computation
        self._sync_manager = MemorySyncManager(
            node_id=node_id,
            config=config,
            mesh_send_callback=mesh_send_callback,
        )
        
        # P2P peers
        self._peers: Set[str] = set()
        
        # Pending batches for retry
        self._pending_batches: Dict[str, MemorySyncBatch] = {}
        
        # Metrics
        self._metrics = {
            "items_stored": 0,
            "items_retrieved": 0,
            "batches_sent": 0,
            "batches_received": 0,
            "p2p_syncs": 0,
            "hub_syncs": 0,
            "improvements_propagated": 0,
        }
        
        logger.info("SpokeMemoryAdapter initialized", node_id=node_id)
    
    def register_tier(self, tier: MemoryTier, instance: LocalTierInterface):
        """Register a local memory tier implementation"""
        self._tiers[tier] = instance
        self._sync_manager.register_memory_tier(tier, instance)
        logger.info("Tier registered", tier=tier.name, node=self.node_id)
    
    def add_peer(self, peer_id: str):
        """Add a P2P peer for direct sync"""
        self._peers.add(peer_id)
        logger.debug("Peer added", peer=peer_id)
    
    def remove_peer(self, peer_id: str):
        """Remove a P2P peer"""
        self._peers.discard(peer_id)
    
    # ========================================================================
    # Local Storage Operations
    # ========================================================================
    
    async def store(
        self,
        tier: MemoryTier,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        sync_priority: SyncPriority = SyncPriority.NORMAL,
    ) -> str:
        """
        Store data in a local tier and optionally sync.
        
        Returns:
            Item ID
        """
        if tier not in self._tiers:
            raise ValueError(f"Tier {tier.name} not registered")
        
        item_id = str(uuid4())
        metadata = metadata or {}
        
        # Store locally
        tier_impl = self._tiers[tier]
        await tier_impl.store(item_id, data, metadata)
        
        # Create sync item for tracking
        sync_item = MemorySyncItem.create(
            payload=data,
            tier=tier,
            operation=SyncOperation.INSERT,
            priority=sync_priority,
            source_node=self.node_id,
        )
        sync_item.item_id = uuid4()
        sync_item.metadata = metadata
        
        # Store in sync manager
        await self._sync_manager.store_local(sync_item)
        
        self._metrics["items_stored"] += 1
        
        logger.debug(
            "Item stored",
            item_id=item_id,
            tier=tier.name,
            size=len(data),
        )
        
        return item_id
    
    async def retrieve(self, tier: MemoryTier, item_id: str) -> Optional[bytes]:
        """Retrieve data from a local tier"""
        if tier not in self._tiers:
            return None
        
        data = await self._tiers[tier].retrieve(item_id)
        if data:
            self._metrics["items_retrieved"] += 1
        return data
    
    async def delete(
        self,
        tier: MemoryTier,
        item_id: str,
        sync_priority: SyncPriority = SyncPriority.NORMAL,
    ) -> bool:
        """Delete data from a local tier and propagate deletion"""
        if tier not in self._tiers:
            return False
        
        result = await self._tiers[tier].delete(item_id)
        
        if result:
            # Create delete sync item
            sync_item = MemorySyncItem(
                item_id=uuid4(),
                timestamp_ns=int(time.time() * 1e9),
                tier=tier,
                operation=SyncOperation.DELETE,
                priority=sync_priority,
                payload=item_id.encode(),
                source_node=self.node_id,
            )
            await self._sync_manager.store_local(sync_item)
        
        return result
    
    # ========================================================================
    # Delta Batch Creation
    # ========================================================================
    
    async def create_delta_batch(
        self,
        tier: MemoryTier,
        target_node: str = "*",
        priority: SyncPriority = SyncPriority.NORMAL,
    ) -> MemorySyncBatch:
        """
        Create a delta batch for sync to target node(s).
        
        Args:
            tier: Memory tier to sync
            target_node: Target node or "*" for broadcast
            priority: Sync priority
        
        Returns:
            MemorySyncBatch with delta items
        """
        return await self._sync_manager.create_delta_batch(
            peer_id=target_node if target_node != "*" else self.hub_node_id,
            tier=tier,
            priority=priority,
        )
    
    # ========================================================================
    # Batch Application
    # ========================================================================
    
    async def apply_sync_batch(
        self,
        batch: MemorySyncBatch,
    ) -> Dict[str, Any]:
        """
        Apply an incoming sync batch.
        
        Uses MemorySyncManager for conflict resolution.
        """
        # Apply through sync manager
        result = await self._sync_manager.apply_sync_batch(batch)
        
        # Also apply to local tier storage
        if result["success"]:
            tier = batch.tier
            if tier in self._tiers:
                tier_impl = self._tiers[tier]
                for item in batch.items:
                    if item.operation == SyncOperation.DELETE:
                        await tier_impl.delete(str(item.item_id))
                    else:
                        payload = item.decompress_payload() if item.is_compressed else item.payload
                        await tier_impl.store(str(item.item_id), payload, item.metadata)
        
        self._metrics["batches_received"] += 1
        
        return result
    
    # ========================================================================
    # Sync Operations
    # ========================================================================
    
    async def sync_to_hub(
        self,
        tier: MemoryTier,
        priority: SyncPriority = SyncPriority.NORMAL,
    ) -> bool:
        """
        Sync a tier to the hub.
        
        Creates delta batch and sends via hub relay.
        """
        batch = await self.create_delta_batch(tier, self.hub_node_id, priority)
        
        if not batch.items:
            logger.debug("No items to sync to hub", tier=tier.name)
            return True
        
        return await self._send_batch(batch, priority.to_memshadow_priority())
    
    async def sync_to_peer(
        self,
        peer_id: str,
        tier: MemoryTier,
        priority: SyncPriority = SyncPriority.NORMAL,
    ) -> bool:
        """
        Sync a tier directly to a P2P peer.
        
        Used for CRITICAL/EMERGENCY priority syncs.
        """
        if peer_id not in self._peers:
            logger.warning("Unknown peer", peer_id=peer_id)
            return False
        
        batch = await self.create_delta_batch(tier, peer_id, priority)
        
        if not batch.items:
            return True
        
        return await self._send_batch(batch, priority.to_memshadow_priority())
    
    async def _send_batch(
        self,
        batch: MemorySyncBatch,
        priority: Priority,
    ) -> bool:
        """
        Send a batch using appropriate routing.
        
        Priority routing rules:
        - CRITICAL/EMERGENCY: Direct P2P + hub notification
        - HIGH: Hub-relayed with priority
        - NORMAL/LOW: Standard hub routing
        """
        if not self.mesh_send:
            logger.warning("No mesh callback configured")
            return False
        
        msg = batch.to_memshadow_message()
        packed = msg.pack()
        
        try:
            if should_route_p2p(priority):
                # P2P: Send to all peers directly, plus hub notification
                self._metrics["p2p_syncs"] += 1
                
                for peer_id in self._peers:
                    if peer_id != batch.source_node:
                        try:
                            await self.mesh_send(peer_id, packed)
                        except Exception as e:
                            logger.warning("P2P send failed", peer=peer_id, error=str(e))
                
                # Also notify hub
                await self.mesh_send(self.hub_node_id, packed)
            else:
                # Hub-relayed
                self._metrics["hub_syncs"] += 1
                await self.mesh_send(self.hub_node_id, packed)
            
            self._metrics["batches_sent"] += 1
            
            logger.debug(
                "Batch sent",
                batch_id=batch.batch_id,
                items=len(batch.items),
                priority=priority.name,
                p2p=should_route_p2p(priority),
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to send batch", error=str(e))
            # Queue for retry
            self._pending_batches[batch.batch_id] = batch
            return False
    
    # ========================================================================
    # Improvement Propagation
    # ========================================================================
    
    async def propagate_improvement(
        self,
        improvement_type: str,
        improvement_data: bytes,
        gain_percent: float,
        affected_tiers: List[MemoryTier],
    ) -> bool:
        """
        Propagate a self-improvement update via MEMSHADOW.
        
        Creates sync items for affected tiers and routes based on gain.
        """
        # Determine priority based on gain
        if gain_percent > 20:
            priority = SyncPriority.URGENT  # Maps to EMERGENCY
        elif gain_percent > 10:
            priority = SyncPriority.HIGH
        else:
            priority = SyncPriority.LOW
        
        # Create sync items for each affected tier
        for tier in affected_tiers:
            if tier not in self._tiers:
                continue
            
            sync_item = MemorySyncItem.create(
                payload=improvement_data,
                tier=tier,
                operation=SyncOperation.MERGE,
                priority=priority,
                source_node=self.node_id,
            )
            sync_item.metadata = {
                "improvement_type": improvement_type,
                "gain_percent": gain_percent,
            }
            
            await self._sync_manager.store_local(sync_item)
        
        # Also send improvement announcement
        if self.mesh_send:
            from dsmil_protocol import create_improvement_announce
            
            msg = create_improvement_announce(
                improvement_id=str(uuid4()),
                improvement_type=improvement_type,
                gain_percent=gain_percent,
                priority=priority.to_memshadow_priority(),
            )
            
            await self.mesh_send(self.hub_node_id, msg.pack())
            self._metrics["improvements_propagated"] += 1
        
        logger.info(
            "Improvement propagated",
            type=improvement_type,
            gain_percent=gain_percent,
            priority=priority.name,
        )
        
        return True
    
    # ========================================================================
    # Message Handling
    # ========================================================================
    
    async def handle_incoming_message(
        self,
        data: bytes,
        source_node: str,
    ) -> Dict[str, Any]:
        """
        Handle incoming MEMSHADOW message.
        
        Routes to appropriate handler based on message type.
        """
        try:
            header = MemshadowHeader.unpack(data[:HEADER_SIZE])
            
            if not header.validate():
                return {"error": "Invalid header"}
            
            if header.msg_type == MessageType.MEMORY_SYNC:
                msg = MemshadowMessage.unpack(data)
                batch = MemorySyncBatch.from_memshadow_message(msg)
                return await self.apply_sync_batch(batch)
            
            elif header.msg_type == MessageType.IMPROVEMENT_PAYLOAD:
                # Apply improvement payload
                msg = MemshadowMessage.unpack(data)
                return await self._handle_improvement_payload(msg, source_node)
            
            else:
                logger.debug("Unhandled message type", msg_type=header.msg_type.name)
                return {"status": "ignored", "msg_type": header.msg_type.name}
                
        except Exception as e:
            logger.error("Failed to handle message", error=str(e))
            return {"error": str(e)}
    
    async def _handle_improvement_payload(
        self,
        msg: MemshadowMessage,
        source_node: str,
    ) -> Dict[str, Any]:
        """Handle an improvement payload message"""
        try:
            payload = json.loads(msg.payload.decode())
            improvement_type = payload.get("type", "unknown")
            
            logger.info(
                "Improvement payload received",
                type=improvement_type,
                source=source_node,
            )
            
            # ACK the improvement
            if self.mesh_send:
                ack_msg = MemshadowMessage.create(
                    msg_type=MessageType.IMPROVEMENT_ACK,
                    payload=json.dumps({
                        "improvement_id": payload.get("improvement_id"),
                        "status": "applied",
                    }).encode(),
                    priority=Priority.HIGH,
                )
                await self.mesh_send(source_node, ack_msg.pack())
            
            return {"status": "applied", "type": improvement_type}
            
        except Exception as e:
            return {"error": str(e)}
    
    # ========================================================================
    # Stats
    # ========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics"""
        return {
            "node_id": self.node_id,
            "registered_tiers": [t.name for t in self._tiers.keys()],
            "peers": list(self._peers),
            "pending_batches": len(self._pending_batches),
            "sync_manager_stats": self._sync_manager.get_stats(),
            **self._metrics,
        }


# ============================================================================
# Spoke Client
# ============================================================================

class SpokeClient:
    """
    Node client for the DSMIL Brain Federation.
    
    Integrates SpokeMemoryAdapter for memory tier access and sync.
    """
    
    def __init__(
        self,
        node_id: str,
        hub_endpoint: str,
        capabilities: Optional[Set[str]] = None,
        data_domains: Optional[Set[str]] = None,
        mesh_port: int = 8889,
        use_mesh: bool = True,
        sync_config: Optional[SyncConfig] = None,
    ):
        self.node_id = node_id
        self.hub_endpoint = hub_endpoint
        self.capabilities = capabilities or {"search", "correlate"}
        self.data_domains = data_domains or set()
        self.mesh_port = mesh_port
        self.use_mesh = use_mesh
        
        # Connection state
        self._connected = False
        self._hub_node_id = "hub"  # Will be updated on connect
        
        # Mesh send callback
        self._mesh_send: Optional[Callable] = None
        
        # Memory adapter
        self._memory_adapter = SpokeMemoryAdapter(
            node_id=node_id,
            hub_node_id=self._hub_node_id,
            mesh_send_callback=self._mesh_send_wrapper,
            config=sync_config,
        )
        
        # Message handlers
        self._handlers: Dict[MessageType, Callable] = {}
        self._setup_handlers()
        
        # Stats
        self._stats = {
            "queries_processed": 0,
            "improvements_announced": 0,
            "improvements_received": 0,
        }
        
        logger.info("SpokeClient initialized", node_id=node_id, hub=hub_endpoint)
    
    def _setup_handlers(self):
        """Register message handlers"""
        self._handlers[MessageType.QUERY_DISTRIBUTE] = self._handle_query
        self._handlers[MessageType.MEMORY_SYNC] = self._handle_memory_sync
        self._handlers[MessageType.IMPROVEMENT_ANNOUNCE] = self._handle_improvement_announce
        self._handlers[MessageType.IMPROVEMENT_PAYLOAD] = self._handle_improvement_payload
        self._handlers[MessageType.INTEL_PROPAGATE] = self._handle_intel
    
    async def _mesh_send_wrapper(self, peer_id: str, data: bytes):
        """Wrapper for mesh send"""
        if self._mesh_send:
            await self._mesh_send(peer_id, data)
    
    def set_mesh_callback(self, callback: Callable):
        """Set the mesh send callback"""
        self._mesh_send = callback
        self._memory_adapter.mesh_send = self._mesh_send_wrapper
    
    @property
    def memory_adapter(self) -> SpokeMemoryAdapter:
        """Access the memory adapter"""
        return self._memory_adapter
    
    # ========================================================================
    # Connection
    # ========================================================================
    
    async def connect(self) -> bool:
        """
        Connect to the hub.
        
        Sends registration message with capabilities and memory tier info.
        """
        try:
            # Create registration message
            reg_data = {
                "node_id": self.node_id,
                "endpoint": f"localhost:{self.mesh_port}",
                "capabilities": list(self.capabilities),
                "data_domains": list(self.data_domains),
                "memory_capabilities": {
                    "supports_l1": True,
                    "supports_l2": True,
                    "supports_l3": True,
                    "max_batch_size": 100,
                    "compression_supported": True,
                },
            }
            
            msg = MemshadowMessage.create(
                msg_type=MessageType.NODE_REGISTER,
                payload=json.dumps(reg_data).encode(),
                priority=Priority.HIGH,
            )
            
            if self._mesh_send:
                await self._mesh_send(self._hub_node_id, msg.pack())
            
            self._connected = True
            
            logger.info("Connected to hub", hub=self.hub_endpoint)
            return True
            
        except Exception as e:
            logger.error("Failed to connect to hub", error=str(e))
            return False
    
    async def disconnect(self):
        """Disconnect from hub"""
        if self._mesh_send:
            msg = MemshadowMessage.create(
                msg_type=MessageType.NODE_DEREGISTER,
                payload=json.dumps({"node_id": self.node_id}).encode(),
            )
            await self._mesh_send(self._hub_node_id, msg.pack())
        
        self._connected = False
        logger.info("Disconnected from hub")
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    # ========================================================================
    # Query Processing
    # ========================================================================
    
    async def process_query(
        self,
        query: Dict[str, Any],
        source: QuerySource,
    ) -> QueryResponse:
        """
        Process an incoming query.
        
        Performs local correlation and analysis.
        """
        start_time = time.time()
        query_id = query.get("query_id", str(uuid4()))
        query_text = query.get("query_text", "")
        
        # Placeholder: real implementation would perform actual analysis
        results = [{
            "match": "example",
            "score": 0.85,
            "source": self.node_id,
        }]
        
        self._stats["queries_processed"] += 1
        
        return QueryResponse(
            query_id=query_id,
            results=results,
            confidence=0.85,
            processing_time_ms=(time.time() - start_time) * 1000,
            source=source,
        )
    
    # ========================================================================
    # Improvement Propagation
    # ========================================================================
    
    async def announce_improvement(
        self,
        improvement_type: str,
        improvement_data: bytes,
        gain_percent: float,
    ):
        """
        Broadcast a local improvement to the network.
        
        Routes based on gain_percent:
        - >20%: CRITICAL (P2P + hub)
        - 10-20%: HIGH (hub relay)
        - <10%: LOW (background)
        """
        # Use memory adapter for propagation
        await self._memory_adapter.propagate_improvement(
            improvement_type=improvement_type,
            improvement_data=improvement_data,
            gain_percent=gain_percent,
            affected_tiers=[MemoryTier.WORKING, MemoryTier.EPISODIC],
        )
        
        self._stats["improvements_announced"] += 1
    
    # ========================================================================
    # Message Handlers
    # ========================================================================
    
    async def _handle_query(self, data: bytes, peer_id: str):
        """Handle query from hub"""
        try:
            msg = MemshadowMessage.unpack(data)
            query = json.loads(msg.payload.decode())
            
            response = await self.process_query(query, QuerySource.HUB)
            
            # Send response
            if self._mesh_send:
                resp_msg = MemshadowMessage.create(
                    msg_type=MessageType.QUERY_RESPONSE,
                    payload=json.dumps({
                        "query_id": response.query_id,
                        "results": response.results,
                        "confidence": response.confidence,
                        "node_id": self.node_id,
                    }).encode(),
                    priority=Priority.NORMAL,
                )
                await self._mesh_send(peer_id, resp_msg.pack())
                
        except Exception as e:
            logger.error("Failed to handle query", error=str(e))
    
    async def _handle_memory_sync(self, data: bytes, peer_id: str):
        """Handle memory sync message"""
        await self._memory_adapter.handle_incoming_message(data, peer_id)
    
    async def _handle_improvement_announce(self, data: bytes, peer_id: str):
        """Handle improvement announcement"""
        try:
            msg = MemshadowMessage.unpack(data)
            announcement = json.loads(msg.payload.decode())
            
            improvement_id = announcement.get("improvement_id")
            improvement_type = announcement.get("type")
            gain = announcement.get("gain_percent", 0)
            
            # Decide if we want this improvement
            if gain >= 5:  # Threshold for requesting
                # Request the improvement
                if self._mesh_send:
                    req_msg = MemshadowMessage.create(
                        msg_type=MessageType.IMPROVEMENT_REQUEST,
                        payload=json.dumps({
                            "improvement_id": improvement_id,
                            "requester": self.node_id,
                        }).encode(),
                        priority=Priority.HIGH,
                    )
                    await self._mesh_send(peer_id, req_msg.pack())
            
            logger.info(
                "Improvement announced",
                type=improvement_type,
                gain_percent=gain,
                source=peer_id,
            )
            
        except Exception as e:
            logger.error("Failed to handle improvement announce", error=str(e))
    
    async def _handle_improvement_payload(self, data: bytes, peer_id: str):
        """Handle improvement payload"""
        result = await self._memory_adapter.handle_incoming_message(data, peer_id)
        if result.get("status") == "applied":
            self._stats["improvements_received"] += 1
    
    async def _handle_intel(self, data: bytes, peer_id: str):
        """Handle intel propagation"""
        try:
            msg = MemshadowMessage.unpack(data)
            intel = json.loads(msg.payload.decode())
            
            logger.info("Intel received", type=intel.get("type"), source=peer_id)
            
        except Exception as e:
            logger.error("Failed to handle intel", error=str(e))
    
    # ========================================================================
    # Dispatch
    # ========================================================================
    
    async def dispatch_message(self, data: bytes, peer_id: str) -> bool:
        """Dispatch incoming message to appropriate handler"""
        try:
            header = MemshadowHeader.unpack(data[:HEADER_SIZE])
            
            if not header.validate():
                logger.warning("Invalid header", peer=peer_id)
                return False
            
            handler = self._handlers.get(header.msg_type)
            if handler:
                await handler(data, peer_id)
                return True
            else:
                logger.debug("No handler", msg_type=header.msg_type.name)
                return False
                
        except Exception as e:
            logger.error("Dispatch error", error=str(e))
            return False
    
    # ========================================================================
    # Stats
    # ========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            "node_id": self.node_id,
            "connected": self._connected,
            "hub_endpoint": self.hub_endpoint,
            "capabilities": list(self.capabilities),
            "memory_adapter": self._memory_adapter.get_stats(),
            **self._stats,
        }


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "QuerySource",
    "QueryResponse",
    "LocalTierInterface",
    "InMemoryTier",
    "SpokeMemoryAdapter",
    "SpokeClient",
]
