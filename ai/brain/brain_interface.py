#!/usr/bin/env python3
"""
DSMIL Second Brain Interface

Main entry point for the distributed intelligence system.
Integrates all components:
- Security layer (CNSA 2.0)
- Memory fabric (Working/Episodic/Semantic)
- Federation (Hub/Spoke)
- Self-improvement and cross-correlation

Central Hub Query Model:
- All NL queries originate from hub
- Nodes receive, correlate, return
- Hub aggregates and synthesizes
"""

import asyncio
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Set
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path

logger = logging.getLogger(__name__)


class BrainMode(Enum):
    """Operating mode of the brain"""
    HUB = auto()          # Central hub mode
    SPOKE = auto()        # Spoke node mode
    STANDALONE = auto()   # Standalone (offline) mode


class BrainState(Enum):
    """State of the brain"""
    INITIALIZING = auto()
    READY = auto()
    PROCESSING = auto()
    SYNCING = auto()
    DEGRADED = auto()
    COMPROMISED = auto()
    SHUTTING_DOWN = auto()


@dataclass
class BrainConfig:
    """Configuration for the brain"""
    node_id: str
    mode: BrainMode

    # Network
    hub_endpoint: Optional[str] = None  # For spoke mode
    listen_port: int = 8900

    # Security
    enable_cnsa: bool = True
    enable_tamper_detection: bool = True

    # Memory
    working_memory_mb: Optional[int] = None  # Auto-detect if None
    enable_consolidation: bool = True
    consolidation_interval: int = 300  # seconds

    # Federation
    enable_federation: bool = True
    heartbeat_interval: float = 5.0

    # Storage
    data_directory: Path = field(default_factory=lambda: Path.home() / ".dsmil" / "brain")

    # Performance
    max_concurrent_queries: int = 10
    query_timeout: float = 30.0


@dataclass
class QueryResult:
    """Result of a brain query"""
    query_id: str
    success: bool

    # Results
    answer: Optional[str] = None
    data: Optional[Dict] = None

    # Sources
    sources: List[Dict] = field(default_factory=list)
    node_responses: List[Dict] = field(default_factory=list)

    # Confidence
    confidence: float = 0.0
    consensus_reached: bool = False

    # Timing
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Errors
    errors: List[str] = field(default_factory=list)


class DSMILBrain:
    """
    DSMIL Second Brain - Distributed Intelligence System

    The central intelligence system that:
    - Manages distributed memory across nodes
    - Performs constant cross-correlation and analysis
    - Coordinates queries across the network
    - Self-improves through continuous learning

    Usage (Hub Mode):
        brain = DSMILBrain(BrainConfig(
            node_id="dsmil-central",
            mode=BrainMode.HUB
        ))
        await brain.initialize()

        # Query the network
        result = await brain.query("What threats target our infrastructure?")

        # Propagate intelligence
        brain.propagate_intel({"type": "ioc", "value": "192.168.1.100"})

    Usage (Spoke Mode):
        brain = DSMILBrain(BrainConfig(
            node_id="node-001",
            mode=BrainMode.SPOKE,
            hub_endpoint="hub.local:8900"
        ))
        await brain.initialize()
        # Brain now receives queries from hub
    """

    _instance: Optional["DSMILBrain"] = None
    _lock = threading.Lock()

    def __init__(self, config: BrainConfig):
        """
        Initialize DSMIL Brain

        Args:
            config: Brain configuration
        """
        self.config = config
        self.node_id = config.node_id
        self.mode = config.mode
        self.state = BrainState.INITIALIZING

        # Components (lazy-initialized)
        self._crypto = None
        self._key_store = None
        self._tamper_detector = None
        self._authenticator = None

        self._working_memory = None
        self._episodic_memory = None
        self._semantic_memory = None
        self._consolidator = None

        self._hub_orchestrator = None
        self._spoke_client = None
        self._offline_coordinator = None
        self._intel_propagator = None
        self._sync_protocol = None

        # Event handlers
        self.on_state_change: Optional[Callable[[BrainState], None]] = None
        self.on_intel_received: Optional[Callable[[Dict], None]] = None
        self.on_query_received: Optional[Callable[[Dict], None]] = None

        # Background tasks
        self._running = False
        self._background_threads: List[threading.Thread] = []

        # Statistics
        self.stats = {
            "queries_processed": 0,
            "intel_received": 0,
            "correlations_found": 0,
            "uptime_start": None,
        }

        logger.info(f"DSMILBrain created: {config.node_id} ({config.mode.name})")

    @classmethod
    def get_instance(cls) -> "DSMILBrain":
        """Get singleton instance"""
        with cls._lock:
            if cls._instance is None:
                # Create with default config
                config = BrainConfig(
                    node_id="dsmil-default",
                    mode=BrainMode.STANDALONE,
                )
                cls._instance = cls(config)
            return cls._instance

    @classmethod
    def create_instance(cls, config: BrainConfig) -> "DSMILBrain":
        """Create singleton instance with config"""
        with cls._lock:
            cls._instance = cls(config)
            return cls._instance

    async def initialize(self) -> bool:
        """
        Initialize all brain components

        Returns:
            True if initialization successful
        """
        logger.info(f"Initializing brain in {self.mode.name} mode...")

        try:
            # Initialize security
            await self._init_security()

            # Initialize memory
            await self._init_memory()

            # Initialize federation
            if self.config.enable_federation:
                await self._init_federation()

            # Start background tasks
            self._start_background_tasks()

            self.state = BrainState.READY
            self.stats["uptime_start"] = datetime.now(timezone.utc)

            logger.info("Brain initialization complete")
            return True

        except Exception as e:
            logger.error(f"Brain initialization failed: {e}")
            self.state = BrainState.DEGRADED
            return False

    async def _init_security(self):
        """Initialize security components"""
        from .security import CNSACrypto, SecureKeyStore, TamperDetector, ContinuousAuthenticator

        if self.config.enable_cnsa:
            self._crypto = CNSACrypto()
            self._key_store = SecureKeyStore(
                storage_path=self.config.data_directory / "keystore"
            )
            logger.info("CNSA 2.0 crypto initialized")

        if self.config.enable_tamper_detection:
            self._tamper_detector = TamperDetector()
            self._tamper_detector.on_critical = self._handle_critical_tamper
            self._tamper_detector.start_monitoring()
            logger.info("Tamper detection started")

        self._authenticator = ContinuousAuthenticator(
            self.node_id,
            is_hub=(self.mode == BrainMode.HUB),
            crypto=self._crypto,
        )
        logger.info("Authentication initialized")

    async def _init_memory(self):
        """Initialize memory components"""
        from .memory import WorkingMemory, EpisodicMemory, SemanticMemory, MemoryConsolidator

        # Working memory with auto-sizing
        self._working_memory = WorkingMemory(
            max_size_bytes=self.config.working_memory_mb * 1024**2 if self.config.working_memory_mb else None
        )

        # Episodic memory
        self._episodic_memory = EpisodicMemory()

        # Semantic memory
        self._semantic_memory = SemanticMemory()

        # Memory consolidator
        if self.config.enable_consolidation:
            self._consolidator = MemoryConsolidator(
                self._working_memory,
                self._episodic_memory,
                self._semantic_memory,
            )
            self._consolidator.start_background_consolidation(
                interval=self.config.consolidation_interval
            )

        logger.info("Memory fabric initialized")

    async def _init_federation(self):
        """Initialize federation components"""
        from .federation import (
            HubOrchestrator, SpokeClient, OfflineCoordinator,
            IntelPropagator, SyncProtocol, NodeCapability
        )

        if self.mode == BrainMode.HUB:
            self._hub_orchestrator = HubOrchestrator(self.node_id, brain_interface=self)
            self._hub_orchestrator.on_intel_received = self._handle_intel_from_node
            self._hub_orchestrator.start_health_monitoring()

            self._intel_propagator = IntelPropagator(self.node_id)
            logger.info("Hub orchestrator initialized")

        elif self.mode == BrainMode.SPOKE:
            self._spoke_client = SpokeClient(
                self.node_id,
                self.config.hub_endpoint,
                capabilities={"search", "correlate", "analyze"},
                data_domains=set(),  # Will be populated
            )

            # Connect to hub
            await self._spoke_client.connect()
            logger.info("Spoke client connected to hub")

        # Offline coordinator (for all modes)
        self._offline_coordinator = OfflineCoordinator(self.node_id)

        # Sync protocol
        self._sync_protocol = SyncProtocol(
            self.node_id,
            is_hub=(self.mode == BrainMode.HUB),
        )

        logger.info("Federation initialized")

    def _handle_critical_tamper(self, evidence):
        """Handle critical tamper detection"""
        logger.critical(f"CRITICAL TAMPER DETECTED: {evidence.tamper_type.name}")
        self.state = BrainState.COMPROMISED

        # In production, would trigger self-destruct protocol
        # For now, just log and alert
        if self.on_state_change:
            self.on_state_change(BrainState.COMPROMISED)

    def _handle_intel_from_node(self, intel: Dict):
        """Handle intel received from a node"""
        self.stats["intel_received"] += 1

        # Store in memory
        self._working_memory.store(
            f"intel:{intel.get('type', 'unknown')}",
            intel,
            metadata={"source": "node", "priority": "high"}
        )

        if self.on_intel_received:
            self.on_intel_received(intel)

    async def query(self, natural_language: str,
                   timeout: Optional[float] = None,
                   require_consensus: bool = False) -> QueryResult:
        """
        Query the distributed brain

        This is the main entry point for intelligence queries.
        In HUB mode: Distributes to nodes and aggregates
        In SPOKE mode: Only processes locally (queries should come from hub)
        In STANDALONE mode: Processes locally only

        Args:
            natural_language: Natural language query
            timeout: Query timeout
            require_consensus: Require consensus from nodes

        Returns:
            QueryResult with answer and metadata
        """
        import hashlib
        import time

        query_id = hashlib.sha256(
            f"{self.node_id}:{natural_language}:{time.time()}".encode()
        ).hexdigest()[:16]

        start_time = time.time()
        self.state = BrainState.PROCESSING

        timeout = timeout or self.config.query_timeout

        try:
            if self.mode == BrainMode.HUB:
                # Distribute to network
                result = await self._distributed_query(
                    query_id, natural_language, timeout, require_consensus
                )
            else:
                # Local query only
                result = await self._local_query(query_id, natural_language)

            result.processing_time_ms = (time.time() - start_time) * 1000
            self.stats["queries_processed"] += 1

            # Record in episodic memory
            from .memory.episodic_memory import EventType
            self._episodic_memory.record_event(
                EventType.QUERY,
                natural_language,
                importance=0.6,
                metadata={"query_id": query_id, "success": result.success}
            )

            self.state = BrainState.READY
            return result

        except Exception as e:
            logger.error(f"Query failed: {e}")
            self.state = BrainState.READY
            return QueryResult(
                query_id=query_id,
                success=False,
                errors=[str(e)],
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    async def _distributed_query(self, query_id: str, natural_language: str,
                                 timeout: float, require_consensus: bool) -> QueryResult:
        """Execute distributed query across network"""
        from .federation.hub_orchestrator import QueryPriority

        # Query hub orchestrator
        response = await self._hub_orchestrator.query(
            natural_language,
            priority=QueryPriority.NORMAL,
            timeout=timeout,
            require_consensus=require_consensus,
        )

        # Also query local memory
        local_result = await self._local_query(query_id, natural_language)

        # Combine results
        return QueryResult(
            query_id=query_id,
            success=response.nodes_responded > 0 or local_result.success,
            answer=self._synthesize_answer(response, local_result),
            data={
                "network_response": response.synthesized_result,
                "local_response": local_result.data,
            },
            node_responses=[r.result for r in response.individual_responses],
            confidence=max(response.consensus_confidence, local_result.confidence),
            consensus_reached=response.consensus_reached,
        )

    async def _local_query(self, query_id: str, natural_language: str) -> QueryResult:
        """Execute query against local memory only"""
        # Search working memory
        working_results = self._working_memory.search_by_context(
            {"query": natural_language},
            top_k=10
        )

        # Search semantic memory
        semantic_results = []
        keywords = natural_language.split()
        for keyword in keywords[:5]:  # Top 5 keywords
            result = self._semantic_memory.query(keyword)
            if result:
                semantic_results.append(result)

        # Calculate confidence
        confidence = min(0.9, 0.3 + len(working_results) * 0.05 + len(semantic_results) * 0.1)

        return QueryResult(
            query_id=query_id,
            success=len(working_results) > 0 or len(semantic_results) > 0,
            data={
                "working_memory": [item.content for item in working_results],
                "semantic_memory": semantic_results,
            },
            confidence=confidence,
            sources=[{"type": "local", "count": len(working_results) + len(semantic_results)}],
        )

    def _synthesize_answer(self, network_response, local_result) -> str:
        """Synthesize final answer from multiple sources"""
        # Simple synthesis - would use LLM in production
        parts = []

        if network_response.synthesized_result:
            parts.append(f"Network: {network_response.synthesized_result}")

        if local_result.data:
            working = local_result.data.get("working_memory", [])
            if working:
                parts.append(f"Local memory: {len(working)} items found")

        return " | ".join(parts) if parts else "No results found"

    def propagate_intel(self, intel: Dict, priority: str = "normal"):
        """
        Propagate intelligence to the network

        Args:
            intel: Intelligence data to propagate
            priority: Priority level
        """
        if self.mode != BrainMode.HUB:
            logger.warning("Intel propagation only available in HUB mode")
            return

        from .federation.intel_propagator import IntelType, PropagationPriority

        # Determine intel type
        intel_type = IntelType.THREAT_INDICATOR
        if intel.get("type") == "pattern":
            intel_type = IntelType.PATTERN
        elif intel.get("type") == "correlation":
            intel_type = IntelType.CORRELATION

        # Determine priority
        priority_map = {
            "background": PropagationPriority.BACKGROUND,
            "normal": PropagationPriority.NORMAL,
            "high": PropagationPriority.HIGH,
            "critical": PropagationPriority.CRITICAL,
            "emergency": PropagationPriority.EMERGENCY,
        }
        prop_priority = priority_map.get(priority.lower(), PropagationPriority.NORMAL)

        # Create and propagate
        report = self._intel_propagator.create_report(
            intel_type=intel_type,
            priority=prop_priority,
            content=intel,
            summary=intel.get("summary", ""),
        )

        self._intel_propagator.queue_for_propagation(report)

        # Also store locally
        self._working_memory.store(
            f"propagated:{report.report_id}",
            intel,
            metadata={"propagated": True}
        )

    def add_knowledge(self, subject: str, predicate: str, obj: str,
                     confidence: float = 0.5):
        """
        Add knowledge to the semantic memory

        Args:
            subject: Subject of the fact
            predicate: Relationship type
            obj: Object of the fact
            confidence: Confidence level
        """
        self._semantic_memory.add_fact(subject, predicate, obj, confidence=confidence)

        # Record change for sync
        self._sync_protocol.record_change(
            "add",
            f"semantic/facts/{subject}:{predicate}:{obj}",
            {"subject": subject, "predicate": predicate, "object": obj, "confidence": confidence}
        )

    def get_knowledge(self, concept: str) -> Optional[Dict]:
        """Get knowledge about a concept"""
        return self._semantic_memory.query(concept)

    def _start_background_tasks(self):
        """Start background tasks"""
        self._running = True

        # Self-improvement task
        def self_improvement_loop():
            import time
            while self._running:
                try:
                    self._run_self_improvement()
                except Exception as e:
                    logger.error(f"Self-improvement error: {e}")
                time.sleep(60)  # Every minute

        thread = threading.Thread(target=self_improvement_loop, daemon=True)
        thread.start()
        self._background_threads.append(thread)

    def _run_self_improvement(self):
        """Run self-improvement cycle"""
        # Cross-correlate working memory items
        items = list(self._working_memory._items.values())

        if len(items) < 2:
            return

        # Look for patterns
        for i, item1 in enumerate(items[:-1]):
            for item2 in items[i+1:]:
                # Simple correlation: shared metadata
                shared = set(item1.metadata.keys()) & set(item2.metadata.keys())
                if shared:
                    for key in shared:
                        if item1.metadata[key] == item2.metadata[key]:
                            # Found correlation
                            self.stats["correlations_found"] += 1

                            # Add relationship to semantic memory
                            self._semantic_memory.add_fact(
                                str(item1.content)[:50],
                                "CORRELATES_WITH",
                                str(item2.content)[:50],
                                confidence=0.5,
                            )

    async def shutdown(self):
        """Gracefully shutdown the brain"""
        logger.info("Shutting down brain...")
        self.state = BrainState.SHUTTING_DOWN
        self._running = False

        # Stop background tasks
        for thread in self._background_threads:
            thread.join(timeout=5.0)

        # Stop components
        if self._consolidator:
            self._consolidator.stop_background_consolidation()

        if self._tamper_detector:
            self._tamper_detector.stop_monitoring()

        if self._hub_orchestrator:
            self._hub_orchestrator.stop_health_monitoring()

        if self._spoke_client:
            await self._spoke_client.disconnect()

        logger.info("Brain shutdown complete")

    def get_status(self) -> Dict:
        """Get comprehensive brain status"""
        status = {
            "node_id": self.node_id,
            "mode": self.mode.name,
            "state": self.state.name,
            "uptime": None,
            "stats": self.stats,
        }

        if self.stats["uptime_start"]:
            uptime = (datetime.now(timezone.utc) - self.stats["uptime_start"]).total_seconds()
            status["uptime"] = f"{uptime:.0f}s"

        # Memory status
        if self._working_memory:
            status["memory"] = {
                "working": self._working_memory.get_stats(),
                "episodic": self._episodic_memory.get_stats() if self._episodic_memory else None,
                "semantic": self._semantic_memory.get_stats() if self._semantic_memory else None,
            }

        # Federation status
        if self._hub_orchestrator:
            status["federation"] = {
                "role": "hub",
                "stats": self._hub_orchestrator.get_hub_stats(),
            }
        elif self._spoke_client:
            status["federation"] = {
                "role": "spoke",
                "status": self._spoke_client.get_status(),
            }

        return status


# Factory functions for easy instantiation
def create_hub_brain(node_id: str = "dsmil-central",
                     data_dir: Optional[Path] = None) -> DSMILBrain:
    """Create a hub brain"""
    config = BrainConfig(
        node_id=node_id,
        mode=BrainMode.HUB,
        data_directory=data_dir or Path.home() / ".dsmil" / "brain",
    )
    return DSMILBrain.create_instance(config)


def create_spoke_brain(node_id: str, hub_endpoint: str,
                       data_dir: Optional[Path] = None) -> DSMILBrain:
    """Create a spoke brain"""
    config = BrainConfig(
        node_id=node_id,
        mode=BrainMode.SPOKE,
        hub_endpoint=hub_endpoint,
        data_directory=data_dir or Path.home() / ".dsmil" / f"brain-{node_id}",
    )
    return DSMILBrain(config)


def create_standalone_brain(node_id: str = "dsmil-local",
                           data_dir: Optional[Path] = None) -> DSMILBrain:
    """Create a standalone brain"""
    config = BrainConfig(
        node_id=node_id,
        mode=BrainMode.STANDALONE,
        enable_federation=False,
        data_directory=data_dir or Path.home() / ".dsmil" / "brain",
    )
    return DSMILBrain(config)


if __name__ == "__main__":
    print("DSMIL Second Brain Self-Test")
    print("=" * 60)

    import asyncio

    async def test_brain():
        # Create standalone brain for testing
        brain = create_standalone_brain("test-brain")

        print(f"\n[1] Initialize Brain")
        success = await brain.initialize()
        print(f"    Initialization: {'Success' if success else 'Failed'}")

        print(f"\n[2] Add Knowledge")
        brain.add_knowledge("APT29", "IS_A", "Threat Actor", confidence=0.95)
        brain.add_knowledge("APT29", "USES", "Cobalt Strike", confidence=0.85)
        brain.add_knowledge("Cobalt Strike", "IS_A", "Malware", confidence=0.99)
        print(f"    Added 3 facts")

        print(f"\n[3] Query Knowledge")
        knowledge = brain.get_knowledge("APT29")
        if knowledge:
            print(f"    Found: {knowledge['concept']['name']}")
            print(f"    Relationships: {len(knowledge['relationships'])}")

        print(f"\n[4] Natural Language Query")
        result = await brain.query("What threats use Cobalt Strike?")
        print(f"    Success: {result.success}")
        print(f"    Confidence: {result.confidence:.2f}")
        print(f"    Processing time: {result.processing_time_ms:.1f}ms")

        print(f"\n[5] Brain Status")
        status = brain.get_status()
        print(f"    Mode: {status['mode']}")
        print(f"    State: {status['state']}")
        print(f"    Queries: {status['stats']['queries_processed']}")
        print(f"    Correlations: {status['stats']['correlations_found']}")

        if status.get('memory'):
            mem = status['memory']
            print(f"    Working memory items: {mem['working']['item_count']}")
            if mem['semantic']:
                print(f"    Semantic concepts: {mem['semantic']['concept_count']}")

        print(f"\n[6] Shutdown")
        await brain.shutdown()
        print(f"    Shutdown complete")

    asyncio.run(test_brain())

    print("\n" + "=" * 60)
    print("DSMIL Second Brain test complete")

