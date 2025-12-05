import json

import pytest

from ai.brain.plugins.ingest.memshadow_ingest import (
    BrainMemoryFacade,
    MemshadowIntelEdgeProcessor,
    PsychIntelRecord,
    ThreatIntelRecord,
    MemoryIntelRecord,
    FederationIntelRecord,
    ImprovementIntelRecord,
)
from ai.brain.metrics.memshadow_metrics import get_memshadow_metrics_registry
from config.memshadow_config import get_memshadow_config

try:
    from dsmil_protocol import (
        MemshadowHeader,
        MessageType,
        Priority,
        PsychEvent,
        create_psych_message,
    )
    from ai.brain.memory.memory_sync_protocol import (
        MemorySyncBatch,
        MemorySyncItem,
        MemoryTier,
        SyncOperation,
        SyncPriority,
    )
    PROTOCOL_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROTOCOL_AVAILABLE = False


class FakeWorkingMemory:
    def __init__(self):
        self.items = []

    def store(self, *args, **kwargs):
        self.items.append((args, kwargs))


class FakeEpisodicMemory:
    def __init__(self):
        self.events = []

    def record_event(self, *args, **kwargs):
        self.events.append((args, kwargs))


class FakeSemanticMemory:
    def __init__(self):
        self.facts = []

    def add_fact(self, *args, **kwargs):
        self.facts.append((args, kwargs))

    def store(self, *args, **kwargs):  # pragma: no cover - fallback path
        self.facts.append((args, kwargs))


class FakeBrain:
    def __init__(self):
        self.working_memory = FakeWorkingMemory()
        self.episodic_memory = FakeEpisodicMemory()
        self.semantic_memory = FakeSemanticMemory()


class FakeMemorySyncManager:
    def __init__(self):
        self.batches = []

    def apply_sync_batch(self, batch):
        self.batches.append(batch)
        return len(batch.items), 0


class FakeHubGateway:
    def __init__(self):
        self.calls = []

    async def handle_memshadow_message(self, header, payload, source_node):
        self.calls.append((header.msg_type, source_node))
        return {"status": "ok"}


class FakeImprovementTracker:
    def __init__(self):
        self.records = []

    def record_improvement(self, improvement):
        self.records.append(improvement)


@pytest.fixture
def processor():
    if not PROTOCOL_AVAILABLE:
        pytest.skip("MEMSHADOW protocol not available")
    brain = FakeBrain()
    gateway = FakeHubGateway()
    mem_manager = FakeMemorySyncManager()
    improvement = FakeImprovementTracker()
    metrics = get_memshadow_metrics_registry()
    metrics.reset()
    return MemshadowIntelEdgeProcessor(
        config=get_memshadow_config(),
        metrics=metrics,
        brain_memory_facade=BrainMemoryFacade(brain),
        hub_gateway=gateway,
        memory_sync_manager=mem_manager,
        improvement_tracker=improvement,
    ), brain, gateway, mem_manager, improvement


def _build_threat_message(indicator: str) -> bytes:
    body = {
        "indicator": indicator,
        "severity": "high",
        "confidence": 0.8,
    }
    payload = json.dumps(body).encode()
    header = MemshadowHeader(
        msg_type=MessageType.THREAT_REPORT,
        priority=Priority.HIGH,
        payload_len=len(payload),
    )
    return header.pack() + payload


def _build_memory_message() -> bytes:
    item = MemorySyncItem(
        item_id="abc123",
        timestamp_ns=1,
        tier=MemoryTier.WORKING,
        operation=SyncOperation.INSERT,
        priority=SyncPriority.NORMAL,
        payload=b"{}",
    )
    batch = MemorySyncBatch(
        batch_id="batch-1",
        source_node="node-a",
        target_node="node-b",
        tier=MemoryTier.WORKING,
        items=[item],
    )
    return batch.pack()


def _build_federation_message() -> bytes:
    payload = json.dumps({"node": "spoke-1"}).encode()
    header = MemshadowHeader(
        msg_type=MessageType.NODE_REGISTER,
        priority=Priority.NORMAL,
        payload_len=len(payload),
    )
    return header.pack() + payload


def _build_improvement_message() -> bytes:
    payload = json.dumps({"improvement_id": "imp-1", "gain": 12.5}).encode()
    header = MemshadowHeader(
        msg_type=MessageType.IMPROVEMENT_ANNOUNCE,
        priority=Priority.HIGH,
        payload_len=len(payload),
    )
    return header.pack() + payload


@pytest.mark.skipif(not PROTOCOL_AVAILABLE, reason="MEMSHADOW protocol not available")
def test_processor_handles_psych(processor):
    proc, brain, *_ = processor
    event = PsychEvent(session_id=99)
    message = create_psych_message([event])
    records = proc.ingest_bytes(message.pack(), source="test", source_type="unit")
    assert any(isinstance(r, PsychIntelRecord) for r in records)
    assert brain.working_memory.items


@pytest.mark.skipif(not PROTOCOL_AVAILABLE, reason="MEMSHADOW protocol not available")
def test_processor_handles_threat(processor):
    proc, brain, *_ = processor
    records = proc.ingest_bytes(_build_threat_message("indicator-1"), source="tgcollector", source_type="json")
    assert any(isinstance(r, ThreatIntelRecord) for r in records)
    assert brain.semantic_memory.facts


@pytest.mark.skipif(not PROTOCOL_AVAILABLE, reason="MEMSHADOW protocol not available")
def test_processor_handles_memory_batches(processor):
    proc, _brain, _gateway, mem_manager, _improvement = processor
    records = proc.ingest_bytes(_build_memory_message(), source="node-a", source_type="memsync")
    assert any(isinstance(r, MemoryIntelRecord) for r in records)
    assert mem_manager.batches


@pytest.mark.skipif(not PROTOCOL_AVAILABLE, reason="MEMSHADOW protocol not available")
def test_processor_handles_federation_messages(processor):
    proc, _brain, gateway, *_ = processor
    records = proc.ingest_bytes(_build_federation_message(), source="spoke-1", source_type="federation")
    assert any(isinstance(r, FederationIntelRecord) for r in records)
    assert gateway.calls


@pytest.mark.skipif(not PROTOCOL_AVAILABLE, reason="MEMSHADOW protocol not available")
def test_processor_handles_improvement_messages(processor):
    proc, _brain, _gateway, _mem_manager, improvement = processor
    records = proc.ingest_bytes(_build_improvement_message(), source="node-a", source_type="improvement")
    assert any(isinstance(r, ImprovementIntelRecord) for r in records)
    assert improvement.records
