import pytest

from ai.brain.api.shrink_endpoint import ShrinkIntelEndpoint

try:
    from dsmil_protocol import PsychEvent, create_psych_message, MessageType, Priority
    PROTOCOL_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROTOCOL_AVAILABLE = False


class FakeWorkingMemory:
    def __init__(self):
        self.items = []

    def store(self, key, content, content_type, priority, metadata=None):
        self.items.append(
            {
                "key": key,
                "content": content,
                "content_type": content_type,
                "priority": priority.name if hasattr(priority, "name") else priority,
                "metadata": metadata or {},
            }
        )


class FakeEpisodicMemory:
    def __init__(self):
        self.events = []

    def record_event(self, *args, **kwargs):
        self.events.append({"args": args, "kwargs": kwargs})


class FakeSemanticMemory:
    def __init__(self):
        self.facts = []

    def add_fact(self, subject, predicate, obj, **kwargs):
        self.facts.append((subject, predicate, obj, kwargs))


class FakeBrain:
    def __init__(self):
        self.working_memory = FakeWorkingMemory()
        self.episodic_memory = FakeEpisodicMemory()
        self.semantic_memory = FakeSemanticMemory()


@pytest.mark.skipif(not PROTOCOL_AVAILABLE, reason="MEMSHADOW protocol not available")
def test_shrink_ingest_binary_happy_path():
    fake_brain = FakeBrain()
    endpoint = ShrinkIntelEndpoint(fake_brain)

    psych_event = PsychEvent(
        session_id=123,
        timestamp_offset_us=500,
        event_type=1,
        acute_stress=0.8,
        machiavellianism=0.4,
        narcissism=0.5,
        psychopathy=0.2,
        burnout_probability=0.1,
        espionage_exposure=0.9,
        confidence=0.95,
    )
    message = create_psych_message([psych_event], priority=Priority.HIGH)
    payload = message.pack()

    body, status = endpoint.handle_post(payload, "application/octet-stream")

    assert status == 200
    assert body["success"]
    assert fake_brain.working_memory.items
    assert fake_brain.episodic_memory.events
    assert fake_brain.semantic_memory.facts
    assert body["records_ingested"].get("psych") == 1


@pytest.mark.skipif(not PROTOCOL_AVAILABLE, reason="MEMSHADOW protocol not available")
def test_shrink_ingest_invalid_magic_returns_400():
    fake_brain = FakeBrain()
    endpoint = ShrinkIntelEndpoint(fake_brain)

    body, status = endpoint.handle_post(b"invalid-bytes", "application/octet-stream")

    assert status == 400
    assert not body["success"]


@pytest.mark.skipif(not PROTOCOL_AVAILABLE, reason="MEMSHADOW protocol not available")
def test_shrink_ingest_accepts_non_psych_messages():
    fake_brain = FakeBrain()
    endpoint = ShrinkIntelEndpoint(fake_brain)

    # Build a MEMORY_SYNC message, which should now be accepted as generic intel
    event = PsychEvent(session_id=1)
    message = create_psych_message([event])
    message.header.msg_type = MessageType.MEMORY_SYNC
    data = message.pack()

    body, status = endpoint.handle_post(data, "application/octet-stream")

    assert status == 200
    assert body["success"]
    assert body["records_ingested"].get("memory", 0) >= 1
