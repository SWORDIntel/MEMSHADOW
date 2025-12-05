#!/usr/bin/env python3
"""MEMSHADOW Intel Edge Processor and legacy ingest glue."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config.memshadow_config import MemshadowConfig, get_memshadow_config

LIBS_PATH = (
    Path(__file__).parent.parent.parent.parent.parent / "libs" / "memshadow-protocol" / "python"
)
if LIBS_PATH.exists() and str(LIBS_PATH) not in sys.path:
    sys.path.insert(0, str(LIBS_PATH))

logger = logging.getLogger(__name__)

try:
    from dsmil_protocol import (
        HEADER_SIZE,
        MEMSHADOW_VERSION,
        MemshadowHeader,
        MessageFlags,
        MessageType,
        Priority,
        PsychEvent,
        PSYCH_EVENT_SIZE,
    )

    PROTOCOL_AVAILABLE = True
except ImportError:  # pragma: no cover - defensive guard
    PROTOCOL_AVAILABLE = False
    logger.error("MEMSHADOW protocol library not available - ingest disabled")

try:  # Heavy imports kept optional to avoid circular references
    from ...memory.memory_sync_protocol import MemorySyncBatch, MemorySyncManager
except Exception:  # pragma: no cover
    MemorySyncBatch = None  # type: ignore
    MemorySyncManager = None  # type: ignore

try:
    from ...federation.memshadow_gateway import HubMemshadowGateway
except Exception:  # pragma: no cover
    HubMemshadowGateway = None  # type: ignore

try:
    from ...federation.improvement_tracker import ImprovementTracker
except Exception:  # pragma: no cover
    ImprovementTracker = None  # type: ignore

from ...metrics.memshadow_metrics import get_memshadow_metrics_registry, MemshadowMetricsRegistry
from ...memory.episodic_memory import EventType
from ...memory.working_memory import ItemPriority
from ..ingest_framework import IngestPlugin, IngestResult

PSYCH_MESSAGE_TYPES: Sequence[MessageType] = (
    MessageType.PSYCH_ASSESSMENT,
    MessageType.DARK_TRIAD_UPDATE,
    MessageType.RISK_UPDATE,
    MessageType.NEURO_UPDATE,
    MessageType.TMI_UPDATE,
    MessageType.COGNITIVE_UPDATE,
    MessageType.FULL_PSYCH,
    MessageType.PSYCH_THREAT_ALERT,
    MessageType.PSYCH_ANOMALY,
    MessageType.PSYCH_RISK_THRESHOLD,
)


@dataclass
class PsychEventRecord:
    """Normalized representation of a parsed PSYCH event."""

    message_type: str
    session_id: int
    timestamp_ns: int
    timestamp_offset_us: int
    event_type: int
    window_size: int
    context_hash: int
    priority: str
    flags: int
    batch_index: int
    scores: Dict[str, float]
    dark_triad_average: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - convenience helper
        return {
            "message_type": self.message_type,
            "session_id": self.session_id,
            "timestamp_ns": self.timestamp_ns,
            "timestamp_offset_us": self.timestamp_offset_us,
            "event_type": self.event_type,
            "window_size": self.window_size,
            "context_hash": self.context_hash,
            "priority": self.priority,
            "flags": self.flags,
            "batch_index": self.batch_index,
            "scores": self.scores,
            "dark_triad_average": self.dark_triad_average,
            "confidence": self.confidence,
        }


@dataclass
class BaseIntelRecord:
    category: str
    source: str
    source_type: str
    msg_type: str
    priority: str
    timestamp_ns: int
    raw_payload: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PsychIntelRecord(BaseIntelRecord):
    events: List[PsychEventRecord] = field(default_factory=list)


@dataclass
class ThreatIntelRecord(BaseIntelRecord):
    intel: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryIntelRecord(BaseIntelRecord):
    batch_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FederationIntelRecord(BaseIntelRecord):
    action: str = ""
    result: Optional[Any] = None


@dataclass
class ImprovementIntelRecord(BaseIntelRecord):
    improvement: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnknownIntelRecord(BaseIntelRecord):
    reason: str = ""


@dataclass
class ParsedMemshadowMessage:
    header: MemshadowHeader
    payload: bytes
    events: List[PsychEvent]
    raw: bytes


class BrainMemoryFacade:
    """Helper that writes events into the available memory tiers."""

    def __init__(self, brain_interface: Any):
        self._brain = brain_interface
        self.working = self._resolve_attr("working_memory")
        self.episodic = self._resolve_attr("episodic_memory")
        self.semantic = self._resolve_attr("semantic_memory")

    def _resolve_attr(self, name: str) -> Optional[Any]:
        if not self._brain:
            return None
        if hasattr(self._brain, name):
            return getattr(self._brain, name)
        private_name = f"_{name}"
        return getattr(self._brain, private_name, None)

    def store_psych_events(self, events: Sequence[PsychEventRecord]) -> None:
        if not self._brain:
            return
        for record in events:
            self._store_working(record)
            self._store_episodic(record)
            self._store_semantic(record)

    def store_threat_record(self, record: ThreatIntelRecord) -> None:
        if not self._brain:
            return
        try:
            if self.working and hasattr(self.working, "store"):
                self.working.store(
                    key=f"threat:{record.metadata.get('indicator', 'unknown')}",
                    content=record.intel,
                    content_type="threat_intel",
                    priority=ItemPriority.HIGH,
                    metadata={"source": record.source, "severity": record.intel.get("severity", "unknown")},
                )
            if self.episodic and hasattr(self.episodic, "record_event"):
                self.episodic.record_event(
                    EventType.ALERT,
                    content=record.intel,
                    context={"source": record.source, "priority": record.priority},
                    importance=0.7,
                    metadata={"category": "threat"},
                    tags={"threat", record.source_type},
                )
            if self.semantic and hasattr(self.semantic, "store"):
                try:
                    self.semantic.store(
                        concept=record.intel.get("indicator", "threat_intel"),
                        knowledge=record.intel,
                        domain="threat_intel",
                    )
                except Exception:
                    # Fallback to add_fact if `store` is not implemented
                    if hasattr(self.semantic, "add_fact"):
                        self.semantic.add_fact(
                            record.intel.get("actor", "unknown_actor"),
                            "TARGETS",
                            record.intel.get("target", "unknown_target"),
                            confidence=record.intel.get("confidence", 0.5),
                        )
        except Exception as exc:  # pragma: no cover
            logger.debug("Threat intel store failed: %s", exc)

    def _store_working(self, record: PsychEventRecord) -> None:
        if not self.working or not hasattr(self.working, "store"):
            return
        exposure = record.scores.get("espionage_exposure", 0.0)
        priority = (
            ItemPriority.CRITICAL if exposure >= 0.85 else ItemPriority.HIGH if exposure >= 0.65 else ItemPriority.NORMAL
        )
        metadata = {
            "source": "shrink",
            "message_type": record.message_type,
            "dark_triad_average": record.dark_triad_average,
        }
        try:
            self.working.store(
                key=f"psych_session:{record.session_id}",
                content=record.to_dict(),
                content_type="psych_event",
                priority=priority,
                metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Working memory store failed: %s", exc)

    def _store_episodic(self, record: PsychEventRecord) -> None:
        if not self.episodic or not hasattr(self.episodic, "record_event"):
            return
        importance = min(1.0, 0.4 + record.dark_triad_average * 0.5)
        context = {
            "session_id": record.session_id,
            "priority": record.priority,
            "message_type": record.message_type,
            "scores": record.scores,
        }
        try:
            self.episodic.record_event(
                EventType.OBSERVATION,
                content=record.to_dict(),
                context=context,
                importance=importance,
                metadata={"source": "shrink"},
                tags={"psych", "shrink"},
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("Episodic memory record failed: %s", exc)

    def _store_semantic(self, record: PsychEventRecord) -> None:
        if not self.semantic or not hasattr(self.semantic, "add_fact"):
            return
        try:
            subject = f"psych_session:{record.session_id}"
            risk_label = (
                "CRITICAL_RISK"
                if record.scores.get("espionage_exposure", 0.0) >= 0.85
                else "HIGH_RISK"
                if record.scores.get("espionage_exposure", 0.0) >= 0.65
                else "BASELINE"
            )
            self.semantic.add_fact(
                subject=subject,
                predicate="INDICATES",
                obj=risk_label,
                confidence=record.confidence,
                metadata={"source": "shrink"},
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("Semantic memory add_fact failed: %s", exc)


class MemshadowIntelEdgeProcessor:
    """Canonical MEMSHADOW edge intel processor."""

    def __init__(
        self,
        config: Optional[MemshadowConfig] = None,
        metrics: Optional[MemshadowMetricsRegistry] = None,
        brain_memory_facade: Optional[BrainMemoryFacade] = None,
        hub_gateway: Optional[HubMemshadowGateway] = None,
        memory_sync_manager: Optional[MemorySyncManager] = None,
        improvement_tracker: Optional[ImprovementTracker] = None,
    ):
        self.config = config or get_memshadow_config()
        self.metrics = metrics or get_memshadow_metrics_registry()
        self.brain_facade = brain_memory_facade
        self.hub_gateway = hub_gateway
        self.memory_sync_manager = memory_sync_manager
        self.improvement_tracker = improvement_tracker

    # ------------------------------------------------------------------ Public API
    def ingest_bytes(self, payload: bytes, source: str, source_type: str = "shrink") -> List[BaseIntelRecord]:
        if not PROTOCOL_AVAILABLE:
            raise ValueError("MEMSHADOW protocol library not available")

        messages = self._parse_messages(payload)
        records: List[BaseIntelRecord] = []

        for parsed in messages:
            self.metrics.increment("memshadow_batches_received")
            category = self._categorize(parsed.header.msg_type)
            if not self._category_enabled(category):
                logger.debug("Skipping %s ingest because category disabled", category)
                continue

            handler = getattr(self, f"_handle_{category}", None)
            if handler is None:
                record = self._handle_unknown(parsed, source, source_type, reason="no_handler")
                records.append(record)
                continue

            produced = handler(parsed, source, source_type)
            if isinstance(produced, list):
                records.extend(produced)
            elif produced:
                records.append(produced)

        return records

    # ------------------------------------------------------------------ Category handlers
    def _handle_psych(self, parsed: ParsedMemshadowMessage, source: str, source_type: str) -> List[PsychIntelRecord]:
        if not parsed.events:
            return [self._handle_unknown(parsed, source, source_type, reason="missing_psych_events")]

        psych_records: List[PsychEventRecord] = []
        for idx, event in enumerate(parsed.events):
            psych_records.append(self._build_psych_record(parsed.header, event, idx))

        if self.brain_facade:
            self.brain_facade.store_psych_events(psych_records)

        self.metrics.increment("memshadow_psych_messages")
        record = PsychIntelRecord(
            category="psych",
            source=source,
            source_type=source_type,
            msg_type=parsed.header.msg_type.name,
            priority=self._priority_name(parsed.header.priority),
            timestamp_ns=parsed.header.timestamp_ns,
            raw_payload=parsed.payload,
            events=psych_records,
        )
        return [record]

    def _handle_threat(self, parsed: ParsedMemshadowMessage, source: str, source_type: str) -> ThreatIntelRecord:
        intel = self._decode_json_payload(parsed.payload)
        intel.setdefault("raw", parsed.payload.hex())
        record = ThreatIntelRecord(
            category="threat",
            source=source,
            source_type=source_type,
            msg_type=parsed.header.msg_type.name,
            priority=self._priority_name(parsed.header.priority),
            timestamp_ns=parsed.header.timestamp_ns,
            raw_payload=parsed.payload,
            intel=intel,
            metadata={
                "indicator": intel.get("indicator"),
                "severity": intel.get("severity", "unknown"),
            },
        )
        if self.brain_facade:
            self.brain_facade.store_threat_record(record)
        self.metrics.increment("memshadow_threat_messages")
        return record

    def _handle_memory(self, parsed: ParsedMemshadowMessage, source: str, source_type: str) -> MemoryIntelRecord:
        summary: Dict[str, Any] = {}
        if not self.memory_sync_manager or MemorySyncBatch is None:
            logger.debug("Memory sync manager unavailable; recording metadata only")
        else:
            raw_message = parsed.raw
            try:
                batch = MemorySyncBatch.unpack(raw_message)
                applied, conflicts = self.memory_sync_manager.apply_sync_batch(batch)
                summary = {
                    "items": len(batch.items),
                    "tier": batch.tier.name if hasattr(batch.tier, "name") else batch.tier,
                    "applied": applied,
                    "conflicts": conflicts,
                }
            except Exception as exc:
                logger.warning("Failed to apply memory sync batch: %s", exc)
                summary["error"] = str(exc)
        self.metrics.increment("memshadow_memory_messages")
        return MemoryIntelRecord(
            category="memory",
            source=source,
            source_type=source_type,
            msg_type=parsed.header.msg_type.name,
            priority=self._priority_name(parsed.header.priority),
            timestamp_ns=parsed.header.timestamp_ns,
            raw_payload=parsed.payload,
            batch_metadata=summary,
        )

    def _handle_federation(self, parsed: ParsedMemshadowMessage, source: str, source_type: str) -> FederationIntelRecord:
        result = None
        if self.hub_gateway and hasattr(self.hub_gateway, "handle_memshadow_message"):
            try:
                maybe_coro = self.hub_gateway.handle_memshadow_message(
                    parsed.header,
                    parsed.payload,
                    source,
                )
                result = self._maybe_await(maybe_coro)
            except Exception as exc:
                logger.warning("Hub gateway handling failed: %s", exc)
                result = {"error": str(exc)}
        self.metrics.increment("memshadow_federation_messages")
        return FederationIntelRecord(
            category="federation",
            source=source,
            source_type=source_type,
            msg_type=parsed.header.msg_type.name,
            priority=self._priority_name(parsed.header.priority),
            timestamp_ns=parsed.header.timestamp_ns,
            raw_payload=parsed.payload,
            action=parsed.header.msg_type.name,
            result=result,
        )

    def _handle_improvement(self, parsed: ParsedMemshadowMessage, source: str, source_type: str) -> ImprovementIntelRecord:
        improvement = self._decode_json_payload(parsed.payload)
        if self.improvement_tracker and hasattr(self.improvement_tracker, "record_improvement"):
            try:
                self.improvement_tracker.record_improvement(improvement)
            except Exception as exc:
                logger.debug("Improvement tracker rejected payload: %s", exc)
        self.metrics.increment("memshadow_improvement_messages")
        return ImprovementIntelRecord(
            category="improvement",
            source=source,
            source_type=source_type,
            msg_type=parsed.header.msg_type.name,
            priority=self._priority_name(parsed.header.priority),
            timestamp_ns=parsed.header.timestamp_ns,
            raw_payload=parsed.payload,
            improvement=improvement,
        )

    def _handle_unknown(
        self,
        parsed: ParsedMemshadowMessage,
        source: str,
        source_type: str,
        reason: str = "unknown",
    ) -> UnknownIntelRecord:
        self.metrics.increment("memshadow_unknown_messages")
        if reason == "unsupported_msg_type":
            self.metrics.increment("memshadow_unknown_msg_type")
        return UnknownIntelRecord(
            category="unknown",
            source=source,
            source_type=source_type,
            msg_type=parsed.header.msg_type.name,
            priority=self._priority_name(parsed.header.priority),
            timestamp_ns=parsed.header.timestamp_ns,
            raw_payload=parsed.payload,
            reason=reason,
        )

    # ------------------------------------------------------------------ Helpers
    def _parse_messages(self, data: bytes) -> List[ParsedMemshadowMessage]:
        messages: List[ParsedMemshadowMessage] = []
        offset = 0
        try:
            while offset + HEADER_SIZE <= len(data):
                header = MemshadowHeader.unpack(data[offset : offset + HEADER_SIZE])
                total_size = HEADER_SIZE + header.payload_len
                if offset + total_size > len(data):
                    raise ValueError("Truncated MEMSHADOW payload")

                payload = data[offset + HEADER_SIZE : offset + total_size]
                events = self._parse_psych_events(payload)
                raw = data[offset : offset + total_size]
                messages.append(ParsedMemshadowMessage(header=header, payload=payload, events=events, raw=raw))
                offset += total_size
        except ValueError as exc:
            self.metrics.increment("memshadow_parse_errors")
            raise

        if offset != len(data):
            logger.debug("Trailing bytes detected after MEMSHADOW parsing (%d bytes)", len(data) - offset)
        return messages

    def _parse_psych_events(self, payload: bytes) -> List[PsychEvent]:
        events: List[PsychEvent] = []
        pos = 0
        while pos + PSYCH_EVENT_SIZE <= len(payload):
            chunk = payload[pos : pos + PSYCH_EVENT_SIZE]
            try:
                events.append(PsychEvent.unpack(chunk))
            except ValueError:
                break
            pos += PSYCH_EVENT_SIZE
        return events

    def _build_psych_record(
        self, header: MemshadowHeader, event: PsychEvent, batch_index: int
    ) -> PsychEventRecord:
        scores = {
            "acute_stress": event.acute_stress,
            "machiavellianism": event.machiavellianism,
            "narcissism": event.narcissism,
            "psychopathy": event.psychopathy,
            "burnout_probability": event.burnout_probability,
            "espionage_exposure": event.espionage_exposure,
            "confidence": event.confidence,
        }
        return PsychEventRecord(
            message_type=header.msg_type.name,
            session_id=event.session_id,
            timestamp_ns=header.timestamp_ns,
            timestamp_offset_us=event.timestamp_offset_us,
            event_type=event.event_type,
            window_size=event.window_size,
            context_hash=event.context_hash,
            priority=self._priority_name(header.priority),
            flags=int(header.flags) if isinstance(header.flags, MessageFlags) else int(header.flags or 0),
            batch_index=batch_index,
            scores=scores,
            dark_triad_average=event.dark_triad_average,
            confidence=event.confidence,
        )

    def _decode_json_payload(self, payload: bytes) -> Dict[str, Any]:
        try:
            return json.loads(payload.decode())
        except Exception:
            return {"raw_hex": payload.hex()}

    def _priority_name(self, priority: Priority) -> str:
        try:
            return priority.name  # type: ignore[attr-defined]
        except AttributeError:
            return str(priority)

    def _categorize(self, msg_type: MessageType) -> str:
        value = int(msg_type)
        if 0x0100 <= value <= 0x01FF:
            return "psych"
        if 0x0200 <= value <= 0x02FF:
            return "threat"
        if 0x0300 <= value <= 0x03FF:
            return "memory"
        if 0x0400 <= value <= 0x04FF:
            return "federation"
        if 0x0500 <= value <= 0x05FF:
            return "improvement"
        return "unknown"

    def _category_enabled(self, category: str) -> bool:
        return {
            "psych": self.config.enable_psych_ingest,
            "threat": self.config.enable_threat_ingest,
            "memory": self.config.enable_memory_ingest,
            "federation": self.config.enable_federation_ingest,
            "improvement": self.config.enable_improvement_ingest,
            "unknown": True,
        }.get(category, True)

    def _maybe_await(self, result: Any) -> Any:
        if asyncio.iscoroutine(result):
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    asyncio.create_task(result)
                    return {"status": "scheduled"}
            except RuntimeError:
                pass
            return asyncio.run(result)
        return result


class MemshadowIngestPlugin(IngestPlugin):
    """Legacy plugin wrapper so existing ingest manager integrations keep working."""

    def __init__(
        self,
        brain_interface: Any = None,
        hub_gateway: Optional[HubMemshadowGateway] = None,
        memory_sync_manager: Optional[MemorySyncManager] = None,
        improvement_tracker: Optional[ImprovementTracker] = None,
    ):
        self._config = get_memshadow_config()
        self._metrics = get_memshadow_metrics_registry()
        self._enabled = PROTOCOL_AVAILABLE and self._config.enable_shrink_ingest
        self._brain_interface = brain_interface
        self._processor = MemshadowIntelEdgeProcessor(
            config=self._config,
            metrics=self._metrics,
            brain_memory_facade=BrainMemoryFacade(brain_interface),
            hub_gateway=hub_gateway,
            memory_sync_manager=memory_sync_manager,
            improvement_tracker=improvement_tracker,
        )

    @property
    def name(self) -> str:
        return "memshadow_ingest"

    @property
    def version(self) -> str:
        return "3.0.0"

    @property
    def description(self) -> str:
        return "Canonical MEMSHADOW edge processor"

    @property
    def supported_types(self) -> List[str]:
        return ["memshadow", "binary", "intel"]

    def set_brain_interface(self, brain_interface: Any) -> None:
        self._brain_interface = brain_interface
        self._processor.brain_facade = BrainMemoryFacade(brain_interface)

    def initialize(self, config: Dict[str, Any]) -> bool:
        if not PROTOCOL_AVAILABLE:
            return False
        self._enabled = config.get("enabled", True) and self._config.enable_shrink_ingest
        if "brain_interface" in config:
            self.set_brain_interface(config["brain_interface"])
        return self._enabled

    def ingest(self, source: Any, **kwargs) -> IngestResult:
        if not self._enabled:
            return IngestResult(success=False, plugin_name=self.name, errors=["MEMSHADOW ingest disabled"])

        brain_interface = kwargs.get("brain_interface")
        if brain_interface and brain_interface is not self._brain_interface:
            self.set_brain_interface(brain_interface)

        if isinstance(source, dict):
            return self.ingest_memshadow_legacy(source, brain_interface)

        data = self._coerce_bytes(source)
        if data is None:
            return IngestResult(success=False, plugin_name=self.name, errors=[f"Unsupported source type: {type(source)}"])
        return self.ingest_memshadow_binary(data, brain_interface)

    def ingest_memshadow_binary(self, data: bytes, brain_interface: Any = None) -> IngestResult:
        try:
            processor = self._processor
            if brain_interface and brain_interface is not self._brain_interface:
                self.set_brain_interface(brain_interface)
                processor = self._processor
            records = processor.ingest_bytes(data, source="memshadow", source_type="binary")
            return IngestResult(
                success=True,
                plugin_name=self.name,
                data=[record.__dict__ for record in records],
                metadata={"protocol_version": MEMSHADOW_VERSION, "record_count": len(records)},
                items_ingested=len(records),
                bytes_processed=len(data),
            )
        except ValueError as exc:
            return IngestResult(success=False, plugin_name=self.name, errors=[str(exc)])

    def ingest_memshadow_legacy(self, payload: Dict[str, Any], brain_interface: Any = None) -> IngestResult:
        message_bytes = self._legacy_json_to_memshadow(payload)
        return self.ingest_memshadow_binary(message_bytes, brain_interface)

    def _legacy_json_to_memshadow(self, payload: Dict[str, Any]) -> bytes:
        msg_type = MessageType.THREAT_REPORT
        if payload.get("type") == "psych_event":
            msg_type = MessageType.PSYCH_ASSESSMENT
        body = json.dumps(payload).encode()
        header = MemshadowHeader(
            msg_type=msg_type,
            priority=Priority.NORMAL,
            payload_len=len(body),
        )
        return header.pack() + body

    def _coerce_bytes(self, source: Any) -> Optional[bytes]:
        if isinstance(source, bytes):
            return source
        if isinstance(source, (str, Path)):
            try:
                with open(source, "rb") as handle:
                    return handle.read()
            except Exception as exc:
                logger.error("Failed to read source %s: %s", source, exc)
                return None
        if hasattr(source, "read"):
            return source.read()
        return None


_default_plugin = MemshadowIngestPlugin()


def ingest_memshadow_binary(data: bytes, brain_interface: Any = None) -> IngestResult:
    if brain_interface:
        _default_plugin.set_brain_interface(brain_interface)
    return _default_plugin.ingest_memshadow_binary(data, brain_interface)


def ingest_memshadow_legacy(payload: Dict[str, Any], brain_interface: Any = None) -> IngestResult:
    if brain_interface:
        _default_plugin.set_brain_interface(brain_interface)
    return _default_plugin.ingest_memshadow_legacy(payload, brain_interface)
