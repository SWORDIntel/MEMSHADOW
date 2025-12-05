#!/usr/bin/env python3
"""MEMSHADOW Protocol ingest plugin for SHRINK and general intel streams."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config.memshadow_config import get_memshadow_config

# Add libs path for protocol imports
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
except ImportError:
    PROTOCOL_AVAILABLE = False
    logger.error("MEMSHADOW protocol library not available - ingest disabled")

from ...metrics.memshadow_metrics import get_memshadow_metrics_registry
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

ALL_MESSAGE_TYPES: Sequence[MessageType] = tuple(MessageType) if PROTOCOL_AVAILABLE else tuple()


def _resolve_message_types(names: Sequence[str]) -> List[MessageType]:
    resolved: List[MessageType] = []
    for name in names:
        try:
            resolved.append(MessageType[name])
        except KeyError:
            logger.warning("Unknown MEMSHADOW message type '%s' in config", name)
    return resolved


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

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - convenience
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
        # Allow private attribute naming (_working_memory, etc.)
        private_name = f"_{name}"
        return getattr(self._brain, private_name, None)

    def store_psych_events(self, events: Sequence[PsychEventRecord]) -> None:
        if not self._brain:
            return
        for record in events:
            self._store_working(record)
            self._store_episodic(record)
            self._store_semantic(record)

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


class MemshadowIngestPlugin(IngestPlugin):
    """Binary MEMSHADOW ingest plugin dedicated to SHRINK and intel payloads."""

    def __init__(self, brain_interface: Any = None):
        self._config = get_memshadow_config()
        self._enabled = PROTOCOL_AVAILABLE and self._config.enable_shrink_ingest
        self._brain_interface = brain_interface
        self._metrics = get_memshadow_metrics_registry()
        self._allowed_message_types = set(ALL_MESSAGE_TYPES) if ALL_MESSAGE_TYPES else set()

    @property
    def name(self) -> str:
        return "memshadow_ingest"

    @property
    def version(self) -> str:
        return "2.0.0"

    @property
    def description(self) -> str:
        return "Parse and store MEMSHADOW events from SHRINK and intel sources"

    @property
    def supported_types(self) -> List[str]:
        return ["memshadow", "binary", "psych_event"]

    def set_brain_interface(self, brain_interface: Any) -> None:
        self._brain_interface = brain_interface

    def initialize(self, config: Dict[str, Any]) -> bool:
        if not PROTOCOL_AVAILABLE:
            return False
        self._enabled = config.get("enabled", True) and self._config.enable_shrink_ingest
        self._brain_interface = config.get("brain_interface", self._brain_interface)
        allowed = config.get("allowed_message_types")
        if allowed:
            resolved = _resolve_message_types(allowed)
            if resolved:
                self._allowed_message_types = set(resolved)
        return self._enabled

    # ------------------------------------------------------------------ Ingest
    def ingest(self, source: Any, **kwargs) -> IngestResult:
        if not self._enabled:
            return IngestResult(
                success=False,
                plugin_name=self.name,
                errors=["MEMSHADOW ingest disabled"],
            )

        brain_interface = kwargs.get("brain_interface") or self._brain_interface

        if isinstance(source, dict):
            return self.ingest_memshadow_legacy(source, brain_interface)

        data = self._coerce_bytes(source)
        if data is None:
            return IngestResult(
                success=False,
                plugin_name=self.name,
                errors=[f"Unsupported source type: {type(source)}"],
            )
        return self.ingest_memshadow_binary(data, brain_interface)

    def ingest_memshadow_binary(
        self, data: bytes, brain_interface: Any = None
    ) -> IngestResult:
        if not PROTOCOL_AVAILABLE:
            return IngestResult(
                success=False,
                plugin_name=self.name,
                errors=["Protocol library unavailable"],
            )
        try:
            messages = self._parse_messages(data)
            facade = BrainMemoryFacade(brain_interface or self._brain_interface)
            structured_records: List[Dict[str, Any]] = []
            total_psych = 0

            for header, events, payload in messages:
                extracted, psych_count = self._extract_records(header, events, payload, facade)
                structured_records.extend(extracted)
                total_psych += psych_count

            if total_psych:
                self._metrics.increment("memshadow_psych_events_ingested", total_psych)
            self._metrics.increment("memshadow_batches_received", len(messages))

            return IngestResult(
                success=True,
                plugin_name=self.name,
                data=structured_records,
                metadata={
                    "messages_parsed": len(messages),
                    "events_ingested": total_psych,
                    "protocol_version": MEMSHADOW_VERSION,
                },
                items_ingested=len(structured_records),
                bytes_processed=len(data),
            )
        except ValueError as exc:
            return IngestResult(
                success=False,
                plugin_name=self.name,
                errors=[str(exc)],
            )

    def ingest_memshadow_legacy(
        self, payload: Dict[str, Any], brain_interface: Any = None
    ) -> IngestResult:
        try:
            record = PsychEventRecord(
                message_type=payload.get("message_type", "PSYCH_ASSESSMENT"),
                session_id=payload.get("session_id", 0),
                timestamp_ns=payload.get("timestamp_ns", 0),
                timestamp_offset_us=payload.get("timestamp_offset_us", 0),
                event_type=payload.get("event_type", 0),
                window_size=payload.get("window_size", 0),
                context_hash=payload.get("context_hash", 0),
                priority=payload.get("priority", "NORMAL"),
                flags=payload.get("flags", 0),
                batch_index=payload.get("batch_index", 0),
                scores=payload.get("scores", {}),
                dark_triad_average=payload.get("dark_triad_average", 0.0),
                confidence=payload.get("confidence", 0.5),
            )
            facade = BrainMemoryFacade(brain_interface or self._brain_interface)
            facade.store_psych_events([record])
            self._metrics.increment("memshadow_psych_events_ingested", 1)
            return IngestResult(
                success=True,
                plugin_name=self.name,
                data=[record.to_dict()],
                metadata={"format": "legacy_json"},
                items_ingested=1,
                bytes_processed=len(json.dumps(payload).encode()),
            )
        except Exception as exc:
            return IngestResult(
                success=False,
                plugin_name=self.name,
                errors=[str(exc)],
            )

    # ----------------------------------------------------------------- Helpers
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

    def _parse_messages(self, data: bytes) -> List[Tuple[MemshadowHeader, List[PsychEvent], bytes]]:
        messages: List[Tuple[MemshadowHeader, List[PsychEvent], bytes]] = []
        offset = 0
        while offset + HEADER_SIZE <= len(data):
            header = MemshadowHeader.unpack(data[offset : offset + HEADER_SIZE])
            if self._allowed_message_types and header.msg_type not in self._allowed_message_types:
                logger.debug("Skipping disallowed msg_type=%s", header.msg_type.name)
                offset += HEADER_SIZE + header.payload_len
                continue

            total_size = HEADER_SIZE + header.payload_len
            if offset + total_size > len(data):
                raise ValueError("Truncated MEMSHADOW payload")

            payload = data[offset + HEADER_SIZE : offset + total_size]
            if len(payload) != header.payload_len:
                raise ValueError("Payload length mismatch")

            events = self._parse_psych_events(payload)
            if not events and header.msg_type in PSYCH_MESSAGE_TYPES:
                logger.debug("No psych events decoded for msg_type=%s", header.msg_type.name)
            messages.append((header, events, payload))
            offset += total_size

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
            except ValueError as exc:
                logger.warning("Failed to parse PsychEvent: %s", exc)
                break
            pos += PSYCH_EVENT_SIZE
        return events

    def _extract_records(
        self,
        header: MemshadowHeader,
        events: List[PsychEvent],
        payload: bytes,
        facade: BrainMemoryFacade,
    ) -> Tuple[List[Dict[str, Any]], int]:
        records: List[Dict[str, Any]] = []
        psych_count = 0

        if header.msg_type in PSYCH_MESSAGE_TYPES and events:
            psych_records: List[PsychEventRecord] = []
            for idx, event in enumerate(events):
                psych_records.append(self._build_psych_record(header, event, idx))
            facade.store_psych_events(psych_records)
            records.extend([r.to_dict() for r in psych_records])
            psych_count = len(psych_records)
        else:
            records.append(self._build_generic_record(header, payload))

        return records, psych_count

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
            priority=header.priority.name if isinstance(header.priority, Priority) else str(header.priority),
            flags=int(header.flags) if isinstance(header.flags, MessageFlags) else int(header.flags or 0),
            batch_index=batch_index,
            scores=scores,
            dark_triad_average=event.dark_triad_average,
            confidence=event.confidence,
        )

    def _build_generic_record(self, header: MemshadowHeader, payload: bytes) -> Dict[str, Any]:
        parsed_payload: Any
        try:
            parsed_payload = json.loads(payload.decode())
        except Exception:
            parsed_payload = payload.hex()

        return {
            "message_type": header.msg_type.name,
            "priority": header.priority.name if isinstance(header.priority, Priority) else header.priority,
            "flags": int(header.flags) if isinstance(header.flags, MessageFlags) else int(header.flags or 0),
            "payload_len": header.payload_len,
            "timestamp_ns": header.timestamp_ns,
            "payload": parsed_payload,
        }


# ---------------------------------------------------------------- convenience
_default_plugin = MemshadowIngestPlugin()


def ingest_memshadow_binary(data: bytes, brain_interface: Any = None) -> IngestResult:
    if brain_interface:
        _default_plugin.set_brain_interface(brain_interface)
    return _default_plugin.ingest_memshadow_binary(data, brain_interface)


def ingest_memshadow_legacy(payload: Dict[str, Any], brain_interface: Any = None) -> IngestResult:
    if brain_interface:
        _default_plugin.set_brain_interface(brain_interface)
    return _default_plugin.ingest_memshadow_legacy(payload, brain_interface)
