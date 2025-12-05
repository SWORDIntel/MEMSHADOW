#!/usr/bin/env python3
"""TGcollector → MEMSHADOW ingest bridge."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from dsmil_protocol import MemshadowHeader, MessageType, Priority

from config.memshadow_config import get_memshadow_config
from ...metrics.memshadow_metrics import get_memshadow_metrics_registry
from .memshadow_ingest import BrainMemoryFacade, MemshadowIntelEdgeProcessor


class TGCollectorIngestor:
    """Normalizes TGcollector JSON into MEMSHADOW messages."""

    def __init__(self, brain_interface: Any = None):
        self._config = get_memshadow_config()
        self._metrics = get_memshadow_metrics_registry()
        self._processor = MemshadowIntelEdgeProcessor(
            config=self._config,
            metrics=self._metrics,
            brain_memory_facade=BrainMemoryFacade(brain_interface),
        )

    def ingest(self, payload: Sequence[Dict[str, Any]], source: str = "tgcollector"):
        if not isinstance(payload, list):
            payload = [payload]

        message_bytes = b"".join(self._build_memshadow_message(item) for item in payload)
        return self._processor.ingest_bytes(message_bytes, source=source, source_type="tgcollector")

    def _build_memshadow_message(self, intel: Dict[str, Any]) -> bytes:
        severity = str(intel.get("severity", "medium")).lower()
        priority = {
            "critical": Priority.CRITICAL,
            "high": Priority.HIGH,
            "medium": Priority.NORMAL,
            "low": Priority.LOW,
        }.get(severity, Priority.NORMAL)

        normalized = {
            "source": intel.get("source", "tgcollector"),
            "indicator": intel.get("indicator") or intel.get("ioc"),
            "threat_type": intel.get("threat_type"),
            "severity": severity,
            "confidence": intel.get("confidence", 0.5),
            "tags": intel.get("tags", []),
            "raw_intel": intel,
        }

        body = json.dumps(normalized).encode()
        header = MemshadowHeader(
            msg_type=MessageType.THREAT_REPORT,
            priority=priority,
            payload_len=len(body),
        )
        return header.pack() + body

    @property
    def processor(self) -> MemshadowIntelEdgeProcessor:
        return self._processor
