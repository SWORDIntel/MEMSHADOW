#!/usr/bin/env python3
"""TGcollector Intel Ingest Endpoint for DSMIL Brain."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from config.memshadow_config import get_memshadow_config
from ..metrics.memshadow_metrics import get_memshadow_metrics_registry
from ..plugins.ingest.tgcollector_ingest import TGCollectorIngestor

logger = logging.getLogger(__name__)


class TGcollectorEndpoint:
    """Normalize TGcollector INTEL_* payloads into MEMSHADOW."""

    def __init__(self, brain_interface=None):
        self.brain_interface = brain_interface
        self._config = get_memshadow_config()
        self._metrics = get_memshadow_metrics_registry()
        self._ingestor = TGCollectorIngestor(brain_interface)

    def handle_post(self, request_data: bytes, content_type: str = "application/json") -> Dict[str, Any]:
        if not request_data:
            return self._failure_response("Empty payload")

        try:
            data = json.loads(request_data.decode() or "{}")
            if isinstance(data, dict) and "records" in data:
                data = data["records"]
            payloads: List[Dict[str, Any]] = data if isinstance(data, list) else [data]
            records = self._ingestor.ingest(payloads, source="tgcollector")
            summary = self._summarize_records(records)
            return {
                "success": True,
                "records_ingested": summary,
                "record_count": sum(summary.values()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except json.JSONDecodeError as exc:
            logger.warning("TGcollector ingest JSON parse error: %s", exc)
            return self._failure_response("Invalid JSON payload")
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.error("Unexpected TGcollector ingest failure: %s", exc, exc_info=True)
            return self._failure_response("Internal server error")

    def _failure_response(self, message: str) -> Dict[str, Any]:
        return {
            "success": False,
            "error": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _summarize_records(self, records: List[Any]) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for record in records:
            category = getattr(record, "category", "unknown")
            summary[category] = summary.get(category, 0) + 1
        return summary


def create_fastapi_router(brain_interface=None):
    """Return a FastAPI router for mounting into an API app."""
    from fastapi import APIRouter, Request

    endpoint = TGcollectorEndpoint(brain_interface)
    router = APIRouter()

    @router.post("/api/v1/ingest/tgcollector")
    async def ingest_tgcollector(request: Request):
        body = await request.body()
        return endpoint.handle_post(body, request.headers.get("content-type", "application/json"))

    return router
