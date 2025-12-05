#!/usr/bin/env python3
"""HTTP endpoint for SHRINK → Brain MEMSHADOW ingest."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from config.memshadow_config import get_memshadow_config
from dsmil_protocol import MemshadowHeader, MessageType, Priority
from ..metrics.memshadow_metrics import get_memshadow_metrics_registry
from ..plugins.ingest.memshadow_ingest import (
    BrainMemoryFacade,
    MemshadowIntelEdgeProcessor,
    ingest_memshadow_binary as global_ingest_binary,
    ingest_memshadow_legacy as global_ingest_legacy,
)

logger = logging.getLogger(__name__)


class ShrinkIntelEndpoint:
    """Process POST /api/v1/ingest/shrink requests."""

    def __init__(
        self,
        brain_interface: Any = None,
        processor: Optional[MemshadowIntelEdgeProcessor] = None,
    ):
        self.brain_interface = brain_interface
        self._config = get_memshadow_config()
        self._metrics = get_memshadow_metrics_registry()
        self._processor = processor or MemshadowIntelEdgeProcessor(
            config=self._config,
            metrics=self._metrics,
            brain_memory_facade=BrainMemoryFacade(brain_interface),
        )

    # ------------------------------------------------------------------ routing
    def handle_post(self, request_data: bytes, content_type: str = "application/octet-stream") -> Tuple[Dict[str, Any], int]:
        if not self._config.enable_shrink_ingest:
            return self._failure_response("SHRINK ingest disabled", status=503)

        try:
            if content_type.startswith("application/octet-stream") or content_type.startswith("application/x-"):
                records = self._processor.ingest_bytes(request_data, source="shrink", source_type="shrink")
            elif content_type == "application/json":
                payload = json.loads(request_data.decode() or "{}")
                message_bytes = self._wrap_json_as_memshadow(payload)
                records = self._processor.ingest_bytes(message_bytes, source="shrink", source_type="json")
            else:
                try:
                    records = self._processor.ingest_bytes(request_data, source="shrink", source_type="binary")
                except Exception:
                    payload = json.loads(request_data.decode() or "{}")
                    message_bytes = self._wrap_json_as_memshadow(payload)
                    records = self._processor.ingest_bytes(message_bytes, source="shrink", source_type="json")

            summary = self._summarize_records(records)
            response = {
                "success": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "records_ingested": summary,
                "record_count": sum(summary.values()),
            }
            return response, 200
        except json.JSONDecodeError as exc:
            logger.warning("SHRINK ingest JSON parse error: %s", exc)
            return self._failure_response("Invalid JSON payload", status=400)
        except ValueError as exc:
            logger.warning("SHRINK ingest validation error: %s", exc)
            return self._failure_response(str(exc), status=400)
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.error("Unexpected SHRINK ingest failure: %s", exc, exc_info=True)
            return self._failure_response("Internal server error", status=500)

    # ---------------------------------------------------------------- helpers
    def _failure_response(self, message: str, status: int) -> Tuple[Dict[str, Any], int]:
        return (
            {
                "success": False,
                "error": message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            status,
        )

    def _wrap_json_as_memshadow(self, payload: Dict[str, Any]) -> bytes:
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

    def _summarize_records(self, records: List[Any]) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for record in records:
            category = getattr(record, "category", "unknown")
            summary[category] = summary.get(category, 0) + 1
        return summary


# --------------------------------------------------------------------- adapters
def create_flask_endpoint(brain_interface: Any = None, ingest_plugin: Optional[MemshadowIngestPlugin] = None):
    from flask import jsonify, request

    endpoint = ShrinkIntelEndpoint(brain_interface, ingest_plugin)

    def flask_handler():
        body, status = endpoint.handle_post(request.data, request.content_type or "application/octet-stream")
        return jsonify(body), status

    return flask_handler


def create_fastapi_router(brain_interface: Any = None, ingest_plugin: Optional[MemshadowIngestPlugin] = None):
    from fastapi import APIRouter, Request, Response

    endpoint = ShrinkIntelEndpoint(brain_interface, ingest_plugin)
    router = APIRouter()

    @router.post("/api/v1/ingest/shrink")
    async def shrink_ingest(request: Request):
        data = await request.body()
        body, status = endpoint.handle_post(data, request.headers.get("content-type", "application/octet-stream"))
        return Response(content=json.dumps(body), media_type="application/json", status_code=status)

    return router


def create_http_handler(brain_interface: Any = None, ingest_plugin: Optional[MemshadowIngestPlugin] = None):
    from http.server import BaseHTTPRequestHandler

    endpoint = ShrinkIntelEndpoint(brain_interface, ingest_plugin)

    class ShrinkIntelHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/api/v1/ingest/shrink":
                self.send_response(404)
                self.end_headers()
                return

            content_length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(content_length)
            body, status = endpoint.handle_post(data, self.headers.get("Content-Type", "application/octet-stream"))

            response = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, fmt, *args):  # pragma: no cover - passthrough to logger
            logger.info("%s - %s", self.address_string(), fmt % args)

    return ShrinkIntelHandler


# Convenience wrappers for scripts requiring quick ingest access -----------------
def ingest_shrink_binary(data: bytes, brain_interface: Any = None):
    return global_ingest_binary(data, brain_interface)


def ingest_shrink_legacy(payload: Dict[str, Any], brain_interface: Any = None):
    return global_ingest_legacy(payload, brain_interface)
