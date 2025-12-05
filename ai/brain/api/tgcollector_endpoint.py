#!/usr/bin/env python3
"""
TGcollector Intel Ingest Endpoint for DSMIL Brain

Accepts MEMSHADOW binary INTEL_REPORT messages from TGcollector and routes them
through the existing memshadow ingest plugin into the Brain memory tiers.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

brain_path = Path(__file__).parent.parent
if str(brain_path) not in sys.path:
    sys.path.insert(0, str(brain_path))

try:
    from plugins.ingest.memshadow_ingest import MemshadowIngestPlugin
    from plugins.ingest_framework import IngestPluginManager
    INGEST_AVAILABLE = True
except ImportError:
    INGEST_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Ingest framework not available")

logger = logging.getLogger(__name__)


class TGcollectorEndpoint:
    """Binary ingest endpoint for TGcollector intel."""

    def __init__(self, brain_interface=None, ingest_manager: Optional[Any] = None):
        self.brain_interface = brain_interface
        self.ingest_manager = ingest_manager
        if INGEST_AVAILABLE and ingest_manager is None:
            try:
                self.ingest_manager = IngestPluginManager()
                self.ingest_manager.load_plugin("memshadow_ingest", {"enabled": True})
            except Exception as exc:
                logger.warning(f"Could not initialize ingest manager: {exc}")
        logger.info("TGcollectorEndpoint initialized")

    def handle_post(self, request_data: bytes, content_type: str = "application/octet-stream") -> Dict[str, Any]:
        """Handle POST with MEMSHADOW binary payload."""
        try:
            if content_type == "application/octet-stream" or content_type.startswith("application/x-"):
                return self._handle_binary_ingest(request_data)
            else:
                return self._handle_binary_ingest(request_data)
        except Exception as exc:  # pragma: no cover - runtime only
            logger.error("tgcollector ingest error: %s", exc, exc_info=True)
            return {"success": False, "error": str(exc), "timestamp": datetime.now(timezone.utc).isoformat()}

    def _handle_binary_ingest(self, data: bytes) -> Dict[str, Any]:
        if not self.ingest_manager:
            return {"success": False, "error": "Ingest manager not available"}
        plugin = self.ingest_manager.get_plugin("memshadow_ingest")
        if not plugin:
            return {"success": False, "error": "MEMSHADOW ingest plugin not loaded"}

        result = plugin.ingest(data)
        if result.success:
            if self.brain_interface and result.data:
                for item in result.data:
                    self._store_intel(item)
            return {
                "success": True,
                "items_ingested": result.items_ingested,
                "bytes_processed": result.bytes_processed,
                "messages_parsed": result.metadata.get("messages_parsed", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        return {"success": False, "errors": result.errors, "timestamp": datetime.now(timezone.utc).isoformat()}

    def _store_intel(self, item: Dict[str, Any]) -> None:
        """Persist intel into brain memories."""
        if not self.brain_interface:
            return
        try:
            # Default: store in semantic memory
            if hasattr(self.brain_interface, "semantic_memory"):
                self.brain_interface.semantic_memory.store(
                    concept=item.get("source", "tgcollector"),
                    knowledge=item,
                    domain="tgcollector",
                )
            if hasattr(self.brain_interface, "episodic_memory"):
                self.brain_interface.episodic_memory.store(
                    event=item,
                    context={"source": "tgcollector"},
                )
        except Exception as exc:
            logger.warning("Failed to store TGcollector intel: %s", exc)


def create_fastapi_router(brain_interface=None, ingest_manager: Optional[Any] = None):
    """Return a FastAPI router for mounting into an API app."""
    from fastapi import APIRouter, Request

    endpoint = TGcollectorEndpoint(brain_interface, ingest_manager)
    router = APIRouter()

    @router.post("/api/v1/ingest/tgcollector")
    async def ingest_tgcollector(request: Request):
        body = await request.body()
        return endpoint.handle_post(body, request.headers.get("content-type", "application/octet-stream"))

    return router
