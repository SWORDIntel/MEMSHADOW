#!/usr/bin/env python3
"""
SHRINK Intel HTTP API Endpoint for DSMIL Brain

Provides HTTP endpoint for SHRINK to send psychological intelligence data
to the local Brain. The Brain then handles mesh broadcasting.

Endpoint: POST /api/v1/ingest/shrink
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

# Add brain path for imports
brain_path = Path(__file__).parent.parent
if str(brain_path) not in sys.path:
    sys.path.insert(0, str(brain_path))

try:
    from plugins.ingest.memshadow_ingest import MemshadowIngestPlugin
    from plugins.ingest_framework import IngestPluginManager
    INGEST_AVAILABLE = True
except ImportError:
    INGEST_AVAILABLE = False
    logger.warning("Ingest framework not available")

logger = logging.getLogger(__name__)


class ShrinkIntelEndpoint:
    """
    HTTP endpoint handler for SHRINK psychological intelligence ingestion

    This can be integrated into Flask, FastAPI, or any HTTP framework.
    """

    def __init__(self, brain_interface=None, ingest_manager: Optional[Any] = None):
        """
        Initialize endpoint

        Args:
            brain_interface: Reference to BrainInterface instance (optional)
            ingest_manager: IngestPluginManager instance (optional)
        """
        self.brain_interface = brain_interface
        self.ingest_manager = ingest_manager

        # Initialize ingest plugin if available
        if INGEST_AVAILABLE and ingest_manager is None:
            try:
                self.ingest_manager = IngestPluginManager()
                self.ingest_manager.load_plugin("memshadow_ingest", {"enabled": True})
            except Exception as e:
                logger.warning(f"Could not initialize ingest manager: {e}")

        logger.info("ShrinkIntelEndpoint initialized")

    def handle_post(self, request_data: bytes, content_type: str = "application/octet-stream") -> Dict[str, Any]:
        """
        Handle POST request with SHRINK intel data

        Args:
            request_data: Binary MEMSHADOW protocol data or JSON
            content_type: Content type of request

        Returns:
            Response dictionary with status and message
        """
        try:
            # Parse based on content type
            if content_type == "application/octet-stream" or content_type.startswith("application/x-"):
                # Binary MEMSHADOW protocol
                return self._handle_binary_ingest(request_data)
            elif content_type == "application/json":
                # JSON format (legacy/fallback)
                import json
                data = json.loads(request_data.decode())
                return self._handle_json_ingest(data)
            else:
                # Try binary first, fallback to JSON
                try:
                    return self._handle_binary_ingest(request_data)
                except:
                    import json
                    data = json.loads(request_data.decode())
                    return self._handle_json_ingest(data)

        except Exception as e:
            logger.error(f"Error handling SHRINK intel request: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def _handle_binary_ingest(self, data: bytes) -> Dict[str, Any]:
        """Handle binary MEMSHADOW protocol ingestion"""
        if not self.ingest_manager:
            return {
                "success": False,
                "error": "Ingest manager not available",
            }

        # Use memshadow ingest plugin
        plugin = self.ingest_manager.get_plugin("memshadow_ingest")
        if not plugin:
            return {
                "success": False,
                "error": "MEMSHADOW ingest plugin not loaded",
            }

        # Ingest the data
        result = plugin.ingest(data)

        if result.success:
            # Store in memory tiers if brain_interface available
            if self.brain_interface and result.data:
                self._store_in_memory_tiers(result.data)

            return {
                "success": True,
                "items_ingested": result.items_ingested,
                "bytes_processed": result.bytes_processed,
                "messages_parsed": result.metadata.get("messages_parsed", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            return {
                "success": False,
                "errors": result.errors,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def _handle_json_ingest(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle JSON format ingestion (legacy/fallback)"""
        # Convert JSON to structured format
        # This is a fallback for non-binary clients

        items_ingested = 0

        # Store in memory tiers
        if self.brain_interface:
            self._store_in_memory_tiers([data])
            items_ingested = 1

        return {
            "success": True,
            "items_ingested": items_ingested,
            "format": "json",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _store_in_memory_tiers(self, extracted_data: list):
        """
        Store extracted data in appropriate memory tiers

        Args:
            extracted_data: List of extracted data dictionaries from ingest plugin
        """
        if not self.brain_interface:
            return

        try:
            # Route to appropriate memory tier based on data type
            for item in extracted_data:
                if not isinstance(item, dict):
                    continue

                data_type = item.get("type")

                if data_type == "psych_event":
                    # Store in working memory (L1) for active correlation
                    if hasattr(self.brain_interface, 'working_memory'):
                        self.brain_interface.working_memory.store(
                            item_id=f"psych_{item.get('session_id', 'unknown')}_{item.get('timestamp_ns', 0)}",
                            content=item,
                            content_type="psych_event",
                            priority="HIGH" if item.get("scores", {}).get("espionage_exposure", 0) > 0.7 else "NORMAL",
                        )

                    # Also store in episodic memory (L2) for long-term patterns
                    if hasattr(self.brain_interface, 'episodic_memory'):
                        self.brain_interface.episodic_memory.store(
                            event=item,
                            context={"source": "shrink", "type": "psych_assessment"},
                        )

                elif data_type == "improvement_announcement":
                    # Store improvement announcements in semantic memory (L3)
                    if hasattr(self.brain_interface, 'semantic_memory'):
                        self.brain_interface.semantic_memory.store(
                            concept=f"improvement_{item.get('improvement_id', 'unknown')}",
                            knowledge=item,
                            domain="self_improvement",
                        )

                elif data_type == "improvement_package":
                    # Store full improvement packages
                    if hasattr(self.brain_interface, 'semantic_memory'):
                        self.brain_interface.semantic_memory.store(
                            concept=f"improvement_{item.get('improvement_id', 'unknown')}",
                            knowledge=item,
                            domain="self_improvement",
                        )

        except Exception as e:
            logger.error(f"Error storing data in memory tiers: {e}", exc_info=True)


# Flask integration example
def create_flask_endpoint(brain_interface=None, ingest_manager=None):
    """
    Create Flask route handler

    Usage:
        from flask import Flask, request
        from ai.brain.api.shrink_endpoint import create_flask_endpoint

        app = Flask(__name__)
        handler = create_flask_endpoint(brain_interface=my_brain)

        @app.route('/api/v1/ingest/shrink', methods=['POST'])
        def shrink_ingest():
            return handler.handle_post(
                request.data,
                request.content_type
            )
    """
    endpoint = ShrinkIntelEndpoint(brain_interface, ingest_manager)

    def flask_handler():
        from flask import request, jsonify
        result = endpoint.handle_post(request.data, request.content_type)
        return jsonify(result), 200 if result.get("success") else 400

    return flask_handler


# FastAPI integration example
def create_fastapi_router(brain_interface=None, ingest_manager=None):
    """
    Create FastAPI router

    Usage:
        from fastapi import APIRouter, Request
        from ai.brain.api.shrink_endpoint import create_fastapi_router

        router = APIRouter()
        endpoint = ShrinkIntelEndpoint(brain_interface=my_brain)

        @router.post("/api/v1/ingest/shrink")
        async def shrink_ingest(request: Request):
            data = await request.body()
            content_type = request.headers.get("content-type", "application/octet-stream")
            return endpoint.handle_post(data, content_type)
    """
    from fastapi import APIRouter, Request, Response

    router = APIRouter()
    endpoint = ShrinkIntelEndpoint(brain_interface, ingest_manager)

    @router.post("/api/v1/ingest/shrink")
    async def shrink_ingest(request: Request):
        data = await request.body()
        content_type = request.headers.get("content-type", "application/octet-stream")
        result = endpoint.handle_post(data, content_type)

        if result.get("success"):
            return result
        else:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=result.get("error", "Ingestion failed"))

    return router


# Standalone HTTP server example (using http.server)
def create_http_handler(brain_interface=None, ingest_manager=None):
    """
    Create http.server BaseHTTPRequestHandler

    Usage:
        from http.server import HTTPServer
        from ai.brain.api.shrink_endpoint import create_http_handler

        handler_class = create_http_handler(brain_interface=my_brain)
        server = HTTPServer(('localhost', 8000), handler_class)
        server.serve_forever()
    """
    from http.server import BaseHTTPRequestHandler

    endpoint = ShrinkIntelEndpoint(brain_interface, ingest_manager)

    class ShrinkIntelHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/api/v1/ingest/shrink":
                content_length = int(self.headers.get('Content-Length', 0))
                data = self.rfile.read(content_length)
                content_type = self.headers.get('Content-Type', 'application/octet-stream')

                result = endpoint.handle_post(data, content_type)

                import json
                response = json.dumps(result).encode()

                self.send_response(200 if result.get("success") else 400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response)))
                self.end_headers()
                self.wfile.write(response)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            logger.info(f"{self.address_string()} - {format % args}")

    return ShrinkIntelHandler

