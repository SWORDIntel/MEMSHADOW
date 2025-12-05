#!/usr/bin/env python3
"""
API Ingest Plugin for DSMIL Brain

Ingests data from REST API endpoints.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.error import URLError

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plugins.ingest_framework import IngestPlugin, IngestResult


class APIIngestPlugin(IngestPlugin):
    """
    API Ingest Plugin

    Ingests data from REST API endpoints.
    """

    @property
    def name(self) -> str:
        return "api_ingest"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Ingest data from REST API endpoints"

    @property
    def supported_types(self) -> List[str]:
        return ["api", "rest", "http", "https"]

    def __init__(self):
        self._timeout = 30
        self._headers: Dict[str, str] = {}
        self._auth_token: Optional[str] = None

    def initialize(self, config: Dict[str, Any]) -> bool:
        self._timeout = config.get("timeout", self._timeout)
        self._headers = config.get("headers", {})
        self._auth_token = config.get("auth_token")
        return True

    def ingest(self, source: Any, **kwargs) -> IngestResult:
        """
        Ingest from API endpoint

        Args:
            source: URL string
            method: HTTP method (GET, POST)
            body: Request body for POST
            headers: Additional headers
        """
        try:
            url = str(source)
            method = kwargs.get("method", "GET")
            body = kwargs.get("body")
            headers = {**self._headers, **kwargs.get("headers", {})}

            # Add auth if configured
            if self._auth_token:
                headers["Authorization"] = f"Bearer {self._auth_token}"

            # Build request
            if body and isinstance(body, dict):
                body = json.dumps(body).encode("utf-8")
                headers["Content-Type"] = "application/json"

            request = Request(url, data=body, headers=headers, method=method)

            # Make request
            with urlopen(request, timeout=self._timeout) as response:
                data = response.read()

                # Try to parse JSON
                content_type = response.headers.get("Content-Type", "")
                if "json" in content_type:
                    try:
                        parsed_data = json.loads(data.decode("utf-8"))
                    except:
                        parsed_data = data
                else:
                    parsed_data = data

                metadata = {
                    "url": url,
                    "method": method,
                    "status": response.status,
                    "content_type": content_type,
                    "size": len(data),
                    "checksum": hashlib.sha256(data).hexdigest(),
                }

                return IngestResult(
                    success=True,
                    plugin_name=self.name,
                    data=parsed_data,
                    metadata=metadata,
                    items_ingested=1,
                    bytes_processed=len(data),
                )

        except URLError as e:
            return IngestResult(
                success=False,
                plugin_name=self.name,
                errors=[f"URL error: {e.reason}"],
            )
        except Exception as e:
            return IngestResult(
                success=False,
                plugin_name=self.name,
                errors=[str(e)],
            )


# Export plugin class
Plugin = APIIngestPlugin

