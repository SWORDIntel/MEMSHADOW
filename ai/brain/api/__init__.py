#!/usr/bin/env python3
"""
DSMIL Brain API Module

HTTP endpoints for external integrations:
- SHRINK psychological intelligence ingestion
- Memory tier queries
- Federation management
"""

from .shrink_endpoint import ShrinkIntelEndpoint, create_flask_endpoint, create_fastapi_router
from .tgcollector_endpoint import TGcollectorEndpoint, create_fastapi_router as create_tgcollector_router

__all__ = [
    "ShrinkIntelEndpoint",
    "TGcollectorEndpoint",
    "create_tgcollector_router",
    "create_flask_endpoint",
    "create_fastapi_router",
]
