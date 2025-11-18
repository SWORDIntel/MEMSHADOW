"""
MCP (Model Context Protocol) Integration Layer for MEMSHADOW

This module provides MCP-compatible endpoints for all MEMSHADOW modules:
- Document processing
- Memory management
- SWARM orchestration
- Security analysis

All endpoints are designed to be called by MCP servers and AI assistants.
"""

from fastapi import APIRouter

from .documents import router as documents_router
from .memory import router as memory_router
from .swarm import router as swarm_router

# Create main MCP router
mcp_router = APIRouter(prefix="/mcp", tags=["mcp"])

# Include sub-routers
mcp_router.include_router(documents_router, prefix="/documents", tags=["mcp-documents"])
mcp_router.include_router(memory_router, prefix="/memory", tags=["mcp-memory"])
mcp_router.include_router(swarm_router, prefix="/swarm", tags=["mcp-swarm"])

__all__ = ["mcp_router"]
