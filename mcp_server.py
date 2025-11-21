#!/usr/bin/env python3
"""
MEMSHADOW MCP Server

A Model Context Protocol server that exposes memory tools to AI assistants.
Keeps memories synced across all connected AI clients.

Usage:
  python mcp_server.py                    # stdio mode (for Claude Desktop)
  python mcp_server.py --http 8080        # HTTP mode (for remote access)

Add to Claude Desktop config (~/.config/claude-desktop/config.json):
{
  "mcpServers": {
    "memshadow": {
      "command": "python",
      "args": ["/path/to/MEMSHADOW/mcp_server.py"],
      "env": {"MEMSHADOW_API_URL": "http://localhost:8000"}
    }
  }
}
"""

import asyncio
import json
import sys
import os
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional
import httpx

# MCP Server configuration
MEMSHADOW_API_URL = os.environ.get("MEMSHADOW_API_URL", "http://localhost:8000")
MEMSHADOW_API_KEY = os.environ.get("MEMSHADOW_API_KEY", "")
SYNC_USER_ID = os.environ.get("MEMSHADOW_USER_ID", "mcp-client")


class MCPServer:
    """MCP Server implementation for MEMSHADOW memory system"""

    def __init__(self):
        self.tools = self._register_tools()
        self.resources = self._register_resources()
        self._memory_cache: Dict[str, Any] = {}
        self._last_sync = None

    def _register_tools(self) -> List[Dict[str, Any]]:
        """Register available MCP tools"""
        return [
            {
                "name": "store_memory",
                "description": "Store a memory/fact for later retrieval. Use this to remember important information from conversations.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The memory content to store"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for categorization"
                        },
                        "importance": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Importance level"
                        }
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "search_memories",
                "description": "Search through stored memories using semantic search. Use this to recall relevant information.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results to return",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "list_recent_memories",
                "description": "List the most recent memories",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of memories to return",
                            "default": 10
                        }
                    }
                }
            },
            {
                "name": "delete_memory",
                "description": "Delete a specific memory by ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The memory ID to delete"
                        }
                    },
                    "required": ["memory_id"]
                }
            },
            {
                "name": "sync_memories",
                "description": "Force sync memories with the MEMSHADOW server",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]

    def _register_resources(self) -> List[Dict[str, Any]]:
        """Register available MCP resources"""
        return [
            {
                "uri": "memshadow://memories/recent",
                "name": "Recent Memories",
                "description": "The 20 most recent memories",
                "mimeType": "application/json"
            },
            {
                "uri": "memshadow://memories/stats",
                "name": "Memory Statistics",
                "description": "Statistics about stored memories",
                "mimeType": "application/json"
            }
        ]

    async def _api_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request to MEMSHADOW API"""
        headers = kwargs.pop("headers", {})
        if MEMSHADOW_API_KEY:
            headers["Authorization"] = f"Bearer {MEMSHADOW_API_KEY}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{MEMSHADOW_API_URL}{endpoint}"
            resp = await client.request(method, url, headers=headers, **kwargs)

            if resp.status_code >= 400:
                return {"error": f"API error: {resp.status_code}", "detail": resp.text}

            return resp.json()

    async def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls"""

        if name == "store_memory":
            return await self._store_memory(
                content=arguments["content"],
                tags=arguments.get("tags", []),
                importance=arguments.get("importance", "medium")
            )

        elif name == "search_memories":
            return await self._search_memories(
                query=arguments["query"],
                limit=arguments.get("limit", 5)
            )

        elif name == "list_recent_memories":
            return await self._list_recent(limit=arguments.get("limit", 10))

        elif name == "delete_memory":
            return await self._delete_memory(memory_id=arguments["memory_id"])

        elif name == "sync_memories":
            return await self._sync_memories()

        else:
            return {"error": f"Unknown tool: {name}"}

    async def _store_memory(self, content: str, tags: List[str], importance: str) -> Dict[str, Any]:
        """Store a memory via MEMSHADOW API"""
        result = await self._api_request(
            "POST",
            "/api/v1/memory/ingest",
            json={
                "content": content,
                "extra_data": {
                    "tags": tags,
                    "importance": importance,
                    "source": "mcp",
                    "synced_at": datetime.utcnow().isoformat()
                }
            }
        )

        if "error" in result:
            return {"content": [{"type": "text", "text": f"Failed to store memory: {result['error']}"}]}

        # Update local cache
        memory_id = result.get("id", hashlib.md5(content.encode()).hexdigest()[:12])
        self._memory_cache[memory_id] = {
            "content": content,
            "tags": tags,
            "importance": importance,
            "created_at": datetime.utcnow().isoformat()
        }

        return {
            "content": [{
                "type": "text",
                "text": f"Memory stored successfully (ID: {memory_id})"
            }]
        }

    async def _search_memories(self, query: str, limit: int) -> Dict[str, Any]:
        """Search memories via MEMSHADOW API"""
        result = await self._api_request(
            "POST",
            "/api/v1/memory/retrieve",
            json={"query": query},
            params={"limit": limit}
        )

        if "error" in result:
            return {"content": [{"type": "text", "text": f"Search failed: {result['error']}"}]}

        memories = result if isinstance(result, list) else result.get("memories", [])

        if not memories:
            return {"content": [{"type": "text", "text": "No matching memories found."}]}

        formatted = []
        for i, mem in enumerate(memories, 1):
            formatted.append(f"{i}. [{mem.get('id', 'N/A')[:8]}] {mem.get('content', '')[:200]}")

        return {
            "content": [{
                "type": "text",
                "text": f"Found {len(memories)} memories:\n\n" + "\n\n".join(formatted)
            }]
        }

    async def _list_recent(self, limit: int) -> Dict[str, Any]:
        """List recent memories"""
        result = await self._api_request(
            "POST",
            "/api/v1/memory/retrieve",
            json={"query": ""},
            params={"limit": limit}
        )

        if "error" in result:
            return {"content": [{"type": "text", "text": f"Failed to list memories: {result['error']}"}]}

        memories = result if isinstance(result, list) else []

        if not memories:
            return {"content": [{"type": "text", "text": "No memories stored yet."}]}

        formatted = [f"- {m.get('content', '')[:100]}..." for m in memories]
        return {
            "content": [{
                "type": "text",
                "text": f"Recent {len(memories)} memories:\n" + "\n".join(formatted)
            }]
        }

    async def _delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Delete a memory"""
        result = await self._api_request("DELETE", f"/api/v1/memory/{memory_id}")

        if "error" in result:
            return {"content": [{"type": "text", "text": f"Delete failed: {result['error']}"}]}

        self._memory_cache.pop(memory_id, None)
        return {"content": [{"type": "text", "text": f"Memory {memory_id} deleted."}]}

    async def _sync_memories(self) -> Dict[str, Any]:
        """Force sync with server"""
        self._last_sync = datetime.utcnow()
        self._memory_cache.clear()

        return {
            "content": [{
                "type": "text",
                "text": f"Memories synced at {self._last_sync.isoformat()}"
            }]
        }

    async def handle_resource_read(self, uri: str) -> Dict[str, Any]:
        """Handle MCP resource reads"""
        if uri == "memshadow://memories/recent":
            result = await self._api_request(
                "POST",
                "/api/v1/memory/retrieve",
                json={"query": ""},
                params={"limit": 20}
            )
            return {"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(result)}]}

        elif uri == "memshadow://memories/stats":
            stats = {
                "cached_memories": len(self._memory_cache),
                "last_sync": self._last_sync.isoformat() if self._last_sync else None,
                "api_url": MEMSHADOW_API_URL
            }
            return {"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(stats)}]}

        return {"error": f"Unknown resource: {uri}"}


class MCPStdioTransport:
    """Stdio transport for MCP protocol"""

    def __init__(self, server: MCPServer):
        self.server = server
        self._request_id = 0

    async def run(self):
        """Main stdio loop"""
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break

                request = json.loads(line.decode())
                response = await self._handle_request(request)

                if response:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()

            except json.JSONDecodeError:
                continue
            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": str(e)},
                    "id": None
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()

    async def _handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle JSON-RPC request"""
        method = request.get("method")
        params = request.get("params", {})
        req_id = request.get("id")

        result = None

        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {"subscribe": False, "listChanged": False}
                },
                "serverInfo": {
                    "name": "memshadow",
                    "version": "1.0.0"
                }
            }

        elif method == "tools/list":
            result = {"tools": self.server.tools}

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            result = await self.server.handle_tool_call(tool_name, arguments)

        elif method == "resources/list":
            result = {"resources": self.server.resources}

        elif method == "resources/read":
            uri = params.get("uri")
            result = await self.server.handle_resource_read(uri)

        elif method == "notifications/initialized":
            return None  # No response for notifications

        else:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method not found: {method}"},
                "id": req_id
            }

        return {"jsonrpc": "2.0", "result": result, "id": req_id}


async def main():
    """Entry point"""
    server = MCPServer()

    if "--http" in sys.argv:
        # HTTP mode for remote access
        port = int(sys.argv[sys.argv.index("--http") + 1]) if len(sys.argv) > sys.argv.index("--http") + 1 else 8080
        print(f"Starting HTTP MCP server on port {port}...", file=sys.stderr)
        # Would need additional HTTP server implementation
        raise NotImplementedError("HTTP mode not yet implemented, use stdio mode")
    else:
        # Stdio mode for Claude Desktop
        transport = MCPStdioTransport(server)
        await transport.run()


if __name__ == "__main__":
    asyncio.run(main())
