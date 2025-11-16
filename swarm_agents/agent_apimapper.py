"""
API Endpoint Mapper Agent

Discovers and maps API endpoints on web services.
Attempts to fetch OpenAPI specs and brute-forces common paths.
"""

import httpx
import uuid
from typing import Dict, Any, List
from base_agent import BaseAgent
import structlog

logger = structlog.get_logger()


class AgentAPIMapper(BaseAgent):
    """
    API endpoint discovery and mapping agent
    """

    # Common API paths to check
    COMMON_PATHS = [
        "/",
        "/api",
        "/api/v1",
        "/api/v2",
        "/v1",
        "/v2",
        "/health",
        "/healthcheck",
        "/status",
        "/metrics",
        "/docs",
        "/swagger",
        "/openapi.json",
        "/api-docs",
        "/api/docs",
        "/api/health",
        "/api/status",
        "/auth",
        "/api/auth",
        "/api/v1/auth",
        "/login",
        "/api/login",
        "/api/v1/users",
        "/api/v1/data",
        "/api/v1/items",
        "/admin",
        "/api/admin"
    ]

    def __init__(self, agent_id: str = None):
        agent_id = agent_id or f"apimapper-{uuid.uuid4().hex[:8]}"
        super().__init__(agent_id=agent_id, agent_type="apimapper")

        logger.info("API mapper agent initialized", agent_id=self.agent_id)

    async def execute_task(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute API endpoint mapping

        Task Payload:
            - target_host: Host to scan
            - target_port: Port to scan
            - use_https: Use HTTPS (default: False)
            - timeout: Request timeout in seconds (default: 5)

        Returns:
            Dictionary with discovered endpoints
        """
        target_host = task_payload.get('target_host')
        target_port = task_payload.get('target_port', 80)
        use_https = task_payload.get('use_https', False)
        timeout = task_payload.get('timeout', 5)

        if not target_host:
            raise ValueError("target_host is required")

        scheme = "https" if use_https else "http"
        base_url = f"{scheme}://{target_host}:{target_port}"

        logger.info(
            "Starting API endpoint mapping",
            agent_id=self.agent_id,
            base_url=base_url
        )

        discovered_endpoints = []
        openapi_spec = None

        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            # Try to fetch OpenAPI spec
            openapi_paths = ["/openapi.json", "/swagger.json", "/api/openapi.json", "/api/swagger.json"]

            for path in openapi_paths:
                try:
                    response = await client.get(f"{base_url}{path}")

                    if response.status_code == 200:
                        openapi_spec = response.json()
                        logger.info(
                            "OpenAPI spec found",
                            agent_id=self.agent_id,
                            path=path
                        )

                        # Extract endpoints from OpenAPI spec
                        if 'paths' in openapi_spec:
                            for endpoint_path in openapi_spec['paths'].keys():
                                discovered_endpoints.append({
                                    "path": endpoint_path,
                                    "methods": list(openapi_spec['paths'][endpoint_path].keys()),
                                    "source": "openapi",
                                    "status_code": 200
                                })

                        break

                except Exception as e:
                    logger.debug(
                        "OpenAPI fetch failed",
                        agent_id=self.agent_id,
                        path=path,
                        error=str(e)
                    )

            # Brute-force common paths
            for path in self.COMMON_PATHS:
                try:
                    response = await client.get(f"{base_url}{path}")

                    # Consider successful if not 404
                    if response.status_code != 404:
                        endpoint_info = {
                            "path": path,
                            "status_code": response.status_code,
                            "source": "brute_force",
                            "content_type": response.headers.get('content-type', ''),
                            "server": response.headers.get('server', '')
                        }

                        # Check if it's already in discovered endpoints
                        if not any(e['path'] == path for e in discovered_endpoints):
                            discovered_endpoints.append(endpoint_info)

                        logger.debug(
                            "Endpoint discovered",
                            agent_id=self.agent_id,
                            path=path,
                            status_code=response.status_code
                        )

                except httpx.TimeoutException:
                    logger.debug(
                        "Request timeout",
                        agent_id=self.agent_id,
                        path=path
                    )

                except Exception as e:
                    logger.debug(
                        "Request failed",
                        agent_id=self.agent_id,
                        path=path,
                        error=str(e)
                    )

        logger.info(
            "API endpoint mapping completed",
            agent_id=self.agent_id,
            endpoints_discovered=len(discovered_endpoints)
        )

        return {
            "discovered_endpoints": discovered_endpoints,
            "num_discovered_endpoints": len(discovered_endpoints),
            "has_openapi_spec": openapi_spec is not None,
            "base_url": base_url,
            "api_endpoints_mapped": True
        }


if __name__ == "__main__":
    agent = AgentAPIMapper()
    agent.run()
