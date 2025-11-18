"""
HYDRA SWARM API Mapper Agent
Phase 7: API discovery, mapping, and documentation

The API mapper agent performs:
- REST API endpoint discovery
- GraphQL schema introspection
- OpenAPI/Swagger parsing
- Parameter enumeration
- Response schema analysis
- Rate limiting detection
"""

from typing import Dict, List, Optional, Any, Set
import asyncio
import json
import re
from urllib.parse import urlparse, urljoin
from datetime import datetime
import structlog

try:
    import httpx
except ImportError:
    httpx = None

from .agent_base import (
    BaseAgent,
    AgentStatus,
    AgentCapability,
    Finding,
    FindingSeverity
)

logger = structlog.get_logger()


class APIEndpoint:
    """Represents a discovered API endpoint"""
    def __init__(
        self,
        path: str,
        methods: List[str],
        parameters: Optional[Dict] = None,
        authentication_required: bool = False
    ):
        self.path = path
        self.methods = methods
        self.parameters = parameters or {}
        self.authentication_required = authentication_required
        self.response_schemas: Dict[str, Any] = {}


class APIMapperAgent(BaseAgent):
    """
    API Mapper Agent for HYDRA SWARM.

    Specializes in:
    - REST API discovery and mapping
    - GraphQL introspection
    - OpenAPI/Swagger documentation parsing
    - Parameter fuzzing and enumeration
    - Response structure analysis
    - Rate limiting detection

    Example:
        agent = APIMapperAgent()
        findings = await agent.execute(
            target="https://api.example.com",
            blackboard=blackboard,
            mission_params={"deep_scan": True}
        )
    """

    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(
            agent_id=agent_id,
            agent_type="APIMapperAgent",
            capabilities=[
                AgentCapability.API_MAPPING,
                AgentCapability.RECONNAISSANCE,
                AgentCapability.FUZZING
            ]
        )

        # Configuration
        self.timeout_seconds = 10
        self.max_concurrent_requests = 5
        self.user_agent = "MEMSHADOW-APIMapper/1.0"

        # Discovered endpoints
        self.discovered_endpoints: Dict[str, APIEndpoint] = {}

        # Common API documentation paths
        self.doc_paths = [
            "/swagger.json",
            "/swagger.yaml",
            "/openapi.json",
            "/openapi.yaml",
            "/api/swagger.json",
            "/api/openapi.json",
            "/api-docs",
            "/api/docs",
            "/docs",
            "/redoc",
            "/graphql",
            "/graphiql",
            "/.well-known/openapi.json"
        ]

        # Common HTTP methods to test
        self.http_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]

        # Common parameter names for fuzzing
        self.common_parameters = [
            "id", "user_id", "userId", "uid",
            "page", "limit", "offset", "size",
            "sort", "order", "orderBy",
            "filter", "search", "query", "q",
            "fields", "include", "expand",
            "format", "callback", "jsonp"
        ]

    async def execute(
        self,
        target: str,
        blackboard: Any,
        mission_params: Optional[Dict] = None
    ) -> List[Finding]:
        """
        Execute API mapping on target.

        Args:
            target: Target API URL
            blackboard: Shared blackboard for intelligence
            mission_params: Mission-specific parameters

        Returns:
            List of findings
        """
        logger.info("API mapper agent starting", agent_id=self.agent_id, target=target)
        self.status = AgentStatus.SCANNING

        params = mission_params or {}
        deep_scan = params.get("deep_scan", False)

        findings: List[Finding] = []

        # Send heartbeat
        await self._heartbeat(blackboard)

        # Phase 1: Check for API documentation
        doc_findings = await self._discover_documentation(target, blackboard)
        findings.extend(doc_findings)

        # Phase 2: Discover REST endpoints
        rest_findings = await self._discover_rest_endpoints(target, blackboard)
        findings.extend(rest_findings)

        # Phase 3: GraphQL introspection
        graphql_findings = await self._introspect_graphql(target, blackboard)
        findings.extend(graphql_findings)

        # Phase 4: Parameter enumeration
        if deep_scan:
            param_findings = await self._enumerate_parameters(target, blackboard)
            findings.extend(param_findings)

        # Phase 5: Rate limiting detection
        rate_findings = await self._detect_rate_limiting(target, blackboard)
        findings.extend(rate_findings)

        # Share API intelligence
        await self._share_api_intelligence(blackboard)

        # Report findings
        for finding in findings:
            await self.report_finding(blackboard, finding)

        self.status = AgentStatus.IDLE
        logger.info(
            "API mapper agent completed",
            agent_id=self.agent_id,
            target=target,
            findings=len(findings),
            endpoints_discovered=len(self.discovered_endpoints)
        )

        return findings

    async def _discover_documentation(
        self,
        target: str,
        blackboard: Any
    ) -> List[Finding]:
        """Discover API documentation endpoints"""
        findings = []

        if not httpx:
            return findings

        parsed = urlparse(target)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for doc_path in self.doc_paths:
                try:
                    url = urljoin(base_url, doc_path)
                    response = await client.get(
                        url,
                        headers={"User-Agent": self.user_agent}
                    )

                    if response.status_code == 200:
                        # Found documentation
                        findings.append(Finding(
                            title=f"API Documentation Exposed: {doc_path}",
                            description=f"API documentation is publicly accessible",
                            severity=FindingSeverity.INFO,
                            location=url,
                            evidence={
                                "path": doc_path,
                                "content_type": response.headers.get("content-type", ""),
                                "size": len(response.content)
                            }
                        ))

                        # Parse OpenAPI/Swagger if applicable
                        if "swagger" in doc_path or "openapi" in doc_path:
                            await self._parse_openapi_spec(response, base_url, findings)

                except Exception as e:
                    logger.debug("Doc probe failed", path=doc_path, error=str(e))

        return findings

    async def _parse_openapi_spec(
        self,
        response: Any,
        base_url: str,
        findings: List[Finding]
    ):
        """Parse OpenAPI/Swagger specification"""
        try:
            spec = response.json()

            # Extract endpoints from paths
            paths = spec.get("paths", {})
            endpoint_count = len(paths)

            if endpoint_count > 0:
                findings.append(Finding(
                    title=f"OpenAPI Specification Discovered",
                    description=f"Found API specification with {endpoint_count} endpoints",
                    severity=FindingSeverity.INFO,
                    location=base_url,
                    evidence={
                        "endpoint_count": endpoint_count,
                        "version": spec.get("openapi") or spec.get("swagger"),
                        "title": spec.get("info", {}).get("title", "Unknown")
                    }
                ))

                # Store discovered endpoints
                for path, methods_config in paths.items():
                    methods = [m.upper() for m in methods_config.keys() if m.upper() in self.http_methods]

                    endpoint = APIEndpoint(
                        path=path,
                        methods=methods,
                        parameters=self._extract_parameters_from_spec(methods_config)
                    )
                    self.discovered_endpoints[path] = endpoint

                # Check for security definitions
                security = spec.get("security") or spec.get("securityDefinitions")
                if not security:
                    findings.append(Finding(
                        title="No Security Scheme Defined",
                        description="OpenAPI specification does not define security requirements",
                        severity=FindingSeverity.MEDIUM,
                        location=base_url,
                        evidence={"spec_version": spec.get("openapi") or spec.get("swagger")}
                    ))

        except Exception as e:
            logger.error("OpenAPI parsing error", error=str(e))

    def _extract_parameters_from_spec(
        self,
        methods_config: Dict
    ) -> Dict[str, List[str]]:
        """Extract parameters from OpenAPI method configuration"""
        all_params = {}

        for method, config in methods_config.items():
            if method.upper() not in self.http_methods:
                continue

            parameters = config.get("parameters", [])
            param_names = [p.get("name") for p in parameters if p.get("name")]

            if param_names:
                all_params[method.upper()] = param_names

        return all_params

    async def _discover_rest_endpoints(
        self,
        target: str,
        blackboard: Any
    ) -> List[Finding]:
        """Discover REST API endpoints"""
        findings = []

        if not httpx:
            return findings

        # Get shared intelligence from recon agent
        recon_endpoints = await blackboard.get("intel:endpoints") or []

        # Filter for API-like endpoints
        api_endpoints = [
            ep for ep in recon_endpoints
            if any(keyword in ep.lower() for keyword in ["/api", "/v1", "/v2", "/v3"])
        ]

        if not api_endpoints and "/api" not in target:
            # Fallback: probe common API patterns
            parsed = urlparse(target)
            base_url = f"{parsed.scheme}://{parsed.netloc}"

            api_endpoints = [
                urljoin(base_url, path)
                for path in ["/api", "/api/v1", "/api/v2"]
            ]

        # Test each endpoint with different HTTP methods
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for endpoint in api_endpoints[:10]:  # Limit to avoid overwhelming
                methods_allowed = await self._test_endpoint_methods(client, endpoint)

                if methods_allowed:
                    # Record endpoint
                    parsed_ep = urlparse(endpoint)
                    path = parsed_ep.path

                    self.discovered_endpoints[path] = APIEndpoint(
                        path=path,
                        methods=methods_allowed
                    )

                    findings.append(Finding(
                        title=f"REST Endpoint Discovered: {path}",
                        description=f"Endpoint responds to methods: {', '.join(methods_allowed)}",
                        severity=FindingSeverity.INFO,
                        location=endpoint,
                        evidence={"methods": methods_allowed}
                    ))

        return findings

    async def _test_endpoint_methods(
        self,
        client: Any,
        endpoint: str
    ) -> List[str]:
        """Test which HTTP methods an endpoint accepts"""
        allowed_methods = []

        # First try OPTIONS request
        try:
            response = await client.options(
                endpoint,
                headers={"User-Agent": self.user_agent}
            )

            if "Allow" in response.headers:
                allowed_methods = [
                    m.strip() for m in response.headers["Allow"].split(",")
                ]
                return allowed_methods
        except Exception:
            pass

        # Fallback: test common methods individually
        for method in ["GET", "POST", "PUT", "DELETE"]:
            try:
                response = await client.request(
                    method,
                    endpoint,
                    headers={"User-Agent": self.user_agent}
                )

                # Consider it allowed if not 404/405
                if response.status_code not in [404, 405]:
                    allowed_methods.append(method)

            except Exception:
                pass

        return allowed_methods

    async def _introspect_graphql(
        self,
        target: str,
        blackboard: Any
    ) -> List[Finding]:
        """Introspect GraphQL API"""
        findings = []

        if not httpx:
            return findings

        parsed = urlparse(target)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # Common GraphQL paths
        graphql_paths = ["/graphql", "/graphiql", "/api/graphql"]

        introspection_query = {
            "query": """
                {
                    __schema {
                        types {
                            name
                            kind
                        }
                        queryType { name }
                        mutationType { name }
                    }
                }
            """
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for path in graphql_paths:
                try:
                    url = urljoin(base_url, path)
                    response = await client.post(
                        url,
                        json=introspection_query,
                        headers={
                            "User-Agent": self.user_agent,
                            "Content-Type": "application/json"
                        }
                    )

                    if response.status_code == 200:
                        data = response.json()

                        if "data" in data and "__schema" in data["data"]:
                            # GraphQL introspection successful
                            schema = data["data"]["__schema"]
                            type_count = len(schema.get("types", []))

                            findings.append(Finding(
                                title="GraphQL Introspection Enabled",
                                description=f"GraphQL API allows introspection. Discovered {type_count} types.",
                                severity=FindingSeverity.MEDIUM,
                                location=url,
                                evidence={
                                    "path": path,
                                    "type_count": type_count,
                                    "query_type": schema.get("queryType", {}).get("name"),
                                    "mutation_type": schema.get("mutationType", {}).get("name")
                                }
                            ))

                except Exception as e:
                    logger.debug("GraphQL introspection failed", path=path, error=str(e))

        return findings

    async def _enumerate_parameters(
        self,
        target: str,
        blackboard: Any
    ) -> List[Finding]:
        """Enumerate potential parameters for discovered endpoints"""
        findings = []

        if not httpx or not self.discovered_endpoints:
            return findings

        parsed = urlparse(target)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            # Test common parameters on discovered GET endpoints
            for path, endpoint in list(self.discovered_endpoints.items())[:5]:  # Limit
                if "GET" not in endpoint.methods:
                    continue

                url = urljoin(base_url, path)

                # Test common parameters
                for param in self.common_parameters[:5]:  # Limit
                    try:
                        # Test with dummy value
                        test_url = f"{url}?{param}=test"
                        response = await client.get(
                            test_url,
                            headers={"User-Agent": self.user_agent}
                        )

                        # If we get a different response, parameter might be valid
                        if response.status_code == 200:
                            if param not in endpoint.parameters.get("GET", []):
                                if "GET" not in endpoint.parameters:
                                    endpoint.parameters["GET"] = []
                                endpoint.parameters["GET"].append(param)

                    except Exception:
                        pass

        return findings

    async def _detect_rate_limiting(
        self,
        target: str,
        blackboard: Any
    ) -> List[Finding]:
        """Detect rate limiting on API"""
        findings = []

        if not httpx:
            return findings

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            # Send rapid requests
            start_time = datetime.utcnow()
            rate_limited = False
            request_count = 0

            try:
                for i in range(20):  # Test with 20 requests
                    response = await client.get(
                        target,
                        headers={"User-Agent": self.user_agent}
                    )

                    request_count += 1

                    # Check for rate limiting indicators
                    if response.status_code == 429:
                        rate_limited = True
                        retry_after = response.headers.get("Retry-After", "unknown")

                        findings.append(Finding(
                            title="Rate Limiting Detected",
                            description=f"API rate limiting triggered after {request_count} requests",
                            severity=FindingSeverity.INFO,
                            location=target,
                            evidence={
                                "requests_before_limit": request_count,
                                "retry_after": retry_after,
                                "rate_limit_header": response.headers.get("X-RateLimit-Limit")
                            }
                        ))
                        break

                    # Small delay between requests
                    await asyncio.sleep(0.1)

                if not rate_limited:
                    findings.append(Finding(
                        title="No Rate Limiting Detected",
                        description=f"API did not enforce rate limits after {request_count} rapid requests",
                        severity=FindingSeverity.LOW,
                        location=target,
                        evidence={"requests_sent": request_count}
                    ))

            except Exception as e:
                logger.error("Rate limiting detection error", error=str(e))

        return findings

    async def _share_api_intelligence(self, blackboard: Any):
        """Share discovered API intelligence with swarm"""
        if self.discovered_endpoints:
            # Share endpoint map
            endpoint_data = {
                path: {
                    "methods": ep.methods,
                    "parameters": ep.parameters,
                    "auth_required": ep.authentication_required
                }
                for path, ep in self.discovered_endpoints.items()
            }

            await self.share_intelligence(
                blackboard,
                "api_endpoints",
                {
                    "endpoint_count": len(self.discovered_endpoints),
                    "endpoints": endpoint_data
                }
            )

            logger.info(
                "Shared API intelligence",
                agent_id=self.agent_id,
                endpoints=len(self.discovered_endpoints)
            )
