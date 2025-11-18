"""
HYDRA SWARM Reconnaissance Agent
Phase 7: Intelligence gathering and target enumeration

The recon agent performs:
- Endpoint discovery
- Technology detection
- Header analysis
- Subdomain enumeration
- Port scanning
- Service fingerprinting
"""

from typing import Dict, List, Optional, Any
import asyncio
import re
from urllib.parse import urlparse, urljoin
from datetime import datetime
import structlog

try:
    import httpx
except ImportError:
    httpx = None  # Mock for testing

from .agent_base import (
    BaseAgent,
    AgentStatus,
    AgentCapability,
    Finding,
    FindingSeverity
)

logger = structlog.get_logger()


class ReconAgent(BaseAgent):
    """
    Reconnaissance Agent for HYDRA SWARM.

    Specializes in:
    - Endpoint discovery (crawling, brute force)
    - Technology stack detection
    - Security header analysis
    - Subdomain enumeration
    - Service fingerprinting

    Example:
        agent = ReconAgent()
        findings = await agent.execute(
            target="https://api.example.com",
            blackboard=blackboard,
            mission_params={"max_depth": 3}
        )
    """

    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(
            agent_id=agent_id,
            agent_type="ReconAgent",
            capabilities=[
                AgentCapability.RECONNAISSANCE,
                AgentCapability.API_MAPPING,
                AgentCapability.VULNERABILITY_SCANNING
            ]
        )

        # Configuration
        self.timeout_seconds = 10
        self.max_concurrent_requests = 10
        self.user_agent = "MEMSHADOW-Recon/1.0"

        # Common endpoints to probe
        self.common_endpoints = [
            # API endpoints
            "/api", "/api/v1", "/api/v2", "/api/v3",
            "/api/docs", "/api/swagger", "/swagger", "/swagger-ui",
            "/graphql", "/graphiql",
            "/openapi.json", "/swagger.json",

            # Authentication
            "/auth", "/login", "/signin", "/signup", "/register",
            "/oauth", "/oauth2", "/token", "/refresh",

            # Admin/Debug
            "/admin", "/dashboard", "/console", "/debug",
            "/health", "/status", "/metrics", "/actuator",

            # Common files
            "/robots.txt", "/sitemap.xml", "/.well-known/security.txt",
            "/package.json", "/composer.json",

            # Security
            "/.git", "/.env", "/config", "/backup"
        ]

        # Technology signatures
        self.tech_signatures = {
            "Express": ["X-Powered-By: Express"],
            "Django": ["csrftoken", "django"],
            "Flask": ["Werkzeug", "Flask"],
            "FastAPI": ["fastapi", "/docs", "/redoc"],
            "Rails": ["X-Runtime", "rails"],
            "Spring": ["Spring", "Tomcat"],
            "ASP.NET": ["X-AspNet-Version", "ASP.NET"],
            "PHP": ["X-Powered-By: PHP", "PHPSESSID"],
            "Node.js": ["X-Powered-By: Express", "node"],
            "Nginx": ["Server: nginx"],
            "Apache": ["Server: Apache"]
        }

    async def execute(
        self,
        target: str,
        blackboard: Any,
        mission_params: Optional[Dict] = None
    ) -> List[Finding]:
        """
        Execute reconnaissance on target.

        Args:
            target: Target URL to reconnaissance
            blackboard: Shared blackboard for intelligence
            mission_params: Mission-specific parameters

        Returns:
            List of findings
        """
        logger.info("Recon agent starting", agent_id=self.agent_id, target=target)
        self.status = AgentStatus.SCANNING

        params = mission_params or {}
        max_depth = params.get("max_depth", 2)
        scan_ports = params.get("scan_ports", False)

        findings: List[Finding] = []

        # Send heartbeat
        await self._heartbeat(blackboard)

        # Phase 1: Initial reconnaissance
        initial_findings = await self._initial_recon(target, blackboard)
        findings.extend(initial_findings)

        # Phase 2: Endpoint discovery
        endpoint_findings = await self._discover_endpoints(target, blackboard, max_depth)
        findings.extend(endpoint_findings)

        # Phase 3: Technology detection
        tech_findings = await self._detect_technologies(target, blackboard)
        findings.extend(tech_findings)

        # Phase 4: Security header analysis
        header_findings = await self._analyze_security_headers(target, blackboard)
        findings.extend(header_findings)

        # Share discovered intelligence
        await self._share_recon_intelligence(blackboard, findings)

        # Report findings
        for finding in findings:
            await self.report_finding(blackboard, finding)

        self.status = AgentStatus.IDLE
        logger.info(
            "Recon agent completed",
            agent_id=self.agent_id,
            target=target,
            findings=len(findings)
        )

        return findings

    async def _initial_recon(
        self,
        target: str,
        blackboard: Any
    ) -> List[Finding]:
        """Initial target reconnaissance"""
        findings = []

        if not httpx:
            logger.warning("httpx not available, using mock recon")
            return findings

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                # Probe root endpoint
                response = await client.get(
                    target,
                    headers={"User-Agent": self.user_agent},
                    follow_redirects=True
                )

                # Check for information disclosure in response
                if response.status_code == 200:
                    # Check for debug mode indicators
                    debug_indicators = [
                        "DEBUG=True",
                        "DJANGO_DEBUG",
                        "flask_debug",
                        "stack trace",
                        "traceback"
                    ]

                    content_lower = response.text.lower()
                    for indicator in debug_indicators:
                        if indicator.lower() in content_lower:
                            findings.append(Finding(
                                title="Debug Mode Enabled",
                                description=f"Target appears to have debug mode enabled. Indicator: {indicator}",
                                severity=FindingSeverity.MEDIUM,
                                location=target,
                                evidence={"indicator": indicator}
                            ))
                            break

                # Check response time (potential DoS indicator)
                if response.elapsed.total_seconds() > 5:
                    findings.append(Finding(
                        title="Slow Response Time",
                        description=f"Target responded in {response.elapsed.total_seconds():.2f}s (>5s threshold)",
                        severity=FindingSeverity.LOW,
                        location=target,
                        evidence={"response_time": response.elapsed.total_seconds()}
                    ))

        except httpx.TimeoutException:
            findings.append(Finding(
                title="Request Timeout",
                description=f"Target timed out after {self.timeout_seconds}s",
                severity=FindingSeverity.LOW,
                location=target,
                evidence={"timeout": self.timeout_seconds}
            ))
        except Exception as e:
            logger.error("Initial recon error", error=str(e), target=target)

        return findings

    async def _discover_endpoints(
        self,
        target: str,
        blackboard: Any,
        max_depth: int
    ) -> List[Finding]:
        """Discover API endpoints"""
        findings = []
        discovered_endpoints = []

        if not httpx:
            return findings

        # Normalize target URL
        parsed = urlparse(target)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            # Probe common endpoints
            tasks = []
            for endpoint in self.common_endpoints[:20]:  # Limit to avoid overwhelming
                url = urljoin(base_url, endpoint)
                tasks.append(self._probe_endpoint(client, url))

            # Execute concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, dict) and result.get("accessible"):
                    discovered_endpoints.append(result["url"])

                    # Report interesting endpoints
                    if any(keyword in result["url"] for keyword in ["/admin", "/debug", "/.git", "/.env"]):
                        findings.append(Finding(
                            title=f"Sensitive Endpoint Exposed: {result['url']}",
                            description=f"Potentially sensitive endpoint is publicly accessible",
                            severity=FindingSeverity.HIGH,
                            location=result["url"],
                            evidence={"status_code": result["status_code"]}
                        ))
                    elif "/api" in result["url"] or "/swagger" in result["url"]:
                        findings.append(Finding(
                            title=f"API Endpoint Discovered: {result['url']}",
                            description=f"API endpoint found during reconnaissance",
                            severity=FindingSeverity.INFO,
                            location=result["url"],
                            evidence={"status_code": result["status_code"]}
                        ))

        return findings

    async def _probe_endpoint(
        self,
        client: Any,
        url: str
    ) -> Dict[str, Any]:
        """Probe a single endpoint"""
        try:
            response = await client.get(
                url,
                headers={"User-Agent": self.user_agent},
                follow_redirects=False
            )

            return {
                "url": url,
                "accessible": response.status_code < 500,
                "status_code": response.status_code,
                "content_length": len(response.content)
            }
        except Exception:
            return {
                "url": url,
                "accessible": False,
                "status_code": None
            }

    async def _detect_technologies(
        self,
        target: str,
        blackboard: Any
    ) -> List[Finding]:
        """Detect technologies used by target"""
        findings = []
        detected_techs = []

        if not httpx:
            return findings

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(
                    target,
                    headers={"User-Agent": self.user_agent}
                )

                # Check headers
                headers_str = str(response.headers).lower()
                response_text = response.text.lower()

                for tech, signatures in self.tech_signatures.items():
                    for signature in signatures:
                        if signature.lower() in headers_str or signature.lower() in response_text:
                            detected_techs.append(tech)
                            break

                if detected_techs:
                    findings.append(Finding(
                        title="Technologies Detected",
                        description=f"Identified technologies: {', '.join(detected_techs)}",
                        severity=FindingSeverity.INFO,
                        location=target,
                        evidence={"technologies": detected_techs}
                    ))

        except Exception as e:
            logger.error("Technology detection error", error=str(e))

        return findings

    async def _analyze_security_headers(
        self,
        target: str,
        blackboard: Any
    ) -> List[Finding]:
        """Analyze security headers"""
        findings = []

        if not httpx:
            return findings

        # Critical security headers
        security_headers = {
            "Strict-Transport-Security": "HSTS not enabled",
            "X-Frame-Options": "Clickjacking protection missing",
            "X-Content-Type-Options": "MIME-sniffing protection missing",
            "Content-Security-Policy": "CSP not implemented",
            "X-XSS-Protection": "XSS protection header missing"
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(
                    target,
                    headers={"User-Agent": self.user_agent}
                )

                # Check for missing security headers
                missing_headers = []
                for header, description in security_headers.items():
                    if header not in response.headers:
                        missing_headers.append(header)

                if missing_headers:
                    findings.append(Finding(
                        title="Missing Security Headers",
                        description=f"Target is missing {len(missing_headers)} security headers",
                        severity=FindingSeverity.MEDIUM,
                        location=target,
                        evidence={
                            "missing_headers": missing_headers,
                            "recommendations": [
                                f"Add {h} header" for h in missing_headers
                            ]
                        }
                    ))

                # Check for information disclosure headers
                disclosure_headers = [
                    "Server",
                    "X-Powered-By",
                    "X-AspNet-Version",
                    "X-AspNetMvc-Version"
                ]

                disclosed_info = {}
                for header in disclosure_headers:
                    if header in response.headers:
                        disclosed_info[header] = response.headers[header]

                if disclosed_info:
                    findings.append(Finding(
                        title="Information Disclosure in Headers",
                        description="Server headers reveal technology stack information",
                        severity=FindingSeverity.LOW,
                        location=target,
                        evidence={"disclosed_headers": disclosed_info}
                    ))

        except Exception as e:
            logger.error("Security header analysis error", error=str(e))

        return findings

    async def _share_recon_intelligence(
        self,
        blackboard: Any,
        findings: List[Finding]
    ):
        """Share reconnaissance intelligence with swarm"""
        # Extract discovered endpoints
        endpoints = [
            f.location for f in findings
            if "Endpoint Discovered" in f.title or "Endpoint Exposed" in f.title
        ]

        if endpoints:
            await self.share_intelligence(
                blackboard,
                "endpoints",
                {"endpoints": endpoints, "count": len(endpoints)}
            )

        # Extract detected technologies
        tech_findings = [f for f in findings if "Technologies Detected" in f.title]
        if tech_findings:
            technologies = tech_findings[0].evidence.get("technologies", [])
            await self.share_intelligence(
                blackboard,
                "technologies",
                {"technologies": technologies}
            )

        # Share security posture
        header_findings = [f for f in findings if "Security Headers" in f.title]
        if header_findings:
            await self.share_intelligence(
                blackboard,
                "security_posture",
                {
                    "missing_headers": header_findings[0].evidence.get("missing_headers", []),
                    "severity": "medium" if len(header_findings[0].evidence.get("missing_headers", [])) > 3 else "low"
                }
            )
