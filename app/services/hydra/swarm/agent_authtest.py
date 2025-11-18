"""
HYDRA SWARM Authentication Test Agent
Phase 7: Authentication and authorization security testing

The auth test agent performs:
- Weak credential testing
- JWT token analysis and manipulation
- Session management testing
- OAuth flow analysis
- Access control bypasses
- Privilege escalation testing
"""

from typing import Dict, List, Optional, Any
import asyncio
import json
import base64
import hashlib
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
import structlog

try:
    import httpx
except ImportError:
    httpx = None

try:
    import jwt as pyjwt
except ImportError:
    pyjwt = None

from .agent_base import (
    BaseAgent,
    AgentStatus,
    AgentCapability,
    Finding,
    FindingSeverity
)

logger = structlog.get_logger()


class AuthTestAgent(BaseAgent):
    """
    Authentication Test Agent for HYDRA SWARM.

    Specializes in:
    - Weak credential testing (common passwords)
    - JWT token security analysis
    - Session management vulnerabilities
    - OAuth/OIDC flow testing
    - Access control bypass attempts
    - Privilege escalation vectors

    Example:
        agent = AuthTestAgent()
        findings = await agent.execute(
            target="https://api.example.com",
            blackboard=blackboard,
            mission_params={"test_credentials": True}
        )
    """

    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(
            agent_id=agent_id,
            agent_type="AuthTestAgent",
            capabilities=[
                AgentCapability.AUTH_TESTING,
                AgentCapability.SESSION_HIJACKING,
                AgentCapability.PRIVILEGE_ESCALATION
            ]
        )

        # Configuration
        self.timeout_seconds = 10
        self.user_agent = "MEMSHADOW-AuthTest/1.0"

        # Common weak credentials to test (PRODUCTION: Use with extreme caution!)
        self.weak_credentials = [
            ("admin", "admin"),
            ("admin", "password"),
            ("admin", "admin123"),
            ("test", "test"),
            ("user", "password"),
            ("demo", "demo")
        ]

        # Common authentication endpoints
        self.auth_endpoints = [
            "/login",
            "/signin",
            "/auth",
            "/authenticate",
            "/api/auth",
            "/api/login",
            "/oauth/token",
            "/token"
        ]

        # JWT algorithm security issues
        self.jwt_algorithms_to_test = ["none", "HS256", "RS256"]

    async def execute(
        self,
        target: str,
        blackboard: Any,
        mission_params: Optional[Dict] = None
    ) -> List[Finding]:
        """
        Execute authentication testing on target.

        Args:
            target: Target URL
            blackboard: Shared blackboard for intelligence
            mission_params: Mission-specific parameters

        Returns:
            List of findings
        """
        logger.info("Auth test agent starting", agent_id=self.agent_id, target=target)
        self.status = AgentStatus.TESTING

        params = mission_params or {}
        test_credentials = params.get("test_credentials", False)
        test_jwt = params.get("test_jwt", True)

        findings: List[Finding] = []

        # Send heartbeat
        await self._heartbeat(blackboard)

        # Phase 1: Discover authentication endpoints
        auth_findings = await self._discover_auth_endpoints(target, blackboard)
        findings.extend(auth_findings)

        # Phase 2: Test for weak credentials (ONLY in test environments!)
        if test_credentials:
            cred_findings = await self._test_weak_credentials(target, blackboard)
            findings.extend(cred_findings)

        # Phase 3: JWT token analysis
        if test_jwt:
            jwt_findings = await self._analyze_jwt_security(target, blackboard)
            findings.extend(jwt_findings)

        # Phase 4: Session management testing
        session_findings = await self._test_session_management(target, blackboard)
        findings.extend(session_findings)

        # Phase 5: OAuth/OIDC analysis
        oauth_findings = await self._analyze_oauth(target, blackboard)
        findings.extend(oauth_findings)

        # Share auth intelligence
        await self._share_auth_intelligence(blackboard, findings)

        # Report findings
        for finding in findings:
            await self.report_finding(blackboard, finding)

        self.status = AgentStatus.IDLE
        logger.info(
            "Auth test agent completed",
            agent_id=self.agent_id,
            target=target,
            findings=len(findings)
        )

        return findings

    async def _discover_auth_endpoints(
        self,
        target: str,
        blackboard: Any
    ) -> List[Finding]:
        """Discover authentication endpoints"""
        findings = []

        if not httpx:
            return findings

        parsed = urlparse(target)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # Get shared endpoint intelligence
        api_endpoints = await blackboard.get("intel:api_endpoints") or {}
        discovered = api_endpoints.get("endpoints", {})

        # Check discovered endpoints for auth-related paths
        auth_paths = [
            path for path in discovered.keys()
            if any(keyword in path.lower() for keyword in ["auth", "login", "token", "oauth"])
        ]

        if auth_paths:
            findings.append(Finding(
                title="Authentication Endpoints Discovered",
                description=f"Found {len(auth_paths)} authentication-related endpoints",
                severity=FindingSeverity.INFO,
                location=base_url,
                evidence={"endpoints": auth_paths}
            ))

        # Probe common auth endpoints not yet discovered
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for endpoint in self.auth_endpoints:
                url = urljoin(base_url, endpoint)

                try:
                    response = await client.post(
                        url,
                        headers={"User-Agent": self.user_agent},
                        json={}  # Empty payload
                    )

                    # Check if endpoint exists and accepts authentication
                    if response.status_code in [200, 400, 401, 422]:
                        findings.append(Finding(
                            title=f"Authentication Endpoint Found: {endpoint}",
                            description="Endpoint appears to accept authentication requests",
                            severity=FindingSeverity.INFO,
                            location=url,
                            evidence={"status_code": response.status_code}
                        ))

                except Exception:
                    pass

        return findings

    async def _test_weak_credentials(
        self,
        target: str,
        blackboard: Any
    ) -> List[Finding]:
        """
        Test for weak/default credentials.

        WARNING: Only use in authorized test environments!
        """
        findings = []

        if not httpx:
            return findings

        # Get auth endpoints from previous phase
        parsed = urlparse(target)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # Test login endpoints
        login_endpoints = [urljoin(base_url, ep) for ep in ["/api/login", "/login"]]

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for url in login_endpoints[:1]:  # Limit to 1 to avoid lockout
                for username, password in self.weak_credentials[:3]:  # Test only first 3
                    try:
                        # Attempt login
                        response = await client.post(
                            url,
                            headers={"User-Agent": self.user_agent},
                            json={"username": username, "password": password}
                        )

                        # Check for successful authentication indicators
                        if response.status_code == 200:
                            response_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}

                            # Look for tokens or session indicators
                            if any(key in response_data for key in ["token", "access_token", "session"]):
                                findings.append(Finding(
                                    title="Weak Credentials Accepted",
                                    description=f"System accepts weak credentials: {username}:{password}",
                                    severity=FindingSeverity.CRITICAL,
                                    location=url,
                                    evidence={
                                        "username": username,
                                        "password_length": len(password),
                                        "response_keys": list(response_data.keys())
                                    }
                                ))
                                break  # Stop testing once we find weak creds

                        # Add delay to avoid rate limiting/lockout
                        await asyncio.sleep(1)

                    except Exception as e:
                        logger.debug("Credential test error", error=str(e))

        return findings

    async def _analyze_jwt_security(
        self,
        target: str,
        blackboard: Any
    ) -> List[Finding]:
        """Analyze JWT token security"""
        findings = []

        if not httpx:
            return findings

        # Try to obtain a JWT token
        token = await self._attempt_get_jwt(target)

        if token:
            # Analyze the token
            jwt_issues = await self._analyze_jwt_token(token, target)
            findings.extend(jwt_issues)

        # Test for algorithm confusion (none algorithm)
        none_findings = await self._test_jwt_none_algorithm(target)
        findings.extend(none_findings)

        return findings

    async def _attempt_get_jwt(self, target: str) -> Optional[str]:
        """Attempt to obtain a JWT token from the target"""
        if not httpx:
            return None

        parsed = urlparse(target)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # Try common paths that might return JWT
        token_endpoints = [
            "/api/token",
            "/oauth/token",
            "/auth/token"
        ]

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for endpoint in token_endpoints:
                try:
                    url = urljoin(base_url, endpoint)
                    response = await client.post(
                        url,
                        headers={"User-Agent": self.user_agent},
                        json={"username": "test", "password": "test"}
                    )

                    if response.status_code == 200:
                        data = response.json()
                        token = data.get("access_token") or data.get("token")

                        if token and token.count(".") == 2:  # JWT format
                            return token

                except Exception:
                    pass

        return None

    async def _analyze_jwt_token(
        self,
        token: str,
        target: str
    ) -> List[Finding]:
        """Analyze JWT token for security issues"""
        findings = []

        try:
            # Decode without verification to inspect claims
            parts = token.split(".")
            if len(parts) != 3:
                return findings

            # Decode header
            header = json.loads(base64.urlsafe_b64decode(parts[0] + "=="))
            payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=="))

            # Check algorithm
            algorithm = header.get("alg")

            if algorithm == "none":
                findings.append(Finding(
                    title="JWT Using 'none' Algorithm",
                    description="JWT token uses 'none' algorithm, allowing unsigned tokens",
                    severity=FindingSeverity.CRITICAL,
                    location=target,
                    evidence={"algorithm": algorithm}
                ))

            # Check for missing expiration
            if "exp" not in payload:
                findings.append(Finding(
                    title="JWT Missing Expiration",
                    description="JWT token does not have an expiration claim (exp)",
                    severity=FindingSeverity.HIGH,
                    location=target,
                    evidence={"payload_keys": list(payload.keys())}
                ))

            # Check for long expiration
            if "exp" in payload:
                exp_timestamp = payload["exp"]
                exp_datetime = datetime.fromtimestamp(exp_timestamp)
                now = datetime.utcnow()

                if (exp_datetime - now) > timedelta(days=7):
                    findings.append(Finding(
                        title="JWT Long Expiration Time",
                        description=f"JWT token has very long expiration ({(exp_datetime - now).days} days)",
                        severity=FindingSeverity.MEDIUM,
                        location=target,
                        evidence={"expiration_days": (exp_datetime - now).days}
                    ))

            # Check for sensitive data in payload
            sensitive_keys = ["password", "secret", "key", "private"]
            found_sensitive = [k for k in payload.keys() if any(s in k.lower() for s in sensitive_keys)]

            if found_sensitive:
                findings.append(Finding(
                    title="Sensitive Data in JWT Payload",
                    description=f"JWT contains potentially sensitive keys: {', '.join(found_sensitive)}",
                    severity=FindingSeverity.MEDIUM,
                    location=target,
                    evidence={"sensitive_keys": found_sensitive}
                ))

        except Exception as e:
            logger.error("JWT analysis error", error=str(e))

        return findings

    async def _test_jwt_none_algorithm(self, target: str) -> List[Finding]:
        """Test if API accepts JWT with 'none' algorithm"""
        findings = []

        # This is a theoretical test - would need actual API interaction
        # In production, this would attempt to use a crafted JWT with alg: none

        return findings

    async def _test_session_management(
        self,
        target: str,
        blackboard: Any
    ) -> List[Finding]:
        """Test session management security"""
        findings = []

        if not httpx:
            return findings

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            try:
                # Make initial request
                response = await client.get(
                    target,
                    headers={"User-Agent": self.user_agent}
                )

                # Check for session cookies
                cookies = response.cookies

                for cookie_name, cookie_value in cookies.items():
                    # Check for secure flag
                    cookie_obj = response.cookies.get(cookie_name)

                    if cookie_obj and not cookie_obj.secure and "https" in target:
                        findings.append(Finding(
                            title=f"Session Cookie Missing Secure Flag: {cookie_name}",
                            description="Session cookie transmitted over HTTPS without Secure flag",
                            severity=FindingSeverity.MEDIUM,
                            location=target,
                            evidence={"cookie_name": cookie_name}
                        ))

                    # Check for HttpOnly flag
                    # Note: httpx doesn't expose this, would need raw headers
                    set_cookie_header = response.headers.get("set-cookie", "").lower()
                    if cookie_name.lower() in set_cookie_header and "httponly" not in set_cookie_header:
                        findings.append(Finding(
                            title=f"Session Cookie Missing HttpOnly Flag: {cookie_name}",
                            description="Session cookie is accessible via JavaScript (XSS risk)",
                            severity=FindingSeverity.HIGH,
                            location=target,
                            evidence={"cookie_name": cookie_name}
                        ))

            except Exception as e:
                logger.error("Session management test error", error=str(e))

        return findings

    async def _analyze_oauth(
        self,
        target: str,
        blackboard: Any
    ) -> List[Finding]:
        """Analyze OAuth/OIDC implementation"""
        findings = []

        if not httpx:
            return findings

        parsed = urlparse(target)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # Check for OAuth discovery endpoint
        discovery_urls = [
            "/.well-known/oauth-authorization-server",
            "/.well-known/openid-configuration"
        ]

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for discovery_path in discovery_urls:
                try:
                    url = urljoin(base_url, discovery_path)
                    response = await client.get(
                        url,
                        headers={"User-Agent": self.user_agent}
                    )

                    if response.status_code == 200:
                        config = response.json()

                        findings.append(Finding(
                            title="OAuth/OIDC Discovery Endpoint Found",
                            description=f"OAuth discovery metadata is publicly accessible",
                            severity=FindingSeverity.INFO,
                            location=url,
                            evidence={
                                "issuer": config.get("issuer"),
                                "authorization_endpoint": config.get("authorization_endpoint"),
                                "token_endpoint": config.get("token_endpoint")
                            }
                        ))

                        # Check for insecure configuration
                        supported_grants = config.get("grant_types_supported", [])
                        if "implicit" in supported_grants:
                            findings.append(Finding(
                                title="OAuth Implicit Grant Enabled",
                                description="Implicit grant flow is enabled (deprecated for security reasons)",
                                severity=FindingSeverity.MEDIUM,
                                location=url,
                                evidence={"grant_types": supported_grants}
                            ))

                except Exception:
                    pass

        return findings

    async def _share_auth_intelligence(
        self,
        blackboard: Any,
        findings: List[Finding]
    ):
        """Share authentication intelligence with swarm"""
        # Extract auth endpoints
        auth_endpoints = [
            f.location for f in findings
            if "Authentication Endpoint" in f.title
        ]

        if auth_endpoints:
            await self.share_intelligence(
                blackboard,
                "auth_endpoints",
                {"endpoints": auth_endpoints, "count": len(auth_endpoints)}
            )

        # Share critical vulnerabilities
        critical_findings = [
            f for f in findings
            if f.severity == FindingSeverity.CRITICAL
        ]

        if critical_findings:
            await self.share_intelligence(
                blackboard,
                "critical_auth_issues",
                {
                    "count": len(critical_findings),
                    "issues": [f.title for f in critical_findings]
                }
            )
