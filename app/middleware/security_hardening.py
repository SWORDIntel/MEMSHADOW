"""
APT-Grade API Security Hardening
Classification: UNCLASSIFIED
Defense against Advanced Persistent Threats
"""

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import structlog
import time
import hashlib
import hmac
import secrets
from typing import Optional, Dict, Set
from collections import defaultdict
from datetime import datetime, timedelta
import re
import ipaddress

logger = structlog.get_logger()

# ============================================================================
# IP Blacklist/Whitelist Management
# ============================================================================

class IPAccessControl:
    """IP-based access control with reputation tracking"""

    def __init__(self):
        self.blacklist: Set[str] = set()
        self.whitelist: Set[str] = set()
        self.reputation: Dict[str, int] = defaultdict(int)  # Higher = worse
        self.last_seen: Dict[str, datetime] = {}

        # Known bad IP ranges (example - expand with threat intel)
        self.bad_ranges = [
            ipaddress.IPv4Network('10.0.0.0/8'),  # Private - shouldn't reach prod
            ipaddress.IPv4Network('172.16.0.0/12'),
            ipaddress.IPv4Network('192.168.0.0/16'),
        ]

    def is_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        if ip in self.whitelist:
            return False
        if ip in self.blacklist:
            return True
        if self.reputation.get(ip, 0) > 100:  # Auto-block high reputation score
            self.blacklist.add(ip)
            logger.warning("IP auto-blocked due to reputation", ip=ip, score=self.reputation[ip])
            return True
        return False

    def record_suspicious_activity(self, ip: str, severity: int = 10):
        """Record suspicious activity from IP"""
        self.reputation[ip] += severity
        self.last_seen[ip] = datetime.utcnow()

    def is_private_ip(self, ip: str) -> bool:
        """Check if IP is from private range"""
        try:
            ip_obj = ipaddress.IPv4Address(ip)
            return any(ip_obj in network for network in self.bad_ranges)
        except:
            return False


# ============================================================================
# Request Signature Verification (HMAC)
# ============================================================================

class RequestSignatureValidator:
    """Verify HMAC signatures on critical requests"""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode()

    def generate_signature(self, method: str, path: str, body: bytes, timestamp: str) -> str:
        """Generate HMAC signature for request"""
        message = f"{method}:{path}:{timestamp}".encode() + body
        signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()
        return signature

    def verify_signature(self, request_signature: str, method: str, path: str,
                        body: bytes, timestamp: str) -> bool:
        """Verify request signature"""
        expected = self.generate_signature(method, path, body, timestamp)
        return hmac.compare_digest(request_signature, expected)


# ============================================================================
# Rate Limiting (Token Bucket with Per-Endpoint Limits)
# ============================================================================

class AdvancedRateLimiter:
    """Advanced rate limiting with per-endpoint and per-IP tracking"""

    def __init__(self):
        # Per-IP rate limits
        self.ip_buckets: Dict[str, Dict] = {}

        # Per-endpoint limits (requests per minute)
        self.endpoint_limits = {
            '/api/v1/auth/login': 5,           # Strict: prevent brute force
            '/api/v1/c2/register': 10,         # Moderate: legitimate C2 traffic
            '/api/v1/tempest/dashboard': 60,   # Generous: dashboard polling
            '/api/v1/metrics': 120,            # Very generous: monitoring
            'default': 100,                    # Default limit
        }

        # Burst allowance
        self.burst_multiplier = 2

    def _get_bucket(self, ip: str, endpoint: str) -> Dict:
        """Get or create rate limit bucket for IP+endpoint"""
        key = f"{ip}:{endpoint}"
        if key not in self.ip_buckets:
            limit = self.endpoint_limits.get(endpoint, self.endpoint_limits['default'])
            self.ip_buckets[key] = {
                'tokens': limit * self.burst_multiplier,
                'max_tokens': limit * self.burst_multiplier,
                'refill_rate': limit / 60.0,  # tokens per second
                'last_refill': time.time(),
            }
        return self.ip_buckets[key]

    def is_allowed(self, ip: str, endpoint: str) -> tuple[bool, Optional[int]]:
        """Check if request is allowed. Returns (allowed, retry_after_seconds)"""
        bucket = self._get_bucket(ip, endpoint)

        # Refill tokens
        now = time.time()
        time_passed = now - bucket['last_refill']
        bucket['tokens'] = min(
            bucket['max_tokens'],
            bucket['tokens'] + (time_passed * bucket['refill_rate'])
        )
        bucket['last_refill'] = now

        # Check if request allowed
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True, None
        else:
            # Calculate retry-after
            tokens_needed = 1 - bucket['tokens']
            retry_after = int(tokens_needed / bucket['refill_rate']) + 1
            return False, retry_after


# ============================================================================
# SQL Injection Detection
# ============================================================================

class SQLInjectionDetector:
    """Detect SQL injection attempts in request parameters"""

    # SQL injection patterns
    SQL_PATTERNS = [
        r"(\bunion\b.*\bselect\b)",
        r"(\bselect\b.*\bfrom\b)",
        r"(\binsert\b.*\binto\b)",
        r"(\bdelete\b.*\bfrom\b)",
        r"(\bdrop\b.*\btable\b)",
        r"(--|\#|\/\*)",  # SQL comments
        r"(\bor\b\s+\d+\s*=\s*\d+)",  # OR 1=1
        r"(\band\b\s+\d+\s*=\s*\d+)",
        r"('.*or.*'.*=.*')",
        r"(exec\s*\()",
        r"(execute\s*\()",
        r"(xp_cmdshell)",
    ]

    def __init__(self):
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.SQL_PATTERNS]

    def detect(self, value: str) -> bool:
        """Detect SQL injection attempt"""
        return any(pattern.search(value) for pattern in self.compiled_patterns)


# ============================================================================
# XSS Detection
# ============================================================================

class XSSDetector:
    """Detect Cross-Site Scripting attempts"""

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",  # onerror=, onclick=, etc.
        r"<iframe",
        r"<object",
        r"<embed",
        r"<img[^>]+src",
        r"eval\s*\(",
        r"alert\s*\(",
        r"document\.cookie",
        r"document\.write",
    ]

    def __init__(self):
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.XSS_PATTERNS]

    def detect(self, value: str) -> bool:
        """Detect XSS attempt"""
        return any(pattern.search(value) for pattern in self.compiled_patterns)


# ============================================================================
# Path Traversal Detection
# ============================================================================

class PathTraversalDetector:
    """Detect path traversal attempts"""

    TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.",
        r"%2e%2e",
        r"\.\.%2f",
        r"%252e%252e",
        r"..\\",
        r"%00",  # Null byte
    ]

    def __init__(self):
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.TRAVERSAL_PATTERNS]

    def detect(self, value: str) -> bool:
        """Detect path traversal attempt"""
        return any(pattern.search(value) for pattern in self.compiled_patterns)


# ============================================================================
# Security Hardening Middleware
# ============================================================================

class APTSecurityMiddleware(BaseHTTPMiddleware):
    """APT-grade security middleware for all API requests"""

    def __init__(self, app: ASGIApp, secret_key: str):
        super().__init__(app)
        self.secret_key = secret_key

        # Security components
        self.ip_control = IPAccessControl()
        self.rate_limiter = AdvancedRateLimiter()
        self.signature_validator = RequestSignatureValidator(secret_key)
        self.sqli_detector = SQLInjectionDetector()
        self.xss_detector = XSSDetector()
        self.path_detector = PathTraversalDetector()

        # Attack tracking
        self.attack_attempts: Dict[str, int] = defaultdict(int)

    async def dispatch(self, request: Request, call_next):
        """Process request with security checks"""

        # Get client IP
        client_ip = request.client.host

        # ----------------------------------------------------------------
        # Check 1: IP Access Control
        # ----------------------------------------------------------------
        if self.ip_control.is_blocked(client_ip):
            logger.warning("Blocked request from blacklisted IP", ip=client_ip)
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Access denied"},
                headers={"X-Security-Block": "IP-BLACKLIST"}
            )

        # Check for private IPs in production
        if self.ip_control.is_private_ip(client_ip):
            logger.warning("Request from private IP range", ip=client_ip)
            self.ip_control.record_suspicious_activity(client_ip, severity=5)

        # ----------------------------------------------------------------
        # Check 2: Rate Limiting
        # ----------------------------------------------------------------
        endpoint = request.url.path
        allowed, retry_after = self.rate_limiter.is_allowed(client_ip, endpoint)

        if not allowed:
            logger.warning("Rate limit exceeded", ip=client_ip, endpoint=endpoint)
            self.ip_control.record_suspicious_activity(client_ip, severity=2)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"},
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Remaining": "0"
                }
            )

        # ----------------------------------------------------------------
        # Check 3: Request Signature (for critical endpoints)
        # ----------------------------------------------------------------
        critical_endpoints = ['/api/v1/c2/', '/api/v1/tempest/']

        if any(endpoint.startswith(ce) for ce in critical_endpoints):
            signature = request.headers.get('X-Request-Signature')
            timestamp = request.headers.get('X-Request-Timestamp')

            if not signature or not timestamp:
                logger.warning("Missing signature on critical endpoint",
                             ip=client_ip, endpoint=endpoint)
                self.ip_control.record_suspicious_activity(client_ip, severity=20)
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Request signature required"},
                    headers={"X-Security-Block": "MISSING-SIGNATURE"}
                )

            # Verify timestamp (prevent replay attacks)
            try:
                ts = float(timestamp)
                if abs(time.time() - ts) > 300:  # 5 minute window
                    logger.warning("Request timestamp out of window",
                                 ip=client_ip, endpoint=endpoint)
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"detail": "Request timestamp invalid"}
                    )
            except ValueError:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Invalid timestamp format"}
                )

        # ----------------------------------------------------------------
        # Check 4: Attack Pattern Detection
        # ----------------------------------------------------------------

        # Check query parameters
        for param, value in request.query_params.items():
            value_str = str(value)

            # SQL Injection
            if self.sqli_detector.detect(value_str):
                logger.critical("SQL injection attempt detected",
                              ip=client_ip, endpoint=endpoint, param=param)
                self.ip_control.record_suspicious_activity(client_ip, severity=50)
                self.attack_attempts[client_ip] += 1
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "Malicious request detected"},
                    headers={"X-Security-Block": "SQLI-DETECTED"}
                )

            # XSS
            if self.xss_detector.detect(value_str):
                logger.critical("XSS attempt detected",
                              ip=client_ip, endpoint=endpoint, param=param)
                self.ip_control.record_suspicious_activity(client_ip, severity=40)
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "Malicious request detected"},
                    headers={"X-Security-Block": "XSS-DETECTED"}
                )

            # Path Traversal
            if self.path_detector.detect(value_str):
                logger.critical("Path traversal attempt detected",
                              ip=client_ip, endpoint=endpoint, param=param)
                self.ip_control.record_suspicious_activity(client_ip, severity=45)
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "Malicious request detected"},
                    headers={"X-Security-Block": "PATH-TRAVERSAL"}
                )

        # Check path itself
        if self.path_detector.detect(endpoint):
            logger.critical("Path traversal in URL", ip=client_ip, endpoint=endpoint)
            self.ip_control.record_suspicious_activity(client_ip, severity=50)
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Malicious request detected"},
                headers={"X-Security-Block": "PATH-TRAVERSAL-URL"}
            )

        # ----------------------------------------------------------------
        # Check 5: Header Security
        # ----------------------------------------------------------------

        # Ensure User-Agent is present (block bots without UA)
        if not request.headers.get('user-agent'):
            logger.warning("Request without User-Agent", ip=client_ip)
            self.ip_control.record_suspicious_activity(client_ip, severity=5)

        # Check for common attack tools in User-Agent
        ua = request.headers.get('user-agent', '').lower()
        attack_tools = ['sqlmap', 'nmap', 'nikto', 'burp', 'metasploit', 'nessus']
        if any(tool in ua for tool in attack_tools):
            logger.critical("Attack tool detected in User-Agent",
                          ip=client_ip, ua=ua)
            self.ip_control.record_suspicious_activity(client_ip, severity=100)
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Access denied"},
                headers={"X-Security-Block": "ATTACK-TOOL-DETECTED"}
            )

        # ----------------------------------------------------------------
        # Process Request
        # ----------------------------------------------------------------

        # Add security headers to request context
        request.state.client_ip = client_ip
        request.state.security_score = self.ip_control.reputation.get(client_ip, 0)

        # Call next middleware/endpoint
        response = await call_next(request)

        # Add security headers to response
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Remove server information disclosure
        response.headers.pop("Server", None)
        response.headers.pop("X-Powered-By", None)

        return response
