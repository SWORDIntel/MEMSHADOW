"""
Security Middleware for MEMSHADOW Web API
Additional security layers beyond authentication
"""

from typing import Callable, Optional
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import structlog
import re

from app.web.rate_limiter import get_rate_limiter

logger = structlog.get_logger()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses.

    Headers added:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block
    - Strict-Transport-Security: max-age=31536000
    - Content-Security-Policy: default-src 'self'
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using token bucket algorithm.
    """

    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self.rate_limiter = get_rate_limiter()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)

        # Get client identifier (IP address)
        client_ip = request.client.host if request.client else "unknown"

        # Check rate limit
        allowed, reason = await self.rate_limiter.check_rate_limit(
            identifier=client_ip,
            endpoint=request.url.path
        )

        if not allowed:
            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                path=request.url.path,
                method=request.method
            )

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": reason
                },
                headers={"Retry-After": "60"}
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        stats = await self.rate_limiter.get_stats()
        response.headers["X-RateLimit-Limit"] = "60"
        response.headers["X-RateLimit-Remaining"] = str(max(0, 60 - stats.get("active_identifiers", 0)))

        return response


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Validate requests for suspicious patterns and potential attacks.
    """

    # Suspicious patterns (SQL injection, XSS, path traversal)
    SUSPICIOUS_PATTERNS = [
        r"(\.\./|\.\.\\)",  # Path traversal
        r"(<script|<iframe|javascript:)",  # XSS
        r"(union\s+select|insert\s+into|delete\s+from)",  # SQL injection
        r"(exec\(|eval\(|__import__)",  # Code injection
        r"(\$\{|<%=)",  # Template injection
    ]

    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.SUSPICIOUS_PATTERNS]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)

        # Check URL path
        if self._is_suspicious(request.url.path):
            logger.warning(
                "Suspicious request pattern detected in URL",
                path=request.url.path,
                client_ip=request.client.host if request.client else "unknown"
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Invalid request"}
            )

        # Check query parameters
        for key, value in request.query_params.items():
            if self._is_suspicious(value):
                logger.warning(
                    "Suspicious request pattern detected in query",
                    param=key,
                    client_ip=request.client.host if request.client else "unknown"
                )
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"error": "Invalid request parameters"}
                )

        return await call_next(request)

    def _is_suspicious(self, text: str) -> bool:
        """Check if text contains suspicious patterns"""
        for pattern in self.patterns:
            if pattern.search(text):
                return True
        return False


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Audit logging for sensitive operations.
    """

    # Sensitive endpoints to log
    SENSITIVE_ENDPOINTS = [
        "/api/auth/login",
        "/api/auth/logout",
        "/api/self-modifying/improve",
        "/api/self-modifying/start",
        "/api/config",
    ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if endpoint is sensitive
        is_sensitive = any(
            request.url.path.startswith(endpoint)
            for endpoint in self.SENSITIVE_ENDPOINTS
        )

        if is_sensitive:
            client_ip = request.client.host if request.client else "unknown"

            logger.info(
                "Sensitive operation",
                method=request.method,
                path=request.url.path,
                client_ip=client_ip,
                user_agent=request.headers.get("user-agent", "unknown")
            )

        # Process request
        response = await call_next(request)

        # Log response status for sensitive endpoints
        if is_sensitive:
            logger.info(
                "Sensitive operation completed",
                path=request.url.path,
                status_code=response.status_code
            )

        return response
