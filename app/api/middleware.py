import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from app.db.redis import redis_client

logger = structlog.get_logger()

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())

        # Add to request state
        request.state.request_id = request_id

        # Add to logger context
        structlog.contextvars.bind_contextvars(request_id=request_id)

        # Process request
        response = await call_next(request)

        # Add to response headers
        response.headers["X-Request-ID"] = request_id

        # Clear context
        structlog.contextvars.clear_contextvars()

        return response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware using Redis"""

    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host
        key = f"rate-limit:{client_ip}"

        if not await redis_client.check_rate_limit(key, self.calls, self.period):
            logger.warning("Rate limit exceeded", client_ip=client_ip)
            return Response(content="Rate limit exceeded", status_code=429)

        return await call_next(request)