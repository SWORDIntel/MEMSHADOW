"""
Rate Limiter for MEMSHADOW Web API
Prevents abuse and DoS attacks with configurable rate limiting
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import structlog

logger = structlog.get_logger()


class RateLimiter:
    """
    Token bucket rate limiter.

    Features:
    - Per-IP rate limiting
    - Per-user rate limiting
    - Configurable rates for different endpoints
    - Automatic cleanup of old entries
    """

    def __init__(
        self,
        default_requests_per_minute: int = 60,
        default_burst: int = 10,
        cleanup_interval_seconds: int = 300
    ):
        """
        Initialize rate limiter.

        Args:
            default_requests_per_minute: Default rate limit
            default_burst: Maximum burst size
            cleanup_interval_seconds: How often to clean up old entries
        """
        self.default_rpm = default_requests_per_minute
        self.default_burst = default_burst
        self.cleanup_interval = cleanup_interval_seconds

        # Token buckets: {identifier: (tokens, last_update)}
        self.buckets: Dict[str, tuple[float, datetime]] = {}

        # Per-endpoint limits
        self.endpoint_limits: Dict[str, tuple[int, int]] = {
            "/api/auth/login": (5, 3),  # 5 per minute, burst of 3 (brute force protection)
            "/api/self-modifying/improve": (10, 2),  # Conservative for expensive operations
            "/api/federated/update": (30, 5),  # Higher for federation
        }

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Statistics
        self.total_requests = 0
        self.blocked_requests = 0

        logger.info(
            "Rate limiter initialized",
            default_rpm=default_requests_per_minute,
            burst=default_burst
        )

    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Check if request is within rate limit.

        Args:
            identifier: IP address or user ID
            endpoint: API endpoint path

        Returns:
            (allowed, reason) - True if allowed, False if rate limited
        """
        async with self._lock:
            self.total_requests += 1

            # Get limits for this endpoint
            if endpoint and endpoint in self.endpoint_limits:
                rpm, burst = self.endpoint_limits[endpoint]
            else:
                rpm, burst = self.default_rpm, self.default_burst

            # Calculate tokens per second
            tokens_per_second = rpm / 60.0

            # Get current bucket state
            now = datetime.utcnow()

            if identifier not in self.buckets:
                # New identifier - initialize bucket
                self.buckets[identifier] = (burst - 1, now)

                logger.debug(
                    "Rate limit: new identifier",
                    identifier=identifier,
                    tokens_remaining=burst - 1
                )
                return True, None

            # Get current tokens and last update time
            current_tokens, last_update = self.buckets[identifier]

            # Calculate time elapsed and tokens to add
            elapsed = (now - last_update).total_seconds()
            new_tokens = min(burst, current_tokens + (elapsed * tokens_per_second))

            # Check if we have tokens available
            if new_tokens >= 1.0:
                # Allow request - consume 1 token
                self.buckets[identifier] = (new_tokens - 1.0, now)

                logger.debug(
                    "Rate limit: request allowed",
                    identifier=identifier,
                    tokens_remaining=new_tokens - 1.0,
                    endpoint=endpoint
                )
                return True, None
            else:
                # Rate limit exceeded
                self.blocked_requests += 1

                # Calculate retry after
                tokens_needed = 1.0 - new_tokens
                retry_after_seconds = int(tokens_needed / tokens_per_second) + 1

                logger.warning(
                    "Rate limit exceeded",
                    identifier=identifier,
                    endpoint=endpoint,
                    retry_after=retry_after_seconds
                )

                return False, f"Rate limit exceeded. Retry after {retry_after_seconds} seconds."

    async def cleanup_old_entries(self, max_age_minutes: int = 60):
        """
        Clean up old bucket entries to prevent memory leak.

        Args:
            max_age_minutes: Remove entries older than this
        """
        async with self._lock:
            now = datetime.utcnow()
            cutoff = now - timedelta(minutes=max_age_minutes)

            # Remove old entries
            old_count = len(self.buckets)
            self.buckets = {
                identifier: (tokens, last_update)
                for identifier, (tokens, last_update) in self.buckets.items()
                if last_update > cutoff
            }

            removed = old_count - len(self.buckets)

            if removed > 0:
                logger.info(
                    "Rate limiter cleanup",
                    removed_entries=removed,
                    remaining_entries=len(self.buckets)
                )

    async def get_stats(self) -> Dict:
        """Get rate limiter statistics"""
        async with self._lock:
            return {
                "total_requests": self.total_requests,
                "blocked_requests": self.blocked_requests,
                "block_rate": self.blocked_requests / max(1, self.total_requests),
                "active_identifiers": len(self.buckets)
            }

    async def reset_identifier(self, identifier: str):
        """Reset rate limit for specific identifier (admin use)"""
        async with self._lock:
            if identifier in self.buckets:
                del self.buckets[identifier]
                logger.info("Rate limit reset", identifier=identifier)


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
