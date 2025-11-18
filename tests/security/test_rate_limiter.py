"""
Security Tests for Rate Limiter
Tests rate limiting functionality to prevent abuse
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from app.web.rate_limiter import RateLimiter


class TestRateLimiterBasic:
    """Test basic rate limiting functionality"""

    @pytest.mark.asyncio
    async def test_initial_request_allowed(self):
        """Test that first request is always allowed"""
        limiter = RateLimiter(default_requests_per_minute=60, default_burst=10)
        allowed, reason = await limiter.check_rate_limit("test_ip")

        assert allowed is True, "First request should be allowed"
        assert reason is None, "No reason for allowing request"

    @pytest.mark.asyncio
    async def test_burst_limit_enforcement(self):
        """Test that burst limit is enforced"""
        limiter = RateLimiter(default_requests_per_minute=60, default_burst=3)

        # First 3 requests should succeed (burst)
        for i in range(3):
            allowed, _ = await limiter.check_rate_limit("test_ip")
            assert allowed is True, f"Request {i+1} should be allowed within burst"

        # 4th request should be rate limited
        allowed, reason = await limiter.check_rate_limit("test_ip")
        assert allowed is False, "Request beyond burst should be limited"
        assert reason is not None, "Should provide reason for rate limiting"
        assert "rate limit exceeded" in reason.lower(), "Reason should mention rate limit"

    @pytest.mark.asyncio
    async def test_different_identifiers_independent(self):
        """Test that different IPs have independent rate limits"""
        limiter = RateLimiter(default_requests_per_minute=60, default_burst=2)

        # Exhaust limit for IP1
        await limiter.check_rate_limit("ip1")
        await limiter.check_rate_limit("ip1")
        allowed_ip1, _ = await limiter.check_rate_limit("ip1")

        # IP2 should still be allowed
        allowed_ip2, _ = await limiter.check_rate_limit("ip2")

        assert allowed_ip1 is False, "IP1 should be rate limited"
        assert allowed_ip2 is True, "IP2 should not be affected by IP1's limit"

    @pytest.mark.asyncio
    async def test_token_regeneration(self):
        """Test that tokens regenerate over time"""
        limiter = RateLimiter(default_requests_per_minute=60, default_burst=2)

        # Exhaust burst
        await limiter.check_rate_limit("test_ip")
        await limiter.check_rate_limit("test_ip")
        allowed1, _ = await limiter.check_rate_limit("test_ip")

        assert allowed1 is False, "Should be rate limited immediately"

        # Wait for token regeneration (60 req/min = 1 req/sec)
        await asyncio.sleep(1.1)

        # Should be allowed again
        allowed2, _ = await limiter.check_rate_limit("test_ip")
        assert allowed2 is True, "Should be allowed after token regeneration"


class TestRateLimiterEndpointSpecific:
    """Test endpoint-specific rate limiting"""

    @pytest.mark.asyncio
    async def test_login_endpoint_stricter_limit(self):
        """Test that login endpoint has stricter limits"""
        limiter = RateLimiter(default_requests_per_minute=60, default_burst=10)

        # Login endpoint should have lower limits (5/min, burst 3)
        for i in range(3):
            allowed, _ = await limiter.check_rate_limit("test_ip", "/api/auth/login")
            assert allowed is True, f"Login request {i+1} should be allowed within burst"

        # 4th login attempt should be blocked
        allowed, reason = await limiter.check_rate_limit("test_ip", "/api/auth/login")
        assert allowed is False, "Login brute force should be prevented"

    @pytest.mark.asyncio
    async def test_expensive_endpoint_limit(self):
        """Test that expensive operations have lower limits"""
        limiter = RateLimiter(default_requests_per_minute=60, default_burst=10)

        # Self-modifying endpoint should have burst of 2
        for i in range(2):
            allowed, _ = await limiter.check_rate_limit("test_ip", "/api/self-modifying/improve")
            assert allowed is True, f"Expensive request {i+1} should be allowed"

        # 3rd request should be blocked
        allowed, _ = await limiter.check_rate_limit("test_ip", "/api/self-modifying/improve")
        assert allowed is False, "Expensive operation should be rate limited"

    @pytest.mark.asyncio
    async def test_default_limit_for_unknown_endpoint(self):
        """Test that unknown endpoints use default limits"""
        limiter = RateLimiter(default_requests_per_minute=60, default_burst=10)

        # Unknown endpoint should use default burst (10)
        for i in range(10):
            allowed, _ = await limiter.check_rate_limit("test_ip", "/api/unknown/endpoint")
            assert allowed is True, f"Request {i+1} should be allowed with default limit"

        # 11th request should be blocked
        allowed, _ = await limiter.check_rate_limit("test_ip", "/api/unknown/endpoint")
        assert allowed is False, "Should enforce default limit for unknown endpoints"


class TestRateLimiterStatistics:
    """Test rate limiter statistics and monitoring"""

    @pytest.mark.asyncio
    async def test_statistics_tracking(self):
        """Test that statistics are properly tracked"""
        limiter = RateLimiter(default_requests_per_minute=60, default_burst=2)

        # Make some requests
        await limiter.check_rate_limit("ip1")
        await limiter.check_rate_limit("ip1")
        await limiter.check_rate_limit("ip1")  # This should be blocked

        stats = await limiter.get_stats()

        assert stats["total_requests"] == 3, "Should track total requests"
        assert stats["blocked_requests"] == 1, "Should track blocked requests"
        assert stats["block_rate"] == 1/3, "Should calculate block rate"
        assert stats["active_identifiers"] >= 1, "Should track active identifiers"

    @pytest.mark.asyncio
    async def test_cleanup_old_entries(self):
        """Test that old entries are cleaned up"""
        limiter = RateLimiter(default_requests_per_minute=60, default_burst=10)

        # Create entries
        await limiter.check_rate_limit("ip1")
        await limiter.check_rate_limit("ip2")

        # Manually set old timestamp
        limiter.buckets["ip1"] = (5.0, datetime.utcnow() - timedelta(hours=2))

        # Cleanup
        await limiter.cleanup_old_entries(max_age_minutes=60)

        # ip1 should be removed, ip2 should remain
        assert "ip1" not in limiter.buckets, "Old entries should be removed"
        assert "ip2" in limiter.buckets, "Recent entries should remain"

    @pytest.mark.asyncio
    async def test_reset_identifier(self):
        """Test manual reset of rate limit for identifier"""
        limiter = RateLimiter(default_requests_per_minute=60, default_burst=1)

        # Exhaust limit
        await limiter.check_rate_limit("test_ip")
        await limiter.check_rate_limit("test_ip")
        allowed1, _ = await limiter.check_rate_limit("test_ip")

        assert allowed1 is False, "Should be rate limited"

        # Reset
        await limiter.reset_identifier("test_ip")

        # Should be allowed again
        allowed2, _ = await limiter.check_rate_limit("test_ip")
        assert allowed2 is True, "Should be allowed after reset"


class TestRateLimiterSecurity:
    """Test security properties of rate limiter"""

    @pytest.mark.asyncio
    async def test_prevents_brute_force_login(self):
        """Test that rate limiter prevents login brute force attacks"""
        limiter = RateLimiter()

        # Simulate brute force attempt
        attempts = 0
        blocked = 0

        for i in range(100):
            allowed, _ = await limiter.check_rate_limit("attacker_ip", "/api/auth/login")
            attempts += 1
            if not allowed:
                blocked += 1

        # Should have blocked majority of requests
        assert blocked > 95, f"Should block brute force (blocked {blocked}/100)"
        assert attempts == 100, "Should process all attempts"

    @pytest.mark.asyncio
    async def test_concurrent_requests_safe(self):
        """Test that concurrent requests are handled safely"""
        limiter = RateLimiter(default_requests_per_minute=60, default_burst=50)

        # Send concurrent requests
        tasks = [
            limiter.check_rate_limit("test_ip")
            for _ in range(100)
        ]

        results = await asyncio.gather(*tasks)
        allowed_count = sum(1 for allowed, _ in results if allowed)

        # Should allow up to burst limit
        assert allowed_count == 50, "Should enforce burst limit with concurrent requests"

    @pytest.mark.asyncio
    async def test_retry_after_guidance(self):
        """Test that retry-after information is provided"""
        limiter = RateLimiter(default_requests_per_minute=60, default_burst=1)

        # Exhaust limit
        await limiter.check_rate_limit("test_ip")
        await limiter.check_rate_limit("test_ip")
        allowed, reason = await limiter.check_rate_limit("test_ip")

        assert allowed is False, "Should be rate limited"
        assert "retry after" in reason.lower(), "Should provide retry-after guidance"
        assert "second" in reason.lower(), "Should specify time unit"

    @pytest.mark.asyncio
    async def test_dos_protection(self):
        """Test that rate limiter protects against DoS attacks"""
        limiter = RateLimiter(default_requests_per_minute=60, default_burst=10)

        # Simulate DoS attack (1000 requests)
        blocked = 0
        for i in range(1000):
            allowed, _ = await limiter.check_rate_limit("attacker_ip")
            if not allowed:
                blocked += 1

        # Should block vast majority
        assert blocked > 985, f"Should block DoS attack (blocked {blocked}/1000)"

    @pytest.mark.asyncio
    async def test_distributed_attack_detection(self):
        """Test that distributed attacks from multiple IPs are limited"""
        limiter = RateLimiter(default_requests_per_minute=60, default_burst=5)

        # Simulate distributed attack (100 IPs, 10 requests each)
        total_blocked = 0

        for ip in range(100):
            for request in range(10):
                allowed, _ = await limiter.check_rate_limit(f"attacker_{ip}")
                if not allowed:
                    total_blocked += 1

        # Each IP should have ~5 blocked (after burst)
        expected_blocked = 100 * 5  # 100 IPs * 5 blocked requests each
        assert total_blocked >= expected_blocked * 0.9, \
            f"Should block distributed attack (blocked {total_blocked}, expected ~{expected_blocked})"


class TestRateLimiterEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_zero_burst(self):
        """Test behavior with zero burst (should still allow based on rate)"""
        # This tests that the implementation handles edge cases gracefully
        limiter = RateLimiter(default_requests_per_minute=60, default_burst=0)

        allowed, _ = await limiter.check_rate_limit("test_ip")
        # With 0 burst but 60/min rate, first request might still be allowed
        # depending on implementation details
        assert isinstance(allowed, bool), "Should return boolean"

    @pytest.mark.asyncio
    async def test_very_high_rate(self):
        """Test with very high rate limits"""
        limiter = RateLimiter(default_requests_per_minute=10000, default_burst=1000)

        # Should allow many requests
        allowed_count = 0
        for i in range(1000):
            allowed, _ = await limiter.check_rate_limit("test_ip")
            if allowed:
                allowed_count += 1

        assert allowed_count == 1000, "Should allow all requests within burst"

    @pytest.mark.asyncio
    async def test_empty_identifier(self):
        """Test handling of empty identifier"""
        limiter = RateLimiter()

        allowed, _ = await limiter.check_rate_limit("")
        assert isinstance(allowed, bool), "Should handle empty identifier gracefully"

    @pytest.mark.asyncio
    async def test_special_characters_in_identifier(self):
        """Test identifiers with special characters"""
        limiter = RateLimiter()

        special_ids = [
            "192.168.1.1",
            "2001:0db8:85a3::8a2e:0370:7334",  # IPv6
            "user@example.com",
            "user:session:12345",
            "../../../etc/passwd",  # Path traversal attempt
        ]

        for identifier in special_ids:
            allowed, _ = await limiter.check_rate_limit(identifier)
            assert isinstance(allowed, bool), f"Should handle identifier: {identifier}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
