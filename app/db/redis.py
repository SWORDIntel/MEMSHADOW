import redis.asyncio as redis
from typing import Optional, Any
import json
import structlog
import time

from app.core.config import settings

logger = structlog.get_logger()

class RedisClient:
    def __init__(self):
        self.pool = None

    async def init_pool(self):
        """Initialize Redis connection pool"""
        try:
            self.pool = redis.ConnectionPool.from_url(
                str(settings.REDIS_URL),
                max_connections=50,
                decode_responses=True
            )

            # Test connection
            async with redis.Redis(connection_pool=self.pool) as conn:
                await conn.ping()

            logger.info("Redis pool initialized")
        except Exception as e:
            logger.error("Redis initialization failed", error=str(e))
            raise

    async def close_pool(self):
        """Close Redis connection pool"""
        if self.pool:
            await self.pool.disconnect()
            logger.info("Redis pool closed")

    async def get_client(self) -> redis.Redis:
        """Get Redis client from pool"""
        return redis.Redis(connection_pool=self.pool)

    # Cache operations
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        r = await self.get_client()
        value = await r.get(key)
        if value:
            return json.loads(value)
        return None

    async def cache_set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Set value in cache"""
        r = await self.get_client()
        serialized = json.dumps(value)
        if ttl:
            await r.setex(key, ttl, serialized)
        else:
            await r.set(key, serialized)

    async def cache_delete(self, key: str):
        """Delete value from cache"""
        r = await self.get_client()
        await r.delete(key)

    # Rate limiting
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window: int
    ) -> bool:
        """Check if rate limit is exceeded"""
        r = await self.get_client()
        pipe = r.pipeline()
        now = int(time.time())
        window_start = now - window

        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        # Add current request
        pipe.zadd(key, {str(now): now})
        # Count requests in window
        pipe.zcard(key)
        # Set expiry
        pipe.expire(key, window + 1)

        results = await pipe.execute()
        request_count = results[2]

        return request_count <= limit

    async def lpush(self, key: str, value: Any):
        r = await self.get_client()
        await r.lpush(key, json.dumps(value))

    async def expire(self, key: str, ttl: int):
        r = await self.get_client()
        await r.expire(key, ttl)

    async def ltrim(self, key: str, start: int, end: int):
        r = await self.get_client()
        await r.ltrim(key, start, end)

    async def lrange(self, key: str, start: int, end: int):
        r = await self.get_client()
        values = await r.lrange(key, start, end)
        return [json.loads(v) for v in values]


# Global client instance
redis_client = RedisClient()

async def init_pool():
    await redis_client.init_pool()

async def close_pool():
    await redis_client.close_pool()