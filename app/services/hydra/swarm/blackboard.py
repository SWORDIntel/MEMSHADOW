"""
HYDRA SWARM Redis Blackboard
Phase 7: Communication and shared state for agent swarm

The blackboard pattern allows agents to:
- Share discovered intelligence
- Coordinate activities
- Avoid duplicate work
- Report findings
- Maintain shared state
"""

from typing import Dict, List, Any, Optional
import json
import asyncio
from datetime import datetime
import structlog

logger = structlog.get_logger()


class RedisBlackboard:
    """
    Redis-based blackboard for agent communication.

    Provides:
    - Key-value storage for shared intelligence
    - Pub/sub for real-time communication
    - TTL support for temporary data
    - Atomic operations

    Example:
        blackboard = RedisBlackboard()

        # Share discovered endpoint
        await blackboard.set(
            "intel:endpoints",
            ["/api/login", "/api/users"]
        )

        # Get shared intelligence
        endpoints = await blackboard.get("intel:endpoints")

        # Publish finding
        await blackboard.publish(
            "findings:coord_123",
            {"severity": "HIGH", "title": "SQL Injection"}
        )
    """

    def __init__(self, redis_client=None):
        """
        Initialize blackboard.

        Args:
            redis_client: Redis client (optional, uses mock if None)
        """
        self.redis = redis_client
        self.subscribers = {}
        self.mock_storage: Dict[str, Any] = {}  # For testing without Redis

        logger.info("Redis blackboard initialized", has_redis=redis_client is not None)

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """
        Set a value in the blackboard.

        Args:
            key: Storage key
            value: Value to store (will be JSON serialized)
            ttl: Time-to-live in seconds (optional)
        """
        serialized = json.dumps(value) if not isinstance(value, str) else value

        if self.redis:
            if ttl:
                await self.redis.setex(key, ttl, serialized)
            else:
                await self.redis.set(key, serialized)
        else:
            # Mock storage
            self.mock_storage[key] = {
                "value": serialized,
                "expires_at": datetime.utcnow().timestamp() + ttl if ttl else None
            }

        logger.debug("Blackboard set", key=key, has_ttl=ttl is not None)

    async def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get a value from the blackboard.

        Args:
            key: Storage key
            default: Default value if key not found

        Returns:
            Stored value (JSON deserialized) or default
        """
        if self.redis:
            value = await self.redis.get(key)
            if value is None:
                return default

            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        else:
            # Mock storage
            if key not in self.mock_storage:
                return default

            item = self.mock_storage[key]

            # Check expiration
            if item["expires_at"] and datetime.utcnow().timestamp() > item["expires_at"]:
                del self.mock_storage[key]
                return default

            try:
                return json.loads(item["value"])
            except json.JSONDecodeError:
                return item["value"]

    async def delete(self, key: str):
        """Delete a key from the blackboard"""
        if self.redis:
            await self.redis.delete(key)
        else:
            self.mock_storage.pop(key, None)

        logger.debug("Blackboard delete", key=key)

    async def append(
        self,
        key: str,
        value: Any
    ):
        """
        Append to a list in the blackboard.

        Args:
            key: Storage key
            value: Value to append
        """
        existing = await self.get(key, [])
        if not isinstance(existing, list):
            existing = [existing]

        existing.append(value)
        await self.set(key, existing)

    async def increment(
        self,
        key: str,
        amount: int = 1
    ) -> int:
        """
        Increment a counter.

        Args:
            key: Counter key
            amount: Amount to increment

        Returns:
            New value
        """
        if self.redis:
            return await self.redis.incrby(key, amount)
        else:
            current = await self.get(key, 0)
            new_value = int(current) + amount
            await self.set(key, new_value)
            return new_value

    async def publish(
        self,
        channel: str,
        message: Dict[str, Any]
    ):
        """
        Publish message to a channel.

        Args:
            channel: Channel name
            message: Message to publish
        """
        serialized = json.dumps(message)

        if self.redis:
            await self.redis.publish(channel, serialized)
        else:
            # Mock pub/sub
            if channel in self.subscribers:
                for callback in self.subscribers[channel]:
                    await callback(message)

        logger.debug("Message published", channel=channel)

    async def subscribe(
        self,
        channel: str,
        callback: callable
    ):
        """
        Subscribe to a channel.

        Args:
            channel: Channel name
            callback: Async function to call on message
        """
        if self.redis:
            # Redis pub/sub implementation
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(channel)

            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        await callback(data)
                    except Exception as e:
                        logger.error("Subscriber callback error", error=str(e))
        else:
            # Mock subscription
            if channel not in self.subscribers:
                self.subscribers[channel] = []
            self.subscribers[channel].append(callback)

        logger.debug("Subscribed to channel", channel=channel)

    async def get_keys_pattern(
        self,
        pattern: str
    ) -> List[str]:
        """
        Get keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "intel:*")

        Returns:
            List of matching keys
        """
        if self.redis:
            keys = await self.redis.keys(pattern)
            return [k.decode() if isinstance(k, bytes) else k for k in keys]
        else:
            # Simple pattern matching for mock
            import fnmatch
            return [
                k for k in self.mock_storage.keys()
                if fnmatch.fnmatch(k, pattern)
            ]

    async def acquire_lock(
        self,
        lock_name: str,
        timeout: int = 10
    ) -> bool:
        """
        Acquire a distributed lock.

        Args:
            lock_name: Lock identifier
            timeout: Lock timeout in seconds

        Returns:
            True if lock acquired
        """
        key = f"lock:{lock_name}"

        if self.redis:
            return await self.redis.set(key, "1", ex=timeout, nx=True)
        else:
            # Mock lock
            if key in self.mock_storage:
                item = self.mock_storage[key]
                if item["expires_at"] and datetime.utcnow().timestamp() < item["expires_at"]:
                    return False

            await self.set(key, "1", ttl=timeout)
            return True

    async def release_lock(self, lock_name: str):
        """Release a distributed lock"""
        await self.delete(f"lock:{lock_name}")

    async def get_agent_count(self) -> int:
        """Get count of active agents"""
        keys = await self.get_keys_pattern("agent:*:heartbeat")
        return len(keys)

    async def get_findings_count(self, coordinator_id: str) -> int:
        """Get count of findings for coordinator"""
        return await self.get(f"findings:count:{coordinator_id}", 0)

    async def get_shared_intelligence_summary(self) -> Dict[str, int]:
        """
        Get summary of shared intelligence.

        Returns:
            Dict of intelligence type -> count
        """
        intel_keys = await self.get_keys_pattern("intel:*")

        summary = {}
        for key in intel_keys:
            intel_type = key.replace("intel:", "")
            data = await self.get(key, [])
            count = len(data) if isinstance(data, list) else 1
            summary[intel_type] = count

        return summary

    async def clear_mission_data(self, mission_id: str):
        """
        Clear all data for a mission.

        Args:
            mission_id: Mission ID
        """
        pattern = f"mission:{mission_id}:*"
        keys = await self.get_keys_pattern(pattern)

        for key in keys:
            await self.delete(key)

        logger.info("Mission data cleared", mission_id=mission_id, keys_deleted=len(keys))


# Global instance (can be initialized with actual Redis client)
blackboard = RedisBlackboard()
