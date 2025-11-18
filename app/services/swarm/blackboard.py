"""
Blackboard Communication Protocol for SWARM

The Blackboard is a shared knowledge base using Redis that allows agents to:
- Communicate asynchronously
- Share discovered information
- Coordinate tasks
- Report results

All communication happens through Redis lists and keys.
"""

import redis
import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog

from app.core.config import settings

logger = structlog.get_logger()


class Blackboard:
    """
    Redis-based blackboard for agent communication and coordination.

    Key Patterns:
    - hydra:tasks:{agent_type} - Task queues for specific agent types
    - hydra:reports:ingress - Report queue where agents post results
    - hydra:blackboard:{key} - Shared state and discovered data
    - hydra:mission:{mission_id} - Mission-specific data
    """

    def __init__(self, redis_url: str = None):
        """
        Initialize blackboard with Redis connection

        Args:
            redis_url: Redis connection URL (defaults to settings)
        """
        self.redis_url = redis_url or str(settings.REDIS_URL)
        self.redis_client = redis.from_url(
            self.redis_url,
            decode_responses=True
        )

        # Key prefixes
        self.TASK_PREFIX = "hydra:tasks:"
        self.REPORT_QUEUE = "hydra:reports:ingress"
        self.BLACKBOARD_PREFIX = "hydra:blackboard:"
        self.MISSION_PREFIX = "hydra:mission:"

        logger.info("Blackboard initialized", redis_url=self.redis_url)

    # === Task Queue Operations ===

    def publish_task(self, agent_type: str, task: Dict[str, Any]) -> str:
        """
        Publish a task to an agent type's queue

        Args:
            agent_type: Type of agent (e.g., 'recon', 'apimapper')
            task: Task payload

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())

        task_message = {
            "task_id": task_id,
            "agent_type_target": agent_type,
            "payload": task,
            "timestamp": datetime.utcnow().isoformat()
        }

        queue_key = f"{self.TASK_PREFIX}{agent_type}"
        self.redis_client.lpush(queue_key, json.dumps(task_message))

        logger.debug(
            "Task published",
            agent_type=agent_type,
            task_id=task_id,
            queue=queue_key
        )

        return task_id

    def get_task(self, agent_type: str, timeout: int = 5) -> Optional[Dict[str, Any]]:
        """
        Get a task from agent type's queue (blocking with timeout)

        Args:
            agent_type: Type of agent
            timeout: Timeout in seconds

        Returns:
            Task dict or None
        """
        queue_key = f"{self.TASK_PREFIX}{agent_type}"

        result = self.redis_client.brpop(queue_key, timeout=timeout)

        if result:
            _, task_json = result
            task = json.loads(task_json)

            logger.debug(
                "Task retrieved",
                agent_type=agent_type,
                task_id=task.get('task_id')
            )

            return task

        return None

    def get_task_count(self, agent_type: str) -> int:
        """
        Get number of pending tasks for an agent type

        Args:
            agent_type: Type of agent

        Returns:
            Number of tasks in queue
        """
        queue_key = f"{self.TASK_PREFIX}{agent_type}"
        return self.redis_client.llen(queue_key)

    # === Report Operations ===

    def publish_report(self, report: Dict[str, Any]) -> bool:
        """
        Publish an agent report to the ingress queue

        Args:
            report: Report payload containing task results

        Returns:
            Success boolean
        """
        report_message = {
            **report,
            "report_timestamp": datetime.utcnow().isoformat()
        }

        self.redis_client.lpush(self.REPORT_QUEUE, json.dumps(report_message))

        logger.debug(
            "Report published",
            task_id=report.get('task_id'),
            agent_id=report.get('agent_id'),
            status=report.get('status')
        )

        return True

    def get_report(self, timeout: int = 5) -> Optional[Dict[str, Any]]:
        """
        Get a report from the ingress queue (blocking with timeout)

        Args:
            timeout: Timeout in seconds

        Returns:
            Report dict or None
        """
        result = self.redis_client.brpop(self.REPORT_QUEUE, timeout=timeout)

        if result:
            _, report_json = result
            report = json.loads(report_json)

            logger.debug(
                "Report retrieved",
                task_id=report.get('task_id'),
                agent_id=report.get('agent_id')
            )

            return report

        return None

    def get_pending_reports_count(self) -> int:
        """
        Get number of pending reports

        Returns:
            Number of reports in queue
        """
        return self.redis_client.llen(self.REPORT_QUEUE)

    # === Blackboard Data Operations ===

    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        Set a value on the blackboard

        Args:
            key: Key name (will be prefixed with blackboard prefix)
            value: Value to store (will be JSON serialized)
            ttl: Time-to-live in seconds (optional)

        Returns:
            Success boolean
        """
        full_key = f"{self.BLACKBOARD_PREFIX}{key}"

        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        elif not isinstance(value, str):
            value = str(value)

        self.redis_client.set(full_key, value)

        if ttl:
            self.redis_client.expire(full_key, ttl)

        logger.debug("Blackboard value set", key=key, has_ttl=ttl is not None)

        return True

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the blackboard

        Args:
            key: Key name (will be prefixed)

        Returns:
            Value or None
        """
        full_key = f"{self.BLACKBOARD_PREFIX}{key}"
        value = self.redis_client.get(full_key)

        if value is None:
            return None

        # Try to parse as JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    def exists(self, key: str) -> bool:
        """
        Check if a key exists on the blackboard

        Args:
            key: Key name

        Returns:
            True if exists
        """
        full_key = f"{self.BLACKBOARD_PREFIX}{key}"
        return self.redis_client.exists(full_key) > 0

    def append_to_list(self, key: str, value: Any) -> int:
        """
        Append a value to a list on the blackboard

        Args:
            key: List key name
            value: Value to append

        Returns:
            New length of list
        """
        full_key = f"{self.BLACKBOARD_PREFIX}{key}"

        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        elif not isinstance(value, str):
            value = str(value)

        length = self.redis_client.lpush(full_key, value)

        logger.debug("Value appended to blackboard list", key=key, new_length=length)

        return length

    def get_list(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """
        Get all values from a list on the blackboard

        Args:
            key: List key name
            start: Start index
            end: End index (-1 for all)

        Returns:
            List of values
        """
        full_key = f"{self.BLACKBOARD_PREFIX}{key}"
        values = self.redis_client.lrange(full_key, start, end)

        # Try to parse each value as JSON
        parsed_values = []
        for value in values:
            try:
                parsed_values.append(json.loads(value))
            except (json.JSONDecodeError, TypeError):
                parsed_values.append(value)

        return parsed_values

    def delete(self, key: str) -> bool:
        """
        Delete a key from the blackboard

        Args:
            key: Key name

        Returns:
            Success boolean
        """
        full_key = f"{self.BLACKBOARD_PREFIX}{key}"
        deleted = self.redis_client.delete(full_key)

        logger.debug("Blackboard key deleted", key=key, was_present=deleted > 0)

        return deleted > 0

    def get_all_keys(self, pattern: str = "*") -> List[str]:
        """
        Get all blackboard keys matching a pattern

        Args:
            pattern: Key pattern (supports wildcards)

        Returns:
            List of keys (without prefix)
        """
        full_pattern = f"{self.BLACKBOARD_PREFIX}{pattern}"
        keys = self.redis_client.keys(full_pattern)

        # Remove prefix from keys
        prefix_len = len(self.BLACKBOARD_PREFIX)
        return [key[prefix_len:] for key in keys]

    # === Mission-Specific Operations ===

    def set_mission_data(self, mission_id: str, key: str, value: Any) -> bool:
        """
        Set mission-specific data

        Args:
            mission_id: Mission ID
            key: Data key
            value: Value

        Returns:
            Success boolean
        """
        mission_key = f"{self.MISSION_PREFIX}{mission_id}:{key}"

        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        elif not isinstance(value, str):
            value = str(value)

        self.redis_client.set(mission_key, value)

        return True

    def get_mission_data(self, mission_id: str, key: str) -> Optional[Any]:
        """
        Get mission-specific data

        Args:
            mission_id: Mission ID
            key: Data key

        Returns:
            Value or None
        """
        mission_key = f"{self.MISSION_PREFIX}{mission_id}:{key}"
        value = self.redis_client.get(mission_key)

        if value is None:
            return None

        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    def clear_mission(self, mission_id: str) -> int:
        """
        Clear all data for a mission

        Args:
            mission_id: Mission ID

        Returns:
            Number of keys deleted
        """
        pattern = f"{self.MISSION_PREFIX}{mission_id}:*"
        keys = self.redis_client.keys(pattern)

        if keys:
            deleted = self.redis_client.delete(*keys)
            logger.info("Mission data cleared", mission_id=mission_id, keys_deleted=deleted)
            return deleted

        return 0

    def flush_all(self) -> bool:
        """
        DANGER: Flush all HYDRA data from Redis
        Only use for testing or cleanup

        Returns:
            Success boolean
        """
        patterns = [
            f"{self.TASK_PREFIX}*",
            self.REPORT_QUEUE,
            f"{self.BLACKBOARD_PREFIX}*",
            f"{self.MISSION_PREFIX}*"
        ]

        all_keys = []
        for pattern in patterns:
            all_keys.extend(self.redis_client.keys(pattern))

        if all_keys:
            deleted = self.redis_client.delete(*all_keys)
            logger.warning("All SWARM data flushed from Redis", keys_deleted=deleted)
            return True

        return True

    def close(self):
        """Close Redis connection"""
        if self.redis_client:
            self.redis_client.close()
            logger.info("Blackboard connection closed")
