"""
MEMSHADOW Client
Base client for interacting with MEMSHADOW API
"""

import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MemshadowClient:
    """
    Base client for MEMSHADOW memory persistence API.

    Example:
        >>> client = MemshadowClient(
        ...     api_url="http://localhost:8000/api/v1",
        ...     api_key="your-api-key"
        ... )
        >>> client.ingest("I love Python programming")
        >>> results = client.retrieve("programming languages")
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        user_id: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize MEMSHADOW client.

        Args:
            api_url: Base URL of the MEMSHADOW API (e.g., "http://localhost:8000/api/v1")
            api_key: API authentication token
            user_id: Optional user ID for multi-user scenarios
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.user_id = user_id
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })

    def ingest(
        self,
        content: str,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a new memory into MEMSHADOW.

        Args:
            content: The text content to store
            extra_data: Optional metadata to attach

        Returns:
            Memory object with ID and metadata

        Raises:
            requests.HTTPError: If the API request fails
        """
        payload = {
            "content": content,
            "extra_data": extra_data or {}
        }

        try:
            response = self.session.post(
                f"{self.api_url}/memory/ingest",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            memory = response.json()
            logger.debug(f"Memory ingested: {memory['id']}")
            return memory

        except requests.RequestException as e:
            logger.error(f"Failed to ingest memory: {e}")
            raise

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories using semantic search.

        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filter criteria

        Returns:
            List of memory objects

        Raises:
            requests.HTTPError: If the API request fails
        """
        payload = {
            "query": query,
            "filters": filters
        }

        try:
            response = self.session.post(
                f"{self.api_url}/memory/retrieve",
                json=payload,
                params={"limit": limit},
                timeout=self.timeout
            )
            response.raise_for_status()

            memories = response.json()
            logger.debug(f"Retrieved {len(memories)} memories")
            return memories

        except requests.RequestException as e:
            logger.error(f"Failed to retrieve memories: {e}")
            raise

    def create_reminder(
        self,
        title: str,
        reminder_date: datetime,
        description: Optional[str] = None,
        due_date: Optional[datetime] = None,
        priority: str = "medium",
        associated_memory_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a task reminder.

        Args:
            title: Title of the reminder
            reminder_date: When to send the reminder
            description: Optional detailed description
            due_date: Optional task due date
            priority: Priority level (low, medium, high)
            associated_memory_id: Optional linked memory ID

        Returns:
            Reminder object

        Raises:
            requests.HTTPError: If the API request fails
        """
        payload = {
            "title": title,
            "reminder_date": reminder_date.isoformat(),
            "description": description,
            "priority": priority,
            "associated_memory_id": associated_memory_id
        }

        if due_date:
            payload["due_date"] = due_date.isoformat()

        try:
            response = self.session.post(
                f"{self.api_url}/reminders/",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            reminder = response.json()
            logger.debug(f"Reminder created: {reminder['id']}")
            return reminder

        except requests.RequestException as e:
            logger.error(f"Failed to create reminder: {e}")
            raise

    def list_reminders(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        List task reminders.

        Args:
            status: Filter by status (pending, reminded, completed, cancelled)
            priority: Filter by priority (low, medium, high)
            limit: Maximum number of results

        Returns:
            List of reminder objects

        Raises:
            requests.HTTPError: If the API request fails
        """
        params = {"limit": limit}
        if status:
            params["status"] = status
        if priority:
            params["priority"] = priority

        try:
            response = self.session.get(
                f"{self.api_url}/reminders/",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()

            reminders = response.json()
            logger.debug(f"Retrieved {len(reminders)} reminders")
            return reminders

        except requests.RequestException as e:
            logger.error(f"Failed to list reminders: {e}")
            raise

    def complete_reminder(self, reminder_id: str) -> Dict[str, Any]:
        """
        Mark a reminder as completed.

        Args:
            reminder_id: ID of the reminder to complete

        Returns:
            Updated reminder object

        Raises:
            requests.HTTPError: If the API request fails
        """
        try:
            response = self.session.post(
                f"{self.api_url}/reminders/{reminder_id}/complete",
                timeout=self.timeout
            )
            response.raise_for_status()

            reminder = response.json()
            logger.debug(f"Reminder completed: {reminder_id}")
            return reminder

        except requests.RequestException as e:
            logger.error(f"Failed to complete reminder: {e}")
            raise
