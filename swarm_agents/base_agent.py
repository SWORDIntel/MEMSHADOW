"""
Base Agent Framework for SWARM

All SWARM agents inherit from this base class which provides:
- Connection to the blackboard
- Task polling and report publishing
- Logging and error handling
- Graceful shutdown
"""

import time
import signal
import sys
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import structlog

# Add parent directory to path for imports
sys.path.insert(0, '/app')

from app.services.swarm.blackboard import Blackboard

logger = structlog.get_logger()


class BaseAgent(ABC):
    """
    Abstract base class for all SWARM agents
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        blackboard: Blackboard = None
    ):
        """
        Initialize the base agent

        Args:
            agent_id: Unique identifier for this agent instance
            agent_type: Type of agent (e.g., 'recon', 'apimapper')
            blackboard: Blackboard instance (will create if not provided)
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.blackboard = blackboard or Blackboard()

        self.running = False
        self.tasks_processed = 0

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(
            "Agent initialized",
            agent_id=self.agent_id,
            agent_type=self.agent_type
        )

    def _signal_handler(self, signum, frame):
        """
        Handle shutdown signals gracefully

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info(
            "Shutdown signal received",
            agent_id=self.agent_id,
            signal=signum
        )
        self.shutdown()

    @abstractmethod
    async def execute_task(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task. Must be implemented by subclasses.

        Args:
            task_payload: Task parameters

        Returns:
            Task result data

        Raises:
            Exception: If task execution fails
        """
        pass

    def run(self, poll_interval: int = 2):
        """
        Main agent loop - polls for tasks and executes them

        Args:
            poll_interval: Seconds between polls
        """
        self.running = True

        logger.info(
            "Agent starting main loop",
            agent_id=self.agent_id,
            agent_type=self.agent_type
        )

        while self.running:
            try:
                # Poll for tasks
                task = self.blackboard.get_task(self.agent_type, timeout=poll_interval)

                if task:
                    # Process the task
                    self._process_task(task)
                else:
                    # No task available
                    logger.debug(
                        "No tasks available",
                        agent_id=self.agent_id,
                        agent_type=self.agent_type
                    )

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received", agent_id=self.agent_id)
                self.shutdown()
                break

            except Exception as e:
                logger.error(
                    "Error in agent main loop",
                    agent_id=self.agent_id,
                    error=str(e),
                    exc_info=True
                )
                time.sleep(poll_interval)

        logger.info(
            "Agent stopped",
            agent_id=self.agent_id,
            tasks_processed=self.tasks_processed
        )

    def _process_task(self, task: Dict[str, Any]):
        """
        Process a single task

        Args:
            task: Task message from coordinator
        """
        task_id = task.get('task_id')
        payload = task.get('payload', {})

        logger.info(
            "Processing task",
            agent_id=self.agent_id,
            task_id=task_id
        )

        try:
            # Execute the task
            import asyncio
            result_data = asyncio.run(self.execute_task(payload))

            # Publish success report
            self._publish_report(
                task_id=task_id,
                status="SUCCESS",
                data=result_data
            )

            self.tasks_processed += 1

            logger.info(
                "Task completed successfully",
                agent_id=self.agent_id,
                task_id=task_id
            )

        except Exception as e:
            logger.error(
                "Task execution failed",
                agent_id=self.agent_id,
                task_id=task_id,
                error=str(e),
                exc_info=True
            )

            # Publish failure report
            self._publish_report(
                task_id=task_id,
                status="FAILURE",
                data={},
                error_message=str(e)
            )

    def _publish_report(
        self,
        task_id: str,
        status: str,
        data: Dict[str, Any],
        error_message: Optional[str] = None
    ):
        """
        Publish a task report to the blackboard

        Args:
            task_id: Task ID
            status: SUCCESS or FAILURE
            data: Result data
            error_message: Error message if failed
        """
        report = {
            "task_id": task_id,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": status,
            "data": data,
            "error_message": error_message
        }

        self.blackboard.publish_report(report)

        logger.debug(
            "Report published",
            agent_id=self.agent_id,
            task_id=task_id,
            status=status
        )

    def shutdown(self):
        """
        Shutdown the agent gracefully
        """
        logger.info("Agent shutting down", agent_id=self.agent_id)
        self.running = False

    def __del__(self):
        """
        Cleanup on destruction
        """
        if hasattr(self, 'blackboard') and self.blackboard:
            self.blackboard.close()
