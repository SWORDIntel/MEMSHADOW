"""
SWARM Coordinator Node

The Coordinator is the central intelligence of the SWARM system. It:
- Loads and manages missions
- Dispatches tasks to agents
- Collects and processes reports
- Maintains the blackboard state
- Generates final mission reports

MCP Integration: Can be controlled via API endpoints for mission management
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from enum import Enum
import structlog

from .blackboard import Blackboard
from .mission import Mission, MissionStage, MissionTask

logger = structlog.get_logger()


class StageStatus(str, Enum):
    """Status of a mission stage"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class MissionStatus(str, Enum):
    """Status of overall mission"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class SwarmCoordinator:
    """
    Central coordinator for the SWARM autonomous agent system
    """

    def __init__(self, blackboard: Blackboard = None):
        """
        Initialize the coordinator

        Args:
            blackboard: Blackboard instance (will create new if not provided)
        """
        self.blackboard = blackboard or Blackboard()
        self.mission: Optional[Mission] = None
        self.mission_status = MissionStatus.INITIALIZED

        # Track stage statuses
        self.stage_statuses: Dict[str, StageStatus] = {}
        self.completed_stages: Set[str] = set()
        self.failed_stages: Set[str] = set()

        # Track dispatched tasks
        self.dispatched_tasks: Dict[str, Dict[str, Any]] = {}
        self.pending_reports: Set[str] = set()

        # Mission start time
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Report accumulator
        self.mission_report: Dict[str, Any] = {}

        logger.info("SwarmCoordinator initialized")

    def load_mission(self, mission: Mission) -> bool:
        """
        Load a mission into the coordinator

        Args:
            mission: Mission object

        Returns:
            Success boolean
        """
        try:
            self.mission = mission
            self.mission_status = MissionStatus.INITIALIZED

            # Initialize stage statuses
            for stage in mission.objective_stages:
                self.stage_statuses[stage.stage_id] = StageStatus.PENDING

            # Store mission in blackboard
            self.blackboard.set_mission_data(
                mission.mission_id,
                "definition",
                mission.dict()
            )

            logger.info(
                "Mission loaded into coordinator",
                mission_id=mission.mission_id,
                num_stages=len(mission.objective_stages)
            )

            return True

        except Exception as e:
            logger.error("Failed to load mission", error=str(e))
            return False

    async def execute_mission(self, timeout: int = 3600) -> Dict[str, Any]:
        """
        Execute the loaded mission

        Args:
            timeout: Maximum mission duration in seconds

        Returns:
            Mission execution report
        """
        if not self.mission:
            raise ValueError("No mission loaded")

        self.mission_status = MissionStatus.RUNNING
        self.start_time = datetime.utcnow()

        logger.info(
            "Starting mission execution",
            mission_id=self.mission.mission_id,
            mission_name=self.mission.mission_name
        )

        try:
            # Main coordination loop
            start_time = time.time()

            while True:
                # Check timeout
                if time.time() - start_time > timeout:
                    logger.warning("Mission timeout exceeded", mission_id=self.mission.mission_id)
                    self.mission_status = MissionStatus.ABORTED
                    break

                # Check if mission is complete
                if self.mission.is_complete(self.completed_stages):
                    logger.info("Mission completed successfully", mission_id=self.mission.mission_id)
                    self.mission_status = MissionStatus.COMPLETED
                    break

                # Check for failed stages
                if self.failed_stages:
                    logger.error(
                        "Mission failed due to stage failures",
                        mission_id=self.mission.mission_id,
                        failed_stages=list(self.failed_stages)
                    )
                    self.mission_status = MissionStatus.FAILED
                    break

                # Process ready stages
                await self._process_ready_stages()

                # Process incoming reports
                await self._process_reports()

                # Brief sleep to prevent tight loop
                await asyncio.sleep(1)

            self.end_time = datetime.utcnow()

            # Generate final report
            report = await self._generate_final_report()

            return report

        except Exception as e:
            logger.error(
                "Mission execution failed",
                mission_id=self.mission.mission_id,
                error=str(e),
                exc_info=True
            )
            self.mission_status = MissionStatus.FAILED
            self.end_time = datetime.utcnow()

            return await self._generate_final_report()

    async def _process_ready_stages(self):
        """
        Process all stages that are ready to execute
        """
        if not self.mission:
            return

        # Get independent stages that haven't started
        independent_stages = self.mission.get_independent_stages()

        for stage in independent_stages:
            if self.stage_statuses[stage.stage_id] == StageStatus.PENDING:
                await self._execute_stage(stage)

        # Get stages whose dependencies are met
        for completed_stage_id in list(self.completed_stages):
            dependent_stages = self.mission.get_dependent_stages(completed_stage_id)

            for stage in dependent_stages:
                if self.stage_statuses[stage.stage_id] == StageStatus.PENDING:
                    # Check if all dependencies are met
                    if self._are_dependencies_met(stage):
                        await self._execute_stage(stage)

    def _are_dependencies_met(self, stage: MissionStage) -> bool:
        """
        Check if all dependencies for a stage are met

        Args:
            stage: MissionStage to check

        Returns:
            True if dependencies are met
        """
        if not stage.depends_on:
            return True

        return stage.depends_on in self.completed_stages

    async def _execute_stage(self, stage: MissionStage):
        """
        Execute a mission stage by dispatching tasks

        Args:
            stage: MissionStage to execute
        """
        logger.info(
            "Executing stage",
            stage_id=stage.stage_id,
            description=stage.description
        )

        self.stage_statuses[stage.stage_id] = StageStatus.IN_PROGRESS

        # Dispatch tasks for this stage
        for task in stage.tasks:
            task_id = await self._dispatch_task(stage.stage_id, task)

            self.dispatched_tasks[task_id] = {
                "stage_id": stage.stage_id,
                "agent_type": task.agent_type,
                "dispatched_at": datetime.utcnow().isoformat()
            }

            self.pending_reports.add(task_id)

    async def _dispatch_task(self, stage_id: str, task: MissionTask) -> str:
        """
        Dispatch a task to an agent

        Args:
            stage_id: ID of the stage this task belongs to
            task: MissionTask to dispatch

        Returns:
            Task ID
        """
        # Prepare task payload
        task_payload = dict(task.params or {})

        # Resolve parameters from blackboard if specified
        if task.params_from_blackboard:
            for param_name, blackboard_key in task.params_from_blackboard.items():
                value = self.blackboard.get(blackboard_key)

                if value is not None:
                    task_payload[param_name] = value
                else:
                    logger.warning(
                        "Blackboard key not found for task param",
                        param_name=param_name,
                        blackboard_key=blackboard_key
                    )

        # Add stage context
        task_payload["stage_id"] = stage_id
        task_payload["mission_id"] = self.mission.mission_id

        # Publish task to agent queue
        task_id = self.blackboard.publish_task(task.agent_type, task_payload)

        logger.info(
            "Task dispatched",
            task_id=task_id,
            stage_id=stage_id,
            agent_type=task.agent_type
        )

        return task_id

    async def _process_reports(self):
        """
        Process incoming agent reports
        """
        # Check for reports (non-blocking)
        report = self.blackboard.get_report(timeout=0)

        if not report:
            return

        task_id = report.get('task_id')
        agent_id = report.get('agent_id')
        status = report.get('status')
        data = report.get('data', {})

        logger.info(
            "Processing report",
            task_id=task_id,
            agent_id=agent_id,
            status=status
        )

        if task_id not in self.dispatched_tasks:
            logger.warning("Received report for unknown task", task_id=task_id)
            return

        # Get stage for this task
        stage_id = self.dispatched_tasks[task_id]['stage_id']
        stage = self.mission.get_stage(stage_id)

        if not stage:
            logger.error("Stage not found for task", stage_id=stage_id, task_id=task_id)
            return

        # Store report data in blackboard
        await self._store_report_data(task_id, agent_id, data)

        # Remove from pending
        self.pending_reports.discard(task_id)

        # Check if stage is complete
        await self._check_stage_completion(stage)

    async def _store_report_data(self, task_id: str, agent_id: str, data: Dict[str, Any]):
        """
        Store report data in the blackboard

        Args:
            task_id: Task ID
            agent_id: Agent ID
            data: Report data
        """
        # Store report
        report_key = f"reports:{task_id}"
        self.blackboard.set(report_key, {
            "agent_id": agent_id,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Update blackboard with discovered data
        for key, value in data.items():
            # Store discovered data with proper keys
            if isinstance(value, list) and key.startswith("discovered_"):
                for item in value:
                    self.blackboard.append_to_list(key, item)
            else:
                existing = self.blackboard.get(key)
                if existing is None:
                    self.blackboard.set(key, value)

    async def _check_stage_completion(self, stage: MissionStage):
        """
        Check if a stage has completed successfully

        Args:
            stage: MissionStage to check
        """
        # Check if all tasks for this stage have reported
        stage_tasks = [
            task_id for task_id, info in self.dispatched_tasks.items()
            if info['stage_id'] == stage.stage_id
        ]

        pending_stage_tasks = [tid for tid in stage_tasks if tid in self.pending_reports]

        if pending_stage_tasks:
            # Still waiting for reports
            return

        # All tasks reported, check success criteria
        success = await self._evaluate_success_criteria(stage)

        if success:
            logger.info(
                "Stage completed successfully",
                stage_id=stage.stage_id
            )
            self.stage_statuses[stage.stage_id] = StageStatus.COMPLETED
            self.completed_stages.add(stage.stage_id)
        else:
            logger.error(
                "Stage failed to meet success criteria",
                stage_id=stage.stage_id
            )
            self.stage_statuses[stage.stage_id] = StageStatus.FAILED
            self.failed_stages.add(stage.stage_id)

    async def _evaluate_success_criteria(self, stage: MissionStage) -> bool:
        """
        Evaluate success criteria for a stage

        Args:
            stage: MissionStage to evaluate

        Returns:
            True if all criteria are met
        """
        for criterion in stage.success_criteria:
            if not await self._evaluate_criterion(criterion):
                logger.warning(
                    "Success criterion not met",
                    stage_id=stage.stage_id,
                    criterion=criterion
                )
                return False

        return True

    async def _evaluate_criterion(self, criterion: str) -> bool:
        """
        Evaluate a single success criterion

        Args:
            criterion: Criterion string

        Returns:
            True if criterion is met
        """
        # Simple criterion evaluation
        # Format: "blackboard_key_exists:key_name"
        if criterion.startswith("blackboard_key_exists:"):
            key = criterion.split(":", 1)[1]
            return self.blackboard.exists(key)

        # Format: "num_known_hosts >= 1"
        if "num_" in criterion and (">=" in criterion or "<=" in criterion or ">" in criterion or "<" in criterion):
            parts = criterion.split()
            if len(parts) >= 3:
                key = parts[0].replace("num_", "")
                operator = parts[1]
                threshold = int(parts[2])

                value = self.blackboard.get(key)

                if value is None:
                    return False

                # Get count
                if isinstance(value, list):
                    count = len(value)
                elif isinstance(value, (int, float)):
                    count = value
                else:
                    return False

                # Evaluate operator
                if operator == ">=":
                    return count >= threshold
                elif operator == "<=":
                    return count <= threshold
                elif operator == ">":
                    return count > threshold
                elif operator == "<":
                    return count < threshold

        # Custom criterion (just check if key exists and is truthy)
        if self.blackboard.exists(criterion):
            value = self.blackboard.get(criterion)
            return bool(value)

        return False

    async def _generate_final_report(self) -> Dict[str, Any]:
        """
        Generate final mission report

        Returns:
            Report dictionary
        """
        if not self.mission:
            return {"error": "No mission loaded"}

        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()

        report = {
            "mission_id": self.mission.mission_id,
            "mission_name": self.mission.mission_name,
            "status": self.mission_status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": duration,
            "stages": {},
            "discovered_data": {},
            "summary": ""
        }

        # Add stage information
        for stage in self.mission.objective_stages:
            stage_id = stage.stage_id

            report["stages"][stage_id] = {
                "description": stage.description,
                "status": self.stage_statuses[stage_id].value,
                "tasks_dispatched": len([
                    tid for tid, info in self.dispatched_tasks.items()
                    if info['stage_id'] == stage_id
                ])
            }

        # Collect discovered data from blackboard
        all_keys = self.blackboard.get_all_keys()

        for key in all_keys:
            if not key.startswith("reports:"):
                value = self.blackboard.get(key)
                report["discovered_data"][key] = value

        # Generate summary
        completed_count = len(self.completed_stages)
        total_count = len(self.mission.objective_stages)

        report["summary"] = (
            f"Mission {self.mission_status.value}. "
            f"Completed {completed_count}/{total_count} stages. "
            f"Duration: {duration:.2f}s" if duration else "Unknown duration"
        )

        # Store report in blackboard
        self.blackboard.set_mission_data(
            self.mission.mission_id,
            "final_report",
            report
        )

        logger.info(
            "Final mission report generated",
            mission_id=self.mission.mission_id,
            status=self.mission_status.value
        )

        return report

    def abort_mission(self) -> bool:
        """
        Abort the current mission

        Returns:
            Success boolean
        """
        if not self.mission:
            return False

        logger.warning("Mission aborted by coordinator", mission_id=self.mission.mission_id)

        self.mission_status = MissionStatus.ABORTED
        self.end_time = datetime.utcnow()

        return True

    def get_status(self) -> Dict[str, Any]:
        """
        Get current mission status

        Returns:
            Status dictionary
        """
        if not self.mission:
            return {"status": "no_mission_loaded"}

        return {
            "mission_id": self.mission.mission_id,
            "mission_name": self.mission.mission_name,
            "status": self.mission_status.value,
            "completed_stages": len(self.completed_stages),
            "total_stages": len(self.mission.objective_stages),
            "failed_stages": len(self.failed_stages),
            "pending_reports": len(self.pending_reports),
            "stage_statuses": {
                stage_id: status.value
                for stage_id, status in self.stage_statuses.items()
            }
        }
