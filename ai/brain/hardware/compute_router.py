#!/usr/bin/env python3
"""
Compute Router for DSMIL Brain

Optimal task routing based on hardware resources:
- Match tasks to appropriate compute resources
- Load balancing across resources
- Priority-based scheduling
"""

import threading
import logging
import heapq
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime, timezone
from enum import Enum, auto

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of compute resources"""
    CPU = auto()
    GPU = auto()
    TPU = auto()
    NETWORK = auto()
    STORAGE = auto()


@dataclass
class TaskRequirements:
    """Requirements for a task"""
    task_id: str

    # Resource needs
    cpu_cores: int = 1
    memory_mb: int = 512
    gpu_required: bool = False
    network_required: bool = False

    # Priority (lower = higher priority)
    priority: int = 100

    # Timing
    estimated_duration_s: float = 1.0
    deadline: Optional[datetime] = None


@dataclass
class RoutingDecision:
    """Decision on how to route a task"""
    task_id: str

    # Routing
    resource_type: ResourceType
    node_id: str = ""

    # Scheduling
    priority: int = 100
    estimated_start: Optional[datetime] = None
    estimated_end: Optional[datetime] = None

    # Reasoning
    reason: str = ""


class ComputeRouter:
    """
    Compute Router

    Routes tasks to optimal compute resources.

    Usage:
        router = ComputeRouter()

        # Register resources
        router.register_resource("node-1", ResourceType.CPU, capacity=8)

        # Route task
        decision = router.route(task_requirements)

        # Execute on decided resource
        result = router.execute(decision, task_fn)
    """

    def __init__(self):
        self._resources: Dict[str, Dict] = {}  # node_id -> {type, capacity, used}
        self._task_queue: List[tuple] = []  # Priority queue (priority, task_id, requirements)
        self._active_tasks: Dict[str, RoutingDecision] = {}

        self._lock = threading.RLock()

        logger.info("ComputeRouter initialized")

    def register_resource(self, node_id: str, resource_type: ResourceType,
                         capacity: int = 1, memory_mb: int = 0):
        """Register a compute resource"""
        with self._lock:
            self._resources[node_id] = {
                "type": resource_type,
                "capacity": capacity,
                "used": 0,
                "memory_mb": memory_mb,
                "memory_used_mb": 0,
            }
            logger.info(f"Registered resource: {node_id} ({resource_type.name})")

    def unregister_resource(self, node_id: str):
        """Unregister a resource"""
        with self._lock:
            self._resources.pop(node_id, None)

    def route(self, requirements: TaskRequirements) -> RoutingDecision:
        """
        Route a task to optimal resource
        """
        with self._lock:
            # Determine required resource type
            if requirements.gpu_required:
                required_type = ResourceType.GPU
            else:
                required_type = ResourceType.CPU

            # Find best matching resource
            best_node = None
            best_score = -1

            for node_id, resource in self._resources.items():
                if resource["type"] != required_type:
                    continue

                # Check capacity
                available = resource["capacity"] - resource["used"]
                if available < requirements.cpu_cores:
                    continue

                # Check memory
                mem_available = resource["memory_mb"] - resource["memory_used_mb"]
                if mem_available < requirements.memory_mb:
                    continue

                # Score based on availability
                score = available / resource["capacity"]

                if score > best_score:
                    best_score = score
                    best_node = node_id

            if not best_node:
                # No suitable resource found
                return RoutingDecision(
                    task_id=requirements.task_id,
                    resource_type=required_type,
                    reason="No suitable resource available",
                )

            # Create decision
            decision = RoutingDecision(
                task_id=requirements.task_id,
                resource_type=required_type,
                node_id=best_node,
                priority=requirements.priority,
                estimated_start=datetime.now(timezone.utc),
                reason=f"Best available resource (score={best_score:.2f})",
            )

            # Reserve resources
            self._resources[best_node]["used"] += requirements.cpu_cores
            self._resources[best_node]["memory_used_mb"] += requirements.memory_mb

            self._active_tasks[requirements.task_id] = decision

            return decision

    def release(self, task_id: str, requirements: TaskRequirements):
        """Release resources after task completion"""
        with self._lock:
            decision = self._active_tasks.pop(task_id, None)
            if decision and decision.node_id in self._resources:
                self._resources[decision.node_id]["used"] -= requirements.cpu_cores
                self._resources[decision.node_id]["memory_used_mb"] -= requirements.memory_mb

    def execute(self, decision: RoutingDecision,
               task_fn: Callable, *args, **kwargs) -> Any:
        """
        Execute task on routed resource

        Note: In real implementation, would dispatch to remote node.
        """
        if not decision.node_id:
            raise ValueError("No resource assigned")

        try:
            result = task_fn(*args, **kwargs)
            return result
        finally:
            pass  # Resources released by caller

    def queue_task(self, requirements: TaskRequirements):
        """Add task to priority queue"""
        with self._lock:
            heapq.heappush(
                self._task_queue,
                (requirements.priority, requirements.task_id, requirements)
            )

    def get_next_task(self) -> Optional[TaskRequirements]:
        """Get next task from queue"""
        with self._lock:
            if self._task_queue:
                _, _, requirements = heapq.heappop(self._task_queue)
                return requirements
            return None

    def get_resource_status(self) -> Dict:
        """Get status of all resources"""
        with self._lock:
            status = {}
            for node_id, resource in self._resources.items():
                status[node_id] = {
                    "type": resource["type"].name,
                    "capacity": resource["capacity"],
                    "used": resource["used"],
                    "available": resource["capacity"] - resource["used"],
                    "utilization": resource["used"] / max(resource["capacity"], 1),
                }
            return status

    def get_stats(self) -> Dict:
        """Get router statistics"""
        with self._lock:
            total_capacity = sum(r["capacity"] for r in self._resources.values())
            total_used = sum(r["used"] for r in self._resources.values())

            return {
                "resources": len(self._resources),
                "total_capacity": total_capacity,
                "total_used": total_used,
                "utilization": total_used / max(total_capacity, 1),
                "queued_tasks": len(self._task_queue),
                "active_tasks": len(self._active_tasks),
            }


if __name__ == "__main__":
    print("Compute Router Self-Test")
    print("=" * 50)

    router = ComputeRouter()

    print("\n[1] Register Resources")
    router.register_resource("cpu-node-1", ResourceType.CPU, capacity=8, memory_mb=16384)
    router.register_resource("cpu-node-2", ResourceType.CPU, capacity=4, memory_mb=8192)
    router.register_resource("gpu-node-1", ResourceType.GPU, capacity=1, memory_mb=8192)
    print("    Registered 3 resources")

    print("\n[2] Route CPU Task")
    task1 = TaskRequirements(
        task_id="task-1",
        cpu_cores=2,
        memory_mb=1024,
        priority=50,
    )
    decision = router.route(task1)
    print(f"    Task: {decision.task_id}")
    print(f"    Routed to: {decision.node_id}")
    print(f"    Reason: {decision.reason}")

    print("\n[3] Route GPU Task")
    task2 = TaskRequirements(
        task_id="task-2",
        cpu_cores=1,
        memory_mb=4096,
        gpu_required=True,
    )
    decision = router.route(task2)
    print(f"    Task: {decision.task_id}")
    print(f"    Routed to: {decision.node_id}")

    print("\n[4] Resource Status")
    status = router.get_resource_status()
    for node_id, info in status.items():
        print(f"    {node_id}: {info['type']}, used={info['used']}/{info['capacity']}")

    print("\n[5] Release Resources")
    router.release("task-1", task1)
    print("    Released task-1 resources")

    print("\n[6] Queue Tasks")
    for i in range(3):
        router.queue_task(TaskRequirements(
            task_id=f"queued-{i}",
            cpu_cores=1,
            priority=100 - i * 10,  # Higher priority for later tasks
        ))
    print("    Queued 3 tasks")

    next_task = router.get_next_task()
    print(f"    Next task: {next_task.task_id} (priority={next_task.priority})")

    print("\n[7] Statistics")
    stats = router.get_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.2%}")
        else:
            print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Compute Router test complete")

