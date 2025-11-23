"""
Dynamic CPU Manager - Adaptive Resource Allocation

Implements intelligent CPU power allocation for database operations:
- Only uses resources when needed
- Scales up for intensive operations
- Scales down during idle periods
- Monitors system load and adjusts accordingly

This mimics how the brain allocates cognitive resources dynamically.
"""

import asyncio
import multiprocessing
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Any, Awaitable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import structlog

logger = structlog.get_logger()


class WorkloadIntensity(Enum):
    """Classification of workload intensity"""
    IDLE = 0        # No active work
    LIGHT = 1       # Simple queries, single operations
    MODERATE = 2    # Multiple concurrent operations
    HEAVY = 3       # Batch processing, migrations
    INTENSIVE = 4   # Full system operations, rebuilds


@dataclass
class ResourceAllocation:
    """Current resource allocation state"""
    cpu_cores: int
    thread_pool_size: int
    process_pool_size: int
    max_concurrent_tasks: int
    batch_size: int
    priority: int = 5  # 1-10 scale


@dataclass
class TaskMetrics:
    """Metrics for a single task execution"""
    task_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    cpu_time_ms: float = 0
    wall_time_ms: float = 0
    memory_bytes: int = 0
    intensity: WorkloadIntensity = WorkloadIntensity.LIGHT


class DynamicCPUManager:
    """
    Manages CPU resource allocation dynamically based on workload.

    Features:
    - Automatic scaling of thread/process pools
    - Task queue with priority scheduling
    - Resource monitoring and adaptation
    - Burst handling for intensive operations
    """

    def __init__(
        self,
        min_workers: int = 1,
        max_workers: Optional[int] = None,
        target_cpu_utilization: float = 0.7,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        adaptation_interval_seconds: float = 5.0
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.target_utilization = target_cpu_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.adaptation_interval = adaptation_interval_seconds

        # Current allocation
        self.current_allocation = ResourceAllocation(
            cpu_cores=min_workers,
            thread_pool_size=min_workers * 2,
            process_pool_size=min_workers,
            max_concurrent_tasks=min_workers * 4,
            batch_size=100
        )

        # Executors
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None

        # Task tracking
        self.active_tasks: Dict[str, TaskMetrics] = {}
        self.completed_tasks: List[TaskMetrics] = []
        self.max_completed_history = 1000

        # Workload tracking
        self.current_workload = WorkloadIntensity.IDLE
        self.workload_history: List[tuple] = []  # (timestamp, intensity)

        # Monitoring
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Statistics
        self.stats = {
            "total_tasks": 0,
            "total_cpu_time_ms": 0,
            "scale_ups": 0,
            "scale_downs": 0,
            "peak_workers": min_workers,
            "avg_response_time_ms": 0,
        }

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(self.current_allocation.max_concurrent_tasks)

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        logger.info("DynamicCPUManager initialized",
                   min_workers=min_workers,
                   max_workers=self.max_workers,
                   target_utilization=target_cpu_utilization)

    async def start(self):
        """Start the CPU manager and monitoring"""
        self._initialize_pools()
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("DynamicCPUManager started")

    async def stop(self):
        """Stop the CPU manager and cleanup"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
        if self._process_pool:
            self._process_pool.shutdown(wait=True)

        logger.info("DynamicCPUManager stopped")

    def _initialize_pools(self):
        """Initialize thread and process pools"""
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self.current_allocation.thread_pool_size,
            thread_name_prefix="neural_storage_"
        )
        self._process_pool = ProcessPoolExecutor(
            max_workers=self.current_allocation.process_pool_size
        )

    async def _monitor_loop(self):
        """Background monitoring and adaptation loop"""
        while self._monitoring:
            try:
                await asyncio.sleep(self.adaptation_interval)
                await self._adapt_resources()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Monitor loop error", error=str(e))

    async def _adapt_resources(self):
        """Adapt resources based on current workload"""
        async with self._lock:
            # Calculate current utilization
            active_count = len(self.active_tasks)
            max_tasks = self.current_allocation.max_concurrent_tasks
            utilization = active_count / max_tasks if max_tasks > 0 else 0

            # Determine workload intensity
            new_intensity = self._classify_workload(active_count, utilization)

            if new_intensity != self.current_workload:
                self.workload_history.append((datetime.utcnow(), new_intensity))
                self.current_workload = new_intensity

            # Scale decision
            if utilization > self.scale_up_threshold:
                await self._scale_up()
            elif utilization < self.scale_down_threshold and active_count == 0:
                await self._scale_down()

    def _classify_workload(self, active_count: int, utilization: float) -> WorkloadIntensity:
        """Classify current workload intensity"""
        if active_count == 0:
            return WorkloadIntensity.IDLE
        elif utilization < 0.25:
            return WorkloadIntensity.LIGHT
        elif utilization < 0.5:
            return WorkloadIntensity.MODERATE
        elif utilization < 0.8:
            return WorkloadIntensity.HEAVY
        else:
            return WorkloadIntensity.INTENSIVE

    async def _scale_up(self):
        """Scale up resources"""
        current_cores = self.current_allocation.cpu_cores
        new_cores = min(current_cores + 1, self.max_workers)

        if new_cores > current_cores:
            self.current_allocation.cpu_cores = new_cores
            self.current_allocation.thread_pool_size = new_cores * 2
            self.current_allocation.process_pool_size = new_cores
            self.current_allocation.max_concurrent_tasks = new_cores * 4

            # Recreate pools with new size
            self._reinitialize_pools()

            self.stats["scale_ups"] += 1
            self.stats["peak_workers"] = max(self.stats["peak_workers"], new_cores)

            logger.info("Scaled up resources",
                       cores=new_cores,
                       thread_pool=self.current_allocation.thread_pool_size)

    async def _scale_down(self):
        """Scale down resources"""
        current_cores = self.current_allocation.cpu_cores
        new_cores = max(current_cores - 1, self.min_workers)

        if new_cores < current_cores:
            self.current_allocation.cpu_cores = new_cores
            self.current_allocation.thread_pool_size = new_cores * 2
            self.current_allocation.process_pool_size = new_cores
            self.current_allocation.max_concurrent_tasks = new_cores * 4

            # Recreate pools with new size
            self._reinitialize_pools()

            self.stats["scale_downs"] += 1

            logger.info("Scaled down resources",
                       cores=new_cores,
                       thread_pool=self.current_allocation.thread_pool_size)

    def _reinitialize_pools(self):
        """Reinitialize pools with new sizes"""
        # Shutdown old pools gracefully
        if self._thread_pool:
            self._thread_pool.shutdown(wait=False)
        if self._process_pool:
            self._process_pool.shutdown(wait=False)

        # Create new pools
        self._initialize_pools()

        # Update semaphore
        self._semaphore = asyncio.Semaphore(self.current_allocation.max_concurrent_tasks)

    async def execute_task(
        self,
        task_id: str,
        func: Callable,
        *args,
        intensity: WorkloadIntensity = WorkloadIntensity.LIGHT,
        use_process_pool: bool = False,
        **kwargs
    ) -> Any:
        """
        Execute a task with dynamic resource allocation.

        Args:
            task_id: Unique identifier for the task
            func: Function to execute
            intensity: Expected workload intensity
            use_process_pool: Use process pool for CPU-bound work
        """
        await self._semaphore.acquire()

        metrics = TaskMetrics(
            task_id=task_id,
            start_time=datetime.utcnow(),
            intensity=intensity
        )
        self.active_tasks[task_id] = metrics

        try:
            start_cpu = time.process_time()
            start_wall = time.perf_counter()

            if use_process_pool and self._process_pool:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._process_pool, func, *args
                )
            elif self._thread_pool:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._thread_pool, func, *args
                )
            else:
                # Fallback to direct execution
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

            end_cpu = time.process_time()
            end_wall = time.perf_counter()

            metrics.cpu_time_ms = (end_cpu - start_cpu) * 1000
            metrics.wall_time_ms = (end_wall - start_wall) * 1000
            metrics.end_time = datetime.utcnow()

            self.stats["total_tasks"] += 1
            self.stats["total_cpu_time_ms"] += metrics.cpu_time_ms

            return result

        finally:
            del self.active_tasks[task_id]
            self.completed_tasks.append(metrics)
            if len(self.completed_tasks) > self.max_completed_history:
                self.completed_tasks.pop(0)
            self._semaphore.release()

    async def execute_batch(
        self,
        tasks: List[tuple],  # [(task_id, func, args), ...]
        intensity: WorkloadIntensity = WorkloadIntensity.HEAVY,
        max_concurrent: Optional[int] = None
    ) -> List[Any]:
        """
        Execute a batch of tasks with controlled concurrency.
        Automatically scales resources for batch operations.
        """
        if not tasks:
            return []

        # Scale up for batch operations
        if intensity.value >= WorkloadIntensity.HEAVY.value:
            await self._scale_up()

        max_concurrent = max_concurrent or self.current_allocation.max_concurrent_tasks
        batch_semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_semaphore(task_id, func, args):
            async with batch_semaphore:
                return await self.execute_task(task_id, func, *args, intensity=intensity)

        # Execute all tasks concurrently with semaphore control
        coros = [
            execute_with_semaphore(task_id, func, args)
            for task_id, func, args in tasks
        ]

        results = await asyncio.gather(*coros, return_exceptions=True)
        return results

    def get_optimal_batch_size(self, total_items: int) -> int:
        """Calculate optimal batch size based on current resources"""
        base_batch = self.current_allocation.batch_size
        workers = self.current_allocation.cpu_cores

        # Adjust based on total items
        if total_items < base_batch:
            return total_items

        # Ensure batches are evenly distributed across workers
        optimal = max(
            base_batch,
            (total_items + workers - 1) // workers
        )

        return min(optimal, base_batch * 2)

    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource statistics"""
        # Calculate average response time from recent tasks
        recent_tasks = self.completed_tasks[-100:]
        if recent_tasks:
            avg_response = sum(t.wall_time_ms for t in recent_tasks) / len(recent_tasks)
        else:
            avg_response = 0

        self.stats["avg_response_time_ms"] = avg_response

        return {
            "allocation": {
                "cpu_cores": self.current_allocation.cpu_cores,
                "thread_pool_size": self.current_allocation.thread_pool_size,
                "process_pool_size": self.current_allocation.process_pool_size,
                "max_concurrent_tasks": self.current_allocation.max_concurrent_tasks,
                "batch_size": self.current_allocation.batch_size,
            },
            "workload": {
                "current_intensity": self.current_workload.name,
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
            },
            "performance": {
                "total_tasks": self.stats["total_tasks"],
                "total_cpu_time_ms": self.stats["total_cpu_time_ms"],
                "avg_response_time_ms": avg_response,
                "scale_ups": self.stats["scale_ups"],
                "scale_downs": self.stats["scale_downs"],
                "peak_workers": self.stats["peak_workers"],
            },
            "system": {
                "available_cores": multiprocessing.cpu_count(),
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
            }
        }

    async def request_burst(self, duration_seconds: float = 10.0):
        """
        Request temporary burst capacity for intensive operations.
        Scales to maximum resources temporarily.
        """
        logger.info("Burst capacity requested", duration=duration_seconds)

        # Store original allocation
        original_cores = self.current_allocation.cpu_cores

        # Scale to maximum
        self.current_allocation.cpu_cores = self.max_workers
        self.current_allocation.thread_pool_size = self.max_workers * 2
        self.current_allocation.process_pool_size = self.max_workers
        self.current_allocation.max_concurrent_tasks = self.max_workers * 4

        self._reinitialize_pools()

        # Schedule return to normal
        async def restore_allocation():
            await asyncio.sleep(duration_seconds)
            async with self._lock:
                self.current_allocation.cpu_cores = original_cores
                self.current_allocation.thread_pool_size = original_cores * 2
                self.current_allocation.process_pool_size = original_cores
                self.current_allocation.max_concurrent_tasks = original_cores * 4
                self._reinitialize_pools()
                logger.info("Burst capacity ended, restored normal allocation")

        asyncio.create_task(restore_allocation())

    def set_priority(self, priority: int):
        """Set processing priority (1-10, higher = more resources)"""
        priority = max(1, min(10, priority))
        self.current_allocation.priority = priority

        # Adjust resources based on priority
        priority_factor = priority / 5.0  # 1.0 at priority 5
        target_cores = int(self.min_workers +
                         (self.max_workers - self.min_workers) * (priority_factor - 0.2))
        target_cores = max(self.min_workers, min(self.max_workers, target_cores))

        if target_cores != self.current_allocation.cpu_cores:
            self.current_allocation.cpu_cores = target_cores
            self.current_allocation.thread_pool_size = target_cores * 2
            self.current_allocation.process_pool_size = target_cores
            self.current_allocation.max_concurrent_tasks = target_cores * 4
            self._reinitialize_pools()

            logger.info("Priority changed", priority=priority, cores=target_cores)
