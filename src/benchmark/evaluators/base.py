"""
Base Task Evaluator for Benchmark Evaluation.

This module provides the abstract base class for all benchmark evaluators.
Each benchmark (AgentDojo, AgentBench, etc.) should implement their own
evaluator by subclassing BaseTaskEvaluator.
"""

import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from react_loop.utils.task_context import init_task_context, clear_task_context

from .task_types import BenchmarkTask, TaskEvaluationResult, TaskCategory

if TYPE_CHECKING:
    from .log_summary_base import LogSummaryBase


class BaseTaskEvaluator(ABC):
    """Abstract base class for benchmark task evaluators.

    This class defines the interface that all benchmark evaluators must
    implement. It provides common functionality for task loading and evaluation.

    Subclasses must implement:
        - benchmark_name: Property returning the benchmark name
        - load_tasks(): Method to load tasks from the benchmark
        - evaluate_task(): Method to evaluate a single task
    """

    @property
    @abstractmethod
    def benchmark_name(self) -> str:
        """Return the benchmark name this evaluator handles.

        Returns:
            Benchmark name (e.g., "agentdojo", "agentbench")
        """
        pass

    @property
    def supported_categories(self) -> List[TaskCategory]:
        """Return supported task categories.

        Returns:
            List of supported TaskCategory enums
        """
        return [TaskCategory.UTILITY, TaskCategory.SECURITY]

    @abstractmethod
    def load_tasks(
        self,
        suite: Optional[str] = None,
        categories: Optional[List[TaskCategory]] = None,
    ) -> List[BenchmarkTask]:
        """Load tasks from the benchmark.

        Args:
            suite: Benchmark suite name (e.g., "workspace")
            categories: Task categories to load (utility, security, or both)

        Returns:
            List of BenchmarkTask objects
        """
        pass

    @abstractmethod
    def evaluate_task(
        self,
        task: BenchmarkTask,
        solver: Callable,
    ) -> TaskEvaluationResult:
        """Evaluate a solver on a single task.

        Args:
            task: The task to evaluate
            solver: The solver function to evaluate

        Returns:
            TaskEvaluationResult with evaluation results
        """
        pass

    def evaluate_tasks(
        self,
        tasks: List[BenchmarkTask],
        solver: Callable,
        progress_callback: Optional[Callable[[int, int, TaskEvaluationResult], None]] = None,
        parallel_workers: int = 1,
    ) -> List[TaskEvaluationResult]:
        """Evaluate solver on multiple tasks.

        Args:
            tasks: List of tasks to evaluate
            solver: The solver function to evaluate
            progress_callback: Optional callback(current, total, result) called after each task
            parallel_workers: Number of parallel workers (1=serial, default)

        Returns:
            List of TaskEvaluationResult objects (in original task order)
        """
        total = len(tasks)

        if parallel_workers <= 1:
            results = []
            for i, task in enumerate(tasks, 1):
                init_task_context()
                try:
                    result = self.evaluate_task(task, solver)
                finally:
                    clear_task_context()
                results.append(result)
                if progress_callback:
                    progress_callback(i, total, result)
            return results

        # Parallel mode
        indexed_results: List[Optional[TaskEvaluationResult]] = [None] * total
        lock = threading.Lock()
        completed = [0]

        def _run(indexed_task):
            idx, task = indexed_task
            init_task_context()
            try:
                result = self.evaluate_task(task, solver)
            except Exception as e:
                import traceback
                result = TaskEvaluationResult(
                    task_id=task.task_id, success=False, output=None,
                    execution_time=0.0,
                    metadata={"error": f"{e}\n{traceback.format_exc()}", "category": task.category.value},
                    interaction_log=[],
                )
            finally:
                clear_task_context()
            with lock:
                completed[0] += 1
                indexed_results[idx] = result
                if progress_callback:
                    progress_callback(completed[0], total, result)

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [executor.submit(_run, (i, t)) for i, t in enumerate(tasks)]
            for f in futures:
                f.result()

        return indexed_results

    def get_log_summary_handler(self) -> Optional["LogSummaryBase"]:
        """Return the log summary handler for this benchmark.

        Subclasses can override this method to return a benchmark-specific
        log summary implementation. Default returns None, which means
        the generic fallback in agent_action.py will be used.

        Returns:
            A LogSummaryBase instance or None to use generic fallback
        """
        return None
