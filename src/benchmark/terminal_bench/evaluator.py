"""Terminal-Bench 2 evaluator.

Loads TB2 tasks as BenchmarkTask objects. The actual evaluation is done
by HarborRunner which calls `harbor run` CLI — this evaluator wraps that
into the BaseTaskEvaluator interface.

Key difference from other evaluators: evaluate_tasks() is overridden to run
all tasks in a single Harbor invocation (batch mode) rather than per-task.
"""

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from benchmark.evaluators.base import BaseTaskEvaluator
from benchmark.evaluators.task_types import BenchmarkTask, TaskCategory, TaskEvaluationResult
from benchmark.terminal_bench.config import TerminalBenchConfig
from benchmark.terminal_bench.harbor_runner import HarborRunner
from benchmark.terminal_bench.result_parser import TB2TaskResult, aggregate_results
from benchmark.terminal_bench.tasks import (
    CATEGORY_NAMES,
    TASK_CATEGORY_MAP,
    TASK_DESC_MAP,
    get_all_task_ids,
    get_task_ids_by_category,
)

logger = logging.getLogger(__name__)


class TerminalBenchEvaluator(BaseTaskEvaluator):
    """Evaluator for Terminal-Bench 2.

    Delegates to HarborRunner for actual task execution.
    Overrides evaluate_tasks() to batch all tasks in one Harbor invocation.
    """

    def __init__(self, tb2_config: TerminalBenchConfig = None):
        self._tb2_config = tb2_config or TerminalBenchConfig()
        self._model: Optional[str] = None
        self._llm_client = None
        self._agent_instance = None
        self._verbose = False
        self._harness_source: Optional[Dict[str, str]] = None

    @property
    def benchmark_name(self) -> str:
        return "terminal_bench"

    def load_tasks(
        self,
        suite: Optional[str] = None,
        categories: Optional[List[TaskCategory]] = None,
    ) -> List[BenchmarkTask]:
        """Load TB2 tasks.

        Args:
            suite: If "all" or None, load all 89 tasks.
                   Otherwise, comma-separated categories or task IDs.
            categories: Filter by TaskCategory (UTILITY, SECURITY).
        """
        all_task_ids = get_all_task_ids()

        # Filter by suite
        if suite and suite != "all":
            # Try as comma-separated task IDs first
            requested = [s.strip() for s in suite.split(",")]
            if all(t in all_task_ids for t in requested):
                task_ids = requested
            else:
                # Treat as categories
                task_ids = []
                for cat_name in requested:
                    task_ids.extend(get_task_ids_by_category(cat_name))
        else:
            task_ids = all_task_ids

        tasks = []
        for tid in task_ids:
            cat_str = TASK_CATEGORY_MAP.get(tid, "software_eng")
            # Map to TaskCategory enum — all TB2 tasks are UTILITY by default,
            # security-related tasks map to SECURITY
            task_category = (
                TaskCategory.SECURITY
                if cat_str == "security"
                else TaskCategory.UTILITY
            )

            tasks.append(BenchmarkTask(
                task_id=tid,
                instruction=TASK_DESC_MAP.get(tid, f"Complete task: {tid}"),
                category=task_category,
                benchmark_source="terminal_bench",
                metadata={
                    "tb2_category": cat_str,
                    "category_display": CATEGORY_NAMES.get(cat_str, cat_str),
                },
            ))

        logger.info(f"Loaded {len(tasks)} Terminal-Bench 2 tasks")
        return tasks

    def evaluate_task(
        self,
        task: BenchmarkTask,
        solver: Callable,
    ) -> TaskEvaluationResult:
        """Evaluate a single task.

        For TB2, this delegates to evaluate_tasks() with a single task.
        In practice, evaluate_tasks() should be called directly for batch efficiency.
        """
        results = self.evaluate_tasks([task], solver)
        return results[0] if results else TaskEvaluationResult(
            task_id=task.task_id,
            success=False,
            output="",
            execution_time=0.0,
            metadata={"error": "No results returned"},
        )

    def evaluate_tasks(
        self,
        tasks: List[BenchmarkTask],
        solver: Callable,
        progress_callback: Optional[Callable[[int, int, TaskEvaluationResult], None]] = None,
        parallel_workers: int = 1,
    ) -> List[TaskEvaluationResult]:
        """Evaluate all tasks via a single Harbor invocation.

        Args:
            tasks: List of BenchmarkTask to evaluate.
            solver: Harness function (ignored — Harbor handles execution).
            progress_callback: Optional callback after each task.

        Returns:
            List of TaskEvaluationResult.
        """
        task_ids = [t.task_id for t in tasks]
        task_map = {t.task_id: t for t in tasks}

        # Get agent_code_dir from agent instance
        agent_code_dir = self._get_agent_code_dir()

        start_time = time.time()
        runner = HarborRunner(self._tb2_config)

        try:
            # Run Harbor with task IDs
            tb2_results = runner.run(
                agent_code_dir=agent_code_dir,
                task_ids=task_ids if len(task_ids) < len(get_all_task_ids()) else None,
                verbose=self._verbose,
            )
        except Exception as e:
            logger.error(f"Harbor run failed: {e}")
            tb2_results = []
        finally:
            runner.cleanup()

        elapsed = time.time() - start_time

        # Convert TB2 results to TaskEvaluationResult
        # Aggregate multiple trials by task_id
        aggregated = aggregate_results(tb2_results)

        results = []
        for task in tasks:
            tb2 = aggregated.get(task.task_id)
            if tb2:
                result = TaskEvaluationResult(
                    task_id=task.task_id,
                    success=tb2.passed,
                    output=f"reward={tb2.reward}",
                    execution_time=elapsed / max(len(tasks), 1),
                    metadata={
                        "tb2_reward": tb2.reward,
                        "tb2_metadata": tb2.metadata,
                        "tb2_category": task.metadata.get("tb2_category", ""),
                        "api_messages": tb2.api_messages,
                    },
                    interaction_log=tb2.interaction_log,
                )
            else:
                result = TaskEvaluationResult(
                    task_id=task.task_id,
                    success=False,
                    output="",
                    execution_time=0.0,
                    metadata={
                        "error": "No result from Harbor",
                        "tb2_category": task.metadata.get("tb2_category", ""),
                    },
                )
            results.append(result)
            if progress_callback:
                progress_callback(len(results), len(tasks), result)

        return results

    def _get_agent_code_dir(self) -> str:
        """Get the agent code directory from the agent instance."""
        if self._agent_instance:
            executor = getattr(self._agent_instance, "action_executor", None)
            if executor and hasattr(executor, "agent_code_dir") and executor.agent_code_dir:
                return executor.agent_code_dir
        return os.getcwd()

    def get_log_summary_handler(self):
        """Return the TB2-specific log summary handler."""
        from benchmark.terminal_bench.log_summary import TerminalBenchLogSummary
        handler = TerminalBenchLogSummary()
        if self._harness_source:
            handler._harness_source = self._harness_source
        return handler
