"""
Benchmark Evaluator Adapter (Base).

This module provides the abstract base adapter that converts benchmark evaluation
to the external evaluator format expected by the agent's evaluate action.

External evaluator signature: (agent_instance, test_cases) -> (reward, metrics)

Subclasses must implement:
    - _calculate_reward(): Calculate reward from evaluation results
"""

import json
import logging
import os
import random
import tempfile
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from benchmark.config import BenchmarkConfig
from benchmark.evaluators.task_types import TaskCategory, TaskEvaluationResult, BenchmarkTask
from benchmark.evaluators.registry import get_evaluator
from react_loop.utils.harness_loader import HarnessLoader, HarnessInfo

logger = logging.getLogger(__name__)


class BenchmarkEvaluatorAdapter(ABC):
    """Base adapter that wraps benchmark evaluators for use as external evaluators.

    This adapter allows benchmark evaluations to be used seamlessly with
    the agent's evaluate action. It fresh-imports the harness function from
    disk on every evaluate() call (unique package name per call) so the bytes
    that run == the bytes that get hashed, then cleans up sys.modules after.

    Subclasses must implement _calculate_reward().
    """

    def __init__(self, config: BenchmarkConfig):
        """Initialize the benchmark evaluator adapter.

        Args:
            config: BenchmarkConfig instance with all benchmark settings.
        """
        self.config = config
        self.benchmark_type = config.type
        self.suite = config.suite
        self.categories = config.categories
        self.max_tasks_per_category = config.max_tasks_per_category
        self.max_tasks_per_category_test = config.max_tasks_per_category_test
        self.verbose = config.verbose

        # Get the evaluator from registry
        self._evaluator = get_evaluator(config.type)
        if self._evaluator is None:
            raise ValueError(
                f"Benchmark evaluator '{config.type}' not found. "
                f"Available evaluators: {self._list_available_evaluators()}"
            )

        # Harness loader (agent_code_dir will be set lazily from agent_instance)
        self._harness_loader = HarnessLoader()

        # Dev/test split cache (populated on first evaluate call)
        self._dev_tasks: Optional[List[BenchmarkTask]] = None
        self._val_tasks: Optional[List[BenchmarkTask]] = None
        self._test_tasks: Optional[List[BenchmarkTask]] = None

        # Dynamic sampling: full non-test pool (not limited, not split)
        self._non_test_pool: Optional[List[BenchmarkTask]] = None
        self._is_dynamic: bool = False

        # Print lock for thread-safe progress output
        self._print_lock = threading.Lock()

    def _list_available_evaluators(self) -> List[str]:
        """List available evaluators."""
        from benchmark.evaluators.registry import list_evaluators
        return list_evaluators()

    def get_log_summary_handler(self) -> Optional[Any]:
        """Get the log summary handler from the evaluator.

        Returns:
            LogSummaryBase instance if available, None otherwise
        """
        if hasattr(self._evaluator, 'get_log_summary_handler'):
            return self._evaluator.get_log_summary_handler()
        return None

    def get_evolution_goal(self) -> Optional[str]:
        """Return the default evolution goal for this benchmark.

        Override in subclasses to provide a benchmark-specific goal.
        Returns None if no default goal is available.
        """
        return None

    def get_task_summary(self) -> Dict[str, Any]:
        """Return the number of tasks evaluated per mode (dev/val/test).

        Lazily loads and splits tasks if not already done. For dynamic
        sampling, dev_size reflects the actual number sampled per evaluate()
        call (limited by max_tasks_per_category).

        IMPORTANT: Callers must ensure _inject_llm_config() has been called
        before this method if _load_and_split_tasks() may be triggered,
        otherwise the evaluator will use its default config (e.g. default
        num_episodes instead of the user-configured value).

        Returns:
            Dict with "dev_size", "val_size", "test_size" keys.
        """
        if self._dev_tasks is None and self._non_test_pool is None:
            self._load_and_split_tasks()

        if self._is_dynamic:
            pool = self._non_test_pool or []
            if self.max_tasks_per_category and pool:
                limited = self._apply_max_tasks_limit(
                    pool, self.max_tasks_per_category, self._effective_categories(pool)
                )
                dev_size = len(limited)
            else:
                dev_size = len(pool)
        else:
            dev_size = len(self._dev_tasks or [])

        return {
            "dev_size": dev_size,
            "val_size": len(self._val_tasks or []),
            "test_size": len(self._test_tasks or []),
        }

    # =========================================================================
    # Dev/Test split
    # =========================================================================

    def _apply_max_tasks_limit(
        self,
        tasks: List[BenchmarkTask],
        limit: Optional[int],
        effective_categories: List[str],
    ) -> List[BenchmarkTask]:
        """Limit tasks per category. Returns all tasks if limit is None or 0."""
        if not limit:
            return tasks
        limited = []
        for cat in effective_categories:
            cat_tasks = [t for t in tasks if t.category.value == cat]
            if cat_tasks:
                limited.extend(cat_tasks[:limit])
        return limited

    def _effective_categories(self, tasks: List[BenchmarkTask]) -> List[str]:
        """Get effective categories, deriving from tasks if not configured."""
        return self.categories or sorted(set(t.category.value for t in tasks))

    def _split_tasks_by_val_ratio(
        self,
        tasks: List[BenchmarkTask],
        effective_categories: List[str],
        seed: int,
    ) -> Tuple[List[BenchmarkTask], List[BenchmarkTask]]:
        """Split tasks into dev and val by val_ratio, stratified by category."""
        if self.config.val_ratio <= 0 or not tasks:
            return tasks, []

        val_rng = random.Random(seed)
        dev_tasks = []
        val_tasks = []
        for cat in effective_categories:
            cat_tasks = [t for t in tasks if t.category.value == cat]
            if not cat_tasks:
                continue
            val_rng.shuffle(cat_tasks)
            val_idx = max(1, int(len(cat_tasks) * (1 - self.config.val_ratio)))
            dev_tasks.extend(cat_tasks[:val_idx])
            val_tasks.extend(cat_tasks[val_idx:])
        return dev_tasks, val_tasks

    def _load_and_split_tasks(self) -> None:
        """Load tasks and split into dev/val/test sets by category.

        When dynamic_sample=True and test set is non-empty, stores the full
        non-test pool for re-sampling on each evaluate call.
        """
        all_tasks = self._evaluator.load_tasks(
            suite=self.suite,
            categories=self.categories,
        )

        if not all_tasks:
            self._dev_tasks = []
            self._val_tasks = []
            self._test_tasks = []
            self._non_test_pool = []
            self._is_dynamic = False
            return

        effective_categories = self._effective_categories(all_tasks)

        if self.config.dev_ratio >= 1.0:
            dev_candidates = all_tasks
            test_candidates = []
            logger.info(f"dev_ratio=1.0: all {len(all_tasks)} tasks assigned to dev set")
        else:
            rng = random.Random(self.config.split_seed)
            dev_candidates = []
            test_candidates = []

            for cat in effective_categories:
                cat_tasks = [t for t in all_tasks if t.category.value == cat]
                if not cat_tasks:
                    continue

                rng.shuffle(cat_tasks)
                split_idx = max(1, int(len(cat_tasks) * self.config.dev_ratio))
                dev_candidates.extend(cat_tasks[:split_idx])
                test_candidates.extend(cat_tasks[split_idx:])

            logger.info(
                f"Dev/test split (seed={self.config.split_seed}, ratio={self.config.dev_ratio}): "
                f"dev_candidates={len(dev_candidates)}, test_candidates={len(test_candidates)}"
            )

        test_tasks = self._apply_max_tasks_limit(
            test_candidates, self.max_tasks_per_category_test, effective_categories,
        )
        if self.max_tasks_per_category_test:
            logger.info(
                f"Test limited to {self.max_tasks_per_category_test} per category, "
                f"total {len(test_tasks)}"
            )

        # Dynamic vs static mode
        if self.config.dynamic_sample:
            self._is_dynamic = True

            # Pre-split val from full dev_candidates (fixed, like test)
            dev_pool, val_tasks = self._split_tasks_by_val_ratio(
                dev_candidates, effective_categories, self.config.split_seed + 1000,
            )

            self._non_test_pool = dev_pool  # dev-only pool for dynamic sampling
            self._dev_tasks = None
            self._val_tasks = val_tasks  # fixed val set
            self._test_tasks = test_tasks
            logger.info(
                f"Dynamic sampling enabled: "
                f"dev_pool={len(dev_pool)}, val={len(val_tasks)} (fixed), test={len(test_tasks)}"
            )
        else:
            self._is_dynamic = False
            self._non_test_pool = None

            dev_tasks = self._apply_max_tasks_limit(
                dev_candidates, self.max_tasks_per_category, effective_categories,
            )
            if self.max_tasks_per_category:
                logger.info(
                    f"Dev limited to {self.max_tasks_per_category} per category, "
                    f"total {len(dev_tasks)}"
                )

            dev_tasks, val_tasks = self._split_tasks_by_val_ratio(
                dev_tasks, effective_categories, self.config.split_seed + 1000,
            )

            self._dev_tasks = dev_tasks
            self._val_tasks = val_tasks
            self._test_tasks = test_tasks
            logger.info(
                f"Final split: dev={len(dev_tasks)}, val={len(val_tasks)}, test={len(test_tasks)}"
            )

    def _get_tasks_for_eval(self, eval_mode: str = "dev", evaluate_seq: int = 0) -> List[BenchmarkTask]:
        """Get tasks for evaluation based on mode and eval_mode.

        Args:
            eval_mode: "dev" for detailed feedback, "val" for black-box evaluation, "test" for test set.
            evaluate_seq: Sequence number for dynamic sampling (different each evaluate call).

        Returns:
            List of tasks to evaluate.
        """
        if self._dev_tasks is None and self._non_test_pool is None:
            self._load_and_split_tasks()

        if eval_mode == "test":
            return self._test_tasks

        if self._is_dynamic:
            if eval_mode == "val":
                if not self._val_tasks:
                    logger.warning("val set is empty — falling back to dev tasks")
                    return []
                return self._val_tasks
            dev_tasks = self._dynamic_sample_from_pool(
                self._non_test_pool, seed_offset=evaluate_seq * 7919,
            )
            return dev_tasks
        else:
            # Static mode
            if eval_mode == "val":
                if not self._val_tasks:
                    logger.warning("val set is empty — falling back to dev tasks")
                    return []
                return self._val_tasks

            if self.config.mode == "batch":
                return self._dev_tasks
            elif self.config.mode == "onthefly":
                raise NotImplementedError("onthefly mode is not yet implemented")
            else:
                raise ValueError(f"Unknown benchmark mode: {self.config.mode}")

    def _get_all_tasks(self) -> List[BenchmarkTask]:
        """Get all tasks (dev + val + test) combined.

        In dynamic mode returns non_test_pool + val_tasks + test_tasks.
        In static mode returns dev + val + test.

        Returns:
            Combined list of all tasks.
        """
        if self._dev_tasks is None and self._non_test_pool is None:
            self._load_and_split_tasks()
        if self._is_dynamic:
            return (self._non_test_pool or []) + (self._val_tasks or []) + (self._test_tasks or [])
        return (self._dev_tasks or []) + (self._val_tasks or []) + (self._test_tasks or [])

    def _dynamic_sample_from_pool(
        self,
        pool: List[BenchmarkTask],
        seed_offset: int,
    ) -> List[BenchmarkTask]:
        """Sample dev tasks from pool with a different seed each time.

        Val is pre-split and fixed in _load_and_split_tasks().
        This method only samples dev tasks with max_tasks_per_category limit.
        """
        effective_categories = self._effective_categories(pool)

        rng = random.Random(self.config.split_seed + seed_offset)
        shuffled = list(pool)
        rng.shuffle(shuffled)

        return self._apply_max_tasks_limit(
            shuffled, self.max_tasks_per_category, effective_categories,
        )

    def evaluate_test_set(
        self,
        agent_instance: Any,
        harness_func: Callable = None,
        func_names: List[str] = None,
        code_dir: str = None,
        repeat_idx: int = 0,
    ) -> Tuple[Optional[float | dict], Dict[str, Any]]:
        """Evaluate on the test set after evolution is complete.

        repeat_idx rotates the map/seed across test_repeats passes for
        benchmarks that support deterministic per-pass seeding (e.g. balrog).
        Benchmarks that ignore it just re-run the same tasks each pass.

        Args:
            agent_instance: The GodelAgent instance
            harness_func: Optional pre-loaded harness function
            func_names: Optional function names for harness loader
            code_dir: Optional directory to load harness from (e.g. best_agent exported path).
                      When set, fresh-imports from this directory under a unique package name.

        Returns:
            Tuple of (reward, metrics). If no test set, returns (None, {"message": "No test set"}).
        """
        # Save agent_instance for _calculate_reward
        self._current_agent_instance = agent_instance

        try:
            self._inject_llm_config(agent_instance)

            # Ensure tasks are loaded and split (must be after _inject_llm_config
            # so that benchmark-specific config like num_episodes is correct)
            if self._dev_tasks is None and self._non_test_pool is None:
                self._load_and_split_tasks()

            # Select task set based on run_all_tasks config
            if self.config.run_all_tasks:
                eval_tasks = self._get_all_tasks()
                eval_label = "ALL (dev+val+test)"
                if not eval_tasks:
                    return None, {"message": "No tasks available"}
            else:
                eval_tasks = self._test_tasks
                eval_label = "test set"
                if not eval_tasks:
                    return None, {"message": "No test set (dev_ratio=1.0)"}

            if code_dir:
                logger.info(f"Evaluating {eval_label} using best version from: {code_dir}")
                print(f"  Evaluating {eval_label} using best version from: {code_dir}", flush=True)
            logger.info(f"Evaluating {eval_label}: {len(eval_tasks)} tasks")
            harness = self._get_harness(agent_instance, harness_func, func_names, code_dir=code_dir)
            results = self._evaluator.evaluate_tasks(
                eval_tasks, harness, self._progress_callback,
                parallel_workers=self.config.parallel_workers,
            )
            reward, metrics = self._calculate_reward(results, evaluate_seq=0, iteration=-1, eval_mode="test")

            # Add test set indicator to metrics
            metrics["test_set"] = True
            metrics["test_task_count"] = len(eval_tasks)
            if self.config.run_all_tasks:
                metrics["all_tasks_mode"] = True

            metrics = HarnessLoader.normalize_metrics(metrics)
            return reward, metrics
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            logger.error(f"Test set evaluation failed:\n{tb_str}")
            return None, HarnessLoader.normalize_metrics({
                "error": tb_str,
                "benchmark": self.benchmark_type,
                "suite": self.suite,
                "test_set": True,
            })
        finally:
            # Same anti-leak / anti-stale cleanup as __call__. Export is one-shot,
            # but cleaning up keeps sys.modules tidy if the process continues.
            cleanup_dir = code_dir or (self._harness_loader.agent_code_dir or None)
            if cleanup_dir:
                HarnessLoader.cleanup_loaded(cleanup_dir)

    # =========================================================================
    # LLM config injection
    # =========================================================================

    def _inject_llm_config(self, agent_instance: Any) -> None:
        """Propagate agent's LLM model, client, instance, and verbose flag to the evaluator.

        Args:
            agent_instance: The GodelAgent instance
        """
        model = None
        llm_client = None

        if agent_instance:
            config = getattr(agent_instance, 'config', None)
            if config:
                model = getattr(config, 'model', None)
            llm_client = getattr(agent_instance, 'llm_client', None)

        # Inject into evaluator
        if hasattr(self._evaluator, '_model'):
            self._evaluator._model = model
        if hasattr(self._evaluator, '_llm_client'):
            self._evaluator._llm_client = llm_client
        if hasattr(self._evaluator, '_agent_instance'):
            self._evaluator._agent_instance = agent_instance
        if hasattr(self._evaluator, '_verbose'):
            self._evaluator._verbose = self.verbose

        # Inject harness source code for code-aware summaries
        executor = getattr(agent_instance, 'action_executor', None)
        if executor and hasattr(executor, 'agent_codes') and executor.agent_codes:
            if hasattr(self._evaluator, '_harness_source'):
                self._evaluator._harness_source = dict(executor.agent_codes)

    def _progress_callback(self, current: int, total: int, result: TaskEvaluationResult) -> None:
        """Progress callback for task evaluation.

        Args:
            current: Current task number (1-indexed)
            total: Total number of tasks
            result: Task evaluation result
        """
        status = "✓" if result.success else "✗"
        ilog = result.interaction_log or []
        ilog_count = len(ilog)
        # Truncate api_messages in metadata for console display
        display_meta = {}
        for k, v in (result.metadata or {}).items():
            if k == "api_messages" and isinstance(v, list):
                display_meta[k] = f"<{len(v)} messages>"
            else:
                display_meta[k] = v
        lines = [
            f"  [Task {current}/{total}] {result.task_id}: {status} ({result.execution_time:.2f}s)",
            f"    metadata: {display_meta}",
            f"    interaction_log: {ilog_count} entries",
        ]
        # Verbose mode: print full interaction log to console
        if self.verbose and ilog:
            lines.append("    --- Interaction Log ---")
            for entry in ilog:
                role = entry.get("role", "")
                if role == "assistant":
                    tools = entry.get("tool_calls", [])
                    if tools:
                        tool_strs = [
                            f"{t.get('name', '?')}({t.get('args', '') or t.get('arguments', '')})"
                            for t in tools
                        ]
                        lines.append(f"      Agent → {', '.join(tool_strs)}")
                    else:
                        lines.append(f"      Agent → {entry.get('content', '')}")
                elif role == "tool":
                    name = entry.get("name", "?")
                    result_str = str(entry.get("result", ""))
                    lines.append(f"      Tool({name}) → {result_str}")
                elif role == "user":
                    lines.append(f"      User: {entry.get('content', '')}")
            lines.append("    --- End Log ---")
        with self._print_lock:
            print("\n".join(lines), flush=True)

    # =========================================================================
    # Condensed + per-task log saving
    # =========================================================================

    def _build_log_dir(self, seq: int, iteration: int) -> str:
        """Build the eval log directory path (including the eval_XXX subdirectory)."""
        try:
            base_dir = os.getcwd()
        except FileNotFoundError:
            # CWD was deleted (e.g. by a bash action) — fall back to a safe dir
            base_dir = tempfile.gettempdir()
        if hasattr(self, "_current_agent_instance") and self._current_agent_instance:
            executor = getattr(self._current_agent_instance, "action_executor", None)
            if executor and hasattr(executor, "agent_code_dir") and executor.agent_code_dir:
                base_dir = os.path.dirname(executor.agent_code_dir)

        if iteration == -1:
            iter_dir = os.path.join(base_dir, "eval_logs", f"test_iter_{seq:03d}")
        else:
            iter_dir = os.path.join(base_dir, "eval_logs", f"iter_{iteration:03d}")

        log_dir = os.path.join(iter_dir, f"eval_{seq:03d}")
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _log_prefix(self, seq: int, iteration: int) -> str:
        """Build shared file prefix for eval log files."""
        if iteration == -1:
            return f"eval_test_{seq:03d}"
        return f"eval_{seq:03d}"

    def _save_condensed_log(self, metrics: Dict[str, Any], seq: int, iteration: int) -> None:
        """Save a condensed log (no step_traces) alongside the full log."""
        handler = self.get_log_summary_handler()
        if not handler:
            return

        log_dir = metrics.get("log_dir")
        if not log_dir:
            return

        condensed = handler.build_condensed_log(metrics)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"{self._log_prefix(seq, iteration)}_{ts}_condensed.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(condensed, f, ensure_ascii=False, indent=2)
        metrics["condensed_log_file"] = path
        logger.info(f"Saved condensed log: {path}")
        print(f"    [Eval] Saved condensed log to: {path}", flush=True)

    def _save_per_task_logs(self, metrics: Dict[str, Any], seq: int, iteration: int) -> None:
        """Save per-task step trace files for on-demand inspection."""
        log_dir = metrics.get("log_dir")
        if not log_dir:
            return

        task_results = metrics.get("task_results", [])
        if not task_results:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = self._log_prefix(seq, iteration)

        task_log_files = []
        for r in task_results:
            task_id = r.get("task_id", "unknown")
            safe_id = task_id.replace("/", "_")

            raw_log = r.get("interaction_log", [])
            # Balrog uses episode-based logs with episode_idx/step_traces;
            # other benchmarks (agentdojo etc.) use flat role-based logs.
            if raw_log and isinstance(raw_log[0], dict) and "episode_idx" in raw_log[0]:
                interaction_log = [
                    {"episode_idx": ep.get("episode_idx"), "step_traces": ep.get("step_traces", [])}
                    for ep in raw_log
                ]
            else:
                interaction_log = raw_log

            task_data = {
                "task_id": task_id,
                "interaction_log": interaction_log,
            }

            path = os.path.join(log_dir, f"{prefix}_{ts}_task_{safe_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(task_data, f, ensure_ascii=False, indent=2)
            task_log_files.append(path)

        metrics["task_log_files"] = task_log_files
        logger.info(f"Saved {len(task_log_files)} per-task log files")
        print(f"    [Eval] Saved {len(task_log_files)} per-task log files", flush=True)

    # =========================================================================
    # __call__: external evaluator interface
    # =========================================================================

    def __call__(
        self,
        agent_instance: Any,
        harness_func: Callable,
        test_cases: List[Any],
        func_names: List[str] = None,
        evaluate_seq: int = 0,
        iteration: int = 0,
        eval_mode: str = "dev",
        num_tasks: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
    ) -> Tuple[Optional[float | dict], Dict[str, Any]]:
        """Execute benchmark evaluation (dev or val set).

        This is the external evaluator interface expected by AgentActionExecutor.

        Args:
            agent_instance: The GodelAgent instance
            harness_func: Pre-loaded harness function (usually None — the loader fresh-imports from disk)
            test_cases: Optional test cases (ignored for benchmark evaluation)
            func_names: Optional list of function names to search for
            evaluate_seq: Sequence number of this evaluate call
            iteration: Current evolution iteration number
            eval_mode: "dev" for detailed feedback, "val" for black-box reward-only
            num_tasks: Debug only — limit evaluation to N tasks
            task_ids: DEBUG ONLY — evaluate specific task IDs (always dev debug mode,
                      reward NOT tracked). Mutually exclusive with num_tasks
                      (task_ids wins if both provided — enforced in caller).

        Returns:
            Tuple of (reward, metrics). For val mode, metrics are stripped to only reward info.
        """
        logger.info(f"Starting benchmark evaluation ({eval_mode}): {self.benchmark_type}/{self.suite}")

        # Save agent_instance for use in _calculate_reward
        self._current_agent_instance = agent_instance

        try:
            # 0. Propagate agent's LLM config to evaluator
            self._inject_llm_config(agent_instance)

            # 1. Get harness function
            harness = self._get_harness(agent_instance, harness_func, func_names)

            # 2. Get tasks for evaluation based on eval_mode (dynamic sampling supported)
            tasks = self._get_tasks_for_eval(eval_mode=eval_mode, evaluate_seq=evaluate_seq)

            if not tasks:
                logger.warning("No benchmark tasks loaded")
                if eval_mode == "val":
                    return None, {
                        "error": "Val set is empty (val_ratio too small or too few dev tasks). "
                                 "Please use eval_mode='dev' to continue iteration.",
                        "benchmark": self.benchmark_type,
                        "suite": self.suite,
                    }
                return None, {
                    "error": "No tasks loaded",
                    "benchmark": self.benchmark_type,
                    "suite": self.suite,
                }

            # 2b. Debug: limit task count or filter by task_ids
            original_count = None
            if task_ids is not None:
                # Filter tasks by exact task_id match (deduplicated).
                # Build a lookup dict once (O(N)) so each task_id lookup is O(1)
                # instead of scanning the full list per task_id (O(M*N)).
                task_by_id = {t.task_id: t for t in tasks}
                seen: set = set()
                filtered = []
                not_found = []
                for tid in task_ids:
                    if tid in seen:
                        continue
                    seen.add(tid)
                    match = task_by_id.get(tid)
                    if match:
                        filtered.append(match)
                    else:
                        not_found.append(tid)

                if not_found:
                    available_preview = [t.task_id for t in tasks[:5]]
                    suffix = "..." if len(tasks) > 5 else ""
                    return None, {
                        "error": f"Task IDs not found in {eval_mode} set: {not_found}. "
                                 f"Available task IDs: {available_preview}{suffix}",
                        "benchmark": self.benchmark_type,
                        "suite": self.suite,
                    }

                tasks = filtered

            elif num_tasks is not None and num_tasks > 0:
                original_count = len(tasks)
                tasks = random.sample(tasks, min(num_tasks, len(tasks)))
                if num_tasks > original_count:
                    logger.warning(f"num_tasks={num_tasks} exceeds available tasks ({original_count})")

            # 3. Execute evaluation with progress callback
            results = self._evaluator.evaluate_tasks(
                tasks, harness, self._progress_callback,
                parallel_workers=self.config.parallel_workers,
            )

            # 4. Calculate reward and metrics
            reward, metrics = self._calculate_reward(
                results,
                evaluate_seq=evaluate_seq,
                iteration=iteration,
                eval_mode=eval_mode,
            )

            if reward is not None:
                if isinstance(reward, dict):
                    reward_str = ", ".join(f"{k}={v:.4f}" for k, v in reward.items() if isinstance(v, (int, float)))
                    logger.info(f"Benchmark evaluation ({eval_mode}) complete: {reward_str}")
                else:
                    logger.info(f"Benchmark evaluation ({eval_mode}) complete: reward={reward:.4f}")

                # Inject task count info for auto-upgrade in agent_evaluator
                if original_count is not None:
                    metrics["num_tasks_requested"] = num_tasks
                    metrics["total_available_tasks"] = original_count
            else:
                logger.warning(f"Benchmark evaluation ({eval_mode}) produced no reward: {metrics}")

            # Scrub the loader's anonymous _harness_pkg_N names from every error string
            # before handing metrics to the agent, so it sees `prompts.py` (not
            # `_harness_pkg_3.prompts`) and can fix the bug in one step instead of first
            # deciphering the temp package name. Recurses into task_results[*].metadata.
            # execution_error and interaction_log[i].error. The "_harness_pkg_" pre-filter
            # makes this a no-op when no string carries the marker.
            metrics = HarnessLoader.normalize_metrics(metrics)
            # Expose dynamic-sampling state so the evaluate feedback can remind the
            # agent that dev reward is sampled (variance), not a deterministic signal.
            metrics["dynamic_sample"] = bool(getattr(self.config, "dynamic_sample", False))
            if getattr(self, "_non_test_pool", None) is not None:
                metrics["dev_pool_size"] = len(self._non_test_pool)
            return reward, metrics

        except ValueError as e:
            logger.error(f"Config error: {e}")
            return None, {
                "error": str(e),
                "benchmark": self.benchmark_type,
                "suite": self.suite,
            }
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            logger.error(f"Benchmark evaluation failed:\n{tb_str}")
            return None, HarnessLoader.normalize_metrics({
                "error": tb_str,
                "benchmark": self.benchmark_type,
                "suite": self.suite,
            })
        finally:
            # Purge sys.modules of every module the fresh harness import registered
            # under agent_code_dir. Without this, a long run (tens of evaluate()
            # calls per iteration × many iterations) leaks one package's worth of
            # modules per call, AND absolute-import bare names (e.g. 'utils') go
            # stale — the next evaluate() would get the previous version from
            # sys.modules instead of reading the just-edited disk. finally runs on
            # every path (success, error, early return), so nothing leaks.
            agent_code_dir = self._harness_loader.agent_code_dir
            if agent_code_dir:
                HarnessLoader.cleanup_loaded(agent_code_dir)

    # =========================================================================
    # Harness loading
    # =========================================================================

    def _get_harness(self, agent_instance: Any, harness_func: Callable = None, func_names: List[str] = None, code_dir: str = None) -> Callable:
        """Get the harness function, fresh-imported from disk.

        Both branches load via ``HarnessLoader``, which imports the harness under
        a unique package name (``_harness_pkg_<N>``) on every call. That guarantees
        the bytes that run == the bytes on disk == the bytes that get hashed — no
        in-memory reloader cache can diverge from disk (the bug that previously let
        a stale, working module earn reward while the on-disk version was broken).
        Callers clean up with ``HarnessLoader.cleanup_loaded`` after evaluation
        (done in ``__call__`` / ``evaluate_test_set``).

        Args:
            agent_instance: The GodelAgent instance
            harness_func: Optional pre-loaded harness function (skips disk import)
            func_names: Optional list of function names to search for
            code_dir: Optional directory to load the harness from (export path);
                when None, loads from the live ``action_executor.agent_code_dir``.

        Returns:
            A wrapped harness function with signature (instruction: str) -> str

        Raises:
            ValueError: If harness function is not found
        """
        # Export path: load fresh from the explicitly given directory (one-shot).
        if code_dir:
            temp_loader = HarnessLoader(code_dir)
            harness_info = temp_loader.load(
                agent_instance=agent_instance,
                harness_func=None,
                func_names=func_names,
            )
            return self._wrap_harness(harness_info, agent_instance)

        # Iteration-eval path: load fresh from the live repo dir.
        # Lazily set agent_code_dir from the action_executor.
        if not self._harness_loader.agent_code_dir and agent_instance:
            executor = getattr(agent_instance, 'action_executor', None)
            if executor and hasattr(executor, 'agent_code_dir') and executor.agent_code_dir:
                # executor.agent_code_dir may be a relative path (repo_path relative
                # to CWD). Normalize to absolute (independent of a live CWD), matching
                # HarnessLoader.__init__, to avoid cascading crashes when CWD is
                # deleted during the iteration-eval phase.
                self._harness_loader.agent_code_dir = HarnessLoader._normalize_code_dir(
                    executor.agent_code_dir
                )

        harness_info = self._harness_loader.load(
            agent_instance=agent_instance,
            harness_func=harness_func,
            func_names=func_names,
        )
        return self._wrap_harness(harness_info, agent_instance)

    def _wrap_harness(self, harness_info: HarnessInfo, agent_instance: Any) -> Callable:
        """Wrap the harness function with a unified signature (instruction: str) -> str.

        ``harness_info.func`` is already the fresh import for this evaluate call, and
        the agent will not edit the harness during evaluation, so there is no need to
        re-fetch func on every call. The legacy signature ``func(instruction)``
        receives the live agent via the module-level ``set_current_agent(agent)``
        — fall back to a direct getattr on the module, no longer relying on the reloader.
        """
        def wrapped_harness(instruction: str) -> str:
            """Unified harness entry."""
            if agent_instance and hasattr(agent_instance, '_thread_local'):
                agent_instance._thread_local.harness_mode = True
            try:
                if harness_info.has_agent_param:
                    return harness_info.func(agent_instance, instruction)

                # Legacy single-arg signature: hand the harness the live agent
                # via a module-level set_current_agent() if it defines one.
                set_agent_func = (
                    getattr(harness_info.module, 'set_current_agent', None)
                    if harness_info.module is not None else None
                )
                if callable(set_agent_func):
                    set_agent_func(agent_instance)
                return harness_info.func(instruction)
            finally:
                if agent_instance and hasattr(agent_instance, '_thread_local'):
                    agent_instance._thread_local.harness_mode = False

        return wrapped_harness


def create_benchmark_evaluator(config: BenchmarkConfig) -> BenchmarkEvaluatorAdapter:
    """Factory function to create a benchmark evaluator adapter.

    Dispatches to the appropriate subclass based on config.type.

    Args:
        config: BenchmarkConfig instance with all benchmark settings.

    Returns:
        BenchmarkEvaluatorAdapter subclass instance

    Raises:
        ValueError: If config.type is not supported
    """
    if config.type == "agentdojo":
        from benchmark.agentdojo.adapter import AgentDojoAdapter
        return AgentDojoAdapter(config)

    if config.type == "paper_review":
        from benchmark.paper_review.adapter import PaperReviewAdapter
        return PaperReviewAdapter(config)

    if config.type == "search_arena":
        from benchmark.search_arena.adapter import SearchArenaAdapter
        return SearchArenaAdapter(config)

    if config.type == "imo_grading":
        from benchmark.imo_grading.adapter import IMOGradingAdapter
        return IMOGradingAdapter(config)

    if config.type == "polyglot":
        from benchmark.polyglot.adapter import PolyglotAdapter
        return PolyglotAdapter(config)

    if config.type == "genesis":
        from benchmark.genesis.adapter import GenesisAdapter
        return GenesisAdapter(config)

    if config.type == "balrog":
        from benchmark.balrog.adapter import BalrogAdapter
        return BalrogAdapter(config)

    if config.type == "terminal_bench":
        from benchmark.terminal_bench.adapter import TerminalBenchAdapter
        return TerminalBenchAdapter(config)

    raise ValueError(
        f"Unsupported benchmark type: '{config.type}'. "
        f"Supported types: 'agentdojo', 'paper_review', 'search_arena', 'imo_grading', 'polyglot', 'genesis', 'balrog', 'terminal_bench'"
    )
