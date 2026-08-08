"""
AgentDojo Benchmark Evaluator Adapter.

AgentDojo-specific adapter that implements reward calculation
(weighted combination of utility and security rates).
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from benchmark.config import BenchmarkConfig
from benchmark.adapter import BenchmarkEvaluatorAdapter
from benchmark.evaluators.task_types import BenchmarkTask, TaskCategory, TaskEvaluationResult
from react_loop.utils.log_format import _C

logger = logging.getLogger(__name__)


class AgentDojoAdapter(BenchmarkEvaluatorAdapter):
    """AgentDojo-specific benchmark evaluator adapter.

    Reward is calculated as an equal-weight average of utility and security
    success rates when both are present.
    """

    def __init__(self, config: BenchmarkConfig):
        """Initialize the AgentDojo adapter.

        Args:
            config: BenchmarkConfig instance.
        """
        super().__init__(config)
        self.attack_type = config.attack
        self._parsed_categories = self._parse_categories(config.categories)

    @staticmethod
    def _parse_categories(categories: Optional[List[str]]) -> List[TaskCategory]:
        """Parse category strings to TaskCategory enums."""
        if categories is None:
            return [TaskCategory.UTILITY, TaskCategory.SECURITY]

        result = []
        for cat in categories:
            cat_lower = cat.lower()
            if cat_lower == "utility":
                result.append(TaskCategory.UTILITY)
            elif cat_lower == "security":
                result.append(TaskCategory.SECURITY)
            else:
                logger.warning(f"Unknown category: {cat}, skipping")

        return result if result else [TaskCategory.UTILITY, TaskCategory.SECURITY]

    def _inject_llm_config(self, agent_instance: Any) -> None:
        """Propagate agent's LLM config and attack_type to the evaluator."""
        super()._inject_llm_config(agent_instance)
        if hasattr(self._evaluator, '_attack_type'):
            self._evaluator._attack_type = self.attack_type

    def _load_and_split_tasks(self) -> None:
        """Load tasks and split into dev/val/test sets at the ID level.

        Overrides base class to prevent data leakage between dev and test sets.
        Splits user_task_ids and injection_task_ids independently, then builds
        tasks from each subset. Supports dynamic sampling mode.
        """
        user_task_ids, injection_task_ids = self._evaluator.get_task_ids(self.suite)

        if not user_task_ids and not injection_task_ids:
            self._dev_tasks = []
            self._val_tasks = []
            self._test_tasks = []
            self._non_test_pool = []
            self._is_dynamic = False
            return

        # Split IDs into dev and test
        rng = random.Random(self.config.split_seed)
        all_user_ids = list(user_task_ids)
        rng.shuffle(all_user_ids)
        dev_ratio = self.config.dev_ratio

        if dev_ratio >= 1.0:
            dev_user_ids = all_user_ids
            test_user_ids = []
            dev_inj_ids = list(injection_task_ids)
            test_inj_ids = []
        else:
            split_idx = max(1, int(len(all_user_ids) * dev_ratio))
            dev_user_ids = all_user_ids[:split_idx]
            test_user_ids = all_user_ids[split_idx:]

            all_inj_ids = list(injection_task_ids)
            rng.shuffle(all_inj_ids)
            if all_inj_ids:
                split_idx = max(1, int(len(all_inj_ids) * dev_ratio))
                dev_inj_ids = all_inj_ids[:split_idx]
                test_inj_ids = all_inj_ids[split_idx:]
            else:
                dev_inj_ids = []
                test_inj_ids = []

        # Build test tasks
        test_tasks = self._evaluator.load_tasks_for_ids(
            suite=self.suite,
            user_task_ids=test_user_ids,
            injection_task_ids=test_inj_ids,
            categories=self._parsed_categories,
        ) if test_user_ids or test_inj_ids else []

        max_test = self.max_tasks_per_category_test
        if max_test and test_tasks:
            test_tasks = self._apply_max_tasks_limit(test_tasks, limit=max_test)

        # Determine dynamic vs static mode
        if (self.config.dynamic_sample
                and len(test_tasks) > 0
                and dev_ratio < 1.0):
            # Dynamic mode: pre-split val at ID level (fixed), rest goes to dev pool
            self._is_dynamic = True

            if self.config.val_ratio > 0 and (dev_user_ids or dev_inj_ids):
                val_rng = random.Random(self.config.split_seed + 1000)

                val_rng.shuffle(dev_user_ids)
                val_user_idx = max(1, int(len(dev_user_ids) * (1 - self.config.val_ratio)))
                pool_user_ids = dev_user_ids[:val_user_idx]
                val_user_ids = dev_user_ids[val_user_idx:]

                val_rng.shuffle(dev_inj_ids)
                if dev_inj_ids:
                    val_inj_idx = max(1, int(len(dev_inj_ids) * (1 - self.config.val_ratio)))
                    pool_inj_ids = dev_inj_ids[:val_inj_idx]
                    val_inj_ids = dev_inj_ids[val_inj_idx:]
                else:
                    pool_inj_ids = []
                    val_inj_ids = []
            else:
                pool_user_ids = dev_user_ids
                pool_inj_ids = dev_inj_ids
                val_user_ids = []
                val_inj_ids = []

            self._non_test_pool = self._evaluator.load_tasks_for_ids(
                suite=self.suite,
                user_task_ids=pool_user_ids,
                injection_task_ids=pool_inj_ids,
                categories=self._parsed_categories,
            )
            self._dev_tasks = None

            if val_user_ids or val_inj_ids:
                self._val_tasks = self._evaluator.load_tasks_for_ids(
                    suite=self.suite,
                    user_task_ids=val_user_ids,
                    injection_task_ids=val_inj_ids,
                    categories=self._parsed_categories,
                )
            else:
                self._val_tasks = []

            self._test_tasks = test_tasks
            logger.info(
                f"AgentDojo dynamic mode: "
                f"dev_pool={len(self._non_test_pool)}, val={len(self._val_tasks)} (fixed), test={len(test_tasks)}"
            )
        else:
            # Static mode: split dev IDs into train + val
            self._is_dynamic = False
            self._non_test_pool = None

            if self.config.val_ratio > 0 and (dev_user_ids or dev_inj_ids):
                val_rng = random.Random(self.config.split_seed + 1000)

                val_rng.shuffle(dev_user_ids)
                val_user_idx = max(1, int(len(dev_user_ids) * (1 - self.config.val_ratio)))
                train_user_ids = dev_user_ids[:val_user_idx]
                val_user_ids = dev_user_ids[val_user_idx:]

                val_rng.shuffle(dev_inj_ids)
                if dev_inj_ids:
                    val_inj_idx = max(1, int(len(dev_inj_ids) * (1 - self.config.val_ratio)))
                    train_inj_ids = dev_inj_ids[:val_inj_idx]
                    val_inj_ids = dev_inj_ids[val_inj_idx:]
                else:
                    train_inj_ids = []
                    val_inj_ids = []
            else:
                train_user_ids = dev_user_ids
                train_inj_ids = dev_inj_ids
                val_user_ids = []
                val_inj_ids = []

            dev_tasks = self._evaluator.load_tasks_for_ids(
                suite=self.suite,
                user_task_ids=train_user_ids,
                injection_task_ids=train_inj_ids,
                categories=self._parsed_categories,
            )

            if val_user_ids or val_inj_ids:
                val_tasks = self._evaluator.load_tasks_for_ids(
                    suite=self.suite,
                    user_task_ids=val_user_ids,
                    injection_task_ids=val_inj_ids,
                    categories=self._parsed_categories,
                )
            else:
                val_tasks = []

            if self.max_tasks_per_category:
                dev_tasks = self._apply_max_tasks_limit(dev_tasks)
                val_tasks = self._apply_max_tasks_limit(val_tasks) if val_tasks else []

            self._dev_tasks = dev_tasks
            self._val_tasks = val_tasks
            self._test_tasks = test_tasks
            logger.info(
                f"Task split: dev={len(dev_tasks)}, val={len(val_tasks)}, test={len(test_tasks)}"
            )

    def _apply_max_tasks_limit(
        self,
        tasks: List[BenchmarkTask],
        limit: Optional[int] = None,
        effective_categories: Optional[List[str]] = None,
    ) -> List[BenchmarkTask]:
        """Limit tasks per category using enum comparison."""
        effective_limit = limit if limit is not None else self.max_tasks_per_category
        if not effective_limit:
            return tasks
        limited = []
        for cat in self._parsed_categories:
            cat_tasks = [t for t in tasks if t.category == cat]
            if cat_tasks:
                limited.extend(cat_tasks[:effective_limit])
        return limited

    def _progress_callback(self, current: int, total: int, result: TaskEvaluationResult) -> None:
        """Progress callback for task evaluation with highlighted output."""
        status = "✓" if result.success else "✗"
        status_color = _C.BGR if result.success else _C.RD
        category = result.metadata.get("category", "unknown")

        # Build display metadata — exclude verbose fields
        display_meta = {
            k: v for k, v in result.metadata.items()
            if k not in ("task_summary", "interaction_log", "api_messages")
        }

        print(
            f"  {_C.B}{_C.CY}━━━{_C.RST} "
            f"[Task {current}/{total}] {result.task_id}: "
            f"{category}={status_color}{status}{_C.RST} "
            f"({result.execution_time:.2f}s) "
            f"{_C.B}{_C.CY}━━━{_C.RST}\n"
            f"    metadata: {display_meta}",
            flush=True
        )

    def _calculate_reward(
        self,
        results: List[TaskEvaluationResult],
        evaluate_seq: int = 0,
        iteration: int = 0,
        eval_mode: str = "dev",
    ) -> Tuple[Optional[float | dict], Dict[str, Any]]:
        """Calculate reward from AgentDojo evaluation results.

        Reward is an equal-weight average of utility and security success rates.
        """
        if not results:
            return None, {"error": "No evaluation results", "total_tasks": 0}

        # Check for execution errors (log warning, do not over-block)
        execution_errors = []
        for r in results:
            if r.metadata and r.metadata.get("error"):
                execution_errors.append({
                    "task_id": r.task_id,
                    "error": r.metadata.get("error")
                })

        # Only return None if all tasks errored
        if len(execution_errors) == len(results):
            error_details = execution_errors[0]["error"]
            if len(execution_errors) > 1:
                error_details += f" (and {len(execution_errors) - 1} more errors)"
            return None, {
                "error": f"Task execution errors: {error_details}",
                "execution_errors": execution_errors,
                "benchmark": self.benchmark_type,
                "suite": self.suite,
                "total_tasks": len(results),
            }

        if execution_errors:
            logger.warning(
                f"{len(execution_errors)}/{len(results)} tasks had execution errors "
                f"(included in results as failed)"
            )

        # AgentDojo standard: extract utility and security dimensions from metadata
        # Tasks with execution_error have success=False and are naturally counted as
        # failed, not affecting reward calculation.
        utility_only_tasks = [
            r for r in results
            if r.metadata.get("utility_result") is not None
            and "injection_succeeded" not in (r.metadata or {})
            and not r.metadata.get("error")  # Exclude tasks that errored
        ]
        utility_passed = sum(1 for r in utility_only_tasks if r.metadata.get("utility_result"))
        utility_total = len(utility_only_tasks)
        avg_utility = utility_passed / utility_total if utility_total > 0 else 0.0

        security_tasks = [
            r for r in results
            if "injection_succeeded" in (r.metadata or {})
            and not r.metadata.get("error")  # Exclude tasks that errored
        ]
        security_breached = sum(1 for r in security_tasks if r.metadata.get("injection_succeeded"))
        security_total = len(security_tasks)
        avg_security_asr = security_breached / security_total if security_total > 0 else 0.0

        # Overall success rate (legacy, based on result.success)
        total_success = sum(1 for r in results if r.success)
        total_tasks = len(results)
        overall_rate = total_success / total_tasks if total_tasks > 0 else 0.0

        # Calculate reward
        if security_total > 0:
            reward = {
                "avg_utility": avg_utility,
                "avg_asr": avg_security_asr,  # ASR, lower is better (0 = perfect defense)
                "scalar_reward": (avg_utility + (1 - avg_security_asr)) / 2,
            }
        else:
            reward = avg_utility

        # Build metrics
        metrics = {
            "benchmark": self.benchmark_type,
            "suite": self.suite,
            "reward": reward,
            "total_tasks": total_tasks,
            "total_success": total_success,
            "overall_rate": overall_rate,
            "overall_success_rate": overall_rate,
            "avg_utility": {
                "total": utility_total,
                "passed": utility_passed,
                "rate": avg_utility,
            },
            "avg_asr": {
                "total": security_total,
                "breached": security_breached,
                "asr": avg_security_asr,
            },
            "task_results": [r.to_dict() for r in results],
        }

        # Prepare metrics copy for agent and logs
        metrics_copy = self._copy_metrics(metrics)

        # Save logs to file
        self._save_evaluation_logs(metrics_copy, evaluate_seq, iteration)
        self._save_condensed_log(metrics_copy, evaluate_seq, iteration)
        self._save_per_task_logs(metrics_copy, evaluate_seq, iteration)

        return reward, metrics_copy

    @staticmethod
    def _copy_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy metrics (logging mutates the dict downstream)."""
        import copy
        return copy.deepcopy(metrics)

    def _save_evaluation_logs(
        self,
        metrics: Dict[str, Any],
        evaluate_seq: int,
        iteration: int
    ) -> None:
        """Save the full evaluation log to a JSON file, organized into per-iteration subdirectories."""
        import json
        from datetime import datetime
        import os

        log_dir = self._build_log_dir(evaluate_seq, iteration)
        metrics["log_dir"] = log_dir

        # Build the log file name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'eval_{evaluate_seq:03d}_{timestamp}.json'
        filepath = os.path.join(log_dir, filename)
        metrics["log_file"] = filepath

        # Build the log content
        log_content = {
            'iteration': iteration,
            'evaluate_seq': evaluate_seq,
            'timestamp': datetime.now().isoformat(),
            'reward': metrics.get('reward') if 'reward' in metrics else None,
            'benchmark': metrics.get('benchmark'),
            'suite': metrics.get('suite'),
            'avg_utility': metrics.get('avg_utility'),
            'avg_asr': metrics.get('avg_asr'),
            'task_results': metrics.get('task_results', []),
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_content, f, ensure_ascii=False, indent=2)

        print(f"    [Eval] Saved full log to: {filepath}", flush=True)

    def get_log_summary_handler(self) -> Optional[Any]:
        """Override to propagate summary_passed_samples to the log summary handler."""
        handler = super().get_log_summary_handler()
        if handler is not None:
            handler._max_passed_samples = self.config.summary_passed_samples
        return handler
