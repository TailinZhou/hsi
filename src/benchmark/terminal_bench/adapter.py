"""Terminal-Bench 2 benchmark evaluator adapter.

Calculates reward as passed/total across all evaluated terminal tasks.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from benchmark.adapter import BenchmarkEvaluatorAdapter
from benchmark.config import BenchmarkConfig
from benchmark.evaluators.task_types import TaskEvaluationResult
from benchmark.terminal_bench.config import TerminalBenchConfig

logger = logging.getLogger(__name__)


class TerminalBenchAdapter(BenchmarkEvaluatorAdapter):
    """Adapter for Terminal-Bench 2 — reward = passed/total."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self._tb2_config = TerminalBenchConfig.from_benchmark_config(config)

    def _inject_llm_config(self, agent_instance: Any) -> None:
        """Inject LLM config and TB2 config into the evaluator."""
        super()._inject_llm_config(agent_instance)
        if hasattr(self._evaluator, "_tb2_config"):
            self._evaluator._tb2_config = self._tb2_config

    def get_evolution_goal(self) -> Optional[str]:
        """Load evolution goal from benchmark_config_goal/terminal_bench/goal.md."""
        goal_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..",
            "benchmark_config_goal", "terminal_bench", "goal.md",
        )
        goal_path = os.path.normpath(goal_path)
        if os.path.exists(goal_path):
            with open(goal_path, "r", encoding="utf-8") as f:
                return f.read()
        return "Maximize pass rate on Terminal-Bench 2 terminal tasks"

    def _calculate_reward(
        self,
        results: List[TaskEvaluationResult],
        evaluate_seq: int = 0,
        iteration: int = 0,
        eval_mode: str = "dev",
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """Calculate reward from TB2 evaluation results.

        Reward = passed_count / total_count
        """
        if not results:
            return None, {"error": "No results", "total_tasks": 0}

        # Check for execution errors
        execution_errors = [r for r in results if r.metadata.get("execution_error")]
        if execution_errors and len(execution_errors) == len(results):
            return None, {
                "error": f"All tasks failed: {execution_errors[0].metadata.get('execution_error', '')}",
                "total_tasks": len(results),
            }

        passed = sum(1 for r in results if r.success)
        total = len(results)
        reward = passed / total if total > 0 else 0.0

        # Per-category breakdown
        by_category: Dict[str, Dict] = {}
        for r in results:
            cat = r.metadata.get("tb2_category", "unknown")
            if cat not in by_category:
                by_category[cat] = {"passed": 0, "failed": 0, "tasks": []}
            if r.success:
                by_category[cat]["passed"] += 1
            else:
                by_category[cat]["failed"] += 1
            by_category[cat]["tasks"].append({
                "task_id": r.task_id,
                "success": r.success,
                "execution_time": r.execution_time,
            })

        # Category pass rates
        category_rates = {}
        for cat, data in by_category.items():
            total_cat = data["passed"] + data["failed"]
            category_rates[cat] = data["passed"] / total_cat if total_cat > 0 else 0.0

        metrics = {
            "benchmark": "terminal_bench",
            "reward": reward,
            "total_tasks": total,
            "passed": passed,
            "failed": total - passed,
            "by_category": category_rates,
            "category_details": by_category,
            "task_results": [r.to_dict() if hasattr(r, 'to_dict') else {
                "task_id": r.task_id,
                "success": r.success,
                "execution_time": r.execution_time,
                "metadata": r.metadata,
            } for r in results],
        }

        self._save_logs(metrics, evaluate_seq, iteration)
        self._save_condensed_log(metrics, evaluate_seq, iteration)
        self._save_per_task_logs(metrics, evaluate_seq, iteration)

        return reward, metrics

    def _save_logs(self, metrics: Dict, seq: int, iteration: int):
        """Save evaluation logs to disk."""
        log_dir = self._build_log_dir(seq, iteration)
        metrics["log_dir"] = log_dir

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"eval_{seq:03d}_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"    [Eval] Saved terminal_bench log to: {path}", flush=True)
