"""Balrog benchmark evaluator adapter.

Calculates reward as average progression across all evaluated game environments.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from benchmark.adapter import BenchmarkEvaluatorAdapter
from benchmark.balrog.config import BalrogConfig
from benchmark.config import BenchmarkConfig
from benchmark.evaluators.task_types import TaskEvaluationResult

logger = logging.getLogger(__name__)


class AdaptiveThresholdTracker:
    """Per-task adaptive threshold tracker.

    Each task_id gets its own independently tracked bar.  On first observation
    for a task, threshold = reward.  On subsequent observations, if the reward
    exceeds the best seen for that task, threshold is raised to
    (reward + margin), clamped to 1.0.  Thresholds never decrease.
    """

    def __init__(self, margin: float):
        self.margin = margin
        self._task_thresholds: Dict[str, float] = {}
        self._task_best_rewards: Dict[str, float] = {}

    def get_threshold(self, task_id: str) -> Optional[float]:
        """Returns per-task threshold, or None if task hasn't been seen yet."""
        return self._task_thresholds.get(task_id)

    def get_snapshot(self) -> Dict[str, Any]:
        """Returns a snapshot of current state for injection into evaluator/metrics."""
        return {
            "task_thresholds": dict(self._task_thresholds),
            "task_best_rewards": dict(self._task_best_rewards),
        }

    def update(self, task_id: str, reward: float) -> float:
        """Update per-task threshold based on observed reward. Returns the new threshold.

        First call for a task: set threshold to reward.
        Subsequent: if reward > best, raise to reward + margin.
        Only increases, clamps to max 1.0.
        """
        if task_id not in self._task_thresholds:
            self._task_best_rewards[task_id] = reward
            self._task_thresholds[task_id] = min(reward + self.margin, 1.0)
            return self._task_thresholds[task_id]

        if reward > self._task_best_rewards[task_id]:
            self._task_best_rewards[task_id] = reward
            new_threshold = min(reward + self.margin, 1.0)
            self._task_thresholds[task_id] = max(
                self._task_thresholds[task_id], new_threshold
            )

        return self._task_thresholds[task_id]


class BalrogAdapter(BenchmarkEvaluatorAdapter):
    """Adapter for Balrog benchmark — reward = mean progression across environments."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self._balrog_config = BalrogConfig.from_benchmark_config(config)
        self._threshold_tracker: Optional[AdaptiveThresholdTracker] = None
        if self._balrog_config.is_adaptive_threshold:
            self._threshold_tracker = AdaptiveThresholdTracker(
                margin=self._balrog_config.adaptive_threshold_margin,
            )

    def _inject_llm_config(self, agent_instance: Any) -> None:
        """Inject LLM config, BalrogConfig, and harness package name into the evaluator."""
        super()._inject_llm_config(agent_instance)
        # Inject balrog config into evaluator
        if hasattr(self._evaluator, '_balrog_config'):
            self._evaluator._balrog_config = self._balrog_config
            # LCB z-score: evolution.lcb_zscore (via agent config) is the single
            # source; the balrog yaml key is only a fallback default.
            self._evaluator._balrog_config.lcb_zscore = getattr(
                agent_instance.config, 'lcb_zscore', self._balrog_config.lcb_zscore
            )
        if self._threshold_tracker is not None:
            snapshot = self._threshold_tracker.get_snapshot()
            self._evaluator._balrog_config.task_thresholds = snapshot["task_thresholds"]
        # Inject the harness package name so the evaluator can resolve the
        # correct context module (not the stale godel_harness_init copy).
        if hasattr(self._evaluator, '_harness_package_name'):
            executor = getattr(agent_instance, 'action_executor', None)
            if executor and hasattr(executor, 'agent_code_dir') and executor.agent_code_dir:
                self._evaluator._harness_package_name = os.path.basename(executor.agent_code_dir)

    def __call__(self, *args, eval_mode: str = "dev", **kwargs):
        self._evaluator._eval_mode = eval_mode
        return super().__call__(*args, eval_mode=eval_mode, **kwargs)

    def evaluate_test_set(self, *args, repeat_idx: int = 0, **kwargs):
        # repeat_idx rotates the deterministic test seed across test_repeats
        # passes → each pass is a different map set, still reproducible. Reset
        # in finally so a later dev/val eval (which ignores it) sees the default.
        self._evaluator._eval_mode = "test"
        self._evaluator._test_repeat_idx = repeat_idx
        try:
            return super().evaluate_test_set(*args, **kwargs)
        finally:
            self._evaluator._test_repeat_idx = 0

    def _calculate_reward(
        self,
        results: List[TaskEvaluationResult],
        evaluate_seq: int = 0,
        iteration: int = 0,
        eval_mode: str = "dev",
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """Calculate reward from Balrog evaluation results.

        Reward signal depends on eval_mode:
        - dev/val (evolution): mean of per-task E_LCB (lcb_progression) — the
          uncertainty-aware mean−z·std/√n keeps version selection conservative
          against episode noise.
        - test: mean of raw avg_progression, matching the native Balrog
          benchmark's reported mean progression (LCB would under-report it).
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

        # Per-environment stats
        by_env: Dict[str, dict] = {}
        total_progression = 0.0
        total_raw_progression = 0.0
        total_episodes = 0

        for r in results:
            env_name = r.metadata.get("env_name", "unknown")
            # Reward signal = per-task E_LCB (lcb_progression); fall back to the
            # raw avg_progression for legacy logs that predate the LCB field.
            progression = r.metadata.get(
                "lcb_progression", r.metadata.get("avg_progression", 0.0)
            )
            raw_progression = r.metadata.get("avg_progression", 0.0)
            num_eps = r.metadata.get("total_episodes", 1)

            if env_name not in by_env:
                by_env[env_name] = {
                    "total_tasks": 0,
                    "total_progression": 0.0,
                    "total_episodes": 0,
                    "task_progressions": [],
                    "task_raw_progressions": [],
                }
            by_env[env_name]["total_tasks"] = by_env[env_name]["total_tasks"] + 1
            by_env[env_name]["total_progression"] += progression * num_eps
            by_env[env_name]["total_episodes"] += num_eps
            by_env[env_name]["task_progressions"].append(progression)
            by_env[env_name]["task_raw_progressions"].append(raw_progression)

            total_progression += progression
            total_raw_progression += raw_progression
            total_episodes += num_eps

        # Headline reward:
        # - dev/val (evolution): mean of per-task E_LCB → uncertainty-aware, so
        #   version selection stays conservative against episode noise.
        # - test: mean of raw avg_progression → matches the native benchmark's
        #   reported mean progression. Using LCB here would systematically
        #   under-report and diverge from the native number.
        use_lcb_reward = eval_mode != "test"
        if use_lcb_reward:
            reward = total_progression / len(results)
        else:
            reward = total_raw_progression / len(results)
        avg_raw_progression = total_raw_progression / len(results)

        # Per-env summary
        env_summary = {}
        for env_name, data in by_env.items():
            progs = data["task_progressions"]
            raw_progs = data["task_raw_progressions"]
            env_summary[env_name] = {
                "avg_progression": sum(raw_progs) / len(raw_progs) if raw_progs else 0.0,
                "avg_lcb_progression": sum(progs) / len(progs) if progs else 0.0,
                "num_tasks": data["total_tasks"],
                "total_episodes": data["total_episodes"],
                "task_progressions": [round(p, 4) for p in raw_progs],
                "task_lcb_progressions": [round(p, 4) for p in progs],
            }

        metrics = {
            "benchmark": "balrog",
            "reward": reward,
            "reward_type": "lcb" if use_lcb_reward else "mean",
            "lcb_zscore": getattr(self._balrog_config, "lcb_zscore", 1.0),
            "avg_raw_progression": avg_raw_progression,
            "total_tasks": len(results),
            "total_episodes": total_episodes,
            "by_environment": env_summary,
            "execution_errors": len(execution_errors),
            "task_results": [r.to_dict() for r in results],
        }

        self._save_logs(metrics, evaluate_seq, iteration)
        self._save_condensed_log(metrics, evaluate_seq, iteration)
        self._save_per_task_logs(metrics, evaluate_seq, iteration)

        if self._threshold_tracker is not None:
            pre_snapshot = self._threshold_tracker.get_snapshot()
            if eval_mode == "dev":
                for r in results:
                    self._threshold_tracker.update(
                        r.task_id, r.metadata.get("avg_progression", 0.0)
                    )
            snapshot = self._threshold_tracker.get_snapshot()
            metrics["adaptive_threshold"] = {
                "enabled": True,
                "margin": self._threshold_tracker.margin,
                "pre_task_thresholds": pre_snapshot["task_thresholds"],
                "pre_task_best_rewards": pre_snapshot["task_best_rewards"],
                **snapshot,
            }

        return reward, metrics

    def _save_logs(self, metrics: Dict, seq: int, iteration: int):
        """Save evaluation logs to disk."""
        log_dir = self._build_log_dir(seq, iteration)
        metrics["log_dir"] = log_dir

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"eval_{seq:03d}_{ts}.json")
        metrics["log_file"] = path
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"    [Eval] Saved balrog log to: {path}", flush=True)
