"""
Benchmark Configuration.

Centralized configuration for benchmark evaluation, independent from agent config.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation.

    Attributes:
        type: Benchmark type (e.g., "agentdojo")
        suite: Benchmark suite name (e.g., "workspace", "banking")
        attack: Attack type for security evaluation (direct, important_instructions, tool_knowledge, dos)
        categories: Task categories to evaluate (e.g., ["utility", "security"])
        max_tasks_per_category: Maximum tasks per category for dev/val sets (None for all)
        max_tasks_per_category_test: Maximum tasks per category for test set (None for all, independent from dev/val)
        verbose: Print detailed eval interaction logs
        dev_ratio: Fraction of tasks for dev set. 1.0 = no test split (backward compatible)
        split_seed: Random seed for deterministic dev/test split
        val_ratio: Fraction of dev set to further split into val. 0.0 = no val set
        mode: "batch" (evaluate all dev tasks) | "onthefly" (reserved, not implemented)
        dynamic_sample: True = re-sample dev only from non-test pool on each evaluate call (val/test pre-fixed); False = dev/val/test all fixed
    """
    type: str = "agentdojo"
    suite: str = "workspace"
    attack: str = "important_instructions"
    categories: Optional[List[str]] = None
    max_tasks_per_category: Optional[int] = None
    max_tasks_per_category_test: Optional[int] = None
    verbose: bool = False
    dev_ratio: float = 1.0
    split_seed: int = 42
    val_ratio: float = 0.0
    mode: str = "batch"
    data_root: Optional[str] = None
    run_all_tasks: bool = False
    dynamic_sample: bool = True
    parallel_workers: int = 1  # Parallel evaluation worker threads (1 = serial)
    summary_passed_samples: int = 0  # Number of passed tasks sampled in eval summary (0 = summarize failed tasks only)
