"""
Benchmark Evaluators Package.

This package provides the core evaluator infrastructure for benchmark evaluation.
"""

from .task_types import (
    TaskCategory,
    BenchmarkTask,
    TaskEvaluationResult,
)
from .base import BaseTaskEvaluator
from .registry import (
    register_evaluator,
    get_evaluator,
    list_evaluators,
    is_evaluator_registered,
)

__all__ = [
    # Task types
    "TaskCategory",
    "BenchmarkTask",
    "TaskEvaluationResult",
    # Base class
    "BaseTaskEvaluator",
    # Registry functions
    "register_evaluator",
    "get_evaluator",
    "list_evaluators",
    "is_evaluator_registered",
]
