"""
Unified Task Types for Benchmark Evaluation.

This module defines common data structures for benchmark task evaluation
that are shared across all benchmark implementations.

Classes:
    TaskCategory: Enum for task categories (UTILITY, SECURITY)
    BenchmarkTask: Unified task representation from any benchmark
    TaskEvaluationResult: Simple evaluation result
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskCategory(str, Enum):
    """Task evaluation categories."""
    UTILITY = "utility"      # Task completion capability
    SECURITY = "security"    # Injection defense capability


@dataclass
class BenchmarkTask:
    """Unified task representation from any benchmark.

    Attributes:
        task_id: Unique identifier for the task
        instruction: The task instruction/prompt
        category: Task category (UTILITY or SECURITY)
        benchmark_source: Source benchmark name (e.g., "agentdojo")
        metadata: Additional task-specific information
    """
    task_id: str
    instruction: str
    category: TaskCategory
    benchmark_source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "category": self.category.value,
            "benchmark_source": self.benchmark_source,
            "metadata": self.metadata,
        }


@dataclass
class TaskEvaluationResult:
    """Simple evaluation result for a single task.

    Returns only raw data - agent should analyze results independently.

    Attributes:
        task_id: ID of the evaluated task
        success: Whether the task was completed successfully
        output: The actual output from the solver
        expected: Expected output (if available)
        execution_time: Time taken to execute (seconds)
        metadata: Additional result metadata (e.g., error message)
        interaction_log: List of interaction records (user, assistant, tool calls/results)
    """
    task_id: str
    success: bool
    output: Any
    expected: Optional[Any] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    interaction_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "execution_time": round(self.execution_time, 2),
            "metadata": self.metadata,
            "interaction_log": self.interaction_log,
        }
