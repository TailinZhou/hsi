"""
Benchmark Package.

This package provides benchmark evaluation infrastructure for the agent.
It supports various benchmarks (AgentDojo, etc.) through a unified interface.

Usage:
    from benchmark import BenchmarkEvaluatorAdapter, create_benchmark_evaluator

    # Create an adapter
    adapter = create_benchmark_evaluator(
        benchmark_type="agentdojo",
        suite="workspace",
    )

    # Use as external evaluator with GodelAgent
    config = GodelAgentConfig(
        benchmark_type="agentdojo",
        benchmark_suite="workspace",
    )
    agent = GodelAgent(..., config=config)
"""

from .adapter import BenchmarkEvaluatorAdapter, create_benchmark_evaluator
from .config import BenchmarkConfig
from .evaluators import (
    # Types
    TaskCategory,
    BenchmarkTask,
    TaskEvaluationResult,
    # Base class
    BaseTaskEvaluator,
    # Registry
    register_evaluator,
    get_evaluator,
    list_evaluators,
)

# Import AgentDojo components (may fail if agentdojo not installed)
try:
    from .agentdojo import (
        AgentDojoBenchmarkConfig,
        AgentDojoEvaluator,
        SuiteLoader,
        LoadedSuite,
        AVAILABLE_SUITES,
        AVAILABLE_ATTACKS,
        DEFAULT_SUITES,
        DEFAULT_BENCHMARK_VERSION,
    )
except ImportError:
    # AgentDojo not available
    pass

# Import Terminal-Bench 2 components (may fail if harbor/litellm not installed)
try:
    from .terminal_bench import (
        TerminalBenchAdapter,
        TerminalBenchConfig,
        TerminalBenchEvaluator,
        get_all_task_ids as terminal_bench_task_ids,
        get_categories as terminal_bench_categories,
    )
except ImportError:
    # Terminal-Bench 2 not available
    pass

__all__ = [
    # Adapter
    "BenchmarkEvaluatorAdapter",
    "create_benchmark_evaluator",
    "BenchmarkConfig",
    # Core types
    "TaskCategory",
    "BenchmarkTask",
    "TaskEvaluationResult",
    # Base class
    "BaseTaskEvaluator",
    # Registry
    "register_evaluator",
    "get_evaluator",
    "list_evaluators",
]
