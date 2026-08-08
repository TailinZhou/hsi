"""Terminal-Bench 2 benchmark integration package.

Provides adapter, evaluator, and runner for evaluating agents on
Terminal-Bench 2's 89 terminal tasks via Harbor.
"""

# Lazy imports to avoid circular dependency during registry auto-registration.
# TerminalBenchAdapter imports benchmark.adapter which may not be fully loaded yet.


def __getattr__(name):
    if name == "TerminalBenchAdapter":
        from benchmark.terminal_bench.adapter import TerminalBenchAdapter
        return TerminalBenchAdapter
    if name == "TerminalBenchConfig":
        from benchmark.terminal_bench.config import TerminalBenchConfig
        return TerminalBenchConfig
    if name == "TerminalBenchEvaluator":
        from benchmark.terminal_bench.evaluator import TerminalBenchEvaluator
        return TerminalBenchEvaluator
    if name in ("get_all_task_ids", "get_categories"):
        from benchmark.terminal_bench import tasks
        return getattr(tasks, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
