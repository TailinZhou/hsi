"""
AgentDojo Benchmark Package.

This package provides AgentDojo benchmark integration.
"""

from .config import (
    AgentDojoBenchmarkConfig,
    AVAILABLE_SUITES,
    AVAILABLE_ATTACKS,
    DEFAULT_SUITES,
    DEFAULT_BENCHMARK_VERSION,
)
from .suite_loader import SuiteLoader, LoadedSuite
from .agentdojo_evaluator import AgentDojoEvaluator
from .output_normalizer import apply_patches, revert_patches

__all__ = [
    # Config
    "AgentDojoBenchmarkConfig",
    "AVAILABLE_SUITES",
    "AVAILABLE_ATTACKS",
    "DEFAULT_SUITES",
    "DEFAULT_BENCHMARK_VERSION",
    # Suite loader
    "SuiteLoader",
    "LoadedSuite",
    # Evaluator
    "AgentDojoEvaluator",
    # Output normalization
    "apply_patches",
    "revert_patches",
]
