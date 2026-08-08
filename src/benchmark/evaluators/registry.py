"""
Evaluator Registry for Benchmark Evaluation.

This module provides a registry mechanism for benchmark evaluators,
allowing dynamic registration and retrieval of evaluators by name.
"""

import logging
from typing import Dict, List, Optional, Type

from .base import BaseTaskEvaluator

logger = logging.getLogger(__name__)

# Global evaluator registry
_EVALUATOR_REGISTRY: Dict[str, Type[BaseTaskEvaluator]] = {}


def register_evaluator(evaluator_class: Type[BaseTaskEvaluator]) -> None:
    """Register an evaluator class.

    Args:
        evaluator_class: The evaluator class to register
    """
    try:
        instance = evaluator_class()
        name = instance.benchmark_name
        _EVALUATOR_REGISTRY[name] = evaluator_class
        logger.debug(f"Registered evaluator: {name}")
    except Exception as e:
        logger.warning(f"Failed to register evaluator {evaluator_class}: {e}")


def get_evaluator(benchmark_name: str) -> Optional[BaseTaskEvaluator]:
    """Get an evaluator instance by benchmark name.

    Args:
        benchmark_name: Name of the benchmark (e.g., "agentdojo")

    Returns:
        Evaluator instance or None if not found
    """
    evaluator_class = _EVALUATOR_REGISTRY.get(benchmark_name)
    if evaluator_class:
        try:
            return evaluator_class()
        except Exception as e:
            logger.error(f"Failed to instantiate evaluator {benchmark_name}: {e}")
            return None
    return None


def list_evaluators() -> List[str]:
    """List all registered evaluator names.

    Returns:
        List of registered benchmark names
    """
    return list(_EVALUATOR_REGISTRY.keys())


def is_evaluator_registered(benchmark_name: str) -> bool:
    """Check if an evaluator is registered.

    Args:
        benchmark_name: Name of the benchmark

    Returns:
        True if evaluator is registered, False otherwise
    """
    return benchmark_name in _EVALUATOR_REGISTRY



def _auto_register() -> None:
    """Auto-register built-in evaluators.

    This function attempts to import and register all built-in evaluators.
    It handles ImportError gracefully to allow partial functionality.
    """
    # Try to register AgentDojo evaluator
    try:
        from benchmark.agentdojo.agentdojo_evaluator import AgentDojoEvaluator
        register_evaluator(AgentDojoEvaluator)
    except ImportError as e:
        logger.debug(f"AgentDojoEvaluator not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to register AgentDojoEvaluator: {e}")

    # Register classification evaluators (one per domain)
    try:
        from benchmark.classification.classification_evaluator import ClassificationEvaluator
        for domain_name in ("paper_review", "search_arena", "imo_grading"):
            try:
                _EVALUATOR_REGISTRY[domain_name] = lambda dn=domain_name: ClassificationEvaluator(domain_name=dn)
                logger.debug(f"Registered evaluator: {domain_name}")
            except Exception as e:
                logger.warning(f"Failed to register classification evaluator for {domain_name}: {e}")
    except ImportError as e:
        logger.debug(f"ClassificationEvaluator not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to register classification evaluators: {e}")

    # Register polyglot evaluator
    try:
        from benchmark.polyglot.evaluator import PolyglotEvaluator
        register_evaluator(PolyglotEvaluator)
    except ImportError as e:
        logger.debug(f"PolyglotEvaluator not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to register PolyglotEvaluator: {e}")

    # Register genesis evaluator
    try:
        from benchmark.genesis.evaluator import GenesisEvaluator
        register_evaluator(GenesisEvaluator)
    except ImportError as e:
        logger.debug(f"GenesisEvaluator not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to register GenesisEvaluator: {e}")

    # Register balrog evaluator
    try:
        from benchmark.balrog.evaluator import BalrogEvaluator
        register_evaluator(BalrogEvaluator)
    except ImportError as e:
        logger.debug(f"BalrogEvaluator not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to register BalrogEvaluator: {e}")

    # Register terminal_bench evaluator
    try:
        from benchmark.terminal_bench.evaluator import TerminalBenchEvaluator
        register_evaluator(TerminalBenchEvaluator)
    except ImportError as e:
        logger.debug(f"TerminalBenchEvaluator not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to register TerminalBenchEvaluator: {e}")


# Run auto-registration on module import
_auto_register()
