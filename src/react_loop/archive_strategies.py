"""
Archive Strategy Library — pluggable seed selection strategies.

Provides a registry of named strategies that the select_seed.py template can look up.
Each strategy is a callable::

    strategy(agent) -> SeedResult

Strategies are discovered from ``evolution/strategies/*.py`` via
``discover_strategies()``.  Additional strategies can also be registered
manually via ``register_strategy()``.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ── Result type ──────────────────────────────────────────────────────────────

@dataclass
class SeedResult:
    """Unified return type for all archive strategies.

    Attributes:
        git_hash: Target commit to switch working tree to.
        strategy_hint: Human-readable hint (shown in logs).
        hypothesis: Multi-part hypothesis for the evolve agent (rationale,
            expected improvement, falsification criteria, bootstrap permission).
        metadata: Arbitrary strategy-specific data.
        merge_ops: Optional list of merge operations to apply *after* checkout.
            Each element is a dict, e.g. {"source_hash": str, "files": [str, ...]}.
    """
    git_hash: str = ""
    strategy_hint: str = ""
    hypothesis: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    merge_ops: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "git_hash": self.git_hash,
            "strategy_hint": self.strategy_hint,
            "hypothesis": self.hypothesis,
            "metadata": self.metadata,
            "merge_ops": self.merge_ops,
        }


# ── Registry ─────────────────────────────────────────────────────────────────

DEFAULT_STRATEGY = "greedy"

STRATEGY_REGISTRY: Dict[str, Callable] = {}


def register_strategy(name: str):
    """Decorator that registers a strategy callable under *name*."""
    def decorator(fn: Callable):
        STRATEGY_REGISTRY[name] = fn
        return fn
    return decorator


def get_strategy(name: str) -> Optional[Callable]:
    """Look up a strategy by name.  Returns None if not found."""
    return STRATEGY_REGISTRY.get(name)


# ── Strategy discovery ────────────────────────────────────────────────────────

def discover_strategies(evolution_dir: str) -> List[str]:
    """Scan evolution/strategies/*.py and register each strategy.

    Args:
        evolution_dir: Path to the evolution/ directory (e.g. repo/evolution).

    Returns:
        List of newly registered strategy names.
    """
    import importlib.util
    from pathlib import Path

    # Accept both parent dir (containing strategies/) and strategies/ dir itself
    candidate = Path(evolution_dir)
    if candidate.name == "strategies" and candidate.is_dir():
        strategies_dir = str(candidate)
    else:
        strategies_dir = str(candidate / "strategies")
    if not os.path.isdir(strategies_dir):
        return []

    registered = []
    for py_file in sorted(Path(strategies_dir).glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                f"evolution_strategy_{py_file.stem}", str(py_file)
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            name = getattr(mod, "STRATEGY_NAME", py_file.stem)
            fn = getattr(mod, "strategy", None)
            if fn:
                STRATEGY_REGISTRY[name] = fn
                registered.append(name)
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Skipping strategy %s: %s", py_file.name, e
            )

    return registered
