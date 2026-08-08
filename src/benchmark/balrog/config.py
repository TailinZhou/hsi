"""Balrog benchmark configuration.

Replaces Hydra/OmegaConf config from HyperAgents with a simple dataclass.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class _AttrDict(dict):
    """Dict subclass that allows attribute-style access (like OmegaConf).

    Recursively converts nested dicts so config.envs.babyai_kwargs works.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, _AttrDict):
                self[key] = _AttrDict(value)

    def __deepcopy__(self, memo):
        import copy
        new = self.__class__()
        memo[id(self)] = new
        for key, value in self.items():
            new[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return new

    def __repr__(self):
        return dict.__repr__(self)


# Default number of episodes per environment
DEFAULT_NUM_EPISODES: Dict[str, int] = {
    "nle": 5,
    "minihack": 5,
    "babyai": 10,
    "crafter": 10,
    "babaisai": 3,
    "textworld": 10,
}

# Default environment kwargs (from HyperAgents config.yaml)
DEFAULT_ENV_KWARGS: Dict[str, Dict[str, Any]] = {
    "nle": {
        "character": "@",
        "max_episode_steps": 100_000,
        "no_progress_timeout": 150,
        "savedir": None,
        "save_ttyrec_every": 0,
        "skip_more": True,
    },
    "minihack": {
        "character": "@",
        "max_episode_steps": 100,
        "penalty_step": -0.01,
        "penalty_time": 0.0,
        "penalty_mode": "constant",
        "savedir": None,
        "save_ttyrec_every": 0,
        "autopickup": False,
        "skip_more": True,
    },
    "babyai": {
        "num_dists": 0,
    },
    "crafter": {
        "area": [64, 64],
        "view": [9, 9],
        "size": [256, 256],
        "reward": True,
        "seed": None,
        "max_episode_steps": 2000,
    },
    "textworld": {
        "objective": True,
        "description": True,
        "score": True,
        "max_score": True,
        "won": True,
        "max_episode_steps": 80,
        "textworld_games_path": "tw_games",
    },
    "babaisai": {
        "add_ruleset": True,
    },
}


def _resolve_env_key(env_name: str) -> str:
    """Resolve sub-suite env name to base env for config fallback.

    babaisai_goto → babaisai, etc. Top-level names pass through.
    """
    for base in ("babaisai",):
        if env_name == base or env_name.startswith(base + "_"):
            return base
    return env_name


@dataclass
class BalrogConfig:
    """Configuration for Balrog benchmark evaluation.

    Attributes:
        env_names: List of environment names to evaluate
        num_episodes: Per-environment episode counts
        max_steps_per_episode: Override max steps (None = use env default)
        feedback_on_invalid_action: Whether to give feedback on invalid actions
        failed_threshold: avg_progression below this value counts as failed (default 0.0)
        env_kwargs: Per-environment keyword arguments
        seed: Global seed for reproducible episode generation (None = non-deterministic)
    """

    env_names: List[str] = field(default_factory=lambda: ["babyai"])
    num_episodes: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_NUM_EPISODES))
    num_episodes_dev: Dict[str, int] = field(default_factory=dict)
    max_steps_per_episode: Optional[int] = None
    max_steps_per_episode_dev: Optional[int] = None
    max_steps_per_episode_non_dev: Optional[int] = None
    feedback_on_invalid_action: bool = True
    failed_threshold: float = 0.0
    adaptive_threshold_margin: float = 0.05
    task_thresholds: Dict[str, float] = field(default_factory=dict)
    env_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: dict(DEFAULT_ENV_KWARGS))
    seed: Optional[int] = None
    include_ascii_map: bool = False
    episode_workers: int = 1
    lcb_zscore: float = 1.0  # LCB uncertainty penalty (fallback; single source is agent.config.lcb_zscore)

    @property
    def is_adaptive_threshold(self) -> bool:
        """True when failed_threshold < 0 (sentinel for adaptive mode)."""
        return self.failed_threshold < 0

    @classmethod
    def from_benchmark_config(cls, benchmark_config) -> "BalrogConfig":
        """Create BalrogConfig from a BenchmarkConfig instance.

        Reads balrog-specific params from benchmark_config._raw_yaml (the
        "balrog" section in config.yaml) if present, otherwise uses defaults.
        """
        suite = getattr(benchmark_config, "suite", "babyai") or "babyai"

        if suite == "all":
            env_names = list(DEFAULT_NUM_EPISODES.keys())
        else:
            env_names = [s.strip() for s in suite.split("-")]

        # Read balrog-specific overrides from yaml if available
        raw = getattr(benchmark_config, '_raw_yaml', None) or {}
        balrog_cfg = raw.get("balrog", {}) if isinstance(raw, dict) else {}

        # num_episodes: merge defaults with yaml overrides
        num_episodes = dict(DEFAULT_NUM_EPISODES)
        yaml_episodes = balrog_cfg.get("num_episodes", {})
        if isinstance(yaml_episodes, dict):
            num_episodes.update(yaml_episodes)

        # num_episodes_dev: cheap per-env override for dev and val mode (exploration).
        # Optional, per-env; an env without an entry falls back to num_episodes for dev/val.
        yaml_episodes_dev = balrog_cfg.get("num_episodes_dev", {})
        num_episodes_dev = dict(yaml_episodes_dev) if isinstance(yaml_episodes_dev, dict) else {}

        # env_kwargs: merge defaults with yaml overrides
        env_kwargs = {k: dict(v) for k, v in DEFAULT_ENV_KWARGS.items()}
        yaml_kwargs = balrog_cfg.get("env_kwargs", {})
        if isinstance(yaml_kwargs, dict):
            for env_key, kwarg_overrides in yaml_kwargs.items():
                if not isinstance(kwarg_overrides, dict):
                    continue
                if env_key in env_kwargs:
                    env_kwargs[env_key].update(kwarg_overrides)
                else:
                    # Fallback: babaisai_goto → babaisai for sub-suite envs
                    base = _resolve_env_key(env_key)
                    if base != env_key and base in env_kwargs:
                        env_kwargs[base].update(kwarg_overrides)

        return cls(
            env_names=env_names,
            num_episodes=num_episodes,
            num_episodes_dev=num_episodes_dev,
            max_steps_per_episode=balrog_cfg.get("max_steps_per_episode", None),
            max_steps_per_episode_dev=balrog_cfg.get("max_steps_per_episode_dev", None),
            max_steps_per_episode_non_dev=balrog_cfg.get("max_steps_per_episode_non_dev", None),
            feedback_on_invalid_action=balrog_cfg.get("feedback_on_invalid_action", True),
            failed_threshold=balrog_cfg.get("failed_threshold", 0.0),
            adaptive_threshold_margin=balrog_cfg.get("adaptive_threshold_margin", 0.05),
            env_kwargs=env_kwargs,
            seed=balrog_cfg.get("seed"),
            include_ascii_map=balrog_cfg.get("include_ascii_map", False),
            episode_workers=balrog_cfg.get("episode_workers", 1),
            lcb_zscore=float(balrog_cfg.get("lcb_zscore", 1.0)),
        )

    def get_max_steps(self, eval_mode: str) -> Optional[int]:
        """Return max_steps based on eval_mode. Fallback to max_steps_per_episode."""
        specific = self.max_steps_per_episode_dev if eval_mode == "dev" else self.max_steps_per_episode_non_dev
        return specific if specific is not None else self.max_steps_per_episode

    def get_num_episodes(self, env_name: str, eval_mode: str, fallback: int = 5) -> int:
        """Per-task episode count for an eval mode.

        dev and val use the cheap num_episodes_dev override when one exists for
        the env (else num_episodes); test/final always use num_episodes (the
        honest value for the post-evolution test eval).
        """
        if eval_mode in ("dev", "val") and env_name in self.num_episodes_dev:
            return self.num_episodes_dev[env_name]
        # Fallback: babaisai_goto → babaisai for envs not explicitly configured
        if eval_mode in ("dev", "val"):
            base = _resolve_env_key(env_name)
            if base != env_name and base in self.num_episodes_dev:
                return self.num_episodes_dev[base]
        if env_name not in self.num_episodes:
            base = _resolve_env_key(env_name)
            if base != env_name:
                return self.num_episodes.get(base, fallback)
        return self.num_episodes.get(env_name, fallback)

    def to_hyperagents_config(self) -> _AttrDict:
        """Convert to an AttrDict mimicking the HyperAgents OmegaConf config structure.

        Environment factories use attribute access (config.envs.babyai_kwargs),
        so we return _AttrDict which supports both dict and attribute access.
        """
        return _AttrDict({
            "envs": {
                "names": "-".join(self.env_names),
                "include_ascii_map": self.include_ascii_map,
                "env_kwargs": {"seed": None},
                "nle_kwargs": self.env_kwargs.get("nle", {}),
                "minihack_kwargs": self.env_kwargs.get("minihack", {}),
                "babyai_kwargs": self.env_kwargs.get("babyai", {}),
                "crafter_kwargs": self.env_kwargs.get("crafter", {}),
                "textworld_kwargs": self.env_kwargs.get("textworld", {}),
                "babaisai_kwargs": self.env_kwargs.get("babaisai", {}),
            },
            "eval": {
                "num_episodes": dict(self.num_episodes),
                "max_steps_per_episode": self.max_steps_per_episode,
                "feedback_on_invalid_action": self.feedback_on_invalid_action,
            },
            "tasks": TASK_DEFINITIONS,
        })


# Lazy import to avoid circular dependency
from .tasks import TASK_DEFINITIONS  # noqa: E402
