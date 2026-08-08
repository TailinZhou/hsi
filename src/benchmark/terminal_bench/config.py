"""Terminal-Bench 2 configuration."""

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TerminalBenchConfig:
    """Configuration for Terminal-Bench 2 evaluation.

    Harbor runs tasks in Runloop sandboxes via CLI.
    """

    # Harbor CLI settings
    harbor_executable: str = "harbor"  # or full path
    dataset: str = "terminal-bench@2.0"
    sandbox: str = "runloop"  # runloop | docker

    # Agent settings
    litellm_model: str = "openai/gpt-4o"  # model for GodelAgentProxy via litellm
    litellm_api_base: Optional[str] = None  # custom API endpoint (e.g. "https://open.bigmodel.cn/api/paas/v4")
    litellm_api_key: Optional[str] = None  # API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env)
    litellm_temperature: Optional[float] = None  # from harness.temperature in config.yaml
    litellm_thinking_enabled: Optional[bool] = None  # from llm.thinking_enabled
    litellm_reasoning_effort: Optional[str] = None  # from llm.reasoning_effort
    max_concurrent: int = 4  # harbor --n-concurrent

    # Timeout
    task_timeout: int = 600  # seconds per task
    overall_timeout: int = 3600  # seconds for full run
    agent_setup_timeout_multiplier: Optional[float] = None  # multiplier for Harbor agent setup timeout (default 360s)
    environment_build_timeout_multiplier: Optional[float] = None  # multiplier for Harbor env build timeout

    # Task filtering
    task_ids: Optional[list] = None  # None = all tasks
    categories: Optional[list] = None

    # Paths
    output_dir: Optional[str] = None  # harbor output dir, auto-generated if None

    # Environment
    env_vars: dict = field(default_factory=dict)

    @classmethod
    def from_benchmark_config(cls, config) -> "TerminalBenchConfig":
        """Create from a BenchmarkConfig instance.

        Reads terminal_bench-specific params from config._raw_yaml
        (the "terminal_bench" section in config.yaml).
        Environment variables take precedence over yaml values.
        """
        # Read terminal_bench section from raw yaml (same pattern as BalrogConfig)
        raw = getattr(config, "_raw_yaml", None) or {}
        tb2_cfg = raw.get("terminal_bench", {}) if isinstance(raw, dict) else {}

        # litellm model: env > yaml > default
        litellm_model = (
            os.environ.get("TB2_LITELLM_MODEL")
            or tb2_cfg.get("litellm_model")
            or "openai/gpt-4o"
        )

        # litellm API base: env > yaml > None
        litellm_api_base = (
            os.environ.get("TB2_LITELLM_API_BASE")
            or tb2_cfg.get("litellm_api_base")
            or None
        )

        # litellm API key: env > yaml > OPENAI_API_KEY fallback
        litellm_api_key = (
            os.environ.get("TB2_LITELLM_API_KEY")
            or tb2_cfg.get("litellm_api_key")
            or os.environ.get("OPENAI_API_KEY")
            or None
        )

        # Harbor executable: env > yaml > PATH lookup
        harbor_executable = (
            os.environ.get("HARBOR_EXECUTABLE")
            or tb2_cfg.get("harbor_executable")
            or shutil.which("harbor")
            or "harbor"
        )

        # Temperature: harness.temperature > llm.temperature > None
        harness_cfg = raw.get("harness", {}) if isinstance(raw, dict) else {}
        litellm_temperature = harness_cfg.get("temperature")
        if litellm_temperature is None:
            llm_cfg = raw.get("llm", {}) if isinstance(raw, dict) else {}
            litellm_temperature = llm_cfg.get("temperature")
        litellm_temperature = os.environ.get("TB2_LITELLM_TEMPERATURE") or litellm_temperature
        if litellm_temperature is not None:
            litellm_temperature = float(litellm_temperature)

        # Thinking: llm.thinking_enabled / llm.reasoning_effort
        llm_cfg = raw.get("llm", {}) if isinstance(raw, dict) else {}
        litellm_thinking_enabled = (
            os.environ.get("TB2_THINKING_ENABLED", "").lower() in ("true", "1")
            if "TB2_THINKING_ENABLED" in os.environ
            else llm_cfg.get("thinking_enabled", None)
        )
        litellm_reasoning_effort = (
            os.environ.get("TB2_REASONING_EFFORT")
            or llm_cfg.get("reasoning_effort")
            or None
        )

        # Timeout multipliers: env > yaml > None
        agent_setup_timeout_multiplier = (
            float(os.environ["TB2_AGENT_SETUP_TIMEOUT_MULTIPLIER"])
            if "TB2_AGENT_SETUP_TIMEOUT_MULTIPLIER" in os.environ
            else tb2_cfg.get("agent_setup_timeout_multiplier")
        )
        environment_build_timeout_multiplier = (
            float(os.environ["TB2_ENV_BUILD_TIMEOUT_MULTIPLIER"])
            if "TB2_ENV_BUILD_TIMEOUT_MULTIPLIER" in os.environ
            else tb2_cfg.get("environment_build_timeout_multiplier")
        )

        return cls(
            harbor_executable=harbor_executable,
            litellm_model=litellm_model,
            litellm_api_base=litellm_api_base,
            litellm_api_key=litellm_api_key,
            litellm_temperature=litellm_temperature,
            litellm_thinking_enabled=litellm_thinking_enabled,
            litellm_reasoning_effort=litellm_reasoning_effort,
            dataset=tb2_cfg.get("dataset", "terminal-bench@2.0"),
            sandbox=tb2_cfg.get("sandbox", "docker"),
            max_concurrent=int(tb2_cfg.get("max_concurrent", 4)),
            task_timeout=int(tb2_cfg.get("task_timeout", 600)),
            overall_timeout=int(tb2_cfg.get("overall_timeout", 3600)),
            agent_setup_timeout_multiplier=agent_setup_timeout_multiplier,
            environment_build_timeout_multiplier=environment_build_timeout_multiplier,
            task_ids=tb2_cfg.get("task_ids"),
            categories=config.categories if hasattr(config, "categories") else None,
            env_vars={
                k: v
                for k, v in os.environ.items()
                if k.startswith(("ANTHROPIC_", "OPENAI_", "RUNLOOP_", "TB2_"))
            },
        )
