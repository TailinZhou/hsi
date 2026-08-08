"""
AgentDojo Benchmark Configuration.

Provides configuration options for running AgentDojo benchmarks.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional


# Available AgentDojo suites
AVAILABLE_SUITES = [
    "workspace",
    "slack",
    "banking",
    "travel",
]

# Available attack types
AVAILABLE_ATTACKS = [
    "important_instructions",
    "tool_knowledge",
    "direct",
    "dos",
]

# Default suites if none specified
DEFAULT_SUITES = ["workspace"]

# Default benchmark version
DEFAULT_BENCHMARK_VERSION = "v1.2.2"


@dataclass
class AgentDojoBenchmarkConfig:
    """Configuration for AgentDojo benchmark runs.

    This configuration controls which AgentDojo suites and attacks to run,
    as well as various execution parameters.

    Attributes:
        benchmark_name: Name of the benchmark (default: "agentdojo")
        benchmark_version: AgentDojo benchmark version to use
        suites: List of task suites to run
        attack_type: Type of attack to use for injection tasks
        user_tasks: Specific user tasks to run (None for all)
        injection_tasks: Specific injection tasks to run (None for all)
        force_rerun: Force rerun even if cached results exist
        llm_model: LLM model to use for the agent
        llm_provider: LLM provider (openai, anthropic, etc.)
        max_retries: Maximum retries for failed tasks
        timeout_seconds: Timeout for each task execution
    """
    benchmark_name: str = "agentdojo"
    benchmark_version: str = DEFAULT_BENCHMARK_VERSION
    suites: List[str] = field(default_factory=lambda: DEFAULT_SUITES)
    attack_type: str = "important_instructions"
    user_tasks: Optional[List[str]] = None
    injection_tasks: Optional[List[str]] = None
    force_rerun: bool = False
    defense_mode: Literal["adaptive", "conservative", "aggressive"] = "adaptive"
    llm_model: str = "gpt-4o-mini"
    llm_provider: str = "openai"
    max_retries: int = 3
    timeout_seconds: int = 120
    use_real_llm: bool = True
    fail_fast: bool = False
    run_user_tasks: bool = True
    run_injection_tasks: bool = True
    log_dir: Optional[Path] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate suites
        invalid = [s for s in self.suites if s not in AVAILABLE_SUITES]
        if invalid:
            raise ValueError(
                f"Invalid suites: {invalid}. "
                f"Available suites: {AVAILABLE_SUITES}"
            )

        # Validate attack type
        if self.attack_type not in AVAILABLE_ATTACKS:
            raise ValueError(
                f"Invalid attack type: {self.attack_type}. "
                f"Available attacks: {AVAILABLE_ATTACKS}"
            )

    @property
    def agent_defense_mode(self) -> str:
        """Alias for defense_mode for compatibility."""
        return self.defense_mode

    @property
    def agentdojo_log_dir(self) -> Optional[Path]:
        """Get log directory for AgentDojo logs."""
        if self.log_dir:
            return self.log_dir / "agentdojo"
        return None
