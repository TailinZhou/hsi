"""
React Loop Agent Framework

A Godel-style self-evolving agent framework with anti-fragility focus.

Evolution formula: π_{t+1}, I_{t+1} = I_t(π_t, S_t, r_t, g)
- π_t: Current agent strategy/code
- S_t: Environment state
- r_t: Reward (including anti-fragility metrics)
- g: Goal
- I_t: Improver

Built-in core tools:
- bash/powershell: Shell command execution; after .py files are modified, the
  agent_codes mirror is synced automatically.
- read_history_self: read historical iteration conversations (@history syntax).
- read_file/edit_file/write_file: file operations.

Required tools:
- evaluate: external evaluation (collects r_t); the harness is fresh-imported
  from disk each time, guaranteeing the running bytes == code_hash.

Robustness improvements:
- validator: AST-level code safety validation (executed on edit/write).
"""

from .agent import GodelAgent, GodelAgentConfig
from .actions.agent_action import AgentActionExecutor
from .state import AgentState, AgentAction, ActionType, EvolutionPhase

__version__ = "0.4.0"
__all__ = [
    # Core components
    "GodelAgent",
    "GodelAgentConfig",
    "AgentActionExecutor",
    "AgentState",
    "AgentAction",
    "ActionType",
    "EvolutionPhase",
]
