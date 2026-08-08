"""AgentDojo solver - Evolvable task solving strategy.

Components: harness.py, prompts.py, hooks.py, tools_harness.py, context.py
"""
from .harness import using_harness as solve, StepResult
from .prompts import SYSTEM_PROMPT
from .hooks import hook_on_request, hook_on_response, hook_on_complete
from .context import HarnessContext

__all__ = [
    "solve",
    "StepResult",
    "SYSTEM_PROMPT",
    "hook_on_request", "hook_on_response", "hook_on_complete",
    "HarnessContext",
]
