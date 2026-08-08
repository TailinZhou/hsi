"""Balrog solver — evolvable game-playing strategy for 6 text-based environments."""

from .harness import using_harness as solve
from .prompts import SYSTEM_PROMPT
from .hooks import hook_on_request, hook_on_response, hook_on_complete
from .context import HarnessContext

__all__ = [
    "solve",
    "SYSTEM_PROMPT",
    "hook_on_request", "hook_on_response", "hook_on_complete",
    "HarnessContext",
]
