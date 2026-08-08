"""React hooks — no-op implementation for Balrog harness.

# ─── EVOLVABLE ───────────────────────────────────────────────────
"""

from typing import Any, Dict, List, Tuple


def hook_on_request(messages: List[Dict], tools: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    return messages, tools


def hook_on_response(response: Any) -> Any:
    return response


def hook_on_complete(response: Any, tool_calls_made: List[Dict], tool_results: List[str]) -> None:
    pass
