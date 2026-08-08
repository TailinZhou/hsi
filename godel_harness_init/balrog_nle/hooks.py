"""React hooks for NLE harness — tail instruction injection.

# ─── EVOLVABLE ───────────────────────────────────────────────────"""

from typing import Any, Dict, List, Tuple

TAIL_INSTRUCTION = (
    "\n\nYou always have to output one of the above actions at a time and no other text. "
    "You always have to output an action until the episode terminates."
)


def hook_on_request(messages: List[Dict], tools: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    # Append tail instruction to last user message
    if messages and messages[-1].get("role") == "user":
        messages[-1]["content"] = messages[-1]["content"] + TAIL_INSTRUCTION
    return messages, tools


def hook_on_response(response: Any) -> Any:
    return response


def hook_on_complete(response: Any, tool_calls_made: List[Dict], tool_results: List[str]) -> None:
    pass
