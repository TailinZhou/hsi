"""Context management — Episode message history for Balrog harness.

Uses incremental accumulation of raw message dicts instead of rebuilding
from simplified triples. This preserves the exact API response structure
(content, tool_calls, reasoning_content) to maintain cache prefix stability.

Message History Rules:
    The message list must follow the OpenAI Chat API conversation format.
    Violating these rules will cause API 400 errors ("messages parameter invalid").

    1. Only ONE system message is allowed, and it MUST be the first message.
       Do NOT append {"role": "system", ...} after user/assistant/tool messages.

    2. Messages must follow a valid role sequence:
       system -> user -> (assistant <-> tool)* -> user -> ...

    3. An "assistant" message with tool_calls may omit "content" (set to None or omit
       the key), but it MUST include the "tool_calls" field.

    4. get_messages() returns the internal list — hooks may modify it in-place.
"""

from typing import Any, Dict, List


class HarnessContext:
    """Episode-aware message history for Balrog game-playing agent.

    Accumulates raw message dicts incrementally. Each game step appends
    user/assistant/tool messages directly, preserving the exact structure
    returned by the API (content, tool_calls, reasoning_content).

    Evolution targets:
        MAX_TEXT_HISTORY   — tunable: max (obs, action) pairs to retain
        _reduce_history()  — evolvable: how to reduce when history exceeds limit
    """

    def __init__(self, system_prompt: str = None):
        self.message_history: List[Dict[str, Any]] = []
        if system_prompt:
            self.message_history.append({"role": "system", "content": system_prompt})

    # ─── FIXED: message accumulation ─────────────────────────────────
    def reset(self, system_prompt: str = None) -> None:
        """Clear all history and reset system prompt (new episode)."""
        self.message_history.clear()
        if system_prompt:
            self.message_history.append({"role": "system", "content": system_prompt})

    def add_user(self, content: str) -> None:
        self.message_history.append({"role": "user", "content": content})

    def add_assistant(self, message) -> None:
        """Store assistant message. Accepts plain string or raw API response."""
        if isinstance(message, str):
            self.message_history.append({"role": "assistant", "content": message})
            return
        entry: Dict[str, Any] = {"role": "assistant"}
        if isinstance(message, dict):
            if message.get("content"):
                entry["content"] = message["content"]
            if "tool_calls" in message and message["tool_calls"]:
                entry["tool_calls"] = message["tool_calls"]
            if message.get("reasoning_content"):
                entry["reasoning_content"] = message["reasoning_content"]
        else:
            if message.content:
                entry["content"] = message.content
            if hasattr(message, 'tool_calls') and message.tool_calls:
                entry["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ]
            if getattr(message, "reasoning_content", None):
                entry["reasoning_content"] = message.reasoning_content
        if not entry.get("content") and not entry.get("tool_calls"):
            entry["content"] = ""
        self.message_history.append(entry)

    def add_tool_result(self, tool_call_id: str, result: Any) -> None:
        """Append a tool result message."""
        self.message_history.append({
            "role": "tool", "tool_call_id": tool_call_id,
            "content": str(result)
        })

    # ─── EVOLVABLE: history management ───────────────────────────────
    # Tunable: max complete (obs, action) pairs to retain.
    MAX_TEXT_HISTORY = 16

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return messages for LLM API call. Triggers _reduce_history() when needed."""
        user_count = sum(1 for m in self.message_history if m.get("role") == "user")
        if user_count > 2 * self.MAX_TEXT_HISTORY and len(self.message_history) > 3:
            self._reduce_history()
        return self.message_history

    def _reduce_history(self):
        """Evolvable: how to reduce history when it exceeds MAX_TEXT_HISTORY.

        Default: cliff truncation — keep system prompt + last MAX_TEXT_HISTORY pairs.
        Evolve to: summarize old turns, prioritize key events, always keep first obs,
        compress repetitive observations, etc.
        """
        user_indices = [i for i, m in enumerate(self.message_history)
                        if m.get("role") == "user"]
        keep_from = user_indices[-(self.MAX_TEXT_HISTORY - 1)]
        truncated = []
        if self.message_history and self.message_history[0].get("role") == "system":
            truncated.append(self.message_history[0])
        truncated.extend(self.message_history[keep_from:])
        self.message_history = truncated
