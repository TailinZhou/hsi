"""Message context management for Terminal-Bench 2 harness."""

import copy
from typing import Any, Dict, List

try:
    import tiktoken
    _tiktoken_enc = tiktoken.encoding_for_model("gpt-4")
except (ImportError, Exception):
    tiktoken = None
    _tiktoken_enc = None


class HarnessContext:
    """Manages conversation message history for the harness react loop.

    Evolution targets:
        MAX_CONTEXT_TOKENS  — tunable: token budget
        needs_summarization() — evolvable: when to trigger summarization
        _reduce_history()   — evolvable: how to reduce when context exceeds limit
    """

    def __init__(self, system_prompt: str):
        self.message_history: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

    # ─── FIXED: message accumulation ─────────────────────────────

    def add_user(self, content: str):
        """Add a user message."""
        self.message_history.append({"role": "user", "content": content})

    def add_assistant_with_tools(self, message: Dict[str, Any]):
        """Add an assistant message (may include tool_calls)."""
        self.message_history.append({"role": "assistant", **message})

    def add_assistant_text(self, content: str):
        """Add a plain text assistant message (no tool_calls)."""
        self.message_history.append({"role": "assistant", "content": content})

    def add_tool_result(self, tool_call_id: str, result: str):
        """Add a tool result message."""
        self.message_history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        })

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return a copy of the current message history."""
        return copy.deepcopy(self.message_history)

    def get_final_answer(self) -> str:
        """Extract the final answer from the last assistant message."""
        for msg in reversed(self.message_history):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]
        return ""

    def get_system_message(self):
        """Return the system message (first entry in history)."""
        return self.message_history[0]

    def replace_messages(self, new_messages):
        """Replace entire message history (for summarization handoff)."""
        self.message_history = list(new_messages)

    # ─── EVOLVABLE: context management ────────────────────────────

    MAX_CONTEXT_TOKENS = 200000

    def count_tokens(self) -> int:
        """Count approximate tokens in message history."""
        if _tiktoken_enc is not None:
            total = 0
            for msg in self.message_history:
                total += len(_tiktoken_enc.encode(str(msg.get("content", ""))))
                total += 4
            return total
        total = 0
        for msg in self.message_history:
            total += len(str(msg.get("content", ""))) // 4
            total += 4
        return total

    def get_free_tokens(self, max_tokens: int = MAX_CONTEXT_TOKENS) -> int:
        """Return remaining token space."""
        return max_tokens - self.count_tokens()

    def needs_summarization(self, threshold: int = 8000) -> bool:
        """Evolvable: when to trigger summarization."""
        return self.get_free_tokens() < threshold

    def _estimate_msg_tokens(self, msg: Dict) -> int:
        """Rough token estimate for a single message."""
        content = str(msg.get("content", ""))
        if _tiktoken_enc is not None:
            return len(_tiktoken_enc.encode(content)) + 4
        return len(content) // 4 + 4

    def unwind_messages(self, target_free_tokens: int = 4000) -> None:
        """Emergency: remove oldest messages to free up token space."""
        current = self.count_tokens()
        while self.MAX_CONTEXT_TOKENS - current < target_free_tokens and len(self.message_history) > 3:
            removed = self.message_history.pop(1)
            current -= self._estimate_msg_tokens(removed)

    def compress_history(self, system_msg, summary_msg, recent):
        """Replace message history with compressed version."""
        self.message_history = [system_msg, summary_msg] + recent
