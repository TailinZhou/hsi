"""Message context management for AgentDojo harness."""

from typing import Any, Dict, List


class HarnessContext:
    """Manages conversation message history for the harness react loop."""

    def __init__(self, system_prompt: str = None):
        self.message_history: List[Dict] = []
        if system_prompt:
            self.message_history.append({"role": "system", "content": system_prompt})

    def add_user(self, content: str) -> None:
        """Add a user message."""
        self.message_history.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        """Add a plain text assistant message (no tool_calls)."""
        self.message_history.append({"role": "assistant", "content": content})

    def add_assistant_with_tools(self, message) -> None:
        """Add an assistant message with tool calls.

        Args:
            message: dict format with content, tool_calls, and optional reasoning_content
        """
        entry = {"role": "assistant"}

        # Support dict format
        if isinstance(message, dict):
            if message.get("content"):
                entry["content"] = message["content"]
            if "tool_calls" in message and message["tool_calls"]:
                entry["tool_calls"] = message["tool_calls"]
            if message.get("reasoning_content"):
                entry["reasoning_content"] = message["reasoning_content"]
        else:
            # Support OpenAI message object
            if message.content:
                entry["content"] = message.content
            if hasattr(message, 'tool_calls') and message.tool_calls:
                entry["tool_calls"] = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ]
            if getattr(message, "reasoning_content", None):
                entry["reasoning_content"] = message.reasoning_content
        self.message_history.append(entry)

    def add_tool_result(self, tool_call_id: str, result: Any) -> None:
        """Add a tool result message."""
        self.message_history.append({"role": "tool", "tool_call_id": tool_call_id, "content": str(result)})

    def get_messages(self) -> List[Dict]:
        """Return the current message history."""
        return self.message_history

    def get_final_answer(self) -> str:
        """Extract the final answer from the last assistant message with content."""
        for msg in reversed(self.message_history):
            if msg["role"] == "assistant" and msg.get("content"):
                return msg["content"]
        return "Task not completed"
