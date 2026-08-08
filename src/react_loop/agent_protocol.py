"""GodelHarness Protocol — the interface that harness functions expect from agent.

GodelAgent already satisfies this via duck typing (it has react(), call_llm(), get_tools()).
GodelAgentProxy explicitly implements it for use in Harbor subprocesses where
the real GodelAgent is not available.
"""

from typing import Any, Dict, List, Optional, Protocol


class GodelHarness(Protocol):
    """The agent interface expected by harness functions.

    Any object with these three methods can be passed to using_harness().
    """

    def react(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        **kwargs: Any,
    ) -> Any:
        """Multi-turn LLM call with tool execution.

        Returns (response, tool_calls_made, tool_results) tuple.
        """
        ...

    def call_llm(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        **kwargs: Any,
    ) -> Any:
        """Single LLM call — returns an OpenAI-compatible response object."""
        ...

    def get_tools(
        self,
        scope: str = "harness",
        **kwargs: Any,
    ) -> List[Dict]:
        """Return tool schemas for the given scope."""
        ...
