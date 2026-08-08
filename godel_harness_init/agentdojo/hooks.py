"""React hooks - Pre/post react hooks for agent customization."""
from typing import Any, Dict, List, Tuple


def hook_on_request(messages: List[Dict], tools: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Hook on request - Passthrough implementation.

    Args:
        messages: Message history
        tools: Tool list

    Returns:
        (messages, tools) unchanged
    """
    return messages, tools


def hook_on_response(response: Any) -> Any:
    """Hook on response - Passthrough implementation.

    Args:
        response: LLM response with response.choices[0].message.tool_calls

    Returns:
        response unchanged
    """
    return response


def hook_on_complete(response: Any, tool_calls_made: List[Dict], tool_results: List[str]) -> None:
    """Hook on complete - Called after tool execution.

    Args:
        response: LLM response object
        tool_calls_made: List of tool calls made, format: [{"id": str, "name": str, "args": dict}]
        tool_results: List of tool execution results
    """
    pass
