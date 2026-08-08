"""
Message management helpers for maintaining the LLM conversation context.
"""
from typing import Dict, List, Any, Optional
from ..state import MessageHistory


def extract_tool_calls_for_history(tool_calls) -> Optional[List[Dict[str, Any]]]:
    """Extract tool call info for message history."""
    if not tool_calls:
        return None

    extracted = []
    for tc in tool_calls:
        extracted.append({
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
        })
    return extracted


def append_response_to_messages(messages, response, tool_calls_made, tool_results):
    """Append an assistant message + tool results from one react step to a plain list.

    Used by evolvable react loops (ensemble) that drive their own message
    list. Mirrors MessageHistory.add_assistant/add_tool but for
    a raw list, so the three templates share one implementation instead of
    copy-pasting it.
    """
    msg = response.choices[0].message
    assistant_msg = {"role": "assistant"}
    if msg.content:
        assistant_msg["content"] = msg.content
    tc_list = extract_tool_calls_for_history(msg.tool_calls)
    if tc_list:
        assistant_msg["tool_calls"] = tc_list
    messages.append(assistant_msg)

    for tc_info, result in zip(tool_calls_made, tool_results):
        messages.append({
            "role": "tool",
            "tool_call_id": tc_info["id"],
            "content": str(result),
        })


def update_history_from_response(
    response,
    message_history: MessageHistory,
    reasoning_content: Optional[str] = None,
) -> bool:
    """Update message history from a react response (add the assistant message).

    Returns:
        True if there were tool_calls, False otherwise.
    """
    message = response.choices[0].message
    if message.tool_calls:
        history_tool_calls = extract_tool_calls_for_history(message.tool_calls)
        message_history.add_assistant(
            content=message.content,
            tool_calls=history_tool_calls,
            reasoning_content=reasoning_content,
        )
        return True
    else:
        message_history.add_assistant(
            content=message.content or "",
            reasoning_content=reasoning_content,
        )
        return False
