"""Custom LLM-callable tools for the task.

Tools defined here are passed to agent.react() alongside framework-injected tools.
The LLM decides when to call them based on their descriptions.

## Tool Format (Harness Format)

Each tool is a dict with:
- info: Dict containing metadata:
    - name: Tool name (string)
    - description: What the tool does (string)
    - input_schema: JSON schema for parameters (dict)
- function: Python callable that implements the tool

NOTE: agent.get_tools() converts this harness format to OpenAI function calling format
automatically: {"info": {...}, "function": ...} -> {"type": "function", "function": {"name": ...}}

Example:

    def my_tool(query: str) -> str:
        '''Does something useful'''
        return f"Result for: {query}"

    TOOLS = [
        {
            "info": {
                "name": "my_tool",
                "description": "Does something useful",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The query to process"}
                    },
                    "required": ["query"]
                },
            },
            "function": my_tool,
        }
    ]
"""

from typing import List, Dict, Any

TOOLS: List[Dict[str, Any]] = []
