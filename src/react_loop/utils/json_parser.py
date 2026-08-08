"""
JSON parsing utilities for handling LLM-generated JSON with common issues.

Handles common JSON problems from LLM outputs:
- Single quotes instead of double quotes
- Extra quotes in values
- Trailing commas
- Comments
- Triple quote escape issues
- Backslash escape issues
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Callable, List, Optional

# Lazy imports to avoid circular dependency
# from ..state import AgentAction, ActionType


# Code fields that should not have quotes cleaned (contain actual code)
CODE_FIELDS = {'new_code', 'code', 'source_code', 'content', 'script'}


def fix_and_parse_json(
    raw_args: str,
    function_name: str = "unknown",
    logging: Optional[Callable[[str], None]] = None
) -> dict:
    """
    Try to fix and parse potentially problematic JSON strings.

    Handles common issues:
    - Single quotes instead of double quotes
    - Extra quotes (like "modify")
    - Trailing commas
    - Comments
    - Triple quote escape issues
    - Backslash escape issues

    Args:
        raw_args: Raw JSON string to parse
        function_name: Function name for debug logging
        logging: Optional logging function

    Returns:
        Parsed dict, or empty dict if all parsing attempts fail
    """
    def log(msg: str) -> None:
        if logging:
            logging(msg)

    fixed_args = raw_args.strip()

    # Strategy 1: Remove comments and parse directly
    try:
        cleaned = re.sub(r'//.*$', '', fixed_args, flags=re.MULTILINE)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Replace single quotes with double quotes (simple case)
    try:
        cleaned = fixed_args.replace("'", '"')
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Remove trailing commas
    try:
        cleaned = re.sub(r',\s*([}\]])', r'\1', fixed_args)
        cleaned = cleaned.replace("'", '"')
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Fix triple quote escape issues
    try:
        cleaned = fix_triple_quotes_in_json(fixed_args)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 5: Fix backslash escape issues
    try:
        cleaned = fix_backslash_escapes(fixed_args)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 6: Try json5 parser (optional dependency)
    try:
        import json5
        return json5.loads(fixed_args)
    except ImportError:
        pass
    except Exception:
        pass

    # Strategy 7: Use ast.literal_eval as last resort
    try:
        import ast
        return ast.literal_eval(raw_args)
    except Exception:
        pass

    # Save failed JSON for debugging
    save_failed_json_debug(raw_args, function_name, logging)
    log("Failed to parse JSON after all attempts, using empty dict")
    return {}


def fix_triple_quotes_in_json(json_str: str) -> str:
    """
    Fix triple quote escape issues in JSON code fields.

    Problem: LLM generates code with triple quotes, but JSON escaping is incomplete
    Example: {"new_code": "def foo():\n    return '''bar'''"}
    """
    code_fields = ['new_code', 'code', 'source_code', 'content']

    def fix_field_value(match):
        field_name = match.group(1)
        value = match.group(2)

        # Check for improperly escaped triple quotes
        if '"""' in value or "'''" in value:
            # Escape triple quotes for JSON
            value = value.replace('\\', '\\\\')  # Escape backslashes first
            value = value.replace('"""', '\\"\\"\\"')  # Escape triple double quotes
            value = value.replace("'''", "\\'\\'\\'")  # Escape triple single quotes
            # Handle newlines
            value = value.replace('\n', '\\n')
            value = value.replace('\r', '\\r')
            value = value.replace('\t', '\\t')

        return f'"{field_name}": "{value}"'

    # Try to match and fix code fields
    result = json_str
    for field in code_fields:
        # Match "field_name": "value" pattern (simplified, non-nested)
        pattern = rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"\s*([,}}])'
        result = re.sub(pattern, lambda m: f'"{field}": "{m.group(1)}"{m.group(2)}', result)

    return result


def fix_backslash_escapes(json_str: str) -> str:
    """
    Fix backslash escape issues in JSON.

    Common issues:
    - Single backslash needs to become double backslash
    - But already correctly escaped ones should not be re-escaped
    """
    def fix_string_content(match):
        content = match.group(1)
        # Check for unescaped backslashes
        # Exclude already escaped cases: \\, \", \n, \r, \t, \uXXXX
        if re.search(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', content):
            # Escape single backslashes
            content = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', content)
        return f'"{content}"'

    # Match JSON string values
    result = re.sub(r'"((?:[^"\\]|\\.)*)"', fix_string_content, json_str)
    return result


def save_failed_json_debug(
    raw_args: str,
    function_name: str,
    logging: Optional[Callable[[str], None]] = None
) -> None:
    """
    Save failed JSON parsing data for debugging.

    Files are saved in .evolution_debug/ directory.
    """
    def log(msg: str) -> None:
        if logging:
            logging(msg)

    debug_dir = os.path.join(os.getcwd(), ".evolution_debug")
    os.makedirs(debug_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"failed_json_{function_name}_{timestamp}.txt"
    filepath = os.path.join(debug_dir, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Failed JSON parsing debug info\n")
            f.write(f"# Function: {function_name}\n")
            f.write(f"# Time: {datetime.now().isoformat()}\n")
            f.write(f"# Length: {len(raw_args)} chars\n")
            f.write(f"\n{'='*60}\n")
            f.write(raw_args)
            f.write(f"\n{'='*60}\n")

        log(f"Saved failed JSON to {filepath}")
    except Exception as e:
        log(f"Failed to save debug file: {e}")


def clean_json_values(
    obj: Any,
    skip_fields: set = None,
    current_key: str = None
) -> Any:
    """
    Clean common issues in JSON values:
    - Extra quotes at end of string values (e.g., modify" -> modify)
    - Extra quotes at start of string values (e.g., "modify -> modify)
    - Entire value wrapped in quotes (e.g., "modify" -> modify)
    - Leading/trailing whitespace

    Args:
        obj: Object to clean
        skip_fields: Set of field names to skip cleaning (default: CODE_FIELDS)
        current_key: Current key name (for recursion)

    Returns:
        Cleaned object
    """
    if skip_fields is None:
        skip_fields = CODE_FIELDS

    if isinstance(obj, dict):
        return {k: clean_json_values(v, skip_fields, k) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_values(item, skip_fields, current_key) for item in obj]
    elif isinstance(obj, str):
        # If it's a code field, only do basic cleaning, don't modify quotes
        if current_key and current_key in skip_fields:
            # Only strip whitespace, preserve all internal characters
            return obj.strip() if obj != obj.strip() else obj

        # Normal cleaning logic for non-code fields
        # Clean string values
        cleaned = obj.strip()

        # Handle trailing extra quotes
        if cleaned.endswith('"') or cleaned.endswith("'"):
            # Check if entire value is wrapped in quotes
            if len(cleaned) >= 2:
                if (cleaned.startswith('"') and cleaned.endswith('"')) or \
                   (cleaned.startswith("'") and cleaned.endswith("'")):
                    inner = cleaned[1:-1]
                    # Only remove outer quotes if inner has no quotes
                    if '"' not in inner and "'" not in inner:
                        cleaned = inner
                    else:
                        # Only trailing quote, remove it
                        cleaned = cleaned[:-1]

        # Clean any remaining quotes
        cleaned = cleaned.strip()
        if cleaned.endswith('"') or cleaned.endswith("'"):
            # If still has trailing quote, remove it
            quote_char = cleaned[-1]
            if not cleaned.startswith(quote_char):
                cleaned = cleaned[:-1]

        return cleaned
    else:
        return obj


def parse_action_from_response(
    response,
    external_tools: List[dict],
    logging: Optional[Callable[[str], None]] = None
):
    """
    Parse LLM response into an AgentAction.

    Args:
        response: LLM response object with tool_calls
        external_tools: List of external tool definitions
        logging: Optional logging function

    Returns:
        AgentAction object, or None if no tool call
    """
    # Lazy import to avoid circular dependency
    from ..state import AgentAction, ActionType

    def log(msg: str) -> None:
        if logging:
            logging(msg)

    message = response.choices[0].message

    # Check for tool call
    _tool_calls = getattr(message, 'tool_calls', None)
    if _tool_calls:
        tool_call = _tool_calls[0]
        function_name = tool_call.function.name

        # Record raw response (for debugging)
        raw_arguments = tool_call.function.arguments
        log(f"[DEBUG] Tool call: {function_name}")
        log(f"[DEBUG] Raw arguments length: {len(raw_arguments)} chars")
        if len(raw_arguments) < 500:
            log(f"[DEBUG] Raw arguments: {raw_arguments}")
        else:
            log(f"[DEBUG] Raw arguments (first 200 chars): {raw_arguments[:200]}...")

        # Safely parse JSON parameters
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as e:
            log(f"JSON parse error for tool '{function_name}': {e}")
            log(f"Raw arguments: {raw_arguments}")
            # Try to fix common JSON issues (GLM model uses single quotes, extra quotes, etc.)
            raw_args = raw_arguments.strip()
            if not raw_args or raw_args == "{}":
                arguments = {}
            else:
                # Pass function_name for debugging
                arguments = fix_and_parse_json(raw_args, function_name, logging=logging)

        # Clean extra quotes and whitespace in parameter values
        if isinstance(arguments, dict):
            arguments = clean_json_values(arguments)

        # Map function name to ActionType
        action_map = {
            # Built-in core
            "bash": ActionType.BASH,
            "read_history_self": ActionType.READ_HISTORY_SELF,
            # File operation tools
            "read_file": ActionType.READ_FILE,
            "edit_file": ActionType.EDIT_FILE,
            "write_file": ActionType.WRITE_FILE,
            # Required external
            "evaluate": ActionType.EVALUATE,
            # Historical version
            "get_historic_version": ActionType.GET_HISTORIC_VERSION,
        }

        # Dynamically add external tool mappings
        for tool in external_tools:
            tool_name = tool["info"]["name"]
            if tool_name not in action_map:
                action_map[tool_name] = ActionType.EXTERNAL_TOOL

        action_type = action_map.get(function_name, ActionType.EXTERNAL_TOOL)

        # Built-in tools use arguments directly as params
        # External tools use the {tool_name, arguments} structure
        if action_type == ActionType.EXTERNAL_TOOL:
            params = {
                "tool_name": function_name,
                "arguments": arguments,
            }
        else:
            params = arguments

        return AgentAction(
            action_type=action_type,
            params=params,
        )

    # No tool call - the LLM considers the task done
    # Return None; the main loop decides how to handle it
    return None
