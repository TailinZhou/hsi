"""
Utility modules for React Loop Agent.

Contains:
- validator: AST-level code security validation
- message_utils: Message history management utilities
- tools: OpenAI tools construction utilities
"""

from .validator import (
    ValidationResult,
    CodeValidator,
)
from .message_utils import (
    extract_tool_calls_for_history,
)
from .tools import (
    build_openai_tools,
    scan_external_tools,
)
from .json_parser import (
    fix_and_parse_json,
    clean_json_values,
    parse_action_from_response,
)
from .harness_loader import (
    HarnessLoader,
    HarnessInfo,
)
from .log_format import (
    _C,
    _tool_color,
)

__all__ = [
    # Validator
    "ValidationResult",
    "CodeValidator",
    # Message utils
    "extract_tool_calls_for_history",
    # Tools
    "build_openai_tools",
    "scan_external_tools",
    # JSON Parser
    "fix_and_parse_json",
    "clean_json_values",
    "parse_action_from_response",
    # Harness Loader
    "HarnessLoader",
    "HarnessInfo",
    # Log Format
    "_C",
    "_tool_color",
]
