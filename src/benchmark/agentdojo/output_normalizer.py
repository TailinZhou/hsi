"""Output normalizer for AgentDojo evaluation.

Provides normalization functions and runtime monkey-patches to fix false-negative
evaluations caused by number comma-formatting in the upstream AgentDojo utility checks.

These patches are purely evaluation-layer fixes — they do not change agent behavior.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

_NUMBER_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3})")


def normalize_numbers_in_text(text: str) -> str:
    """Strip comma separators from numbers (e.g. '1,000' -> '1000').

    Non-numeric commas (e.g. in prose) are left untouched.
    """
    if "," not in text:
        return text
    return _NUMBER_COMMA_RE.sub("", text)


# ---------------------------------------------------------------------------
# Patch machinery
# ---------------------------------------------------------------------------

_originals: dict[tuple[Any, str], Any] = {}


def apply_patches() -> None:
    """Apply runtime patches to AgentDojo evaluation functions.

    Idempotent — calling multiple times has no additional effect.
    """
    if _originals:
        return

    try:
        _patch_get_text_content_as_str()
    except Exception as e:
        logger.warning("Failed to apply patch %s: %s", _patch_get_text_content_as_str.__name__, e)


def revert_patches() -> None:
    """Restore all original functions for clean teardown."""
    for (module, attr), original in _originals.items():
        try:
            setattr(module, attr, original)
        except Exception:
            pass
    _originals.clear()


def _save_original(module: Any, attr: str) -> None:
    """Record the original value before patching (no-op if already saved)."""
    key = (module, attr)
    if key not in _originals:
        _originals[key] = getattr(module, attr)


# ---------------------------------------------------------------------------
# Patch 1 — normalize numbers in model output
# ---------------------------------------------------------------------------

def _patch_get_text_content_as_str() -> None:
    import agentdojo.types as types_module

    original = types_module.get_text_content_as_str
    _save_original(types_module, "get_text_content_as_str")

    def patched(content_blocks):
        return normalize_numbers_in_text(original(content_blocks))

    types_module.get_text_content_as_str = patched

    # Also update the reference that task_suite may have imported directly.
    try:
        import agentdojo.task_suite.task_suite as ts_module
        if getattr(ts_module, "get_text_content_as_str", None) is original:
            _save_original(ts_module, "get_text_content_as_str")
            ts_module.get_text_content_as_str = patched
    except (ImportError, AttributeError):
        pass
