"""
Terminal log formatting utilities.

Provides ANSI color constants and tool-specific color mapping
for structured, hierarchical log output.
"""


class _C:
    """ANSI color codes for structured terminal output."""
    RST = "\033[0m"
    B   = "\033[1m"     # Bold
    D   = "\033[2m"     # Dim
    CY  = "\033[36m"    # Cyan
    GR  = "\033[32m"    # Green
    YE  = "\033[33m"    # Yellow
    BL  = "\033[34m"    # Blue
    MA  = "\033[35m"    # Magenta
    RD  = "\033[31m"    # Red
    BCY = "\033[96m"    # Bright Cyan
    BGR = "\033[92m"    # Bright Green
    BYE = "\033[93m"    # Bright Yellow
    BBL = "\033[94m"    # Bright Blue
    BMA = "\033[95m"    # Bright Magenta
    BWH = "\033[97m"    # Bright White


def _tool_color(name: str) -> str:
    """Get color for a tool name based on its category."""
    _MAP = {
        "read_file": _C.BBL, "read_history_self": _C.BBL,
        "get_historic_version": _C.BBL,
        "edit_file": _C.BYE, "write_file": _C.BYE,
        "bash": _C.BGR,
        "evaluate": _C.BMA,
        "compact_context": _C.BCY, "end_evolution": _C.BCY,
    }
    return _MAP.get(name, _C.BWH)


# ─── Unified presentation layer ─────────────────────────────────────
# These front-of-stage printers live in the framework (NOT under evolution/),
# so meta-evolve cannot touch them and the seed/commit/best modules stay pure
# strategy code. Each select_*.py delegates its display to these instead of
# carrying its own log-formatting helpers.


def log_phase_banner(agent, title, info="", color=_C.BCY):
    """Print a uniform phase banner (╔═╗ centered title + optional info line).

    Used by meta-evolve and any future seed/commit react phase entry. Replaces
    the duplicated banner code that used to live in ``meta_evolve.run()``.

    Args:
        agent: anything with an ``_log(str)`` method.
        title: banner title text (auto-centered, width auto-fits).
        info: optional extra line printed under the box (e.g. candidate count).
        color: border color (default bright cyan; meta-evolve passes magenta).
    """
    W = max(60, len(title))
    pad = max(0, W - len(title))
    left = pad // 2
    right = pad - left
    agent._log(f"\n{_C.B}{color}{'╔' + '═' * W + '╗'}{_C.RST}")
    agent._log(f"{color}║{' ' * left}{_C.RST}{_C.B}{_C.BWH}{title}{_C.RST}{color}{' ' * right}║{_C.RST}")
    agent._log(f"{_C.B}{color}{'╚' + '═' * W + '╝'}{_C.RST}")
    if info:
        agent._log(info)


def log_react_step(agent, step, max_steps, tool_calls_made, tool_results,
                   prefix="", prefix_color=_C.BYE):
    """Print one react step's tool-call display box (uniform across react loops).

    Used by any react loop (ensemble strategy, seed/commit react). Keeps the
    special case that ``evaluate`` results are never truncated (they carry the
    reward the decision hinges on).

    Args:
        agent: anything with an ``_log(str)`` method.
        step: 1-based current step index.
        max_steps: step budget for the loop. None → no denominator (the loop
            has no hard step cap; e.g. the probe sub-agent past its soft budget).
        tool_calls_made: list of {"name", "args", ...} dicts from the response.
        tool_results: list of tool result objects, parallel to tool_calls_made.
        prefix: optional label printed before "Step" (e.g. "probe") so a
            sub-agent's trajectory is visually distinguishable from the outer
            loop's own steps when they interleave in the log. Empty → no prefix.
        prefix_color: color for the prefix label (default bright yellow, chosen
            to stand out against the dim "Step" text and meta-evolve's magenta).
    """
    import json

    prefix_part = f"{prefix_color}{prefix}{_C.RST} " if prefix else ""
    step_label = f"Step {step}/{max_steps}" if max_steps is not None else f"Step {step}"
    agent._log(f"  {prefix_part}{_C.D}{step_label}{_C.RST}")
    if not tool_calls_made:
        agent._log(f"  {_C.D}[LLM Response] No tool calls{_C.RST}")
        return
    names = [tc['name'] for tc in tool_calls_made]
    agent._log(f"  {_C.D}[LLM Response] Tool calls:{_C.RST} {names}")
    for idx, (tc, result) in enumerate(zip(tool_calls_made, tool_results), start=1):
        tool_name = tc['name']
        c = _tool_color(tool_name)
        agent._log(f"  {_C.D}┌─{_C.RST}{_C.B}{c} [{idx}/{len(tool_calls_made)}] {tool_name}{_C.RST}")
        if tc.get('args'):
            try:
                params_str = json.dumps(tc['args'], ensure_ascii=False, indent=4)
                if len(params_str) > 500:
                    params_str = params_str[:1000] + '...'
                agent._log(f"  {_C.D}│{_C.RST} {_C.D}Params:{_C.RST}")
                for line in params_str.split('\n'):
                    agent._log(f"  {_C.D}│{_C.RST}   {line}")
            except Exception:
                agent._log(f"  {_C.D}│{_C.RST} {_C.D}Params:{_C.RST} {tc.get('args')}")
        agent._log(f"  {_C.D}│{_C.RST} {_C.D}Result:{_C.RST}")
        result_str = str(result)
        if tool_name != 'evaluate' and len(result_str) > 500:
            result_str = result_str[:1000] + '...'
        for line in result_str.split('\n'):
            agent._log(f"  {_C.D}│{_C.RST}   {line}")
        agent._log(f"  {_C.D}└──{_C.RST}")


def log_selection_result(agent, dimension, value, hint=""):
    """Print one uniform selection-result line for a dimension.

    Args:
        agent: anything with an ``_log(str)`` method.
        dimension: one of "seed", "commit", "best".
        value: the chosen hash (git_hash / code_hash / commit_hash); truncated to 7.
        hint: optional label (strategy_hint / commit_hint / submit_hint).
    """
    short = (value or "")[:7]
    hint_part = f" (hint: {hint})" if hint else ""
    agent._log(f"  Archive select_{dimension}: {short}{hint_part}")
