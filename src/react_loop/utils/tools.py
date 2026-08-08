"""
OpenAI tool definitions and external tool management.
"""
import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Any, Optional, Tuple

from ..state import ActionType


# Scopes that pick / fuse versions (NOT the solving evolve scope). These may NOT
# read benchmark source/answers — that would let the agent overfit or cheat. The
# evolve scope is excluded because it legitimately reads task files while solving.
# The guard itself lives once in GodelAgent._execute_tool_impl; add a scope here
# to shield it.
_BENCHMARK_SHIELDED_SCOPES = frozenset({"meta_evolve", "atom", "pick_seed", "submit_best", "probe"})


def benchmark_access_blocked(path_or_command: str) -> bool:
    """True if a read_file path or bash command touches benchmark source/answers
    (``src/benchmark/`` — task source + ground-truth answers)."""
    return "src/benchmark/" in (path_or_command or "").replace("\\", "/")


def benchmark_block_message(tool_name: str, args: Dict) -> Optional[str]:
    """Return a BLOCKED message if this tool call would touch ``src/benchmark/``,
    else None. Covers read_file (path arg) and bash/powershell (command arg).
    Called from ``_execute_tool_impl`` for every scope in ``_BENCHMARK_SHIELDED_SCOPES``.
    """
    if tool_name == ActionType.READ_FILE.value and benchmark_access_blocked(args.get("path", "")):
        return "BLOCKED: reading benchmark source/answers (src/benchmark/) is not allowed in this scope."
    if tool_name in (ActionType.BASH.value, ActionType.POWERSHELL.value) and benchmark_access_blocked(args.get("command", "")):
        return "BLOCKED: accessing benchmark paths (src/benchmark/) is not allowed in this scope (cat/git show/etc on benchmark source/answers)."
    return None


def scan_external_tools(
    agent_code_dir: str,
    tool_scope: str,
    log_func: Callable = None
) -> List[Dict]:
    """
    Scan the agent code directory for external tools (single-file mode).

    External tool placement convention:
    - tools_{scope}.py single file: defines a TOOLS list and a same-named Python function inside the file.

    Args:
        agent_code_dir: Path to the agent code directory.
        tool_scope: Tool scope, "evolve" or "harness".
        log_func: Optional logging function.

    Returns:
        List of external tools.
    """
    external_tools = []

    if not agent_code_dir:
        return external_tools

    if tool_scope not in ("evolve", "harness"):
        if log_func:
            log_func(f"Invalid tool_scope: {tool_scope}, must be 'evolve' or 'harness'")
        return external_tools

    single_file = os.path.join(agent_code_dir, f"tools_{tool_scope}.py")

    if os.path.isfile(single_file):
        if log_func:
            log_func(f"Loading {tool_scope} tools from {single_file}")
        try:
            spec = importlib.util.spec_from_file_location("tools_module", single_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if log_func:
                log_func(f"Module loaded, checking for TOOLS attribute...")
            if hasattr(module, "TOOLS"):
                tools_list = module.TOOLS
                if log_func:
                    log_func(f"TOOLS list found with {len(tools_list)} items")
                for item in tools_list:
                    info = item.get("info")
                    func = item.get("function")

                    # Support function as a callable or a string (need to fetch from module)
                    if isinstance(func, str):
                        func = getattr(module, func, None)

                    if info:
                        # harness format: info + function
                        name = info.get("name", "")
                        if name and func and callable(func):
                            external_tools.append({
                                "info": info,
                                "function": func,
                                "name": name,
                                "scope": tool_scope,
                            })
                            if log_func:
                                log_func(f"Loaded {tool_scope} tool (single file): {name}")
                        else:
                            if log_func:
                                log_func(f"Invalid tool entry in {single_file}: name={name}, func={func}")
                    else:
                        # Backward-compatible old flat format: name, description, parameters, function
                        name = item.get("name")
                        if name and func and callable(func):
                            external_tools.append({
                                "info": {
                                    "name": name,
                                    "description": item.get("description", ""),
                                    "input_schema": item.get("parameters", {"type": "object", "properties": {}}),
                                },
                                "function": func,
                                "name": name,
                                "scope": tool_scope,
                            })
                            if log_func:
                                log_func(f"Loaded {tool_scope} tool (single file, flat format): {name}")
                        else:
                            if log_func:
                                log_func(f"Invalid tool entry in {single_file}: name={name}, func={func}")
            else:
                if log_func:
                    log_func(f"No TOOLS list found in {single_file}")
        except Exception as e:
            import traceback
            if log_func:
                log_func(f"Failed to load single-file tools from {single_file}: {e}")
                log_func(f"Traceback: {traceback.format_exc()}")

    # Also scan evolution/evolution_tools.py when scope is "evolve"
    if tool_scope == "evolve":
        evo_tools_file = os.path.join(agent_code_dir, "evolution", "evolution_tools.py")
        if os.path.isfile(evo_tools_file):
            if log_func:
                log_func(f"Scanning evolution tools: {evo_tools_file}")
            try:
                spec = importlib.util.spec_from_file_location("evolution_tools", evo_tools_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "TOOLS") and module.TOOLS:
                    for item in module.TOOLS:
                        info = item.get("info")
                        func = item.get("function")
                        if isinstance(func, str):
                            func = getattr(module, func, None)
                        if info and func and callable(func):
                            name = info.get("name", "")
                            if name:
                                external_tools.append({
                                    "info": info,
                                    "function": func,
                                    "name": name,
                                    "scope": "evolve",
                                })
                                if log_func:
                                    log_func(f"Loaded evolution tool: {name}")
            except Exception as e:
                if log_func:
                    log_func(f"Warning: Failed to load evolution tools: {e}")

    return external_tools


def get_shell_tool_schema(is_windows: bool = False) -> Dict[str, Any]:
    """Get shell tool schema for bash or PowerShell."""
    name = "powershell" if is_windows else "bash"
    shell_name = "PowerShell" if is_windows else "bash"
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Execute {shell_name} commands. Git version control is available: "
                           "save checkpoints with 'git add -A && git commit -m checkpoint', "
                           "revert files with 'git checkout <commit> -- .', "
                           "view history with 'git log', compare with 'git diff'. "
                           "Commit frequently with meaningful messages describing key changes. "
                           "For reading/writing code files, prefer read_file/edit_file/write_file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": f"The {shell_name} command to run."
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 120)"
                    }
                },
                "required": ["command"]
            }
        }
    }


def build_external_tool_schema(tool: Dict) -> Dict[str, Any]:
    """Build external tool schema."""
    info = tool.get("info", {})
    return {
        "type": "function",
        "function": {
            "name": info.get("name", tool.get("name", "unknown")),
            "description": info.get("description", ""),
            "parameters": info.get("input_schema", {"type": "object", "properties": {}})
        }
    }


# Filter-style scope -> whitelist of built-in tool names visible in that scope. Add a scope = add a row.
# evolve / harness / all are not in this table (they are construct-style, not filter-style, and go through their own branches).
_SCOPE_TOOL_WHITELIST: Dict[str, set] = {
    "meta_evolve": {"read_file", "edit_file", "write_file", "bash", "powershell",
                    "end_meta_evolution", "validate_archive", "meta_bootstrap", "probe",
                    "lesson"},
    "atom": {"read_file", "edit_file", "write_file", "bash", "powershell"},
    # seed tool surface: read-only + bash + evaluate, no direct editing.
    # Merge/fusion is done by a strategy tool (ensemble), not by the agent manually editing via edit/write.
    "pick_seed": {"read_file", "bash", "powershell", "evaluate"},
    # submit_best tool surface: mirrors pick_seed — read-only + bash + evaluate.
    # final best-version selection differs from the fusion sub-agent (atom scope): the submit_best agent
    # only inspects candidates and live-tests fusion; it does not hand-edit the harness (fusion is done inside the ensemble strategy tool).
    "submit_best": {"read_file", "bash", "powershell", "evaluate"},
    # probe: read-only investigation sub-agent — read_file + bash + end_probe only
    "probe": {"read_file", "bash", "powershell", "end_probe"},
}


def build_openai_tools(
    external_tools: List[Dict],
    enable_bash: bool = True,
    scope: str = "evolve",
    description_overrides: Dict[str, str] = None,
    extra_tools: List[Dict] = None,
    exclude_tools: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build OpenAI function calling tools.

    Args:
        external_tools: List of external tools.
        enable_bash: Whether to enable the bash tool.
        scope: "evolve" (all tools) | "meta_evolve" (read_file, edit_file, write_file, bash, end_evolution)
        description_overrides: {tool_name: override_desc} — only replaces the description field.
        extra_tools: List of additional tool schemas to append (e.g. compact_meta).

    Returns:
        List of OpenAI tool definitions.
    """
    description_overrides = description_overrides or {}
    tools = []

    # ========== Built-in core tools ==========
    # read_history_self
    tools.append({
        "type": "function",
        "function": {
            "name": "read_history_self",
            "description": "Search your conversation history from COMPLETED iterations by keyword. "
                           "IMPORTANT: Always prefer keyword search over loading full iteration history "
                           "to avoid context overflow. Full iteration loading (@history:N) should only "
                           "be used as a last resort when you cannot narrow down with keywords. "
                           "For reading code files, use read_file instead. "
                           "NOTE: You can only read COMPLETED iterations, not the current one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "History targets using @history syntax (1-based iteration numbers). "
                                       "PREFER keyword search (lightweight, recommended): "
                                       "[\"@history:hook design\"] - find iterations about 'hook design', "
                                       "[\"@history:timeout\"] - find iterations about 'timeout'. "
                                       "Scoped search (when you know the iteration): "
                                       "[\"@history:2:caching\"] - search 'caching' only in iteration 2. "
                                       "Full load (AVOID unless necessary - very large output): "
                                       "[\"@history:1\"] - load ALL of iteration 1, use sparingly."
                    }
                },
                "required": ["targets"]
            }
        }
    })

    # bash / powershell (built-in core, can be disabled via config) - command execution
    if enable_bash:
        tools.append(get_shell_tool_schema(is_windows=(sys.platform == 'win32')))

    # ========== File operation tools (split out from the original editor) ==========
    # read_file - read a file
    tools.append({
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file with line numbers, or list directory contents (2 levels). "
                           "Use this instead of bash 'cat' for viewing code. "
                           "Long outputs are truncated with `<response clipped>` marker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file or directory. Absolute paths (e.g. `/repo/file.py`) are always accepted. Relative paths (e.g. `harness.py`, `BOOTSTRAP.md`) are resolved against the repo root directory."
                    }
                },
                "required": ["path"]
            }
        }
    })

    # edit_file - precise replacement
    tools.append({
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit an existing file by replacing specific text. "
                           "Preferred for small, targeted changes. "
                           "AST validation is performed for .py files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file. Absolute paths (e.g. `/repo/file.py`) are always accepted. Relative paths (e.g. `harness.py`) are resolved against the repo root directory."
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find and replace. "
                                        "Must match exactly including whitespace/indentation. "
                                        "Must be unique in the file (or use `replace_all=true`)."
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The text to replace `old_string` with. "
                                        "Can be empty string to delete."
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "If true, replace all occurrences of `old_string`. Default: false."
                    }
                },
                "required": ["path", "old_string"]
            }
        }
    })

    # write_file - create/overwrite file
    tools.append({
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new file or completely overwrite an existing file. "
                           "Use for creating new files or major rewrites. "
                           "For small edits, prefer `edit_file` instead. "
                           "AST validation is performed for .py files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file. Absolute paths (e.g. `/repo/file.py`) are always accepted. Relative paths (e.g. `harness.py`) are resolved against the repo root directory."
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete content to write to the file."
                    },
                    "create_only": {
                        "type": "boolean",
                        "description": "If true, only create new file (error if exists). Default: false (allow overwrite)."
                    }
                },
                "required": ["path", "content"]
            }
        }
    })

    # ========== Required external tools ==========
    # evaluate
    tools.append({
        "type": "function",
        "function": {
            "name": "evaluate",
            "description": "Request evaluation to get your current reward (collect r_t). "
                           "Uses external evaluator if provided, otherwise internal evaluator. "
                           "If you renamed your entry function to something else, "
                           "specify the new name in func_names. "
                           "Before calling evaluate, review your changes — read the "
                           "edited files to verify they match your hypothesis. Evaluate "
                           "measures; it does not review or debug.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_cases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional test cases for evaluation"
                    },
                    "func_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of entry function names to search for (in order). "
                                       "Default: ['using_harness', 'harness']. "
                                       "Use this if you renamed your entry function."
                    },
                    "eval_mode": {
                        "type": "string",
                        "enum": ["dev", "val"],
                        "description": "Evaluation mode. 'dev' (default): returns detailed feedback — "
                                       "failed tasks, logs, suggestions. "
                                       "'val': returns ONLY reward number — black box for honest "
                                       "self-assessment, detects overfitting to dev tasks. "
                                       "Recommendation: use 'dev' for learning, 'val' periodically "
                                       "for generalization check."
                    },
                    "num_tasks": {
                        "type": "integer",
                        "description": "Fallback for a quick RANDOM sanity check when you have NO specific task in mind "
                                       "and just want fast feedback (e.g. task set is large, typing names impractical). "
                                       "Reward is NOT tracked for evolution (does NOT affect best-version selection). "
                                       "If num_tasks >= total available tasks, it auto-upgrades to a full tracked evaluation. "
                                       "Prefer task_ids when you know which task you care about. "
                                       "Mutually exclusive with task_ids (task_ids wins if both given)."
                    },
                    "task_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "PREFERRED way to verify a targeted fix. Pins the EXACT tasks to re-run by ID — "
                                       "deterministic, always re-runs what you named (dev re-samples randomly each call, "
                                       "so a plain dev eval may not even hit the task you just fixed). "
                                       "Reward is NOT tracked (hand-picked set is not representative). "
                                       "IDs appear in the per-task breakdown of any dev evaluate result — copy them verbatim. "
                                       "Mutually exclusive with num_tasks. Example: ['textworld/coin_collector']"
                    }
                }
            }
        }
    })

    # ========== Evolution control tools ==========
    # compact_context
    tools.append({
        "type": "function",
        "function": {
            "name": "compact_context",
            "description": "End the current iteration and compact context. "
                           "Call lesson(lesson=..., confidence=...) BEFORE this to record this iteration's cross-iteration verdict. "
                           "Call this when you have finished modifying code and evaluating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Concise summary of what was accomplished."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for ending this iteration."
                    }
                },
                "required": ["summary", "reason"]
            }
        }
    })

    # end_evolution
    tools.append({
        "type": "function",
        "function": {
            "name": "end_evolution",
            "description": "End the entire evolution process. "
                           "Use ONLY when you are confident the strategy is near-perfect and further iterations "
                           "are unlikely to improve performance. This is a FINAL action - the evolution will stop "
                           "after the current iteration is committed. "
                           "Call lesson(lesson=..., confidence=...) BEFORE this to record this iteration's cross-iteration verdict. "
                           "WARNING: Only use this when you have strong evidence (e.g., consistently high reward) "
                           "that the strategy cannot be further improved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A final summary of the entire evolution outcome."
                    },
                    "reason": {
                        "type": "string",
                        "description": "The reason for ending evolution (e.g., 'reward plateaued at 0.95', 'strategy is optimal')."
                    }
                },
                "required": ["summary", "reason"]
            }
        }
    })

    # end_meta_evolution
    tools.append({
        "type": "function",
        "function": {
            "name": "end_meta_evolution",
            "description": "End the current meta-evolution phase. "
                           "Call this when you have finished modifying the archive strategy and want to "
                           "signal completion of this meta-evolve phase. "
                           "The main evolution loop will continue to the next iteration after this.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A summary of what was changed in this meta-evolve phase."
                    },
                    "reason": {
                        "type": "string",
                        "description": "The reason for ending this meta-evolve phase."
                    }
                },
                "required": ["summary", "reason"]
            }
        }
    })

    # ========== Historical version tools ==========
    # get_historic_version
    tools.append({
        "type": "function",
        "function": {
            "name": "get_historic_version",
            "description": "Get code from a historical version. Use this to compare with previous versions or learn from past iterations. IMPORTANT: Always specify 'file' when you know which file you need — returning all files wastes context and may exceed token limits. NOTE: Only COMPLETED iterations are available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "iteration": {
                        "type": "string",
                        "description": "Iteration number (1-based, e.g., '1', '2', '3') or 'best' for the best version"
                    },
                    "file": {
                        "type": "string",
                        "description": "Specific file name to retrieve (e.g., 'harness.py', 'tools.py'). STRONGLY RECOMMENDED — omitting this returns all files which is expensive and often unnecessary."
                    }
                },
                "required": ["iteration"]
            }
        }
    })

    # get_historic_eval_code
    tools.append({
        "type": "function",
        "function": {
            "name": "get_historic_eval_code",
            "description": "View code snapshot from a specific evaluation in the current iteration. "
                           "Use the eval number from Evaluate History to review code at that point. "
                           "Helps you compare different versions tried within this iteration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "eval_index": {
                        "type": "integer",
                        "description": "Eval number from Evaluate History (1-based, e.g., 1, 2, 3)"
                    },
                    "file": {
                        "type": "string",
                        "description": "Specific file to retrieve (e.g., 'harness.py'). Omit to see all files."
                    }
                },
                "required": ["eval_index"]
            }
        }
    })

    # ========== Evolution notebook tools (split: plan = this round's working memory, lesson = cross-round accumulation) ==========
    # plan — writes plan.md (iteration-scoped, ephemeral, gitignored)
    tools.append({
        "type": "function",
        "function": {
            "name": "plan",
            "description": "Update your iteration-scoped working notebook (plan.md). "
                           "plan.md is EPHEMERAL — it is cleared at the start of each iteration and rolls back with the code. "
                           "The framework writes the seed-selection Hypothesis into it at iteration start; you own the Plan and Progress sections. "
                           "Use this to think through your approach (Plan) before coding and to track what you tried (Progress) as the iteration unfolds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "string",
                        "description": "Your working plan for THIS iteration: the failure mode you are targeting, the change you intend, and the mechanism by which it should help (so it can be falsified). Overwrites the prior Plan section if non-empty."
                    },
                    "progress": {
                        "type": "string",
                        "description": "What you changed and the reward before → after, plus what the trace confirmed or refuted. Overwrites the prior Progress section if non-empty."
                    }
                },
                "required": ["plan"]
            }
        }
    })

    # lesson — writes BOOTSTRAP.md ## Lesson (cross-iteration, never rolls back)
    tools.append({
        "type": "function",
        "function": {
            "name": "lesson",
            "description": "Record ONE cross-iteration lesson into BOOTSTRAP.md (which never rolls back with git). "
                           "BOOTSTRAP.md is the ONLY memory the next iteration and seed selection inherit — future iterations start with no conversation history, so write as if explaining to yourself with no other context. "
                           "The framework records exactly one `[Iter N|conf=X.XX]` line per iteration (later calls for the same N revise that line). "
                           "By default, N = the current evolution iteration. Pass `iteration` explicitly to overwrite a past iteration's lesson — use this when you discover a prior lesson was wrong and needs correction. "
                           "You MUST call this before compact_context / end_evolution so the verdict survives into the next iteration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lesson": {
                        "type": "string",
                        "description": "A self-contained verdict on the hypothesis you tested this iteration (see plan.md): did it hold or break, the root cause, and the transferable takeaway. No 'see above' — make it stand alone."
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Your confidence in this lesson (0.0-1.0). 0.8+ = directly confirmed by trace evidence (failure mode disappeared from failing tasks). 0.5 = plausible but unverified (score moved but noise-band unclear). 0.2-0.4 = speculative direction. Default 0.5. A low-confidence lesson is better than a confident falsehood."
                    },
                    "iteration": {
                        "type": "integer",
                        "description": "Optional. Which iteration's lesson line to write (the N in [Iter N]). Defaults to the current evolution iteration. Pass a past iteration number to directly overwrite that iter's lesson — e.g., meta-evolve discovers iter 1's lesson was toxic, call with iteration=1 to replace it."
                    }
                },
                "required": ["lesson"]
            }
        }
    })

    # meta_bootstrap (cumulative cross-phase notebook for meta-evolve)
    tools.append({
        "type": "function",
        "function": {
            "name": "meta_bootstrap",
            "description": "Cumulative cross-round notebook (evolution/meta_bootstrap.md). Persists across ALL "
                           "meta-evolves (never rolls back with git). Each call APPENDS one numbered Meta#N record "
                           "(latest on top) — it does NOT overwrite. Read it FIRST each meta-evolve "
                           "(read_file evolution/meta_bootstrap.md) to see ALL past records AND to verify last "
                           "round's Prediction against the actual outcome. Then append this round's record before ending. "
                           "Every record has four sections — What / Why / Lesson / Prediction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "what": {
                        "type": "string",
                        "description": "What you changed THIS meta-evolve: which dimension (seed/commit/prompt), "
                                       "which file/function, and the concrete edit. e.g. 'seed: strategies/greedy.py — "
                                       "made select_seed prefer top-2 by val reward'."
                    },
                    "why": {
                        "type": "string",
                        "description": "Why you made this change: the problem you observed (e.g. analysis paralysis, "
                                       "reward plateau at iter N, overfitting dev) and the mechanism by which the "
                                       "change should help."
                    },
                    "lesson": {
                        "type": "string",
                        "description": "Transferable lesson: what worked/failed in PAST rounds, pitfalls to avoid, "
                                       "reusable patterns across the seed/commit/prompt dimensions."
                    },
                    "prediction": {
                        "type": "string",
                        "description": "Prediction for the NEXT iteration: expected reward / behavior change. "
                                       "Will be verified against actual outcome at the start of the next meta-evolve "
                                       "(record hit/miss + why)."
                    }
                },
                "required": ["what"]
            }
        }
    })

    # ========== Probe sub-agent tools ==========
    # probe — spawn a read-only investigation sub-agent
    tools.append({
        "type": "function",
        "function": {
            "name": "probe",
            "description": "Spawn a read-only investigation sub-agent. Target depends on your scope: "
                           "(1) In the evolve main loop, investigate per-task traces under eval_logs/ "
                           "to understand WHY a task failed — write a SPECIFIC instruction naming the "
                           "exact task file, the ONE question to answer, and the concrete signals to "
                           "look for (room sequences, action patterns, specific events). The evaluate "
                           "result shows which trace files exist; use those paths to form your own "
                           "diagnostic question. Vague instructions produce vague findings. "
                           "(2) In meta-evolve, investigate cross-iteration conversation logs under "
                           ".evolution_context/ (reward trends, decision quality, what was already "
                           "tried) when the question spans multiple files or phases — quick "
                           "single-file lookups are fine to do yourself with bash+jq. The sub-agent uses "
                           "read_file + bash (jq/grep) + end_probe and returns a cited findings "
                           "summary, saving your context for the decision. Max 50 steps. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instructions": {
                        "type": "string",
                        "description": "What to investigate. MUST be specific — the sub-agent is a "
                                       "search tool, not an analyst. Give it: (1) exact file path(s) "
                                       "from the evaluate result, (2) ONE question to answer, (3) concrete "
                                       "signals to look for (patterns, keywords, counts). Vague instructions "
                                       "like 'analyze the trace' will produce useless results. The "
                                       "sub-agent uses bash+jq to extract signal from large log files."
                    }
                },
                "required": ["instructions"]
            }
        }
    })

    # end_probe — probe sub-agent completion signal
    tools.append({
        "type": "function",
        "function": {
            "name": "end_probe",
            "description": "Signal the end of a probe investigation and return your findings. "
                           "Only available inside the probe sub-agent. Group findings as your "
                           "system prompt instructs (evolve: by task / failure theme; meta-evolve: "
                           "by dimension prompt/seed/commit/best), cite specific evidence (step "
                           "numbers, iteration numbers, reward deltas), and keep it concise but "
                           "actionable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "string",
                        "description": "Structured investigation findings summary."
                    }
                },
                "required": ["findings"]
            }
        }
    })

    # ========== Archive validation tools ==========
    # validate_archive
    tools.append({
        "type": "function",
        "function": {
            "name": "validate_archive",
            "description": "Validate the current select_*.py modules and strategy files by dry-running select_seed() "
                           "and select_commit(). Checks for: syntax errors, missing imports, unregistered "
                           "strategy names in _STRATEGY_SCHEDULE, whether select_seed() returns a valid result, "
                           "and whether select_commit() (when present) runs without error on a mock state. "
                           "Use this after modifying select_seed.py / select_commit.py or "
                           "strategy files in meta-evolve to verify changes before ending the meta-evolve phase.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    })

    # ========== Optional external tools (dynamically loaded) ==========
    if scope == "evolve":
        for tool in external_tools:
            tools.append(build_external_tool_schema(tool))

    # ========== scope filtering (data-driven whitelist; submit_best_pick etc. are
    # extended via extra_tools AFTER filtering, so they are retained) ==========
    allow = _SCOPE_TOOL_WHITELIST.get(scope)
    if allow is not None:
        tools = [t for t in tools if t.get("function", {}).get("name", "") in allow]

    # ========== exclude_tools filtering (evolve scope) ==========
    if exclude_tools and scope == "evolve":
        exclude_set = set(exclude_tools)
        tools = [t for t in tools if t.get("function", {}).get("name", "") not in exclude_set]

    # ========== description overrides ==========
    for tool in tools:
        name = tool.get("function", {}).get("name", "")
        if name in description_overrides:
            tool["function"]["description"] = description_overrides[name]

    # ========== Append extra tools ==========
    if extra_tools:
        tools.extend(extra_tools)

    return tools
