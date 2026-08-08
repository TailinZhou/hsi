"""
Iteration-level operations for GodelAgent evolution.

This module contains helper methods for managing single iteration operations
in the evolution process, separated from agent.py for better code organization.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from .state import (
    AgentState,
    AgentAction,
    ActionType,
    ACTION_TYPE_MAP,
    EvolutionPhase,
    IterationSummary,
    METADATA_FILES,
    reward_to_scalar,
    rank_versions_by_best_reward,
    fmt_reward,
)
from .utils.log_format import _C, _tool_color, log_phase_banner
from .utils.message_utils import update_history_from_response

if TYPE_CHECKING:
    from .agent import GodelAgent


def _parse_snapshot(snap):
    """Parse evaluation snapshot, handling both 3-element and legacy 2-element format."""
    if len(snap) >= 3:
        return snap[0], snap[1], snap[2]
    return snap[0], snap[1], "dev"


def _snapshot_execution_errors(snap) -> int:
    """Execution-error count stored as the snapshot's 4th element, else 0.

    A non-zero value means the harness CRASHED during that eval (e.g. a
    NameError). Used to veto tainted code versions from best-version
    selection. Returns 0 for legacy 3-element snapshots and for malformed
    entries so old resumed runs and agent-edited select_commit mocks stay
    backward-compatible (treated as "no crash → no veto").
    """
    try:
        if len(snap) > 3 and snap[3] is not None:
            return int(snap[3])
    except (TypeError, ValueError):
        pass
    return 0


def _snapshot_modified_files(snap) -> list:
    """Modified-files list stored as the snapshot's 5th element, else empty list.

    Each evaluate() call snapshots the current _modified_files so that
    per-commit KG nodes reflect the files actually modified to produce that
    version, not the entire iteration's accumulated set.

    Returns [] for legacy snapshots without the 5th element.
    """
    try:
        if len(snap) > 4 and snap[4] is not None:
            return list(snap[4])
    except (TypeError, ValueError):
        pass
    return []


def _snapshot_change_tags(snap) -> list:
    """Operation-type tags from the snapshot's 6th element (modifications_made list).

    Returns a sorted list of unique operation types, e.g. ["edit_file", "write_file"].
    Legacy snapshots without the 6th element return [].
    """
    try:
        if len(snap) > 5 and snap[5] is not None:
            mods = snap[5]
            if isinstance(mods, list):
                return sorted({
                    str(m.get("operation", ""))
                    for m in mods
                    if isinstance(m, dict) and m.get("operation")
                })
    except (TypeError, ValueError):
        pass
    return []


def _lookup_modified_files_for_hash(snapshots: list, code_hash: str) -> list:
    """Return the per-snapshot modified_files for a given code_hash.

    Searches evaluation_snapshots for entries matching code_hash and returns
    the modified_files from the first matching snapshot. Falls back to empty
    list if no snapshot has the 5th element (legacy data).
    """
    for snap in snapshots:
        ch, _r, _m = _parse_snapshot(snap)
        if ch == code_hash:
            mf = _snapshot_modified_files(snap)
            if mf:
                return mf
    return []


def _lookup_change_tags_for_hash(snapshots: list, code_hash: str) -> list:
    """Return the per-snapshot change_tags for a given code_hash.

    Same pattern as _lookup_modified_files_for_hash but reads the 6th element
    (modifications_made snapshot) and extracts operation types.
    """
    for snap in snapshots:
        ch, _r, _m = _parse_snapshot(snap)
        if ch == code_hash:
            tags = _snapshot_change_tags(snap)
            if tags:
                return tags
    return []


# pick_commit_version — the tool the choosing agent calls to commit its decision
# during the per-iteration commit-selection mini react loop. Injected as an extra
# tool (survives the scope whitelist filter, which only trims the base tool set).
# pick_commit_version — permanently in the evolve tool set (injected via extra_tools
# in agent.get_tools(scope="evolve")). The agent calls it during the main loop as a
# BOOKMARK, not as the final decision. At iteration end, a nudge prompt is appended
# to the SAME conversation (cache-preserving) for final confirmation.
PICK_COMMIT_VERSION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "pick_commit_version",
        "description": (
            "Add a code version to the commit pool for this iteration. "
            "The pool collects promising versions that may become seeds for "
            "future iterations. Call this after evaluate to add to the pool — "
            "you can call it multiple times; each call adds another version. "
            "The code_hash must come from an evaluate result. At iteration end "
            "you will finalize the pool by selecting which versions to commit."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code_hash": {
                    "type": "string",
                    "description": (
                        "The code_hash (content hash) from an evaluate result. "
                        "Must match exactly one of the hashes you've seen in evaluate output."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Brief reason for adding to the pool, e.g. 'val-validated', "
                        "'generalizes best across multiple evals', 'avoids dev-overfit', "
                        "'most reliable after N evaluations', 'new architectural direction'."
                    ),
                },
            },
            "required": ["code_hash"],
        },
    },
}


# finalize_commit_pool — the tool the agent calls during the nudge phase to
# select MULTIPLE versions for committing (replaces the old single pick_commit_version
# nudge pattern). Each selected code_hash becomes a separate pool commit with
# its own git tag, available as a seed for future iterations.
FINALIZE_COMMIT_POOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "finalize_commit_pool",
        "description": (
            "Finalize the commit pool by selecting code_hashes to commit as seeds "
            "for future iterations. Choose versions that represent DIFFERENT "
            "exploration directions or have HIGH potential for further optimization. "
            "Each selected version becomes a separate pool commit with its own git tag. "
            "List order is significant. The first version you list is left on disk "
            "and inspected by meta-evolution as this iteration's representative/primary "
            "direction — put your main intended direction first, not necessarily the "
            "highest-reward one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code_hashes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "code_hash": {
                                "type": "string",
                                "description": "The code_hash to commit (from the candidate table).",
                            },
                            "description": {
                                "type": "string",
                                "description": "One-line description of what this version changes and why it was selected.",
                            },
                        },
                        "required": ["code_hash"],
                    },
                    "description": (
                        "List of versions to commit. Each must include the code_hash "
                        "and a one-line description. Select 2-5 versions that "
                        "represent different exploration directions. Avoid duplicates "
                        "(same code_hash appearing multiple times will be deduplicated). "
                        "List order is significant. The first version you list is left "
                        "on disk and inspected by meta-evolution as this iteration's "
                        "representative/primary direction — put your main intended "
                        "direction first, not necessarily the highest-reward one."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why these versions were chosen.",
                },
            },
            "required": ["code_hashes"],
        },
    },
}


class EvolveHelper:
    """
    Iteration-level operations for GodelAgent evolution.

    This class encapsulates all the methods that operate on a single iteration,
    making agent.py more readable by separating iteration logic from core capabilities.
    """

    def __init__(self, agent: "GodelAgent"):
        """
        Initialize EvolveHelper.

        Args:
            agent: The GodelAgent instance to operate on.
        """
        self.agent = agent

    def _log(self, message: str) -> None:
        """Delegate logging to agent."""
        self.agent._log(message)

    def _load_evolution_module(self, module_name: str):
        """Load a Python module from the evolution/ subdirectory.

        Uses importlib to dynamically load modules from the agent's
        evolution/ directory, avoiding sys.modules caching issues.

        Args:
            module_name: Module filename without extension (e.g., "archive")

        Returns:
            Loaded module object, or None if not found
        """
        import importlib.util

        file_path = os.path.join(self.agent.agent_code_dir, "evolution", f"{module_name}.py")
        if not os.path.exists(file_path):
            return None
        try:
            spec = importlib.util.spec_from_file_location(f"evolution.{module_name}", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            self._log(f"Warning: Failed to load evolution module {module_name}: {e}")
            return None

    def _load_evolution_prompt_md(self) -> str | None:
        """Load evolution_base_prompt.md from the evolution/ subdirectory.

        Returns:
            Prompt text, or None if file not found.
        """
        md_path = os.path.join(self.agent.agent_code_dir, "evolution", "evolution_base_prompt.md")
        if not os.path.exists(md_path):
            return None
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            self._log(f"Warning: Failed to load evolution_base_prompt.md: {e}")
            return None

    def _build_git_status_section(self) -> str:
        """
        Build the Git Status section, injected as S_t environment info.

        Returns:
            Formatted Git Status string.
        """
        git_ctrl = self.agent.git_controller

        # Edge case: not a git repository
        if not git_ctrl.is_git_repo():
            return "Not a git repository."

        # Collect git info
        current_branch = git_ctrl.get_current_branch()
        main_branch = git_ctrl.get_main_branch()
        recent_commits = git_ctrl.get_recent_commits(n=5)
        wd_status = git_ctrl.get_working_directory_status()

        # Edge case: detached HEAD
        is_detached = current_branch is None

        lines = []

        # Branch info
        if is_detached:
            commit = git_ctrl.get_current_commit()
            short_commit = commit[:7] if commit else "unknown"
            lines.append(f"**HEAD:** detached at `{short_commit}`")
        else:
            lines.append(f"**Current Branch:** `{current_branch}`")

        if main_branch:
            lines.append(f"**Main Branch:** `{main_branch}`")

        # Recent commits
        if recent_commits:
            lines.append("\n**Recent Commits:**")
            for commit in recent_commits:
                msg_preview = commit['message'][:50] + ("..." if len(commit['message']) > 50 else "")
                lines.append(f"  - `{commit['short_hash']}` {msg_preview}")

        # Working-directory status
        staged = wd_status.get("staged", [])
        modified = wd_status.get("modified", [])
        untracked = wd_status.get("untracked", [])

        if staged or modified or untracked:
            lines.append("\n**Working Directory Changes:**")
            if staged:
                files_str = ", ".join(f"`{f}`" for f in staged[:5])
                suffix = f" ... and {len(staged) - 5} more" if len(staged) > 5 else ""
                lines.append(f"  - Staged ({len(staged)}): {files_str}{suffix}")
            if modified:
                files_str = ", ".join(f"`{f}`" for f in modified[:5])
                suffix = f" ... and {len(modified) - 5} more" if len(modified) > 5 else ""
                lines.append(f"  - Modified ({len(modified)}): {files_str}{suffix}")
            if untracked:
                files_str = ", ".join(f"`{f}`" for f in untracked[:5])
                suffix = f" ... and {len(untracked) - 5} more" if len(untracked) > 5 else ""
                lines.append(f"  - Untracked ({len(untracked)}): {files_str}{suffix}")
        else:
            lines.append("\n**Working Directory:** Clean (no uncommitted changes)")

        return "\n".join(lines)

    def process_tool_calls(
        self,
        tool_calls: List[Dict],
        tool_results: List[str],
        *,
        messages: "MessageHistory" = None,
    ) -> None:
        """
        Process tool calls from react() response.

        Args:
            tool_calls: List of tool calls made, format: [{"id": str, "name": str, "args": dict}]
            tool_results: List of tool execution results
            messages: Custom message history; defaults to agent.message_history.
        """
        msgs = messages or self.agent.message_history
        n_calls = len(tool_calls)
        for idx, (tc, result) in enumerate(zip(tool_calls, tool_results), start=1):
            self.agent._actions_in_iteration += 1

            # Map tool name to ActionType
            action_type = ACTION_TYPE_MAP.get(tc['name'], ActionType.EXTERNAL_TOOL)

            # Create AgentAction
            action = AgentAction(
                action_type=action_type,
                params=tc['args'],
                result=result,
            )

            # Log: tool-execution details (with color and hierarchy)
            tool_name = tc['name']
            color = _tool_color(tool_name)
            self._log(f"  {_C.D}┌─{_C.RST}{_C.B}{color} [{idx}/{len(tool_calls)}] {tool_name}{_C.RST}")
            if tc['args']:
                try:
                    params_str = json.dumps(tc['args'], ensure_ascii=False, indent=4)
                    if len(params_str) > 500:
                        params_str = params_str[:1000] + '...'
                    self._log(f"  {_C.D}│{_C.RST} {_C.D}Params:{_C.RST}")
                    for line in params_str.split('\n'):
                        self._log(f"  {_C.D}│{_C.RST}   {line}")
                except Exception:
                    self._log(f"  {_C.D}│{_C.RST} {_C.D}Params:{_C.RST} {tc['args']}")
            self._log(f"  {_C.D}│{_C.RST} {_C.D}Result:{_C.RST}")
            if tool_name == 'evaluate':
                result_display = result
            else:
                result_display = result[:1000] + '...' if len(result) > 500 else result
            for line in result_display.split('\n'):
                self._log(f"  {_C.D}│{_C.RST}   {line}")
            self._log(f"  {_C.D}└──{_C.RST}")

            # The last tool result of this step gets a status suffix appended so
            # Step 2+ can continue without a user prompt. record_action still
            # records the original result (the status suffix does not pollute
            # the action history).
            msg_content = result
            if idx == n_calls:
                msg_content = result + self._build_step_status_suffix()

            # Update message history (add tool response)
            msgs.add_tool(msg_content, tc['id'])

            # Record action
            self.agent.state.record_action(action)

        # Truncate message history (keep the most recent N messages)
        max_history = getattr(self.agent.config, 'max_history_messages', 15)
        msgs.truncate(max_messages=max_history)

    def build_decision_prompt(self, seed_hypothesis: str = "") -> str:
        """Build the iteration's first user prompt (injected only at Step 1).

        Step 2+ no longer receives a user prompt; continuation relies on the
        tool result (status line in ``_build_step_status_suffix``). Environment
        / git status / code files / lifecycle are in the system prompt
        (``_build_system_env_section``); evaluate history is in evaluate's tool
        result. Only Step-1-specific dynamic content is kept here: the iteration
        header + baseline/reward/steps + the seed hypothesis (iter 2+ only).
        """
        from .evolution_prompt import get_iteration_begin_prompt

        state_summary = {
            "steps_taken": self.agent._step_in_iteration,
            "max_steps_per_iteration": self.agent.config.max_steps_per_iteration,
            "reward": self.agent.state.reward,
        }

        # Use cached baseline reward (computed once per iteration)
        code_baseline_reward = self._cached_baseline_reward

        # Init harness's measured "experience": injected only at iteration 1
        # (a bootstrap product) so the first self-rewrite is driven by the
        # harness's lived feedback. Empty for other iterations or on resume.
        init_experience = (
            self.agent._init_experience
            if self.agent.iteration == 1 and getattr(self.agent, "_init_experience", None)
            else ""
        )

        # Hypothesis only injected from iteration 2 onwards.
        # Iteration 1 has no seed selection — the init harness IS the start.
        is_iter1 = self.agent.iteration == 1
        prompt = get_iteration_begin_prompt(
            iteration=self.agent.iteration,
            state_summary=state_summary,
            code_baseline_reward=code_baseline_reward,
            repo_path=self.agent.agent_code_dir,
            init_experience=init_experience,
            seed_hypothesis=seed_hypothesis if not is_iter1 else "",
            hypotheses=getattr(self, '_seed_hypotheses', []) if not is_iter1 else [],
        )

        # Prepend goal (moved from system prompt) so it's the opening section
        # of this single user message.
        if self.agent.goal:
            prompt = f"## Evolution Goal\n\n{self.agent.goal}\n\n{prompt}"

        return prompt

    def _build_benchmark_info_section(self) -> str:
        """Build the benchmark task-count info section (duck-typed, no import coupling).

        Before calling get_task_summary(), the user's benchmark config is first
        injected into the evaluator via _inject_llm_config(), so when
        _load_and_split_tasks() is triggered the evaluator sees the user's
        config rather than defaults (e.g. num_episodes).

        Returns:
            Formatted task-num section, e.g. "dev=1 | val=0 | test=0".
            Returns an empty string when external_evaluator is absent or lacks
            get_task_summary.
        """
        adapter = self.agent.external_evaluator
        if adapter is None:
            return ""
        if not hasattr(adapter, 'get_task_summary'):
            return ""
        # Ensure the evaluator has the user's benchmark config before
        # get_task_summary() may trigger _load_and_split_tasks().
        # _inject_llm_config is idempotent — safe to call every iteration.
        if hasattr(adapter, '_inject_llm_config'):
            try:
                adapter._inject_llm_config(self.agent)
            except Exception:
                pass
        try:
            s = adapter.get_task_summary()
        except Exception:
            return ""
        return (
            f"### Task num\n\n"
            f"dev={s['dev_size']} | val={s['val_size']} | test={s['test_size']}\n\n"
        )

    def _build_sandbox_info_section(self, scope: str = "evolve") -> str:
        """Build the bwrap-sandbox filesystem-visibility description (injected into the Environment section).

        When scope="evolve", notes that evolution/ is hidden;
        when scope="meta_evolve", notes that evolution/ is fully visible and writable.
        """
        repo = str(self.agent.repo_path)
        run_dir = os.path.dirname(repo)
        eval_logs = os.path.join(run_dir, 'eval_logs')
        # evolve.py lives under src/react_loop/; the framework dir is os.path.dirname(__file__).
        framework = os.path.realpath(os.path.dirname(__file__))

        lines = [
            "### Sandbox (bwrap filesystem isolation)",
            "",
            "Your shell commands run inside a **bwrap sandbox**. "
            "Only the paths listed below are visible:",
            "",
            "| Path | Access | Notes |",
            "|------|--------|-------|",
            f"| `{repo}/` | **rw** | Your harness code (working directory) |",
            f"| `{eval_logs}/` | ro | Evaluation logs |",
            f"| `{framework}/` | ro | Framework source (`react_loop`) |",
            "| `/usr`, `/lib`, `/lib64`, `/bin`, `/etc` | ro | System basics |",
            "| `/tmp` | tmpfs | Temporary files (not shared with host) |",
            "",
            "**Everything else is invisible** — including `/home`, `/mnt` "
            "outside the mounted subtrees, and the project root (`.env`, "
            "`config.yaml`, `benchmark_config_goal/`, `src/benchmark/`).",
            "",
            f"**PYTHONPATH** includes the `src/` parent of `react_loop/`, "
            f"so you can `from react_loop.xxx import yyy`.",
        ]

        if scope == "evolve":
            evo_dir = os.path.join(repo, 'evolution')
            lines.append("")
            lines.append(
                f"**`{evo_dir}/` is hidden** (tmpfs overlay) — you cannot "
                f"see or modify evolution strategy files. Focus on your harness."
            )
        else:
            lines.append("")
            lines.append(
                f"**`{repo}/evolution/` is fully visible and writable** — "
                f"this is YOUR domain to improve (archive strategies, prompts)."
            )

        return "\n".join(lines)

    def _build_system_env_section(self) -> str:
        """Build the `## Environment (S_t)` section injected into the system prompt (static within an iteration).

        Includes the working directory / OS / Python / git baseline, git status,
        available code files, and the task lifecycle (YOUR_TASK_TEXT). Static
        context that does not change per step lives in the system prompt so it
        is not repeated in every user prompt and the prompt-cache prefix is
        preserved.
        """
        from .evolution_prompt import YOUR_TASK_TEXT

        files_list = "\n".join(
            f"- {f}" for f in sorted(self.agent.action_executor.agent_codes.keys())
        )
        if not files_list:
            files_list = "(none loaded yet)"
        git_status = self._build_git_status_section()
        platform_info = "Windows (PowerShell)" if sys.platform == 'win32' else "Linux/macOS (bash)"
        benchmark_info = self._build_benchmark_info_section()
        sandbox_info = self._build_sandbox_info_section("evolve")
        framework_dir = os.path.realpath(os.path.dirname(__file__))

        return (
            f"\n\n## Environment (S_t)\n\n"
            f"**Working Directory:** `{self.agent.agent_code_dir}`\n"
            f"**OS:** {platform_info}\n"
            f"**Python:** available as `python`\n"
            f"**Git:** available for version control\n\n"
            f"{sandbox_info}\n\n"
            f"### Git Status\n{git_status}\n\n"
            f"### Available Code Files\n\n"
            f"The following files are in your strategy code directory:\n{files_list}\n\n"
            f"### Framework Source (read-only)\n\n"
            f"Your runtime is driven by the framework at `{framework_dir}/`. "
            f"It is read-only (bwrap ro mount); you can inspect it with "
            f"`read_file(\"<absolute_path>\")` but you cannot modify it.\n\n"
            f"**Do not deep-dive into the framework unless you hit a bug "
            f"or an unexpected error.** Your job is to improve your harness, "
            f"not to study the runtime. The `framework_contract.md` already "
            f"documents the key mechanisms you need.\n\n"
            f"When you do need to debug, the key files are:\n"
            f"- `evolve.py` — per-iteration driver\n"
            f"- `agent.py` — agent loop & LLM interaction\n"
            f"- `actions/agent_evaluator.py` — evaluate() & reward collection\n"
            f"- `actions/agent_action.py` — tool dispatch & sandbox\n"
            f"- `actions/agent_file_ops.py` — read/write/edit resolution\n"
            f"- `state.py` — state machine, ActionType, reward tracking\n\n"
            f"{benchmark_info}"
            f"---\n\n{YOUR_TASK_TEXT}"
        )

    def _build_step_status_suffix(self) -> str:
        """Build the status line appended to the last tool result of this step.

        In continuation-only mode, Step 2+ has no user prompt; this status line
        lets the agent see at a glance the current step / context usage / current
        reward / current best version, and decide whether to keep editing or
        wrap up. The reward portion is omitted before the first evaluate
        (state.reward is None).
        """
        agent = self.agent
        step = agent._step_in_iteration
        max_steps = agent.config.max_steps_per_iteration
        ctx_tokens = agent._iteration_prompt_tokens
        max_ctx = getattr(agent.config, 'max_context_tokens', 128000)
        ctx_pct = int(ctx_tokens / max_ctx * 100) if max_ctx else 0

        bits = [f"Step {step}/{max_steps}", f"ctx {ctx_pct}%"]

        eval_n = agent._eval_count_in_iteration
        if eval_n > 0:
            bits.append(f"evals:{eval_n}")

        cur = agent.state.reward
        eval_mode = agent.state.last_eval_mode
        if cur is not None and eval_mode:
            bits.append(f"cur={fmt_reward(cur)}({eval_mode})")

        best_version = self._find_best_code_version()
        if best_version:
            _, best_reward, best_mode = best_version
            bits.append(f"best★={fmt_reward(best_reward)}({best_mode})")

        line = "— [" + " | ".join(bits) + "]"
        return "\n" + line

    def _execute_react_step(
        self,
        tool_executor,
        *,
        messages: "MessageHistory" = None,
        tools: List[Dict] = None,
    ) -> tuple:
        """Run one react step: call LLM, process response and tool calls.

        Args:
            tool_executor: (tool_name, args) -> result
            messages: Custom message history; defaults to agent.message_history.
            tools: Custom tool list; defaults to the evolve tools.

        Returns:
            (has_tool_calls, tool_calls_made, tool_results)
        """
        msgs = messages or self.agent.message_history
        used_tools = tools or self.agent.get_tools(scope="evolve")

        response, tool_calls_made, tool_results = self.agent.react(
            messages=msgs.get_messages_for_llm(),
            tools=used_tools,
            tool_executor=tool_executor,
        )

        message = response.choices[0].message

        # Extract reasoning_content (tolerant of varying model attribute names)
        reasoning_content = getattr(message, 'reasoning_content', None)
        if reasoning_content is None:
            reasoning_content = getattr(message, 'reason_content', None)
        if reasoning_content:
            self.agent.state.reasoning_contents.append(reasoning_content)

        # Log: tool-call names
        if tool_calls_made:
            names = [tc['name'] for tc in tool_calls_made]
            self._log(f"  {_C.D}[LLM Response] Tool calls:{_C.RST} {names}")

        # Update message history
        update_history_from_response(response, msgs, reasoning_content=reasoning_content)

        if tool_calls_made:
            self.process_tool_calls(tool_calls_made, tool_results, messages=msgs)

        return (bool(tool_calls_made), tool_calls_made, tool_results)

    def run_init_eval(self) -> Optional[str]:
        """Bootstrap: evaluate the init harness once on the dev set before iteration 1.

        Produces two grounded signals that drive the first self-rewrite:
        - The init harness's own lived experience (an eval-summary text) → injected
          into the iteration-1 prompt as "Init Harness Experience", so iteration 1
          is driven by the harness's measured pain-points instead of cold code reading.
        - A real dev reward for the iteration-0 seed record (replacing the placeholder
          0.0), so ``select_seed`` and the iteration-1 baseline reflect actual perf.

        Calls the external evaluator **directly**, bypassing
        ``AgentActionExecutor.evaluate`` so the iteration-1 ``AgentState`` /
        ``evaluation_snapshots`` stay clean. To run the init harness under the same
        conditions as a normal evaluate, it first mirrors ``reset_for_iteration``'s
        setup (scan harness tools + load codes, the latter feeds ``_harness_source``
        for code-aware eval summaries).

        Returns the experience text, or None if the eval produced no reward / the
        experience text came back empty (caller treats None as "no injection").
        """
        agent = self.agent
        evaluator = agent.external_evaluator
        if evaluator is None:
            return None

        agent._log(agent._format_banner(
            "  INIT HARNESS EVALUATION  ", "BMA"))
        self._log("  Evaluating the init harness on the dev set before iteration 1...")

        # Mirror reset_for_iteration so the init harness runs under the same
        # conditions as a normal evaluate. Safe at bootstrap: load_codes() guards
        # its state.update_pi() call behind `if self.state:` (state is None here).
        try:
            agent._scan_external_tools()
        except Exception as e:
            self._log(f"  Warning: tool scan before init eval failed: {e}")
        try:
            agent.action_executor.load_codes()
        except Exception as e:
            self._log(f"  Warning: load_codes before init eval failed: {e}")

        try:
            reward, metrics = evaluator(
                agent, None, [], None,
                evaluate_seq=0, iteration=0,
                eval_mode="dev",
            )
        finally:
            # Residue cleanup: the harness may have parked its interaction log on
            # the agent via set_harness_interaction_log(). Bootstrap is off the
            # normal per-task evaluate path, so drop it before iteration 1.
            agent.clear_harness_interaction_log()

        if reward is None:
            err = metrics.get("error") if isinstance(metrics, dict) else None
            self._log(
                "  Init eval produced no reward; skipping experience injection."
                + (f" ({err})" if err else "")
            )
            return None

        # Experience text via the standalone summary helper. It is pure-read re:
        # AgentState (no state mutation); it may mutate the local `metrics` dict
        # (strips api_messages) which we do not reuse, so that is harmless here.
        try:
            experience = agent.action_executor._summarize_evaluation(reward, metrics)
        except Exception as e:
            self._log(f"  Warning: init eval summary failed: {e}")
            experience = ""

        # Backfill the iteration-0 seed record in place with the real dev reward.
        # No tracker "update" API exists; record_iteration only appends and would
        # duplicate the seed. get_iteration(0) returns the seed (metadata={} ⇒
        # is_main_iteration=True). Its reward rides the tracker, persisted via
        # _save_metadata() — the only persistence hook.
        try:
            rec = agent.evolution_tracker.get_iteration(0)
            if rec is not None:
                rec.reward = [reward_to_scalar(reward)]
                rec.metadata["committed_code_reward"] = [reward]
                rec.metadata["committed_eval_mode"] = ["dev"]
                agent.evolution_tracker._save_metadata()
        except Exception as e:
            self._log(f"  Warning: failed to record init seed reward: {e}")

        self._log(f"  Init harness dev reward: {fmt_reward(reward)}")
        return experience or None

    def run_iteration(self, strategy_hint: str = "", hypothesis: str = "",
                      hypotheses: list = None) -> str:
        """
        Execute one evolution iteration using react loop.

        The agent autonomously decides actions via self.react() until:
        - Agent calls compact_context (ends iteration normally)
        - Agent calls end_evolution (ends entire evolution)
        - Max actions reached (auto-compacts and continues to next iteration)

        Args:
            strategy_hint: Optional hint from archive.select_seed() injected into system prompt.
            hypothesis: Optional seed hypothesis from archive.select_seed() for system prompt injection.
                Controlled by config.inject_seed_hypothesis.
            hypotheses: Optional list of competing hypothesis dicts from seed selection.

        Returns:
            "end_evolution" or "continue"
        """
        # ── Phase banner ──────────────────────────────────────────
        log_phase_banner(
            self.agent,
            f"MAIN EVOLVE (iter {self.agent.iteration})",
            info=f"  max_steps: {self.agent.config.max_steps_per_iteration} | "
                 f"hypothesis: {'yes' if hypothesis else 'none'}",
        )

        # Reset and prepare for iteration
        parent_commit = self.reset_for_iteration(strategy_hint=strategy_hint, hypothesis=hypothesis,
                                                 hypotheses=hypotheses)

        # Create tool executor for this iteration (inline).
        # Intercepts pick_commit_version to record bookmarks during the main loop.
        # The final commit decision happens as a continuation of this SAME conversation
        # (cache-preserving nudge), not a separate react loop with filtered tools.
        def tool_executor(tool_name: str, args: Dict) -> str:
            if tool_name == "pick_commit_version":
                self.agent.state.commit_picks.append({
                    "code_hash": args.get("code_hash", ""),
                    "reason": args.get("reason", ""),
                    "step": self.agent._step_in_iteration,
                })
                ch = args.get("code_hash", "")[:12]
                return (
                    f"Added to commit pool ({len(self.agent.state.commit_picks)} versions total): "
                    f"{ch}... (step {self.agent._step_in_iteration}). "
                    f"Call again to add more promising versions to the pool. "
                    f"At iteration end you will finalize the pool."
                )
            if tool_name == "finalize_commit_pool":
                return (
                    "finalize_commit_pool is NOT available during the main "
                    "evolve loop. Use pick_commit_version to add versions to "
                    "the pool. At iteration end, a nudge prompt will ask you "
                    "to call finalize_commit_pool to select which versions to "
                    "actually commit."
                )
            return self.agent.execute_tool(tool_name, args, scope="evolve")

        # Agent decision loop using react
        while not self.agent.state.is_iteration_complete():
            if self.agent._step_in_iteration >= self.agent.config.max_steps_per_iteration:
                self._log("Max steps reached, auto-compacting context and continuing")
                self.agent.state.max_steps_reached = True
                self.agent.state.mark_iteration_ended(
                    summary="Auto-ended: max steps reached",
                    reason="max_steps_limit"
                )
                break

            # Increment the step counter (one LLM call = one step)
            self.agent._step_in_iteration += 1

            # Inject the first user prompt only at Step 1; Step 2+ continues
            # from tool results (status line at the end of the last tool
            # result — see _build_step_status_suffix).
            if self.agent._step_in_iteration == 1:
                prompt = self.build_decision_prompt(
                    seed_hypothesis=getattr(self, '_seed_hypothesis_text', '')
                )
                self.agent.message_history.add_user(prompt)

            # Step header
            step_label = f" Step {self.agent._step_in_iteration} "
            raw_reward = self.agent.state.reward
            if raw_reward is None:
                reward_val = "None"
            elif isinstance(raw_reward, dict):
                reward_val = ", ".join(f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in raw_reward.items())
            else:
                reward_val = f"{raw_reward:.4f}"
            # Show eval mode tag if available
            eval_mode = self.agent.state.last_eval_mode
            reward_label = f"Reward({eval_mode})" if eval_mode else "Reward"
            detail = f" Steps: {self.agent._step_in_iteration}/{self.agent.config.max_steps_per_iteration} | Actions: {self.agent._actions_in_iteration} | {reward_label}: {reward_val} "
            line_len = max(0, 55 - len(step_label) - len(detail))
            self._log(
                f"\n{_C.B}{_C.BBL}>{step_label}--{_C.RST}{_C.D}{detail}{'-' * line_len}{_C.RST}"
            )

            # Use react for decision-making and execution
            has_tools, tool_calls_made, tool_results = self._execute_react_step(tool_executor)

            # Handle the no-tool-calls case
            if not has_tools:
                if self.agent.state.is_iteration_complete():
                    break  # Iteration complete; exit the loop
                else:
                    # Prompt the agent to use compact_context or continue
                    self.agent.message_history.add_user(
                        "No tool calls made. Call compact_context to end this iteration, "
                        "or continue with other tools (read_file, edit_file, evaluate, etc.)."
                    )
                    continue

            # Update phase
            self.agent.phase = EvolutionPhase.EVOLVING

            # Detect evolution_ended (the agent called end_evolution)
            if self.agent.state.evolution_ended:
                break

        # ── Max-steps version selection: give the agent one last chance to pick a commit version ──
        if self.agent.state.max_steps_reached and not self.agent.state.evolution_ended:
            version_prompt = (
                "MAXIMUM STEPS REACHED. This iteration is ending now.\n"
                "IMPORTANT: You MUST call lesson and compact_context to end this iteration.\n"
                "First, record this iteration's cross-iteration lesson:\n"
                "  lesson(lesson=\"...\", confidence=0.5)\n"
                "Then, end the iteration:\n"
                "  compact_context(summary=\"...\", reason=\"max_steps\")\n"
                "If you saved a better version as a git commit, "
                "use bash to run 'git checkout <commit> -- .' first.\n"
                "If you stashed better changes, run 'git stash pop' first.\n"
                "Your selection will be archived as this iteration's final result."
            )
            self.agent.message_history.add_user(version_prompt)

            # Give the agent up to 3 react turns to complete git checkout + compact_context
            max_final_steps = 3
            for _ in range(max_final_steps):
                if self.agent.state.is_iteration_complete():
                    break
                self._execute_react_step(tool_executor)

            # If the agent still hasn't called compact_context, force-end it
            if not self.agent.state.is_iteration_complete():
                self._log("  Max steps final steps exhausted, forcing compact_context")
                self.agent.state.mark_iteration_ended(
                    summary="Auto-ended: max steps reached (forced)",
                    reason="max_steps_limit"
                )

        # Iteration end-of-life handling
        is_max_steps = self.agent.state.max_steps_reached
        is_evolution_end = self.agent.state.evolution_ended
        is_agent_ended = self.agent.state.iteration_ended and not is_max_steps

        # Generate summary (max_steps also counts as success)
        summary = self.generate_iteration_summary(
            success=True,
            max_steps_reached=is_max_steps,
        )

        # Print summary to console
        if is_max_steps:
            status = f"{_C.BYE}MAX_STEPS{_C.RST} (auto-compacted)"
        elif is_evolution_end:
            status = f"{_C.RD}EVOLUTION_ENDED{_C.RST} (agent requested)"
        else:
            status = f"{_C.BGR}SUCCESS{_C.RST} (agent ended)"
        self._log(f"\n  {_C.D}┌─{_C.RST}{_C.B}{_C.BCY} ITERATION {self.agent.iteration} SUMMARY {_C.RST}")
        self._log(f"  {_C.D}│{_C.RST} Status: {status}")
        self._log(f"  {_C.D}│{_C.RST} Reward: {_C.B}{fmt_reward(summary.reward)}{_C.RST}  |  Modifications: {summary.modifications_count}")
        self._log(f"  {_C.D}│{_C.RST} {summary.summary_text}")
        self._log(f"  {_C.D}└──{_C.RST}\n")

        # Lesson-nudge fallback: if the agent never called lesson() during the
        # iteration (or the max-steps path), nudge it to record ONE cross-iteration
        # verdict before committing. No silent framework synthesis — the agent
        # writes the lesson itself, grounded in plan.md's hypothesis + the
        # traces/rewards already in-context. Runs as a continuation of the SAME
        # conversation (cache-preserving), before the commit-pool nudge.
        self._ensure_lesson_recorded()

        # Commit iteration BEFORE saving context so summary.commit_hash is
        # populated (commit_iteration backfills it from the new commit).
        # Must stay a single save call: EvolutionContext.add_summary appends
        # without dedup, so calling save both before and after would duplicate.
        chosen_hashes = self._finalize_commit_pick()
        self.commit_iteration(summary, chosen_hashes=chosen_hashes)

        self.save_iteration_context(summary)

        # Determine the end reason
        if is_evolution_end:
            end_reason = "end_evolution"
        elif is_max_steps:
            end_reason = "max_steps"
        else:
            end_reason = "compact_context"
        self.agent._last_iteration_end_reason = end_reason

        # Return signal
        if is_evolution_end:
            return "end_evolution"
        return "continue"

    def reset_for_iteration(self, strategy_hint: str = "", hypothesis: str = "",
                            hypotheses: list = None) -> str:
        """
        Reset state and prepare for a new iteration.

        Args:
            strategy_hint: Optional hint from archive.select_seed() to inject into system prompt.
            hypothesis: Optional seed hypothesis from archive.select_seed() for system prompt injection.
            hypotheses: Optional list of competing hypothesis dicts from seed selection.

        Returns:
            parent_commit: The commit hash before this iteration (for rollback)
        """
        # Reset state for new iteration
        self.agent.state = AgentState(
            iteration=self.agent.iteration,
            goal=self.agent.goal,
        )
        self.agent._actions_in_iteration = 0  # Tool-call counter
        self.agent._step_in_iteration = 0     # LLM-call counter
        self.agent._iteration_prompt_tokens = 0  # Reset token counter
        self.agent._eval_count_in_iteration = 0
        self.agent.phase = EvolutionPhase.EVOLVING

        # Reset action_executor's modified-files set
        self.agent.action_executor._modified_files = set()
        # Sync _file_hashes to the current on-disk state. _file_hashes is
        # lazy-initialized (scanned only when empty) and does not auto-track the
        # disk — if files changed by the previous round of meta-evolve / seed
        # switching (e.g. evolution/*.py) are not re-scanned, this round's
        # _auto_reload_if_changed will mis-judge disk(new) ≠ cache(old) as
        # "modified this round", polluting _modified_files → KG modified_files
        # → structural_similarity.
        self.agent.action_executor._file_hashes = self.agent.action_executor._scan_py_files()

        # Reset message history
        self.agent.message_history = type(self.agent.message_history)()  # Create new instance

        # Load code (via action_executor)
        codes = self.agent.action_executor.load_codes()

        # Scan external tools (each iteration)
        self.agent._scan_external_tools()

        self.agent.action_executor.set_state(self.agent.state, codes=codes)

        # Log available tools for this iteration (single compact line)
        evolve_tools = self.agent.get_tools(scope="evolve")
        tool_names = [t["function"]["name"] for t in evolve_tools]
        shell_type = "PowerShell" if sys.platform == 'win32' else "bash"

        # Add the system message (with historical context).
        # The system prompt is two-source concatenation: the FIXED framework
        # contract (identity/architecture/lifecycle/eval — from src/, meta
        # cannot edit it) + the EVOLVABLE strategy half (evolution/ copy
        # preferred, src fallback). Order: framework first (grounding),
        # strategy second.
        from .evolution_prompt import get_framework_contract
        framework = get_framework_contract()
        evolvable_prompt = self._load_evolution_prompt_md()
        if evolvable_prompt is None:
            from .evolution_prompt import get_base_system_prompt
            evolvable_prompt = get_base_system_prompt()
        # Resume guard: a pre-split run's evolution/ copy still embeds the
        # full contract, so concatenating duplicates Lifecycle / Evaluation /
        # Strategy Code. Don't rewrite meta's file — just warn so the operator
        # starts a new run (new runs copy the split seed automatically).
        if any(marker in evolvable_prompt for marker in (
            "## Lifecycle", "## Evaluation & Data Split", "## Your Strategy Code",
        )):
            self._log(
                "Warning: evolution/evolution_base_prompt.md predates the "
                "framework/strategy split — its sections duplicate the framework "
                "contract. Start a new run to get the split layout."
            )
        # Static environment (S_t) follows the framework contract so the agent
        # perceives the environment before seeing the strategy.
        system_prompt = framework + "\n\n" + self._build_system_env_section()

        # The evolution strategy (evolution_base_prompt.md) is placed after the
        # environment info, labeled "Your Evolution Strategy", so the agent
        # clearly perceives this as the strategy layer that meta-evolve evolves.
        # A primer is injected at the top of the prompt via the primacy effect,
        # to keep the bottom strategy from being drowned by the long contract +
        # environment info.
        strategy_header = (
            "\n\n---\n"
            "## Your Evolution Strategy (your core operating manual for this iteration)\n\n"
        )
        strategy_footer = (
            "\n\n---\n"
            "**The Evolution Strategy above is your core operating manual. "
            "The framework contract defines tools and rules; the strategy "
            "defines HOW you think, decide, and act. Internalize it.**"
        )
        system_prompt = (
            "> **Read `## Your Evolution Strategy` at the bottom of this prompt "
            "before making decisions — it defines how you think and act this "
            "iteration.**\n\n"
            + system_prompt
        )
        system_prompt += strategy_header + evolvable_prompt + strategy_footer

        # Add a note based on the previous iteration's end reason
        if self.agent.iteration > 0 and self.agent._last_iteration_end_reason:
            if self.agent._last_iteration_end_reason == "max_steps":
                note = "\n\n## Previous Iteration Note\n\nThe previous iteration ended because the maximum steps limit was reached. Your changes were committed. Continue from where you left off.\n"
                system_prompt = system_prompt + note

        # Inject seed hypothesis from archive.select_seed() (or fall back to strategy_hint)
        inject = getattr(self.agent.config, 'inject_seed_hypothesis', True)
        if inject and hypothesis:
            self._seed_hypothesis_text = hypothesis
        elif inject and strategy_hint:
            # backward compat: fall back to old strategy_hint if no hypothesis
            self._seed_hypothesis_text = f"## Archive Strategy Hint\n\nSeed: {strategy_hint}"
        else:
            self._seed_hypothesis_text = ""

        # Store competing hypotheses from seed selection (new format)
        self._seed_hypotheses = hypotheses or []

        # plan.md: clear the ephemeral working notebook and write the seed
        # hypothesis into it (framework-owned, reliable). The agent updates
        # Plan/Progress via the `plan` tool during the iteration; this seeds
        # the Hypothesis section so it always reflects what seed selection
        # chose — even when (not if) the agent forgets to write it down.
        # `state.lesson_recorded` starts False on the fresh AgentState above;
        # the iteration-end lesson-nudge flips it when the agent calls lesson().
        self._init_plan_md(hypothesis, hypotheses)

        self.agent.message_history.add_system(system_prompt)

        # Get starting commit (saved as the rollback anchor).
        # Final safety net: resolve away any meta_evolve commit sitting on HEAD
        # so the recorded parent (and commit_iteration's `reset --soft parent`)
        # never points at a meta commit. Any meta/switch commit wedged in
        # between gets squashed into the new main commit by commit_iteration.
        head = self.agent.git_controller.get_current_commit() or ""
        parent_commit = self.agent.archive_manager._resolve_non_meta_commit(head)
        self.agent.state.parent_commit = parent_commit

        # Cache baseline reward for this iteration (avoids per-step lookup)
        self._cached_baseline_reward = self._get_code_baseline_reward()

        self._log(
            f"Loaded {len(codes)} files, "
            f"{len(tool_names)} tools ({', '.join(tool_names)}), "
            f"shell: {shell_type}"
        )

        return parent_commit

    def _init_plan_md(self, hypothesis: str = "", hypotheses: list = None) -> None:
        """Clear ``plan.md`` and write the seed-selection Hypothesis section.

        Called at iteration start (``reset_for_iteration``). ``plan.md`` is the
        ephemeral working notebook: it is cleared each iteration, then the
        framework writes the hypothesis the seed-selection chose so the agent
        always has a reliable reference of what it set out to test. The agent
        owns the Plan/Progress sections (via the ``plan`` tool); the Hypothesis
        section is framework-owned and preserved across ``plan`` calls.

        - 2+ competing ``hypotheses`` → rendered as ### H1/H2/... under
          ``## Hypothesis``.
        - single ``hypothesis`` markdown → body placed under ``## Hypothesis``
          (a leading ``## ...`` header line is stripped to avoid nesting).
        - neither (iteration 1, or injection disabled) → plan.md left empty.

        Best-effort: a write failure logs a warning but never breaks the
        iteration — plan.md is a convenience, not a correctness dependency.
        """
        agent = self.agent
        plan_path = os.path.join(agent.agent_code_dir, "plan.md")
        hyps = hypotheses or []
        body = ""
        if len(hyps) >= 2:
            lines = []
            for h in hyps:
                hid = h.get("id", "?")
                hconf = h.get("confidence", 0.5)
                htext = h.get("hypothesis", "")
                hpred = h.get("prediction", "")
                hfals = h.get("falsification", "")
                lines.append(f"### {hid} (conf={hconf:.2f}): {htext}")
                if hpred:
                    lines.append(f"- **Predicts**: {hpred}")
                if hfals:
                    lines.append(f"- **Falsified if**: {hfals}")
                lines.append("")
            body = "\n".join(lines).strip()
        elif (hypothesis or "").strip():
            hyp_text = hypothesis.strip()
            # Strip a leading "## ..." header line so plan.md has a single
            # clean ## Hypothesis section (the seed-selection hypothesis
            # markdown otherwise starts with "## Seed Selection Hypothesis").
            nl = hyp_text.find("\n")
            if 0 < nl and hyp_text[:nl].strip().startswith("## "):
                hyp_text = hyp_text[nl:].strip()
            body = hyp_text

        content = f"## Hypothesis\n\n{body}\n" if body else ""
        if not content:
            return  # Nothing to write — iteration 1 or no seed hypothesis
        try:
            with open(plan_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            self._log(f"Warning: failed to init plan.md: {e}")

    def _extract_summary_metrics(self, eval_metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Extract scalar metrics from the full evaluation metrics for IterationSummary.metrics.

        IterationSummary.metrics has type Dict[str, float] and keeps scalar values only.

        Args:
            eval_metrics: The metrics dict returned by the external evaluator.

        Returns:
            Dict of scalar metrics, e.g. {"utility_rate": 0.8, "security_asr": 0.1, "overall_rate": 0.85}.
        """
        if not eval_metrics or not isinstance(eval_metrics, dict):
            return {}

        scalar_metrics: Dict[str, float] = {}

        # Pull top-level scalar fields directly
        for key in ("overall_rate", "overall_success_rate", "total_tasks", "total_success"):
            if key in eval_metrics and isinstance(eval_metrics[key], (int, float)):
                scalar_metrics[key] = float(eval_metrics[key])

        # Pull rate-style fields from nested structures
        if "avg_utility" in eval_metrics and isinstance(eval_metrics["avg_utility"], dict):
            rate = eval_metrics["avg_utility"].get("rate")
            if isinstance(rate, (int, float)):
                scalar_metrics["utility_rate"] = float(rate)

        if "avg_asr" in eval_metrics and isinstance(eval_metrics["avg_asr"], dict):
            asr = eval_metrics["avg_asr"].get("asr")
            if isinstance(asr, (int, float)):
                scalar_metrics["security_asr"] = float(asr)

        # Pull each dimension out of the reward dict (when reward is a dict)
        reward = eval_metrics.get("reward")
        if isinstance(reward, dict):
            for k, v in reward.items():
                if isinstance(v, (int, float)) and k != "scalar_reward":
                    scalar_metrics[f"reward_{k}"] = float(v)

        return scalar_metrics

    def generate_iteration_summary(self, success: bool = True, max_steps_reached: bool = False) -> IterationSummary:
        """
        Generate a summary of the current iteration.

        The agent usually provides its own summary via compact_context /
        end_evolution; that summary is used directly with no extra LLM call.
        An LLM is invoked only when:
        1. iteration_summary_text is empty (the agent did not provide a summary);
        2. max_steps auto-terminated and the agent did not call compact_context.

        Args:
            success: Whether the iteration succeeded.
            max_steps_reached: Whether the max-step limit was hit.

        Returns:
            An IterationSummary object.
        """
        modifications = self.agent.state.modifications_made
        reward = self.agent.state.reward

        # Use val-preferred best reward (consistent with commit_iteration logic)
        best_version = self._find_best_code_version()
        if best_version:
            _, best_reward, _ = best_version
            scalar_reward = reward_to_scalar(best_reward)
        else:
            scalar_reward = self.agent.state.get_scalar_reward()

        action_count = self.agent._actions_in_iteration
        agent_summary = self.agent.state.iteration_summary_text

        # Decide whether an LLM-generated summary is needed
        needs_llm = (
            not agent_summary
            or (max_steps_reached and self.agent.state.iteration_end_reason == "max_steps_limit")
        )

        if not needs_llm:
            # The agent already supplied a valid summary (compact_context / end_evolution)
            summary_text = agent_summary
            key_decisions = [f"Action: {a.action_type.value}" for a in self.agent.state.action_history]
        else:
            summary_text, key_decisions = self._generate_summary_via_llm(
                modifications=modifications,
                reward=reward,
                action_count=action_count,
                success=success,
                max_steps_reached=max_steps_reached,
            )

        # Extract scalar metrics from last evaluation for summary
        metrics = self._extract_summary_metrics(self.agent.state.last_evaluation_metrics)

        return IterationSummary(
            iteration=self.agent.iteration,
            reward=scalar_reward,
            metrics=metrics,
            summary_text=summary_text,
            modifications_count=len(modifications),
            key_decisions=key_decisions,
            commit_hash=self.agent.state.commit_hash if success else self.agent.state.parent_commit,
            success=success,
        )

    def _generate_summary_via_llm(
        self,
        modifications: list,
        reward,
        action_count: int,
        success: bool,
        max_steps_reached: bool,
    ) -> tuple:
        """Generate summary_text and key_decisions via LLM (only invoked when the agent did not provide a summary)."""
        action_history_str = [
            f"- {a.action_type.value}: {json.dumps(a.params, ensure_ascii=False)}"
            for a in self.agent.state.action_history
        ]

        from .evolution_prompt import get_summary_generation_prompt
        prompt = get_summary_generation_prompt(
            iteration=self.agent.iteration,
            modifications=modifications,
            reward=reward,
            action_count=action_count,
            step_count=self.agent._step_in_iteration,
            action_history=action_history_str,
            max_steps=self.agent.config.max_steps_per_iteration,
            success=success,
            max_steps_reached=max_steps_reached,
            agent_summary=self.agent.state.iteration_summary_text,
            end_reason=self.agent.state.iteration_end_reason,
        )

        pre_summary_len = len(self.agent.message_history.messages)
        try:
            # Append the summary prompt to the existing message_history (reuses the cache prefix)
            self.agent.message_history.add_user(prompt)

            # Pass the evolve tools to keep the cache prefix consistent (the react loop passes these same tools)
            evolve_tools = self.agent.get_tools(scope="evolve")
            response = self.agent.call_llm(
                messages=self.agent.message_history.get_messages_for_llm(),
                tools=evolve_tools,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned no text content (tool call or empty)")
            content = content.strip()

            # Append the assistant response to history (save_iteration_context needs the full Q&A pair)
            update_history_from_response(response, self.agent.message_history)

            # Try to extract JSON (handles markdown code-block wrapping)
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()

            result = json.loads(content)
            return (result.get("summary_text", ""), result.get("key_decisions", [])[:5])
        except Exception as e:
            self._log(f"Failed to generate summary via LLM: {e}, using rule-based fallback")
            while len(self.agent.message_history.messages) > pre_summary_len:
                self.agent.message_history.remove_last()
            if success:
                modified_files = self.agent.action_executor._modified_files
                eval_count = len(self.agent.state.evaluation_snapshots)
                if modified_files:
                    summary_text = (
                        f"Modified {len(modified_files)} file(s): "
                        f"{', '.join(sorted(modified_files)[:10])}. "
                        f"Ran {eval_count} evaluation(s)."
                    )
                elif eval_count > 0:
                    summary_text = f"Ran {eval_count} evaluation(s) without file modifications."
                else:
                    summary_text = "No file modifications or evaluations in this iteration."
                key_decisions = [f"Action: {a.action_type.value}" for a in self.agent.state.action_history]
            else:
                summary_text = self.agent.state.iteration_summary_text or f"FAILED: max actions reached ({action_count})"
                key_decisions = []
            return (summary_text, key_decisions)

    def save_iteration_context(self, summary: IterationSummary) -> None:
        """
        Save iteration context to the filesystem.

        Args:
            summary: Iteration summary.
        """
        if not self.agent.context_persistence:
            return

        # Save message history
        self.agent.context_persistence.save_message_history(self.agent.iteration, self.agent.message_history)

        # Load and update the evolution context
        from .state import EvolutionContext
        context = self.agent.context_persistence.load_evolution_context()
        context.add_summary(summary)

        # Save the evolution context and summaries
        self.agent.context_persistence.save_evolution_context(context)
        self.agent.context_persistence.save_summaries(context)

        self._log(f"Saved iteration {self.agent.iteration} context to {self.agent.context_persistence.context_dir}")

    def _ensure_lesson_recorded(self) -> None:
        """Nudge the agent to call ``lesson()`` if it didn't during the iteration.

        Lesson is the ONLY memory the next iteration (and seed selection)
        inherits. If the agent compacted/ended without recording one, append a
        short user prompt to the SAME conversation (cache-preserving) and run
        up to ``lesson_nudge_max_steps`` react steps so it records a verdict
        grounded in the hypothesis it tested (plan.md) and how the iteration
        went (traces + rewards already in-context).

        The ``lesson`` action handler sets ``state.lesson_recorded = True`` on
        fire, so the loop bails out as soon as the agent complies. No silent
        framework synthesis — if the nudge exhausts without a lesson, the
        iteration simply commits without a new lesson line (the previous
        iterations' lessons remain in BOOTSTRAP.md).

        ``lesson_nudge_max_steps=0`` skips the nudge entirely.
        """
        agent = self.agent
        if getattr(agent.state, "lesson_recorded", False):
            return

        max_steps = getattr(agent.config, "lesson_nudge_max_steps", 2)
        if max_steps <= 0:
            return

        log_phase_banner(
            agent,
            f"LESSON NUDGE (iter {agent.iteration})",
            info="  agent didn't call lesson() — nudging (ground it in plan.md + the iteration's outcome)",
            color=_C.BMA,
        )

        nudge = (
            "Before this iteration's code is committed, record ONE cross-iteration "
            "lesson so the next iteration and seed selection can learn from it.\n"
            "Call `lesson(lesson=\"...\", confidence=0.5)` with a self-contained "
            "verdict on the hypothesis you tested this iteration (re-read plan.md "
            "for the hypothesis) and how the iteration went: did the change hold or "
            "break, the root cause, the transferable takeaway. Set `confidence` "
            "honestly — 0.8+ if confirmed by trace evidence, 0.5 if plausible, "
            "0.2-0.4 if speculative. BOOTSTRAP.md is the only memory the next "
            "iteration inherits, so make the line stand alone (no 'see above')."
        )
        agent.message_history.add_user(nudge)

        try:
            for nudge_step in range(1, max_steps + 1):
                self._log(
                    f"\n  {_C.D}—{_C.RST} {_C.BMA}Lesson Nudge "
                    f"{nudge_step}/{max_steps}{_C.RST}"
                )
                self._execute_react_step(
                    lambda name, args: agent.execute_tool(name, args, scope="evolve")
                )
                if getattr(agent.state, "lesson_recorded", False):
                    self._log(
                        f"  {_C.BGR}✓ Lesson recorded{_C.RST}"
                    )
                    break
        except Exception as e:
            agent._log(f"  Lesson nudge react failed: {e}")

    def _finalize_commit_pick(self) -> Optional[List[Dict[str, str]]]:
        """Finalize the commit pool as a continuation of the SAME conversation.

        Replaces the old single-hash pick pattern. Tools never change →
        cache prefix is preserved across the entire iteration.

        Short-circuits when:
        - evolvable_commit_strategy is disabled → returns None (fallback)
        - 0 evaluation snapshots → returns None (no candidates)
        - ≤1 unique code hash → returns [{"code_hash": hash, "description": ""}] immediately

        Otherwise appends a nudge prompt to agent.message_history (same conversation!)
        and runs up to commit_nudge_max_steps _execute_react_step() calls.

        Returns:
            list of dicts with "code_hash" and "description" keys, empty list, or None to fall back to default ranking.
        """
        agent = self.agent

        snapshots = agent.state.evaluation_snapshots

        # ── Phase banner (before all short-circuits, so every path is visible) ──
        iter_label = f"for iter {agent.iteration}"
        evolvable = getattr(agent.config, 'evolvable_commit_strategy', False)
        if not evolvable:
            log_phase_banner(
                agent, f"COMMIT SELECTION ({iter_label})",
                info="  disabled (evolvable_commit_strategy=false) | fallback to max-reward ranking",
                color=_C.BMA,
            )
            return None

        if not snapshots:
            log_phase_banner(
                agent, f"COMMIT SELECTION ({iter_label})",
                info="  Candidates: 0 (no evaluations) | fallback to max-reward ranking",
                color=_C.BMA,
            )
            return None

        # Group by unique code_hash
        seen: Dict[str, bool] = {}
        for snap in snapshots:
            ch, _r, _m = _parse_snapshot(snap)
            seen[ch] = True

        unique_hashes = list(seen.keys())

        # ── Phase banner ──────────────────────────────────────────
        info_parts = [f"Candidates: {len(unique_hashes)}"]
        if len(unique_hashes) <= 1:
            info_parts.append("auto-pick (≤1 unique)")
        else:
            info_parts.append(f"nudge max {agent.config.commit_nudge_max_steps} steps")
        log_phase_banner(
            agent,
            f"COMMIT SELECTION ({iter_label})",
            info="  " + " | ".join(info_parts),
            color=_C.BMA,
        )

        # Short-circuit: ≤1 unique code hash
        if len(unique_hashes) <= 1:
            if unique_hashes:
                chosen = unique_hashes[0]
                agent._log(
                    f"  ✓ Single candidate {chosen[:12]} — auto-confirmed"
                )
                return [{"code_hash": chosen, "description": ""}]
            agent._log("  ✗ No candidates — falling back to default ranking")
            return None

        agent._log(
            f"  {len(unique_hashes)} unique candidates — "
            f"running LLM-driven nudge confirmation"
        )

        return self._finalize_commit_pick_using_module(seen)

    def _finalize_commit_pick_using_module(
        self, seen: Dict[str, bool]
    ) -> Optional[List[Dict[str, str]]]:
        """Load select_commit.py prompts, append nudge to SAME message_history, run
        up to commit_nudge_max_steps continue steps with identical tools (cache-preserving).

        Fallback chain on failure: last bookmark in seen → None (max-reward ranking).

        Returns:
            list of dicts with "code_hash" and "description" keys, or None for fallback.
        """
        agent = self.agent

        # Load select_commit module for prompt functions
        commit_module = self._load_evolution_module("select_commit")
        if commit_module is None:
            agent._log("  ✗ select_commit.py not found — falling back to default ranking")
            return None

        # Check for required evolvable functions
        if not hasattr(commit_module, 'get_commit_nudge_prompt'):
            agent._log("  ✗ get_commit_nudge_prompt not found — falling back to default ranking")
            return None

        try:
            commit_prompt = commit_module.get_commit_nudge_prompt(agent)
            candidate_table = self._build_candidate_table(
                agent.state.evaluation_snapshots
            )
        except Exception as e:
            agent._log(f"  ✗ Failed to build commit nudge prompts: {e}")
            return None

        # Format pick history (framework-owned, not evolvable)
        pick_history = self._format_pick_history()

        # Get max steps from config (0 = skip nudge entirely)
        max_steps = agent.config.commit_nudge_max_steps
        if max_steps <= 0:
            agent._log("  commit_nudge_max_steps=0 — falling back to last bookmark")
            return self._last_bookmark_in_seen(seen)

        # Build eval_data for the {eval_data} placeholder in the evolvable prompt.
        # Framework-owned content: candidate table + pick history + decision nudge.
        eval_data = candidate_table
        if pick_history:
            eval_data += f"\n\n{pick_history}"
        eval_data += (
            f"\n\n**Decision required:** Call `finalize_commit_pool(code_hashes=[...])` "
            f"with your chosen hashes from the candidate table above. Each entry must be an object "
            f"with `code_hash` and `description` (a one-line summary of what changed and why). "
            f"Select 2-5 versions representing different exploration directions. "
            f"List them in priority order; the first becomes the on-disk representative "
            f"meta-evolution inspects as this iteration's primary direction. "
            f"You have up to {max_steps} step(s) to inspect (read_file/bash) and decide."
        )

        # Build nudge user message — appended to the SAME message_history
        # (same conversation, same tools → cache prefix preserved).
        # The evolvable _COMMIT_NUDGE_PROMPT may use {eval_data} to position
        # where the framework-injected evidence + decision nudge appears.
        # Use str.replace() instead of .format() — evolvable prompts may
        # contain literal curly braces (e.g. JSON examples) that .format()
        # would misinterpret as placeholders, causing KeyError.
        if "{eval_data}" in commit_prompt:
            user_content = commit_prompt.replace("{eval_data}", eval_data)
        else:
            user_content = commit_prompt + "\n\n" + eval_data

        # Record message count before the nudge so we can extract and save
        # the commit-selection conversation separately.
        pre_nudge_len = len(agent.message_history.messages)

        agent.message_history.add_user(user_content)

        # Nudge react loop: up to max_steps, SAME evolve tools
        decision: Dict[str, Any] = {}
        done = [False]

        def nudge_tool_executor(name, args):
            if name == "finalize_commit_pool":
                decision.update(args or {})
                done[0] = True
                raw_hashes = (args or {}).get("code_hashes", [])
                reason = (args or {}).get("reason", "")
                # Display: handle both string and dict formats
                display_hashes = []
                for h in raw_hashes:
                    if isinstance(h, str):
                        display_hashes.append(h[:12])
                    elif isinstance(h, dict):
                        display_hashes.append(h.get("code_hash", "")[:12])
                return (
                    f"Pool finalized: {len(raw_hashes)} version(s) selected. "
                    + (f"Reason: {reason}" if reason else "")
                    + f"\nSelected hashes: {', '.join(display_hashes)}"
                )
            return agent.execute_tool(name, args, scope="evolve")

        # ── Nudge react loop ──────────────────────────────────────
        try:
            for nudge_step in range(1, max_steps + 1):
                # Step header
                self._log(
                    f"\n  {_C.D}—{_C.RST} {_C.BMA}Nudge {nudge_step}/{max_steps}{_C.RST}"
                )
                has_tools, tool_calls_made, tool_results = self._execute_react_step(
                    nudge_tool_executor
                )

                if done[0]:
                    raw_hashes = decision.get("code_hashes", [])
                    reason = decision.get("reason", "")
                    display_hashes = []
                    for h in raw_hashes:
                        if isinstance(h, str):
                            display_hashes.append(h[:12])
                        elif isinstance(h, dict):
                            display_hashes.append(h.get("code_hash", "")[:12])
                    hashes_str = ", ".join(display_hashes)
                    self._log(
                        f"  {_C.BGR}✓ Pool finalized: {len(raw_hashes)} version(s)"
                        + (f" ({reason})" if reason else "")
                        + f" — {hashes_str}{_C.RST}"
                    )
                    break

                # Append commit-specific status suffix to the last tool
                # result (the main-loop suffix from process_tool_calls is
                # already there).  This replaces the old per-step user
                # message so the agent perceives time pressure without an
                # extra round-trip.
                if tool_calls_made:
                    step_pct = int(nudge_step / max_steps * 100) if max_steps else 0
                    suffix = f"\n— [Commit Step {nudge_step}/{max_steps}]"
                    if step_pct >= 80:
                        suffix += (
                            " ⚠️ running low — call `finalize_commit_pool` now"
                        )
                    elif step_pct >= 50:
                        suffix += (
                            " ⏳ past halfway — converge; "
                            "call `finalize_commit_pool` soon"
                        )
                    msgs = agent.message_history.messages
                    for i in range(len(msgs) - 1, -1, -1):
                        if msgs[i].get("role") == "tool":
                            msgs[i]["content"] = str(msgs[i]["content"]) + suffix
                            self._log(f"  {suffix.strip()}")
                            break
        except Exception as e:
            agent._log(f"  ✗ Nudge react failed: {e}")
            self._save_commit_nudge_messages(pre_nudge_len)
            return None

        # ── Save commit-selection conversation ────────────────────
        self._save_commit_nudge_messages(pre_nudge_len)

        chosen_hashes = decision.get("code_hashes", [])

        # Normalize: accept both plain strings (backward compat) and {code_hash, description} objects
        normalized: List[Dict[str, str]] = []
        for item in chosen_hashes:
            if isinstance(item, str):
                normalized.append({"code_hash": item, "description": ""})
            elif isinstance(item, dict):
                normalized.append({
                    "code_hash": item.get("code_hash", ""),
                    "description": item.get("description", ""),
                })

        # Resolve possibly-truncated hashes to full code_hashes.
        resolved_hashes: List[Dict[str, str]] = []
        for entry in normalized:
            ch = entry["code_hash"]
            resolved = self._resolve_code_hash(ch, seen)
            if resolved:
                resolved_hashes.append({
                    "code_hash": resolved,
                    "description": entry["description"],
                })
            else:
                agent._log(f"  ✗ Invalid hash ({ch[:12]}) — skipped")

        if resolved_hashes:
            reason = decision.get("reason", "llm_pick")
            agent._log(
                f"  ✓ LLM finalized pool: {len(resolved_hashes)} version(s) ({reason})"
            )
            return resolved_hashes

        if chosen_hashes:
            agent._log(
                f"  ✗ No valid hashes in selection — "
                f"falling back to last bookmark"
            )
        else:
            agent._log(
                "  ✗ No decision made — falling back to last bookmark"
            )

        # Fallback: use the single last bookmark
        last = self._last_bookmark_in_seen(seen)
        return [{"code_hash": last, "description": ""}] if last else None

    def _resolve_code_hash(self, partial: str, seen: Dict[str, bool]) -> Optional[str]:
        """Resolve a (possibly truncated) code_hash to the full hash key in seen.

        The candidate table renders hashes as ch[:12] to save tokens; the agent
        naturally copies what it sees. This method maps back to the full hash
        stored in evaluation_snapshots so downstream exact-match checks pass.

        Returns:
            Full code_hash string if found, None otherwise.
        """
        if not partial:
            return None
        # Exact match first (agent may have the full hash from another source)
        if partial in seen:
            return partial
        # Prefix match: agent copied truncated hash from candidate table
        for full in seen:
            if full.startswith(partial):
                return full
        return None

    def _last_bookmark_in_seen(self, seen: Dict[str, bool]) -> Optional[str]:
        """Iterate commit_picks in reverse; return first whose code_hash is in seen.

        Returns None if no valid bookmarks exist — the caller (commit_iteration)
        will then fall through to _find_best_code_version() for max-reward ranking.
        """
        for pick in reversed(self.agent.state.commit_picks):
            ch = pick.get("code_hash", "")
            resolved = self._resolve_code_hash(ch, seen)
            if resolved:
                self._log(
                    f"  ✓ Last bookmark: {resolved[:12]} "
                    f"(step {pick.get('step', '?')})"
                )
                return resolved
        self._log("  ✗ No valid bookmarks — falling back to max-reward ranking")
        return None

    def _save_commit_nudge_messages(self, pre_nudge_len: int) -> None:
        """Extract nudge messages from agent.message_history and save to select_commit/.

        Args:
            pre_nudge_len: Number of messages in agent.message_history before the
                           commit nudge prompt was appended.
        """
        agent = self.agent
        persistence = getattr(agent, "context_persistence", None)
        if persistence is None:
            return
        nudge_msgs = agent.message_history.messages[pre_nudge_len:]
        if not nudge_msgs:
            return
        try:
            saved = persistence.save_phase_messages(
                "select_commit", agent.iteration, nudge_msgs
            )
            if saved:
                agent._log(
                    f"  Saved {len(nudge_msgs)} select_commit messages to "
                    f"{saved.parent.name}/{saved.name}"
                )
        except Exception as e:
            agent._log(f"  Warning: Failed to save select_commit messages: {e}")

    def _build_candidate_table(self, snapshots) -> str:
        """Group snapshots by code_hash, render as markdown table.

        Pure framework data pipeline — no evolvable hooks. The real strategic
        lever is _COMMIT_NUDGE_PROMPT in select_commit.py, not table layout.

        Returns:
            Formatted markdown string (table + guidance).
        """
        from collections import defaultdict

        by_hash = defaultdict(lambda: {"dev": [], "val": []})
        for snap in snapshots:
            ch, r, m = _parse_snapshot(snap)
            mode = "val" if m == "val" else "dev"
            by_hash[ch][mode].append(r)

        lines = ["## Evaluation Candidates\n"]
        lines.append("| # | Code Hash | N Dev | N Val | Best Dev | Best Val | Modified Files |")
        lines.append("|---|-----------|-------|-------|----------|----------|----------------|")

        for i, (ch, modes) in enumerate(by_hash.items()):
            best_dev = fmt_reward(max(modes["dev"])) if modes["dev"] else "-"
            best_val = fmt_reward(max(modes["val"])) if modes["val"] else "-"
            mf = _lookup_modified_files_for_hash(snapshots, ch)
            mf_str = ", ".join(mf[:4]) if mf else "-"
            lines.append(
                f"| {i+1} | `{ch[:12]}` | {len(modes['dev'])} | "
                f"{len(modes['val'])} | {best_dev} | {best_val} | {mf_str} |"
            )

        lines.append("")
        lines.append(
            "Select 2-5 versions for the commit pool. Prefer val-validated versions "
            "over dev-only ones, and choose versions representing DIFFERENT exploration "
            "directions. Use `read_file` or `bash` to inspect candidates, "
            "then call `finalize_commit_pool(code_hashes=[{\"code_hash\": \"...\", \"description\": \"...\"}, ...])` "
            "with your choices. Write a one-line description for each selected version "
            "explaining what changed and why it was chosen. "
            "List them in priority order; the first becomes the on-disk representative "
            "meta-evolution inspects as this iteration's primary direction."
        )

        return "\n".join(lines)

    def _format_pick_history(self) -> str:
        """Format the agent's bookmark history from agent.state.commit_picks.

        Framework-owned (not evolvable) — the bookmark table format is
        infrastructure, not strategy. select_commit.py only owns the nudge
        prompt text and candidate formatting.

        Returns:
            Formatted markdown string, or empty string if no bookmarks.
        """
        picks = self.agent.state.commit_picks
        if not picks:
            return ""

        lines = ["## Your Commit Pool (cumulative — each call adds a version)", ""]
        lines.append("| Step | Code Hash | Reason |")
        lines.append("|------|-----------|--------|")

        for p in picks:
            ch = p.get("code_hash", "")[:12]
            step = p.get("step", "?")
            reason = p.get("reason", "") or "-"
            lines.append(f"| {step} | `{ch}` | {reason} |")

        lines.append("")
        lines.append(
            "Your pool entries are listed in order. You will select which "
            "ones to commit from the candidate table above using `finalize_commit_pool`."
        )

        return "\n".join(lines)

    def commit_iteration(self, summary: IterationSummary = None, chosen_hashes: Optional[List[Dict[str, str]]] = None) -> None:
        """Commit iteration results as one or more pool commits.

        For each selected code_hash, restores the snapshot, squashes changes
        into a single commit on top of the parent, tags it as a pool entry,
        and records it in the tracker and knowledge graph.

        Dedup: if two selected hashes produce identical code, only the first
        is committed. HEAD auto-commit: if the agent's final working state
        is not already in the pool, it gets committed as an additional entry.
        """
        self.agent.phase = EvolutionPhase.COMMITTING
        agent = self.agent
        parent_commit = agent.state.parent_commit

        bootstrap_path, bootstrap_backup = self._backup_bootstrap(agent.agent_code_dir)

        # ── Build description map from chosen_hashes ───────────────────
        description_map: Dict[str, str] = {}
        if chosen_hashes:
            for entry in chosen_hashes:
                ch = entry.get("code_hash", "")
                desc = entry.get("description", "")
                if ch:
                    description_map[ch] = desc

        # ── Determine which hashes to commit ──────────────────────────
        hashes_to_commit: List[str] = []
        best_version = None

        if chosen_hashes:
            hashes_to_commit = [e.get("code_hash", "") for e in chosen_hashes if e.get("code_hash")]
            self._log(
                f"  Pool commit: {len(hashes_to_commit)} version(s) selected"
            )
        else:
            # Fallback: use the single best version
            best_version = self._find_best_code_version()
            if best_version:
                hashes_to_commit = [best_version[0]]

        if not hashes_to_commit:
            self._log("  No versions to commit — skipping")
            if bootstrap_backup is not None:
                self._restore_bootstrap(bootstrap_path, bootstrap_backup)
            self.agent.phase = EvolutionPhase.COMPLETED
            return

        # ── Dedup: compute code_hash for HEAD; merge duplicates ──────
        head_hash = agent.action_executor._compute_code_hash()
        seen_code_hashes: set = set()
        deduped_hashes = []
        for h in hashes_to_commit:
            if h in seen_code_hashes:
                self._log(f"  Dedup: skipping duplicate hash {h[:7]}")
                continue
            seen_code_hashes.add(h)
            deduped_hashes.append(h)

        total = len(deduped_hashes)
        self._log(f"  Committing {total} pool version(s) after dedup")

        # ── Generate summary text (shared across all pool commits) ───
        summary_text = ""
        modifications = agent.state.modifications_made
        modified_files_set = agent.action_executor._modified_files
        eval_count = len(agent.state.evaluation_snapshots)
        if modifications:
            mod_types = [m.get("operation", "unknown") for m in modifications]
            summary_text = f"Made {len(modifications)} modifications: {', '.join(set(mod_types))}."
        elif modified_files_set:
            summary_text = (
                f"Modified {len(modified_files_set)} file(s): "
                f"{', '.join(sorted(modified_files_set)[:10])}. "
                f"Ran {eval_count} evaluation(s)."
            )
        else:
            summary_text = "No modifications made in this iteration."

        # ── Commit each version ──────────────────────────────────────
        pool_commits_created: List[str] = []
        pool_entries: List[Dict[str, Any]] = []
        first_new_commit = None

        # Pre-compute shared data to avoid O(P*S) scans inside the loop
        reward_history_snapshot = self._build_enriched_reward_history()

        # Save KG to disk BEFORE commits so each pool commit snapshots the
        # current graph (nodes from previous iterations + INIT). Nodes for
        # THIS iteration's pool commits are added in-memory (skip_save=True)
        # and batch-saved AFTER all commits.
        kg = getattr(agent, "_knowledge_graph", None)
        if kg is not None:
            try:
                kg.save()
            except Exception as e:
                self._log(f"Warning: KG pre-commit save failed: {e}")

        for idx, ch in enumerate(deduped_hashes):
            # Restore code snapshot for this version
            self._log(
                f"  [{idx+1}/{total}] Committing pool entry: {ch[:7]}"
            )
            current_hash = agent.action_executor._compute_code_hash()
            if ch != current_hash:
                if not self._restore_code_snapshot(ch):
                    self._log(f"  Warning: Could not restore snapshot {ch[:7]}, skipping")
                    continue

            if bootstrap_backup is not None:
                self._restore_bootstrap(bootstrap_path, bootstrap_backup)

            # Stage everything
            agent.git_controller._run_git_command(["add", "-A"])

            # Also stage context files
            if agent.context_persistence:
                context_files = agent.context_persistence.get_context_files_for_commit()
                for f in context_files:
                    result = agent.git_controller._run_git_command(["add", f], check=False)
                    if result.returncode != 0 and result.stderr:
                        self._log(f"Warning: git add {f} failed: {result.stderr.strip()}")

            # Soft reset to parent — squash intermediate commits
            if parent_commit and ch != head_hash:
                # Only reset if this version came from a snapshot (not HEAD which
                # already has the intermediate commits squashed in the working tree)
                agent.git_controller._run_git_command(
                    ["reset", "--soft", parent_commit]
                )

            # Align reward
            aligned_reward, committed_eval_mode, committed_exec_errors = (
                self._align_reward_with_committed_code(ch)
            )
            if aligned_reward is None:
                aligned_reward = agent.state.reward
                committed_eval_mode = "dev"
                last_metrics = agent.state.last_evaluation_metrics
                committed_exec_errors = agent.action_executor._extract_execution_errors(
                    last_metrics
                ) if last_metrics else 0
            aligned_scalar = reward_to_scalar(aligned_reward)

            # Create commit
            pool_label = f" pool={idx+1}/{total}" if total > 1 else ""
            desc = description_map.get(ch, "")
            desc_suffix = f" — {desc}" if desc else ""
            commit_message = (
                f"iteration={agent.iteration}{pool_label} "
                f"reward={aligned_scalar:.4f} "
                f"actions={len(agent.state.action_history)}"
                f"{desc_suffix}"
            )
            new_commit = agent.git_controller.create_evolution_commit(
                iteration=agent.iteration,
                message=commit_message,
                files=None,
            )

            if new_commit:
                if first_new_commit is None:
                    first_new_commit = new_commit
                pool_commits_created.append(new_commit)

                # Tag as pool entry
                agent.git_controller._run_git_command(
                    ["tag", "-f", f"pool-iter{agent.iteration}-{idx+1}", new_commit],
                    check=False,
                )

                # Accumulate entry for the single record_pool_iteration call
                pool_entries.append({
                    "new_commit": new_commit,
                    "reward": aligned_scalar,
                    "timestamp": datetime.now().isoformat(),
                    "committed_code_reward": aligned_reward,
                    "committed_eval_mode": committed_eval_mode,
                    "execution_errors": committed_exec_errors,
                })

                # Knowledge graph node
                if getattr(agent, "_knowledge_graph", None) is not None:
                    try:
                        _kg_summary = (
                            summary.summary_text
                            if summary is not None and getattr(summary, "summary_text", None)
                            else (getattr(agent.state, "iteration_summary_text", "") or summary_text)
                        )
                        # Prepend description to summary_text for KG node readability
                        _kg_desc = description_map.get(ch, "")
                        if _kg_desc:
                            _kg_summary = f"{_kg_desc} | {_kg_summary}" if _kg_summary else _kg_desc
                        _kg_tags = _lookup_change_tags_for_hash(
                            agent.state.evaluation_snapshots, ch
                        ) or sorted({
                            str(m.get("operation", ""))
                            for m in agent.state.modifications_made
                            if m.get("operation")
                        })
                        _kg_modified = _lookup_modified_files_for_hash(
                            agent.state.evaluation_snapshots, ch
                        ) or list(agent.action_executor._modified_files)
                        agent._knowledge_graph.add_node(
                            iteration=agent.iteration,
                            git_hash=new_commit,
                            parent_hash=parent_commit or "",
                            reward=aligned_scalar,
                            eval_mode=committed_eval_mode,
                            summary_text=_kg_summary or "",
                            modified_files=_kg_modified,
                            change_tags=_kg_tags,
                            is_meta=False,
                            skip_save=True,  # batch: save once after loop
                        )
                    except Exception as e:
                        self._log(f"Warning: KG add_node failed: {e}")

                self._log(f"  [{idx+1}/{total}] ✓ {new_commit[:7]} (reward={aligned_scalar:.4f})")
            else:
                self._log(f"  [{idx+1}/{total}] ✗ Commit failed")

        # Record the iteration ONCE (one record, parallel-list pool entries)
        if pool_entries:
            try:
                agent.evolution_tracker.record_pool_iteration(
                    iteration=agent.iteration,
                    parent_commit=parent_commit,
                    pool_entries=pool_entries,
                    state_summary=agent.state.environment_summary,
                    action_count=len(agent.state.action_history),
                    shared_metadata={
                        "modifications_count": len(agent.state.modifications_made),
                        "modified_files": list(agent.action_executor._modified_files),
                        "summary_text": (
                            (summary.summary_text if summary and summary.summary_text else None)
                            or agent.state.iteration_summary_text
                            or summary_text
                        ),
                        "iteration_end_reason": agent.state.iteration_end_reason,
                        "success": True,
                        "max_steps_reached": agent.state.max_steps_reached,
                        "operation_type": "pool",
                        "reward_history": [
                            {"reward": r, "eval_mode": m, "code_hash": h}
                            for r, m, h in reward_history_snapshot
                        ],
                        "seed_info": (
                            {k: v for k, v in agent._current_seed_info.items()
                             if k in ("strategy_hint", "metadata")}
                            if getattr(agent, "_current_seed_info", None) else None
                        ),
                    },
                )
            except Exception as e:
                self._log(f"Warning: record_pool_iteration failed: {e}")

        # Batch save KG after all pool commits (avoids O(P) serializations)
        if getattr(agent, "_knowledge_graph", None) and pool_commits_created:
            try:
                agent._knowledge_graph.save()
            except Exception as e:
                self._log(f"Warning: KG batch save failed: {e}")

        # ── Update state ─────────────────────────────────────────────
        if first_new_commit:
            agent.state.commit_hash = first_new_commit
            agent.state.pool_commits = pool_commits_created
            if summary is not None:
                summary.commit_hash = first_new_commit

        # ── Restore FIRST pool version to working tree ─────────────────
        # The first entry in the agent's finalize_commit_pool order is its
        # declared primary direction for this iteration. It is left on disk
        # for the subsequent meta-evolution phase to inspect (meta can read
        # the harness but only edit evolution/). Using the first version —
        # not a greedy max-reward pick — gives meta a faithful picture of
        # the agent's intent. (seed-selection's reset --hard overwrites this
        # before the next iteration, so it does not affect the next seed;
        # it exists to inform meta-evolution.)
        if pool_commits_created and deduped_hashes:
            first_hash = deduped_hashes[0]
            if first_hash != agent.action_executor._compute_code_hash():
                self._restore_code_snapshot(first_hash)
            self._log(f"  Restored first pool version {first_hash[:7]} to working tree "
                      f"(for meta-evolve inspection)")

        self.agent.phase = EvolutionPhase.COMPLETED

    def _align_reward_with_committed_code(self, committed_hash: str = None):
        """Align reward with the snapshot that matches the committed code hash.

        Searches evaluation_snapshots for a matching code_hash. If found, returns
        the **best single-eval reward** over that hash's matching evals (per mode,
        val-preferred) — no cross-evaluate aggregation. The returned reward flows
        into ``record.reward`` → ``controller.get_best_version`` (seed +
        submit_best), so cross-iteration selection follows the same single-eval
        rule with no controller change: max-of-best-reward across iterations
        simply picks the version with the single highest eval.

        Falls back to (None, None, 0) if no snapshot matches the committed code
        (code modified after the last eval); the caller then uses state.reward.

        Args:
            committed_hash: Pre-computed code hash. If None, computes it from disk.

        Returns:
            Tuple of (best_reward, eval_mode, exec_errors) where eval_mode is
            "val" or "dev", and exec_errors is the max execution-error count
            observed across ALL evals of this committed code (any crash taints
            the version, regardless of which eval layer the reward came from).
            (None, None, 0) if no snapshot matches the committed code hash.
        """
        if committed_hash is None:
            committed_hash = self.agent.action_executor._compute_code_hash()
        snapshots = self.agent.state.evaluation_snapshots

        # Group matching snapshots by eval mode → take the best single eval per
        # mode (max). No cross-evaluate aggregation: the same code evaluated K
        # times is represented by its single highest reward.
        by_mode = {"val": [], "dev": []}
        committed_exec_errors = 0
        for snap in snapshots:
            code_hash, reward, mode = _parse_snapshot(snap)
            if code_hash != committed_hash:
                continue
            by_mode["val" if mode == "val" else "dev"].append(reward)
            # Veto signal: a crash on ANY eval of this code taints the version.
            committed_exec_errors = max(
                committed_exec_errors, _snapshot_execution_errors(snap)
            )

        # Val-preferred: use the val best-reward if any val eval exists, else dev.
        for mode in ("val", "dev"):
            scalars = [reward_to_scalar(r) for r in by_mode[mode]]
            if scalars:
                return max(scalars), mode, committed_exec_errors

        # No snapshot matches committed code hash — code was modified after last eval
        return None, None, 0

    def _get_code_baseline_reward(self):
        """Get the reward from the previous iteration's committed code.

        Reads committed_code_reward from the last iteration's tracker record.
        For iteration 1 this is the iteration-0 seed record — which carries a real
        dev reward only when the bootstrap init-eval ran (run_init_eval backfills
        it). Absent (resume / init_eval disabled / eval produced no reward) the
        seed's metadata is empty ⇒ returns None ⇒ baseline renders "—".

        Returns:
            The baseline reward (float, dict, or None)
        """
        if self.agent.iteration < 1:
            return None

        tracker = self.agent.evolution_tracker
        if not tracker:
            return None

        record = tracker.get_iteration(self.agent.iteration - 1)
        if not record or not record.metadata:
            return None

        reward = record.metadata.get("committed_code_reward")
        # Normalize: tracker stores rewards as lists (one per commit);
        # extract the first element for single-commit iterations.
        if isinstance(reward, list) and len(reward) > 0:
            reward = reward[0]
        return reward

    def _find_best_code_version(self):
        """Find the code version with the best val-preferred single-eval reward.

        Delegates to rank_versions_by_best_reward (the single source of truth,
        shared with select_commit) so the commit pick and the committed-reward
        alignment never disagree. The returned reward is the version's best
        single-eval reward, matching what _align_reward_with_committed_code will
        recompute for the same hash.

        Returns:
            (best_code_hash, best_reward, best_eval_mode) or None if no snapshots.
        """
        snapshots = self.agent.state.evaluation_snapshots
        code_snapshots = self.agent.state.code_snapshots
        if not snapshots:
            return None

        ranked = rank_versions_by_best_reward(snapshots, code_snapshots)
        if not ranked:
            return None
        best = ranked[0]
        return (best["code_hash"], best["best_reward"], best["mode"])

    def _restore_code_snapshot(self, code_hash: str) -> bool:
        """Write snapshot files back to disk for the given code_hash.

        Args:
            code_hash: The hash key in state.code_snapshots

        Returns:
            True if restored successfully, False otherwise.
        """
        snapshot = self.agent.state.code_snapshots.get(code_hash)
        if not snapshot:
            return False

        agent_code_dir = self.agent.agent_code_dir

        # Never touch evolution/ during a main-iteration best-version restore —
        # it's meta-evolve territory. Its .py isn't a "harness version" to
        # restore, and its .md (e.g. evolution_base_prompt.md) isn't in the
        # .py-only snapshot and would be deleted by the cleanup below, silently
        # dropping staged meta changes. Exclude evolution/ from both the
        # write-back and the cleanup sweep.
        snapshot = {rel: content for rel, content in snapshot.items()
                    if not rel.replace(os.sep, '/').startswith('evolution/')}

        bootstrap_path, bootstrap_backup = self._backup_bootstrap(agent_code_dir)

        try:
            # Create all needed directories first
            dirs_needed = set()
            for rel_path in snapshot:
                full_path = os.path.join(agent_code_dir, rel_path)
                parent = os.path.dirname(full_path)
                if parent:
                    dirs_needed.add(parent)
            for d in dirs_needed:
                os.makedirs(d, exist_ok=True)
            # Write files
            for rel_path, content in snapshot.items():
                full_path = os.path.join(agent_code_dir, rel_path)
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            # Clean up .py/.md files on disk that are not in the snapshot
            snapshot_rel_paths = set(snapshot.keys())
            for root, dirs, files in os.walk(agent_code_dir):
                dirs[:] = [d for d in dirs if not d.startswith('.')
                           and d != '__pycache__' and d != 'evolution']
                for fname in files:
                    # Skip harness-root metadata notebooks — they are memory,
                    # not the version being restored.
                    if fname in METADATA_FILES:
                        continue
                    if fname.endswith('.py') or fname.endswith('.md'):
                        full_path = os.path.join(root, fname)
                        rel = os.path.relpath(full_path, agent_code_dir)
                        # Normalize to forward slashes for consistency
                        rel = rel.replace(os.sep, '/')
                        if rel not in snapshot_rel_paths:
                            os.remove(full_path)
                            self._log(f"Removed extra file not in snapshot: {rel}")
            if bootstrap_backup is not None:
                self._restore_bootstrap(bootstrap_path, bootstrap_backup)
            return True
        except Exception as e:
            self._log(f"Warning: Failed to restore code snapshot {code_hash[:7]}: {e}")
            return False

    @staticmethod
    def _backup_bootstrap(agent_code_dir: str) -> tuple:
        """Read and return BOOTSTRAP.md content for later restore. Returns (path, content) or (path, None)."""
        bootstrap_path = os.path.join(agent_code_dir, "BOOTSTRAP.md")
        try:
            with open(bootstrap_path, 'r', encoding='utf-8') as f:
                return bootstrap_path, f.read()
        except (FileNotFoundError, OSError):
            return bootstrap_path, None

    @staticmethod
    def _restore_bootstrap(bootstrap_path: str, content: str) -> None:
        """Write BOOTSTRAP.md content back to disk."""
        try:
            with open(bootstrap_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception:
            pass

    def _build_enriched_reward_history(self):
        """Build reward history enriched with eval_mode and code_hash.

        Zips iteration_rewards with evaluation_snapshots (1:1 aligned).

        Returns:
            List of (reward, eval_mode, code_hash) tuples.
        """
        rewards = self.agent.state.iteration_rewards
        snapshots = self.agent.state.evaluation_snapshots

        enriched = []
        for i, reward in enumerate(rewards):
            if i < len(snapshots):
                code_hash, _, mode = _parse_snapshot(snapshots[i])
            else:
                mode = "dev"
                code_hash = ""
            enriched.append((reward, mode, code_hash))
        return enriched
