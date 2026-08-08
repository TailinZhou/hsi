"""
Meta-Evolve Helper — Auxiliary class for agent.meta_evolve().

Uses React Core as the core capability, following the Godel design principle:
  react + atomic tools = core capability
  meta_evolve is built from react
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List

from .actions.agent_action import _LESSON_LINE_RE, parse_sections
from .context_persistence import ContextPersistence
from .probe_agent import run_probe
from .state import (
    ActionType,
    AgentState,
    EvolutionPhase,
    IterationSummary,
    MessageHistory,
    fmt_reward,
    reward_to_scalar,
)
from .utils.log_format import _C
from .utils import log_format


class MetaEvolveHelper:
    """Helper for the meta_evolve() method on GodelAgent.

    Runs a mini react loop where the agent modifies files in the
    evolution/ subdirectory. Changes take effect from the next iteration.

    Like EvolveHelper is to agent.evolve(), this class is to
    agent.meta_evolve().
    """

    def __init__(self, agent):
        """
        Args:
            agent: Reference to the GodelAgent instance.
        """
        self.agent = agent
        self._meta_iteration = 0
        self._meta_action_history: List[Dict] = []
        self._meta_edits_need_validation: bool = False
        self._last_modifications_count = 0
        self._last_summary_text = ""
        self._pre_meta_commit: str = ""
        self.meta_context = ContextPersistence(
            agent.agent_code_dir,
            context_dir_name=".meta_evolution_context",
        )

    # -----------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------

    def run(self) -> str:
        """
        Run a meta-evolution phase after an iteration.

        Reuses iter_helper._execute_react_step() for each step,
        sharing the same model/temperature as the main evolution.

        Returns:
            "continue" or "end_evolution"
        """
        agent = self.agent
        agent.phase = EvolutionPhase.META_EVOLVING
        self._meta_iteration += 1
        self._meta_action_history = []
        self._meta_edits_need_validation = False

        # Save commit before meta-evolve for potential rollback
        self._pre_meta_commit = agent.git_controller.get_current_commit() or ""

        log_format.log_phase_banner(
            agent, f"META-EVOLVE after iter {agent.iteration}", color=_C.BMA
        )

        evo_dir = os.path.join(agent.agent_code_dir, "evolution")
        if not os.path.isdir(evo_dir):
            agent._log("No evolution/ directory, skipping meta-evolve")
            return "continue"

        # ── State isolation (Bug #7) ───────────────────────────────────────
        # _execute_react_step → process_tool_calls writes into agent.state
        # (action_history, modifications_made, reasoning_contents) and bumps
        # agent._actions_in_iteration. None of that belongs to the just-finished
        # main iteration. Swap in a throwaway AgentState + saved counters and
        # restore them in finally so the main state is untouched.
        saved_state = agent.state
        saved_executor_state = agent.action_executor.state
        saved_actions = agent._actions_in_iteration
        throwaway = AgentState(iteration=agent.iteration, goal=agent.goal)
        agent.state = throwaway
        # No codes arg → agent_codes cache is left untouched.
        agent.action_executor.set_state(throwaway)

        try:
            meta_messages = MessageHistory()

            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt()
            meta_messages.add_system(system_prompt)
            meta_messages.add_user(user_prompt)

            meta_tools = self._get_tools()
            tool_executor = self._execute_tool

            max_steps = agent.config.meta_evolve_max_steps
            for step in range(1, max_steps + 1):
                agent._log(f"  {_C.D}Meta-Evolve Step {step}/{max_steps}{_C.RST}")

                try:
                    has_tools, tool_calls_made, tool_results = agent.iter_helper._execute_react_step(
                        tool_executor=tool_executor,
                        messages=meta_messages,
                        tools=meta_tools,
                    )
                except Exception as e:
                    agent._log(f"  Meta-evolve react call failed: {e}")
                    break

                if not has_tools:
                    meta_messages.add_user(
                        "No tool calls made. Call end_meta_evolution to end this "
                        "meta-evolve phase, or continue with tools (read_file, edit_file, "
                        "write_file, bash)."
                    )
                    continue

                # Track action history for context saving
                for tc_info in tool_calls_made:
                    self._meta_action_history.append({
                        "name": tc_info["name"],
                        "args": tc_info.get("args", {}),
                    })

                # Meta-specific: check flow-control tools
                for tc_info in tool_calls_made:
                    tool_name = tc_info["name"]
                    if tool_name == ActionType.END_META_EVOLUTION.value:
                        if self._meta_edits_need_validation:
                            # Gate already returned BLOCKED via _execute_tool; break to
                            # let the step loop continue so the agent can validate.
                            break
                        self._save_context(meta_messages)
                        self._commit("agent requested end of meta-evolution")
                        self._reload_modules()
                        return "continue"

            self._save_context(meta_messages)
            self._commit()
            self._reload_modules()

            return "continue"
        finally:
            # Restore the main iteration's state and counters.
            agent.state = saved_state
            agent.action_executor.set_state(saved_executor_state)
            agent._actions_in_iteration = saved_actions

    # -----------------------------------------------------------------
    # Prompt builders
    # -----------------------------------------------------------------

    # Marker-wrapped prompt sections that are conditionally stripped, one per
    # evolvable dimension. Entry: (config flag, full-block regex, marker-only
    # regex). When the flag is False the whole block (markers + body) is removed;
    # when True only the marker comment lines are stripped so the body stays
    # visible. Regexes are pre-compiled once at class load.
    _STRIP_DIMENSIONS = [
        (
            "evolvable_commit_strategy",
            re.compile(r'# ┌── SELECT_COMMIT.*?# └── END SELECT_COMMIT ──\n*', re.DOTALL),
            re.compile(r'# [┌└]── (?:END )?SELECT_COMMIT[^\n]*\n?'),
        ),
    ]

    def _build_system_prompt(self) -> str:
        """Build system prompt for the meta-evolve phase — focused on select_*.py."""
        prompt_path = Path(__file__).parent / "meta_evolve_prompt.md"
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        for flag, full_re, markers_re in self._STRIP_DIMENSIONS:
            if not getattr(self.agent.config, flag):
                content = full_re.sub('', content)
            else:
                content = markers_re.sub('', content)

        return content

    def _build_user_prompt(self) -> str:
        """Build a compact user prompt — ~700-1200 tokens across 4 sections.

        Sections: Evolution History (compact table), Bootstrap History (last
        entry with prediction verification), Knowledge Graph (when nodes exist),
        Log Exploration Guide (format hints + jq examples).
        """
        agent = self.agent

        # ── Section 1: Compact Evolution History ──
        history_section = self._build_compact_history()

        # ── Section 2: Per-Dimension Summary (this iteration's decisions) ──
        dimension_section = self._build_per_dimension_summary()

        # ── Section 3: Bootstrap History ──
        bootstrap_section = self._build_bootstrap_section()

        # ── Section 4: Lesson Audit ──
        lesson_audit_section = self._build_lesson_audit_section()

        # ── Section 5: Knowledge Graph (only when nodes exist) ──
        graph_section = ""
        kg = getattr(agent, "_knowledge_graph", None)
        if kg is not None and kg.nodes:
            graph_section = "\n### Knowledge Graph\n\n" + kg.render_for_prompt() + "\n"

        # ── Section 6: Log Exploration Guide ──
        ctx_dir = os.path.join(agent.agent_code_dir, ".evolution_context")
        log_dir = os.path.join(ctx_dir, "main_evolve")
        log_section = self._build_log_guide(log_dir, ctx_dir)

        # ── Assemble ──
        return f"""## Iteration {agent.iteration} — Meta-Evolve
**Working Directory:** `{agent.agent_code_dir}`

{history_section}
{dimension_section}
{bootstrap_section}
{lesson_audit_section}
{graph_section}
{log_section}"""

    # -----------------------------------------------------------------
    # User-prompt section builders
    # -----------------------------------------------------------------

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        """Truncate text to max_len characters, adding '…' if truncated."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 1] + "…"

    def _build_per_dimension_summary(self) -> str:
        """Build a compact per-dimension summary of this iteration's phases.

        Surfaces the key decision + log path for main_evolve, select_seed,
        and select_commit so meta-evolve knows what happened in
        each dimension without digging through logs first.

        Returns:
            Markdown table string, or empty string if no iteration data exists.
        """
        agent = self.agent
        iteration = agent.iteration
        persistence = getattr(agent, "context_persistence", None)

        # ── Resolve log paths ──
        ctx_rel = ".evolution_context"

        def _log_path(phase: str) -> str:
            """Return a relative log path if the phase log exists, else '—'."""
            if persistence is None:
                return "—"
            iters = persistence.list_phase_iterations(phase)
            if phase == "main_evolve":
                # Check main message_history dir directly
                mh_dir = persistence.message_history_dir
                log_file = mh_dir / f"iter_{iteration}.json"
                if log_file.exists():
                    return f"`{ctx_rel}/{phase}/iter_{iteration}.json`"
                return "—"
            if iteration in iters:
                return f"`{ctx_rel}/{phase}/iter_{iteration}.json`"
            return "—"

        # ── main_evolve: latest tracker record ──
        main_line = ""
        tracker = agent.evolution_tracker
        if tracker and tracker.records:
            rec = tracker.get_iteration(iteration)
            if rec:
                reward_str = fmt_reward(rec.primary_reward())
                # committed_eval_mode is now a list (one per pool entry);
                # collapse to a single display value.
                modes = (rec.metadata or {}).get("committed_eval_mode") or []
                if isinstance(modes, list):
                    uniq_modes = sorted({m for m in modes if m})
                    mode = "/".join(uniq_modes) if uniq_modes else "?"
                else:
                    mode = modes or "?"
                mods = (rec.metadata or {}).get("modifications_count", 0)
                files = (rec.metadata or {}).get("modified_files") or []
                eval_count = len((rec.metadata or {}).get("reward_history") or [])
                summary = self._truncate(
                    (rec.metadata or {}).get("summary_text", ""), 80
                )
                main_line = (
                    f"| main_evolve | reward={reward_str}({mode}), "
                    f"{eval_count} eval(s), {mods} mod(s) "
                    f"({', '.join(files[:3])}{'…' if len(files) > 3 else ''}) | "
                    f"{_log_path('main_evolve')} |"
                )
        if not main_line:
            main_line = f"| main_evolve | (no data yet) | {_log_path('main_evolve')} |"

        # ── select_seed: agent._current_seed_info ──
        seed_line = ""
        seed_info = getattr(agent, "_current_seed_info", None) or {}
        if seed_info:
            shash = seed_info.get("git_hash", "")[:7]
            hint = seed_info.get("strategy_hint", "?")
            hypothesis = self._truncate(
                seed_info.get("hypothesis", ""), 60
            )
            seed_eval = seed_info.get("seed_eval_reward")
            if seed_eval is not None:
                seed_str = f"→ {shash} ({hint}, eval={seed_eval:.4f})"
            else:
                seed_str = f"→ {shash} ({hint})"
            if hypothesis:
                seed_str += f"\n  hypothesis: {hypothesis}"
            seed_line = (
                f"| select_seed | {seed_str} | {_log_path('select_seed')} |"
            )
        else:
            seed_line = f"| select_seed | (archive disabled or no seed info) | — |"

        # ── select_commit: latest tracker record + log existence ──
        commit_line = ""
        if tracker and tracker.records:
            rec = tracker.get_iteration(iteration)
            if rec:
                primary_commit = rec.primary_commit()
                commit_hash = primary_commit[:7] if primary_commit else "?"
                evolvable = getattr(agent.config, "evolvable_commit_strategy", False)
                if evolvable:
                    # Nudge ran — check if log exists
                    log_path = _log_path("select_commit")
                    if log_path != "—":
                        how = "nudge"
                    else:
                        how = "bookmark/fallback"
                else:
                    how = "max-reward"
                commit_line = (
                    f"| select_commit | → {commit_hash} ({how}) | "
                    f"{_log_path('select_commit')} |"
                )
        if not commit_line:
            commit_line = (
                f"| select_commit | (no data) | {_log_path('select_commit')} |"
            )

        # ── Assemble table ──
        return (
            "### This Iteration — Per-Dimension Summary\n\n"
            "| Dimension | Key Decision | Log |\n"
            "|-----------|-------------|-----|\n"
            f"{main_line}\n"
            f"{seed_line}\n"
            f"{commit_line}\n"
        )

    def _build_compact_history(self) -> str:
        """Build a compact iteration table from tracker records.

        Only includes non-meta main iterations since the last meta-evolve.
        Best iteration is marked with ★.
        """
        agent = self.agent
        records = agent.evolution_tracker.records if agent.evolution_tracker else []
        if not records:
            return "### Evolution History\n\n(No iterations recorded yet.)\n"

        last_meta = agent._last_meta_evolve_iteration

        # Single pass: collect visible rows + track best simultaneously
        rows = []
        best_iter = -1
        best_scalar = float("-inf")
        for r in records:
            if r.metadata.get("type") == "meta_evolve":
                continue
            s = reward_to_scalar(r.primary_reward())
            if s > best_scalar:
                best_scalar = s
                best_iter = r.iteration
            if r.iteration <= last_meta:
                continue
            mods = r.metadata.get("modifications_count", 0)
            summary = self._truncate(r.metadata.get("summary_text", ""), 60)
            strategy = r.metadata.get("strategy_hint", "")
            rows.append((r.iteration, fmt_reward(r.primary_reward()), strategy, mods, summary))

        if not rows:
            return "### Evolution History\n\n(No new iterations since last meta-evolve.)\n"

        header = (
            "### Evolution History (since last meta)\n\n"
            "| ★Iter | Reward | Strategy | Mods | Summary (60ch) |\n"
            "|-------|--------|----------|------|----------------|"
        )
        lines = []
        for (it, rw, st, md, sm) in rows:
            star = "★" if it == best_iter else " "
            lines.append(f"| {star}Iter {it} | {rw} | {st} | {md} | {sm} |")
        return header + "\n" + "\n".join(lines) + "\n"

    # Single regex extracting all bootstrap fields from the canonical format
    # (### What\n...\n### Why\n... etc. — written by meta_bootstrap in agent_action.py).
    _BOOTSTRAP_FIELDS_RE = re.compile(
        r"### What\s*\n(?P<what>.+?)\n\n"
        r"### Why\s*\n(?P<why>.+?)\n\n"
        r"### Lesson\s*\n(?P<lesson>.+?)\n\n"
        r"### Prediction\s*\n(?P<prediction>.+?)(?=\n## Meta #|\Z)",
        re.DOTALL,
    )

    def _build_bootstrap_section(self) -> str:
        """Parse meta_bootstrap.md and display the last entry.

        Shows What/Why/Lesson/Prediction from the most recent ## Meta #N
        entry, plus the actual reward since that bootstrap was written for
        the agent to compare.
        """
        agent = self.agent
        bootstrap_path = os.path.join(agent.agent_code_dir, "evolution", "meta_bootstrap.md")

        if not os.path.isfile(bootstrap_path):
            return "### Bootstrap History\n\n(No bootstrap entries yet.)\n"

        try:
            with open(bootstrap_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return "### Bootstrap History\n\n(Could not read meta_bootstrap.md)\n"

        # Split into ## Meta #N entries
        entries = re.split(r"\n(?=## Meta #\d+)", content)
        if not entries:
            return "### Bootstrap History\n\n(No `## Meta #N` entries found.)\n"

        last_entry = entries[-1].strip()
        if not last_entry.startswith("## Meta #"):
            return "### Bootstrap History\n\n(Last entry is not a `## Meta #N` block.)\n"

        # Count total entries
        meta_count = sum(1 for e in entries if e.strip().startswith("## Meta #"))

        # Extract all four fields in one regex pass
        m = self._BOOTSTRAP_FIELDS_RE.search(last_entry)
        what = m.group("what").strip() if m else "(not provided)"
        why = m.group("why").strip() if m else "(not provided)"
        lesson = m.group("lesson").strip() if m else "(not provided)"
        prediction = m.group("prediction").strip() if m else "(not provided)"

        # Show recent actual rewards for the agent to compare against prediction
        records = agent.evolution_tracker.records if agent.evolution_tracker else []
        recent_rewards = []
        for r in reversed(records):
            if r.metadata.get("type") != "meta_evolve":
                recent_rewards.append(f"Iter {r.iteration}: {fmt_reward(r.primary_reward())}")
            if len(recent_rewards) >= 3:
                break

        reward_line = ""
        if recent_rewards:
            reward_line = "**Recent actual rewards:** " + " | ".join(reversed(recent_rewards)) + "\n"

        # Extract split() out of the f-string: a backslash inside an f-string
        # expression part is a SyntaxError on Python < 3.12.
        latest_first_line = last_entry.split('\n')[0]
        return (
            "### Bootstrap History\n\n"
            f"Latest: {latest_first_line}\n"
            f"**What:** {what}\n"
            f"**Why:** {why}\n"
            f"**Lesson:** {lesson}\n"
            f"**Prediction:** {prediction}\n"
            f"{reward_line}"
            f"({meta_count} entries total)\n"
        )

    def _build_lesson_audit_section(self) -> str:
        """Build a Lesson Audit data block for the user prompt.

        Reads BOOTSTRAP.md → extracts ALL ``[Iter N|conf=X.XX]`` lines (full
        history, not just the current iteration). Reads plan.md → extracts
        ``## Hypothesis`` section.

        When no lesson exists for the current iteration, flags it as a red flag
        rather than saying "nothing to audit."
        """
        agent = self.agent
        iteration = agent.iteration

        # ── Read BOOTSTRAP.md → extract ALL [Iter N|conf=X.XX] lines ──
        bootstrap_path = os.path.join(agent.agent_code_dir, "BOOTSTRAP.md")
        all_lessons: List[str] = []
        current_lesson: Optional[str] = None
        if os.path.isfile(bootstrap_path):
            try:
                with open(bootstrap_path, "r", encoding="utf-8") as f:
                    content = f.read()
                for line in content.splitlines():
                    stripped = line.strip()
                    m = _LESSON_LINE_RE.match(stripped)
                    if m:
                        n = int(m.group(1))
                        all_lessons.append(stripped)
                        if n == iteration:
                            current_lesson = stripped
            except Exception:
                pass

        # ── Read plan.md → extract ## Hypothesis section ──
        plan_path = os.path.join(agent.agent_code_dir, "plan.md")
        hypothesis = None
        if os.path.isfile(plan_path):
            try:
                with open(plan_path, "r", encoding="utf-8") as f:
                    sections = parse_sections(f.read(), ["Hypothesis"])
                    hypothesis = (sections.get("hypothesis", "") or "").strip() or None
            except Exception:
                pass

        lines = ["### Lesson Audit\n"]

        # Full lesson history (all iterations)
        if all_lessons:
            lines.append("**All recorded lessons (full history):**")
            for entry in all_lessons:
                lines.append(f"- `{entry}`")
        else:
            lines.append("**All recorded lessons:** (none — BOOTSTRAP.md is empty or missing)")

        lines.append("")

        # Current iteration status
        if current_lesson:
            lines.append(f"**Current iteration ({iteration}) lesson:** `{current_lesson}`")
        else:
            lines.append(
                f"**Current iteration ({iteration}) lesson:** "
                f"⚠️ RED FLAG — the evolve agent did NOT record a lesson this iteration. "
                f"This may indicate the agent is stuck, gave up, or found nothing worth recording. "
                f"Investigate: did it form a hypothesis? Did it evaluate? Why no verdict?"
            )

        if hypothesis:
            lines.append(f"**Hypothesis (from plan.md):** {hypothesis}")

        return "\n".join(lines) + "\n"

    # Template for probe instructions — appended to the Log Exploration section so
    # the meta-evolve agent sees it every time it decides to use the probe tool.
    # Four required elements: phase, iteration range, ONE question, patterns.
    _PROBE_INSTRUCTION_TEMPLATE = (
        "\n### Probe Instructions Template\n\n"
        "When calling `probe(instructions=...)`, your instructions MUST contain:\n\n"
        "1. **Phase** — which log phase(s) to examine (main_evolve / select_seed / "
        "select_commit)\n"
        "2. **Iteration range** — which iterations (e.g. \"iter_1 through iter_5\" or "
        "\"all iterations\")\n"
        "3. **ONE question** — a single, focused diagnosis question. NOT a list of "
        "sub-questions.\n"
        "4. **Patterns to look for** — what specific signals answer the question\n\n"
        "**Good example** (specific, single question):\n"
        "\"Examine select_seed/ iter_1-5. Question: is the seed strategy picking "
        "versions that beat the previous best, or is it re-seeding the same plateau? "
        "Look for pick_seed arguments — what git hash is chosen, and is the reward "
        "trending up or flat?\"\n\n"
        "**Bad example** (too broad):\n"
        "\"Check all logs for patterns and tell me what's wrong.\"\n\n"
        "**Very bad example** (too vague):\n"
        "\"Look at logs.\"\n"
    )

    # Template for the Log Exploration section — {log_dir} is the main evolve
    # directory; {ctx_dir} is the .evolution_context root so phase subdirectories
    # can be referenced directly.
    _LOG_GUIDE_TEMPLATE = (
        "### Log Exploration\n\n"
        "Conversation logs (one per phase per iteration, JSON format "
        "`{{\"messages\": [{{\"role\": \"...\", \"content\": \"...\", \"tool_calls\": [...]}}]}}`):\n\n"
        "| Phase | Path |\n"
        "|-------|------|\n"
        "| main_evolve | `{ctx_dir}/main_evolve/iter_N.json` |\n"
        "| select_seed | `{ctx_dir}/select_seed/iter_N.json` |\n"
        "| select_commit | `{ctx_dir}/select_commit/iter_N.json` |\n\n"
        "**Never `read_file` an entire log** — use `bash` with `jq` instead.\n\n"
        "**Phase correspondence**: diagnose select_seed → look at its own log; "
        "diagnose select_commit → look at its own log; etc. Each dimension's "
        "decision-making is captured in its own conversation file.\n\n"
        "Useful `jq` queries (works on any of the above):\n"
        "```bash\n"
        "# List all tool calls in an iteration\n"
        "cat {log_dir}/iter_N.json | jq -r '.messages[] | select(.tool_calls) | .tool_calls[].function | \"\\(.name)(\\(.arguments[:120]))\"'\n\n"
        "# Find failed evaluations\n"
        "cat {log_dir}/iter_N.json | jq -r '.messages[] | select(.role==\"tool\") | select(.content | test(\"fail|error|0\\\\.0\"; \"i\")) | .content[:300]'\n\n"
        "# See evaluate results\n"
        "cat {log_dir}/iter_N.json | jq -r '.messages[] | select(.role==\"tool\") | select(.content | test(\"reward|passed|score\"; \"i\")) | .content[:500]'\n"
        "```"
    )

    @classmethod
    def _build_log_guide(cls, log_dir: str, ctx_dir: str = None) -> str:
        """Format the log exploration guide with the correct log directories.

        Args:
            log_dir: Path to the main evolve message_history/ directory.
            ctx_dir: Path to .evolution_context/ root (for phase subdirectory refs).
        """
        if ctx_dir is None:
            ctx_dir = os.path.dirname(log_dir)
        return cls._LOG_GUIDE_TEMPLATE.format(log_dir=log_dir, ctx_dir=ctx_dir) + cls._PROBE_INSTRUCTION_TEMPLATE

    # -----------------------------------------------------------------
    # Tools
    # -----------------------------------------------------------------

    def _get_tools(self) -> List[Dict]:
        """Build the OpenAI tools list for meta-evolve phase."""
        return self.agent.get_tools(scope="meta_evolve")

    def _execute_tool(self, tool_name: str, args: Dict) -> str:
        """Execute a tool in the meta-evolve context."""
        # probe: delegate investigation to a read-only sub-agent
        if tool_name == "probe":
            instructions = args.get("instructions", "")
            self.agent._log(f"  Spawning probe sub-agent...")
            return run_probe(self.agent, instructions, scope="meta_evolve")

        if tool_name == ActionType.END_META_EVOLUTION.value and self._meta_edits_need_validation:
            return (
                "BLOCKED: You modified files under evolution/ but haven't validated "
                "the changes since the last edit. Call validate_archive now to dry-run "
                "select_seed() and verify your changes. Only call end_meta_evolution "
                "after validate_archive has been called."
            )

        # bash/powershell can edit files via sed/redirect, bypassing
        # edit_file/write_file and thus the validate_archive gate. Snapshot two
        # signatures before execution:
        #   - evolution/ .py   -> if changed, require validate_archive afterwards
        #   - harness .py      -> meta must NEVER touch harness code; if changed,
        #                          revert the escape from the pre-meta commit so
        #                          it can't persist into the next main iteration
        is_shell = tool_name in (ActionType.BASH.value, ActionType.POWERSHELL.value)
        before_sig = self._evolution_py_signature() if is_shell else None
        before_harness_sig = self._harness_py_signature() if is_shell else None

        result = self.agent.execute_tool(tool_name, args, scope="meta_evolve")

        if tool_name in (ActionType.EDIT_FILE.value, ActionType.WRITE_FILE.value):
            path = args.get("path", "")
            if "evolution/" in path.replace("\\", "/"):
                self._meta_edits_need_validation = True
        elif tool_name == ActionType.VALIDATE_ARCHIVE.value:
            self._meta_edits_need_validation = False
        elif is_shell and before_sig is not None:
            # A bash edit to evolution/*.py slipped past edit_file/write_file.
            if self._evolution_py_signature() != before_sig:
                self._meta_edits_need_validation = True
            # A bash edit to harness code — a meta sandbox escape. Revert it.
            if before_harness_sig is not None:
                after_harness_sig = self._harness_py_signature()
                escaped = [
                    rel for rel in set(after_harness_sig) | set(before_harness_sig)
                    if after_harness_sig.get(rel) != before_harness_sig.get(rel)
                ]
                if escaped:
                    self._revert_harness_escape(escaped)

        return result

    @staticmethod
    def _collect_py_signatures(root_dir: str, base_dir: str,
                               skip_dirs: set = None) -> Dict[str, tuple]:
        """Stat-based change-sentinel for .py files under *root_dir*.

        Returns ``{rel_path: (mtime_ns, size)}`` where rel_path is relative to
        *base_dir*. Skips directories in *skip_dirs* and any starting with ".".
        """
        sig: Dict[str, tuple] = {}
        if not os.path.isdir(root_dir):
            return sig
        skip = skip_dirs or set()
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
            for f in files:
                if not f.endswith(".py"):
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, base_dir).replace("\\", "/")
                try:
                    st = os.stat(full)
                    sig[rel] = (st.st_mtime_ns, st.st_size)
                except OSError:
                    pass
        return sig

    def _evolution_py_signature(self) -> Dict[str, tuple]:
        """Stat-snapshot of every .py file under evolution/."""
        evo_dir = os.path.join(self.agent.agent_code_dir, "evolution")
        return self._collect_py_signatures(evo_dir, self.agent.agent_code_dir, {"__pycache__"})

    def _harness_py_signature(self) -> Dict[str, tuple]:
        """Stat-snapshot of .py files outside evolution/ (harness code)."""
        return self._collect_py_signatures(
            self.agent.agent_code_dir,
            self.agent.agent_code_dir,
            {"__pycache__", "evolution", ".evolution",
             ".meta_evolution_context", ".evolution_context"},
        )

    def _revert_harness_escape(self, escaped_files: List[str]) -> None:
        """Revert bash/powershell edits to harness files during meta-evolve.

        The meta sandbox only permits edits under ``evolution/``. A bash edit
        to harness code bypasses edit_file/write_file (and thus the
        validate_archive gate); left alone it stays in the working tree, gets
        read by the next iteration's ``load_codes()``, and is folded into that
        iteration's commit — silently persisting a forbidden change. We restore
        each escaped file from the pre-meta commit (the authoritative main
        baseline) so the escape can't leak forward. Files created by bash that
        don't exist in the baseline are removed.
        """
        agent = self.agent
        if not self._pre_meta_commit:
            agent._log("  Warning: cannot revert meta bash harness escape "
                       "(no pre-meta commit saved)")
            return
        for rel in escaped_files:
            try:
                content = agent.git_controller.get_file_at_commit(
                    rel, self._pre_meta_commit
                )
                full = os.path.join(agent.agent_code_dir, rel)
                if content is not None:
                    with open(full, "w", encoding="utf-8") as f:
                        f.write(content)
                elif os.path.exists(full):
                    # bash-created file absent from the baseline -> delete it
                    os.remove(full)
                agent._log(f"  {_C.YE}Reverted meta bash harness escape: "
                           f"{rel}{_C.RST}")
            except Exception as e:
                agent._log(f"  Warning: failed to revert harness escape "
                           f"{rel}: {e}")
        # Keep the action executor's modified-set honest about the revert so a
        # later meta commit / next-iteration reset doesn't list ghost edits.
        try:
            agent.action_executor._modified_files.difference_update(escaped_files)
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Context saving
    # -----------------------------------------------------------------

    def _save_context(self, meta_messages: MessageHistory) -> None:
        agent = self.agent

        modifications_count = sum(
            1 for a in self._meta_action_history
            if a.get("name") in ("edit_file", "write_file")
        )

        summary_text, key_decisions = self._generate_summary(modifications_count)

        summary = IterationSummary(
            iteration=self._meta_iteration,
            reward=0.0,
            metrics={},
            summary_text=summary_text,
            modifications_count=modifications_count,
            key_decisions=key_decisions,
            commit_hash="",
            success=True,
        )

        self.meta_context.save_message_history(self._meta_iteration, meta_messages)

        from .state import EvolutionContext
        context = self.meta_context.load_evolution_context()
        # The meta-evolve phase doesn't evaluate code and has no concept of
        # reward — disable reward-based best tracking to avoid misleading
        # best_reward=0.0 / best_iteration=0 entries in evolution_context.json
        # / summaries.json.
        context.track_best_reward = False
        context.add_summary(summary)
        self.meta_context.save_evolution_context(context)
        self.meta_context.save_summaries(context)

        self._last_summary_text = summary_text
        self._last_modifications_count = modifications_count

    def _generate_summary(self, modifications_count: int) -> tuple:
        """Generate summary_text and key_decisions via LLM, with lightweight fallback."""
        agent = self.agent
        fallback_text = (
            f"Meta-evolve phase after main iteration {agent.iteration}. "
            f"Modifications: {modifications_count}."
        )

        if modifications_count == 0:
            return fallback_text, []

        action_history_str = [
            f"- {a.get('name', 'unknown')}: {json.dumps(a.get('args', {}), ensure_ascii=False)}"
            for a in self._meta_action_history
        ]

        try:
            from .evolution_prompt import get_summary_generation_prompt
            from .utils.json_parser import fix_and_parse_json
            prompt = get_summary_generation_prompt(
                iteration=self._meta_iteration,
                modifications=[],
                reward=0.0,
                action_count=len(self._meta_action_history),
                step_count=0,
                action_history=action_history_str,
                max_steps=agent.config.meta_evolve_max_steps,
                success=True,
                max_steps_reached=False,
                agent_summary=f"Meta-evolve phase after main iteration {agent.iteration}",
                end_reason="meta_evolve",
            )

            response = agent.llm_client.chat.completions.create(
                model=agent.config.model,
                messages=[
                    {"role": "system", "content": "You are a concise code evolution analyst. Always respond with valid JSON only, no markdown formatting."},
                    {"role": "user", "content": prompt}
                ]
            )
            content = response.choices[0].message.content.strip()

            # Strip markdown code block wrapper if present
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()

            result = fix_and_parse_json(content)
            return result.get("summary_text", ""), result.get("key_decisions", [])
        except Exception as e:
            agent._log(f"  Meta-evolve summary generation failed: {e}")
            return fallback_text, []

    # -----------------------------------------------------------------
    # Reload & Commit
    # -----------------------------------------------------------------

    def _reload_modules(self) -> None:
        """Re-validate the archive after meta-evolve edits (no in-memory reload).

        Strategy modules and select_*.py are always loaded fresh from disk by
        ``_ensure_strategies_discovered`` / ``_load_evolution_module``
        (``importlib.util.spec_from_file_location``, not cached in sys.modules),
        so there is nothing to reload into memory — meta-evolve's edits are
        already on disk and will be picked up on the next access. We only reset
        the discovery flag (so the registry is rebuilt from the new disk state)
        and validate; on failure, roll back evolution/.
        """
        agent = self.agent
        evo_dir = os.path.join(agent.agent_code_dir, "evolution")
        if not os.path.isdir(evo_dir):
            return

        # Reset strategy discovery flag so next iteration re-scans strategies
        if hasattr(agent, 'archive_manager') and hasattr(agent.archive_manager, '_strategies_discovered'):
            agent.archive_manager._strategies_discovered = False
            agent._log("  Reset strategy discovery flag")

        # Validate the on-disk archive — rollback on failure
        if hasattr(agent, 'archive_manager'):
            validation = agent.archive_manager.validate_archive()
            if not validation["valid"]:
                agent._log(
                    f"  {_C.RD}Archive validation FAILED after meta-evolve: {validation['error']}{_C.RST}"
                )
                if validation.get("unknown_strategies"):
                    agent._log(
                        f"  Unknown strategies: {validation['unknown_strategies']}"
                    )
                self._rollback_evo_dir()
            else:
                agent._log(f"  {_C.GR}Archive validation passed{_C.RST}")

    def _rollback_evo_dir(self) -> None:
        """Roll back evolution/ (and its summaries) to the pre-meta commit.

        Restores both ``evolution/`` and ``.meta_evolution_context/`` so the
        rolled-back summaries match the rolled-back archive (no "summaries
        claim select_*.py changed but it was just reverted" contradiction).
        The result is staged but NOT committed: Layer B already parked HEAD on
        the last main commit, so the good files persist as staged changes and
        get folded into the next main iteration's commit (same fold semantics
        as the meta changes themselves).
        """
        agent = self.agent
        if not self._pre_meta_commit:
            agent._log("  Warning: no pre-meta commit saved, cannot rollback")
            return

        try:
            agent.git_controller._run_git_command(
                ["checkout", self._pre_meta_commit, "--", "evolution/"]
            )
            # Restore summaries to match the rolled-back archive.
            agent.git_controller._run_git_command(
                ["checkout", self._pre_meta_commit, "--", ".meta_evolution_context/"],
                check=False,
            )
            agent.git_controller._run_git_command(
                ["add", "evolution/", ".meta_evolution_context/"]
            )
            agent._log(
                f"  {_C.YE}Rolled back evolution/ + summaries to pre-meta commit "
                f"{self._pre_meta_commit[:7]} (staged, no rollback commit){_C.RST}"
            )

            # Reset discovery flag — the rolled-back disk files will be loaded
            # fresh on the next strategy access (no in-memory reload needed).
            if hasattr(agent, 'archive_manager'):
                agent.archive_manager._strategies_discovered = False
        except Exception as e:
            agent._log(f"  Warning: rollback of evolution/ failed: {e}")

    def _commit(self, reason: str = "") -> None:
        """Commit meta-evolve changes; HEAD always returns to the main baseline.

        The agent is free to use git during meta-evolve (including ``git commit``
        via bash). Any such intermediate commits become **dangling** — they never
        enter the main lineage — because we reset HEAD back to ``_pre_meta_commit``
        (the main commit that existed before meta-evolve started), keeping all
        meta changes staged in the index. The next main iteration's
        ``commit_iteration`` (``git add -A`` + ``reset --soft parent``) folds them
        into the new main commit, so meta work is never wasted.
        """
        agent = self.agent
        agent.phase = EvolutionPhase.META_COMMITTING

        try:
            evo_dir = os.path.join(agent.agent_code_dir, "evolution")
            if not os.path.isdir(evo_dir):
                return

            # The authoritative main baseline: the commit that was HEAD when
            # meta-evolve started (saved in run()). Using this — NOT
            # get_current_commit(), which the agent may have advanced via bash
            # `git commit` — guarantees HEAD returns to the real main commit.
            if self._pre_meta_commit:
                main_baseline = self._pre_meta_commit
            else:
                main_baseline = agent.git_controller.get_current_commit() or ""
                agent._log("  Warning: _pre_meta_commit not saved; falling back to "
                           "current HEAD as meta baseline (main-line cleanliness "
                           "not guaranteed if the agent committed mid-meta).")

            commit_msg = f"[Meta-Evolve iter={agent.iteration}]"
            if reason:
                commit_msg += f" {reason}"

            # Create/record the meta commit (for the tracker / evolution graph).
            # If the agent already committed evolution/, this may be a no-op
            # returning the current HEAD; either way we record parent=baseline.
            meta_commit = agent.git_controller.create_evolution_commit(
                iteration=agent.iteration,
                message=commit_msg,
                files=["evolution/", ".meta_evolution_context/"],
            )

            if meta_commit and meta_commit != main_baseline:
                agent._log(f"  Meta-evolve recorded: {meta_commit[:7]} "
                          f"(baseline {main_baseline[:7]})")
                agent.evolution_tracker.record_iteration(
                    iteration=self._meta_iteration,
                    parent_commit=main_baseline,
                    new_commit=meta_commit,
                    reward=0.0,
                    state_summary="meta_evolve",
                    action_count=len(self._meta_action_history),
                    metadata={
                        "type": "meta_evolve",
                        "main_iteration": agent.iteration,
                        "modifications_count": self._last_modifications_count,
                        "summary_text": self._last_summary_text,
                    },
                )

                # Pin a ref on the meta commit. _commit() below resets HEAD back
                # to the main baseline (by design — meta changes fold into the
                # next main commit), which leaves this commit dangling and
                # eventually garbage-collectable. The ref keeps it durable and
                # discoverable via `git log --all` without polluting refs/heads.
                if agent.git_controller.create_meta_ref(agent.iteration, meta_commit):
                    agent._log(f"  Pinned ref refs/meta_evolve/iter-"
                              f"{agent.iteration} -> {meta_commit[:7]}")

                # Backfill the just-saved meta summary with the real commit hash.
                # _save_context runs before _commit in run(), so it persisted an
                # empty commit_hash; patch the matching summary in place here.
                try:
                    _ctx = self.meta_context.load_evolution_context()
                    # Stay consistent with _save_context: meta context does not
                    # track reward-based best.
                    _ctx.track_best_reward = False
                    if _ctx.iteration_summaries:
                        _last = _ctx.iteration_summaries[-1]
                        if _last.iteration == self._meta_iteration:
                            _last.commit_hash = meta_commit
                            self.meta_context.save_evolution_context(_ctx)
                            self.meta_context.save_summaries(_ctx)
                except Exception:
                    pass
            else:
                agent._log("  Meta-evolve produced no new commit (nothing to record)")

            # Unconditionally restore HEAD to the main baseline so the next
            # iteration never seeds from a meta/agent commit. reset --soft keeps
            # the index, so all meta changes (evolution/ + context, plus any
            # bash-staged edits) stay staged and get folded in by commit_iteration.
            current = agent.git_controller.get_current_commit() or ""
            if main_baseline and current != main_baseline:
                agent.git_controller._run_git_command(
                    ["reset", "--soft", main_baseline], check=False
                )
                agent._log(f"  Reset HEAD to main baseline {main_baseline[:7]} "
                          f"(meta changes staged for fold-in)")
        finally:
            agent.phase = EvolutionPhase.EVOLVING
