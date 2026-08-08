"""Balrog log summary — formats evaluation results for LLM analysis."""

import random
from typing import Any, Dict, List, Optional

from benchmark.evaluators.log_summary_base import LogSummaryBase

# Shared note explaining summary numbers are LCB (not raw means).
# Used by build_eval_summary_prompt and format_precomputed_summaries.
_LCB_NOTE_LINES = [
    "Note: task/env/reward numbers below are LCB (lower-confidence bound =",
    "mean − z·std/√n), not raw means — a single lucky episode doesn't inflate them.",
    "Per-episode `progression` is the raw single-episode value (the E_LCB input).",
]


class BalrogLogSummary(LogSummaryBase):
    """Log summary handler for Balrog benchmark.

    Formats per-task/per-env progression, failed episode details,
    and action analysis for LLM consumption.
    """

    @staticmethod
    def _task_lcb(meta: Dict[str, Any]) -> float:
        """Per-task progression, preferring the E_LCB field with a raw fallback."""
        return meta.get("lcb_progression", meta.get("avg_progression", 0))

    @staticmethod
    def _env_lcb(env_data: Dict[str, Any]) -> float:
        """Per-env mean progression, preferring the LCB field with a raw fallback."""
        return env_data.get("avg_lcb_progression", env_data.get("avg_progression", 0))

    def _append_adaptive_threshold_info(
        self, lines: list, metrics: Dict[str, Any], bold: bool = False,
    ) -> None:
        """Append compact adaptive threshold summary and per-task detail table."""
        at_line = self.format_adaptive_threshold(metrics, bold=bold)
        if at_line:
            lines.append(at_line)
            lines.append("")
        at_detail = self.format_adaptive_threshold_detail(metrics)
        if at_detail:
            lines.append(at_detail)
            lines.append("")

    def build_eval_summary_prompt(self, metrics: Dict[str, Any], max_passed_samples: int = 10) -> str:
        """Build a detailed evaluation summary prompt for LLM analysis."""
        task_results = metrics.get("task_results", [])
        reward = metrics.get("reward", 0.0)
        by_env = metrics.get("by_environment", {})

        # Separate passed (progression > 0) and failed tasks
        passed = [r for r in task_results if r.get("success", False)]
        failed = [r for r in task_results if not r.get("success", True)]

        lines = [
            "# Balrog Benchmark Evaluation Summary",
            f"**Overall Reward (LCB)**: {reward:.4f}",
            f"**Total Tasks**: {len(task_results)} | Passed: {len(passed)} | Failed: {len(failed)}",
            "",
            *_LCB_NOTE_LINES,
            "",
        ]

        # Adaptive threshold context
        self._append_adaptive_threshold_info(lines, metrics, bold=True)

        # Per-environment breakdown
        if by_env:
            lines.append("## Per-Environment Results")
            for env_name, env_data in by_env.items():
                lcb_prog = self._env_lcb(env_data)
                num_tasks = env_data.get("num_tasks", 0)
                progs = env_data.get("task_lcb_progressions", env_data.get("task_progressions", []))
                lines.append(f"- **{env_name}**: lcb_progression={lcb_prog:.4f} ({num_tasks} tasks, scores={[f'{p:.3f}' for p in progs]})")
            lines.append("")

        # Failed tasks detail
        if failed:
            lines.append(f"## Failed Tasks (progression < threshold)")
            for r in failed[:20]:  # Limit to 20
                meta = r.get("metadata", {})
                env_name = meta.get("env_name", "unknown")
                task_name = meta.get("task_name", "unknown")
                error = meta.get("execution_error")
                lines.append(f"- [{env_name}] {task_name}")
                if error:
                    lines.append(f"  Error: {error}")
                # Episode details
                ep_logs = r.get("interaction_log", [])
                for ep in ep_logs[:3]:
                    if "error" in ep:
                        lines.append(f"  Episode {ep.get('episode_idx', '?')}: ERROR - {ep['error']}")
                    else:
                        lines.append(
                            f"  Episode {ep.get('episode_idx', '?')}: "
                            f"steps={ep.get('num_steps', 0)}, "
                            f"progression={ep.get('progression', 0):.3f}, "
                            f"failed_actions={len(ep.get('failed_candidates', []))}"
                        )
                        # Show step traces for all steps
                        traces = ep.get("step_traces", [])
                        if traces:
                            lines.append(f"    All {len(traces)} step traces:")
                            for t in traces:
                                obs_preview = t.get("observation", "")
                                resp = t.get("agent_response", "")
                                valid = t.get("valid_action", "")
                                is_valid = t.get("is_valid", True)
                                rw = t.get("reward", 0)
                                step_n = t.get("step", "?")
                                lines.append(f"    Step {step_n}: reward={rw}")
                                lines.append(f"      Observation: {obs_preview}")
                                lines.append(f"      Agent said: {resp}")
                                if not is_valid:
                                    lines.append(f"      [INVALID] corrected to: {valid}")
            lines.append("")

        # Sample of passed tasks
        if passed:
            sample = random.sample(passed, min(max_passed_samples, len(passed)))
            lines.append("## Sample of Passed Tasks")
            for r in sample:
                meta = r.get("metadata", {})
                env_name = meta.get("env_name", "unknown")
                lcb_prog = self._task_lcb(meta)
                lines.append(f"- [{env_name}] lcb_progression={lcb_prog:.4f}")
            lines.append("")

        # Analysis questions
        lines.extend([
            "## Analysis Questions",
            "1. Which environments have the lowest progression? What patterns explain the failures?",
            "2. Are there common invalid action patterns? Check failed_candidates in episode logs.",
            "3. Is the system prompt giving adequate guidance for each game type?",
            "4. Should the action parsing be more robust?",
            "5. Would per-environment prompts improve performance?",
        ])

        return "\n".join(lines)

    def format_failed_tasks_only(self, reward: float | dict, metrics: Dict[str, Any]) -> str:
        """Format concise failed tasks summary with full step traces."""
        task_results = metrics.get("task_results", [])
        by_env = metrics.get("by_environment", {})
        failed = [r for r in task_results if not r.get("success", True)]

        lines = [f"Reward (LCB): {reward:.4f}", f"Failed: {len(failed)}/{len(task_results)} tasks"]

        # Adaptive threshold context
        at_line = self.format_adaptive_threshold(metrics)
        if at_line:
            lines.append(at_line)
        at_detail = self.format_adaptive_threshold_detail(metrics)
        if at_detail:
            lines.append("")
            lines.append(at_detail)

        if by_env:
            for env_name, env_data in by_env.items():
                lines.append(f"  {env_name}: lcb_prog={self._env_lcb(env_data):.4f}")

        if failed:
            lines.append("Failed tasks:")
            for r in failed[:10]:
                meta = r.get("metadata", {})
                lines.append(f"  [{meta.get('env_name', '?')}] {meta.get('task_name', '?')}")
                # Show full step traces for each episode
                ep_logs = r.get("interaction_log", [])
                for ep in ep_logs[:2]:
                    if "error" in ep:
                        lines.append(f"    ep{ep.get('episode_idx', '?')}: ERROR - {ep['error']}")
                        continue
                    traces = ep.get("step_traces", [])
                    total_steps = len(traces)
                    invalid_count = sum(1 for t in traces if not t.get("is_valid", True))
                    lines.append(
                        f"    ep{ep.get('episode_idx', '?')}: "
                        f"{invalid_count}/{total_steps} invalid actions, "
                        f"progression={ep.get('progression', 0):.3f}"
                    )
                    for t in traces:
                        step_n = t.get("step", "?")
                        obs_preview = t.get("observation", "")
                        resp = t.get("agent_response", "")
                        valid = t.get("valid_action", "")
                        is_valid = t.get("is_valid", True)
                        rw = t.get("reward", 0)
                        lines.append(f"      Step {step_n}:")
                        lines.append(f"        Observation: {obs_preview}")
                        if is_valid:
                            lines.append(f"        Agent action: '{resp}'")
                        else:
                            lines.append(
                                f"        Agent action: '{resp}' → "
                                f"corrected: '{valid}' [INVALID]"
                            )
                        lines.append(f"        Reward: {rw}")

        return "\n".join(lines)

    def build_probe_seeds(
        self, metrics: Dict[str, Any], condensed_file: Optional[str]
    ) -> List[str]:
        """1 probe seed targeting this eval's worst (failed / lowest
        avg_progression) task.

        Points probe at the condensed log's per-episode ``episode_summary`` +
        ``action_frequency`` — Balrog's richest first-person diagnostic fields.
        Returns [] when nothing failed (no failure to drill).
        """
        if not condensed_file or not isinstance(metrics, dict):
            return []
        scored = []
        for t in metrics.get("task_results", []):
            if not isinstance(t, dict) or not t.get("task_id"):
                continue
            md = t.get("metadata", {}) or {}
            prog = md.get("avg_progression")
            try:
                prog_f = float(prog) if prog is not None else None
            except (TypeError, ValueError):
                prog_f = None
            il = t.get("interaction_log") or []
            steps = il[0].get("num_steps", "") if il and isinstance(il[0], dict) else ""
            scored.append((t["task_id"], prog_f, steps, bool(t.get("success"))))
        if not scored or all(ok for *_, ok in scored):
            return []  # nothing failed → no failure-drill seeds
        # Failures first, then lowest progression (None treated as worst tiebreak).
        scored.sort(key=lambda r: (0 if not r[3] else 1, r[1] if r[1] is not None else 1.0))
        seeds: List[str] = []
        for tid, prog, steps, ok in scored[:1]:
            bits = []
            if prog is not None:
                bits.append(f"progression {prog:.3f}")
            if steps != "":
                bits.append(f"{steps} steps")
            tail = f" ({', '.join(bits)})" if bits else ""
            head = f"{condensed_file}: task '{tid}' {'failed' if not ok else 'passed'}{tail}."
            if not ok:
                seeds.append(
                    f"{head} Read its episode_summary + action_frequency to find "
                    f"where/why the harness got stuck, and report the concrete failure "
                    f"mode + the harness change that would fix it."
                )
            else:
                seeds.append(
                    f"{head} Read its episode_summary to confirm what made the harness "
                    f"succeed, in case that mechanism generalizes to the failing tasks."
                )
        return seeds

    def build_log_schema_description(self) -> str:
        """Balrog condensed-log schema for the probe sub-agent.

        Injected into the probe system prompt via run_probe(log_schema=...).
        Describes the ACTUAL condensed-log shape (no step_traces — those live
        only in the per-task files) plus the diagnostic field semantics the
        probe needs to read a failure. The evolve agent never sees this — it
        delegates to probe instead of reading logs itself.
        """
        return (
            "## Balrog Log Schema\n"
            "Two files per eval share a shape:\n"
            "- `*_condensed.json` — READ FIRST. Small; no step_traces.\n"
            "- `*_task_<task_id>.json` — same shape, but each episode ALSO has\n"
            "  `step_traces[]` (per-step observation / agent_response / valid_action\n"
            "  / is_valid / reward). Open one only for step-level detail.\n"
            "\n"
            "Top-level (condensed): `reward` (E_LCB progression), `avg_raw_progression`\n"
            "(raw mean, pre-LCB), `total_tasks`, `total_episodes`, `execution_errors`\n"
            "(>0 means some task CRASHED via a harness exception), `by_environment`,\n"
            "and `task_results[]` = `{ task_id, success, metadata{env_name, task_name,\n"
            "avg_progression, total_episodes}, interaction_log[EPISODE] }`.\n"
            "\n"
            "EPISODE is the unit of diagnosis:\n"
            "- `action_frequency: {<action>: count}` — what the agent ACTUALLY did. One\n"
            "  action dominating (e.g. `\"go north\": 20`) means it was stuck in a loop.\n"
            "- `failed_candidates: [str]` — actions the harness rejected/corrected\n"
            "  (INVALID actions).\n"
            "- `done: bool` — `false` means the episode TIMED OUT without finishing.\n"
            "- `target_plan: str | null` — the intended solution (some envs only;\n"
            "  absent in textworld). Compare against action_frequency to see whether\n"
            "  the agent even attempted the right strategy.\n"
            "- `episode_summary: str | null` — first-person diary the harness wrote\n"
            "  about what hurt. Usually the FASTEST signal — read it first.\n"
            "\n"
            "jq recipes (start from the condensed file):\n"
            "```bash\n"
            "# Worst tasks first\n"
            "cat *_condensed.json | jq -r '.task_results | sort_by(.metadata.avg_progression) | .[] | \"\\(.task_id): \\(.metadata.avg_progression)\"'\n"
            "\n"
            "# First-person diary of every episode that has one\n"
            "cat *_condensed.json | jq -r '.task_results[] | .task_id as $t | .interaction_log[] | select(.episode_summary) | \"== \\($t) ep\\(.episode_idx) ==\\n\\(.episode_summary)\"'\n"
            "```"
        )

    def build_consolidation_input(self, metrics: Dict[str, Any]) -> str:
        """Build consolidation input with per-environment grouping for Balrog.

        Delegates per-task formatting to build_consolidation_input_for_task.
        """
        parts = []
        for r in self.get_multi_episode_task_results(metrics):
            parts.append(self.build_consolidation_input_for_task(r))
        return "\n\n".join(parts)

    def _task_header(self, task_result: dict, *, short: bool = False) -> str:
        """Build the `### [env_name/task_name] lcb=...` header for a task.

        Shared by ``build_consolidation_input_for_task`` (long form:
        ``lcb_progression=…``, ``episodes``) and ``format_task_consolidation_output``
        (short form: ``lcb=…``, ``eps``).
        """
        meta = task_result.get("metadata", {})
        env_name = meta.get("env_name", "unknown")
        task_name = meta.get("task_name", task_result.get("task_id", "unknown"))
        lcb_prog = self._task_lcb(meta)
        success = task_result.get("success", False)
        num_eps = len(task_result.get("interaction_log", []))
        if short:
            return (
                f"### [{env_name}/{task_name}] lcb={lcb_prog:.2f} "
                f"({'PASS' if success else 'FAIL'}, {num_eps} eps)"
            )
        return (
            f"### [{env_name}/{task_name}] lcb_progression={lcb_prog:.4f} "
            f"({'PASS' if success else 'FAIL'}, {num_eps} episodes)"
        )

    def get_multi_episode_task_results(self, metrics: Dict[str, Any]) -> List[dict]:
        """Balrog: return multi-episode tasks that have summaries or errors.

        Tasks where every episode is near-perfect (no diary, no error) are
        excluded — there is nothing to consolidate. Does NOT call super()
        because the base default returns [] (safe for non-multi-episode
        benchmarks); Balrog re-implements the multi-episode filter directly.
        """
        return [
            r for r in metrics.get("task_results", [])
            if len(r.get("interaction_log", [])) > 1
            and any(
                ep.get("episode_summary") or "error" in ep
                for ep in r.get("interaction_log", [])
            )
        ]

    def build_consolidation_input_for_task(self, task_result: dict) -> str:
        """Balrog-style per-task consolidation input.

        Uses [env_name/task_name] lcb_progression=... format with
        per-episode compact lines.
        """
        interaction_log = task_result.get("interaction_log", [])
        lines = [self._task_header(task_result)]
        lines.extend(self._format_episode_lines(interaction_log))
        return "\n".join(lines)

    def format_task_consolidation_output(
        self, task_result: dict, consolidated_text: str
    ) -> str:
        """Balrog-style per-task consolidation output header."""
        return f"{self._task_header(task_result, short=True)}\n{consolidated_text.strip()}"

    def format_task_diaries_fallback(self, task_result: dict) -> str:
        """Balrog fallback: show raw diaries when per-task LLM consolidation fails."""
        return self.build_consolidation_input_for_task(task_result)

    def format_non_consolidated_tasks(self, metrics: Dict[str, Any]) -> str:
        """Format single-episode tasks in Balrog style."""
        parts = []
        for r in metrics.get("task_results", []):
            interaction_log = r.get("interaction_log", [])
            if len(interaction_log) > 1:
                continue
            meta = r.get("metadata", {})
            env_name = meta.get("env_name", "unknown")
            task_name = meta.get("task_name", r.get("task_id", "unknown"))
            lcb_prog = self._task_lcb(meta)
            success = r.get("success", False)
            summary = ""
            if interaction_log:
                summary = interaction_log[0].get("episode_summary", "")
            line = (
                f"- [{env_name}/{task_name}] lcb_progression={lcb_prog:.4f} "
                f"({'PASS' if success else 'FAIL'})"
            )
            if summary:
                line += f": {summary}"
            parts.append(line)
        return "\n".join(parts)

    def _condense_task_results(self, task_results: list) -> None:
        """Balrog-specific condense: remove step_traces, trim action_frequency to top-5."""
        for result in task_results:
            for ep in result.get("interaction_log", []):
                ep.pop("step_traces", None)
                ep.pop("api_messages", None)
                freq = ep.get("action_frequency")
                if isinstance(freq, dict) and len(freq) > 5:
                    top5 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
                    ep["action_frequency"] = dict(top5)

    def format_precomputed_summaries(self, metrics: Dict[str, Any]) -> Optional[str]:
        """Return the evaluation header (reward, per-env, thresholds, LCB note).

        No longer includes the raw ``## Episode Analysis`` diary dump — that is
        now handled by per-task consolidation in ``_consolidate_multi_episode_summaries``.
        Returns None only when task_results is empty.
        """
        task_results = metrics.get("task_results", [])
        if not task_results:
            return None

        reward = metrics.get("reward", 0.0)
        by_env = metrics.get("by_environment", {})
        total = len(task_results)
        failed = sum(1 for r in task_results if not r.get("success", False))

        lines = [
            "# Balrog Evaluation Summary",
            f"Reward (LCB): {reward:.4f} | Total: {total}, Failed: {failed}",
            "",
            *_LCB_NOTE_LINES,
            "",
        ]

        # Adaptive threshold context
        self._append_adaptive_threshold_info(lines, metrics)

        if by_env:
            lines.append("## Per-Environment")
            for env_name, data in by_env.items():
                avg = self._env_lcb(data)
                n = data.get("num_tasks", 0)
                lines.append(f"- {env_name}: lcb_progression={avg:.4f} ({n} tasks)")
            lines.append("")

        return "\n".join(lines)
