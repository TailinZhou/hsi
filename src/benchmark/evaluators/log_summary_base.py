"""
Log Summary Base Class.

This module defines the abstract base class for benchmark-specific
log summary implementations. Each benchmark can provide its own
summary prompt generation logic.
"""

import copy
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LogSummaryBase(ABC):
    """Abstract base class for log summary handlers.

    Each benchmark can implement its own log summary logic by subclassing
    this class and returning it from BaseTaskEvaluator.get_log_summary_handler().

    The agent_action.py evaluate() method will use this handler to generate
    LLM summary prompts for evaluation results.
    """

    @staticmethod
    def _extract_task_progressions(metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract task_id -> avg_progression mapping from task_results."""
        return {
            r.get("task_id", ""): r.get("metadata", {}).get("avg_progression", 0.0)
            for r in metrics.get("task_results", [])
        }

    @staticmethod
    def _get_pre_thresholds(at: Dict[str, Any]) -> tuple:
        """Return (pre_task_thresholds, pre_task_best_rewards) with fallback."""
        thresholds = at.get("pre_task_thresholds", at.get("task_thresholds", {}))
        best_rewards = at.get("pre_task_best_rewards", at.get("task_best_rewards", {}))
        return thresholds, best_rewards

    @staticmethod
    def format_adaptive_threshold(metrics: Dict[str, Any], bold: bool = False) -> Optional[str]:
        """Format compact adaptive threshold summary, or None if not enabled."""
        at = metrics.get("adaptive_threshold")
        if not at or not at.get("enabled"):
            return None
        task_thresholds, _ = LogSummaryBase._get_pre_thresholds(at)
        if not task_thresholds:
            prefix = "**Adaptive Thresholds**" if bold else "Adaptive Thresholds"
            return f"{prefix}: no thresholds set yet (first evaluation)"
        task_progs = LogSummaryBase._extract_task_progressions(metrics)
        above = sum(
            1 for tid, thresh in task_thresholds.items()
            if task_progs.get(tid, 0.0) >= thresh
        )
        total = len(task_thresholds)
        prefix = "**Adaptive Thresholds**" if bold else "Adaptive Thresholds"
        return (
            f"{prefix}: {above}/{total} tasks above their personal bars "
            f"(margin={at['margin']})"
        )

    @staticmethod
    def format_adaptive_threshold_detail(metrics: Dict[str, Any]) -> Optional[str]:
        """Format detailed per-task threshold table with pass/fail status (weakest first), or None."""
        at = metrics.get("adaptive_threshold")
        if not at or not at.get("enabled"):
            return None
        task_thresholds, task_best_rewards = LogSummaryBase._get_pre_thresholds(at)
        if not task_thresholds:
            return None
        task_progs = LogSummaryBase._extract_task_progressions(metrics)
        sorted_tasks = sorted(
            task_thresholds.items(),
            key=lambda x: task_progs.get(x[0], 0.0) - x[1],
        )
        lines = [
            "| Task | Current | Threshold | Best | Status |",
            "|------|---------|-----------|------|--------|",
        ]
        for task_id, threshold in sorted_tasks:
            best = task_best_rewards.get(task_id, 0.0)
            current = task_progs.get(task_id, 0.0)
            status = "PASS" if current >= threshold else "FAIL"
            lines.append(f"| {task_id} | {current:.4f} | {threshold:.4f} | {best:.4f} | {status} |")
        return "\n".join(lines)

    @abstractmethod
    def build_eval_summary_prompt(self, metrics: Dict[str, Any], max_passed_samples: int = 10) -> str:
        """Build the evaluation summary prompt for LLM.

        This method should generate a prompt that includes:
        - Task results summary (passed/failed counts)
        - Detailed information for failed tasks (including interaction logs)
        - Randomly sampled passed tasks (up to max_passed_samples)
        - Specific analysis questions for the LLM

        Args:
            metrics: Evaluation metrics containing:
                - task_results: List of task evaluation results
                - reward: Overall reward value
                - Any benchmark-specific metrics
            max_passed_samples: Maximum number of passed tasks to randomly sample (default: 10)

        Returns:
            A formatted prompt string for the LLM to analyze
        """
        pass

    @abstractmethod
    def format_failed_tasks_only(self, reward: float | dict, metrics: Dict[str, Any]) -> str:
        """Format failed tasks only (concise mode).

        Used when evaluate_llm_summary=False in config.

        Args:
            reward: The evaluation reward (float or dict with per-category rates)
            metrics: Evaluation metrics containing task_results

        Returns:
            A concise string showing only failed tasks
        """
        pass

    def build_summary_instruction(self) -> str:
        """Build summary instruction text for appending to api_messages.

        Used in messages-based summary mode: the agent's original API messages
        are reused and this instruction is appended as the final user message,
        enabling prompt cache hits.

        Returns:
            Instruction text string.
        """
        return (
            "The interaction above is YOUR run — you lived it step by step. Write a "
            "short, blunt first-person diary entry about what hurt.\n\n"
            "Speak as \"I\" and point at the concrete instruction, prompt, or action "
            "shape you actually saw and executed that let you down — name it. Do NOT "
            "blame luck, randomness, the environment, or \"bad draws\", and do NOT blame "
            "yourself for \"not trying hard enough\": if you couldn't do the right thing, "
            "your harness didn't give you the means — say which means. State what that "
            "mechanism should have given you as a need, not a code change. Do NOT propose "
            "diffs or \"change X to Y\"."
        )

    @staticmethod
    def _build_harness_code_block(harness_source: Optional[Dict[str, str]]) -> str:
        """Format harness_source dict into a sorted, truncated text block."""
        if not harness_source:
            return ""
        priority = ["harness.py", "prompts.py", "context.py", "hooks.py", "tools_harness.py"]
        files = {
            k: v for k, v in harness_source.items()
            if not k.startswith("evolution/") and os.path.basename(k) != "__init__.py"
        }
        sorted_names = []
        for p in priority:
            if p in files:
                sorted_names.append(p)
        for name in sorted(files):
            if name not in sorted_names:
                sorted_names.append(name)

        max_chars = 16000
        parts = []
        total = 0
        for name in sorted_names:
            content = files[name]
            entry = f"=== {name} ===\n{content}\n\n"
            if total + len(entry) > max_chars:
                remaining = max_chars - total
                if remaining > 100:
                    truncated = content[:remaining - 60]
                    parts.append(f"=== {name} (truncated) ===\n{truncated}\n... [truncated]\n\n")
                break
            parts.append(entry)
            total += len(entry)
        return "".join(parts)

    @staticmethod
    def _build_harness_section(harness_source: Optional[Dict[str, str]]) -> str:
        """Build a full harness section with header, or empty string if no source."""
        block = LogSummaryBase._build_harness_code_block(harness_source)
        if not block:
            return ""
        return (
            "\n\n## Current Harness Source Code\n"
            "Below is the harness code that controlled this evaluation.\n\n"
            + block + "\n"
        )

    @staticmethod
    def _harness_diagnostic_suffix() -> str:
        """Standard prompt suffix: diagnose failures, trace to code, no suggestions."""
        return (
            "Do NOT suggest code changes. Just diagnose what went wrong and where in "
            "the code the problem originates."
        )

    def build_passed_summary(self, passed_count: int, total_count: int) -> str:
        """Build a summary line for passed tasks stats.

        Args:
            passed_count: Number of passed tasks.
            total_count: Total number of tasks.

        Returns:
            Summary string.
        """
        return f"Passed: {passed_count}/{total_count} tasks"

    def get_condensed_description(self) -> str:
        """Return a short description of what the condensed log omits.

        Used by _build_log_path_hint() to tell the agent what to expect
        without hardcoding benchmark-specific details.
        """
        return "all tasks, no step-level traces"

    def build_probe_seeds(
        self, metrics: Dict[str, Any], condensed_file: Optional[str]
    ) -> List[str]:
        """Build 1-2 ready-to-use `probe(instructions=...)` seeds for this eval.

        Each seed points the probe sub-agent at a concrete failing task in this
        evaluation's condensed log, so the evolve agent has a ready probe call
        instead of exploring eval_logs itself. Seeds reference the benchmark's
        own diagnostic fields (episode diaries, tool-call logs, shell commands,
        test output, ...).

        Trace-bearing benchmarks override this to return up to 2 seeds from the
        worst failing task(s). The default returns ``[]`` for benchmarks without
        a step-level trajectory to probe (e.g. classification, genesis) — probe
        would have nothing actionable to drill there.

        Args:
            metrics: This evaluation's metrics dict (task_results, ...).
            condensed_file: Path to the condensed log, or None.

        Returns:
            A list of 0-2 instruction strings (empty when nothing failed or the
            benchmark has no probeable trajectory).
        """
        return []

    def build_log_schema_description(self) -> str:
        """Return a human-readable description of the log JSON structure.

        Subclasses should override to describe their benchmark-specific schema
        and provide example queries (jq/python3).
        """
        return (
            "## Log JSON Schema\n"
            "```json\n"
            "{\n"
            '  "reward": float,\n'
            '  "total_tasks": int,\n'
            '  "task_results": [\n'
            "    {\n"
            '      "task_id": str,\n'
            '      "success": bool,\n'
            '      "metadata": {...},\n'
            '      "interaction_log": [...]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n"
            "### Example queries\n"
            "```bash\n"
            "# List failed task IDs\n"
            'python3 -c "import json; d=json.load(open(\'LOG_FILE\')); [print(r[\\\"task_id\\\"]) for r in d[\\\"task_results\\\"] if not r[\\\"success\\\"]]"\n'
            "```"
        )

    def build_condensed_log(self, metrics: Dict[str, Any]) -> dict:
        """Build a condensed copy of metrics without step-level traces.

        Removes step_traces and api_messages from every episode. Subclasses
        can override _condense_task_results() for benchmark-specific trimming.
        """
        condensed = copy.deepcopy(metrics)
        task_results = condensed.get("task_results", [])
        self._condense_task_results(task_results)
        condensed.pop("log_file", None)
        condensed.pop("log_dir", None)
        condensed.pop("condensed_log_file", None)
        condensed.pop("task_log_files", None)
        return condensed

    def _condense_task_results(self, task_results: List[dict]) -> None:
        """Remove verbose fields from task results in-place.

        Default: remove step_traces and api_messages from each episode in
        interaction_log. Subclasses can override for custom trimming.
        """
        for result in task_results:
            for ep in result.get("interaction_log", []):
                ep.pop("step_traces", None)
                ep.pop("api_messages", None)

    def format_precomputed_summaries(self, metrics: Dict[str, Any]) -> str | None:
        """Format pre-computed episode summaries if available.

        Returns None to indicate no precomputed summaries (caller falls through
        to _try_messages_summary or JSON dump).
        """
        return None

    @staticmethod
    def _format_episode_lines(interaction_log: List[dict]) -> List[str]:
        """Format episodes into indented lines. Shared by consolidation and precomputed paths."""
        lines = []
        for ep in interaction_log:
            if "error" in ep:
                lines.append(f"  Episode {ep.get('episode_idx', '?')}: ERROR - {ep['error']}")
                continue
            ep_idx = ep.get("episode_idx", "?")
            ep_prog = ep.get("progression", 0)
            ep_steps = ep.get("num_steps", 0)
            ep_invalid = len(ep.get("failed_candidates", []))
            s = ep.get("episode_summary", "")
            lines.append(
                f"  Episode {ep_idx} (progression={ep_prog:.3f}, "
                f"steps={ep_steps}, invalid={ep_invalid}):\n  {s}"
            )
        return lines

    def needs_consolidation(self, metrics: Dict[str, Any]) -> bool:
        """Return True if any task has multi-episode interaction_log with summaries.

        Single-episode benchmarks (AgentDojo etc.) return False → zero overhead.
        """
        for r in metrics.get("task_results", []):
            interaction_log = r.get("interaction_log", [])
            if len(interaction_log) > 1:
                for ep in interaction_log:
                    if ep.get("episode_summary"):
                        return True
        return False

    def build_consolidation_input(self, metrics: Dict[str, Any]) -> str:
        """Build structured text of per-task per-episode summaries for consolidation LLM.

        Only includes multi-episode tasks (single-episode tasks are formatted
        separately via format_non_consolidated_tasks). Delegates per-task
        formatting to build_consolidation_input_for_task.
        """
        parts = []
        for r in self.get_multi_episode_task_results(metrics):
            parts.append(self.build_consolidation_input_for_task(r))
        return "\n\n".join(parts)

    def format_non_consolidated_tasks(self, metrics: Dict[str, Any]) -> str:
        """Format single-episode tasks that don't need consolidation.

        Returns empty string if no single-episode tasks exist.
        """
        parts = []
        for r in metrics.get("task_results", []):
            interaction_log = r.get("interaction_log", [])
            if len(interaction_log) > 1:
                continue
            task_id = r.get("task_id", "unknown")
            success = r.get("success", False)
            summary = ""
            if interaction_log:
                summary = interaction_log[0].get("episode_summary", "")
            line = f"- {task_id} ({'PASS' if success else 'FAIL'})"
            if summary:
                line += f": {summary}"
            parts.append(line)
        return "\n".join(parts)

    def get_multi_episode_task_results(self, metrics: Dict[str, Any]) -> List[dict]:
        """Return task_results entries that need per-task consolidation.

        Default returns [] — safe for benchmarks with role-based turn logs
        (AgentDojo, TerminalBench, Polyglot, Classification, Genesis). Only
        benchmarks with genuine multi-episode semantics (Balrog: each entry
        in interaction_log is an episode dict with episode_idx/progression/
        episode_summary) MUST override this to return their multi-episode
        task list.
        """
        return []

    def build_consolidation_input_for_task(self, task_result: dict) -> str:
        """Build LLM consolidation input for a single multi-episode task.

        Formats the task header + per-episode diary lines. Subclasses
        (e.g. Balrog) override to add benchmark-specific metadata.
        """
        interaction_log = task_result.get("interaction_log", [])
        task_id = task_result.get("task_id", "unknown")
        success = task_result.get("success", False)
        lines = [f"### Task: {task_id} ({'PASS' if success else 'FAIL'})"]
        lines.extend(self._format_episode_lines(interaction_log))
        return "\n".join(lines)

    def format_task_consolidation_output(
        self, task_result: dict, consolidated_text: str
    ) -> str:
        """Wrap per-task LLM consolidation with a task header.

        Subclasses override to add benchmark-specific header formatting.
        """
        task_id = task_result.get("task_id", "unknown")
        success = task_result.get("success", False)
        header = f"### {task_id} ({'PASS' if success else 'FAIL'})"
        return f"{header}\n{consolidated_text.strip()}"

    def format_task_diaries_fallback(self, task_result: dict) -> str:
        """Fallback when per-task LLM consolidation fails.

        Default: return the raw consolidation input (header + episode diaries).
        Subclasses may override to use a different fallback format.
        """
        return self.build_consolidation_input_for_task(task_result)

    def get_interesting_task_ids(self, metrics: Dict[str, Any]) -> Optional[set]:
        """Return set of task IDs that should be highlighted in log file listing.

        Returns None to indicate all tasks should be shown (default).
        Override in subclasses to filter to failed + sampled passed tasks.
        """
        return None
