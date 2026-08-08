"""Terminal-Bench 2 log summary — formats evaluation results for LLM analysis."""

import json
from typing import Any, Dict, List, Optional

from benchmark.evaluators.log_summary_base import LogSummaryBase


class TerminalBenchLogSummary(LogSummaryBase):
    """Log summary handler for Terminal-Bench 2 benchmark."""

    def build_probe_seeds(
        self, metrics: Dict[str, Any], condensed_file: Optional[str]
    ) -> List[str]:
        """1-2 probe seeds targeting this eval's worst (failed / lowest
        tb2_reward) tasks.

        Points probe at the per-task ``interaction_log`` (the shell commands the
        harness ran) and the ``tb2_category`` so it can spot category-specific
        failure modes. Returns [] when nothing failed.
        """
        if not condensed_file:
            return []

        def reward_key(r: dict) -> float:
            rw = r.get("metadata", {}).get("tb2_reward")
            try:
                return float(rw) if rw is not None else 1.0  # None → sort last (worst)
            except (TypeError, ValueError):
                return 1.0

        failed = [r for r in metrics.get("task_results", []) if not r.get("success")]
        if not failed:
            return []
        failed.sort(key=reward_key)
        seeds: List[str] = []
        for r in failed[:2]:
            meta = r.get("metadata", {})
            tid = r.get("task_id", "unknown")
            cat = meta.get("tb2_category", "?")
            seeds.append(
                f"{condensed_file}: task '{tid}' failed [category={cat}]. Read its "
                f"interaction_log shell commands to find where the harness went wrong "
                f"(wrong command, missing step, or misread verifier), and report the "
                f"concrete failure mode + the harness change that would fix it."
            )
        return seeds

    def build_log_schema_description(self) -> str:
        """Terminal-Bench-specific log JSON schema description."""
        return (
            "## Terminal-Bench 2 Log JSON Schema\n"
            "### Top-level structure\n"
            "```json\n"
            "{\n"
            '  "reward": float,\n'
            '  "total_tasks": int,\n'
            '  "passed": int,\n'
            '  "failed": int,\n'
            '  "by_category": {"<cat>": float},\n'
            '  "task_results": [\n'
            "    {\n"
            '      "task_id": str,\n'
            '      "success": bool,\n'
            '      "metadata": {\n'
            '        "tb2_reward": float,\n'
            '        "tb2_category": str\n'
            "      },\n"
            '      "interaction_log": [\n'
            "        {\n"
            '          "role": "user" | "assistant" | "tool",\n'
            '          "content": str,\n'
            '          "tool_calls": [{"name": str, "args": str}]\n'
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n"
            "### Example queries\n"
            "```bash\n"
            "# List failed tasks by category\n"
            "python3 -c \"import json; d=json.load(open('LOG_FILE')); \\\n"
            "  [print(f\\\"{r['task_id']}: {r['metadata'].get('tb2_category','?')}\\\") \\\n"
            "   for r in d['task_results'] if not r['success']]\"\n"
            "```"
        )

    def __init__(self):
        self._harness_source: Optional[Dict[str, str]] = None

    def build_eval_summary_prompt(self, metrics: Dict[str, Any], **kwargs) -> str:
        """Build evaluation summary prompt with complete interaction logs.

        Dumps full task_results (including interaction_log) as JSON so the
        LLM sees every turn and can produce an accurate summary in one call.
        """
        harness_section = self._build_harness_section(self._harness_source)
        if self._harness_source:
            analysis = (
                "1. Overall pass rate and per-category breakdown\n"
                "2. Root cause analysis for each failed task (based on its interaction log)\n"
                "3. Which part of the harness code is responsible for each failure? "
                "Trace failures back to specific functions, prompt text, or logic.\n"
                "4. Common failure patterns across tasks and what code patterns cause them\n\n"
                + self._harness_diagnostic_suffix()
            )
        else:
            analysis = (
                "1. Overall pass rate and per-category breakdown\n"
                "2. Root cause analysis for each failed task (based on its interaction log)\n"
                "3. Common failure patterns across tasks\n"
                "4. Concrete, actionable suggestions for improving the agent's "
                "harness code, prompts, or tool usage\n"
            )
        return (
            "# Terminal-Bench 2 Evaluation Results\n\n"
            "Below is the complete evaluation data including interaction logs "
            f"for every task.  Analyze the results and provide:\n{analysis}\n"
            "```json\n"
            + json.dumps(metrics, ensure_ascii=False, indent=2, default=str)
            + "\n```"
            + harness_section
        )

    def build_summary_instruction(self) -> str:
        """Build summary instruction text for appending to api_messages.

        Used in messages-based summary mode: the agent's original API messages
        are reused and this instruction is appended as the final user message,
        enabling prompt cache hits.
        """
        if self._harness_source:
            harness_section = self._build_harness_section(self._harness_source)
            return (
                "The above is your complete interaction with the terminal for this task. "
                "This task FAILED the verifier check.\n\n"
                "Analyze why this task failed and provide:\n"
                "1. **Root cause**: What specific mistake, wrong command, or missing step "
                "caused the failure? Quote the exact commands or outputs that went wrong.\n"
                "2. **Code trace**: Which part of the harness code is responsible for the failure? "
                "Trace it back to the specific function, prompt text, or logic.\n"
                "3. **Pattern**: Is this a one-off mistake or a systematic issue that "
                "likely affects multiple tasks?\n\n"
                + harness_section
                + self._harness_diagnostic_suffix()
            )
        return (
            "The above is your complete interaction with the terminal for this task. "
            "This task FAILED the verifier check.\n\n"
            "Analyze why this task failed and provide:\n"
            "1. **Root cause**: What specific mistake, wrong command, or missing step "
            "caused the failure? Quote the exact commands or outputs that went wrong.\n"
            "2. **Actionable fix**: What concrete change to the agent's harness code, "
            "system prompt, or tool usage strategy would prevent this failure?\n"
            "3. **Pattern**: Is this a one-off mistake or a systematic issue that "
            "likely affects multiple tasks?"
        )

    def build_passed_summary(self, passed_count: int, total_count: int) -> str:
        """Build a summary line for passed tasks."""
        failed_count = total_count - passed_count
        return (
            f"## Evaluation Summary\n"
            f"- Total tasks: {total_count}\n"
            f"- Passed: {passed_count}\n"
            f"- Failed: {failed_count}\n"
            f"- Pass rate: {passed_count / total_count:.1%}" if total_count > 0 else
            f"## Evaluation Summary\n- No tasks evaluated"
        )

    def format_failed_tasks_only(self, reward: float | dict, metrics: Dict[str, Any]) -> str:
        """Format failed tasks with complete interaction logs as JSON."""
        task_results = metrics.get("task_results", [])
        failed = [r for r in task_results if not r.get("success", True)]

        reward_str = (
            ", ".join(f"{k}={v:.4f}" for k, v in reward.items() if isinstance(v, (int, float)))
            if isinstance(reward, dict) else f"{reward:.4f}"
        )

        return (
            f"Reward: {reward_str}\n"
            f"Failed: {len(failed)}/{len(task_results)} tasks\n\n"
            "Failed task results (with interaction logs):\n"
            "```json\n"
            + json.dumps(failed, ensure_ascii=False, indent=2, default=str)
            + "\n```"
        )
