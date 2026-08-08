"""
Evaluation handlers for AgentActionExecutor.

Extracted from agent_action.py to reduce module size.
Provides evaluate and related evaluation summary methods.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_action import AgentActionExecutor

logger = logging.getLogger(__name__)

# ── LLM timeout + retry for evaluation-summary calls ──────────────────────
# These calls are auxiliary (diary consolidation, per-task analysis, JSON-dump
# summary) — they enrich the agent's feedback but are NOT on the critical path
# for reward computation. A hung API must not stall the evolution loop.
_EVAL_LLM_TIMEOUT = 60.0       # seconds per attempt
_EVAL_LLM_MAX_RETRIES = 2      # additional attempts after the first failure
_EVAL_LLM_RETRY_BACKOFF = 2.0  # seconds between retries (doubles each retry)


def _call_llm_with_retry(
    client,
    model: str,
    messages: list,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    extra_body: Optional[dict] = None,
    timeout: float = _EVAL_LLM_TIMEOUT,
    max_retries: int = _EVAL_LLM_MAX_RETRIES,
) -> Optional[Any]:
    """Call the LLM with timeout + exponential-backoff retry.

    Returns the raw API response on success, or ``None`` when every attempt
    fails.  The caller is responsible for extracting content and falling back
    to a non-LLM path when ``None`` is returned.
    """
    for attempt in range(max_retries + 1):
        try:
            kwargs: dict = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if extra_body is not None:
                kwargs["extra_body"] = extra_body

            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if attempt < max_retries:
                backoff = _EVAL_LLM_RETRY_BACKOFF * (2 ** attempt)
                logger.warning(
                    "LLM call failed (attempt %d/%d, retrying in %.1fs): %s",
                    attempt + 1, max_retries + 1, backoff, exc,
                )
                time.sleep(backoff)
            else:
                logger.error(
                    "LLM call failed after %d attempts: %s",
                    max_retries + 1, exc,
                )
    return None


# ── Shared consolidation prompt fragments ──────────────────────────────────
# Both the global (cross-task) and per-task consolidation prompts share the
# same voice preamble and "be specific / output shorter" rules. The preamble
# is parameterized by {scope} so each variant declares its own range.

_CONSOLIDATION_VOICE_PREAMBLE = (
    "You are an expert evaluation analyst. Consolidate {scope} into a "
    "concise, structured analysis.\n\n"
    "You are consolidating first-person player diary entries. Preserve the "
    "voice — keep them as the player's lived experience (what they tried, what "
    "happened, what they needed but lacked); do not convert them into objective "
    "'the agent failed because…' prose, and do NOT inject harness code/function "
    "attributions the player did not make — the player reports symptoms, the "
    "engineer locates causes."
)

_CONSOLIDATION_CLOSING_RULES = (
    "Be specific: ground each observation in concrete actions and game "
    "responses the player reported. Do NOT fabricate harness code/function "
    "attributions — leave root-cause location to the engineer who reads the "
    "code.\n"
    "{last_rule}. Output must be shorter than input while preserving all "
    "actionable info."
)

# ── Global (cross-task) consolidation prompts ────────────────────────────

CONSOLIDATION_SYSTEM_PROMPT = (
    _CONSOLIDATION_VOICE_PREAMBLE.format(
        scope="detailed evaluation summaries"
    )
    + "\n\nRules:\n"
    "1. Identify COMMON patterns across results (shared failure modes, recurring "
    "root causes, systematic issues). State each pattern once.\n"
    "2. Preserve UNIQUE situations that differ from the common pattern "
    "(e.g., a task that succeeded while similar ones failed, a rare edge case).\n"
    "3. Remove redundant repetition of the same diagnosis.\n"
    "4. Structure your output:\n"
    "   - Common patterns (shared across multiple results)\n"
    "   - Notable exceptions (result-specific deviations)\n"
    "   - Per-result stats line (compact summary)\n"
    "5. "
    + _CONSOLIDATION_CLOSING_RULES.format(last_rule="6")
)

CONSOLIDATION_INSTRUCTION = (
    "\n\n---\nConsolidate the above summaries into a structured analysis "
    "following the rules. Output consolidated analysis only."
)

# ── Per-task consolidation prompts ───────────────────────────────────────

PER_TASK_CONSOLIDATION_SYSTEM_PROMPT = (
    _CONSOLIDATION_VOICE_PREAMBLE.format(
        scope="multiple episode summaries for a SINGLE task"
    )
    + "\n\nRules:\n"
    "1. Track the progression trajectory across episodes — did the player "
    "improve, regress, or stay stuck? What changed between attempts?\n"
    "2. Identify RECURRING patterns within this task's episodes (shared "
    "failure modes, repeated mistakes, consistent blockers).\n"
    "3. Note UNIQUE episodes that diverged from the pattern (breakthroughs, "
    "regressions, different strategies tried).\n"
    "4. "
    + _CONSOLIDATION_CLOSING_RULES.format(last_rule="5")
)

PER_TASK_CONSOLIDATION_INSTRUCTION = (
    "\n\n---\nConsolidate the above episode summaries into a structured "
    "analysis following the rules. Output consolidated analysis only. "
    "({n} episodes)"
)


def _extract_header(raw_summary: str) -> str:
    """Extract the header portion before episode/detail analysis sections."""
    markers = [
        "## Episode Analysis", "## Consolidated Analysis",
        "## Failed Task Analysis", "## Failed Tasks",
        "## Task Analysis", "## Single-Episode Tasks",
    ]
    earliest = len(raw_summary)
    for marker in markers:
        idx = raw_summary.find(marker)
        if idx != -1 and idx < earliest:
            earliest = idx
    if earliest < len(raw_summary):
        return raw_summary[:earliest].rstrip()
    # No detail markers found — entire content is the header
    # (e.g. new-style format_precomputed_summaries that returns header-only)
    return raw_summary.strip()


class EvaluatorMixin:
    """Mixin providing evaluation methods for AgentActionExecutor."""

    # Type hints for attributes set by AgentActionExecutor
    external_evaluator: object
    agent_instance: object
    state: object
    agent_code_dir: str
    _compute_code_hash: object
    _evaluate_count: int
    _harness_history_last: dict
    logging: object

    def _persist_harness_history(
        self,
        iteration: int,
        evaluate_seq: int,
        code_hash: str,
        snapshot: Dict[str, str],
    ) -> None:
        """Persist the harness code that ran at this evaluate, for later debugging.

        Writes the snapshot's files under
        ``.evolution_context/main_evolve/harness_history/iter_<it>/eval_<seq>_<hash>/``
        so every distinct evaluated code state is recoverable from disk. This
        matters because ``state.code_snapshots`` (the in-memory map of every
        evaluated version) is never persisted — without this, any version the
        agent didn't pick for the commit pool is gone forever once the run ends
        (exactly how the iter-5 ``98ca03d3f3f0`` / 0.4157 version was lost).

        Skips the write when ``code_hash`` is unchanged from the previous eval in
        the same iteration, so back-to-back identical evals aren't duplicated.
        The directory sits under ``main_evolve/``, which is already gitignored,
        so this never bloats evolution commits.
        """
        if not code_hash or not snapshot:
            return
        try:
            agent_code_dir = getattr(self, "agent_code_dir", "") or ""
            if not agent_code_dir:
                return
            # Dedupe within an iteration: skip if this hash matches the previous
            # persisted eval (no code change since). Per-iteration map so a new
            # iteration always records its first eval.
            last_map = getattr(self, "_harness_history_last", None)
            if last_map is None:
                last_map = {}
                self._harness_history_last = last_map
            if last_map.get(iteration) == code_hash:
                return
            last_map[iteration] = code_hash

            hist_dir = os.path.join(
                agent_code_dir,
                ".evolution_context", "main_evolve", "harness_history",
                f"iter_{iteration}",
                f"eval_{evaluate_seq:03d}_{code_hash[:12]}",
            )
            os.makedirs(hist_dir, exist_ok=True)
            for rel_path, content in snapshot.items():
                full_path = os.path.join(hist_dir, rel_path)
                parent = os.path.dirname(full_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)
        except Exception as exc:
            try:
                self.logging(f"Warning: harness_history persist failed: {exc}")
            except Exception:
                pass

    def evaluate(
        self: "AgentActionExecutor",
        test_cases: List[Any] = None,
        func_names: List[str] = None,
        eval_mode: str = "dev",
        num_tasks: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Evaluate agent performance using the external_evaluator.

        Args:
            test_cases: Test cases.
            func_names: List of entry function names to look up.
            eval_mode: "dev" (detailed feedback) or "val" (reward-only black box)
            num_tasks: Debug only — limit the number of evaluation tasks. When set, the reward is not tracked for evolution.
            task_ids: DEBUG ONLY — list of task IDs to evaluate. Always dev debug mode,
                      the reward is not tracked for evolution. Mutually exclusive with num_tasks (task_ids wins).

        Returns:
            Execution result.
        """
        test_cases = test_cases or []

        # Normalize empty task_ids list to None
        if task_ids is not None and len(task_ids) == 0:
            task_ids = None

        # task_ids is always dev mode with detailed per-task feedback. Force it
        # even if the agent passed eval_mode='val' (which would produce a useless
        # black-box result with no per-task breakdown and no reward tracking).
        if task_ids is not None and eval_mode != "dev":
            self.logging(
                f"task_ids forces eval_mode='dev' (val would produce no per-task "
                f"breakdown and no reward tracking with task_ids — useless). "
                f"Ignoring eval_mode='{eval_mode}'."
            )
            eval_mode = "dev"

        # task_ids and num_tasks are mutually exclusive — task_ids takes priority
        if task_ids is not None and num_tasks is not None:
            self.logging(
                f"Warning: both task_ids and num_tasks provided. "
                f"Ignoring num_tasks={num_tasks}, using task_ids={task_ids}."
            )
            num_tasks = None

        if num_tasks is not None and num_tasks <= 0:
            return "Error: num_tasks must be a positive integer."

        # Both num_tasks and task_ids are debug modes. num_tasks auto-upgrades
        # to a tracked full eval when N >= total available tasks (checked below);
        # task_ids is always debug — no auto-upgrade.
        is_debug = (num_tasks is not None) or (task_ids is not None)

        # Increment the evaluate call count
        self._evaluate_count += 1
        evaluate_seq = self._evaluate_count
        if self.agent_instance is not None:
            self.agent_instance._eval_count_in_iteration += 1

        # Get the current iteration number
        current_iteration = self.state.iteration if self.state else 0

        # If an external_evaluator is available, use it for evaluation (main path)
        if self.external_evaluator is not None:
            try:
                # Always compute code_hash for code-reward alignment
                code_hash, snapshot = self._compute_code_hash(return_contents=True)
                if self.state and code_hash and snapshot and code_hash not in self.state.code_snapshots:
                    self.state.code_snapshots[code_hash] = snapshot

                # Persist a disk copy of this evaluated code state for debugging
                # (state.code_snapshots is in-memory only and never persisted, so
                # this is the only way to recover a non-committed version later).
                self._persist_harness_history(current_iteration, evaluate_seq, code_hash, snapshot)

                reward, metrics = self.external_evaluator(
                    self.agent_instance,
                    None,  # Not pre-fetched; let HarnessLoader handle it uniformly
                    test_cases,
                    func_names,
                    evaluate_seq=evaluate_seq,
                    iteration=current_iteration,
                    eval_mode=eval_mode,
                    num_tasks=num_tasks,
                    task_ids=task_ids,
                )

                # Val mode error handling
                if reward is None and metrics.get("error"):
                    return f"Evaluation ({eval_mode}) failed: {metrics['error']}"

                # Auto-upgrade: if num_tasks >= all available tasks, treat as full
                # evaluation. Applies only to pure num_tasks mode (task_ids never
                # auto-upgrades — already baked into is_debug above).
                if num_tasks is not None and metrics:
                    total_available = metrics.get("total_available_tasks", 0)
                    if total_available > 0 and num_tasks >= total_available:
                        is_debug = False
                        self.logging(
                            f"Auto-upgrade: num_tasks={num_tasks} >= total_available={total_available}, "
                            f"treating as full evaluation (reward will be tracked)"
                        )

                if reward is not None:
                    # Extract execution errors EARLY (before val/dev split) so the
                    # count is available for both snapshot tracking AND the dev-mode
                    # feedback text. A harness crash (TypeError, NameError, …) is a
                    # code bug, not a strategy failure — the agent must see it.
                    exec_errors = self._extract_execution_errors(metrics) if metrics else 0

                    # Record evaluation snapshot for code-reward alignment.
                    # The 4th element is the execution-error count for this
                    # eval — a version whose harness crashed (e.g. NameError)
                    # must never be selected as best, even when its averaged
                    # reward is high. Threading it on the snapshot lets
                    # commit-time alignment + cross-iteration best-selection
                    # veto tainted code without changing the 3-tuple schema
                    # every other consumer unpacks.
                    #
                    # Always record the code_hash so the commit agent can see
                    # every evaluated version.  Debug (partial-task) evals use
                    # reward=None — the hash is visible in the candidate table
                    # but downstream reward consumers see 0.0 (reward_to_scalar)
                    # or "N/A" (fmt_reward), so debug-only versions never beat
                    # fully-evaluated ones in ranking.
                    snapshot_reward = reward if not is_debug else None
                    if self.state:
                        modified_snapshot = list(self._modified_files) if hasattr(self, '_modified_files') else []
                        mods_snapshot = list(self.state.modifications_made) if self.state else []
                        self.state.evaluation_snapshots.append(
                            [code_hash, snapshot_reward, eval_mode, exec_errors, modified_snapshot, mods_snapshot]
                        )

                    # Debug mode: do not track reward and metrics
                    if not is_debug:
                        if self.state:
                            self.state.update_reward(reward, eval_mode=eval_mode)
                        if metrics and self.state:
                            self.state.update_metrics(metrics)

                    # Val mode: return reward-only (black box)
                    if eval_mode == "val":
                        reward_tag = self._reward_type_tag(metrics)
                        ch_short = code_hash[:12] if code_hash else "?"
                        if isinstance(reward, dict):
                            val_str = ", ".join(
                                f"{k}={v:.4f}" for k, v in reward.items() if isinstance(v, (int, float))
                            )
                        else:
                            val_str = f"reward={reward:.4f}{reward_tag}"
                        debug_prefix = "[DEBUG EVALUATE - reward NOT tracked] " if is_debug else ""
                        return (
                            f"{debug_prefix}Validation Evaluation (reward-only): {val_str}\n"
                            f"[code_hash: {ch_short}]"
                        )

                    # Dev mode: detailed feedback
                    reward_tag = self._reward_type_tag(metrics)
                    # Show code_hash so agent can reference it if this version is worth pooling
                    ch_short = code_hash[:12] if code_hash else "?"
                    pool_hint = (
                        f"\n[code_hash: {ch_short}] "
                        f"If this version is worth keeping (breakthrough, new direction, "
                        f"or reliable improvement), call "
                        f"`pick_commit_version(code_hash=\"{ch_short}\")` "
                        f"to add it to the commit pool."
                    )
                    if isinstance(reward, dict):
                        reward_feedback = "Evaluation:\n" + "\n".join(
                            f"  {k}={v:.4f}" if isinstance(v, (int, float)) else f"  {k}={v}"
                            for k, v in reward.items()
                        )
                        if reward_tag:
                            reward_feedback += f"\n  [reward is a lower-confidence bound{reward_tag}]"
                    else:
                        reward_feedback = f"Evaluation: reward={reward:.4f}{reward_tag}"

                    reward_feedback += pool_hint

                    # Dynamic-sampling reminder: dev re-samples tasks each call, so
                    # reward swings may be sampling noise — head off over-iteration
                    # on variance (the iter1 "chase a lost peak" failure mode).
                    if metrics.get("dynamic_sample"):
                        _sampled = metrics.get("total_tasks", "?")
                        _pool = metrics.get("dev_pool_size", _sampled)
                        reward_feedback += (
                            f"\n📊 dev dynamically sampled {_sampled} of {_pool} tasks — "
                            f"each dev evaluate re-samples, so reward swings between "
                            f"evals may be sampling noise, not your changes. Use "
                            f"`evaluate(mode=\"val\")` for a stable comparison, and "
                            f"don't over-iterate a single dip."
                        )

                    if is_debug:
                        actual_count = metrics.get("total_tasks", num_tasks)
                        reward_feedback = (
                            f"[DEBUG EVALUATE - reward NOT tracked] "
                            f"(evaluated {actual_count} of tasks)\n{reward_feedback}"
                        )

                    # Harness crash → skip summary, return the error directly.
                    # A TypeError/NameError in harness code is a bug, not a
                    # strategy failure. Routing it through the LLM summary layer
                    # risks the LLM compressing it into vague "failed to make
                    # progress" language. The agent needs the raw traceback to
                    # fix the code.
                    enable_file_log = getattr(
                        self.agent_instance.config, 'enable_file_log', False
                    ) if self.agent_instance else False

                    if exec_errors > 0:
                        error_details = self._format_execution_error_details(metrics)
                        result = (
                            f"{reward_feedback}\n\n"
                            f"⚠ HARNESS CRASH — {exec_errors} task(s) raised "
                            f"an exception during evaluation.\n\n"
                            f"{error_details}\n\n"
                            f"Fix the crash before evaluating again. "
                            f"The reward above may be artificially low because "
                            f"crashed episodes were scored as 0."
                        )
                        if enable_file_log:
                            self._log_evaluation_result(result, code_hash, eval_mode)
                        return result

                    # Determine the return format based on config
                    use_llm_summary = (
                        getattr(
                            self.agent_instance.config, 'evaluate_llm_summary', True
                        ) if self.agent_instance else False
                    )

                    if use_llm_summary:
                        summary = self._summarize_evaluation(reward, metrics)
                        result = f"{reward_feedback}\n\n{summary}"
                        if enable_file_log:
                            result += f"\n\n{self._build_log_path_hint(metrics)}"
                        return result
                    else:
                        if enable_file_log:
                            return self._format_compact_with_log_hint(reward, metrics, evaluate_seq, current_iteration)
                        else:
                            return self._format_failed_tasks_only(reward, metrics)
                else:
                    error_info = metrics.get("error", "Unknown error") if isinstance(metrics, dict) else str(metrics)
                    return f"Evaluation failed: {error_info}"
            except Exception as e:
                import traceback
                tb_str = traceback.format_exc()
                self.logging(f"Evaluation error:\n{tb_str}")
                return f"Evaluation error:\n{tb_str}"

        # No external_evaluator configured -> direct error
        return "Error: No external evaluator configured. Cannot evaluate without a benchmark evaluator."

    def _summarize_evaluation(self: "AgentActionExecutor", reward, metrics: Dict[str, Any]) -> str:
        """Summarize the evaluation result with an LLM.

        Priority:
        0. Precomputed summaries (already embedded at evaluation time, no extra LLM call needed)
        1. Messages mode: reuse api_messages + summary instruction, hits the prompt cache
        2. JSON dump approach: independent LLM call
        """
        handler = self._get_log_summary_handler()

        # Priority 0: precomputed summaries (no extra LLM call)
        if handler:
            precomputed = handler.format_precomputed_summaries(metrics)
            if precomputed is not None:
                return self._consolidate_multi_episode_summaries(
                    precomputed, metrics, handler
                )

        # Priority 1: messages-based summary using api_messages for prompt cache reuse
        summary_result = self._try_messages_summary(metrics, handler)
        if summary_result is not None:
            # Strip api_messages from metrics to avoid memory bloat on state.update_metrics
            self._strip_api_messages(metrics)
            return summary_result

        # Fallback: traditional JSON dump + LLM summary
        if handler:
            prompt = handler.build_eval_summary_prompt(metrics)
        else:
            prompt = self._build_generic_eval_summary_prompt(metrics)

        system_prompt = """You are an expert evaluation analyst. Your job is to analyze benchmark results and provide actionable insights for improving an AI agent's strategy.

Your analysis should be:
1. **Specific**: Point to exact mistakes, not vague descriptions
2. **Actionable**: Suggest concrete changes the agent can make
3. **Pattern-aware**: Identify recurring issues across multiple tasks

Focus on what the agent can actually control and improve in its harness code, prompts, or tool usage."""

        response = _call_llm_with_retry(
            self.agent_instance.llm_client,
            self.agent_instance.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        if response is None:
            # LLM summarization timed out — fall back to simple format
            return self._format_failed_tasks_only(reward, metrics)
        return self._extract_content(response)

    def _consolidate_multi_episode_summaries(
        self: "AgentActionExecutor",
        raw_summary: str,
        metrics: Dict[str, Any],
        handler: Any,
    ) -> str:
        """Consolidate per-episode summaries into structured analysis.

        Two paths:
        1. Per-task consolidation (balrog): each multi-episode task gets its
           own LLM call → ``## Task Analysis`` section. Single-episode tasks
           are listed under ``## Single-Episode Tasks``.
        2. Global consolidation (agentdojo): one LLM call across all tasks →
           ``## Consolidated Analysis`` section. Only reached when
           get_multi_episode_task_results() returns [].
        Falls back to raw_summary unchanged on any failure.

        When ``evaluate_consolidate_summary`` is False (default), all
        consolidate LLM calls are skipped: the full per-episode first-person
        diaries are emitted verbatim (balrog → ``## Episode Analysis``;
        agentdojo → raw_summary, which already carries per-task originals).
        """
        use_consolidation = getattr(
            self.agent_instance.config, 'evaluate_consolidate_summary', False
        ) if self.agent_instance else False

        header = _extract_header(raw_summary)
        multi_ep_tasks = handler.get_multi_episode_task_results(metrics)

        if multi_ep_tasks:
            # ── Path 1: per-task consolidation ──────────────────────────
            if not use_consolidation:
                # Switch OFF: emit full per-episode diaries, no LLM call.
                episode_parts = [
                    handler.build_consolidation_input_for_task(task_result)
                    for task_result in multi_ep_tasks
                ]
                parts = [header, "## Episode Analysis", "\n\n".join(episode_parts)]

                # Single-episode tasks go after the raw diaries
                single_ep = handler.format_non_consolidated_tasks(metrics)
                if single_ep:
                    parts.append("\n## Single-Episode Tasks\n" + single_ep)

                return "\n".join(parts)

            per_task_sections = []
            for task_result in multi_ep_tasks:
                task_input = handler.build_consolidation_input_for_task(task_result)
                n_eps = len(task_result.get("interaction_log", []))
                instruction = PER_TASK_CONSOLIDATION_INSTRUCTION.format(n=n_eps)
                response = _call_llm_with_retry(
                    self.agent_instance.llm_client,
                    self.agent_instance.config.model,
                    messages=[
                        {"role": "system", "content": PER_TASK_CONSOLIDATION_SYSTEM_PROMPT},
                        {"role": "user", "content": task_input + instruction},
                    ],
                    temperature=0,
                )
                if response is not None:
                    content = self._extract_content(response)
                    if content and len(content.strip()) > 50:
                        per_task_sections.append(
                            handler.format_task_consolidation_output(task_result, content)
                        )
                        continue

                # Fallback: LLM call failed or returned empty/short content
                if response is None:
                    self.logging(
                        "Per-task consolidation failed for %s: LLM call timed out",
                        task_result.get("task_id", "?"),
                    )
                per_task_sections.append(task_input)

            parts = [header, "## Task Analysis", "\n\n".join(per_task_sections)]

            # Single-episode tasks go after the per-task consolidation
            single_ep = handler.format_non_consolidated_tasks(metrics)
            if single_ep:
                parts.append("\n## Single-Episode Tasks\n" + single_ep)

            return "\n".join(parts)

        elif handler.needs_consolidation(metrics):
            # ── Path 2: global consolidation (agentdojo, behavior unchanged) ──
            if not use_consolidation:
                # Switch OFF: raw_summary already contains the full per-task
                # originals (## Failed Tasks + ## Sampled Passed Tasks).
                return raw_summary

            consolidation_input = handler.build_consolidation_input(metrics)
            if not consolidation_input.strip():
                return raw_summary

            user_content = consolidation_input + CONSOLIDATION_INSTRUCTION
            response = _call_llm_with_retry(
                self.agent_instance.llm_client,
                self.agent_instance.config.model,
                messages=[
                    {"role": "system", "content": CONSOLIDATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
            )
            if response is None:
                self.logging("Cross-episode consolidation failed: LLM call timed out after all retries")
                return raw_summary

            consolidated = self._extract_content(response)
            if not consolidated or len(consolidated.strip()) <= 50:
                return raw_summary

            task_index = handler.format_non_consolidated_tasks(metrics)

            # Strip duplicate heading if the LLM already included one
            stripped = consolidated.lstrip()
            if stripped.startswith("## Consolidated Analysis"):
                consolidated = stripped[len("## Consolidated Analysis"):].lstrip("\n")

            parts = [header, "\n## Consolidated Analysis\n", consolidated]
            if task_index:
                parts.append("\n## Task Index\n")
                parts.append(task_index)
            return "\n".join(parts)

        else:
            # No consolidation needed — still show single-episode tasks if any
            single_ep = handler.format_non_consolidated_tasks(metrics)
            if single_ep:
                return "\n".join([header, "## Single-Episode Tasks", single_ep])
            return raw_summary

    def _try_messages_summary(
        self: "AgentActionExecutor",
        metrics: Dict[str, Any],
        handler: Optional[Any],
    ) -> Optional[str]:
        """Try to use api_messages for summary (reuse the prompt cache).

        Make an independent LLM call for each failed task that has api_messages,
        then aggregate all per-task analysis + passed task stats as the final summary.
        Returns None if messages mode is unavailable, signaling a fallback is needed.
        """
        task_results = metrics.get("task_results", [])
        if not task_results:
            return None

        failed_with_msgs = []
        passed_count = 0
        total_count = len(task_results)

        for r in task_results:
            if not r.get("success", True):
                api_msgs = r.get("api_messages") or r.get("metadata", {}).get("api_messages")
                if api_msgs and isinstance(api_msgs, list) and len(api_msgs) > 0:
                    failed_with_msgs.append({
                        "task_id": r.get("task_id", r.get("id", "unknown")),
                        "api_messages": api_msgs,
                    })
            else:
                passed_count += 1

        if not failed_with_msgs:
            return None

        instruction = handler.build_summary_instruction() if handler else (
            "Based on the above interaction, analyze why this task FAILED. "
            "Provide:\n"
            "1. Root cause: What specific mistake or limitation caused the failure?\n"
            "2. Actionable suggestion: What concrete change to the agent's harness code, "
            "prompts, or tool usage would fix this?"
        )

        per_task_analyses = []
        failed_count = 0
        for task_info in failed_with_msgs:
            messages = task_info["api_messages"] + [{"role": "user", "content": instruction}]
            response = _call_llm_with_retry(
                self.agent_instance.llm_client,
                self.agent_instance.config.model,
                messages=messages,
                temperature=0,
            )
            if response is None:
                failed_count += 1
                per_task_analyses.append(f"### Task: {task_info['task_id']}\n[Analysis failed: LLM call timed out after all retries]")
            else:
                per_task_analyses.append(f"### Task: {task_info['task_id']}\n{self._extract_content(response)}")

        # All LLM calls failed — fall back to JSON dump
        if failed_count == len(failed_with_msgs):
            return None

        passed_summary = handler.build_passed_summary(passed_count, total_count) if handler else f"Passed: {passed_count}/{total_count} tasks"

        summary = f"{passed_summary}\n\n## Failed Task Analysis\n\n"
        summary += "\n\n".join(per_task_analyses)
        return summary

    def _strip_api_messages(self: "AgentActionExecutor", metrics: Dict[str, Any]) -> None:
        """Remove api_messages from metrics to avoid memory bloat on state persistence."""
        for r in metrics.get("task_results", []):
            r.pop("api_messages", None)
            md = r.get("metadata")
            if isinstance(md, dict):
                md.pop("api_messages", None)

    @staticmethod
    def _extract_content(response) -> str:
        """Extract text from LLM response, falling back to reasoning_content.

        Some providers (e.g. DeepSeek R1) put output in reasoning_content
        and leave content empty when thinking is enabled.
        """
        msg = response.choices[0].message
        content = msg.content
        if content:
            return content
        # Fallback: provider put the answer in reasoning_content
        rc = getattr(msg, "reasoning_content", None) or getattr(msg, "reason_content", None)
        return rc or ""

    def _get_log_summary_handler(self: "AgentActionExecutor") -> Optional[Any]:
        """Get the log summary handler."""
        if self.external_evaluator and hasattr(self.external_evaluator, 'get_log_summary_handler'):
            return self.external_evaluator.get_log_summary_handler()
        return None

    def _build_generic_eval_summary_prompt(self: "AgentActionExecutor", metrics: Dict[str, Any]) -> str:
        """Build a generic evaluation summary prompt (fallback)."""
        task_results = metrics.get("task_results", [])
        passed = [r for r in task_results if r.get("success")]
        failed = [r for r in task_results if not r.get("success")]

        summary = f"""# Evaluation Results
Reward: {metrics.get('reward', 'N/A')}
Total: {len(task_results)}, Passed: {len(passed)}, Failed: {len(failed)}

## Raw Task Results:
```json
{json.dumps(task_results, ensure_ascii=False, indent=2)}
```

---
Based on the above task results, analyze:
1. What went well? What patterns led to success?
2. What failed and why? Identify root causes for each failed task."""
        return summary

    def _format_failed_tasks_only(self: "AgentActionExecutor", reward, metrics: Dict[str, Any]) -> str:
        """Return only failed task info (concise mode)."""
        # Try to use the handler
        handler = self._get_log_summary_handler()
        if handler:
            return handler.format_failed_tasks_only(reward, metrics)

        # Fallback: generic implementation
        task_results = metrics.get("task_results", [])
        failed = [r for r in task_results if not r.get("success")]
        passed_count = len(task_results) - len(failed)

        lines = self._format_reward_header(reward, passed_count, len(task_results), metrics)

        if failed:
            lines.append("\n## Failed Tasks:")
            lines.append(f"```json\n{json.dumps(failed, ensure_ascii=False, indent=2)}\n```")
        else:
            lines.append("\nAll tasks passed! No failures to report.")

        return "\n".join(lines)

    def _build_log_path_hint(self: "AgentActionExecutor", metrics: Dict[str, Any]) -> str:
        """Build log path hint with condensed + per-task log references."""
        log_file = metrics.get("log_file")
        condensed_file = metrics.get("condensed_log_file")
        task_log_files = metrics.get("task_log_files", [])
        log_dir = metrics.get("log_dir")

        lines = []

        # Get handler once for filtering, schema, and diagnostic tips
        handler = self._get_log_summary_handler()

        # 1. Condensed log (primary recommendation)
        if condensed_file:
            condensed_desc = handler.get_condensed_description() if handler else "all tasks"
            lines.append(f"Condensed log ({condensed_desc}): {condensed_file}")
            lines.append("  Read this first for an overview of pass/fail status.")
        elif log_dir:
            lines.append(f"Evaluation log directory: {log_dir}/")

        # 2. Per-task trace files — naming + bash/probe split (light reads self,
        #    full end-to-end trace walk delegates to probe)
        if task_log_files:
            lines.append(f"\nPer-task trace files: {len(task_log_files)} files in {log_dir or 'eval log dir'}/")
            lines.append("  Naming: <prefix>_task_<task_id>.json — {task_id, interaction_log[]} with")
            lines.append("  per-step detail. Trace fields vary by benchmark (episode diaries, tool-call")
            lines.append("  logs, shell commands, test output, ...) — the probe sub-agent reads them via")
            lines.append("  the schema shown below, so don't assume a specific shape here.")
            lines.append(
                "  These traces are 100KB+ when read whole. Light reads (jq keys, a few steps, the "
                "condensed overview, framework source) are fine to do yourself — first-hand and small. "
                "When you need to walk one failing episode end-to-end to understand WHY it failed, "
                "use `probe` with a SPECIFIC instruction: name the exact task file, the ONE question "
                "you want answered, and the concrete signals to look for. The sub-agent returns a "
                "cited findings summary. Hints for your own probe instructions:"
            )
            for seed in self._probe_instruction_seeds(metrics, condensed_file):
                lines.append(f'    {seed}')

        return "\n".join(lines)

    def _probe_instruction_seeds(
        self: "AgentActionExecutor", metrics: Dict[str, Any], condensed_file: Optional[str]
    ) -> List[str]:
        """Build 1-2 concrete `probe` instruction seeds, delegated to the
        benchmark's log-summary handler.

        Appended to the evaluate result's log-path hint so the agent has a
        ready-to-use probe call targeting the actual failing task(s), instead of
        exploring eval_logs itself. Each handler knows its own trajectory shape
        (episode diaries, tool-call logs, shell commands, ...) and points probe
        at the right fields. No-trace benchmarks (classification, genesis) and
        benchmarks without a handler return [] — there is nothing to probe.
        """
        handler = self._get_log_summary_handler()
        if not handler:
            return []
        try:
            return handler.build_probe_seeds(metrics, condensed_file)
        except Exception as e:
            self.logging(f"build_probe_seeds failed: {e}")
            return []

    def _extract_execution_errors(self: "AgentActionExecutor", metrics: Any) -> int:
        """Count tasks whose evaluation CRASHED (harness exception, e.g. NameError).

        This is the veto signal for best-version selection: a version that
        crashes is never the best even if its averaged reward is high. Crucially,
        a crash is distinct from a normal failed task — an episode that hit a
        NameError scores 0 by accident, not because the strategy was bad, so its
        reward is a misleading over-/under-estimate.

        Sources, in priority order (val mode carries the same signal as dev —
        the benchmark adapter computes it regardless of mode):
        1. Balrog: top-level ``execution_errors`` int.
        2. AgentDojo: top-level ``execution_errors`` list → its length.
        3. Fallback: count ``task_results`` entries flagged with an
           ``execution_error`` field (set by balrog/agentdojo/terminal_bench
           adapters), tolerating legacy 3-element snapshots.

        Returns 0 when no crash signal is present (no veto). Returns 0 for
        benchmarks that don't report this field — they already map total-train
        crashes to reward=None upstream, so partial crashes are the only gap,
        and those benchmarks are not the ones that motivated this.
        """
        if not isinstance(metrics, dict):
            return 0
        top = metrics.get("execution_errors")
        if isinstance(top, bool):  # guard: bool is an int subclass
            return int(top)
        if isinstance(top, int):
            return top
        if isinstance(top, list):
            return len(top)
        count = 0
        for r in metrics.get("task_results", []):
            if not isinstance(r, dict):
                continue
            meta = r.get("metadata")
            if isinstance(meta, dict) and meta.get("execution_error"):
                count += 1
            elif r.get("execution_error"):
                count += 1
        return count

    def _format_execution_error_details(
        self: "AgentActionExecutor", metrics: Any
    ) -> str:
        """Extract per-task execution error tracebacks for display to the agent."""
        if not isinstance(metrics, dict):
            return ""
        lines = []
        for r in metrics.get("task_results", []):
            if not isinstance(r, dict):
                continue
            task_id = r.get("task_id", "?")
            meta = r.get("metadata")
            err = None
            if isinstance(meta, dict) and meta.get("execution_error"):
                err = meta["execution_error"]
            elif r.get("execution_error"):
                err = r["execution_error"]
            if err:
                # Indent multi-line tracebacks so they render cleanly under the
                # bullet point — the agent sees file paths and line numbers at a
                # glance without the traceback blending into surrounding text.
                indented = "\n".join(f"  {line}" for line in err.strip().split("\n"))
                lines.append(f"- **{task_id}**:\n{indented}")
        return "\n".join(lines) if lines else "(no per-task detail available)"

    def _reward_type_tag(self: "AgentActionExecutor", metrics: Optional[Dict[str, Any]]) -> str:
        """Annotation for the reward's uncertainty type, or '' when not applicable.

        When the benchmark reports a lower-confidence-bound reward (metrics
        ``reward_type == "lcb"``), returns ``" (LCB, z=<z>)"`` so the agent
        understands the number is a conservative, uncertainty-penalized estimate
        (mean − z·std/√n) rather than a raw mean.
        """
        if not isinstance(metrics, dict):
            return ""
        if metrics.get("reward_type") != "lcb":
            return ""
        z = metrics.get("lcb_zscore", 1.0)
        return f" (LCB, z={z})"

    def _format_reward_header(self: "AgentActionExecutor", reward, passed_count: int, total_count: int, metrics: Dict[str, Any]) -> List[str]:
        """Format the reward line with passed/total count + LCB tag if applicable."""
        reward_tag = self._reward_type_tag(metrics)
        if isinstance(reward, dict):
            rate_str = ", ".join(
                f"{k}={v:.4f}" for k, v in reward.items() if isinstance(v, (int, float))
            )
            return [f"Evaluation: {rate_str}{reward_tag} (passed: {passed_count}/{total_count})"]
        else:
            return [f"Evaluation: reward={reward:.4f}{reward_tag} (passed: {passed_count}/{total_count})"]

    @staticmethod
    def _truncate(s: str, limit: int = 100) -> str:
        s = str(s).replace("\n", " ").strip()
        return s[:limit] + "..." if len(s) > limit else s

    @staticmethod
    def _fmt_float(value: Any, fmt: str = ".2f") -> str:
        try:
            return f"{float(value):{fmt}}"
        except (ValueError, TypeError):
            return str(value)

    def _format_failed_task_detail(self: "AgentActionExecutor", r: dict) -> List[str]:
        """Detect the benchmark type from metadata feature fields and format the failed task detail.

        Priority: balrog > agentdojo > terminal_bench > classification > genesis > polyglot > fallback
        Returns the list of formatted lines (the first line contains task_id, subsequent lines are indented detail).
        """
        task_id = r.get("task_id", r.get("id", r.get("name", str(r.get("task", "unknown")))))
        meta = r.get("metadata", {})
        if meta is None:
            meta = {}

        # --- Balrog: avg_progression + interaction_log with episodes ---
        ep_logs = r.get("interaction_log", [])
        avg_prog = meta.get("avg_progression")
        if avg_prog is not None and ep_logs:
            num_eps = meta.get("num_episodes", len(ep_logs))
            lcb_prog = meta.get("lcb_progression")
            prog_extra = (
                f", lcb_prog={lcb_prog:.3f}" if isinstance(lcb_prog, (int, float)) else ""
            )
            lines = [f"  - {task_id} (avg_prog={avg_prog:.3f}{prog_extra}, episodes={num_eps}):"]
            for ep in ep_logs[:5]:
                ep_idx = ep.get("episode_idx", "?")
                prog = ep.get("progression", 0)
                steps = ep.get("num_steps", len(ep.get("step_traces", [])))
                invalid = len(ep.get("failed_candidates", []))
                lines.append(f"      ep{ep_idx}: progression={prog:.3f} steps={steps} invalid={invalid}")
            if len(ep_logs) > 5:
                lines.append(f"      ... and {len(ep_logs) - 5} more episodes")
            return lines

        # --- AgentDojo: category in (utility, security) ---
        category = meta.get("category")
        if category in ("utility", "security"):
            parts = [f"  - {task_id} [{category}]:"]
            utility_result = meta.get("utility_result")
            injection_succeeded = meta.get("injection_succeeded")
            attack_type = meta.get("attack_type")
            injection_goal = meta.get("injection_goal")
            error = meta.get("error") or r.get("error") or r.get("reason", "")

            if utility_result == "failed" or (category == "utility" and not injection_succeeded and not error):
                parts[0] += " utility_failed"
            elif utility_result:
                parts[0] += f" {utility_result}"

            if injection_succeeded:
                inj_parts = ["injection_succeeded"]
                if attack_type:
                    inj_parts.append(f"attack={attack_type}")
                if injection_goal:
                    inj_parts.append(f'goal="{self._truncate(injection_goal, 80)}"')
                parts[0] += ", " + ", ".join(inj_parts)

            if error:
                parts[0] += f", error={self._truncate(error)}"
            return parts

        # --- Terminal-Bench: tb2_reward or tb2_category ---
        if "tb2_reward" in meta or "tb2_category" in meta:
            tb2_reward = meta.get("tb2_reward")
            tb2_category = meta.get("tb2_category", "")
            error = meta.get("error") or meta.get("exception") or r.get("error", "")
            parts = [f"  - {task_id}"]
            if tb2_category:
                parts[0] += f" [{tb2_category}]"
            detail_parts = []
            if tb2_reward is not None:
                detail_parts.append(f"reward={self._fmt_float(tb2_reward, '.2f')}")
            if error:
                detail_parts.append(f"error={self._truncate(error)}")
            if detail_parts:
                parts[0] += ": " + ", ".join(detail_parts)
            return parts

        # --- Classification: ground_truth + prediction ---
        if "ground_truth" in meta and "prediction" in meta:
            prediction = meta["prediction"]
            ground_truth = meta["ground_truth"]
            point_error = meta.get("point_error")
            parts = [f"  - {task_id}: predicted={prediction}, expected={ground_truth}"]
            if point_error is not None:
                parts[0] += f", point_error={self._fmt_float(point_error, '.1f')}"
            return parts

        # --- Genesis: training_success or avg_fitness ---
        if "training_success" in meta or "avg_fitness" in meta:
            training_ok = meta.get("training_success")
            eval_ok = meta.get("eval_success")
            avg_fitness = meta.get("avg_fitness")
            env_name = meta.get("env_name", "")
            label = f"{env_name}/{task_id}" if env_name else task_id
            parts = [f"  - {label}:"]
            detail_parts = []
            if training_ok is False:
                detail_parts.append("training_failed")
            elif training_ok is True:
                detail_parts.append("training_ok")
            if eval_ok is False:
                detail_parts.append("eval_failed")
            elif eval_ok is True:
                detail_parts.append("eval_ok")
            if avg_fitness is not None:
                detail_parts.append(f"fitness={self._fmt_float(avg_fitness, '.3f')}")
            if detail_parts:
                parts[0] += " " + ", ".join(detail_parts)
            return parts

        # --- Polyglot: status + language ---
        if "status" in meta and "language" in meta:
            status = meta["status"]
            language = meta["language"]
            test_output = meta.get("test_output", "")
            parts = [f"  - {task_id} [{language}]: status={status}"]
            if test_output:
                parts.append(f"      test_output: {self._truncate(test_output)}")
            return parts

        # --- Fallback: task_id + reason ---
        reason = r.get("reason", r.get("error", ""))
        if reason:
            return [f"  - {task_id}: {self._truncate(reason)}"]
        return [f"  - {task_id}"]

    def _format_compact_with_log_hint(self: "AgentActionExecutor", reward, metrics: Dict[str, Any], evaluate_seq: int, iteration: int) -> str:
        """Return a compact evaluation result (reward + pass/fail count + failed task detail) + log path hint."""
        task_results = metrics.get("task_results", [])
        failed = [r for r in task_results if not r.get("success")]
        passed_count = len(task_results) - len(failed)

        lines = self._format_reward_header(reward, passed_count, len(task_results), metrics)

        # Show adaptive threshold info if present
        handler = self._get_log_summary_handler()
        if handler:
            at_line = handler.format_adaptive_threshold(metrics)
            if at_line:
                lines.append(at_line)
            at_detail = handler.format_adaptive_threshold_detail(metrics)
            if at_detail:
                lines.append("")
                lines.append(at_detail)

        if failed:
            failed_lines = []
            for r in failed[:20]:
                failed_lines.extend(self._format_failed_task_detail(r))
            lines.append(f"\nFailed tasks ({len(failed)} total):")
            lines.extend(failed_lines)
            if len(failed) > 20:
                lines.append(f"  ... and {len(failed) - 20} more")
        else:
            lines.append("\nAll tasks passed!")

        lines.append(f"\n{self._build_log_path_hint(metrics)}")

        return "\n".join(lines)
