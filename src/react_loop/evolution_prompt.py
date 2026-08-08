"""
Prompt templates for React Loop Agent evolution.

These prompts guide the agent's decision-making process
and the improver's code generation.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from .state import fmt_reward, fmt_iteration_status, reward_to_scalar

if TYPE_CHECKING:
    from .state import MessageHistory, AgentAction
    from .git_version.controller import EvolutionTracker


def get_framework_contract() -> str:
    """
    Get the FIXED framework contract for the agent — identity, architecture,
    lifecycle, and evaluation rules.

    This is the non-evolvable half of the system prompt. It is concatenated
    BEFORE the evolvable half (evolution_base_prompt.md) to form the full
    system prompt. meta-evolve cannot edit it: it lives in src/, outside the
    evolution/ sandbox (edits to non-evolution/ files are reverted).
    """
    contract_path = Path(__file__).parent / "framework_contract.md"
    with open(contract_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_base_system_prompt() -> str:
    """
    Get the EVOLVABLE half of the system prompt (exploit/explore strategy +
    task evolution strategy).

    This is the fallback for the evolvable half when
    evolution/evolution_base_prompt.md is absent. Callers must prepend
    get_framework_contract() — this is NOT the full system prompt on its own.
    meta-evolve edits the evolution/ copy of this file.
    """
    prompt_path = Path(__file__).parent / "evolution_base_prompt.md"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


# Task-lifecycle text. Lives in the system prompt's Environment (S_t) section
# (static per iteration) rather than re-emitted on every step's user prompt.
# Extracted from the former get_decision_prompt() tail so the lifecycle guidance
# is never lost when the per-step user prompt was slimmed down to Step-1-only.
YOUR_TASK_TEXT = """## Your Task

Work in phases. Do NOT evaluate after every small edit — review your code first.
The seed-selection hypothesis is already in `plan.md` — read it first (`read_file plan.md`).

**1. DIAGNOSE** — Read previous eval failure analysis; identify the single most impactful failure mode. Record your working plan in `plan(plan=...)`.
**2. DESIGN** — ONE hypothesis. Plan which files to change and why.
**3. IMPLEMENT & REVIEW** — Edit code, then `read_file` every changed file to verify correctness and edge cases BEFORE evaluate.
**4. VALIDATE** — `evaluate(eval_mode="dev")`. Read the trace, not just the score. Did the target failure mode disappear? Record progress in `plan(progress=...)`. If it held, `pick_commit_version`.
**5. DECIDE** — Hypothesis confirmed → next failure mode (back to 1). Refuted → diagnose why, pivot. 2–5 cycles is healthy; past 7, you're using evaluate as a debugger. When done: record one cross-iteration `lesson(lesson=..., confidence=...)` (the only memory the next iteration inherits), then `compact_context`."""


def get_iteration_begin_prompt(
    iteration: int,
    state_summary: Dict[str, Any],
    code_baseline_reward: float | dict | None = None,
    repo_path: str = "",
    init_experience: str = "",
    seed_hypothesis: str = "",
    hypotheses: list = None,
) -> str:
    """First (and only) user prompt for an iteration's react loop.

    Injected only at Step 1; subsequent steps continue from tool results with
    no intervening user message (standard OpenAI function-calling continuation).
    Environment, git status, code files, and the task lifecycle already live in
    the system prompt; the per-evaluate reward/history already lives in the
    evaluate tool result. So this prompt carries only Step-1-specific dynamic
    content: the iteration header, the baseline/reward/steps one-liner, the
    init-harness experience (iteration 1 only), and the seed hypothesis.

    Args:
        iteration: Current iteration number.
        state_summary: Summary of current state (steps, max steps, reward).
        code_baseline_reward: Reward from previous iteration's committed code
            (None for the first iteration → shown as "—").
        repo_path: Absolute path to the agent's code directory (used to construct
            absolute paths for read_file instructions).
        init_experience: The init harness's own lived eval feedback, produced by
            the bootstrap init-eval (run_init_eval). Injected ONLY at iteration 1
            to drive the first self-rewrite from measured pain-points instead of
            cold code reading. Empty/absent → section omitted.
        seed_hypothesis: The seed hypothesis from archive.select_seed().
            Not injected in iteration 1 (no seed selection needed).
            Empty → omitted.
        hypotheses: List of competing hypothesis dicts from seed selection.
            When 2+ items, rendered as "Competing Hypotheses" section.
            When empty/None, falls back to legacy seed_hypothesis rendering.
            Not injected in iteration 1.

    Returns:
        Formatted first-user prompt string.
    """
    steps_taken = state_summary.get('steps_taken', 1)
    max_steps = state_summary.get('max_steps_per_iteration', 50)
    reward = state_summary.get('reward')
    reward_display = fmt_reward(reward)

    if code_baseline_reward is not None:
        # Iteration 1's baseline is the init harness's own bootstrap-eval reward
        # (the iteration-0 seed), not a "previous committed" iteration's reward.
        baseline_label = "(init harness)" if iteration == 1 else "(prev committed)"
        baseline_line = f"{fmt_reward(code_baseline_reward)} {baseline_label}"
    else:
        baseline_line = "—"

    # Build hypothesis section: new multi-hypothesis format takes priority
    hyps = hypotheses or []
    if len(hyps) >= 2:
        lines = [
            "\n\n## Competing Hypotheses (from Seed Selection)",
            "",
            "These are NOT established truths — they are competing directions to explore.",
            "Your job: design changes that test them, then update your beliefs based on evidence.",
            "You may confirm, refute, or synthesize them.",
            "",
        ]
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
        hypothesis_section = "\n".join(lines)
    else:
        hypothesis_text = (seed_hypothesis or "").strip()
        hypothesis_section = (
            f"\n\n{hypothesis_text}" if hypothesis_text else ""
        )

    experience = (init_experience or "").strip()
    experience_section = ""
    if experience:
        experience_section = (
            f"\n\n## Init Harness Experience (your harness on the task, before any edits)\n"
            f"{experience}\n\n"
            f"This is your harness's own lived feedback. Rewrite your harness to fix "
            f"what it complains about. Distill the top pain-points into a cross-iteration "
            f"`lesson(lesson=..., confidence=...)` so they survive context reset into iteration 2+."
        )

    return (
        f"## Iteration {iteration} — begin\n"
        f"Improve the harness to maximize reward. Environment, git status, code "
        f"files, and the task lifecycle are in the system prompt above.\n\n"
        f"**Baseline:** {baseline_line}  |  **Reward:** {reward_display}  |  "
        f"**Steps:** {steps_taken}/{max_steps}"
        f"{experience_section}"
        f"{hypothesis_section}\n\n"
        f"Use `read_file(\"{repo_path}/plan.md\")` to read your current iteration "
        f"plan/progress (the seed hypothesis is already written there), and "
        f"`read_file(\"{repo_path}/BOOTSTRAP.md\")` for the accumulated cross-iteration "
        f"lessons (memory that never rolls back)." if repo_path else
        f"Your current iteration plan/progress is in `plan.md` (the seed hypothesis "
        f"is written there) and accumulated cross-iteration lessons are in "
        f"`BOOTSTRAP.md` — both readable via read_file using the absolute paths "
        f"from your Working Directory in the system prompt."
    )


def _build_prompt_from_tracker(
    base_prompt: str,
    tracker: "EvolutionTracker",
    context_dir: str = None
) -> str:
    """Build a system prompt from EvolutionTracker."""
    if not tracker.records:
        return base_prompt

    # Get the best iteration
    best_iteration = tracker.get_best_iteration()
    # Build a pointer + summary per iteration
    iteration_pointers = []
    meta_pointers = []
    for record in tracker.records:
        is_meta = record.metadata.get("type") == "meta_evolve"
        status = fmt_iteration_status(record.metadata)

        summary_text = record.metadata.get("summary_text", "")
        modifications_count = record.metadata.get("modifications_count", 0)

        if is_meta:
            main_iter = record.metadata.get("main_iteration", "?")
            meta_pointers.append(
                f"  [Meta] after main iter {main_iter} (mods: {modifications_count}): {summary_text}"
            )
        else:
            best_marker = "★" if record.iteration == best_iteration else " "
            trajectory_info = ""
            rw_hist = record.metadata.get("reward_history", [])
            if rw_hist:
                rewards = [reward_to_scalar(e.get("reward", 0)) for e in rw_hist]
                trajectory_info = f" [trajectory: {'>'.join(f'{r:.3f}' for r in rewards)}]"
            iteration_pointers.append(
                f"{best_marker} Iteration {record.iteration} [{status}] "
                f"(reward: {fmt_reward(record.primary_reward())}, mods: {modifications_count}){trajectory_info}: {summary_text}"
            )

    pointers_text = "\n".join(iteration_pointers)
    if meta_pointers:
        pointers_text += "\n\n## Meta-Evolve History\n" + "\n".join(meta_pointers)

    context_section = f"""

## Evolution History

**Total:** {len(iteration_pointers)} iterations, {len(meta_pointers)} meta-evolve phases

**Status meanings:**
- SUCCEEDED: Iteration completed all required actions (introspect, evaluate)
- MAX_STEPS: Iteration ended due to step limit (changes committed, not a failure)
- FAILED: Iteration did not complete
- ★ marks the iteration with highest reward

{pointers_text}

Learn from past iterations to improve your evolution strategy."""

# **Read detailed history:**
# - `read_history_self(["@history:N"])` - Read iteration N's conversation (N is 1-based)
# - `read_history_self(["@history:keyword"])` - Search history by keyword (e.g., "error handling", "caching")
# - `read_history_self(["@history:N:keyword"])` - Search keyword within iteration N

# Examples:
# - `read_history_self(["@history:{best_iteration}"])` - Read best iteration (marked with ★)
# - `read_history_self(["@history:caching"])` - Find iterations related to caching
# - `read_history_self(["@history:2:caching"])` - Search "caching" in iteration 2

# **Note:** You can only read COMPLETED iterations. The current iteration is still in progress.

    return base_prompt + context_section


def get_summary_generation_prompt(
    iteration: int,
    modifications: List[Dict[str, Any]],
    reward: float | dict,
    action_count: int,
    action_history: List[str],
    max_steps: int,
    step_count: int = 0,
    success: bool = True,
    max_steps_reached: bool = False,
    agent_summary: str = "",
    end_reason: str = "",
) -> str:
    """
    Get the prompt used to generate the iteration summary.

    The LLM produces summary_text and key_decisions; the statistical info is
    rule-based.

    Args:
        iteration: Iteration number.
        modifications: List of modifications.
        reward: Reward value.
        action_count: Number of actions.
        action_history: Action history (list of strings).
        max_steps: Maximum number of steps.
        step_count: Number of steps.
        success: Whether the iteration succeeded.
        max_steps_reached: Whether the max-step limit was hit.
        agent_summary: Summary provided by the agent.
        end_reason: End reason provided by the agent.

    Returns:
        The summary-generation prompt.
    """
    # Statistical info (rule-based)
    mod_summary = ""
    if modifications:
        mod_types = [m.get("type", "unknown") for m in modifications]
        mod_summary = f"Modifications made: {len(modifications)} ({', '.join(set(mod_types))})"

    action_history_str = "\n".join(action_history) if action_history else "None"

    if success:
        agent_context = f"\n- Agent summary: {agent_summary}" if agent_summary else ""
        agent_context += f"\n- End reason: {end_reason}" if end_reason else ""

        if reward is None:
            reward_display = "None"
        elif isinstance(reward, dict):
            reward_display = ', '.join(f'{k}={v:.4f}' for k, v in reward.items() if isinstance(v, (int, float)))
        else:
            reward_display = f'{reward:.4f}'
        return f"""This iteration has ended. Do NOT call any tools — respond with JSON text only.

Analyze iteration {iteration} and provide:
1. A 1-2 sentence summary of what was accomplished
2. The 3-5 most critical decisions made (key_decisions)

Key facts:
- Reward: {reward_display}
- Steps: {step_count}/{max_steps}, Actions: {action_count}
- {mod_summary}{agent_context}

Action history:
{action_history_str}

Respond in JSON format:
{{"summary_text": "...", "key_decisions": ["...", "..."]}}"""
    else:
        failure_info = "Reached max steps limit" if max_steps_reached else "Did not complete all required actions"

        return f"""This iteration has ended. Do NOT call any tools — respond with JSON text only.

Analyze this FAILED iteration {iteration} and provide:
1. A 1-2 sentence summary explaining the failure and lesson learned
2. The 3-5 most critical decisions that led to failure (key_decisions)

Failure context:
- Reason: {failure_info}
- Steps: {step_count}/{max_steps}, Actions: {action_count}
- Agent summary: {agent_summary or 'N/A'}
- End reason: {end_reason or 'N/A'}

Action history:
{action_history_str}

Respond in JSON format:
{{"summary_text": "...", "key_decisions": ["...", "..."]}}"""
