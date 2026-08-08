"""Recursive Strategy — Always continue from current HEAD.

Builds a linear chain of incremental improvements. Each iteration
starts from the latest commit, continuously refining the strategy.
"""
from src.react_loop.archive_strategies import SeedResult

# ─── FIXED: Strategy metadata (must preserve) ─────────────────────
STRATEGY_NAME = "recursive"
STRATEGY_DESCRIPTION = "Always continue from current HEAD — builds a linear chain of incremental improvements."
# ─── END FIXED ────────────────────────────────────────────────────


# ─── FIXED: Interface (must preserve) ─────────────────────────────
def strategy(agent, tool_args=None) -> SeedResult:
    return _strategy(agent)
# ─── END FIXED ────────────────────────────────────────────────────


# ─── EVOLVABLE: Implementation (modify freely) ────────────────────
def _strategy(agent):
    head = agent.git_controller.get_current_commit() or ""
    # Get current iteration context for richer hypothesis
    tracker = agent.evolution_tracker
    last_reward = None
    if tracker and tracker.records:
        main_records = [r for r in tracker.records if r.is_main_iteration]
        if main_records:
            last_reward = main_records[-1].reward

    reward_context = f" (last reward: {last_reward:.4f})" if last_reward is not None else ""
    return SeedResult(
        git_hash=head,
        strategy_hint="recursive",
        hypothesis=(
            "## Seed Selection Hypothesis\n\n"
            "**Selected seed**: {hash}{reward_context}\n\n"
            "**Selection rationale**: Recursive strategy continues the "
            "linear chain from current HEAD. This assumes incremental "
            "improvement compounds — each iteration builds directly on "
            "the last.\n\n"
            "**Hypothesis**: The most recent iteration's changes provide "
            "the best foundation for the next improvement because they "
            "represent the freshest exploration direction. Continuing "
            "from HEAD preserves momentum.\n\n"
            "**Falsification criteria**: If reward declines for 2+ "
            "consecutive iterations, the linear chain may be diverging. "
            "If reward is flat while an earlier version had higher "
            "reward, greedy backtracking may be better.\n\n"
            "**Bootstrap**: You may adjust or abandon this hypothesis "
            "based on evidence discovered during this iteration. If "
            "the current trajectory is not working, consider requesting "
            "a greedy or ensemble strategy next iteration."
        ).format(hash=head[:7], reward_context=reward_context),
    )
# ─── END EVOLVABLE ────────────────────────────────────────────────
