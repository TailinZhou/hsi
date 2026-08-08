"""Greedy Strategy — Always start from the historical best version.

Selects the version with the highest reward as the seed for each iteration,
ensuring every iteration builds on proven success rather than the latest commit.
"""
from src.react_loop.archive_strategies import SeedResult

# ─── FIXED: Strategy metadata (must preserve) ─────────────────────
STRATEGY_NAME = "greedy"
STRATEGY_DESCRIPTION = "Always start from the historical best version — maximises reward per iteration."
# ─── END FIXED ────────────────────────────────────────────────────


# ─── FIXED: Interface (must preserve) ─────────────────────────────
def strategy(agent, tool_args=None) -> SeedResult:
    return _strategy(agent)
# ─── END FIXED ────────────────────────────────────────────────────


# ─── EVOLVABLE: Implementation (modify freely) ────────────────────
def _strategy(agent):
    best = agent.evolution_tracker.get_best_version("highest_reward")
    if best is None:
        current_head = agent.git_controller.get_current_commit() or ""
        return SeedResult(
            git_hash=current_head,
            strategy_hint="greedy:cold_start",
            hypothesis=(
                "## Seed Selection Hypothesis\n\n"
                "**Selected seed**: {hash} (iteration 0, cold start)\n\n"
                "**Selection rationale**: No historical data yet — "
                "starting from the initial commit is the only option.\n\n"
                "**Hypothesis**: The initial code provides a baseline. "
                "Improvements will come from the first evolution iteration.\n\n"
                "**Falsification criteria**: N/A — first iteration, no "
                "prior to compare against.\n\n"
                "**Bootstrap**: You may adjust or abandon this hypothesis "
                "based on evidence discovered during this iteration."
            ).format(hash=current_head[:7]),
        )

    best_hash, best_reward = best

    # Checkout so working tree matches the returned hash — consistent
    # with ensemble's behavior. Preserve evolution/ (meta-evolve territory).
    try:
        agent.git_controller._run_git_command(
            ["checkout", best_hash, "--", ".",
             ":(exclude)evolution", ":(exclude).evolution_context",
             ":(exclude).meta_evolution_context"],
            check=True,
        )
        agent._log(f"  [Greedy] Checked out best version {best_hash[:7]}")
    except Exception as e:
        agent._log(f"  [Greedy] Checkout failed ({e}), hash still valid")

    return SeedResult(
        git_hash=best_hash,
        strategy_hint="greedy",
        hypothesis=(
            "## Seed Selection Hypothesis\n\n"
            "**Selected seed**: {hash} (best reward: {reward:.4f})\n\n"
            "**Selection rationale**: This is the historically "
            "highest-reward version. Greedy selection assumes that "
            "building on proven success is the safest path to further "
            "improvement.\n\n"
            "**Hypothesis**: Starting from the best-known version will "
            "produce reward ≥ current best ({reward:.4f}) because the "
            "evolve agent can identify and amplify what already works.\n\n"
            "**Falsification criteria**: If reward does not improve "
            "after 2 iterations seeded from this version, the greedy "
            "assumption may be wrong — the agent may be overfitting to "
            "dev or stuck in a local optimum. Consider ensemble "
            "strategy.\n\n"
            "**Bootstrap**: You may adjust or abandon this hypothesis "
            "based on evidence discovered during this iteration. If you "
            "find contrary evidence, consider requesting a different "
            "seed strategy next iteration."
        ).format(hash=best_hash[:7], reward=best_reward),
        metadata={"reward": best_reward},
    )
# ─── END EVOLVABLE ────────────────────────────────────────────────
