"""
Select-Seed Module — Iteration start version selection (the "seed" dimension).

This is one of three independent archive modules (seed / commit / best), each
owning one dimension of version-selection strategy. meta-evolve edits this file
to change how the next iteration *starts*.

## Layering

  FIXED (public interface — meta-evolve must NOT change these):
    - select_seed(agent)          — entry: loads prompts/tools, calls framework runner.
    - get_seed_system_prompt(agent) → str
    - get_seed_user_guidance(agent) → str
    - get_seed_tools(agent) → list[dict]

  EVOLVABLE (strategy — modify freely):
    - _SEED_SELECTION_SYSTEM_PROMPT — role, context, strategy tools, decision rules.
    - _SEED_USER_GUIDANCE — procedural "how to decide" steps (appended to user prompt).
    - _SEED_TOOLS — which strategy tools are enabled + their param overrides.

  FRAMEWORK (not here — meta-evolve cannot edit):
    - src/react_loop/seed_selection.py: run_seed_selection, PICK_SEED_SCHEMA,
      candidate table, react loop, HEAD restoration.

INTERFACE (must preserve):
- select_seed(agent) -> dict

Return format (must match exactly — downstream code reads these keys):

  select_seed(agent) -> dict:
      {
          "git_hash": str,       # Target git commit to checkout. "" = no switch.
          "strategy_hint": str,  # Label for logging (e.g. "greedy", "llm_pick").
          "hypothesis": str,     # Multi-part hypothesis for the evolve agent
                                 # (rationale, expected improvement, falsification
                                 # criteria, bootstrap permission). "" = none.
          "metadata": dict,      # Strategy-specific data (can be empty {}).
          "merge_ops": list,     # [{"source_hash": str, "files": [str, ...]}]. Use [] for none.
      }
      Return {} only as a "do nothing" signal.

agent key properties:
- agent.evolution_tracker: records, get_best_version(), get_nodes(), get_graph_data(), ...
- agent.git_controller: get_current_commit(), ...
- agent._knowledge_graph (optional): edges with llm_diff_analysis, structural_similarity
- agent._log(msg): logging
"""

# ─── FIXED: Public interface (must preserve) ────────────────────────
def select_seed(agent):
    """Select which git version to start the next iteration from.

    Loads the evolvable prompts + tool list, then delegates to the
    framework runner ``run_seed_selection`` which handles the candidate pool,
    react loop, and HEAD restoration.

    Returns:
        dict with keys: git_hash (str), strategy_hint (str),
                        metadata (dict), merge_ops (list)
        Or {} to fall back to the configured archive_strategy.
    """
    from src.react_loop.seed_selection import run_seed_selection

    system_prompt = get_seed_system_prompt(agent)
    user_guidance = get_seed_user_guidance(agent)
    seed_tools = get_seed_tools(agent)
    return run_seed_selection(
        agent, system_prompt, seed_tools, user_guidance=user_guidance,
    )


def get_seed_system_prompt(agent) -> str:
    """Return the system prompt for the seed-selection react loop.

    Stable role/context/strategy framing — cached by the prompt cache.
    meta-evolve edits ``_SEED_SELECTION_SYSTEM_PROMPT`` to change this.
    """
    return _SEED_SELECTION_SYSTEM_PROMPT


def get_seed_user_guidance(agent) -> str:
    """Return the procedural guidance appended to the user prompt.

    Per-instance "how to decide" steps — separate from the system prompt so
    meta-evolve can edit them independently, and so dynamic config values
    (eval budget) can be filled in by the FIXED layer.

    The ``{{seed_eval_step}}`` placeholder is filled from agent.config.
    """
    seed_eval_enabled = getattr(agent.config, "seed_eval_enabled", False)
    seed_eval_max = getattr(agent.config, "seed_eval_max_calls", 1)

    if seed_eval_enabled:
        eval_step = (
            f"\n**Evaluate NEW versions only.** Candidates in the table already "
            f"have known rewards — re-evaluating a candidate whose code hasn't "
            f"changed is wasteful. Only call `evaluate(eval_mode=\"dev\")` if you "
            f"create or discover a NEW version (e.g. via ensemble merge) that has "
            f"no reward yet. You have up to {seed_eval_max} evaluate call(s).\n\n"
            f"**Make your decision** by calling `pick_seed(git_hash=\"...\", "
            f"strategy_hint=\"...\")` with your chosen commit from the candidate "
            f"table (or a new hash from ensemble)."
        )
    else:
        eval_step = (
            f"\n**Make your decision** by calling `pick_seed(git_hash=\"...\", "
            f"strategy_hint=\"...\")` with your chosen commit from the candidate "
            f"table (or a new hash from ensemble)."
        )

    return _SEED_USER_GUIDANCE + "\n" + eval_step


def get_seed_tools(agent) -> list:
    """Return the list of evolvable strategy tool configs.

    Each entry is a dict with at minimum:
        {"name": str, "enabled": bool, ...optional param overrides...}

    meta-evolve can enable/disable tools, add new ones (after creating the
    corresponding evolution/strategies/<name>.py), or adjust parameters
    (e.g. ensemble's k).
    """
    return _SEED_TOOLS
# ─── END FIXED ──────────────────────────────────────────────────────


# ─── EVOLVABLE: System prompt (modify freely) ──────────────────────
_SEED_SELECTION_SYSTEM_PROMPT = """\
# Seed Selection

You are choosing which git version to start the next evolution iteration from.
Your job: pick the version with the greatest *potential* — the one most likely
to lead to improvement, not necessarily the one with the highest reward.

## Core Mental Model: Continue or Pivot?

Every seed selection is a choice between two strategies:

- **Continue (exploit):** Stay on the current trajectory. Pick the best version
  from the current lineage and let the evolve loop keep improving it. This is
  the right move when reward is still climbing — the current direction has
  momentum and the evolve loop can squeeze more out of it.

- **Pivot (explore):** Start from a new combination. Either fuse complementary
  versions via ensemble, or pick a version from a different branch that was
  abandoned but had a different architectural approach. This is the right move
  when the current trajectory has plateaued — continuing the same direction
  would waste an iteration.

The core question: **is the current trajectory still producing improvements,
or has it exhausted its potential?**

## Task-Specific Strategy (per-benchmark, tuned by meta-evolve)

This section is EMPTY on cold start. Meta-evolve fills it for THIS benchmark's
reward structure. When present, it OVERRIDES the generic defaults below.

<!-- META-EVOLVE INJECTS HERE. Template:
**Benchmark:** <suite, e.g. minihack>
**Explore/Exploit bias:** explore-heavy | balanced | exploit-heavy
**Knob overrides (override the generic defaults below):**
- plateau_window: <N>   (flat/declining iters before mandatory pivot; lower = pivot sooner = more explore)
- ensemble_trigger: strong | moderate | weak   (complementarity evidence needed to fuse versions)
- early_phase_fraction: <0–1>   (fraction of max_iterations that is "early = explore-friendly")
**Why:** <1-2 lines: why this benchmark rewards this balance>
-->

## Generic defaults (used when Task-Specific Strategy is empty)

Balanced — pivot after **2+** flat/declining iterations, **moderate** ensemble
trigger (fuse when versions show real complementarity), early-third of the run
is explore-friendly. Meta-evolve should replace these with a benchmark-specific
call once it has read the reward structure.

## How to decide

1. **Read the evidence first** — the candidate table, Evolution Lineage tree,
   Cross-Version Correlations, and Lessons from past iterations are all in the
   data below. Study them before any tool call: lineage shows which branches
   were explored/abandoned, correlations show hidden connections (a version
   from iter2 may share architectural DNA with the current best), lessons show
   what hypotheses were confirmed/refuted (a refuted ensemble is a dead end;
   a confirmed-but-underexploited direction has unfinished potential).

2. **Assess the trajectory.** Best reward still climbing → CONTINUE. Flat/declining
   for `plateau_window` iterations → PIVOT. Within `early_phase_fraction` of the
   run affords more exploration; late should converge.

3. **Inspect 2+ candidates** before deciding — `bash` (`git show <hash>:harness.py`,
   `git diff <a> <b>`), `read_file`, `checkout_version(git_hash)` (writes a version's
   files to disk for full inspection; HEAD stays unchanged), `view_node(git_hash)`.

4. **Act.** Continue → `pick_seed` (best from current lineage). Pivot via ensemble →
   `pick_seed_ensemble(source_hashes=[...])` — choose versions with complementary
   mechanisms from different branches; the first hash is the foundation (checked out
   as base, others contribute code to merge). Pivot via branch → `pick_seed` (a pool
   commit marked [POOL], or a version from a different branch). You are making a
   judgment call, not running a flowchart — your reasoning goes in `hypothesis`.

## Guardrails

- **Don't repeat a dead seed.** Same version seeded for `plateau_window`+ flat/declining
  iterations → you MUST pivot. Repeating a seed into a plateau is the only wrong answer.
- **Inspect before deciding.** You MUST inspect 2+ candidates before `pick_seed`.
- **Prefer val-validated versions.** Val is more trustworthy than dev-only.
- **Ensemble eval is a potential signal, not a bar.** A fused version is freshly combined
  and un-iterated; it usually scores AT OR BELOW current best on first eval. Don't reject
  it for that — judge by the combined mechanisms' ceiling, not the opening shot.

## Ensemble: Fusion AND Repair

pick_seed_ensemble works in TWO modes:

- **Fusion (2+ hashes)**: Merge complementary mechanisms from different versions.
  The sub-agent reads all source code and fuses the best elements.

- **Repair (1 hash)**: Fix a specific bug or issue in a single version.
  Use merge_instructions to describe what's broken and how to fix it. The sub-agent
  will make targeted fixes — you evaluate the result and decide.

A fusion that scores below its sources on first eval is NOT automatically a dead end.
Fusions often need adjustment — the sub-agent may have missed a mechanism interaction
or introduced a conflict. If you believe the combined mechanisms are promising:
(1) identify the specific issue via bash/read_file,
(2) call pick_seed_ensemble with 1 hash (repair mode) and merge_instructions
describing what to fix, (3) evaluate the repaired version, (4) repeat if needed.

If after inspection you conclude the fusion is genuinely unpromising (mechanisms
are fundamentally incompatible, not just incorrectly merged), then and only then
reject it and pick a different seed. But do not reject solely because the first
eval score was low — most fusions need at least one round of repair.

## Strategy tools

- **pick_seed_ensemble(source_hashes, merge_instructions)** — Code fusion
  sub-agent. YOU choose which versions to fuse by passing their commit hashes
  in `source_hashes`. The first hash is checked out as the base; additional
  hashes contribute code to merge in. Use this to create a new starting point
  from complementary versions when the current trajectory has stalled.

- **pick_seed(git_hash, strategy_hint, hypothesis, hypotheses)** — Submit
  your seed choice. Use this whether you are continuing (picking the best
  version from the current lineage) or pivoting (picking a version from
  a different branch or an ensemble result).

You MUST explain your reasoning in the `hypothesis` field of `pick_seed`.
Generate 2-4 competing hypotheses with different core claims. Each must make
a distinct, falsifiable prediction. The evolve agent will test them — they are
exploration GUIDES, not final decisions.
"""


_SEED_USER_GUIDANCE = """\
## How to decide

The candidate table, Evolution Lineage, Cross-Version Correlations, and Lessons
from past iterations are all in the data below — no tool calls needed to read
them. Follow the system prompt's framework:

1. **Read that evidence first** — lineage, reward trend, past lesson verdicts,
   your iteration counter (in the Environment section).
2. **Assess trajectory:** climbing → continue; flat for `plateau_window` → pivot.
   Check the **Task-Specific Strategy** section for THIS benchmark's bias and
   knob overrides — they take precedence over the generic defaults.
3. **Inspect 2+ candidates** with `bash`/`read_file`/`checkout_version`/`view_node`.
4. **If pivoting via ensemble**, call `pick_seed_ensemble(source_hashes=[...])`
   (foundation first; complementary mechanisms). It checks out the merged version —
   `evaluate()` will test it.
4a. **If a fused version scores low**, diagnose the conflict with bash/read_file,
    then call `pick_seed_ensemble` with 1 hash (repair mode) to fix it.
    You can repair multiple times — each repair is a targeted fix, not a full re-fusion.
5. **Call `pick_seed(git_hash, strategy_hint, hypothesis, hypotheses)`** with
   2-4 competing, falsifiable hypotheses. Your hypothesis MUST include: why this
   version, your trajectory assessment, what you expect, what would falsify it.
"""
# ─── END EVOLVABLE ──────────────────────────────────────────────────


# ─── EVOLVABLE: Strategy tool configuration (modify freely) ────────
# Each entry: {"name": str, "enabled": bool, ...optional param overrides...}
# - "enabled": true → the tool appears in the react loop.
# - Extra keys become param overrides (e.g. "k": 5 for ensemble).
# To add a new tool:
#   1. Create evolution/strategies/<new_name>.py with strategy(agent) → SeedResult.
#   2. Add {"name": "pick_seed_<new_name>", "enabled": True} to this list.
#   3. The framework picks up the schema from STRATEGY_TOOL_SCHEMAS (if it
#      exists there) or builds a generic one from the strategy metadata.
#
# Framework-provided inspection tools (always available, no config needed):
#   - checkout_version(git_hash) — write a version's files to working tree
#   - view_node(git_hash) — show a version's full KG relations (lineage + correlations)
_SEED_TOOLS = [
    {"name": "pick_seed_ensemble", "enabled": True},
]
# ─── END EVOLVABLE ──────────────────────────────────────────────────
