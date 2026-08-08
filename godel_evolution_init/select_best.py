"""
Select-Best Module — Final best-version selection STRATEGY (the "submit_best" dimension).

A FIXED, non-evolvable strategy. This file ships in the init template
(godel_evolution_init/select_best.py) and is loaded AS-IS every run by
``ArchiveManager._load_best_module()`` — it is NEVER copied to evolution/, so
meta-evolve (whose edit sandbox is restricted to evolution/) physically cannot
edit it. A developer tunes it here directly.

Unified with select_seed.py — same FIXED interface pattern (system_prompt,
user_guidance, strategy_tools), same framework runner pattern. The difference
is the GOAL: seed picks the most promising starting point; best picks the
strongest (or fuses a better) final submission.

The react MECHANISM (the loop, the ``submit_best_pick`` tool, the
fusion-must-be-evaluated invariant, the post-fusion import smoke, HEAD
restoration) lives in the framework layer
``src.react_loop.submit_best.run_submit_best``, NOT in this file and NOT editable.

INTERFACE:
- select_best(agent) -> dict

Return format (must match exactly — downstream code reads these keys):

  select_best(agent) -> dict:
      {
          "commit_hash": str,    # A git commit to submit/export as final best.
          "submit_hint": str,    # "best_single" | "llm_pick" | "fusion:validated".
          "metadata": {
              "reward": float,         # scalar reward of the chosen commit.
              "fusion_sources": list,  # source commits — only when fusion was used.
              "validated": bool,       # whether a fusion was evaluate()-validated.
          },
      }
      Return {} to fall back to the default best-version logic (get_best_version).

## Layering

  FIXED (public interface — this whole file is non-evolvable; developer-edited):
    - select_best(agent)              — entry: loads prompts/tools, calls framework runner.
    - get_best_system_prompt(agent)   → str
    - get_best_user_guidance(agent)   → str
    - get_best_tools(agent)           → list[dict]
    - _SUBMIT_BEST_SYSTEM_PROMPT      — role, context, strategy tools, decision rules.
    - _BEST_USER_GUIDANCE             — procedural "how to decide" steps (appended to user prompt).
    - _BEST_TOOLS                     — which strategy tools are enabled + their param overrides.
    - _get_candidates(agent)          — which versions are eligible to be submitted.

  FRAMEWORK (not here — meta-evolve cannot edit):
    - src/react_loop/submit_best.py: run_submit_best, SUBMIT_BEST_PICK_SCHEMA,
      candidate table, react loop, fusion guard, post-fusion import smoke, HEAD restoration.

select_best works by REACT: the prompts + tool config are handed to
run_submit_best, which runs the react loop at the end of every evolution (the
choosing agent inspects code, may call ensemble to fuse, evaluates, and submits
via submit_best_pick). Two paths never enter the react loop: (1) no candidates →
returns {} → caller falls back to max-reward; (2) single candidate →
run_submit_best short-circuits to best_single without a react loop.

agent key properties:
- agent.evolution_tracker: records, get_best_version(), get_nodes(), get_graph_data(), ...
- agent.git_controller: get_current_commit(), ...
- agent._knowledge_graph (optional): edges with llm_diff_analysis, structural_similarity
- agent._log(msg): logging
"""

# ─── FIXED: Public interface (must preserve) ────────────────────────
def select_best(agent):
    """Select which version to submit/export as the FINAL best after evolution ends.

    FIXED orchestration: loads the evolvable prompts + tool config, filters
    candidates via ``_get_candidates``, then hands everything to the framework
    runner ``run_submit_best``. Returning ``{}`` falls back to the hardcoded
    ``get_best_version("highest_reward")``.

    Returns:
        dict with keys: commit_hash (str), submit_hint (str),
                        metadata (dict with "reward")
        Or {} to use the default best-version selection.
    """
    from src.react_loop.submit_best import run_submit_best

    candidates = _get_candidates(agent)
    if not candidates:
        return {}

    system_prompt = get_best_system_prompt(agent)
    user_guidance = get_best_user_guidance(agent)
    strategy_tools = get_best_tools(agent)

    # Environment context is injected into the system prompt by the framework
    # layer (run_submit_best → build_environment_context), so we pass the raw
    # prompts here.
    return run_submit_best(
        agent,
        candidates,
        system_prompt,
        strategy_tools,
        user_guidance=user_guidance,
    )


def get_best_system_prompt(agent) -> str:
    """Return the system prompt for the submit-best react loop.

    Stable role/context/strategy framing — cached by the prompt cache.
    A DEVELOPER edits ``_SUBMIT_BEST_SYSTEM_PROMPT`` to change this — this
    whole file is non-evolvable (meta-evolve cannot reach it; it is never
    copied to evolution/).
    """
    return _SUBMIT_BEST_SYSTEM_PROMPT


def get_best_user_guidance(agent) -> str:
    """Return the procedural guidance appended to the user prompt.

    Per-instance "how to decide" steps — separate from the system prompt so
    meta-evolve can edit them independently.
    """
    return _BEST_USER_GUIDANCE


def get_best_tools(agent) -> list:
    """Return the list of evolvable strategy tool configs.

    Each entry is a dict with at minimum:
        {"name": str, "enabled": bool, ...optional param overrides...}

    meta-evolve can enable/disable tools, add new ones (after creating the
    corresponding evolution/strategies/<name>.py), or adjust parameters
    (e.g. ensemble's k).
    """
    return _BEST_TOOLS
# ─── END FIXED ──────────────────────────────────────────────────────


# ─── EVOLVABLE: Strategy (modify freely) ────────────────────────────
_SUBMIT_BEST_SYSTEM_PROMPT = """\
# Submit-Best Selection

Evolution has ended. You are choosing which version to SUBMIT as the final best.
The exported version is later evaluated on a held-out test set, so GENERALIZATION
matters — not just the single highest dev reward.

## Strategy tools

You have access to an algorithmic strategy tool:

- **pick_seed_ensemble(source_hashes, merge_instructions)** — delegates code
  fusion to a sub-agent. You specify `source_hashes` (which versions to merge)
  and a `merge_instructions` directive telling the sub-agent WHAT to focus on
  (e.g. "take error handling from X, prompt structure from Y"). The sub-agent
  reads, fuses, and commits — then returns the hash. Unlike seed selection,
  where a fusion is a high-potential *starting point* that gets an iteration to
  mature, here the goal is to **merge out a version that BEATS the best single
  candidate**. That is why you **MUST `evaluate` the result before submitting**:
  the evaluated reward is the only thing that puts the fusion on the same
  comparison axis as every other version — an unevaluated fusion is a
  reward-less commit and can never be justifiably picked over a measured one.
  Use only when complementary strengths genuinely combine into something better
  than any single version.

## Inspection tools

- **checkout_version(git_hash)** — write a candidate's files to the working tree
  (HEAD unchanged) so you can read or evaluate its actual code.
- **view_node(git_hash)** — deep-dive ONE version's complete graph relations: its
  reward/summary, lineage neighbors (parent + children), and cross-version
  correlation edges with their similarity and LLM diff analyses. Use this to
  inspect a specific candidate in depth instead of reading the whole graph.
- `bash` (`git show <hash>:harness.py`, `git diff <a> <b> -- harness.py`) and
  `read_file` for anything else.

The candidate table lists the candidates with their rewards, a lineage tree, and
cross-version correlations — enough to see the landscape; use `view_node` to
deep-dive a specific version and `checkout_version` to read its code.

## Decision rules

1. **High dev/val performance AND generalization — both required.** The exported
   version is scored on a held-out test set, so you need a version that is BOTH
   strong on dev/val AND likely to generalize. Do NOT pick a single high dev
   spike that overfits (it won't transfer to test), and do NOT pick a
   consistently mediocre version just because it's stable. Favor versions with
   high absolute dev/val reward that are ALSO consistent across tasks and
   val-validated. Generalization and raw performance are co-objectives, not a
   trade-off — a version must clear a high bar on both.

2. **Prefer val-validated versions.** A version with a val evaluation is more
   trustworthy than one with only dev evals.

3. **Single pick is the default.** Choose the clearly-best candidate unless
   distinct, complementary strengths across versions genuinely combine into
   something better. Fusion adds risk — only fuse when the benefit is clear.

4. **Fusion MUST be evaluated — it's the only way to compare.** A freshly fused
   version has no reward until you `evaluate` it. Without that reward it cannot
   be compared to the candidates, so you could never justify picking it over a
   measured one. You MUST `evaluate` any fusion before submitting, and it has to
   EARN its place: only submit a fusion whose evaluated reward is competitive
   with — ideally better than — the best single candidate, and that still clears
   the generalization bar from rule 1. Never submit merged code whose
   performance you haven't measured.

5. **When uncertain, default to the single highest-reward candidate.** The
   candidate table's Reference line names it ("Best historical reward"); call
   `submit_best_pick` with that hash. It's the safest bet when no clear
   alternative stands out.
"""


_BEST_USER_GUIDANCE = """\
## How to decide

1. **Inspect candidates** — deep-dive a version with `view_node(git_hash)` (its
   lineage neighbors + correlation edges + LLM analyses) and
   `checkout_version(git_hash)` to read its actual code; or use `bash`
   (`git show <hash>:harness.py`, `git diff <a> <b> -- harness.py`) / `read_file`
   to compare code quality.

2. **Fuse only when complementary.** Call
   `pick_seed_ensemble(source_hashes=[...], merge_instructions="...")` to combine
   genuinely complementary strengths from different versions.

3. **Evaluate any fusion, then compare.** If you use ensemble to create a merged
   version, call `evaluate(eval_mode="dev")` to give it a reward — that reward
   is how it competes with the candidates. A fusion is accepted ONLY if you've
   evaluated it AND it is competitive with (ideally better than) the best single
   candidate. An unevaluated fusion has no reward and cannot be compared, so it
   can never be picked.

4. **Make your decision** by calling `submit_best_pick(commit_hash="...",
   submit_hint="...")` with your chosen commit (a candidate hash, or a
   fusion hash you've validated).
"""


# Mirrors select_seed._SEED_TOOLS: ensemble only. greedy was removed — the
# highest-reward candidate is always visible in the candidate table (and named
# in its Reference line), so a greedy tool is redundant; pick it directly via
# submit_best_pick.
_BEST_TOOLS = [
    {"name": "pick_seed_ensemble", "enabled": True},
]
# ─── END EVOLVABLE ──────────────────────────────────────────────────


# ─── EVOLVABLE: Candidate pool (modify freely) ──────────────────────
def _get_candidates(agent):
    """Which versions are eligible to be submitted as best.

    Default: every main-line iteration's final commit with a positive reward.
    ``is_main_iteration`` excludes meta / submit_best_fusion / ensemble records
    (single source of truth shared with resume-state and get_iteration).

    NOTE: use ``r.primary_reward()`` (a scalar — the max of the reward pool), NOT
    ``r.reward`` (which is a *list*, the pool). ``reward_to_scalar`` returns 0.0
    for a list, so filtering on ``r.reward`` would silently exclude every record
    and select_best would never activate.

    Evolve this to widen/narrow the pool: a different reward threshold, an
    iteration range, including ensembles, etc.
    """
    from src.react_loop.state import reward_to_scalar
    tracker = agent.evolution_tracker
    records = tracker.records or []
    return [
        r for r in records
        if r.is_main_iteration and r.new_commit
        and reward_to_scalar(r.primary_reward()) > 0
    ]
# ─── END EVOLVABLE ──────────────────────────────────────────────────
