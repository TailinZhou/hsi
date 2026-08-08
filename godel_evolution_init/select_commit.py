"""
Select-Commit Module — Iteration-end commit nudge prompt.

meta-evolve edits this file to improve the decision guidance the agent sees
when finalizing which evaluated code version to commit.

INTERFACE (must preserve):
- get_commit_nudge_prompt(agent) -> str  # nudge prompt for LLM-driven final confirmation
"""

# ─── FIXED ────────────────────────────────────────────────────────────
def get_commit_nudge_prompt(agent) -> str:
    """Return the nudge prompt for cache-preserving commit finalization."""
    return _COMMIT_NUDGE_PROMPT
# ─── END FIXED ────────────────────────────────────────────────────────


# ─── EVOLVABLE ────────────────────────────────────────────────────────
_COMMIT_NUDGE_PROMPT = """\
You are finalizing the commit pool for this iteration — selecting MULTIPLE
code versions that will become seeds for future iterations.

## Your task

You see every evaluated code version from this iteration, plus your pool
history from the main loop (shown below after this prompt). Your job: select
2-5 versions that represent DIFFERENT exploration directions. For each version,
write a one-line description explaining what changed and why it was selected.
A blind heuristic would just commit the top dev score; the framework trusts your
judgment to build a diverse pool that gives future iterations options.

## Decision rules

1. **Diversity over max reward.** Select versions that represent different
   approaches — e.g. one with prompt improvements, one with structural loop
   changes, one with error handling. Two very similar versions waste a pool slot.

2. **Val > Dev.** Versions with val evaluations are more trustworthy than
   dev-only ones. Prefer val-validated versions.

3. **High-potential over high-reward.** A version with moderate reward but
   an obvious improvement path is more valuable than one that maxed out.

4. **Cover breakthroughs.** If you had a "aha!" moment this iteration,
   ENSURE that version is in the pool — breakthroughs are fragile.

5. **Inspect before deciding.** Use `read_file` or `bash` to diff between
   candidates. Don't pick blind.

6. **Recommended pool size: 2-5 versions.** Too few limits future options;
   too many dilutes the pool with noise.

7. **Write meaningful descriptions.** Each committed version gets a one-line
   summary — future iterations (and the human reading git log) need to
   understand the intent behind each version.

8. **Order matters.** List your primary intended direction FIRST. The first
   version in your `finalize_commit_pool` list is left on disk and inspected
   by meta-evolution as this iteration's representative code, so put your
   main strategic direction first — not necessarily the highest-reward one.
   Keep the rest of the pool diverse.

## Output

When ready, call `finalize_commit_pool(code_hashes=[{"code_hash": "...", "description": "..."}, ...])` with
your selected versions from the evaluation data below. Each entry must
include both the code_hash and a one-line description. List order is
significant: the first entry becomes the on-disk representative that
meta-evolution inspects as this iteration's primary direction, so order
deliberately — your main direction first, diversity for the rest.

{eval_data}

## Commit Pool Balance ★ TUNABLE ★

The thresholds in this section control whether the commit pool is permissive
(more exploration nodes, lower bar) or strict (fewer nodes, higher quality).
Meta-evolve can adjust these to shift commit behavior along the spectrum.

- **Pool size range** (Decision rule 6, above): currently **2-5** versions.
  Raise the upper bound (e.g. 3-8) when exploration is starved — more
  committed versions = more seeds for future iterations to try. Lower
  (e.g. 1-3) when you need quality over quantity — fewer but stronger seeds.
  This is the highest-leverage knob.

- **Diversity vs. reward priority** (Decision rule 1, above): currently
  "Diversity over max reward." When exploration is needed, strengthen this
  to "Diversity first, reward secondary — commit versions with different
  mechanisms even if their raw scores are lower." When high-value commits
  are needed, flip to "Reward first, diversity second — prefer versions
  with proven performance; only diversify within the top tier."

- **Inclusion threshold** (Decision rule 3, above): currently "High-potential
  over high-reward" — implicit, qualitative. Make it explicit: "Commit
  versions even 20% below best reward if they use a different mechanism"
  (permissive) vs. "Commit only versions within 10% of best reward" (strict).

- **Val gate** (Decision rule 2, above): currently "Prefer val-validated."
  Soften to "Dev-only is acceptable for novel mechanisms" when exploration
  nodes are scarce. Harden to "Only commit val-validated versions" when
  quality matters most and dev-overfit is a risk.

When editing: change ONE knob per meta-evolve round, make a falsifiable
prediction about how it will shift commit-pool composition, and verify
via `select_commit/iter_N.json` logs in the next round.
"""
# ─── END EVOLVABLE ────────────────────────────────────────────────────
