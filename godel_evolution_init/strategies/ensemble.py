"""Ensemble Strategy — LLM-powered code fusion sub-agent from agent-picked versions.

Delegated by the seed selector via ``pick_seed_ensemble(source_hashes, merge_instructions)``.
The seed-selection agent explicitly chooses which commit hashes to fuse based on
mechanism complementarity. The first hash is checked out as the working base;
additional hashes contribute code to merge in. Then runs a sub-react-loop where
the LLM reads, analyzes, fuses, and commits. NO evaluate — the seed selector
evaluates the returned commit at its own level.
"""
from src.react_loop.archive_strategies import SeedResult
from src.react_loop.utils.message_utils import append_response_to_messages

# ─── FIXED: Strategy metadata (must preserve) ─────────────────────
STRATEGY_NAME = "ensemble"
STRATEGY_DESCRIPTION = "LLM-powered ensemble — delegated code fusion sub-agent from agent-picked versions."
# ─── END FIXED ────────────────────────────────────────────────────


# ─── FIXED: Interface (must preserve) ─────────────────────────────
def strategy(agent, tool_args=None):
    return _strategy(agent, tool_args)
# ─── END FIXED ────────────────────────────────────────────────────


# ─── EVOLVABLE: Strategy configuration ────────────────────────────
DEFAULT_MAX_STEPS = 50

_ENSEMBLE_SYSTEM_PROMPT = """\
# Ensemble Task — Code Fusion / Repair Sub-Agent

You are an **ensemble sub-agent**: your job is to either merge the best elements
from multiple high-performing versions of agent harness code into a single improved
version, OR fix a specific bug in a single version.

You have ONE job: read → analyze → fuse/repair → commit. Do NOT evaluate — the
seed selector will handle evaluation after you return your result.

## Two Modes

### Fusion Mode (2+ source hashes)
Merge complementary mechanisms from multiple versions into one improved version.
Follow the merge directive to prioritize which version's patterns to use where.

### Repair Mode (1 source hash)
Fix a specific bug or issue in a single version. The merge_instructions describe
what's broken and how to fix it. You are debugging, not merging — make targeted
fixes only. Do NOT rewrite the entire file.

## Merge Directive
{merge_instructions}

## Instructions
1. **Read** the code from each version provided below.
2. **Follow the merge directive** — it tells you what to focus on and which
   versions to prioritize for which aspects. In repair mode, it describes the
   specific bug to fix.
3. **Fuse or fix** the best elements into the working directory. In fusion mode,
   combine complementary strengths — do NOT just copy one version verbatim. In
   repair mode, make targeted fixes only — do NOT rewrite entire files.
4. **Commit** your result with `bash` (git add -A; git commit -m "...").
5. **Signal completion.** After committing, reply with a brief summary of what
   you merged/fixed and why — **without calling any tools.** A tool-free response
   is your completion signal. The framework will finalize your commit, and the
   seed selector will evaluate it.

## Constraints
- Only modify files in the working directory (the agent's code directory).
- Preserve the overall architecture and interfaces.
- Keep changes focused on the merge directive — don't rewrite everything.
"""

_ENSEMBLE_USER_TEMPLATE = """\
## Version(s) to {mode} (agent-selected, {n} version(s))

{context}

{version_sections}

{diff_summaries}

## Task

{task_description}

Use `read_file` to inspect current state, `edit_file` for targeted changes,
`write_file` for complete rewrites, and `bash` for git operations.
"""


def _resolve_hash(git_controller, h: str) -> str:
    """Resolve a (possibly abbreviated) commit hash to its full form."""
    try:
        result = git_controller._run_git_command(
            ["rev-parse", "--verify", h], check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return h


def _read_code_at_commit(git_controller, commit_hash: str,
                         py_files: list = None) -> str:
    """Read the full harness code (.py) snapshot at a commit for ensemble.

    Returns each top-level .py file's complete content (not the commit diff)
    so the LLM can fuse whole versions rather than incremental patches.
    ALL .py files are included in full — no truncation. The sub-agent needs
    every line to fuse/repair correctly. Large harnesses may produce
    substantial prompts (up to ~230K chars per commit for typical harnesses);
    the seed agent should limit source_hashes count accordingly.

    Args:
        git_controller: GitController instance.
        commit_hash: The commit to read from.
        py_files: Optional pre-computed list of top-level .py file names.
                  When omitted, discovered from the commit tree (one git
                  ls-tree call). Pre-compute and pass to avoid redundant
                  tree lookups across multiple versions.
    """
    try:
        if py_files is None:
            files = git_controller.get_tracked_files_at_commit(commit_hash, "")
            py_files = sorted(
                f for f in files
                if f.endswith(".py") and "/" not in f and "\\" not in f
            )
        parts = []
        for rel in py_files:
            content = git_controller.get_file_at_commit(rel, commit_hash)
            if not content:
                continue
            parts.append(f"### {rel}\n{content}\n")
        if parts:
            return "\n".join(parts)
    except Exception:
        pass
    return f"[No readable harness code at {commit_hash[:7]}]"


def _git_diff_stat(git_controller, hash_a: str, hash_b: str) -> str:
    """Get git diff --stat between two commits for structural comparison."""
    try:
        result = git_controller._run_git_command(
            ["diff", "--stat", hash_a, hash_b,
             "--", ".", ":(exclude)evolution", ":(exclude).evolution_context",
             ":(exclude).meta_evolution_context"],
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "(no diff available)"
# ─── END EVOLVABLE ────────────────────────────────────────────────


# ─── EVOLVABLE: Strategy logic (modify freely) ────────────────────
def _make_too_few_result(agent):
    """Build a SeedResult for the ensemble:too_few fallback path."""
    current_head = agent.git_controller.get_current_commit() or ""
    reason = (
        f"No valid commits found in source_hashes. "
        f"Falling back to current HEAD."
    )
    return SeedResult(
        git_hash=current_head,
        strategy_hint="ensemble:too_few",
        hypothesis=(
            "## Seed Selection Hypothesis\n\n"
            "**Selected seed**: {hash} (current HEAD)\n\n"
            "**Selection rationale**: {reason}\n\n"
            "**Hypothesis**: Continuing from HEAD until more "
            "versions accumulate.\n\n"
            "**Falsification criteria**: N/A — ensemble was not "
            "performed.\n\n"
            "**Bootstrap**: You may adjust or abandon this "
            "hypothesis based on evidence."
        ).format(hash=current_head[:7], reason=reason),
    )


def _strategy(agent, tool_args=None):
    if tool_args is None:
        tool_args = {}

    source_hashes = tool_args.get("source_hashes", [])
    if not isinstance(source_hashes, list) or len(source_hashes) == 0:
        return _make_too_few_result(agent)

    # Validate each hash is a real commit; filter to valid ones.
    valid_hashes = []
    for h in source_hashes:
        if not isinstance(h, str) or not h.strip():
            continue
        h = h.strip()
        try:
            type_res = agent.git_controller._run_git_command(
                ["cat-file", "-t", h], check=False
            )
            if type_res.returncode == 0 and type_res.stdout.strip() == "commit":
                valid_hashes.append(h)
        except Exception:
            pass

    if len(valid_hashes) == 0:
        return _make_too_few_result(agent)

    max_steps = tool_args.get("max_steps", DEFAULT_MAX_STEPS)
    if not isinstance(max_steps, int) or max_steps < 1:
        max_steps = DEFAULT_MAX_STEPS

    merge_instructions = tool_args.get("merge_instructions", "").strip()
    is_repair = len(valid_hashes) == 1
    if not merge_instructions:
        if is_repair:
            merge_instructions = (
                "No specific directive provided. Analyze the version, identify "
                "any bugs or issues, and make targeted fixes to improve it."
            )
        else:
            merge_instructions = (
                "No specific directive provided. Analyze all versions, identify "
                "the best elements from each, and create a fused version that "
                "combines their complementary strengths."
            )

    # ── Look up rewards from tracker records for display ──────────
    tracker = agent.evolution_tracker
    # new_commit / reward are parallel LISTS (a record's pool may hold several
    # commits) — iterate pool entries so each commit→scalar maps cleanly. Using
    # r.new_commit (a list) as a dict key raises "unhashable type: 'list'" and
    # crashes the ensemble sub-agent before it even starts.
    reward_map = {}
    if tracker and tracker.records:
        for r in tracker.records:
            for entry in r.iter_pool():
                reward_map[entry["new_commit"]] = entry["reward"]

    base_hash = valid_hashes[0]

    try:
        mode_label = "repair" if is_repair else "fusion"
        agent._log(
            f"  [Ensemble] Starting {mode_label} of {len(valid_hashes)} version(s) "
            f"(merge_instructions={'yes' if merge_instructions else 'none'})"
        )

        # Checkout the first hash as base, PRESERVING evolution/
        agent.git_controller._run_git_command(
            ["checkout", base_hash, "--", ".",
             ":(exclude)evolution", ":(exclude).evolution_context",
             ":(exclude).meta_evolution_context"],
            check=True,
        )

        # ── Re-stage evolution/ after partial checkout ──────────────
        # git checkout <tree> -- . :(exclude)evolution leaves evolution/
        # files from the previous HEAD in both working tree and index.
        # This mixed-index state can cause subsequent git add -A / commit
        # to silently drop evolution/ files — git sees index entries for
        # evolution/ that don't match base_hash's tree and may resolve
        # the discrepancy by deleting them during the sub-agent's git ops.
        #
        # Fix: explicitly re-stage evolution/ from the working tree
        # (which preserves the meta-evolved versions from the previous
        # HEAD).  We do NOT checkout from base_hash because the base may
        # be a harness-only commit that predates meta-evolve and carries
        # the original template — that would revert meta-evolve changes.
        agent.git_controller._run_git_command(
            ["add", "evolution/"], check=False
        )

        # ── Build prompts ────────────────────────────────────────────
        # Pre-compute the file list once — all versions share the same
        # harness .py file names, so one ls-tree call suffices.
        py_files = None
        if valid_hashes:
            try:
                files = agent.git_controller.get_tracked_files_at_commit(
                    valid_hashes[0], ""
                )
                py_files = sorted(
                    f for f in files
                    if f.endswith(".py") and "/" not in f and "\\" not in f
                )
            except Exception:
                pass

        version_sections = []
        for h in valid_hashes:
            code = _read_code_at_commit(agent.git_controller, h, py_files=py_files)
            full = _resolve_hash(agent.git_controller, h)
            r = reward_map.get(full, None)
            if r is not None:
                reward_str = f"{r:.4f}" if isinstance(r, (int, float)) else str(r)
                version_sections.append(
                    f"### Version {h[:7]} (reward: {reward_str})\n```\n{code}\n```"
                )
            else:
                version_sections.append(
                    f"### Version {h[:7]} (reward: unknown)\n```\n{code}\n```"
                )

        system_prompt = _ENSEMBLE_SYSTEM_PROMPT.format(
            merge_instructions=merge_instructions,
        )

        # ── Build user prompt with dual-mode awareness ──────────────
        n = len(valid_hashes)
        if is_repair:
            mode = "Repair"
            context = (
                "This version was chosen for REPAIR — a specific bug or issue "
                "needs to be fixed. The merge_instructions in the system prompt "
                "describe what's broken and how to fix it. Make targeted fixes "
                "only — do NOT rewrite the entire file."
            )
            task_description = (
                "Analyze the version above. Follow the merge directive from the "
                "system prompt. Fix the described issue, **commit** your result, "
                "then signal completion by replying with a summary — **without "
                "calling any tools.**"
            )
            diff_summaries = ""  # No diffs for single version
        else:
            mode = "Fuse"
            context = (
                "These versions were explicitly chosen by the seed-selection agent "
                "for their complementary mechanisms — not merely sorted by reward. "
                "The first version below is checked out as the working base; "
                "remaining versions contribute code to merge in."
            )
            task_description = (
                "Analyze each version above. Follow the merge directive from the "
                "system prompt. Fuse the best elements, **commit** your result, "
                "then signal completion by replying with a summary — **without "
                "calling any tools.**"
            )
            # Build diff --stat summaries between adjacent pairs (multi-hash only)
            diff_parts = []
            for i in range(len(valid_hashes) - 1):
                a = valid_hashes[i]
                b = valid_hashes[i + 1]
                stat = _git_diff_stat(agent.git_controller, a, b)
                diff_parts.append(f"#### {a[:7]} ↔ {b[:7]}\n```\n{stat}\n```")
            diff_summaries = (
                "## Structural Differences (git diff --stat between adjacent versions)\n\n"
                + "\n\n".join(diff_parts) if diff_parts
                else "(no adjacent pairs to compare)"
            )

        user_prompt = _ENSEMBLE_USER_TEMPLATE.format(
            n=n,
            mode=mode,
            context=context,
            version_sections="\n\n".join(version_sections),
            diff_summaries=diff_summaries,
            task_description=task_description,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Sub-agent tools: atom scope only (file ops + bash). NO evaluate.
        tools = agent.get_tools(scope="atom")

        def tool_executor(tool_name: str, args: dict) -> str:
            return agent.execute_tool(tool_name, args, scope="atom")

        # ── Run sub-react-loop (no evaluate, pure code fusion) ─────
        for step in range(max_steps):
            response, tool_calls_made, tool_results = agent.react(
                messages=messages,
                tools=tools,
                tool_executor=tool_executor,
            )

            append_response_to_messages(
                messages, response, tool_calls_made, tool_results
            )

            if not tool_calls_made:
                messages.append({
                    "role": "user",
                    "content": (
                        "You did not use any tools in your last response. "
                        "If you have finished the code {mode}, reply with a "
                        "brief confirmation of what you {past_tense}. "
                        "If you still need to make changes, please use the "
                        "available tools (read_file, edit_file, write_file, "
                        "bash) to continue."
                    ).format(
                        mode="repair" if is_repair else "fusion",
                        past_tense="fixed" if is_repair else "merged",
                    ),
                })
                confirm_response, confirm_tool_calls, confirm_results = agent.react(
                    messages=messages,
                    tools=tools,
                    tool_executor=tool_executor,
                )
                append_response_to_messages(
                    messages, confirm_response, confirm_tool_calls, confirm_results
                )

                if not confirm_tool_calls:
                    break

        # ── Commit the result ──────────────────────────────────────
        if is_repair:
            commit_msg = f"[Ensemble] Repaired {base_hash[:7]}"
            strategy_hint = "ensemble:repaired"
            result_label = "repair"
            hypothesis_template = (
                "## Seed Selection Hypothesis\n\n"
                "**Selected seed**: {hash} (ensemble repair)\n\n"
                "**Selection rationale**: Ensemble repaired version "
                "{base} with directive: \"{directive}\". The source "
                "version had a specific issue that needed fixing.\n\n"
                "**Hypothesis**: The repaired version fixes the "
                "identified issue while preserving the source version's "
                "strengths. Initial eval reward should be at or above "
                "the source.\n\n"
                "**Falsification criteria**: If reward is lower than "
                "the source version, the repair may have introduced a "
                "regression.\n\n"
                "**Bootstrap**: You may adjust or abandon this "
                "hypothesis based on evidence."
            )
        else:
            commit_msg = f"[Ensemble] Merged {len(valid_hashes)} versions"
            strategy_hint = "ensemble:fused"
            result_label = "fusion"
            hypothesis_template = (
                "## Seed Selection Hypothesis\n\n"
                "**Selected seed**: {hash} (ensemble fusion)\n\n"
                "**Selection rationale**: Ensemble fused {n} "
                "agent-selected versions ({sources}) with directive: "
                "\"{directive}\". These versions were explicitly "
                "chosen by the seed-selection agent for their "
                "complementary mechanisms — not merely sorted by "
                "reward. Source {base} provides the foundation; "
                "other sources contribute mechanisms the foundation "
                "lacks.\n\n"
                "**Hypothesis**: The fusion's value is whether it "
                "creates a capability combination that no single "
                "source version possesses. Initial eval reward may "
                "be below or above individual sources — this is "
                "NORMAL for fusion. The question is not immediate "
                "reward but whether the evolve loop can build on "
                "the combined mechanisms.\n\n"
                "**Falsification criteria**: If the evolve loop "
                "makes no progress in 2 iterations starting from "
                "this seed, the fusion hypothesis is falsified — "
                "the combined mechanisms may be incompatible or "
                "the bottleneck may be elsewhere.\n\n"
                "**Bootstrap**: You may adjust or abandon this "
                "hypothesis based on evidence discovered during "
                "this iteration. If the fusion underperforms, "
                "the component versions may be incompatible — "
                "consider a different strategy next iteration."
            )

        new_commit = agent.git_controller.create_evolution_commit(
            iteration=agent.iteration,
            message=commit_msg,
            files=None,
        )

        source_list = ", ".join(h[:7] for h in valid_hashes)
        base_short = base_hash[:7]

        if new_commit:
            agent._log(f"  [Ensemble] Committed {result_label}: {new_commit[:7]}")
            return SeedResult(
                git_hash=new_commit,
                strategy_hint=strategy_hint,
                hypothesis=hypothesis_template.format(
                    hash=new_commit[:7], n=len(valid_hashes),
                    sources=source_list, base=base_short,
                    directive=merge_instructions[:200],
                ),
                metadata={"sources": valid_hashes},
            )

        agent._log("  [Ensemble] No changes produced, using base")
        return SeedResult(
            git_hash=base_hash,
            strategy_hint="ensemble:no_changes",
            hypothesis=(
                "## Seed Selection Hypothesis\n\n"
                "**Selected seed**: {hash} (ensemble base — no changes)\n\n"
                "**Selection rationale**: Ensemble attempted to {action} {n} "
                "version(s) but produced no meaningful changes — the code "
                "may already be optimal. Using the base source.\n\n"
                "**Hypothesis**: The selected version is already "
                "near-optimal; the base is a reasonable seed.\n\n"
                "**Falsification criteria**: If reward is flat, the "
                "version may be stuck in a local optimum.\n\n"
                "**Bootstrap**: You may adjust or abandon this hypothesis "
                "based on evidence."
            ).format(
                hash=base_hash[:7], n=len(valid_hashes),
                action="repair" if is_repair else "fuse",
            ),
            metadata={"sources": valid_hashes},
        )

    except Exception as e:
        agent._log(f"  [Ensemble] Failed ({e}), restoring base {base_hash[:7]}")
        try:
            agent.git_controller._run_git_command(
                ["checkout", base_hash, "--", ".",
                 ":(exclude)evolution", ":(exclude).evolution_context",
                 ":(exclude).meta_evolution_context"], check=False
            )
        except Exception:
            pass

        return SeedResult(
            git_hash=base_hash,
            strategy_hint="ensemble:error",
            hypothesis=(
                "## Seed Selection Hypothesis\n\n"
                "**Selected seed**: {hash} (ensemble fallback — error)\n\n"
                "**Selection rationale**: Ensemble failed with error: "
                "{error}. Falling back to the base source version.\n\n"
                "**Hypothesis**: The base version is still the "
                "strongest starting point.\n\n"
                "**Falsification criteria**: Standard iteration evaluation.\n\n"
                "**Bootstrap**: You may adjust or abandon this hypothesis "
                "based on evidence."
            ).format(hash=base_hash[:7], error=str(e)[:200]),
            metadata={"error": str(e)},
        )
# ─── END EVOLVABLE ────────────────────────────────────────────────
