"""
Submit-Best Runner — framework-layer MECHANISM for final-best selection.

A FIXED, non-evolvable stage that runs a react loop at the end of EVERY
evolution to pick which version to export as the final best (with optional
ensemble fusion). The strategy (prompts, candidate pool, tool config) lives in
the FIXED init template godel_evolution_init/select_best.py, loaded as-is every
run — NOT under evolution/, so meta-evolve cannot edit it.

Unified with seed_selection.py — same react loop pattern, same tool scope
({read_file, bash, evaluate}), same strategy-tool bridge. The differences
are the decision tool (submit_best_pick), the fusion-must-be-evaluated
guard, and the post-fusion import smoke.

This module is NOT under evolution/ — meta-evolve cannot edit it. A developer
who wants a different selection strategy edits the init template
select_best.py: tune the prompts, change the candidate pool, or replace the
react call with a pure rule.

Invariants enforced here (not evolvable):
  - 0 candidates -> {} ; 1 candidate -> short-circuit (no react).
  - >=2 candidates -> react loop in the ``submit_best`` scope + submit_best_pick.
  - A fusion is accepted ONLY after the agent evaluated ITS OWN commit (an
    eval_results entry whose commit == the fusion hash — an eval of a different
    commit doesn't count) AND its harness imports cleanly (post-fusion smoke).
  - Every exit path restores HEAD / working tree.
"""
from typing import Dict, List, Any

from .state import reward_to_scalar
from .utils import log_format
from .utils.message_utils import append_response_to_messages
from .utils.harness_loader import HarnessLoader
from .seed_selection import (
    build_environment_context,
    _build_candidate_table,
    _build_strategy_tool_schemas,
    _build_strategy_tool_executors,
    _discover_extra_strategies,
    CHECKOUT_VERSION_SCHEMA,
    VIEW_NODE_SCHEMA,
    _exec_checkout_version,
    _exec_view_node,
)


# submit_best_pick — the tool the choosing agent calls to commit its decision.
SUBMIT_BEST_PICK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_best_pick",
        "description": (
            "Submit your final best-version decision. Call this once you are satisfied "
            "with your pick — either a single candidate commit, or a fusion you built "
            "(via an ensemble strategy tool) and evaluated. Ends the selection react loop."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "commit_hash": {
                    "type": "string",
                    "description": (
                        "The chosen git commit hash — either a candidate hash (single pick) "
                        "or a fusion commit returned by a strategy tool."
                    ),
                },
                "submit_hint": {
                    "type": "string",
                    "description": "Short reason label, e.g. 'llm_pick' or 'fusion:validated'.",
                },
                "fusion_sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "If this is a fusion, list the source commit hashes that were combined.",
                },
            },
            "required": ["commit_hash"],
        },
    },
}


def run_submit_best(
    agent,
    candidates,
    system_prompt: str,
    strategy_tools: List[dict],
    user_guidance: str = "",
    max_steps: int = None,
) -> dict:
    """Run the submit_best react loop; return the select_best result dict.

    Unified with ``run_seed_selection`` — same pattern, same tool scope,
    different decision tool and fusion guard.

    Args:
        agent: GodelAgent instance.
        candidates: list of EvolutionRecord — the iteration-final commits.
        system_prompt: str — role/strategy framing (evolvable, → system message).
        strategy_tools: list of evolvable tool configs from _BEST_TOOLS
                    (each: {"name": str, "enabled": bool, ...}).
        user_guidance: str — procedural "how to decide" steps (evolvable,
                    appended to the candidate table in the user message).
        max_steps: optional budget; defaults to agent.config.submit_best_max_steps.

    Returns:
        {} on no-decision/fallback, else {commit_hash, submit_hint, metadata{...}}.
    """
    if max_steps is None:
        max_steps = getattr(agent.config, "submit_best_max_steps", 50)
    kg = getattr(agent, "_knowledge_graph", None)
    has_kg = bool(kg is not None and getattr(kg, "edges", None))
    dry_run = getattr(agent, "_submit_best_dry_run", False)

    # ── Phase banner (before short-circuit, so every path is visible) ──
    info_parts = [f"Candidates: {len(candidates)}"]
    if not candidates:
        info_parts.append("fallback (empty)")
    elif len(candidates) == 1:
        info_parts.append("auto-pick (single)")
    else:
        info_parts.append(f"max {max_steps} steps")
    if has_kg:
        info_parts.append("KG: yes")

    if not dry_run:
        log_format.log_phase_banner(
            agent,
            f"SELECT BEST (after iter {agent.iteration})",
            info="  " + " | ".join(info_parts),
        )

    # Short-circuit: 0 or 1 candidates — no react loop needed
    if not candidates:
        if not dry_run:
            agent._log("  ✗ No candidates — falling back to max-reward ranking")
        return {}
    if len(candidates) == 1:
        only = candidates[0]
        only_commit = only.primary_commit()
        only_reward = only.primary_reward()
        if not dry_run:
            agent._log(
                f"  ✓ Single candidate {only_commit[:12]} — auto-confirmed "
                f"(reward={reward_to_scalar(only_reward):.4f})"
            )
        return {
            "commit_hash": only_commit,
            "submit_hint": "best_single",
            "metadata": {"reward": reward_to_scalar(only_reward), "validated": False},
        }

    # Read pre-HEAD only when we're entering the react loop (≥2 candidates).
    # 0/1-candidate short-circuits above don't modify the working tree.
    pre_head = agent.git_controller.get_current_commit() or ""

    # ── Build tools: framework base + strategy tools + decision + inspection ──
    extra = _discover_extra_strategies(agent, strategy_tools)
    framework_tools = agent.get_tools(scope="submit_best")
    strategy_schemas = _build_strategy_tool_schemas(agent, strategy_tools, extra=extra)
    strategy_executors = _build_strategy_tool_executors(agent, strategy_tools, extra=extra)
    tools = framework_tools + strategy_schemas + [
        SUBMIT_BEST_PICK_SCHEMA, CHECKOUT_VERSION_SCHEMA, VIEW_NODE_SCHEMA,
    ]

    # ── Build messages ──
    # User prompt = data (candidate table + lineage + correlations) + procedural
    # guidance. Per-node deep-dives are left to the view_node tool (on demand),
    # not dumped inline. System prompt = role/strategy framing (fixed template)
    # + environment (framework).
    user_content = _build_candidate_table(agent, candidates) + user_guidance

    messages = [
        {"role": "system", "content": system_prompt + "\n" + build_environment_context(agent)},
        {"role": "user", "content": user_content},
    ]

    # closure-mutable containers so the nested tool_executor can record state.
    decision: Dict[str, Any] = {}
    # Per-commit eval results: [{"reward": scalar, "commit": full_hash}, ...].
    # Ties each reward to the commit that was actually evaluated (mirrors
    # seed_selection), so a fusion's reward can't be misattributed to a later
    # eval of a different commit.
    eval_results: List[Dict[str, Any]] = []
    done = [False]              # set when submit_best_pick fires

    def tool_executor(name, args):
        # 1. Strategy tool?
        if name in strategy_executors:
            return strategy_executors[name](agent, args)

        # 2. Decision tool?
        if name == "submit_best_pick":
            decision.update(args or {})
            done[0] = True
            return (
                f"Decision recorded: {(args or {}).get('commit_hash', '')[:7]}. "
                f"Selection complete."
            )

        # 3. Inspection tools (checkout_version, view_node)
        if name == "checkout_version":
            return _exec_checkout_version(agent, args)
        if name == "view_node":
            return _exec_view_node(agent, args)

        # 4. Evaluate tool — track reward keyed by the commit actually on disk,
        #    so a fusion's reward is never misattributed to another commit's eval.
        if name == "evaluate":
            res = agent.execute_tool(name, args, scope="submit_best")
            eval_results.append({
                "reward": reward_to_scalar(getattr(agent.state, "reward", None)),
                "commit": agent.git_controller.get_current_commit() or "",
            })
            return res

        # 4. Framework tool (read_file / bash / powershell)
        return agent.execute_tool(name, args, scope="submit_best")

    def _append_status_suffix(step):
        """Append a step-progress suffix to the last tool result."""
        step_pct = int(step / max_steps * 100) if max_steps else 0
        suffix = f"\n— [Best Step {step}/{max_steps}]"
        if step_pct >= 80:
            suffix += " ⚠️ running low — call `submit_best_pick` now"
        elif step_pct >= 50:
            suffix += " ⏳ past halfway — converge; call `submit_best_pick` soon"
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "tool":
                messages[i]["content"] = str(messages[i]["content"]) + suffix
                if not dry_run:
                    agent._log(f"  {suffix.strip()}")
                return

    try:
        for step in range(1, max_steps + 1):
            response, tool_calls_made, tool_results = agent.react(
                messages=messages, tools=tools, tool_executor=tool_executor,
            )
            append_response_to_messages(
                messages, response, tool_calls_made, tool_results
            )
            if not dry_run:
                log_format.log_react_step(
                    agent, step, max_steps, tool_calls_made, tool_results
                )

            if done[0]:
                break

            if tool_calls_made:
                _append_status_suffix(step)

            if not tool_calls_made:
                messages.append({
                    "role": "user",
                    "content": (
                        "You did not use any tools. If you are ready to decide, call "
                        "`submit_best_pick` with your chosen commit_hash. Otherwise use "
                        "`bash`/`read_file`/`evaluate` or consult a strategy tool "
                        "to inspect candidates further."
                    ),
                })
                c_response, c_tool_calls, c_results = agent.react(
                    messages=messages, tools=tools, tool_executor=tool_executor,
                )
                append_response_to_messages(
                    messages, c_response, c_tool_calls, c_results
                )
                if not dry_run:
                    log_format.log_react_step(
                        agent, step, max_steps, c_tool_calls, c_results
                    )
                if done[0]:
                    break
                if not c_tool_calls:
                    break

        # ── Bonus react: one final chance if agent exhausted all steps ──
        if not done[0]:
            agent._log("  [best] max steps reached without decision — final nudge")
            messages.append({
                "role": "user",
                "content": (
                    "⏰ **TIME'S UP.** You have used all {max_steps} steps without "
                    "calling `submit_best_pick`. This is your ABSOLUTE last chance — "
                    "call `submit_best_pick(commit_hash=\"...\")` NOW. If you do not "
                    "decide, your selection will be discarded and the default "
                    "max-reward version used instead."
                ).format(max_steps=max_steps),
            })
            try:
                b_response, b_tool_calls, b_results = agent.react(
                    messages=messages, tools=tools, tool_executor=tool_executor,
                )
                append_response_to_messages(
                    messages, b_response, b_tool_calls, b_results
                )
                if not done[0]:
                    agent._log("  [best] still no decision after bonus react")
            except Exception:
                agent._log("  [best] bonus react failed")

    except Exception as e:
        agent._log(f"  [best] react loop failed: {e}")

    # ── Save submit-best conversation to .evolution_context/select_best/ ──
    _save_best_messages(agent, messages)

    return _finalize_pick(agent, decision, eval_results, candidates, pre_head)


def _save_best_messages(agent, messages) -> None:
    """Persist the submit-best conversation to .evolution_context/select_best/."""
    persistence = getattr(agent, "context_persistence", None)
    if persistence is None:
        return
    if not messages:
        return
    try:
        saved = persistence.save_phase_messages(
            "select_best", agent.iteration, messages
        )
        if saved:
            agent._log(f"  [best] Saved conversation to {saved}")
    except Exception as e:
        agent._log(f"  [best] Failed to save messages: {e}")


def _finalize_pick(agent, decision, eval_results, candidates, pre_head):
    """Validate the agent's submit_best_pick decision and assemble the return dict.

    - No commit_hash, or not a valid commit -> restore + {} (fallback).
    - commit_hash is a candidate -> single pick (accept).
    - commit_hash is a fusion (new hash from ensemble) -> accept ONLY if the agent
      evaluated THIS fusion commit (an eval_results entry matches its hash);
      otherwise reject -> {}. A fusion's reward is taken only from an eval whose
      commit == the fusion hash, never from an eval of some other commit.

    Every path restores HEAD / working tree.
    """
    if eval_results is None:
        eval_results = []
    # Dry-run gate (validate_archive): silence user-facing logs so a stubbed
    # dry-run doesn't leak result lines like "✓ LLM picked ..." (mirrors
    # run_submit_best's _submit_best_dry_run gate).
    dry_run = getattr(agent, "_submit_best_dry_run", False)

    def reject(reason=None):
        if reason and not dry_run:
            agent._log(f"  ✗ {reason}")
        _restore_head(agent, pre_head, accept=False)
        return {}

    h = (decision or {}).get("commit_hash", "")
    if not h:
        return reject("No commit_hash in decision — falling back to max-reward ranking")

    # Must be a real commit
    try:
        type_res = agent.git_controller._run_git_command(
            ["cat-file", "-t", h], check=False
        )
    except Exception as e:
        return reject(f"cat-file failed for {h[:7]}: {e}")
    if not (type_res.returncode == 0 and type_res.stdout.strip() == "commit"):
        return reject(f"{h[:7]} is not a valid commit — rejecting")

    # Normalize to full hash — the LLM may submit a short hash (7–12 chars from
    # display output like view_node / git log --oneline), but candidate_by_hash
    # keys are full 40-char hashes from git rev-parse.  Resolve once up front so
    # both single-pick and fusion paths use the same canonical form.
    try:
        resolve = agent.git_controller._run_git_command(
            ["rev-parse", "--verify", h], check=False
        )
        full_h = resolve.stdout.strip() if resolve.returncode == 0 else h
    except Exception:
        full_h = h

    candidate_by_hash = {entry["new_commit"]: (r, entry)
                         for r in candidates
                         for entry in r.iter_pool()}

    # --- Single pick ---
    if full_h in candidate_by_hash:
        rec, entry = candidate_by_hash[full_h]
        if not dry_run:
            agent._log(
                f"  ✓ LLM picked: {full_h[:12]} "
                f"(reward={reward_to_scalar(entry['reward']):.4f})"
            )
        _restore_head(agent, full_h, accept=True)
        return {
            "commit_hash": full_h,
            "submit_hint": (decision or {}).get("submit_hint") or "llm_pick",
            "metadata": {"reward": reward_to_scalar(entry["reward"]), "validated": False},
        }

    # --- Fusion (new commit from ensemble strategy tool) ---
    # Guard: the fusion commit itself MUST have been evaluated. Only an eval
    # whose commit == the fusion hash counts — an eval of some other commit
    # (e.g. the agent inspected a candidate after fusing) does NOT validate
    # this fusion and must not lend its reward to it. (Mirrors seed_selection's
    # commit-keyed eval tracking.)
    matching_evals = [e for e in eval_results if e.get("commit", "") == full_h]
    if not matching_evals:
        return reject(
            f"fusion {h[:7]} submitted but its own commit was never evaluated "
            f"— rejecting (evaluating a different commit doesn't validate this fusion)"
        )
    fusion_reward = max(e["reward"] for e in matching_evals)

    # Post-fusion import smoke (R2): the eval guard above ties the reward to the
    # fusion commit, but it doesn't verify the fusion's harness actually imports
    # cleanly — a freshly-fused commit whose cross-file imports misalign (e.g.
    # context.py importing a name utils.py doesn't export — the old 757fb4e crash)
    # can still pass an eval that ran against a transient tree. Materialize the
    # fusion's tree and fresh-import the harness from disk; reject on any import
    # error (which resets HEAD to pre_head). Benchmark-agnostic; validates the
    # bytes that will actually be exported.
    smoke_failed = False
    try:
        agent.git_controller._run_git_command(
            ["checkout", h, "--", ".",
             ":(exclude)evolution", ":(exclude).evolution_context",
             ":(exclude).meta_evolution_context"],
            check=False,
        )
        HarnessLoader(agent.agent_code_dir).load(agent)
    except Exception as e:
        smoke_failed = True
        if not dry_run:
            agent._log(f"  ✗ Fusion import smoke failed for {h[:7]}: {e}")
    finally:
        # Purge sys.modules of the fresh-imported harness package so it doesn't
        # leak / go stale for later loads (mirrors adapter._get_harness).
        try:
            HarnessLoader.cleanup_loaded(agent.agent_code_dir)
        except Exception:
            pass
    if smoke_failed:
        return reject("fusion harness failed to import — rejecting")

    sources = (decision or {}).get("fusion_sources") or []
    try:
        agent.evolution_tracker.record_iteration(
            iteration=-agent.iteration,
            parent_commit=pre_head,
            new_commit=full_h,
            reward=fusion_reward,
            state_summary=f"Submit-best fusion of {len(sources) or 'several'} versions",
            action_count=0,
            metadata={
                "operation_type": "submit_best_fusion",
                "sources": sources,
                "validated": True,
                "summary_text": "LLM fusion validated via evaluate()",
            },
        )
    except Exception as e:
        if not dry_run:
            agent._log(f"  ✗ Failed to record fusion node: {e}")

    if not dry_run:
        agent._log(
            f"  ✓ Fusion accepted: {full_h[:12]} "
            f"(reward={fusion_reward:.4f}, {len(sources)} source(s))"
        )
    _restore_head(agent, full_h, accept=True)
    return {
        "commit_hash": full_h,
        "submit_hint": "fusion:validated",
        "metadata": {
            "reward": fusion_reward,
            "fusion_sources": sources,
            "validated": True,
        },
    }


def _restore_head(agent, target, accept):
    """Restore HEAD / working tree on every exit path."""
    try:
        if accept:
            agent.git_controller._run_git_command(
                ["checkout", target, "--", ".",
                 ":(exclude)evolution", ":(exclude).evolution_context",
                 ":(exclude).meta_evolution_context"],
                check=False,
            )
        else:
            agent.git_controller._run_git_command(
                ["reset", "--hard", target], check=False
            )
    except Exception:
        pass
