"""
Seed-Selection Runner — framework-layer MECHANISM for iteration-start version selection.

Holds the react loop, the ``pick_seed`` tool schema, strategy-tool bridge executors,
candidate table + KG diff, and HEAD restoration. The STRATEGY (which tools are enabled,
how the choosing agent is instructed) lives in the evolvable
``godel_evolution_init/select_seed.py``, which calls run_seed_selection.

This module is NOT under evolution/ — meta-evolve cannot edit it. An agent that
wants a different seed-selection strategy edits select_seed.py: tune the system
prompt, enable/disable strategy tools, or register new ones.

Invariants enforced here (not evolvable):
  - 0 candidates -> {} (fallback to configured strategy).
  - 1 candidate -> short-circuit (no react).
  - >=2 candidates -> fresh react loop + pick_seed decision tool.
  - HEAD restored on every exit path.
  - Strategy tools dynamically load from evolution/strategies/<name>.py.
"""

import importlib.util
import os
import sys
from typing import Dict, List, Any

from .state import reward_to_scalar
from .utils import log_format
from .utils.message_utils import append_response_to_messages


# ── Decision tool ──────────────────────────────────────────────────────────────

PICK_SEED_SCHEMA = {
    "type": "function",
    "function": {
        "name": "pick_seed",
        "description": (
            "Submit your seed selection — the git version to start the next iteration from. "
            "Call this once you have chosen which version is the strongest starting point. "
            "Ends the seed-selection react loop."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "git_hash": {
                    "type": "string",
                    "description": "The chosen git commit hash to seed the next iteration.",
                },
                "strategy_hint": {
                    "type": "string",
                    "description": (
                        "Short label for logging, e.g. 'ensemble', 'ensemble:fused', "
                        "'greedy', 'best_pick', 'llm_pick'. Do NOT use "
                        "'single_candidate' — that value is reserved for the "
                        "framework's auto-short-circuit when only one candidate "
                        "exists; if you pick a single whole version yourself, use "
                        "'greedy' or 'best_pick'."
                    ),
                },
                "hypothesis": {
                    "type": "string",
                    "description": (
                        "Your hypothesis about WHY this seed is the best starting point. "
                        "Include: (1) selection rationale — why this version over others, "
                        "(2) expected improvement — what you predict will happen, "
                        "(3) falsification criteria — signs that this hypothesis is wrong, "
                        "(4) bootstrap permission — acknowledge the evolve agent may adjust "
                        "or abandon this hypothesis based on evidence. "
                        "Format as markdown: ## Seed Selection Hypothesis\\n\\n"
                        "**Selected seed**: <hash> (iteration <n>, reward <r>)\\n\\n"
                        "**Selection rationale**: ...\\n\\n"
                        "**Hypothesis**: ...\\n\\n"
                        "**Falsification criteria**: ...\\n\\n"
                        "**Bootstrap**: ..."
                    ),
                },
                "hypotheses": {
                    "type": "array",
                    "description": (
                        "2-4 COMPETING hypotheses about what to improve and why. "
                        "These are exploration GUIDES, not truths — the evolve agent "
                        "should test them against evidence. Each hypothesis must make "
                        "different, falsifiable predictions."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Short label e.g. H1, H2, H3"},
                            "hypothesis": {"type": "string", "description": "Core claim: what's wrong and what change would help"},
                            "prediction": {"type": "string", "description": "Concrete verifiable prediction: 'If H1 is correct, then after change X, task Y should show improvement Z'"},
                            "falsification": {"type": "string", "description": "What evidence would prove this hypothesis WRONG"},
                            "confidence": {"type": "number", "description": "Initial confidence 0.0-1.0", "default": 0.5}
                        },
                        "required": ["id", "hypothesis", "prediction", "falsification"]
                    }
                },
                "merge_ops": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_hash": {"type": "string"},
                            "files": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                    "description": "Optional: merge specific files from other commits after checkout.",
                },
            },
            "required": ["git_hash"],
        },
    },
}


# ── Strategy tool schemas (framework-owned, not evolvable — evolvable controls
#    enable/disable + param overrides via _SEED_TOOLS in select_seed.py) ──────

STRATEGY_TOOL_SCHEMAS: Dict[str, dict] = {
    "ensemble": {
        "name": "pick_seed_ensemble",
        "description": (
            "LLM-powered code sub-agent. TWO MODES: (1) FUSION (2+ hashes) — merge "
            "complementary mechanisms from multiple versions. (2) REPAIR (1 hash) — "
            "fix bugs in a single version per the merge_instructions. The first hash "
            "is always checked out as the working base. The sub-agent has NO evaluate "
            "— you evaluate the returned commit yourself at the seed selection level."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_hashes": {
                    "type": "array",
                    "description": (
                        "List of commit hashes. YOU choose these. TWO MODES: "
                        "**Fusion (2+ hashes)**: pick versions with COMPLEMENTARY "
                        "mechanisms from different lineages or with different "
                        "failure-class coverage. The first hash is the foundation "
                        "(checked out as base); remaining hashes contribute code to "
                        "merge in. **Repair (1 hash)**: fix a specific bug in a "
                        "single version. Use merge_instructions to describe what's "
                        "broken and how to fix it."
                    ),
                    "items": {
                        "type": "string",
                        "description": "A git commit hash (full or abbreviated).",
                    },
                    "minItems": 1,
                },
                "merge_instructions": {
                    "type": "string",
                    "description": (
                        "Directive for the sub-agent. In FUSION mode: which version's "
                        "patterns to prefer for which modules, what problem the merge "
                        "should solve. Example: 'Take the error handling from version X, "
                        "the prompt structure from version Y, keep the archive strategy "
                        "from Z.' In REPAIR mode: describe the specific bug and how to "
                        "fix it. Example: 'The evaluate loop crashes on None reward — "
                        "add a None check before reward_to_scalar.'"
                    ),
                },
                "max_steps": {
                    "type": "integer",
                    "description": (
                        "Maximum steps for the ensemble sub-agent react loop "
                        "(default 50). Increase for complex merges across many "
                        "files; decrease for quick simple fusions."
                    ),
                },
            },
            "required": ["source_hashes"],
        },
    },
}

# ── Inspection tool schemas (framework-owned, made available to seed selection) ──

CHECKOUT_VERSION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "checkout_version",
        "description": (
            "Checkout a candidate version's harness files to disk for inspection. "
            "Does NOT change HEAD — only writes files to the working tree so you "
            "can read and evaluate them. Use this to inspect a candidate before deciding."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "git_hash": {
                    "type": "string",
                    "description": "The git commit hash to checkout files from.",
                },
            },
            "required": ["git_hash"],
        },
    },
}

VIEW_NODE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "view_node",
        "description": (
            "Deep-dive ONE version's complete relations in the knowledge graph: its own "
            "reward/summary, its lineage neighbors (parent + children), and its cross-version "
            "correlation edges (each with similarity + the LLM diff analysis). Use this to "
            "inspect a specific candidate in depth instead of reading the whole graph. "
            "Pass the candidate's commit hash (full hash or a unique prefix)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "git_hash": {
                    "type": "string",
                    "description": "The commit hash to inspect (full hash or unique prefix).",
                },
            },
            "required": ["git_hash"],
        },
    },
}


# ── Inspection tool executors (framework-owned, shared by seed + best) ──────

def _exec_checkout_version(agent, args) -> str:
    """checkout_version: write a version's files to the working tree so the
    agent can read/evaluate them. HEAD is unchanged."""
    target_hash = args.get("git_hash", "")
    if not target_hash:
        return "Error: git_hash is required."
    try:
        type_res = agent.git_controller._run_git_command(
            ["cat-file", "-t", target_hash], check=False
        )
        if not (type_res.returncode == 0 and type_res.stdout.strip() == "commit"):
            return f"Error: {target_hash[:12]} is not a valid commit."
    except Exception as e:
        return f"Error verifying commit: {e}"
    try:
        agent.git_controller._run_git_command(
            ["checkout", target_hash, "--", "."], check=False
        )
        return (
            f"Working tree now contains files from commit {target_hash[:7]}. "
            f"HEAD is unchanged. Use read_file or evaluate to inspect."
        )
    except Exception as e:
        return f"Error checking out version: {e}"


def _exec_view_node(agent, args) -> str:
    """view_node: on-demand neighborhood query for one node — its own info,
    lineage neighbors (parent/children), and correlation edges (with similarity
    + LLM diff analysis). The complete relations for a single version, without
    dumping the whole graph into the prompt. On-demand, so NOTHING is truncated:
    the agent asked for this node — return its full summary and every edge's full
    LLM analysis verbatim. ``node_id == git_hash`` (full); a unique prefix is
    also accepted."""
    target = args.get("git_hash", "")
    if not target:
        return "Error: git_hash is required."
    kg = getattr(agent, "_knowledge_graph", None)
    if kg is None:
        return "Knowledge graph not available."

    node = kg.nodes.get(target)
    if node is None:
        # Accept a unique hash prefix (the candidate table may show truncated hashes).
        matches = [nid for nid in kg.nodes if nid.startswith(target)]
        if len(matches) == 1:
            target = matches[0]
            node = kg.nodes[target]
        elif not matches:
            return f"Node {target[:7]} not found in knowledge graph."
        else:
            return (f"Hash {target[:7]} is ambiguous ({len(matches)} matches); "
                    f"use a longer prefix.")

    def _label(nid):
        n = kg.nodes.get(nid)
        if n is None:
            return nid[:7]
        return f"{n.short()} (iter {n.iteration}, reward {n.reward:.4f})"

    lines = [
        f"Node {node.short()} (iter {node.iteration}, reward {node.reward:.4f}, "
        f"mode {node.eval_mode or '?'})"
    ]
    summ = (node.summary_text or "").strip().replace("\n", " ")
    if summ:
        lines.append(f"  summary: {summ}")

    # Lineage neighbors
    parents, children = [], []
    for e in kg.lineage_edges.values():
        if e.dst_id == target:
            parents.append(e.src_id)
        if e.src_id == target:
            children.append(e.dst_id)
    lines.append("Lineage:")
    lines.append(
        f"  parent(s): {', '.join(_label(p) for p in parents) or '(root — no parent)'}"
    )
    lines.append(
        f"  children: {', '.join(_label(c) for c in children) or '(leaf — no children)'}"
    )

    # Correlation (semantic) edges
    corr = kg.get_correlations(target)
    lines.append(f"Correlations ({len(corr)} semantic edge(s)):")
    if not corr:
        lines.append("  (none)")
    else:
        for e in sorted(corr, key=lambda x: x.structural_similarity, reverse=True):
            other = e.dst_id if e.src_id == target else e.src_id
            analysis = (e.llm_diff_analysis or "").strip().replace("\n", " ")
            line = f"  ↔ {_label(other)} (sim={e.structural_similarity:.2f})"
            if analysis:
                line += f": {analysis}"
            lines.append(line)
    return "\n".join(lines)


def _load_strategy_module(agent, strategy_name: str):
    """Load a strategy module from evolution/strategies/<name>.py.

    Returns the loaded module, or None if not found / load failed.
    """
    agent_code_dir = agent.agent_code_dir
    strat_path = os.path.join(
        agent_code_dir, "evolution", "strategies", f"{strategy_name}.py"
    )
    if not os.path.isfile(strat_path):
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            f"evolution_strategy_{strategy_name}", strat_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        agent._log(f"  [seed] Failed to load strategy module '{strategy_name}': {e}")
        return None


def _strategy_name_from_tool(tool_name: str) -> str | None:
    """Derive strategy name from tool name: pick_seed_ensemble → ensemble.

    Returns None if the tool name doesn't follow the convention.
    """
    prefix = "pick_seed_"
    if tool_name.startswith(prefix) and len(tool_name) > len(prefix):
        return tool_name[len(prefix):]
    return None


def _discover_extra_strategies(agent, enabled_tools: List[dict]) -> Dict[str, dict]:
    """Discover strategy tools not pre-registered in STRATEGY_TOOL_SCHEMAS.

    For each tool name in enabled_tools that follows the ``pick_seed_<name>``
    convention but isn't in STRATEGY_TOOL_SCHEMAS, try to load the strategy
    module and read its metadata (STRATEGY_NAME, STRATEGY_DESCRIPTION).

    Returns a dict mapping tool_name → schema suitable for merging into the
    tool schemas list. These are "extra" strategies — meta-evolve can create
    new ones without touching framework code.
    """
    known_names = {s["name"] for s in STRATEGY_TOOL_SCHEMAS.values()}
    extra = {}
    for tool_cfg in enabled_tools:
        tool_name = tool_cfg.get("name", "")
        if not tool_cfg.get("enabled", True):
            continue
        if tool_name in known_names:
            continue
        strategy_name = _strategy_name_from_tool(tool_name)
        if not strategy_name:
            continue
        mod = _load_strategy_module(agent, strategy_name)
        if mod is None:
            agent._log(
                f"  [seed] Warning: _SEED_TOOLS references '{tool_name}' "
                f"but evolution/strategies/{strategy_name}.py not found"
            )
            continue
        # Read strategy metadata
        desc = getattr(mod, "STRATEGY_DESCRIPTION", f"Strategy: {strategy_name}")
        params_override = {
            k: v for k, v in tool_cfg.items()
            if k not in ("name", "enabled")
        }
        schema = {
            "name": tool_name,
            "description": desc,
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
        # Inject parameter overrides into description
        if params_override:
            overrides_str = ", ".join(f"{k}={v}" for k, v in params_override.items())
            schema["description"] = f"{desc} (params: {overrides_str})"
        extra[tool_name] = schema
    return extra


def _build_enabled_tool_map(enabled_tools: List[dict]) -> Dict[str, dict]:
    """Pre-build {name: cfg} dict from enabled_tools for O(1) lookup."""
    return {t.get("name", ""): t for t in enabled_tools if t.get("enabled", True)}


def _build_strategy_tool_executors(agent, enabled_tools: List[dict],
                                   extra: Dict[str, dict] = None):
    """Build a dict of {tool_name: executor_fn} for enabled strategy tools.

    Each executor dynamically loads the corresponding evolution/strategies/<name>.py
    module and calls its ``strategy(agent)`` function, returning ``SeedResult.to_dict()``.

    Supports BOTH pre-registered strategies (in STRATEGY_TOOL_SCHEMAS) AND
    meta-evolved additions (tools in _SEED_TOOLS whose names follow the
    ``pick_seed_<name>`` convention and whose strategy module exists).
    """
    if extra is None:
        extra = _discover_extra_strategies(agent, enabled_tools)
    executors = {}

    # Map tool names back to strategy names (pre-registered)
    tool_to_strategy = {}
    for sname, schema in STRATEGY_TOOL_SCHEMAS.items():
        tool_to_strategy[schema["name"]] = sname

    # Also map extra strategies (meta-evolved additions)
    for tool_name in extra:
        strategy_name = _strategy_name_from_tool(tool_name)
        if strategy_name and strategy_name not in tool_to_strategy.values():
            tool_to_strategy[tool_name] = strategy_name

    def _make_executor(sname, tname):
        def executor(agent_ref, args):
            agent_ref._log(f"  [seed] Consulting strategy tool: {tname}")
            mod = _load_strategy_module(agent_ref, sname)
            if mod is None or not hasattr(mod, "strategy"):
                return (
                    f"Strategy '{sname}' could not be loaded. "
                    f"It may have been deleted or renamed."
                )
            try:
                result = mod.strategy(agent_ref, args)
                if hasattr(result, "to_dict"):
                    d = result.to_dict()
                else:
                    d = result

                # All strategy tools checkout to the returned hash so
                # evaluate() runs against the right code. Tell the agent.
                if isinstance(d, dict) and d.get("git_hash"):
                    d = dict(d)
                    h = d["git_hash"][:7]
                    hint = d.get("strategy_hint", "strategy")
                    d["_working_tree"] = (
                        f"Working tree is NOW at commit {h}. "
                        f"The code on disk IS the {hint} result. "
                        f"Call evaluate(eval_mode='dev') to test it, "
                        f"or read_file/bash to inspect."
                    )
                return d
            except Exception as e:
                agent_ref._log(f"  [seed] Strategy '{sname}' raised: {e}")
                return f"Strategy '{sname}' failed: {e}"

        return executor

    enabled = _build_enabled_tool_map(enabled_tools)
    for tool_name in enabled:
        strategy_name = tool_to_strategy.get(tool_name)
        if not strategy_name:
            continue
        executors[tool_name] = _make_executor(strategy_name, tool_name)

    return executors


def _build_strategy_tool_schemas(agent, enabled_tools: List[dict],
                                 extra: Dict[str, dict] = None) -> List[dict]:
    """Build OpenAI tool schemas for enabled strategy tools.

    Uses STRATEGY_TOOL_SCHEMAS for descriptions/params, then overlays any
    overrides from the evolvable _SEED_TOOLS config (e.g. k for ensemble).

    Also auto-discovers meta-evolved additions: tools in _SEED_TOOLS whose
    names follow the ``pick_seed_<name>`` convention and whose strategy
    modules exist at ``evolution/strategies/<name>.py``, building schemas
    from the module's STRATEGY_NAME / STRATEGY_DESCRIPTION metadata.
    """
    if extra is None:
        extra = _discover_extra_strategies(agent, enabled_tools)
    schemas = []
    enabled = _build_enabled_tool_map(enabled_tools)

    # Pre-registered strategies
    for sname, schema in STRATEGY_TOOL_SCHEMAS.items():
        tool_name = schema["name"]
        if tool_name not in enabled:
            continue

        cfg = enabled[tool_name]
        desc = schema["description"]
        params = dict(schema["parameters"])  # shallow copy

        # Apply param overrides from evolvable config
        # e.g. ensemble: {"k": 5} → inject max into k's description
        for key, val in cfg.items():
            if key in ("name", "enabled"):
                continue
            props = params.setdefault("properties", {})
            if key in props and isinstance(props[key], dict):
                props[key]["description"] = (
                    f"{props[key].get('description', '')} (max: {val})"
                )

        schemas.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": desc,
                "parameters": params,
            },
        })

    # Extra strategies (meta-evolved additions not in STRATEGY_TOOL_SCHEMAS)
    for tool_name, schema in extra.items():
        schemas.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": schema["description"],
                "parameters": schema["parameters"],
            },
        })

    return schemas


# ── Shared environment context (framework-owned, injected into system prompts ──

def build_environment_context(agent) -> str:
    """Build the ``## Environment`` context block for seed/submit_best system prompts.

    Framework-owned — injected into the system prompt so the choosing agent always
    knows the real working directory, OS, and available tools, even if the evolvable
    strategy prompts (select_seed.py / select_best.py) get rewritten by meta-evolve.
    Without this the agent guesses a wrong cwd (e.g. ``/repo``), wasting steps
    rediscovering it via ``pwd``.
    """
    platform = "Windows (PowerShell)" if sys.platform == "win32" else "Linux/macOS (bash)"
    iteration = getattr(agent, "iteration", 0)
    max_iterations = getattr(agent.config, "max_iterations", "?")
    iter_info = f"{iteration} / {max_iterations}" if max_iterations != "?" else str(iteration)
    lines = [
        f"## Environment\n",
        f"**Working Directory:** `{agent.agent_code_dir}`",
        f"**OS:** {platform}",
        f"**Current Iteration:** {iter_info}",
        f"**Python:** available as `python`",
        f"**Git:** available for version control",
    ]
    # Include KG path if available
    kg = getattr(agent, "_knowledge_graph", None)
    if kg is not None:
        kg_path = getattr(kg, "storage_path", "")
        if kg_path:
            lines.append(f"**Knowledge Graph:** `{kg_path}` (lineage + correlations)")
    lines.append("")
    return "\n".join(lines)


# ── Candidate table (framework-owned, shared by seed + best) ────────

def _build_lessons_block(agent) -> str:
    """Read BOOTSTRAP.md's ``## Lesson`` section → a ``## Lessons from past
    iterations`` block injected into the seed-selection user message.

    Closes the hypothesis-verification loop: seed selection emits competing
    hypotheses; this feeds their verdicts back so the next round's hypotheses
    are informed by what prior iterations actually confirmed/refuted. Pure-data
    injection (before the evolvable ``user_guidance``) — the seed SYSTEM prompt
    stays untouched for cache stability.

    Returns "" when BOOTSTRAP.md is absent or has no Lesson section.
    """
    from src.react_loop.actions.agent_action import parse_sections

    bootstrap_path = os.path.join(agent.agent_code_dir, "BOOTSTRAP.md")
    if not os.path.isfile(bootstrap_path):
        return ""
    try:
        with open(bootstrap_path, "r", encoding="utf-8") as f:
            sections = parse_sections(f.read(), ["Lesson"])
    except Exception:
        return ""
    lesson_text = sections.get("lesson", "")
    if not lesson_text:
        return ""
    return f"## Lessons from past iterations\n\n{lesson_text}"


def _build_candidate_table(agent, candidates) -> str:
    """Build the candidate overview table + lineage tree + cross-candidate KG
    correlation analyses.

    Pure framework data pipeline — shared by run_seed_selection and
    run_submit_best. Shows every non-meta main-iteration commit with its reward,
    eval mode, modified files, tags, and summary, plus a text lineage tree and
    selective correlation analyses from the knowledge graph. Deep per-node
    inspection (full neighborhood: lineage + correlations + analyses) is left to
    the on-demand ``view_node`` tool rather than dumped inline.

    Pool-aware: each pool entry within a candidate record becomes its own row.
    """
    parts = []

    # Candidate table — expand pool entries (each entry is one row)
    candidate_set = set()
    rows = []
    for r in candidates:
        for entry in r.iter_pool():
            candidate_set.add(entry["new_commit"])
            rows.append((r, entry))
    lines = ["## Candidate versions (iteration-final commits)\n"]
    for r, entry in rows:
        h = entry["new_commit"]
        scalar = reward_to_scalar(entry["reward"])
        meta = r.metadata or {}
        mode = entry.get("committed_eval_mode") or "?"
        summary = (meta.get("summary_text") or "").replace("\n", " ").strip()
        mod_files = meta.get("modified_files") or []
        change_tags = meta.get("change_tags") or []
        op_type = meta.get("operation_type", "")
        is_pool = op_type == "pool"
        is_init = op_type == "init"
        if is_init:
            tag = " [INIT]"
        elif is_pool:
            tag = " [POOL]"
        else:
            tag = ""
        # INIT commits carry seed_info with the seed hash and strategy hint
        seed_info = meta.get("seed_info", {}) if is_init else {}
        seed_hash = seed_info.get("git_hash", "")
        seed_hint = seed_info.get("strategy_hint", "")
        seed_str = f" seed={seed_hash[:7]}" if seed_hash else ""
        seed_str += f" ({seed_hint})" if seed_hint else ""
        iter_label = f"iter={r.iteration}" if r.iteration >= 0 else f"INIT→iter={-r.iteration}"
        files_str = ", ".join(mod_files) if mod_files else "(none)"
        line = (
            f"- {iter_label} hash={h[:7]}{tag}{seed_str} reward={scalar:.4f} "
            f"mode={mode} files=[{files_str}]"
        )
        if change_tags:
            line += f" tags=[{', '.join(change_tags)}]"
        if summary:
            line += f"\n  summary: {summary}"
        lines.append(line)
    parts.append("\n".join(lines))

    # ── KG: lineage tree + cross-candidate correlations ──
    kg = getattr(agent, "_knowledge_graph", None)
    if kg is not None:
        # Lineage tree (text visualization)
        try:
            lineage_text = kg.render_lineage_for_prompt(token_budget=2000)
            if lineage_text:
                parts.append("\n" + lineage_text)
        except Exception:
            pass

        # Cross-candidate correlation analyses (v3: selective edges only)
        try:
            corr_text = kg.render_correlations_for_prompt(
                node_ids=list(candidate_set), token_budget=2000
            )
            if corr_text:
                parts.append("\n" + corr_text)
        except Exception:
            pass

    # Best / head anchors
    best = agent.evolution_tracker.get_best_version("highest_reward")
    head = agent.git_controller.get_current_commit() or ""
    parts.append("")
    parts.append(
        f"**Reference:** Best historical reward = "
        + (f"{best[1]:.4f} (commit {best[0][:7]})" if best else "N/A")
        + f"  |  Current HEAD = {head[:7]}"
    )
    # Add KG path for convenience
    if kg is not None:
        kg_path = getattr(kg, "storage_path", "")
        if kg_path:
            parts.append(f"  |  Knowledge Graph: `{kg_path}`")

    return "\n".join(parts)


# ── Main runner ─────────────────────────────────────────────────────────────────

def run_seed_selection(
    agent,
    system_prompt: str,
    seed_tools: List[dict],
    user_guidance: str = "",
    max_steps: int = None,
) -> dict:
    """Run the seed-selection react loop; return a seed_info dict.

    Args:
        agent: GodelAgent instance.
        system_prompt: str — role/strategy framing (evolvable, → system message).
        seed_tools: list of evolvable tool configs from _SEED_TOOLS
                    (each: {"name": str, "enabled": bool, ...}).
        user_guidance: str — procedural "how to decide" steps (evolvable,
                    appended to the candidate table in the user message).
        max_steps: react loop budget. Falls back to
                   agent.config.seed_selection_max_steps (default 10).

    Returns:
        {} on no-decision/fallback, else {git_hash, strategy_hint, metadata, merge_ops}.
    """
    if max_steps is None:
        max_steps = getattr(agent.config, "seed_selection_max_steps", 10)
    seed_eval_enabled = getattr(agent.config, "seed_eval_enabled", False)
    seed_eval_max = getattr(agent.config, "seed_eval_max_calls", 1)
    # Dry-run gate: validate_archive() stubs react to verify this module loads
    # and runs end-to-end. Silence all user-facing logs below so the dry-run
    # doesn't masquerade as a real seed selection (mirrors submit_best's
    # _submit_best_dry_run gate on the select_best dimension).
    dry_run = getattr(agent, "_seed_dry_run", False)
    eval_count = [0]
    eval_results: List[Dict[str, Any]] = []  # [{"reward": scalar, "mode": str, "commit": str}, ...]
    tracker = agent.evolution_tracker

    # ── Build candidate pool ──
    # Includes: main-iteration commits (pool, regular) + INIT commits.
    # INIT commits (operation_type="init") are INCLUDED — when seed selection
    # uses ensemble/recursive strategies the INIT commit IS the new version
    # (seed checkout + meta fold-in); filtering it out would lose that lineage.
    # Excludes: meta_evolve, crossover, ensemble (intermediate artifacts).
    candidates = [
        r for r in (tracker.records or [])
        if r.primary_commit()
        and r.metadata.get("type") != "meta_evolve"
        and r.metadata.get("operation_type") not in ("crossover", "ensemble")
        and (
            r.metadata.get("operation_type") == "init"
            or (r.is_main_iteration and r.iteration >= 0)
        )
    ]

    iter_label = f"for iter {agent.iteration}"

    # 0 candidates → fallback (should not happen after the initial seed record
    # is created during agent init; kept as a safety net for edge cases).
    if not candidates:
        if not dry_run:
            log_format.log_phase_banner(
                agent, f"SEED SELECTION ({iter_label})",
                info="  Candidates: 0 | fallback to configured strategy",
            )
        agent.archive_manager._ensure_strategies_discovered()
        if not dry_run:
            agent._log("  [seed] 0 candidates — no tracker records yet, returning {}")
        return {}

    # 1 candidate → short-circuit (no react loop)
    if len(candidates) == 1:
        only = candidates[0]
        scalar = reward_to_scalar(only.primary_reward())
        only_commit = only.primary_commit()
        if not dry_run:
            log_format.log_phase_banner(
                agent, f"SEED SELECTION ({iter_label})",
                info=(
                    f"  Candidates: 1 (iter {only.iteration}, "
                    f"reward {scalar:.4f}) | auto-pick (single)"
                ),
            )
        agent.archive_manager._ensure_strategies_discovered()
        if not dry_run:
            agent._log(
                f"  [seed] Single candidate (iter {only.iteration}, "
                f"reward {scalar:.4f}), short-circuit"
            )
        result: Dict[str, Any] = {
            "git_hash": only_commit,
            "strategy_hint": "single_candidate",
            "hypothesis": "",
            "hypotheses": [],
            "metadata": {},
            "merge_ops": [],
        }
        # Include seed eval fields even in short-circuit (always None here)
        result["seed_eval_reward"] = None
        result["seed_eval_mode"] = None
        # If seed eval is enabled and we have a single candidate, the agent
        # could still evaluate it — but short-circuit skips the react loop.
        return result

    # ≥2 candidates → react loop
    if not dry_run:
        agent._log(
            f"  [seed] {len(candidates)} candidates, "
            f"running agentic seed selection (max {max_steps} steps)"
        )

    pre_head = agent.git_controller.get_current_commit() or ""

    # ── Build tools: framework base + strategy tools + decision tool ──
    # Discover extra strategies once (shared between schemas and executors)
    extra = _discover_extra_strategies(agent, seed_tools)
    framework_tools = agent.get_tools(scope="pick_seed")
    strategy_schemas = _build_strategy_tool_schemas(agent, seed_tools, extra=extra)
    strategy_executors = _build_strategy_tool_executors(agent, seed_tools, extra=extra)
    tools = framework_tools + strategy_schemas + [
        CHECKOUT_VERSION_SCHEMA, VIEW_NODE_SCHEMA, PICK_SEED_SCHEMA
    ]

    # ── Build messages ──
    # User prompt = data (candidate table + past lessons) + procedural guidance (evolvable).
    # System prompt = role/strategy framing (evolvable) + environment (framework).
    # meta-evolve can edit system_prompt and user_guidance independently.
    # Lessons block is pure-data injection (closes the hypothesis→verdict loop):
    # placed before user_guidance so the evolvable guidance still anchors the tail.
    user_content = _build_candidate_table(agent, candidates)
    lessons_block = _build_lessons_block(agent)
    if lessons_block:
        user_content += "\n\n" + lessons_block
    user_content += user_guidance

    kg = getattr(agent, "_knowledge_graph", None)
    has_kg = bool(kg is not None and getattr(kg, "edges", None))

    if not dry_run:
        log_format.log_phase_banner(
            agent,
            f"SEED SELECTION ({iter_label})",
            info=(
                f"  Candidates: {len(candidates)} | "
                f"knowledge graph: {'yes' if has_kg else 'no'} | max_steps: {max_steps}"
            ),
        )

    agent.archive_manager._ensure_strategies_discovered()

    # Environment is objective fact — inject into system prompt so the
    # choosing agent always knows the real working directory, even if the
    # evolvable _SEED_SELECTION_SYSTEM_PROMPT gets rewritten by meta-evolve.
    messages = [
        {"role": "system", "content": system_prompt + "\n" + build_environment_context(agent)},
        {"role": "user", "content": user_content},
    ]

    # ── React loop ──
    decision: Dict[str, Any] = {}
    done = [False]

    def tool_executor(name, args):
        # 1. Strategy tool?
        if name in strategy_executors:
            return strategy_executors[name](agent, args)

        # 2. Decision tool?
        if name == "pick_seed":
            h = (args or {}).get("git_hash", "")
            if not h:
                return (
                    "Error: pick_seed requires a non-empty 'git_hash'. "
                    "You must provide the full or short commit hash of the version "
                    "you are selecting as the seed. Example: pick_seed(git_hash=\"abc1234\"). "
                    "Do not call pick_seed without this argument."
                )
            decision.update(args or {})
            done[0] = True
            return (
                f"Decision recorded: {h[:7]}. "
                f"Seed selection complete."
            )

        # 3. Inspection tools (checkout_version, view_node)
        if name == "checkout_version":
            return _exec_checkout_version(agent, args)

        if name == "view_node":
            return _exec_view_node(agent, args)

        # 4. Evaluate tool (seed eval — validate hypothesis before picking)
        if name == "evaluate":
            if not seed_eval_enabled:
                return "Error: evaluate is not available in seed selection scope."
            if eval_count[0] >= seed_eval_max:
                return (
                    f"Maximum seed eval calls ({seed_eval_max}) reached. "
                    f"Call pick_seed to decide."
                )
            eval_count[0] += 1
            result = agent.execute_tool(name, args, scope="pick_seed")
            # Extract scalar reward from agent state.
            # agent.state may be None during seed selection (no full AgentState
            # is created until reset_for_iteration in the main evolve loop).
            reward_val = agent.state.reward if agent.state else None
            scalar = reward_to_scalar(reward_val) if reward_val is not None else 0.0
            mode = getattr(agent.state, "last_eval_mode", "dev") if agent.state else "dev"
            # Capture the commit being evaluated so we only credit
            # seed_eval_reward when the agent actually picks THIS version.
            current_eval_commit = agent.git_controller.get_current_commit() or ""
            eval_results.append({
                "reward": scalar, "mode": mode, "commit": current_eval_commit,
            })
            remaining = seed_eval_max - eval_count[0]
            hint = (
                f"\n\n[Seed eval {eval_count[0]}/{seed_eval_max}] "
                f"Reward: {scalar:.4f} ({mode}). "
            )
            if remaining > 0:
                hint += (
                    f"{remaining} eval(s) remaining. "
                    f"Evaluate again or continue inspecting."
                )
            else:
                hint += (
                    "No evals remaining. Continue inspecting "
                    "or call pick_seed to decide."
                )
            return result + hint

        # 5. Framework tool (read_file / bash / powershell / evaluate)
        return agent.execute_tool(name, args, scope="pick_seed")

    def _build_seed_status_suffix(step):
        """Return a status suffix for the last tool result, matching the main
        evolve loop's ``_build_step_status_suffix`` pattern (appended to tool
        output, NOT a separate user message)."""
        step_pct = int(step / max_steps * 100) if max_steps else 0
        parts = [f"Step {step}/{max_steps}"]

        if seed_eval_enabled:
            eval_used = eval_count[0]
            eval_left = seed_eval_max - eval_used
            parts.append(f"evals {eval_used}/{seed_eval_max}")

        suffix = "\n— [" + " | ".join(parts) + "]"

        # Step-budget hints
        if step_pct >= 80:
            suffix += (
                " ⚠️ running low — you should call `pick_seed` now "
                "unless you have a critical unanswered question"
            )
        elif step_pct >= 50:
            suffix += (
                " ⏳ past halfway — converge toward a decision; "
                "call `pick_seed` soon"
            )

        # Eval-budget guidance (actionable, not just a count)
        if seed_eval_enabled and eval_left == 0 and eval_used > 0:
            suffix += " | no evals left — inspect further or call `pick_seed`"

        return suffix

    def _append_status_to_last_tool_result(step):
        """Find the last tool-role message and append the status suffix to it.

        This mirrors ``EvolveHelper.process_tool_calls()`` line 284:
        ``msg_content = result + self._build_step_status_suffix()``.
        The agent reads the status as part of the tool output on its next turn,
        without an extra user-message round-trip."""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "tool":
                suffix = _build_seed_status_suffix(step)
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
                _append_status_to_last_tool_result(step)

            if not tool_calls_made:
                # No tool call — nudge to decide or keep inspecting, then
                # give one more react within this same step.
                messages.append({
                    "role": "user",
                    "content": (
                        "You did not use any tools. If you are ready to decide, call "
                        "`pick_seed` with your chosen git_hash. Otherwise use "
                        "`bash`/`read_file`/`checkout_version` to inspect candidates "
                        "or `pick_seed_ensemble` to fuse versions."
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
            if not dry_run:
                agent._log("  [seed] max steps reached without decision — final nudge")
            messages.append({
                "role": "user",
                "content": (
                    "⏰ **TIME'S UP.** You have used all {max_steps} steps without "
                    "calling `pick_seed`. This is your ABSOLUTE LAST chance — call "
                    "`pick_seed(git_hash=\"...\")` NOW. If you do not decide, your "
                    "selection will be discarded and the default strategy used instead."
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
                    if not dry_run:
                        agent._log("  [seed] still no decision after bonus react")
            except Exception:
                if not dry_run:
                    agent._log("  [seed] bonus react failed")

    except Exception as e:
        if not dry_run:
            agent._log(f"  [seed] react loop failed: {e}")

    # ── Save seed-selection conversation to .evolution_context/select_seed/ ──
    _save_seed_messages(agent, messages)

    return _finalize_seed(agent, decision, candidates, pre_head, eval_results)


# ── Finalize ────────────────────────────────────────────────────────────────────

def _save_seed_messages(agent, messages) -> None:
    """Persist the seed-selection conversation to .evolution_context/select_seed/."""
    persistence = getattr(agent, "context_persistence", None)
    if persistence is None:
        return
    if not messages:
        return
    try:
        saved = persistence.save_phase_messages(
            "select_seed", agent.iteration, messages
        )
        if saved:
            agent._log(f"  [seed] Saved conversation to {saved}")
    except Exception as e:
        agent._log(f"  [seed] Failed to save messages: {e}")


def _finalize_seed(agent, decision, candidates, pre_head,
                   eval_results=None) -> dict:
    """Validate the agent's pick_seed decision and assemble the seed_info dict.

    - No commit_hash, or not a valid commit → restore HEAD + {} (fallback).
    - commit_hash is a meta_evolve commit → reject → restore + {}.
    - commit_hash is valid → restore HEAD to pre_head (the agent's dirty tree
      from the react loop is discarded) and return seed_info dict.

    HEAD restore on every exit path uses ``reset --hard pre_head`` (like
    submit_best's _restore_head(reject) / _finalize_pick). The caller
    (ArchiveManager.select_seed) applies the actual version switch via
    apply_version_switch().
    """
    if eval_results is None:
        eval_results = []

    # Dry-run gate (validate_archive): silence the reject reason so a stubbed
    # dry-run doesn't leak result lines (symmetric to run_seed_selection's gate
    # and to submit_best's _finalize_pick).
    dry_run = getattr(agent, "_seed_dry_run", False)

    def reject(reason=None):
        if reason and not dry_run:
            agent._log(f"  [seed] {reason}")
        _restore_head(agent, pre_head)
        return {}

    h = (decision or {}).get("git_hash", "")
    if not h:
        return reject("no commit_hash in pick_seed decision")

    # Must be a real commit
    try:
        type_res = agent.git_controller._run_git_command(
            ["cat-file", "-t", h], check=False
        )
    except Exception as e:
        return reject(f"cat-file failed for {h[:7]}: {e}")
    if not (type_res.returncode == 0 and type_res.stdout.strip() == "commit"):
        return reject(f"{h[:7]} is not a valid commit — rejecting")

    # Normalize to full hash at the boundary — the LLM may submit a short hash
    # (7 chars from view_node / git log --oneline), but all internal lookups
    # (tracker records, eval_results) store full 40-char hashes.  Resolve once
    # here so everything downstream uses the canonical form.
    try:
        resolve = agent.git_controller._run_git_command(
            ["rev-parse", "--verify", h], check=False
        )
        full_h = resolve.stdout.strip() if resolve.returncode == 0 else h
    except Exception:
        full_h = h

    # Reject meta_evolve commits
    if agent.archive_manager.is_meta_evolve_commit(full_h):
        return reject(f"{h[:7]} is a meta_evolve commit — rejecting")

    hint = (decision or {}).get("strategy_hint") or "llm_pick"
    hypothesis = (decision or {}).get("hypothesis") or ""
    merge_ops = (decision or {}).get("merge_ops") or []
    metadata = (decision or {}).get("metadata") or {}

    # Extract hypotheses (new format) or wrap legacy hypothesis
    hypotheses = (decision or {}).get("hypotheses") or []
    if not hypotheses:
        legacy_hypothesis = (decision or {}).get("hypothesis") or ""
        if legacy_hypothesis:
            hypotheses = [{
                "id": "H1",
                "hypothesis": legacy_hypothesis,
                "prediction": "",
                "falsification": "",
                "confidence": 0.5
            }]

    _restore_head(agent, pre_head)

    # Compute best seed eval reward for the CHOSEN seed.
    #
    # Priority:
    #   1. Fresh eval run during seed selection against this exact commit
    #      (the agent evaluated the version and confirmed it works).
    #   2. Historical reward from the candidate's tracker record
    #      (every candidate was evaluated at the end of its own iteration).
    #
    # Strategy tools (e.g. ensemble) may switch the working
    # tree mid-loop; evals of other versions are excluded so the reward
    # on the INIT commit always reflects the seed that was picked.
    best_eval_reward = None
    best_eval_mode = None
    # Compute best seed eval reward for the CHOSEN seed (full_h already resolved above).
    if full_h:
        # 1) Fresh eval during seed selection (evaluated this exact version)
        matching_evals = [
            e for e in (eval_results or [])
            if e.get("commit", "") == full_h
        ]
        if matching_evals:
            best = max(matching_evals, key=lambda e: e["reward"])
            best_eval_reward = best["reward"]
            best_eval_mode = best["mode"]
        else:
            # 2) Fall back to candidate's historical reward
            for c in (candidates or []):
                for entry in c.iter_pool():
                    if entry["new_commit"] == full_h:
                        best_eval_reward = reward_to_scalar(entry["reward"])
                        best_eval_mode = entry.get("committed_eval_mode", "")
                        break
                if best_eval_reward is not None:
                    break

    return {
        "git_hash": full_h,  # canonical full hash — downstream lookups expect full hashes
        "strategy_hint": hint,
        "hypothesis": hypothesis,
        "hypotheses": hypotheses,
        "metadata": metadata,
        "merge_ops": merge_ops,
        "seed_eval_reward": best_eval_reward,
        "seed_eval_mode": best_eval_mode,
    }


def _restore_head(agent, target):
    """Restore HEAD / working tree on every exit path.

    Uses ``reset --hard`` to discard the agent's dirty tree from the react loop.
    The caller (ArchiveManager.select_seed) is responsible for applying the
    actual version switch via apply_version_switch().

    During a validate_archive dry-run (``agent._seed_dry_run``), the reset is
    skipped: the stubbed react doesn't dirty the tree, and the reset would
    destroy pre-existing working-tree edits (e.g. meta-evolve edits to
    ``evolution/`` made before validate_archive was called).

    NOTE: Staged meta-evolve changes must already be committed by the caller
    (via _fold_in_staged_meta_changes) before this function runs — reset --hard
    wipes both the index and working tree.
    """
    # Defensive check: warn if staged changes would be destroyed.
    # Silenced during validate_archive dry-runs — the stubbed react doesn't
    # actually dirty the tree, so the warning would be a false alarm there.
    dry_run = getattr(agent, "_seed_dry_run", False)
    try:
        check = agent.git_controller._run_git_command(
            ["diff", "--cached", "--quiet"], check=False
        )
        if check.returncode != 0 and not dry_run:
            agent._log(
                "  [seed] WARNING: _restore_head would destroy staged changes. "
                "Ensure _fold_in_staged_meta_changes() was called before seed selection."
            )
    except Exception:
        pass
    if dry_run:
        # Dry-run (validate_archive): the stubbed react doesn't dirty the
        # tree, so there is nothing to clean up. The reset --hard below would
        # destroy pre-existing working-tree edits (e.g. meta-evolve edits to
        # evolution/ made before validate_archive was called). Skip it.
        return
    try:
        agent.git_controller._run_git_command(
            ["reset", "--hard", target], check=False
        )
    except Exception:
        pass
