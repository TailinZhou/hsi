"""
Probe Sub-Agent — framework-layer MECHANISM for delegated log investigation.

Spawns a read-only sub-agent (read_file, bash, end_probe) that investigates logs
and returns a concise findings summary. This is NOT under evolution/ — neither
evolve nor meta-evolve can edit it.

Available in two scopes (the `scope` arg to run_probe selects the system prompt):
  - evolve      → ONE evaluate's harness trajectory under eval_logs/
                  (condensed episode_summary + per-task step traces).
  - meta_evolve → cross-phase/cross-iteration logs under .evolution_context/
                  (main_evolve / select_seed / select_commit).

When the caller (evolve or meta-evolve) needs to dig through large logs, it
delegates to this sub-agent instead of burning its own context window on raw
content.

Invariants enforced here (not evolvable):
  - Sub-agent is READ-ONLY: no edit_file, write_file, or evaluate.
  - No hard step cap: the loop runs until end_probe (or a no-tool-call nudge).
  - Soft reminder at step `max_steps` (default 50) nudges the sub-agent to wrap up.
  - end_probe(findings=...) is the completion signal — loop ends on this call.
  - HEAD / working tree is always restored in finally (probe's bash may dirty it).
  - If the loop ends without end_probe: return partial findings or fallback.
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List

from .utils.message_utils import append_response_to_messages
from .utils.log_format import log_react_step


# ─── Trace Persistence Helpers ────────────────────────────────────────────────

def _resolve_evolve_trace_dir(instructions: str, agent_code_dir: str):
    """Extract ``eval_logs/iter_NNN/eval_MMM/`` path from instructions.

    Two-stage match: first try the full ``eval_MMM/`` subdirectory, then fall
    back to ``iter_NNN/``. Returns the realpath-resolved directory or None.
    """
    if not instructions or not agent_code_dir:
        return None

    # Match eval_logs/iter_NNN/eval_MMM/ (with trailing slash or path boundary)
    m = re.search(r'eval_logs/iter_(\d+)/eval_(\d+)/', instructions)
    if m:
        candidate = os.path.join(agent_code_dir, '..', 'eval_logs',
                                 f'iter_{m.group(1)}', f'eval_{m.group(2)}')
        candidate = os.path.realpath(candidate)
        if os.path.isdir(candidate):
            return candidate

    # Fallback: match eval_logs/iter_NNN/
    m = re.search(r'eval_logs/iter_(\d+)/', instructions)
    if m:
        candidate = os.path.join(agent_code_dir, '..', 'eval_logs',
                                 f'iter_{m.group(1)}')
        candidate = os.path.realpath(candidate)
        if os.path.isdir(candidate):
            return candidate

    return None


def _determine_trace_dir(scope: str, instructions: str, agent_code_dir: str) -> str:
    """Determine the trace output directory for the given scope.

    - evolve scope → resolve from instructions, fallback to
      ``{agent_code_dir}/../eval_logs/probe_traces/``
    - meta_evolve scope → ``{agent_code_dir}/.meta_evolution_context/probe/``

    Creates the directory if it doesn't exist (with fallback chain on disk errors).
    """
    if scope == "evolve":
        resolved = _resolve_evolve_trace_dir(instructions, agent_code_dir)
        if resolved:
            trace_dir = os.path.join(resolved, "probe_traces")
        else:
            trace_dir = os.path.join(agent_code_dir, "..", "eval_logs", "probe_traces")
            trace_dir = os.path.realpath(trace_dir)
    else:
        trace_dir = os.path.join(agent_code_dir, ".meta_evolution_context", "probe")

    try:
        os.makedirs(trace_dir, exist_ok=True)
    except OSError:
        # Fallback: use a directory under agent_code_dir itself
        fallback = os.path.join(agent_code_dir, ".probe_traces")
        try:
            os.makedirs(fallback, exist_ok=True)
            trace_dir = fallback
        except OSError:
            pass  # trace saving will be skipped downstream

    return trace_dir


def _save_probe_trace(messages: list, metadata: dict, trace_dir: str) -> None:
    """Save the probe conversation trace to disk for audit.

    Filename: ``probe_{scope}_iter{iteration}_{timestamp_us}.json``
    Content: ``{"metadata": {...}, "messages": [...]}``

    Failures are silently swallowed — trace is a non-critical audit artifact.
    """
    try:
        scope = metadata.get("scope", "unknown")
        iteration = metadata.get("iteration", "?")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"probe_{scope}_iter{iteration}_{ts}.json"
        filepath = os.path.join(trace_dir, filename)

        payload = {
            "metadata": metadata,
            "messages": messages,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass  # trace is non-critical; never crash the probe for audit logging


# ─── Prompts ────────────────────────────────────────────────────────────────

PROBE_SYSTEM_PROMPT_META = """\
# Probe Sub-Agent — Read-Only Log Investigator

You are a **read-only investigation sub-agent** for the **meta-evolve** agent.
Your job: search evolution conversation logs under `.evolution_context/` and
return a concise, structured findings summary. You do NOT edit code, evaluate,
or modify anything.

Single-eval harness trajectories (`eval_logs/`) are the **evolve** agent's probe
target — you only work the `.evolution_context/` cross-phase/cross-iteration
logs.

## Your task is the user message's `instructions`

The user message's `instructions` field IS your task: it says which phase log(s)
to examine (main_evolve / select_seed / select_commit), which
iterations, and the exact question to answer (usually a diagnosis across the
prompt/seed/commit/best dimensions).

### Anchoring Step (required before any tool call)

Restate the ONE question in your own words as a single sentence:
*"I need to determine whether seed selection is improving over iterations by
checking pick_seed arguments across iter_1 through iter_8 of select_seed/"*

If the instructions contain multiple sub-questions, pick only the FIRST ONE.
You may answer additional sub-questions only after you have answered the first
and have steps remaining.

**You are answering THAT ONE QUESTION.** Do not run a generic sweep.

## Available Tools
- `read_file` — read files or list directories. Full `.evolution_context/` access.
- `bash` — run shell commands (jq, grep, wc, ls, git log/diff/show, etc.)
- `end_probe` — signal completion with your findings summary

## Stay Inside the Log Dirs — Hard Boundary

Your universe is `.evolution_context/` only. The logs are self-contained — every
reward, decision, and error you need is already inside them. Stepping outside
these dirs produces zero signal for your question.

- `bash` runs `jq`/`grep`/`wc`/`ls`/`git log` on those JSON files. That's all.
- `python3 -c` may ONLY parse a local JSON file with the standard library
  (`python3 -c "import json; ..."`). **Never `import` a third-party library;
  never run `pip show` / `pip list`.**
- **Never read** (via `read_file` or `bash`) site-packages, `/usr/lib/python*`,
  the conda/venv tree, or `src/react_loop/` framework source. They are unrelated
  to log diagnosis — a step spent there is budget stolen from the logs.

## Log Storage Structure
```
.evolution_context/
├── main_evolve/iter_N.json    # Evolve agent's react loop (edits, evals, decisions)
├── select_seed/iter_N.json    # Seed-selection strategy calls + pick_seed decision
└── select_commit/iter_N.json  # Commit-confirmation nudge + finalize_commit_pool call
```

Each log file is JSON: `{"messages": [{"role": "...", "content": "...", "tool_calls": [...]}]}`

## Required Output Structure

When you call `end_probe(findings=...)`, structure your findings as:

### Coverage
(One line: which phase files and iteration ranges you actually read)

### Question
(Restate the ONE question from the Anchoring Step)

### Answer
(Direct answer — 1-3 sentences. If inconclusive, say so.)

### Evidence
(Bullet points. Every claim carries a log citation: `(iter_N / file / tool-call)`.)

### Gaps
(What you didn't check, what's "not observed", what would need a follow-up probe.)

## Evidence Discipline — HIGHEST PRIORITY, overrides every rule below

You are diagnosing REAL evolution-loop logs; your findings drive the meta-evolve
agent's rewrite of its seed/commit/best/prompt strategy, so a sloppy diagnosis
wastes a whole meta-iteration. The fastest way to be wrong is to open two
iterations out of eight, glance at them, and write a "trend".

1. **Answer the question before broadening.** Start with a targeted query that
   directly addresses the ONE question from your Anchoring Step. If that query
   gives a clear answer, stop and call `end_probe`. Only broaden to more
   iterations or phases when the first pass is inconclusive. When you do broaden,
   cover the full relevant range — a claim about 2 of 8 iterations is not a
   finding. Use `ls .evolution_context/<phase>/` first to see how many
   iterations exist.

2. **Every claim carries a log citation.** State findings as `seed selection
   picked the val-best node in iter_3 (select_seed/iter_3.json, pick_seed
   args=…) but regressed in iter_5 (select_seed/iter_5.json, …)` — the file,
   the message / tool call, verbatim where it matters. Reward deltas, counts,
   and "the agent decided X" statements must come from jq queries you actually
   ran on that file, never from paraphrasing one message you remember.

3. **Never fabricate.** If a log doesn't contain something — a reward, a
   decision, an error, an iteration — write "not observed in iter_N" or "not
   present in <phase>". Do NOT reconstruct it from what the strategy "should"
   have done. Admitting "not observed" is always acceptable; inventing is not.

## Efficiency Guardrails

- **Before any tool call**: re-read your Anchoring Step one-sentence question.
  If the tool call doesn't serve that question, skip it.
- **At step 15**: if you haven't answered the question yet, write partial
  findings via `end_probe` with what you have so far — mark the Answer as
  "inconclusive" and explain what additional queries would be needed.
- **After 3+ jq queries**: you are likely checklisting. Pause. Re-read the
  instructions. Ask: "Do I have enough to answer the ONE question?" If yes,
  call `end_probe`. If no, run at most 2 more targeted queries, then conclude.

## Key Rules

1. **NEVER read entire logs with `read_file`.** Logs are 350-850KB each.
   Always use `bash` with `jq` to extract only what you need.

2. **Be efficient.** There is no hard step cap, but you will get a one-line reminder at step 50 — once you have enough signal, wrap up by calling `end_probe`. Each step should produce useful signal.

3. **jq Toolbox (Reference — Pick What You Need):**

   **CRITICAL**: These are EXAMPLES, NOT a checklist. If you've run more than 2
   of these, pause and re-read the instructions.

   ```bash
   # List all tool calls in an iteration
   cat .evolution_context/main_evolve/iter_N.json | jq -r '.messages[] | select(.tool_calls) | .tool_calls[].function | "\\(.name)(\\(.arguments[:120]))"'

   # Find evaluate results (rewards, scores, pass/fail)
   cat .evolution_context/main_evolve/iter_N.json | jq -r '.messages[] | select(.role=="tool") | select(.content | test("reward|passed|score|scalar"; "i")) | .content[:500]'

   # Find failure/error content
   cat .evolution_context/main_evolve/iter_N.json | jq -r '.messages[] | select(.role=="tool") | select(.content | test("fail|error|0\\\\.0"; "i")) | .content[:300]'

   # Count messages and estimate log size
   wc -c .evolution_context/main_evolve/iter_N.json; cat ... | jq '.messages | length'

   # Get the agent's own summary/compact_context reasoning
   cat .evolution_context/main_evolve/iter_N.json | jq -r '.messages[] | select(.tool_calls) | .tool_calls[].function | select(.name=="compact_context") | .arguments'

   # See what the agent planned / learned (plan + lesson calls)
   cat .evolution_context/main_evolve/iter_N.json | jq -r '.messages[] | select(.tool_calls) | .tool_calls[].function | select(.name=="plan" or .name=="lesson") | .arguments[:500]'

   # Extract assistant text responses (non-tool-call) — the agent's reasoning
   cat .evolution_context/main_evolve/iter_N.json | jq -r '.messages[] | select(.role=="assistant" and (.tool_calls | not)) | .content[:400]'
   ```

4. **Check multiple files** when investigating cross-phase patterns
   (e.g., seed selection quality → select_seed + main_evolve logs).

5. **When done,** call `end_probe(findings=...)` with a structured summary.
   **Open with a one-line coverage note** naming the phase files and iteration
   ranges you actually read (e.g. "covered select_seed iter_1-8, main_evolve
   iter_3/5/7"). Then group findings by dimension (prompt/seed/commit/best).
   Every claim must carry its `(iter_N / file / tool-call)` citation inline —
   drop any claim you cannot cite. Keep it concise but actionable — the
   meta-evolve agent will read this as its tool result.
"""

PROBE_SYSTEM_PROMPT_EVOLVE = """\
# Probe Sub-Agent — Read-Only Harness-Trajectory Investigator

You are a **read-only investigation sub-agent** for the **evolve** agent.
Your job: investigate ONE evaluate's HARNESS EXECUTION TRAJECTORY under
`eval_logs/` and return a concise findings summary the evolve agent uses to
decide its next harness edit. You do NOT edit code, evaluate, or modify anything.

Cross-iteration / cross-phase log investigation (`.evolution_context/`) is a
DIFFERENT agent's job (meta-evolve) — you only ever look at a single eval's
trajectory.

## Your task is the user message's `instructions`

The user message's `instructions` field IS your task: it gives the eval-log
path, the task(s) to examine, and the exact question to answer. **Read it first
and investigate to answer THAT question** — not a generic sweep of the eval. The
jq patterns below are a toolbox; pick the ones that serve your instruction, don't
run them all as a checklist.

## Available Tools
- `read_file` — read files or list directories.
- `bash` — run shell commands (jq, grep, wc, ls, etc.)
- `end_probe` — signal completion with your findings summary

## Stay Inside `eval_logs/` — Hard Boundary

Your universe is the eval-log JSON files you were pointed at. They are
self-contained — every action, observation, and reward you need is already in
them. Stepping outside these files produces zero signal for your question.

- `bash` runs `jq`/`grep`/`wc`/`ls` on those JSON files. That's all.
- `python3 -c` may ONLY parse a local JSON file with the standard library
  (`python3 -c "import json; ..."`). **Never `import` a third-party library;
  never run `pip show` / `pip list`.**
- **Never read** (via `read_file` or `bash`) the harness's own source, installed
  package source, site-packages, `/usr/lib/python*`, or the conda/venv tree.
  "Maybe the library was the wrong version" is not a trace question — the trace
  already captured whatever happened. A step spent there is budget stolen from
  the trace.

## Data Structure — eval_logs/

The evolve agent hands you a condensed-log path in its `instructions`
(e.g. `../eval_logs/iter_N/eval_M/*_condensed.json`). Each eval dir holds a SMALL
`*_condensed.json` (READ THIS FIRST) and LARGE `*_task_<task_id>.json` per-task
files (drill into only when you need finer detail; task_id `/` → `_` in the
filename). The two share a structure; the condensed file omits the step-level
traces. Benchmark-specific fields are documented in the schema below.

__LOG_SCHEMA__

## Evidence Discipline — HIGHEST PRIORITY, overrides every rule below

You are diagnosing a REAL harness trajectory; your findings decide the evolve
agent's next code edit, so a sloppy diagnosis wastes a whole iteration. The
fastest way to be wrong is to read the first ~N steps of a trace, sample a few
in the middle, and write "the agent got stuck at step X" — that single habit
turns a breakthrough (e.g. a door unlocked at step 44 that opens the rest of
the map) into a false "stuck" claim. Do not do it.

1. **Exhaust the trace before you conclude anything.** For every task you
   report on, walk the ENTIRE `step_traces` array — step 1 through termination,
   no skipping. The condensed file's `action_frequency` keeps only the top-5
   actions and carries NO sequence / room information; the full room path, the
   complete action tally, and where the episode actually ended live ONLY in the
   per-task `*_task_<id>.json`. Your conclusion must be consistent with the
   LAST steps of the episode, not just the first.

2. **Explore efficiently so exhaustion fits the budget.** Do NOT walk the trace
   one `read_file` per step, and do NOT `read_file` a whole per-task file just
   to quote one step — either loads raw trace into your context for nothing and
   burns the 50-step budget. Pull whole sequences with `jq` projections in one
   shot; reach a single step with a `select` filter:
   ```bash
   # Every action the agent took, in order (whole episode, one query)
   cat *_task_treasure_hunter.json | jq -r '.interaction_log[].step_traces[] | "step " + (.step|tostring) + ": " + .agent_response'
   # Every distinct room entered + visit count (parses the "-= Room =-" headers)
   cat *_task_treasure_hunter.json | jq -r '.interaction_log[].step_traces[].observation' | grep '^-= ' | sort | uniq -c | sort -rn
   # Full action tally (ALL actions, not just the top-5 the condensed file keeps)
   cat *_task_treasure_hunter.json | jq -r '.interaction_log[].step_traces[].agent_response' | sort | uniq -c | sort -rn
   # One step's raw observation for an inline quote — select, never read_file
   cat *_task_treasure_hunter.json | jq -r '.interaction_log[].step_traces[] | select(.step==38) | .observation'
   ```

3. **Every claim carries a trace citation.** State findings as
   `the agent unlocked the box at step 38 ("You unlock the TextWorld style
   box.") then explored 10 more rooms through step 74` — assertion + step
   number + verbatim quote. Counts (rooms visited, actions, keys found) must be
   numbers you actually tallied with the queries above, never estimates or
   round numbers.

4. **Never fabricate.** If the trace doesn't show something — a key, a reward,
   a room, a loop — write "not observed in trace". Do NOT fill the gap from
   what a TextWorld game "would" contain, and do NOT trust the episode_summary
   as ground truth (it is the harness's own self-critique and can be wrong
   about locations and events — e.g. it may name the wrong room). Verify any
   summary claim against the step traces before echoing it. Admitting "not
   observed" is always acceptable; inventing is not.

## Key Rules

1. **NEVER `read_file` a whole log.** Use `bash`+`jq` (or `python3 -c`) to
   extract fields. Start from the condensed file; open a per-task file only when
   you need finer detail than it carries.

2. **Be efficient.** There is no hard step cap, but you will get a one-line reminder at step 50 — once you have enough signal, wrap up by calling `end_probe`. Each step should produce useful signal.

3. **Query recipes live in the schema above.** The injected schema includes
   `jq`/`python3` recipes tailored to THIS benchmark's log shape — adapt them
   rather than guessing field names.

4. **Focus on what's actionable for a harness edit.** The evolve agent wants to
   know: where the harness got stuck, which mechanism failed, and what harness
   change would fix it. If a passing and a failing task differ in an instructive
   way, surface that contrast.

5. **When done,** call `end_probe(findings=...)` with a structured summary.
   **Open with a one-line coverage note** naming the files and step ranges you
   actually read (e.g. "covered treasure_hunter 1-74, coin_collector 1-80").
   Then group findings by task / failure theme. Every claim must carry its
   `(step N: "…")` citation inline — drop any claim you cannot cite. End with a
   concrete suggested harness fix per theme. Keep it concise — the evolve agent
   reads this as its tool result and uses it to decide its next edit.
"""

PROBE_USER_TEMPLATE = """\
## Investigation

**Instructions:** {instructions}
**Iteration:** {iteration}
**Working Directory:** {agent_code_dir}

Treat the Instructions above as your task — investigate to answer exactly what
they ask, using `bash`+`jq` on the logs they point you to. Don't run a generic
sweep; be targeted — but once targeted, **exhaust the relevant logs/traces
fully before concluding, and ground every claim in a citation (step number /
file / verbatim quote). Never fabricate: if you didn't observe it, say "not
observed".** When finished, call `end_probe(findings=...)` with your structured
summary.
"""


# Generic log-structure doc injected into the EVOLVE probe prompt when the caller
# has no benchmark log-summary handler (so no benchmark-specific schema). Tells
# the sub-agent to discover the structure itself with ls + jq instead of assuming
# a shape.
GENERIC_SCHEMA_FALLBACK = """\
## Log JSON Schema

Each task result has `{task_id, success, metadata, interaction_log}`. The exact
fields vary by benchmark — discover the structure yourself with `ls` + `jq`:
```bash
# Top-level keys
cat ../eval_logs/iter_N/eval_M/*_condensed.json | jq 'keys'

# Shape of one task and its first trace entry
cat ../eval_logs/iter_N/eval_M/*_condensed.json | jq '.task_results[0] | keys'
cat ../eval_logs/iter_N/eval_M/*_condensed.json | jq '.task_results[0].interaction_log[0]'

# List failed task IDs
cat ../eval_logs/iter_N/eval_M/*_condensed.json | jq -r '.task_results[] | select(.success|not) | .task_id'
```
"""


def _select_probe_system_prompt(scope: str, log_schema: str = "") -> str:
    """Build the probe system prompt for the given scope.

    EVOLVE scope injects the benchmark log schema (or GENERIC_SCHEMA_FALLBACK when
    none is provided) into the ``__LOG_SCHEMA__`` sentinel of the EVOLVE prompt.
    META scope uses its fixed prompt unchanged (it probes ``.evolution_context/``,
    not eval_logs — no eval schema applies).
    """
    if scope == "evolve":
        return PROBE_SYSTEM_PROMPT_EVOLVE.replace(
            "__LOG_SCHEMA__", log_schema or GENERIC_SCHEMA_FALLBACK
        )
    return PROBE_SYSTEM_PROMPT_META


# ─── Runner ─────────────────────────────────────────────────────────────────

def _snapshot_harness_files(agent_code_dir: str) -> Dict[str, str]:
    """Snapshot every evolvable harness file's contents under ``agent_code_dir``.

    Captures all ``.py`` and ``.md`` files (excluding ``evolution/``,
    ``__pycache__``, and hidden dirs) — the full surface the agent may hold
    uncommitted edits in. ``run_probe`` writes these bytes back after its
    defensive ``git reset --hard`` so the agent's in-progress edits survive the
    probe (see ``run_probe`` for why that matters).
    """
    contents: Dict[str, str] = {}
    if not agent_code_dir or not os.path.isdir(agent_code_dir):
        return contents
    for root, dirs, files in os.walk(agent_code_dir):
        dirs[:] = [
            d for d in dirs
            if not d.startswith('.') and d != '__pycache__' and d != 'evolution'
        ]
        for fname in files:
            if not (fname.endswith('.py') or fname.endswith('.md')):
                continue
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, agent_code_dir).replace(os.sep, '/')
            try:
                with open(full_path, 'r', encoding='utf-8') as fh:
                    contents[rel_path] = fh.read()
            except Exception:
                pass
    return contents


def _restore_harness_files(agent_code_dir: str, contents: Dict[str, str]) -> None:
    """Write a snapshotted ``{rel_path: content}`` mapping back to disk.

    Only writes files present in the snapshot (does not delete others); the
    preceding ``git reset --hard`` already reconciled the git-tracked set.
    """
    for rel_path, content in contents.items():
        full_path = os.path.join(agent_code_dir, rel_path)
        parent = os.path.dirname(full_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception:
            pass


def run_probe(agent, instructions: str, max_steps: int = 50, scope: str = "meta_evolve", log_schema: str = "") -> str:
    """Spawn a read-only probe sub-agent to investigate logs.

    The sub-agent has access to read_file, bash (for jq), and end_probe
    (completion signal). It cannot edit, write, or evaluate.

    Args:
        agent: GodelAgent instance (provides react(), execute_tool(), get_tools()).
        instructions: What to investigate — passed to the sub-agent as its task.
        max_steps: Soft reminder threshold (default 50). The probe loop no longer
            hard-stops at this count — it runs until end_probe or a no-tool-call
            nudge — but when step reaches max_steps a single user message reminds
            the sub-agent to wrap up via end_probe if its investigation is done.
        scope: "evolve" or "meta_evolve" — selects the system prompt. The evolve
            prompt targets eval_logs/ (this eval's harness trajectory) and groups
            findings by task/failure-theme; the meta_evolve prompt targets
            .evolution_context/ cross-phase logs and groups by dimension
            (prompt/seed/commit/best).
        log_schema: Benchmark-specific log structure doc (from the log-summary
            handler's ``build_log_schema_description()``) injected into the
            EVOLVE prompt. Ignored for META scope. Empty → GENERIC_SCHEMA_FALLBACK.

    Returns:
        The findings string from end_probe, or a fallback message if the probe
        ended (no-tool-call nudge exhausted) without calling end_probe.
    """
    agent_code_dir = getattr(agent, "agent_code_dir", "") or ""

    # ── Build messages ──
    system_prompt = _select_probe_system_prompt(scope, log_schema)
    user_prompt = PROBE_USER_TEMPLATE.format(
        instructions=instructions,
        iteration=agent.iteration if hasattr(agent, "iteration") else "?",
        agent_code_dir=agent_code_dir,
    )

    messages: List[Dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # ── Tools: probe scope ──
    tools = agent.get_tools(scope="probe")

    # closure-mutable containers
    findings: List[str] = []
    done = [False]
    step = 0

    def tool_executor(tool_name: str, args: dict) -> str:
        if tool_name == "end_probe":
            findings.append(args.get("findings", ""))
            done[0] = True
            return "Investigation complete. Findings recorded."
        return agent.execute_tool(tool_name, args, scope="probe")

    # ── Save pre-probe HEAD for restoration ──
    pre_head = ""
    try:
        pre_head = agent.git_controller.get_current_commit() or ""
    except Exception:
        pass

    # ── Snapshot the agent's working-tree contents BEFORE probe ──
    # The finally below runs `git reset --hard pre_head` to clean up the probe
    # sub-agent's bash (it may run git checkout/reset and dirty the tree). But
    # --hard to pre_head ALSO discards the agent's OWN uncommitted edits that
    # existed before probe was called — a silent data-loss bug: the agent's
    # in-progress edit vanishes the moment it probes, and since an eval is
    # usually followed by a probe, the agent ends up re-applying the same edit
    # every cycle and/or mistaking the wipe for "eval reverted my code".
    # Snapshot the evolvable file *contents* here; write them back after the
    # reset so the agent's edits survive the probe intact.
    pre_contents: Dict[str, str] = _snapshot_harness_files(agent_code_dir)

    try:
        # ── Run sub-react-loop ──
        # No hard step cap: the loop runs until end_probe (or a no-tool-call
        # nudge). When step reaches `max_steps` we inject ONE soft reminder so
        # the sub-agent wraps up rather than drifting indefinitely.
        while True:
            step += 1

            response, tool_calls_made, tool_results = agent.react(
                messages=messages,
                tools=tools,
                tool_executor=tool_executor,
            )

            append_response_to_messages(
                messages, response, tool_calls_made, tool_results
            )

            # Surface this probe step inline (same box format as the main evolve
            # loop) so the operator can watch the sub-agent investigate in real
            # time. No step cap — always show "Step N" without denominator.
            log_react_step(agent, step, None, tool_calls_made, tool_results,
                           prefix="probe")

            if done[0]:
                break

            # Soft reminder once the step budget is reached: nudge the sub-agent
            # to finish via end_probe. Injected exactly once — only when step
            # first equals max_steps — as a user message the next react sees.
            if step == max_steps:
                messages.append({
                    "role": "user",
                    "content": (
                        f"You have reached {max_steps} steps. If your "
                        f"investigation is complete, call "
                        f"`end_probe(findings=...)` with your structured summary "
                        f"to finish. If not, continue investigating with "
                        f"`bash`+`jq` or `read_file`."
                    ),
                })

            if not tool_calls_made:
                # Nudge: one more chance, then break
                messages.append({
                    "role": "user",
                    "content": (
                        "You did not use any tools in your last response. "
                        "If your investigation is complete, call "
                        "`end_probe(findings=...)` with your structured summary. "
                        "Otherwise, use `bash` with `jq` or `read_file` to "
                        "continue investigating."
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
                # Render the nudge's follow-up step too (same step index — it is a
                # retry of this step after the no-tool-call nudge).
                log_react_step(agent, step, None, confirm_tool_calls, confirm_results,
                               prefix="probe")
                if done[0]:
                    break
                if not confirm_tool_calls:
                    break

        # ── Return findings ──
        if findings and findings[0].strip():
            return findings[0].strip()

        # Loop ended without end_probe (no-tool-call nudge exhausted) — return
        # partial findings / fallback.
        return (
            "[Probe] Investigation ended without explicit findings. "
            "The sub-agent stopped without calling end_probe. "
            "Consider re-running with more specific instructions or a narrower scope."
        )

    except Exception as e:
        # Save trace even on exception (finally runs too, but messages may be
        # more complete here — save what we have before the return unwinds).
        try:
            trace_dir = _determine_trace_dir(scope, instructions, agent_code_dir)
            metadata = {
                "scope": scope,
                "iteration": getattr(agent, "iteration", "?"),
                "step_count": step,
                "end_probe_called": False,
                "end_reason": f"exception: {e}",
                "instructions_preview": instructions[:300] if instructions else "",
                "final_findings": "",
                "agent_code_dir": agent_code_dir,
            }
            _save_probe_trace(messages, metadata, trace_dir)
        except Exception:
            pass
        return f"[Probe] Investigation failed: {e}"

    finally:
        # Save probe trace for audit (before HEAD restore)
        try:
            trace_dir = _determine_trace_dir(scope, instructions, agent_code_dir)
            metadata = {
                "scope": scope,
                "iteration": getattr(agent, "iteration", "?"),
                "step_count": step,
                "end_probe_called": bool(findings and findings[0].strip()),
                "end_reason": "end_probe" if done[0] else "no_tool_call_exhaustion",
                "instructions_preview": instructions[:300] if instructions else "",
                "final_findings": findings[0] if findings else "",
                "agent_code_dir": agent_code_dir,
            }
            _save_probe_trace(messages, metadata, trace_dir)
        except Exception:
            pass

        # Always restore HEAD / working tree. The probe's bash may have
        # dirtied the working tree (git checkout, etc.).
        if pre_head:
            try:
                agent.git_controller._run_git_command(
                    ["reset", "--hard", pre_head], check=False
                )
            except Exception:
                pass

        # Restore the agent's pre-probe working-tree edits. `reset --hard` above
        # reconciled the git-tracked set (undoing the probe's commits/checkouts,
        # including evolution/); now overwrite the evolvable files with the
        # agent's snapshotted content so its uncommitted edits are back. Without
        # this, every probe silently destroys the agent's in-progress work.
        if pre_contents:
            try:
                _restore_harness_files(agent_code_dir, pre_contents)
                # Re-sync the in-memory agent_codes mirror so the agent's next
                # action (edit_file reads disk directly; bash code-views and
                # _harness_source read the mirror) sees the restored bytes.
                action_executor = getattr(agent, "action_executor", None)
                if action_executor is not None:
                    for rel_path, content in pre_contents.items():
                        if rel_path.endswith('.py'):
                            action_executor._reload_single_file(
                                rel_path, content, skip_validation=True
                            )
            except Exception:
                pass
