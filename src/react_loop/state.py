"""
State machine and data structures for React Loop Agent.

Core design: Agent autonomously decides actions, not a fixed SOP.

The agent collects π_t (strategy), S_t (environment), r_t (reward)
through self-directed actions until an iteration is complete.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import math
import statistics
import uuid


def fmt_reward(r) -> str:
    """Format reward for display. Handles None, float, dict, and list rewards."""
    if r is None:
        return "N/A"
    if isinstance(r, list):
        if len(r) == 0:
            return "[]"
        if len(r) == 1:
            return fmt_reward(r[0])
        return "[" + ", ".join(fmt_reward(x) for x in r) + "]"
    if isinstance(r, dict):
        vals = ", ".join(f"{k}={v:.4f}" for k, v in r.items() if isinstance(v, (int, float)))
        return "{" + vals + "}"
    return f"{r:.4f}"


def fmt_iteration_status(metadata: dict) -> str:
    """Derive iteration status from record metadata."""
    if metadata.get("max_steps_reached", False):
        return "MAX_STEPS"
    if metadata.get("success", True):
        return "SUCCEEDED"
    return "FAILED"


def reward_to_scalar(reward) -> float:
    """Convert reward (float, dict, or None) to a scalar value.

    - float/int reward: return as float
    - dict reward: use 'scalar_reward' key if present, else average of numeric values
    - None: return 0.0
    """
    if reward is None:
        return 0.0
    if isinstance(reward, (int, float)):
        return float(reward)
    if isinstance(reward, dict):
        if "scalar_reward" in reward and isinstance(reward["scalar_reward"], (int, float)):
            return float(reward["scalar_reward"])
        vals = [v for k, v in reward.items()
                if isinstance(v, (int, float)) and k != "scalar_reward"]
        return sum(vals) / len(vals) if vals else 0.0
    return 0.0


def lower_confidence_bound(values, z: float = 1.0) -> float:
    """Lower Confidence Bound (LCB): mean − z·stdev(ddof=1)/√n.

    Used for uncertainty-aware reward at the per-evaluate layer (E_LCB): the
    caller passes the list of episode progressions for a single task (e.g. the
    balrog evaluator's ``lcb_progression``).

    When there is no uncertainty the LCB is the mean (a no-op): n < 2 leaves
    stdev undefined → mean; an all-identical list has stdev=0 → mean. A higher
    z penalizes jittery estimates harder. Mirrors the _summarize_reward_runs
    pattern in main.py (same sample-std ddof=1 + n>=2 guard).

    Args:
        values: list of numeric rewards.
        z: z-score (conservatism). Default 1.0 ≈ ~84th-percentile lower bound.

    Returns:
        The lower-confidence bound as a float (0.0 for empty input).
    """
    n = len(values)
    if n == 0:
        return 0.0
    mean = statistics.fmean(values)
    if n < 2:
        return mean
    std = statistics.stdev(values)  # sample stdev, ddof=1
    if std == 0:
        return mean
    return mean - z * std / math.sqrt(n)


def rank_versions_by_best_reward(
    snapshots: List[list],
    code_snapshots: Dict[str, Any] = None,
) -> List[dict]:
    """Rank evaluated code versions by their best single-eval reward.

    Single source of truth for best-reward version ranking — shared by
    ``select_commit`` and ``EvolveHelper._find_best_code_version`` so the two
    never drift apart.

    Groups ``evaluation_snapshots`` entries by ``(code_hash, mode)``, takes
    ``max(per-eval scalar rewards)`` as ``best_reward`` for each group, and
    returns the groups of the **winning layer** sorted best-first (val layer if
    any hash has a val eval, else the dev layer). This preserves the existing
    val-preferred selection semantics: a val eval anywhere makes dev evals
    irrelevant for picking the best.

    Each entry is a dict::

        {code_hash, mode, best_reward, n_evals, mean}

    sorted by best_reward desc, then n_evals desc (reliability), then mean desc.
    The head ``[0]`` is the best version. Empty list when no valid snapshots.

    Args:
        snapshots: ``state.evaluation_snapshots`` — list of
            ``[code_hash, reward, eval_mode]`` (2-element legacy form tolerated).
        code_snapshots: ``state.code_snapshots`` — hashes not present here are
            skipped (their file contents can't be restored). None skips the filter.

    Returns:
        Ranked list of per-version dicts (winning layer only, best first).
    """
    # Gather scalar rewards per (code_hash, mode)
    per_group: Dict[tuple, List[float]] = {}
    for snap in snapshots:
        if len(snap) >= 3:
            code_hash, reward, mode = snap[0], snap[1], snap[2]
        elif len(snap) >= 2:
            code_hash, reward, mode = snap[0], snap[1], "dev"
        else:
            continue
        if code_snapshots is not None and code_hash not in code_snapshots:
            continue
        mode = "val" if mode == "val" else "dev"
        per_group.setdefault((code_hash, mode), []).append(reward_to_scalar(reward))

    if not per_group:
        return []

    # Val-preferred: if any hash was evaluated on val, rank val evals only;
    # otherwise fall back to dev evals. Never mix the two layers.
    layer = "val" if any(mode == "val" for (_, mode) in per_group) else "dev"

    rows = []
    for (code_hash, mode), scalars in per_group.items():
        if mode != layer:
            continue
        n = len(scalars)
        rows.append({
            "code_hash": code_hash,
            "mode": mode,
            "best_reward": max(scalars),
            "n_evals": n,
            "mean": statistics.fmean(scalars),
        })
    # best_reward desc → n_evals desc (more evals = more reliable) → mean desc
    rows.sort(key=lambda e: (-e["best_reward"], -e["n_evals"], -e["mean"]))
    return rows


# Harness-root Markdown files that are NOTEBOOK / MEMORY, not strategy code.
# Used by code_hash computation, file scanning, orphan cleanup, and best-version
# restore — all of which must skip them because they describe the evolution, not
# the version being evaluated/committed. Single source of truth; import this
# everywhere rather than hardcoding filenames.
#   - BOOTSTRAP.md: cross-iteration lesson memory (tracked + protected).
#   - plan.md: iteration-scoped working notebook (gitignored, ephemeral).
METADATA_FILES = frozenset({"BOOTSTRAP.md", "plan.md"})


class ActionType(Enum):
    """
    Action types available to the agent.

    Tool architecture (simplified):
    - Built-in core tools: bash (auto-syncs the agent_codes mirror after .py edits), read_history_self
    - File-operation tools: read_file, edit_file, write_file (split out from the former editor)
    - Required external tool (1): evaluate
    - Optional external tools (dynamic): external_tool
    - Historical-version tool (1): get_historic_version

    bash is a built-in core tool; after .py files are modified it auto-syncs
    on-disk changes into the agent_codes mirror (evaluate fresh-imports from
    disk each time, so no hot-reload layer is needed).
    read_history_self is used only to read historical iteration conversations.
    read_file/edit_file/write_file are file-operation tools supporting exact
    replacement and full overwrite.
    """
    BASH = "bash"                       # Execute shell commands + disk→agent_codes sync
    POWERSHELL = "powershell"            # Windows PowerShell command + disk→agent_codes sync
    READ_HISTORY_SELF = "read_history_self"  # Read historical iteration conversations (@history syntax)
    EVALUATE = "evaluate"               # External evaluation (collects r_t)
    EXTERNAL_TOOL = "external_tool"     # Execute external tools (dynamically loaded)
    GET_HISTORIC_VERSION = "get_historic_version"  # Get historical-version code
    GET_HISTORIC_EVAL_CODE = "get_historic_eval_code"  # Get the code snapshot evaluate saw within the current iteration
    READ_FILE = "read_file"             # Read a file (with line numbers)
    EDIT_FILE = "edit_file"             # Exact string replacement
    WRITE_FILE = "write_file"           # Create / overwrite a file
    PLAN = "plan"                       # Write plan.md: this iteration's working hypothesis/plan/progress (ephemeral)
    LESSON = "lesson"                   # Write BOOTSTRAP.md: cumulative cross-iteration lesson (not rolled back)
    COMPACT_CONTEXT = "compact_context"  # End the current iteration (agent-initiated)
    END_EVOLUTION = "end_evolution"      # End the entire evolution (agent-initiated)
    END_META_EVOLUTION = "end_meta_evolution"  # End the current meta-evolve phase
    VALIDATE_ARCHIVE = "validate_archive"  # Validate that select_*.py edits are well-formed
    END_PROBE = "end_probe"            # End the probe sub-agent investigation


# Single-source mapping from tool name to ActionType
ACTION_TYPE_MAP: Dict[str, ActionType] = {
    "bash": ActionType.BASH,
    "powershell": ActionType.POWERSHELL,
    "read_history_self": ActionType.READ_HISTORY_SELF,
    "evaluate": ActionType.EVALUATE,
    "get_historic_version": ActionType.GET_HISTORIC_VERSION,
    "get_historic_eval_code": ActionType.GET_HISTORIC_EVAL_CODE,
    "read_file": ActionType.READ_FILE,
    "edit_file": ActionType.EDIT_FILE,
    "write_file": ActionType.WRITE_FILE,
    "compact_context": ActionType.COMPACT_CONTEXT,
    "end_evolution": ActionType.END_EVOLUTION,
    "end_meta_evolution": ActionType.END_META_EVOLUTION,
    "validate_archive": ActionType.VALIDATE_ARCHIVE,
    "plan": ActionType.PLAN,
    "lesson": ActionType.LESSON,
    "end_probe": ActionType.END_PROBE,
}


@dataclass
class AgentAction:
    """
    Represents a single action taken by the agent.

    Records what the agent did, with what parameters, and what resulted.
    """
    action_type: ActionType
    params: Dict[str, Any] = field(default_factory=dict)
    result: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize action to dictionary."""
        return {
            "action_type": self.action_type.value,
            "params": self.params,
            "result": self.result,
            "timestamp": self.timestamp,
        }

    def to_llm_dict(self) -> Dict[str, Any]:
        """Serialize action for LLM context (excludes timestamp)."""
        return {
            "action_type": self.action_type.value,
            "params": dict(self.params),
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentAction":
        """Deserialize action from dictionary."""
        return cls(
            action_type=ActionType(data["action_type"]),
            params=data.get("params", {}),
            result=data.get("result", ""),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class AgentState:
    """
    Agent state for tracking evolution iteration progress.

    Key insight: The agent autonomously decides when to end
    an iteration (via compact_context) and when to end the
    entire evolution (via end_evolution).

    Code is stored as a dict of {relative_path: code} in pi_codes,
    with pi_code_dir as the base directory.
    """
    # Iteration number
    iteration: int = 0

    # π_t - Current strategy (agent's own code)
    # Stored as a {relative_path: code} dict
    pi_codes: Dict[str, str] = field(default_factory=dict)
    pi_code_dir: str = ""  # Base directory path

    # S_t - Environment state (collected through sensing actions)
    environment_summary: str = ""
    environment_data: Dict[str, Any] = field(default_factory=dict)

    # r_t - Reward (obtained through evaluation action)
    # float for single-category, dict for multi-category (e.g. {"utility_rate": ..., "security_rate": ..., "overall_rate": ...})
    reward: float | dict | None = None
    iteration_rewards: List[float | dict] = field(default_factory=list)  # All rewards within this iteration
    last_eval_mode: str = ""  # "dev" or "val" — mode of most recent evaluate call

    # Code-reward alignment: snapshots of (code_hash, reward, eval_mode) taken at each evaluate call
    evaluation_snapshots: List[list] = field(default_factory=list)  # [[code_hash, reward, eval_mode], ...]

    # Code snapshots per unique hash — maps code_hash → {rel_path: file_content}
    # Used to restore the best-performing code at commit time
    code_snapshots: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # Evaluation metrics (detailed benchmark results from external_evaluator)
    iteration_metrics: List[Dict[str, Any]] = field(default_factory=list)  # All metrics within this iteration
    last_evaluation_metrics: Optional[Dict[str, Any]] = None  # Latest evaluation metrics

    # g - Goal (provided by user)
    goal: str = ""

    # Agent-autonomous iteration control
    iteration_ended: bool = False
    evolution_ended: bool = False
    iteration_summary_text: str = ""
    iteration_end_reason: str = ""

    # Reasoning contents - per-step reasoning content (from the LLM's reasoning_content)
    reasoning_contents: List[str] = field(default_factory=list)

    # Action history for this iteration
    action_history: List[AgentAction] = field(default_factory=list)

    # Git version information
    commit_hash: str = ""
    parent_commit: str = ""

    # Self-modification tracking
    modifications_made: List[Dict[str, Any]] = field(default_factory=list)
    last_modification_result: str = ""

    # Failure tracking
    max_steps_reached: bool = False

    # Commit pool — agent calls pick_commit_version during main loop to ADD
    # versions to a cumulative pool (not a single bookmark). Each call appends;
    # the final commit decision is made as a continuation of the same conversation
    # (cache-preserving nudge) using finalize_commit_pool to select multiple versions.
    commit_picks: List[Dict[str, Any]] = field(default_factory=list)
    # Each entry: {"code_hash": str, "reason": str, "step": int}

    # Pool commits created during commit_iteration (multiple per iteration).
    # Used by select_seed to discover pool entries as candidates.
    pool_commits: List[str] = field(default_factory=list)

    # Lesson tracking — True once the agent has called lesson() this iteration.
    # Reset to False on each fresh AgentState (reset_for_iteration). Drives the
    # iteration-end lesson-nudge fallback: if still False after the main loop,
    # the framework nudges the agent to record a cross-iteration lesson.
    lesson_recorded: bool = False

    def is_iteration_complete(self) -> bool:
        """
        Check if the iteration is complete.

        An iteration is complete when the agent calls compact_context
        or end_evolution, or when max_steps is reached.

        Returns:
            True if the iteration has been ended.
        """
        return self.iteration_ended

    def mark_iteration_ended(self, summary: str = "", reason: str = "") -> None:
        """Mark the current iteration as ended by the agent."""
        self.iteration_ended = True
        self.iteration_summary_text = summary
        self.iteration_end_reason = reason

    def mark_evolution_ended(self, summary: str = "", reason: str = "") -> None:
        """Mark the entire evolution as ended by the agent."""
        self.iteration_ended = True
        self.evolution_ended = True
        self.iteration_summary_text = summary
        self.iteration_end_reason = reason

    def record_action(self, action: AgentAction) -> None:
        """
        Record an action in the history.

        Args:
            action: The action to record.
        """
        action.timestamp = datetime.now().isoformat()
        self.action_history.append(action)

    def update_pi(self, codes: Dict[str, str], code_dir: str = "") -> None:
        """
        Update π_t (strategy/code).

        Args:
            codes: Dict of {relative_path: code}
            code_dir: Base directory path
        """
        self.pi_codes = codes
        if code_dir:
            self.pi_code_dir = code_dir

    def get_all_codes(self) -> Dict[str, str]:
        """
        Get all code content.

        Returns:
            Dict of {path: code}
        """
        return self.pi_codes.copy()

    def get_first_code(self) -> Tuple[str, str]:
        """
        Get the first file's code (for backward compatibility).

        Returns:
            Tuple of (path, code)
        """
        if self.pi_codes:
            first_file = sorted(self.pi_codes.keys())[0]
            return first_file, self.pi_codes[first_file]
        return "", ""

    def update_environment(self, summary: str, data: Dict[str, Any] = None) -> None:
        """Update S_t (environment state)."""
        self.environment_summary = summary
        if data:
            self.environment_data.update(data)

    def update_reward(self, reward: float | dict, eval_mode: str = "") -> None:
        """Update r_t (reward) and track in-iteration history."""
        self.reward = reward
        self.iteration_rewards.append(reward)
        if eval_mode:
            self.last_eval_mode = eval_mode

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update evaluation metrics and track in-iteration history."""
        self.last_evaluation_metrics = metrics
        self.iteration_metrics.append(metrics)

    def get_scalar_reward(self) -> float:
        """Extract a scalar value from reward for comparison/tracking."""
        return reward_to_scalar(self.reward)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "iteration": self.iteration,
            "pi_codes": self.pi_codes,
            "pi_code_dir": self.pi_code_dir,
            "environment_summary": self.environment_summary,
            "environment_data": self.environment_data,
            "reward": self.reward,
            "iteration_rewards": self.iteration_rewards,
            "last_eval_mode": self.last_eval_mode,
            "evaluation_snapshots": self.evaluation_snapshots,
            "code_snapshots": self.code_snapshots,
            "iteration_metrics": self.iteration_metrics,
            "last_evaluation_metrics": self.last_evaluation_metrics,
            "goal": self.goal,
            "iteration_ended": self.iteration_ended,
            "evolution_ended": self.evolution_ended,
            "iteration_summary_text": self.iteration_summary_text,
            "iteration_end_reason": self.iteration_end_reason,
            "reasoning_contents": self.reasoning_contents,
            "action_history": [a.to_dict() for a in self.action_history],
            "commit_hash": self.commit_hash,
            "parent_commit": self.parent_commit,
            "modifications_made": self.modifications_made,
            "max_steps_reached": self.max_steps_reached,
            "commit_picks": self.commit_picks,
            "pool_commits": self.pool_commits,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Deserialize state from dictionary."""
        return cls(
            iteration=data.get("iteration", 0),
            pi_codes=data.get("pi_codes", {}),
            pi_code_dir=data.get("pi_code_dir", ""),
            environment_summary=data.get("environment_summary", ""),
            environment_data=data.get("environment_data", {}),
            reward=data.get("reward", 0.0),
            iteration_rewards=data.get("iteration_rewards", []),
            last_eval_mode=data.get("last_eval_mode", ""),
            evaluation_snapshots=data.get("evaluation_snapshots", []),
            code_snapshots=data.get("code_snapshots", {}),
            iteration_metrics=data.get("iteration_metrics", []),
            last_evaluation_metrics=data.get("last_evaluation_metrics"),
            goal=data.get("goal", ""),
            iteration_ended=data.get("iteration_ended", False),
            evolution_ended=data.get("evolution_ended", False),
            iteration_summary_text=data.get("iteration_summary_text", ""),
            iteration_end_reason=data.get("iteration_end_reason", ""),
            reasoning_contents=data.get("reasoning_contents", []),
            action_history=[
                AgentAction.from_dict(a) for a in data.get("action_history", [])
            ],
            commit_hash=data.get("commit_hash", ""),
            parent_commit=data.get("parent_commit", ""),
            modifications_made=data.get("modifications_made", []),
            max_steps_reached=data.get("max_steps_reached", False),
            commit_picks=data.get("commit_picks", []),
            pool_commits=data.get("pool_commits", []),
        )


class EvolutionPhase(Enum):
    """
    Phases of the evolution process.

    Note: These are for tracking/logging purposes only.
    The agent still autonomously decides its actions.
    """
    INITIALIZING = "initializing"
    EVOLVING = "evolving"
    COMPACTING = "compacting"
    COMMITTING = "committing"
    COMPLETED = "completed"
    ERROR = "error"
    META_EVOLVING = "meta_evolving"
    META_COMMITTING = "meta_committing"


@dataclass
class MessageHistory:
    """
    Message-history manager for the LLM conversation context.

    Supports system/user/assistant/tool message types and enables
    multi-turn conversation and cache hits.

    Attributes:
        messages: List of messages; each carries a role and content.
    """
    messages: List[Dict[str, Any]] = field(default_factory=list)

    def add_system(self, content: str) -> None:
        """Add a system message (typically added only once at the start of an iteration)."""
        self.messages.append({
            "role": "system",
            "content": content
        })

    def add_user(self, content: str) -> None:
        """Add a user message."""
        self.messages.append({
            "role": "user",
            "content": content
        })

    def add_assistant(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        reasoning_content: Optional[str] = None,
    ) -> None:
        """Add an assistant message (may include text content, tool calls, and reasoning content)."""
        message: Dict[str, Any] = {"role": "assistant"}
        if content:
            message["content"] = content
        if tool_calls:
            message["tool_calls"] = tool_calls
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        # API requires content or tool_calls on every assistant message
        if not message.get("content") and not message.get("tool_calls"):
            message["content"] = ""
        self.messages.append(message)

    def add_tool(self, content: str, tool_call_id: str) -> None:
        """Add a tool-response message."""
        self.messages.append({
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call_id
        })

    def remove_last(self) -> Optional[Dict[str, Any]]:
        """Pop and return the last message; return None when the list is empty."""
        if self.messages:
            return self.messages.pop()
        return None

    def get_messages_for_llm(self) -> List[Dict[str, Any]]:
        """Get the message list for the LLM call (deep-copied).

        Note: filtering of reasoning_content is handled centrally in
        GodelAgent.call_llm() based on the thinking_enabled setting.
        """
        return [msg.copy() for msg in self.messages]

    def truncate(self, max_messages: int = 50) -> None:
        """
        Truncate the message history, keeping system messages and the most recent conversation.

        Guarantees:
        1. assistant(tool_calls) messages are not split from their corresponding tool responses.
        2. The truncation start is rewound to the most recent ``assistant(tool_calls)`` — a
           necessary companion to continuation-only mode (after the first user message,
           continuation relies on tool results): we no longer depend on "the first non-system
           message must be a user message". If the first non-system message after truncation is
           not a user message (i.e. starts with assistant), insert a placeholder user message
           to stay compatible with endpoints that reject assistant-first conversations.

        Args:
            max_messages: Max number of messages (excluding system messages).
        """
        if len(self.messages) <= max_messages:
            return

        # Separate system messages from the rest
        system_messages = [m for m in self.messages if m["role"] == "system"]
        other_messages = [m for m in self.messages if m["role"] != "system"]

        if len(other_messages) <= max_messages:
            return

        # Keep the most recent non-system messages
        all_other = other_messages
        start_idx = len(all_other) - max_messages

        # Walk back to the most recent assistant(tool_calls): guarantees the
        # first retained message carries tool_calls, so the tool responses that
        # immediately follow it don't become orphans (split from their assistant).
        while start_idx > 0:
            m = all_other[start_idx]
            if m["role"] == "assistant" and m.get("tool_calls"):
                break
            start_idx -= 1

        kept = all_other[start_idx:]

        # Fallback: if the first non-system message after truncation is not a
        # user message (pure assistant-first), some endpoints (certain Jinja
        # templates that require user-first) will reject it. Insert a
        # placeholder user for compatibility. glm accepts system→assistant
        # openings; the insertion is harmless for it (just one extra short user).
        if kept and kept[0]["role"] != "user":
            kept = [{"role": "user", "content": "[earlier steps truncated]"}] + kept

        self.messages = system_messages + kept

    def clear(self) -> None:
        """Clear the message history."""
        self.messages = []

    def __len__(self) -> int:
        return len(self.messages)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict."""
        return {
            "messages": self.messages.copy()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessageHistory":
        """Deserialize from a dict."""
        return cls(messages=data.get("messages", []))


@dataclass
class IterationSummary:
    """
    Summary info for a single iteration.

    Used to pass context across iterations, helping the LLM understand
    evolution history.

    Attributes:
        iteration: Iteration number.
        reward: This iteration's reward value.
        metrics: Detailed anti-fragility metrics.
        summary_text: A 1-2 sentence summary produced by the LLM.
        modifications_count: Number of modifications.
        key_decisions: List of key decisions.
        commit_hash: Git commit hash.
        timestamp: Timestamp.
        success: Whether the iteration completed successfully.
    """
    iteration: int
    reward: float
    metrics: Dict[str, float] = field(default_factory=dict)
    summary_text: str = ""
    modifications_count: int = 0
    key_decisions: List[str] = field(default_factory=list)
    commit_hash: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict."""
        return {
            "iteration": self.iteration,
            "reward": self.reward,
            "metrics": self.metrics,
            "summary_text": self.summary_text,
            "modifications_count": self.modifications_count,
            "key_decisions": self.key_decisions,
            "commit_hash": self.commit_hash,
            "timestamp": self.timestamp,
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationSummary":
        """Deserialize from a dict."""
        return cls(
            iteration=data.get("iteration", 0),
            reward=data.get("reward", 0.0),
            metrics=data.get("metrics", {}),
            summary_text=data.get("summary_text", ""),
            modifications_count=data.get("modifications_count", 0),
            key_decisions=data.get("key_decisions", []),
            commit_hash=data.get("commit_hash", ""),
            timestamp=data.get("timestamp", ""),
            success=data.get("success", True),
        )


@dataclass
class EvolutionContext:
    """
    Evolution context, manages cross-iteration historical info.

    Used for:
    1. Tracking the best version and the evolution trend.
    2. Providing historical context to the LLM.
    3. Persisting to the filesystem.

    Attributes:
        iteration_summaries: List of summaries for all iterations.
        best_reward: Historical best reward.
        best_iteration: The iteration number that produced the best reward.
        track_best_reward: Whether to track the best version by reward. The
            main evolution context is True; the meta-evolve context should be
            False — the meta-evolve phase does not evaluate code and has no
            concept of reward, so best_reward/best_iteration do not apply and
            are serialized as null.
    """
    iteration_summaries: List[IterationSummary] = field(default_factory=list)
    best_reward: float = 0.0
    best_iteration: int = 0
    track_best_reward: bool = True

    def add_summary(self, summary: IterationSummary) -> None:
        """
        Add an iteration summary and update statistics.

        Args:
            summary: Iteration summary.
        """
        self.iteration_summaries.append(summary)

        # Update best record (reward tracking is disabled in the meta-evolve context)
        if self.track_best_reward and summary.reward > self.best_reward:
            self.best_reward = summary.reward
            self.best_iteration = summary.iteration

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict."""
        return {
            "iteration_summaries": [s.to_dict() for s in self.iteration_summaries],
            # When track_best_reward=False these two fields do not apply; write null to avoid misleading values.
            "best_reward": self.best_reward if self.track_best_reward else None,
            "best_iteration": self.best_iteration if self.track_best_reward else None,
            "track_best_reward": self.track_best_reward,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionContext":
        """Deserialize from a dict."""
        return cls(
            iteration_summaries=[
                IterationSummary.from_dict(s)
                for s in data.get("iteration_summaries", [])
            ],
            # Backward-compatible with old files (no track_best_reward key → default True) and null values.
            best_reward=(data.get("best_reward") or 0.0),
            best_iteration=(data.get("best_iteration") or 0),
            track_best_reward=data.get("track_best_reward", True),
        )
