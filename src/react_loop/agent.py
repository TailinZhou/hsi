"""
Godel Agent - Self-evolving agent with autonomous decision-making.

Core design: The agent autonomously decides its actions, NOT a fixed SOP.

Core principles:
- `react` + atomic tools are the core capabilities of the Godel Agent.
- `evolve` is built on top of `react`.
- `harness` should be passed into `agent.react` for construction.

Evolution formula: π_{t+1}, I_{t+1} = I_t(π_t, S_t, r_t, g)
- π_t: Current agent strategy/code
- S_t: Environment state
- r_t: Reward (including anti-fragility metrics)
- g: Goal
- I_t: Improver (the agent itself, deciding actions via _decide_action)

The agent collects π_t, S_t, r_t through self-directed actions
until an iteration is complete.

Uses folder mode: godel_harness_init_path points to a directory with multiple .py files (default: godel_harness_init/agentdojo).
"""

import os
import re
import sys, json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple

from .state import (
    AgentState,
    AgentAction,
    ActionType,
    ACTION_TYPE_MAP,
    EvolutionPhase,
    MessageHistory,
    IterationSummary,
    fmt_reward,
)
from .git_version.controller import GitController, EvolutionTracker
from .utils.json_parser import (
    fix_and_parse_json,
    clean_json_values,
    parse_action_from_response,
)
from .utils.tools import (
    build_openai_tools,
    get_shell_tool_schema,
    build_external_tool_schema,
    scan_external_tools,
    benchmark_block_message,
    _BENCHMARK_SHIELDED_SCOPES,
)
from .utils.log_format import _C, _tool_color
from .utils.usage_logger import UsageLogger, extract_usage, format_usage_report
from .actions.agent_action import AgentActionExecutor
from .context_persistence import ContextPersistence
from .evolve import EvolveHelper, PICK_COMMIT_VERSION_SCHEMA, FINALIZE_COMMIT_POOL_SCHEMA



@dataclass
class GodelAgentConfig:
    """Configuration for the Godel Agent."""
    max_iterations: int = 10
    max_steps_per_iteration: int = 50
    godel_harness_init_path: str = ""  # Initial code path (read-only; copied into repo_path)
    output_dir: str = "./evolved_agents"
    model: str = "gpt-4"
    verbose: bool = True
    enable_bash: bool = True  # Whether to enable the bash tool (evolve stage)
    harness_enable_bash: bool = False  # Whether to enable the bash tool (harness/evaluate stage)

    temperature: float = 0.7  # LLM sampling temperature (evolve stage)
    harness_temperature: float = None  # LLM sampling temperature (harness stage); None = follow temperature
    thinking_enabled: bool = False  # Whether to enable thinking mode
    harness_thinking_enabled: Optional[bool] = None  # Thinking mode for harness/evaluate stage; None = follow thinking_enabled
    harness_max_tokens: Optional[int] = None  # max_tokens for harness stage; None = no limit
    reasoning_effort: str = "high"  # Reasoning effort: "low", "high", "max"
    max_context_tokens: int = 68000  # Context window token cap

    # Message history truncation config
    max_history_messages: int = 15  # Max number of messages retained in history (excluding system messages)
    max_tool_result_length: int = 100000  # Max characters for a tool result (large value = effectively no truncation)

    # Evaluate result handling config
    evaluate_llm_summary: bool = True  # Whether to use LLM to summarize evaluate results (false = return only failed tasks)
    evaluate_consolidate_summary: bool = False  # Whether to run cross-episode LLM consolidation on multi-episode summaries
                                                # (false = keep the full per-episode raw diary; no consolidate LLM call)
    # Bootstrap: evaluate the init harness once on the dev set before evolution starts. Two outputs:
    #   1. The init harness's own "experience" text → injected into iteration 1's first prompt
    #      (so the first self-rewrite is driven by the harness's measured pain-points,
    #      not a cold-start code read).
    #   2. A real dev reward written back to the iteration-0 seed record (replacing the
    #      placeholder 0.0), giving select_seed + iteration-1's baseline a real grounding.
    # Disable to skip bootstrap (fresh runs also cold-start; seed reward stays 0.0).
    init_eval_enabled: bool = True
    eval_feedback_style: str = "diary"  # Episode feedback tone: "diary" (player-first-person harness pain log) | "summary" (third-person objective, A/B comparison)
    enable_file_log: bool = False  # When True, return a condensed eval result + a log path hint

    # LCB (Lower-Confidence-Bound) reward — z-score for uncertainty penalty.
    # Single source: evolution.lcb_zscore in config.yaml. Higher z penalizes
    # jittery reward estimates harder (reward = mean − z·std/√n).
    lcb_zscore: float = 1.0

    # Meta-evolution config
    godel_evolution_init_path: str = ""  # Initial evolution-strategy path
    meta_evolve_enabled: bool = True     # Whether meta-evolve is enabled
    meta_evolve_max_steps: int = 10      # Max steps for meta-evolve
    meta_evolve_enable_bash: bool = True  # Whether to enable the bash tool (meta-evolve stage)
    iter_per_metaevolve: int = 1         # Run meta-evolve every N iterations
    archive_enabled: bool = True         # Whether to enable archive-based evolution graph version management
    archive_strategy: str = "greedy"  # Initial strategy name
    evolvable_commit_strategy: bool = True  # Whether to enable LLM-nudge commit confirmation (nudge prompt provided by select_commit.py)
    commit_nudge_max_steps: int = 3       # Max react steps during the commit nudge phase (0 = skip nudge, go straight to fallback)
    lesson_nudge_max_steps: int = 2       # Max react steps during the lesson nudge phase (0 = skip; fallback when agent didn't call lesson())
    seed_selection_max_steps: int = 10    # Step cap for the seed-selection react loop
    inject_seed_hypothesis: bool = True   # Whether to inject the seed hypothesis into the evolve system prompt
    seed_eval_enabled: bool = False       # Whether evaluate is allowed during the seed-selection phase
    seed_eval_max_calls: int = 1          # Max number of evaluate calls during seed selection

    # Submit-best — fixed, non-evolvable agentic stage that picks which version
    # to export as the final best. Runs a react loop at the end of every
    # evolution (with optional ensemble fusion). The strategy loads from the
    # init template (godel_evolution_init/), NOT evolution/, so meta-evolve
    # physically cannot edit it. Falls back to get_best_version("highest_reward")
    # when disabled or on any miss.
    submit_best_enabled: bool = True      # Whether to enable the agentic select_best stage
    submit_best_max_steps: int = 50       # Step cap for the submit_best react loop

    # Knowledge graph — fully-connected evolution knowledge graph (cross-iteration experience sharing)
    meta_evolve_kg_enabled: bool = True        # Whether to enable the evolution knowledge graph (default on: experience-sharing companion to meta-evolve)
    meta_evolve_kg_max_nodes: int = 100        # Upper bound on graph nodes (older low-value nodes are pruned when exceeded)
    meta_evolve_kg_concurrency: int = 2        # Concurrency for the LLM analysis that builds semantic edges

    evolve_exclude_tools: List[str] = field(default_factory=list)

    resume_from: Optional[dict] = None

_NO_OVERRIDE = object()  # sentinel for thread-local overrides


class GodelAgent:
    """
    Godel Agent - Self-evolving agent with autonomous decision-making.

    Key principle: Agent autonomously decides actions, not a fixed SOP.

    The agent uses LLM function calling to select actions, collecting
    π_t, S_t, r_t through self-directed actions until iteration is complete.
    """

    def __init__(
        self,
        repo_path: str,
        llm_client,
        goal: str,
        config: GodelAgentConfig = None,
        logging: Callable = print,
        external_evaluator: Callable = None,
        test_cases: List[Any] = None,
    ):
        """
        Initialize the Godel Agent.

        Args:
            repo_path: Path to the repository for version control.
            llm_client: OpenAI-compatible LLM client.
            goal: The evolution goal.
            config: Agent configuration.
            logging: Logging function.
            external_evaluator: External evaluation function; signature (agent_instance, test_cases) -> (reward, metrics).
            test_cases: List of test cases.
        """
        self.repo_path = Path(repo_path).resolve()
        self.llm_client = llm_client
        self.goal = goal
        self.config = config or GodelAgentConfig()
        self.logging = logging
        self.test_cases = test_cases or []

        # Thread-local overrides for parallel evaluation
        self._thread_local = threading.local()

        # LLM config (model/temperature/thinking/reasoning_effort)
        self._llm_model = self.config.model
        self._llm_temperature = self.config.temperature
        self._harness_temperature = (
            self.config.harness_temperature
            if self.config.harness_temperature is not None
            else self.config.temperature
        )
        self._thinking_enabled = self.config.thinking_enabled
        self._harness_thinking_enabled = (
            self.config.harness_thinking_enabled
            if self.config.harness_thinking_enabled is not None
            else self.config.thinking_enabled
        )
        self._reasoning_effort = self.config.reasoning_effort
        self._harness_max_tokens = self.config.harness_max_tokens

        # External evaluator (created and passed in by the caller)
        self.external_evaluator = external_evaluator

        # Initialize git controller
        self.git_controller = GitController(self.repo_path)

        # Initialize agent code directory:
        # - Copy initial code from godel_harness_init_path to repo_path
        # - This ensures each run has isolated code and git history
        self._init_agent_code_dir(self.config.godel_harness_init_path)

        # Initialize components
        self.evolution_tracker = EvolutionTracker(self.config.output_dir)

        # Usage logger — records token usage for each LLM call (including cache
        # hits), so API usage can be reconstructed after the fact. An atexit hook
        # writes the final summary at process exit; evolve() also writes a
        # mid-run snapshot when it finishes.
        self._usage_logger = UsageLogger(
            log_path=os.path.join(self.config.output_dir, "usage_log.jsonl"),
            summary_path=os.path.join(self.config.output_dir, "usage_summary.json"),
            log=self._log,
        )

        # Knowledge graph — None when the flag is off; does not affect iteration behavior
        self._knowledge_graph = None
        if getattr(self.config, "meta_evolve_kg_enabled", False):
            try:
                from .knowledge_graph import EvolutionKnowledgeGraph
                self._knowledge_graph = EvolutionKnowledgeGraph(
                    agent_code_dir=self.agent_code_dir,
                    git_controller=self.git_controller,
                    call_llm=self.call_llm,
                    max_nodes=getattr(self.config, "meta_evolve_kg_max_nodes", 100),
                    concurrency=getattr(self.config, "meta_evolve_kg_concurrency", 2),
                    log=self._log,
                )
                self._knowledge_graph.load()
                # First-time enable: backfill historical nodes from existing
                # tracker records (semantic edges left empty)
                if not self._knowledge_graph.nodes and self.evolution_tracker.records:
                    self._knowledge_graph.backfill_from_tracker(self.evolution_tracker)
            except Exception as e:
                self._log(f"Warning: knowledge graph init failed: {e}")
                self._knowledge_graph = None

        # External tools (scanned each iteration)
        self.evolve_tools = []  # Tools specific to the evolve stage
        self._harness_tools = []   # Tools specific to the harness stage (backing store; accessed via property)
        self.external_tools = []  # All tools (used by action_executor)
        self._injected_harness_functions = {}  # Functions injected into the harness scope
        self.openai_tools = build_openai_tools(self.evolve_tools, self.config.enable_bash, exclude_tools=self.config.evolve_exclude_tools)

        # State
        self.state: Optional[AgentState] = None
        self.iteration = 0  # Number of completed iterations; the next one is iteration + 1
        # Bootstrap: the measured "experience" text from the init harness (produced by run_init_eval).
        # Injected into the first prompt only at iteration 1; not re-run on resume, so may be None.
        self._init_experience: Optional[str] = None
        self._last_meta_evolve_iteration = 0  # Iteration value at the last meta-evolve
        self.phase = EvolutionPhase.INITIALIZING
        self._actions_in_iteration = 0  # Tool-call counter
        self._step_in_iteration = 0     # LLM-call counter (one LLM call may include multiple tool calls)
        self._eval_count_in_iteration = 0  # Number of evaluate calls (per iteration)
        # Cached best-commit resolution (commit_hash, full_reward). Set once by
        # _resolve_best_commit() at the end of evolve(); None = not yet resolved.
        # _generate_evolution_summary() reads this passively so the summary and
        # the exported commit always agree (see R1).
        self._best_resolved: Optional[tuple] = None

        # Code mode (folder mode for multi-file support)
        self._code_mode = "folder"

        # Iteration end tracking
        self._last_iteration_end_reason: str = ""  # "compact_context" | "max_steps" | "end_evolution" | ""
        self._iteration_prompt_tokens: int = 0     # prompt_tokens of the most recent LLM call

        # Context management - message history
        self.message_history = MessageHistory()
        self.context_persistence: Optional[ContextPersistence] = None

        # Initialize git if needed
        if not self.git_controller.is_git_repo():
            self.git_controller.init_repo()

        # Create initial commit so get_current_commit() never returns None
        if self.git_controller.get_current_commit() is None:
            self.git_controller.create_evolution_commit(
                iteration=0,
                message="Initial commit: agent code + evolution strategy",
                files=None,
            )

        # Initialize context persistence FIRST (needed by action_executor)
        self._init_context_persistence()

        # Initialize action executor (after context_persistence is ready)
        self.action_executor = AgentActionExecutor(
            llm_client=llm_client,
            model=self.config.model,
            repo_path=self.repo_path,
            agent_code_dir=self.agent_code_dir,  # Use repo_path as working directory
            tools=self.external_tools,  # Pass in the external tools list
            logging=logging,
            external_evaluator=self.external_evaluator,
            agent_instance=self,
            context_persistence=self.context_persistence,  # Pass in the context-persistence instance
        )

        # Initialize iteration helper
        self.iter_helper = EvolveHelper(self)

        # Initialize meta-evolve helper and archive manager
        from .meta_evolve import MetaEvolveHelper
        from .archive_manager import ArchiveManager
        self._meta_evolve_helper = MetaEvolveHelper(self)
        self.archive_manager = ArchiveManager(self)

        # Record the initial code as a seed node in the tracker so select_seed
        # has at least one candidate from the start — no more "0 candidates"
        # fallback on the first iteration. Skip on resume (tracker already has
        # records from the previous run).
        if not self.config.resume_from:
            init_head = self.git_controller.get_current_commit() or ""
            if init_head:
                self.evolution_tracker.record_iteration(
                    iteration=0,
                    parent_commit="",
                    new_commit=init_head,
                    reward=0.0,
                    state_summary="Initial seed — agent code before any evolution",
                    action_count=0,
                    metadata={},
                )
                self._log(f"  Recorded initial seed: {init_head[:7]}")

                # Add iter-0 to knowledge graph so backbone edges have a root.
                kg = getattr(self, "_knowledge_graph", None)
                if kg is not None:
                    try:
                        kg.add_node(
                            iteration=0,
                            git_hash=init_head,
                            parent_hash="",
                            reward=0.0,
                            eval_mode="dev",
                            summary_text="Initial seed — agent code before any evolution",
                            modified_files=[],
                            change_tags=["init"],
                            is_meta=False,
                            # skip_save=False: persist immediately so the
                            # file exists before the first commit_iteration.
                        )
                    except Exception as e:
                        self._log(f"Warning: iter-0 KG add_node failed: {e}")

        # Interaction log transport: per-thread via property below

        # Resume: restore runtime state from previous run
        if self.config.resume_from:
            self._restore_resume_state()

    # =====================================================================
    # Thread-local overrides (for parallel evaluation)
    # =====================================================================

    def set_thread_override(self, key, value):
        if not hasattr(self._thread_local, 'overrides'):
            self._thread_local.overrides = {}
        self._thread_local.overrides[key] = value

    def get_thread_override(self, key):
        if not hasattr(self._thread_local, 'overrides'):
            return _NO_OVERRIDE
        return self._thread_local.overrides.get(key, _NO_OVERRIDE)

    def clear_thread_overrides(self):
        if hasattr(self._thread_local, 'overrides'):
            self._thread_local.overrides.clear()

    @property
    def harness_tools(self):
        override = self.get_thread_override('harness_tools')
        return override if override is not _NO_OVERRIDE else self._harness_tools

    @harness_tools.setter
    def harness_tools(self, value):
        self._harness_tools = value

    def _get_injected_functions(self):
        override = self.get_thread_override('_injected_harness_functions')
        return override if override is not _NO_OVERRIDE else self._injected_harness_functions

    # -- Interaction log (thread-local) --

    @property
    def _current_interaction_log(self):
        return getattr(self._thread_local, 'interaction_log', None)

    @_current_interaction_log.setter
    def _current_interaction_log(self, value):
        self._thread_local.interaction_log = value

    # =====================================================================
    # Core capability interface — agent.react as the core capability
    # =====================================================================

    def react(
        self,
        messages: List[Dict],
        tools: List[Dict] = None,
        tool_executor: Callable[[str, Dict], str] = None,
        hook_on_request: Callable[[List[Dict], List[Dict]], Tuple[List[Dict], List[Dict]]] = None,
        hook_on_response: Callable[[Any], Any] = None,
        hook_on_complete: Callable[[Any, List[Dict], List[str]], None] = None,
    ) -> Tuple[Any, List[Dict], List[str]]:
        """
        Core capability: a single React (one LLM call + tool execution).

        This is the core-capability interface of the Godel Agent. The caller is
        responsible for looping and message-history management.

        Args:
            messages: Message history.
            tools: List of available tools (OpenAI format); None means no tools.
            tool_executor: Tool executor (tool_name, args) -> result.
            hook_on_request: Pre-LLM-request hook.
                Signature: (messages, tools) -> (messages, tools).
                Returns the (possibly modified) messages and tools.
            hook_on_response: Hook fired after the LLM response and before
                tool execution.
                Signature: (response) -> response.
                Returns the (possibly modified) response.
            hook_on_complete: Hook fired after tool execution.
                Signature: (response, tool_calls_made, tool_results) -> None.
                Used for any post-tool-completion logic.

        Returns:
            (response, tool_calls_made, tool_results)
            - response: the LLM response object.
            - tool_calls_made: list of tools called this turn, format [{"id": str, "name": str, "args": dict}].
            - tool_results: list of tool-execution results.
        """
        # 1. hook_on_request: before the LLM request
        if hook_on_request:
            messages, tools = hook_on_request(messages, tools)

        # 2. Call the LLM
        response = self.call_llm(messages, tools)
        message = response.choices[0].message

        # 3. No tool calls → return immediately
        _tool_calls = getattr(message, 'tool_calls', None)
        if not _tool_calls:
            return (response, [], [])

        # 4. hook_on_response: after the LLM response, before tool execution
        if hook_on_response:
            response = hook_on_response(response)
            message = response.choices[0].message

        # 5. Execute all tool calls
        tool_calls_made, tool_results = self._execute_tool_calls(
            getattr(message, 'tool_calls', []), tool_executor
        )

        # 6. hook_on_complete: after tool execution
        if hook_on_complete:
            hook_on_complete(response, tool_calls_made, tool_results)

        return (response, tool_calls_made, tool_results)

    @staticmethod
    def _execute_tool_calls(tool_calls, tool_executor) -> tuple:
        """Execute a group of tool calls."""
        from .utils.json_parser import fix_and_parse_json
        tool_calls_made = []
        tool_results = []
        for tc in tool_calls:
            if tc.function.arguments:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = fix_and_parse_json(tc.function.arguments, tc.function.name)
            else:
                args = {}
            result = tool_executor(tc.function.name, args) if tool_executor else ""
            tool_calls_made.append({"id": tc.id, "name": tc.function.name, "args": args})
            tool_results.append(result)
        return tool_calls_made, tool_results

    def call_llm(self, messages: List[Dict], tools: List[Dict] = None) -> Any:
        """Capability: call the LLM (dispatcher — supports thread-local override).

        Every LLM call (evolve / harness / meta) goes through this entry point,
        so token usage is logged here centrally.
        """
        override = self.get_thread_override('call_llm')
        if override is not _NO_OVERRIDE:
            response = override(messages, tools)
        else:
            response = self._call_llm_impl(messages, tools)
        self._log_llm_usage(response)
        return response

    def _in_harness(self) -> bool:
        """Whether the current thread is in the harness/evaluate stage (single source, reused widely)."""
        return bool(getattr(self._thread_local, 'harness_mode', False))

    def _log_llm_usage(self, response: Any) -> None:
        """Extract usage from the LLM response and persist it (thread-safe). Failures do not affect the main flow.

        Token usage is read here centrally (_call_llm_impl no longer parses it
        itself), and _iteration_prompt_tokens is refreshed for the evolve
        context-bar.
        """
        try:
            usage = extract_usage(response)
            if usage is None:
                return
            in_harness = self._in_harness()
            thinking = self._harness_thinking_enabled if in_harness else self._thinking_enabled
            self._iteration_prompt_tokens = usage.get("prompt_tokens") or 0
            entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "scope": self._current_llm_scope(),
                "iteration": self.iteration,
                "model": self._llm_model,
                "thinking_enabled": bool(thinking),
                **usage,
            }
            self._usage_logger.record(entry)
        except Exception as e:
            self._log(f"[usage] log failed: {e}")

    def _current_llm_scope(self) -> str:
        """Infer the stage this LLM call belongs to: harness > meta > evolve."""
        if self._in_harness():
            return "harness"
        if self.phase in (EvolutionPhase.META_EVOLVING, EvolutionPhase.META_COMMITTING):
            return "meta"
        return "evolve"

    def print_usage_report(self) -> None:
        """Refresh and print the LLM usage report for this run (including test-evaluation usage)."""
        try:
            summary = self._usage_logger.summarize()
            if not (summary.get("total") or {}).get("calls"):
                print("\n(LLM usage: no records)")
                return
            print("\n" + format_usage_report(summary))
            print(f"\nDetails: {self._usage_logger.log_path}")
            print(f"Summary: {self._usage_logger.summary_path}")
        except Exception as e:
            self._log(f"Warning: usage report failed: {e}")

    def _call_llm_impl(self, messages: List[Dict], tools: List[Dict] = None) -> Any:
        """
        Capability: call the LLM (implementation).

        Args:
            messages: Message history.
            tools: List of available tools (OpenAI format).

        Returns:
            The LLM response object.
        """
        # Sanitize messages for API compliance:
        # - non-thinking mode: strip ALL reasoning_content (not a valid API field)
        # - thinking mode: keep reasoning_content only on assistant messages that
        #   have tool_calls (API requires it for tool-call turns); strip from
        #   non-tool-call assistant messages (API ignores it between user messages)
        in_harness = self._in_harness()
        thinking_enabled = (
            self._harness_thinking_enabled if in_harness
            else self._thinking_enabled
        )
        if thinking_enabled:
            sanitized_messages = []
            for msg in messages:
                if msg.get("role") == "assistant" and not msg.get("tool_calls"):
                    m = {k: v for k, v in msg.items() if k != "reasoning_content"}
                    sanitized_messages.append(m)
                else:
                    sanitized_messages.append(msg)
        else:
            sanitized_messages = []
            for msg in messages:
                m = {k: v for k, v in msg.items() if k != "reasoning_content"}
                sanitized_messages.append(m)

        # Normalize and sanitize tools before sending to API:
        # 1. Strip non-JSON-serializable values (e.g., callable)
        # 2. Convert harness-format tools to OpenAI format
        if tools:
            normalized = []
            for tool in tools:
                cleaned = {k: v for k, v in tool.items() if not callable(v)}
                if "info" in cleaned and "type" not in cleaned:
                    normalized.append(build_external_tool_schema(cleaned))
                else:
                    normalized.append(cleaned)
            tools = normalized

        kwargs = {
            "model": self._llm_model,
            "messages": sanitized_messages,
            "tools": tools,
            "tool_choice": "auto" if tools else None,
        }
        # Harness mode with no tools: apply max_tokens to prevent runaway
        # token generation (e.g., model outputting 8192 tokens for a single
        # game action like "go north").
        if in_harness and not tools and self._harness_max_tokens is not None:
            kwargs["max_tokens"] = self._harness_max_tokens
        if thinking_enabled:
            kwargs["reasoning_effort"] = self._reasoning_effort
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            kwargs["temperature"] = (
                self._harness_temperature if in_harness
                else self._llm_temperature
            )
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        # Token usage is extracted centrally in _log_llm_usage at the dispatcher
        # layer (robust); no duplicate parsing here.
        return self.llm_client.chat.completions.create(**kwargs)

    def meta_evolve(self) -> str:
        """Run a meta-evolution phase after an iteration.

        Delegates to MetaEvolveHelper, which uses react() internally.

        Returns:
            "continue" or "end_evolution"
        """
        return self._meta_evolve_helper.run()

    def evolve(self) -> Dict[str, Any]:
        """
        Main evolution loop.

        Returns:
            Summary of evolution results.
        """
        self._log(f"Starting evolution with goal: {self.goal}")
        self._log(f"Max iterations: {self.config.max_iterations}")

        # Early exit if all iterations already completed (resume with no work left)
        if self.iteration >= self.config.max_iterations:
            self._log("Run already completed — all iterations done. Nothing to resume.")
            summary = self._generate_evolution_summary()
            return {
                "iterations_completed": self.iteration,
                "best_version": self.evolution_tracker.get_best_version(),
                "summary": summary,
                "exported_path": None,
                "visualization_path": None,
            }

        # Bootstrap: on a fresh run, evaluate the init harness once on the dev set
        # before iteration 1. This (a) records a real dev reward on the iteration-0
        # seed (replacing placeholder 0.0) and (b) produces an experience text that
        # drives iteration 1's first self-rewrite from the harness's lived feedback.
        # Resume skips it: the seed reward already rides the tracker
        # (evolution_metadata.json), and the experience text is in-memory only.
        self._init_experience = None
        if (not self.config.resume_from
                and getattr(self.config, 'init_eval_enabled', True)
                and self.external_evaluator is not None):
            try:
                self._init_experience = self.iter_helper.run_init_eval()
            except Exception as e:
                self._log(f"Warning: init eval failed: {e}")

        while self.iteration < self.config.max_iterations:
            self.iteration += 1  # Increment first; iteration now denotes the iteration currently being executed

            # Iteration banner
            self._log(self._format_banner(
                f"  ITERATION {self.iteration}/{self.config.max_iterations}  ", "BCY"
            ))

            # --- Fold in staged meta-evolve changes before seed selection ---
            # meta_evolve._commit() leaves them staged via reset --soft;
            # committing them now protects them from seed_selection's
            # _restore_head() reset --hard.  The commit will be amended into
            # the INIT commit by create_init_commit() below.
            self._fold_in_staged_meta_changes()

            # --- Archive: Select seed version ---
            strategy_hint = ""
            hypothesis = ""
            hypotheses = []
            seed_info = None
            self._current_seed_info = None
            if self.config.archive_enabled:
                try:
                    seed_info = self.archive_manager.select_seed()
                    if seed_info:
                        target_hash = seed_info.get("git_hash", "")
                        strategy_hint = seed_info.get("strategy_hint", "")
                        hypothesis = seed_info.get("hypothesis", "")
                        hypotheses = seed_info.get("hypotheses", [])
                        self._current_seed_info = seed_info
                except Exception as e:
                    self._log(f"Warning: archive select_seed failed: {e}")

            # --- Create INIT commit (folds meta changes + version switch) ---
            # Replaces the old apply_version_switch() call — creates a single
            # tracked INIT commit that carries operation_type="init" and the
            # seed hypothesis + optional seed_eval_reward.
            self.archive_manager.create_init_commit(seed_info)

            try:
                result = self.iter_helper.run_iteration(
                    strategy_hint=strategy_hint,
                    hypothesis=hypothesis,
                    hypotheses=hypotheses,
                )
            except Exception as e:
                import traceback as _tb
                self._log(f"Error in iteration {self.iteration}: {e}\n{_tb.format_exc()}")
                self.phase = EvolutionPhase.ERROR
                self._save_resume_state()
                continue

            if result == "end_evolution":
                self._log("Agent requested end of evolution.")
                self._save_resume_state()
                break

            # Meta-evolve phase: runs on interval, but NOT after the final
            # iteration — meta-evolve only improves the archive strategy
            # (select_seed/select_commit + base prompt) that the NEXT
            # iteration consumes. With no next iteration, it burns tokens
            # and its output is never applied.
            should_meta_evolve = (
                self.config.meta_evolve_enabled
                and self.iteration % self.config.iter_per_metaevolve == 0
                and self.iteration < self.config.max_iterations
            )
            if should_meta_evolve:
                try:
                    meta_result = self.meta_evolve()
                    self._last_meta_evolve_iteration = self.iteration
                    self._save_resume_state()
                    if meta_result == "end_evolution":
                        self._log("Agent requested end of evolution during meta-evolve.")
                        break
                except Exception as e:
                    self._log(f"Error in meta-evolve after iteration {self.iteration}: {e}")
                    # Don't break — meta-evolve failure shouldn't stop evolution
            elif self.config.meta_evolve_enabled:
                if self.iteration >= self.config.max_iterations:
                    self._log(f"  Skipping meta-evolve (iter {self.iteration} is the final iteration — nothing downstream would use it)")
                else:
                    next_meta = ((self.iteration // self.config.iter_per_metaevolve) + 1) * self.config.iter_per_metaevolve
                    self._log(f"  Skipping meta-evolve (iter {self.iteration}, next at iter {next_meta})")

            # Save resume state after each completed iteration (if no meta-evolve ran)
            if not should_meta_evolve:
                self._save_resume_state()

        # Run the agentic select_best stage (cached): loads the FIXED, non-evolvable
        # strategy from the init template and runs a react loop to pick the final
        # best version (with optional ensemble fusion). Falls back to
        # get_best_version("highest_reward") on any miss. Runs BEFORE the summary so
        # the summary reflects the chosen commit and its banner sits under the last
        # iteration.
        self._resolve_best_commit()

        # Generate summary
        summary = self._generate_evolution_summary()
        W = 60
        self._log(f"\n{_C.B}{_C.BGR}{'═' * W}{_C.RST}")
        self._log(f"{_C.B}{_C.BGR}  EVOLUTION COMPLETE{_C.RST}")
        self._log(f"{_C.B}{_C.BGR}{'═' * W}{_C.RST}")
        self._log(summary)

        # Export best version to agent_code_best folder
        exported_path = self.export_best_version()

        # Generate evolution graph visualization
        vis_path = None
        try:
            from .visualization import generate_evolution_html
            html = generate_evolution_html(
                tracker=self.evolution_tracker,
                git_controller=self.git_controller,
                goal=self.goal,
            )
            vis_path = os.path.join(self.config.output_dir, "evolution_graph.html")
            with open(vis_path, "w", encoding="utf-8") as f:
                f.write(html)
            self._log(f"Evolution graph: {vis_path}")
        except Exception as e:
            self._log(f"Warning: visualization generation failed: {e}")

        # Write a mid-run usage snapshot (atexit will overwrite once more at
        # process exit, including test-evaluation usage).
        try:
            self._usage_logger.summarize()
        except Exception as e:
            self._log(f"Warning: usage summary failed: {e}")

        # Use _resolve_best_commit() (the agentic select_best decision) so the
        # summary's "best version" agrees with the exported commit (R1).
        resolved = self._best_resolved
        if resolved is None:
            resolved = self.evolution_tracker.get_best_version("highest_reward")
        return {
            "iterations_completed": self.iteration,
            "best_version": resolved,
            "summary": summary,
            "exported_path": exported_path,
            "visualization_path": vis_path,
        }

    def get_tools(self, scope: str = "harness", injected_tools: List[Dict] = None) -> List[Dict]:
        """
        Get the tool list (OpenAI format), filtered by scope.

        Args:
            scope: "evolve" | "meta_evolve" | "harness" | "all"
                - "evolve": evolution tools + bash + evolve_tools
                - "meta_evolve": read_file/edit_file/write_file/bash/end_meta_evolution
                - "harness": bash + harness_tools (default)
                - "all": all tools
            injected_tools: Tool list passed in from harness.py (harness format).
                Format: [{"info": {"name", "description", "input_schema"}, "function": <callable>}, ...]
                Appended directly to self.harness_tools (no conversion needed).

        Returns:
            Tool list (OpenAI format).
        """
        if scope == "meta_evolve":
            return build_openai_tools(
                external_tools=[],
                enable_bash=self.config.meta_evolve_enable_bash,
                scope="meta_evolve",
                description_overrides={
                    "edit_file": "Edit a file by replacing specific text. In meta-evolve, ONLY files in evolution/ directory or at the repo root are allowed.",
                    "write_file": "Create or overwrite a file. In meta-evolve, ONLY files in evolution/ directory or at the repo root are allowed.",
                    "lesson": "Audit and revise the evolve agent's cross-iteration lesson in BOOTSTRAP.md. "
                              "Compare the lesson against the hypothesis (from plan.md) and the evidence "
                              "in the conversation logs. If the conclusion is unsupported or the confidence "
                              "is miscalibrated, call this tool with a corrected lesson and honest confidence. "
                              "Pass iteration=N to directly overwrite a past iteration's lesson line. "
                              "Only use this to fix wrong lessons — don't rewrite accurate ones.",
                },
            )

        if scope == "atom":
            return build_openai_tools(
                external_tools=[],
                enable_bash=True,
                scope="atom",
            )

        if scope == "pick_seed":
            # Unified react scope for seed selection: file operations + bash.
            # Strategy-tool schemas and the pick_seed schema are appended by the
            # caller (run_seed_selection).
            return build_openai_tools(
                external_tools=[],
                enable_bash=self.config.enable_bash,
                scope="pick_seed",
            )

        if scope == "submit_best":
            # Unified react scope for final best-version selection: read-only +
            # bash + evaluate. The submit_best_pick schema and strategy-tool
            # schemas are appended by the caller (run_submit_best). Fusion is
            # performed by an atom-scope sub-agent inside the ensemble strategy
            # tool, not by hand-edits from the submit_best agent.
            return build_openai_tools(
                external_tools=[],
                enable_bash=self.config.enable_bash,
                scope="submit_best",
            )

        if scope == "probe":
            # Read-only investigation sub-agent: read_file + bash + end_probe.
            return build_openai_tools(
                external_tools=[],
                enable_bash=self.config.enable_bash,
                scope="probe",
            )

        if scope == "evolve":
            # inject pick_commit_version + finalize_commit_pool as extra_tools so they
            # survive the scope whitelist filter. This keeps tools static across the
            # entire iteration (main loop + commit nudge) → zero cache misses from
            # tool-set changes.
            return build_openai_tools(
                self.evolve_tools, self.config.enable_bash,
                exclude_tools=self.config.evolve_exclude_tools,
                extra_tools=[PICK_COMMIT_VERSION_SCHEMA, FINALIZE_COMMIT_POOL_SCHEMA],
            )

        # harness or all scope: uniformly use self.harness_tools

        # injected_tools supports two formats:
        #   harness format: {"info": {"name": ..., "input_schema": ...}, "function": <callable>}
        #   OpenAI format:  {"type": "function", "function": {"name": ..., "parameters": ...}, "callable": <callable>}
        if scope in ["harness", "all"] and injected_tools:
            existing_names = {t.get("info", {}).get("name", t.get("name", "")) for t in self.harness_tools}
            for tool in injected_tools:
                # Extract name from either format
                info = tool.get("info", {})
                func_spec = tool.get("function", {})
                name = info.get("name", "") or (func_spec.get("name", "") if isinstance(func_spec, dict) else "")
                # Extract callable
                func = tool.get("function")
                if not callable(func):
                    func = tool.get("callable")
                if name:
                    # Always update injected functions (even if tool already exists
                    # from _scan_external_tools) so execute_tool() can find them
                    if func and callable(func):
                        self._get_injected_functions()[name] = func
                    if name not in existing_names:
                        self.harness_tools.append(tool)
                        existing_names.add(name)

        tools = []
        existing_names = set()

        # bash / powershell tool (harness scope uses harness_enable_bash)
        if self.config.harness_enable_bash:
            tools.append(get_shell_tool_schema(is_windows=(sys.platform == 'win32')))

        # Read uniformly from self.harness_tools and convert to OpenAI format
        if scope in ["harness", "all"] and self.harness_tools:
            for tool in self.harness_tools:
                tool_info = tool.get("info", {})
                func_spec = tool.get("function", {})
                # harness format: info.name + info.input_schema
                # OpenAI format:  function.name + function.parameters
                if isinstance(func_spec, dict) and func_spec.get("name"):
                    # Already OpenAI format — use directly (drop non-standard fields like callable)
                    name = func_spec["name"]
                    if name and name not in existing_names:
                        tools.append({
                            "type": "function",
                            "function": {
                                "name": name,
                                "description": func_spec.get("description", ""),
                                "parameters": func_spec.get("parameters", {"type": "object", "properties": {}}),
                            }
                        })
                        existing_names.add(name)
                else:
                    # harness format
                    name = tool_info.get("name", tool.get("name", ""))
                    if name and name not in existing_names:
                        tools.append({
                            "type": "function",
                            "function": {
                                "name": name,
                                "description": tool_info.get("description", ""),
                                "parameters": tool_info.get("input_schema", {"type": "object", "properties": {}}),
                            }
                        })
                        existing_names.add(name)

        # If scope is "all", also add the evolve tools
        if scope == "all":
            for tool in self.evolve_tools:
                tool_info = tool.get("info", {})
                name = tool_info.get("name", tool.get("name", ""))
                if name and name not in existing_names:
                    tools.append(build_external_tool_schema(tool))
                    existing_names.add(name)

        return tools

    def execute_tool(
        self,
        tool_name: str,
        args: Dict,
        scope: str = "harness",
    ) -> str:
        """Unified tool executor (dispatcher — supports thread-local override)."""
        override = self.get_thread_override('execute_tool')
        if override is not _NO_OVERRIDE:
            return override(tool_name, args, scope=scope)
        return self._execute_tool_impl(tool_name, args, scope=scope)

    def _execute_tool_impl(
        self,
        tool_name: str,
        args: Dict,
        scope: str = "harness",
    ) -> str:
        """
        Unified tool executor (implementation); supports evolve, harness, and meta_evolve scopes.

        Args:
            tool_name: Tool name.
            args: Tool arguments.
            scope: "evolve" | "harness" | "meta_evolve".

        Returns:
            Tool-execution result.
        """
        # Benchmark shield: version-picking scopes (meta_evolve / atom) must
        # NOT read benchmark source/answers — that would be cheating.
        # Encoded as a scope set so every shielded scope is covered by one check;
        # evolve (solving) is intentionally NOT shielded.
        if scope in _BENCHMARK_SHIELDED_SCOPES:
            _blocked = benchmark_block_message(tool_name, args)
            if _blocked:
                return _blocked

        # 0. Harness scope: injected functions take priority (e.g. Docker-aware bash)
        injected = self._get_injected_functions()
        if scope == "harness" and tool_name in injected:
            func = injected[tool_name]
            try:
                return func(**args)
            except Exception as e:
                return f"Tool error: {e}"

        # 0.5. Harness scope: harness_tools fallback (before the hard-coded bash check).
        # This lets Docker-aware bash/editor-style tools take priority over the
        # built-in host bash.
        if scope == "harness":
            for tool in self.harness_tools:
                tool_info = tool.get("info", {})
                name = tool_info.get("name", tool.get("name", ""))
                if name == tool_name:
                    func = tool.get("function")
                    if func and callable(func):
                        try:
                            return func(**args)
                        except Exception as e:
                            return f"Tool error: {e}"

        # 1. Shell tool (bash / powershell; the scope picks which enable flag applies).
        # atom is an evolve-style editing scope (used by the ensemble strategy to
        # modify the harness); get_tools(scope="atom") hard-codes enable_bash=True
        # to provide bash. The same gate must let it through here, otherwise bash
        # would be advertised but rejected at the execution layer (older code
        # routed atom to harness_enable_bash).
        if tool_name in ("bash", "powershell"):
            bash_enabled = (
                self.config.enable_bash
                if scope in ("evolve", "atom", "pick_seed", "submit_best", "probe")
                else self.config.meta_evolve_enable_bash
                if scope == "meta_evolve"
                else self.config.harness_enable_bash
            )
            if not bash_enabled:
                return f"Tool '{tool_name}' is not available in {scope} scope."
            return self.action_executor.execute_shell(
                command=args.get("command"),
                timeout=args.get("timeout"),
                scope=scope
            )

        # 2. Evolution tools (available in evolve scope; end_evolution also
        # allowed in meta_evolve scope). Tools restricted to evolve scope only
        # (general-purpose file operations are not in this group).
        evolution_only_tools = ["evaluate", "compact_context", "plan", "lesson"]
        # evolve-only history tools
        evolve_history_tools = ["read_history_self", "get_historic_version", "get_historic_eval_code"]

        if tool_name in evolution_only_tools:
            # lesson is also available in meta_evolve scope for auditing
            if tool_name == "lesson" and scope == "meta_evolve":
                return self.action_executor.execute(tool_name, args)
            # pick_seed scope also allows evaluate (the agent self-validates seed
            # hypotheses during seed selection); submit_best scope also allows
            # evaluate (post-fusion reward must be measured). compact_context /
            # plan / lesson are allowed here, but the pick_seed/submit_best
            # whitelists don't expose them.
            if scope not in ("evolve", "pick_seed", "submit_best"):
                return f"Tool '{tool_name}' is only available in evolve scope."
            if tool_name in self.config.evolve_exclude_tools:
                return f"Tool '{tool_name}' is disabled in current configuration."
            return self.action_executor.execute(tool_name, args)

        if tool_name in evolve_history_tools:
            if scope != "evolve":
                return f"Tool '{tool_name}' is only available in evolve scope."
            if tool_name in self.config.evolve_exclude_tools:
                return f"Tool '{tool_name}' is disabled in current configuration."
            return self.action_executor.execute(tool_name, args)

        # end_evolution: available in both evolve and meta_evolve scopes
        if tool_name == "end_evolution":
            if scope not in ("evolve", "meta_evolve"):
                return f"Tool '{tool_name}' is only available in evolve/meta_evolve scope."
            return self.action_executor.execute(tool_name, args)

        # meta_evolve-only tools
        if tool_name in ("end_meta_evolution", "validate_archive", "meta_bootstrap"):
            if scope != "meta_evolve":
                return f"Tool '{tool_name}' is only available in meta_evolve scope."
            return self.action_executor.execute(tool_name, args)

        # 3. File-operation tools (available in evolve, meta_evolve, atom,
        # pick_seed, submit_best scopes). Note: these are built-in tools; in
        # harness scope, callers should fall through to the external-tool branch
        # to use the benchmark-provided tools of the same name.
        file_tools = ["read_file", "edit_file", "write_file"]
        if tool_name in file_tools and scope in ("evolve", "meta_evolve", "atom", "pick_seed", "submit_best", "probe"):
            path = args.get("path", "")
            abs_repo = os.path.realpath(self.agent_code_dir)
            # Relative paths are resolved against agent_code_dir (consistent
            # with agent_file_ops._resolve_path()); otherwise os.path.realpath
            # would resolve against CWD and the whitelist could be tripped.
            rp = os.path.realpath(path) if os.path.isabs(path) else os.path.realpath(os.path.join(abs_repo, path))

            # evolve: path sandbox check
            if scope == "evolve":
                abs_evo_dir = os.path.realpath(os.path.join(abs_repo, "evolution"))
                # evolution/ directory is always forbidden (for all file_tools)
                if rp == abs_evo_dir or rp.startswith(abs_evo_dir + os.sep):
                    return f"Error: Cannot access evolution/ directory in evolve scope. Focus on your harness code."
                # read_file whitelist: only repo/, eval_logs/, src/react_loop/
                if tool_name == "read_file":
                    abs_react_loop = os.path.realpath(os.path.dirname(__file__))
                    abs_eval_logs = os.path.realpath(os.path.join(abs_repo, '..', 'eval_logs'))
                    in_repo = rp == abs_repo or rp.startswith(abs_repo + os.sep)
                    in_logs = rp == abs_eval_logs or rp.startswith(abs_eval_logs + os.sep)
                    in_fw = rp == abs_react_loop or rp.startswith(abs_react_loop + os.sep)
                    if not (in_repo or in_logs or in_fw):
                        return (f"Error: read_file in evolve scope is restricted to "
                                f"repo/, eval_logs/, and src/react_loop/. "
                                f"'{path}' is outside.")
                self.action_executor._restricted_dirs = {"evolution"}
            else:
                self.action_executor._restricted_dirs = set()

            # meta_evolve: path sandbox check (allows modifying files under
            # evolution/ and at the repo root).
            if scope == "meta_evolve" and tool_name in ("edit_file", "write_file"):
                abs_evo_dir = os.path.realpath(os.path.join(abs_repo, "evolution"))
                in_evolution = rp == abs_evo_dir or rp.startswith(abs_evo_dir + os.sep)
                in_harness = rp == abs_repo or os.path.dirname(rp) == abs_repo
                if not (in_evolution or in_harness):
                    return (f"Error: meta-evolve can only modify files under evolution/ "
                            f"or at the repo root. Path '{path}' is not allowed.")
            # meta_evolve: read_file sandbox — only allows reading evolution/
            # (the meta-evolvable tree), src/react_loop/ (the framework mechanisms
            # that select_seed/select_commit plug into), and the harness code at
            # the repo root (harness.py/prompts.py/hooks.py/..., top-level files
            # only — non-recursive, so the large logs under .evolution_context/
            # still require bash+jq). src/benchmark/ is already intercepted by
            # the benchmark shield (cheating). Other read-only exploration
            # (git show / git diff / jq) goes through bash.
            if scope == "meta_evolve" and tool_name == "read_file":
                abs_react_loop = os.path.realpath(os.path.dirname(__file__))
                abs_evo_dir = os.path.realpath(os.path.join(abs_repo, "evolution"))
                in_evolution = rp == abs_evo_dir or rp.startswith(abs_evo_dir + os.sep)
                in_react_loop = rp == abs_react_loop or rp.startswith(abs_react_loop + os.sep)
                in_harness = rp == abs_repo or os.path.dirname(rp) == abs_repo
                if not (in_evolution or in_react_loop or in_harness):
                    return (f"Error: meta-evolve read_file is sandboxed to evolution/, "
                            f"src/react_loop/, and the harness files at the repo root. "
                            f"'{path}' is outside all. Use bash (git show / cat / jq) "
                            f"for other read-only exploration.")
            # atom / pick_seed: no path sandbox; can edit agent harness files

            # probe: read-only defense in depth — edit_file/write_file are not
            # in the probe whitelist, but block them here as a safety net.
            if scope == "probe" and tool_name in ("edit_file", "write_file"):
                return f"Tool '{tool_name}' is not available in probe scope (read-only)."

            return self.action_executor.execute(tool_name, args)

        # end_probe: only available in probe scope (defense in depth — the
        # probe sub-agent's tool_executor intercepts this first, so this guard
        # only fires if someone calls end_probe outside a probe context).
        if tool_name == "end_probe":
            if scope != "probe":
                return f"Tool '{tool_name}' is only available in probe scope."
            return self.action_executor.execute(tool_name, args)

        # probe: spawn a read-only investigation sub-agent.
        # Available for any scope whose tool list includes "probe" (evolve,
        # meta_evolve). meta_evolve has its own interceptor in meta_evolve.py
        # that fires first; this is the fallback for evolve and any future scopes.
        if tool_name == "probe":
            from .probe_agent import run_probe
            instructions = args.get("instructions", "")
            self._log(f"  Spawning probe sub-agent...")
            # Inject the benchmark's log-structure schema so the probe sub-agent
            # reads the right fields for THIS benchmark (episode diaries vs
            # tool-call logs vs shell commands) instead of a hardcoded shape.
            # META scope ignores it (its prompt probes .evolution_context/, not eval_logs).
            schema = ""
            try:
                handler = self.action_executor._get_log_summary_handler()
                if handler:
                    schema = handler.build_log_schema_description() or ""
            except Exception:
                pass
            return run_probe(self, instructions, scope=scope, log_schema=schema)

        # 4. External tools (fallback to evolve_tools when not in harness scope)
        if scope != "harness":
            for tool in self.evolve_tools:
                tool_info = tool.get("info", {})
                name = tool_info.get("name", tool.get("name", ""))
                if name == tool_name:
                    func = tool.get("function")
                    if func:
                        try:
                            return func(**args)
                        except Exception as e:
                            return f"Tool error: {e}"

        return f"Unknown tool: {tool_name}"

    # =====================================================================
    # Internal methods
    # =====================================================================

    # =====================================================================
    # Resume state persistence
    # =====================================================================

    def _save_resume_state(self) -> None:
        """Write resume_state.json to output_dir after each iteration."""
        state = {
            "version": 1,
            "completed_iterations": self.iteration,
            "max_iterations": self.config.max_iterations,
            "last_meta_evolve_iteration": self._last_meta_evolve_iteration,
            "last_iteration_end_reason": self._last_iteration_end_reason,
            "last_updated": datetime.now().isoformat(),
        }
        path = os.path.join(self.config.output_dir, "resume_state.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self._log(f"Warning: failed to save resume state: {e}")

    @staticmethod
    def load_resume_state(run_dir: str) -> dict:
        """Load resume state from run_dir/resume_state.json.

        Falls back to derivation from evolution_metadata.json if
        resume_state.json doesn't exist (backward compat).
        """
        resume_path = os.path.join(run_dir, "resume_state.json")
        if os.path.exists(resume_path):
            with open(resume_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return GodelAgent._derive_resume_state(run_dir)

    @staticmethod
    def _derive_resume_state(run_dir: str) -> dict:
        """Derive resume state from evolution_metadata.json records."""
        from .git_version.controller import EvolutionTracker
        tracker = EvolutionTracker(run_dir)

        if not tracker.records:
            return {
                "version": 1,
                "completed_iterations": 0,
                "max_iterations": 0,
                "last_meta_evolve_iteration": 0,
                "last_iteration_end_reason": "",
            }

        completed = 0
        last_meta = 0
        last_reason = ""

        for rec in tracker.records:
            if rec.is_main_iteration:
                completed = max(completed, rec.iteration)
                last_reason = rec.metadata.get("iteration_end_reason", "")
            elif rec.metadata.get("type") == "meta_evolve":
                last_meta = max(last_meta, rec.metadata.get("main_iteration", 0))

        return {
            "version": 1,
            "completed_iterations": completed,
            "max_iterations": completed,
            "last_meta_evolve_iteration": last_meta,
            "last_iteration_end_reason": last_reason,
        }

    def _restore_resume_state(self) -> None:
        """Restore runtime state from resume_from after full initialization."""
        rs = self.config.resume_from
        completed = rs.get("completed_iterations", 0)

        # Restore iteration counter
        self.iteration = completed
        self._last_meta_evolve_iteration = rs.get("last_meta_evolve_iteration", 0)
        self._last_iteration_end_reason = rs.get("last_iteration_end_reason", "")

        # Main-line records (excludes meta records).
        # Used both for the count check and the HEAD check below.
        main_records = [r for r in self.evolution_tracker.records if r.is_main_iteration]
        if len(main_records) != completed:
            self._log(
                f"Warning: tracker has {len(main_records)} non-meta records "
                f"but resume claims {completed} completed iterations"
            )

        # Validate git HEAD matches last MAIN record's commit
        if main_records:
            last_main = main_records[-1]
            current_head = self.git_controller.get_current_commit() or ""
            # HEAD may point to a meta-evolve commit after the last MAIN commit — that's OK
            # Only warn if HEAD doesn't match ANY recent record (MAIN or META)
            if current_head != last_main.primary_commit():
                # Check if HEAD matches the META record right after last MAIN (if any)
                all_commits = {entry["new_commit"]
                               for r in self.evolution_tracker.records
                               for entry in r.iter_pool()}
                if current_head not in all_commits:
                    self._log(
                        f"Warning: git HEAD ({current_head[:7]}) doesn't match "
                        f"any tracked record. The run may have crashed mid-commit."
                    )

        # Resume banner
        self._log(self._format_banner(
            f"  RESUMING from iteration {completed}  ", "BYE"
        ))
        self._log(f"  Completed: {completed}/{self.config.max_iterations} iterations")
        self._log(f"  Last meta-evolve: iteration {self._last_meta_evolve_iteration}")
        self._log(f"  Last end reason: {self._last_iteration_end_reason}")

    def _fold_in_staged_meta_changes(self):
        """Commit staged meta-evolve changes into the main line.

        meta_evolve._commit() does reset --soft <main_baseline>, which keeps
        meta changes staged but unstaged commits. If seed_selection runs a
        react loop, its _restore_head does reset --hard which would wipe them.
        Committing them first makes them durable.
        """
        try:
            result = self.git_controller._run_git_command(
                ["diff", "--cached", "--quiet"], check=False
            )
            if result.returncode != 0:
                self.git_controller._run_git_command(
                    ["commit", "-m",
                     f"[Meta fold-in] Fold meta-evolve changes into main line "
                     f"before iteration {self.iteration + 1}"],
                    check=False
                )
                new_head = self.git_controller.get_current_commit() or ""
                self._log(f"  Folded staged meta changes into main line: "
                          f"{new_head[:7] if new_head else '?'}")
        except Exception as e:
            self._log(f"  Warning: fold-in of staged meta changes failed: {e}")

    def _init_agent_code_dir(self, source_path: str) -> None:
        """
        Initialize agent code directory by copying initial code to repo_path.

        This ensures each run has:
        - Isolated code directory (in run_xxx/repo/)
        - Independent git history
        - No interference with other runs

        Args:
            source_path: Path to the initial agent code (godel_harness_init/agentdojo)
        """
        import shutil

        source_dir = Path(source_path).resolve()
        target_dir = self.repo_path

        # Check if repo already has code (resuming from existing run)
        if target_dir.exists() and self.git_controller.is_git_repo():
            # Check if there are any tracked files
            tracked = self.git_controller.get_tracked_files_at_commit("HEAD", "")
            if tracked:
                self._log(f"Resuming from existing repo with {len(tracked)} tracked files")
                self.agent_code_dir = str(target_dir)
                return

        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy initial code to repo directory
        # Exclude: .git, .evolution_context (start fresh for each run)
        exclude_dirs = {".git", ".evolution_context"}

        if source_dir.exists() and source_dir != target_dir:
            self._log(f"Copying initial code from {source_dir} to {target_dir}")

            # Copy all files except excluded directories
            for item in source_dir.iterdir():
                if item.name in exclude_dirs:
                    continue
                target_item = target_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, target_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, target_item)

            self._log(f"Initial code copied successfully")
        elif source_dir == target_dir:
            self._log(f"Source and target are the same: {target_dir}")

        # Set agent_code_dir to repo_path (where code will be modified)
        self.agent_code_dir = str(target_dir)

        # Copy evolution init if configured
        self._init_evolution_dir(self.config.godel_evolution_init_path)

    def _init_evolution_dir(self, source_path: str) -> None:
        """
        Initialize the evolution/ subdirectory by copying from godel_evolution_init_path.

        This provides the initial evolution strategy that the agent can modify
        during meta-evolve phases.

        Args:
            source_path: Path to the initial evolution strategy directory
        """
        import shutil

        if not source_path:
            self._log("No evolution init path configured, skipping evolution dir setup")
            return

        source_dir = Path(source_path).resolve()
        target_dir = Path(self.agent_code_dir) / "evolution"

        # If evolution dir already exists (resuming), skip
        if target_dir.exists() and any(target_dir.iterdir()):
            self._log(f"Evolution dir already exists, skipping copy: {target_dir}")
            return

        if not source_dir.exists():
            self._log(f"Warning: Evolution init path does not exist: {source_dir}")
            return

        target_dir.mkdir(parents=True, exist_ok=True)

        exclude_dirs = {".git", "__pycache__"}
        self._log(f"Copying evolution init from {source_dir} to {target_dir}")

        # Dimensions whose flag is off are simply NOT copied — loading the
        # missing module later returns None and the delegator falls back to
        # the configured strategy (seed) / {} (commit/best). This replaces the
        # old in-file marker-block stripping on a single archive.py.
        skip_files = set()
        if not self.config.evolvable_commit_strategy:
            skip_files.add("select_commit.py")
        # select_best is a FIXED, non-evolvable module: it loads from the init
        # template (godel_evolution_init/select_best.py), NOT evolution/, so
        # meta-evolve (sandboxed to evolution/) physically cannot edit it. Never
        # copied to evolution/.
        skip_files.add("select_best.py")
        if skip_files:
            self._log(f"Not copying disabled dimension module(s): {sorted(skip_files)}")

        for item in source_dir.iterdir():
            if item.name in exclude_dirs or item.name in skip_files:
                continue
            target_item = target_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target_item, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target_item)

        self._log(f"Evolution init copied successfully ({len(list(target_dir.iterdir()))} files)")

    def _init_context_persistence(self) -> None:
        """Initialize the context-persistence manager."""
        if not self.agent_code_dir:
            self._log("Warning: No agent_code_dir, context persistence disabled")
            return

        self.context_persistence = ContextPersistence(self.agent_code_dir)
        self._log(f"Context persistence initialized at {self.context_persistence.context_dir}")

    def _scan_external_tools(self, scope: str = None) -> None:
        """
        Scan the agent code directory for external tools.

        Args:
            scope: None = all, "evolve" = evolve only, "harness" = harness only.
        """
        if scope is None or scope == "evolve":
            self.evolve_tools = scan_external_tools(self.agent_code_dir, "evolve", self._log)
        if scope is None or scope == "harness":
            self.harness_tools = scan_external_tools(self.agent_code_dir, "harness", self._log)
        self.external_tools = self.evolve_tools + self.harness_tools
        self.openai_tools = build_openai_tools(self.evolve_tools, self.config.enable_bash, exclude_tools=self.config.evolve_exclude_tools)
        self.action_executor.tools = self.external_tools

    # =====================================================================
    # React-based evolution helpers
    # =====================================================================

    def _generate_evolution_summary(self) -> str:
        """Generate a summary of the evolution process.

        PASSIVE w.r.t. select_best: reads the cached ``_best_resolved`` when the
        caller (evolve() tail) already ran ``_resolve_best_commit()``; otherwise
        falls back to get_best_version("highest_reward"). Must NOT itself trigger
        select_best — the early-exit path (all iterations already done) calls this
        directly and must not spawn an agentic pick (R1).
        """
        # Resolve the best commit + its full reward. When _best_resolved is set
        # the full reward is already cached (no get_full_reward re-scan);
        # otherwise derive it from the fallback highest-reward version.
        best_commit = None
        best_full_reward = None
        if self._best_resolved is not None:
            best_commit, best_full_reward = self._best_resolved
        else:
            best = self.evolution_tracker.get_best_version("highest_reward")
            if best:
                best_commit, best_scalar = best
                best_full_reward = self.evolution_tracker.get_full_reward(
                    best_commit, best_scalar
                )

        lines = [
            f"Goal: {self.goal}",
            f"Iterations completed: {self.iteration}",
            "",
            "Evolution History:",
        ]

        for record in self.evolution_tracker.records:
            is_meta = record.metadata.get("type") == "meta_evolve"
            primary_commit_str = record.primary_commit()[:7] if record.primary_commit() else "?"

            if is_meta:
                main_iter = record.metadata.get("main_iteration", "?")
                lines.append(
                    f"  [meta-evolve after iter {main_iter}]: commit={primary_commit_str}"
                )
                meta_summary = record.metadata.get("summary_text", "")
                if meta_summary:
                    lines.append(f"    [summary]: {meta_summary}")
            else:
                # Regular evolution iteration
                reward_history = record.metadata.get("reward_history", [])

                # Build seed source line
                seed_info = record.metadata.get("seed_info")
                seed_tag = ""
                if seed_info:
                    hint = seed_info.get("strategy_hint", "")
                    if hint:
                        seed_tag = f" seed={hint}"

                if len(reward_history) > 1:
                    history_strs = []
                    for entry in reward_history:
                        if isinstance(entry, dict) and "reward" in entry and "eval_mode" in entry:
                            mode_tag = entry["eval_mode"]
                            history_strs.append(f"[{mode_tag}]{fmt_reward(entry['reward'])}")
                        else:
                            history_strs.append(fmt_reward(entry))
                    history_str = " -> ".join(history_strs)
                    lines.append(
                        f"  iter={record.iteration}: rewards=[{history_str}] "
                        f"actions={record.action_count}{seed_tag} commit={primary_commit_str}"
                    )
                else:
                    lines.append(
                        f"  iter={record.iteration}: reward={fmt_reward(record.primary_reward())} "
                        f"actions={record.action_count}{seed_tag} commit={primary_commit_str}"
                    )

                # Show iteration summary if available
                iter_summary = record.metadata.get("summary_text", "")
                if iter_summary:
                    lines.append(f"    [summary]: {iter_summary}")

        if best_commit:
            lines.append(
                f"\nBest version: {best_commit[:7]} with reward {fmt_reward(best_full_reward)}"
            )

        return "\n".join(lines)

    @staticmethod
    def _format_banner(title_text: str, color: str) -> str:
        """Format a boxed banner line. color is a _C attribute name like 'BCY'."""
        W = 70
        pad = max(0, W - len(title_text))
        left = pad // 2
        right = pad - left
        c = getattr(_C, color)
        return (
            f"\n{_C.B}{c}{'╔' + '═' * W + '╗'}{_C.RST}\n"
            f"{c}║{' ' * left}{_C.RST}{_C.B}{_C.BWH}{title_text}{_C.RST}{c}{' ' * right}║{_C.RST}\n"
            f"{_C.B}{c}{'╚' + '═' * W + '╝'}{_C.RST}"
        )

    def _log(self, message: str) -> None:
        """Log a message."""
        if self.config.verbose:
            self.logging(message)

    def set_harness_interaction_log(self, log: list) -> None:
        """Store interaction log from harness for evaluator retrieval.

        Harness calls this to pass its conversation history back to the
        evaluator through the agent instance.
        """
        self._current_interaction_log = log

    def clear_harness_interaction_log(self) -> None:
        """Clear interaction log before a new task evaluation."""
        self._current_interaction_log = None

    def export_best_version(self, output_name: str = "agent_code_best") -> Optional[str]:
        """
        Export the best version to a separate folder.

        Which version is "best" is decided by ``_resolve_best_commit()``: the
        agentic select_best stage picks a commit when ``config.submit_best_enabled``
        is on (falling back to the highest-reward version on any miss/exception).

        Args:
            output_name: Name of the output folder (default: "agent_code_best")
                        If output_dir contains a timestamp (run_YYYYMMDD_HHMMSS),
                        the timestamp will be appended to the output name.

        Returns:
            Path to the exported folder, or None if export failed.
        """
        resolved = self._resolve_best_commit()
        if not resolved:
            self._log("No best version found to export")
            return None

        best_commit, best_full_reward = resolved
        return self._export_commit(best_commit, best_full_reward, output_name)

    def _resolve_best_commit(self) -> Optional[Tuple[str, Any]]:
        """Resolve which commit to export as the final best.

        Runs the agentic select_best stage when ``config.submit_best_enabled``
        is on: archive_manager.select_best() loads the FIXED, non-evolvable
        strategy from the init template (NOT evolution/, so meta-evolve cannot
        edit it) and its framework runner executes a react loop to pick a
        commit (with optional ensemble fusion). Falls back to
        get_best_version("highest_reward") when disabled, the module returns
        no commit_hash, or anything raises.

        Result is cached in ``self._best_resolved`` so the summary and the
        exported commit always agree (R1). Returns None when no best version
        exists.
        """
        if self._best_resolved is not None:
            return self._best_resolved

        best_commit = None
        best_scalar = None

        if getattr(self.config, "submit_best_enabled", True):
            try:
                result = self.archive_manager.select_best()
                if result and result.get("commit_hash"):
                    best_commit = result["commit_hash"]
                    meta = result.get("metadata") or {}
                    best_scalar = meta.get("reward")
            except Exception as e:
                self._log(f"Warning: select_best failed: {e}")

        if not best_commit:
            best = self.evolution_tracker.get_best_version("highest_reward")
            if not best:
                return None
            best_commit, best_scalar = best

        best_full_reward = self.evolution_tracker.get_full_reward(
            best_commit, best_scalar
        )
        self._best_resolved = (best_commit, best_full_reward)
        return self._best_resolved

    def _export_commit(
        self,
        best_commit: str,
        best_full_reward: Any,
        output_name: str = "agent_code_best",
    ) -> Optional[str]:
        """Export a specific commit's code to a separate folder.

        This is the original export logic (path resolution, file materialization,
        .evolution copy), factored out so ``export_best_version`` can pass in the
        chosen commit.
        """
        import shutil

        # Determine output path: use output_dir as target directory
        # output_dir is like "./evolution_results/agentdojo/run_20260322_153045"
        # We want to export to "./evolution_results/agentdojo/run_20260322_153045/agent_code_best_20260322_153045"
        output_dir = self.config.output_dir

        # Extract timestamp from output_dir if present (run_YYYYMMDD_HHMMSS)
        timestamp_match = re.search(r'run_(\d{8}_\d{6})', output_dir)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            full_output_name = f"{output_name}_{timestamp}"
        else:
            full_output_name = output_name

        output_path = os.path.join(output_dir, full_output_name)

        reward_str = fmt_reward(best_full_reward)
        self._log(f"\nExporting best version (commit {best_commit[:7]}, reward: {reward_str})...")
        self._log(f"Target: {output_path}")

        # Remove existing output folder if exists
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
            self._log(f"Removed existing folder: {output_path}")

        # Determine directory prefix relative to repo root
        # If agent_code_dir == repo_path, files are at repo root (no prefix)
        # If agent_code_dir is a subdir of repo_path, use relative path as prefix
        if self.agent_code_dir and os.path.normpath(self.agent_code_dir) == os.path.normpath(str(self.repo_path)):
            dir_prefix = ""
        else:
            dir_prefix = str(Path(self.agent_code_dir).name)

        tracked_files = self.git_controller.get_tracked_files_at_commit(
            best_commit,
            dir_prefix,
        )

        if not tracked_files:
            self._log("No tracked files found, copying current agent_code_dir...")
            # Fallback: copy current directory
            shutil.copytree(self.agent_code_dir, output_path)
        else:
            # Create output directory
            os.makedirs(output_path, exist_ok=True)

            # Export each tracked file
            for rel_file in tracked_files:
                try:
                    content = self.git_controller.get_file_content_at_commit(
                        best_commit, rel_file
                    )
                    if content is not None:
                        # Strip the directory prefix to get path relative to agent_code_dir
                        if dir_prefix:
                            rel_to_agent = rel_file.replace(dir_prefix + "/", "", 1)
                        else:
                            rel_to_agent = rel_file

                        if "/" in rel_to_agent:
                            sub_dir = os.path.join(output_path, os.path.dirname(rel_to_agent))
                            os.makedirs(sub_dir, exist_ok=True)
                        target_file = os.path.join(output_path, rel_to_agent)

                        with open(target_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                except Exception as e:
                    self._log(f"Warning: Failed to export {rel_file}: {e}")

        # Copy .evolution context if exists
        evolution_src = os.path.join(self.agent_code_dir, ".evolution")
        if os.path.exists(evolution_src):
            evolution_dst = os.path.join(output_path, ".evolution")
            shutil.copytree(evolution_src, evolution_dst)
            self._log(f"Copied evolution context to {evolution_dst}")

        self._log(f"Successfully exported best version to: {output_path}")
        return output_path
