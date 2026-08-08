"""Balrog benchmark evaluator.

Runs multi-step episodes (observe -> act -> reward -> repeat) against
6 text-based game environments. Each task is an (env_name, task_name) pair
and runs num_episodes episodes internally, averaging progression scores.
"""

import copy
import logging
import random
import re
import sys
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from benchmark.balrog.config import BalrogConfig
from benchmark.balrog.tasks import get_tasks_for_env
from benchmark.balrog.utils import get_unique_seed, to_jsonable
from benchmark.evaluators.base import BaseTaskEvaluator
from benchmark.evaluators.log_summary_base import LogSummaryBase
from benchmark.evaluators.task_types import BenchmarkTask, TaskCategory, TaskEvaluationResult

logger = logging.getLogger(__name__)

# Only write a per-episode harness "pain-point diary" for episodes that fell
# short of near-perfect — a fully/near-solved episode carries no diagnostic
# value, and skipping it saves an LLM call. This is an episode-level gate; it is
# deliberately NOT the task-level `failed_threshold` ability bar.
_EPISODE_DIARY_SUCCESS_BAR = 0.99

# ── LLM timeout + retry for per-episode diary calls ──────────────────────
# These calls enrich the agent's feedback with first-person episode diaries.
# A hung API must not stall episode evaluation — the diary is auxiliary.
_DIARY_LLM_TIMEOUT = 45.0       # seconds per attempt (shorter: per-episode, many calls)
_DIARY_LLM_MAX_RETRIES = 1      # additional attempts (2 total)
_DIARY_LLM_RETRY_BACKOFF = 1.0  # seconds between retries


# ── Episode early-stop guard: detect action loops ─────────────────────────
# When the LLM gets stuck (e.g. repeatedly emitting an invalid action like
# "idle" that harness fallbacks cycle through, or a fixed oscillation), the
# episode burns the prompt cache for no gain running to max_steps. This pure
# evaluator-side guard breaks such doomed episodes early. Period-1 (straight-
# line navigation) is deliberately NOT flagged, so legitimate moves survive.
_ACTION_LOOP_WINDOW = 12  # must be divisible by 2 and 3


def _is_action_loop(actions: List[str], window: int = _ACTION_LOOP_WINDOW) -> bool:
    """True if the last ``window`` actions form a strict period-2 (ABAB…) or
    period-3 (ABCABC…) loop — a strong stuck signal. Period-1 is excluded so
    that straight-line navigation (e.g. go-to-win moving ``right`` repeatedly
    toward the target) is never killed."""
    if len(actions) < window:
        return False
    recent = actions[-window:]
    if len(set(recent)) < 2:
        return False  # all-identical = period-1 straight line; do not kill navigation
    return any(
        all(recent[i] == recent[i % p] for i in range(window))
        for p in (2, 3)
    )


class BalrogEvaluator(BaseTaskEvaluator):
    """Evaluator for Balrog benchmark — multi-step game episodes."""

    def __init__(self, balrog_config: BalrogConfig = None):
        self._balrog_config = balrog_config or BalrogConfig()
        self._model: Optional[str] = None
        self._llm_client = None
        self._agent_instance = None
        self._verbose = False
        self._eval_mode = "dev"
        # test_repeats pass index (set by adapter.evaluate_test_set). Folded into
        # get_unique_seed's deterministic string so repeat N>0 draws a different
        # map set than repeat 0. Default 0 = unchanged behavior for dev/val and
        # for callers that never set it.
        self._test_repeat_idx: int = 0
        self._cached_config_dict: Optional[dict] = None
        self._harness_package_name: Optional[str] = None
        self._harness_source: Optional[Dict[str, str]] = None

    @property
    def benchmark_name(self) -> str:
        return "balrog"

    def _get_config_dict(self) -> dict:
        """Get hyperagents config dict, cached for the lifetime of the evaluator."""
        if self._cached_config_dict is None:
            self._cached_config_dict = self._balrog_config.to_hyperagents_config()
        return self._cached_config_dict

    def load_tasks(
        self,
        suite: Optional[str] = None,
        categories: Optional[List[TaskCategory]] = None,
    ) -> List[BenchmarkTask]:
        """Load tasks from the Balrog benchmark.

        Args:
            suite: Hyphen-separated env names (e.g., "babyai", "babyai-crafter", "all")
            categories: Ignored — all tasks are UTILITY
        """
        if suite == "all" or suite is None:
            env_names = ["nle", "minihack", "babyai", "crafter", "textworld", "babaisai"]
        else:
            env_names = [s.strip() for s in suite.split("-")]

        tasks = []
        for env_name in env_names:
            for task_name in get_tasks_for_env(env_name):
                tasks.append(BenchmarkTask(
                    task_id=f"{env_name}/{task_name}",
                    instruction=f"Play the {env_name} game task: {task_name}",
                    category=TaskCategory.UTILITY,
                    benchmark_source="balrog",
                    metadata={
                        "env_name": env_name,
                        "task_name": task_name,
                        "num_episodes": self._balrog_config.num_episodes.get(env_name, 5),
                        "episode_workers": self._balrog_config.episode_workers,
                    },
                ))

        logger.info(f"Loaded {len(tasks)} Balrog tasks from environments: {env_names}")
        return tasks

    def evaluate_task(
        self,
        task: BenchmarkTask,
        solver: Callable,
    ) -> TaskEvaluationResult:
        """Evaluate a solver on a single (env_name, task_name) pair.

        Runs multiple episodes and averages the progression score.
        Dispatches to serial or parallel execution based on episode_workers.
        """
        env_name = task.metadata["env_name"]
        task_name = task.metadata["task_name"]
        # dev/val exploration use the cheap num_episodes_dev override; test (and the
        # final post-evolution eval) use the honest num_episodes. metadata holds the
        # honest baseline.
        num_episodes = self._balrog_config.get_num_episodes(
            env_name, self._eval_mode, task.metadata.get("num_episodes", 5),
        )
        episode_workers = task.metadata.get("episode_workers", 1)
        config_dict = self._get_config_dict()

        start_time = time.time()

        if episode_workers <= 1 or num_episodes <= 1:
            episode_progressions, episode_logs, execution_error = (
                self._run_episodes_serial(
                    env_name, task_name, config_dict, solver, num_episodes,
                )
            )
        else:
            episode_progressions, episode_logs, execution_error = (
                self._run_episodes_parallel(
                    env_name, task_name, config_dict, solver,
                    num_episodes, episode_workers,
                )
            )

        elapsed = time.time() - start_time
        avg_progression = sum(episode_progressions) / len(episode_progressions)

        # Per-task Lower-Confidence-Bound (E_LCB): mean(eps) − z·std(eps)/√n.
        # Episode noise (temp-driven 0↔1 flips) is isolated here, per task, so a
        # version that's reliably good at one task isn't penalized for being
        # reliably bad at another. Isolating per task (rather than pooling all
        # episodes) separates episode-noise from task-difficulty spread.
        from react_loop.state import lower_confidence_bound
        lcb_zscore = getattr(self._balrog_config, "lcb_zscore", 1.0)
        lcb_progression = lower_confidence_bound(episode_progressions, z=lcb_zscore)

        threshold = self._balrog_config.task_thresholds.get(
            task.task_id, self._balrog_config.failed_threshold
        )

        # success/threshold keeps the RAW avg_progression: threshold is a raw
        # capability bar, separate from the selection conservatism of the LCB.
        return TaskEvaluationResult(
            task_id=task.task_id,
            success=avg_progression >= threshold,
            output=f"avg_progression={avg_progression:.4f}",
            expected=None,
            execution_time=elapsed,
            metadata={
                "env_name": env_name,
                "task_name": task_name,
                "num_episodes": num_episodes,
                "avg_progression": avg_progression,
                "lcb_progression": lcb_progression,
                "episode_progressions": episode_progressions,
                "total_episodes": num_episodes,
                "successful_episodes": sum(1 for p in episode_progressions if p > 0),
                "execution_error": execution_error,
                "failed_threshold_used": threshold,
            },
            interaction_log=episode_logs,
        )

    def _run_episodes_serial(
        self,
        env_name: str,
        task_name: str,
        config_dict: dict,
        solver: Callable,
        num_episodes: int,
    ) -> tuple:
        """Run episodes sequentially (original behavior)."""
        episode_progressions = []
        episode_logs = []
        execution_error = None

        for ep_idx in range(num_episodes):
            try:
                progression, ep_log = self._run_episode(
                    env_name, task_name, config_dict, solver, ep_idx,
                )
                episode_progressions.append(progression)

                self._append_episode_summary(ep_log, env_name, task_name, progression)

                episode_logs.append(ep_log)
                print(
                    f"      [{env_name}/{task_name}] ep {ep_idx+1}/{num_episodes}: "
                    f"progression={progression:.3f}, steps={ep_log['num_steps']}, "
                    f"invalid_actions={len(ep_log['failed_candidates'])}",
                    flush=True,
                )
            except Exception as e:
                tb_str = traceback.format_exc()
                logger.error(f"Balrog episode error ({env_name}/{task_name} ep={ep_idx}): {tb_str}")
                execution_error = tb_str
                episode_progressions.append(0.0)
                episode_logs.append({"error": tb_str, "episode_idx": ep_idx})

        return episode_progressions, episode_logs, execution_error

    def _run_episodes_parallel(
        self,
        env_name: str,
        task_name: str,
        config_dict: dict,
        solver: Callable,
        num_episodes: int,
        episode_workers: int,
    ) -> tuple:
        """Run episodes in parallel using a thread pool."""
        workers = min(episode_workers, num_episodes)
        results = [None] * num_episodes
        lock = threading.Lock()

        def _thread_fn(ep_idx):
            return self._run_episode_in_thread(
                ep_idx, env_name, task_name, config_dict, solver,
                num_episodes, lock,
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_thread_fn, i): i for i in range(num_episodes)}
            for future in as_completed(futures):
                ep_idx, progression, ep_log, error = future.result()
                with lock:
                    results[ep_idx] = (progression, ep_log, error)

        episode_progressions = []
        episode_logs = []
        execution_error = None

        for ep_idx in range(num_episodes):
            progression, ep_log, error = results[ep_idx]
            if error is not None:
                execution_error = error
            episode_progressions.append(progression)
            episode_logs.append(ep_log)

        return episode_progressions, episode_logs, execution_error

    def _run_episode_in_thread(
        self,
        ep_idx: int,
        env_name: str,
        task_name: str,
        config_dict: dict,
        solver: Callable,
        num_episodes: int,
        print_lock: threading.Lock,
    ) -> tuple:
        """Run a single episode inside a thread. Returns (ep_idx, progression, ep_log, error)."""
        try:
            progression, ep_log = self._run_episode(
                env_name, task_name, config_dict, solver, ep_idx,
            )
            self._append_episode_summary(ep_log, env_name, task_name, progression)
            with print_lock:
                print(
                    f"      [{env_name}/{task_name}] ep {ep_idx+1}/{num_episodes}: "
                    f"progression={progression:.3f}, steps={ep_log['num_steps']}, "
                    f"invalid_actions={len(ep_log['failed_candidates'])}",
                    flush=True,
                )
            return (ep_idx, progression, ep_log, None)
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Balrog episode error ({env_name}/{task_name} ep={ep_idx}): {tb_str}")
            ep_log = {"error": tb_str, "episode_idx": ep_idx}
            with print_lock:
                print(
                    f"      [{env_name}/{task_name}] ep {ep_idx+1}/{num_episodes}: "
                    f"ERROR: {e}",
                    flush=True,
                )
            return (ep_idx, 0.0, ep_log, tb_str)

    def _run_episode(
        self,
        env_name: str,
        task_name: str,
        config_dict: dict,
        solver: Callable,
        episode_idx: int,
    ) -> tuple:
        """Run a single episode and return (progression, log)."""
        from benchmark.balrog.environments import make_env
        import numpy as np
        from react_loop.utils.task_context import get_task_context

        # Seed policy:
        #   - test: balrog_config.seed (deterministic; task_id+episode_idx+repeat_idx
        #     vary). repeat_idx rotates maps across test_repeats passes so each pass
        #     is a different map set, while staying reproducible run-to-run.
        #   - dev/val: None → get_unique_seed uses PID+time (random maps)
        if self._eval_mode == "test":
            base_seed = self._balrog_config.seed
        else:
            base_seed = None
        seed = get_unique_seed(
            episode_idx=episode_idx,
            base_seed=base_seed,
            task_id=f"{env_name}/{task_name}",
            repeat_idx=self._test_repeat_idx,
        )
        _rng = random.Random(seed)
        _np_rng = np.random.default_rng(seed)

        config = copy.deepcopy(config_dict)
        envs_cfg = config.setdefault("envs", {})
        envs_cfg.setdefault("env_kwargs", {})["seed"] = seed
        if env_name == "crafter":
            per_env_key = f"{env_name}_kwargs"
            if per_env_key in envs_cfg:
                envs_cfg[per_env_key]["seed"] = seed

        env = make_env(env_name, task_name, config)

        if env_name == "nle":
            obs, info = env.reset()
        else:
            obs, info = env.reset(seed=seed)

        max_steps = int(env.max_steps)
        max_steps_override = self._balrog_config.get_max_steps(self._eval_mode)
        if max_steps_override is not None:
            max_steps = min(max_steps, int(max_steps_override))

        # ── task_context: all data for harness, no markers/encoding on obs ──
        task_ctx = get_task_context()
        task_ctx['env_name'] = env_name

        instruction = ""
        if env_name == "babyai":
            instruction = env.get_instruction_prompt(
                instructions=obs.get("mission", "")
            )
        else:
            instruction = env.get_instruction_prompt()
        task_ctx['instruction'] = instruction

        # obs stays clean — long_term only, no markers/concatenation
        observation_text = self._extract_obs_long_term(obs)

        episode_return = 0.0
        action_frequency = defaultdict(int)
        num_steps = 0
        failed_candidates = []
        recent_valid_actions: List[str] = []  # for action-loop early-stop guard
        step_traces = []
        prev_step_info = None  # carry reward/achievements from previous step

        for step in range(max_steps):
            task_ctx['is_new_episode'] = (step == 0)

            # short_term goes into task_context, not onto obs
            short_term = self._extract_obs_short_term(obs)
            task_ctx['short_term'] = short_term or ""

            # naive_instruction — applied to ALL environments (matches original BALROG NaiveAgent)
            naive_instruction = (
                "You always have to output one of the above actions at a time "
                "and no other text. You always have to output an action until "
                "the episode terminates."
            )
            task_ctx['naive_instruction'] = naive_instruction

            # Pass previous step's reward & achievements to harness
            if prev_step_info is not None:
                task_ctx['last_step_reward'] = to_jsonable(prev_step_info.get('reward', 0.0))
                achievements = prev_step_info.get('achievements')
                if isinstance(achievements, dict):
                    task_ctx['achievements'] = {
                        k: v for k, v in achievements.items() if v
                    }
                unlocked = prev_step_info.get('unlocked')
                if isinstance(unlocked, (set, list)):
                    task_ctx['recent_unlocked'] = list(unlocked)
            else:
                task_ctx['last_step_reward'] = 0.0
                task_ctx['achievements'] = {}
                task_ctx['recent_unlocked'] = []

            response_str = solver(observation_text)

            # Action cleaning — filter_letters applied to ALL environments (matches original BALROG NaiveAgent)
            response_str = self._clean_action_textworld(response_str)

            action = env.check_action_validity(response_str)
            action_frequency[action] += 1

            # Early-stop guard: break doomed episodes stuck in an action loop
            # (e.g. LLM emitting "idle" → harness fallback cycling), which
            # otherwise burns the prompt cache running to max_steps.
            recent_valid_actions.append(action)
            if _is_action_loop(recent_valid_actions):
                if self._verbose:
                    print(
                        f"        step {step+1}: ⚠️ action loop detected "
                        f"({_ACTION_LOOP_WINDOW}-step period-2/3 repeat: "
                        f"{recent_valid_actions[-_ACTION_LOOP_WINDOW:]}), "
                        f"early-stopping episode to preserve prompt cache",
                        flush=True,
                    )
                num_steps = step + 1
                break

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward

            # Store step info for next iteration's task_context
            prev_step_info = {'reward': reward}
            if isinstance(info, dict):
                if 'achievements' in info:
                    prev_step_info['achievements'] = info['achievements']
                if 'unlocked' in info:
                    prev_step_info['unlocked'] = info['unlocked']

            step_traces.append({
                "step": step + 1,
                "observation": observation_text,
                "agent_response": response_str,
                "valid_action": action,
                "is_valid": response_str == action,
                "reward": to_jsonable(reward),
                "terminated": terminated,
                "truncated": truncated,
            })

            if self._verbose:
                corrected = f" → corrected={repr(action)}" if action != response_str else ""
                obs_preview = observation_text[:2000]
                print(
                    f"        step {step+1}: obs='{obs_preview}'"
                    f"\n          action={repr(response_str)}{corrected}"
                    f", reward={to_jsonable(reward)}",
                    flush=True,
                )

            valid_actions = env.get_action_list_str()

            if action != response_str:
                failed_candidates.append(response_str)
                if self._balrog_config.feedback_on_invalid_action:
                    feedback = (
                        f"\n\nYour action '{response_str}' is not valid. "
                        f"Valid actions: {valid_actions}\n"
                        f"Defaulted to: {action}\n\nObservation:\n"
                    )
                    text = obs["text"]
                    if isinstance(text, str):
                        obs["text"] = feedback + text
                    else:
                        obs["text"]["long_term_context"] = feedback + text.get("long_term_context", "")

            observation_text = self._extract_obs_long_term(obs)
            num_steps = step + 1
            if done:
                break

        stats = to_jsonable(env.get_stats())
        progression = stats.get("progression", 0.0)

        episode_log = {
            "episode_idx": episode_idx,
            "num_steps": num_steps,
            "episode_return": to_jsonable(episode_return),
            "progression": progression,
            "action_frequency": dict(action_frequency),
            "failed_candidates": failed_candidates,
            "seed": seed,
            "done": done,
            "step_traces": step_traces,
        }
        episode_log.update(stats)

        return progression, episode_log

    @staticmethod
    def _clean_action_textworld(response: str) -> str:
        raw = (response or "").strip()
        return re.sub(r"[^a-zA-Z\s:]", "", raw)

    def _build_harness_code_block(self) -> str:
        """Format _harness_source into a text block for the summary prompt."""
        return LogSummaryBase._build_harness_code_block(self._harness_source)

    def _append_episode_summary(
        self, ep_log: dict, env_name: str, task_name: str, progression: float,
    ) -> None:
        """Append per-episode summary via LLM call reusing warm message history.

        Must apply the same sanitization as agent._call_llm_impl() so the
        message prefix is byte-identical to what the API cached during the
        episode — otherwise the prompt cache misses.
        """
        # Near-perfect episodes carry no diagnostic value — skip the LLM call.
        # This gate is episode-level; it is deliberately NOT the task-level
        # `failed_threshold` ability bar (which compares a task's avg ability).
        if progression >= _EPISODE_DIARY_SUCCESS_BAR:
            return

        if self._agent_instance and not getattr(
            self._agent_instance.config, 'evaluate_llm_summary', True
        ):
            return

        style = getattr(
            self._agent_instance.config, 'eval_feedback_style', 'diary'
        ) if self._agent_instance else 'diary'

        if not self._llm_client or not self._model or not self._harness_package_name:
            return
        try:
            from react_loop.utils.task_context import get_task_context
            task_ctx = get_task_context()
            ctx = task_ctx.get('harness_context')
            if not ctx or not ctx.message_history:
                return
            # Strip reasoning_content to match _call_llm_impl sanitization.
            # BALROG never uses tools (tools=None), so all assistant messages
            # lack tool_calls and reasoning_content is always stripped.
            summary_messages = [
                {k: v for k, v in msg.items() if k != "reasoning_content"}
                for msg in ctx.message_history
            ]
            harness_section = LogSummaryBase._build_harness_section(self._harness_source)

            # Per-episode summary/diary describes the lived run — WHAT happened
            # and WHAT the player needed — WITHOUT naming the responsible harness
            # mechanism. Pre-naming a culprit biases the engineer toward fixing
            # whatever the player guessed instead of tracing the phenomenon to
            # the real root cause (a frequent rabbit-hole on noisy benchmarks:
            # the player fingers a real-but-secondary bug, the engineer polishes
            # it for iterations while the actual capability gap stays at 0).
            # Player reports the symptom; engineer (who reads the code) attributes.
            if style == "summary":
                # Third-person objective account of WHAT happened in this episode.
                instruction = (
                    f"The above is the complete episode ({env_name}/{task_name}, "
                    f"progression={progression:.3f}) played using the harness code below.\n"
                    + harness_section +
                    "Account for this episode objectively:\n"
                    "1. What happened? Identify the key failure phenomena: invalid actions, "
                    "missed goals, action loops, wasted steps. Be specific about which steps "
                    "went wrong and what the harness did at each.\n"
                    "2. What capability was missing — what did the harness need (information, "
                    "memory, a way to choose) to handle the situation it fumbled? State as a "
                    "need, not a code location.\n"
                    "3. Is this failure pattern systematic (repeats across this environment "
                    "type) or a one-off?\n\n"
                    "Do NOT suggest code changes. Do NOT pin each failure to a specific "
                    "function or line — describe the phenomenon; the engineer locates the cause "
                    "in the code themselves."
                )
            else:
                # Default "diary" arm: first-person account of the lived run.
                # Goal 1: kill luck-blaming mis-attribution (force the player to
                # describe a concrete stuck-point, not dismiss it as a bad draw).
                # Goal 2: do NOT pre-name the responsible harness mechanism — that
                # biases the engineer into fixing whatever the player guessed
                # instead of tracing the phenomenon to the real root cause. The
                # player reports WHAT happened and WHAT they needed; the engineer
                # (who reads the code) does the attribution.
                instruction = (
                    f"The episode above is YOUR run ({env_name}/{task_name}, "
                    f"progression={progression:.3f}) — you lived it step by step. "
                    f"The harness source code that controlled your run follows.\n"
                    + harness_section +
                    "You are the player. Write a short, blunt first-person diary entry about "
                    "what happened to you.\n\n"
                    "Ground rules:\n"
                    "- Speak as \"I\". You played this; describe it from the inside.\n"
                    "- Describe WHAT YOU EXPERIENCED, not which line of code is at fault. "
                    "You don't see the engineer's code clearly enough to diagnose it — report "
                    "the symptom; the engineer traces it.\n"
                    "- Do NOT blame luck, randomness, the environment, or \"bad draws\". A 0/5 "
                    "is not \"9% bad luck\" — something you needed was missing. Say what.\n"
                    "- Do NOT blame yourself for \"not trying hard enough\". If you couldn't do "
                    "the right thing, the means were missing — say which means.\n"
                    "- Do NOT name functions, quote code, or propose fixes. Just the lived run.\n\n"
                    "In 4-6 sentences cover:\n"
                    "1. What I was trying to do and what actually happened — the specific action "
                    "and the game's response (e.g. \"I went north repeatedly but kept landing "
                    "back in the same room\", \"I had the key but never took the target\").\n"
                    "2. Where I got stuck or sent in circles — the moment the run stopped "
                    "making progress.\n"
                    "3. What I NEEDED but didn't have — a piece of information, a way to do X, "
                    "memory of something I saw earlier. Stated as a need, not a code change.\n"
                    "4. Whether this felt like one stuck point or a repeated pattern in how I "
                    "was driven through the run.\n\n"
                    "Do NOT propose code, diffs, or \"change X to Y\". Do NOT name the function "
                    "or rule you think is broken. Just describe what playing this run was like "
                    "and what you needed — the engineer reads the code and does the diagnosis."
                )

            summary_messages.append({
                "role": "user",
                "content": instruction,
            })
            # Match the agent's thinking/reasoning config so the API
            # processes messages the same way as during the episode.
            thinking_enabled = getattr(
                self._agent_instance, "_harness_thinking_enabled", False
            ) if self._agent_instance else False
            api_kwargs = dict(
                model=self._model,
                messages=summary_messages,
                temperature=0,
                max_tokens=1536,
                timeout=_DIARY_LLM_TIMEOUT,
            )
            if thinking_enabled:
                effort = getattr(
                    getattr(self._agent_instance, "config", None),
                    "reasoning_effort", None,
                ) or "medium"
                api_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
                api_kwargs["reasoning_effort"] = effort
            else:
                api_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            # Retry loop: a hung per-episode diary call must not stall the
            # episode thread. Fall back gracefully (no diary) on exhaustion.
            resp = None
            last_error = None
            for attempt in range(_DIARY_LLM_MAX_RETRIES + 1):
                try:
                    resp = self._llm_client.chat.completions.create(**api_kwargs)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < _DIARY_LLM_MAX_RETRIES:
                        backoff = _DIARY_LLM_RETRY_BACKOFF * (2 ** attempt)
                        logger.warning(
                            "Diary LLM call failed (attempt %d/%d, retrying in %.1fs): %s",
                            attempt + 1, _DIARY_LLM_MAX_RETRIES + 1, backoff, exc,
                        )
                        time.sleep(backoff)
                    else:
                        raise  # re-raise to outer except → graceful fallback

            if resp is None:
                raise RuntimeError("Diary LLM call failed after all retries")

            msg = resp.choices[0].message
            content = msg.content or ""
            # Fallback: some providers put analysis in reasoning_content
            if not content.strip():
                rc = getattr(msg, "reasoning_content", None) or ""
                if rc.strip():
                    content = rc
            ep_log["episode_summary"] = content
        except Exception:
            logger.warning("Per-episode summary failed", exc_info=True)

    def get_log_summary_handler(self):
        """Return the Balrog-specific log summary handler."""
        from benchmark.balrog.log_summary import BalrogLogSummary
        return BalrogLogSummary()

    @staticmethod
    def _extract_obs_text(obs):
        """Extract observation text from env output.

        NLE/MiniHack return a plain string; other envs (babyai, crafter,
        textworld, babaisai) return a dict with long_term_context /
        short_term_context keys.
        """
        text = obs.get("text", "")
        if isinstance(text, str):
            return text
        long_term = text.get("long_term_context", "")
        short_term = text.get("short_term_context", "")
        if short_term:
            return long_term + "\n\n" + short_term
        return long_term

    @staticmethod
    def _extract_obs_long_term(obs):
        """Extract only long_term_context from env output."""
        text = obs.get("text", "")
        if isinstance(text, str):
            return text
        return text.get("long_term_context", "")

    @staticmethod
    def _extract_obs_short_term(obs):
        """Extract only short_term_context from env output."""
        text = obs.get("text", "")
        if isinstance(text, str):
            return ""
        return text.get("short_term_context", "")
