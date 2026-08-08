"""Balrog harness — LLM-centric, task-generic.

task_context keys (set by evaluator each step):
    env_name          str   — game environment name (e.g. "crafter", "textworld")
    instruction       str   — environment instruction / action list (empty if none)
    short_term        str   — status info: inventory, stats, etc. (empty if none)
    naive_instruction str   — output constraint for LLM (empty if none)
    is_new_episode    bool  — True on first step of each episode
    last_step_reward  float — reward from previous step (0.0 on first step)
    achievements      dict  — all unlocked achievements {name: count} (empty on first step)
    recent_unlocked   list  — achievements newly unlocked on previous step (empty on first step)
"""

import importlib
from typing import Any, Dict, List, Optional

from react_loop.utils.task_context import get_task_context


# ─── FIXED: using_harness ────────────────────────────────────────
def _build_system_prompt(base_prompt: str, instruction: str) -> str:
    """Build system prompt with instruction appended once per episode.

    Original BALROG only delivers instruction at step 0; we embed it in the
    system prompt so it survives history truncation.
    """
    if instruction:
        return base_prompt + "\n\n" + instruction
    return base_prompt


def _inject_short_term(messages: List[Dict], short_term: str) -> List[Dict]:
    """Inject short_term into the last user message for the current LLM call only.

    Historical user messages remain plain long_term observations — matching
    the original BALROG where only the latest observation carries short_term.
    Returns a shallow-copied list so the original history is not modified.
    """
    if not short_term:
        return messages
    result = list(messages)
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "user":
            result[i] = dict(result[i])
            result[i]["content"] = (
                f"{short_term}\n\n{result[i]['content']}"
            )
            break
    return result


def using_harness(agent: Any, obs: str) -> str:
    """Fixed interface — evaluator calls this once per game step."""
    pkg = __package__
    context_mod = importlib.import_module(".context", pkg)
    hooks_mod = importlib.import_module(".hooks", pkg)
    prompts_mod = importlib.import_module(".prompts", pkg)
    task_ctx = get_task_context()

    system_prompt = _build_system_prompt(
        prompts_mod.SYSTEM_PROMPT, task_ctx.get('instruction', ''))

    # Lifecycle management — fixed, never modify
    if 'harness_context' not in task_ctx:
        task_ctx['harness_context'] = context_mod.HarnessContext(system_prompt)
    ctx = task_ctx['harness_context']
    if task_ctx.get('is_new_episode', False):
        ctx.reset(system_prompt=system_prompt)

    if 'harness_policy' not in task_ctx:
        task_ctx['harness_policy'] = HarnessPolicy()

    action = task_ctx['harness_policy'].workflow(agent, obs, ctx, hooks_mod, prompts_mod)
    return action


# ─── EVOLVABLE: HarnessPolicy ────────────────────────────────────
class HarnessPolicy:
    """LLM-centric policy for Balrog tasks.

    Owns whatever auxiliary state and processing the agent evolves. Each step
    assembles a message from task_context + internal state, calls the LLM,
    and returns its action.

    Step lifecycle:
        _on_new_episode()                          — first step of each episode
        _process_observation(obs, task_ctx)        — before message assembly
        _build_context(state, task_ctx)            — assemble context (non-obs info)
        _build_message(obs, state, task_ctx)       — assemble full LLM input
        get_agent_decision(agent, messages, hooks) — one agent.react() call
        _postprocess_action(action, obs, task_ctx) — validate or correct the LLM output
        _update_state(obs, action, task_ctx)       — after action is chosen

    The workflow() method orchestrates these stages. The default is a single
    react() call per step — if needed, evolve it to use multi-react workflows (e.g.,
    analyze-then-act, plan-then-execute, or act-then-validate).
    """

    def __init__(self):
        self.state: Dict[str, Any] = {}
        self._on_new_episode()

    def _on_new_episode(self) -> None:
        """Called at the start of each new episode."""
        self.state = {}

    def _process_observation(self, obs: str, task_ctx: Dict) -> str:
        """Process the raw observation. Returns the observation for the LLM."""
        return obs

    def _build_context(self, state: Dict, task_ctx: Dict) -> str:
        """Assemble the context section (everything except the observation).

        Note:
        - instruction (action list + goal) is appended to the system prompt
          once per episode — do NOT repeat it here.
        - short_term (inventory/status) is NOT baked into history messages;
          it is injected into the latest user message at LLM call time only.
        """
        parts = []

        naive_instruction = task_ctx.get('naive_instruction', '')
        if naive_instruction:
            parts.append(naive_instruction)

        if state.get('steps', 0) > 0:
            parts.append(f"[Step {state['steps']}, last action: {state.get('last_action', '')}]")

        last_reward = task_ctx.get('last_step_reward', 0.0)
        if state.get('steps', 0) > 0 and last_reward != 0.0:
            parts.append(f"Last step reward: {last_reward}")

        recent = task_ctx.get('recent_unlocked', [])
        if recent:
            parts.append(f"Recently unlocked: {', '.join(recent)}")

        return "\n\n".join(parts)

    def _build_message(self, obs: str, state: Dict, task_ctx: Dict) -> str:
        """Assemble the message sent to the LLM."""
        context = self._build_context(state, task_ctx)
        if context:
            return (
                f"## Context\n{context}\n\n"
                f"## Your Observation\n{obs}"
            )
        return f"## Your Observation\n{obs}"

    def _update_state(self, obs: str, action: str, task_ctx: Dict) -> None:
        """Update internal state after each step."""
        self.state['steps'] = self.state.get('steps', 0) + 1
        self.state['last_action'] = action

    def _postprocess_action(self, action: str, obs: str, task_ctx: Dict) -> str:
        """Validate or correct the LLM's output action. Returns the final action."""
        return action

    def workflow(self, agent, obs, ctx, hooks_mod, prompts_mod) -> str:
        """Orchestrate one step. Default: single react() call.

        Key design: short_term is NOT baked into the user message added to
        history. It is injected into the latest message only at LLM call time,
        so historical observations contain only long_term context.

        Evolve this method to change the per-step workflow — e.g., call
        agent.react() multiple times with different messages, use
        self.state to carry context between calls, or add pre/post
        processing stages.
        """
        task_ctx = get_task_context()

        if task_ctx.get('is_new_episode', False):
            self._on_new_episode()

        obs = self._process_observation(obs, task_ctx)
        # History stores observation WITHOUT short_term
        message = self._build_message(obs, self.state, task_ctx)
        ctx.add_user(message)

        # Inject short_term into latest message for this call only
        short_term = task_ctx.get('short_term', '')
        messages = ctx.get_messages()
        messages = _inject_short_term(messages, short_term)

        action, raw_msg = self.get_agent_decision(agent, messages, hooks_mod)
        action = self._postprocess_action(action, obs, task_ctx)

        ctx.add_assistant(raw_msg)
        self._update_state(obs, action, task_ctx)
        return action

    def get_agent_decision(self, agent, messages, hooks_mod):
        """Single agent.react() call. Returns (action_text, raw_message)."""
        response, _, _ = agent.react(
            messages=messages,
            tools=None,
            hook_on_request=hooks_mod.hook_on_request,
            hook_on_response=hooks_mod.hook_on_response,
            hook_on_complete=None,
        )
        msg = response.choices[0].message
        action = (msg.content or "").strip()
        return action, msg
