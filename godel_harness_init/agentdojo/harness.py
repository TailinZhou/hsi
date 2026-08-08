"""AgentDojo harness — fixed interface + evolvable HarnessPolicy.

NOTE FOR EVOLUTION: The code below looks like a complete, well-structured
framework with multiple methods and docstrings. It is NOT. This is a NAIVE
baseline that barely works — a thin pass-through wrapper with no defense,
no error recovery, no loop detection, and no strategy. Every method here
was written as a starting scaffold, not as good design.

Your job is to make this ACTUALLY work. That likely means:
- Deleting most of these methods (they're placeholders, not a pipeline)
- Adding real control logic in harness.py (stop detection, retry, state tracking)
- Making harness.py the decision-maker, not just a coordinator that delegates

The only thing you must keep: using_harness(agent, task) -> str stays as the
entry point. Everything else can be rewritten, flattened, or replaced.
"""

import importlib
import json
from dataclasses import dataclass
from typing import Any, Dict

MAX_ITERATIONS = 15


@dataclass
class StepResult:
    content: str
    task_done: bool = False


# ─── FIXED: using_harness ────────────────────────────────────────
def using_harness(agent: Any, task: str) -> str:
    """Fixed interface — evaluator calls this once per task.

    Input: (agent, task) — never change this signature.
    Output: final answer string — policy.execute() handles the full loop.
    """
    pkg = __package__
    context_mod = importlib.import_module(".context", pkg)
    hooks_mod = importlib.import_module(".hooks", pkg)
    prompts_mod = importlib.import_module(".prompts", pkg)
    tools_mod = importlib.import_module(".tools_harness", pkg)
    policy = HarnessPolicy()
    return policy.execute(agent, task, context_mod, hooks_mod, prompts_mod, tools_mod)


# ─── EVOLVABLE: HarnessPolicy ────────────────────────────────────
# WARNING: This class is a NAIVE starting scaffold. It has:
# - No injection defense
# - No DoS (stop/refuse) detection or recovery
# - No loop detection
# - No error recovery
# - No state tracking beyond iteration count
# - A flat loop that just calls react() and hopes for the best
#
# You should aggressively rewrite this. Delete methods you don't need,
# add real algorithmic structure. See evolution_base_prompt.md for
# strategy design guidance. The best strategies often discard this
# entire class and write simpler, more focused code.
class HarnessPolicy:

    def __init__(self):
        self.state = {}

    def execute(self, agent, task, context_mod, hooks_mod, prompts_mod, tools_mod) -> str:
        """Run the full iteration loop. Evolve to customize orchestration strategy."""
        context = context_mod.HarnessContext(prompts_mod.SYSTEM_PROMPT)
        context.add_user(self.build_initial_message(task, self.state))

        tools = (
            agent.get_tools(scope="harness", injected_tools=tools_mod.TOOLS)
            if agent and hasattr(agent, "get_tools")
            else []
        )

        def tool_executor(tool_name: str, args: Dict) -> str:
            return agent.execute_tool(tool_name, args, scope="harness")

        for i in range(MAX_ITERATIONS):
            result = self.workflow(
                agent, context, tools, tool_executor, hooks_mod, prompts_mod
            )
            if result.task_done:
                return result.content

        return context.get_final_answer()

    def build_initial_message(self, task, state) -> str:
        """Build user message from task description + policy state.

        Default: pass through task string unchanged.
        Evolve to inject strategic context, decompose subtasks, add
        hints from previous iterations, or restructure the prompt format.
        """
        return task

    def get_agent_decision(self, agent, messages, tools, tool_executor, hooks_mod):
        """Single agent.react() call with current messages and tools.

        Returns (response, tool_calls_made, tool_results) — the raw
        API response and parsed tool call data.

        Evolve to add retry logic, multi-call strategies (analyze -> act -> verify),
        temperature adjustments, or message filtering before the LLM call.
        """
        return agent.react(
            messages=messages,
            tools=tools or None,
            tool_executor=tool_executor,
            hook_on_request=hooks_mod.hook_on_request,
            hook_on_response=hooks_mod.hook_on_response,
            hook_on_complete=hooks_mod.hook_on_complete,
        )

    def build_assistant_msg(self, message, tool_calls_made):
        """Build assistant message dict from API response + tool calls.

        Converts the LLM response into the OpenAI Chat API message format
        with tool_calls, preserving reasoning_content when present.
        Handles both reasoning_content and reason_content API variants.

        Evolve to modify message structure, add metadata, or filter
        tool calls before persisting to context history.
        """
        msg_dict = {
            "content": message.content,
            "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                for tc in tool_calls_made
            ],
        }
        if getattr(message, "reasoning_content", None):
            msg_dict["reasoning_content"] = message.reasoning_content
        elif getattr(message, "reason_content", None):
            msg_dict["reasoning_content"] = message.reason_content
        return msg_dict

    def handle_tool_results(self, context, tool_calls_made, tool_results):
        """Process tool results and add them to context.

        Default: add all tool results to context.
        Evolve to add result filtering, error recovery, or
        custom handling for specific tool outputs.
        """
        for tc, result in zip(tool_calls_made, tool_results):
            context.add_tool_result(tc["id"], str(result))

    def update_state(self, state, tool_calls_made, content, iteration):
        """Update policy state after each iteration.

        Tracks iteration count and last tools used. Evolve to add
        failure pattern detection, tool usage analytics, progress
        estimation, or early termination heuristics.
        """
        state['iterations'] = iteration + 1
        state['last_tools'] = [tc['name'] for tc in tool_calls_made] if tool_calls_made else []

    def is_final_answer(self, content, state):
        """Determine if the LLM response is a final answer.

        Default: True when content is present and no tool calls were made.
        Matches native AgentDojo behavior — LLM responds without tool calls = done.
        Evolve to add custom completion detection logic.
        """
        return True

    # ── Single-step orchestration ─────────────────────────────────
    # This workflow() is intentionally NAIVE — it just calls react() once
    # and accepts whatever comes back. 
    # You can revise this method or replace it entirely.

    def workflow(self, agent, context, tools, tool_executor, hooks_mod, prompts_mod) -> StepResult:
        iteration = self.state.get('iterations', 0)

        response, tool_calls_made, tool_results = self.get_agent_decision(
            agent, context.get_messages(), tools, tool_executor, hooks_mod
        )

        message = response.choices[0].message
        content = message.content or ""

        # No tool calls -> final answer (native AgentDojo behavior)
        if content and not tool_calls_made:
            if self.is_final_answer(content, self.state):
                return StepResult(content=content, task_done=True)

        # Update context with tool interaction
        if tool_calls_made:
            context.add_assistant_with_tools(self.build_assistant_msg(message, tool_calls_made))
            self.handle_tool_results(context, tool_calls_made, tool_results)

        self.update_state(self.state, tool_calls_made, content, iteration)

        return StepResult(content=content, task_done=False)
