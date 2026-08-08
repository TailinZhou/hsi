"""Terminal-Bench 2 harness — fixed interface + evolvable HarnessPolicy.

The evaluator calls using_harness(agent, task, terminal_state) once per task.
using_harness() owns the iteration loop; policy.workflow() handles a single step.
Task completes immediately when task_complete tool is called.
"""

import importlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class StepResult:
    content: str
    task_done: bool = False


# ─── FIXED: using_harness ────────────────────────────────────────
def using_harness(agent: Any, task: str, terminal_state: str = "") -> str:
    """Fixed interface — evaluator calls this once per task.

    Input: (agent, task, terminal_state) — never change this signature.
    Output: final answer string — policy.execute() handles the full loop.
    """
    pkg = __package__
    context_mod = importlib.import_module(".context", pkg)
    hooks_mod = importlib.import_module(".hooks", pkg)
    prompts_mod = importlib.import_module(".prompts", pkg)
    tools_mod = importlib.import_module(".tools_harness", pkg)

    policy = HarnessPolicy()
    return policy.execute(agent, task, terminal_state, context_mod, hooks_mod, prompts_mod, tools_mod)


# ─── EVOLVABLE: HarnessPolicy ────────────────────────────────────
MAX_ITERATIONS = 1000

class HarnessPolicy:
    """Evolvable task-solving policy — a strategy, not a pipeline.

    This is your algorithm for solving tasks. The LLM (via react()) is your
    reasoning engine; YOU design the decision flow.

    Everything in this class is yours to reshape. The initial methods are
    scaffolding — NOT a fixed contract. You can delete, rename, add, or
    restructure any method to serve your strategy. The only constraint is
    that the entry method called by using_harness() returns the expected type.
    """

    def __init__(self):
        self.state = {}

    def execute(self, agent, task, terminal_state, context_mod, hooks_mod, prompts_mod, tools_mod) -> str:
        """Run the full iteration loop. Evolve to customize orchestration strategy."""
        context = context_mod.HarnessContext(prompts_mod.SYSTEM_PROMPT)
        context.add_user(self.build_initial_message(task, terminal_state, self.state))

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

    def build_initial_message(self, task, terminal_state, state) -> str:
        """Build user message from task + terminal state + policy state.

        Default: concatenate task with terminal_state if present.
        Evolve to inject planning context, add state from previous
        iterations, or restructure the prompt for better task decomposition.
        """
        user_msg = task
        if terminal_state:
            user_msg += f"\n\nCurrent terminal state:\n{terminal_state}"
        return user_msg

    def get_agent_decision(self, agent, messages, tools, tool_executor, hooks_mod):
        """Single agent.react() call with current messages and tools.

        Returns (response, tool_calls_made, tool_results) — the raw
        API response and parsed tool call data.

        Evolve to add retry logic, multi-call strategies (plan -> execute -> verify),
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
        """Process tool results. task_complete exits immediately.

        Returns True if task_complete was called (loop should exit).
        All other tool results are added to context normally.
        """
        completed = False
        for tc, result in zip(tool_calls_made, tool_results):
            if tc["name"] == "task_complete":
                context.add_tool_result(tc["id"], "Task complete.")
                completed = True
            else:
                context.add_tool_result(tc["id"], str(result))
        return completed

    def update_state(self, state, tool_calls_made, content, iteration):
        """Update policy state after each iteration.

        Tracks iteration count and last tools used. Evolve to add
        failure pattern detection, task progress estimation, or
        early termination heuristics based on state history.
        """
        state['iterations'] = iteration + 1
        state['last_tools'] = [tc['name'] for tc in tool_calls_made] if tool_calls_made else []

    def validate_and_inject_feedback(self, content, tool_calls_made, context):
        """Non-blocking compliance check — inject FORMAT REMINDER if Analysis/Plan missing."""
        if not tool_calls_made or not content:
            return
        missing = []
        if not re.search(r'\*?\*?Analysis\*?\*?\s*:', content, re.IGNORECASE):
            missing.append("**Analysis:** (what the terminal shows, accomplished, remaining)")
        if not re.search(r'\*?\*?Plan\*?\*?\s*:', content, re.IGNORECASE):
            missing.append("**Plan:** (what commands to run next and why)")
        if missing:
            context.add_user(
                "[FORMAT REMINDER] Your previous response was missing required sections:\n"
                + "\n".join(f"- {m}" for m in missing)
                + "\n\nPlease include these in your next response."
            )

    # ── 3-step summarization ─────────────────────────────────────

    def run_summarization(self, agent, task_desc, terminal_now, context, prompts_mod):
        """Attempt 3-step summarization handoff. Returns True if context was replaced.

        Steps:
            1. Full-context LLM generates a summary of the conversation
            2. Fresh-context LLM asks questions about gaps in the summary
            3. Full-context LLM answers those questions

        On success, replaces message history with handoff structure:
            [system, question_prompt, questions, handoff_msg]
        """
        messages = context.message_history

        # Step 1: summarize
        try:
            prompt = prompts_mod.SUMMARY_GENERATION_PROMPT.format(task=task_desc)
            resp = agent.call_llm(messages + [{"role": "user", "content": prompt}], tools=None)
            summary = resp.choices[0].message.content or ""
        except Exception:
            return False
        if not summary:
            return False

        # Step 2: ask questions (fresh context)
        try:
            qp = prompts_mod.QUESTION_ASKING_PROMPT.format(
                task=task_desc, summary=summary, terminal_state=terminal_now)
            resp = agent.call_llm([{"role": "user", "content": qp}], tools=None)
            questions = resp.choices[0].message.content or ""
        except Exception:
            return False
        if not questions:
            return False

        # Step 3: answer questions (full context + summary conversation)
        try:
            summary_prompt = prompts_mod.SUMMARY_GENERATION_PROMPT.format(task=task_desc)
            ap = prompts_mod.ANSWER_PROVIDING_PROMPT.format(questions=questions)
            resp = agent.call_llm(
                messages + [
                    {"role": "user", "content": summary_prompt},
                    {"role": "assistant", "content": summary},
                    {"role": "user", "content": ap},
                ],
                tools=None,
            )
            answers = resp.choices[0].message.content or ""
        except Exception:
            return False
        if not answers:
            return False

        # Handoff: preserve question-answer conversation structure
        question_prompt = prompts_mod.QUESTION_ASKING_PROMPT.format(
            task=task_desc, summary=summary, terminal_state=terminal_now)
        handoff = prompts_mod.HANDOFF_PROMPT.format(answers=answers)
        context.replace_messages([
            context.get_system_message(),
            {"role": "user", "content": question_prompt},
            {"role": "assistant", "content": questions},
            {"role": "user", "content": handoff},
        ])
        self.state['summarization_method'] = '3-step'
        return True

    # ── Single-step orchestration ─────────────────────────────────

    def workflow(self, agent, context, tools, tool_executor, hooks_mod, prompts_mod) -> StepResult:
        """One reasoning phase within your algorithm.

        NOT limited to a single react() call. Evolve to call react() multiple
        times with different prompts (analyze → plan → execute → verify),
        or implement conditional sub-strategies within this phase.

        Returns StepResult(content, task_done).
        """
        iteration = self.state.get('iterations', 0)

        response, tool_calls_made, tool_results = self.get_agent_decision(
            agent, context.get_messages(), tools, tool_executor, hooks_mod
        )

        message = response.choices[0].message
        content = message.content or ""

        # No tool calls -> inject reminder and continue loop
        if content and not tool_calls_made:
            context.add_assistant_text(content)
            context.add_user(
                "You responded without using any tools. Please continue working on the task "
                "by calling the appropriate tools (bash, task_complete, etc.)."
            )
            return StepResult(content=content, task_done=False)

        # Update context with tool interaction
        if tool_calls_made:
            context.add_assistant_with_tools(self.build_assistant_msg(message, tool_calls_made))
            task_done = self.handle_tool_results(
                context, tool_calls_made, tool_results
            )
            if task_done:
                return StepResult(content=content, task_done=True)
            self.validate_and_inject_feedback(content, tool_calls_made, context)

        self.update_state(self.state, tool_calls_made, content, iteration)

        # ── Proactive 3-step summarization ───────────────────────
        if context.needs_summarization() and len(context.message_history) > 6:
            task_desc = ""
            for msg in context.message_history:
                if msg.get("role") == "user":
                    task_desc = str(msg.get("content", ""))
                    break

            terminal_now = ""
            try:
                terminal_now = agent.execute_tool(
                    "bash", {"command": "", "duration": 0.5}, scope="harness")
            except Exception:
                pass

            self.run_summarization(agent, task_desc, terminal_now, context, prompts_mod)

        return StepResult(content=content, task_done=False)
