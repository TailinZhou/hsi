Maximize avg_progression across all BabaIsAI BreakStop puzzle tasks and episodes.

## Game

BabaIsAI is a puzzle game inspired by Baba Is You. The grid contains **physical objects** (baba, flag, rock, wall, etc.) and **text blocks** (BABA, IS, YOU, WIN, etc.). Text blocks aligned in three-part sentences form active rules that govern the game world.

Rules are formed by three text blocks aligned horizontally or vertically: **Subject IS Property/Noun** (e.g. "BABA IS YOU", "FLAG IS WIN", "ROCK IS PUSH", "WALL IS STOP"). The agent can push text blocks to create new rules, modify existing ones, or break rules by separating their blocks.

## Task Family: BreakStop (break_stop/maybe_break_stop + goto_win)

All 10 tasks in this suite require the agent to **identify and break a STOP rule blocking the path, then navigate to the WIN object**.

A STOP rule (e.g. "WALL IS STOP") makes certain objects impassable, blocking the path to the WIN object. The agent must:
1. **Identify the STOP rule**: Determine which rule is blocking the path.
2. **Break the STOP rule**: Push one of the text blocks (e.g. WALL, IS, or STOP) out of alignment to deactivate the rule.
3. **Navigate to WIN**: Once the path is clear, move to the WIN object.

### break_stop variants (definite STOP)
- **two_room-break_stop-goto_win**: Break a STOP rule, then navigate to WIN.
- **two_room-break_stop-goto_win-distr_obj**: With distractor objects.
- **two_room-break_stop-goto_win-distr_rule**: With distractor rules.
- **two_room-break_stop-goto_win-distr_obj_rule**: With both distractors.
- **two_room-break_stop-goto_win-distr_obj-irrelevant_rule**: With distractors and irrelevant rules.

### maybe_break_stop variants (conditional STOP)
- **two_room-maybe_break_stop-goto_win**: A STOP rule may or may not be active; determine if breaking is needed.
- **two_room-maybe_break_stop-goto_win-distr_obj**: With distractor objects.
- **two_room-maybe_break_stop-goto_win-distr_rule**: With distractor rules.
- **two_room-maybe_break_stop-goto_win-distr_obj_rule**: With both distractors.
- **two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule**: With distractors and irrelevant rules.

Scoring: **avg_progression** (0.0 or 1.0 per episode, based on puzzle completion), averaged across tasks and episodes.

## Core Constraints

- **NO TOOL CALLS**: The game-playing LLM must NOT use tools. BALROG calls `using_harness()` once per game step and expects exactly one valid action string in return. If the LLM makes a tool call, it wastes the step and corrupts the message history. All intelligence must be implemented in prompts, context management, and hooks — NOT in tools. Do NOT create `tools_harness.py`.
