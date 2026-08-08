Maximize avg_progression across all BabaIsAI Make puzzle tasks and episodes.

## Game

BabaIsAI is a puzzle game inspired by Baba Is You. The grid contains **physical objects** (baba, flag, rock, wall, etc.) and **text blocks** (BABA, IS, YOU, WIN, etc.). Text blocks aligned in three-part sentences form active rules that govern the game world.

Rules are formed by three text blocks aligned horizontally or vertically: **Subject IS Property/Noun** (e.g. "BABA IS YOU", "FLAG IS WIN", "ROCK IS PUSH", "WALL IS STOP"). The agent can push text blocks to create new rules, modify existing ones, or break rules by separating their blocks.

## Task Family: Make (make_win, with distractors and break_stop)

All 16 tasks in this suite require the agent to **construct a new WIN rule, then navigate to the newly-WIN object**. No object initially has the WIN property — the agent must create the rule by pushing text blocks into alignment.

The core sequence:
1. **Identify the tools**: Find Subject, IS, and WIN text blocks on the grid.
2. **Push blocks into alignment**: Push text blocks so they form a valid rule (e.g. "FLAG IS WIN").
3. **Navigate to the WIN object**: Once the rule is active, move to the object that now has WIN.

### Single-room variants
- **make_win**: Pure make — push blocks to form a WIN rule, then reach the target.
- **make_win-distr_obj**: With distractor objects present.
- **make_win-distr_rule**: With distractor (irrelevant) rules present.
- **make_win-distr_obj_rule**: With both distractor objects and rules.
- **make_win-distr_obj-irrelevant_rule**: With distractors and irrelevant rules.

### Two-room variants
- **two_room-make_win**: Make a WIN rule across two rooms.
- **two_room-make_win-distr_obj**: With distractor objects.
- **two_room-make_win-distr_rule**: With distractor rules.
- **two_room-make_win-distr_obj_rule**: With both distractors.
- **two_room-make_win-distr_obj-irrelevant_rule**: With distractors and irrelevant rules.
- **two_room-make_win-distr_win_rule**: With a misleading second WIN rule as distractor.

### break_stop-make variants
- **two_room-break_stop-make_win**: Break a STOP rule first, then construct a WIN rule.
- **two_room-break_stop-make_win-distr_obj**: With distractor objects.
- **two_room-break_stop-make_win-distr_rule**: With distractor rules.
- **two_room-break_stop-make_win-distr_obj_rule**: With both distractors.
- **two_room-break_stop-make_win-distr_obj-irrelevant_rule**: With distractors and irrelevant rules.

Scoring: **avg_progression** (0.0 or 1.0 per episode, based on puzzle completion), averaged across tasks and episodes.

## Core Constraints

- **NO TOOL CALLS**: The game-playing LLM must NOT use tools. BALROG calls `using_harness()` once per game step and expects exactly one valid action string in return. If the LLM makes a tool call, it wastes the step and corrupts the message history. All intelligence must be implemented in prompts, context management, and hooks — NOT in tools. Do NOT create `tools_harness.py`.
