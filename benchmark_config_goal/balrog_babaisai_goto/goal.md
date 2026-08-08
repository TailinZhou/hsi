Maximize avg_progression across all BabaIsAI GoTo puzzle tasks and episodes.

## Game

BabaIsAI is a puzzle game inspired by Baba Is You. The grid contains **physical objects** (baba, flag, rock, wall, etc.) and **text blocks** (BABA, IS, YOU, WIN, etc.). Text blocks aligned in three-part sentences form active rules that govern the game world.

Rules are formed by three text blocks aligned horizontally or vertically: **Subject IS Property/Noun** (e.g. "BABA IS YOU", "FLAG IS WIN", "ROCK IS PUSH", "WALL IS STOP"). The agent can push text blocks to create new rules, modify existing ones, or break rules by separating their blocks.

## Task Family: GoTo (goto_win, with distractors)

All 11 tasks in this suite are GoTo variants. The objective is simple: **navigate to an existing WIN object**.

An active rule already gives some object the WIN property (e.g. "FLAG IS WIN"). The agent controls the YOU object (by default baba) and must navigate to the WIN object. Distractor objects and irrelevant rules may be present but do not need to be modified — the agent only needs to plan a path and execute movement actions.

### Single-room variants
- **goto_win**: Pure navigation to a WIN object in a single room.
- **goto_win-distr_obj**: Navigation with distractor objects present.
- **goto_win-distr_rule**: Navigation with distractor (irrelevant) rules present.
- **goto_win-distr_obj_rule**: Navigation with both distractor objects and rules.
- **goto_win-distr_obj-irrelevant_rule**: Navigation with distractor objects and irrelevant rules.

### Two-room variants
- **two_room-goto_win**: Navigate between two rooms to reach the WIN object.
- **two_room-goto_win-distr_obj**: Two-room navigation with distractor objects.
- **two_room-goto_win-distr_rule**: Two-room navigation with distractor rules.
- **two_room-goto_win-distr_obj_rule**: Two-room navigation with both distractors.
- **two_room-goto_win-distr_obj-irrelevant_rule**: Two-room with distractors and irrelevant rules.
- **two_room-goto_win-distr_win_rule**: Two-room with a misleading second WIN rule as distractor.

Scoring: **avg_progression** (0.0 or 1.0 per episode, based on puzzle completion), averaged across tasks and episodes.

## Core Constraints

- **NO TOOL CALLS**: The game-playing LLM must NOT use tools. BALROG calls `using_harness()` once per game step and expects exactly one valid action string in return. If the LLM makes a tool call, it wastes the step and corrupts the message history. All intelligence must be implemented in prompts, context management, and hooks — NOT in tools. Do NOT create `tools_harness.py`.
