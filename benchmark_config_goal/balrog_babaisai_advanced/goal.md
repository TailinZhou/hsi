Maximize avg_progression across all BabaIsAI Advanced puzzle tasks and episodes.

## Game

BabaIsAI is a puzzle game inspired by Baba Is You. The grid contains **physical objects** (baba, flag, rock, wall, etc.) and **text blocks** (BABA, IS, YOU, WIN, etc.). Text blocks aligned in three-part sentences form active rules that govern the game world.

Rules are formed by three text blocks aligned horizontally or vertically: **Subject IS Property/Noun** (e.g. "BABA IS YOU", "FLAG IS WIN", "ROCK IS PUSH", "WALL IS STOP"). The agent can push text blocks to create new rules, modify existing ones, or break rules by separating their blocks.

## Task Family: Advanced (identity manipulation, composite reasoning)

All 3 tasks in this suite require **non-trivial reasoning about object identity and multi-step rule manipulation**. These are the hardest BabaIsAI puzzles.

### make_you variants
- **two_room-make_you**: The current YOU object cannot reach the goal. The agent must push text blocks to form a rule that reassigns YOU to a different object (e.g. "ROCK IS YOU"), then proceed with that new controlled object. Requires understanding that changing identity changes which object can act.
- **two_room-make_you-make_win**: Composite puzzle: first reassign YOU to a different object, then construct a WIN rule for yet another object, then navigate to it. Requires chaining two rule-creation steps in sequence.

### make_wall_win
- **two_room-make_wall_win**: Give WALL the WIN property, then touch any wall. Challenging because walls are immovable (cannot be pushed) — the agent must bring text blocks TO the wall rather than pushing the wall TO text blocks. Requires reasoning about immovable targets.

Scoring: **avg_progression** (0.0 or 1.0 per episode, based on puzzle completion), averaged across tasks and episodes.

## Core Constraints

- **NO TOOL CALLS**: The game-playing LLM must NOT use tools. BALROG calls `using_harness()` once per game step and expects exactly one valid action string in return. If the LLM makes a tool call, it wastes the step and corrupts the message history. All intelligence must be implemented in prompts, context management, and hooks — NOT in tools. Do NOT create `tools_harness.py`.
