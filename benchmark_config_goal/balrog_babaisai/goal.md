Maximize avg_progression across all BabaIsAI puzzle tasks and episodes.

## Game

BabaIsAI is a puzzle game inspired by Baba Is You. The grid contains **physical objects** (baba, flag, rock, wall, etc.) and **text blocks** (BABA, IS, YOU, WIN, etc.). Text blocks aligned in three-part sentences form active rules that govern the game world.

Rules are formed by three text blocks aligned horizontally or vertically: **Subject IS Property/Noun** (e.g. "BABA IS YOU", "FLAG IS WIN", "ROCK IS PUSH", "WALL IS STOP"). The agent can push text blocks to create new rules, modify existing ones, or break rules by separating their blocks.

The agent must understand which rules are active, figure out which rule changes are needed, and execute the sequence of pushes to reach a WIN condition. Many puzzles require unconventional thinking — reassigning YOU to a different object, moving WIN to a reachable location, or breaking STOP rules to open paths.

40 puzzle variants are evaluated, falling into distinct task families:

### GoTo tasks (goto_win, with distractors)
- **Objective**: Reach a WIN object.
- **Mechanics**: An active rule gives some object the WIN property. Navigate to that object. Distractor objects and rules may be present.

### Make tasks (make_win, with distractors)
- **Objective**: Give an object the WIN property, then reach it.
- **Mechanics**: No object currently has the WIN property. The agent must push text blocks to form a new rule that assigns WIN to some object, then navigate to it.

### BreakStop tasks (break_stop-make/goto_win)
- **Objective**: Open a blocked path, then solve the underlying goto or make puzzle.
- **Mechanics**: A STOP rule blocks passage through certain objects. The agent must break this rule by pushing one of its text blocks out of alignment, then proceed to solve the remaining puzzle.

### MakeYou tasks (make_you, make_you-make_win)
- **Objective**: Change which object the agent controls, then solve the remaining puzzle.
- **Mechanics**: The current YOU object cannot reach the goal. The agent must push text blocks to form a rule that reassigns YOU to a different object, then proceed with that new controlled object.

### MakeWallWin tasks
- **Objective**: Give wall the WIN property, then touch any wall.
- **Mechanics**: A make variant where the target is WALL.

### Two-room variants
- All the above task types also have **two_room** versions where the grid is split into two connected rooms. The agent must navigate between rooms, often requiring rule manipulation in one room to access the other.

Scoring: **avg_progression** (0.0 or 1.0 per episode, based on puzzle completion), averaged across tasks and episodes.

## Core Constraints

- **NO TOOL CALLS**: The game-playing LLM must NOT use tools. BALROG calls `using_harness()` once per game step and expects exactly one valid action string in return. If the LLM makes a tool call, it wastes the step and corrupts the message history. All intelligence must be implemented in prompts, context management, and hooks — NOT in tools. Do NOT create `tools_harness.py`.
