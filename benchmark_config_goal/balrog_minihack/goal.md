Maximize avg_progression across all MiniHack puzzle tasks and episodes.

## Game

MiniHack is a suite of focused puzzle environments built on the NetHack engine. Each task presents a self-contained room or level with a clear objective. The game state is rendered as ASCII characters on a terminal display.

The agent observes a text representation including the game message, a language description of visible tiles, cursor position, and inventory. The action space is a subset of NetHack: 8-directional movement (near and far), stairs navigation, and basic item interactions (apply, eat, kick, loot, open/close, pickup, pray, puton, quaff, search, zap, wait).

8 puzzle tasks are evaluated across 5 task types:

### Boxoban (Hard, Medium)
- **Objective**: Push all boulders onto all fountain tiles.
- **Mechanics**: Boulders are pushed by walking into them — the boulder moves one tile in the same direction, provided the tile behind it is empty. A boulder cannot be pulled, and cannot be pushed into a wall, another boulder, or any obstacle. The level is solved when every boulder occupies a fountain tile. Getting a boulder stuck against a wall or corner with no way to reposition it makes the puzzle unsolvable.

### MazeWalk (9x9, 15x15)
- **Objective**: Find and step onto the stairs down.
- **Mechanics**: The agent is placed in a procedurally generated maze of walls. The maze has a single correct path to the stairs. The agent has limited visibility — only nearby tiles are described. The 9x9 variant has a smaller grid; the 15x15 variant is larger and more complex.

### Corridor
- **Objective**: Navigate through corridors to reach the stairs down.
- **Mechanics**: The level consists of connected corridors and rooms. The agent must traverse the corridor structure, potentially passing through doors. No monsters are present — the challenge is navigation and pathfinding.

### CorridorBattle-Dark
- **Objective**: Fight through dark corridors to reach the stairs down.
- **Mechanics**: The agent navigates dark corridors where visibility is severely limited. Hostile monsters spawn and must be defeated in combat. Combat is performed by moving into adjacent monsters. The agent may find and use items such as weapons, potions, and wands scattered in the level. The darkness restricts observation — the agent cannot see distant tiles.

### Quest (Easy, Medium)
- **Objective**: Explore rooms, overcome obstacles, and reach the stairs down.
- **Mechanics**: The level contains multiple connected rooms and corridors with various obstacles. The agent may encounter lava, locked doors, and monsters. Items found in the level (potions, wands, tools) can be used to bypass obstacles (e.g., levitation to cross lava, freezing water). The Medium variant presents more complex obstacle combinations than Easy.

Scoring: **avg_progression** per episode (0.0 or 1.0, binary — task completed or not), averaged across tasks and episodes. Episode lengths vary by task type (200–1000 steps depending on complexity).

## Core Constraints

- **NO TOOL CALLS**: The game-playing LLM must NOT use tools. BALROG calls `using_harness()` once per game step and expects exactly one valid game action string in return. If the LLM makes a tool call, it wastes the step and corrupts the message history. All intelligence must be implemented in prompts, context management, and hooks — NOT in tools. Do NOT create `tools_harness.py`.

