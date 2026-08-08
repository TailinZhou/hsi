Maximize avg_progression across all TextWorld tasks and episodes.

## Game

TextWorld is an interactive fiction engine with text-based adventure games. You explore rooms, collect items, and complete multi-step objectives through text commands (e.g. "go north", "take key", "open door", "insert carrot into oven").

Three task variants are evaluated:

### Treasure Hunter (~20 rooms)
- **Objective**: Find and take a specific target object.
- **Environment**: ~20 rooms in a procedural maze. Rooms have themed names and are connected by cardinal directions. Some passages have doors that may be locked.
- **Objects**: One target object, keys, containers (chests, safes, boxes), doors. The target may be in various locations throughout the maze.
- **Mechanics**:
  - Doors and containers can be in three states: open, closed (unlocked), or locked.
  - Locked things require a matching key — keys are adjective-matched to their locks (e.g. "round keycard" matches "round safe"). You must have the key in inventory to unlock.
  - Keys may be lying in rooms or inside containers (which may themselves be locked).
  - After `unlock`, the door/container remains closed — you must `open` it separately to access contents or pass through.
  - Taking the target object wins the game.
- **Available actions**: `look`, `goal`, `inventory`, `go <dir>`, `open <thing>`, `close <thing>`, `take <thing>`, `take <thing> from <container>`, `drop <thing>`, `examine <thing>`, `unlock <thing> with <key>`

### The Cooking Game (~12 rooms)
- **Objective**: Follow a recipe to prepare and eat a meal.
- **Environment**: ~12 rooms including a kitchen, supermarket, and other household rooms. Ingredients and tools are distributed across rooms in containers (fridge, cupboard, pantry) and on supporters (counter, table).
- **Objects**: Cookbook, raw ingredients (with colors, e.g. "red apple", "green pepper"), knife, cooking appliances (BBQ, stove, oven), containers, supporters.
- **Mechanics**:
  - A cookbook in one room contains the recipe — `examine cookbook` to read it.
  - The recipe specifies: required ingredients (with exact colors), cutting method (chop/slice/dice), and cooking method (grill/fry/roast).
  - Cooking appliances: BBQ is for grilling, stove is for frying, oven is for roasting.
  - Ingredients must be processed (chop/slice/dice with knife) BEFORE cooking — cooking a raw unprocessed ingredient fails.
  - Incorrect cooking (wrong method, wrong state of ingredient) leads to failure.
  - Ingredient colors must match the recipe exactly (if recipe says "red apple", only "red apple" works, not "green apple").
  - `prepare meal` only works in the kitchen room.
  - After preparing the meal, `eat meal` wins the game.
- **Available actions**: `look`, `goal`, `inventory`, `go <dir>`, `examine <thing>`, `open <thing>`, `take <thing>`, `take <thing> from <container>`, `drop <thing>`, `cook <food> with <appliance>`, `chop <food> with knife`, `slice <food> with knife`, `dice <food> with knife`, `prepare meal`, `eat meal`

### Coin Collector (~58 rooms)
- **Objective**: Find the coin and take it.
- **Win condition**: Executing `take coin` while in the room that contains the coin.
- **Environment**: ~58 rooms in a large procedural maze. No locked doors, no containers, no keys — only rooms connected by cardinal exits. This is the largest maze of the three tasks.
- **Objects**: Exactly one coin, placed in exactly one room.
- **Mechanics**:
  - The coin is visible in the room description text when you enter the room that contains it (e.g. "There is a coin on the floor."). In rooms without the coin, no mention of a coin appears.
  - `take coin` in a room without a visible coin returns "You can't see any such thing." and wastes the step.
  - No other objects or interactions exist — movement and `take coin` are the only meaningful actions.
- **Available actions**: `goal`, `go <dir>`, `take coin`

Each task generates procedural maps, so memorization does not help — the agent must reason from observations.

Scoring: **avg_progression** (0.0–1.0) per episode, averaged across tasks and episodes.

## Core Constraints

- **NO TOOL CALLS**: The game-playing LLM must NOT use tools. BALROG calls `using_harness()` once per game step and expects exactly one valid game action string in return. If the LLM makes a tool call, it wastes the step and corrupts the message history. All intelligence must be implemented in prompts, context management, and hooks — NOT in tools. Do NOT create `tools_harness.py`.
