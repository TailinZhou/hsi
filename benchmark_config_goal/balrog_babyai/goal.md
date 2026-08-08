Maximize avg_progression across all BabyAI tasks and episodes.

## Game

BabyAI is a grid-world environment where the agent receives a text instruction (mission) and must execute it by navigating the grid. Instructions involve going to objects, picking them up, putting them next to or in front of other objects, or opening doors.

Example missions: "go to the red ball", "pick up the green key", "open the red door", "put the blue ball next to the grey key".

The grid contains colored objects (balls, keys, boxes) and doors. The agent has a limited field of view and must explore to find target objects.

Five task variants are evaluated with varying complexity of instructions and grid configurations:

### GoTo
- **Objective**: Navigate to a specific target object described in the mission.
- **Objects**: Colored objects (balls, keys, boxes) and doors.
- **Mechanics**:
  - The agent must locate the target object in the grid and move adjacent to it.
  - The mission specifies the object type and color (e.g., "go to the red ball").

### PickUp
- **Objective**: Pick up a specific object described in the mission.
- **Objects**: Colored objects (balls, keys, boxes).
- **Mechanics**:
  - The agent must locate the target object, move on top of it, and pick it up.

### Open
- **Objective**: Open a specific door described in the mission.
- **Objects**: Colored doors.
- **Mechanics**:
  - The agent must locate the target door, face it, and toggle it open.

### PutNext
- **Objective**: Pick up one object and place it next to another target object.
- **Objects**: Colored objects (balls, keys, boxes).
- **Mechanics**:
  - The agent must pick up the first object, navigate to the second object, and drop the first object adjacent to it.

### PickUpSeqGoTo
- **Objective**: Pick up a sequence of objects in order, then navigate to a final target.
- **Objects**: Colored objects (balls, keys, boxes) and doors.
- **Mechanics**:
  - The mission specifies multiple objects to pick up in sequence, followed by a navigation goal.

Each task generates procedural grid layouts, so memorization does not help — the agent must reason from observations.

Scoring: **avg_progression** (0.0 or 1.0 per episode, based on mission completion), averaged across tasks and episodes.

## Observation System

- **Line-of-sight**: Objects are reported with relative direction and distance. An object may be visible without a clear path to reach it.
- **No collision feedback**: Walking into a wall silently keeps the agent in place. The observation regenerates from the unchanged position, producing identical text.

## Core Constraints

- **NO TOOL CALLS**: The game-playing LLM must NOT use tools. BALROG calls `using_harness()` once per game step and expects exactly one valid game action string in return. If the LLM makes a tool call, it wastes the step and corrupts the message history. All intelligence must be implemented in prompts, context management, and hooks — NOT in tools. Do NOT create `tools_harness.py`.
