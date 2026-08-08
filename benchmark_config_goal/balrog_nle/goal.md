Maximize avg_progression on the NLE/NetHack task across all episodes.

> **Note for the evolution agent**: The detailed game mechanics below are factual reference for you as the "developer" writing harness code. Do NOT directly inject internal values (damage formulas, spawn rates, hidden mechanics, item probabilities) into game-playing prompts — the game-playing LLM does not receive these in the original BALROG setting and discovering strategy through observation and evaluation feedback is part of the challenge. Use this knowledge to design better prompt structures, context management, and hooks — not to dump raw game internals into prompts.

## Game

NLE (NetHack Learning Environment) wraps NetHack — one of the deepest and most complex roguelike games ever created. The agent controls a character in a procedurally generated dungeon, descending through levels while surviving, fighting, and managing resources.

The agent observes a text representation each step: a **message** line describing recent events, a **language observation** describing visible tiles and creatures in text, a **cursor** position, and **statistics** (or ASCII map). Inventory is provided separately. The observation includes bottom-line stats: HP, max HP, power, max power, armor class, gold, depth (Dlvl), experience level (Xp), experience points, time, hunger state, and condition flags.

The action space is the full NetHack keyboard — 90+ actions including movement, combat, magic, inventory management, and navigation.

### Dungeon Structure
- **Levels**: The dungeon consists of multiple levels connected by staircases. Each level is procedurally generated with rooms, corridors, doors, traps, shops, altars, fountains, sinks, and thrones.
- **Stairs**: Descending stairs (`>`) lead to deeper levels. Ascending stairs (`<`) lead back up. The agent must be standing on the stairs to use the `down` or `up` action.
- **Rooms and corridors**: Levels are composed of rooms connected by corridors. Doors can be open, closed, or locked. Hidden doors and corridors can be discovered with the `search` action.

### Movement and Navigation
- **Objective**: Explore the dungeon and descend to deeper levels.
- **Mechanics**: 8-directional movement (near: one tile; far: multiple tiles in one action). The `travel` action moves toward a location or landmark (`>` for down stairs, `<` for up stairs). Moving into a closed door opens it; moving into a locked door fails. The `search` action checks adjacent tiles for hidden doors and traps.

### Combat
- **Objective**: Defeat hostile monsters encountered during exploration.
- **Mechanics**: Move into an adjacent monster to attack with the wielded weapon (melee). Monsters move and attack each turn. The `kick` action deals damage at close range and can break locked doors or chests. The `zap` action fires a wand in a chosen direction. The `throw` action hurls an item at a target. The `fire` action shoots ammunition from the quiver. The `cast` action casts a spell (requires knowing the spell). The `fight` action attacks a square even if no monster is visible there.

### Survival
- **Objective**: Keep the character alive.
- **Mechanics**:
  - **Hit points (HP)**: Damaged by monster attacks, traps, and other hazards. HP regenerates slowly over time. Death occurs when HP reaches 0.
  - **Hunger**: The hunger state drains over time. When hungry, the agent must `eat` food from inventory or from corpses on the ground. Starvation causes damage and eventually death.
  - **Power (energy)**: Consumed by casting spells. Regenerates over time.
  - **Armor class**: Lower is better. Improved by `wear`-ing armor.

### Inventory Management
- **Objective**: Carry and use items effectively.
- **Mechanics**:
  - `pickup` picks up items at the agent's feet. `drop` places an item on the ground.
  - `wield` equips a weapon. `wear` puts on armor. `takeoff` removes armor. `puton` / `remove` manage accessories (rings, amulets).
  - `quiver` selects ammunition. `swap` switches between primary and secondary weapons.
  - `inventory` displays all carried items with their letter labels.
  - `apply` uses a tool (e.g., applying a key to a locked door, a whistle, a camera).
  - Many item prompts ask for a letter (e.g., "What do you want to eat? [dgh or ?*]") — respond with the corresponding single character.

### Item Types
- **Weapons and armor**: Wielded and worn to improve combat stats.
- **Scrolls**: `read` to activate effects (identification, enchantment, teleportation, etc.).
- **Potions**: `quaff` to drink for various effects (healing, gain ability, invisibility, etc.).
- **Wands**: `zap` in a direction for ranged effects (striking, teleportation, polymorph, etc.).
- **Food**: `eat` to reduce hunger. Comestibles include food rations, fruit, tins, and monster corpses (some corpses have side effects).
- **Tools**: `apply` to use (keys, whistles, mirrors, cameras, etc.).
- **Spellbooks**: `read` to learn spells, then `cast` to use them.
- **Gems**: Valuable for score; some have magical properties.

### Religion
- **Objective**: Use divine intervention when in dire need.
- **Mechanics**: `pray` to the agent's god — may heal, remove curses, or save from death, but cannot be used too frequently. `offer` a sacrifice on a co-aligned altar (requires standing on an altar and holding a corpse).

### Progression
- **Objective**: Descend to deeper dungeon levels and gain experience.
- **Mechanics**: Progression is tracked by two metrics:
  - **Dlvl** (dungeon depth): The current dungeon level number (1–50, then Astral Plane). Deeper levels contribute higher progression values.
  - **Xp** (experience level): Gained by defeating monsters. Higher experience levels contribute higher progression values.
  - The highest progression value across both Dlvl and Xp is used as the episode score. The maximum progression (1.0) is achieved by ascending (winning the game).
- Scoring: **avg_progression** based on the highest Dlvl/Xp achievement reached, averaged across episodes. Extremely long horizon: default 100000 steps per episode.

## Core Constraints

- **NO TOOL CALLS**: The game-playing LLM must NOT use tools. BALROG calls `using_harness()` once per game step and expects exactly one valid game action string in return. If the LLM makes a tool call, it wastes the step and corrupts the message history. All intelligence must be implemented in prompts, context management, and hooks — NOT in tools. Do NOT create `tools_harness.py`.
