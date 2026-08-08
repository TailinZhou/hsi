Maximize avg_progression on the Crafter survival task across all episodes.

## Game

Crafter is a 2D tile-based survival crafting game. The agent must manage four survival bars — **health**, **food**, **drink**, and **energy** (each 0–9) — while exploring, gathering resources, crafting tools, and defeating enemies.

The game features 22 achievements organized in a dependency tree:
- **Collection**: collect wood, collect sapling, collect stone, collect coal, collect iron, collect diamond, collect drink
- **Crafting**: make wood pickaxe, make stone pickaxe, make iron pickaxe, make wood sword, make stone sword, make iron sword
- **Combat & Food**: defeat zombie, defeat skeleton, eat cow, eat plant
- **Building**: place stone, place table, place furnace, place plant
- **Survival**: wake up

Achievements have crafting dependencies (e.g. furnace requires table on the ground, iron tools require furnace). The game world is procedurally generated each episode.

### Facing Direction

The agent has a facing direction (north/south/east/west), defaulting to south. Each Move action changes facing to match the movement direction (Move West → agent faces west). The Do and Place actions target the single tile directly in front of the agent based on facing direction.

### Resource Collection

Move adjacent to resources, face them, and use Do to collect. Harder materials require appropriate tools in inventory:

| Resource | Tool Required | Yields |
|----------|--------------|--------|
| Wood (from trees) | None | 1 wood |
| Sapling (from grass) | None | 1 sapling (10% chance per attempt) |
| Drink (from water) | None | +1 drink |
| Stone | Wood Pickaxe | 1 stone |
| Coal | Wood Pickaxe | 1 coal |
| Iron | Stone Pickaxe | 1 iron |
| Diamond | Iron Pickaxe | 1 diamond |

Tools do not break — once crafted, they persist for the rest of the episode.

### Crafting

Use Make actions to craft items from inventory. The agent must stand adjacent to the required structures.

| Item | Materials Consumed | Nearby Structures Required |
|------|--------------------|---------------------------|
| Wood Pickaxe | 1 wood | table |
| Stone Pickaxe | 1 wood + 1 stone | table |
| Iron Pickaxe | 1 wood + 1 coal + 1 iron | table + furnace |
| Wood Sword | 1 wood | table |
| Stone Sword | 1 wood + 1 stone | table |
| Iron Sword | 1 wood + 1 coal + 1 iron | table + furnace |

### Combat & Food

Face adjacent enemies or animals and use Do to attack. Damage depends on weapon: bare hands = 1, wood sword = 2, stone sword = 3, iron sword = 5.

**Zombie**: 5 HP. Chases within 8 tiles (90% move chance, 80% direct path). Deals 2 damage melee (7 if player sleeping). Attack cooldown: 5 ticks.

**Skeleton**: 3 HP. Shoots arrows for 2 damage at range up to 5 tiles. Retreats if player is within 3 tiles. Arrow reload: 4 ticks. Detection range: 8 tiles.

**Cow**: 3 HP. Passive — 50% chance to move randomly each tick. Killing one grants +6 food and the "eat cow" achievement.

**Plant**: Placed plants grow over time and can be harvested with Do when ripe, granting +4 food and the "eat plant" achievement.

### Survival Bars

All bars range 0–9. Food, drink, and energy drain over time (not per action):

**Food**: decreases 1 point every 25 ticks while awake, every 50 ticks while sleeping.

**Drink**: decreases 1 point every 20 ticks while awake, every 40 ticks while sleeping.

**Energy**: fatigue accumulates at +1 per tick awake. When fatigue exceeds 30, energy decreases by 1 and fatigue resets. When sleeping, fatigue decreases; when it drops below -10, energy recovers by 1 and fatigue resets.

**Health**: regenerates +1 per 25 ticks when food > 0 AND drink > 0 AND (energy > 0 OR sleeping). When any bar is 0, health degenerates at 1 per 15 ticks. Sleeping doubles regeneration rate.

### Sleep

Can only sleep when energy < 9 (max). While sleeping: cannot take any other action; food and drink drain at half rate; energy and health recover. Wakes automatically when energy reaches 9 (grants "wake up"). Wakes immediately if damaged.

### Death

Health drops to 0 from combat, survival bar depletion, or stepping on lava (instantly fatal). After death, the episode continues for remaining steps but the agent sees only "You died." and cannot act meaningfully.

### Terrain

Walkable: grass, path, sand. Water, trees, stone, coal, iron, diamond, and placed objects are NOT walkable. Lava is walkable but instantly fatal.

### Building

Use Place actions to put objects on the tile directly in front of the agent. The target tile must be empty of objects.

| Place Action | Materials Consumed | Valid Terrain |
|-------------|--------------------|--------------|
| Place Stone | 1 stone | grass, sand, path, water, lava |
| Place Table | 2 wood | grass, sand, path |
| Place Furnace | 4 stone | grass, sand, path |
| Place Plant | 1 sapling | grass only |

## Actions

The LLM must output exactly one of these 17 case-sensitive action strings each step:

Noop, Move West, Move East, Move North, Move South, Do, Sleep, Place Stone, Place Table, Place Furnace, Place Plant, Make Wood Pickaxe, Make Stone Pickaxe, Make Iron Pickaxe, Make Wood Sword, Make Stone Sword, Make Iron Sword

## Observation Format

Each step the game provides two text observations:

**Environment text** (passed as `obs` to harness):
- Status line: "You are sleeping, and will not be able take actions until energy is full." / "You died." / (empty when normal)
- Environment: "You see:\n- wood 2 steps to your north\n- stone 3 steps to your west and 1 step to your south\n..."
- Facing: "You face tree at your front." / "You face nothing at your front."

**Inventory text** (passed as `short_term` in task_context):
```
Your status:
- health: 5/9
- food: 3/9
- drink: 7/9
- energy: 2/9

Your inventory:
- wood: 5
- stone: 2
```
Or "You have nothing in your inventory." when all material/tool counts are 0.

**task_context keys** available to harness code:
- `env_name`: "crafter"
- `instruction`: the 17 actions list + 22 achievements prompt (constant across episode)
- `short_term`: inventory/status text (see above)
- `is_new_episode`: True on first step of each episode
- `last_step_reward`: reward from previous step (0.0 on first step, non-zero when achievement unlocks)
- `achievements`: dict of all unlocked achievements with counts (empty on first step)
- `recent_unlocked`: list of newly unlocked achievement names from previous step
- `naive_instruction`: always empty for Crafter

## Scoring

Per-episode: score = number of achievements with count > 0 / 22. Repeat completions of the same achievement do not increase score further.

**avg_progression** = mean of per-episode scores across all evaluation episodes.

Episodes run 2000 steps each. The world is procedurally generated each episode.

## Core Constraints

- **NO TOOL CALLS**: The game-playing LLM must NOT use tools. BALROG calls `using_harness()` once per game step and expects exactly one valid game action string in return. If the LLM makes a tool call, it wastes the step and corrupts the message history. All intelligence must be implemented in prompts, context management, and hooks — NOT in tools. Do NOT create `tools_harness.py`.
