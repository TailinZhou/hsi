# HSI — Hierarchical Self-Improvement for Hot-Swappable Agent Harnesses

HSI is a Godel-style self-improvement framework in which a single frozen LLM $M$ operates across three layered scopes — **task harness**, **evolver**, and **meta-evolver** — to rewrite its own harness code, the strategy that governs the rewriting, and the selector that exports the final deployed version. A thinking-on/off design isolates the harness contribution: thinking is disabled at task time to cap the model's per-step ceiling, and enabled when rewriting the harness to give self-modification its best chance.

The frozen-backbone constraint rules out any system that recruits an external or stronger proposer. The only seam between scopes is message-history ownership: each scope is a memory boundary, not a separate agent.

## Pipeline

Five stages structure the loop. The first three operate on the task harness $H$. The fourth rewrites the evolver strategy $\Sigma$ under `evolution/`. The fifth terminates the run by selecting the version for held-out test evaluation.

| Stage | Edits | Tool Scope |
|---|---|---|
| 1. Seed selection | Reads $\mathcal{G}_t$, emits hypothesis $h_t$ + seed | Read on `evolution/` + dev eval (3 calls) |
| 2. Main evolution | Rewrites $H$ (per-step policy, prompts, hooks, memory, tools) | File-system on harness dir + `evaluate` + `plan`/`lesson`/`probe` |
| 3. Commit selection | Picks 2-5 diverse versions per iteration | Appended to main evolution's history |
| 4. Meta-evolution | Rewrites `select_seed`/`select_commit` under `evolution/` | File-system confined to `evolution/` |
| 5. Best-version selection | Picks the exported version on `val` | Fixed, non-evolvable agentic stage |

Two structural invariants make the harness hot-swappable:

- The benchmark injection entry signature `using_harness(agent, task)` is held fixed. Any internal of $H$ may change; this seam cannot.
- The meta-evolver's own execution logic is loaded from `godel_evolution_init/` and never modified. It is the outer frozen anchor.

Two memory channels carry context across the iteration: `plan.md` (ephemeral, rolled back with the code if the iteration is abandoned) and `BOOTSTRAP.md` (permanent, read by future seed selections).

## Repository Layout

```
hsi/
├── src/
│   ├── react_loop/          # HSI core: GodelAgent, EvolveHelper, MetaEvolveHelper, ArchiveManager
│   └── benchmark/           # BALROG evaluator (with agentdojo, terminal_bench as extension examples)
├── godel_harness_init/      # Frozen task-harness templates (one per BALROG env)
│   ├── balrog/              # Multi-env combined
│   ├── balrog_babyai/       # Per-env templates
│   ├── balrog_crafter/
│   ├── balrog_minihack/
│   ├── balrog_nle/
│   ├── balrog_textworld/
│   └── balrog_babaisai/
├── godel_evolution_init/    # Frozen meta-evolver anchor (select_seed/select_commit/select_best + strategies/)
├── benchmark_config_goal/   # Per-suite config.yaml + goal.md
├── reported_results/        # 13 final runs feeding the paper tables + setup_*.md + trajectory plots
├── baseline_results/        # Init-harness baselines (DeepSeek-V4-Flash)
├── scripts/
│   ├── download_balrog_data.py
│   └── eval_harness_snapshot.py
├── main.py                  # Entry point
├── config.yaml              # Default config (BabyAI Setup A)
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

HSI needs a BALROG data download (one-time):

```bash
python scripts/download_balrog_data.py
```

Set up the LLM endpoint in `.env`:

```
OPENAI_API_KEY=your-key
OPENAI_API_BASE=https://api.deepseek.com   # for DeepSeek-V4-Flash
```

## Quick Start

Run the default config (BabyAI, Setup A, meta-on, $T=5$ iterations, 80 react() steps per iteration):

```bash
python main.py
```

To run a different suite, point `main.py` at the corresponding config:

```bash
python main.py benchmark_config_goal/balrog_babyai/config.yaml          # Setup A
python main.py benchmark_config_goal/balrog_textworld/config.yaml       # Setup A
python main.py benchmark_config_goal/balrog_crafter/config.yaml         # Setup A
python main.py benchmark_config_goal/balrog_minihack/config.yaml        # Setup A
python main.py benchmark_config_goal/balrog_nle/config.yaml             # Setup A
python main.py benchmark_config_goal/balrog_babaisai_breakstop/config.yaml   # Setup B (held-out)
python main.py benchmark_config_goal/balrog_babaisai_goto/config.yaml        # Setup B (held-out)
python main.py benchmark_config_goal/balrog_babaisai_make/config.yaml        # Setup B (held-out)
```

To resume an interrupted run:

```bash
python main.py --resume evolution_results/balrog_babyai/run_<timestamp>
```

## Configuration

`config.yaml` is the single source of truth. Key sections:

- `llm` — model, `thinking_enabled` (true for evolver/meta-evolver), `reasoning_effort: "max"`
- `evolution` — `max_iterations: 5`, `max_steps_per_iteration: 80`, `lcb_zscore: 0.5`, `evaluate_llm_summary: true`, `init_eval_enabled: false`
- `harness` — `thinking_enabled: false` (task-time thinking is OFF by design), `temperature: 0.0`
- `init` — paths to harness and evolution init templates
- `meta_evolve` — `enabled`, `max_steps: 50`, `archive_strategy: "greedy"`, `inject_seed_hypothesis: true`, `seed_eval_enabled: true`, `evolvable_commit_strategy: true`, `submit_best_enabled: true`, `submit_best_max_steps: 80`
- `benchmark` — `type: balrog`, `suite`, `dev_ratio` (1.0 for Setup A, 0.8 for Setup B), `val_ratio`, `dynamic_sample`, `test_repeats: 3`

The reward is the stochastic lower-confidence bound $r = \mu - z \cdot \sigma / \sqrt{n}$ with $z = 0.5$, computed per task across episodes then averaged across tasks.


## Output Structure

Each run produces:

```
evolution_results/<suite>/run_<timestamp>/
├── repo/                       # Git-tracked evolution history
│   ├── harness.py              # Evolved harness
│   ├── prompts.py, hooks.py, context.py
│   ├── evolution/              # Evolver strategy Σ (meta-editable)
│   │   ├── select_seed.py, select_commit.py
│   │   └── strategies/
│   └── .evolution/             # Persistent context (summaries, message history)
├── agent_code_best_<ts>/       # Exported deployed harness
├── evolution_graph.html        # Visualized cumulative graph $\mathcal{G}_T$
├── evolution_metadata.json
├── final_results.json
├── test_repeat_results.json
├── usage_summary.json
└── context.json                # For resumption
```

## What HSI Is Not

- **Not test-time search.** One candidate per iteration; no population-based parallel scaling. Population-level diversity is preserved via the commit pool, not via parallel rollouts.
- **Not external-proposer.** The same frozen $M$ that executes the harness also rewrites it. The meta-evolver's own execution logic is loaded from `godel_evolution_init/` and is never edited by the agent.
- **Not universal.** Harness evolution is bounded by the backbone's intellectual ceiling. On tasks beyond that ceiling (e.g. NLE under DeepSeek-V4-Flash-Preview), no harness redesign closes the gap. This is consistent with the VC-dimension limit on self-improving agents.

## License

MIT
