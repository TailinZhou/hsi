#!/usr/bin/env python3
"""
React Loop Agent - Entry Point

Usage:
    python main.py                                    # use default config.yaml
    python main.py benchmark_config_goal/paper_review/config.yaml
    python main.py --resume evolution_results/balrog/run_20260604_093750
    python main.py --resume evolution_results/balrog/run_20260604_093750 --config benchmark_config_goal/balrog/config.yaml
"""

import os
import sys
import json
import shutil
import statistics
from pathlib import Path
from datetime import datetime

# Windows Unicode fix: force UTF-8 for stdout/stderr before any output
# This must run before any print() or import that might trigger output
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add src to sys.path
# Prefer resolve() to get an absolute path; if CWD has been deleted causing
# getcwd() to fail, fall back to a relative path (the script lives in the
# project root, so a relative path usually works too).
_script_dir = Path(__file__).parent
try:
    _script_dir = _script_dir.resolve()
except (FileNotFoundError, OSError):
    pass
sys.path.insert(0, str(_script_dir / "src"))

# CWD captured at process startup (absolute path). During long self-evolving runs
# the process CWD may be deleted by the agent's own bash actions, after which
# os.getcwd() raises FileNotFoundError and breaks all CWD-dependent code in
# waves (harness loader abspath→getcwd, log directories, benchmark env, etc.).
# Capture an absolute CWD here at startup (when CWD is guaranteed to exist) for
# later CWD recovery / relative-path resolution, so runtime code never depends
# on getcwd again.
try:
    _startup_cwd = Path(os.getcwd())
except (FileNotFoundError, OSError):
    _startup_cwd = _script_dir  # Extreme fallback: use the script directory

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from react_loop.agent import GodelAgent, GodelAgentConfig
from react_loop.state import reward_to_scalar
from benchmark.config import BenchmarkConfig
from benchmark.adapter import create_benchmark_evaluator

# Project root directory
ROOT_DIR = Path(__file__).parent


def get_evolution_goal(config_path: str) -> str:
    """Load evolution goal from the same directory as config.yaml."""
    goal_path = Path(config_path).parent / "goal.md"
    if goal_path.exists():
        return goal_path.read_text(encoding="utf-8")
    return "Self-evolve your strategy."


def load_secrets():
    """Load secrets from .env."""
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_API_BASE')
    if not api_key:
        print("Error: OPENAI_API_KEY is not set, please check the .env file")
        sys.exit(1)
    return {
        'api_key': api_key,
        'base_url': base_url,
    }


def load_config(config_path: str) -> dict:
    """Load configuration from YAML."""
    p = Path(config_path)
    if not p.exists():
        print(f"Error: Config file {p} does not exist")
        sys.exit(1)
    print(f"Loading config: {p}")
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_agent_config(cfg: dict) -> GodelAgentConfig:
    """Build GodelAgentConfig from yaml config."""
    llm = cfg.get('llm', {})
    evo = cfg.get('evolution', {})
    harness = cfg.get('harness', {})
    init = cfg.get('init', {})
    out = cfg.get('output', {})
    meta = cfg.get('meta_evolve', {})

    return GodelAgentConfig(
        # LLM
        model=llm.get('model', 'gpt-4'),
        temperature=float(llm.get('temperature', 0.7)),
        thinking_enabled=bool(llm.get('thinking_enabled', False)),
        reasoning_effort=str(llm.get('reasoning_effort', 'high')),
        # Evolution
        max_iterations=int(evo.get('max_iterations', 10)),
        max_steps_per_iteration=int(evo.get('max_steps_per_iteration', 50)),
        enable_bash=bool(evo.get('enable_bash', True)),
        max_context_tokens=int(evo.get('max_context_tokens', 68000)),
        # History truncation
        max_history_messages=int(evo.get('max_history_messages', 10000)),
        max_tool_result_length=int(evo.get('max_tool_result_length', 100000)),
        # Evaluate summary
        evaluate_llm_summary=evo.get('evaluate_llm_summary', True),
        evaluate_consolidate_summary=evo.get('evaluate_consolidate_summary', False),
        # Bootstrap init-harness evaluation before iteration 1
        init_eval_enabled=bool(evo.get('init_eval_enabled', True)),
        eval_feedback_style=str(evo.get('eval_feedback_style', 'diary')),
        enable_file_log=bool(evo.get('enable_file_log', False)),
        # LCB uncertainty penalty (single source for reward + version selection)
        lcb_zscore=float(evo.get('lcb_zscore', 1.0)),
        # Lesson-nudge fallback: if the agent didn't call lesson() before
        # compacting, append a short nudge to the same conversation and run
        # up to this many react steps so it records a cross-iteration lesson.
        # 0 = skip the nudge entirely (no lesson lands if the agent forgot).
        lesson_nudge_max_steps=int(evo.get('lesson_nudge_max_steps', 2)),
        # Archive
        archive_enabled=bool(meta.get('archive_enabled', True)),
        archive_strategy=str(meta.get('archive_strategy', 'recursive')),
        # Exclude tools
        evolve_exclude_tools=evo.get('exclude_tools', []),
        # Harness
        harness_enable_bash=bool(harness.get('enable_bash', False)),
        harness_temperature=(
            float(harness['temperature']) if 'temperature' in harness else None
        ),
        harness_thinking_enabled=(
            bool(harness['thinking_enabled']) if 'thinking_enabled' in harness else None
        ),
        harness_max_tokens=(
            int(harness['max_tokens']) if 'max_tokens' in harness else None
        ),
        # Init
        godel_harness_init_path=init.get('godel_harness_init_path', ''),
        godel_evolution_init_path=init.get('godel_evolution_init_path', ''),
        # Output
        output_dir=out.get('directory', './evolved_agents'),
        verbose=out.get('verbose', True),
        # Meta-evolution
        meta_evolve_enabled=bool(meta.get('enabled', True)),
        meta_evolve_max_steps=int(meta.get('max_steps', 10)),
        meta_evolve_enable_bash=bool(meta.get('enable_bash', True)),
        iter_per_metaevolve=max(1, int(meta.get('iter_per_metaevolve', 1))),
        evolvable_commit_strategy=bool(meta.get('evolvable_commit_strategy', True)),
        commit_nudge_max_steps=int(meta.get('commit_nudge_max_steps', 3)),
        seed_selection_max_steps=int(meta.get('seed_selection_max_steps', 10)),
        # Seed hypothesis injection
        inject_seed_hypothesis=bool(meta.get('inject_seed_hypothesis', True)),
        # Seed eval (validate seed hypothesis during seed selection)
        seed_eval_enabled=bool(meta.get('seed_eval_enabled', False)),
        seed_eval_max_calls=int(meta.get('seed_eval_max_calls', 1)),
        # Submit-best — fixed, non-evolvable agentic final-best selection stage.
        submit_best_enabled=bool(meta.get('submit_best_enabled', True)),
        submit_best_max_steps=int(meta.get('submit_best_max_steps', 50)),
        # Knowledge graph
        meta_evolve_kg_enabled=bool(meta.get('kg_enabled', True)),
        meta_evolve_kg_max_nodes=int(meta.get('kg_max_nodes', 100)),
        meta_evolve_kg_concurrency=int(meta.get('kg_concurrency', 2)),
    )


def create_llm_client(secrets: dict):
    """Create the LLM client."""
    import httpx
    return OpenAI(
        api_key=secrets['api_key'],
        base_url=secrets['base_url'],
        timeout=httpx.Timeout(1200.0, connect=60.0),
    )


def parse_cli_args(args):
    """Parse CLI arguments for --resume and --config flags."""
    resume_dir = None
    config_path = None
    i = 0
    while i < len(args):
        if args[i] == "--resume" and i + 1 < len(args):
            resume_dir = os.path.abspath(args[i + 1])
            i += 2
        elif args[i] == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
            i += 2
        elif not config_path and not args[i].startswith("--"):
            config_path = args[i]
            i += 1
        else:
            i += 1
    return resume_dir, config_path


def validate_resume_dir(resume_dir):
    """Validate that a resume directory has all required files."""
    if not os.path.isdir(resume_dir):
        print(f"Error: Resume directory does not exist: {resume_dir}")
        sys.exit(1)

    metadata_path = os.path.join(resume_dir, "evolution_metadata.json")
    repo_path = os.path.join(resume_dir, "repo")
    if not os.path.exists(metadata_path):
        print(f"Error: evolution_metadata.json not found in {resume_dir}")
        sys.exit(1)
    if not os.path.isdir(repo_path):
        print(f"Error: repo/ directory not found in {resume_dir}")
        sys.exit(1)


def _summarize_reward_runs(values):
    """Compute descriptive statistics over a list of scalar reward values.

    Returns n/mean/std/variance/min/max. std/variance are sample estimates
    (ddof=1) and fall back to 0.0 when fewer than 2 samples are available,
    since statistics.stdev/variance would otherwise raise.
    """
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "variance": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": n,
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if n >= 2 else 0.0,
        "variance": statistics.variance(values) if n >= 2 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main():
    print("=" * 60)
    print("React Loop Agent - Starting Evolution")
    print("=" * 60)

    # Parse CLI arguments
    resume_dir, config_path = parse_cli_args(sys.argv[1:])

    # Resolve config path
    if resume_dir:
        validate_resume_dir(resume_dir)
        # Config: --config override > run_dir/config.yaml
        if not config_path:
            config_path = os.path.join(resume_dir, "config.yaml")
        print(f"Resuming from: {resume_dir}")
        print(f"Config: {config_path}")
    else:
        if not config_path:
            config_path = str(ROOT_DIR / "config.yaml")

    # 1. Load configuration
    cfg = load_config(config_path)
    secrets = load_secrets()
    agent_config = build_agent_config(cfg)

    print(f"Model: {agent_config.model} (temperature={agent_config.temperature}, thinking={agent_config.thinking_enabled}, reasoning_effort={agent_config.reasoning_effort})")
    print(f"API Base: {secrets['base_url']}")
    print(f"enable_bash (evolve): {agent_config.enable_bash}")
    print(f"harness_enable_bash: {agent_config.harness_enable_bash}")
    print(f"harness_temperature: {agent_config.harness_temperature}, harness_thinking_enabled: {agent_config.harness_thinking_enabled}")

    # 2. Create client
    client = create_llm_client(secrets)

    if resume_dir:
        # Resume: reuse existing run directory
        run_output_dir = resume_dir
        agent_config.output_dir = run_output_dir

        # Load resume state
        resume_state = GodelAgent.load_resume_state(run_output_dir)
        agent_config.resume_from = resume_state

        completed = resume_state.get("completed_iterations", 0)
        original_max = resume_state.get("max_iterations", 0)
        print(f"Completed iterations: {completed}/{original_max}")
        print(f"New max_iterations: {agent_config.max_iterations}")

        # Warn if already completed and no config override
        if completed >= agent_config.max_iterations:
            print(f"\nWarning: Run already completed ({completed} iterations).")
            print(f"  Use --config with a higher max_iterations to extend the run.")
            sys.exit(0)
    else:
        # Normal: create new timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = f"{agent_config.output_dir}/run_{timestamp}"
        agent_config.output_dir = run_output_dir

        # Copy config.yaml to run directory
        os.makedirs(run_output_dir, exist_ok=True)
        shutil.copy2(config_path, run_output_dir)
        print(f"Config file backed up to: {run_output_dir}")

    # 4. Create benchmark evaluator (standalone config)
    bench_cfg = cfg.get('benchmark', {})
    if bench_cfg.get('type'):
        benchmark_config = BenchmarkConfig(
            type=bench_cfg['type'],
            suite=bench_cfg.get('suite', 'workspace'),
            attack=bench_cfg.get('attack', 'important_instructions'),
            categories=bench_cfg.get('categories'),
            max_tasks_per_category=bench_cfg.get('max_tasks_per_category'),
            max_tasks_per_category_test=bench_cfg.get('max_tasks_per_category_test'),
            verbose=bench_cfg.get('verbose', False),
            dev_ratio=bench_cfg.get('dev_ratio', 1.0),
            split_seed=bench_cfg.get('split_seed', 42),
            val_ratio=bench_cfg.get('val_ratio', 0.0),
            mode=bench_cfg.get('mode', 'batch'),
            data_root=bench_cfg.get('data_root'),
            run_all_tasks=bench_cfg.get('run_all_tasks', False),
            dynamic_sample=bench_cfg.get('dynamic_sample', True),
            parallel_workers=int(bench_cfg.get('parallel_workers', 1)),
            summary_passed_samples=int(bench_cfg.get('summary_passed_samples', 0)),
        )
        benchmark_config._raw_yaml = cfg
        benchmark_evaluator = create_benchmark_evaluator(benchmark_config)
        evolution_goal = get_evolution_goal(config_path)
    else:
        benchmark_evaluator = None
        evolution_goal = "Self-evolve your strategy."

    # 5. Create agent
    agent = GodelAgent(
        repo_path=f"{run_output_dir}/repo",
        llm_client=client,
        goal=evolution_goal,
        config=agent_config,
        external_evaluator=benchmark_evaluator,
    )

    bench_type = benchmark_config.type if benchmark_evaluator else "none"
    print(f"\nConfiguration complete:")
    print(f"  Initial strategy: {agent_config.godel_harness_init_path}")
    print(f"  Run directory: {run_output_dir}")
    print(f"  Iterations: {agent_config.max_iterations}")
    print(f"  Benchmark: {bench_type} ({bench_cfg.get('suite', 'N/A')})")
    if benchmark_evaluator:
        print(f"  Dev/Val/Test split: dev_ratio={benchmark_config.dev_ratio}, val_ratio={benchmark_config.val_ratio}, mode={benchmark_config.mode}, dynamic_sample={benchmark_config.dynamic_sample}")
        if benchmark_config.run_all_tasks:
            print(f"  Post-evolution eval: ALL tasks (dev+val+test)")
    print("=" * 60)

    # 6. Run evolution
    print("\nStarting...")

    try:
        result = agent.evolve()
        print("\n" + "=" * 60)
        print("Evolution Complete!")
        print("=" * 60)
        print(f"Iterations: {result['iterations_completed']}")
        if result['best_version']:
            print(f"Best version: {result['best_version'][0][:7]} (reward: {result['best_version'][1]:.4f})")
        if result.get('exported_path'):
            print(f"Best strategy exported to: {result['exported_path']}")
        if result.get('visualization_path'):
            print(f"Evolution graph visualization: {result['visualization_path']}")

        # 7. Run test set evaluation (if any) — repeated multiple times to reduce reward variance
        if benchmark_evaluator:
            # The process CWD may have been deleted by the agent's own bash actions during
            # evolution. test-eval and its downstream (harness loader abspath→getcwd, log
            # directories, benchmark env) all depend on CWD, and break in waves if it is
            # gone. Ensure CWD is valid before test-eval: if invalid, switch back to the
            # absolute startup CWD, or the script directory as a last resort. Restoring
            # CWD is the root fix (makes all downstream getcwd calls work again); resolving
            # code_dir to absolute below is a symptomatic fallback.
            try:
                os.getcwd()
            except FileNotFoundError:
                for _fallback in (str(_startup_cwd), str(_script_dir)):
                    try:
                        os.chdir(_fallback)
                        print(f"  [warn] Process CWD is invalid, switched to: {_fallback}", flush=True)
                        break
                    except OSError:
                        continue

            eval_label = "ALL tasks (dev+val+test)" if benchmark_config.run_all_tasks else "test set"
            test_repeats = max(1, int(bench_cfg.get('test_repeats', 5)))
            code_dir = result.get('exported_path')
            # exported_path is a relative path (./evolution_results/...). Even after
            # restoring CWD above, resolve it to absolute (anchored to startup CWD,
            # never calling getcwd) so HarnessLoader does not depend on a live CWD —
            # this is the root-cause fix for harness loader crashes (abspath→getcwd).
            if code_dir and not os.path.isabs(code_dir):
                code_dir = os.path.normpath(os.path.join(str(_startup_cwd), code_dir))

            print("\n" + "-" * 60)
            print(f"Running {eval_label} evaluation ({test_repeats} repeats)...")
            print("-" * 60)

            run_scalars = []
            run_raw_rewards = []
            run_task_counts = []
            skipped_reasons = []

            # Test is OFF the self-evolution loop: main.py aggregates reward only,
            # so per-episode LLM diary summaries would burn tokens on text nobody
            # consumes. Scoped override — flip evaluate_llm_summary off around the
            # test loop and restore it in finally. The balrog/agentdojo evaluators
            # read this off the shared agent.config; balrog uses ThreadPoolExecutor,
            # so the flip is visible to every worker thread.
            _saved_eval_summary = agent.config.evaluate_llm_summary
            agent.config.evaluate_llm_summary = False
            try:
                for run_i in range(test_repeats):
                    run_num = run_i + 1
                    if test_repeats > 1:
                        print(f"\n>>> {eval_label} run {run_num}/{test_repeats}")
                    reward, metrics = benchmark_evaluator.evaluate_test_set(
                        agent, code_dir=code_dir, repeat_idx=run_i,
                    )
                    if reward is None:
                        reason = metrics.get("message", metrics.get("error", "unknown"))
                        skipped_reasons.append(f"run {run_num}: {reason}")
                        print(f"  run {run_num} skipped: {reason}")
                        continue
                    run_raw_rewards.append(reward)
                    run_task_counts.append(metrics.get('test_task_count'))
                    scalar = reward_to_scalar(reward)
                    run_scalars.append(scalar)
                    if isinstance(reward, dict):
                        parts = [f"{k}={v:.4f}" for k, v in reward.items()
                                 if isinstance(v, (int, float))]
                        detail = ", ".join(parts) if parts else "n/a"
                        print(f"  run {run_num} reward: {detail} (scalar={scalar:.4f})")
                    else:
                        print(f"  run {run_num} reward: {scalar:.4f}")
            finally:
                agent.config.evaluate_llm_summary = _saved_eval_summary

            print("\n" + "-" * 60)
            print(f"{eval_label} aggregate results ({len(run_scalars)}/{test_repeats} successful runs)")
            print("-" * 60)

            if not run_scalars:
                reason = skipped_reasons[-1] if skipped_reasons else "unknown"
                print(f"  {eval_label} evaluation skipped/failed: {reason}")
            else:
                agg = _summarize_reward_runs(run_scalars)
                print(f"  reward mean:     {agg['mean']:.4f}")
                print(f"  reward std:      {agg['std']:.4f}")
                print(f"  reward variance: {agg['variance']:.4f}")
                print(f"  reward min/max:  {agg['min']:.4f} / {agg['max']:.4f}")
                tc = run_task_counts[0] if run_task_counts else "?"
                print(f"  tasks per run:   {tc}")

                # Per-category breakdown when any run returns a multi-category dict reward
                cat_keys = []
                seen = set()
                for r in run_raw_rewards:
                    if isinstance(r, dict):
                        for k, v in r.items():
                            if isinstance(v, (int, float)) and k != "scalar_reward" and k not in seen:
                                seen.add(k)
                                cat_keys.append(k)
                per_category = {}
                if cat_keys:
                    print("  per-category:")
                    for k in cat_keys:
                        vals = [float(r[k]) for r in run_raw_rewards
                                if isinstance(r, dict) and isinstance(r.get(k), (int, float))]
                        ca = _summarize_reward_runs(vals)
                        per_category[k] = ca
                        print(f"    {k}: mean={ca['mean']:.4f} std={ca['std']:.4f} (n={ca['n']})")

                # Persist a compact summary for later analysis
                try:
                    summary_path = os.path.join(run_output_dir, "test_repeat_results.json")
                    summary = {
                        "eval_label": eval_label,
                        "test_repeats": test_repeats,
                        "successful_runs": len(run_scalars),
                        "scalar_rewards": run_scalars,
                        "statistics": agg,
                        "per_category": per_category,
                        "skipped": skipped_reasons,
                    }
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)
                    print(f"  saved: {summary_path}")
                except Exception as save_err:
                    print(f"  (failed to save test summary: {save_err})")

    except Exception as e:
        print(f"\nRuntime error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'agent' in locals() and agent is not None:
            agent.print_usage_report()

        if benchmark_evaluator and benchmark_config.type == "terminal_bench":
            from benchmark.terminal_bench.harbor_runner import cleanup_harbor_docker_resources
            print("\nCleaning up Harbor Docker resources...")
            cleanup_harbor_docker_resources()


if __name__ == "__main__":
    main()
