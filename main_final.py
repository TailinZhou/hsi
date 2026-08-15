#!/usr/bin/env python3
"""
Final Evaluation — Evolved Harness (Post-Evolution)

Evaluates the best evolved agent from a completed evolution run.

Usage:
    python main_final.py                                         # use default config.yaml
    python main_final.py benchmark_config_goal/balrog_babyai/config.yaml
    python main_final.py evolution_results/balrog_babyai/run_20260511_231336
    python main_final.py evolution_results/balrog_babyai/run_20260511_231336/agent_code_best_20260511_231336
"""

import os
import sys
import glob
import json
from pathlib import Path
from datetime import datetime

# Windows Unicode fix
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import load_config, load_secrets, create_llm_client, build_agent_config, _summarize_reward_runs
from react_loop.agent import GodelAgent, GodelAgentConfig
from react_loop.state import reward_to_scalar
from benchmark.config import BenchmarkConfig
from benchmark.adapter import create_benchmark_evaluator

ROOT_DIR = Path(__file__).parent


def resolve_run_path(raw_path: str) -> tuple[str, str]:
    """Resolve input path to (run_dir, code_dir).

    Accepts:
      - run dir:       evolution_results/balrog/run_20260511_231336
      - code dir:      evolution_results/balrog/run_20260511_231336/agent_code_best_20260511_231336
      - benchmark dir: evolution_results/balrog  (auto-pick latest run)
    Returns (run_dir, code_dir).
    """
    p = Path(raw_path).resolve()

    # Case 1: already a agent_code_best_* directory
    if p.name.startswith("agent_code_best_"):
        return str(p.parent), str(p)

    # Case 2: a run_* directory
    if p.name.startswith("run_"):
        candidates = sorted(p.glob("agent_code_best_*"), key=lambda x: x.name, reverse=True)
        if not candidates:
            print(f"Error: no agent_code_best_* found in {p}")
            sys.exit(1)
        return str(p), str(candidates[0])

    # Case 3: benchmark output dir — pick latest run
    candidates = sorted(p.glob("run_*"), key=lambda x: x.name, reverse=True)
    if not candidates:
        print(f"Error: no run_* found in {p}")
        sys.exit(1)
    run_dir = candidates[0]
    code_candidates = sorted(run_dir.glob("agent_code_best_*"), key=lambda x: x.name, reverse=True)
    if not code_candidates:
        print(f"Error: no agent_code_best_* found in {run_dir}")
        sys.exit(1)
    return str(run_dir), str(code_candidates[0])


def resolve_from_config(cfg: dict, config_path: str | None = None) -> tuple[str, str]:
    """Resolve (run_dir, code_dir) from a config dict.

    If config_path is inside a run_* directory, use that directly.
    Otherwise, fall back to output.directory and pick the latest run.
    """
    # If config is inside a run_* directory, use it directly
    if config_path:
        config_parent = Path(config_path).resolve().parent
        if config_parent.name.startswith("run_"):
            run_dir, code_dir = resolve_run_path(str(config_parent))
            return str(run_dir), code_dir

    bench_type = cfg.get('benchmark', {}).get('type')
    if not bench_type:
        print("Error: no benchmark.type in config")
        sys.exit(1)

    output_dir = cfg.get('output', {}).get('directory', '')
    if not output_dir:
        print("Error: no output.directory in config")
        sys.exit(1)

    bench_output = Path(output_dir)
    if not bench_output.exists():
        print(f"Error: no evolution results found at {bench_output}")
        sys.exit(1)

    run_dir, code_dir = resolve_run_path(str(bench_output))
    return str(run_dir), code_dir


def main():
    W = 60
    print("=" * W)
    print("Final Evaluation (Evolved Harness)")
    print("=" * W)

    # 1. Resolve paths
    raw_path = sys.argv[1] if len(sys.argv) > 1 else None

    if raw_path is None:
        # No argument — use default config.yaml
        config_path = str(Path("config.yaml").resolve())
        cfg = load_config()
        run_dir, code_dir = resolve_from_config(cfg, config_path)
    elif raw_path.endswith(('.yaml', '.yml')):
        # Config yaml path — same style as main_init.py
        config_path = str(Path(raw_path).resolve())
        cfg = load_config(raw_path)
        run_dir, code_dir = resolve_from_config(cfg, config_path)
    else:
        # Direct path to run_dir / code_dir / benchmark output dir
        run_dir, code_dir = resolve_run_path(raw_path)
        config_path = os.path.join(run_dir, "config.yaml")
        cfg = load_config(config_path)

    if not os.path.exists(config_path):
        print(f"Error: config.yaml not found at {config_path}")
        sys.exit(1)

    print(f"Run dir:    {run_dir}")
    print(f"Best code:  {code_dir}")
    print(f"Config:     {config_path}")

    # 2. Load secrets + build agent config
    secrets = load_secrets()
    agent_config = build_agent_config(cfg)
    client = create_llm_client(secrets)

    print(f"Model: {agent_config.model}")

    # 3. Create benchmark evaluator (force run_all_tasks=True)
    bench_cfg = cfg.get('benchmark', {})
    bench_type = bench_cfg.get('type')
    if not bench_type:
        print("Error: no benchmark.type in config")
        sys.exit(1)

    benchmark_config = BenchmarkConfig(
        type=bench_type,
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
    )
    benchmark_config._raw_yaml = cfg
    benchmark_evaluator = create_benchmark_evaluator(benchmark_config)

    print(f"Benchmark: {bench_type} ({bench_cfg.get('suite', 'N/A')})")

    # 4. Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./final_results/{bench_type}_{timestamp}"
    agent_config.output_dir = output_dir

    os.makedirs(output_dir, exist_ok=True)
    import shutil
    shutil.copy2(config_path, os.path.join(output_dir, "config.yaml"))

    print(f"Output: {output_dir}")

    # 5. Create GodelAgent — use evolved code as init so agent_code_dir
    #    and action_executor.agent_codes match the post-evolution state in main.py
    agent_config.godel_harness_init_path = code_dir
    agent = GodelAgent(
        repo_path=f"{output_dir}/repo",
        llm_client=client,
        goal="Final evaluation (evolved harness)",
        config=agent_config,
        external_evaluator=None,
    )

    # 6. Run evaluation using the evolved harness —— 重复多次以降低 reward 波动（对齐 main.py 结束后跑 test 的设置）
    eval_label = "ALL tasks (dev+val+test)" if bench_cfg.get('run_all_tasks', False) else "test set"
    test_repeats = max(1, int(bench_cfg.get('test_repeats', 5)))

    print()
    print("-" * W)
    print(f"Running {eval_label} evaluation with evolved harness ({test_repeats} repeats)...")
    print("-" * W)

    run_scalars = []
    run_raw_rewards = []
    run_task_counts = []
    skipped_reasons = []

    for run_i in range(test_repeats):
        run_num = run_i + 1
        if test_repeats > 1:
            print(f"\n>>> {eval_label} run {run_num}/{test_repeats}")
        # repeat_idx rotates the deterministic test seed across test_repeats
        # passes → each pass is a different map set (mirrors main.py).
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

    # 7. Output results
    print()
    print("=" * W)
    print(f"{eval_label} aggregate results ({len(run_scalars)}/{test_repeats} successful runs)")
    print("=" * W)

    agg = None
    per_category = {}
    mean_reward = None
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
        mean_reward = agg['mean']

        # Per-category breakdown when any run returns a multi-category dict reward
        cat_keys = []
        seen = set()
        for r in run_raw_rewards:
            if isinstance(r, dict):
                for k, v in r.items():
                    if isinstance(v, (int, float)) and k != "scalar_reward" and k not in seen:
                        seen.add(k)
                        cat_keys.append(k)
        if cat_keys:
            print("  per-category:")
            for k in cat_keys:
                vals = [float(r[k]) for r in run_raw_rewards
                        if isinstance(r, dict) and isinstance(r.get(k), (int, float))]
                ca = _summarize_reward_runs(vals)
                per_category[k] = ca
                print(f"    {k}: mean={ca['mean']:.4f} std={ca['std']:.4f} (n={ca['n']})")

    # Persist repeat summary (对齐 main.py 的 test_repeat_results.json)
    try:
        summary = {
            "eval_label": eval_label,
            "test_repeats": test_repeats,
            "successful_runs": len(run_scalars),
            "scalar_rewards": run_scalars,
            "statistics": agg,
            "per_category": per_category,
            "skipped": skipped_reasons,
        }
        summary_path = os.path.join(output_dir, "test_repeat_results.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  saved: {summary_path}")
    except Exception as save_err:
        print(f"  (failed to save test summary: {save_err})")

    # Save results to JSON
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "benchmark": bench_type,
        "suite": bench_cfg.get('suite'),
        "model": agent_config.model,
        "evolved_harness": code_dir,
        "source_run": run_dir,
        "reward": mean_reward,
        "test_repeats": test_repeats,
        "statistics": agg,
        "per_category": per_category,
        "raw_rewards": run_raw_rewards,
        "timestamp": timestamp,
    }
    results_path = os.path.join(output_dir, "final_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_dir}/")
    print("=" * W)

    # Cleanup Harbor Docker resources
    if bench_type == "terminal_bench":
        from benchmark.terminal_bench.harbor_runner import cleanup_harbor_docker_resources
        print("\nCleaning up Harbor Docker resources...")
        cleanup_harbor_docker_resources()


if __name__ == "__main__":
    main()
