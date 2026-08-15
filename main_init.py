#!/usr/bin/env python3
"""
Baseline Evaluation — Initial Harness (No Evolution)

Usage:
    python main_init.py                                    # use default config.yaml
    python main_init.py benchmark_config_goal/agentdojo/config.yaml
"""

import os
import sys
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


def main():
    # ── banner ────────────────────────────────────────────────
    W = 60
    print("=" * W)
    print("Baseline Evaluation (Initial Harness, No Evolution)")
    print("=" * W)

    # 1. Load config + secrets (reuse main.py helpers)
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(config_path)
    secrets = load_secrets()
    agent_config = build_agent_config(cfg)
    client = create_llm_client(secrets)

    print(f"Model: {agent_config.model}")
    print(f"Initial harness: {agent_config.godel_harness_init_path}")

    # 2. Create benchmark evaluator (force run_all_tasks=True)
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

    # 3. Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./baseline_results/{bench_type}_{timestamp}"
    agent_config.output_dir = output_dir

    os.makedirs(output_dir, exist_ok=True)
    import shutil
    shutil.copy2(config_path, os.path.join(output_dir, "config.yaml"))

    print(f"Output: {output_dir}")

    # 4. Create GodelAgent (copies initial harness into repo_path)
    agent = GodelAgent(
        repo_path=f"{output_dir}/repo",
        llm_client=client,
        goal="Baseline evaluation (no evolution)",
        config=agent_config,
        external_evaluator=None,
    )

    # 5. Run baseline evaluation —— 重复多次以降低 reward 波动（对齐 main.py 结束后跑 test 的设置）
    eval_label = "ALL tasks (dev+val+test)" if bench_cfg.get('run_all_tasks', False) else "test set"
    test_repeats = max(1, int(bench_cfg.get('test_repeats', 5)))

    print()
    print("-" * W)
    print(f"Running {eval_label} evaluation ({test_repeats} repeats)...")
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
        # passes → each pass is a different map set (mirrors main.py). Without
        # it every repeat uses repeat_idx=0 and yields identical maps/rewards.
        reward, metrics = benchmark_evaluator.evaluate_test_set(
            agent, code_dir=agent.agent_code_dir, repeat_idx=run_i,
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

    # 6. Output results
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
        "harness_init": agent_config.godel_harness_init_path,
        "reward": mean_reward,
        "test_repeats": test_repeats,
        "statistics": agg,
        "per_category": per_category,
        "raw_rewards": run_raw_rewards,
        "timestamp": timestamp,
    }
    results_path = os.path.join(output_dir, "baseline_results.json")
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
