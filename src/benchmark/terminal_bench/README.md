# Terminal-Bench 2 Integration

Terminal-Bench 2 (TB2) contains 89 real-world terminal tasks spanning system administration, software engineering, security, machine learning, and more. This module integrates it into the GodelHarness benchmark framework.

## Architecture

```
GodelAgent calls evaluate
  → TerminalBenchAdapter.__call__()
    → HarborRunner.run() invokes the harbor CLI
      → Harbor starts GodelAgentOnHarbor in a Runloop sandbox
        → GodelAgentProxy (litellm) calls the LLM, bash is routed to tmux
        → using_harness(proxy, instruction) executes the harness strategy
      → Harbor verifier decides pass/fail
    → parses result.json
  → _calculate_reward() → reward = passed / total
```

Core design: the harness source is decoupled from the agent via the `GodelHarness` Protocol. During local evolution, the `agent` is a GodelAgent (in-process); during TB2 evaluation, the `agent` is a GodelAgentProxy (inside the Harbor subprocess, calling the LLM via litellm). The harness code is the same single copy and does not care which implementation is behind it.

## File overview

| File | Responsibility |
|------|----------------|
| `adapter.py` | `TerminalBenchAdapter` — reward = passed/total, log saving |
| `evaluator.py` | `TerminalBenchEvaluator(BaseTaskEvaluator)` — task loading + Harbor batch execution |
| `harbor_runner.py` | `HarborRunner` — wraps the `harbor run` CLI, temporary directory management |
| `bridge_agent_template.py` | Dynamically generates `GodelAgentOnHarbor` + `GodelAgentProxy` |
| `config.py` | `TerminalBenchConfig` — Harbor paths, litellm model, timeouts, etc. |
| `tasks.py` | The 89 task_ids and their category definitions |
| `result_parser.py` | Parses Harbor's result.json output |
| `log_summary.py` | `TerminalBenchLogSummary` — evaluation summary generation |
| `requirements.txt` | Optional dependencies: `litellm`, `harbor-ai` |

## Install dependencies

```bash
pip install -r requirements.txt
```

If not installed, the registry will skip it gracefully and other benchmarks are unaffected.

## Configuration

In `config.yaml`:

```yaml
benchmark:
  type: terminal_bench
  suite: all            # all | comma-separated categories or task_ids
  max_tasks_per_category: 3
  dev_ratio: 1.0
```

Environment variables:

| Variable | Description |
|----------|-------------|
| `TB2_LITELLM_MODEL` | The litellm model used by GodelAgentProxy, defaults to `openai/gpt-4o` |
| `HARBOR_EXECUTABLE` | Path to the Harbor CLI, auto-detected by default |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | LLM API key, propagated to the subprocess |

## The 89 task categories

| Category | Count | Examples |
|----------|-------|----------|
| Software Engineering | 22 | build-cython-ext, compile-compcert, make-mips-interpreter |
| Machine Learning & AI | 14 | caffe-cifar-10, hf-model-inference, train-fasttext |
| Data Science | 13 | count-dataset-tokens, query-optimize, portfolio-optimization |
| Scientific Computing | 13 | dna-assembly, protein-assembly, mcmc-sampling-stan |
| System Administration | 12 | configure-git-webserver, nginx-request-logging, qemu-startup |
| Security | 10 | fix-code-vulnerability, crack-7z-hash, git-leak-recovery |
| Debugging | 5 | fix-git, custom-memory-heap-crash, pytorch-model-recovery |

## How to run

### Single-task smoke test

```bash
harbor run --dataset terminal-bench@2.0 \
   --agent-import-path bridge_agent:GodelAgentOnHarbor \
   -i extract-elf
```

### Full evaluation through GodelAgent

```bash
python main.py  # set type: terminal_bench in config.yaml
```

## Related files

- Initial harness: `godel_harness_init/terminal_bench/`
- Evolution goal: `benchmark_config_goal/terminal_bench/goal.md`
- GodelHarness Protocol: `src/react_loop/agent_protocol.py`
