Self-evolve your harness strategy to maximize the pass rate on Terminal-Bench 2.

## Tasks

Terminal-Bench 2 (TB2) evaluates agents on 89 realistic terminal tasks across 7 categories:
- **System Administration**: Server configuration, git management, SSH setup, container operations
- **Software Engineering**: Building compilers, extensions, interpreters, renderers
- **Data Science**: Token counting, query optimization, data merging, scheduling
- **Security**: Cryptanalysis, vulnerability fixing, hash cracking, git leak recovery
- **Machine Learning & AI**: Model training, inference, batch scheduling, segmentation
- **Debugging**: Fixing async code, memory heaps, OCaml GC, PyTorch model recovery
- **Scientific Computing**: Protein assembly, DNA processing, circuit design, MCMC sampling

Each task runs in a **sandboxed terminal environment** (Harbor/Runloop). Your harness must complete tasks by executing bash commands interactively.

## Task Rules

Your harness is called once per task with the task instruction. A sandbox container is created with the specific setup (code, data, tools) for that task.

The harness runs a multi-turn react loop (up to 30 iterations) using one tool:

- **bash** — Execute commands in the sandboxed terminal

When the LLM outputs text without tool calls → harness returns as final answer.

After your harness completes, Harbor's **verifier** checks whether the task was completed correctly. The result is binary: **passed** (reward = 1) or **failed** (reward = 0).

Scoring: **passed_count / total_count** (0.0 to 1.0)

Note: Your harness code runs inside a Harbor subprocess (Runloop sandbox), not in the main GodelAgent process. LLM calls go through litellm independently.

## Your Strategy Files

All files in the current folder are yours to modify freely — evolve any combination to build your best strategy:

- **harness.py** — Entry point. `using_harness()` is FIXED. `HarnessPolicy.execute()` is your EVOLVABLE core.
- **prompts.py** — System prompt for the terminal-task LLM.
- **hooks.py** — Hooks to intercept and transform messages before/after LLM calls.
- **tools_harness.py** — Custom tools the LLM can call.
- **context.py** — Conversation history and state management.
