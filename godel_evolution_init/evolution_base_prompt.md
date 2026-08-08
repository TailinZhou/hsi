## Designing Your Policy

`harness.py` *is* your policy — its decision loop is a design choice, not a given. The naive loop (one `react()`, pass through the answer) is the floor, not the ceiling. Your biggest leverage is **reshaping the loop itself**, not tuning its wording or bolting on overrides.

**The loop is a design space.** A single `react()` is one LLM step; how you compose it is the policy:
- **Single-step** — one `react()`, pass through. Weak: the LLM acts with no structure.
- **Multi-stage** — compose several `react()` calls into a pipeline (e.g. *analyze → plan → act*). Build stages in `harness.py`, prompts in `prompts.py`, per-stage context in `hooks.py`, and inter-stage state in `context.py`. **The composition IS the policy.**
- **Structured control flow** — branching, loop/stuck detection with recovery, memory across calls, state machines for deterministic sub-problems.
- **Hybrid — code + LLM** — not every step needs `call_llm()`. Algorithmic decisions (graph search, systematic exploration) are cheaper and more reliable as code — compute them in `workflow()` and return without the LLM. Use the LLM for semantic decisions (what to interact with), code for algorithmic ones (where to go next).

**Improvement comes from adding structure, not tuning text.** A longer prompt squeezes marginal gains from the same loop; a new stage, a state machine, a recovery path, or replacing LLM calls with code for algorithmic sub-problems changes what the loop can do. When stuck, ask: what capability does the policy lack? Where would its code live — a stage in `harness.py`, a data structure in `context.py`, a validator in `utils.py`? Could this be done without the LLM?

**Build shared capabilities.** Code that provides a general capability (e.g. tracking state the LLM can't reliably remember, making decisions the LLM is inconsistent at, processing patterns that repeat across tasks) lifts the whole benchmark. When adding code, ask: does this encode a task solution, or a capability other tasks can use too? Prefer the latter. **When no shared approach fits, per-task branching is a valid fallback — it's better to solve tasks differently than to solve none of them well.**

## The Evolution Loop: Hypothesis → Verify → Learn

An iteration is a **hypothesis-driven search** across multiple experiment cycles. You don't form one hypothesis and stick to it — each cycle's verdict feeds back into the next, like a scientist adjusting their theory as experiments come in. Reward is too noisy to judge a single edit; the real signal is the failure traces.

1. **Orient** — digest the seed hypothesis and `BOOTSTRAP.md` lessons. Don't guess from scratch.
2. **Hypothesize** — identify the most impactful failure mode. Form a falsifiable prediction: what root cause, what change, what effect? Record via `plan(plan=...)`.
3. **Test** — one targeted edit → review → evaluate. **Read the failure traces, not the score.** Reward jitters inside a noise band; a single evaluate can't tell you whether the edit helped. Verify: did the failure mode disappear from the failing tasks? That's your verdict. **One hypothesis per cycle** — bundling unrelated changes tells you nothing about which worked.
4. **Update the hypothesis** — the verdict drives the next cycle: supported → deepen; partially supported → refine; refuted → pivot to a new root cause. The hypothesis sharpens with each cycle.
5. **Record & bookmark** along the way — `plan(progress=...)` for progress, `pick_commit_version` for breakthroughs, `lesson(...)` for cross-iteration memory.
6. **Repeat or consolidate** — the updated hypothesis starts the next cycle. When you've made meaningful progress or exhausted productive directions, consolidate and end. A healthy iteration runs 2–5 evaluate calls; past 7 is debugger territory — consolidate and compact. The final `lesson(...)` captures the consolidated verdict.

