"""UsageLogger - records the token usage of every LLM call, for post-hoc review of API usage.

Design notes
--------
- **Single recording point**: the caller (the agent.call_llm dispatcher) calls ``record()``
  after every LLM call, passing the usage dict extracted from the response + meta info
  (scope / iteration / model / thinking).
- **Hot-path optimization**: ``call_llm`` is invoked thousands of times per run (including
  ~40 concurrent evaluation threads), so ``record()`` reuses a single always-open append
  handle (line-buffered) and only does one ``write`` inside the lock, avoiding the
  system-call overhead of ``open/close`` on every call and the in-lock ``json.dumps``.
- **Aggregation always re-reads from JSONL**: ``summarize()`` re-parses the whole
  ``usage_log.jsonl`` to aggregate, rather than relying on memory - so it is accurate
  whether called at the end of ``evolve()`` or at process exit (including test evaluations
  outside of evolve).
- **atexit fallback**: the process automatically calls ``summarize()`` at exit and closes
  the handle, ensuring a final summary is available even on abnormal exit / when no
  explicit call is made.

Field compatibility
--------
- DeepSeek style: ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``
- OpenAI style: ``usage.prompt_tokens_details.cached_tokens`` /
  ``usage.completion_tokens_details.reasoning_tokens``
``extract_usage()`` uses ``getattr`` defensively for both styles; missing fields are
recorded as 0 (consistent across aggregation).
"""

import json
import os
import threading
import atexit
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


def _val(x: Any) -> int:
    """None / missing -> 0; otherwise coerce to int (ignore unparseable values)."""
    if x is None:
        return 0
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def _get(obj: Any, key: str) -> Any:
    """Get a (top-level) field from a dict or object."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def extract_usage(response: Any) -> Optional[Dict[str, Any]]:
    """Robustly extract the usage field from an LLM response object.

    Compatible with both DeepSeek and OpenAI field styles; for non-standard objects
    returned on the override path, ``getattr`` is fully defensive, returning None when
    no usage can be obtained.
    """
    u = _get(response, "usage")
    if u is None:
        return None

    prompt = _val(_get(u, "prompt_tokens"))
    completion = _val(_get(u, "completion_tokens"))
    total = _val(_get(u, "total_tokens"))

    # cache hit: DeepSeek direct field takes priority, fall back to OpenAI prompt_tokens_details.cached_tokens
    cache_hit = _get(u, "prompt_cache_hit_tokens")
    cache_miss = _get(u, "prompt_cache_miss_tokens")
    if cache_hit is None:
        cache_hit = _get(_get(u, "prompt_tokens_details"), "cached_tokens")

    # reasoning tokens: OpenAI completion_tokens_details.reasoning_tokens
    reasoning = _get(_get(u, "completion_tokens_details"), "reasoning_tokens")

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cache_hit_tokens": _val(cache_hit),
        "cache_miss_tokens": _val(cache_miss),
        "reasoning_tokens": _val(reasoning),
    }


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file as a list of dicts, skipping blank lines and lines that fail to parse (pure function, reused in several places)."""
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _new_acc() -> Dict[str, int]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cache_hit_tokens": 0,
        "cache_miss_tokens": 0,
        "reasoning_tokens": 0,
    }


def _add(acc: Dict[str, int], r: Dict[str, Any]) -> None:
    acc["calls"] += 1
    acc["prompt_tokens"] += _val(r.get("prompt_tokens"))
    acc["completion_tokens"] += _val(r.get("completion_tokens"))
    acc["total_tokens"] += _val(r.get("total_tokens"))
    acc["cache_hit_tokens"] += _val(r.get("cache_hit_tokens"))
    acc["cache_miss_tokens"] += _val(r.get("cache_miss_tokens"))
    acc["reasoning_tokens"] += _val(r.get("reasoning_tokens"))


def _finalize(acc: Dict[str, int]) -> Dict[str, Any]:
    """Compute two cache-hit-rate conventions (fill fields in place and return)."""
    hit, miss = acc["cache_hit_tokens"], acc["cache_miss_tokens"]
    prompt = acc["prompt_tokens"]
    acc["cache_hit_rate"] = round(hit / (hit + miss), 4) if (hit + miss) > 0 else 0.0
    # OpenAI convention: cached_tokens is part of prompt
    acc["cache_hit_of_prompt"] = round(hit / prompt, 4) if prompt > 0 else 0.0
    return acc


def aggregate_usage(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a list of raw usage records -> {total, by_scope, by_iteration, by_scope_model}.

    Pure function, neither reads nor writes files; shared by ``UsageLogger.summarize()``
    and ``print_usage_report()`` to keep the "recompute from log" convention consistent.
    """
    total = _new_acc()
    by_scope: Dict[str, Dict[str, int]] = {}
    by_iteration: Dict[str, Dict[str, int]] = {}
    by_scope_model: Dict[str, Dict[str, int]] = {}
    for r in records:
        _add(total, r)
        scope = r.get("scope") or "unknown"
        _add(by_scope.setdefault(scope, _new_acc()), r)
        ikey = str(r.get("iteration")) if r.get("iteration") is not None else "n/a"
        _add(by_iteration.setdefault(ikey, _new_acc()), r)
        sm = f"{scope}/{r.get('model') or '?'}"
        _add(by_scope_model.setdefault(sm, _new_acc()), r)
    return {
        "total": _finalize(total),
        "by_scope": {k: _finalize(v) for k, v in by_scope.items()},
        "by_iteration": {k: _finalize(v) for k, v in by_iteration.items()},
        "by_scope_model": {k: _finalize(v) for k, v in by_scope_model.items()},
    }


class UsageLogger:
    """Thread-safely record the usage of every LLM call, and provide a summary recomputed from the log."""

    def __init__(
        self,
        log_path: str,
        summary_path: str,
        log: Optional[Callable] = None,
    ) -> None:
        self.log_path = log_path
        self.summary_path = summary_path
        self._log = log or (lambda m: None)
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        # Reuse a single append handle (line-buffered) for the entire run, avoiding open/close on every record() on the hot path
        self._fh = open(log_path, "a", encoding="utf-8", buffering=1)
        atexit.register(self._safe_finalize)

    def record(self, entry: Dict[str, Any]) -> None:
        """Append a usage record (thread-safe). entry should contain the six usage fields + meta info."""
        line = json.dumps(entry, ensure_ascii=False, default=str)  # dumps outside the lock
        with self._lock:
            self._fh.write(line + "\n")

    def summarize(self) -> Dict[str, Any]:
        """Re-read all records from usage_log.jsonl, aggregate, write usage_summary.json, and return it."""
        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "log_path": self.log_path,
            **aggregate_usage(read_jsonl(self.log_path)),
        }
        with self._lock:
            with open(self.summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        self._log(
            f"[usage] summary → {self.summary_path} "
            f"(calls={summary['total']['calls']}, "
            f"cache_hit_rate={summary['total']['cache_hit_rate']:.1%})"
        )
        return summary

    def _safe_finalize(self) -> None:
        """atexit fallback: write the final summary and close the handle at process exit."""
        try:
            if os.path.exists(self.log_path):
                self.summarize()
        except Exception:
            pass
        finally:
            try:
                self._fh.close()
            except Exception:
                pass


# ─── Human-readable report (printed by main.py after evolution / also usable for historical runs) ──────────

def format_usage_report(summary: Dict[str, Any]) -> str:
    """Format the dict produced by aggregate_usage / summarize into a terminal-friendly multi-line report."""
    t = summary.get("total") or _new_acc()
    L = ["=" * 64, "LLM Usage Report", "=" * 64]
    L.append(f"Total calls : {t.get('calls', 0):,}")
    L.append(f"Tokens      : prompt {t.get('prompt_tokens', 0):,} | "
             f"completion {t.get('completion_tokens', 0):,} | "
             f"total {t.get('total_tokens', 0):,}")
    if t.get("reasoning_tokens"):
        L.append(f"              reasoning(thinking) {t['reasoning_tokens']:,}")
    L.append(f"Cache       : hit {t.get('cache_hit_tokens', 0):,} | "
             f"miss {t.get('cache_miss_tokens', 0):,} | "
             f"hit_rate {t.get('cache_hit_rate', 0):.1%} "
             f"(of prompt {t.get('cache_hit_of_prompt', 0):.1%})")

    by_scope = summary.get("by_scope", {})
    if by_scope:
        L.append("")
        L.append(f"{'scope':<10}{'calls':>8}{'prompt':>13}{'compl':>11}"
                 f"{'cache_hit':>13}{'hit_rate':>10}")
        L.append("-" * 64)
        order = ["evolve", "harness", "meta"]
        keys = order + [k for k in by_scope if k not in order]
        for sc in keys:
            if sc not in by_scope:
                # A scope in `order` may not exist in this run (e.g. "meta" is missing when nometa / no meta calls)
                continue
            r = by_scope[sc]
            L.append(f"{sc:<10}{r['calls']:>8,}{r['prompt_tokens']:>13,}"
                     f"{r['completion_tokens']:>11,}{r['cache_hit_tokens']:>13,}"
                     f"{r['cache_hit_rate']:>10.1%}")

    by_it = summary.get("by_iteration", {})
    if by_it:
        L.append("")
        L.append("By iteration:")

        def _itkey(k: str):
            try:
                return (0, int(k))
            except ValueError:
                return (1, k)

        for k in sorted(by_it, key=_itkey):
            r = by_it[k]
            L.append(f"  iter {k:>3}: {r['calls']:>5} calls | hit_rate {r['cache_hit_rate']:.1%} "
                     f"| prompt {r['prompt_tokens']:,}")

    by_sm = summary.get("by_scope_model", {})
    if len(by_sm) > 1:
        L.append("")
        L.append("By scope/model:")
        for k in sorted(by_sm):
            r = by_sm[k]
            L.append(f"  {k:<28} {r['calls']:>5} calls | hit_rate {r['cache_hit_rate']:.1%}")

    L.append("=" * 64)
    return "\n".join(L)


def print_usage_report(run_dir: str) -> None:
    """Read ``run_dir/usage_log.jsonl`` to recompute and print the report; also usable for historical runs.

    Prefer recompute from jsonl (most up-to-date); fall back to usage_summary.json when
    jsonl is absent; prompt no-data when neither exists or there are no records. Does
    not instantiate a UsageLogger, avoiding duplicate atexit registration.
    """
    log_path = os.path.join(run_dir, "usage_log.jsonl")
    summ_path = os.path.join(run_dir, "usage_summary.json")
    summary: Optional[Dict[str, Any]] = None
    if os.path.exists(log_path):
        summary = aggregate_usage(read_jsonl(log_path))
    elif os.path.exists(summ_path):
        with open(summ_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    if not summary or not (summary.get("total") or {}).get("calls"):
        print("\n(LLM usage: no records)")
        return
    print("\n" + format_usage_report(summary))
    print(f"\nDetails: {log_path}")
    print(f"Summary: {summ_path}")
