"""Regression test: every distinct evaluated code state is persisted to disk.

``EvaluateMixin._persist_harness_history`` writes a copy of the harness files
that ran at each ``evaluate()`` call to
``.evolution_context/main_evolve/harness_history/iter_<it>/eval_<seq>_<hash>/``.
This is the on-disk fallback for ``state.code_snapshots`` (which is in-memory
only and never persisted) — it's how a non-committed version can be recovered
after a run ends (cf. the iter-5 ``98ca03d3f3f0`` / 0.4157 version that was lost).
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.react_loop.actions.agent_evaluator import EvaluatorMixin  # noqa: E402


class _FakeExec(EvaluatorMixin):
    def __init__(self, code_dir):
        self.agent_code_dir = code_dir
        self._logs = []

    def logging(self, msg):
        self._logs.append(msg)


def _hist_dir(code_dir):
    return os.path.join(code_dir, ".evolution_context", "main_evolve", "harness_history")


def test_persists_each_distinct_hash_and_dedupes_consecutive():
    d = tempfile.mkdtemp(prefix="hh_")
    ex = _FakeExec(d)
    snap_a = {"harness.py": "BASELINE\n", "prompts.py": "P\n"}
    snap_b = {"harness.py": "BASELINE\nEDIT\n", "prompts.py": "P\n"}
    ha = "a" * 32
    hb = "b" * 32

    ex._persist_harness_history(5, 41, ha, snap_a)   # new hash -> write
    ex._persist_harness_history(5, 42, ha, snap_a)   # same hash -> skip
    ex._persist_harness_history(5, 43, hb, snap_b)   # new hash -> write

    iter5 = sorted(os.listdir(os.path.join(_hist_dir(d), "iter_5")))
    assert iter5 == ["eval_041_aaaaaaaaaaaa", "eval_043_bbbbbbbbbbbb"], iter5


def test_per_iteration_reset_records_repeated_hash():
    d = tempfile.mkdtemp(prefix="hh_")
    ex = _FakeExec(d)
    snap = {"harness.py": "X\n"}
    ha = "a" * 32

    ex._persist_harness_history(5, 41, ha, snap)
    ex._persist_harness_history(6, 44, ha, snap)   # same hash, NEW iteration -> write

    assert os.path.isdir(os.path.join(_hist_dir(d), "iter_5", "eval_041_aaaaaaaaaaaa"))
    assert os.path.isdir(os.path.join(_hist_dir(d), "iter_6", "eval_044_aaaaaaaaaaaa"))


def test_writes_snapshot_files_at_relative_paths():
    d = tempfile.mkdtemp(prefix="hh_")
    ex = _FakeExec(d)
    snap = {"harness.py": "BASELINE\nEDIT\n", "prompts.py": "PROMPT\n"}
    ex._persist_harness_history(5, 41, "c" * 32, snap)

    base = os.path.join(_hist_dir(d), "iter_5", "eval_041_cccccccccccc")
    assert open(os.path.join(base, "harness.py")).read() == "BASELINE\nEDIT\n"
    assert open(os.path.join(base, "prompts.py")).read() == "PROMPT\n"


def test_empty_hash_or_snapshot_is_noop():
    d = tempfile.mkdtemp(prefix="hh_")
    ex = _FakeExec(d)
    ex._persist_harness_history(5, 41, "", {"harness.py": "x"})
    ex._persist_harness_history(5, 42, "a" * 32, {})
    assert not os.path.exists(_hist_dir(d))
