"""Regression test: probe must not destroy the agent's uncommitted edits.

Background: ``run_probe`` ends with a defensive ``git reset --hard <pre_head>``
in its ``finally`` to clean up the probe sub-agent's bash. Bare ``--hard`` to
``pre_head`` also wipes the agent's OWN uncommitted working-tree edits that
existed before probe was called — a silent data-loss bug (the agent's
in-progress edit vanished the moment it probed, and since each eval is usually
followed by a probe, the agent re-applied the same edit every cycle and/or
mistook the wipe for "eval reverted my code").

The fix snapshots the evolvable harness file *contents* before the probe and
writes them back after the reset. This test exercises that snapshot/restore
pair directly against a real temp git repo.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add repo root to path for imports (matches the convention in the other tests).
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.react_loop.probe_agent import _snapshot_harness_files, _restore_harness_files  # noqa: E402


def _git(cwd, *args, env=None):
    e = {**os.environ, **(env or {})}
    return subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True, env=e)


def _make_repo():
    d = tempfile.mkdtemp(prefix="probe_iso_")
    genv = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    }
    with open(os.path.join(d, "harness.py"), "w") as f:
        f.write("BASELINE\n")
    with open(os.path.join(d, "BOOTSTRAP.md"), "w") as f:
        f.write("boot-base\n")
    _git(d, "init", "-q")
    _git(d, "add", "-A")
    _git(d, "commit", "-q", "-m", "init", env=genv)
    head = _git(d, "rev-parse", "HEAD").stdout.strip()
    return d, head


def test_uncommitted_edits_survive_probe_reset_hard():
    """The agent's in-progress edits must survive the probe's git reset --hard."""
    d, head = _make_repo()

    # 1) agent makes an UNCOMMITTED edit before calling probe
    with open(os.path.join(d, "harness.py"), "w") as f:
        f.write("BASELINE\nMY_UNCOMMITTED_EDIT\n")
    with open(os.path.join(d, "BOOTSTRAP.md"), "w") as f:
        f.write("boot-edit\n")

    # 2) run_probe captures the snapshot here (after the edit, before probe bash)
    pre_contents = _snapshot_harness_files(d)
    assert pre_contents["harness.py"] == "BASELINE\nMY_UNCOMMITTED_EDIT\n"
    assert pre_contents["BOOTSTRAP.md"] == "boot-edit\n"

    # 3) probe's bash dirties the tree (the original finally did reset --hard head)
    _git(d, "reset", "--hard", head)
    # OLD bug: edit is gone now
    assert "MY_UNCOMMITTED_EDIT" not in open(os.path.join(d, "harness.py")).read()
    assert open(os.path.join(d, "BOOTSTRAP.md")).read() == "boot-base\n"

    # 4) NEW fix: write the snapshot back
    _restore_harness_files(d, pre_contents)

    assert "MY_UNCOMMITTED_EDIT" in open(os.path.join(d, "harness.py")).read()
    assert open(os.path.join(d, "BOOTSTRAP.md")).read() == "boot-edit\n"


def test_snapshot_excludes_evolution_and_pycache():
    """Snapshot must capture harness files but skip evolution/ and __pycache__."""
    d, _ = _make_repo()
    os.makedirs(os.path.join(d, "evolution", "strategies"), exist_ok=True)
    with open(os.path.join(d, "evolution", "archive.py"), "w") as f:
        f.write("# should NOT be snapshotted\n")
    os.makedirs(os.path.join(d, "__pycache__"), exist_ok=True)
    with open(os.path.join(d, "__pycache__", "harness.cpython-310.pyc"), "w") as f:
        f.write("junk")

    snap = _snapshot_harness_files(d)
    assert "harness.py" in snap
    assert "BOOTSTRAP.md" in snap
    assert all(not k.startswith("evolution/") for k in snap)
    assert all(not k.startswith("__pycache__") for k in snap)
