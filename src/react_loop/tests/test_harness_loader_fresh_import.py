"""Regression tests for the code_hash ↔ running-bytes alignment fix.

These pin the invariant ``HarnessLoader`` restores:

    the bytes that RUN during evaluate == the bytes on DISK == the bytes HASHED.

The old hot-reload path let ``sys.modules`` hold a stale, *working* module while
the on-disk version had drifted to a *broken* state. evaluate() then earned its
reward in memory, the reward got stamped onto the broken disk hash, and the
broken state was committed/exported — the final eval (which fresh-imports from
disk) crashed. The fix: evaluate fresh-imports the harness under a **unique
package name** each call, then ``cleanup_loaded`` purges ``sys.modules``. So a
broken disk always surfaces as a load error, never a stale reward.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

# Repo root (for `src.*`) AND src/ (for `benchmark`/`react_loop` top-level, matching main.py).
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from src.react_loop.utils.harness_loader import HarnessLoader  # noqa: E402


class TestFreshImportInvariance(unittest.TestCase):
    """The core invariant: disk is truth; sys.modules never goes stale."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="harness_loader_test_")
        # Minimal package harness: __init__.py => loaded as a package, so the
        # entry file's `from .context import ...` relative imports resolve.
        Path(self.tmp, "__init__.py").write_text("")
        self._write_utils("def helper():\n    return 'v1'\n", "WAS_AMBIGUOUS = True\n")
        Path(self.tmp, "context.py").write_text(
            "from .utils import helper, WAS_AMBIGUOUS\n"
            "def ctx():\n    return helper(), WAS_AMBIGUOUS\n"
        )
        Path(self.tmp, "harness.py").write_text(
            "from .context import ctx\n"
            "def using_harness(agent, task):\n    return ctx()\n"
        )

    def tearDown(self):
        HarnessLoader.cleanup_loaded(self.tmp)
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _join(parts):
        return "".join(parts)

    def _write_utils(self, *parts):
        Path(self.tmp, "utils.py").write_text(self._join(parts))

    def _load(self):
        loader = HarnessLoader(self.tmp)
        return loader.load(agent_instance=None, harness_func=None)

    # ------------------------------------------------------------------
    # The invariant
    # ------------------------------------------------------------------

    def test_first_import_resolves(self):
        """Sanity: a clean harness loads and runs."""
        info = self._load()
        self.assertTrue(info.has_agent_param)
        self.assertEqual(info.func(None, ""), ("v1", True))
        self.assertIsNotNone(info.module)

    def test_edited_disk_takes_effect_next_load(self):
        """Editing utils.py changes what the NEXT load returns — no stale cache."""
        self._load()  # warm: populates sys.modules under _harness_pkg_<N>
        # Edit utils.py: helper() -> 'v2'; WAS_AMBIGUOUS unchanged.
        self._write_utils("def helper():\n    return 'v2'\n", "WAS_AMBIGUOUS = True\n")
        HarnessLoader.cleanup_loaded(self.tmp)
        info = self._load()
        self.assertEqual(info.func(None, ""), ("v2", True))

    def test_broken_disk_surfaces_as_load_error_not_stale_reward(self):
        """The exact bug: removing a symbol mid-iteration must surface as a load
        error on the next evaluate(), NOT return a stale working module that
        earns reward against the broken on-disk hash.

        Before the fix (hot-reload): ``sys.modules['repo.utils']`` still held the
        working v1 module with ``helper``, so context's import resolved from the
        stale cache and the harness kept earning reward — while disk utils.py was
        broken. After the fix: fresh import + cleanup => broken disk => error.
        """
        self._load()  # v1 working module now resident in sys.modules
        # Break the disk: drop helper() entirely so context's
        # `from .utils import helper` can no longer resolve.
        self._write_utils("WAS_AMBIGUOUS = True\n")
        HarnessLoader.cleanup_loaded(self.tmp)
        loader = HarnessLoader(self.tmp)
        with self.assertRaises(ValueError):
            loader.load(agent_instance=None, harness_func=None)
        # A failed load may leave partial modules behind (e.g. utils loaded fine
        # before context failed). The adapter's finally clause runs cleanup_loaded
        # exactly like this — it must sweep the leftovers so the next evaluate()
        # starts clean, and a second sweep is a no-op.
        first_sweep = HarnessLoader.cleanup_loaded(self.tmp)
        self.assertGreaterEqual(first_sweep, 0)
        self.assertEqual(HarnessLoader.cleanup_loaded(self.tmp), 0)

    # ------------------------------------------------------------------
    # Mechanism: unique name + cleanup
    # ------------------------------------------------------------------

    def test_unique_package_name_each_load(self):
        """Consecutive loads use distinct package names, so the staleness guard
        in _load_dependencies (``if module_name in sys.modules: continue``) never
        returns a previous evaluate()'s cached module."""
        info1 = self._load()
        pkg1 = info1.module.__package__
        HarnessLoader.cleanup_loaded(self.tmp)
        info2 = self._load()
        pkg2 = info2.module.__package__
        self.assertTrue(pkg1.startswith("_harness_pkg_"), pkg1)
        self.assertTrue(pkg2.startswith("_harness_pkg_"), pkg2)
        self.assertNotEqual(pkg1, pkg2)

    def test_cleanup_purges_loaded_modules(self):
        """cleanup_loaded removes every sys.modules entry under agent_code_dir,
        covering relative-import names (``_pkg.utils``) and the package stub."""
        self._load()
        # Filter by the unique package prefix — NOT a substring match (which would
        # wrongly catch unrelated modules like `email.utils`, `src.react_loop.utils.*`).
        leaky = [k for k in sys.modules if k.startswith("_harness_pkg_")]
        self.assertTrue(leaky, f"expected harness package modules in sys.modules, got {leaky}")
        removed = HarnessLoader.cleanup_loaded(self.tmp)
        self.assertGreaterEqual(removed, 1)
        survivors = [k for k in leaky if k in sys.modules]
        self.assertEqual(survivors, [], f"cleanup left modules behind: {survivors}")

    def test_cleanup_leaves_shared_deps(self):
        """cleanup_loaded must NOT touch modules outside agent_code_dir (numpy,
        framework code, etc.) — it filters by realpath under the dir."""
        # Simulate a shared dep resident in sys.modules pointing elsewhere.
        fake = "def helper():\n    return 'v1'\n"
        Path(self.tmp, "shared_dep_marker.txt").write_text(fake)  # dir stays the harness dir
        self._load()
        removed = HarnessLoader.cleanup_loaded(self.tmp)
        self.assertGreaterEqual(removed, 1)
        # Framework module (not under self.tmp) must survive.
        self.assertIn("src.react_loop.utils.harness_loader", sys.modules)

    def test_cleanup_empty_or_missing_dir_is_noop(self):
        """cleanup_loaded must not raise on falsy / nonexistent paths."""
        self.assertEqual(HarnessLoader.cleanup_loaded(""), 0)
        self.assertEqual(HarnessLoader.cleanup_loaded("/no/such/dir/anywhere"), 0)

    def test_load_returns_none_on_empty_dir(self):
        """A dir with no entry file yields a clear ValueError, not a crash."""
        empty = tempfile.mkdtemp(prefix="harness_empty_")
        try:
            Path(empty, "__init__.py").write_text("")
            with self.assertRaises(ValueError) as cm:
                HarnessLoader(empty).load(agent_instance=None, harness_func=None)
            self.assertIn("entry function", str(cm.exception))
        finally:
            shutil.rmtree(empty, ignore_errors=True)


class TestAdapterHarnessGlue(unittest.TestCase):
    """Exercises BenchmarkEvaluatorAdapter._get_harness + _wrap_harness + cleanup
    glue without the full benchmark registry — covers both harness signatures.

    Risk point called out in the plan: the legacy ``func(instruction)`` signature
    must still receive the live agent via module-level ``set_current_agent`` now
    that the reloader (which used to fetch it) is gone.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="adapter_glue_")
        Path(self.tmp, "__init__.py").write_text("")

    def tearDown(self):
        HarnessLoader.cleanup_loaded(self.tmp)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_adapter(self):
        # Bypass __init__ (which needs a registered evaluator + config); the
        # harness-loading glue only touches self._harness_loader.
        from benchmark.adapter import BenchmarkEvaluatorAdapter
        adapter = BenchmarkEvaluatorAdapter.__new__(BenchmarkEvaluatorAdapter)
        adapter._harness_loader = HarnessLoader()
        return adapter

    def _agent(self):
        return SimpleNamespace(action_executor=SimpleNamespace(agent_code_dir=self.tmp))

    def test_new_signature_loads_and_wraps(self):
        """func(agent, task) -> str: _get_harness fresh-imports, _wrap_harness passes
        the live agent through (proven by reading agent.tag inside the harness)."""
        Path(self.tmp, "harness.py").write_text(
            "def using_harness(agent, task):\n    return f'ran:{task}:{agent.tag}'\n"
        )
        agent = SimpleNamespace(
            action_executor=SimpleNamespace(agent_code_dir=self.tmp),
            tag="LIVE",
        )
        wrapped = self._make_adapter()._get_harness(agent, None, None)
        self.assertEqual(wrapped("go"), "ran:go:LIVE")
        self.assertGreaterEqual(HarnessLoader.cleanup_loaded(self.tmp), 1)

    def test_legacy_signature_uses_set_current_agent(self):
        """func(instruction) + module-level set_current_agent: _wrap_harness must
        hand the live agent over via getattr(module, 'set_current_agent')."""
        Path(self.tmp, "harness.py").write_text(
            "_agent = None\n"
            "def set_current_agent(a):\n    global _agent\n    _agent = a\n"
            "def using_harness(task):\n    return f'agent_set:{_agent is not None}'\n"
        )
        agent = self._agent()
        wrapped = self._make_adapter()._get_harness(agent, None, None)
        self.assertEqual(wrapped("task"), "agent_set:True")
        HarnessLoader.cleanup_loaded(self.tmp)


if __name__ == "__main__":
    unittest.main()