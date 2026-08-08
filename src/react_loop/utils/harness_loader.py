"""
Harness Loader - fresh-imports the agent's harness entry function from disk.

Core invariant: **the bytes executed at evaluate time == the bytes on disk == the bytes hashed**.

To this end, every ``load()`` re-imports the entry file and its sibling modules from disk
under a **unique package name** (``_harness_pkg_<N>``). The unique package name means the
stale guard in ``_load_dependencies`` (``if module_name in sys.modules: continue``) never
hits entries left over from the previous evaluate — every import reads the current disk
content. This is the key detail that fixes the reward <-> code_hash misalignment: reusing a
fixed basename (e.g. ``repo``) would return the module cached from the first evaluate and
now stale, leaving the reward stamped onto a corrupted disk hash.

The caller is responsible for calling ``HarnessLoader.cleanup_loaded(agent_code_dir)`` after
evaluation to clean ``sys.modules``, covering absolute imports (``from utils import X``
registers the bare name ``utils``) and avoiding long-run accumulation. The old "hot reloader"
path has been removed — evaluate reads disk, edit writes disk, no intermediate layer needed.
"""

import importlib.util
import inspect
import itertools
import logging
import os
import re
import shutil
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Monotonically increasing counter that generates a unique package name per fresh-import.
# Thread-safe (atomic under CPython GIL via itertools.count.__next__); uniqueness is enough,
# ordering does not matter.
_package_counter = itertools.count()


# Normalization of anonymous package names to agent-readable file names.
#
# The temporary package name ``_harness_pkg_<N>`` injected by the loader in tracebacks is
# noise to the agent: it cannot find that path in the filesystem (cleared from
# ``sys.modules`` once evaluate ends), and can only translate ``_harness_pkg_3.prompts`` to
# ``prompts.py`` by inference — it may even mistakenly assume there is a directory called
# ``_harness_pkg_3`` and look for it in bash. This performs mechanical normalization before
# the error text is fed back to the agent, so locating it is one step. Only the loader's
# anonymous package name is matched; real file paths (e.g.
# ``File ".../repo/harness.py"``) are not affected.
_HARNESS_PKG_RE = re.compile(r"_harness_pkg_\d+(?:\.([A-Za-z_]\w*))?")


def normalize_error_text(text: str) -> str:
    """Normalize the anonymous package name ``_harness_pkg_N[.mod]`` in a string to an agent-readable form.

    - ``_harness_pkg_3.prompts`` -> ``prompts.py`` (module qualified name -> file name)
    - ``_harness_pkg_3`` (bare name) -> ``<harness>``

    Strings that do not contain ``_harness_pkg_`` are returned as-is, so callers can wrap
    unconditionally — pre-filtering is handled by ``"_harness_pkg_" in s`` with no regex
    cost.
    """
    if not isinstance(text, str) or "_harness_pkg_" not in text:
        return text

    def _repl(m: "re.Match[str]") -> str:
        mod = m.group(1)
        return f"{mod}.py" if mod else "<harness>"

    return _HARNESS_PKG_RE.sub(_repl, text)


def normalize_metrics(metrics: Any) -> Any:
    """Recursively scrub in place all string values in metrics that contain ``_harness_pkg_N``.

    Walks dict values and list elements (keys are fixed field names, not modified). For each
    string, do a ``"_harness_pkg_" in s`` pre-filter first — most strings (reward, progression,
    step traces, ...) pass instantly; only error text that actually contains an anonymous
    package name goes through the regex. Called once before evaluate returns; all downstream
    display paths (top-level ``metrics['error']``, per-task ``metadata['execution_error']``,
    per-episode ``interaction_log[i]['error']``) become clean automatically, and newly added
    benchmarks benefit too.
    """
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            if isinstance(v, str):
                if "_harness_pkg_" in v:
                    metrics[k] = normalize_error_text(v)
            else:
                normalize_metrics(v)
    elif isinstance(metrics, list):
        for i, v in enumerate(metrics):
            if isinstance(v, str):
                if "_harness_pkg_" in v:
                    metrics[i] = normalize_error_text(v)
            else:
                normalize_metrics(v)
    return metrics


@dataclass
class HarnessInfo:
    """Harness function info."""
    func: Callable
    source: str  # Source description
    has_agent_param: bool  # Whether it is the new-style signature func(agent, task) -> str
    module: Any = None  # Module where the entry function lives (getattr fallback, e.g. legacy set_current_agent)


class HarnessLoader:
    """Unified harness function loader (always fresh-imports from disk)."""

    # Constants
    ENTRY_FILES = ['harness.py']
    FALLBACK_FILES = ['solve.py', 'solver.py']
    # Default function names to look up (by priority)
    DEFAULT_FUNC_NAMES = ['using_harness', 'harness']

    def __init__(self, agent_code_dir: str = None):
        """
        Initialize HarnessLoader.

        Args:
            agent_code_dir: Path to the agent code directory.
        """
        # Normalize to an absolute path, without depending on a live CWD throughout.
        # os.path.abspath() internally calls os.getcwd(), and during a long self-evolving
        # run the process CWD may be deleted by a bash command the agent itself wrote
        # (see the same failure mode in adapter._build_log_dir) — at that point abspath
        # raises FileNotFoundError, and even "load harness" cannot proceed. Absolute paths
        # do not need getcwd; relative paths are only resolved when CWD is still alive,
        # otherwise they are kept as-is (normpath, without touching getcwd), and left to
        # callers that can provide absolute paths (e.g. main.py anchors the launch CWD).
        self.agent_code_dir = self._normalize_code_dir(agent_code_dir)
        self._load_errors: list = []  # Collect all load errors for agent visibility

    @staticmethod
    def _normalize_code_dir(path: Optional[str]) -> Optional[str]:
        """Normalize agent_code_dir to an absolute path, without depending on a live CWD throughout.

        - Absolute path: returned directly after normpath (no getcwd call).
        - Relative path + CWD alive: abspath resolves it to absolute.
        - Relative path + CWD deleted: abspath raises FileNotFoundError, fall back to
          normpath (kept relative). The caller should pass an absolute path in this case;
          this method at least ensures no exception is raised and construction is not blocked.
        """
        if not path:
            return path
        if os.path.isabs(path):
            return os.path.normpath(path)
        try:
            return os.path.abspath(path)
        except FileNotFoundError:
            return os.path.normpath(path)

    def _add_error(self, msg: str) -> None:
        """Record a load error for later reporting to agent."""
        self._load_errors.append(msg)
        logger.warning(msg)  # Still log for debugging

    def get_error_summary(self) -> str:
        """Get summary of all load errors for agent visibility."""
        if not self._load_errors:
            return ""
        return "\n".join(f"- {err}" for err in self._load_errors)

    @staticmethod
    def _fresh_package_name() -> str:
        """Generate a unique package name, ensuring this import does not hit a stale entry in sys.modules."""
        return f"_harness_pkg_{next(_package_counter)}"

    def load(
        self,
        agent_instance: Any = None,
        harness_func: Callable = None,
        func_names: list = None,
        package_name: Optional[str] = None,
    ) -> HarnessInfo:
        """
        Load the harness function.

        Priority:
        1. Directly passed function (``harness_func``) — does not go through disk; the
           caller bears the alignment responsibility.
        2. Fresh-import from the ``self.agent_code_dir`` filesystem (unique package name).

        Args:
            agent_instance: Agent instance (used by the new-style signature).
            harness_func: Directly passed function (highest priority, skips disk import).
            func_names: List of function names to look up (default: ['using_harness', 'harness']).
            package_name: Package name used during import. When None, a unique name is
                auto-generated (recommended), guaranteeing current disk content is read
                rather than the sys.modules cache.

        Returns:
            HarnessInfo object.

        Raises:
            ValueError: The harness function could not be found.
        """
        search_names = func_names if func_names else self.DEFAULT_FUNC_NAMES

        # 1. Directly passed function
        if harness_func and callable(harness_func):
            logger.info("Using pre-loaded harness function")
            return self._create_harness_info(harness_func, "pre-loaded", None)

        # 2. Fresh-import from the filesystem (unique package name -> always reads disk)
        if self.agent_code_dir:
            info = self._load_from_filesystem(agent_instance, search_names, package_name)
            if info:
                return info

        raise ValueError(
            f"Agent code must have an entry function. "
            f"Searched names: {search_names}. "
            f"Signatures: func(agent, task) -> str or func(instruction: str) -> Any\n"
            f"Load errors:\n{self.get_error_summary()}"
        )

    @staticmethod
    def cleanup_loaded(agent_code_dir: str) -> int:
        """Clean up ``sys.modules`` after evaluation: remove all entries whose
        ``__file__``/``__path__`` lies under ``agent_code_dir``.

        Why it's needed: the unique package name only guarantees that **relative imports**
        (``from .utils import X`` registered as ``_pkg.utils``) do not hit stale entries;
        **absolute imports** (``from utils import X``) register the bare name ``utils``,
        which the next evaluate would return as a stale version straight from
        ``sys.modules``. This method filters by disk path, covering both import styles,
        while preserving shared dependencies (numpy, etc.).
        Use ``realpath`` to resolve symlinks so path comparison does not miss them.

        Returns:
            Number of sys.modules entries removed.
        """
        if not agent_code_dir:
            return 0
        # realpath calls getcwd() when resolving a relative path; if CWD was deleted by an
        # agent-written bash command, fall back to normpath (no getcwd). The path comparison
        # below (real == base / startswith) goes through the same logic on both sides to stay
        # consistent, without depending on getcwd.
        try:
            base = os.path.realpath(agent_code_dir)
        except FileNotFoundError:
            base = os.path.normpath(agent_code_dir)
        removed = 0
        for key in list(sys.modules.keys()):
            mod = sys.modules.get(key)
            if mod is None:
                continue
            # Collect every possible disk path for this module: __file__ (regular module) + __path__[] (package)
            paths = []
            f = getattr(mod, '__file__', None)
            if f:
                paths.append(f)
            p = getattr(mod, '__path__', None)
            if p:
                try:
                    paths.extend(iter(p))
                except TypeError:
                    pass
            if not paths:
                continue
            for path in paths:
                try:
                    real = os.path.realpath(path)
                except (ValueError, TypeError, FileNotFoundError):
                    continue
                # startswith(base + sep) prevents prefix collisions (e.g. /foo/repo vs /foo/repository);
                # == base covers the package directory itself (a stub package's __path__ points at agent_code_dir).
                if real == base or real.startswith(base + os.sep):
                    del sys.modules[key]
                    removed += 1
                    break
        return removed

    # ── error text normalization (along with cleanup_loaded, both "post-processing of loader products") ──
    # cleanup_loaded clears runtime residue from sys.modules; normalize_* clears the
    # anonymous package name residue from error text, so the agent doesn't have to decode
    # ``_harness_pkg_3``. Both are exposed as staticmethods for the adapter to call before
    # evaluate returns. See module-level functions for the implementation.
    @staticmethod
    def normalize_error_text(text: str) -> str:
        """See module-level :func:`normalize_error_text`."""
        return normalize_error_text(text)

    @staticmethod
    def normalize_metrics(metrics: Any) -> Any:
        """See module-level :func:`normalize_metrics`."""
        return normalize_metrics(metrics)

    def _load_from_filesystem(
        self, agent_instance: Any, func_names: list = None, package_name: Optional[str] = None
    ) -> Optional[HarnessInfo]:
        """Fresh-import the harness from the filesystem."""
        if not self.agent_code_dir or not os.path.exists(self.agent_code_dir):
            return None

        # Default generates a unique package name: ensures this import is not polluted by
        # stale entries from the previous round in sys.modules.
        # The caller can also pass package_name explicitly (e.g. the export path prefers a
        # human-readable package name).
        if package_name is None:
            package_name = self._fresh_package_name()

        # Invalidate the harness bytecode cache (.pyc). .pyc is indexed by **source file
        # path** and is unrelated to the module name, so the unique package name does not
        # block it; CPython validates .pyc using the source file's ``int(st_mtime)``
        # (truncated to whole seconds) + size. Edits within the same second with unchanged
        # size will reuse a stale .pyc — running old bytecode while disk is already new
        # (possibly broken) source, exactly reopening the reward<->hash misalignment this
        # loader is meant to plug.
        # Deleting the cache forces recompilation from the current source.
        self._invalidate_bytecode_cache()

        if os.path.isdir(self.agent_code_dir):
            # Try package mode first
            info = self._load_as_package(agent_instance, func_names, package_name)
            if info:
                return info
            # Then try single-file mode
            return self._load_as_files(agent_instance, func_names, package_name)
        else:
            # Single file
            return self._load_single_file(self.agent_code_dir, agent_instance, func_names)

    def _invalidate_bytecode_cache(self) -> None:
        """Delete the top-level ``__pycache__`` of ``agent_code_dir`` to force the next import to compile from source.

        Only clears the top level: the harness entry + sibling ``.py`` files are all at the
        ``agent_code_dir`` root (flat contract), and their ``.pyc`` land in the top-level
        ``__pycache__``. Does not clear the caches of subdirectories such as ``evolution/``
        — those are meta-evolve strategy modules; clearing them would force recompilation
        every time, and it would also avoid doing an ``os.walk`` over the whole
        ``agent_code_dir`` on every evaluate.
        """
        cache = os.path.join(self.agent_code_dir, "__pycache__")
        if os.path.isdir(cache):
            shutil.rmtree(cache, ignore_errors=True)

    def _load_as_package(
        self, agent_instance: Any, func_names: list, package_name: str
    ) -> Optional[HarnessInfo]:
        """Import as a Python package (fresh-import under the unique package name).

        ``package_name`` is generated by ``_load_from_filesystem`` and passed in (unique).
        """
        init_path = os.path.join(self.agent_code_dir, '__init__.py')
        if not os.path.exists(init_path):
            return None

        search_names = func_names if func_names else self.DEFAULT_FUNC_NAMES

        # Add the parent directory to sys.path (so the package is importable)
        # agent_code_dir is already normalized to an absolute path in __init__, so dirname
        # suffices; normally getcwd is not needed. abspath is still wrapped to handle the
        # extreme case where CWD was deleted and FileNotFoundError is raised (fall back to
        # dirname directly).
        try:
            parent_dir = os.path.dirname(os.path.abspath(self.agent_code_dir))
        except FileNotFoundError:
            parent_dir = os.path.dirname(self.agent_code_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        # Add agent_code_dir to sys.path (so the module can import sibling modules)
        if self.agent_code_dir not in sys.path:
            sys.path.insert(0, self.agent_code_dir)

        try:
            # Register the package stub (do not execute __init__.py, avoiding circular dependencies).
            # Unique name -> if a stale entry with the same name already exists, clear it first,
            # ensuring the stub's __path__ points at the current directory.
            import types
            pkg = types.ModuleType(package_name)
            pkg.__package__ = package_name
            pkg.__path__ = [self.agent_code_dir]
            sys.modules[package_name] = pkg

            # Load the dependency modules (tolerate individual failures)
            self._load_dependencies(package_name)

        except Exception:
            self._add_error(f"Failed to setup package dependencies:\n{traceback.format_exc()}")

        # Look up the entry file (separate try/except so a failure in one entry file does not affect others)
        for entry_file in self.ENTRY_FILES:
            entry_path = os.path.join(self.agent_code_dir, entry_file)
            if os.path.exists(entry_path):
                try:
                    module = self._load_module(entry_path, f"{package_name}.{entry_file[:-3]}")
                    if module:
                        info = self._find_harness_in_module(module, agent_instance, search_names)
                        if info:
                            return info
                except Exception:
                    self._add_error(f"Failed to load entry file {entry_file}:\n{traceback.format_exc()}")

        return None

    def _load_as_files(
        self, agent_instance: Any, func_names: list, package_name: str
    ) -> Optional[HarnessInfo]:
        """Independent-file fallback: ``_load_as_package`` returns None only when there
        is no ``__init__.py``, so the directory this method handles **has no package
        semantics** — no stub registration, no relative imports.
        The harness should reference siblings via absolute imports (``from utils import X``,
        resolved via sys.path).

        ``package_name`` is still used as the module name prefix (ensures uniqueness,
        avoiding sys.modules collisions); ``module.__package__`` is set to
        ``_harness_pkg_N`` by ``module_from_spec`` according to that prefix — there is no
        corresponding stub, so relative imports are unavailable, but that is exactly the
        semantics of a directory without ``__init__.py``.
        """
        search_names = func_names if func_names else self.DEFAULT_FUNC_NAMES

        # Sibling dependencies are loaded only once (rather than re-running listdir for every candidate entry file).
        self._load_dependencies(package_name)

        all_files = self.ENTRY_FILES + self.FALLBACK_FILES
        for filename in all_files:
            filepath = os.path.join(self.agent_code_dir, filename)
            if os.path.exists(filepath):
                module_name = f"{package_name}.{filename[:-3]}"
                try:
                    module = self._load_module(filepath, module_name)
                    if module:
                        info = self._find_harness_in_module(module, agent_instance, search_names)
                        if info:
                            return info
                except Exception:
                    self._add_error(f"Failed to load {filename} as file:\n{traceback.format_exc()}")
        return None

    def _load_single_file(self, filepath: str, agent_instance: Any, func_names: list = None) -> Optional[HarnessInfo]:
        """Load a single file."""
        search_names = func_names if func_names else self.DEFAULT_FUNC_NAMES
        module = self._load_module(filepath, "agent_module")
        if module:
            return self._find_harness_in_module(module, agent_instance, search_names)
        return None

    def _load_dependencies(self, package_name: str) -> None:
        """Dynamically load dependency modules - scan all .py files in the directory.

        The unique package name guarantees ``module_name`` is brand-new, so the
        ``if module_name in sys.modules: continue`` guard here almost never fires — unless
        the same evaluate() calls it repeatedly (harmless).
        """
        if not self.agent_code_dir or not os.path.isdir(self.agent_code_dir):
            return

        for filename in os.listdir(self.agent_code_dir):
            # Skip entry files and non-Python files
            if not filename.endswith('.py') or filename in self.ENTRY_FILES + self.FALLBACK_FILES:
                continue
            # Skip __init__.py (package init is not a module)
            if filename == '__init__.py':
                continue

            dep_path = os.path.join(self.agent_code_dir, filename)
            module_name = f"{package_name}.{filename[:-3]}"

            # Under a unique name it cannot hit a stale entry; defensively still skip those already loaded this round
            if module_name in sys.modules:
                continue

            # Load the module
            try:
                self._load_module(dep_path, module_name)
            except Exception:
                self._add_error(f"Failed to load dependency {filename}:\n{traceback.format_exc()}")

    def _load_module(self, filepath: str, module_name: str) -> Optional[Any]:
        """Generic module loading method.

        ``module.__package__`` is set to ``spec.parent`` automatically by
        ``module_from_spec`` according to ``module_name`` (``_harness_pkg_N.harness`` ->
        ``_harness_pkg_N``); no manual override is needed — overriding it manually can
        instead disagree with ``__spec__.parent`` and trigger a DeprecationWarning.
        """
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
                return module
            except Exception:
                # Cleanup: Remove stale module entry on failure to prevent cached empty module
                if module_name in sys.modules:
                    del sys.modules[module_name]
                raise  # Re-raise to propagate error
        return None

    def _find_harness_in_module(self, module: Any, agent_instance: Any, func_names: list = None) -> Optional[HarnessInfo]:
        """Find the harness function in the module."""
        search_names = func_names if func_names else self.DEFAULT_FUNC_NAMES
        for func_name in search_names:
            func = getattr(module, func_name, None)
            if callable(func):
                logger.info(f"Found {func_name} in module")
                return self._create_harness_info(func, f"module.{func_name}", module=module)
        return None

    def _create_harness_info(
        self, func: Callable, source: str, module: Any = None
    ) -> HarnessInfo:
        """Create a HarnessInfo object."""
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        has_agent_param = len(params) >= 2 and params[0] in ['agent', 'self']
        return HarnessInfo(
            func=func, source=source, has_agent_param=has_agent_param, module=module,
        )
