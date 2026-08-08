"""
Archive Manager — Evolution version management strategy layer.

Delegates seed / commit selection to the optional select_seed.py /
select_commit.py modules in the evolution/ directory. select_best.py is the
exception: it is a FIXED, non-evolvable module loaded from the init template
(godel_evolution_init/), NOT evolution/, so meta-evolve (whose edit sandbox
is restricted to evolution/) physically cannot edit it. All data queries go
through EvolutionTracker. No internal data storage — ArchiveManager is a
thin strategy layer.
"""

import os
import sys
from typing import List, Dict, Any

from .archive_strategies import get_strategy, discover_strategies, DEFAULT_STRATEGY, STRATEGY_REGISTRY
from .state import METADATA_FILES
from .utils import log_format
from .utils.log_format import _C

# When main.py adds src/ to sys.path, the relative import above registers
# the module as "react_loop.archive_strategies".  Strategy files and
# the select_*.py templates use the absolute import "from src.react_loop.archive_strategies
# import ...", which creates a SEPARATE module instance under the key
# "src.react_loop.archive_strategies".  We need to keep both in sync.
_ALTERNATE_KEYS = [
    "src.react_loop.archive_strategies",
    "react_loop.archive_strategies",
]


class ArchiveManager:
    """Manages version selection strategy for non-linear evolution.

    Delegates to the optional select_seed.py / select_commit.py modules in the
    evolution/ directory for custom selection strategies. select_best.py is
    FIXED and non-evolvable: it is loaded from the init template
    (godel_evolution_init/), not evolution/, so meta-evolve cannot edit it.
    Falls back to the configured archive_strategy (default "greedy") when a
    module is absent.
    """

    def __init__(self, agent):
        """
        Args:
            agent: Reference to the GodelAgent instance.
        """
        self.agent = agent
        self._strategies_discovered = False

    # -----------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------

    def select_seed(self) -> dict:
        """Choose starting version for next iteration.

        Returns:
            Seed info dict: {"git_hash": str, "strategy_hint": str, "metadata": dict}
            Falls back to configured strategy on any failure.
        """
        agent = self.agent

        module = self._load_seed_module()

        if not module or not hasattr(module, 'select_seed'):
            return self._call_configured_strategy("no seed module")

        try:
            seed_info = module.select_seed(agent)
            if seed_info and seed_info.get("git_hash"):
                # Runtime guard: reject meta_evolve commits as seed
                if self.is_meta_evolve_commit(seed_info["git_hash"]):
                    agent._log(
                        f"  Warning: archive returned meta_evolve commit "
                        f"{seed_info['git_hash'][:7]}, falling back to configured strategy"
                    )
                    return self._call_configured_strategy(
                        "meta_evolve commit rejected", use_head_fallback=True
                    )
                returned_hint = seed_info.get("strategy_hint", "")
                if returned_hint in ("fallback", ""):
                    override = self._call_configured_strategy(
                        f"archive returned '{returned_hint}'"
                    )
                    if override:
                        return override
                log_format.log_selection_result(
                    agent, "seed",
                    seed_info.get("git_hash", ""),
                    seed_info.get("strategy_hint", ""),
                )
                # The early guard above already rejected any meta_evolve commit,
                # and _resolve_non_meta_commit's own first-line check would just
                # no-op here — so the seed hash is guaranteed non-meta already.
                return seed_info
        except Exception as e:
            agent._log(f"Warning: select_seed.select_seed() raised: {e}")

        return self._call_configured_strategy("archive exception", use_head_fallback=True)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _ensure_strategies_discovered(self):
        """Lazy strategy discovery, then sync registry to all module keys."""
        if self._strategies_discovered:
            return
        agent = self.agent
        evo_dir = os.path.join(agent.agent_code_dir, "evolution")
        registered = discover_strategies(evo_dir)
        self._strategies_discovered = True
        if registered:
            agent._log(f"  Archive: discovered {len(registered)} strategies: {registered}")
        else:
            agent._log(f"  Warning: no strategies discovered from {evo_dir}")

        # Sync the populated STRATEGY_REGISTRY to every module key so that
        # both "react_loop.archive_strategies" (relative import) and
        # "src.react_loop.archive_strategies" (absolute import) see the
        # same populated dict.
        self._sync_registry()

    def _sync_registry(self):
        """Copy the populated STRATEGY_REGISTRY to alternate sys.modules entries.

        main.py adds src/ to sys.path, so the relative import in this file
        registers the module as "react_loop.archive_strategies".  Evolvable
        templates (select_*.py, strategy files) use the absolute path
        "src.react_loop.archive_strategies", which may resolve to a different
        module instance with its own empty STRATEGY_REGISTRY.  This method
        copies the populated dict to all known keys.
        """
        for key in _ALTERNATE_KEYS:
            mod = sys.modules.get(key)
            if mod is not None and id(mod.STRATEGY_REGISTRY) != id(STRATEGY_REGISTRY):
                mod.STRATEGY_REGISTRY = STRATEGY_REGISTRY

    def _call_configured_strategy(self, reason: str, use_head_fallback: bool = False) -> dict | None:
        """Execute the configured archive strategy by name.

        Args:
            reason: Why we're calling the strategy directly (for logging).
            use_head_fallback: If True, return current HEAD when strategy fails.
                If False, return None on failure (caller has other options).
        """
        agent = self.agent
        self._ensure_strategies_discovered()
        strategy_name = getattr(agent.config, 'archive_strategy', DEFAULT_STRATEGY)
        fn = get_strategy(strategy_name)
        if fn:
            agent._log(f"  Archive: using configured '{strategy_name}' ({reason})")
            try:
                result = fn(agent)
                seed_dict = result.to_dict() if hasattr(result, "to_dict") else result
                # Ensure hypothesis field exists (backward compat with strategies
                # that don't generate one)
                if isinstance(seed_dict, dict) and "hypothesis" not in seed_dict:
                    seed_dict["hypothesis"] = ""
                return seed_dict
            except Exception as e:
                agent._log(f"  Warning: strategy '{strategy_name}' raised: {e}")
        if use_head_fallback:
            head = agent.git_controller.get_current_commit() or ""
            # Resolve away any meta commit sitting on HEAD, otherwise the guard
            # above rejects a meta seed only for this fallback to return the
            # very same meta commit — a self-defeating loop.
            head = self._resolve_non_meta_commit(head)
            return {"git_hash": head,
                    "strategy_hint": strategy_name, "hypothesis": "", "metadata": {}}
        return None

    def _load_seed_module(self):
        """Load select_seed.py from evolution/ via EvolveHelper's shared loader."""
        return self.agent.iter_helper._load_evolution_module("select_seed")

    def _load_commit_module(self):
        """Load select_commit.py from evolution/ via EvolveHelper's shared loader."""
        return self.agent.iter_helper._load_evolution_module("select_commit")

    def _load_best_module(self):
        """Load select_best.py from the godel_evolution_init path (NOT evolution/).

        select_best is a FIXED, non-evolvable module: it ships in the init
        template and is loaded as-is every run. meta-evolve's edit sandbox is
        restricted to evolution/ (the edit_file/write_file path guard in
        agent.py _execute_tool_impl), and select_best.py is never copied there
        (see GodelAgent._init_evolution_dir skip_files), so loading from the
        init path makes it physically uneditable by meta-evolve while its
        framework runner (submit_best.run_submit_best) still executes a react
        loop every run.

        Returns None if the init path is unset, the file is absent, or loading
        raises — the caller then falls back to get_best_version("highest_reward").
        """
        import importlib.util

        agent = self.agent
        init_path = getattr(agent.config, "godel_evolution_init_path", "") or ""
        if not init_path:
            return None
        best_path = os.path.join(init_path, "select_best.py")
        if not os.path.isfile(best_path):
            return None
        try:
            spec = importlib.util.spec_from_file_location(
                "godel_evolution_init.select_best", best_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            agent._log(
                f"Warning: failed to load select_best from init template: {e}"
            )
            return None

    # -----------------------------------------------------------------
    # Validation & guard
    # -----------------------------------------------------------------

    def validate_archive(self) -> Dict[str, Any]:
        """Dry-run validate the current select_*.py modules and strategies.

        Checks:
        1. Whether select_seed.py can be loaded and select_seed() called without error.
        2. Whether all strategy names referenced in _STRATEGY_SCHEDULE are registered.
        3. Whether select_seed() returns a valid dict with git_hash.
        4. Whether select_commit() (when present) runs without error and returns a
           valid dict — exercised on a minimal mock state so the non-empty logic
           path (where agent edits most often break) is covered.

        Returns:
            {"valid": bool, "error": str, "seed_info": dict|None,
             "commit_info": dict|None, "unknown_strategies": [str]}
        """
        agent = self.agent
        result = {"valid": False, "error": "", "seed_info": None,
                  "commit_info": None, "unknown_strategies": []}

        # 1. Check unknown strategy names in select_seed.py source
        unknown = self._check_strategy_names_in_archive()
        result["unknown_strategies"] = unknown
        if unknown:
            result["error"] = (
                f"Unknown strategies in _STRATEGY_SCHEDULE: {unknown}. "
                f"Registered: {list(STRATEGY_REGISTRY.keys())}"
            )
            return result

        # 2. Dry-run select_seed() — now agentic (react loop), must be stubbed
        self._ensure_strategies_discovered()
        seed_module = self._load_seed_module()
        if not seed_module:
            result["error"] = "No select_seed.py module found"
            return result

        if not hasattr(seed_module, 'select_seed'):
            result["error"] = "select_seed.py has no select_seed() function"
            return result

        seed_err, seed_info = self._dry_run_select_seed(seed_module, agent)
        result["seed_info"] = seed_info
        if seed_err:
            result["error"] = seed_err
            return result

        # 3. Verify select_commit.py (only when present — not copied when
        #    evolvable_commit_strategy=false). Only checks that the module
        #    loads and get_commit_nudge_prompt() returns a non-empty string.
        commit_module = self._load_commit_module()
        if commit_module and hasattr(commit_module, 'get_commit_nudge_prompt'):
            try:
                prompt = commit_module.get_commit_nudge_prompt(agent)
                if not prompt or not isinstance(prompt, str):
                    result["error"] = "select_commit.get_commit_nudge_prompt() returned empty or non-string"
                    return result
                result["commit_info"] = {"prompt_ok": True, "len": len(prompt)}
            except Exception as e:
                result["error"] = f"select_commit.get_commit_nudge_prompt() raised: {e}"
                return result

        result["valid"] = True
        return result

    @staticmethod
    def _select_result_error(out, key: str, fn_name: str) -> str | None:
        """Validate a select_*() return value.

        Must be a dict; if non-empty it must carry a truthy ``key``. Returns an
        error string if invalid, else None. Shared by the production delegators
        and the validate_archive dry-runs so the "valid select result" contract
        lives in one place.
        """
        if not isinstance(out, dict):
            return f"{fn_name}() returned invalid type: {type(out)}"
        if out and not out.get(key):
            return f"{fn_name}() returned non-empty dict without {key}"
        return None



    def _dry_run_select_seed(self, seed_module, agent) -> tuple:
        """Dry-run select_seed() without spending tokens.

        The new agentic select_seed() runs a react loop — expensive for a dry-run.
        We stub ``agent.react`` to immediately emit a ``pick_seed`` call returning
        the real current HEAD (so ``cat-file -t`` validation passes). The stubbed
        react is injected only while select_seed() runs; restored afterward.

        Returns ``(error_or_None, seed_info_or_None)``.
        """
        from types import SimpleNamespace

        mock_hash = agent.git_controller.get_current_commit() or "deadbee1"
        saved_react = agent.react

        def stub_react(messages, tools=None, tool_executor=None, **kw):
            tc = [{"id": "x", "name": "pick_seed",
                   "args": {"git_hash": mock_hash, "strategy_hint": "dry_run"}}]
            if tool_executor:
                tool_executor("pick_seed", tc[0]["args"])
            return (SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content="done", tool_calls=None))]), tc, ["ok"])

        seed_info = None
        saved_dry_flag = getattr(agent, "_seed_dry_run", False)
        try:
            agent.react = stub_react
            # Silence the SEED SELECTION banner + react-step logs so this
            # validation dry-run doesn't masquerade as a real seed selection
            # (symmetric to _submit_best_dry_run on the select_best dimension).
            agent._seed_dry_run = True
            out = seed_module.select_seed(agent)
            err = ArchiveManager._select_result_error(out, "git_hash", "select_seed")
            if err:
                return (err, None)
            if out:
                seed_info = {
                    "git_hash": str(out.get("git_hash", ""))[:7],
                    "strategy_hint": out.get("strategy_hint", ""),
                }
            return (None, seed_info)
        except Exception as e:
            return (f"select_seed() raised: {e}", None)
        finally:
            agent.react = saved_react
            agent._seed_dry_run = saved_dry_flag

    def _check_strategy_names_in_archive(self) -> List[str]:
        """Validate that select_seed.py exports the required FIXED-interface functions.

        The new agentic select_seed.py has ``select_seed``, ``get_seed_system_prompt``,
        and ``get_seed_tools`` as its public API — no more ``_STRATEGY_SCHEDULE``
        constant or direct strategy-name registry. Instead, validate that all tool
        names in ``_SEED_TOOLS`` correspond to existing strategy modules.
        """
        agent = self.agent
        seed_path = os.path.join(agent.agent_code_dir, "evolution", "select_seed.py")
        if not os.path.isfile(seed_path):
            return []

        seed_module = self._load_seed_module()
        if not seed_module:
            return ["select_seed.py could not be loaded"]

        unknown = []

        # Check FIXED interface functions exist
        for fn_name in ("select_seed", "get_seed_system_prompt", "get_seed_tools"):
            if not hasattr(seed_module, fn_name):
                unknown.append(f"missing FIXED function: {fn_name}")

        # Check tool names in _SEED_TOOLS correspond to loadable strategy modules
        seed_tools = getattr(seed_module, "get_seed_tools", None)
        if seed_tools:
            try:
                tools_list = seed_tools(agent)
            except Exception:
                tools_list = getattr(seed_module, "_SEED_TOOLS", [])
            for tool_cfg in tools_list:
                tool_name = tool_cfg.get("name", "")
                if not tool_name.startswith("pick_seed_"):
                    continue
                strategy_name = tool_name[len("pick_seed_"):]
                strat_path = os.path.join(
                    agent.agent_code_dir, "evolution", "strategies",
                    f"{strategy_name}.py"
                )
                if not os.path.isfile(strat_path):
                    unknown.append(
                        f"tool '{tool_name}' references missing strategy: "
                        f"evolution/strategies/{strategy_name}.py"
                    )
                else:
                    # Also verify the module is syntactically loadable
                    # (catches syntax errors that os.path.isfile would miss)
                    try:
                        import importlib.util as _iu
                        spec = _iu.spec_from_file_location(
                            f"_val_{strategy_name}", strat_path
                        )
                        _mod = _iu.module_from_spec(spec)
                        spec.loader.exec_module(_mod)
                    except Exception as e:
                        unknown.append(
                            f"tool '{tool_name}': evolution/strategies/"
                            f"{strategy_name}.py failed to load: {e}"
                        )

        return unknown

    def is_meta_evolve_commit(self, git_hash: str) -> bool:
        """Check if a git commit is a meta_evolve commit.

        Layers of detection:
        0. Tracker record: operation_type == "init" → NOT meta (INIT is a
           tracked main-line commit).
        1. Tracker record: metadata type == "meta_evolve".
        1.5. Commit subject marked [Ensemble] / [Crossover] → NOT meta. A
           fusion seed artifact can have an evolution/-only top-commit diff
           (the fusion splits the harness commit from the evolution/ restore),
           which would otherwise trip Layer 2 and get the agent's ensemble seed
           pick silently rejected → greedy fallback. Such commits are seeds, not
           meta commits, and are checked here BEFORE they land in the tracker.
        2. Git diff-tree fallback: a commit whose changed files are ALL under
           evolution/ is treated as meta (catches meta commits that never made
           it into the tracker, e.g. after a record failure). Mirrors the
           runtime-evolved ``evolution/select_*.py`` equivalent.
        """
        agent = self.agent
        if not git_hash:
            return False

        # Layer 0: tracker record with operation_type "init" → NOT meta.
        # INIT commits fold meta changes + version switch into one tracked
        # main-line commit; they must never be rejected as seed or parent.
        if agent.evolution_tracker:
            for record in agent.evolution_tracker.records:
                if git_hash in record.new_commit:
                    if record.metadata.get("operation_type") == "init":
                        return False
                    return record.metadata.get("type") == "meta_evolve"

        # Layer 1.5: ensemble / crossover fusion commits are SEED artifacts,
        # NOT meta commits. The returned top commit's diff can be confined to
        # evolution/ (the fusion splits the harness commit from the evolution/
        # restore, so the outer commit's only delta is evolution/) — without
        # this exemption the Layer-2 diff-tree heuristic below flags it as meta
        # and the agent's ensemble seed pick is silently rejected → greedy
        # fallback. The fusion commit is checked here BEFORE it lands in the
        # tracker (it is an intermediate selection artifact, invisible to
        # Layer 0/1 above), so we read the commit subject marker directly.
        # Mirrors the Layer-0 "init → not meta" exemption.
        try:
            subj_res = agent.git_controller._run_git_command(
                ["log", "-1", "--format=%s", git_hash], check=False,
            )
            if subj_res.returncode == 0 and (
                "[Ensemble]" in subj_res.stdout or "[Crossover]" in subj_res.stdout
            ):
                return False
        except Exception:
            pass

        # Layer 2: diff-tree fallback. Ignore build artifacts (*.pyc /
        # __pycache__) so a meta commit that dragged in bytecode is still
        # recognized as meta as long as its substantive source changes are
        # all under evolution/.
        try:
            result = agent.git_controller._run_git_command(
                ["diff-tree", "--no-commit-id", "-r", "--name-only", git_hash],
                check=False,
            )
            if result.returncode == 0:
                files = [f for f in result.stdout.strip().split("\n") if f]
                norm = [f.replace("\\", "/") for f in files]
                content = [f for f in norm
                           if not f.endswith(".pyc") and "__pycache__/" not in f]
                if content and all(f.startswith("evolution/") for f in content):
                    return True
        except Exception:
            pass

        return False

    def _resolve_non_meta_commit(self, git_hash: str) -> str:
        """Walk back from ``git_hash`` to the nearest non-meta commit.

        Safety net so a main iteration never seeds from — nor records as
        parent — a meta_evolve commit. Follows the tracker's recorded
        ``parent_commit`` chain first (it may jump straight to a main commit
        even when the git parent is another meta/switch commit), then falls
        back to the git first-parent. Returns the original hash unchanged if
        it is already non-meta or resolution fails.

        Note: ``meta_evolve._commit`` now resets HEAD to ``_pre_meta_commit``
        at the end of every meta phase, so in the normal path HEAD is already
        non-meta by the time this runs. This walker stays as belt-and-suspenders
        for the rare case where that reset was skipped (e.g. ``_pre_meta_commit``
        unset, or meta invoked outside ``run()``).
        """
        if not git_hash or not self.is_meta_evolve_commit(git_hash):
            return git_hash

        agent = self.agent
        visited = {git_hash}
        current = git_hash

        while True:
            nxt = None
            # Prefer the tracker-recorded parent (points at a real main commit)
            if agent.evolution_tracker:
                record = agent.evolution_tracker.get_record_by_commit(current)
                if record and record.parent_commit:
                    nxt = record.parent_commit
            # Fallback: git first-parent
            if not nxt:
                try:
                    result = agent.git_controller._run_git_command(
                        ["rev-parse", current + "^"], check=False
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        nxt = result.stdout.strip()
                except Exception:
                    pass
            if not nxt or nxt in visited:
                break  # exhausted — give up, return original
            visited.add(nxt)
            current = nxt
            if not self.is_meta_evolve_commit(current):
                return current

        return git_hash

    # -----------------------------------------------------------------
    # Public: version switching & merge ops
    # -----------------------------------------------------------------

    def select_best(self) -> dict:
        """Select which version to export as the FINAL best after evolution.

        select_best is a FIXED, non-evolvable stage: the strategy ships in the
        init template (godel_evolution_init/select_best.py) and is loaded as-is
        every run — meta-evolve's edit sandbox is restricted to evolution/, so
        it physically cannot touch this file. The framework runner
        (submit_best.run_submit_best) executes a react loop at the end of every
        evolution to make the pick, with optional ensemble fusion.

        Gated by ``config.submit_best_enabled`` (default on). Returns {} when
        disabled, the module is missing, the call raises, or the module returns
        no commit_hash — the caller then falls back to
        get_best_version("highest_reward").

        Returns:
            {commit_hash, submit_hint, metadata{...}} on success, else {}.
        """
        agent = self.agent

        if not getattr(agent.config, "submit_best_enabled", True):
            agent._log("  select_best: disabled (submit_best_enabled=false) — using highest-reward fallback")
            return {}

        module = self._load_best_module()
        if not module or not hasattr(module, "select_best"):
            agent._log("  select_best: module not loadable from init template — using highest-reward fallback")
            return {}

        try:
            result = module.select_best(agent)
            if result and result.get("commit_hash"):
                log_format.log_selection_result(
                    agent, "best",
                    result.get("commit_hash", ""),
                    result.get("submit_hint", ""),
                )
                return result
            agent._log(
                "  select_best: module returned no commit_hash (no candidates) — "
                "using highest-reward fallback"
            )
        except Exception as e:
            agent._log(f"Warning: select_best.select_best() raised: {e} — using highest-reward fallback")

        return {}

    def _checkout_target_harness(self, target_hash: str) -> None:
        """Check out harness files from *target_hash* while preserving evolution/.

        This is the pure-file-operation core extracted from
        ``apply_version_switch()`` — it does the checkout + orphan cleanup but
        does NOT commit. Reused by both ``apply_version_switch()`` and
        ``create_init_commit()``.  On failure the working tree is best-effort
        restored to HEAD.
        """
        agent = self.agent
        if not target_hash:
            return
        try:
            from .evolve import EvolveHelper
            bootstrap_path, bootstrap_backup = EvolveHelper._backup_bootstrap(
                agent.agent_code_dir
            )

            agent.git_controller._run_git_command(
                ["checkout", target_hash, "--", ".",
                 ":(exclude)evolution", ":(exclude).evolution_context",
                 ":(exclude).meta_evolution_context"],
                check=False,
            )

            if bootstrap_backup is not None:
                EvolveHelper._restore_bootstrap(bootstrap_path, bootstrap_backup)

            self._clean_orphan_harness_files(target_hash)
        except Exception as e:
            agent._log(
                f"Warning: checkout target harness {target_hash[:7]} failed: {e}"
            )
            # Best-effort restore to HEAD, preserving evolution/.
            try:
                agent.git_controller._run_git_command(
                    ["checkout", "HEAD", "--", ".",
                     ":(exclude)evolution", ":(exclude).evolution_context",
                     ":(exclude).meta_evolution_context"],
                    check=False,
                )
            except Exception:
                pass

    def apply_version_switch(self, target_hash: str, hint: str = "",
                             merge_ops: List[Dict[str, Any]] = None) -> bool:
        """Switch working directory files to a different git version.

        Returns:
            True on success, False on failure. On failure a partial checkout
            is best-effort restored to HEAD before returning False, so the
            iteration continues from the current commit rather than silently
            landing on a half-switched working tree.
        """
        agent = self.agent
        if not target_hash:
            return False

        try:
            self._checkout_target_harness(target_hash)

            if merge_ops:
                self._apply_merge_ops(merge_ops)

            agent.git_controller._run_git_command(["add", "-A"])

            commit_msg = f"[Archive version switch] -> {target_hash[:7]}"
            if hint:
                commit_msg += f" ({hint})"

            agent.git_controller.create_evolution_commit(
                iteration=agent.iteration,
                message=commit_msg,
                files=None,
            )

            agent._log(f"  Archive: switched to version {target_hash[:7]}"
                      f"{f' (hint: {hint})' if hint else ''}")
            return True
        except Exception as e:
            agent._log(f"Warning: version switch to {target_hash[:7]} failed: {e}")
            return False

    def apply_merge_ops(self, merge_ops: List[Dict[str, Any]]) -> None:
        """Apply merge operations without a version switch."""
        if not merge_ops:
            return
        try:
            self._apply_merge_ops(merge_ops)
            self.agent.git_controller._run_git_command(["add", "-A"])
            self.agent.git_controller.create_evolution_commit(
                iteration=self.agent.iteration,
                message="[Archive merge-ops]",
                files=None,
            )
        except Exception as e:
            self.agent._log(f"Warning: apply_merge_ops failed: {e}")

    # -----------------------------------------------------------------
    # INIT commit — unified meta fold-in + version switch
    # -----------------------------------------------------------------

    def create_init_commit(self, seed_info: dict | None = None) -> str | None:
        """Create the INIT commit that opens the next main iteration.

        Replaces the old ``apply_version_switch()`` + ``apply_merge_ops()``
        pattern with a single tracked commit that carries
        ``operation_type="init"``.  The caller (``GodelAgent.evolve()``) is
        expected to have already called ``_fold_in_staged_meta_changes()``
        before seed selection — this method picks up from there:

        1. If *seed_info* carries a ``git_hash`` different from HEAD, checks
           out that seed's harness files (preserving ``evolution/``).
        2. Amends the fold-in commit (when present) into the INIT commit, or
           creates a fresh INIT commit when no fold-in was made.
        3. Records the commit in the tracker with ``operation_type="init"``
           and optionally ``seed_eval_reward`` / ``seed_eval_mode``.
        4. Adds a knowledge-graph node (when KG is enabled).

        Args:
            seed_info: The dict returned by ``archive_manager.select_seed()``,
                or None.  Fields used: ``git_hash``, ``strategy_hint``,
                ``hypothesis``, ``merge_ops``, ``seed_eval_reward``,
                ``seed_eval_mode``.

        Returns:
            The INIT commit hash, or None on failure.
        """
        agent = self.agent
        iteration = agent.iteration
        head = agent.git_controller.get_current_commit() or ""
        if not head:
            return None

        seed_info = seed_info or {}
        target = seed_info.get("git_hash", "")
        merge_ops = seed_info.get("merge_ops", [])
        hint = seed_info.get("strategy_hint", "")
        seed_eval_reward = seed_info.get("seed_eval_reward")
        seed_eval_mode = seed_info.get("seed_eval_mode")

        # ── No-op detection: record anyway so every iteration has a
        # corresponding -N INIT record (preserves seed-selection hypothesis
        # + parent linkage for tree/lineage queries).
        _fold_msg = agent.git_controller.get_commit_message("HEAD") or ""
        _can_amend = _fold_msg.startswith("[Meta fold-in]")
        _needs_switch = bool(target and target != head)
        _is_noop = not _needs_switch and not _can_amend and not merge_ops
        if _is_noop:
            agent._log(
                f"  INIT: no-op (seed=HEAD, no fold-in, no merge_ops) — recording anyway"
            )

        # ── Phase banner ──────────────────────────────────────────

        _info_parts = []
        if target:
            _info_parts.append(f"Seed: {target[:7]}")
            if hint:
                _info_parts[-1] += f" ({hint})"
        else:
            _info_parts.append("Seed: HEAD (no switch)")
        _info_parts.append(
            "Fold-in: " + ("amended" if _can_amend else "fresh commit")
        )
        if seed_eval_reward is not None:
            _r = f"{seed_eval_reward:.4f}" if isinstance(seed_eval_reward, (int, float)) else str(seed_eval_reward)
            _info_parts.append(f"Eval: {_r}" + (f" ({seed_eval_mode})" if seed_eval_mode else ""))
        _info = "  " + " | ".join(_info_parts)

        log_format.log_phase_banner(
            agent,
            f"INIT COMMIT for iter {iteration}",
            info=_info,
            color=_C.BCY,
        )

        # ── Phase 1: Version switch (if seed target differs from HEAD) ──
        tree_changed = False
        if not _is_noop:
            if target and target != head:
                self._checkout_target_harness(target)
                tree_changed = True
            if merge_ops:
                self._apply_merge_ops(merge_ops)
                tree_changed = True

        # ── Phase 2: Commit (amend fold-in, or create fresh) ──────────
        # Skip the actual commit when no-op — HEAD stays put. We still
        # record a tracker entry below so every iteration has a -N INIT.
        if not _is_noop:
            commit_msg = f"[INIT iter={iteration}]"
            if target:
                commit_msg += f" Seed: {target[:7]}"
                if hint:
                    commit_msg += f" ({hint})"

            if tree_changed:
                agent.git_controller._run_git_command(["add", "-A"])
            if _can_amend:
                agent.git_controller._run_git_command(
                    ["commit", "--amend", "-m", commit_msg], check=False
                )
            else:
                agent.git_controller._run_git_command(
                    ["commit", "--allow-empty", "-m", commit_msg], check=False
                )
            head = agent.git_controller.get_current_commit() or ""

        if not head:
            return None

        # ── Phase 3: Record in tracker ─────────────────────────────────
        # Determine git parent: walk back from INIT to the nearest non-meta
        # commit (typically the last MAIN iteration commit).
        try:
            result = agent.git_controller._run_git_command(
                ["rev-parse", head + "^"], check=False
            )
            raw_parent = result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            raw_parent = ""
        # Semantic parent: when a seed switch happened (target ≠ original
        # HEAD), the INIT is semantically a child of the seed, NOT the git
        # parent (which traces back through the meta chain). This keeps the
        # lineage tree correctly showing seed→INIT edges.
        # Defer _resolve_non_meta_commit — it walks git log and is only
        # needed in the rare case where target == head (no version switch).
        if target and target != head:
            semantic_parent = target
            git_parent = ""
        else:
            git_parent = self._resolve_non_meta_commit(raw_parent) if raw_parent else ""
            semantic_parent = git_parent

        metadata: Dict[str, Any] = {
            "operation_type": "init",
            "main_iteration": iteration,
        }
        if _is_noop:
            metadata["skipped"] = True
        if seed_info:
            metadata["seed_info"] = {
                k: v for k, v in seed_info.items()
                if k in ("strategy_hint", "git_hash", "hypothesis")
            }
        if seed_eval_reward is not None:
            metadata["seed_eval_reward"] = seed_eval_reward
        if seed_eval_mode:
            metadata["seed_eval_mode"] = seed_eval_mode

        agent.evolution_tracker.record_iteration(
            iteration=-iteration,  # negative → auxiliary, not a main iter
            parent_commit=semantic_parent,
            new_commit=head,
            reward=seed_eval_reward if seed_eval_reward is not None else 0.0,
            state_summary=f"INIT commit for iteration {iteration}",
            action_count=0,
            metadata=metadata,
        )
        _parent_info = (
            f"seed={target[:7]}" if (target and target != head)
            else f"parent={semantic_parent[:7] if semantic_parent else '?'}"
        )
        agent._log(
            f"  INIT: {head[:7]} recorded ({_parent_info}"
            + (f", seed_eval={seed_eval_reward:.4f}" if seed_eval_reward is not None else "")
            + (", skipped" if _is_noop else "")
            + ")"
        )

        # ── Phase 4: Knowledge-graph node ──────────────────────────────
        # Skip KG node creation on no-op INITs: HEAD is unchanged so a node
        # for this hash already exists (or will be created by the actual
        # commit on a future iteration).
        if not _is_noop:
            kg = getattr(agent, "_knowledge_graph", None)
            if kg is not None:
                try:
                    kg.add_node(
                        iteration=iteration,
                        git_hash=head,
                        parent_hash=semantic_parent or "",
                        reward=(
                            seed_eval_reward
                            if seed_eval_reward is not None else 0.0
                        ),
                        eval_mode=seed_eval_mode or "",
                        summary_text=(
                            f"INIT commit for iteration {iteration}"
                            + (f" (seed: {target[:7]})" if target else "")
                        ),
                        modified_files=[],
                        change_tags=["init"],
                        is_meta=False,
                    )
                except Exception as e:
                    agent._log(f"Warning: INIT KG add_node failed: {e}")

        return head

    def _apply_merge_ops(self, merge_ops: List[Dict[str, Any]]) -> None:
        """Apply merge operations: check out specific files from source commits."""
        agent = self.agent
        by_source: Dict[str, List[str]] = {}
        for op in merge_ops:
            src = op.get("source_hash", "")
            files = op.get("files", [])
            if not src or not files:
                continue
            by_source.setdefault(src, []).extend(files)

        for src, files in by_source.items():
            try:
                agent.git_controller._run_git_command(
                    ["checkout", src, "--", *files]
                )
                agent._log(f"    Merge: {', '.join(files)} <- {src[:7]}")
            except Exception as e:
                agent._log(f"    Warning: merge from {src[:7]} failed: {e}")

    def _target_tracked_files(self, target_hash: str) -> set:
        """Return repo-relative file paths tracked at target_hash (forward-slash).

        Empty set on any git failure — callers treat empty as "skip cleanup".
        """
        agent = self.agent
        result = agent.git_controller._run_git_command(
            ["ls-tree", "-r", "--name-only", target_hash], check=False
        )
        if result.returncode != 0:
            return set()
        return {
            line.strip().replace("\\", "/")
            for line in result.stdout.splitlines()
            if line.strip()
        }

    def _clean_orphan_harness_files(self, target_hash: str) -> None:
        """Remove harness .py/.md files not present at target_hash.

        ``git checkout <target> -- .`` restores target's files into the working
        tree but leaves files created in later commits lingering — they'd then be
        committed into the new iteration via ``git add -A``, so the seed wouldn't
        actually be the target version. This sweep deletes those orphans.

        Mirrors ``_restore_code_snapshot``'s cleanup: excludes evolution/
        (meta-evolve territory), hidden dirs, __pycache__, and BOOTSTRAP.md
        (preserved across iterations). Fail-safe — if the target tree can't be
        read, delete nothing rather than risk removing wanted files.
        """
        agent = self.agent
        target_files = self._target_tracked_files(target_hash)
        if not target_files:
            short = target_hash[:7] if target_hash else "?"
            agent._log(f"  Archive: skip orphan cleanup (could not read tree at {short})")
            return

        agent_code_dir = agent.agent_code_dir
        removed = []
        for root, dirs, files in os.walk(agent_code_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.')
                       and d != '__pycache__' and d != 'evolution']
            for fname in files:
                # Skip harness-root metadata notebooks — they describe the
                # evolution, not the version being switched to.
                if fname in METADATA_FILES:
                    continue
                if not (fname.endswith('.py') or fname.endswith('.md')):
                    continue
                full_path = os.path.join(root, fname)
                rel = os.path.relpath(full_path, agent_code_dir).replace(os.sep, '/')
                if rel not in target_files:
                    try:
                        os.remove(full_path)
                        removed.append(rel)
                    except OSError:
                        pass
        if removed:
            agent._log(
                f"  Archive: removed {len(removed)} orphan file(s) absent from seed "
                f"{target_hash[:7]}: {removed[:10]}"
            )
