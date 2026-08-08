"""
Agent Action Executor - Execute actions for the Godel Agent.

This module contains the execution logic for all agent actions:

Built-in core tools (5):
- bash: Execute shell commands; auto-syncs disk changes into the agent_codes mirror
- read_history_self: Read iteration history (@history syntax)
- read_file: Read file content or list directory
- edit_file: Precise string replacement in files
- write_file: Create or overwrite files

Required external tool (1):
- evaluate: External evaluation (collect r_t)

Optional external tools (dynamic):
- execute_external_tool: Run dynamically loaded tools
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple, TYPE_CHECKING

from ..utils import (
    CodeValidator,
    ValidationResult,
)
from ..state import fmt_reward, METADATA_FILES
from ..evolve import _parse_snapshot
from .agent_file_ops import FileOpsMixin
from .agent_history import HistoryMixin
from .agent_evaluator import EvaluatorMixin


if TYPE_CHECKING:
    from ..state import AgentState


# External evaluator signature: (agent_instance, harness_func, test_cases) -> (reward: float | dict, metrics: dict)
ExternalEvaluator = Callable[[Any, Callable, List[Any]], Tuple[float, Dict[str, Any]]]


# Module-level regex for parsing `## <Name>` sections in markdown files.
# Compiled once rather than on every call to parse_sections — plan() and
# lesson() each invoke this multiple times per iteration.
_PARSE_SECTIONS_RE = re.compile(r'^## (.+?)\s*$', re.MULTILINE)

# Module-level regex for parsing `[Iter N|conf=X.XX] lesson text` lines in
# BOOTSTRAP.md. Shared by _merge_cumulative_lesson (agent_action) and
# _build_lesson_audit_section (meta_evolve). Compiled once.
_LESSON_LINE_RE = re.compile(r'^\[Iter\s*(\d+)(?:\|conf=([\d.]+))?\](.*)$')


def parse_sections(content: str, section_names: List[str]) -> dict:
    """Parse ``## <Name>`` sections from markdown content.

    Returns ``{name_lower: body}`` where *body* is the text between the
    ``## <Name>`` header and the next ``## `` header (or EOF), stripped.
    Only sections listed in *section_names* are matched; unmatched ``## ``
    headers are ignored.

    This is a public module-level function — import it directly for
    cross-module use (e.g. from ``seed_selection.py``) instead of reaching
    into ``AgentActionExecutor._parse_sections``.
    """
    sections = {n.lower(): "" for n in section_names}
    # Build a set for O(1) membership, then filter: only matches whose
    # group(1) (the header text) is in the requested set.
    name_set = frozenset(n.lower() for n in section_names)
    matches = list(_PARSE_SECTIONS_RE.finditer(content))
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        if name.lower() not in name_set:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections[name.lower()] = content[start:end].strip()
    return sections


class AgentActionExecutor(FileOpsMixin, HistoryMixin, EvaluatorMixin):
    """
    Executor for agent actions.

    Handles the actual execution of each action type, updating
    the agent state accordingly.

    Core capabilities:
    - edit/write directly to disk; evaluate fresh-imports from disk each call (guarantees runtime bytes == code_hash)
    - evaluate supports external evaluators
    - read_self reads specified targets: source/files/history
    """

    def __init__(
        self,
        llm_client,
        model: str,
        repo_path: Path,
        agent_code_dir: str,
        tools: List[Dict[str, Any]] = None,
        logging: Callable = print,
        external_evaluator: ExternalEvaluator = None,
        agent_instance: Any = None,
        context_persistence: Any = None,
    ):
        """
        Initialize the action executor.

        Args:
            llm_client: OpenAI-compatible LLM client.
            model: Model name to use.
            repo_path: Path to the repository.
            agent_code_dir: Path to the agent's code directory.
            tools: List of available external tools.
            logging: Logging function.
            external_evaluator: External evaluation function.
            agent_instance: Agent instance.
            context_persistence: ContextPersistence instance.
        """
        self.llm_client = llm_client
        self.model = model
        self.repo_path = Path(repo_path)
        self.agent_code_dir = agent_code_dir
        self.tools = tools or []
        self.logging = logging
        self.external_evaluator = external_evaluator
        self.agent_instance = agent_instance
        self.context_persistence = context_persistence

        # Mutable state (updated during execution)
        self.agent_codes: Dict[str, str] = {}  # {relative_path: code}
        self.state: Optional["AgentState"] = None

        # Track modified files
        self._modified_files: set = set()

        # Subdirectory names to hide from directory listings
        self._restricted_dirs: set = set()

        # Track evaluate call count (for log saving)
        self._evaluate_count: int = 0

        # File hash tracking (detect disk changes after bash -> sync agent_codes mirror)
        self._file_hashes: Dict[str, str] = {}  # {rel_path: md5_hash}

    @property
    def validator(self) -> CodeValidator:
        """Get or create a CodeValidator instance (lazily reused)."""
        if not hasattr(self, '_validator') or self._validator is None:
            self._validator = CodeValidator()
        return self._validator

    def _validate_python_code(self, content: str, filename: str) -> Optional[str]:
        """Validate Python code safety and syntax. Returns error string or None."""
        val_result = self.validator.validate(content)
        if not val_result.valid:
            violations = ', '.join(val_result.violations) if val_result.violations else ''
            return f"[{filename}] Validation FAILED: {val_result.message}. {violations}"
        try:
            compile(content, filename, "exec")
        except SyntaxError as e:
            return f"[{filename}] Syntax error: {e}"
        return None

    def _compute_file_hash(self, filepath: str) -> str:
        """Compute the MD5 hash of a file."""
        import hashlib
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def _compute_code_hash(self, return_contents: bool = False):
        """Compute the aggregate MD5 hash of all code files under agent_code_dir.

        Args:
            return_contents: If True, also return {rel_path: content} dict
                alongside the hash, avoiding a second file walk.

        Returns:
            code_hash string, or (code_hash, contents_dict) tuple if return_contents=True.
        """
        import hashlib
        if not self.agent_code_dir or not os.path.isdir(self.agent_code_dir):
            return ("", {}) if return_contents else ""

        file_hashes = []
        contents = {} if return_contents else None
        for root, dirs, files in os.walk(self.agent_code_dir):
            # Exclude evolution/ — it's meta-evolve territory, not part of the
            # harness "version" being evaluated/committed. _restore_code_snapshot
            # already excludes it on both write-back and cleanup; the hash must
            # match that scope, otherwise the committed code's content (which
            # keeps the current evolution/ tree) wouldn't equal the snapshot the
            # reward was attributed to, misaligning (commit, reward) pairs
            # whenever evolution/ changes mid-iteration (meta-evolve / seed
            # checkout).
            dirs[:] = [d for d in dirs if not d.startswith('.')
                       and d != '__pycache__' and d != 'evolution']
            for f in files:
                if not f.endswith('.py') and not f.endswith('.md'):
                    continue
                if f in METADATA_FILES:
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, self.agent_code_dir)
                file_hashes.append((rel_path, self._compute_file_hash(full_path)))
                if return_contents:
                    try:
                        with open(full_path, 'r', encoding='utf-8') as fh:
                            contents[rel_path] = fh.read()
                    except Exception:
                        pass

        file_hashes.sort(key=lambda x: x[0])

        h = hashlib.md5()
        for rel_path, fhash in file_hashes:
            h.update(f"{rel_path}:{fhash}".encode())

        code_hash = h.hexdigest()
        return (code_hash, contents) if return_contents else code_hash

    def _scan_py_files(self) -> Dict[str, str]:
        """Scan agent_code_dir and compute hashes for .py + .md files.

        Aligns file types with _compute_code_hash (.py + .md, excluding METADATA_FILES
        i.e. BOOTSTRAP.md / plan.md) so that bash/git-modified .md files are also
        detected by _auto_reload_if_changed and recorded into _modified_files (the .md
        files edited via the tool path are already tracked directly). evolution/ is
        still included — this way bash edits to select_*.py / strategies can be
        recorded into _modified_files (for commit summary / KG tracking). evaluate()
        always fresh-imports from disk regardless, so this only affects change tracking
        and not runtime bytes (see test_scan_py_files_includes_evolution_dir).
        """
        hashes = {}
        if not self.agent_code_dir or not os.path.isdir(self.agent_code_dir):
            return hashes

        for root, _, files in os.walk(self.agent_code_dir):
            # Skip hidden directories and __pycache__
            root_rel = os.path.relpath(root, self.agent_code_dir)
            if root_rel != '.' and (root_rel.startswith('.') or '__pycache__' in root_rel):
                continue

            for f in files:
                if f.endswith('.py'):
                    if f.startswith('_'):
                        continue
                elif f.endswith('.md'):
                    if f in METADATA_FILES:
                        continue
                else:
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, self.agent_code_dir)
                hashes[rel_path] = self._compute_file_hash(full_path)
        return hashes

    def _reload_single_file(self, rel_path: str, content: str = None, skip_validation: bool = False) -> str:
        """Sync a single .py file into the in-memory ``agent_codes`` cache.

        The historical name ``_reload_single_file`` is retained (edit_file/write_file/auto-detect
        all call it by this name), but it now **no longer hot-reloads anything**: evaluate()
        fresh-imports the harness from disk each time, so disk is the source of truth.
        It only does two things here —
        1. (optional) AST validation, providing immediate feedback to the editor;
        2. update the ``agent_codes`` mirror (the code view of the evolve prompt,
           ``_harness_source`` injection, and KG modified_files all depend on it).
        Returns an empty string on success, an error string on validation failure.
        """
        # Only handle .py files
        if not rel_path.endswith('.py'):
            return ""

        full_path = os.path.join(self.agent_code_dir, rel_path)

        # Read content (if not passed in)
        if content is None:
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                return f"[{rel_path}] Failed to read: {e}"

        # Validate (if needed)
        if not skip_validation:
            error = self._validate_python_code(content, rel_path)
            if error:
                return error

        # Sync the in-memory mirror (consumers: code view in evolve.py, _harness_source in adapter.py)
        self.agent_codes[rel_path] = content
        return ""

    def _auto_reload_if_changed(self) -> List[str]:
        """Sync disk changes into ``agent_codes`` / ``_modified_files`` after a bash command.

        Detects disk changes via hash comparison (no more hot-reloading — evaluate
        fresh-imports from disk each time):
        - .py changes: read from disk -> AST validate -> refresh the ``agent_codes`` mirror
          (validation errors are reported back).
        - .md and other non-module changes: only recorded into ``_modified_files``
          (for commit summary / KG tracking).
        - Tool file changes: trigger the corresponding scope's tool rescan.
        """
        current_hashes = self._scan_py_files()
        modified = []

        # Find modified files
        for path, hash_val in current_hashes.items():
            old_hash = self._file_hashes.get(path)
            if old_hash != hash_val:
                modified.append(path)

        if not modified:
            return []

        results = []
        for rel_path in modified:
            # Only .py is a module and needs agent_codes refresh; .md and other non-module
            # files are recorded into _modified_files below (for commit summary / KG tracking),
            # not into agent_codes.
            if not rel_path.endswith('.py'):
                continue
            result = self._reload_single_file(rel_path, skip_validation=False)
            if result:
                results.append(result)
            else:
                results.append(f"[{rel_path}] synced to cache")

        # Update the hash cache
        self._file_hashes = current_hashes

        # Update the modified files set (used by git commit)
        if modified:
            self._modified_files.update(modified)
            self._rescan_external_tools(" after disk change", modified_files=modified)

        return results

    def set_state(self, state: "AgentState", codes: Dict[str, str] = None) -> None:
        """Set the current execution context."""
        self.state = state
        if codes is not None:
            self.agent_codes = codes
            self.state.update_pi(codes=self.agent_codes, code_dir=self.agent_code_dir)

    # Tool file name -> scope map (single-file mode only)
    TOOL_FILE_MAP = {
        "tools_evolve.py": "evolve",
        "tools_harness.py": "harness",
        "evolution/evolution_tools.py": "evolve",
    }

    def _rescan_external_tools(self, context: str = "", modified_files: list = None) -> None:
        """Only rescan the corresponding scope when a tool file is modified."""
        if modified_files is None or not self.agent_instance or not hasattr(self.agent_instance, '_scan_external_tools'):
            return
        # Get the current mode
        mode = getattr(getattr(self.agent_instance, 'config', None), 'mode', 'evolve')
        scopes_to_rescan = set()
        for f in modified_files:
            basename = os.path.basename(f)
            if basename in self.TOOL_FILE_MAP:
                scope = self.TOOL_FILE_MAP[basename]
                # In evolve mode, only rescan evolve tools
                if mode == "evolve" and scope != "evolve":
                    continue
                scopes_to_rescan.add(scope)
        if not scopes_to_rescan:
            return
        try:
            for scope in scopes_to_rescan:
                self.agent_instance._scan_external_tools(scope=scope)
                self.logging(f"Rescanned {scope} tools{context}")
        except Exception as e:
            self.logging(f"Warning: Failed to rescan tools: {e}")

    def load_codes(self) -> Dict[str, str]:
        """Load all .py files under the code directory."""
        if not self.agent_code_dir or not os.path.isdir(self.agent_code_dir):
            return {}

        self.agent_codes = {}

        for root, dirs, files in os.walk(self.agent_code_dir):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

            for file in files:
                if not file.endswith('.py'):
                    continue

                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.agent_code_dir)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.agent_codes[relative_path] = f.read()
                except Exception as e:
                    self.logging(f"Warning: Failed to load {file_path}: {e}")

        # Update state
        if self.state:
            self.state.update_pi(codes=self.agent_codes, code_dir=self.agent_code_dir)

        return self.agent_codes

    def _maybe_truncate(self, content: str, max_length: int = None) -> str:
        """Truncate long content and add a marker."""
        if max_length is None:
            if self.agent_instance and hasattr(self.agent_instance, 'config'):
                max_length = getattr(self.agent_instance.config, 'max_tool_result_length', 100000)
            else:
                max_length = 100000
        if len(content) > max_length:
            return (
                content[:max_length]
                + "\n\n[Tool result truncated due to length limit. "
                + "The original output was too long to display fully.]"
            )
        return content

    # Short-message actions that do not need truncation
    _NO_TRUNCATE_ACTIONS = frozenset({
        "compact_context", "end_evolution", "end_meta_evolution", "plan", "lesson",
        "validate_archive", "end_probe",
    })

    # Required params per action type (for validation before execution)
    _REQUIRED_PARAMS = {
        "bash": ["command"],
        "powershell": ["command"],
        "read_file": ["path"],
        "edit_file": ["path", "old_string"],
        "write_file": ["path", "content"],
    }

    def execute(self, action_type: str, params: Dict[str, Any]) -> str:
        """Execute an action based on type."""
        # Validate required params
        required = self._REQUIRED_PARAMS.get(action_type)
        if required:
            missing = [p for p in required if params.get(p) is None]
            if missing:
                return f"Error: Missing required parameter(s): {', '.join(missing)}. Received params: {list(params.keys())}"

        if action_type in ("bash", "powershell"):
            result = self.execute_shell(
                command=params.get("command"),
                timeout=params.get("timeout")
            )

        elif action_type == "read_history_self":
            result = self.read_history_self(
                targets=params.get("targets")
            )

        elif action_type == "evaluate":
            result = self.evaluate(
                test_cases=params.get("test_cases"),
                func_names=params.get("func_names"),
                eval_mode=params.get("eval_mode", "dev"),
                num_tasks=params.get("num_tasks"),
                task_ids=params.get("task_ids"),
            )

        elif action_type == "external_tool":
            result = self.execute_external_tool(
                params.get("tool_name"),
                params.get("arguments", {})
            )

        elif action_type == "get_historic_version":
            result = self.get_historic_version(
                iteration=params.get("iteration"),
                file_name=params.get("file")
            )

        elif action_type == "read_file":
            result = self.read_file(path=params.get("path"))

        elif action_type == "edit_file":
            result = self.edit_file(
                path=params.get("path"),
                old_string=params.get("old_string"),
                new_string=params.get("new_string", ""),
                replace_all=params.get("replace_all", False)
            )

        elif action_type == "write_file":
            result = self.write_file(
                path=params.get("path"),
                content=params.get("content"),
                create_only=params.get("create_only", False)
            )

        elif action_type == "compact_context":
            result = self.compact_context(
                summary=params.get("summary", ""),
                reason=params.get("reason", ""),
            )

        elif action_type == "end_evolution":
            result = self.end_evolution(
                summary=params.get("summary", ""),
                reason=params.get("reason", ""),
            )

        elif action_type == "end_meta_evolution":
            result = self.end_meta_evolution(
                summary=params.get("summary", ""),
                reason=params.get("reason", "")
            )

        elif action_type == "plan":
            result = self.plan(
                plan=params.get("plan", ""),
                progress=params.get("progress", ""),
            )

        elif action_type == "lesson":
            result = self.lesson(
                lesson=params.get("lesson", ""),
                confidence=params.get("confidence", 0.5),
                iteration=params.get("iteration"),
            )

        elif action_type == "meta_bootstrap":
            result = self.meta_bootstrap(
                what=params.get("what", ""),
                why=params.get("why", ""),
                lesson=params.get("lesson", ""),
                prediction=params.get("prediction", ""),
            )

        elif action_type == "validate_archive":
            result = self.validate_archive()

        elif action_type == "end_probe":
            result = self.end_probe(
                findings=params.get("findings", ""),
            )

        elif action_type == "get_historic_eval_code":
            result = self.get_historic_eval_code(
                eval_index=params.get("eval_index"),
                file_name=params.get("file")
            )

        else:
            return f"Unknown action type: {action_type}"

        # Uniformly truncate long results (except for short-message actions)
        if action_type not in self._NO_TRUNCATE_ACTIONS:
            result = self._maybe_truncate(result)
        return result

    def compact_context(self, summary: str = "", reason: str = "") -> str:
        """End the current iteration and compact context."""
        if self.state:
            self.state.mark_iteration_ended(summary=summary, reason=reason)
        return f"Iteration ended. Summary: {summary}. Reason: {reason}. Changes will be committed."

    def end_evolution(self, summary: str = "", reason: str = "") -> str:
        """End the entire evolution process."""
        if self.state:
            self.state.mark_evolution_ended(summary=summary, reason=reason)
        return f"Evolution ended. Summary: {summary}. Reason: {reason}. This is the FINAL iteration."

    def end_meta_evolution(self, summary: str = "", reason: str = "") -> str:
        """End the current meta-evolution phase (does not stop the main evolution loop)."""
        return f"Meta-evolution phase ended. Summary: {summary}. Reason: {reason}. The main evolution will continue to the next iteration."

    def plan(self, plan: str = "", progress: str = "") -> str:
        """Update the iteration-scoped working notebook (plan.md).

        plan.md is **ephemeral**: cleared at the start of each iteration and seeded
        with the ``## Hypothesis`` section from seed-selection (framework-owned).
        This tool only merges the ``## Plan`` / ``## Progress`` sections (non-empty
        overwrites, empty preserves); the ``## Hypothesis`` section is preserved as-is.

        plan.md is gitignored (not in git history) and excluded from code_hash /
        file scanning — it is working memory, not the code version being evaluated/
        committed, so it is **not** recorded into ``_modified_files``; only
        ``modifications_made`` (operation="plan") is recorded to reflect agent activity.

        Returns the full merged plan.md content (immediate confirmation) rather than
        a one-line acknowledgment.
        """
        plan_path = os.path.join(self.agent_code_dir, "plan.md")

        # Short-circuit: both empty → no modifications; just return the current
        # content (avoids the full read-parse-write cycle for no-op calls).
        if not plan.strip() and not progress.strip():
            if os.path.exists(plan_path):
                try:
                    with open(plan_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception:
                    pass
            return ""

        existing = ""
        if os.path.exists(plan_path):
            try:
                with open(plan_path, 'r', encoding='utf-8') as f:
                    existing = f.read()
            except Exception:
                pass

        sections = self._parse_sections(existing, ["Hypothesis", "Plan", "Progress"])
        hypothesis = sections.get("hypothesis", "")
        # plan / progress: non-empty overwrites, empty preserves existing
        merged_plan = plan.strip() or sections.get("plan", "")
        merged_progress = progress.strip() or sections.get("progress", "")

        parts = []
        if hypothesis:
            parts.append(f"## Hypothesis\n{hypothesis}")
        if merged_plan:
            parts.append(f"## Plan\n{merged_plan}")
        if merged_progress:
            parts.append(f"## Progress\n{merged_progress}")
        content = ("\n\n".join(parts) + "\n") if parts else ""

        try:
            with open(plan_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            return f"Error saving plan.md: {e}"

        # plan.md is ephemeral working memory (gitignored) and does not enter _modified_files —
        # it is not a code change and should not enter git/KG/commit-summary tracking. Only activity is recorded.
        if self.state is not None:
            self.state.modifications_made.append({
                "operation": "plan",
                "file": "plan.md",
            })

        return content

    def lesson(self, lesson: str = "", confidence: float = 0.5,
               iteration: Optional[int] = None) -> str:
        """Write the ``## Lesson`` section of BOOTSTRAP.md (cumulative across iterations, never rolled back).

        BOOTSTRAP.md is now **lesson-only** (no longer the Plan/Progress/Lesson three-section format).
        Exactly one ``[Iter N|conf=X.XX]`` aggregated lesson per iteration, accumulating across
        iterations; it is the only experience thread inherited by the next iteration + seed selection
        (see ``_merge_cumulative_lesson``).

        - ``iteration``: which iteration's lesson line to write. Default None = the current iteration
          (``self.state.iteration``). Passing N explicitly directly overwrites the ``[Iter N]`` line,
          used by meta-evolve to correct previous iterations' wrong lessons.
        - ``confidence``: your confidence in this lesson (0.0-1.0). 0.8+ = confirmed by trace evidence.
          0.5 = plausible but unverified. 0.2-0.4 = speculative direction. Default 0.5.
        - Multiple calls within the same iteration → rewrite the same iter's line (whether current
          iter or explicitly specified).

        Calling marks ``state.lesson_recorded = True``, driving the iteration-end lesson-nudge
        fallback (see ``EvolveHelper._ensure_lesson_recorded``).

        Returns the full merged BOOTSTRAP.md content (immediate confirmation + equivalent notebook snapshot).
        """
        bootstrap_path = os.path.join(self.agent_code_dir, "BOOTSTRAP.md")

        existing_lesson = ""
        if os.path.exists(bootstrap_path):
            try:
                with open(bootstrap_path, 'r', encoding='utf-8') as f:
                    sections = self._parse_sections(f.read(), ["Lesson"])
                    existing_lesson = sections.get("lesson", "")
            except Exception:
                pass

        iter_num = iteration if iteration is not None else (
            self.state.iteration if self.state is not None else 0
        )
        merged_lesson = self._merge_cumulative_lesson(
            existing_lesson, lesson, iter_num, confidence=confidence,
        )

        # BOOTSTRAP.md now only keeps the ## Lesson section (lesson-only)
        content = f"## Lesson\n{merged_lesson}\n"
        try:
            with open(bootstrap_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            return f"Error saving BOOTSTRAP.md: {e}"

        # Add to the modified files set -> included in git commit (BOOTSTRAP.md is tracked + protected)
        if self.agent_code_dir:
            rel_path = os.path.relpath(bootstrap_path, self.agent_code_dir)
            self._modified_files.add(rel_path)

            if self.state is not None:
                self.state.modifications_made.append({
                    "operation": "lesson",
                    "file": rel_path,
                })
                # Mark that the agent recorded a lesson this iteration —
                # suppresses the iteration-end lesson-nudge fallback.
                self.state.lesson_recorded = True

        return content

    @staticmethod
    def _merge_cumulative_lesson(
        existing_lesson: str, new_lesson: str, iteration: int,
        confidence: float = 0.5,
    ) -> str:
        """Merge the lesson section into the cumulative format: exactly one ``[Iter N|conf=X.XX]`` aggregated lesson per iteration.

        - Parses the existing lesson line by line into ``[Iter N]`` or ``[Iter N|conf=X.XX]``.
        - If ``[Iter {iteration}]`` already exists: replace with the latest aggregation (multiple
          updates within an iteration -> always keep the latest line for that iter; passing an
          iteration explicitly can directly overwrite a historical iter).
        - Otherwise append a new line ``[Iter {iteration}|conf={confidence:.2f}] {new_lesson}`` (new iter settles).
        - Old text that does not match ``[Iter N]`` is preserved as a legacy preamble (backward compatible with the old format).
        - When ``new_lesson`` is empty, return existing unchanged (no rewrite).
        - Old format ``[Iter N]`` (no |conf=) -> parsed as conf=0.50 (neutral default).
        """
        new_lesson = (new_lesson or "").strip()
        if not new_lesson:
            return existing_lesson

        # group(1)=iteration, group(2)=confidence (None→0.50), group(3)=lesson text
        line_re = _LESSON_LINE_RE
        iter_lines: Dict[int, str] = {}
        order: List[int] = []
        legacy: List[str] = []
        for line in (existing_lesson or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            m = line_re.match(stripped)
            if m:
                n = int(m.group(1))
                if n not in iter_lines:
                    order.append(n)
                iter_lines[n] = stripped
            else:
                legacy.append(stripped)

        new_line = f"[Iter {iteration}|conf={confidence:.2f}] {new_lesson}"
        if iteration in iter_lines:
            iter_lines[iteration] = new_line
        else:
            order.append(iteration)
            iter_lines[iteration] = new_line

        parts = list(legacy) + [iter_lines[n] for n in sorted(iter_lines.keys())]
        return "\n".join(parts)

    @staticmethod
    def _parse_sections(content: str, section_names: List[str]) -> dict:
        """Delegate to the module-level ``parse_sections`` (shared with
        ``seed_selection._build_lessons_block`` via direct import)."""
        return parse_sections(content, section_names)

    def meta_bootstrap(self, what: str = "", why: str = "", lesson: str = "", prediction: str = "") -> str:
        """Cumulative meta-evolve notebook (evolution/meta_bootstrap.md).

        Persistent across meta-evolves: lives under evolution/, persisted by meta commits,
        not rolled back with the main iteration git.
        At the end of each meta-evolve, **append** one Meta#N record (newest on top),
        recording what/why/lesson/prediction.
        At the start of the next meta-evolve, read first to see the full history + verify the
        last prediction, closing the loop.
        """
        import re

        bootstrap_path = os.path.join(self.agent_code_dir, "evolution", "meta_bootstrap.md")

        # Number N = the just-completed main iteration (meta-evolve runs after main iter N)
        agent = self.agent_instance
        iteration = getattr(agent, "iteration", None) if agent else None
        meta_num = iteration if isinstance(iteration, int) else "?"

        # reward = the latest non-meta main record's reward (meta-evolve uses a throwaway state and cannot take reward from state.reward)
        reward_str = "N/A"
        if agent is not None:
            tracker = getattr(agent, "evolution_tracker", None)
            records = getattr(tracker, "records", None) if tracker is not None else None
            if records:
                for rec in reversed(records):
                    meta = getattr(rec, "metadata", None) or {}
                    if meta.get("type") == "meta_evolve":
                        continue
                    # Skip non-main-line records: meta_evolve, crossover (backward compat), ensemble
                    if meta.get("operation_type") == "crossover":
                        continue
                    if meta.get("operation_type") == "ensemble":
                        continue
                    reward_str = fmt_reward(rec.primary_reward() if hasattr(rec, "primary_reward") else 0.0)
                    break

        # Build the new record (newest on top, so the next start can verify the last prediction at a glance)
        new_entry = (
            f"## Meta #{meta_num} (after iter {meta_num}, reward={reward_str})\n"
            f"### What\n{what.strip() or '(none)'}\n\n"
            f"### Why\n{why.strip() or '(none)'}\n\n"
            f"### Lesson\n{lesson.strip() or '(none)'}\n\n"
            f"### Prediction\n{prediction.strip() or '(none)'}"
        )

        # Read existing content, split out existing `## Meta #` records + legacy residue (old Plan/Progress/Lesson format, backward compatible)
        existing_blocks = []
        legacy = ""
        if os.path.exists(bootstrap_path):
            try:
                with open(bootstrap_path, "r", encoding="utf-8") as f:
                    existing = f.read()
            except Exception:
                existing = ""
            block_re = re.compile(r"^## Meta #[^\n]*\n(?:.*?)(?=^## Meta #|\Z)", re.MULTILINE | re.DOTALL)
            existing_blocks = [b.strip() for b in block_re.findall(existing)]
            legacy = block_re.sub("", existing).strip()

        # Prepend the new record to the top, cap the most recent 10 entries (prevents meta_bootstrap.md from bloating when read into the prompt)
        new_blocks = ([new_entry] + existing_blocks)[:10]

        body = "\n\n".join(new_blocks)
        if legacy:
            body += "\n\n" + legacy
        content = body + "\n"

        try:
            os.makedirs(os.path.dirname(bootstrap_path), exist_ok=True)
            with open(bootstrap_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            return f"Error saving meta_bootstrap: {e}"

        if self.agent_code_dir:
            rel_path = os.path.relpath(bootstrap_path, self.agent_code_dir)
            self._modified_files.add(rel_path)
            if self.state is not None:
                self.state.modifications_made.append({
                    "operation": "meta_bootstrap", "file": rel_path,
                })

        updated = []
        if what.strip(): updated.append("what")
        if why.strip(): updated.append("why")
        if lesson.strip(): updated.append("lesson")
        if prediction.strip(): updated.append("prediction")
        total = len(new_blocks)
        return (f"Meta-bootstrap appended: Meta #{meta_num} ({total} entries total). "
                f"Updated: {', '.join(updated) or 'no content'}")

    def validate_archive(self) -> str:
        """Validate the current select_*.py modules by dry-running select_seed()/select_commit()."""
        from ..archive_strategies import STRATEGY_REGISTRY

        agent = self.agent_instance
        if not agent or not hasattr(agent, 'archive_manager'):
            return "Error: archive_manager not available."

        archive_mgr = agent.archive_manager
        archive_mgr._strategies_discovered = False
        archive_mgr._ensure_strategies_discovered()

        validation = archive_mgr.validate_archive()
        if validation["valid"]:
            seed_info = validation.get("seed_info", {})
            commit_info = validation.get("commit_info")
            lines = [
                "Archive validation PASSED.",
                f"Dry-run select_seed() → git_hash={seed_info.get('git_hash', 'N/A')}, "
                f"strategy_hint={seed_info.get('strategy_hint', 'N/A')}",
            ]
            if commit_info is not None:
                lines.append(
                    f"select_commit.py get_commit_nudge_prompt() → "
                    f"prompt_ok={commit_info.get('prompt_ok', False)}, "
                    f"len={commit_info.get('len', 0)}"
                )
            else:
                lines.append("select_commit.py not present (evolvable_commit_strategy=false).")
            lines.append("All strategy names in _STRATEGY_SCHEDULE are registered.")
            return "\n".join(lines)
        parts = [f"Archive validation FAILED: {validation['error']}"]
        unknown = validation.get("unknown_strategies", [])
        if unknown:
            parts.append(
                f"Unknown strategies: {unknown}. "
                f"Registered: {list(STRATEGY_REGISTRY.keys())}"
            )
        return "\n".join(parts)

    def end_probe(self, findings: str = "") -> str:
        """Complete a probe investigation and return findings."""
        return f"Investigation complete.\n\n{findings}"

    def execute_shell(self, command: str, timeout: float = None, scope: str = "evolve") -> str:
        """Execute a shell command, auto-detect disk changes and sync them into the agent_codes mirror."""
        if not command:
            return "Error: 'command' parameter is required and cannot be empty."

        # Before execution: only record file hashes on the first call
        if not self._file_hashes:
            self._file_hashes = self._scan_py_files()

        # bwrap sandbox wrapping (auto-enabled when bwrap is available, silently skipped when not)
        final_command = self._wrap_bwrap(command, scope) if self._bwrap_available else command

        try:
            timeout_val = timeout or 120

            if sys.platform == 'win32':
                # Windows: PowerShell
                result = subprocess.run(
                    ['powershell', '-NoProfile', '-NonInteractive', '-Command', final_command],
                    capture_output=True,
                    text=True,
                    timeout=timeout_val,
                    cwd=str(self.repo_path),
                    encoding='utf-8',
                    errors='replace',
                )
            else:
                # Linux/macOS: bash
                result = subprocess.run(
                    final_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout_val,
                    cwd=str(self.repo_path),
                    encoding='utf-8',
                    errors='replace',
                )

            output = []
            if result.stdout:
                output.append(result.stdout)
            if result.stderr:
                # Filter common noise
                stderr = result.stderr
                noise_patterns = [
                    "Inappropriate ioctl for device",
                    "cannot set terminal process group",
                    "job control turned off",
                    "To update your account to follow the current behavior",
                    "See the documentation",
                ]
                for pattern in noise_patterns:
                    if pattern in stderr:
                        stderr = stderr.replace(pattern, "")
                if stderr.strip():
                    output.append(f"[stderr]:\n{stderr.strip()}")

            result_str = "\n".join(output) if output else "Command completed with no output."

            # After execution: sync disk changes into agent_codes / _modified_files (no more hot-reloading)
            reload_results = self._auto_reload_if_changed()

            if reload_results:
                result_str += "\n\n[Disk Change]:\n" + "\n".join(reload_results)

            return f"[Command]: {command}\n[Output]:\n{result_str}"

        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout or 120} seconds."
        except Exception as e:
            return f"Error: {str(e)}"

    @property
    def _bwrap_static_mounts(self) -> list:
        """bwrap static mount prefix (system base + Python + framework + network), invariant for the process lifetime."""
        if not hasattr(self, '_cached_bwrap_mounts'):
            framework = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
            src_dir = os.path.dirname(framework)  # src/ — for PYTHONPATH so Python can import react_loop
            self._cached_bwrap_mounts = [
                '--ro-bind', '/usr', '/usr',
                '--ro-bind', '/lib', '/lib',
                '--ro-bind', '/lib64', '/lib64',
                '--ro-bind', '/bin', '/bin',
                '--ro-bind', '/etc', '/etc',
                '--proc', '/proc',
                '--dev', '/dev',
                '--tmpfs', '/tmp',
                '--ro-bind', sys.prefix, sys.prefix,
                '--ro-bind', framework, framework,
                '--setenv', 'PYTHONPATH', src_dir,
                '--share-net',
            ]
        return self._cached_bwrap_mounts

    @property
    def _bwrap_available(self) -> bool:
        """Whether bwrap is available (lazily detected, only once)."""
        if not hasattr(self, '_bwrap_bin'):
            self._bwrap_bin = shutil.which('bwrap')
        return self._bwrap_bin is not None

    def _wrap_bwrap(self, command: str, scope: str) -> str:
        """Wrap a bash command with bwrap to restrict filesystem visibility.

        evolve/meta_evolve/atom/submit_best scope:
          - repo/ rw, eval_logs/ ro, src/react_loop/ ro, system base ro
        harness scope:
          - repo/ ro, eval_logs/ rw, src/react_loop/ ro, system base ro
        """
        repo = str(self.repo_path)
        run_dir = os.path.dirname(repo)
        eval_logs = os.path.join(run_dir, 'eval_logs')

        # Mount the run directory according to scope
        if scope in ('evolve', 'meta_evolve', 'atom', 'submit_best', 'probe'):
            repo_flag, logs_flag = '--bind', '--ro-bind'
        elif scope == 'harness':
            repo_flag, logs_flag = '--ro-bind', '--bind'
        else:
            return command  # Unknown scope: no sandbox, transparent passthrough

        # Shallow-copy the static prefix + append dynamic mounts
        mounts = list(self._bwrap_static_mounts)
        mounts += [repo_flag, repo, repo]
        if os.path.isdir(eval_logs):
            mounts += [logs_flag, eval_logs, eval_logs]
        # Non-meta_evolve scope: hide the evolution/ directory (cover with tmpfs, invisible and unmodifiable by agent)
        if scope in ('evolve', 'atom', 'submit_best'):
            evo_dir = os.path.join(repo, 'evolution')
            if os.path.isdir(evo_dir):
                mounts += ['--tmpfs', evo_dir]

        # shlex.join correctly handles paths with spaces
        escaped = command.replace("'", "'\\''")
        return f"{self._bwrap_bin} {shlex.join(mounts)} bash -c '{escaped}'"

    def execute_external_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute an external tool (dynamically loaded)."""
        for tool in self.tools:
            if tool["name"] == tool_name:
                try:
                    return tool["function"](**arguments)
                except Exception as e:
                    return f"External tool error: {e}"

        return f"Unknown external tool: {tool_name}. Available tools: {[t['name'] for t in self.tools]}"

    def get_historic_version(self, iteration: Any, file_name: str = None) -> str:
        """Get code from a historical version."""
        if self.agent_instance is None:
            return "Error: Agent instance not available for accessing git history."

        git_controller = getattr(self.agent_instance, 'git_controller', None)
        evolution_tracker = getattr(self.agent_instance, 'evolution_tracker', None)

        if git_controller is None or evolution_tracker is None:
            return "Error: Git controller or evolution tracker not available."

        # Parse the commit
        commit = None
        iteration_label = str(iteration)

        if iteration == "best":
            best = evolution_tracker.get_best_version("highest_reward")
            if not best:
                return "No best version found yet. Complete at least one iteration first."
            commit = best[0]
            iteration_label = "best"
        else:
            try:
                iter_num = int(iteration)
                if iter_num < 1:
                    return f"Invalid iteration number: {iter_num}. Iteration numbers start at 1."
                record = evolution_tracker.get_iteration(iter_num)
                commit = record.primary_commit() if record else None
                iteration_label = f"iteration {iter_num}"
            except (ValueError, TypeError):
                return f"Invalid iteration value: {iteration}. Use a number (1, 2, 3...) or 'best'."

        if not commit:
            # Get the list of available iterations
            available = [r.iteration for r in evolution_tracker.records]
            if available:
                return f"Version not found for iteration {iteration}. Available completed iterations: {sorted(available)}"
            else:
                return f"Version not found for iteration {iteration}. No iterations have been completed yet."

        # Get the directory name of agent_code_dir (relative to the repo root)
        dir_name = Path(self.agent_code_dir).name if self.agent_code_dir else ""

        if file_name:
            # Get a single file
            file_path = f"{dir_name}/{file_name}" if dir_name else file_name
            content = git_controller.get_file_at_commit(file_path, commit)
            if content:
                return f"### {file_name} ({iteration_label})\n```python\n{content}\n```"
            return f"File '{file_name}' not found in {iteration_label}."
        else:
            # Get all files
            files = git_controller.get_tracked_files_at_commit(commit, dir_name)
            if not files:
                return f"No files found in {iteration_label}."

            results = []
            for f in files:
                content = git_controller.get_file_at_commit(f, commit)
                if content:
                    # Extract the file name relative to agent_code_dir
                    rel_name = f.split('/')[-1] if '/' in f else f
                    results.append(f"### {rel_name} ({iteration_label})\n```python\n{content}\n```")

            return "\n\n".join(results)

    def get_historic_eval_code(self, eval_index: int, file_name: str = None) -> str:
        """Get code snapshot from a specific evaluation in the current iteration."""
        snapshots = self.state.evaluation_snapshots if self.state else []
        if not snapshots:
            return "No evaluations in this iteration yet."
        if eval_index is None or eval_index < 1 or eval_index > len(snapshots):
            return f"Invalid eval_index: {eval_index}. Available: 1-{len(snapshots)}"
        code_hash, reward, eval_mode = _parse_snapshot(snapshots[eval_index - 1])
        snapshot = self.state.code_snapshots.get(code_hash)
        if not snapshot:
            return f"Code snapshot not found for eval #{eval_index}."
        if file_name:
            content = snapshot.get(file_name)
            if content is None:
                available = sorted(snapshot.keys())
                return f"File '{file_name}' not found in eval #{eval_index}. Available: {available}"
            return f"### {file_name} (eval #{eval_index}, {eval_mode}, reward={fmt_reward(reward)})\n```python\n{content}\n```"
        else:
            results = []
            for rel_path, content in sorted(snapshot.items()):
                results.append(f"### {rel_path} (eval #{eval_index}, {eval_mode})\n```python\n{content}\n```")
            return f"Eval #{eval_index} ({eval_mode}, reward={fmt_reward(reward)}):\n\n" + "\n\n".join(results)

    def _write_file_impl(self, path: Path, content: str) -> None:
        """Internal file-write implementation. For .py files, AST validation and syntax check are performed."""
        # .py files: AST validation
        if str(path).endswith('.py'):
            error = self._validate_python_code(content, str(path))
            if error:
                raise ValueError(error)

        try:
            path.write_text(content)
        except Exception as e:
            raise ValueError(f"Failed to write file: {e}")
