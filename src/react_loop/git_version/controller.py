"""
Git version controller and evolution tracker for React Loop Agent.

Provides:
- GitController: Core git operations (commit, diff, reset, apply_patch)
- EvolutionTracker: Track evolution history with tree structure support

Adapted from HGM project's git_utils.py.
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

try:
    import git
    GITPYTHON_AVAILABLE = True
except ImportError:
    GITPYTHON_AVAILABLE = False

from ..state import fmt_reward


@dataclass
class EvolutionRecord:
    """Record of a single evolution step.

    A single iteration may produce multiple commits (a "pool" — one per
    selected code version). ``new_commit`` / ``reward`` / ``timestamp`` are
    parallel lists (one entry per pool member, length 1 for single-commit
    iterations). The metadata fields ``committed_code_reward`` /
    ``committed_eval_mode`` / ``execution_errors`` mirror this shape.
    """
    iteration: int
    parent_commit: str
    new_commit: List[str] = field(default_factory=list)
    reward: List[float] = field(default_factory=list)
    timestamp: List[str] = field(default_factory=list)
    state_summary: str = ""
    action_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # operation_type values for "auxiliary" commits that must NOT count as a
    # main iteration (they reuse the main iteration number space via negative
    # numbers / separate metadata). Single source of truth for the filter that
    # resume-state, get_iteration, and best-version selection all share.
    AUX_OPERATION_TYPES = frozenset({"crossover", "init"})

    @property
    def is_main_iteration(self) -> bool:
        """True if this is a real main-line iteration record (not meta_evolve
        or crossover). Use this everywhere a caller needs to filter auxiliary
        commits out of main-iteration logic."""
        meta = self.metadata or {}
        return (meta.get("type") != "meta_evolve"
                and meta.get("operation_type") not in self.AUX_OPERATION_TYPES)

    def primary_commit(self) -> str:
        """Best-reward commit in the pool (or first if no rewards).
        Use this when a caller needs a single representative commit."""
        if not self.new_commit:
            return ""
        if self.reward and len(self.reward) == len(self.new_commit):
            idx = max(range(len(self.reward)), key=lambda i: self.reward[i])
            return self.new_commit[idx]
        return self.new_commit[0]

    def primary_reward(self) -> float:
        """Highest reward in the pool (or 0 if none)."""
        return max(self.reward) if self.reward else 0.0

    def primary_timestamp(self) -> str:
        """First timestamp in the pool (or empty string)."""
        return self.timestamp[0] if self.timestamp else ""

    def iter_pool(self):
        """Yield one dict per pool entry with parallel-list fields joined."""
        ccr = self.metadata.get("committed_code_reward", [])
        cem = self.metadata.get("committed_eval_mode", [])
        exe = self.metadata.get("execution_errors", [])
        if not isinstance(ccr, list):
            ccr = [ccr]
        if not isinstance(cem, list):
            cem = [cem]
        if not isinstance(exe, list):
            exe = [exe]
        n = len(self.new_commit)
        for i in range(n):
            yield {
                "new_commit": self.new_commit[i],
                "reward": self.reward[i] if i < len(self.reward) else 0.0,
                "timestamp": self.timestamp[i] if i < len(self.timestamp) else "",
                "committed_code_reward": ccr[i] if i < len(ccr) else None,
                "committed_eval_mode": cem[i] if i < len(cem) else "",
                "execution_errors": exe[i] if i < len(exe) else 0,
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "parent_commit": self.parent_commit,
            "new_commit": self.new_commit,
            "reward": self.reward,
            "timestamp": self.timestamp,
            "state_summary": self.state_summary,
            "action_count": self.action_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionRecord":
        # Backward compat: older files stored scalar new_commit/reward/timestamp
        new_commit = data.get("new_commit")
        if isinstance(new_commit, str):
            new_commit = [new_commit] if new_commit else []
        elif new_commit is None:
            new_commit = []

        reward = data.get("reward")
        if isinstance(reward, (int, float)):
            reward = [float(reward)]
        elif reward is None:
            reward = []

        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = [timestamp] if timestamp else []
        elif timestamp is None:
            timestamp = []

        metadata = dict(data.get("metadata", {}) or {})
        # Wrap commit-level metadata fields that used to be scalars into lists.
        for k in ("committed_code_reward", "committed_eval_mode", "execution_errors"):
            v = metadata.get(k)
            if v is not None and not isinstance(v, list):
                metadata[k] = [v]

        return cls(
            iteration=data["iteration"],
            parent_commit=data["parent_commit"],
            new_commit=new_commit,
            reward=reward,
            timestamp=timestamp,
            state_summary=data.get("state_summary", ""),
            action_count=data.get("action_count", 0),
            metadata=metadata,
        )


class GitController:
    """
    Git operations controller for agent evolution.

    Handles version control operations needed for tracking
    agent code changes across evolution iterations.
    """

    def __init__(self, repo_path: str):
        """
        Initialize the Git controller.

        Args:
            repo_path: Path to the git repository.
        """
        self.repo_path = Path(repo_path).resolve()
        self._repo = None

        if GITPYTHON_AVAILABLE:
            try:
                self._repo = git.Repo(self.repo_path)
            except Exception:
                pass  # Will use subprocess commands instead

    def _run_git_command(
        self,
        args: List[str],
        check: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a git command using subprocess."""
        cmd = ["git", "-C", str(self.repo_path)] + args
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
            encoding='utf-8',
            errors='replace',
        )

    def is_git_repo(self) -> bool:
        """Check if the path is a git repository."""
        git_dir = self.repo_path / ".git"
        return git_dir.exists()

    def init_repo(self) -> bool:
        """Initialize a git repository if it doesn't exist."""
        if self.is_git_repo():
            # Even on an existing repo, ensure bytecode artifacts are
            # ignored/untracked (keeps the main lineage free of *.pyc).
            self._ensure_gitignore()
            return True

        try:
            self._run_git_command(["init"])
            # Configure local user info (only for this repo's commit records)
            # No GitHub login needed, just to identify the commit author
            self._run_git_command(["config", "user.email", "evolution@localhost"])
            self._run_git_command(["config", "user.name", "Evolution Agent"])
            self._ensure_gitignore()
            return True
        except subprocess.CalledProcessError:
            return False

    def _ensure_gitignore(self) -> None:
        """Append Python artifact rules + plan.md to .gitignore and untrack cached .pyc.

        Runs on every ``init_repo`` (new or existing repo). The evolution repo
        historically had no ``.gitignore``, so ``__pycache__/*.pyc`` got tracked
        and an agent's ``git add -A`` kept dragging them into commits —
        polluting the main lineage. This appends (never overwrites) the rules
        and removes already-tracked bytecode from the index (worktree files are
        kept; Python regenerates them).

        ``plan.md`` is also ignored: it is an ephemeral per-iteration working
        notebook (cleared at iteration start, never committed). Gitignoring it
        means ``git checkout``/``add -A`` never touch it, so it rolls back
        naturally with the code without backup/restore shenanigans.
        """
        ignore_path = self.repo_path / ".gitignore"
        try:
            existing = ignore_path.read_text(encoding="utf-8") if ignore_path.exists() else ""
        except OSError:
            existing = ""
        needs_bytecode = ("__pycache__/" not in existing) or ("*.py[cod]" not in existing)
        needs_plan = "plan.md" not in existing
        if needs_bytecode or needs_plan:
            chunks = []
            if needs_bytecode:
                chunks.append(
                    "\n# Python bytecode (auto-added by evolution framework)\n"
                    "__pycache__/\n*.py[cod]\n"
                )
            if needs_plan:
                chunks.append(
                    "\n# plan.md — ephemeral per-iteration working notebook "
                    "(auto-added by evolution framework)\n"
                    "plan.md\n"
                )
            with open(ignore_path, "a", encoding="utf-8") as f:
                f.write("".join(chunks))
            self._run_git_command(["add", ".gitignore"], check=False)

        # Untrack any currently-tracked bytecode / plan.md so they're ignored
        # from now on (worktree files kept). One pathspec-driven call — git
        # interprets the patterns itself, so no shell glob or ls-files probe
        # is needed. plan.md never existed before this change, so it's a no-op
        # there; harmless for fresh repos.
        self._run_git_command(
            ["rm", "-r", "--cached", "--ignore-unmatch",
             "*.py[cod]", "__pycache__", "plan.md"],
            check=False,
        )

    def get_current_commit(self) -> Optional[str]:
        """
        Get the current commit hash.

        Returns:
            Commit hash string or None if no commits exist.
        """
        try:
            result = self._run_git_command(["rev-parse", "HEAD"], check=False)
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except Exception:
            return None

    def get_short_commit(self, commit: str = None) -> str:
        """Get short version of commit hash."""
        if commit is None:
            commit = self.get_current_commit()
        if commit:
            return commit[:7]
        return "none"

    def create_evolution_commit(
        self,
        iteration: int,
        message: str,
        files: List[str] = None
    ) -> Optional[str]:
        """
        Create a commit for an evolution step.

        Args:
            iteration: The iteration number.
            message: Commit message.
            files: Optional list of specific files to commit.
                   If None, commits all changes.

        Returns:
            The new commit hash, or None on failure.
        """
        try:
            # Stage files
            if files:
                for f in files:
                    self._run_git_command(["add", f])
            else:
                self._run_git_command(["add", "-A"])

            # Create commit
            full_message = f"[Evolution iter={iteration}] {message}"
            result = self._run_git_command(
                ["commit", "-m", full_message, "--no-verify"],
                check=False
            )

            if result.returncode != 0:
                # Maybe nothing to commit
                if "nothing to commit" in result.stdout:
                    return self.get_current_commit()
                print(f"Commit failed: {result.stderr}")
                return None

            return self.get_current_commit()

        except Exception as e:
            print(f"Error creating commit: {e}")
            return None

    def create_meta_ref(self, main_iteration: int, commit: str) -> bool:
        """Pin a git ref on a meta-evolve commit so it is durable & discoverable.

        Meta-evolve commits are intentionally kept off the main lineage (HEAD is
        reset to the main baseline so meta changes fold into the next main
        commit). Without a ref, such a commit is dangling and eventually
        garbage-collected. This writes a lightweight ref under
        ``refs/meta_evolve/`` — outside ``refs/heads`` so it never shows up as a
        branch and can't confuse branch-based logic, but reachable for GC and
        visible to ``git log --all``.

        Idempotent: re-pinning the same (iteration, commit) overwrites cleanly.

        Args:
            main_iteration: The main iteration this meta phase followed (matches
                the ``[Meta-Evolve iter=N]`` commit message).
            commit: The meta-evolve commit hash to pin.

        Returns:
            True if the ref was created, False otherwise.
        """
        if not commit:
            return False
        try:
            ref = f"refs/meta_evolve/iter-{main_iteration}"
            result = self._run_git_command(
                ["update-ref", ref, commit], check=False
            )
            if result.returncode == 0:
                return True
            print(f"Failed to create meta ref {ref}: {result.stderr}")
            return False
        except Exception as e:
            print(f"Error creating meta ref: {e}")
            return False

    def diff_versus_commit(self, commit: str) -> str:
        """
        Get diff of current state versus a commit, including untracked files.

        Args:
            commit: The commit hash to diff against.

        Returns:
            Diff string including new file contents.
        """
        diff_output = ""

        # Get diff of tracked files
        try:
            result = self._run_git_command(
                ["diff", commit],
                check=False
            )
            diff_output = result.stdout
        except Exception:
            pass

        # Get list of untracked files
        try:
            result = self._run_git_command(
                ["ls-files", "--others", "--exclude-standard"],
                check=False
            )
            untracked_files = result.stdout.splitlines()

            # Generate diffs for untracked files
            for file_path in untracked_files:
                full_path = self.repo_path / file_path
                if full_path.exists() and full_path.is_file():
                    try:
                        content = full_path.read_text(encoding="utf-8")
                        # Create a diff-like format for new files
                        diff_output += f"\ndiff --git a/{file_path} b/{file_path}\n"
                        diff_output += f"new file mode 100644\n"
                        diff_output += f"index 0000000..{'0' * 40}\n"
                        diff_output += f"--- /dev/null\n"
                        diff_output += f"+++ b/{file_path}\n"
                        for i, line in enumerate(content.splitlines(), 1):
                            diff_output += f"+{line}\n"
                    except Exception:
                        continue

        except Exception:
            pass

        return diff_output

    def reset_to_commit(self, commit: str) -> bool:
        """
        Reset the repository to a specific commit.

        Args:
            commit: The commit hash to reset to.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Hard reset tracked files
            result = self._run_git_command(
                ["reset", "--hard", commit],
                check=False
            )
            if result.returncode != 0:
                print(f"Reset failed: {result.stderr}")
                return False

            # Clean untracked files and directories
            self._run_git_command(["clean", "-fd"], check=False)

            return True

        except Exception as e:
            print(f"Error resetting to commit: {e}")
            return False

    def apply_patch(self, patch_str: str) -> Tuple[bool, str]:
        """
        Apply a patch to the repository.

        Args:
            patch_str: The patch content in unified diff format.

        Returns:
            Tuple of (success, error_message).
        """
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo_path), "apply", "--reject", "-"],
                input=patch_str,
                text=True,
                capture_output=True,
                check=False,
                encoding='utf-8',
                errors='replace',
            )

            if result.returncode != 0:
                error_msg = f"Patch did not fully apply. stderr: {result.stderr}"
                print(error_msg)
                return False, error_msg

            return True, ""

        except Exception as e:
            error_msg = f"Error applying patch: {e}"
            print(error_msg)
            return False, error_msg

    def get_commit_message(self, commit: str = None) -> str:
        """Get the commit message for a specific commit."""
        if commit is None:
            commit = "HEAD"
        try:
            result = self._run_git_command(
                ["log", "-1", "--format=%B", commit],
                check=False
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def get_file_at_commit(self, file_path: str, commit: str) -> Optional[str]:
        """Get the content of a file at a specific commit."""
        try:
            result = self._run_git_command(
                ["show", f"{commit}:{file_path}"],
                check=False
            )
            if result.returncode == 0:
                return result.stdout
            return None
        except Exception:
            return None

    def get_tracked_files_at_commit(self, commit: str, directory: str = "") -> List[str]:
        """
        Get list of tracked files in a directory at a specific commit.

        Args:
            commit: The commit hash.
            directory: Directory path relative to repo root (optional).

        Returns:
            List of file paths relative to repo root.
        """
        try:
            # Use git ls-tree to list files at specific commit
            if directory:
                result = self._run_git_command(
                    ["ls-tree", "-r", "--name-only", commit, directory],
                    check=False
                )
            else:
                result = self._run_git_command(
                    ["ls-tree", "-r", "--name-only", commit],
                    check=False
                )

            if result.returncode == 0:
                files = [f for f in result.stdout.strip().split('\n') if f]
                files = [f for f in files
                         if (f.endswith('.py') or f.endswith('.md'))
                         and '/.evolution/' not in f and not f.startswith('.evolution/')]
                return files
            return []
        except Exception:
            return []

    def get_file_content_at_commit(self, commit: str, file_path: str) -> Optional[str]:
        """
        Get content of a file at a specific commit.

        Alias for get_file_at_commit for clarity.

        Args:
            commit: The commit hash.
            file_path: Path to the file relative to repo root.

        Returns:
            File content as string, or None if not found.
        """
        return self.get_file_at_commit(file_path, commit)

    # =====================================================================
    # Git Status Methods (for S_t environment info)
    # =====================================================================

    def get_current_branch(self) -> Optional[str]:
        """
        Get the current branch name.

        Returns:
            Branch name string, or None if in detached HEAD state or not a git repo.
        """
        if not self.is_git_repo():
            return None
        try:
            result = self._run_git_command(
                ["branch", "--show-current"],
                check=False
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            return None  # Detached HEAD
        except Exception:
            return None

    def get_main_branch(self) -> Optional[str]:
        """
        Get the main branch name (main, master, or default branch).

        Returns:
            Main branch name, or None if cannot determine.
        """
        if not self.is_git_repo():
            return None

        # Try common main branch names
        for branch in ["main", "master"]:
            try:
                result = self._run_git_command(
                    ["rev-parse", "--verify", branch],
                    check=False
                )
                if result.returncode == 0:
                    return branch
            except Exception:
                continue

        # Fallback: get default branch from remote symbolic ref
        try:
            result = self._run_git_command(
                ["symbolic-ref", "refs/remotes/origin/HEAD"],
                check=False
            )
            if result.returncode == 0:
                # Output format: refs/remotes/origin/main
                ref = result.stdout.strip()
                if ref.startswith("refs/remotes/origin/"):
                    return ref.split("/")[-1]
        except Exception:
            pass

        return None

    def get_recent_commits(self, n: int = 5) -> List[Dict[str, str]]:
        """
        Get the recent N commits.

        Args:
            n: Number of commits to retrieve.

        Returns:
            List of dicts with keys: hash, short_hash, message, author, date
        """
        if not self.is_git_repo():
            return []

        commits = []
        try:
            # Format: hash|short_hash|subject|author|date
            result = self._run_git_command(
                ["log", f"-{n}", "--format=%H|%h|%s|%an|%ci"],
                check=False
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        parts = line.split("|", 4)
                        if len(parts) >= 5:
                            commits.append({
                                "hash": parts[0],
                                "short_hash": parts[1],
                                "message": parts[2],
                                "author": parts[3],
                                "date": parts[4],
                            })
        except Exception:
            pass

        return commits

    def get_working_directory_status(self) -> Dict[str, List[str]]:
        """
        Get the working directory status (staged, modified, untracked files).

        Returns:
            Dict with keys: staged, modified, untracked, each containing list of file paths.
        """
        status = {
            "staged": [],
            "modified": [],
            "untracked": [],
        }

        if not self.is_git_repo():
            return status

        try:
            # Get porcelain status
            result = self._run_git_command(
                ["status", "--porcelain"],
                check=False
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if not line:
                        continue
                    # Porcelain format: XY filename
                    # X = staged status, Y = unstaged status
                    # XY is exactly 2 chars, filename follows (may have leading space)
                    if len(line) < 3:
                        continue
                    xy = line[:2]
                    filepath = line[2:].strip()  # Skip XY and strip whitespace

                    # Staged files (X is not space or ?)
                    if xy[0] not in (" ", "?"):
                        status["staged"].append(filepath)

                    # Modified files (Y is M, D, etc but not space or ?)
                    if xy[1] not in (" ", "?") and xy[0] != "?":
                        status["modified"].append(filepath)

                    # Untracked files (??)
                    if xy == "??":
                        status["untracked"].append(filepath)

        except Exception:
            pass

        return status


class EvolutionTracker:
    """
    Track evolution history with support for tree structures.

    Maintains a record of all evolution iterations, their relationships,
    and allows querying for best versions or specific lineages.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the evolution tracker.

        Args:
            output_dir: Directory to store evolution metadata.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.output_dir / "evolution_metadata.json"
        self.records: List[EvolutionRecord] = []
        self.tree: Dict[str, List[str]] = {}  # parent -> [children]
        self.metadata: Dict[str, Any] = {}

        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load existing metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.records = [
                        EvolutionRecord.from_dict(r)
                        for r in data.get("records", [])
                    ]
                    self.tree = data.get("tree", {})
                    self.metadata = data.get("metadata", {})
            except Exception as e:
                print(f"Error loading evolution metadata: {e}")

    def _save_metadata(self) -> None:
        """Save metadata to file."""
        try:
            data = {
                "records": [r.to_dict() for r in self.records],
                "tree": self.tree,
                "metadata": self.metadata,
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving evolution metadata: {e}")

    def record_iteration(
        self,
        iteration: int,
        parent_commit: str,
        new_commit: str,
        reward: float,
        state_summary: str = "",
        action_count: int = 0,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Record a single-commit iteration.

        Wraps ``new_commit``/``reward`` into length-1 lists — equivalent to a
        pool of size 1. For multi-commit iterations use ``record_pool_iteration``.
        """
        metadata = dict(metadata or {})
        # Wrap commit-level metadata scalars (legacy callers may pass either).
        for k in ("committed_code_reward", "committed_eval_mode", "execution_errors"):
            v = metadata.get(k)
            if v is not None and not isinstance(v, list):
                metadata[k] = [v]

        record = EvolutionRecord(
            iteration=iteration,
            parent_commit=parent_commit,
            new_commit=[new_commit],
            reward=[float(reward)],
            timestamp=[datetime.now().isoformat()],
            state_summary=state_summary,
            action_count=action_count,
            metadata=metadata,
        )

        self.records.append(record)

        # Update tree structure
        if parent_commit not in self.tree:
            self.tree[parent_commit] = []
        self.tree[parent_commit].append(new_commit)

        self._save_metadata()

    def record_pool_iteration(
        self,
        iteration: int,
        parent_commit: str,
        pool_entries: List[Dict[str, Any]],
        state_summary: str = "",
        action_count: int = 0,
        shared_metadata: Dict[str, Any] = None,
    ) -> None:
        """Record an iteration that produced multiple commits (a "pool").

        Args:
            iteration: The iteration number.
            parent_commit: Parent commit hash shared by all pool entries.
            pool_entries: List of dicts, each containing keys ``new_commit``,
                ``reward``, ``timestamp``, ``committed_code_reward``,
                ``committed_eval_mode``, ``execution_errors``.
            state_summary: Brief summary of the iteration state (shared).
            action_count: Number of actions taken in this iteration (shared).
            shared_metadata: Iteration-level metadata shared across pool entries
                (e.g. ``summary_text``, ``modified_files``, ``seed_info``).
        """
        metadata = dict(shared_metadata or {})
        metadata["committed_code_reward"] = [e.get("committed_code_reward") for e in pool_entries]
        metadata["committed_eval_mode"] = [e.get("committed_eval_mode", "") for e in pool_entries]
        metadata["execution_errors"] = [e.get("execution_errors", 0) for e in pool_entries]
        metadata["pool_size"] = len(pool_entries)

        record = EvolutionRecord(
            iteration=iteration,
            parent_commit=parent_commit,
            new_commit=[e["new_commit"] for e in pool_entries],
            reward=[float(e.get("reward", 0.0)) for e in pool_entries],
            timestamp=[e.get("timestamp") or datetime.now().isoformat() for e in pool_entries],
            state_summary=state_summary,
            action_count=action_count,
            metadata=metadata,
        )
        self.records.append(record)

        # tree: parent → all pool commits as siblings
        if parent_commit not in self.tree:
            self.tree[parent_commit] = []
        self.tree[parent_commit].extend(record.new_commit)

        self._save_metadata()

    def get_best_version(
        self,
        strategy: str = "highest_reward"
    ) -> Optional[Tuple[str, float]]:
        """
        Get the best version according to strategy.

        Args:
            strategy: Selection strategy:
                - "highest_reward": Highest reward with val-preferred logic (default)
                - "latest": Most recent
                - "first": First recorded

        Returns:
            Tuple of (commit_hash, reward) or None if no records.
        """
        if not self.records:
            return None

        if strategy == "highest_reward":
            return self._get_best_version_val_preferred()
        elif strategy == "latest":
            latest = self.records[-1]
            return (latest.primary_commit(), latest.primary_reward())
        elif strategy == "first":
            first = self.records[0]
            return (first.primary_commit(), first.primary_reward())
        else:
            return None

    def _get_best_version_val_preferred(self) -> Optional[Tuple[str, float]]:
        """Cross-iteration best selection with val-preferred logic + crash veto.

        Priority: val records → dev records → fallback to all (backward compatible).

        Crash veto: a version whose evaluation CRASHED (``execution_errors`` >
        0 in metadata, e.g. a harness NameError) is never selected as best,
        even if its averaged reward is the highest — a crash zeroes episodes by
        accident, not strategy, so the reward is unreliable. Within a mode layer
        we prefer clean records; only if EVERY candidate in the layer crashed do
        we fall back to the tainted ones (so selection never returns None solely
        because all versions had errors). When a higher-reward version is skipped
        for crashing, a warning is printed so the silent-selection failure mode
        (exporting an erroring version as "best") stays visible.

        Pool-aware: each iteration's record may carry multiple commits (a pool);
        each pool entry is its own candidate.
        """
        # Build a flat candidate list: one entry per pool entry across records.
        candidates: List[Tuple["EvolutionRecord", Dict[str, Any]]] = []
        for rec in self.records:
            for entry in rec.iter_pool():
                if entry["reward"] > 0:
                    candidates.append((rec, entry))

        if not candidates:
            return None

        def _exec_err(entry: Dict[str, Any]) -> int:
            v = entry.get("execution_errors", 0)
            try:
                return int(v) if v else 0
            except (TypeError, ValueError):
                return 0

        for mode in ("val", "dev"):
            mode_cands = [(r, e) for (r, e) in candidates
                          if e.get("committed_eval_mode") == mode]
            if not mode_cands:
                continue
            clean = [pair for pair in mode_cands if _exec_err(pair[1]) == 0]
            pool = clean if clean else mode_cands
            best_r, best_e = max(pool, key=lambda pair: pair[1]["reward"])
            # Surface a veto only when it actually changed the pick — i.e. the
            # highest-reward candidate crashed and a lower-reward clean one won.
            top_r, top_e = max(mode_cands, key=lambda pair: pair[1]["reward"])
            if top_e is not best_e and _exec_err(top_e) > 0:
                print(
                    f"[best-select] Skipped {top_e['new_commit'][:7]} "
                    f"(reward={fmt_reward(top_e['reward'])}, "
                    f"execution_errors={_exec_err(top_e)}) — harness crashed during "
                    f"eval; reward is unreliable. Selected {best_e['new_commit'][:7]} "
                    f"(reward={fmt_reward(best_e['reward'])}, clean) instead."
                )
            return (best_e["new_commit"], best_e["reward"])

        best_r, best_e = max(candidates, key=lambda pair: pair[1]["reward"])
        return (best_e["new_commit"], best_e["reward"])

    def get_lineage(self, commit: str) -> List[EvolutionRecord]:
        """
        Get the lineage (ancestry) of a commit.

        Args:
            commit: The commit hash to trace.

        Returns:
            List of EvolutionRecord from root to the commit.
        """
        lineage = []
        current_commit = commit

        # Build a map of commit -> record (one entry per pool entry)
        commit_to_record = {entry["new_commit"]: r
                            for r in self.records
                            for entry in r.iter_pool()}

        # Trace backwards
        visited = set()
        while current_commit and current_commit not in visited:
            visited.add(current_commit)

            if current_commit in commit_to_record:
                record = commit_to_record[current_commit]
                lineage.append(record)
                current_commit = record.parent_commit
            else:
                break

        # Reverse to get root -> commit order
        lineage.reverse()
        return lineage

    def get_children(self, commit: str) -> List[str]:
        """Get all children of a commit."""
        return self.tree.get(commit, [])

    def get_all_commits(self) -> List[str]:
        """Get all commit hashes in evolution history."""
        return [entry["new_commit"]
                for r in self.records
                for entry in r.iter_pool()]

    def get_iteration(self, iteration: int) -> Optional[EvolutionRecord]:
        """Get the main-line record for a specific iteration.

        Skips auxiliary records (meta_evolve / crossover) so
        callers (baseline reward lookup, historic_version) only ever see real main
        iterations — via ``EvolutionRecord.is_main_iteration``, the single source
        of truth shared with ``_derive_resume_state`` and
        ``_generate_evolution_summary``. Meta records reuse the main ``iteration``
        number space via ``_meta_iteration`` and would otherwise collide; the
        ``main_iteration`` metadata is the source of truth.
        """
        for record in self.records:
            if record.iteration != iteration:
                continue
            if not record.is_main_iteration:
                continue
            return record
        return None

    def get_record_by_commit(self, commit_hash: str) -> Optional[EvolutionRecord]:
        """Get record by its new_commit hash (searches across all pool entries)."""
        if not commit_hash:
            return None
        for r in self.records:
            for entry in r.iter_pool():
                if entry["new_commit"] == commit_hash:
                    return r
        return None

    def get_full_reward(self, commit_hash: str, fallback_scalar: float) -> Any:
        """Get the full reward (dict or float) for a commit.

        Uses committed_code_reward (val-preferring aligned reward) from metadata,
        falls back to the scalar reward from the tracker.

        Args:
            commit_hash: The commit hash to look up.
            fallback_scalar: Scalar reward to return if no record or no history.

        Returns:
            The full reward (dict or float), or the fallback scalar.
        """
        if not commit_hash:
            return fallback_scalar
        for r in self.records:
            for entry in r.iter_pool():
                if entry["new_commit"] == commit_hash:
                    committed_reward = entry.get("committed_code_reward")
                    if committed_reward is not None:
                        return committed_reward
                    return fallback_scalar
        return fallback_scalar

    def get_summary(self) -> str:
        """Get a summary of the evolution history."""
        if not self.records:
            return "No evolution records yet."

        best = self.get_best_version("highest_reward")
        latest = self.get_best_version("latest")

        lines = [
            f"Total iterations: {len(self.records)}",
            f"Best reward: {fmt_reward(best[1])} (commit {best[0][:7]})" if best else "",
            f"Latest: commit {latest[0][:7]} with reward {fmt_reward(latest[1])}" if latest else "",
            "",
            "Reward progression:",
        ]

        for record in self.records:
            lines.append(
                f"  iter={record.iteration}: reward={fmt_reward(record.primary_reward())} "
                f"actions={record.action_count}"
            )

        return "\n".join(lines)

    def export_evolution_data(self) -> Dict[str, Any]:
        """Export all evolution data for analysis."""
        return {
            "records": [r.to_dict() for r in self.records],
            "tree": self.tree,
            "metadata": self.metadata,
            "best_version": self.get_best_version("highest_reward"),
            "total_iterations": len(self.records),
        }

    def get_history_context(self, last_n: int = 3) -> str:
        """
        Get the historical context text of the most recent N iterations (for the LLM).

        Args:
            last_n: The most recent N iterations.

        Returns:
            Formatted historical context string.
        """
        if not self.records:
            return "No previous iterations."

        recent = self.records[-last_n:]
        lines = []

        for record in recent:
            modifications_count = record.metadata.get("modifications_count", 0)
            lines.append(
                f"- Iteration {record.iteration}: reward={fmt_reward(record.primary_reward())}, "
                f"modifications={modifications_count}"
            )
            # If summary_text is present, add it to the context
            summary_text = record.metadata.get("summary_text", "")
            if summary_text:
                lines.append(f"  Summary: {summary_text}")

        return "\n".join(lines)

    def get_best_iteration(self) -> int:
        """Get the best iteration number."""
        best = self.get_best_version("highest_reward")
        if best:
            for record in self.records:
                for entry in record.iter_pool():
                    if entry["new_commit"] == best[0]:
                        return record.iteration
        return 0

    # =====================================================================
    # Graph query methods (derive graph from records + tree, no extra storage)
    # =====================================================================

    def get_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Build graph-friendly node dict from records.

        Returns:
            Dict mapping short_hash -> node info dict, compatible with
            the old evolution_graph.json "nodes" format.

        Pool-aware: one node per pool entry.
        """
        nodes = {}
        for record in self.records:
            metadata = record.metadata or {}
            summary_text = metadata.get("summary_text", "")
            for entry in record.iter_pool():
                git_hash = entry["new_commit"]
                short_hash = git_hash[:7] if git_hash else ""
                if not short_hash:
                    continue

                fitness_detail = entry.get("committed_code_reward")
                if fitness_detail is None:
                    fitness_detail = {}

                nodes[short_hash] = {
                    "iteration": record.iteration,
                    "git_hash": git_hash,
                    "parent_hash": record.parent_commit,
                    "fitness_score": entry["reward"],
                    "fitness_detail": fitness_detail,
                    "cost_metrics": {
                        "action_count": record.action_count,
                    },
                    "node_summary": summary_text,
                    "operation_type": "refine",
                    "status": "completed",
                }
        return nodes

    def get_edges(self) -> List[Dict[str, str]]:
        """Derive edge list from records.

        Returns:
            List of {source, target, operation_type} dicts.

        Pool-aware: one edge per pool entry (parent → entry commit).
        """
        edges = []
        for record in self.records:
            parent = record.parent_commit
            if parent:
                parent_short = parent[:7] if len(parent) > 7 else parent
                for entry in record.iter_pool():
                    child_short = entry["new_commit"][:7] if entry["new_commit"] else ""
                    if child_short:
                        edges.append({
                            "source": parent_short,
                            "target": child_short,
                            "operation_type": "refine",
                        })
        return edges

    def get_root_node(self) -> Optional[str]:
        """Get the root node (first record's parent commit, short hash)."""
        if not self.records:
            return None
        parent = self.records[0].parent_commit
        return parent[:7] if parent else None

    def get_active_head(self) -> Optional[str]:
        """Get the active head (latest record's primary commit, short hash)."""
        if not self.records:
            return None
        primary = self.records[-1].primary_commit()
        return primary[:7] if primary else None

    def get_graph_data(self) -> Dict[str, Any]:
        """Build the full graph dict (nodes + edges + pointers) from tracker data.

        This replaces the old evolution_graph.json format entirely.
        """
        return {
            "version": 1,
            "nodes": self.get_nodes(),
            "edges": self.get_edges(),
            "root_node": self.get_root_node(),
            "active_head": self.get_active_head(),
        }
