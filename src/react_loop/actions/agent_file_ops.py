"""
File operation handlers for AgentActionExecutor.

Extracted from agent_action.py to reduce module size.
Provides read_file, edit_file, write_file implementations.
"""

import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_action import AgentActionExecutor


class FileOpsMixin:
    """Mixin providing file operation methods for AgentActionExecutor."""

    # Type hint for mypy — these are set by AgentActionExecutor.__init__
    agent_code_dir: str
    agent_instance: object
    _modified_files: set
    _restricted_dirs: set  # Subdirectory names to hide from directory listings
    _reload_single_file: object
    _rescan_external_tools: object
    _write_file_impl: object
    logging: object

    def _resolve_path(self: "AgentActionExecutor", path: str) -> "tuple[Path, str | None]":
        """Resolve a possibly relative path to an absolute path. Returns (Path, None) or (None, error_msg)."""
        p = Path(path)
        if p.is_absolute():
            return p, None
        if self.agent_code_dir:
            return Path(self.agent_code_dir) / p, None
        return None, f"Error: The path {path} is not an absolute path (must start with '/')."

    def read_file(self: "AgentActionExecutor", path: str) -> str:
        """
        Read file content or list a directory.

        Args:
            path: File path (absolute path or relative path against agent_code_dir)

        Returns:
            File content with line numbers, or directory listing.
        """
        try:
            path_obj, err = self._resolve_path(path)
            if err:
                return err

            # Check whether the path exists
            if not path_obj.exists():
                return f"Error: The path {path} does not exist."

            return self._read_file_impl(path_obj)

        except Exception as e:
            return f"Error: {str(e)}"

    def _read_file_impl(self: "AgentActionExecutor", path_obj: Path) -> str:
        """Internal implementation for reading a file or directory."""
        if path_obj.is_dir():
            # Directory: list non-hidden files (up to 2 levels), use os.walk instead of the Unix find command
            try:
                entries = []
                for root, dirs, files in os.walk(str(path_obj)):
                    # Compute the current depth
                    rel_root = os.path.relpath(root, str(path_obj))
                    depth = 0 if rel_root == '.' else rel_root.count(os.sep) + 1
                    if depth > 2:
                        dirs.clear()
                        continue
                    # Skip hidden directories and restricted directories
                    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in self._restricted_dirs]
                    # Skip hidden files
                    for f in files:
                        if not f.startswith('.'):
                            entries.append(os.path.join(root, f))
                listing = '\n'.join(entries)
                return (
                    f"Here's the files and directories up to 2 levels deep in {path_obj}, "
                    f"excluding hidden items:\n{listing}"
                )
            except Exception as e:
                return f"Error listing directory: {e}"

        # File: show content with line numbers
        content = self._read_file_content(path_obj)
        return self._format_file_output(content, str(path_obj))

    def _read_file_content(self: "AgentActionExecutor", path: Path) -> str:
        """Read file content."""
        try:
            return path.read_text()
        except Exception as e:
            raise ValueError(f"Failed to read file: {e}")

    def _format_file_output(self: "AgentActionExecutor", content: str, path: str, init_line: int = 1) -> str:
        """Format the output (with line numbers). Truncation is handled centrally by execute()."""
        content = content.expandtabs()
        numbered_lines = [
            f"{i + init_line:6}\t{line}" for i, line in enumerate(content.split("\n"))
        ]
        return (
            f"Here's the result of running `cat -n` on {path}:\n"
            + "\n".join(numbered_lines)
            + "\n"
        )

    def edit_file(
        self: "AgentActionExecutor",
        path: str,
        old_string: str,
        new_string: str = "",
        replace_all: bool = False
    ) -> str:
        """
        Precise string-replacement edit.

        Args:
            path: Absolute path of the file.
            old_string: The original string to replace (must match exactly).
            new_string: The new string to replace with (default empty, i.e. deletion).
            replace_all: Whether to replace all occurrences (default False).

        Returns:
            Operation result string.
        """
        try:
            path_obj, err = self._resolve_path(path)
            if err:
                return err

            # Check whether the file exists
            if not path_obj.exists():
                return f"Error: The file {path} does not exist."

            if path_obj.is_dir():
                return f"Error: {path} is a directory and cannot be edited as a file."

            return self._edit_file_impl(path_obj, old_string, new_string, replace_all)

        except Exception as e:
            return f"Error: {str(e)}"

    def _edit_file_impl(
        self: "AgentActionExecutor",
        path_obj: Path,
        old_string: str,
        new_string: str,
        replace_all: bool
    ) -> str:
        """Internal implementation of precise string replacement."""
        # Read the original file
        try:
            original_content = path_obj.read_text()
        except Exception as e:
            return f"Error: Failed to read file: {e}"

        # Check whether old_string exists
        if old_string not in original_content:
            return f"Error: `old_string` not found in file. Make sure it matches exactly including whitespace."

        # Check uniqueness (if not replace_all)
        if not replace_all:
            count = original_content.count(old_string)
            if count > 1:
                return (
                    f"Error: `old_string` appears {count} times in file. "
                    f"Either provide a more unique string, or set `replace_all=true`."
                )

        # Perform the replacement
        if replace_all:
            new_content = original_content.replace(old_string, new_string)
        else:
            new_content = original_content.replace(old_string, new_string, 1)

        # Check whether there is an actual change
        if new_content == original_content:
            return "Warning: No changes made (old_string equals new_string)."

        # Write the file (AST validation is performed)
        self._write_file_impl(path_obj, new_content)

        # Update the modified files set
        rel_path = None
        if self.agent_code_dir:
            rel_path = os.path.relpath(str(path_obj), self.agent_code_dir)
            self._modified_files.add(rel_path)

        # Record the modification
        if self.state is not None:
            self.state.modifications_made.append({
                "operation": "edit",
                "file": rel_path or str(path_obj),
            })

            # Sync into the agent_codes mirror (.py files; evaluate fresh-imports from disk each time)
            reload_result = self._reload_single_file(rel_path, new_content, skip_validation=True)
            if reload_result:
                # Sync failed (validation error), but the file modification is not rolled back (already written to disk)
                return f"Warning: File edited but {reload_result}"

            # Rescan external tools (only when a tool file is modified)
            self._rescan_external_tools(" after edit", modified_files=[rel_path])

        # Return a concise diff
        old_lines = old_string.strip().split('\n')
        new_lines = new_string.strip().split('\n') if new_string.strip() else []

        if len(old_lines) == 1 and len(new_lines) <= 1:
            # Single-line replacement
            preview_old = old_string[:50] + "..." if len(old_string) > 50 else old_string
            preview_new = new_string[:50] + "..." if len(new_string) > 50 else new_string
            return f"Replaced: `{preview_old}` -> `{preview_new}`"
        else:
            # Multi-line replacement
            return (
                f"Replaced {len(old_lines)} line(s) with {len(new_lines)} line(s) "
                f"in {path_obj}"
            )

    def write_file(
        self: "AgentActionExecutor",
        path: str,
        content: str,
        create_only: bool = False
    ) -> str:
        """
        Create or overwrite a file.

        Args:
            path: Absolute path of the file.
            content: Full content of the file.
            create_only: Whether to only create (when True, an existing file raises an error).

        Returns:
            Operation result string.
        """
        try:
            path_obj, err = self._resolve_path(path)
            if err:
                return err

            # Check whether it is a directory
            if path_obj.is_dir():
                return f"Error: {path} is a directory."

            # create_only mode: check whether the file already exists
            if create_only and path_obj.exists():
                return f"Error: Cannot create new file; {path} already exists."

            # Write the file (AST validation is performed)
            self._write_file_impl(path_obj, content)

            # Update the modified files set
            if self.agent_code_dir:
                rel_path = os.path.relpath(str(path_obj), self.agent_code_dir)
                self._modified_files.add(rel_path)

                # Record the modification
                if self.state is not None:
                    self.state.modifications_made.append({
                        "operation": "write",
                        "file": rel_path,
                    })

                # Sync into the agent_codes mirror (.py files; evaluate fresh-imports from disk each time)
                reload_result = self._reload_single_file(rel_path, content, skip_validation=True)
                if reload_result:
                    # Sync failed (validation error), but the file modification is not rolled back (already written to disk)
                    return f"Warning: File written but {reload_result}"

                # Rescan external tools (only when a tool file is modified)
                self._rescan_external_tools(" after write", modified_files=[rel_path])

            action = "created" if create_only else "written"
            return f"File {action} successfully at: {path}"

        except Exception as e:
            return f"Error: {str(e)}"
