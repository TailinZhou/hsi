"""
Context Persistence Module for React Loop Agent.

Handles saving and loading of evolution context data to/from the filesystem.
Enables git version control of context data across iterations.

Storage structure:
{agent_code_dir}/.evolution_context/
├── .gitignore
├── evolution_context.json     # Cross-iteration context
├── summaries.json             # Quick view summaries
├── main_evolve/               # Main evolve loop messages
│   ├── iter_0.json
│   ├── iter_1.json
│   └── ...
├── select_seed/               # select_seed phase messages
│   ├── iter_1.json
│   └── ...
├── select_commit/             # select_commit phase messages
│   ├── iter_1.json
│   └── ...
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

from .state import MessageHistory, IterationSummary, EvolutionContext


class ContextPersistence:
    """
    Context-persistence manager.

    Responsibilities:
    1. Save/load message history (per iteration).
    2. Save/load evolution context (cross-iteration).
    3. Manage the persistence directory structure.
    4. Provide the list of files that need to be git-committed.
    """

    CONTEXT_DIR_NAME = ".evolution_context"
    MESSAGE_HISTORY_DIR = "main_evolve"

    # Phase-specific subdirectories under .evolution_context/
    # Each stores the full conversation messages from its phase's react loop.
    PHASE_DIRS = {
        "select_seed": "select_seed",
        "select_commit": "select_commit",
    }

    def __init__(self, agent_code_dir: str, context_dir_name: str = ".evolution_context"):
        """
        Initialize the persistence manager.

        Args:
            agent_code_dir: Path to the agent code directory.
            context_dir_name: Context directory name (default ".evolution_context").
        """
        self.agent_code_dir = Path(agent_code_dir)
        self.context_dir = self.agent_code_dir / context_dir_name
        self.message_history_dir = self.context_dir / self.MESSAGE_HISTORY_DIR

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure the persistence directory structure exists."""
        self.context_dir.mkdir(parents=True, exist_ok=True)
        self.message_history_dir.mkdir(parents=True, exist_ok=True)

        # Create .gitignore (ignore the main_evolve directory since files may be large)
        gitignore_path = self.context_dir / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text(
                "# Ignore main evolve history files (can be large)\n"
                "main_evolve/\n"
                "# Keep evolution context and summaries\n"
                "!evolution_context.json\n"
                "!summaries.json\n"
            )

    def save_message_history(
        self,
        iteration: int,
        message_history: MessageHistory
    ) -> Path:
        """
        Save the message history to a file.

        Args:
            iteration: Iteration number.
            message_history: MessageHistory object.

        Returns:
            Path of the saved file.
        """
        filename = f"iter_{iteration}.json"
        filepath = self.message_history_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(message_history.to_dict(), f, ensure_ascii=False, indent=2)

        return filepath

    def load_message_history(self, iteration: int) -> Optional[MessageHistory]:
        """
        Load the message history.

        Args:
            iteration: Iteration number.

        Returns:
            A MessageHistory object, or None if the file does not exist.
        """
        filename = f"iter_{iteration}.json"
        filepath = self.message_history_dir / filename

        if not filepath.exists():
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return MessageHistory.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load message history for iteration {iteration}: {e}")
            return None

    def save_evolution_context(self, context: EvolutionContext) -> Path:
        """
        Save the evolution context.

        Args:
            context: EvolutionContext object.

        Returns:
            Path of the saved file.
        """
        filepath = self.context_dir / "evolution_context.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(context.to_dict(), f, ensure_ascii=False, indent=2)

        return filepath

    def load_evolution_context(self) -> Optional[EvolutionContext]:
        """
        Load the evolution context.

        Returns:
            An EvolutionContext object, or a fresh empty one if the file does not exist.
        """
        filepath = self.context_dir / "evolution_context.json"

        if not filepath.exists():
            return EvolutionContext()

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return EvolutionContext.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load evolution context: {e}")
            return EvolutionContext()

    def save_summaries(self, context: EvolutionContext) -> Path:
        """
        Save the summaries file (for quick viewing).

        Args:
            context: EvolutionContext object.

        Returns:
            Path of the saved file.
        """
        filepath = self.context_dir / "summaries.json"

        summaries_data = {
            "total_iterations": len(context.iteration_summaries),
            "best_reward": context.best_reward if context.track_best_reward else None,
            "best_iteration": context.best_iteration if context.track_best_reward else None,
            "summaries": [
                {
                    "iteration": s.iteration,
                    "reward": s.reward,
                    "summary_text": s.summary_text,
                    "commit_hash": s.commit_hash[:7] if s.commit_hash else "",
                }
                for s in context.iteration_summaries
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summaries_data, f, ensure_ascii=False, indent=2)

        return filepath

    def save_phase_messages(
        self,
        phase: str,
        iteration: int,
        messages,
    ) -> Optional[Path]:
        """
        Save phase-specific conversation messages to .evolution_context/<phase_dir>/iter_N.json.

        Accepts either a ``MessageHistory`` object or a plain list of message dicts.

        Args:
            phase: Phase key — one of "select_seed", "select_commit".
            iteration: Iteration number.
            messages: ``MessageHistory`` or ``list[dict]`` — the full conversation.

        Returns:
            Saved file path, or None if phase is unknown or messages are empty.
        """
        if phase not in self.PHASE_DIRS:
            return None

        # Accept both MessageHistory and plain list[dict]
        if hasattr(messages, 'to_dict'):
            data = messages.to_dict()
        elif isinstance(messages, list):
            data = {"messages": list(messages)}
        else:
            return None

        if not data.get("messages"):
            return None  # Don't save empty conversations

        phase_dir = self.context_dir / self.PHASE_DIRS[phase]
        phase_dir.mkdir(parents=True, exist_ok=True)

        filepath = phase_dir / f"iter_{iteration}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return filepath

    def load_phase_messages(self, phase: str, iteration: int) -> Optional[List[Dict[str, Any]]]:
        """
        Load phase-specific conversation messages.

        Args:
            phase: Phase key — one of "select_seed", "select_commit".
            iteration: Iteration number.

        Returns:
            List of message dicts, or None if not found / load error.
        """
        if phase not in self.PHASE_DIRS:
            return None

        phase_dir = self.context_dir / self.PHASE_DIRS[phase]
        filepath = phase_dir / f"iter_{iteration}.json"

        if not filepath.exists():
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("messages", [])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load {phase} messages for iteration {iteration}: {e}")
            return None

    def list_phase_iterations(self, phase: str) -> List[int]:
        """
        List all available iteration numbers for a given phase.

        Args:
            phase: Phase key — one of "select_seed", "select_commit".

        Returns:
            Sorted list of iteration numbers.
        """
        if phase not in self.PHASE_DIRS:
            return []

        phase_dir = self.context_dir / self.PHASE_DIRS[phase]
        if not phase_dir.exists():
            return []

        iterations = []
        for filepath in phase_dir.glob("iter_*.json"):
            try:
                iter_num = int(filepath.stem.replace("iter_", ""))
                iterations.append(iter_num)
            except ValueError:
                pass
        return sorted(iterations)

    def get_context_files_for_commit(self) -> List[str]:
        """
        Get the list of context files that need to be git-committed.

        Returns:
            List of file paths (absolute).
        """
        files = []

        # evolution_context.json - the main cross-iteration context
        evolution_context_path = self.context_dir / "evolution_context.json"
        if evolution_context_path.exists():
            files.append(str(evolution_context_path))

        # summaries.json - quick-view summaries
        summaries_path = self.context_dir / "summaries.json"
        if summaries_path.exists():
            files.append(str(summaries_path))

        # BOOTSTRAP.md - cumulative cross-iteration lesson (tracked +
        # protected; needs commit to persist).
        # NOTE: plan.md is deliberately NOT added here — it's ephemeral
        # (cleared each iteration) and gitignored, so it never enters git.
        bootstrap_path = self.agent_code_dir / "BOOTSTRAP.md"
        if bootstrap_path.exists():
            files.append(str(bootstrap_path))

        return files

    def get_all_context_files(self) -> List[str]:
        """
        Get all context files (including message history).

        Returns:
            List of file paths (absolute).
        """
        files = self.get_context_files_for_commit()

        # Add all message-history files
        if self.message_history_dir.exists():
            for filepath in sorted(self.message_history_dir.glob("iter_*.json")):
                files.append(str(filepath))

        # Add all phase message files
        for phase_dir_name in self.PHASE_DIRS.values():
            phase_dir = self.context_dir / phase_dir_name
            if phase_dir.exists():
                for filepath in sorted(phase_dir.glob("iter_*.json")):
                    files.append(str(filepath))

        return files

    def clear_message_history(self) -> None:
        """Clear all message-history files (including phase subdirectories)."""
        if self.message_history_dir.exists():
            for filepath in self.message_history_dir.glob("iter_*.json"):
                filepath.unlink()

        # Clear phase-subdirectory messages
        for phase_dir_name in self.PHASE_DIRS.values():
            phase_dir = self.context_dir / phase_dir_name
            if phase_dir.exists():
                for filepath in phase_dir.glob("iter_*.json"):
                    filepath.unlink()

    def clear_all(self) -> None:
        """Clear all context data."""
        self.clear_message_history()

        # Delete evolution_context.json
        evolution_context_path = self.context_dir / "evolution_context.json"
        if evolution_context_path.exists():
            evolution_context_path.unlink()

        # Delete summaries.json
        summaries_path = self.context_dir / "summaries.json"
        if summaries_path.exists():
            summaries_path.unlink()

    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get context-summary info.

        Returns:
            Summary-info dict.
        """
        context = self.load_evolution_context()

        return {
            "context_dir": str(self.context_dir),
            "total_iterations": len(context.iteration_summaries),
            "best_reward": context.best_reward if context.track_best_reward else None,
            "best_iteration": context.best_iteration if context.track_best_reward else None,
            "message_history_files": len(list(self.message_history_dir.glob("iter_*.json"))),
        }

    def list_available_iterations(self) -> List[int]:
        """
        List all available iteration numbers.

        Returns:
            List of iteration numbers.
        """
        iterations = []
        if self.message_history_dir.exists():
            for filepath in self.message_history_dir.glob("iter_*.json"):
                try:
                    iter_num = int(filepath.stem.replace("iter_", ""))
                    iterations.append(iter_num)
                except ValueError:
                    pass
        return sorted(iterations)

    def search_history_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Search historical conversations by keyword.

        Args:
            keyword: Search keyword.

        Returns:
            List of matching results.
        """
        matches = []
        keyword_lower = keyword.lower()

        for iteration in self.list_available_iterations():
            history = self.load_message_history(iteration)
            if not history:
                continue

            for i, msg in enumerate(history.messages):
                content = msg.get('content', '')
                role = msg.get('role', 'unknown')

                # Inspect tool_calls
                tool_calls = msg.get('tool_calls', [])
                tool_names = []
                for tc in tool_calls:
                    tc_name = tc.get('function', {}).get('name', '')
                    if tc_name:
                        tool_names.append(tc_name)

                # Search content
                search_text = f"{content} {' '.join(tool_names)}".lower()

                if keyword_lower in search_text:
                    # Snippet the surrounding context
                    content_lower = content.lower()
                    idx = content_lower.find(keyword_lower)
                    if idx >= 0:
                        context_start = max(0, idx - 100)
                        context_end = min(len(content), idx + len(keyword) + 200)

                        if context_start > 0:
                            space_idx = content.find(' ', context_start)
                            if space_idx != -1 and space_idx < idx:
                                context_start = space_idx + 1
                        if context_end < len(content):
                            space_idx = content.rfind(' ', 0, context_end)
                            if space_idx > idx:
                                context_end = space_idx

                        snippet = content[context_start:context_end]

                        matches.append({
                            'iteration': iteration,
                            'msg_index': i,
                            'role': role,
                            'tool_calls': tool_names,
                            'snippet': snippet.strip()
                        })

        return matches

    def search_in_iteration(self, iteration: int, keyword: str) -> List[Dict[str, Any]]:
        """
        Search for a keyword within a specific iteration.

        Args:
            iteration: Iteration number.
            keyword: Search keyword.

        Returns:
            List of matching results.
        """
        history = self.load_message_history(iteration)
        if not history:
            return []

        matches = []
        keyword_lower = keyword.lower()

        for i, msg in enumerate(history.messages):
            content = msg.get('content', '')
            role = msg.get('role', 'unknown')

            # Inspect tool_calls
            tool_calls = msg.get('tool_calls', [])
            tool_names = []
            for tc in tool_calls:
                tc_name = tc.get('function', {}).get('name', '')
                if tc_name:
                    tool_names.append(tc_name)

            # Search content
            search_text = f"{content} {' '.join(tool_names)}".lower()

            if keyword_lower in search_text:
                content_lower = content.lower()
                idx = content_lower.find(keyword_lower)
                if idx >= 0:
                    context_start = max(0, idx - 100)
                    context_end = min(len(content), idx + len(keyword) + 200)

                    if context_start > 0:
                        space_idx = content.find(' ', context_start)
                        if space_idx != -1 and space_idx < idx:
                            context_start = space_idx + 1
                    if context_end < len(content):
                        space_idx = content.rfind(' ', 0, context_end)
                        if space_idx > idx:
                            context_end = space_idx

                    snippet = content[context_start:context_end]

                    matches.append({
                        'msg_index': i,
                        'role': role,
                        'tool_calls': tool_names,
                        'snippet': snippet.strip()
                    })

        return matches

    @staticmethod
    def format_messages_for_display(messages: list, label: str = "History") -> str:
        """
        Format a list of messages for display.

        Args:
            messages: List of message dicts.
            label: Title label.

        Returns:
            Formatted history string.
        """
        if not messages:
            return f"### {label}\nNo messages."

        lines = [f"### {label} ({len(messages)} messages)\n"]

        for i, msg in enumerate(messages, 1):
            role = msg.get('role', 'unknown')

            if role == 'system':
                content = msg.get('content', '')
                if len(content) > 500:
                    content = content[:500] + '...'
                lines.append(f"**[System]**:\n{content}\n")

            elif role == 'user':
                content = msg.get('content', '')
                if len(content) > 1000:
                    content = content[:1000] + '...'
                lines.append(f"**[User]**:\n{content}\n")

            elif role == 'assistant':
                content = msg.get('content', '')
                tool_calls = msg.get('tool_calls')

                if content:
                    if len(content) > 500:
                        content = content[:500] + '...'
                    lines.append(f"**[Assistant]**:\n{content}\n")

                if tool_calls:
                    for tc in tool_calls:
                        func_name = tc.get('function', {}).get('name', 'unknown')
                        lines.append(f"**[Tool Call]**: {func_name}\n")

            elif role == 'tool':
                content = msg.get('content', '')
                if len(content) > 500:
                    content = content[:500] + '...'
                lines.append(f"**[Tool Result]**:\n{content}\n")

        return "\n".join(lines)

    def format_history_for_display(self, iteration: int) -> str:
        """
        Format the given iteration's history for display.

        Args:
            iteration: Iteration number.

        Returns:
            Formatted history string.
        """
        history = self.load_message_history(iteration)
        if not history or not hasattr(history, 'messages'):
            return f"### Iteration {iteration} History\nNo messages."

        return self.format_messages_for_display(
            history.messages,
            label=f"Iteration {iteration} History",
        )
