"""
History search handlers for AgentActionExecutor.

Extracted from agent_action.py to reduce module size.
Provides read_history_self and related history search methods.
"""

import json
import os
import re
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_action import AgentActionExecutor


class HistoryMixin:
    """Mixin providing history search methods for AgentActionExecutor."""

    # Type hints for attributes set by AgentActionExecutor
    agent_code_dir: str
    context_persistence: object
    logging: object

    def read_history_self(
        self: "AgentActionExecutor",
        targets: List[str] = None
    ) -> str:
        """
        Read historical iteration conversations.

        Only the @history syntax is supported:
        - @history:N: read the conversation history of the N-th iteration
        - @history:keyword: search historical conversations by keyword
        - @history:N:keyword: search a keyword within the N-th iteration

        Args:
            targets: List of history targets to read.
        """
        if not targets:
            return "Error: 'targets' parameter is required. Use @history:N or @history:keyword format."

        results = []

        for target in targets:
            if target.startswith('@history:'):
                result = self._read_special_target(target)
                results.append(result)
            else:
                results.append(
                    f"Error: '{target}' is not a valid history target. "
                    f"Use @history:N (e.g., @history:1) or @history:keyword (e.g., @history:error). "
                    f"For reading code files, use bash 'cat <filename>'."
                )

        return "\n\n".join(results)

    def _read_special_target(self: "AgentActionExecutor", target: str) -> str:
        """
        Read a special target (@history:N or @history:keyword or @history:N:keyword).

        Args:
            target: Special target name.

        Returns:
            Target content.
        """
        # @history:N or @history:keyword or @history:N:keyword
        if target.startswith('@history:'):
            param = target.split(':', 1)[1]

            # New: support the iteration:keyword combination
            if ':' in param:
                parts = param.split(':', 1)
                try:
                    iteration_num = int(parts[0])
                    keyword = parts[1]
                    if keyword:  # Ensure the keyword is non-empty
                        return self._search_in_iteration(iteration_num, keyword)
                except ValueError:
                    pass  # First part is not a number, fall back to keyword search

            # Original logic
            try:
                iteration_num = int(param)
                return self._load_history_by_iteration(iteration_num)
            except ValueError:
                return self._search_history_by_keyword(param)

        return f"Error: Unknown special target '{target}'"

    def _search_history_by_keyword(self: "AgentActionExecutor", keyword: str) -> str:
        """Search historical conversations by keyword."""
        # Use context_persistence for searching
        if self.context_persistence:
            matches = self.context_persistence.search_history_by_keyword(keyword)

            if not matches:
                return f"### Keyword Search: '{keyword}'\nNo matches found in history."

            # Format the output
            lines = [f"### Keyword Search: '{keyword}' ({len(matches)} matches)\n"]

            # Group by iteration
            by_iteration = {}
            for m in matches:
                it = m['iteration']
                if it not in by_iteration:
                    by_iteration[it] = []
                by_iteration[it].append(m)

            for iteration in sorted(by_iteration.keys(), reverse=True):
                iter_matches = by_iteration[iteration]
                lines.append(f"\n**Iteration {iteration}** ({len(iter_matches)} matches)")

                for m in iter_matches[:5]:  # Show at most 5 matches per iteration
                    role = m['role']
                    tools = f" (tool: {', '.join(m['tool_calls'])})" if m['tool_calls'] else ""
                    lines.append(f"  - [{role}]{tools}: ...{m['snippet']}...")

                if len(iter_matches) > 5:
                    lines.append(f"  - ... and {len(iter_matches) - 5} more matches")

            return "\n".join(lines)

        # Fallback: direct file access (backward compatible)
        return self._search_keyword_fallback(keyword)

    def _search_keyword_fallback(self: "AgentActionExecutor", keyword: str) -> str:
        """Filesystem fallback implementation for keyword search."""
        if not self.agent_code_dir:
            return "Error: No agent_code_dir configured."

        context_dir = os.path.join(self.agent_code_dir, '.context')

        if not os.path.isdir(context_dir):
            return f"Error: No history context found. Context directory does not exist: {context_dir}"

        matches = []
        keyword_lower = keyword.lower()

        # Iterate over all history files
        for filename in os.listdir(context_dir):
            if not filename.startswith('history_') or not filename.endswith('.json'):
                continue

            filepath = os.path.join(context_dir, filename)

            # Extract the iteration number
            try:
                iteration_num = int(filename.replace('history_', '').replace('.json', ''))
            except ValueError:
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                messages = data.get('messages', [])

                # Search for the keyword in each message
                for i, msg in enumerate(messages):
                    content = msg.get('content', '')
                    role = msg.get('role', 'unknown')

                    # Check tool_calls
                    tool_calls = msg.get('tool_calls', [])
                    tool_names = []
                    for tc in tool_calls:
                        tc_name = tc.get('function', {}).get('name', '')
                        if tc_name:
                            tool_names.append(tc_name)

                    # Search the content
                    search_text = f"{content} {' '.join(tool_names)}".lower()

                    if keyword_lower in search_text:
                        # Extract context
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

                            # Highlight the keyword
                            highlighted = snippet.replace(
                                keyword,
                                f"**{keyword}**"
                            )

                            matches.append({
                                'iteration': iteration_num,
                                'msg_index': i,
                                'role': role,
                                'tool_calls': tool_names,
                                'snippet': highlighted.strip()
                            })

            except Exception as e:
                self.logging(f"Warning: Failed to search in {filename}: {e}")
                continue

        if not matches:
            return f"### Keyword Search: '{keyword}'\nNo matches found in history."

        # Format the output
        lines = [f"### Keyword Search: '{keyword}' ({len(matches)} matches)\n"]

        # Group by iteration
        by_iteration = {}
        for m in matches:
            it = m['iteration']
            if it not in by_iteration:
                by_iteration[it] = []
            by_iteration[it].append(m)

        for iteration in sorted(by_iteration.keys(), reverse=True):
            iter_matches = by_iteration[iteration]
            lines.append(f"\n**Iteration {iteration}** ({len(iter_matches)} matches)")

            for m in iter_matches[:5]:  # Show at most 5 matches per iteration
                role = m['role']
                tools = f" (tool: {', '.join(m['tool_calls'])})" if m['tool_calls'] else ""
                lines.append(f"  - [{role}]{tools}: ...{m['snippet']}...")

            if len(iter_matches) > 5:
                lines.append(f"  - ... and {len(iter_matches) - 5} more matches")

        return "\n".join(lines)

    def _search_in_iteration(self: "AgentActionExecutor", iteration: int, keyword: str) -> str:
        """Search for a keyword within a specified iteration."""
        # Use context_persistence for searching
        if self.context_persistence:
            matches = self.context_persistence.search_in_iteration(iteration, keyword)

            if not matches:
                return f"### Search in Iteration {iteration}: '{keyword}'\nNo matches found."

            # Format the output
            lines = [f"### Search in Iteration {iteration}: '{keyword}' ({len(matches)} matches)\n"]

            for m in matches[:10]:  # Show at most 10 matches
                role = m['role']
                tools = f" (tool: {', '.join(m['tool_calls'])})" if m['tool_calls'] else ""
                lines.append(f"  - [{role}]{tools}: ...{m['snippet']}...")

            if len(matches) > 10:
                lines.append(f"  - ... and {len(matches) - 10} more matches")

            return "\n".join(lines)

        # Fallback: direct file access
        return self._search_in_iteration_fallback(iteration, keyword)

    def _search_in_iteration_fallback(self: "AgentActionExecutor", iteration: int, keyword: str) -> str:
        """Filesystem fallback implementation for in-iteration search."""
        if not self.agent_code_dir:
            return "Error: No agent_code_dir configured."

        filepath = os.path.join(self.agent_code_dir, '.evolution_context', 'main_evolve', f'iter_{iteration}.json')

        if not os.path.exists(filepath):
            return f"Error: No history found for iteration {iteration}"

        keyword_lower = keyword.lower()
        matches = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            messages = data.get('messages', [])

            # Search for the keyword in each message
            for i, msg in enumerate(messages):
                content = msg.get('content', '')
                role = msg.get('role', 'unknown')

                # Check tool_calls
                tool_calls = msg.get('tool_calls', [])
                tool_names = []
                for tc in tool_calls:
                    tc_name = tc.get('function', {}).get('name', '')
                    if tc_name:
                        tool_names.append(tc_name)

                # Search the content
                search_text = f"{content} {' '.join(tool_names)}".lower()

                if keyword_lower in search_text:
                    # Extract context
                    content_lower = content.lower()
                    idx = content_lower.find(keyword_lower)
                    if idx >= 0:
                        context_start = max(0, idx - 100)
                        context_end = min(len(content), idx + len(keyword) + 200)

                        if context_start > 0:
                            # Try to start at a word boundary
                            space_idx = content.find(' ', context_start)
                            if space_idx != -1 and space_idx < idx:
                                context_start = space_idx + 1
                        if context_end < len(content):
                            # Try to end at a word boundary
                            space_idx = content.rfind(' ', 0, context_end)
                            if space_idx > idx:
                                context_end = space_idx

                        snippet = content[context_start:context_end]

                        # Highlight the keyword (preserve original case)
                        highlighted = self._highlight_keyword(snippet, keyword)

                        matches.append({
                            'msg_index': i,
                            'role': role,
                            'tool_calls': tool_names,
                            'snippet': highlighted.strip()
                        })

        except Exception as e:
            return f"Error searching in iteration {iteration}: {e}"

        if not matches:
            return f"### Search in Iteration {iteration}: '{keyword}'\nNo matches found."

        # Format the output
        lines = [f"### Search in Iteration {iteration}: '{keyword}' ({len(matches)} matches)\n"]

        for m in matches[:10]:  # Show at most 10 matches
            role = m['role']
            tools = f" (tool: {', '.join(m['tool_calls'])})" if m['tool_calls'] else ""
            lines.append(f"  - [{role}]{tools}: ...{m['snippet']}...")

        if len(matches) > 10:
            lines.append(f"  - ... and {len(matches) - 10} more matches")

        return "\n".join(lines)

    def _highlight_keyword(self: "AgentActionExecutor", text: str, keyword: str) -> str:
        """Highlight the keyword in text (case-insensitive)."""
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        return pattern.sub(f"**{keyword}**", text)

    def _format_message_history(self: "AgentActionExecutor", history, label: str = "History") -> str:
        """Format message history for display."""
        if not history or not hasattr(history, 'messages'):
            return f"### {label}\nNo messages."

        lines = [f"### {label} ({len(history.messages)} messages)\n"]

        for i, msg in enumerate(history.messages, 1):
            role = msg.get('role', 'unknown')

            if role == 'system':
                content = msg.get('content', '')
                # Truncate overly long system prompts
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

    def _load_history_by_iteration(self: "AgentActionExecutor", iteration: int) -> str:
        """Load the conversation history of a specified iteration."""
        # Use context_persistence to load history
        if self.context_persistence:
            history = self.context_persistence.load_message_history(iteration)

            if history is None:
                # List available iterations
                available = self.context_persistence.list_available_iterations()
                if available:
                    return f"Error: No history for iteration {iteration}. Available completed iterations: {sorted(available)}"
                else:
                    return f"Error: No history files found. No iterations have been completed yet."

            return self.context_persistence.format_history_for_display(iteration)

        # Fallback: direct file access
        return self._load_history_fallback(iteration)

    def _load_history_fallback(self: "AgentActionExecutor", iteration: int) -> str:
        """Filesystem fallback implementation for loading history."""
        if not self.agent_code_dir:
            return "Error: No agent_code_dir configured."

        # Find the history file
        context_dir = os.path.join(self.agent_code_dir, '.evolution_context', 'main_evolve')
        history_file = os.path.join(context_dir, f'iter_{iteration}.json')

        if not os.path.isfile(history_file):
            # List available history files
            available = []
            if os.path.isdir(context_dir):
                for f in os.listdir(context_dir):
                    if f.startswith('iter_') and f.endswith('.json'):
                        try:
                            num = int(f.replace('iter_', '').replace('.json', ''))
                            available.append(num)
                        except ValueError:
                            pass

            if available:
                return f"Error: No history for iteration {iteration}. Available completed iterations: {sorted(available)}"
            else:
                return f"Error: No history files found in {context_dir}"

        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            messages = data.get('messages', [])

            # Create a temporary MessageHistory object for formatting
            from .state import MessageHistory
            history = MessageHistory(messages=messages)
            return self._format_message_history(history, f"Iteration {iteration} History")

        except Exception as e:
            return f"Error loading history for iteration {iteration}: {e}"
