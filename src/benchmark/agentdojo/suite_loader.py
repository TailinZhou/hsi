"""
Suite Loader for AgentDojo Benchmark.

Loads task suites from agentDojo and providing utilities
to access tasks, tools, and injection vectors.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    AgentDojoBenchmarkConfig,
    DEFAULT_BENCHMARK_VERSION,
    AVAILABLE_SUITES,
)

logger = logging.getLogger(__name__)


@dataclass
class LoadedSuite:
    """A loaded AgentDojo suite with its components.

    Attributes:
        name: Suite name
        suite: AgentDojo task suite object
        tools: List of available tools/functions
        user_tasks: Dictionary of user tasks
        injection_tasks: Dictionary of injection tasks
        injection_vector_defaults: Default injection vectors
    """
    name: str
    suite: Any  # agentdojo.task_suite.task_suite.TaskSuite
    tools: List[Any] = field(default_factory=list)
    user_tasks: Dict[str, Any] = field(default_factory=dict)
    injection_tasks: Dict[str, Any] = field(default_factory=dict)
    injection_vector_defaults: Dict[str, str] = field(default_factory=dict)
    load_time: datetime = field(default_factory=datetime.utcnow)

    def get_user_task(self, task_id: str) -> Optional[Dict]:
        """Get a user task by ID."""
        return self.user_tasks.get(task_id)

    def get_injection_task(self, task_id: str) -> Optional[Dict]:
        """Get an injection task by ID."""
        return self.injection_tasks.get(task_id)

    def get_all_user_tasks(self) -> Dict[str, Any]:
        """Get all user tasks."""
        return dict(self.user_tasks)

    def get_all_injection_tasks(self) -> Dict[str, Any]:
        """Get all injection tasks."""
        return dict(self.injection_tasks)

    def get_tools(self) -> List:
        """Get available tools."""
        return self.tools

    def get_injection_vector_defaults(self) -> dict[str, str]:
        """Get injection vector defaults."""
        return self.injection_vector_defaults

    @property
    def user_task_count(self) -> int:
        """Get number of user tasks."""
        return len(self.user_tasks)

    @property
    def injection_task_count(self) -> int:
        """Get number of injection tasks."""
        return len(self.injection_tasks)

    @property
    def total_task_count(self) -> int:
        """Get total number of tasks."""
        return self.user_task_count + self.injection_task_count

    @property
    def load_duration_ms(self) -> float:
        """Get suite load duration in milliseconds."""
        return (self.load_time - datetime.utcnow()).total_seconds() * 1000

    def get_user_task_ids(self) -> List[str]:
        """Get list of user task IDs."""
        return list(self.user_tasks.keys())

    def get_injection_task_ids(self) -> List[str]:
        """Get list of injection task IDs."""
        return list(self.injection_tasks.keys())

    def get_tool_names(self) -> List[str]:
        """Get list of tool names."""
        return [t.name if hasattr(t, 'name') else str(t) for t in self.tools]


class SuiteLoader:
    """Loader for AgentDojo task suites.

    This class handles loading and caching of AgentDojo
    task suites, providing utilities to access tasks, tools,
    and injection vectors.
    """

    def __init__(self, config: AgentDojoBenchmarkConfig):
        """Initialize the suite loader.

        Args:
            config: benchmark configuration
        """
        self.config = config
        self._loaded_suites: Dict[str, LoadedSuite] = {}
        self._load_time = datetime.utcnow()

    def load_suite(self, suite_name: str) -> Optional[LoadedSuite]:
        """Load a specific suite by name.

        If the suite is already loaded, return the cached version.

        Args:
            suite_name: Name of the suite to load
                (e.g., "workspace", "slack", "banking", "travel")

        Returns:
            Loaded suite object

        Raises:
            ValueError: if suite is not available
            ImportError: if AgentDojo is not installed
        """
        # Check cache first
        if suite_name in self._loaded_suites:
            logger.info(f"Returning cached suite: {suite_name}")
            return self._loaded_suites[suite_name]

        # Validate suite name
        if suite_name not in AVAILABLE_SUITES:
            raise ValueError(
                f"Unknown suite: {suite_name}. "
                f"Available suites: {AVAILABLE_SUITES}"
            )

        # Load the suite
        start_time = datetime.utcnow()

        try:
            # Import AgentDojo modules
            from agentdojo.task_suite import get_suite

            # Get suite with default version
            suite = get_suite(DEFAULT_BENCHMARK_VERSION, suite_name)

            # Extract components
            loaded = LoadedSuite(
                name=suite.name,
                suite=suite,
                tools=list(suite.tools),
                user_tasks=dict(suite.user_tasks),
                injection_tasks=dict(suite.injection_tasks),
                injection_vector_defaults=suite.get_injection_vector_defaults(),
                load_time=start_time,
            )

            self._loaded_suites[suite_name] = loaded

            logger.info(
                f"Loaded suite: {suite_name} "
                f"({loaded.user_task_count} user tasks, {loaded.injection_task_count} injection tasks)"
            )

            return loaded

        except ImportError as e:
            logger.error(f"Failed to import AgentDojo: {e}")
            raise ImportError(
                "AgentDojo is not installed. "
                "Install it with: pip install agentdojo"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load suite {suite_name}: {e}")
            raise

    def load_all_suites(self) -> Dict[str, LoadedSuite]:
        """Load all configured suites.

        Returns:
            Dictionary mapping suite names to loaded suites
        """
        for suite_name in self.config.suites:
            try:
                self.load_suite(suite_name)
            except Exception as e:
                logger.warning(f"Failed to load suite {suite_name}: {e}")

        return self._loaded_suites

    def get_suite(self, suite_name: str) -> LoadedSuite:
        """Get a loaded suite by name.

        Args:
            suite_name: Name of the suite

        Returns:
            Loaded suite object

        Raises:
            KeyError: if suite is not loaded
        """
        if suite_name not in self._loaded_suites:
            # Try to load it
            self.load_suite(suite_name)

        if suite_name not in self._loaded_suites:
            raise KeyError(f"Suite not loaded: {suite_name}")

        return self._loaded_suites[suite_name]

    def get_user_tasks(self, suite_name: str) -> Dict[str, Any]:
        """Get all user tasks for a suite.

        Args:
            suite_name: Name of the suite

        Returns:
            Dictionary of user tasks
        """
        suite = self.get_suite(suite_name)
        return suite.user_tasks

    def get_injection_tasks(self, suite_name: str) -> Dict[str, Any]:
        """Get all injection tasks for a suite.

        Args:
            suite_name: Name of the suite

        Returns:
            Dictionary of injection tasks
        """
        suite = self.get_suite(suite_name)
        return suite.injection_tasks

    def get_tools(self, suite_name: str) -> List:
        """Get tools available for a suite.

        Args:
            suite_name: Name of the suite

        Returns:
            List of tool functions
        """
        suite = self.get_suite(suite_name)
        return suite.tools

    def get_injection_vector_defaults(self, suite_name: str) -> Dict[str, str]:
        """Get injection vector defaults for a suite.

        Args:
            suite_name: Name of the suite

        Returns:
            Dictionary mapping injection vector names to default values
        """
        suite = self.get_suite(suite_name)
        return suite.injection_vector_defaults

    def create_runtime(self, suite_name: str) -> Any:
        """Create a functions runtime for a suite.

        Args:
            suite_name: Name of the suite

        Returns:
            FunctionsRuntime instance
        """
        from agentdojo.functions_runtime import FunctionsRuntime

        suite = self.get_suite(suite_name)
        return FunctionsRuntime(suite.tools)

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics.

        Returns:
            Dictionary with loader stats
        """
        return {
            "loaded_suites": list(self._loaded_suites.keys()),
            "total_user_tasks": sum(
                s.user_task_count for s in self._loaded_suites.values()
            ),
            "total_injection_tasks": sum(
                s.injection_task_count for s in self._loaded_suites.values()
            ),
            "load_time_seconds": (datetime.utcnow() - self._load_time).total_seconds(),
        }
