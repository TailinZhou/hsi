"""Parse Harbor output results for Terminal-Bench 2.

Harbor outputs results in a directory structure:
    jobs/<job_name>/
      <task_id>__<trial>/
        result.json

Each result.json has:
{
  "verifier_result": { "rewards": { "reward": 0|1 } },
  "agent_result": { ... }
}
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TB2TaskResult:
    """Parsed result for a single TB2 task trial."""

    task_id: str
    trial: int
    passed: bool
    reward: float
    metadata: Dict = field(default_factory=dict)
    agent_result: Optional[Dict] = None
    result_path: Optional[str] = None
    interaction_log: List[Dict] = field(default_factory=list)
    api_messages: List[Dict] = field(default_factory=list)


def parse_harbor_output(output_dir: str) -> List[TB2TaskResult]:
    """Scan Harbor output directory and parse all result.json files.

    Args:
        output_dir: Path to the Harbor jobs output directory.

    Returns:
        List of TB2TaskResult, one per trial directory found.
    """
    results: List[TB2TaskResult] = []

    if not os.path.isdir(output_dir):
        logger.warning(f"Harbor output directory not found: {output_dir}")
        return results

    # Walk the directory tree for result.json files
    for root, dirs, files in os.walk(output_dir):
        if "result.json" not in files:
            continue

        result_path = os.path.join(root, "result.json")
        trial_dir = os.path.basename(root)

        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to parse {result_path}: {e}")
            task_id, trial = _parse_trial_dir(trial_dir)
            results.append(TB2TaskResult(
                task_id=task_id or trial_dir,
                trial=trial,
                passed=False,
                reward=0.0,
                metadata={"error": f"Failed to parse result.json: {e}"},
                result_path=result_path,
            ))
            continue

        # Skip job-level result.json (has no trial_name)
        if "trial_name" not in data:
            logger.debug(f"Skipping job-level result.json: {result_path}")
            continue

        # Use task_name from result.json (reliable) rather than parsing dir name
        task_id = data.get("task_name", "")
        _, trial = _parse_trial_dir(trial_dir)
        if not task_id:
            task_id, trial = _parse_trial_dir(trial_dir)

        # Extract reward from verifier_result
        # Use `or {}` to handle null values (key exists but value is None)
        verifier = data.get("verifier_result") or {}
        rewards = verifier.get("rewards") or {}
        reward_val = rewards.get("reward", 0)
        passed = bool(reward_val)

        # Extract agent_result metadata
        agent_result = data.get("agent_result") or {}
        metadata = {
            "verifier_result": verifier,
        }

        # Try to extract token/cost info from agent_result
        if isinstance(agent_result, dict):
            for key in ("total_tokens", "prompt_tokens", "completion_tokens", "cost", "num_turns"):
                if key in agent_result:
                    metadata[key] = agent_result[key]

        # Look for interaction_log.json in the same directory or subdirectories
        interaction_log = _parse_interaction_log(root)

        # Look for api_messages.json (raw OpenAI messages for prompt cache reuse)
        api_messages = _parse_api_messages(root)

        # Fallback: try extracting from agent_result if file-based log not found
        if not interaction_log and isinstance(agent_result, dict):
            interaction_log = agent_result.get("interaction_log", [])

        # Include exception.txt content if trial errored
        exception_path = os.path.join(root, "exception.txt")
        if os.path.isfile(exception_path):
            try:
                with open(exception_path, "r", encoding="utf-8") as f:
                    metadata["exception"] = f.read().strip()
            except OSError:
                pass

        results.append(TB2TaskResult(
            task_id=task_id,
            trial=trial,
            passed=passed,
            reward=float(reward_val),
            metadata=metadata,
            agent_result=agent_result,
            result_path=result_path,
            interaction_log=interaction_log,
            api_messages=api_messages,
        ))

    logger.info(f"Parsed {len(results)} results from {output_dir}")
    return results


def aggregate_results(
    results: List[TB2TaskResult],
) -> Dict[str, TB2TaskResult]:
    """Aggregate multiple trials per task_id into a single result.

    For each task_id, keeps the best result (highest reward).
    """
    best: Dict[str, TB2TaskResult] = {}
    for r in results:
        if r.task_id not in best or r.reward > best[r.task_id].reward:
            best[r.task_id] = r
    return best


def _parse_trial_dir(dir_name: str) -> tuple:
    """Parse '<task_id>__<trial>' directory name.

    Returns (task_id, trial_number).
    """
    match = re.match(r"^(.+)__(\d+)$", dir_name)
    if match:
        return match.group(1), int(match.group(2))
    return dir_name, 0


def _parse_interaction_log(trial_dir: str) -> List[Dict]:
    """Look for interaction_log.json in trial directory or subdirectories.

    Harbor output structure:
        <trial_dir>/interaction_log.json       (written to trial root)
        <trial_dir>/agent/interaction_log.json  (written to logging_dir)

    Falls back to recursive search if standard paths fail.

    Returns parsed list or empty list if not found.
    """
    candidates = [
        os.path.join(trial_dir, "interaction_log.json"),
        os.path.join(trial_dir, "agent", "interaction_log.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    logger.info(f"Loaded {len(data)} interaction entries from {path}")
                    return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to parse interaction log {path}: {e}")

    # Fallback: recursive search under trial_dir
    if os.path.isdir(trial_dir):
        candidates_set = set(candidates)
        for root, dirs, files in os.walk(trial_dir):
            if "interaction_log.json" in files:
                path = os.path.join(root, "interaction_log.json")
                if path not in candidates_set:
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            logger.info(f"Loaded {len(data)} interaction entries from {path} (fallback)")
                            return data
                    except (json.JSONDecodeError, OSError) as e:
                        logger.warning(f"Failed to parse interaction log {path}: {e}")
    return []


def _parse_api_messages(trial_dir: str) -> List[Dict]:
    """Look for api_messages.json in trial directory or subdirectories.

    api_messages.json contains the raw OpenAI messages sent to the API,
    which can be reused for prompt cache hits in LLM summary calls.

    Returns parsed list or empty list if not found.
    """
    candidates = [
        os.path.join(trial_dir, "api_messages.json"),
        os.path.join(trial_dir, "agent", "api_messages.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    logger.info(f"Loaded {len(data)} api_messages from {path}")
                    return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to parse api_messages {path}: {e}")
    return []
