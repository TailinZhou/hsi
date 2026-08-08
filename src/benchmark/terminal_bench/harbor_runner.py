"""Harbor CLI runner for Terminal-Bench 2.

Encapsulates `harbor run` CLI invocations, temporary directory management,
harness code copying, and result collection.
"""

import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from typing import Dict, List, Optional

from benchmark.terminal_bench.bridge_agent_template import write_bridge_agent
from benchmark.terminal_bench.config import TerminalBenchConfig
from benchmark.terminal_bench.result_parser import (
    TB2TaskResult,
    parse_harbor_output,
)

logger = logging.getLogger(__name__)

# Harbor on WSL2 patches:
# 1) _chown_to_host_user is a no-op (Docker Desktop handles ownership).
# 2) _run_docker_compose_command uses synchronous subprocess via
#    asyncio.to_thread, since the agent now runs in a separate thread
#    (asyncio.to_thread) and no longer uses nest_asyncio — but keeping
#    the subprocess patch as a safety net for any remaining edge cases.
_HARBOR_WRAPPER_TEMPLATE = textwrap.dedent("""\
    #!/usr/bin/env python3
    import asyncio
    import os
    import shlex
    import subprocess
    import sys

    def _patch_harbor_docker():
        try:
            from harbor.environments.docker.docker import DockerEnvironment
            from harbor.environments.docker.docker import ExecResult
        except ImportError:
            return

        # 1) No-op chown — Docker Desktop handles ownership transparently.
        async def _noop_chown(self, path, recursive=False):
            pass
        DockerEnvironment._chown_to_host_user = _noop_chown

        # 2) Replace _run_docker_compose_command with a subprocess
        #    implementation that is immune to the corrupted asyncio state
        #    left by nest_asyncio.
        #    Uses asyncio.to_thread so subprocess.run does NOT block the
        #    event loop — critical because setup() makes 6-8 docker exec calls
        #    that can each take 10-60s on WSL2, and blocking the loop prevents
        #    asyncio.wait_for timeouts from firing.
        _orig = DockerEnvironment._run_docker_compose_command

        async def _sync_docker_compose_command(
            self, command, check=True, timeout_sec=None
        ):
            full_command = [
                "docker", "compose",
                "--project-name", self.session_id.lower(),
                "--project-directory",
                str(self.environment_dir.resolve().absolute()),
            ]
            for p in self._docker_compose_paths:
                full_command.extend(["-f", str(p.resolve().absolute())])
            full_command.extend(command)

            env = self._env_vars.to_env_dict(include_os_env=True)
            if self._compose_task_env:
                env.update(self._compose_task_env)
            if self._persistent_env:
                env.update(self._persistent_env)

            def _run_subprocess():
                return subprocess.run(
                    full_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_sec,
                    env=env,
                )

            try:
                proc = await asyncio.to_thread(_run_subprocess)
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"Command timed out after {timeout_sec} seconds"
                )
            except FileNotFoundError:
                raise RuntimeError("docker compose not found")

            result = ExecResult(
                stdout=proc.stdout,
                stderr=None,
                return_code=proc.returncode,
            )
            if check and result.return_code != 0:
                raise RuntimeError(
                    f"Docker compose command failed for environment "
                    f"{self.environment_name}. "
                    f"Command: {' '.join(full_command)}\\n"
                    f"Exit code: {result.return_code}\\n"
                    f"Output: {result.stdout or ''}"
                )
            return result

        DockerEnvironment._run_docker_compose_command = (
            _sync_docker_compose_command
        )

    _patch_harbor_docker()

    from harbor.cli.main import app
    if __name__ == "__main__":
        sys.argv[0] = sys.argv[0].removesuffix(".exe")
        sys.exit(app())
""")


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07|\x1b\[[\?]?[0-9;]*[a-zA-Z]")


class _LogStreamer:
    """Background thread that tails agent pane files in harbor output."""

    def __init__(self, output_dir: str):
        self._output_dir = output_dir
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._offsets: Dict[str, int] = {}

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            self._poll()
            self._stop.wait(1.0)

    def _poll(self):
        if not os.path.isdir(self._output_dir):
            return
        for job_dir in os.listdir(self._output_dir):
            job_path = os.path.join(self._output_dir, job_dir)
            if not os.path.isdir(job_path):
                continue
            for entry in os.listdir(job_path):
                pane_path = os.path.join(job_path, entry, "agent", "terminus_2.pane")
                offset = self._offsets.get(pane_path, 0)
                try:
                    with open(pane_path, "r", errors="replace") as f:
                        f.seek(offset)
                        new_data = f.read()
                        self._offsets[pane_path] = f.tell()
                except (OSError, FileNotFoundError):
                    continue
                if not new_data:
                    continue
                task_name = entry.split("__")[0]
                clean = _ANSI_RE.sub("", new_data)
                for line in clean.splitlines():
                    stripped = line.strip()
                    if stripped:
                        print(f"  [{task_name}] {stripped}", flush=True)


class HarborRunner:
    """Manages Harbor CLI execution for TB2 tasks.

    Workflow:
    1. Create temp directory with bridge_agent.py + harness_code/
    2. Copy agent code into harness_code/
    3. Run `harbor run` CLI
    4. Parse results from output directory
    """

    def __init__(self, config: TerminalBenchConfig):
        self.config = config
        self._tmp_dir: Optional[str] = None

    def run(
        self,
        agent_code_dir: str,
        task_ids: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> List[TB2TaskResult]:
        """Execute Harbor run and return parsed results.

        Args:
            agent_code_dir: Path to the agent's harness code directory
                (contains harness.py, prompts.py, etc.)
            task_ids: Optional list of task IDs to filter. None = all tasks.
            verbose: Stream agent interaction logs in real time.

        Returns:
            List of TB2TaskResult from all trials.
        """
        # 1. Prepare temporary directory
        work_dir = self._prepare_work_dir(agent_code_dir)
        logger.info(f"Harbor work directory: {work_dir}")
        print(f"  [Harbor] Work dir: {work_dir}", flush=True)

        self._verbose = verbose

        # 2. Build harbor command
        cmd = self._build_command(work_dir, task_ids)
        logger.info(f"Running: {' '.join(cmd)}")

        # 3. Execute
        output_dir = self.config.output_dir or os.path.join(work_dir, "harbor_output")
        env = self._build_env()

        # Start log streaming thread if verbose
        streamer = None
        if verbose:
            streamer = _LogStreamer(output_dir)
            streamer.start()

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=work_dir,
            )
            stdout, stderr = proc.communicate(timeout=self.config.overall_timeout)
            returncode = proc.returncode

            print(f"  [Harbor] Return code: {returncode}", flush=True)
            if stdout and self._verbose:
                for line in stdout.split('\n')[-50:]:
                    if line.strip():
                        print(f"  [Harbor stdout] {line}", flush=True)
            if returncode != 0:
                logger.error(f"Harbor exited with code {returncode}")
                logger.error(f"stderr: {stderr[:3000]}")
                if stderr:
                    stderr_lines = stderr.split('\n')
                    for line in stderr_lines[-30:]:
                        if line.strip():
                            print(f"  [Harbor] {line}", flush=True)
            else:
                logger.info("Harbor completed successfully")
                print(f"  [Harbor] Completed, output at: {output_dir}", flush=True)

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            logger.error(f"Harbor run timed out after {self.config.overall_timeout}s")
            returncode = -1
        except FileNotFoundError:
            logger.error(
                f"Harbor executable not found: {self.config.harbor_executable}. "
                "Install with: pip install harbor-ai or uv tool install harbor-ai"
            )
            returncode = -1
        finally:
            if streamer:
                streamer.stop()

        # 4. Parse results
        if os.path.isdir(output_dir):
            if self._verbose:
                print(f"  [Harbor] Output directory structure:", flush=True)
                for root, dirs, files in os.walk(output_dir):
                    rel = os.path.relpath(root, output_dir)
                    print(f"    {rel}/", flush=True)
                    for f in files:
                        fpath = os.path.join(root, f)
                        if f.endswith(".json"):
                            import json as _json
                            try:
                                with open(fpath) as _f:
                                    _data = _json.load(_f)
                                print(f"      {f}: {_json.dumps(_data, indent=2)[:500]}", flush=True)
                            except Exception:
                                print(f"      {f}: (parse error)", flush=True)
                        else:
                            print(f"      {f}", flush=True)
        else:
            print(f"  [Harbor] Output directory NOT found: {output_dir}", flush=True)

        results = parse_harbor_output(output_dir)
        print(f"  [Harbor] Parsed {len(results)} results", flush=True)
        return results

    def _prepare_work_dir(self, agent_code_dir: str) -> str:
        """Create temporary work directory with bridge_agent.py and harness_code/."""
        work_dir = tempfile.mkdtemp(prefix="tb2_bridge_")
        self._tmp_dir = work_dir

        # Generate bridge_agent.py
        write_bridge_agent(
            target_dir=work_dir,
            litellm_model=self.config.litellm_model,
        )

        # Write harbor wrapper that patches _chown_to_host_user for WSL2
        self._write_harbor_wrapper(work_dir)

        # Copy harness code
        harness_dest = os.path.join(work_dir, "harness_code")
        os.makedirs(harness_dest, exist_ok=True)

        if os.path.isdir(agent_code_dir):
            for fname in os.listdir(agent_code_dir):
                if fname.endswith(".py"):
                    src = os.path.join(agent_code_dir, fname)
                    dst = os.path.join(harness_dest, fname)
                    shutil.copy2(src, dst)
                    logger.debug(f"Copied {src} -> {dst}")

        # Ensure __init__.py exists so harness_code is a proper package
        init_path = os.path.join(harness_dest, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                f.write("")

        return work_dir

    def _write_harbor_wrapper(self, work_dir: str) -> str:
        """Write a harbor CLI wrapper that patches DockerEnvironment.

        The wrapper monkey-patches _run_docker_compose_command to use
        synchronous subprocess calls instead of asyncio.create_subprocess_exec,
        working around CancelledError caused by nest_asyncio corrupting the
        event loop's subprocess transport state.
        """
        wrapper_path = os.path.join(work_dir, "harbor_wrapper.py")
        with open(wrapper_path, "w") as f:
            f.write(_HARBOR_WRAPPER_TEMPLATE)
        os.chmod(wrapper_path, os.stat(wrapper_path).st_mode | stat.S_IEXEC)
        self._harbor_wrapper = wrapper_path
        return wrapper_path

    def _resolve_dataset_path(self) -> Optional[str]:
        """Resolve dataset to a local path if cached, otherwise download it.

        Returns a local directory path usable with `harbor run --path`,
        or None if the dataset must be fetched via `-d` (remote registry).
        """
        dataset = self.config.dataset  # e.g. "terminal-bench@2.0"
        if not dataset:
            return None

        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "harbor", "datasets")
        dataset_name = dataset.split("@")[0]
        local_dir = os.path.join(cache_dir, dataset_name)

        # Already cached locally — use it directly
        if os.path.isdir(local_dir) and os.listdir(local_dir):
            logger.info(f"Using cached dataset: {local_dir}")
            print(f"  [Harbor] Using cached dataset: {local_dir}", flush=True)
            return local_dir

        # Download and cache for offline use
        logger.info(f"Downloading dataset {dataset} for local caching...")
        print(f"  [Harbor] Downloading dataset {dataset}...", flush=True)
        try:
            os.makedirs(cache_dir, exist_ok=True)
            download_proc = subprocess.run(
                ["harbor", "download", dataset, "--export", "-o", cache_dir],
                capture_output=True, text=True, timeout=300,
            )
            if download_proc.returncode == 0 and os.path.isdir(local_dir):
                logger.info(f"Dataset cached to {local_dir}")
                print(f"  [Harbor] Dataset cached to {local_dir}", flush=True)
                return local_dir
            else:
                logger.warning(
                    f"harbor download failed (rc={download_proc.returncode}): "
                    f"{download_proc.stderr[:500]}"
                )
                print(f"  [Harbor] Download failed, falling back to remote registry", flush=True)
                return None
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"harbor download error: {e}")
            print(f"  [Harbor] Download error, falling back to remote registry", flush=True)
            return None

    def _build_command(
        self,
        work_dir: str,
        task_ids: Optional[List[str]],
    ) -> List[str]:
        """Build the harbor run CLI command."""
        # Use the wrapper (with chown patch) on Linux/WSL2 for Docker sandbox
        executable = self.config.harbor_executable
        if (
            self.config.sandbox in ("docker", None)
            and hasattr(self, "_harbor_wrapper")
            and self._harbor_wrapper
        ):
            executable = sys.executable
            cmd = [executable, self._harbor_wrapper, "run"]
        else:
            cmd = [executable, "run"]

        # Resolve dataset: prefer local cache, fall back to remote registry
        dataset_path = self._resolve_dataset_path()
        if dataset_path:
            cmd.extend(["--path", str(dataset_path)])
        else:
            cmd.extend(["-d", self.config.dataset])
        cmd.extend([
            "--agent-import-path", "bridge_agent:GodelAgentOnHarbor",
            "-m", self.config.litellm_model,
            "-n", str(self.config.max_concurrent),
        ])

        # Execution environment: docker (default), runloop, daytona, e2b, modal
        if self.config.sandbox and self.config.sandbox != "docker":
            cmd.extend(["-e", self.config.sandbox])

        # Keep Docker images between runs to avoid rebuilding (~60s saved per run).
        # Containers are still removed; only images and volumes are preserved.
        cmd.append("--no-delete")

        # Timeout multipliers for WSL2 Docker (where containers start slower)
        if self.config.agent_setup_timeout_multiplier is not None:
            cmd.extend([
                "--agent-setup-timeout-multiplier",
                str(self.config.agent_setup_timeout_multiplier),
            ])
        if self.config.environment_build_timeout_multiplier is not None:
            cmd.extend([
                "--environment-build-timeout-multiplier",
                str(self.config.environment_build_timeout_multiplier),
            ])

        # Agent execution timeout multiplier (task.toml default is 900s)
        # task_timeout in config.yaml is the desired absolute timeout;
        # convert to multiplier: task_timeout / 900
        if self.config.task_timeout and self.config.task_timeout != 600:
            multiplier = self.config.task_timeout / 900.0
            cmd.extend(["--agent-timeout-multiplier", str(multiplier)])

        # Output directory
        output_dir = self.config.output_dir or os.path.join(work_dir, "harbor_output")
        cmd.extend(["--jobs-dir", output_dir])

        # Task ID filter
        if task_ids:
            for tid in task_ids:
                cmd.extend(["-i", tid])

        # Pass litellm config to agent via --ae (agent-env)
        cmd.extend(["--ae", f"TB2_LITELLM_MODEL={self.config.litellm_model}"])
        if self.config.litellm_api_base:
            cmd.extend(["--ae", f"TB2_LITELLM_API_BASE={self.config.litellm_api_base}"])
        if self.config.litellm_api_key:
            cmd.extend(["--ae", f"TB2_LITELLM_API_KEY={self.config.litellm_api_key}"])
        if self.config.litellm_temperature is not None:
            cmd.extend(["--ae", f"TB2_LITELLM_TEMPERATURE={self.config.litellm_temperature}"])
        if self.config.litellm_thinking_enabled is not None:
            cmd.extend(["--ae", f"TB2_THINKING_ENABLED={str(self.config.litellm_thinking_enabled).lower()}"])
        if self.config.litellm_reasoning_effort:
            cmd.extend(["--ae", f"TB2_REASONING_EFFORT={self.config.litellm_reasoning_effort}"])

        return cmd

    def _build_env(self) -> Dict[str, str]:
        """Build environment variables for the subprocess."""
        env = os.environ.copy()
        # Forward API keys and TB2 config
        env.update(self.config.env_vars)
        env["TB2_LITELLM_MODEL"] = self.config.litellm_model
        if self.config.litellm_api_base:
            env["TB2_LITELLM_API_BASE"] = self.config.litellm_api_base
        if self.config.litellm_api_key:
            env["TB2_LITELLM_API_KEY"] = self.config.litellm_api_key
        if self.config.litellm_temperature is not None:
            env["TB2_LITELLM_TEMPERATURE"] = str(self.config.litellm_temperature)
        if self.config.litellm_thinking_enabled is not None:
            env["TB2_THINKING_ENABLED"] = str(self.config.litellm_thinking_enabled).lower()
        if self.config.litellm_reasoning_effort:
            env["TB2_REASONING_EFFORT"] = self.config.litellm_reasoning_effort
        return env

    def cleanup(self):
        """Remove temporary directory."""
        if self._tmp_dir and os.path.isdir(self._tmp_dir):
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            logger.debug(f"Cleaned up {self._tmp_dir}")
            self._tmp_dir = None

    def __del__(self):
        self.cleanup()


def cleanup_harbor_docker_resources():
    """Clean up leftover Harbor Docker containers and networks.

    Should be called once at the end of main.py / main_init.py / main_final.py,
    NOT after each Harbor run (containers are kept between evaluations for speed).
    """
    import subprocess

    try:
        # Remove Harbor containers (matching *__*-main-1 naming pattern)
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.ID}} {{.Names}}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return

        harbor_containers = [
            line.split()[0]
            for line in result.stdout.strip().splitlines()
            if "__" in line and "-main-1" in line
        ]

        if harbor_containers:
            subprocess.run(
                ["docker", "rm", "-f"] + harbor_containers,
                capture_output=True, text=True, timeout=30,
            )
            logger.info(f"Cleaned up {len(harbor_containers)} Harbor containers")

        # Prune unused networks
        subprocess.run(
            ["docker", "network", "prune", "-f"],
            capture_output=True, text=True, timeout=10,
        )

    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
