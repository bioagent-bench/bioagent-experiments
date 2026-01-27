#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
import random
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Sequence

import click
from src.dataset import DataSet
from src.logs import RunConfig, configure_logging
from src.models import MODELS
from src.tools import tools_mapping_dict
from src.system_prompts import prompts

OTEL_SINK_HOST = "127.0.0.1:4317"
PROJECT_ROOT = Path(__file__).resolve().parent
# Populated in main() after validating environment variables.
RUN_LOGS: Path | None = None
METADATA_PATH = Path("~/bioagent-bench/src/task_metadata.json").expanduser()
DATA_ROOT = Path("~/bioagent-data").expanduser()
REQUIRED_ENV_VARS = ("RUN_LOGS", "OPENROUTER_API_KEY")


def _ensure_required_env_vars() -> None:
    missing_env = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing_env:
        missing = ", ".join(missing_env)
        raise click.ClickException(
            f"Cannot run evaluations because the following environment variables are unset: {missing}"
        )

def _build_run_config(
    task: DataSet,
    system_prompt_name: str,
    use_reference_data: bool,
    run_logs: Path,
    experiment_name: str,
    model: str,
    tool_names: Sequence[str],
    otel_sink_host: str | None = None,
) -> RunConfig:
    timestamp = datetime.now()

    otel_sink_host = otel_sink_host or OTEL_SINK_HOST
    run_hash = str(uuid.uuid4())
    run_root = run_logs / experiment_name / task.task_id / run_hash
    metadata_path = run_logs / "runs" / f"{run_hash}.json"

    otel_path = run_logs / "otel" / f"otlp-{run_hash}.ndjson"
    return RunConfig(
        metadata_path=metadata_path,
        run_hash=run_hash,
        use_reference_data=use_reference_data,
        timestamp=timestamp,
        task_id=task.task_id,
        task_prompt=task.task_prompt,
        num_tools=len(tool_names),
        tool_names=list(tool_names),
        system_prompt_name=system_prompt_name,
        experiment_name=experiment_name,
        model=model,
        run_dir_path=run_root,
        data_path=Path(task.path),
        otel_sink_host=otel_sink_host,
        otel_sink_path=otel_path,
    )


def _prepare_run_config(run_config: RunConfig) -> None:
    run_config.run_dir_path.mkdir(parents=True, exist_ok=True)
    run_config.otel_sink_path.parent.mkdir(parents=True, exist_ok=True)
    run_config.save_run_metadata()


@contextmanager
def temporary_mamba_environment(env_file: Path, env_name: str) -> Iterator[str]:
    executable = "mamba"
    create_cmd = [
        executable,
        "env",
        "create",
        "--yes",
        "--name",
        env_name,
        "--file",
        str(env_file),
    ]
    logging.info("Provisioning evaluation environment %s", env_name)
    subprocess.run(
        create_cmd,
        check=True,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        yield env_name
    finally:
        remove_cmd = [executable, "env", "remove", "--yes", "--name", env_name]
        logging.info("Removing evaluation environment %s", env_name)
        subprocess.run(
            remove_cmd,
            check=False,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


def _run_agent_subprocess(env_name: str, config_path: Path) -> None:
    executable = "mamba"
    cmd = [
        executable,
        "run",
        "--name",
        env_name,
        "python",
        "-m",
        "src.agent",
        "--config",
        str(config_path),
    ]
    env = os.environ.copy()
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)


def _run_agent_subprocess_with_timeout(
    env_name: str, config_path: Path, timeout_seconds: float | None
) -> None:
    """Run the agent subprocess with an optional timeout.

    Args:
        env_name (str): Mamba environment name to run under.
        config_path (Path): Path to the serialized RunConfig JSON file.
        timeout_seconds (float | None): Timeout in seconds. When None, no timeout is enforced.

    Returns:
        None: Runs the subprocess as a side effect.

    Raises:
        subprocess.TimeoutExpired: If the subprocess exceeds the configured timeout.
        subprocess.CalledProcessError: If the subprocess exits non-zero.
    """

    executable = "mamba"
    cmd = [
        executable,
        "run",
        "--name",
        env_name,
        "python",
        "-m",
        "src.agent",
        "--config",
        str(config_path),
    ]

    env = os.environ.copy()
    if timeout_seconds is None:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)
        return

    # Use a process group so we can terminate the whole tree (mamba -> python -> child tools).
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        start_new_session=True,
    )
    try:
        remaining = max(0.0, timeout_seconds - (time.time() - start))
        proc.wait(timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_seconds) from exc

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _run_single_task(
    env_name: str, run_config: RunConfig, timeout_seconds: float | None
) -> None:
    logging.info(
        "Running task '%s' (run %s) in environment '%s'.",
        run_config.task_id,
        run_config.run_hash,
        env_name,
    )
    _run_agent_subprocess_with_timeout(
        env_name=env_name,
        config_path=run_config.metadata_path,
        timeout_seconds=timeout_seconds,
    )
    logging.info("Completed task '%s'.", run_config.task_id)


def _execute_tasks_in_env(
    tasks: Sequence[DataSet],
    env_file: Path,
    build_run_config: Callable[[DataSet], RunConfig],
    timeout_seconds: float | None,
) -> int:
    """Run tasks sequentially in isolated mamba environments.

    Args:
        tasks (Sequence[DataSet]): Tasks to execute.
        env_file (Path): Conda/mamba environment YAML file.
        build_run_config (Callable[[DataSet], RunConfig]): Factory that creates a RunConfig per task.
        timeout_seconds (float | None): Per-task timeout in seconds. When None, no timeout is enforced.

    Returns:
        int: Number of tasks completed successfully.
    """

    completed = 0
    for task in list(tasks):
        run_config = build_run_config(task)
        _prepare_run_config(run_config)
        env_alias = f"{task.task_id}-{run_config.run_hash[:8]}"
        with temporary_mamba_environment(env_file=env_file, env_name=env_alias) as env_name:
            try:
                _run_single_task(env_name, run_config, timeout_seconds=timeout_seconds)
            except subprocess.TimeoutExpired:
                run_config.error_type = "timeout"
                hours = None if timeout_seconds is None else timeout_seconds / 3600
                run_config.error_message = (
                    "Run exceeded timeout."
                    if hours is None
                    else f"Run exceeded timeout ({hours:.2f} hours)."
                )
                run_config.save_run_metadata()
                logging.exception(
                    "Task '%s' (run %s) exceeded timeout; continuing to next task.",
                    run_config.task_id,
                    run_config.run_hash,
                )
                continue
            except subprocess.CalledProcessError as exc:
                run_config.error_type = "subprocess_failed"
                run_config.error_message = (
                    f"Agent subprocess failed with return code {exc.returncode}."
                )
                run_config.save_run_metadata()
                logging.exception(
                    "Task '%s' (run %s) failed (return code %s); continuing to next task.",
                    run_config.task_id,
                    run_config.run_hash,
                    exc.returncode,
                )
                continue
        completed += 1
    return completed


@contextmanager
def run_otel_module(
    host: str, ndjson_path: str, mode: str = "single"
) -> Iterator[None]:
    """
    Launch the OTEL sink as a separate Python module process:
      python -m otel --host ... --path ...
    Ensures clean shutdown on exit.
    """
    args = [
        sys.executable,
        "-m",
        "otel",
        "--host",
        str(host),
        "--path",
        str(ndjson_path),
        "--mode",
        mode,
    ]
    os.makedirs(os.path.dirname(ndjson_path) or ".", exist_ok=True)

    proc = subprocess.Popen(args, cwd=PROJECT_ROOT)  # inherit stdout/stderr
    try:
        yield
    finally:
        # Graceful stop; send SIGINT then wait, fall back to kill
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def run_environment(
    suite: str,
    model_name: str,
    use_reference_data: bool = False,
    task_ids: Sequence[str] | None = None,
    timeout_hours: float = 4.0,
) -> None:
    """
    Execute evaluation suite sequentially in isolated mamba environments.

    Args:
        suite (str): One of "open", "minimal", "expanded-10", "expanded-20", "expanded-30".
        model_name (str): Model identifier/profile.
        use_reference_data (bool): Whether to include reference data.
        task_ids (Sequence[str] | None): Optional list of task_ids to run. When None, runs all tasks in the suite.
        timeout_hours (float): Per-task timeout in hours.
    """

    datasets = DataSet.load_all(metadata_path=str(METADATA_PATH), data_root=str(DATA_ROOT))
    suite = suite.lower()

    if suite == "open":
        experiment_name = (
            "open-environment-with-reference-data"
            if use_reference_data
            else "open-environment-no-reference-data"
        )
        env_file = Path("envs/open-environment.yml")
        relevant_tasks = datasets
        system_prompt_name = "v1"

        def _tool_names(_: DataSet) -> Sequence[str]:
            return ()

    elif suite == "minimal":
        experiment_name = "minimal-tool-environment"
        env_file = Path("envs/tools-environment.yml")
        relevant_tasks = [task for task in datasets if task.task_id in tools_mapping_dict]
        system_prompt_name = "v2"

        def _tool_names(task: DataSet) -> Sequence[str]:
            return tuple(tools_mapping_dict[task.task_id])

    elif suite.startswith("expanded-"):
        try:
            extra_count = int(suite.split("-", 1)[1])
        except Exception as exc:
            raise click.ClickException(f"Invalid expanded suite '{suite}'.") from exc

        experiment_name = f"expanded-{extra_count}-tool-environment"
        env_file = Path("envs/tools-environment.yml")
        relevant_tasks = [task for task in datasets if task.task_id in tools_mapping_dict]
        system_prompt_name = "v2"

        def _tool_names(task: DataSet) -> Sequence[str]:
            base_tools = tools_mapping_dict[task.task_id]
            other_tools = [
                tool
                for key, tools in tools_mapping_dict.items()
                if key != task.task_id
                for tool in tools
                if tool not in base_tools
            ]
            if not other_tools:
                return tuple(base_tools)
            sample_count = min(extra_count, len(other_tools))
            extra_tools = random.sample(other_tools, sample_count)
            return tuple(base_tools + extra_tools)

    else:
        raise click.ClickException(f"Unknown suite '{suite}'")

    if RUN_LOGS is None:
        raise click.ClickException("RUN_LOGS is not configured. Did you run via main()?")

    if task_ids:
        wanted = set(task_ids)
        relevant_tasks = [task for task in relevant_tasks if task.task_id in wanted]
        missing = sorted(wanted - {task.task_id for task in relevant_tasks})
        if missing:
            raise click.ClickException(
                "Unknown/unavailable task_ids for this suite: " + ", ".join(missing)
            )

    otel_root = RUN_LOGS / "otel"
    otel_root.mkdir(parents=True, exist_ok=True)
    total_tasks = len(relevant_tasks)

    def _build(task: DataSet) -> RunConfig:
        return _build_run_config(
            task=task,
            system_prompt_name=system_prompt_name,
            use_reference_data=use_reference_data,
            run_logs=RUN_LOGS,
            experiment_name=experiment_name,
            model=model_name,
            tool_names=_tool_names(task),
            otel_sink_host=OTEL_SINK_HOST,
        )

    logging.info("Using external OTEL sink on %s.", OTEL_SINK_HOST)
    timeout_seconds = None if timeout_hours <= 0 else timeout_hours * 3600
    completed = _execute_tasks_in_env(
        tasks=relevant_tasks,
        env_file=env_file,
        build_run_config=_build,
        timeout_seconds=timeout_seconds,
    )

    click.echo(f"Completed {completed}/{total_tasks} tasks for suite '{suite}' with model '{model_name}'.")

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--suite",
    type=click.Choice(["open", "minimal", "expanded-10", "expanded-20", "expanded-30"], case_sensitive=False),
    default="open",
    show_default=True,
    help="Which evaluation configuration to run.",
)
@click.option(
    "--reference-mode",
    type=click.Choice(["with", "without"], case_sensitive=False),
    default="without",
    show_default=True,
)
@click.option(
    "--model",
    "models",
    multiple=True,
    help="Repeat to run only the provided model names (defaults to all).",
)
@click.option(
    "--task-id",
    "task_ids",
    multiple=True,
    help="Repeat to run only the provided task ids (defaults to all tasks in the suite).",
)
@click.option(
    "--timeout-hours",
    type=float,
    default=4.0,
    show_default=True,
    help="Per-task timeout in hours. Use 0 or a negative value to disable the timeout.",
)
def main(
    suite: str,
    reference_mode: str,
    models: tuple[str, ...],
    task_ids: tuple[str, ...],
    timeout_hours: float,
) -> None:
    configure_logging()
    _ensure_required_env_vars()

    suite = suite.lower()
    reference_mode = reference_mode.lower()

    global RUN_LOGS
    RUN_LOGS = Path(os.getenv("RUN_LOGS"))

    selected_models = list(models) if models else MODELS
    use_reference = reference_mode == "with"

    for model in selected_models:
        run_environment(
            suite=suite,
            model_name=model,
            use_reference_data=use_reference,
            task_ids=task_ids or None,
            timeout_hours=timeout_hours,
        )


if __name__ == "__main__":
    main()
