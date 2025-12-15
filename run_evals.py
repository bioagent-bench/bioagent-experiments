#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
import random
import socket
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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
RUN_LOGS = Path(os.getenv("RUN_LOGS"))
METADATA_PATH = Path("/home/dionizije/bioagent-bench/src/task_metadata.json")
DATA_ROOT = Path("/home/dionizije/bioagent-data")
REQUIRED_ENV_VARS = ("AZURE_OPENAI_API_KEY", "ANTHROPIC_FOUNDRY_API_KEY", "RUN_LOGS")


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

    otel_sink_host = otel_sink_host or "127.0.0.1:4317"
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
        data_path=Path(task.path) if task.path else None,
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


def _run_single_task(env_name: str, run_config: RunConfig) -> None:
    logging.info(
        "Running task '%s' (run %s) in environment '%s'.",
        run_config.task_id,
        run_config.run_hash,
        env_name,
    )
    _run_agent_subprocess(env_name=env_name, config_path=run_config.metadata_path)
    logging.info("Completed task '%s'.", run_config.task_id)


def _execute_tasks_in_env(
    tasks: Sequence[DataSet],
    env_file: Path,
    max_workers: int,
    build_run_config: Callable[[DataSet], RunConfig],
) -> int:
    task_list = list(tasks)

    worker_count = max(1, min(max_workers, len(task_list)))

    def _run_task(task: DataSet) -> None:
        run_config = build_run_config(task)
        _prepare_run_config(run_config)
        env_alias = f"{task.task_id}-{run_config.run_hash[:8]}"
        with temporary_mamba_environment(
            env_file=env_file, env_name=env_alias
        ) as env_name:
            _run_single_task(env_name, run_config)

    completed = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_run_task, task) for task in task_list]
        try:
            for future in as_completed(futures):
                future.result()
                completed += 1
        except Exception:
            for future in futures:
                future.cancel()
            raise
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
    max_workers: int = 1,
) -> None:
    """
    Execute evaluation suite in a mamba environment. `suite` accepts
    "open", "minimal", or "expanded".
    """

    datasets = DataSet.load_all(metadata_path=METADATA_PATH, data_root=DATA_ROOT)
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

        suite_use_reference = use_reference_data

    elif suite == "minimal":
        experiment_name = "minimal-tool-environment"
        env_file = Path("envs/tools-environment.yml")
        relevant_tasks = [
            task for task in datasets if task.task_id in tools_mapping_dict
        ]
        system_prompt_name = "v2"

        def _tool_names(task: DataSet) -> Sequence[str]:
            return tuple(tools_mapping_dict[task.task_id])

        suite_use_reference = use_reference_data

    elif suite == "expanded":
        experiment_name = f"expanded-{10}-tool-environment"
        env_file = Path("envs/tools-environment.yml")
        relevant_tasks = [
            task for task in datasets if task.task_id in tools_mapping_dict
        ]
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
            extra_tools = random.sample(other_tools, 10)
            return tuple(base_tools + extra_tools)

        suite_use_reference = use_reference_data

    elif suite == "expanded":
        experiment_name = f"expanded-{20}-tool-environment"
        env_file = Path("envs/tools-environment.yml")
        relevant_tasks = [
            task for task in datasets if task.task_id in tools_mapping_dict
        ]
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
            extra_tools = random.sample(other_tools, 20)
            return tuple(base_tools + extra_tools)

        suite_use_reference = use_reference_data

    elif suite == "expanded":
        experiment_name = f"expanded-{30}-tool-environment"
        env_file = Path("envs/tools-environment.yml")
        relevant_tasks = [
            task for task in datasets if task.task_id in tools_mapping_dict
        ]
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
            extra_tools = random.sample(other_tools, 30)
            return tuple(base_tools + extra_tools)

        suite_use_reference = use_reference_data
    else:
        raise ValueError(f"Unknown suite '{suite}'")

    otel_root = RUN_LOGS / "otel"
    otel_root.mkdir(parents=True, exist_ok=True)
    total_tasks = len(relevant_tasks)

    def _build(task: DataSet) -> RunConfig:
        return _build_run_config(
            task=task,
            system_prompt_name=system_prompt_name,
            use_reference_data=suite_use_reference,
            run_logs=RUN_LOGS,
            experiment_name=experiment_name,
            model=model_name,
            tool_names=_tool_names(task),
        )

    logging.info("Starting shared OTEL sink on %s.", OTEL_SINK_HOST)
    with run_otel_module(
        host=OTEL_SINK_HOST,
        ndjson_path=str(otel_root.resolve()),
        mode="multi",
    ):
        completed = _execute_tasks_in_env(
            tasks=relevant_tasks,
            env_file=env_file,
            max_workers=max_workers,
            build_run_config=_build,
        )
    click.echo(
        f"Completed {completed}/{total_tasks} tasks for suite '{suite}' with model '{model_name}'."
    )


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
)
@click.option(
    "--max-workers",
    type=click.IntRange(1, None),
    default=4,
    show_default=True,
    help="Maximum number of tasks to run concurrently.",
)
@click.option(
    "--model",
    "models",
    multiple=True,
    help="Repeat to run only the provided model names (defaults to all).",
)
def main(
    suite: str,
    reference_mode: str,
    max_workers: int,
    models: tuple[str, ...],
) -> None:
    _ensure_required_env_vars()

    suite = suite.lower()
    reference_mode = reference_mode.lower()

    selected_models = list(models) if models else MODELS

    use_reference = False
    if reference_mode == "with":
        use_reference = True

    for model in selected_models:
        run_environment(
            suite=suite,
            model_name=model,
            use_reference_data=use_reference,
            max_workers=max_workers,
        )

if __name__ == "__main__":
    main()
