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

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_LOGS = Path(os.getenv("RUN_LOGS"))
METADATA_PATH = Path("/home/dionizije/bioagent-bench/src/task_metadata.json")
DATA_ROOT = Path("/home/dionizije/bioagent-data")
MAX_WORKERS = 5


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
    system_prompt = prompts.get(system_prompt_name)

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
        system_prompt=system_prompt,
        system_prompt_name=system_prompt_name,
        experiment_name=experiment_name,
        model=model,
        run_dir_path=run_root,
        data_path=Path(task.path) if task.path else None,
        otel_sink_host=otel_sink_host,
        otel_sink_path=otel_path,
    )


def _allocate_otel_endpoint(host: str = "127.0.0.1") -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        _, port = sock.getsockname()
    return f"{host}:{port}"


def _prepare_run_config(run_config: RunConfig) -> None:
    run_config.run_dir_path.mkdir(parents=True, exist_ok=True)
    run_config.otel_sink_path.parent.mkdir(parents=True, exist_ok=True)
    run_config.save_run_metadata()


@contextmanager
def temporary_mamba_environment(env_file: Path) -> Iterator[str]:
    executable = "mamba"
    create_cmd = [
        executable,
        "env",
        "create",
        "--yes",
        "--name",
        "bioinformatics",
        "--file",
        str(env_file),
    ]
    logging.info("Provisioning evaluation environment %s", "bioinformatics")
    subprocess.run(
        create_cmd,
        check=True,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        yield "bioinformatics"
    finally:
        remove_cmd = [executable, "env", "remove", "--yes", "--name", "bioinformatics"]
        logging.info("Removing evaluation environment %s", "bioinformatics")
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
) -> None:
    task_list = list(tasks)
    if not task_list:
        logging.info("No tasks to run for environment file '%s'.", env_file)
        return

    worker_count = max(1, min(max_workers, len(task_list)))
    with temporary_mamba_environment(env_file=env_file) as env_name:
        logging.info(
            "Running %d tasks with up to %d concurrent workers in environment '%s'.",
            len(task_list),
            worker_count,
            env_name,
        )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            active: dict = {}
            tasks_iter = iter(task_list)

            def _submit_next_task() -> bool:
                try:
                    task = next(tasks_iter)
                except StopIteration:
                    return False
                run_config = build_run_config(task)
                _prepare_run_config(run_config)
                otel_ctx = run_otel_module(
                    host=run_config.otel_sink_host,
                    ndjson_path=str(run_config.otel_sink_path.resolve()),
                )
                logging.info(
                    "Starting OTEL sink for task '%s' (run %s) on %s",
                    task.task_id,
                    run_config.run_hash,
                    run_config.otel_sink_host,
                )
                otel_ctx.__enter__()
                future = executor.submit(_run_single_task, env_name, run_config)
                active[future] = (run_config, otel_ctx)
                return True

            for _ in range(worker_count):
                if not _submit_next_task():
                    break

            try:
                while active:
                    future = next(as_completed(list(active.keys())))
                    run_config, otel_ctx = active.pop(future)
                    try:
                        future.result()
                    finally:
                        otel_ctx.__exit__(None, None, None)
                    _submit_next_task()
            except Exception:
                for future, (_, otel_ctx) in active.items():
                    future.cancel()
                    try:
                        otel_ctx.__exit__(None, None, None)
                    except Exception:
                        pass
                raise


@contextmanager
def run_otel_module(host: str, ndjson_path: str) -> Iterator[None]:
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


def open_environment(
    model_name, use_reference_data: bool = False, max_workers: int = 1
) -> None:
    experiment_name = (
        "open-environment-with-reference-data"
        if use_reference_data
        else "open-environment-no-reference-data"
    )

    configure_logging()
    datasets = DataSet.load_all(metadata_path=METADATA_PATH, data_root=DATA_ROOT)

    def _build(task: DataSet) -> RunConfig:
        return _build_run_config(
            task=task,
            system_prompt_name="v1",
            use_reference_data=use_reference_data,
            run_logs=RUN_LOGS,
            experiment_name=experiment_name,
            model=model_name,
            tool_names=(),
            otel_sink_host=_allocate_otel_endpoint(),
        )

    _execute_tasks_in_env(
        tasks=datasets,
        env_file=Path("envs/open-environment.yml"),
        max_workers=max_workers,
        build_run_config=_build,
    )


def minimal_tool_environmet(max_workers: int = 1) -> None:
    """Models have minimal tools"""

    EXPERIMENT_NAME = "minimal-tool-environment"
    MODEL_NAME = "gpt-5-codex(high)"

    configure_logging()
    datasets = DataSet.load_all(
        metadata_path=METADATA_PATH,
        data_root=DATA_ROOT,
    )
    relevant_tasks = [task for task in datasets if task.task_id in tools_mapping_dict]

    def _build(task: DataSet) -> RunConfig:
        tools_config = tools_mapping_dict[task.task_id]
        return _build_run_config(
            task=task,
            system_prompt_name="v2",
            run_logs=RUN_LOGS,
            experiment_name=EXPERIMENT_NAME,
            model=MODEL_NAME,
            tool_names=tools_config,
            otel_sink_host=_allocate_otel_endpoint(),
        )

    _execute_tasks_in_env(
        tasks=relevant_tasks,
        env_file=Path("envs/tools-environment.yml"),
        max_workers=max_workers,
        build_run_config=_build,
    )


def expanded_tool_environmet(num_extra_tools: int = 10, max_workers: int = 1) -> None:
    """Models have minimal tools"""

    EXPERIMENT_NAME = f"expanded-{num_extra_tools}-tool-environment"
    MODEL_NAME = "gpt-5-codex(high)"

    configure_logging()
    datasets = DataSet.load_all(
        metadata_path=METADATA_PATH,
        data_root=DATA_ROOT,
    )
    relevant_tasks = [task for task in datasets if task.task_id in tools_mapping_dict]

    def _build(task: DataSet) -> RunConfig:
        base_tools = tools_mapping_dict[task.task_id]
        other_tools = [
            tool
            for key, tools in tools_mapping_dict.items()
            if key != task.task_id
            for tool in tools
            if tool not in base_tools
        ]
        extra_tools = random.sample(other_tools, num_extra_tools)
        tools_config = base_tools + extra_tools
        return _build_run_config(
            task=task,
            system_prompt_name="v2",
            run_logs=RUN_LOGS,
            experiment_name=EXPERIMENT_NAME,
            model=MODEL_NAME,
            tool_names=tools_config,
            otel_sink_host=_allocate_otel_endpoint(),
        )

    _execute_tasks_in_env(
        tasks=relevant_tasks,
        env_file=Path("envs/tools-environment.yml"),
        max_workers=max_workers,
        build_run_config=_build,
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--suite",
    type=click.Choice(["open", "minimal", "expanded"], case_sensitive=False),
    default="open",
    show_default=True,
    help="Which evaluation configuration to run.",
)
@click.option(
    "--reference-mode",
    type=click.Choice(["with", "without", "both"], case_sensitive=False),
    default="with",
    show_default=True,
    help="Open suite only: whether to use reference data.",
)
@click.option(
    "--max-workers",
    type=click.IntRange(1, None),
    default=5,
    show_default=True,
    help="Maximum number of tasks to run concurrently.",
)
@click.option(
    "--num-extra-tools",
    type=click.IntRange(0, None),
    default=10,
    show_default=True,
    help="Expanded suite only: number of random tools to add.",
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
    num_extra_tools: int,
    models: tuple[str, ...],
) -> None:
    suite = suite.lower()
    reference_mode = reference_mode.lower()
    selected_models = list(models) if models else MODELS

    if suite == "open":
        reference_modes: list[bool] = []
        if reference_mode in ("with", "both"):
            reference_modes.append(True)
        if reference_mode in ("without", "both"):
            reference_modes.append(False)
        for model in selected_models:
            for use_reference in reference_modes or [False]:
                open_environment(
                    model_name=model,
                    use_reference_data=use_reference,
                    max_workers=max_workers,
                )
    elif suite == "minimal":
        minimal_tool_environmet(max_workers=max_workers)
    elif suite == "expanded":
        expanded_tool_environmet(
            num_extra_tools=num_extra_tools,
            max_workers=max_workers,
        )


if __name__ == "__main__":
    main()
