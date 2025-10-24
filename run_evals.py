"""Orchestrate evaluation loops with per-run mamba environments."""

from __future__ import annotations

import os
import subprocess
import uuid
from contextlib import contextmanager
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

from src.dataset import DataSet
from src.logs import RunConfig, configure_logging
from src.tools import REGISTRY
from src.system_prompts import prompts

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_LOGS = Path(os.getenv("RUN_LOGS"))
METADATA_PATH = Path("/home/dionizije/bioagent-bench/src/task_metadata.json")
DATA_ROOT = Path("/home/dionizije/bioagent-data")


def _build_run_config(
    task: DataSet,
    system_prompt_name: str,
    run_logs: Path,
    max_steps: int,
    planning_interval: int,
    experiment_name: str,
    model: str,
    tool_names: Sequence[str],
) -> RunConfig:
    """Construct the ``RunConfig`` for ``src.eval`` using dataset metadata.

    Args:
        task (DataSet): Dataset metadata describing the task to evaluate.
        system_prompt_name (str): Key into ``system_prompts.prompts``.
        run_logs (Path): Base directory for runs and logs.
        max_steps (int): Maximum agent steps.
        planning_interval (int): Planning interval for the agent.
        experiment_name (str): Name of the experiment.
        model (str): Model identifier string (e.g., ``"azure"``).
        tool_names (Sequence[str]): Tool identifiers to hand to the agent.

    Returns:
        RunConfig: Configuration describing the run metadata.
    """
    timestamp = datetime.now()
    system_prompt = prompts.get(system_prompt_name)

    run_hash = str(uuid.uuid4())
    run_root = run_logs / experiment_name / task.task_id / run_hash
    metadata_path = run_logs / f"{run_hash}.json"

    run_config = RunConfig(
        metadata_path=metadata_path,
        run_hash=run_hash,
        timestamp=timestamp,
        task_id=task.task_id,
        task_prompt=task.task_prompt,
        max_steps=max_steps,
        planning_interval=planning_interval,
        num_tools=len(tool_names),
        tool_names=list(tool_names),
        system_prompt=system_prompt,
        system_prompt_name=system_prompt_name,
        experiment_name=experiment_name,
        model=model,
        run_dir_path=run_root,
        data_path=Path(task.path) if task.path else None,
    )

    return run_config


@contextmanager
def temporary_mamba_environment(env_file: Path) -> Iterator[str]:
    """Provision a throwaway mamba environment and clean it up afterwards.
    Args:
        env_file (Path): Path to the environment file to use.

    Yields:
        Iterator[str]: The name of the newly created environment.
    """

    # I wrapped this in a subprocess to avoid propaganting logs to my console
    # There is a lot of spam that made babysitting the agents difficult
    executable = 'mamba'
    env_name = f"bioagent-eval-{uuid.uuid4().hex}"

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
        remove_cmd = [
            executable,
            "env",
            "remove",
            "--yes",
            "--name",
            env_name,
        ]
        logging.info("Removing evaluation environment %s", env_name)
        result = subprocess.run(
            remove_cmd,
            check=False,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

def _run_eval_subprocess(env_name: str, config_path: Path) -> None:
    """Run ``src.eval`` in a subprocess inside the provided environment.

    Args:
        env_name (str): Name of the mamba environment to use.
        config_path (Path): Path to the serialised run configuration.
    """

    executable = "mamba"
    cmd = [
        executable,
        "run",
        "--name",
        env_name,
        "python",
        "-m",
        "src.eval",
        "--config",
        str(config_path),
    ]

    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def open_environment() -> None:
    """Models have no tools"""

    MAX_STEPS = 20
    PLANNING_INTERVAL = 1
    EXPERIMENT_NAME = "open-environment"
    MODEL_NAME = "azure"

    configure_logging()
    datasets = DataSet.load_all(
        metadata_path=METADATA_PATH,
        data_root=DATA_ROOT,
    )

    for task in datasets:
        run_config = _build_run_config(
            task=task,
            system_prompt_name='v1',
            run_logs=RUN_LOGS,
            max_steps=MAX_STEPS,
            planning_interval=PLANNING_INTERVAL,
            experiment_name=EXPERIMENT_NAME,
            model=MODEL_NAME,
            tool_names=("run_terminal_command",),
        )

        run_config.run_dir_path.mkdir(parents=True, exist_ok=True)
        run_config.save_run_metadata()

        with temporary_mamba_environment(env_file=Path("envs/open-environment.yml")) as env_name:
            print(f"Running task '{task.task_id}' in environment '{env_name}'.")
            try:
                _run_eval_subprocess(env_name=env_name, config_path=run_config.metadata_path)
                print(f"Completed task '{task.task_id}'.")
            except subprocess.CalledProcessError as e:
                print(e)


def minimal_tool_environmet() -> None:
    """Models have minimal tools"""

    MAX_STEPS = 20
    PLANNING_INTERVAL = 1
    EXPERIMENT_NAME = "minimal-tool-environment"
    MODEL_NAME = "azure"


    configure_logging()
    datasets = DataSet.load_all(
        metadata_path=METADATA_PATH,
        data_root=DATA_ROOT,
    )
    for task in datasets:
        if task.task_id != "transcript-quant":
            continue

        tool_names = REGISTRY.tool_names_for_task(task.task_id)
        run_config = _build_run_config(
            task=task,
            system_prompt_name='v2',
            run_logs=RUN_LOGS,
            max_steps=MAX_STEPS,
            planning_interval=PLANNING_INTERVAL,
            experiment_name=EXPERIMENT_NAME,
            model=MODEL_NAME,
            tool_names=tool_names,
        )

        run_config.run_dir_path.mkdir(parents=True, exist_ok=True)
        run_config.save_run_metadata()

        with temporary_mamba_environment(env_file=Path("envs/tools-environment.yml")) as env_name:
            print(f"Running task '{task.task_id}' in environment '{env_name}'.")
            try:
                _run_eval_subprocess(env_name=env_name, config_path=run_config.metadata_path)
                print(f"Completed task '{task.task_id}'.")
            except subprocess.CalledProcessError as e:
                print(e)


def expanded_tool_environmet() -> None:
    """Models are expanded with random unnecessary tools"""

    MAX_STEPS = 20
    PLANNING_INTERVAL = 1
    EXPERIMENT_NAME = "expanded-tool-environment"
    MODEL_NAME = "azure"

    configure_logging()
    datasets = DataSet.load_all(
        metadata_path=METADATA_PATH,
        data_root=DATA_ROOT,
    )
    for task in datasets:
        if task.task_id != "transcript-quant":
            continue

        base_tool_names = REGISTRY.tool_names_for_task(task.task_id)
        random_tool_names = REGISTRY.sample_additional_tool_names(
            exclude=base_tool_names,
            sample_size=10,
        )
        tool_names = REGISTRY.tool_names_for_task(
            task.task_id,
            extra_tool_names=random_tool_names,
        )
        run_config = _build_run_config(
            task=task,
            system_prompt_name='v2',
            run_logs=RUN_LOGS,
            max_steps=MAX_STEPS,
            planning_interval=PLANNING_INTERVAL,
            experiment_name=EXPERIMENT_NAME,
            model=MODEL_NAME,
            tool_names=tool_names,
        )

        run_config.run_dir_path.mkdir(parents=True, exist_ok=True)
        run_config.save_run_metadata()

        with temporary_mamba_environment(env_file=Path("envs/tools-environment.yml")) as env_name:
            print(f"Running task '{task.task_id}' in environment '{env_name}'.")
            try:
                _run_eval_subprocess(env_name=env_name, config_path=run_config.metadata_path)
                print(f"Completed task '{task.task_id}'.")
            except subprocess.CalledProcessError as e:
                print(e)


def all_tools_environment() -> None:
    """Models have full tools"""

    MAX_STEPS = 20
    PLANNING_INTERVAL = 1
    EXPERIMENT_NAME = "all-tool-environment"
    MODEL_NAME = "azure"

    configure_logging()
    datasets = DataSet.load_all(
        metadata_path=METADATA_PATH,
        data_root=DATA_ROOT,
    )
    for task in datasets:
        if task.task_id != "transcript-quant":
            continue

        tool_names = REGISTRY.all_tool_names()
        print('All tools environment')
        print(tool_names)
        print('-'*100)
        run_config = _build_run_config(
            task=task,
            system_prompt_name='v2',
            run_logs=RUN_LOGS,
            max_steps=MAX_STEPS,
            planning_interval=PLANNING_INTERVAL,
            experiment_name=EXPERIMENT_NAME,
            model=MODEL_NAME,
            tool_names=tool_names,
        )

        run_config.run_dir_path.mkdir(parents=True, exist_ok=True)
        run_config.save_run_metadata()

        with temporary_mamba_environment(env_file=Path("envs/tools-environment.yml")) as env_name:
            print(f"Running task '{task.task_id}' in environment '{env_name}'.")
            try:
                _run_eval_subprocess(env_name=env_name, config_path=run_config.metadata_path)
                print(f"Completed task '{task.task_id}'.")
            except subprocess.CalledProcessError as e:
                print(e)


if __name__ == "__main__":
    # for i in range(3):
    #     open_environment()
    #     minimal_tool_environmet()
    #     expanded_tool_environmet()
    #     all_tools_environment()
    minimal_tool_environmet()
