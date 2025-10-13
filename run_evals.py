"""Orchestrate evaluation loops with per-run mamba environments."""

from __future__ import annotations

import os
import json
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

from src.dataset import DataSet
from src.logs import RunConfig
from src.models import model_loader_mapping
from src.system_prompts import prompts


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_LOGS = Path(os.getenv("RUN_LOGS"))


METADATA_PATH = Path("/home/dionizije/bioagent-bench/src/task_metadata.json")
DATA_ROOT = Path("/home/dionizije/bioagent-data")
SYSTEM_PROMPT = "v1"
MAX_STEPS = 2
PLANNING_INTERVAL = 1
EXPERIMENT_NAME = "open-environment"
MODEL_NAME = "azure"
TOOL_NAMES: Sequence[str] = (
    "run_terminal_command",
)
BASE_ENV = "base"


def _generate_run_hash(_: str, length: int = 7) -> str:
    """Generate a short, random identifier for a run configuration.

    Args:
        _ (str): Unused input retained for compatibility.
        length (int): Desired number of characters in the identifier.

    Returns:
        str: Numeric run hash of the requested length.
    """

    return str(uuid.uuid4().int % (10**length)).zfill(length)


def _resolve_mamba_executable() -> str:
    """Resolve the executable used to invoke mamba.

    Returns:
        str: Executable name or absolute path used for ``mamba``.
    """

    # TODO: Update this if ``mamba`` is not on PATH or a different executable should be used.
    return "mamba"


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

    run_hash =  str(uuid.uuid4())
    run_root = run_logs / experiment_name / run_hash
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
def temporary_mamba_environment() -> Iterator[str]:
    """Provision a throwaway mamba environment and clean it up afterwards.

    Args:
        base_env (str): Existing environment to clone. Defaults to ``"base"``.

    Yields:
        Iterator[str]: The name of the newly created environment.
    """

    executable = "mamba"
    env_name = f"bioagent-eval-{uuid.uuid4().hex}"

    create_cmd = [
        executable,
        "create",
        "--yes",
        "--name",
        env_name,
        "r-base",
        "r-essentials",
        "-c", "conda-forge",
    ]
    subprocess.run(create_cmd, check=True)

    try:
        yield env_name
    finally:
        remove_cmd = [
            executable,
            "remove",
            "--yes",
            "--name",
            env_name,
            "--all",
        ]
        subprocess.run(remove_cmd, check=False)


def _run_eval_subprocess(env_name: str, config_path: Path) -> None:
    """Run ``src.eval`` in a subprocess inside the provided environment.

    Args:
        env_name (str): Name of the mamba environment to use.
        config_path (Path): Path to the serialised run configuration.
    """

    executable = _resolve_mamba_executable()
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
    datasets = DataSet.load_all(
        metadata_path=METADATA_PATH,
        data_root=DATA_ROOT,
    )

    for task in datasets:
        run_config = _build_run_config(
            task=task,
            system_prompt_name=SYSTEM_PROMPT,
            run_logs=RUN_LOGS,
            max_steps=MAX_STEPS,
            planning_interval=PLANNING_INTERVAL,
            experiment_name=EXPERIMENT_NAME,
            model=MODEL_NAME,
            tool_names=TOOL_NAMES,
        )

        run_config.run_dir_path.mkdir(parents=True, exist_ok=True)
        run_config.save_run_metadata()

        with temporary_mamba_environment() as env_name:
            print(f"Running task '{task.task_id}' in environment '{env_name}'.")
            _run_eval_subprocess(env_name=env_name, config_path=run_config.metadata_path)
            print(f"Completed task '{task.task_id}'.")


if __name__ == "__main__":
    open_environment()
