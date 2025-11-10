#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
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
    timestamp = datetime.now()
    system_prompt = prompts.get(system_prompt_name)

    run_hash = str(uuid.uuid4())
    run_root = run_logs / experiment_name / task.task_id / run_hash
    metadata_path = run_logs / "runs" / f"{run_hash}.json"

    otel_path = run_logs / "otel" / f"otlp-{run_hash}.ndjson"

    return RunConfig(
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
        otel_sink_host="127.0.0.1:4317",
        otel_sink_path=otel_path,
    )


@contextmanager
def temporary_mamba_environment(env_file: Path) -> Iterator[str]:
    executable = "mamba"
    create_cmd = [executable, "env", "create", "--yes", "--name", "bioinformatics", "--file", str(env_file)]
    logging.info("Provisioning evaluation environment %s", "bioinformatics")
    subprocess.run(create_cmd, check=True, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        yield "bioinformatics"
    finally:
        remove_cmd = [executable, "env", "remove", "--yes", "--name", "bioinformatics"]
        logging.info("Removing evaluation environment %s", "bioinformatics")
        subprocess.run(remove_cmd, check=False, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _run_eval_subprocess(env_name: str, config_path: Path) -> None:
    executable = "mamba"
    cmd = [executable, "run", "--name", env_name, "python", "-m", "src.eval", "--config", str(config_path)]
    env = os.environ.copy()
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)


@contextmanager
def run_otel_module(host: str, ndjson_path: str) -> Iterator[None]:
    """
    Launch the OTEL sink as a separate Python module process:
      python -m otel --host ... --port ... --path ...
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


def open_environment() -> None:
    MAX_STEPS = 20
    PLANNING_INTERVAL = 1
    EXPERIMENT_NAME = "open-environment"
    MODEL_NAME = "azure"

    configure_logging()
    datasets = DataSet.load_all(metadata_path=METADATA_PATH, data_root=DATA_ROOT)

    for task in datasets:
        if task.task_id != "transcript-quant":
            continue

        run_config = _build_run_config(
            task=task,
            system_prompt_name="v1",
            run_logs=RUN_LOGS,
            max_steps=MAX_STEPS,
            planning_interval=PLANNING_INTERVAL,
            experiment_name=EXPERIMENT_NAME,
            model=MODEL_NAME,
            tool_names=(),
        )

        # Ensure directories exist & save metadata early
        run_config.run_dir_path.mkdir(parents=True, exist_ok=True)
        run_config.otel_sink_path.parent.mkdir(parents=True, exist_ok=True)
        run_config.save_run_metadata()

        logging.info(
            "Starting per-run OTEL sink (module) port=%s -> %s",
            run_config.otel_sink_port,
            str(run_config.otel_sink_path.resolve()),
        )

        with run_otel_module(
            host=run_config.otel_sink_host,
            ndjson_path=str(run_config.otel_sink_path.resolve()),
        ):
            with temporary_mamba_environment(env_file=Path("envs/open-environment.yml")) as env_name:
                print(f"Running task '{task.task_id}' in environment '{env_name}'.")
                otlp_grpc_target = f"{run_config.otel_sink_host}"
                _run_eval_subprocess(env_name=env_name, config_path=run_config.metadata_path)
                print(f"Completed task '{task.task_id}'.")
                

if __name__ == "__main__":
    open_environment()
