#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Sequence

import click

from src.dataset import DataSet
from src.logs import RunConfig, configure_logging


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("~/bioagent-data").expanduser()
METADATA_PATH = Path("~/bioagent-bench/src/task_metadata.json").expanduser()

OTEL_SINK_HOST = "127.0.0.1:4317"
REQUIRED_ENV_VARS = ("RUN_LOGS",)

ABLATION_ROOT = PROJECT_ROOT / "ablation"
PROMPT_BLOAT_PATH = ABLATION_ROOT / "prompt_bloat.py"

MODEL_NAME = "gpt-5-2"
SYSTEM_PROMPT_NAME = "v3"
USE_REFERENCE_DATA = True
ENV_FILE = Path("envs/open-environment.yml")
TOOL_NAMES: tuple[str, ...] = ()

# Populated in main() after validating environment variables.
RUN_LOGS: Path | None = None

SETTING_CHOICES = ("corrupt", "decoy", "prompt-bloat")


def _ensure_required_env_vars() -> None:
    missing_env = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing_env:
        missing = ", ".join(missing_env)
        raise click.ClickException(
            f"Cannot run ablations because the following environment variables are unset: {missing}"
        )


def _build_run_config(
    task: DataSet,
    task_prompt: str,
    run_logs: Path,
    experiment_name: str,
    model: str,
    system_prompt_name: str,
    use_reference_data: bool,
    tool_names: Sequence[str],
    otel_sink_host: str | None = None,
) -> RunConfig:
    if not task.path:
        raise click.ClickException(f"Task '{task.task_id}' has no local data path.")

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
        task_prompt=task_prompt,
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


def _copy_inputs_to_directory(
    source_root: Path,
    destination_root: Path,
    use_reference_data: bool,
) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    source = Path(source_root)
    if not source.exists():
        return

    excluded_names: set[str] = {"results"}
    if not use_reference_data:
        excluded_names.add("reference")

    for item in source.iterdir():
        if item.name in excluded_names:
            continue
        target = destination_root / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _overlay_directory(source_root: Path, destination_root: Path) -> None:
    source = Path(source_root)
    if not source.exists():
        return
    for item in source.rglob("*"):
        if item.is_dir():
            continue
        rel_path = item.relative_to(source)
        target = destination_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def _prepare_ablation_inputs(
    run_config: RunConfig,
    ablation_mode: str,
    use_reference_data: bool,
    ablation_root: Path,
) -> Path:
    source_root = Path(run_config.data_path)
    if not source_root.exists():
        raise click.ClickException(
            f"Input data directory does not exist: {source_root}"
        )

    staging_root = run_config.run_dir_path / "source_inputs"
    if staging_root.exists():
        shutil.rmtree(staging_root)

    _copy_inputs_to_directory(source_root, staging_root, use_reference_data)

    if ablation_mode == "corrupt":
        override_root = ablation_root / "corrupt" / "data" / run_config.task_id
    elif ablation_mode == "decoy":
        override_root = ablation_root / "decoys" / "data" / run_config.task_id
    else:
        raise click.ClickException(f"Unknown ablation mode '{ablation_mode}'.")

    if not override_root.exists():
        raise click.ClickException(
            f"Ablation data not found for '{run_config.task_id}' at {override_root}"
        )

    _overlay_directory(override_root, staging_root / "data")
    return staging_root


def _load_prompt_bloat_map(prompt_path: Path) -> dict[str, str]:
    import importlib.util

    if not prompt_path.exists():
        raise click.ClickException(f"Prompt bloat file not found at {prompt_path}")

    spec = importlib.util.spec_from_file_location("prompt_bloat", prompt_path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Failed to load prompt bloat module from {prompt_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, "bloats"):
        raise click.ClickException(f"Prompt bloat module does not contain 'bloats' dictionary")
    
    mapping: dict[str, str] = {}
    for label, content in module.bloats.items():
        task_id = label.strip().lower().replace("_", "-")
        mapping[task_id] = content.strip()
    return mapping


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


def _run_agent_subprocess_with_timeout(
    env_name: str, config_path: Path, timeout_seconds: float | None
) -> None:
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
    prepare_inputs: Callable[[RunConfig], Path] | None = None,
) -> int:
    completed = 0
    for task in list(tasks):
        run_config = build_run_config(task)
        staged_path: Path | None = None

        if prepare_inputs:
            run_config.run_dir_path.mkdir(parents=True, exist_ok=True)
            staged_path = prepare_inputs(run_config)
            run_config.data_path = staged_path


        _prepare_run_config(run_config)
        env_alias = f"{task.task_id}-{run_config.run_hash[:8]}"
        try:
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
        finally:
            if staged_path and staged_path.exists():
                shutil.rmtree(staged_path, ignore_errors=True)
    return completed


def _filter_tasks(
    tasks: Sequence[DataSet], task_ids: Sequence[str] | None
) -> list[DataSet]:
    if not task_ids:
        return list(tasks)
    wanted = set(task_ids)
    selected = [task for task in tasks if task.task_id in wanted]
    missing = sorted(wanted - {task.task_id for task in selected})
    if missing:
        raise click.ClickException(
            "Unknown/unavailable task_ids: " + ", ".join(missing)
        )
    return selected


def run_ablations(
    settings: Sequence[str],
    task_ids: Sequence[str] | None,
    timeout_hours: float,
) -> None:
    datasets = DataSet.load_all(metadata_path=str(METADATA_PATH), data_root=str(DATA_ROOT))
    selected_tasks = _filter_tasks(datasets, task_ids)

    if RUN_LOGS is None:
        raise click.ClickException("RUN_LOGS is not configured. Did you run via main()?")

    prompt_bloat_map = {}
    if "prompt-bloat" in settings:
        prompt_bloat_map = _load_prompt_bloat_map(PROMPT_BLOAT_PATH)

    timeout_seconds = None if timeout_hours <= 0 else timeout_hours * 3600

    for setting in settings:
        experiment_name = f"ablation-{setting}"

        if setting == "prompt-bloat":
            filtered_tasks = []
            missing_prompts = []
            for task in selected_tasks:
                if task.task_id in prompt_bloat_map:
                    filtered_tasks.append(task)
                else:
                    missing_prompts.append(task.task_id)

            if task_ids and missing_prompts:
                raise click.ClickException(
                    "Missing prompt bloat text for task ids: "
                    + ", ".join(sorted(missing_prompts))
                )

            if missing_prompts:
                logging.warning(
                    "Skipping tasks without prompt bloat text: %s",
                    ", ".join(sorted(missing_prompts)),
                )
            tasks_to_run = filtered_tasks
        else:
            tasks_to_run = selected_tasks

        if not tasks_to_run:
            logging.warning("No tasks to run for setting '%s'.", setting)
            continue

        def _build(task: DataSet) -> RunConfig:
            task_prompt = task.task_prompt
            if setting == "prompt-bloat":
                bloat_text = prompt_bloat_map.get(task.task_id, "")
                if bloat_text:
                    task_prompt = f"{bloat_text}\n\n{task.task_prompt}"

            return _build_run_config(
                task=task,
                task_prompt=task_prompt,
                run_logs=RUN_LOGS,
                experiment_name=experiment_name,
                model=MODEL_NAME,
                system_prompt_name=SYSTEM_PROMPT_NAME,
                use_reference_data=USE_REFERENCE_DATA,
                tool_names=TOOL_NAMES,
                otel_sink_host=OTEL_SINK_HOST,
            )

        prepare_inputs = None
        if setting in ("corrupt", "decoy"):

            def _prepare_inputs(run_config: RunConfig, mode: str = setting) -> Path:
                return _prepare_ablation_inputs(
                    run_config=run_config,
                    ablation_mode=mode,
                    use_reference_data=USE_REFERENCE_DATA,
                    ablation_root=ABLATION_ROOT,
                )

            prepare_inputs = _prepare_inputs

        completed = _execute_tasks_in_env(
            tasks=tasks_to_run,
            env_file=ENV_FILE,
            build_run_config=_build,
            timeout_seconds=timeout_seconds,
            prepare_inputs=prepare_inputs,
        )

        click.echo(
            f"Completed {completed}/{len(tasks_to_run)} tasks for setting '{setting}' with model '{MODEL_NAME}'."
        )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--setting",
    "settings",
    type=click.Choice(SETTING_CHOICES, case_sensitive=False),
    multiple=True,
    help="Which ablation setting(s) to run. Defaults to all.",
)
@click.option(
    "--task-id",
    "task_ids",
    multiple=True,
    help="Repeat to run only the provided task ids (defaults to all tasks).",
)
@click.option(
    "--timeout-hours",
    type=float,
    default=4.0,
    show_default=True,
    help="Per-task timeout in hours. Use 0 or a negative value to disable the timeout.",
)
def main(
    settings: tuple[str, ...],
    task_ids: tuple[str, ...],
    timeout_hours: float,
) -> None:
    configure_logging()
    _ensure_required_env_vars()

    global RUN_LOGS
    RUN_LOGS = Path(os.getenv("RUN_LOGS"))

    selected_settings = [s.lower() for s in settings] if settings else list(SETTING_CHOICES)
    run_ablations(
        settings=selected_settings,
        task_ids=task_ids or None,
        timeout_hours=timeout_hours,
    )


if __name__ == "__main__":
    main()
