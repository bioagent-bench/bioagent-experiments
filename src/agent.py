from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from otel import sum_token_counts
from .logs import RunConfig, configure_logging
from .system_prompts import prompts


configure_logging()

DOCKER_RUN_FLAG = "BIOAGENT_RUN_IN_DOCKER"
DOCKER_IMAGE_ENV = "BIOAGENT_DOCKER_IMAGE"
IN_DOCKER_FLAG = "BIOAGENT_IN_DOCKER"
OTEL_HOST_ENV = "BIOAGENT_OTEL_HOST"
DEFAULT_DOCKER_IMAGE = "bioagent-experiments:latest"


def _flag_enabled(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_running_in_docker() -> bool:
    if _flag_enabled(os.getenv(IN_DOCKER_FLAG)):
        return True
    return Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _mount_dir_for(path: Path) -> Path:
    if path.exists() and path.is_dir():
        return path
    if path.suffix:
        return path.parent
    return path


def _prune_mounts(mounts: set[Path]) -> list[Path]:
    resolved = [mount.resolve() for mount in mounts]
    ordered = sorted(resolved, key=lambda p: len(str(p)))
    result: list[Path] = []
    for candidate in ordered:
        if any(_is_relative_to(candidate, existing) for existing in result):
            continue
        result.append(candidate)
    return result


def _docker_env_vars(run_config: RunConfig) -> dict[str, str]:
    passthrough_keys = {
        "RUN_LOGS",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "ANTHROPIC_FOUNDRY_API_KEY",
        "CODEX_HOME",
    }
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in passthrough_keys or key.startswith("OTEL_"):
            env[key] = value

    env.pop(DOCKER_RUN_FLAG, None)
    env.pop(IN_DOCKER_FLAG, None)

    otel_host = run_config.otel_sink_host or ""
    if ("127.0.0.1" in otel_host or "localhost" in otel_host) and not env.get(
        OTEL_HOST_ENV
    ):
        env[OTEL_HOST_ENV] = "host.docker.internal:4317"

    return env


def _build_docker_command(config_path: Path, run_config: RunConfig, image: str) -> list[str]:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required but was not found on PATH.")

    project_root = Path(__file__).resolve().parents[1]
    mounts: set[Path] = {
        _mount_dir_for(project_root),
        _mount_dir_for(run_config.data_path),
        _mount_dir_for(run_config.run_dir_path),
        _mount_dir_for(run_config.metadata_path),
        _mount_dir_for(run_config.otel_sink_path),
    }
    mounts = _prune_mounts(mounts)

    cmd = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--add-host",
        "host.docker.internal:host-gateway",
        "-w",
        str(project_root),
        "-e",
        f"{IN_DOCKER_FLAG}=1",
    ]

    for key, value in _docker_env_vars(run_config).items():
        cmd.extend(["-e", f"{key}={value}"])

    for mount in mounts:
        cmd.extend(["-v", f"{mount}:{mount}"])

    codex_home = Path("~/.codex").expanduser()
    if codex_home.exists():
        cmd.extend(["-v", f"{codex_home}:/root/.codex"])

    cmd.append(image)
    cmd.extend(["python", "-m", "src.agent", "--config", str(config_path)])
    return cmd


def _run_in_docker(config_path: Path, image: str | None) -> None:
    run_config = load_run_config(config_path)
    docker_image = image or os.getenv(DOCKER_IMAGE_ENV) or DEFAULT_DOCKER_IMAGE
    cmd = _build_docker_command(config_path, run_config, docker_image)
    subprocess.run(cmd, check=True)


def _should_run_in_docker(args: argparse.Namespace) -> bool:
    if args.docker or args.docker_image:
        return True
    if _flag_enabled(os.getenv(DOCKER_RUN_FLAG)):
        return True
    return bool(os.getenv(DOCKER_IMAGE_ENV))


def load_run_config(config_path: Path) -> RunConfig:
    """Load a ``RunConfig`` instance from a metadata JSON file.

    Args:
        config_path (Path): Path to the run metadata JSON file.

    Returns:
        RunConfig: Deserialized run configuration.
    """

    run_config = RunConfig.load_run_metadata(config_path)
    run_config.num_tools = len(run_config.tool_names)

    return run_config


def create_dirs(prefix: Path) -> None:
    """Create standard output directories for an evaluation run.

    Args:
        prefix (Path): Base directory for the evaluation artifacts.

    Returns:
        None: This function creates directories as a side effect.
    """

    root = Path(prefix)
    directories = (
        root / "outputs",
        root / "results",
    )

    for path in directories:
        path.mkdir(parents=True, exist_ok=True)


def glob_input_data(*input_dirs: Path) -> list[Path]:
    """Collect all files used as input for evaluation.

    Args:
        *input_dirs (Path): Directories containing task-specific input or reference data.

    Returns:
        list[Path]: Sorted list of file paths discovered under the provided directories.
    """

    files: set[Path] = set()
    for directory in input_dirs:
        root = Path(directory)
        if not root.exists():
            continue
        files.update(p for p in root.rglob("*") if p.is_file())

    return sorted(files)


def _build_subprocess_env(run_config: RunConfig) -> dict[str, str]:
    env = os.environ.copy()

    endpoint = run_config.otel_sink_host
    if endpoint:
        endpoint = endpoint if endpoint.startswith("http") else f"http://{endpoint}"
        env["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
        env["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"] = endpoint
        env["OTEL_EXPORTER_OTLP_GRPC_ENDPOINT"] = endpoint
        env["OTEL_EXPORTER_OTLP_PROTOCOL"] = "grpc"

    attrs_raw = env.get("OTEL_RESOURCE_ATTRIBUTES", "")
    attrs = [entry.strip() for entry in attrs_raw.split(",") if entry.strip()]
    attrs = [entry for entry in attrs if not entry.startswith("run_hash=")]
    attrs.append(f"run_hash={run_config.run_hash}")
    env["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(attrs)

    return env


def copy_inputs_to_run_directory(
    source_root: Path,
    destination_root: Path,
    use_reference_data: bool,
) -> None:
    """Copy evaluation inputs into the run directory.

    Args:
        source_root (Path): Directory containing the task inputs.
        destination_root (Path): Destination directory under the run path.

    Returns:
        None: This function copies files as a side effect.
    """

    destination_root.mkdir(parents=True, exist_ok=True)
    source = Path(source_root)
    if not source.exists():
        return

    excluded_names: set[str] = {"results"}

    # If we don't use reference data, we exclude the reference directory
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


@contextmanager
def isolated_run_environment(
    run_dir_path: Path, inputs_root: Path, use_reference_data: bool
) -> Iterator[Path]:
    """Provide an isolated working directory for agent execution.

    Args:
        run_dir_path (Path): Root directory for the current run.
        inputs_root (Path): Source directory containing task inputs.

    Yields:
        Iterator[Path]: Path to the copied inputs directory inside the run path.
    """

    copied_inputs_root = run_dir_path / "inputs"
    logging.debug(f"Copying inputs to run directory: {copied_inputs_root}")
    copy_inputs_to_run_directory(inputs_root, copied_inputs_root, use_reference_data)

    previous_cwd = Path.cwd()
    os.chdir(run_dir_path)
    try:
        yield copied_inputs_root
    finally:
        os.chdir(previous_cwd)


def run_agent_task(run_config: RunConfig) -> RunConfig:
    """Run a single dataset evaluation using the provided configuration.

    Args:
        run_config (RunConfig): Configuration describing how to run the agent.

    Returns:
        RunConfig: Updated run configuration with evaluation results.
    """
    inputs_root = Path(run_config.data_path)
    if not inputs_root.exists():
        raise FileNotFoundError(f"Input data directory does not exist: {inputs_root}")

    otel_host_override = os.getenv(OTEL_HOST_ENV)
    if otel_host_override:
        run_config.otel_sink_host = otel_host_override

    run_dir_path = run_config.run_dir_path
    log_path = run_config.metadata_path

    run_dir_path.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    create_dirs(run_dir_path)

    outputs_root = run_dir_path / "outputs"
    results_root = run_dir_path / "results"

    input_data: list[Path] = []

    # create an isolated run environment
    # use_reference data controls wether we include ref data or not
    with isolated_run_environment(
        run_dir_path, inputs_root, run_config.use_reference_data
    ) as run_inputs_root:
        data_dir = run_inputs_root / "data"
        reference_dir = run_inputs_root / "reference"

        if run_config.use_reference_data:
            input_data = glob_input_data(data_dir, reference_dir)
        else:
            input_data = glob_input_data(data_dir)

        system_prompt_template = prompts.get(run_config.system_prompt_name)
        env_alias = f"{run_config.task_id}-{run_config.run_hash[:8]}"
        system_prompt = system_prompt_template.format(env_name=env_alias)
        prompt = (
            system_prompt
            + "\n\n"
            + run_config.task_prompt
            + f"\n\nThe input data is: {input_data}"
        )

        start_time = time.time()
        logging.debug(f"Starting codex execution at {start_time}")
        process_env = _build_subprocess_env(run_config)
        
        if not run_config.model.startswith("gpt"):
            raise ValueError(
                f"Unsupported model '{run_config.model}'. Only codex-cli profiles are allowed."
            )

        subprocess.run(
            [
                "codex",
                "exec",
                prompt,
                "--profile",
                run_config.model,
                "--skip-git-repo-check",
                "--sandbox",
                "workspace-write",
            ],
            env=process_env,
        )
        end_time = time.time()

        logging.debug(f"Codex execution finished at {end_time}")
        run_config.duration = (end_time - start_time) / 60  # minutes

        try:
            in_tok, out_tok = sum_token_counts(run_config.otel_sink_path)
            run_config.input_tokens = int(in_tok)
            run_config.output_tokens = int(out_tok)
            logging.debug(f"Aggregated tokens — input: {in_tok}, output: {out_tok}")
        except Exception as e:
            logging.exception(f"Failed to aggregate token counts: {e}")

        run_config.save_run_metadata()

        # clean up input data after execution completes
        inputs_folder = run_dir_path / "inputs"
        if inputs_folder.exists():
            logging.debug(f"Deleting inputs folder: {inputs_folder}")
            shutil.rmtree(inputs_folder)

    return run_config


def run_from_config(config_path: Path) -> RunConfig:
    """Load a run configuration from disk and execute the evaluation."""

    run_config = load_run_config(config_path)
    return run_agent_task(run_config)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script."""

    parser = argparse.ArgumentParser(description="Run a single evaluation loop.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the serialized RunConfig JSON file.",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Run the agent inside a Docker container.",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default=None,
        help="Docker image to use for container execution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if _should_run_in_docker(args) and not _is_running_in_docker():
        _run_in_docker(args.config, args.docker_image)
        return
    run_from_config(args.config)


if __name__ == "__main__":
    main()
