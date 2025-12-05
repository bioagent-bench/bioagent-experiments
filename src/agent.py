from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from otel import sum_token_counts
from src.mcp_configs import modify_codex_config, remove_codex_mcp_config
from .logs import RunConfig, configure_logging


configure_logging()


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
    logging.info(f"Copying inputs to run directory: {copied_inputs_root}")
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

        prompt = (
            run_config.system_prompt
            + "\n\n"
            + run_config.task_prompt
            + f"\n\nThe input data is: {input_data}"
        )

        if run_config.experiment_name.startswith("open-environment"):
            # we shouldn't use an old MCP if we run open-environment
            remove_codex_mcp_config()

        # set the required tools in the MCP
        else:
            tools_json = run_config.run_dir_path / "tools.json"
            tools_json.write_text(json.dumps(run_config.tool_names))
            logging.info("Modifying Codex config for enabling tools")
            modify_codex_config(
                username=run_config.experiment_name, tools_config=tools_json
            )
        start_time = time.time()
        logging.info(f"Starting codex execution at {start_time}")
        if run_config.model in ("claude-opus-4-5", "claude-sonnet-4-5"):
            subprocess.run(
                [
                    "claude",
                    "-p",
                    prompt,
                    "--model",
                    run_config.model,
                    "--dangerously-skip-permissions",
                ]
            )
        else:
            subprocess.run(
                [
                    "codex",
                    "exec",
                    prompt,
                    "--profile",
                    run_config.model,
                    "--skip-git-repo-check",
                    "--yolo",
                ]
            )
        end_time = time.time()

        logging.info(f"Codex execution finished at {end_time}")
        run_config.duration = (end_time - start_time) / 60  # minutes

        try:
            in_tok, out_tok = sum_token_counts(run_config.otel_sink_path)
            run_config.input_tokens = int(in_tok)
            run_config.output_tokens = int(out_tok)
            logging.info(f"Aggregated tokens — input: {in_tok}, output: {out_tok}")
        except Exception as e:
            logging.exception(f"Failed to aggregate token counts: {e}")

        run_config.save_run_metadata()

        # clean up input data after execution completes
        inputs_folder = run_dir_path / "inputs"
        if inputs_folder.exists():
            logging.info(f"Deleting inputs folder: {inputs_folder}")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
