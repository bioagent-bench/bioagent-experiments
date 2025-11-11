from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from otel import sum_token_counts

from .judge_agent import (
    EvaluationResults,
    EvaluationResultsGiab,
    EvaluationResultsSchema,
    EvaluationResultsGiabSchema,
    build_judge_prompt_csv,
    build_judge_prompt_giab,
    eval_giab_metrics,
    parse_agent_outputs,
    parse_agent_results,
)
from .logs import RunConfig, configure_logging
from .models import create_azure_model, load_model
from .tools import REGISTRY


configure_logging()

def load_run_config(config_path: Path) -> RunConfig:
    """Load a ``RunConfig`` instance from a metadata JSON file.

    Args:
        config_path (Path): Path to the run metadata JSON file.

    Returns:
        RunConfig: Deserialized run configuration.
    """

    run_config = RunConfig.load_run_metadata(config_path)

    resolved_tools = REGISTRY.resolve_tools(run_config.tool_names)

    run_config.tools = resolved_tools
    run_config.num_tools = len(resolved_tools)

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


def copy_inputs_to_run_directory(source_root: Path, destination_root: Path) -> None:
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

    for item in source.iterdir():
        if item.name in excluded_names:
            continue
        target = destination_root / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


@contextmanager
def isolated_run_environment(run_dir_path: Path, inputs_root: Path) -> Iterator[Path]:
    """Provide an isolated working directory for agent execution.

    Args:
        run_dir_path (Path): Root directory for the current run.
        inputs_root (Path): Source directory containing task inputs.

    Yields:
        Iterator[Path]: Path to the copied inputs directory inside the run path.
    """

    copied_inputs_root = run_dir_path / "inputs"
    logging.info(f"Copying inputs to run directory: {copied_inputs_root}")
    copy_inputs_to_run_directory(inputs_root, copied_inputs_root)

    previous_cwd = Path.cwd()
    os.chdir(run_dir_path)
    try:
        yield copied_inputs_root
    finally:
        os.chdir(previous_cwd)


def evaluate_task(run_config: RunConfig) -> RunConfig:
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

    with isolated_run_environment(run_dir_path, inputs_root) as run_inputs_root:
        data_dir = run_inputs_root / "data"
        reference_dir = run_inputs_root / "reference"

        executor_kwargs: dict[str, Any] = {
            "output_path": str(outputs_root),
            "results_path": str(results_root),
        }
        if data_dir.exists():
            executor_kwargs["data_path"] = str(data_dir)
        if reference_dir.exists():
            executor_kwargs["reference_path"] = str(reference_dir)

        input_data = glob_input_data(data_dir, reference_dir)


        prompt = run_config.system_prompt + "\n\n" + run_config.task_prompt + f"\n\nThe input data is: {input_data}"
        start_time = time.time()
        logging.info(f"Starting codex execution at {start_time}")
        subprocess.run(["codex", "exec", prompt, "--skip-git-repo-check", "--yolo"])
        end_time = time.time()
        logging.info(f"Codex execution finished at {end_time}")
        run_config.duration = (end_time - start_time) / 60 # minutes

        try:
            in_tok, out_tok = sum_token_counts(run_config.otel_sink_path)
            run_config.input_tokens = int(in_tok)
            run_config.output_tokens = int(out_tok)
            logging.info(f"Aggregated tokens — input: {in_tok}, output: {out_tok}")
        except Exception as e:
            logging.exception(f"Failed to aggregate token counts: {e}")

        agent_output_tree = parse_agent_outputs(run_config.run_dir_path / "outputs")

        client = create_azure_model(framework="openai")

        logging.info(f"Running judge LLM to evaluate the results")

        if run_config.task_id == "giab":
            agent_results = eval_giab_metrics(
                run_config.run_dir_path / "results",
                run_config.data_path / "results",
                inputs_root / "data" / "Agilent_v7.chr.bed",
                inputs_root / "reference" / "Homo_sapiens_assembly38.fasta",
            )
            judge_prompt = build_judge_prompt_giab(
                input_data,
                run_config.task_prompt,
                agent_output_tree,
                agent_results,
            )
            completion = client.beta.chat.completions.parse(
                model="gpt-5",
                messages=[{"role": "user", "content": judge_prompt}],
                response_format=EvaluationResultsGiabSchema,
            )
            parsed_response = completion.choices[0].message.parsed
            final_result = EvaluationResultsGiab(
                steps_completed=parsed_response.steps_completed,
                steps_to_completion=parsed_response.steps_to_completion,
                final_results_reached=parsed_response.final_results_reached,
                f1_score=parsed_response.f1_score,
                notes=parsed_response.notes,
            )

        else:
            agent_results = parse_agent_results(run_config.run_dir_path / "results")
            truth_results = parse_agent_results(run_config.data_path / "results")
            judge_prompt = build_judge_prompt_csv(
                input_data,
                run_config.task_prompt,
                agent_output_tree,
                agent_results,
                truth_results,
            )
            completion = client.beta.chat.completions.parse(
                model="gpt-5",
                messages=[{"role": "user", "content": judge_prompt}],
                response_format=EvaluationResultsSchema,
            )
            parsed_response = completion.choices[0].message.parsed
            final_result = EvaluationResults(
                steps_completed=parsed_response.steps_completed,
                steps_to_completion=parsed_response.steps_to_completion,
                final_result_reached=parsed_response.final_result_reached,
                notes=parsed_response.notes,
            )

        logging.info(f"Judge LLM finished running with results: {final_result}")
        run_config.eval_results = final_result
        run_config.save_run_metadata()
        logging.info(f"Run configuration saved with results: {run_config.eval_results}")

    return run_config


def run_from_config(config_path: Path) -> RunConfig:
    """Load a run configuration from disk and execute the evaluation."""

    run_config = load_run_config(config_path)
    return evaluate_task(run_config)


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



        




