from __future__ import annotations

import argparse
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from phoenix.otel import register
from smolagents import CodeAgent
from opentelemetry import trace

from .judge_agent import (
    EvaluationResults,
    build_judge_prompt_csv,
    build_judge_prompt_giab,
    eval_giab_metrics,
    parse_agent_outputs,
    parse_agent_results,
)
from .logs import RunConfig
from .models import create_azure_model, load_model
from .tools import run_terminal_command


SmolagentsInstrumentor().instrument()
register(project_name="bioagent-experiments")

TOOLS_REGISTRY: dict[str, Any] = {
    "run_terminal_command": run_terminal_command,
}


def load_run_config(config_path: Path) -> RunConfig:
    """Load a ``RunConfig`` instance from a metadata JSON file.

    Args:
        config_path (Path): Path to the run metadata JSON file.

    Returns:
        RunConfig: Deserialized run configuration.
    """

    run_config = RunConfig.load_run_metadata(config_path)

    tools: list[Any] = []
    for tool_name in run_config.tool_names:
        tool = TOOLS_REGISTRY.get(tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool requested in configuration: {tool_name}")
        tools.append(tool)

    run_config.tools = tools
    run_config.num_tools = len(tools)

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

    if run_config.data_path is None:
        raise ValueError("RunConfig.data_path must be set before running an evaluation.")

    inputs_root = Path(run_config.data_path)
    if not inputs_root.exists():
        raise FileNotFoundError(f"Input data directory does not exist: {inputs_root}")

    run_dir_path = run_config.run_dir_path
    log_path = run_config.metadata_path

    run_dir_path.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    create_dirs(run_dir_path)

    # Add run_hash as custom attribute to Phoenix traces
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("bioagent_evaluation") as span:
        span.set_attribute("run_hash", run_config.run_hash)
        span.set_attribute("task_id", run_config.task_id)
        span.set_attribute("experiment_name", run_config.experiment_name)
        span.set_attribute("model", run_config.model)

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

            agent = CodeAgent(
                max_steps=run_config.max_steps,
                model=load_model(run_config.model),
                tools=run_config.tools,
                additional_authorized_imports=["*"],
                planning_interval=run_config.planning_interval,
                return_full_result=True,
            )
            agent.prompt_templates["system_prompt"] = run_config.system_prompt
            results = agent.run(run_config.task_prompt + f"\n\nThe input data is: {input_data}")

        # collect stuff from the results
        run_config.input_tokens = results.token_usage.input_tokens
        run_config.output_tokens = results.token_usage.output_tokens
        run_config.duration = results.timing.duration / 60
        run_config.steps = len(results.steps)
        agent_output_tree = parse_agent_outputs(run_config.run_dir_path / "results")

        if run_config.task_id == "giab":
            agent_results = eval_giab_metrics(
                run_config.run_dir_path / "results",
                run_config.run_dir_path / "results",
                inputs_root / "data" / "Agilent_v7.chr.bed",
                inputs_root / "reference" / "Homo_sapiens_assembly38.fasta",
            )
            judge_prompt = build_judge_prompt_giab(
                input_data,
                run_config.task_prompt,
                agent_output_tree,
                agent_results,
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

        client = create_azure_model(framework="openai")
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": judge_prompt}],
        ).choices[0].message.content
        final_result = EvaluationResults(**json.loads(response))

        run_config.eval_results = final_result
        run_config.save_run_metadata()

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



        




