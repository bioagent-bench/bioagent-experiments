
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from phoenix.otel import register
from smolagents import CodeAgent
from opentelemetry import trace

from dataset import DataSet
from docker_sandbox import DockerSandbox
from judge_agent import (
    EvaluationResults,
    build_judge_prompt_csv,
    build_judge_prompt_giab,
    eval_giab_metrics,
    parse_agent_outputs,
    parse_agent_results,
)
from logs import RunConfig
from models import create_azure_model
from system_prompts import prompts
from tools import run_terminal_command


SmolagentsInstrumentor().instrument()
register(
    project_name="bioagent-experiments"
)

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

def evaluate_task(run_config: RunConfig) -> RunConfig:
    """Run a single dataset evaluation using the provided configuration.

    Args:
        run_config (RunConfig): Configuration describing how to run the agent.

    Returns:
        RunConfig: Updated run configuration with evaluation results.
    """

    run_timestamp = run_config.timestamp.strftime("%Y%m%d-%H%M%S")
    run_logs_root = run_config.run_logs_root or Path("./run-logs")
    experiment_root = Path(run_logs_root) / run_config.experiment_name / run_config.task_id
    run_path = experiment_root / run_timestamp
    create_dirs(run_path)

    inputs_root = run_config.data_path
    outputs_root = run_path / "outputs"
    results_root = run_path / "results"

    input_data = glob_input_data(inputs_root / "data", inputs_root / "reference")

    data_dir = inputs_root / "data"
    reference_dir = inputs_root / "reference"
    executor_kwargs: dict[str, str] = {
        "output_path": str(outputs_root),
        "results_path": str(results_root),
    }
    if data_dir.exists():
        executor_kwargs["data_path"] = str(data_dir)
    if reference_dir.exists():
        executor_kwargs["reference_path"] = str(reference_dir)

    agent = CodeAgent(
        max_steps=run_config.max_steps,
        model=create_azure_model(),
        tools=run_config.tools,
        additional_authorized_imports=["*"],
        planning_interval=run_config.planning_interval,
        return_full_result=True,
        executor_type="docker",
        executor_kwargs=executor_kwargs,
    )
    # agent.prompt_templates["system_prompt"] = run_config.system_prompt
    # results = agent.run(run_config.task_prompt + f"\n\nThe input data is: {input_data}")

    # collect stuff from the results
    run_config.input_tokens = 0
    run_config.output_tokens = 0
    run_config.duration = 0
    run_config.steps = 0
    # run_config.input_tokens = results.token_usage.input_tokens
    # run_config.output_tokens = results.token_usage.output_tokens
    # run_config.duration = results.timing.duration / 60 # in minutes
    # run_config.steps = len(results.steps)
    agent_output_tree = parse_agent_outputs(run_path)

    if run_config.task_id == "giab":
        agent_results = eval_giab_metrics(
            run_path / "results",
            run_path / "results",
            inputs_root / "data" / "Agilent_v7.chr.bed",
            inputs_root / "reference" / "Homo_sapiens_assembly38.fasta",
        )
        agent_prompt = build_judge_prompt_giab(
            input_data,
            run_config.task_prompt,
            agent_output_tree,
            agent_results,
        )
    else:
        agent_results = parse_agent_results(run_path / "results")
        truth_results = parse_agent_results(run_path / "results")
        agent_prompt = build_judge_prompt_csv(
            input_data,
            run_config.task_prompt,
            agent_output_tree,
            agent_results,
            truth_results,
        )

    # client = create_azure_model(framework="openai")
    # response = client.chat.completions.create(
    #     model="gpt-5-mini",
    #     messages=[{"role": "user", "content": agent_prompt}],
    # ).choices[0].message.content
    # final_result = EvaluationResults(**json.loads(response))

    # run_config.eval_results = final_result
    run_config.eval_results = "placeholder"
    metadata_path = run_path / "run_metadata.json"
    run_config.save_run_metadata(metadata_path)

    return run_config


def main() -> None:
    """Execute evaluations configured via ``RunConfig`` entries."""

    datasets = DataSet.load_all(
        metadata_path=Path("~/bioagent-bench/src/task_metadata.json").expanduser(),
        data_root=Path("~/bioagent-data").expanduser(),
    )
    tools: list[Iterable] = [run_terminal_command]

    default_run_logs_root = Path(os.getenv("RUN_LOGS_ROOT"))

    for task in datasets:
        if task.task_id != "alzheimer-mouse":
            continue
        run_hash = f"{task.task_id}-{datetime.now().isoformat()}"
        run_config = RunConfig(
            run_hash=run_hash,
            timestamp=datetime.now(),
            task_id=task.task_id,
            task_prompt=task.task_prompt,
            max_steps=1,
            planning_interval=1,
            num_tools=len(tools),
            tools=tools,
            system_prompt=prompts["v1"],
            system_prompt_name="v1",
            experiment_name="open-environment",
            model="azure",
            run_logs_root=default_run_logs_root,
            data_path=Path(task.path),
        )

        print(f"Processing task: {task.task_id} at {task.path}")
        result = evaluate_task(run_config)
        print(f"Completed evaluation for {task.task_id}")


if __name__ == "__main__":
    main()



        




