
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from phoenix.otel import register
from smolagents import CodeAgent

from dataset import DataSet
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


register()
SmolagentsInstrumentor().instrument()


def create_dirs(prefix: Path) -> None:
    """Create standard output directories for an evaluation run.

    Args:
        prefix (Path): Base directory for the evaluation artifacts.

    Returns:
        None: This function creates directories as a side effect.
    """

    root = Path(prefix)
    outputs_path = root / "outputs"
    results_path = root / "results"
    outputs_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)


def glob_input_data(data_dir: Path, ref_dir: Path) -> list[Path]:
    """Collect all files used as input for evaluation.

    Args:
        data_dir (Path): Directory containing task-specific data inputs.
        ref_dir (Path): Directory containing reference materials.

    Returns:
        list[Path]: Sorted list of file paths discovered under ``data_dir`` and ``ref_dir``.
    """

    data_files = [p for p in data_dir.rglob("*") if p.is_file()]
    ref_files = [p for p in ref_dir.rglob("*") if p.is_file()]
    return sorted({*data_files, *ref_files})


def evaluate_task(run_config: RunConfig) -> EvaluationResults:
    """Run a single dataset evaluation using the provided configuration.

    Args:
        run_config (RunConfig): Configuration describing how to run the agent.
        task (DataSet): Dataset metadata for the evaluation task.

    Returns:
        EvaluationResults: Structured results parsed from the judge model output.
    """

    experiment_root = Path(run_config.experiment_name).expanduser()
    test_path = experiment_root / run_config.task_id
    create_dirs(test_path)

    input_data = glob_input_data(test_path / "data", test_path / "reference")

    agent = CodeAgent(
        max_steps=run_config.max_steps,
        model=create_azure_model(),
        tools=run_config.tools,
        additional_authorized_imports=["*"],
        planning_interval=run_config.planning_interval,
    )
    agent.prompt_templates["system_prompt"] = run_config.system_prompt
    results = agent.run(run_config.task_prompt + f"\n\nThe input data is: {input_data}")
    print(results.token_usage)
    print(results.timing)
    agent_output_tree = parse_agent_outputs(test_path)

    if run_config.task_id == "giab":
        agent_results = eval_giab_metrics(
            test_path / "results",
            test_path / "results",
            test_path / "data" / "Agilent_v7.chr.bed",
            test_path / "reference" / "Homo_sapiens_assembly38.fasta",
        )
        agent_prompt = build_judge_prompt_giab(
            input_data,
            run_config.task_prompt,
            agent_output_tree,
            agent_results,
        )
    else:
        agent_results = parse_agent_results(test_path / "results")
        truth_results = parse_agent_results(test_path / "results")
        agent_prompt = build_judge_prompt_csv(
            input_data,
            run_config.task_prompt,
            agent_output_tree,
            agent_results,
            truth_results,
        )

    client = create_azure_model(framework="openai")
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": agent_prompt}],
    ).choices[0].message.content
    final_result = EvaluationResults(**json.loads(response))

    run_config.eval_results = final_result
    run_config.timestamp_end = datetime.now()
    metadata_path = test_path / "results" / "run_metadata.json"
    run_config.save_run_metadata(metadata_path)

    return final_result


def main() -> None:
    """Execute evaluations configured via ``RunConfig`` entries."""

    datasets = DataSet.load_all()
    tools: list[Iterable] = [run_terminal_command]

    for task in datasets:
        if task.task_id != "alzheimer-mouse":
            continue

        run_config = RunConfig(
            task_id=task.task_id,
            task_prompt=task.task_prompt,
            max_steps=1,
            planning_interval=1,
            num_tools=len(tools),
            tools=tools,
            system_prompt=prompts["v1"],
            system_prompt_name="v1",
            experiment_name="../test_outputs",
            model="azure",
        )

        run_config.timestamp_start = datetime.now()
        print(f"Processing task: {task.task_id} at {task.path}")
        result = evaluate_task(run_config)
        print(f"Completed evaluation for {task.task_id}: {result}")


if __name__ == "__main__":
    main()



        




