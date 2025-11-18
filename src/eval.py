
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.judge_agent import EvaluationResults, EvaluationResultsGiab, EvaluationResultsGiabSchema, EvaluationResultsSchema, build_judge_prompt_csv, build_judge_prompt_giab, eval_giab_metrics, parse_agent_outputs, parse_agent_results
from src.logs import RunConfig, configure_logging
from src.models import create_azure_model

configure_logging()

def run_eval(run_config: RunConfig):
    agent_output_tree = parse_agent_outputs(run_config.run_dir_path / "outputs")
    client = create_azure_model(framework="openai")

    logging.info(f"\t\tRunning judge LLM to evaluate the results")
    if run_config.task_id == "giab":
        agent_results = eval_giab_metrics(
            run_config.run_dir_path / "results",
            run_config.data_path / "results",
            run_config.data_path / "data" / "Agilent_v7.chr.bed",
            run_config.data_path / "reference" / "Homo_sapiens_assembly38.fasta",
        )
        judge_prompt = build_judge_prompt_giab(
            run_config.data_path / "data",
            run_config.task_prompt,
            agent_output_tree,
            agent_results,
            run_config.task_id,
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
            results_match=parsed_response.results_match,
            notes=parsed_response.notes,
        )

    else:
        agent_results = parse_agent_results(run_config.run_dir_path / "results")
        truth_results = parse_agent_results(run_config.data_path / "results")
        judge_prompt = build_judge_prompt_csv(
            run_config.data_path / "data",
            run_config.task_prompt,
            agent_output_tree,
            agent_results,
            truth_results,
            run_config.task_id,
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
            results_match=parsed_response.results_match,
            notes=parsed_response.notes,
        )
    logging.info(f"\t\tJudge LLM finished evaluating the results.")
    run_config.eval_results = final_result
    run_config.save_run_metadata()
    logging.info(f"Run configuration saved with results: {run_config.eval_results}")


if __name__ == "__main__":
    run_logs_env = os.getenv("RUN_LOGS")
    if not run_logs_env:
        raise EnvironmentError("RUN_LOGS environment variable is not set")
    logging.info(f"Running evals for {run_logs_env}")
    runs_dir = Path(run_logs_env) / "runs"
    run_files = sorted(runs_dir.glob("*.json"))
    for run_file in tqdm(run_files, desc="Evaluating runs", unit="run"):
        run_config = RunConfig.load_run_metadata(run_file)
        if run_config.eval_results is None:
            logging.info(f"Running eval for {run_config.task_id}-{run_config.experiment_name}")
            run_eval(run_config)
        else:
            logging.info(f"Eval already run for {run_config.task_id}-{run_config.experiment_name}")