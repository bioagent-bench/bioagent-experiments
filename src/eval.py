
import logging
import os
import sys
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.judge_agent import EvaluationResults, EvaluationResultsGiab, EvaluationResultsGiabSchema, EvaluationResultsSchema, build_judge_prompt_csv, build_judge_prompt_giab, eval_giab_metrics, parse_agent_outputs, parse_agent_results
from src.logs import RunConfig, configure_logging

configure_logging()

api_key_path = PROJECT_ROOT / ".keys" / "azure_api.key"
openai_api_key = api_key_path.read_text()
openai_api_url = PROJECT_ROOT / ".keys" / "azure_endpoint.key"
openai_api_url = openai_api_url.read_text()

openai_client = OpenAI(api_key=openai_api_key, base_url=openai_api_url)

def run_eval(run_config: RunConfig):
    agent_output_tree = parse_agent_outputs(run_config.run_dir_path / "outputs")

    if run_config.use_reference_data:
        reference_data = run_config.data_path / "data/reference"
    else:
        reference_data = "No reference data provided - In this mode the agent was supposed to download reference data"

    download_dest = run_config.data_path.parent
    assert (run_config.data_path / "results").exists(), (
        "Results directory does not exist. "
        "Please download from bioagent-bench using: "
        f'uv run python src/dataset.py download --results --no-data --no-reference --dest "{download_dest}"'
    )

    logging.info(f"\t\tRunning judge LLM to evaluate the results")
    if run_config.task_id == "giab":
        agent_results = eval_giab_metrics(
            agent_results_dir=run_config.run_dir_path / "results",
            truth_dir=run_config.data_path / "results",
            input_bed=run_config.data_path / "data" / "Agilent_v7.chr.bed",
            ref_fasta=run_config.data_path / "reference" / "Homo_sapiens_assembly38.fasta",
        )        
        
        
        judge_prompt = build_judge_prompt_giab(
            input_data=run_config.data_path / "data/input",
            reference_data=reference_data,
            task_prompt=run_config.task_prompt,
            processing_tree=agent_output_tree,
            results=agent_results,
            task_id=run_config.task_id,
        )
        response = openai_client.responses.parse(
            model="gpt-5.1",
            reasoning={"effort": "high"},
            text_format=EvaluationResultsGiabSchema,
            input=judge_prompt
        )
        parsed_response = response.output_parsed
        final_result = EvaluationResultsGiab(
            steps_completed=parsed_response.steps_completed,
            steps_to_completion=parsed_response.steps_to_completion,
            final_result_reached=parsed_response.final_result_reached,
            f1_score=parsed_response.f1_score,
            results_match=parsed_response.results_match,
            notes=parsed_response.notes,
        )

    else:
        agent_results = parse_agent_results(run_config.run_dir_path / "results")
        truth_results = parse_agent_results(run_config.data_path / "results")
        judge_prompt = build_judge_prompt_csv(
            input_data=run_config.data_path / "data/input",
            reference_data=reference_data,
            task_prompt=run_config.task_prompt,
            processing_tree=agent_output_tree,
            results=agent_results,
            truth=truth_results,
            task_id=run_config.task_id,
        )
        response = openai_client.responses.parse(
            model="gpt-5.1",
            reasoning={"effort": "medium"},
            text_format=EvaluationResultsGiabSchema,
            input=judge_prompt
        )
        parsed_response = response.output_parsed
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
    run_configs = [RunConfig.load_run_metadata(run_file) for run_file in run_files]

    run_configs_pending = [run_config for run_config in run_configs if run_config.eval_results is None]
    logging.info(f"{len(run_configs_pending)}/{len(run_configs)} runs need evaluation")

    if not run_configs_pending:
        logging.info("No runs pending evaluation.")

    for run_config in tqdm(run_configs_pending, desc="Evaluating runs", unit="run", total=len(run_configs_pending)):
        logging.info(f"Running eval for {run_config.task_id}-{run_config.experiment_name}")
        run_eval(run_config)