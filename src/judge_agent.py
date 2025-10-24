import csv
from dataclasses import dataclass
import subprocess
from tarfile import data_filter
import tempfile
from pathlib import Path
from pydantic import BaseModel

@dataclass
class EvaluationResults:
    steps_completed: int
    steps_to_completion: int
    final_result_reached: bool
    notes: str

@dataclass
class EvaluationResultsGiab:
    steps_completed: int
    steps_to_completion: int
    final_results_reached: bool
    f1_score: int
    notes: str

class EvaluationResultsSchema(BaseModel):
    """Schema for evaluation results used with structured output API.
    
    Attributes:
        steps_completed (int): Number of steps the agent completed.
        steps_to_completion (int): Number of steps needed to reach completion.
        final_result_reached (bool): Whether the agent reached the final result.
        notes (str): Additional evaluation notes.
    """
    steps_completed: int
    steps_to_completion: int
    final_result_reached: bool
    notes: str

class EvaluationResultsGiabSchema(BaseModel):
    """Schema for GIAB evaluation results used with structured output API.
    
    Attributes:
        steps_completed (int): Number of steps the agent completed.
        steps_to_completion (int): Number of steps needed to reach completion.
        final_results_reached (bool): Whether the agent reached the final results.
        f1_score (int): F1 score for the evaluation.
        notes (str): Additional evaluation notes.
    """
    steps_completed: int
    steps_to_completion: int
    final_results_reached: bool
    f1_score: int
    notes: str



def parse_agent_outputs(output_dir: Path) -> list[Path]:
    """Return files and immediate subdirectories under the directory.

    For each top-level directory, returns:
    - All files directly in that directory
    - Names of subdirectories (but not their contents)

    This reduces context spam by not listing all nested files.

    Args:
        output_dir (Path): Root directory to scan.

    Returns:
        list[Path]: File paths and subdirectory names relative to ``output_dir``.
    """
    root = Path(output_dir)
    results = []
    
    for item in root.rglob("*"):
        if ".snakemake" in item.parts:
            continue
            
        rel_path = item.relative_to(root)
        
        # Include files that are at most 2 levels deep (root/dir/file)
        # But not files deeper than that (root/dir/subdir/file)
        if item.is_file():
            # Count path depth: if it's root/file or root/dir/file, include it
            # Exclude root/dir/subdir/file and deeper
            if len(rel_path.parts) <= 2:
                results.append(rel_path)
        # Include directories that are immediate children of top-level dirs
        # (root/dir/subdir) but not deeper
        elif item.is_dir():
            if len(rel_path.parts) == 2:
                results.append(rel_path)
    
    return results


def parse_agent_results(results_dir):
    """Parse results from CSV/TSV files.

    Args:
        results_dir: Path to a directory containing results files, or a path
            to a single results file.

    Returns:
        If a single file path is provided and it has a ``.csv`` or ``.tsv``
        suffix, returns a list of dictionaries (one per row) parsed from the
        file. If a directory path is provided, returns a dictionary mapping
        ``Path`` objects of all discovered ``.csv``/``.tsv`` files under the
        directory to a list of row dictionaries for each file. If no matching
        files are found, returns an empty list (for single-file input) or an
        empty dict (for directory input).
    """

    path = Path(results_dir)

    def _read_table(file_path: Path, delimiter: str):
        with file_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            return list(reader)

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return _read_table(path, ",")
        if suffix == ".tsv":
            return _read_table(path, "\t")
        return []

    # if path is a directory parse all CSV/TSV files recursively
    if path.is_dir():
        results: dict[Path, list[dict]] = {}
        for file_path in list(path.rglob("*.csv")) + list(path.rglob("*.tsv")):
            if file_path.suffix.lower() == ".csv":
                results[file_path] = _read_table(file_path, ",")
            elif file_path.suffix.lower() == ".tsv":
                results[file_path] = _read_table(file_path, "\t")
        return results

    return {}

def eval_giab_metrics(
    agent_results_dir: Path,
    truth_dir: Path,
    input_bed: Path,
    ref_fasta: Path,
    ):
    """Run hap.py evaluation on GIAB VCF files and return summary metrics.

    Args:
        agent_results_dir (Path): Path to the agent results directory containing VCF files.
        truth_dir (Path): Path to the truth data directory containing benchmark files.

    Returns:
        dict: Dictionary containing hap.py summary metrics with keys like
        Type, TRUTH, QUERY, Recall, Precision, F1_Score, or empty dict if evaluation fails.
    """

    # Find agent VCF files
    vcf_candidates = sorted(agent_results_dir.rglob("*.vcf.gz"))
    if not vcf_candidates:
        return {'No VCF files have been generated successfully.'}

    agent_vcf = vcf_candidates[0]

    truth_vcf = truth_dir / "HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
    truth_bed = truth_dir / "HG001_GRCh38_1_22_v4.2.1_benchmark.bed"
    
    # Create temporary directory for hap.py output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_prefix = Path(temp_dir) / "evaluation"
        
        # Run hap.py
        cmd = [
            "mamba",
            "run",
            "-n", "hap",
            "hap.py",
            str(truth_vcf),
            str(agent_vcf),
            "-f", str(truth_bed),
            "-o", str(output_prefix),
            "-T", str(input_bed),
            "-r", str(ref_fasta),
            "--pass-only"
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Parse summary.csv file
        summary_file = Path(f"{output_prefix}.summary.csv")
        if summary_file.exists():
            with summary_file.open("r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Type") == "SNP":  # Return SNP metrics
                        return (
                            "These are benchmarking metrics from hap.py, which compares the "
                            "called variants (QUERY) to the Genome in a Bottle truth set (TRUTH). "
                            "The table shows results for SNPs, including the number of variants in "
                            "the truth set and query, as well as the calculated recall, precision, "
                            f"and F1-score.\n\n"
                            f"Type: {row.get('Type')}\n"
                            f"TRUTH Total: {row.get('TRUTH.TOTAL')}\n"
                            f"QUERY Total: {row.get('QUERY.TOTAL')}\n"
                            f"Recall: {row.get('METRIC.Recall')}\n"
                            f"Precision: {row.get('METRIC.Precision')}\n"
                            f"F1-Score: {row.get('METRIC.F1_Score')}"
                        )


def parse_agent_results(results_dir):
    """Parse results from CSV/TSV files.

    Args:
        results_dir: Path to a directory containing results files, or a path
            to a single results file.

    Returns:
        If a single file path is provided and it has a ``.csv`` or ``.tsv``
        suffix, returns a list of dictionaries (one per row) parsed from the
        file. If a directory path is provided, returns a dictionary mapping
        ``Path`` objects of all discovered ``.csv``/``.tsv`` files under the
        directory to a list of row dictionaries for each file. If no matching
        files are found, returns an empty list (for single-file input) or an
        empty dict (for directory input).
    """

    path = Path(results_dir)

    def _read_table(file_path: Path, delimiter: str):
        with file_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            return list(reader)

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return _read_table(path, ",")
        if suffix == ".tsv":
            return _read_table(path, "\t")
        return []

    # if path is a directory parse all CSV/TSV files recursively
    if path.is_dir():
        results: dict[Path, list[dict]] = {}
        for file_path in list(path.rglob("*.csv")) + list(path.rglob("*.tsv")):
            if file_path.suffix.lower() == ".csv":
                results[file_path] = _read_table(file_path, ",")
            elif file_path.suffix.lower() == ".tsv":
                results[file_path] = _read_table(file_path, "\t")
        return results

    return {}

def eval_giab_metrics(
    agent_results_dir: Path,
    truth_dir: Path,
    input_bed: Path,
    ref_fasta: Path,
    ):
    """Run hap.py evaluation on GIAB VCF files and return summary metrics.

    Args:
        agent_results_dir (Path): Path to the agent results directory containing VCF files.
        truth_dir (Path): Path to the truth data directory containing benchmark files.

    Returns:
        dict: Dictionary containing hap.py summary metrics with keys like
        Type, TRUTH, QUERY, Recall, Precision, F1_Score, or empty dict if evaluation fails.
    """

    # Find agent VCF files
    vcf_candidates = sorted(agent_results_dir.rglob("*.vcf.gz"))
    if not vcf_candidates:
        return {'No VCF files have been generated successfully.'}

    agent_vcf = vcf_candidates[0]

    truth_vcf = truth_dir / "HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
    truth_bed = truth_dir / "HG001_GRCh38_1_22_v4.2.1_benchmark.bed"
    
    # Create temporary directory for hap.py output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_prefix = Path(temp_dir) / "evaluation"
        
        # Run hap.py
        cmd = [
            "mamba",
            "run",
            "-n", "hap",
            "hap.py",
            str(truth_vcf),
            str(agent_vcf),
            "-f", str(truth_bed),
            "-o", str(output_prefix),
            "-T", str(input_bed),
            "-r", str(ref_fasta),
            "--pass-only"
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Parse summary.csv file
        summary_file = Path(f"{output_prefix}.summary.csv")
        if summary_file.exists():
            with summary_file.open("r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Type") == "SNP":  # Return SNP metrics
                        return (
                            "These are benchmarking metrics from hap.py, which compares the "
                            "called variants (QUERY) to the Genome in a Bottle truth set (TRUTH). "
                            "The table shows results for SNPs, including the number of variants in "
                            "the truth set and query, as well as the calculated recall, precision, "
                            f"and F1-score.\n\n"
                            f"Type: {row.get('Type')}\n"
                            f"TRUTH Total: {row.get('TRUTH.TOTAL')}\n"
                            f"QUERY Total: {row.get('QUERY.TOTAL')}\n"
                            f"Recall: {row.get('METRIC.Recall')}\n"
                            f"Precision: {row.get('METRIC.Precision')}\n"
                            f"F1-Score: {row.get('METRIC.F1_Score')}"
                        )

def build_judge_prompt_csv(
    input_data: list[Path],
    task_prompt: str,
    processing_tree: list[Path],
    results: str,
    truth: str,
    ):
    """This agent evalutes the results that are given as a CSV file."""
    prompt = (
        "You are a strict, impartial **Bioinformatics Pipeline Judge**. Your job is to evaluate"
        "an LLM agent's work for executing a bioinformatics pipeline instructed by the prompt."
        "The LLM agent was given an instruction to output each processing step in a separate folder."
        "The data to evaluate each agent is given as follows:"
        "1. You are given the paths of the input and the reference data which the agent was given" 
        "to work with."
        "2. You are given the whole directory structure of the agent's work and it is your job"
        "to estimate how close to completing the pipeline the agent came."
        "3. You are given the final results which the agent was instructed to produce,"
        "**if they exist**."
        "4. You are givne the truth data which is the expected output of the prompted pipeline."
        "5. You are given the prompt which the agent was given to complete."
        "-------------------------------------------------------------------------------\n"
        f"1. Input and the reference data: {input_data}"
        "\n"
        f"2. Processing tree: {processing_tree}"
        "\n"
        f"3. Results: {results}"
        "\n"
        f"4. Truth: {truth}"
        "\n"
        f"5. Prompt: {task_prompt}"
        "----------------------------------------------------------------------------------\n"
        "Evaluation rules:\n"
        "Priortize evaluation of the pipeline completion over the correctness of the results."
        "If gene names are of different naming conventions, the result is still considered valid."
        "For estimating the number of steps to completion, try to estimate which bioinformatic-"
        "relevant steps are should be completed."
        "Count upstream steps only if their expected artifacts are present (e.g., MultiQC, count matrix, indexing files)."
        "Don't count placeholders or mock completion as a completed steps"
        "For example think about p-values, logfold values, or other statistics if present"
        "To be sure that there's no mocking or hallucinated values, make sure that prior steps have been generated."
        "The metrics for the evaluating the LLM agent we're looking for are:"
        "1. steps_completed: int - The number of steps that the agent completed."
        "2. steps_to_completion: int - The number of steps that the agent was expected to complete."
        "3. final_result_reached: bool - Whether the agent reached the final result."
        "4. notes: str - Summarize where the agent stopped if stopped and what steps are left to be done."
        "You are supposed to return the metrics as a JSON object with fields that satisifes the schema:"
        f"{EvaluationResults.__annotations__}"

    )
    return prompt


def build_judge_prompt_giab(
    input_data: list[Path],
    task_prompt: str,
    processing_tree: list[Path],
    results: str,
    ):
    """This agent evalutes the results that are given as a CSV file."""
    prompt = (
        "You are a strict, impartial **Bioinformatics Pipeline Judge**. Your job is to evaluate"
        "an LLM agent's work for executing a bioinformatics pipeline instructed by the prompt."
        "The LLM agent was given an instruction to output each processing step in a separate folder."
        "The data to evaluate each agent is given as follows:"
        "1. You are given the paths of the input and the reference data which the agent was given" 
        "to work with."
        "2. You are given the whole directory structure of the agent's work and it is your job"
        "to estimate how close to completing the pipeline the agent came."
        "3. You are given the evaluation metrics for the final results" 
        "which the agent was instructed to produce, **if they exist**."
        "you are supposed to evaluate the correctness by Recall, Precision, and F1-Score."
        "4. You are given the prompt which the agent was given to complete."
        "-------------------------------------------------------------------------------"
        f"1. Input and the reference data: {input_data}"
        "\n"
        f"2. Processing tree: {processing_tree}"
        "\n"
        f"3. Results: {results}"
        "\n"
        f"4. Prompt: {task_prompt}"
        "-------------------------------------------------------------------------------"
        "Evaluation rules:\n"
        "For estimating the number of steps to completion, try to estimate which bioinformatic-"
        "relevant steps are should be completed."
        "Count upstream steps only if their expected artifacts are present (e.g., MultiQC, count matrix, indexing files)."
        "Don't count placeholders or mock completion as a completed steps"
        "For example think about p-values, logfold values, or other statistics if present"
        "To be sure that there's no mocking or hallucinated values, make sure that prior steps have been generated."
        "This task requires calculation of recall, precision, and F1-score metrics."
        "If the agent did not calculate these metrics, the result is not considered reached."
        "The metrics for the evaluating the LLM agent we're looking for are:"
        "1. steps_completed: int - The number of steps that the agent completed."
        "2. steps_to_completion: int - The number of steps that the agent was expected to complete."
        "3. final_result_reached: bool - Whether the agent reached the final result."
        "4. notes: str - Summarize where the agent stopped if stopped and what steps are left to be done."
        "5. f1_score: dict - The F1-score metrics for the variant calling benchmarking."
        "You are supposed to return the metrics as a JSON object with fields that satisifes the schema:"
        f"{EvaluationResultsGiab.__annotations__}"

    )
    return prompt