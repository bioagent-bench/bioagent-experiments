import csv
from dataclasses import dataclass
from itertools import islice
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
    results_match: bool
    notes: str

@dataclass
class EvaluationResultsGiab:
    steps_completed: int
    steps_to_completion: int
    final_result_reached: bool
    f1_score: int
    results_match: bool
    notes: str

class EvaluationResultsSchema(BaseModel):
    """Schema for evaluation results used with structured output API.
    
    Attributes:
        steps_completed (int): Number of steps the agent completed.
        steps_to_completion (int): Number of steps needed to reach completion.
        final_result_reached (bool): Whether the agent reached the final result.
        results_match (bool): Whether the produced results satisfy the task
            verification checklist.
        notes (str): Additional evaluation notes.
    """
    steps_completed: int
    steps_to_completion: int
    final_result_reached: bool
    results_match: bool
    notes: str

class EvaluationResultsGiabSchema(BaseModel):
    """Schema for GIAB evaluation results used with structured output API.
    
    Attributes:
        steps_completed (int): Number of steps the agent completed.
        steps_to_completion (int): Number of steps needed to reach completion.
        final_result_reached (bool): Whether the agent reached the final results.
        f1_score (int): F1 score for the evaluation.
        results_match (bool): Whether the GIAB run cleared the acceptance
            threshold.
        notes (str): Additional evaluation notes.
    """
    steps_completed: int
    steps_to_completion: int
    final_result_reached: bool
    f1_score: int
    results_match: bool
    notes: str


RESULTS_MATCH_RULES: dict[str, str] = {
    "alzheimer-mouse": (
        "- Task rule: mark `results_match` true only if the agent's CSV shares at least one entry "
        "in the `Pathway` column with the truth file."
    ),
    "comparative-genomics": (
        "- Task rule: set `results_match` true only when at least one `consensus_annotation` string "
        "exactly matches between the agent output and the truth results."
    ),
    "cystic-fibrosis": (
        "- Task rule: set `results_match` true only if the causal CFTR variant is reported exactly "
        "once in the final CSV with chromosome 7, position 117227832, variant_id 7115, reference G, "
        "alternate T, gene CFTR, gene_id ENSG00000001626, annotation stop_gained, impact HIGH, and "
        "transcript ENST00000003084. Any missing field keeps the flag false."
    ),
    "deseq": (
        "- Task rule: require at least five overlapping `gene_id` values between the agent CSV and "
        "the truth DESeq output before returning `results_match=true`."
    ),
    "evolution": (
        "- Task rule: set `results_match` true only if there is at least one variant where both "
        "chromosome/CHROM and position/POS match the truth set."
    ),
    "giab": (
        "- Task rule: set `results_match` true only if the hap.py SNP F1-score exceeds 0.7. "
        "Otherwise it must be false."
    ),
    "metagenomics": (
        "- Task rule: require that the most abundant phylum reported by the agent is "
        "`Pseudomonadota` (consider JP4D/JC1A abundance columns) and that at least two additional "
        "OTUs share the same `Phylum` labels as rows in the truth data. If either condition fails, "
        "return `results_match=false`."
    ),
    "single-cell": (
        "- Task rule: mark `results_match` true only if at least one `(cluster_id, "
        "predicted_cell_type)` pair from the truth CSV is also present in the agent's output. "
        "Cluster numbering can be permuted, but the evidence must show a real match (e.g., cluster 0 "
        "labeled as 'Endothelial cell')."
    ),
    "transcript-quant": (
        "- Task rule: set `results_match` true only when the agent's TSV contains an identical set "
        "of `transcript_id` → `count` pairs as the truth TSV (order/formatting may differ)."
    ),
    "viral-metagenomics": (
        "- Task rule: the agent must explicitly report the species 'Bottlenose dolphin adenovirus 1' "
        "within the Viruses domain. Without that virus, `results_match` stays false."
    ),
}


def _build_results_match_guidance(task_id: str) -> str:
    """Return task-specific instructions for the results_match field."""
    base = (
        "\nSpecial verification flag `results_match` must always be provided as a boolean "
        "(true/false). Mark it true only when the evidence in the provided results and truth "
        "artifacts satisfies the applicable rule. When in doubt, default to false."
    )
    rule = RESULTS_MATCH_RULES.get(task_id)
    return f"{base}\n{rule}"


def parse_agent_outputs(output_dir: Path) -> list[Path]:
    """Return outputs relative to ``output_dir`` up to two levels deep, skipping
    ``.snakemake`` data.
    If for example another software is installed it might create large nesting 
    of installation files which fill up the context of the LLM.

    Args:
        output_dir (Path): Directory whose descendants should be returned.

    Returns:
        list[Path]: Relative file and directory paths found under ``output_dir``,
            excluding anything nested inside a ``.snakemake`` directory.
    """
    root = Path(output_dir)
    outputs: list[Path] = []
    for path in root.rglob("*"):
        relative_path = path.relative_to(root)
        if ".snakemake" in relative_path.parts:
            continue
        if len(relative_path.parts) > 2:
            continue
        outputs.append(relative_path)
    return outputs


def parse_agent_results(results_dir: Path | str, max_rows: int = 100):
    """Parse results from CSV/TSV files, returning only a leading snippet.

    Args:
        results_dir (Path | str): Directory containing results files or a single
            results file path.
        max_rows (int): Maximum number of rows to read from each table in order
            to keep the payload concise.

    Returns:
        dict: If a single file path is provided, returns a dictionary with keys
            ``rows`` (list of dict rows), ``snippet_note`` (str), ``truncated``
            (bool), and ``display_limit`` (int). If a directory path is
            provided, returns a dictionary mapping each discovered ``Path`` to
            the aforementioned metadata dictionary. If no matching files are
            found, returns an empty dict.
    """

    path = Path(results_dir)
    def _read_table(file_path: Path, delimiter: str) -> dict:
        with file_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            rows = list(islice(reader, max_rows))
            truncated = next(reader, None) is not None

        row_count = len(rows)
        snippet_note = (
            f"Showing only the first {max_rows} rows from {file_path.name} "
            "to provide a concise snippet because the file contains additional rows."
            if truncated
            else (
                f"File {file_path.name} has {row_count} rows (<= {max_rows}); "
                "showing full contents."
            )
        )
        return {
            "path": file_path,
            "rows": rows,
            "snippet_note": snippet_note,
            "truncated": truncated,
            "display_limit": max_rows,
            "displayed_rows": row_count,
        }

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return _read_table(path, ",")
        if suffix == ".tsv":
            return _read_table(path, "\t")
        return {}

    if path.is_dir():
        results: dict[Path, dict] = {}
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
    reference_data: list[Path],
    task_prompt: str,
    processing_tree: list[Path],
    results: str,
    truth: str,
    task_id: str,
    ):
    """Build the judging prompt for CSV-based tasks."""
    results_match_guidance = _build_results_match_guidance(task_id)
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
        f"1. Input data: {input_data}"
        "\n"
        f"2. Reference data: {reference_data}"
        "\n"
        f"3. Processing tree: {processing_tree}"
        "\n"
        f"4. Results: {results}"
        "\n"
        f"5. Truth: {truth}"
        "\n"
        f"6. Prompt: {task_prompt}"
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
        f"{results_match_guidance}\n"
        "The metrics for the evaluating the LLM agent we're looking for are:"
        "1. steps_completed: int - The number of steps that the agent completed."
        "2. steps_to_completion: int - The number of steps that the agent was expected to complete."
        "3. final_result_reached: bool - Whether the agent reached the final result."
        "4. notes: str - Summarize where the agent stopped if stopped and what steps are left to be done."
        "5. results_match: bool - Set to true/false (or 1/0) per the rule above."
        "You are supposed to return the metrics as a JSON object with fields that satisifes the schema:"
        f"{EvaluationResults.__annotations__}"

    )
    return prompt


def build_judge_prompt_giab(
    input_data: list[Path],
    reference_data: list[Path],
    task_prompt: str,
    processing_tree: list[Path],
    results: str,
    task_id: str,
    ):
    """Build the judging prompt for GIAB-style tasks."""
    results_match_guidance = _build_results_match_guidance(task_id)
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
        f"2. Reference data: {reference_data}"  
        "\n"
        f"3. Processing tree: {processing_tree}"
        "\n"
        f"4. Results: {results}"
        "\n"
        f"5. Prompt: {task_prompt}"
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
        f"{results_match_guidance}\n"
        "The metrics for the evaluating the LLM agent we're looking for are:"
        "1. steps_completed: int - The number of steps that the agent completed."
        "2. steps_to_completion: int - The number of steps that the agent was expected to complete."
        "3. final_result_reached: bool - Whether the agent reached the final result."
        "4. notes: str - Summarize where the agent stopped if stopped and what steps are left to be done."
        "5. f1_score: dict - The F1-score metrics for the variant calling benchmarking."
        "6. results_match: bool - Set to true/false (or 1/0) per the rule above."
        "You are supposed to return the metrics as a JSON object with fields that satisifes the schema:"
        f"{EvaluationResultsGiab.__annotations__}"

    )
    return prompt