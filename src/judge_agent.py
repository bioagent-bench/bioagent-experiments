import csv
import subprocess
import tempfile
from pathlib import Path


def parse_agent_outputs(output_dir):
    """Return a list of all files and folders in the directory."""
    return list(Path(output_dir).rglob("*"))


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
        return {}

    agent_vcf = vcf_candidates[0]

    truth_vcf = truth_dir / "HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
    truth_bed = truth_dir / "HG001_GRCh38_1_22_v4.2.1_benchmark.bed"
    
    # Create temporary directory for hap.py output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_prefix = Path(temp_dir) / "evaluation"
        
        try:
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
                            return {
                                "Type": row.get("Type"),
                                "TRUTH": row.get("TRUTH.TOTAL"),
                                "QUERY": row.get("QUERY.TOTAL"),
                                "Recall": row.get("METRIC.Recall"),
                                "Precision": row.get("METRIC.Precision"),
                                "F1_Score": row.get("METRIC.F1_Score")
                            }
            
        except Exception as e:
            print(e)
    
    return {}

def build_judge_agent():
    pass

