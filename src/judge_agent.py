import csv
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen
from urllib.error import URLError, HTTPError


def parse_outputs(output_dir):
    """Return a list of all files and folders in the directory."""
    return list(Path(output_dir).rglob("*"))


def parse_results(results_dir):
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

def eval_giab_results(results_dir):
    """Locate GIAB VCF and index files under a results directory.

    Args:
        results_dir (str | Path): Path to the results directory to scan.

    Returns:
        dict[str, Path]: Dictionary with keys ``"vcf_gz"`` and ``"vcf_gz_tbi"``
        pointing to the ``.vcf.gz`` file and its corresponding ``.vcf.gz.tbi``
        tabix index file. Returns an empty dict if either file is not found.
    """

    base_path = Path(results_dir)

    if not base_path.exists() or not base_path.is_dir():
        return {}

    vcf_candidates = sorted(base_path.rglob("*.vcf.gz"))
    tbi_candidates = sorted(base_path.rglob("*.vcf.gz.tbi"))

    if not vcf_candidates or not tbi_candidates:
        return {}

    # Prefer a matching .tbi for a given .vcf.gz if available
    for vcf_path in vcf_candidates:
        tbi_path = Path(f"{vcf_path}.tbi")
        if tbi_path.exists():
            return {"vcf_gz": vcf_path, "vcf_gz_tbi": tbi_path}

    # Fallback to first found of each type
    return {"vcf_gz": vcf_candidates[0], "vcf_gz_tbi": tbi_candidates[0]}

def build_judge_agent():
    pass

