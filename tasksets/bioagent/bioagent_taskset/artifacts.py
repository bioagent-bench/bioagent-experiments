"""Runtime artifact collection and bounded judge evidence."""

from __future__ import annotations

import asyncio
import csv
import shlex
import subprocess
import tempfile
from pathlib import Path

import verifiers.v1 as vf

TABLE_SUFFIXES = {".csv": ",", ".tsv": "\t"}


async def runtime_files(runtime: vf.Runtime, *roots: str) -> list[str]:
    command = (
        "find "
        + " ".join(shlex.quote(root) for root in roots)
        + " -maxdepth 3 -path '*/.snakemake/*' -prune -o -type f -print"
    )
    result = await runtime.run(["sh", "-c", command], {})
    if result.exit_code != 0:
        raise RuntimeError(result.stderr)
    return sorted(line for line in result.stdout.splitlines() if line)


async def export_results(
    runtime: vf.Runtime,
    files: list[str],
    destination: Path,
) -> None:
    for relative in (path for path in files if path.startswith("results/")):
        target = destination / relative
        data = await runtime.read(relative)
        await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)


async def runtime_table_snippets(
    runtime: vf.Runtime,
    files: list[str],
    max_rows: int = 100,
) -> dict[str, list[dict[str, str]]]:
    snippets: dict[str, list[dict[str, str]]] = {}
    for path in files:
        delimiter = TABLE_SUFFIXES.get(Path(path).suffix.lower())
        if delimiter is None or not path.startswith("results/"):
            continue
        result = await runtime.run(["head", "-n", str(max_rows + 1), path], {})
        if result.exit_code != 0:
            raise RuntimeError(result.stderr)
        snippets[path] = list(csv.DictReader(result.stdout.splitlines(), delimiter=delimiter))
    return snippets


def local_table_snippets(root: Path, max_rows: int = 100) -> dict[str, list[dict[str, str]]]:
    snippets: dict[str, list[dict[str, str]]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        delimiter = TABLE_SUFFIXES.get(path.suffix.lower())
        if delimiter is None:
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            snippets[str(path.relative_to(root))] = [
                row for _, row in zip(range(max_rows), csv.DictReader(handle, delimiter=delimiter))
            ]
    return snippets


def list_local_files(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def _giab_metrics(artifact_dir: Path, source_dir: Path) -> dict[str, float | int | str]:
    queries = list((artifact_dir / "results").rglob("*.vcf.gz"))
    if not queries:
        return {"status": "missing_query_vcf", "f1_score": 0.0}
    query = queries[0]
    truth = source_dir / "results" / "HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
    confident = source_dir / "results" / "HG001_GRCh38_1_22_v4.2.1_benchmark.bed"
    regions = source_dir / "data" / "Agilent_v7.chr.bed"
    reference_source = source_dir / "reference" / "Homo_sapiens_assembly38.fasta"

    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        reference = workdir / reference_source.name
        reference.symlink_to(reference_source)
        subprocess.run(
            ["mamba", "run", "-n", "hap", "samtools", "faidx", str(reference)],
            check=True,
            capture_output=True,
        )
        if not Path(f"{query}.tbi").exists():
            subprocess.run(
                ["mamba", "run", "-n", "hap", "bcftools", "index", "-t", str(query)],
                check=True,
                capture_output=True,
            )
        output = workdir / "evaluation"
        subprocess.run(
            [
                "mamba",
                "run",
                "-n",
                "hap",
                "hap.py",
                str(truth),
                str(query),
                "-f",
                str(confident),
                "-o",
                str(output),
                "-T",
                str(regions),
                "-r",
                str(reference),
                "--pass-only",
            ],
            check=True,
            capture_output=True,
        )
        with Path(f"{output}.summary.csv").open(encoding="utf-8", newline="") as handle:
            row = next(row for row in csv.DictReader(handle) if row["Type"] == "SNP")
        return {
            "type": "SNP",
            "truth_total": int(row["TRUTH.TOTAL"]),
            "query_total": int(row["QUERY.TOTAL"]),
            "recall": float(row["METRIC.Recall"]),
            "precision": float(row["METRIC.Precision"]),
            "f1_score": float(row["METRIC.F1_Score"]),
        }


async def benchmark_metrics(
    task_id: str,
    artifact_dir: Path,
    source_dir: Path,
) -> dict[str, float | int | str]:
    if task_id != "giab":
        return {}
    try:
        return await asyncio.to_thread(_giab_metrics, artifact_dir, source_dir)
    except subprocess.CalledProcessError as error:
        return {
            "status": "hap.py_failed",
            "f1_score": 0.0,
            "exit_code": error.returncode,
        }
