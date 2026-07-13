"""Deterministic result scoring for BioAgent tasks."""

from __future__ import annotations

import csv
import io
from decimal import Decimal, InvalidOperation
from pathlib import Path

import verifiers.v1 as vf

DELIMITERS = {".csv": ",", ".tsv": "\t"}


def _normalized(row: dict[str, str]) -> dict[str, str]:
    return {key.strip().casefold(): (value or "").strip() for key, value in row.items()}


def _value(row: dict[str, str], *names: str) -> str:
    normalized = _normalized(row)
    return next(
        (normalized[name.casefold()] for name in names if name.casefold() in normalized), ""
    )


def _local_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(item for item in root.rglob("*") if item.suffix.lower() in DELIMITERS):
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle, delimiter=DELIMITERS[path.suffix.lower()]))
    return rows


def _transcript_counts(text: str, delimiter: str) -> dict[str, str]:
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    if rows and rows[0][:2] == ["transcript_id", "count"]:
        rows = rows[1:]
    return {row[0].strip(): row[1].strip() for row in rows if len(row) >= 2}


def _local_transcript_counts(root: Path) -> dict[str, str]:
    counts: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.suffix.lower() in DELIMITERS):
        counts.update(
            _transcript_counts(
                path.read_text(encoding="utf-8"),
                DELIMITERS[path.suffix.lower()],
            )
        )
    return counts


async def _runtime_transcript_counts(runtime: vf.Runtime, files: list[str]) -> dict[str, str]:
    counts: dict[str, str] = {}
    for path in files:
        delimiter = DELIMITERS.get(Path(path).suffix.lower())
        if delimiter is None or not path.startswith("results/"):
            continue
        counts.update(
            _transcript_counts(
                (await runtime.read(path)).decode("utf-8", errors="replace"),
                delimiter,
            )
        )
    return counts


async def _runtime_rows(runtime: vf.Runtime, files: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in files:
        delimiter = DELIMITERS.get(Path(path).suffix.lower())
        if delimiter is None or not path.startswith("results/"):
            continue
        text = (await runtime.read(path)).decode("utf-8", errors="replace")
        rows.extend(csv.DictReader(io.StringIO(text), delimiter=delimiter))
    return rows


def _numbers_equal(left: str, right: str) -> bool:
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return False


def _variant_key(row: dict[str, str]) -> tuple[str, str]:
    chromosome = _value(row, "chrom", "chromosome", "CHROM").casefold().removeprefix("chr")
    position = _value(row, "pos", "position", "POS")
    return chromosome, position


def _variant_keys(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {key for row in rows if all(key := _variant_key(row))}


def _values(rows: list[dict[str, str]], *names: str) -> set[str]:
    return {value for row in rows if (value := _value(row, *names))}


def _metagenomics_match(generated: list[dict[str, str]], truth: list[dict[str, str]]) -> bool:
    def abundance(row: dict[str, str]) -> float:
        return float(_value(row, "JP4D") or 0) + float(_value(row, "JC1A") or 0)

    if not generated:
        return False
    top = max(generated, key=abundance)
    if _value(top, "Phylum").casefold() != "pseudomonadota":
        return False
    truth_pairs = {(_value(row, "OTU"), _value(row, "Phylum")) for row in truth}
    top_pair = (_value(top, "OTU"), _value(top, "Phylum"))
    overlaps = {
        (_value(row, "OTU"), _value(row, "Phylum"))
        for row in generated
        if (_value(row, "OTU"), _value(row, "Phylum")) in truth_pairs
    }
    return len(overlaps - {top_pair}) >= 2


async def score_results(
    task_id: str,
    runtime: vf.Runtime,
    files: list[str],
    truth_dir: Path,
    benchmark: dict[str, float | int | str],
) -> bool:
    if task_id == "transcript-quant":
        counts = await _runtime_transcript_counts(runtime, files)
        truth_counts = _local_transcript_counts(truth_dir)
        return counts.keys() == truth_counts.keys() and all(
            _numbers_equal(counts[key], truth_counts[key]) for key in counts
        )

    generated = await _runtime_rows(runtime, files)
    truth = _local_rows(truth_dir)

    if task_id == "alzheimer-mouse":
        return bool(_values(generated, "Pathway") & _values(truth, "Pathway"))
    if task_id == "comparative-genomics":
        return bool(
            _values(generated, "consensus_annotation") & _values(truth, "consensus_annotation")
        )
    if task_id == "cystic-fibrosis":
        required = {
            "chromosome": "7",
            "position": "117227832",
            "variant_id": "7115",
            "reference": "G",
            "alternate": "T",
            "gene_name": "CFTR",
            "gene_id": "ENSG00000001626",
            "annotation": "stop_gained",
            "impact": "HIGH",
            "transcript_id": "ENST00000003084",
        }
        aliases = {
            "chromosome": ("chromosome", "chrom", "CHROM"),
            "position": ("position", "pos", "POS"),
            "gene_name": ("gene_name", "gene"),
            "transcript_id": ("transcript_id", "transcript"),
        }
        matches = [
            row
            for row in generated
            if all(
                _value(row, *aliases.get(field, (field,))).casefold() == expected.casefold()
                for field, expected in required.items()
            )
        ]
        return len(matches) == 1
    if task_id == "deseq":
        return len(_values(generated, "gene_id") & _values(truth, "gene_id")) >= 5
    if task_id == "evolution":
        return bool(_variant_keys(generated) & _variant_keys(truth))
    if task_id == "giab":
        return float(benchmark.get("f1_score", 0.0)) > 0.7
    if task_id == "metagenomics":
        return _metagenomics_match(generated, truth)
    if task_id == "single-cell":
        pairs = {
            (_value(row, "cluster_id"), _value(row, "predicted_cell_type"))
            for row in generated
            if _value(row, "cluster_id") and _value(row, "predicted_cell_type")
        }
        truth_pairs = {
            (_value(row, "cluster_id"), _value(row, "predicted_cell_type"))
            for row in truth
            if _value(row, "cluster_id") and _value(row, "predicted_cell_type")
        }
        return bool(pairs & truth_pairs)
    if task_id == "viral-metagenomics":
        return any(
            _value(row, "domain").casefold() == "viruses"
            and _value(row, "species").casefold() == "bottlenose dolphin adenovirus 1"
            for row in generated
        )
    raise ValueError(f"No result scorer for {task_id!r}")
