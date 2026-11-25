"""Deterministic result scoring for BioAgent tasks."""

from __future__ import annotations

import csv
import io
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import verifiers.v1 as vf

DELIMITERS = {".csv": ",", ".tsv": "\t"}

# NCBI TaxID 1569362: pre-2023 name, Kaiju 2024 / pre-binomial ICTV name, and
# ratified ICTV Latinized binomial species name.
_VIRAL_TARGET_SPECIES = frozenset(
    {
        "bottlenose dolphin adenovirus 1",
        "dolphin mastadenovirus b",
        "mastadenovirus delphini",
    }
)
_UNSPECIFIED_SPECIES = frozenset({"", "unclassified", "unknown", "na", "n/a", "none"})

# Common legacy-to-modern NCBI bacterial phylum synonyms.
_PHYLUM_SYNONYMS = {
    "acidobacteria": "acidobacteriota",
    "actinobacteria": "actinomycetota",
    "bacteroidetes": "bacteroidota",
    "chloroflexi": "chloroflexota",
    "cyanobacteria": "cyanobacteriota",
    "firmicutes": "bacillota",
    "fusobacteria": "fusobacteriota",
    "planctomycetes": "planctomycetota",
    "proteobacteria": "pseudomonadota",
    "spirochaetes": "spirochaetota",
    "verrucomicrobia": "verrucomicrobiota",
}

_SPADES_CONTIG_RE = re.compile(r"^node_\d+_length_(\d+)_cov_\d+(?:\.\d+)?(?:e[+-]?\d+)?$")

# Quantitative tolerances for EM isoform quantification and phylum abundance comparisons.
_METAGENOMICS_MIN_PEARSON = 0.90
_METAGENOMICS_MAX_REL_L1 = 0.40
_TRANSCRIPT_MIN_KEY_RECALL = 0.90
_TRANSCRIPT_MIN_KEY_PRECISION = 0.90
_TRANSCRIPT_MAX_TOTAL_REL_ERR = 0.05
_TRANSCRIPT_MAX_REL_L1 = 0.10
_TRANSCRIPT_MIN_PEARSON = 0.98


def _normalized(row: dict[str, str]) -> dict[str, str]:
    return {
        key.strip().casefold(): (value or "").strip()
        for key, value in row.items()
        if key is not None
    }


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
    if rows and [cell.strip().casefold() for cell in rows[0][:2]] == ["transcript_id", "count"]:
        rows = rows[1:]
    return {row[0].strip(): row[1].strip() for row in rows if len(row) >= 2 and row[0].strip()}


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


def _parse_non_negative_decimal(value: str) -> Decimal | None:
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    if not number.is_finite() or number < 0:
        return None
    return number


def _parse_non_negative_float(value: str) -> float | None:
    if not value:
        return 0.0
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _pearson_r(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0.0 or denom_y == 0.0:
        return None
    return num / (denom_x * denom_y)


def _numbers_equal(left: str, right: str) -> bool:
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return False


def _transcript_quant_match(counts: dict[str, str], truth_counts: dict[str, str]) -> bool:
    parsed_pred: dict[str, Decimal] = {}
    for key, raw in counts.items():
        val = _parse_non_negative_decimal(raw)
        if val is None:
            return False
        if val > 0:
            parsed_pred[key] = val

    parsed_truth: dict[str, Decimal] = {}
    for key, raw in truth_counts.items():
        val = _parse_non_negative_decimal(raw)
        if val is None:
            return False
        if val > 0:
            parsed_truth[key] = val

    if not parsed_truth:
        return not parsed_pred
    if not parsed_pred:
        return False
    if parsed_pred == parsed_truth:
        return True

    shared = set(parsed_pred) & set(parsed_truth)
    if len(shared) / len(parsed_truth) < _TRANSCRIPT_MIN_KEY_RECALL:
        return False
    if len(shared) / len(parsed_pred) < _TRANSCRIPT_MIN_KEY_PRECISION:
        return False

    union_keys = sorted(set(parsed_pred) | set(parsed_truth))
    xs = [float(parsed_pred.get(key, 0)) for key in union_keys]
    ys = [float(parsed_truth.get(key, 0)) for key in union_keys]
    pred_total = sum(xs)
    truth_total = sum(ys)
    if abs(pred_total - truth_total) / truth_total > _TRANSCRIPT_MAX_TOTAL_REL_ERR:
        return False
    if sum(abs(x - y) for x, y in zip(xs, ys)) / truth_total > _TRANSCRIPT_MAX_REL_L1:
        return False
    if len(union_keys) >= 2 and any(y != ys[0] for y in ys):
        corr = _pearson_r(xs, ys)
        if corr is None or corr < _TRANSCRIPT_MIN_PEARSON:
            return False
    return True


def _variant_key(row: dict[str, str]) -> tuple[str, str]:
    chromosome = _value(row, "chrom", "chromosome", "CHROM").casefold().removeprefix("chr")
    position = _value(row, "pos", "position", "POS")
    return chromosome, position


def _variant_keys(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {key for row in rows if all(key := _variant_key(row))}


def _canonical_chrom(chrom: str) -> str:
    normalized = chrom.casefold().removeprefix("chr")
    if match := _SPADES_CONTIG_RE.match(normalized):
        return f"spades_length_{match.group(1)}"
    return normalized


def _normalize_alleles(ref: str, alt: str) -> tuple[str, str]:
    ref_norm, alt_norm = ref.upper(), alt.upper()
    while len(ref_norm) > 1 and len(alt_norm) > 1 and ref_norm[-1] == alt_norm[-1]:
        ref_norm = ref_norm[:-1]
        alt_norm = alt_norm[:-1]
    return ref_norm, alt_norm


def _variant_alleles(row: dict[str, str]) -> tuple[str, str] | None:
    ref = _value(row, "ref", "reference", "REF")
    alt = _value(row, "alt", "alternate", "ALT")
    if not ref or not alt:
        return None
    return _normalize_alleles(ref, alt)


def _evolution_match(generated: list[dict[str, str]], truth: list[dict[str, str]]) -> bool:
    truth_exact: dict[tuple[str, str], set[tuple[str, str] | None]] = {}
    truth_canonical: dict[tuple[str, str], set[tuple[str, str] | None]] = {}
    for row in truth:
        raw_key = _variant_key(row)
        if not all(raw_key):
            continue
        alleles = _variant_alleles(row)
        truth_exact.setdefault(raw_key, set()).add(alleles)
        canonical_key = (_canonical_chrom(raw_key[0]), raw_key[1])
        truth_canonical.setdefault(canonical_key, set()).add(alleles)

    for row in generated:
        raw_key = _variant_key(row)
        if not all(raw_key):
            continue
        alleles = _variant_alleles(row)
        if raw_key in truth_exact:
            expected = truth_exact[raw_key]
            if alleles is None or None in expected or alleles in expected:
                return True
        canonical_key = (_canonical_chrom(raw_key[0]), raw_key[1])
        if canonical_key in truth_canonical:
            expected = truth_canonical[canonical_key]
            if alleles is None:
                if None in expected:
                    return True
            elif alleles in expected or None in expected:
                return True
    return False


def _values(rows: list[dict[str, str]], *names: str) -> set[str]:
    return {value for row in rows if (value := _value(row, *names))}


def _canonical_phylum(name: str) -> str:
    key = " ".join(name.split()).casefold()
    return _PHYLUM_SYNONYMS.get(key, key)


def _metagenomics_match(generated: list[dict[str, str]], truth: list[dict[str, str]]) -> bool:
    if not generated:
        return False

    gen_by_phylum: dict[str, tuple[float, float]] = {}
    for row in generated:
        jp4d = _parse_non_negative_float(_value(row, "JP4D"))
        jc1a = _parse_non_negative_float(_value(row, "JC1A"))
        if jp4d is None or jc1a is None:
            return False
        phylum = _canonical_phylum(_value(row, "Phylum"))
        if phylum:
            prev_jp, prev_jc = gen_by_phylum.get(phylum, (0.0, 0.0))
            gen_by_phylum[phylum] = (prev_jp + jp4d, prev_jc + jc1a)

    if not gen_by_phylum:
        return False
    top_phylum = max(gen_by_phylum, key=lambda name: sum(gen_by_phylum[name]))
    if sum(gen_by_phylum[top_phylum]) <= 0 or top_phylum != "pseudomonadota":
        return False

    truth_by_phylum: dict[str, tuple[float, float]] = {}
    for row in truth:
        jp4d = _parse_non_negative_float(_value(row, "JP4D")) or 0.0
        jc1a = _parse_non_negative_float(_value(row, "JC1A")) or 0.0
        phylum = _canonical_phylum(_value(row, "Phylum"))
        if phylum:
            prev_jp, prev_jc = truth_by_phylum.get(phylum, (0.0, 0.0))
            truth_by_phylum[phylum] = (prev_jp + jp4d, prev_jc + jc1a)

    # Preserve exact (OTU, Phylum) overlap check (e.g., phyloseq::tax_glom leaf TaxIDs),
    # while also accepting canonical NCBI Phylum TaxIDs / phylum-name matches when positive.
    truth_pairs = {
        (_value(row, "OTU"), _canonical_phylum(_value(row, "Phylum")))
        for row in truth
        if _value(row, "OTU") and _value(row, "Phylum")
    }
    exact_non_top = {
        (_value(row, "OTU"), _canonical_phylum(_value(row, "Phylum")))
        for row in generated
        if _canonical_phylum(_value(row, "Phylum")) != top_phylum
        and (_value(row, "OTU"), _canonical_phylum(_value(row, "Phylum"))) in truth_pairs
    }
    phylum_non_top = {
        phylum
        for phylum, (jp4d, jc1a) in gen_by_phylum.items()
        if phylum != top_phylum and phylum in truth_by_phylum and (jp4d + jc1a) > 0
    }
    if len(exact_non_top) < 2 and len(phylum_non_top) < 2:
        return False

    # Validate relative abundance concordance across JP4D and JC1A whenever truth provides values.
    union_phyla = sorted(set(gen_by_phylum) | set(truth_by_phylum))
    for sample_idx in (0, 1):
        ys = [truth_by_phylum.get(phylum, (0.0, 0.0))[sample_idx] for phylum in union_phyla]
        truth_sum = sum(ys)
        if truth_sum <= 0:
            continue
        xs = [gen_by_phylum.get(phylum, (0.0, 0.0))[sample_idx] for phylum in union_phyla]
        gen_sum = sum(xs)
        if gen_sum <= 0:
            return False
        x_rel = [x / gen_sum for x in xs]
        y_rel = [y / truth_sum for y in ys]
        if sum(abs(x - y) for x, y in zip(x_rel, y_rel)) > _METAGENOMICS_MAX_REL_L1:
            return False
        if len(union_phyla) >= 2 and any(y != y_rel[0] for y in y_rel):
            corr = _pearson_r(x_rel, y_rel)
            if corr is None or corr < _METAGENOMICS_MIN_PEARSON:
                return False

    return True


def _viral_metagenomics_match(generated: list[dict[str, str]]) -> bool:
    target_counts: list[float | None] = []
    other_named_counts: list[float] = []
    for row in generated:
        if _value(row, "domain").casefold() != "viruses":
            continue
        species = " ".join(_value(row, "species").split()).casefold()
        count_raw = _value(row, "contig_count", "count")
        count: float | None = None
        if count_raw:
            count = _parse_non_negative_float(count_raw)
            if count is None:
                continue
        if species in _VIRAL_TARGET_SPECIES:
            if count is not None and count <= 0:
                continue
            target_counts.append(count)
        elif species not in _UNSPECIFIED_SPECIES and count is not None:
            other_named_counts.append(count)

    if not target_counts:
        return False
    positive_targets = [count for count in target_counts if count is not None]
    if positive_targets and other_named_counts:
        return max(positive_targets) >= max(other_named_counts)
    return True


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
        return _transcript_quant_match(counts, truth_counts)

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
        return _evolution_match(generated, truth)
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
        return _viral_metagenomics_match(generated)
    raise ValueError(f"No result scorer for {task_id!r}")
