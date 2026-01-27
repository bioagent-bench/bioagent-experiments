from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = BASE_DIR / "stability-results"


@dataclass(frozen=True)
class TaskSpec:
    folder: str
    patterns: tuple[str, ...]
    id_candidates: tuple[tuple[str, ...], ...]
    value_cols: tuple[str, ...] | None = None
    exclude_value_cols: tuple[str, ...] = ()
    value_agg: str = "mean"


TASK_SPECS: dict[str, TaskSpec] = {
    "alzheimer": TaskSpec(
        folder="alzheimer",
        patterns=("shared_kegg_pathways*.csv",),
        id_candidates=(("pathway",),),
        value_cols=("5xFAD_pvalue", "3xTG_AD_pvalue", "PS3O1S_pvalue"),
    ),
    "comparative": TaskSpec(
        folder="comparative",
        patterns=("core_cogs_consensus*.csv", "universal_clusters_consensus.csv"),
        id_candidates=(("consensus_annotation",), ("cluster_number",)),
        value_cols=(),
    ),
    "cystic-fibrosis": TaskSpec(
        folder="cystic-fibrosis",
        patterns=("cystic_fibrosis_causal_variant*.csv",),
        id_candidates=(("chromosome", "position", "reference", "alternate"), ("variant_id",)),
        value_cols=(),
    ),
    "evolution": TaskSpec(
        folder="evolution",
        patterns=("shared_*variants*.csv",),
        id_candidates=(("chrom", "pos", "ref", "alt"),),
        value_cols=(),
    ),
    "metagenomics": TaskSpec(
        folder="metagenomics",
        patterns=("*.csv",),
        id_candidates=(("Phylum",), ("OTU",)),
        exclude_value_cols=("OTU", "Kingdom", "Phylum"),
        value_agg="sum",
    ),
    "single-cell": TaskSpec(
        folder="single-cell",
        patterns=("*.csv",),
        id_candidates=(
            ("predicted_cell_type", "gene_name"),
            ("cluster_id", "gene_name"),
            ("gene_name",),
        ),
        value_cols=("logfoldchanges", "abs_logfc", "pvals_adj", "pvals"),
    ),
    "transcript-quant": TaskSpec(
        folder="transcript-quant",
        patterns=("transcript_counts*.tsv",),
        id_candidates=(("transcript_id",),),
        value_cols=("count",),
    ),
    "viral-metagenomics": TaskSpec(
        folder="viral-metagenomics",
        patterns=("*.csv",),
        id_candidates=(("domain", "species"),),
        value_cols=("contig_count",),
    ),
    "deseq": TaskSpec(
        folder="deseq",
        patterns=("differential_expression*.csv",),
        id_candidates=(("gene_id",),),
        value_cols=("log2FoldChange", "padj", "pvalue", "baseMean"),
    ),
}


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def pick_id_cols(df: pd.DataFrame, candidates: Iterable[Iterable[str]]) -> list[str] | None:
    for cols in candidates:
        if all(col in df.columns for col in cols):
            return list(cols)
    return None


def fallback_id_cols(df: pd.DataFrame) -> list[str]:
    priority = [
        "gene_id",
        "transcript_id",
        "pathway",
        "variant_id",
        "gene_name",
        "OTU",
        "Phylum",
        "species",
        "cluster_number",
    ]
    for col in priority:
        if col in df.columns:
            return [col]
    return [df.columns[0]]


def build_id_series(df: pd.DataFrame, id_cols: list[str]) -> pd.Series:
    id_df = df[id_cols].fillna("").astype(str)
    if len(id_cols) == 1:
        return id_df[id_cols[0]]
    return id_df.agg("|".join, axis=1)


def infer_value_cols(
    df: pd.DataFrame,
    id_cols: list[str],
    explicit_cols: tuple[str, ...] | None,
    exclude_cols: Iterable[str],
) -> list[str]:
    if explicit_cols is not None:
        return [col for col in explicit_cols if col in df.columns]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    excluded = set(id_cols) | set(exclude_cols)
    return [col for col in numeric_cols if col not in excluded]


def build_value_table(
    df: pd.DataFrame, id_series: pd.Series, value_cols: list[str], agg: str
) -> pd.DataFrame | None:
    if not value_cols:
        return None
    values = df[value_cols].apply(pd.to_numeric, errors="coerce")
    values = values.join(id_series.rename("_id"))
    grouped = values.groupby("_id")
    if agg == "mean":
        return grouped.mean(numeric_only=True)
    if agg == "sum":
        return grouped.sum(numeric_only=True)
    return grouped.agg(agg)


def mean_or_nan(values: Iterable[float]) -> float:
    cleaned = [v for v in values if v is not None and not np.isnan(v)]
    if not cleaned:
        return float("nan")
    return float(np.mean(cleaned))


def load_runs(results_dir: Path, spec: TaskSpec) -> list[dict]:
    task_dir = results_dir / spec.folder
    files = []
    for pattern in spec.patterns:
        files.extend(task_dir.glob(pattern))
    runs = []
    for path in sorted(set(files)):
        df = read_table(path)
        id_cols = pick_id_cols(df, spec.id_candidates) or fallback_id_cols(df)
        id_series = build_id_series(df, id_cols)
        value_cols = infer_value_cols(df, id_cols, spec.value_cols, spec.exclude_value_cols)
        values = build_value_table(df, id_series, value_cols, spec.value_agg)
        runs.append(
            {
                "path": path,
                "id_cols": id_cols,
                "value_cols": value_cols,
                "ids": set(id_series.dropna().unique()),
                "values": values,
            }
        )
    return runs


def pairwise_jaccard(run_a: dict, run_b: dict) -> float:
    union = run_a["ids"] | run_b["ids"]
    shared = run_a["ids"] & run_b["ids"]
    return len(shared) / len(union) if union else float("nan")


def pairwise_value_corr(run_a: dict, run_b: dict, shared_ids: set[str]) -> float:
    if run_a["values"] is None or run_b["values"] is None or not shared_ids:
        return float("nan")
    cols = [col for col in run_a["values"].columns if col in run_b["values"].columns]
    if not cols:
        return float("nan")
    corrs = []
    shared_list = list(shared_ids)
    for col in cols:
        a = run_a["values"].loc[run_a["values"].index.intersection(shared_list), col]
        b = run_b["values"].loc[run_b["values"].index.intersection(shared_list), col]
        both = pd.concat([a, b], axis=1).dropna()
        if len(both) < 2:
            continue
        corr = both.iloc[:, 0].corr(both.iloc[:, 1])
        if not np.isnan(corr):
            corrs.append(corr)
    return mean_or_nan(corrs)


def summarize_task(task: str, runs: list[dict]) -> tuple[dict, list[dict]]:
    id_sets = [run["ids"] for run in runs]
    union_all = set().union(*id_sets) if id_sets else set()
    intersection_all = set.intersection(*id_sets) if id_sets else set()

    pairwise_rows = []
    jaccards = []
    value_corrs = []

    for run_a, run_b in combinations(runs, 2):
        union = run_a["ids"] | run_b["ids"]
        shared = run_a["ids"] & run_b["ids"]
        jaccard = pairwise_jaccard(run_a, run_b)
        value_corr = pairwise_value_corr(run_a, run_b, shared)
        jaccards.append(jaccard)
        value_corrs.append(value_corr)
        pairwise_rows.append(
            {
                "task": task,
                "run_a": run_a["path"].name,
                "run_b": run_b["path"].name,
                "shared_count": len(shared),
                "union_count": len(union),
                "shared_metric_jaccard": jaccard,
                "value_corr": value_corr,
            }
        )

    summary = {
        "task": task,
        "run_count": len(runs),
        "union_items": len(union_all),
        "intersection_items": len(intersection_all),
        "mean_run_items": mean_or_nan([len(ids) for ids in id_sets]),
        "shared_metric_jaccard": mean_or_nan(jaccards),
        "mean_pairwise_value_corr": mean_or_nan(value_corrs),
        "id_columns": ", ".join(runs[0]["id_cols"]) if runs else "",
        "value_columns": ", ".join(sorted({col for run in runs for col in run["value_cols"]})) if runs else "",
    }
    return summary, pairwise_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare stability task outputs across runs using shared overlap and value agreement metrics."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Path to the stability-results directory.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "robustness_summary.csv",
        help="CSV path for the summary table.",
    )
    parser.add_argument(
        "--pairwise-out",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "robustness_pairwise.csv",
        help="CSV path for per-pair metrics.",
    )
    parser.add_argument("--quiet", action="store_true", help="Do not print the summary table.")
    args = parser.parse_args()

    summary_rows = []
    pairwise_rows = []

    for task, spec in TASK_SPECS.items():
        runs = load_runs(args.results_dir, spec)
        if not runs:
            print(f"Skipping {task}: no files found in {args.results_dir / spec.folder}")
            continue
        summary, pairwise = summarize_task(task, runs)
        summary_rows.append(summary)
        pairwise_rows.extend(pairwise)

    if not summary_rows:
        print("No task outputs found.")
        return 1

    summary_df = pd.DataFrame(summary_rows).sort_values("task")
    summary_df.to_csv(args.summary_out, index=False)

    if pairwise_rows:
        pairwise_df = pd.DataFrame(pairwise_rows).sort_values(["task", "run_a", "run_b"])
        pairwise_df.to_csv(args.pairwise_out, index=False)

    if not args.quiet:
        print(summary_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
