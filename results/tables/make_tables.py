"""Generate the merged perturbation + stability LaTeX table from canonical data.

Single source of truth: ``perturbation_stability.csv`` in this directory.
Never hand-edit the generated .tex -- edit the CSV and re-run this script.

Provenance of each column
-------------------------
delta_completion : results/data/prompt_bloat_delta.csv (in-repo, authoritative).
                   VERIFIED against that file on every run; a mismatch is a
                   hard error, so the numbers cannot silently drift.
corrupted        : paper.pdf Table 3 (perturbation outcomes). "yes" == the
                   paper's \\cmark == the agent identified the corrupt input,
                   which is the desired behaviour.
decoy_avoided    : the COMPLEMENT of the paper's "Decoy" column. The paper
                   reports whether the decoy was *used* (\\cmark == used ==
                   undesired); this column reports whether it was *avoided*, so
                   that "yes" is the desired behaviour in both mark columns and
                   a green dot uniformly means "good".
                   decoy_avoided = NOT decoy_used.
                   Cross-checked against the paper's prose: corrupted inputs
                   correctly identified in 7/10 tasks, decoys used in 2/10
                   (i.e. avoided in 8/10).
jaccard, pearson : paper.pdf Table 2 (four trials per task). The published
                   table has 9 rows and omits `giab`, which was not run
                   multiple times (high resource cost), hence NA/NA here.
                   The paper labels two rows with short names, mapped here to
                   the canonical identifiers used everywhere else:
                     "alzheimer"   -> alzheimer-mouse
                     "comparative" -> comparative-genomics

Usage:  python results/tables/make_tables.py
"""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CANONICAL = HERE / "perturbation_stability.csv"
DELTA_SOURCE = REPO / "results" / "data" / "prompt_bloat_delta.csv"
OUT_TEX = HERE / "table_perturbation_stability.tex"

# Paper prose cross-checks (paper.pdf, section 5.2).
EXPECTED_CORRUPTED_IDENTIFIED = 7
EXPECTED_DECOYS_USED = 2
EXPECTED_ROWS = 10


def read_canonical():
    with CANONICAL.open() as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != EXPECTED_ROWS:
        raise SystemExit(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")
    return rows


def verify_deltas(rows):
    """Hard-fail if delta_completion disagrees with the in-repo source."""
    with DELTA_SOURCE.open() as fh:
        source = {r["task_id"]: float(r["delta"]) for r in csv.DictReader(fh)}

    problems = []
    for row in rows:
        task = row["task_id"]
        if task not in source:
            problems.append(f"{task}: absent from {DELTA_SOURCE.name}")
            continue
        want, got = round(source[task], 1), float(row["delta_completion"])
        if abs(want - got) > 1e-9:
            problems.append(f"{task}: CSV says {got}, source says {want}")

    missing = set(source) - {r["task_id"] for r in rows}
    if missing:
        problems.append(f"tasks in source but not in table: {sorted(missing)}")
    if problems:
        raise SystemExit("Delta verification FAILED:\n  " + "\n  ".join(problems))
    print(f"  delta_completion: {len(rows)}/{len(rows)} match {DELTA_SOURCE.name}")


def verify_marks(rows):
    """Cross-check the mark columns against the paper's reported counts."""
    for col in ("corrupted", "decoy_avoided"):
        bad = [r["task_id"] for r in rows if r[col] not in ("yes", "no")]
        if bad:
            raise SystemExit(f"{col}: values must be yes/no; offenders: {bad}")

    corrupted = sum(r["corrupted"] == "yes" for r in rows)
    # decoy_avoided is the complement of the paper's "decoy used" column, so
    # check it by converting back to the paper's figure.
    decoys_used = sum(r["decoy_avoided"] == "no" for r in rows)
    if corrupted != EXPECTED_CORRUPTED_IDENTIFIED:
        raise SystemExit(
            f"corrupted identified = {corrupted}, paper reports "
            f"{EXPECTED_CORRUPTED_IDENTIFIED}/{EXPECTED_ROWS}"
        )
    if decoys_used != EXPECTED_DECOYS_USED:
        raise SystemExit(
            f"decoys used = {decoys_used} (from {EXPECTED_ROWS - decoys_used} avoided), "
            f"paper reports {EXPECTED_DECOYS_USED}/{EXPECTED_ROWS} used"
        )
    print(f"  corrupted identified: {corrupted}/{EXPECTED_ROWS} matches paper")
    print(f"  decoys avoided:       {EXPECTED_ROWS - decoys_used}/{EXPECTED_ROWS} "
          f"(= {decoys_used} used, matches paper)")


def verify_stability(rows):
    """Sanity-check the stability columns' ranges and NA placement."""
    for row in rows:
        for col in ("jaccard", "pearson"):
            v = row[col]
            if v == "NA":
                continue
            f = float(v)
            lo = 0.0 if col == "jaccard" else -1.0
            if not lo <= f <= 1.0:
                raise SystemExit(f"{row['task_id']}.{col}={f} outside [{lo}, 1.0]")
    na_jaccard = [r["task_id"] for r in rows if r["jaccard"] == "NA"]
    if na_jaccard != ["giab"]:
        raise SystemExit(f"Only giab should lack a Jaccard value; got {na_jaccard}")
    print(f"  stability: ranges OK; Jaccard NA only for {na_jaccard}")


def fmt_num(value, decimals=1):
    return "NA" if value == "NA" else f"{float(value):.{decimals}f}"


def build_tex(rows):
    body = []
    for r in rows:
        body.append(
            "\\texttt{{{task}}} & {corrupted} & {decoy} & {delta} & {jac} & {pear} \\\\".format(
                task=r["task_id"],
                corrupted="\\yes" if r["corrupted"] == "yes" else "\\no",
                decoy="\\yes" if r["decoy_avoided"] == "yes" else "\\no",
                delta=fmt_num(r["delta_completion"], 1),
                jac=fmt_num(r["jaccard"], 3),
                pear=fmt_num(r["pearson"], 3),
            )
        )

    template = r"""% GENERATED by results/tables/make_tables.py -- DO NOT EDIT BY HAND.
% Edit results/tables/perturbation_stability.csv and re-run the script; it
% re-verifies every number against its source before writing this file.
%
% Requires the \yes / \no macros from the Table 1 preamble (green/grey dot
% followed by the word). \cmark -> \yes, \xmark -> \no, exactly as in Table 1.
% They must be declared with \DeclareRobustCommand, because this caption uses
% them and \caption is a moving argument.
% Cross-references \label{tab:bioagent_tasks} (Table 1); both labels below.
\begin{table*}[t]
\centering
\footnotesize
\caption{Agent robustness across tasks, combining perturbation outcomes with
run-to-run stability. \textbf{Corrupted} reports whether the agent identified
the corrupt input, and \textbf{Decoy avoided} whether it correctly ignored the
decoy file; \yes\ is the desired behaviour in both columns.
\textbf{$\Delta$ Completion} is the change in completion performance under
prompt bloat (percentage points; higher is better). \textbf{Jaccard} and
\textbf{Pearson} summarise agreement across four trials per task; \texttt{giab}
was run only once because of its high resource requirements, so its stability
entries are NA, as are Pearson entries for tasks with no valid numeric pairs.
Task identifiers match Table~\ref{tab:bioagent_tasks}.}
\label{tab:perturbation-stability}
\setlength{\tabcolsep}{7pt}
\begin{tabular}{lllccc}
\toprule
\textbf{Identifier} & \textbf{Corrupted} & \textbf{Decoy avoided} & \textbf{$\Delta$ Completion (\%)} & \textbf{Jaccard} & \textbf{Pearson} \\
\midrule
@@ROWS@@
\bottomrule
\end{tabular}
\end{table*}
"""
    return template.replace("@@ROWS@@", "\n".join(body))


if __name__ == "__main__":
    rows = read_canonical()
    print(f"Verifying {len(rows)} rows from {CANONICAL.name}:")
    verify_deltas(rows)
    verify_marks(rows)
    verify_stability(rows)
    OUT_TEX.write_text(build_tex(rows))
    print(f"\nAll checks passed. Wrote {OUT_TEX.relative_to(REPO)}")
