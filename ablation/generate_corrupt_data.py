from pathlib import Path
import random


def generate_gene_table(n_rows: int = 12, seed: int = 0):
    """
    Create a DataFrame with the same columns as the provided table, but with
    intentionally fake/obviously-wrong values (mostly constant / uniform).
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)

    cols = [
        "", "gene_id", "gene_name", "feature", "fc", "log2fc", "pval", "qval",
        "wt.MeanFPKM", "tg.MeanFPKM",
        "FPKM.tau_02", "FPKM.tau_03", "FPKM.tau_04", "FPKM.tau_01", "FPKM.tau_05", "FPKM.tau_06"
    ]

    idx_col = np.arange(1, n_rows + 1)

    gene_ids = [f"ENSMUSG{str(i).zfill(11)}" for i in range(1, n_rows + 1)]
    gene_names = [f"FAKE{i:04d}" for i in range(1, n_rows + 1)]
    feature = ["gene"] * n_rows

    fc = np.full(n_rows, 1.0)
    log2fc = np.full(n_rows, 0.0)

    pval = np.full(n_rows, 1.0)
    qval = np.full(n_rows, 1.0)

    wt_mean = np.full(n_rows, 0.0)
    tg_mean = np.full(n_rows, 0.0)

    base_tau = np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14])
    tau = np.tile(base_tau, (n_rows, 1))

    tau += rng.uniform(-1e-6, 1e-6, size=tau.shape)

    data = {
        "": idx_col,
        "gene_id": gene_ids,
        "gene_name": gene_names,
        "feature": feature,
        "fc": fc,
        "log2fc": log2fc,
        "pval": pval,
        "qval": qval,
        "wt.MeanFPKM": wt_mean,
        "tg.MeanFPKM": tg_mean,
        "FPKM.tau_02": tau[:, 0],
        "FPKM.tau_03": tau[:, 1],
        "FPKM.tau_04": tau[:, 2],
        "FPKM.tau_01": tau[:, 3],
        "FPKM.tau_05": tau[:, 4],
        "FPKM.tau_06": tau[:, 5],
    }

    df = pd.DataFrame(data, columns=cols)
    df.to_csv('DEA_PS3O1.csv')


def corrupt_seq_to_ns(seq: str, rng: random.Random) -> str:
    """Replace ~frac_n of positions with 'N', keeping length identical."""
    seq = seq.rstrip("\n")
    L = len(seq)
    if L == 0:
        return seq
    k = int(round(0.9 * L))
    k = max(0, min(L, k))
    idxs = list(range(L))
    rng.shuffle(idxs)
    to_n = set(idxs[:k])
    return "".join("N" if i in to_n else seq[i] for i in range(L))

from pathlib import Path
import random
import tempfile
import os
import shutil
import gzip

def generate_corrupt_fastq():
    fastq_dir = Path("~/bioagent-experiments/ablation/corrupt/data/deseq/").expanduser()
    out_dir = fastq_dir / "corrupted"
    pattern = "SRR*.fastq"

    frac_n = 0.90
    force_lowq = True

    def corrupt_seq_to_ns(seq: str, rng: random.Random) -> str:
        seq = seq.rstrip("\n")
        L = len(seq)
        if L == 0:
            return seq
        k = int(round(frac_n * L))
        k = max(0, min(L, k))
        idxs = list(range(L))
        rng.shuffle(idxs)
        to_n = set(idxs[:k])
        return "".join("N" if i in to_n else seq[i] for i in range(L))

    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(fastq_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No FASTQ files matched {pattern!r} in {fastq_dir}")

    for in_path in files:
        rng = random.Random()
        out_path = out_dir / in_path.name

        with in_path.open("r", encoding="utf-8", errors="replace") as fin, \
             out_path.open("w", encoding="utf-8") as fout:

            rec = 0
            while True:
                h = fin.readline()
                if not h:
                    break
                s = fin.readline()
                p = fin.readline()
                q = fin.readline()
                if not (s and p and q):
                    raise ValueError(f"Truncated FASTQ record in {in_path} at record {rec+1}")

                if not h.startswith("@"):
                    raise ValueError(f"Invalid FASTQ header in {in_path} at record {rec+1}: {h[:50]!r}")
                if not p.startswith("+"):
                    raise ValueError(f"Invalid FASTQ plus-line in {in_path} at record {rec+1}: {p[:50]!r}")

                s_stripped = s.rstrip("\n")
                q_stripped = q.rstrip("\n")
                if len(s_stripped) != len(q_stripped):
                    raise ValueError(
                        f"Length mismatch in {in_path} at record {rec+1}: "
                        f"seq={len(s_stripped)} qual={len(q_stripped)}"
                    )

                s_cor = corrupt_seq_to_ns(s_stripped, rng)
                q_cor = ("!" * len(q_stripped)) if force_lowq else q_stripped

                fout.write(h)
                fout.write(s_cor + "\n")
                fout.write(p)
                fout.write(q_cor + "\n")
                rec += 1

        print(f"Wrote corrupted copy: {out_path}")

    print(f"\nDone. Corrupted FASTQs are in: {out_dir}\n"
          f"Point your experiment at that directory to use the junk reads.")

def generate_corrupt_fastq_gz():
    in_dir = Path("~/bioagent-experiments/ablation/corrupt/data/evolution").expanduser()
    out_dir = in_dir / "corrupted"
    out_dir.mkdir(parents=True, exist_ok=True)

    frac_n = 0.90
    force_lowq = True

    def corrupt_seq_to_ns(seq: str, rng: random.Random) -> str:
        L = len(seq)
        if L == 0:
            return seq
        k = int(round(frac_n * L))
        k = max(0, min(L, k))
        idxs = list(range(L))
        rng.shuffle(idxs)
        to_n = set(idxs[:k])
        return "".join("N" if i in to_n else seq[i] for i in range(L))

    files = sorted([f for f in in_dir.glob("*.fastq.gz") if 'anc' in f.name])
    if not files:
        raise FileNotFoundError(f"No *.fastq.gz files found in {in_dir}")

    for in_path in files:
        out_path = out_dir / in_path.name
        rng = random.Random()

        with gzip.open(in_path, "rt", encoding="utf-8", errors="replace", newline="") as fin, \
             gzip.open(out_path, "wt", encoding="utf-8", newline="\n") as fout:

            rec = 0
            while True:
                h = fin.readline()
                if not h:
                    break
                s = fin.readline()
                p = fin.readline()
                q = fin.readline()
                if not (s and p and q):
                    raise ValueError(f"Truncated FASTQ record in {in_path} at record {rec+1}")

                h = h.rstrip("\n")
                s = s.rstrip("\n")
                p = p.rstrip("\n")
                q = q.rstrip("\n")

                if not h.startswith("@"):
                    raise ValueError(f"Invalid FASTQ header in {in_path} at record {rec+1}: {h[:60]!r}")
                if not p.startswith("+"):
                    raise ValueError(f"Invalid FASTQ plus-line in {in_path} at record {rec+1}: {p[:60]!r}")
                if len(s) != len(q):
                    raise ValueError(
                        f"Length mismatch in {in_path} at record {rec+1}: seq={len(s)} qual={len(q)}"
                    )

                s_cor = corrupt_seq_to_ns(s, rng)
                q_cor = ("!" * len(q)) if force_lowq else q

                fout.write(h + "\n")
                fout.write(s_cor + "\n")
                fout.write(p + "\n")
                fout.write(q_cor + "\n")
                rec += 1

        print(f"Wrote corrupted copy: {out_path}")

    print(f"\nDone. Corrupted FASTQ.GZ files are in: {out_dir}\n"
          f"Point your pipeline at that directory to use the junk reads.")


def generate_corrupt_fq(path):
    in_dir = Path(path).expanduser()
    out_dir = in_dir / "corrupted"
    out_dir.mkdir(parents=True, exist_ok=True)

    frac_n = 0.90
    force_lowq = True

    def corrupt_seq_to_ns(seq: str, rng: random.Random) -> str:
        L = len(seq)
        if L == 0:
            return seq
        k = int(round(frac_n * L))
        k = max(0, min(L, k))
        idxs = list(range(L))
        rng.shuffle(idxs)
        to_n = set(idxs[:k])
        return "".join("N" if i in to_n else seq[i] for i in range(L))

    files = sorted([f for f in in_dir.glob("*.fastq.gz")])
    if not files:
        raise FileNotFoundError(f"No *.fastq.gz files found in {in_dir}")

    for in_path in files:
        out_path = out_dir / in_path.name
        rng = random.Random()

        with gzip.open(in_path, "rt", encoding="utf-8", errors="replace", newline="") as fin, \
             gzip.open(out_path, "wt", encoding="utf-8", newline="\n") as fout:

            rec = 0
            while True:
                h = fin.readline()
                if not h:
                    break
                s = fin.readline()
                p = fin.readline()
                q = fin.readline()
                if not (s and p and q):
                    raise ValueError(f"Truncated FASTQ record in {in_path} at record {rec+1}")

                h = h.rstrip("\n")
                s = s.rstrip("\n")
                p = p.rstrip("\n")
                q = q.rstrip("\n")

                if not h.startswith("@"):
                    raise ValueError(f"Invalid FASTQ header in {in_path} at record {rec+1}: {h[:60]!r}")
                if not p.startswith("+"):
                    raise ValueError(f"Invalid FASTQ plus-line in {in_path} at record {rec+1}: {p[:60]!r}")
                if len(s) != len(q):
                    raise ValueError(
                        f"Length mismatch in {in_path} at record {rec+1}: seq={len(s)} qual={len(q)}"
                    )

                s_cor = corrupt_seq_to_ns(s, rng)
                q_cor = ("!" * len(q)) if force_lowq else q

                fout.write(h + "\n")
                fout.write(s_cor + "\n")
                fout.write(p + "\n")
                fout.write(q_cor + "\n")
                rec += 1

        print(f"Wrote corrupted copy: {out_path}")

    print(f"\nDone. Corrupted FASTQ.GZ files are in: {out_dir}\n"
          f"Point your pipeline at that directory to use the junk reads.")


def generate_corrupt_single_cell():
    """
    Make MatrixMarket (.mtx) copies where every nonzero entry's value is replaced with `constant`.

    Preserves:
      - header line (%%MatrixMarket ...)
      - all comment lines starting with %
      - dims line: n_rows n_cols nnz
      - the exact (row, col) indices and number of entries

    Writes:
      output_dir/<same filename> for each matched .mtx

    Notes:
      - MatrixMarket coordinate format uses 1-based indices; we keep them unchanged.
      - Many 10x mtx are declared 'integer'; using an int constant is safest.
    """
    in_dir = Path("~/bioagent-experiments/ablation/corrupt/data/single-cell").expanduser().resolve()

    pattern = "*_matrix.mtx"
    mtx_files = sorted(in_dir.glob(pattern))
    if not mtx_files:
        raise FileNotFoundError(f"No files matched {pattern!r} in {in_dir}")

    for in_path in mtx_files:
        out_path = in_dir / "corrupted" / in_path.name

        with in_path.open("r", encoding="utf-8", errors="replace") as fin, \
             out_path.open("w", encoding="utf-8", newline="\n") as fout:

            header = fin.readline()
            if not header.startswith("%%MatrixMarket"):
                raise ValueError(f"{in_path} does not start with a MatrixMarket header")
            fout.write(header)

            dims_line = None
            for line in fin:
                if not line.strip():
                    fout.write(line)
                    continue
                if line.lstrip().startswith("%"):
                    fout.write(line)
                    continue
                dims_line = line
                break

            if dims_line is None:
                raise ValueError(f"{in_path}: missing dims line")

            dims = dims_line.strip().split()
            if len(dims) != 3:
                raise ValueError(f"{in_path}: bad dims line: {dims_line!r}")
            fout.write(dims_line)

            for line in fin:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"{in_path}: malformed entry line: {line!r}")
                i, j = parts[0], parts[1]
                fout.write(f"{i} {j} {777}\n")

        print(f"Wrote: {out_path}")



if __name__=="__main__":
    # generate_gene_table()
    # generate_corrupt_fastq()
    # generate_corrupt_fastq_gz()
    # generate_corrupt_fq("~/bioagent-experiments/ablation/corrupt/data/giab")
    # generate_corrupt_fq("~/bioagent-experiments/ablation/corrupt/data/metagenomics")
    # generate_corrupt_single_cell()
    # generate_corrupt_fq("~/bioagent-experiments/ablation/corrupt/data/transcript-quant")
    generate_corrupt_fq("/home/dionizije/bioagent-experiments/ablation/corrupt/data/viral-metagenomics")