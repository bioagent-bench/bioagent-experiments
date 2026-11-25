"""Unit tests for deterministic BioAgent result scoring edge cases."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import verifiers.v1 as vf
from bioagent_taskset.scoring import score_results


class _MemoryRuntime:
    """Minimal in-memory Runtime stub implementing the `read` contract used by score_results."""

    def __init__(self, files: dict[str, str]) -> None:
        self._files = {path: text.encode("utf-8") for path, text in files.items()}

    async def read(self, path: str) -> bytes:
        return self._files[path]


async def _run_scorer(
    tmp_path: Path,
    task_id: str,
    generated_files: dict[str, str],
    truth_files: dict[str, str],
) -> bool:
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    for name, content in truth_files.items():
        (truth_dir / name).write_text(content, encoding="utf-8")
    runtime = cast(vf.Runtime, _MemoryRuntime(generated_files))
    return await score_results(task_id, runtime, sorted(generated_files), truth_dir, {})


# ---------------------------------------------------------------------------
# 1. viral-metagenomics: ICTV 2023-2024 taxonomy synonyms & FP rejection
# ---------------------------------------------------------------------------

VIRAL_TRUTH = {
    "taxonomy.csv": (
        "contig_count,domain,species\n"
        "2,Viruses,Equid gammaherpesvirus 5\n"
        "44,Viruses,Bottlenose dolphin adenovirus 1\n"
        "68,,Unclassified\n"
    )
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "csv_content",
    [
        # Legacy exact pre-2023 name with contig counts
        (
            "contig_count,domain,species\n"
            "44,Viruses,Bottlenose dolphin adenovirus 1\n"
            "2,Viruses,Equid gammaherpesvirus 5\n"
            "68,,Unclassified\n"
        ),
        # Legacy minimal schema without contig_count column
        "domain,species\nViruses,Bottlenose dolphin adenovirus 1\n",
        # Modern Kaiju 2024-08-15 ICTV output
        (
            "contig_count,domain,species\n"
            "11,Viruses,Dolphin mastadenovirus B\n"
            "5,Unclassified,\n"
            "1,Viruses,Percavirus equidgamma5\n"
        ),
        # Ratified ICTV Latinized binomial species name
        (
            "contig_count,domain,species\n"
            "12,Viruses,Mastadenovirus delphini\n"
            "1,Viruses,Percavirus equidgamma5\n"
        ),
        # High unclassified viral contig count should not disqualify dominant named species
        (
            "contig_count,domain,species\n"
            "50,Viruses,Unclassified\n"
            "11,Viruses,Dolphin mastadenovirus B\n"
        ),
    ],
)
async def test_viral_metagenomics_accepts_legacy_and_ictv_synonyms(
    tmp_path: Path, csv_content: str
) -> None:
    assert await _run_scorer(
        tmp_path,
        "viral-metagenomics",
        {"results/taxonomy.csv": csv_content},
        VIRAL_TRUTH,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "csv_content",
    [
        # Wrong closely related dolphin adenovirus species (AdV-2, not AdV-1)
        "contig_count,domain,species\n44,Viruses,Dolphin mastadenovirus A\n",
        "contig_count,domain,species\n44,Viruses,Mastadenovirus delphinidae\n",
        # Wrong unrelated virus
        "contig_count,domain,species\n44,Viruses,Human mastadenovirus C\n",
        # Secondary virus only without primary target species
        "contig_count,domain,species\n2,Viruses,Percavirus equidgamma5\n",
        "contig_count,domain,species\n2,Viruses,Equid gammaherpesvirus 5\n",
        # Wrong taxonomic domain
        "contig_count,domain,species\n11,Bacteria,Dolphin mastadenovirus B\n",
        # Zero contig count for target species
        "contig_count,domain,species\n0,Viruses,Dolphin mastadenovirus B\n",
        # Extra false-positive named virus dominating the target species
        (
            "contig_count,domain,species\n"
            "200,Viruses,Human mastadenovirus C\n"
            "1,Viruses,Dolphin mastadenovirus B\n"
        ),
    ],
)
async def test_viral_metagenomics_rejects_wrong_and_false_positive_viruses(
    tmp_path: Path, csv_content: str
) -> None:
    assert not await _run_scorer(
        tmp_path,
        "viral-metagenomics",
        {"results/taxonomy.csv": csv_content},
        VIRAL_TRUTH,
    )


# ---------------------------------------------------------------------------
# 2. metagenomics: phyloseq::tax_glom leaf TaxIDs vs canonical Phylum TaxIDs
# ---------------------------------------------------------------------------

METAGENOMICS_TRUTH = {
    "phylum_relative_abundances.csv": (
        '"OTU","Kingdom","Phylum","JP4D","JC1A"\n'
        '"2771361","Bacteria","Pseudomonadota",84.8613,66.4368\n'
        '"1883","Bacteria","Actinomycetota",6.2580,14.2529\n'
        '"2004710","Bacteria","Bacteroidota",4.6397,6.2069\n'
        '"39491","Bacteria","Bacillota",1.0922,4.3678\n'
        '"2718620","Bacteria","Thermodesulfobacteriota",0.3747,5.0575\n'
    )
}


@pytest.mark.asyncio
async def test_metagenomics_legacy_exact_otu_phylum_match(tmp_path: Path) -> None:
    assert await _run_scorer(
        tmp_path,
        "metagenomics",
        {"results/phylum.csv": METAGENOMICS_TRUTH["phylum_relative_abundances.csv"]},
        METAGENOMICS_TRUTH,
    )


@pytest.mark.asyncio
async def test_metagenomics_accepts_canonical_ncbi_phylum_taxids(tmp_path: Path) -> None:
    kraken2_output = (
        "OTU,Kingdom,Phylum,JP4D,JC1A\n"
        "1224,Bacteria,Pseudomonadota,84.8255,66.1765\n"
        "201174,Bacteria,Actinomycetota,7.8427,21.7831\n"
        "976,Bacteria,Bacteroidota,3.5697,2.9944\n"
        "1239,Bacteria,Bacillota,1.5070,2.5252\n"
        "200940,Bacteria,Thermodesulfobacteriota,0.2380,1.8769\n"
        "1297,Bacteria,Deinococcota,0.0863,0.3822\n"
    )
    assert await _run_scorer(
        tmp_path,
        "metagenomics",
        {"results/phylum.csv": kraken2_output},
        METAGENOMICS_TRUTH,
    )


@pytest.mark.asyncio
async def test_metagenomics_accepts_legacy_phylum_synonyms_and_proportions(tmp_path: Path) -> None:
    proportion_output = (
        "OTU,Kingdom,Phylum,JP4D,JC1A\n"
        "1224,Bacteria,Proteobacteria,0.85,0.67\n"
        "201174,Bacteria,Actinobacteria,0.07,0.15\n"
        "976,Bacteria,Bacteroidetes,0.05,0.07\n"
        "1239,Bacteria,Firmicutes,0.02,0.05\n"
    )
    assert await _run_scorer(
        tmp_path,
        "metagenomics",
        {"results/phylum.csv": proportion_output},
        METAGENOMICS_TRUTH,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "csv_content",
    [
        # Wrong dominant phylum
        (
            "OTU,Kingdom,Phylum,JP4D,JC1A\n"
            "201174,Bacteria,Actinomycetota,85.0,66.0\n"
            "1224,Bacteria,Pseudomonadota,6.0,14.0\n"
            "976,Bacteria,Bacteroidota,5.0,6.0\n"
        ),
        # Fewer than two additional expected phyla
        (
            "OTU,Kingdom,Phylum,JP4D,JC1A\n"
            "1224,Bacteria,Pseudomonadota,85.0,66.0\n"
            "201174,Bacteria,Actinomycetota,15.0,34.0\n"
        ),
        # Wrong flat abundances despite valid canonical Phylum TaxIDs
        (
            "OTU,Kingdom,Phylum,JP4D,JC1A\n"
            "1224,Bacteria,Pseudomonadota,35.0,35.0\n"
            "201174,Bacteria,Actinomycetota,33.0,33.0\n"
            "976,Bacteria,Bacteroidota,32.0,32.0\n"
        ),
        # Missing JC1A sample abundances (all zeros)
        (
            "OTU,Kingdom,Phylum,JP4D,JC1A\n"
            "1224,Bacteria,Pseudomonadota,85.0,0.0\n"
            "201174,Bacteria,Actinomycetota,7.0,0.0\n"
            "976,Bacteria,Bacteroidota,5.0,0.0\n"
        ),
        # Large fabricated phylum distorting community composition
        (
            "OTU,Kingdom,Phylum,JP4D,JC1A\n"
            "1224,Bacteria,Pseudomonadota,52.0,52.0\n"
            "999999,Bacteria,Fabricatedota,45.0,45.0\n"
            "201174,Bacteria,Actinomycetota,2.0,2.0\n"
            "976,Bacteria,Bacteroidota,1.0,1.0\n"
        ),
        # Non-top phyla reported with zero abundance
        (
            "OTU,Kingdom,Phylum,JP4D,JC1A\n"
            "1224,Bacteria,Pseudomonadota,100.0,100.0\n"
            "201174,Bacteria,Actinomycetota,0.0,0.0\n"
            "976,Bacteria,Bacteroidota,0.0,0.0\n"
        ),
        # Malformed non-numeric abundance
        (
            "OTU,Kingdom,Phylum,JP4D,JC1A\n"
            "1224,Bacteria,Pseudomonadota,85.0,NA\n"
            "201174,Bacteria,Actinomycetota,7.0,14.0\n"
            "976,Bacteria,Bacteroidota,5.0,6.0\n"
        ),
    ],
)
async def test_metagenomics_rejects_bad_abundances_and_taxa(
    tmp_path: Path, csv_content: str
) -> None:
    assert not await _run_scorer(
        tmp_path,
        "metagenomics",
        {"results/phylum.csv": csv_content},
        METAGENOMICS_TRUTH,
    )


# ---------------------------------------------------------------------------
# 3. transcript-quant: zero-count rows & EM quantification rounding
# ---------------------------------------------------------------------------

TRANSCRIPT_TRUTH = {
    "truth.tsv": (
        "ENST00000221801.8\t3278\n"
        "ENST00000246747.9\t1217\n"
        "ENST00000377205.6\t324\n"
        "ENST00000462686.1\t75\n"
        "ENST00000403197.5\t61\n"
        "ENST00000256078.10\t54\n"
        "ENST00000075322.11\t14\n"
    )
}


@pytest.mark.asyncio
async def test_transcript_quant_legacy_exact_match(tmp_path: Path) -> None:
    pred = "transcript_id\tcount\n" + TRANSCRIPT_TRUTH["truth.tsv"]
    assert await _run_scorer(
        tmp_path,
        "transcript-quant",
        {"results/counts.tsv": pred},
        TRANSCRIPT_TRUTH,
    )


@pytest.mark.asyncio
async def test_transcript_quant_accepts_zero_count_rows_and_em_rounding(tmp_path: Path) -> None:
    salmon_pred = (
        "transcript_id\tcount\n"
        "ENST00000221801.8\t3276\n"
        "ENST00000246747.9\t1219\n"
        "ENST00000377205.6\t323\n"
        "ENST00000462686.1\t76\n"
        "ENST00000403197.5\t60\n"
        "ENST00000256078.10\t55\n"
        "ENST00000075322.11\t14\n"
        "ENST00000999991.1\t0\n"
        "ENST00000999992.1\t0.0\n"
        "ENST00000999993.1\t0\n"
    )
    assert await _run_scorer(
        tmp_path,
        "transcript-quant",
        {"results/counts.tsv": salmon_pred},
        TRANSCRIPT_TRUTH,
    )


@pytest.mark.asyncio
async def test_transcript_quant_accepts_fractional_salmon_numreads(tmp_path: Path) -> None:
    salmon_float_pred = (
        "transcript_id\tcount\n"
        "ENST00000221801.8\t3277.4\n"
        "ENST00000246747.9\t1218.2\n"
        "ENST00000377205.6\t323.1\n"
        "ENST00000462686.1\t75.6\n"
        "ENST00000403197.5\t60.7\n"
        "ENST00000256078.10\t54.0\n"
        "ENST00000075322.11\t14.0\n"
        "ENST00000999999.1\t0.000\n"
    )
    assert await _run_scorer(
        tmp_path,
        "transcript-quant",
        {"results/counts.tsv": salmon_float_pred},
        TRANSCRIPT_TRUTH,
    )


@pytest.mark.asyncio
async def test_transcript_quant_accepts_realistic_isoform_em_reassignment(tmp_path: Path) -> None:
    # 20 expressed transcripts where EM shifts 1 low-abundance isoform to 0 and assigns
    # a few reads to an overlapping isoform, while preserving r > 0.999 and low L1 error.
    truth_lines = [
        f"ENST{i:04d}.1\t{count}" for i, count in enumerate(range(50, 1050, 50), start=1)
    ]
    truth_lines.append("ENST0021.1\t5")
    pred_lines = ["transcript_id\tcount"]
    for i, count in enumerate(range(50, 1050, 50), start=1):
        delta = 1 if i % 2 == 0 else -1
        pred_lines.append(f"ENST{i:04d}.1\t{count + delta}")
    pred_lines.append("ENST0021.1\t0")
    pred_lines.append("ENST0022.1\t5")
    pred_lines.extend(f"ENST{i:04d}.1\t0" for i in range(100, 130))

    assert await _run_scorer(
        tmp_path,
        "transcript-quant",
        {"results/counts.tsv": "\n".join(pred_lines) + "\n"},
        {"truth.tsv": "\n".join(truth_lines) + "\n"},
    )


@pytest.mark.asyncio
async def test_transcript_quant_rejects_single_transcript_wrong_count(tmp_path: Path) -> None:
    assert not await _run_scorer(
        tmp_path,
        "transcript-quant",
        {"results/counts.tsv": "transcript_id\tcount\nENST1\t5\n"},
        {"truth.tsv": "transcript_id\tcount\nENST1\t4\n"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tsv_content",
    [
        # Missing a non-zero truth transcript (recall = 6/7 < 0.90)
        (
            "transcript_id\tcount\n"
            "ENST00000221801.8\t3278\n"
            "ENST00000246747.9\t1217\n"
            "ENST00000377205.6\t324\n"
            "ENST00000462686.1\t75\n"
            "ENST00000403197.5\t61\n"
            "ENST00000256078.10\t54\n"
        ),
        # Spurious non-zero transcript (precision = 7/8 < 0.90)
        (
            "transcript_id\tcount\n"
            "ENST00000221801.8\t3278\n"
            "ENST00000246747.9\t1217\n"
            "ENST00000377205.6\t324\n"
            "ENST00000462686.1\t75\n"
            "ENST00000403197.5\t61\n"
            "ENST00000256078.10\t54\n"
            "ENST00000075322.11\t14\n"
            "ENST00000999999.1\t10\n"
        ),
        # Wildly wrong counts (swapped top two transcripts)
        (
            "transcript_id\tcount\n"
            "ENST00000221801.8\t1217\n"
            "ENST00000246747.9\t3278\n"
            "ENST00000377205.6\t324\n"
            "ENST00000462686.1\t75\n"
            "ENST00000403197.5\t61\n"
            "ENST00000256078.10\t54\n"
            "ENST00000075322.11\t14\n"
        ),
        # Counts doubled (total read mass off by 2x)
        (
            "transcript_id\tcount\n"
            "ENST00000221801.8\t6556\n"
            "ENST00000246747.9\t2434\n"
            "ENST00000377205.6\t648\n"
            "ENST00000462686.1\t150\n"
            "ENST00000403197.5\t122\n"
            "ENST00000256078.10\t108\n"
            "ENST00000075322.11\t28\n"
        ),
        # All-zero output
        ("transcript_id\tcount\nENST00000221801.8\t0\nENST00000246747.9\t0\n"),
        # Negative count
        (
            "transcript_id\tcount\n"
            "ENST00000221801.8\t-3278\n"
            "ENST00000246747.9\t1217\n"
            "ENST00000377205.6\t324\n"
            "ENST00000462686.1\t75\n"
            "ENST00000403197.5\t61\n"
            "ENST00000256078.10\t54\n"
            "ENST00000075322.11\t14\n"
        ),
        # Non-numeric count
        ("transcript_id\tcount\nENST00000221801.8\tNaN\nENST00000246747.9\t1217\n"),
    ],
)
async def test_transcript_quant_rejects_missing_or_wrong_counts(
    tmp_path: Path, tsv_content: str
) -> None:
    assert not await _run_scorer(
        tmp_path,
        "transcript-quant",
        {"results/counts.tsv": tsv_content},
        TRANSCRIPT_TRUTH,
    )


# ---------------------------------------------------------------------------
# 4. evolution: SPAdes contig header canonicalization & allele validation
# ---------------------------------------------------------------------------

EVOLUTION_TRUTH = {
    "variants_shared.csv": (
        "CHROM,POS,REF,ALT,GENE,IMPACT,EFFECT,STATUS\n"
        "NODE_5_length_100915_cov_6.070430,76369,CCCGCGCCGCGCCGCGCCGA,CCCGCGCCGCGCCGA,"
        "G1,HIGH,frameshift_variant,shared\n"
        "NODE_22_length_60506_cov_6.117460,32199,TCT,TGCT,G2,HIGH,frameshift_variant,shared\n"
        "NODE_26_length_50016_cov_6.096117,20446,T,A,G3,MODERATE,missense_variant,shared\n"
    )
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "csv_content",
    [
        # Legacy exact (chrom, pos) match with identical SPAdes header
        ("chrom,pos,ref,alt\nNODE_26_length_50016_cov_6.096117,20446,T,A\n"),
        # Legacy minimal 2-column (chrom, pos) table with exact header
        "chrom,pos\nNODE_26_length_50016_cov_6.096117,20446\n",
        # Shifted SPAdes floating-point coverage + minimal vs right-padded deletion alleles
        (
            "chrom,pos,ref,alt\n"
            "NODE_5_length_100915_cov_5.874551,76369,CCCGCGCCGCGCCGCGCCG,CCCGCGCCGCGCCG\n"
        ),
        # Shifted SPAdes rank index + coverage + minimal vs right-padded insertion alleles
        ("chrom,pos,ref,alt\nNODE_23_length_60506_cov_5.921346,32199,T,TG\n"),
        # Shifted SPAdes rank index + coverage on SNP
        ("chrom,pos,ref,alt\nNODE_28_length_50016_cov_5.968722,20446,T,A\n"),
    ],
)
async def test_evolution_accepts_legacy_and_canonicalized_spades_headers(
    tmp_path: Path, csv_content: str
) -> None:
    assert await _run_scorer(
        tmp_path,
        "evolution",
        {"results/variants.csv": csv_content},
        EVOLUTION_TRUTH,
    )


@pytest.mark.asyncio
async def test_evolution_preserves_non_spades_chrom_matching(tmp_path: Path) -> None:
    non_spades_truth = {"variants.csv": "CHROM,POS,REF,ALT\nchr1,12345,G,A\n"}
    assert await _run_scorer(
        tmp_path,
        "evolution",
        {"results/variants.csv": "chrom,pos,ref,alt\n1,12345,G,A\n"},
        non_spades_truth,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "csv_content",
    [
        # Wrong position on matching SPAdes contig
        "chrom,pos,ref,alt\nNODE_28_length_50016_cov_5.968722,20447,T,A\n",
        # Wrong contig length at matching position
        "chrom,pos,ref,alt\nNODE_28_length_50017_cov_5.968722,20446,T,A\n",
        # Wrong alternate allele on canonicalized SPAdes contig
        "chrom,pos,ref,alt\nNODE_28_length_50016_cov_5.968722,20446,T,C\n",
        # Wrong alternate allele even on exact SPAdes contig header
        "chrom,pos,ref,alt\nNODE_26_length_50016_cov_6.096117,20446,T,C\n",
        # Missing alleles on shifted SPAdes header when truth specifies alleles
        "chrom,pos\nNODE_28_length_50016_cov_5.968722,20446\n",
        # False-positive variants on unrelated contigs only
        (
            "chrom,pos,ref,alt\n"
            "NODE_1_length_157883_cov_5.770357,143695,G,T\n"
            "NODE_3_length_120662_cov_5.768387,66991,G,T\n"
        ),
        # Empty result table
        "chrom,pos,ref,alt\n",
    ],
)
async def test_evolution_rejects_wrong_coords_alleles_and_false_positives(
    tmp_path: Path, csv_content: str
) -> None:
    assert not await _run_scorer(
        tmp_path,
        "evolution",
        {"results/variants.csv": csv_content},
        EVOLUTION_TRUTH,
    )
