SYSTEM_PROMPT = """\
You are an expert bioinformatics agent. Complete the requested analysis using accepted practices
from genomics, transcriptomics, and computational biology.

Work only in the current rollout workspace. Inspect the available files before choosing a pipeline.
Inputs are in `data/`; when supplied, reference resources are in `reference/`. You may install
missing software in the runtime. Never fabricate biological data or a successful result.

Keep intermediate work in numbered, descriptive directories under `outputs/`. Put only the final
deliverable requested by the task in `results/`. Stop once that deliverable is complete.
"""


RESULT_RULES = {
    "alzheimer-mouse": (
        "results_match is true only if the generated CSV and expected CSV share at least one "
        "Pathway value."
    ),
    "comparative-genomics": (
        "results_match is true only if at least one consensus_annotation value exactly matches "
        "the expected results."
    ),
    "cystic-fibrosis": (
        "results_match is true only if the causal CFTR variant is reported exactly once with "
        "chromosome 7, position 117227832, variant_id 7115, reference G, alternate T, gene CFTR, "
        "gene_id ENSG00000001626, annotation stop_gained, impact HIGH, and transcript "
        "ENST00000003084."
    ),
    "deseq": (
        "results_match is true only if at least five gene_id values overlap the expected DESeq "
        "output."
    ),
    "evolution": (
        "results_match is true only if at least one chromosome/CHROM and position/POS pair "
        "(canonicalizing SPAdes NODE_<idx>_length_<L>_cov_<float> headers by contig length when "
        "alleles match) matches the expected variants."
    ),
    "giab": "results_match is true only if the hap.py SNP F1 score is greater than 0.7.",
    "metagenomics": (
        "results_match is true only if the most abundant phylum is Pseudomonadota when comparing "
        "JP4D and JC1A, at least two additional expected phyla overlap (by (OTU, Phylum) or "
        "phylum name), and relative abundances agree quantitatively with the expected results."
    ),
    "single-cell": (
        "results_match is true only if at least one (cluster_id, predicted_cell_type) pair "
        "matches the expected results."
    ),
    "transcript-quant": (
        "results_match is true only if non-zero transcript counts match the expected mapping "
        "exactly or within tight EM quantification tolerances; zero-count rows, row order, and "
        "formatting may differ."
    ),
    "viral-metagenomics": (
        "results_match is true only if Bottlenose dolphin adenovirus 1 (or its ICTV synonyms "
        "Dolphin mastadenovirus B / Mastadenovirus delphini) is explicitly reported as a "
        "dominant positive hit under the Viruses domain."
    ),
}
