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
        "matches the expected variants."
    ),
    "giab": "results_match is true only if the hap.py SNP F1 score is greater than 0.7.",
    "metagenomics": (
        "results_match is true only if the most abundant phylum is Pseudomonadota when comparing "
        "JP4D and JC1A and at least two additional OTUs have the expected Phylum labels."
    ),
    "single-cell": (
        "results_match is true only if at least one (cluster_id, predicted_cell_type) pair "
        "matches the expected results."
    ),
    "transcript-quant": (
        "results_match is true only if the complete transcript_id to count mapping equals the "
        "expected mapping; row order and formatting may differ."
    ),
    "viral-metagenomics": (
        "results_match is true only if Bottlenose dolphin adenovirus 1 is explicitly reported "
        "under the Viruses domain."
    ),
}
