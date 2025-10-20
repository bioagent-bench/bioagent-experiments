from pathlib import Path
from tempfile import TemporaryDirectory
from smolagents import tool

from bio_mcp.tools.snake.variant_processing.snpsift_casecontrol import (
    snpsift_casecontrol,
)
from bio_mcp.tools.snake.variant_processing.snpsift_filter import snpsift_filter
from bio_mcp.tools.snake.variant_processing.snpeff_download import snpeff_download
from bio_mcp.tools.snake.variant_processing.snpeff_annotate import snpeff_annotate

from bio_mcp.tools.snake.preprocessing.fastqc import fastqc
from bio_mcp.tools.snake.preprocessing.trimmomatic import trimmomatic
from bio_mcp.tools.snake.reporting.multiqc import multiqc
from bio_mcp.tools.snake.alignment.star_index import star_index
from bio_mcp.tools.snake.alignment.star_align import star_align
from bio_mcp.tools.snake.reporting.deseq2_deseqdataset import deseq2_deseqdataset
from bio_mcp.tools.snake.reporting.deseq2_wald import deseq2_wald

from bio_mcp.tools.snake.preprocessing.fastp import fastp
from bio_mcp.tools.snake.reporting.quast import quast
from bio_mcp.tools.snake.alignment.bwa_mem2_index import bwa_mem2_index
from bio_mcp.tools.snake.alignment.bwa_mem2_mem import bwa_mem2_mem
from bio_mcp.tools.snake.bam_processing.samtools_faidx import samtools_faidx
from bio_mcp.tools.snake.bam_processing.samtools_sort import samtools_sort
from bio_mcp.tools.snake.bam_processing.samtools_fixmate import samtools_fixmate
from bio_mcp.tools.snake.bam_processing.samtools_markdup import samtools_markdup
from bio_mcp.tools.snake.bam_processing.samtools_view import samtools_view
from bio_mcp.tools.snake.bam_processing.samtools_index import samtools_index
from bio_mcp.tools.snake.variant_calling.freebayes import freebayes
from bio_mcp.tools.snake.variant_processing.bgzip import bgzip
from bio_mcp.tools.snake.variant_processing.tabix_index import tabix_index
from bio_mcp.tools.snake.variant_processing.bcftools_filter import bcftools_filter
from bio_mcp.tools.snake.variant_processing.bcftools_view import bcftools_view
from bio_mcp.tools.snake.variant_processing.bcftools_index import bcftools_index
from bio_mcp.tools.snake.variant_processing.vembrane_filter import vembrane_filter
from bio_mcp.tools.snake.reporting.compleasm_run import compleasm_run
from bio_mcp.tools.snake.reporting.compleasm_download import compleasm_download
from bio_mcp.tools.snake.assembly.metaspades import metaspades

from bio_mcp.tools.snake.variant_processing.picard_createsequencedictionary import (
    picard_createsequencedictionary,
)
from bio_mcp.tools.snake.variant_processing.picard_addorreplacereadgroups import (
    picard_addorreplacereadgroups,
)
from bio_mcp.tools.snake.bam_processing.sambamba_markdup import sambamba_markdup
from bio_mcp.tools.snake.variant_processing.gatk_baserecalibrator import (
    gatk_baserecalibrator,
)
from bio_mcp.tools.snake.variant_processing.gatk_applybqsr import gatk_applybqsr
from bio_mcp.tools.snake.variant_calling.gatk_haplotypecaller import (
    gatk_haplotypecaller,
)
from bio_mcp.tools.snake.variant_processing.bcftools_stats import bcftools_stats
from bio_mcp.tools.snake.reporting.mosdepth import mosdepth
from bio_mcp.tools.snake.reporting.picard_collecthsmetrics import (
    picard_collecthsmetrics,
)
from bio_mcp.tools.snake.kraken2.kraken2_download_library import (
    kraken2_download_library,
)
from bio_mcp.tools.snake.kraken2.kraken2_download_taxonomy import (
    kraken2_download_taxonomy,
)
from bio_mcp.tools.snake.kraken2.kraken2_build import kraken2_build
from bio_mcp.tools.snake.kraken2.kraken2_classify import kraken2_classify
from bio_mcp.tools.snake.metagenomics.kraken_biom import kraken_biom

from bio_mcp.tools.snake.quantification.salmon_index import salmon_index
from bio_mcp.tools.snake.quantification.salmon_quant import salmon_quant

from bio_mcp.tools.snake.preprocessing.fastp import fastp
from bio_mcp.tools.snake.alignment.minimap2_index import minimap2_index
from bio_mcp.tools.snake.alignment.minimap2_align import minimap2_align
from bio_mcp.tools.snake.bam_processing.samtools_fastq_separate import (
    samtools_fastq_separate,
)
from bio_mcp.tools.snake.assembly.megahit import megahit
from bio_mcp.tools.snake.extra.kaiju_classify import kaiju_classify
from bio_mcp.tools.snake.extra.kaiju2krona import kaiju2krona


@tool
def run_terminal_command(command: str) -> str:
    """
    Run a terminal command and return combined stdout and stderr output.

    Args:
        command (str): Command to run in the shell.

    Returns:
        str: Combined stdout and stderr output, or an error description.
    """
    import subprocess

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        combined_output = "\n".join(part for part in (stdout, stderr) if part)
        body = combined_output if combined_output else "<no output>"
        return f"Exit code {result.returncode}\n{body}"
    except Exception as error:
        return f"Unexpected error while running command: {error}"


tool_snpsift_casecontrol = tool(snpsift_casecontrol)
tool_snpsift_filter = tool(snpsift_filter)
tool_snpeff_download = tool(snpeff_download)
tool_snpeff_annotate = tool(snpeff_annotate)
tool_fastqc = tool(fastqc)
tool_trimmomatic = tool(trimmomatic)
tool_multiqc = tool(multiqc)
tool_star_index = tool(star_index)
tool_star_align = tool(star_align)
tool_deseq2_deseqdataset = tool(deseq2_deseqdataset)
tool_deseq2_deseq_wald = tool(deseq2_wald)
tool_fastp = tool(fastp)
tool_fastqc = tool(fastqc)
tool_multiqc = tool(multiqc)
tool_quast = tool(quast)
tool_bwa_mem2_index = tool(bwa_mem2_index)
tool_bwa_mem2_mem = tool(bwa_mem2_mem)
tool_samtools_faidx = tool(samtools_faidx)
tool_samtools_sort = tool(samtools_sort)
tool_samtools_fixmate = tool(samtools_fixmate)
tool_samtools_markdup = tool(samtools_markdup)
tool_samtools_view = tool(samtools_view)
tool_samtools_index = tool(samtools_index)
tool_freebayes = tool(freebayes)
tool_bgzip = tool(bgzip)
tool_tabix_index = tool(tabix_index)
tool_bcftools_filter = tool(bcftools_filter)
tool_bcftools_view = tool(bcftools_view)
tool_bcftools_index = tool(bcftools_index)
tool_vembrane_filter = tool(vembrane_filter)
tool_compleasm_run = tool(compleasm_run)
tool_compleasm_download = tool(compleasm_download)
tool_snpeff_download = tool(snpeff_download)
tool_snpeff_annotate = tool(snpeff_annotate)
tool_metaspades = tool(metaspades)
tool_fastqc = tool(fastqc)
tool_fastp = tool(fastp)
tool_multiqc = tool(multiqc)
tool_samtools_faidx = tool(samtools_faidx)
tool_picard_createsequencedictionary = tool(picard_createsequencedictionary)
tool_bwa_mem2_index = tool(bwa_mem2_index)
tool_bwa_mem2_mem = tool(bwa_mem2_mem)
tool_picard_addorreplacereadgroups = tool(picard_addorreplacereadgroups)
tool_sambamba_markdup = tool(sambamba_markdup)
tool_samtools_index = tool(samtools_index)
tool_gatk_baserecalibrator = tool(gatk_baserecalibrator)
tool_gatk_applybqsr = tool(gatk_applybqsr)
tool_gatk_haplotypecaller = tool(gatk_haplotypecaller)
tool_bcftools_index = tool(bcftools_index)
tool_bcftools_stats = tool(bcftools_stats)
tool_mosdepth = tool(mosdepth)
tool_picard_collecthsmetrics = tool(picard_collecthsmetrics)
tool_fastqc = tool(fastqc)
tool_trimmomatic = tool(trimmomatic)
tool_multiqc = tool(multiqc)
tool_metaspades = tool(metaspades)
tool_kraken2_download_library = tool(kraken2_download_library)
tool_kraken2_download_taxonomy = tool(kraken2_download_taxonomy)
tool_kraken2_build = tool(kraken2_build)
tool_kraken2_classify = tool(kraken2_classify)
tool_kraken_bio = tool(kraken_biom)
tool_salmon_index = tool(salmon_index)
tool_salmon_quant = tool(salmon_quant)
tool_fastp = tool(fastp)
tool_minimap2_index = tool(minimap2_index)
tool_minimap2_align = tool(minimap2_align)
tool_samtools_view = tool(samtools_view)
tool_samtools_fastq_separate = tool(samtools_fastq_separate)
tool_megahit = tool(megahit)
tool_kaiju_classify = tool(kaiju_classify)
tool_kaiju2krona = tool(kaiju2krona)


TASK_TOOL_MAPPING = {
    "cystic-fibrosis": [
        "tool_snpsift_casecontrol",
        "tool_snpsift_filter",
        "tool_snpeff_download",
        "tool_snpeff_annotate",
    ],
    "deseq": [
        "tool_fastqc",
        "tool_trimmomatic",
        "tool_multiqc",
        "tool_star_index",
        "tool_star_align",
        "tool_deseq2_deseqdataset",
        "tool_deseq2_deseq_wald",
    ],
    "evolution": [
        "tool_fastp",
        "tool_fastqc",
        "tool_multiqc",
        "tool_quast",
        "tool_bwa_mem2_index",
        "tool_bwa_mem2_mem",
        "tool_samtools_faidx",
        "tool_samtools_sort",
        "tool_samtools_fixmate",
        "tool_samtools_markdup",
        "tool_samtools_view",
        "tool_samtools_index",
        "tool_freebayes",
        "tool_bgzip",
        "tool_tabix_index",
        "tool_bcftools_filter",
        "tool_bcftools_view",
        "tool_bcftools_index",
        "tool_vembrane_filter",
        "tool_compleasm_run",
        "tool_compleasm_download",
        "tool_snpeff_download",
        "tool_snpeff_annotate",
        "tool_metaspades",
    ],
    "giab": [
        "tool_fastqc",
        "tool_fastp",
        "tool_multiqc",
        "tool_samtools_faidx",
        "tool_picard_createsequencedictionary",
        "tool_bwa_mem2_index",
        "tool_bwa_mem2_mem",
        "tool_picard_addorreplacegroups",
        "tool_sambamba_markdup",
        "tool_samtools_index",
        "tool_gatk_baserecalibrator",
        "tool_gatk_applybqsr",
        "tool_gatk_haplotypecaller",
        "tool_bcftools_index",
        "tool_bcftools_stats",
        "tool_mosdepth",
        "tool_picard_collecthsmetrics",
    ],
    "metagenomics": [
        "tool_fastqc",
        "tool_trimmomatic",
        "tool_multiqc",
        "tool_metaspades",
        "tool_kraken2_download_library",
        "tool_kraken2_download_taxonomy",
        "tool_kraken2_build",
        "tool_kraken2_classify",
        "tool_kraken_bio",
    ],
    "transcript-quant": [
        "tool_salmon_index",
        "tool_salmon_quant",
    ],
    "viral-metagenomics": [
        "tool_fastp",
        "tool_minimap2_index",
        "tool_minimap2_align",
        "tool_samtools_view",
        "tool_samtools_fastq_separate",
        "tool_megahit",
        "tool_kaiju_classify",
        "tool_kaiju2krona",
    ],
}

AVAILABLE_TOOLS_LIST = [
    obj
    for name, obj in globals().items()
    if name.startswith("tool_")
    or (hasattr(obj, "__name__") and obj.__name__ == "run_terminal_command")
]
