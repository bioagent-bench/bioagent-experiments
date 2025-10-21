from typing import Any, Callable, Collection, Sequence

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

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)


@tool
def run_terminal_command(command: str) -> str:
    """Run a terminal command and return combined stdout and stderr output.

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


ToolFn = Callable[..., Any]
ToolSpec = ToolFn | str


class ToolRegistry:
    """Centralise tool definitions and per-task tool selections."""

    def __init__(self, default_tools: Sequence[ToolSpec] | None = None) -> None:
        self._tools_by_name: dict[str, ToolFn] = {}
        self._task_tools: dict[str, list[str]] = {}
        self._default_tool_names: list[str] = []

        if default_tools is not None:
            for tool_spec in default_tools:
                tool_fn = self._resolve_tool_spec(tool_spec)
                name = self.register_tool(tool_fn)
                self._default_tool_names.append(name)

    def _resolve_tool_spec(self, tool_spec: ToolSpec) -> ToolFn:
        if isinstance(tool_spec, str):
            candidate = globals().get(tool_spec)
            if candidate is None:
                raise KeyError(f"Default tool '{tool_spec}' is not defined.")
            if not callable(candidate):
                raise TypeError(
                    f"Default tool '{tool_spec}' must resolve to a callable, got {type(candidate)!r}."
                )
            return candidate
        return tool_spec

    def register_tool(self, tool_fn: ToolFn, *, name: str | None = None) -> str:
        key = name
        self._tools_by_name[key] = tool_fn
        return key

    def register_task_tools(
        self,
        task_id: str,
        tool_names: Collection[str] | None,
        *,
        extra_tool_names: Collection[str] | None = None,
    ) -> None:
        names = []
        if tool_names:
            names.extend(tool_names)
        if extra_tool_names:
            names.extend(extra_tool_names)
        self._task_tools[task_id] = names

    def task_tool_names(self, task_id: str) -> list[str]:
        return self._task_tools.get(task_id, []).copy()

    def tool_names_for_task(
        self,
        task_id: str,
        *,
        include_default_tools: bool = True,
        extra_tool_names: Collection[str] | None = None,
    ) -> list[str]:
        names = self.task_tool_names(task_id)
        if include_default_tools:
            for default_tool in self._default_tool_names:
                if default_tool not in names:
                    names.append(default_tool)
        if extra_tool_names:
            for tool_name in extra_tool_names:
                if tool_name not in names:
                    names.append(tool_name)
        return names

    def resolve_tools(self, tool_names: Collection[str]) -> list[ToolFn]:
        tools: list[ToolFn] = []
        for name in tool_names:
            tool = self._tools_by_name.get(name)
            if tool is None:
                raise KeyError(f"Unknown tool requested: {name}")
            tools.append(tool)
        return tools


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
tool_metaspades = tool(metaspades)
tool_picard_createsequencedictionary = tool(picard_createsequencedictionary)
tool_picard_addorreplacereadgroups = tool(picard_addorreplacereadgroups)
tool_sambamba_markdup = tool(sambamba_markdup)
tool_gatk_baserecalibrator = tool(gatk_baserecalibrator)
tool_gatk_applybqsr = tool(gatk_applybqsr)
tool_gatk_haplotypecaller = tool(gatk_haplotypecaller)
tool_bcftools_stats = tool(bcftools_stats)
tool_mosdepth = tool(mosdepth)
tool_picard_collecthsmetrics = tool(picard_collecthsmetrics)
tool_kraken2_download_library = tool(kraken2_download_library)
tool_kraken2_download_taxonomy = tool(kraken2_download_taxonomy)
tool_kraken2_build = tool(kraken2_build)
tool_kraken2_classify = tool(kraken2_classify)
tool_kraken_bio = tool(kraken_biom)
tool_salmon_index = tool(salmon_index)
tool_salmon_quant = tool(salmon_quant)
tool_minimap2_index = tool(minimap2_index)
tool_minimap2_align = tool(minimap2_align)
tool_samtools_fastq_separate = tool(samtools_fastq_separate)
tool_megahit = tool(megahit)
tool_kaiju_classify = tool(kaiju_classify)
tool_kaiju2krona = tool(kaiju2krona)


REGISTRY = ToolRegistry(default_tools=[run_terminal_command])

for name, obj in list(globals().items()):
    if name.startswith("tool_"):
        REGISTRY.register_tool(obj, name=name)


REGISTRY.register_task_tools(
    "cystic-fibrosis",
    [
        "tool_snpsift_casecontrol",
        "tool_snpsift_filter",
        "tool_snpeff_download",
        "tool_snpeff_annotate",
    ],
)
REGISTRY.register_task_tools(
    "deseq",
    [
        "tool_fastqc",
        "tool_trimmomatic",
        "tool_multiqc",
        "tool_star_index",
        "tool_star_align",
        "tool_deseq2_deseqdataset",
        "tool_deseq2_deseq_wald",
    ],
)
REGISTRY.register_task_tools(
    "evolution",
    [
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
)
REGISTRY.register_task_tools(
    "giab",
    [
        "tool_fastqc",
        "tool_fastp",
        "tool_multiqc",
        "tool_samtools_faidx",
        "tool_picard_createsequencedictionary",
        "tool_bwa_mem2_index",
        "tool_bwa_mem2_mem",
        "tool_picard_addorreplacereadgroups",
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
)
REGISTRY.register_task_tools(
    "metagenomics",
    [
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
)
REGISTRY.register_task_tools(
    "transcript-quant",
    [
        "tool_salmon_index",
        "tool_salmon_quant",
    ],
)
REGISTRY.register_task_tools(
    "viral-metagenomics",
    [
        "tool_fastp",
        "tool_minimap2_index",
        "tool_minimap2_align",
        "tool_samtools_view",
        "tool_samtools_fastq_separate",
        "tool_megahit",
        "tool_kaiju_classify",
        "tool_kaiju2krona",
    ],
)

if __name__ == "__main__":
    result = tool_fastqc(
        fastq="/home/dionizije/bioinformatics-mcp/bio_wrappers/fastqc/test/reads/a.fastq",
        html="~/html.html",
        zip="~/zip.zip",
    )
    print(result)