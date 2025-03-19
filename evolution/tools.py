from smolagents import tool
from typing import Dict, Union
import subprocess
from pathlib import Path


@tool
def read_directory_tool(directory_path: str) -> str:
    """
    List files and directories within a specified directory path.

    Args:
        directory_path: Path to the directory to read

    Returns:
        String representation of the files in the directory with paths relative to working directory
    """
    try:
        path = Path(directory_path)
        contents = list(path.iterdir())

        prefix = (
            f"/{directory_path}"
            if not directory_path.startswith("/")
            else directory_path
        )
        prefix = prefix.rstrip("/")

        files = [f"{prefix}/{item.name}" for item in contents if item.is_file()]
        directories = [f"{prefix}/{item.name}/" for item in contents if item.is_dir()]

        return str(files + directories)
    except Exception as e:
        return str(e)


@tool
def fastqc_tool(input_file: str, output_dir: str) -> Dict[str, Union[str, int]]:
    """
    Perform quality control using FastQC on FASTQ files.

    Args:
        input_file: Path to the FASTQ file
        output_dir: Directory where FastQC results will be stored

    Returns:
        Dictionary containing stdout, stderr, and return code
    """
    result = subprocess.run(
        ["fastqc", input_file, "-o", output_dir], capture_output=True, text=True
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


@tool
def run_fastp(input_files: str, output_files: str, html: str, json: str) -> str:
    """
    Run FastP for quality control on paired-end FASTQ files.

    Args:
        input_files: Input FASTQ files
        output_files: Output trimmed FASTQ files
        html: Output HTML report
        json: Output JSON report

    Returns:
        Output from the FastP command
    """
    result = subprocess.run(
        [
            "fastp",
            "--detect_adapter_for_pe",
            "--overrepresentation_analysis",
            "--correction",
            "--cut_right",
            "--thread",
            "60",
            "--html",
            html,
            "--json",
            json,
        ]
        + input_files.split()
        + output_files.split(),
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


@tool
def run_fastqc(input_files: str, output_dir: str) -> str:
    """
    Run FastQC on FASTQ files.

    Args:
        input_files: Input FASTQ files
        output_dir: Directory to store FastQC results

    Returns:
        Output from the FastQC command
    """
    result = subprocess.run(
        ["fastqc", "-o", output_dir] + input_files.split(),
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


@tool
def run_spades(read1: str, read2: str, output_dir: str) -> str:
    """
    Run SPAdes for genome assembly.

    Args:
        read1: First read file
        read2: Second read file
        output_dir: Directory to store SPAdes results

    Returns:
        Output from the SPAdes command
    """
    result = subprocess.run(
        ["spades.py", "-o", output_dir, "--careful", "-1", read1, "-2", read2],
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


@tool
def run_quast(assembly: str, output_dir: str) -> str:
    """
    Run QUAST to check assembly quality.

    Args:
        assembly: Assembly file
        output_dir: Directory to store QUAST results

    Returns:
        Output from the QUAST command
    """
    result = subprocess.run(
        ["quast", "-o", output_dir, assembly], capture_output=True, text=True
    )
    return result.stdout + result.stderr


@tool
def run_bwa_index(reference: str) -> str:
    """
    Index a reference genome using BWA.

    Args:
        reference: Reference genome file

    Returns:
        Output from the BWA index command
    """
    result = subprocess.run(["bwa", "index", reference], capture_output=True, text=True)
    return result.stdout + result.stderr


@tool
def run_bwa_mem(reference: str, read1: str, read2: str) -> str:
    """
    Run BWA MEM for read mapping.

    Args:
        reference: Reference genome file
        read1: First read file
        read2: Second read file

    Returns:
        Output from the BWA MEM command
    """
    result = subprocess.run(
        ["bwa", "mem", reference, read1, read2], capture_output=True, text=True
    )
    return result.stdout + result.stderr


@tool
def run_samtools_sort(input_bam: str, output_bam: str) -> str:
    """
    Sort a BAM file using Samtools.

    Args:
        input_bam: Input BAM file
        output_bam: Output sorted BAM file

    Returns:
        Output from the Samtools sort command
    """
    result = subprocess.run(
        ["samtools", "sort", "-o", output_bam, input_bam],
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


@tool
def run_freebayes(reference: str, bam_file: str) -> str:
    """
    Call variants using Freebayes.

    Args:
        reference: Reference genome file
        bam_file: Input BAM file

    Returns:
        Output from the Freebayes command
    """
    result = subprocess.run(
        ["freebayes", "-f", reference, bam_file], capture_output=True, text=True
    )
    return result.stdout + result.stderr


@tool
def run_prokka(input_file: str, output_dir: str) -> str:
    """
    Annotate a genome using Prokka.

    Args:
        input_file: Input assembly file
        output_dir: Directory to store Prokka results

    Returns:
        Output from the Prokka command
    """
    result = subprocess.run(
        ["prokka", "--outdir", output_dir, input_file], capture_output=True, text=True
    )
    return result.stdout + result.stderr
