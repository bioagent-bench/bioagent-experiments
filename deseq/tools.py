from smolagents import tool
from typing import Dict, Union
import subprocess
from pathlib import Path


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
def fasterq_dump_tool(sra_file: str, output_dir: str) -> Dict[str, Union[str, int]]:
    """
    Convert SRA files to FASTQ format using fasterq-dump.

    Args:
        sra_file: Path to the SRA file
        output_dir: Directory where FASTQ results will be stored

    Returns:
        Dictionary containing stdout, stderr, and return code
    """
    result = subprocess.run(
        ["fasterq-dump", sra_file, "-O", output_dir], capture_output=True, text=True
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


@tool
def trimmomatic_tool(
    input_file_1: str, input_file_2: str, output_prefix: str
) -> Dict[str, Union[str, int]]:
    """
    Perform trimming on paired-end FASTQ files using Trimmomatic.

    Args:
        input_file_1: Path to the first FASTQ file
        input_file_2: Path to the second FASTQ file
        output_prefix: Prefix for output files

    Returns:
        Dictionary containing stdout, stderr, and return code
    """
    result = subprocess.run(
        [
            "trimmomatic",
            "PE",
            "-threads",
            "32",
            input_file_1,
            input_file_2,
            f"{output_prefix}_1P.fastq",
            f"{output_prefix}_1U.fastq",
            f"{output_prefix}_2P.fastq",
            f"{output_prefix}_2U.fastq",
            "LEADING:20",
            "TRAILING:20",
            "SLIDINGWINDOW:4:20",
            "MINLEN:50",
        ],
        capture_output=True,
        text=True,
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


@tool
def star_tool(
    genome_dir: str,
    gtf_file: str,
    read_file_1: str,
    read_file_2: str,
    output_prefix: str,
) -> Dict[str, Union[str, int]]:
    """
    Map reads to the genome using STAR.

    Args:
        genome_dir: Directory containing the genome index
        gtf_file: Path to the GTF file
        read_file_1: Path to the first FASTQ file
        read_file_2: Path to the second FASTQ file
        output_prefix: Prefix for output files

    Returns:
        Dictionary containing stdout, stderr, and return code
    """
    result = subprocess.run(
        [
            "STAR",
            "--runThreadN",
            "32",
            "--genomeDir",
            genome_dir,
            "--sjdbGTFfile",
            gtf_file,
            "--readFilesIn",
            read_file_1,
            read_file_2,
            "--outFileNamePrefix",
            output_prefix,
            "--outSAMtype",
            "BAM",
            "SortedByCoordinate",
            "--limitBAMsortRAM",
            "100000000000",
            "--quantMode",
            "GeneCounts",
        ],
        capture_output=True,
        text=True,
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


# list all the tools to be able to export them
bioinformatics_tools = [
    fastqc_tool,
    read_directory_tool,
    fasterq_dump_tool,
    trimmomatic_tool,
    star_tool,
]
