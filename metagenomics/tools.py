from smolagents import tool
from typing import Dict, Union
import subprocess
from pathlib import Path
import os


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
def run_trimmomatic(input_file_r1: str, input_file_r2: str, output_prefix: str) -> str:
    """Run Trimmomatic on paired-end FASTQ files.

    Args:
        input_file_r1: Path to the first input FASTQ file
        input_file_r2: Path to the second input FASTQ file
        output_prefix: Prefix for output files
    """
    try:
        result = subprocess.run(
            [
                "trimmomatic",
                "PE",
                "-threads",
                "32",
                input_file_r1,
                input_file_r2,
                f"{output_prefix}_1P.fastq",
                f"{output_prefix}_1U.fastq",
                f"{output_prefix}_2P.fastq",
                f"{output_prefix}_2U.fastq",
                "SLIDINGWINDOW:4:20",
                "MINLEN:35",
                "ILLUMINACLIP:data/seq/TruSeq3-PE.fa:2:40:15",
            ],
            capture_output=True,
            text=True,
        )
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)


@tool
def run_kraken(
    input_file: str, db_path: str, output_file: str, report_file: str
) -> str:
    """Run Kraken2 for taxonomic classification.

    Args:
        input_file: Path to the input FASTQ file
        db_path: Path to the Kraken database
        output_file: Path to the output file for Kraken results
        report_file: Path to the report file for Kraken results
    """
    try:
        result = subprocess.run(
            [
                "kraken2",
                "--db",
                db_path,
                "--threads",
                "32",
                input_file,
                "--output",
                output_file,
                "--report",
                report_file,
            ],
            capture_output=True,
            text=True,
        )
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)


@tool
def run_kraken_biom(report_files: list, output_file: str) -> str:
    """Generate a BIOM file from Kraken reports.

    Args:
        report_files: List of paths to Kraken report files
        output_file: Path to the output BIOM file
    """
    try:
        result = subprocess.run(
            ["kraken-biom"] + report_files + ["--fmt", "json", "-o", output_file],
            capture_output=True,
            text=True,
        )
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)


bioinformatics_tools = [
    read_directory_tool,
    run_fastqc,
    run_trimmomatic,
    run_kraken,
    run_kraken_biom,
]
