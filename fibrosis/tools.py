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
def run_snpEff(input_vcf: str) -> str:
    """
    Run SnpEff on the provided VCF file.

    Args:
        input_vcf: Input VCF file

    Returns:
        Output from the SnpEff command
    """
    result = subprocess.run(
        [
            "snpEff",
            "-Xmx12g",
            "-v",
            "-lof",
            "-motif",
            "-nextProt",
            "GRCh37.75",
            input_vcf,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout + result.stderr


@tool
def run_snpSift(input_vcf: str) -> str:
    """
    Run SnpSift on the provided VCF file.

    Args:
        input_vcf: Input VCF file

    Returns:
        Output from the SnpSift command
    """
    result = subprocess.run(
        [
            "SnpSift",
            "-Xmx100g",
            "caseControl",
            "-v",
            "-tfam",
            "data/pedigree.tfam",
            input_vcf,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    with open("data/ex1.eff.cc.vcf", "w") as output_file:
        output_file.write(result.stdout)
    return result.stdout


@tool
def filter_vcf(input_vcf: str) -> str:
    """
    Filter the VCF file based on specific criteria.

    Args:
        input_vcf: Input VCF file

    Returns:
        Output from the filtering process
    """
    result = subprocess.run(
        [
            "cat",
            input_vcf,
            "|",
            "SnpSift",
            "filter",
            "(Cases[0] = 3) & (Controls[0] = 0) & ((EFF[*].IMPACT = 'HIGH') | (EFF[*].IMPACT = 'MODERATE'))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    with open("data/ex1.filtered.vcf", "w") as output_file:
        output_file.write(result.stdout)
    return result.stdout


@tool
def annotate_vcf(input_vcf: str) -> str:
    """
    Annotate the VCF file with ClinVar data.

    Args:
        input_vcf: Input VCF file

    Returns:
        Output from the annotation process
    """
    result = subprocess.run(
        ["SnpSift", "-Xmx100g", "annotate", "-v", "data/clinvar_2025.vcf", input_vcf],
        capture_output=True,
        text=True,
        check=True,
    )
    with open("data/ex1.eff.cc.clinvar.vcf", "w") as output_file:
        output_file.write(result.stdout)
    return result.stdout


@tool
def filter_clinvar_vcf(input_vcf: str) -> str:
    """
    Filter the annotated VCF file based on specific criteria.

    Args:
        input_vcf: Input VCF file

    Returns:
        Output from the filtering process
    """
    result = subprocess.run(
        [
            "cat",
            input_vcf,
            "|",
            "SnpSift",
            "filter",
            "(GENEINFO[*] =~ 'CFTR') & ((EFF[*].IMPACT = 'HIGH') | (EFF[*].IMPACT = 'MODERATE')) & (Cases[0] = 3) & (Controls[0] = 0)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    with open("output_cf_variant.txt", "w") as output_file:
        output_file.write(result.stdout)
    return result.stdout


bioinformatics_tools = [
    read_directory_tool,
    run_snpEff,
    run_snpSift,
    filter_vcf,
    annotate_vcf,
    filter_clinvar_vcf,
]
