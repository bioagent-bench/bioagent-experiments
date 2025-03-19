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
def extract_sequences_tool(
    pairs: List[Dict], db_path: str, cog_sets: List
) -> Dict[str, Any]:
    """
    Extract sequences for COGs with at least a specified number of orthologs.

    Args:
        pairs: List of pairs of genes
        db_path: Path to the database
        cog_sets: List of COG sets to filter

    Returns:
        Dictionary containing extracted sequences
    """
    sequences = ExtractBy(
        x=pairs, y=db_path, z=cog_sets[lengths(cog_sets) >= 4], Verbose=True
    )
    return sequences


@tool
def build_phylogenetic_trees_tool(matched_sequences: Dict) -> Dict[str, Any]:
    """
    Build phylogenetic trees for each COG.

    Args:
        matched_sequences: Dictionary of matched COG sequences

    Returns:
        Dictionary containing phylogenetic trees
    """
    cog_trees = {}
    for i, current_cog in enumerate(matched_sequences):
        aligned_cog = AlignTranslation(current_cog)
        cog_trees[i] = TreeLine(
            aligned_cog, method="ML", reconstruct=True, maxTime=0.05, processors=None
        )
        print(f"Completed tree {i + 1} of {len(matched_sequences)}")
    return cog_trees


@tool
def annotate_sequences_tool(
    matched_sequences: Dict, training_set: Any
) -> Dict[str, Any]:
    """
    Annotate sequences functionally.

    Args:
        matched_sequences: Dictionary of matched COG sequences
        training_set: Loaded training set for taxonomic classification

    Returns:
        Dictionary containing annotations for each sequence
    """
    cogs_annot = {}
    for i, current_protein in enumerate(matched_sequences):
        cogs_annot[i] = IdTaxa(current_protein, training_set, processors=None)
        print(f"Completed annotation {i + 1} of {len(matched_sequences)}")
    return cogs_annot


bioinformatics_tools = [
    read_directory_tool,
    extract_sequences_tool,
    build_phylogenetic_trees_tool,
    annotate_sequences_tool,
]
