from smolagents import tool
from typing import Dict, Union
import subprocess
from pathlib import Path
from rpy2 import robjects
from rpy2.robjects import pandas2ri


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
    # Convert Python objects to R
    r = robjects.r
    r_pairs = pandas2ri.py2rpy(pairs)
    r_cog_sets = robjects.ListVector(cog_sets)

    # Call R's ExtractBy function
    sequences = r("""
        function(pairs, db_path, cog_sets) {
            ExtractBy(x=pairs, y=db_path, z=cog_sets[lengths(cog_sets) >= 4], Verbose=TRUE)
        }
    """)(r_pairs, db_path, r_cog_sets)

    return robjects.conversion.rpy2py(sequences)


@tool
def build_phylogenetic_trees_tool(matched_sequences: Dict) -> Dict[str, Any]:
    """
    Build phylogenetic trees for each COG.

    Args:
        matched_sequences: Dictionary of matched COG sequences

    Returns:
        Dictionary containing phylogenetic trees
    """
    r = robjects.r
    r_matched_sequences = robjects.ListVector(matched_sequences)

    # Define and call R function
    cog_trees = r("""
        function(matched_sequences) {
            cog_trees <- list()
            for (i in seq_along(matched_sequences)) {
                aligned_cog <- AlignTranslation(matched_sequences[[i]])
                cog_trees[[i]] <- TreeLine(
                    aligned_cog, 
                    method="ML", 
                    reconstruct=TRUE, 
                    maxTime=0.05, 
                    processors=NULL
                )
                print(paste("Completed tree", i, "of", length(matched_sequences)))
            }
            return(cog_trees)
        }
    """)(r_matched_sequences)

    return robjects.conversion.rpy2py(cog_trees)


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
    r = robjects.r
    r_matched_sequences = robjects.ListVector(matched_sequences)
    r_training_set = robjects.conversion.py2rpy(training_set)

    # Define and call R function
    cogs_annot = r("""
        function(matched_sequences, training_set) {
            cogs_annot <- list()
            for (i in seq_along(matched_sequences)) {
                cogs_annot[[i]] <- IdTaxa(matched_sequences[[i]], training_set, processors=NULL)
                print(paste("Completed annotation", i, "of", length(matched_sequences)))
            }
            return(cogs_annot)
        }
    """)(r_matched_sequences, r_training_set)

    return robjects.conversion.rpy2py(cogs_annot)


bioinformatics_tools = [
    read_directory_tool,
    extract_sequences_tool,
    build_phylogenetic_trees_tool,
    annotate_sequences_tool,
]
