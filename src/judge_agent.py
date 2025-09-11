import os
import json
from pathlib import Path


def parse_outputs(output_dir):
    """Return a list of all files and folders in the directory."""
    return list(Path(output_dir).rglob("*"))


def parse_results(results_dir):
    pass


def build_judge_agent():
    pass

