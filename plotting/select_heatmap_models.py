import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_INPUTS = [
    "results/may-24/data/all_models_task_completion_rate.csv",
    "results/data/all_models_task_completion_rate.csv",
]

SELECTED_MODELS = [
    "openrouter/openai/gpt-5.5",
    "openrouter/anthropic/claude-opus-4.7",
    "openrouter/google/gemini-3-pro-preview",
    "openrouter/qwen/qwen3.7-max",
    "openrouter/qwen/qwen3.6-27b",
]

DEFAULT_TASK_LIMIT = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a selected-model output folder and render the heatmap-bar plot."
    )
    parser.add_argument(
        "--input-csv",
        action="append",
        default=[],
        help="Task-by-model completion CSV. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/selected-heatmap-models",
        help="New folder for selected-model data and figures.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Only write selected-model CSVs; do not render heatmap_bar.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Task ID to include. Can be provided multiple times. Defaults to all tasks.",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=DEFAULT_TASK_LIMIT,
        help="Number of tasks to include when --task-id is not provided. Defaults to all tasks.",
    )
    return parser.parse_args()


def read_sources(paths: list[str]) -> list[tuple[Path, pd.DataFrame]]:
    sources = []
    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        df = pd.read_csv(path).set_index("task_id")
        sources.append((path, df))
    return sources


def select_models(sources: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    selected = []
    missing = []

    for model in SELECTED_MODELS:
        for _, df in sources:
            if model in df.columns:
                selected.append(df[[model]])
                break
        else:
            missing.append(model)

    if missing:
        raise ValueError("Missing selected model(s): " + ", ".join(missing))

    selected_df = pd.concat(selected, axis=1)
    selected_df.index.name = "task_id"
    return selected_df.sort_index()


def select_tasks(selected_df: pd.DataFrame, task_ids: list[str], task_limit: int) -> pd.DataFrame:
    if task_ids:
        missing = [task_id for task_id in task_ids if task_id not in selected_df.index]
        if missing:
            raise ValueError("Missing selected task(s): " + ", ".join(missing))
        return selected_df.loc[task_ids]

    if task_limit < 0:
        raise ValueError("--task-limit must be non-negative")
    if task_limit == 0:
        return selected_df
    return selected_df.head(task_limit)


def write_outputs(selected_df: pd.DataFrame, output_dir: Path) -> Path:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    selected_csv = data_dir / "selected_models_task_completion_rate.csv"
    selected_df.to_csv(selected_csv)

    model_summary = (
        selected_df.mean(axis=0, skipna=True)
        .sort_values(ascending=False)
        .rename("completion_rate")
        .reset_index()
        .rename(columns={"index": "model"})
    )
    model_summary.to_csv(data_dir / "selected_models_completion_rate.csv", index=False)
    return selected_csv


def render_plot(selected_csv: Path, output_dir: Path):
    figures_dir = output_dir / "figures"
    command = [
        sys.executable,
        "plotting/heatmap_bar.py",
        "--input-csv",
        str(selected_csv),
        "--output-dir",
        str(figures_dir),
        "--models",
        *SELECTED_MODELS,
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    args = parse_args()

    input_paths = args.input_csv or DEFAULT_INPUTS
    output_dir = Path(args.output_dir)
    sources = read_sources(input_paths)
    selected_df = select_models(sources)
    selected_df = select_tasks(selected_df, args.task_id, args.task_limit)
    selected_csv = write_outputs(selected_df, output_dir)

    if not args.skip_plot:
        render_plot(selected_csv, output_dir)

    print(f"Wrote selected model outputs to {output_dir}")
