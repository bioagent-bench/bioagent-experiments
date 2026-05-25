import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_MODELS = [
    'gpt-5-1-codex-max',
    'gpt-5-2',
    'gpt-gemini',
    'gpt-opus',
    'gpt-sonnet',
    'openrouter/mistralai/devstral-2512:free',
    'openrouter/moonshotai/kimi-k2-thinking',
    'openrouter/qwen/qwen3-coder:free',
    'openrouter/z-ai/glm-4.7',
    'openrouter/minimax/minimax-m2.1',
]

MODEL_LABELS = {
    'gpt-5-1-codex-max': 'gpt-5-1',
    'gpt-5-2': 'gpt-5-2',
    'gpt-gemini': 'gemini-3-pro-preview',
    'gpt-opus': 'claude-opus-4-5',
    'gpt-sonnet': 'claude-sonnet-4-5',
    'openrouter/mistralai/devstral-2512:free': 'devstral-2512',
    'openrouter/moonshotai/kimi-k2-thinking': 'kimi-k2-thinking',
    'openrouter/qwen/qwen3-coder:free': 'qwen3-coder',
    'openrouter/z-ai/glm-4.7': 'glm-4.7',
    'openrouter/minimax/minimax-m2.1': 'minimax-m2.1',
    'openrouter/openai/gpt-5.5': 'gpt-5.5',
    'openrouter/openai/gpt-5.1': 'gpt-5.1',
    'openrouter/moonshotai/kimi-k2.6': 'kimi-k2.6',
    'openrouter/anthropic/claude-opus-4.7': 'claude-opus-4.7',
    'openrouter/deepseek/deepseek-v4-pro': 'deepseek-v4-pro',
    'openrouter/qwen/qwen3.7-max': 'qwen3.7-max',
    'openrouter/qwen/qwen3.6-27b': 'qwen3.6-27b',
    'openrouter/anthropic/claude-opus-4.5': 'claude-opus-4.5',
    'openrouter/google/gemini-3-pro-preview': 'gemini-3-pro-preview',
}


def parse_args():
    parser = argparse.ArgumentParser(description="Render task completion heatmap with average bars.")
    parser.add_argument(
        "--input-csv",
        default="results/data/all_models_task_completion_rate.csv",
        help="Task-by-model completion-rate CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/figures",
        help="Directory for heatmap_bar.png and heatmap_bar.pdf.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional model IDs to plot. Defaults to the paper model set when present, otherwise all columns.",
    )
    parser.add_argument(
        "--rows",
        choices=["models", "tasks"],
        default="models",
        help="Whether heatmap rows should be models or tasks.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot window after saving.")
    return parser.parse_args()


def display_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.split("/")[-1] or model)


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    df.set_index("task_id", inplace=True)

    if args.models:
        best_models = args.models
    else:
        default_models = [model for model in DEFAULT_MODELS if model in df.columns]
        best_models = default_models or df.columns.tolist()

    missing_models = [model for model in best_models if model not in df.columns]
    if missing_models:
        print(f"Skipping missing models: {', '.join(missing_models)}")
    best_models = [model for model in best_models if model in df.columns]
    if not best_models:
        raise ValueError("No requested models were found in the input CSV.")

    df = df[best_models]
    if args.rows == "models":
        df = df.T
        averages = df.mean(axis=1)

        # Sort models by their average completion rate (descending) and re-index df/averages accordingly
        sorted_idx = averages.sort_values(ascending=False, kind="mergesort").index
        df = df.loc[sorted_idx]
        averages = averages.loc[sorted_idx]
        row_labels = [display_label(model) for model in df.index]
        x_label = "Task"
        y_label = "Model"
    else:
        averages = df.mean(axis=1)
        row_labels = df.index.tolist()
        df.columns = [display_label(model) for model in df.columns]
        x_label = "Model"
        y_label = "Task"

    df.index = row_labels

    print(df)

    # Custom colormap: mild orange to blue
    colors = ['#FFE5D9', '#FFC9B3', '#FFFFFF', '#B3D4FF', '#80BFFF']
    cmap_custom = LinearSegmentedColormap.from_list('orange_blue', colors, N=100)

    row_count, column_count = df.shape
    axis_fontsize = 24
    tick_fontsize = 22 if max(row_count, column_count) <= 9 else 19
    annot_fontsize = 20 if max(row_count, column_count) <= 9 else 16
    bar_value_fontsize = 22 if row_count <= 10 else 18

    # Create figure with 2 subplots: heatmap on left, horizontal bar chart on right
    fig_width = max(9, 1.08 * column_count + 3.8)
    fig_height = max(6, 0.56 * row_count + 2.4)
    fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(fig_width, fig_height),
                                           gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.01})

    # Plot heatmap
    sns.heatmap(df, annot=True, fmt=".0f", cmap=cmap_custom, ax=ax_heat,
                cbar=False,
                linewidths=0.5, linecolor='white',
                annot_kws={'fontsize': annot_fontsize, 'fontfamily': 'Linux Libertine O', 'color': 'black'},
                square=False,
                vmin=0, vmax=100)

    # Style heatmap
    ax_heat.set_xlabel(x_label, fontsize=axis_fontsize, fontfamily="Linux Libertine O", color='black')
    ax_heat.set_ylabel(y_label, fontsize=axis_fontsize, fontfamily="Linux Libertine O", color='black')

    # Set tick labels
    ax_heat.set_xticklabels(df.columns, fontsize=tick_fontsize, fontfamily="Linux Libertine O", rotation=45, ha='right', color='black')
    ax_heat.set_yticklabels(df.index, fontsize=tick_fontsize, fontfamily="Linux Libertine O", rotation=0, color='black')
    ax_heat.tick_params(colors='black', length=0)

    # Plot horizontal bar chart - bars aligned with heatmap rows
    bar_positions = np.arange(row_count) + 0.5  # Center bars on heatmap cells

    # Color bars based on average using the same colormap
    max_avg = 100
    bar_colors = [cmap_custom(val / max_avg) for val in averages.values]

    bars = ax_bar.barh(bar_positions, averages.values, height=0.7, color=bar_colors, edgecolor='white')

    # Style bar chart - remove bounding box, keep only axis lines
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_visible(True)
    ax_bar.spines['left'].set_color('black')
    ax_bar.spines['bottom'].set_color('black')
    ax_bar.set_ylim(ax_heat.get_ylim())
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Average", fontsize=axis_fontsize, fontfamily="Linux Libertine O", color='black')
    ax_bar.set_xlim(0, 100)
    ax_bar.tick_params(axis='x', labelsize=tick_fontsize, colors='black')
    for label in ax_bar.get_xticklabels():
        label.set_fontfamily('Linux Libertine O')
        label.set_color('black')

    # Add value labels at end of bars
    for bar, val in zip(bars, averages.values):
        ax_bar.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}', ha='left', va='center', fontsize=bar_value_fontsize, fontfamily='Linux Libertine O', color='black')

    plt.tight_layout()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'heatmap_bar.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'heatmap_bar.pdf', dpi=300, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close(fig)
