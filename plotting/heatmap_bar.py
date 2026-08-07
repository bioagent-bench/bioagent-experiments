import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_DIR = ASSETS_DIR / "logos"
TASK_ICON_DIR = ASSETS_DIR / "tasks"
FONT_DIR = ASSETS_DIR / "fonts"
# Static Figtree instances by weight (matplotlib only reads the variable
# font's Light default, so we ship pre-instanced weights).
FONT_WEIGHTS = {
    "medium": FONT_DIR / "Figtree-Medium.ttf",
    "semibold": FONT_DIR / "Figtree-SemiBold.ttf",
}
FONT_FALLBACK = FONT_DIR / "Figtree.ttf"

# Color anchoring a 100% completion rate; the cell gradient runs white -> this.
FULL_RATE_COLOR = "#13682A"

# Font sizes, shared with plotting/scatter_plan_completion.py so the two
# figures are typographically consistent. TICK_FONTSIZE is the value used for
# the 10-model/10-task paper figures; smaller grids bump up (see below).
AXIS_FONTSIZE = 29
TICK_FONTSIZE = 23


DEFAULT_MODELS = [
    'gpt-5-1-codex-max',
    'gpt-5-2',
    'gpt-gemini',
    'gpt-opus',
    'gpt-sonnet',
    'openrouter/mistralai/devstral-2512:free',
    'openrouter/moonshotai/kimi-k2-thinking',
    'openrouter/qwen/qwen3-coder:free',
    'gpt-glm',
    'openrouter/minimax/minimax-m2.1',
]

# Human-facing model names (real product names, not run identifiers).
MODEL_LABELS = {
    'gpt-5-1': 'GPT-5.1',
    'gpt-5-1-codex-max': 'GPT-5.1-Codex-Max',
    'gpt-5-2': 'GPT-5.2',
    'gpt-gemini': 'Gemini 3 Pro',
    'gpt-opus': 'Claude Opus 4.5',
    'gpt-sonnet': 'Claude Sonnet 4.5',
    'openrouter/mistralai/devstral-2512:free': 'Devstral 2512',
    'openrouter/moonshotai/kimi-k2-thinking': 'Kimi K2 Thinking',
    'openrouter/qwen/qwen3-coder:free': 'Qwen3 Coder',
    'openrouter/z-ai/glm-4.7': 'GLM-4.7',
    'gpt-glm': 'GLM-4.7',
    'openrouter/minimax/minimax-m2.1': 'MiniMax M2.1',
    'openrouter/openai/gpt-5.5': 'GPT-5.5',
    'openrouter/openai/gpt-5.2': 'GPT-5.2',
    'openrouter/openai/gpt-5.1': 'GPT-5.1',
    'openrouter/moonshotai/kimi-k2.6': 'Kimi K2.6',
    'openrouter/anthropic/claude-opus-4.7': 'Claude Opus 4.7',
    'openrouter/deepseek/deepseek-v4-pro': 'DeepSeek V4 Pro',
    'openrouter/qwen/qwen3.7-max': 'Qwen3.7 Max',
    'openrouter/qwen/qwen3.6-27b': 'Qwen3.6 27B',
    'openrouter/anthropic/claude-opus-4.5': 'Claude Opus 4.5',
    'openrouter/google/gemini-3-pro-preview': 'Gemini 3 Pro',
}


def setup_font(weight: str = "medium") -> str:
    """Register the bundled Figtree font and make it the default family.

    ``weight`` selects a pre-instanced static weight ("medium" or "semibold").
    """
    path = FONT_WEIGHTS.get(weight, FONT_WEIGHTS["medium"])
    if not path.exists():
        path = FONT_FALLBACK
    if path.exists():
        font_manager.fontManager.addfont(str(path))
        name = font_manager.FontProperties(fname=str(path)).get_name()
        plt.rcParams["font.family"] = name
        return name
    return plt.rcParams.get("font.family", "sans-serif")


def provider_for(model_id: str) -> str | None:
    """Map a model identifier to its provider logo (order matters: the
    Codex-harness ``gpt-*`` ids embed the underlying model name)."""
    m = model_id.lower()
    if "gemini" in m:
        return "gemini"
    if any(k in m for k in ("opus", "sonnet", "claude", "anthropic")):
        return "anthropic"
    if any(k in m for k in ("glm", "z-ai", "zhipu", "chatglm")):
        return "zai"
    if any(k in m for k in ("gpt", "openai", "codex")):
        return "openai"
    if any(k in m for k in ("kimi", "moonshot")):
        return "moonshot"
    if "minimax" in m:
        return "minimax"
    if "qwen" in m:
        return "qwen"
    if any(k in m for k in ("devstral", "mistral")):
        return "mistral"
    if "deepseek" in m:
        return "deepseek"
    return None


def load_logo(model_id: str):
    provider = provider_for(model_id)
    if provider is None:
        return None
    path = LOGO_DIR / f"{provider}.png"
    if not path.exists():
        return None
    return mpimg.imread(str(path))


def load_task_icon(task_id: str):
    """Task icons are named after the task label, e.g. ``deseq.png``."""
    path = TASK_ICON_DIR / f"{task_id}.png"
    if not path.exists():
        return None
    return mpimg.imread(str(path))


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
    parser.add_argument("--show", action="store_true", help="Display the plot window after saving.")
    return parser.parse_args()


def display_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.split("/")[-1] or model)


if __name__ == "__main__":
    args = parse_args()
    font_name = setup_font("medium")

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
    # Rows are models; sort by average completion (descending).
    df = df.T
    averages = df.mean(axis=1)
    sorted_idx = averages.sort_values(ascending=False, kind="mergesort").index
    df = df.loc[sorted_idx]
    averages = averages.loc[sorted_idx]

    model_ids = list(df.index)
    task_ids = list(df.columns)
    row_labels = [display_label(model) for model in model_ids]

    print(df)

    # White -> #13682A gradient; 0% completion is white, 100% is the full color.
    cmap_custom = LinearSegmentedColormap.from_list("white_green", ["#FFFFFF", FULL_RATE_COLOR], N=256)

    row_count, column_count = df.shape
    axis_fontsize = AXIS_FONTSIZE
    tick_fontsize = 25 if max(row_count, column_count) <= 9 else TICK_FONTSIZE
    annot_fontsize = 22 if max(row_count, column_count) <= 9 else 19
    bar_value_fontsize = 25 if row_count <= 10 else 20

    data = df.to_numpy(dtype=float)

    # Square cells: 1 data unit per cell on both axes; the figure is sized so
    # the heatmap block is square and the bar panel sits to its right.
    cell_in = 0.62
    heat_w = column_count * cell_in
    heat_h = row_count * cell_in
    bar_w = heat_h * 0.45
    fig_w = heat_w + bar_w + 6.0   # left room for logos + names
    fig_h = heat_h + 3.0           # room for the Task label below

    fig = plt.figure(figsize=(fig_w, fig_h))
    # Manual axes placement so the heatmap cells stay perfectly square.
    left_pad = 4.4 / fig_w
    bottom_pad = 1.9 / fig_h
    ax_heat = fig.add_axes([left_pad, bottom_pad, heat_w / fig_w, heat_h / fig_h])
    gap = 0.55 / fig_w
    ax_bar = fig.add_axes([left_pad + heat_w / fig_w + gap, bottom_pad, bar_w / fig_w, heat_h / fig_h])

    # --- Heatmap cells ---------------------------------------------------
    ax_heat.imshow(
        data,
        cmap=cmap_custom,
        vmin=0,
        vmax=100,
        aspect="equal",
        extent=(0, column_count, row_count, 0),
        interpolation="nearest",
    )
    ax_heat.set_xlim(0, column_count)
    ax_heat.set_ylim(row_count, 0)

    # Black gridlines between cells.
    for x in range(column_count + 1):
        ax_heat.axvline(x, color="black", linewidth=1.0, zorder=3)
    for y in range(row_count + 1):
        ax_heat.axhline(y, color="black", linewidth=1.0, zorder=3)

    # Black border around the whole heatmap.
    for spine in ax_heat.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(2.2)

    # Cell annotations with adaptive contrast (dark cells -> white text).
    full_rgb = np.array(to_rgb(FULL_RATE_COLOR))
    for i in range(row_count):
        for j in range(column_count):
            val = data[i, j]
            if np.isnan(val):
                continue
            # Perceived luminance of the blended cell color.
            cell_rgb = (1 - val / 100) * np.ones(3) + (val / 100) * full_rgb
            lum = 0.299 * cell_rgb[0] + 0.587 * cell_rgb[1] + 0.114 * cell_rgb[2]
            text_color = "white" if lum < 0.55 else "black"
            ax_heat.text(
                j + 0.5, i + 0.5, f"{val:.0f}",
                ha="center", va="center",
                fontsize=annot_fontsize, color=text_color, zorder=4,
            )

    # Remove task identifiers on the x-axis (icons will be added later).
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    ax_heat.tick_params(length=0)

    # Task icons stand in for the x tick labels, one per column. Icons vary in
    # aspect ratio, so each is scaled to fit a common box (never stretched).
    icon_box_in = cell_in * 0.88
    icon_y = row_count + 0.62
    for j, task_id in enumerate(task_ids):
        icon = load_task_icon(task_id)
        if icon is None:
            print(f"No task icon for '{task_id}'; falling back to text label.")
            ax_heat.text(
                j + 0.5, icon_y, task_id, rotation=45, ha="right", va="center",
                fontsize=tick_fontsize * 0.7, color="black", clip_on=False,
            )
            continue
        h, w = icon.shape[0], icon.shape[1]
        zoom = min(icon_box_in * 72 / w, icon_box_in * 72 / h)
        ab = AnnotationBbox(
            OffsetImage(icon, zoom=zoom), (j + 0.5, icon_y), frameon=False,
            box_alignment=(0.5, 0.5), annotation_clip=False, zorder=5,
        )
        ax_heat.add_artist(ab)

    # "Task" label, placed a bit more than one cell height below the heatmap.
    ax_heat.text(
        column_count / 2, row_count + 1.35, "Task",
        ha="center", va="top", fontsize=axis_fontsize, color="black",
        clip_on=False,
    )
    # --- Model names + logos (left of the heatmap) -----------------------
    # Order, left to right: "Model" label | model name | logo | heatmap.
    # Logos sit in a fixed column hugging the heatmap edge (all logos render
    # at a uniform 256x256, so the column stays aligned).
    from matplotlib.transforms import blended_transform_factory
    blend = blended_transform_factory(ax_heat.transAxes, ax_heat.transData)
    save_dpi = 300
    logo_zoom = (cell_in * save_dpi * 0.74) / 256 / (save_dpi / 72)
    logo_half_w = (cell_in * 0.74 / heat_w)   # logo half-width in axes fraction
    logo_center_x = -0.005 - logo_half_w      # right edge ~0.005 left of heatmap
    name_right_x = logo_center_x - logo_half_w - 0.007
    name_texts = []
    for i, (mid, label) in enumerate(zip(model_ids, row_labels)):
        y = i + 0.5
        name_texts.append(ax_heat.text(
            name_right_x, y, label, transform=blend,
            ha="right", va="center", fontsize=tick_fontsize, color="black",
            clip_on=False,
        ))
        logo = load_logo(mid)
        if logo is None:
            continue
        im = OffsetImage(logo, zoom=logo_zoom)
        ab = AnnotationBbox(
            im, (logo_center_x, y), xycoords=blend, frameon=False,
            box_alignment=(0.5, 0.5), annotation_clip=False, zorder=5,
        )
        ax_heat.add_artist(ab)

    # "Model" axis label, just left of the longest model name (measured, so it
    # never collides regardless of font size).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_inv = ax_heat.transAxes.inverted()
    names_left_x = min(
        axes_inv.transform((t.get_window_extent(renderer=renderer).x0, 0))[0]
        for t in name_texts
    )
    ax_heat.text(
        names_left_x - 0.045, row_count / 2, "Model", transform=blend,
        ha="center", va="center", rotation=90,
        fontsize=axis_fontsize, color="black", clip_on=False,
    )

    # --- Average bar chart -----------------------------------------------
    bar_positions = np.arange(row_count) + 0.5
    bar_colors = [cmap_custom(val / 100) for val in averages.values]
    bars = ax_bar.barh(
        bar_positions, averages.values, height=0.7,
        color=bar_colors, edgecolor="black", linewidth=1.4,
    )

    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_visible(True)
    ax_bar.spines["left"].set_color("black")
    ax_bar.spines["bottom"].set_color("black")
    ax_bar.set_ylim(ax_heat.get_ylim())
    ax_bar.set_yticks([])
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xticks([0, 50, 100])
    ax_bar.tick_params(axis="x", labelsize=tick_fontsize, colors="black")
    ax_bar.text(
        50, row_count + 1.35, "Average",
        ha="center", va="top", fontsize=axis_fontsize, color="black",
        clip_on=False,
    )

    for bar, val in zip(bars, averages.values):
        ax_bar.text(
            bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", ha="left", va="center",
            fontsize=bar_value_fontsize, color="black",
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "heatmap_bar.png", dpi=500, bbox_inches="tight", transparent=True)
    plt.savefig(output_dir / "heatmap_bar.pdf", dpi=500, bbox_inches="tight", transparent=True)
    if args.show:
        plt.show()
    plt.close(fig)
