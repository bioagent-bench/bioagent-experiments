import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D

# Reuse the Figtree font setup and provider-logo loading from the heatmap.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heatmap_bar import setup_font, load_logo, AXIS_FONTSIZE, TICK_FONTSIZE


# Real product names keyed by the plan_vs_completion model_id.
REAL_NAMES = {
    "openrouter_anthropic_claude-opus-4.5": "Claude Opus 4.5",
    "openrouter_anthropic_claude-sonnet-4.5": "Claude Sonnet 4.5",
    "openrouter_google_gemini-3-pro-preview": "Gemini 3 Pro",
    "openrouter_minimax_minimax-m2.1": "MiniMax M2.1",
    "openrouter_mistralai_devstral-2512": "Devstral 2512",
    "openrouter_moonshotai_kimi-k2-thinking": "Kimi K2 Thinking",
    "openrouter_openai_gpt-5.2": "GPT-5.2",
    "openrouter_qwen_qwen3-coder": "Qwen3 Coder",
    "openrouter_z-ai_glm-4.7": "GLM-4.7",
}

SAVE_DPI = 500
# bbox_inches="tight" pads the trimmed bbox by this much on every side.
SAVE_PAD_IN = 0.1

# Match the heatmap's saved size so the figures line up consistently.
SIZE_REFERENCE = Path("results/selected-heatmap-models/figures/heatmap_bar.png")
FALLBACK_SIZE_IN = (14.6, 7.72)


def real_name(model_id: str) -> str:
    return REAL_NAMES.get(model_id, model_id.split("_")[-1])


def reference_size_px(path: Path) -> tuple[int, int] | None:
    """Saved pixel size (width, height) of a reference figure."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def _saved_px_size(fig) -> tuple[int, int]:
    """Pixel size this figure would actually be saved at."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=SAVE_DPI, bbox_inches="tight",
                pad_inches=SAVE_PAD_IN, transparent=True)
    buf.seek(0)
    with Image.open(buf) as im:
        return im.size


def fit_saved_size(fig, target: tuple[int, int], max_iter: int = 60) -> tuple[int, int]:
    """Resize the figure so its saved raster is exactly ``target`` pixels.

    Saving with ``bbox_inches="tight"`` trims to the drawn content and then
    pads, so setting ``figsize`` alone would not match the output. Each pass
    measures the real saved size and corrects only the dimension still off;
    previously seen sizes short-circuit any ±1px oscillation.
    """
    target_w, target_h = target
    current = _saved_px_size(fig)
    seen = set()
    for _ in range(max_iter):
        if current == target or current in seen:
            break
        seen.add(current)
        width, height = fig.get_size_inches()
        dw = (target_w - current[0]) / SAVE_DPI if current[0] != target_w else 0.0
        dh = (target_h - current[1]) / SAVE_DPI if current[1] != target_h else 0.0
        fig.set_size_inches(width + dw, height + dh)
        current = _saved_px_size(fig)
    return current


if __name__ == "__main__":
    setup_font()

    df = pd.read_csv("results/data/plan_vs_completion.csv")

    fig, ax = plt.subplots(figsize=(15, 7))

    # Regression line with correlation coefficient.
    z = np.polyfit(df["rating"], df["completion_rate"], 1)
    p = np.poly1d(z)
    r = np.corrcoef(df["rating"], df["completion_rate"])[0, 1]
    x_line = np.linspace(df["rating"].min() - 0.2, df["rating"].max() + 0.2, 100)
    ax.plot(x_line, p(x_line), "--", color="gray", alpha=0.8, linewidth=3.5, zorder=1)

    # Logo size (~0.55in); labels sit consistently at the top-right of each
    # logo (offset in points so spacing is uniform regardless of data scale).
    logo_zoom = 0.40 * 72 / 256
    label_offset = (15, 9)  # points: right and up from the logo center

    for _, row in df.iterrows():
        x, y = row["rating"], row["completion_rate"]

        logo = load_logo(row["model_id"])
        if logo is not None:
            im = OffsetImage(logo, zoom=logo_zoom)
            ab = AnnotationBbox(
                im, (x, y), frameon=False, box_alignment=(0.5, 0.5),
                annotation_clip=False, zorder=3,
            )
            ax.add_artist(ab)
        else:
            ax.scatter([x], [y], c="black", s=200, zorder=3)

        ax.annotate(
            real_name(row["model_id"]), (x, y),
            xytext=label_offset, textcoords="offset points",
            ha="left", va="bottom", fontsize=TICK_FONTSIZE, color="black", zorder=4,
        )

    ax.set_xlabel("Plan rating", fontsize=AXIS_FONTSIZE, color="black")
    ax.set_ylabel("Completion rate (%)", fontsize=AXIS_FONTSIZE, color="black")
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors="black")
    ax.set_xlim(2.3, 5.3)
    ax.set_ylim(58, 106)

    # Legend: regression line only.
    handles = [
        Line2D([0], [0], linestyle="--", color="gray", linewidth=3.5,
               label=f"Regression line (r = {r:.2f})"),
    ]
    ax.legend(handles=handles, prop={"size": TICK_FONTSIZE}, loc="lower right",
              frameon=True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Match the heatmap's saved size exactly.
    target_px = reference_size_px(SIZE_REFERENCE) or tuple(
        round(v * SAVE_DPI) for v in FALLBACK_SIZE_IN
    )
    achieved_px = fit_saved_size(fig, target_px)
    print(f"Size: target {target_px[0]}x{target_px[1]} px "
          f"-> achieved {achieved_px[0]}x{achieved_px[1]} px "
          f"({achieved_px[0] / SAVE_DPI:.2f}x{achieved_px[1] / SAVE_DPI:.2f} in)")

    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "scatter_plan_completion.png", dpi=SAVE_DPI,
                bbox_inches="tight", pad_inches=SAVE_PAD_IN, transparent=True)
    plt.savefig(output_dir / "scatter_plan_completion.pdf", dpi=SAVE_DPI,
                bbox_inches="tight", pad_inches=SAVE_PAD_IN, transparent=True)
    plt.close(fig)
