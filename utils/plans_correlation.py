from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CLOSED_MODEL_MAP = {
    "openrouter_openai_gpt-5.2": "gpt-5-2",
    "openrouter_google_gemini-3-pro-preview": "gpt-gemini",
    "openrouter_anthropic_claude-opus-4.5": "gpt-opus",
    "openrouter_anthropic_claude-sonnet-4.5": "gpt-sonnet",
    "openrouter_z-ai_glm-4.7": "gpt-glm",
}


def plan_model_to_perf(model_id: str) -> str | None:
    if model_id in CLOSED_MODEL_MAP:
        return CLOSED_MODEL_MAP[model_id]
    if model_id.startswith("openrouter_"):
        _, vendor, model = model_id.split("_", 2)
        return f"openrouter/{vendor}/{model}"
    return None


def load_plan_ratings(plan_dir: Path) -> pd.DataFrame:
    rows = []
    for path in plan_dir.glob("*.json"):
        data = json.loads(path.read_text())
        for ev in data.get("evaluations", []):
            rows.append({"model_id": ev["model_id"], "rating": ev["rating"]})
    return pd.DataFrame(rows)


def main() -> None:
    plan_df = load_plan_ratings(Path("plan_evals"))
    plan_scores = (
        plan_df.groupby("model_id", as_index=False)["rating"].mean()
        .assign(model_key=lambda df: df["model_id"].map(plan_model_to_perf))
        .dropna(subset=["model_key"])
    )

    perf = pd.concat(
        [
            pd.read_csv("results/data/best_closed_models_completion_rate.csv").assign(split="closed"),
            pd.read_csv("results/data/best_open_models_completion_rate.csv").assign(split="open"),
        ],
        ignore_index=True,
    )
    perf["model_key"] = perf["model"].str.replace(":free", "", regex=False)

    merged = plan_scores.merge(perf, on="model_key", how="inner")
    data_dir = Path("results/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "plan_vs_completion.csv"
    merged.to_csv(data_path, index=False)
    corr = merged["rating"].corr(merged["completion_rate"])

    colors = {"closed": "#2b6cb0", "open": "#2f855a"}
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for split, group in merged.groupby("split"):
        ax.scatter(
            group["rating"],
            group["completion_rate"],
            s=70,
            label=split,
            color=colors.get(split, "#4a5568"),
            alpha=0.9,
        )
        for _, row in group.iterrows():
            ax.annotate(
                row["model"],
                (row["rating"], row["completion_rate"]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=10,
            )

    ax.set_xlabel("Avg plan rating", fontsize=14)
    ax.set_ylabel("Completion rate (%)", fontsize=14)
    ax.tick_params(labelsize=12)

    out_path = Path("results/charts/plans_vs_completion.pdf")
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")
    print(f"Wrote data to {data_path}")


if __name__ == "__main__":
    main()
