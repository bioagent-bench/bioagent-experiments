from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
runs_dir = BASE_DIR / "runs-models"

AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
ANNOTATION_SIZE = 10

models = {}
harness_model_map = {
    "codex-cli": [
        "gpt-gemin",
        "gpt-opus",
        "gpt-kimi",
        "gpt-minimax",
        "gpt-5-1",
        "gpt-5-2",
        "gpt-5-1-codex",
        "gpt-glm",
        "gpt-sonnet",
    ],
    "claude-code": [
        "claude-sonnet-4-5",
        "claude-opus-4-5",
    ],
    "opencode": [
        "openrouter/z-ai/glm-4.7",
        "openrouter/moonshotai/kimi-k2-thinking",
        "openrouter/anthropic/claude-sonnet-4.5",
        "openrouter/minimax/minimax-m2.1",
        "openrouter/mistralai/devstral-2512:free",
        "openrouter/google/gemini-3-pro-preview",
        "openrouter/openai/gpt-5.2",
        "openrouter/qwen/qwen3-coder:free",
        "openrouter/anthropic/claude-opus-4.5",
    ],
}

closed_models_filter = ["gemini", "opus", "sonnet", "gpt-5.2"]


def load_runs_data(runs_dir: Path) -> list[dict]:
    """Load and process run data from JSON files.
    
    Args:
        runs_dir: Path to directory containing run JSON files.
    
    Returns:
        List of dictionaries containing processed run data with keys:
        model, task_id, use_reference_data, experiment_name, steps_completed,
        final_result_reached, steps_to_completion, completion_rate,
        adjusted_completion_rate.
    """
    rows = []
    for i in runs_dir.glob("*.json"):
        with i.open() as f:
            run = json.load(f)
            model = run["model"]
            eval_results = run["eval_results"]
            task_id = run["task_id"]
            run_hash = run["run_hash"]
            use_reference_data = run["use_reference_data"]
            experiment_name = run["experiment_name"]
            completion_rate = (
                eval_results["steps_completed"] / eval_results["steps_to_completion"]
            )
            completion_rate_adj = (
                eval_results["steps_completed"] + eval_results["final_result_reached"]
            ) / (eval_results["steps_to_completion"] + 1)
            rows.append(
                {
                    "model": model,
                    "task_id": task_id,
                    "run_hash": run_hash,
                    "use_reference_data": use_reference_data,
                    "experiment_name": experiment_name,
                    "steps_completed": eval_results["steps_completed"],
                    "final_result_reached": eval_results["final_result_reached"],
                    "steps_to_completion": eval_results["steps_to_completion"],
                    "completion_rate": completion_rate,
                    "adjusted_completion_rate": completion_rate_adj,
                }
            )
    return pd.DataFrame(rows)


df = load_runs_data(runs_dir)
df = df.loc[df["experiment_name"] == "open-environment-with-reference-data"]

# add harness table
harness_by_model = {
    model: harness
    for harness, models in harness_model_map.items()
    for model in models
}
df["harness"] = df["model"].map(harness_by_model)
df.loc[df["harness"].isna() & df["model"].str.startswith("openrouter/"), "harness"] = "opencode"
df.loc[df["harness"].isna() & df["model"].str.startswith("claude"), "harness"] = "claude-code"
df.loc[df["harness"].isna() & df["model"].str.startswith("gpt-"), "harness"] = "codex-cli"
df = df.loc[df["harness"].notna()]

harness_display_labels = {
    "codex-cli": "Codex CLI",
    "claude-code": "Claude Code",
    "opencode": "OpenCode",
}
harness_order = list(harness_model_map.keys())
harness_task_df = (
    df.groupby(["task_id", "harness"])["completion_rate"].mean().reset_index()
)
harness_task_df["completion_rate"] *= 100
harness_task_pivot = harness_task_df.pivot(
    index="task_id",
    columns="harness",
    values="completion_rate",
).reindex(columns=harness_order)
harness_task_pivot = harness_task_pivot.sort_index()
harness_task_pivot.index.name = "task_id"
harness_means = harness_task_pivot.mean().to_frame().T
harness_means.index = ["mean"]
harness_table_df = pd.concat([harness_task_pivot, harness_means])
harness_table_df = harness_table_df.rename(columns=harness_display_labels)

model_summary = (
    df.groupby(["model", "harness"])["completion_rate"].mean().reset_index()
)
model_summary = model_summary.loc[model_summary["harness"].isin(harness_order)]
model_summary["completion_rate"] *= 100
model_summary["label"] = model_summary["model"].str.split("/").str[-1]
harness_data_path = BASE_DIR / "results" / "data" / "harness_success_rate.csv"
harness_data_path.parent.mkdir(parents=True, exist_ok=True)
model_summary[["model", "label", "harness", "completion_rate"]].to_csv(
    harness_data_path, index=False
)
harness_positions = {harness: idx for idx, harness in enumerate(harness_order)}
model_summary["x"] = model_summary["harness"].map(harness_positions)

fig, ax = plt.subplots(figsize=(10.8, 6.2))
for harness, group in model_summary.groupby("harness"):
    ax.scatter(
        group["x"],
        group["completion_rate"],
        s=45,
        alpha=0.85,
        edgecolor="#333333",
        linewidth=0.4,
        label=harness_display_labels.get(harness, harness),
    )
    for _, row in group.iterrows():
        ax.text(
            row["x"] + 0.01,
            row["completion_rate"],
            row["label"],
            fontsize=ANNOTATION_SIZE,
            ha="left",
            va="center",
        )
ax.set_ylabel("Completion rate (%)", fontsize=AXIS_LABEL_SIZE)
ax.set_ylim(0, 100)
ax.set_xlabel("")
ax.set_axisbelow(True)
ax.tick_params(labelsize=TICK_LABEL_SIZE)
for y in [0, 20, 40, 60, 80]:
    ax.axhline(y, color="#666666", linestyle="--", linewidth=0.6, alpha=0.5)
ax.set_xlim(-0.5, len(harness_order) - 0.5)
ax.set_xticks(list(harness_positions.values()))
ax.set_xticklabels([harness_display_labels.get(h, h) for h in harness_order])
plt.setp(ax.get_xticklabels(), rotation=8, ha="center")
fig.tight_layout()
plot_path = BASE_DIR / "results" / "charts/harness_success_rate.pdf"
fig.savefig(plot_path, format="pdf", bbox_inches="tight")
plt.close(fig)

best_closed_models = [
    "gpt-opus",
    "gpt-gemini",
    "gpt-5-2",
    "gpt-sonnet",
]
best_open_models = [
    "gpt-glm",
    "openrouter/moonshotai/kimi-k2-thinking",
    "openrouter/minimax/minimax-m2.1",
    "openrouter/qwen/qwen3-coder:free",
    "openrouter/mistralai/devstral-2512:free",
]

# raw best results
best_closed_models_df = df.loc[df["model"].isin(best_closed_models)]
best_open_models_df = df.loc[df["model"].isin(best_open_models)]
best_closed_models_df = (
    best_closed_models_df.groupby("model")["completion_rate"].mean() * 100
)
best_open_models_df = (
    best_open_models_df.groupby("model")["completion_rate"].mean() * 100
)
best_closed_models_df.to_csv("results/data/best_closed_models_completion_rate.csv")
best_open_models_df.to_csv("results/data/best_open_models_completion_rate.csv")

combined_df = pd.concat(
    [
        best_closed_models_df.rename("completion_rate")
        .reset_index()
        .assign(weight_type="closed"),
        best_open_models_df.rename("completion_rate")
        .reset_index()
        .assign(weight_type="open"),
    ],
    ignore_index=True,
)
combined_df["label"] = combined_df["model"].str.split("/").str[-1]
label_categories = [
    model.split("/")[-1] for model in best_closed_models + best_open_models
]
combined_df["label"] = pd.Categorical(
    combined_df["label"],
    categories=label_categories,
    ordered=True,
)
combined_df = combined_df.sort_values("label")

colors = {"closed": "#2b6cb0", "open": "#2f855a"}
bar_colors = [colors[wt] for wt in combined_df["weight_type"]]

fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.bar(combined_df["label"], combined_df["completion_rate"], color=bar_colors)
ax.set_ylabel("Completion rate (%)", fontsize=AXIS_LABEL_SIZE)
ax.set_ylim(0, 100)
ax.set_xlabel("")
ax.set_axisbelow(True)
ax.tick_params(labelsize=TICK_LABEL_SIZE)
for y in [0, 20, 40, 60, 80]:
    ax.axhline(y, color="#666666", linestyle="--", linewidth=0.6, alpha=0.5)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
output_path = BASE_DIR / "results" / "charts/open_closed_completion_rate.pdf"
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, format="pdf", bbox_inches="tight")
plt.close(fig)

best_models = best_closed_models + best_open_models
best_task_model_df = (
    df.loc[df["model"].isin(best_models)]
    .groupby(["task_id", "model"])["completion_rate"]
    .mean()
    .mul(100)
    .reset_index()
)
best_task_model_pivot = best_task_model_df.pivot(
    index="task_id",
    columns="model",
    values="completion_rate",
).reindex(columns=best_models)
best_task_model_pivot = best_task_model_pivot.sort_index()
best_task_model_pivot.index.name = "task_id"

task_data_path = BASE_DIR / "results" / "data" / "best_models_task_completion_rate.csv"
task_data_path.parent.mkdir(parents=True, exist_ok=True)
best_task_model_pivot.to_csv(task_data_path)

task_labels = best_task_model_pivot.index.tolist()
model_labels = [model.split("/")[-1] for model in best_task_model_pivot.columns]
values = best_task_model_pivot.to_numpy(dtype=float)
masked_values = np.ma.masked_invalid(values)
cmap = plt.get_cmap("YlGnBu").copy()
cmap.set_bad("#f0f0f0")

width = max(8, 0.6 * len(model_labels) + 3)
height = max(4.5, 0.45 * len(task_labels) + 2.5)
fig, ax = plt.subplots(figsize=(width, height))
im = ax.imshow(masked_values, aspect="auto", cmap=cmap, vmin=0, vmax=100)
ax.set_xlabel("Model", fontsize=AXIS_LABEL_SIZE)
ax.set_ylabel("Task", fontsize=AXIS_LABEL_SIZE)
ax.set_xticks(range(len(model_labels)))
ax.set_xticklabels(
    model_labels, rotation=30, ha="right", fontsize=TICK_LABEL_SIZE
)
ax.set_yticks(range(len(task_labels)))
ax.set_yticklabels(task_labels, fontsize=TICK_LABEL_SIZE)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Completion rate (%)", fontsize=AXIS_LABEL_SIZE)
cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)
fig.tight_layout()
output_path = BASE_DIR / "results" / "charts/best_models_task_completion_rate.pdf"
fig.savefig(output_path, format="pdf", bbox_inches="tight")
plt.close(fig)

# best results by harness
best_harness_df = df.groupby("harness")

"""perturbation"""

# calculate difference from prompt bloated to regular runs
df_bloat = load_runs_data(Path('./runs-bloat/'))
gpt_5 = df[df["model"] == "gpt-5-2"][['task_id', 'completion_rate']]
bloat = df_bloat[['task_id', 'completion_rate']]
cmp = gpt_5.merge(
    bloat,
    on="task_id",
    how="outer",
    suffixes=("_base", "_bloat")
)
cmp["delta"] = (cmp["completion_rate_bloat"] - cmp["completion_rate_base"]) * 100
print(cmp["delta"].mean())
cmp.to_csv('results/data/prompt_bloat_delta.csv')

# corrupt data
df_corrupt = load_runs_data(Path('./runs-corrupt/'))
df_decoy = load_runs_data(Path('./runs-decoy'))
df_stability = load_runs_data(Path('./runs-stability'))

df_stability[["run_hash", "model", "task_id"]].to_csv('stability-logs.csv')

df_stability = df_stability.loc[df_stability["model"] == "gpt-5-2"]
df_stability = df_stability.loc[df_stability["task_id"] == 'transcript-quant']
