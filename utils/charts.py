from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
runs_dir = BASE_DIR / "runs-models"

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
df["harness"] = "unknown"
df.loc[df["model"].str.startswith("openrouter/"), "harness"] = "opencode"
df.loc[df["model"].str.startswith("claude"), "harness"] = "claude"
df.loc[df["model"].str.startswith("gpt-"), "harness"] = "codex"

df = df.loc[df["harness"] != "unknown"]

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
best_closed_models_df.to_csv("results/best_closed_models_completion_rate.csv")
best_open_models_df.to_csv("results/best_open_models_completion_rate.csv")

# best results by harness
best_harness_df = df.groupby("harness")
print(best_harness_df.head())

"""ablation"""
df_bloat = load_runs_data(Path('./runs-bloat/'))

df_corrupt = load_runs_data(Path('./runs-corrupt/'))

df_decoy = load_runs_data(Path('./runs-decoy'))

df_stability = load_runs_data(Path('./runs-stability'))