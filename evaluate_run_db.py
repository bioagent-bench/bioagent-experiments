import duckdb
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

df = duckdb.sql("""
    SELECT 
        *
    FROM read_json(?, ignore_errors=true, union_by_name=true, maximum_depth=2)
""", params=[[str(p) for p in Path('~/run_logs/runs/').expanduser().glob('*.json')]]).to_df()

if 'eval_results' in df.columns:
    df['steps_completed'] = df['eval_results'].apply(lambda x: x.get('steps_completed') if isinstance(x, dict) else None)
    df['steps_to_completion'] = df['eval_results'].apply(lambda x: x.get('steps_to_completion') if isinstance(x, dict) else None)
    df['final_result_reached'] = df['eval_results'].apply(lambda x: x.get('final_result_reached') if isinstance(x, dict) else None)

df['steps_completed'] = pd.to_numeric(df['steps_completed'], errors='coerce').fillna(0)
df['steps_to_completion'] = pd.to_numeric(df['steps_to_completion'], errors='coerce').fillna(0)
df['final_result_reached'] = pd.to_numeric(df['final_result_reached'], errors='coerce').fillna(False)
percent_completed = df['steps_completed'] / df['steps_to_completion'].replace(0, pd.NA)
df['percent_completed'] = percent_completed.fillna(0).mul(100)

print(df)

model_summary = (
    df.groupby("model", dropna=False)["percent_completed"]
    .agg(["mean", "std", "count"])
    .rename(columns={"mean": "avg_percent_completed", "std": "std_percent_completed"})
    .reset_index()
    .sort_values("avg_percent_completed", ascending=False)
)

print("\nAverage percent completion per model:")
print(model_summary)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(
    model_summary["model"],
    model_summary["avg_percent_completed"],
    yerr=model_summary["std_percent_completed"].fillna(0),
    capsize=4,
    color="#4C72B0",
)
ax.set_xlabel("Model")
ax.set_ylabel("Completion (%)")
ax.set_title("Average Percent Completed by Model")
plt.xticks(rotation=20, ha="right")
ax.set_ylim(0, 100)
plt.tight_layout()
output_path = Path("percent_completed_by_model.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\nPlot saved to {output_path}")
plt.show()

task_summary = (
    df.groupby("task_id", dropna=False)["percent_completed"]
    .agg(["mean", "std", "count"])
    .rename(columns={"mean": "avg_percent_completed", "std": "std_percent_completed"})
    .reset_index()
    .sort_values("avg_percent_completed", ascending=False)
)

print("\nAverage percent completion per task:")
print(task_summary)

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(
    task_summary["task_id"],
    task_summary["avg_percent_completed"],
    yerr=task_summary["std_percent_completed"].fillna(0),
    capsize=4,
    color="#55A868",
)
ax.set_xlabel("Task ID")
ax.set_ylabel("Completion (%)")
ax.set_title("Average Percent Completed by Task")
plt.xticks(rotation=45, ha="right")
ax.set_ylim(0, 100)
plt.tight_layout()
task_output_path = Path("percent_completed_by_task.png")
plt.savefig(task_output_path, dpi=300, bbox_inches="tight")
print(f"\nPlot saved to {task_output_path}")
plt.show()