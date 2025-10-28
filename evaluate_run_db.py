import duckdb
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

df = duckdb.sql("""
    SELECT 
        *
    FROM read_json(?, ignore_errors=true, union_by_name=true, maximum_depth=2)
""", params=[[str(p) for p in Path('~/run_logs').expanduser().glob('*.json')]]).to_df()

# Extract nested fields if they exist
if 'eval_results' in df.columns:
    df['steps_completed'] = df['eval_results'].apply(lambda x: x.get('steps_completed') if isinstance(x, dict) else None)
    df['steps_to_completion'] = df['eval_results'].apply(lambda x: x.get('steps_to_completion') if isinstance(x, dict) else None)

# Convert to numeric and replace None/NaN values with 0
df['steps_completed'] = pd.to_numeric(df['steps_completed'], errors='coerce').fillna(0)
df['steps_to_completion'] = pd.to_numeric(df['steps_to_completion'], errors='coerce').fillna(0)

print(df)

# Create pivot table for plotting (mean values)
pivot_df = df.pivot_table(
    values='steps_completed',
    index='task_id',
    columns='experiment_name',
    aggfunc='mean'  # Use mean in case there are multiple runs per task/experiment
)

# Calculate standard deviation for error bars
pivot_std = df.pivot_table(
    values='steps_completed',
    index='task_id',
    columns='experiment_name',
    aggfunc='std'
).fillna(0)  # Fill NaN with 0 for cases with single runs

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))
pivot_df.plot(kind='bar', ax=ax, yerr=pivot_std, capsize=4)

ax.set_xlabel('Task ID')
ax.set_ylabel('Steps Completed')
ax.set_title('Steps Completed by Task ID across Different Experiments')
ax.legend(title='Experiment Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('steps_completed_by_task.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to steps_completed_by_task.png")
plt.show()