import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the benchmark data from CSV
df = pd.read_csv("./GLBIO_results_known.csv")

# Summarize performance per model
summary = df.groupby("model").sum(numeric_only=True)
summary["percent_complete"] = (
    summary["completed_steps"] / summary["total_steps"]
) * 100
summary = summary.sort_values("percent_complete", ascending=False).reset_index()

# Set Seaborn theme
sns.set_theme(style="whitegrid")

# Create the barplot
plt.figure(figsize=(10, 6))
barplot = sns.barplot(x="percent_complete", y="model", data=summary, palette="viridis")

# Add value labels
for index, row in summary.iterrows():
    plt.text(
        row["percent_complete"] + 1,
        index,
        f"{row['percent_complete']:.1f}%",
        va="center",
        fontsize=10,
    )

# Chart aesthetics
plt.title("LLM Benchmark - Completion Rate by Model", fontsize=16, weight="bold")
plt.xlabel("Completion Rate (%)")
plt.ylabel("Model")
plt.xlim(0, 100)
plt.tight_layout()

# Save to PNG
plt.savefig("llm_benchmark_completion.png", dpi=300, bbox_inches="tight")
plt.show()
