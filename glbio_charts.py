import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_benchmark_chart(csv_filename, chart_title):
    # Load the benchmark data from CSV
    df = pd.read_csv(csv_filename, sep=";")

    # Summarize performance per model
    print(df)

    summary = df.groupby("model").sum(numeric_only=True)
    summary["percent_complete"] = (
        summary["completed_steps"] / summary["total_steps"]
    ) * 100
    summary = summary.sort_values("percent_complete", ascending=False).reset_index()

    # Set Seaborn theme
    sns.set_theme(style="whitegrid")

    # Create the barplot
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(
        y="percent_complete", x="model", data=summary, palette="viridis"
    )

    # Add value labels
    for index, row in summary.iterrows():
        plt.text(
            index,
            row["percent_complete"] + 1,
            f"{row['percent_complete']:.1f}%",
            ha="center",
            fontsize=10,
        )

    # Chart aesthetics
    plt.title(chart_title, fontsize=16, weight="bold")
    plt.ylabel("Completion Rate (%)")
    plt.xlabel("Model")
    plt.ylim(0, 100)
    plt.tight_layout()

    # Save to PNG
    output_filename = (
        f"llm_benchmark_completion_{chart_title.lower().replace(' ', '_')}.png"
    )
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.show()


# Create both charts
create_benchmark_chart("./GLBIO_results_known.csv", "Preconfigured Environment")
create_benchmark_chart("./GLBIO_results_unknown.csv", "Open Environment")
