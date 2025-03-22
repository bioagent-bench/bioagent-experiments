import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_benchmark_charts(known_csv, unknown_csv):
    # Set up a single figure
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    # Read and process both datasets
    def process_data(csv_filename):
        df = pd.read_csv(csv_filename, sep=";")
        summary = df.groupby("model").sum(numeric_only=True)
        summary["percent_complete"] = (
            summary["completed_steps"] / summary["total_steps"]
        ) * 100
        return summary.sort_values("percent_complete", ascending=False).reset_index()

    known_summary = process_data(known_csv)
    unknown_summary = process_data(unknown_csv)

    # Create the overlapping barplots
    bar_width = 0.35
    x = range(len(known_summary))

    # Create bars for both environments
    plt.bar(
        x,
        known_summary["percent_complete"],
        bar_width,
        label="Preconfigured Environment",
        alpha=0.7,
    )
    plt.bar(
        [i + bar_width for i in x],
        unknown_summary["percent_complete"],
        bar_width,
        label="Open Environment",
        alpha=0.7,
    )

    # Add value labels
    def add_labels(x_pos, data):
        for i, value in enumerate(data["percent_complete"]):
            plt.text(
                x_pos[i],
                value + 1,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    add_labels(x, known_summary)
    add_labels([i + bar_width for i in x], unknown_summary)

    # Chart aesthetics
    plt.ylabel("Completion Rate (%)")
    plt.xlabel("Model")
    plt.title("Benchmark Completion Rates by Environment", fontsize=14, weight="bold")
    plt.xticks([i + bar_width / 2 for i in x], known_summary["model"])
    plt.ylim(0, 100)
    plt.legend()

    # Save and show
    plt.tight_layout()
    plt.savefig("llm_benchmark_completion_combined.png", dpi=300, bbox_inches="tight")
    plt.show()


# Create the combined chart
create_benchmark_charts("./GLBIO_results_known.csv", "./GLBIO_results_unknown.csv")
