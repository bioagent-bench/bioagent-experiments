import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("results/data/plan_vs_completion.csv")

    # Separate by open/closed weights
    open_models = df[df['split'] == 'open']
    closed_models = df[df['split'] == 'closed']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot regression line with correlation coefficient
    z = np.polyfit(df['rating'], df['completion_rate'], 1)
    p = np.poly1d(z)
    r = np.corrcoef(df['rating'], df['completion_rate'])[0, 1]
    x_line = np.linspace(df['rating'].min() - 0.1, df['rating'].max() + 0.1, 100)
    ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.7, zorder=1, label=f'Regression line (r = {r:.2f})')

    # Plot closed weights models
    ax.scatter(closed_models['rating'], closed_models['completion_rate'],
               c='#1f77b4', s=200, label='Closed weights', zorder=3)

    # Plot open weights models
    ax.scatter(open_models['rating'], open_models['completion_rate'],
               c='#ff7f0e', s=200, label='Open weights', zorder=3)

    # Add model labels
    for _, row in df.iterrows():
        ax.annotate(row['model_id'].split('_')[-1],
                    (row['rating'], row['completion_rate']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=22, fontfamily='Linux Libertine O')

    # Style
    ax.set_xlabel("Plan Rating", fontsize=26, fontfamily="Linux Libertine O")
    ax.set_ylabel("Completion Rate (%)", fontsize=26, fontfamily="Linux Libertine O")
    ax.tick_params(axis='both', labelsize=22)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Linux Libertine O')

    # Legend outside and above the plot
    ax.legend(prop={'family': 'Linux Libertine O', 'size': 16},
              loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=3, frameon=True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('results/figures/scatter_plan_completion.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/scatter_plan_completion.pdf', dpi=300, bbox_inches='tight')
    plt.show()
