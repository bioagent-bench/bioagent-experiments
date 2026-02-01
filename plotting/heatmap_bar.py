import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


if __name__ == "__main__":
    df = pd.read_csv("results/data/all_models_task_completion_rate.csv")
    df.set_index("task_id", inplace=True)
    df = df.T

    best_models = ['gpt-5-1-codex-max', 'gpt-5-2', 'gpt-gemini', 'gpt-opus', 'gpt-sonnet', 'openrouter/mistralai/devstral-2512:free', 'openrouter/moonshotai/kimi-k2-thinking', 'openrouter/qwen/qwen3-coder:free', 'openrouter/z-ai/glm-4.7', 'openrouter/minimax/minimax-m2.1']

    df = df.loc[best_models]
    averages = df.mean(axis=1)

    # Sort models by their average completion rate (descending) and re-index df/averages accordingly
    sorted_idx = averages.sort_values(ascending=False).index
    df = df.loc[sorted_idx]
    averages = averages.loc[sorted_idx]
    best_models = list(sorted_idx)

    df.index = [
        'claude-opus-4-5',
        'gemini-3-pro-preview',
        'claude-sonnet-4-5',
        'gpt-5-2',
        'gpt-5-1',
        'kimi-k2-thinking',
        'glm-4.7',
        'minimax-m2.1',
        'qwen3-coder',
        'devstral-2512',
    ]

    print(df)

    # Custom colormap: mild orange to blue
    colors = ['#FFE5D9', '#FFC9B3', '#FFFFFF', '#B3D4FF', '#80BFFF']
    cmap_custom = LinearSegmentedColormap.from_list('orange_blue', colors, N=100)

    # Create figure with 2 subplots: heatmap on left, horizontal bar chart on right
    fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(9, 6),
                                           gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.01})

    # Plot heatmap
    sns.heatmap(df, annot=True, fmt=".0f", cmap=cmap_custom, ax=ax_heat,
                cbar=False,
                linewidths=0.5, linecolor='white',
                annot_kws={'fontsize': 20, 'fontfamily': 'Linux Libertine O', 'color': 'black'},
                square=True,
                vmin=0, vmax=100)

    # Style heatmap
    ax_heat.set_xlabel("Task", fontsize=24, fontfamily="Linux Libertine O", color='black')
    ax_heat.set_ylabel("Model", fontsize=24, fontfamily="Linux Libertine O", color='black')

    # Set tick labels
    ax_heat.set_xticklabels(df.columns, fontsize=22, fontfamily="Linux Libertine O", rotation=45, ha='right', color='black')
    ax_heat.set_yticklabels(df.index, fontsize=22, fontfamily="Linux Libertine O", rotation=0, color='black')
    ax_heat.tick_params(colors='black')

    # Plot horizontal bar chart - bars aligned with heatmap rows
    bar_positions = np.arange(len(best_models)) + 0.5  # Center bars on heatmap cells

    # Color bars based on average using the same colormap
    max_avg = 100
    bar_colors = [cmap_custom(val / max_avg) for val in averages.values]

    bars = ax_bar.barh(bar_positions, averages.values, height=0.7, color=bar_colors, edgecolor='white')

    # Style bar chart - remove bounding box, keep only axis lines
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_visible(True)
    ax_bar.spines['left'].set_color('black')
    ax_bar.spines['bottom'].set_color('black')
    ax_bar.set_ylim(len(best_models), 0)  # Invert to match heatmap row order
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Average", fontsize=24, fontfamily="Linux Libertine O", color='black')
    ax_bar.set_xlim(0, 100)
    ax_bar.tick_params(axis='x', labelsize=22, colors='black')
    for label in ax_bar.get_xticklabels():
        label.set_fontfamily('Linux Libertine O')
        label.set_color('black')

    # Add value labels at end of bars
    for bar, val in zip(bars, averages.values):
        ax_bar.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}', ha='left', va='center', fontsize=22, fontfamily='Linux Libertine O', color='black')

    plt.tight_layout()
    plt.savefig('results/figures/heatmap_bar.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/heatmap_bar.pdf', dpi=300, bbox_inches='tight')
    plt.show()
