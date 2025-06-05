import matplotlib.pyplot as plt

# Assuming `metrics` is a pandas DataFrame with columns: 'Model', 'Precision', 'Recall', 'F1 score'
models = metrics['Model']
metrics_list = ['Precision', 'Recall', 'F1 score']

fig, axes = plt.subplots(nrows=3, figsize=(8, 8), sharex=True)

for i, metric in enumerate(metrics_list):
    values = metrics[metric]
    ax = axes[i]
    
    bars = ax.bar(models, values, color='#006B3C')  # Cadmium Green approximation
    
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.1)

# Bottom plot: add title, xlabel, and xticks
axes[-1].set_title(f'Metric Comparison for {country}')
axes[-1].set_xlabel('Model')
axes[-1].set_xticks(range(len(models)))
axes[-1].set_xticklabels(models, rotation=45)

# Remove x-axis tick labels for top plots
for ax in axes[:-1]:
    ax.tick_params(labelbottom=False)

plt.tight_layout()
plt.show()
