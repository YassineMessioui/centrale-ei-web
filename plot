import matplotlib.pyplot as plt

# List of metrics to plot
for metric in ['Precision', 'Recall', 'F1 score']:
    plt.figure(figsize=(6, 4))

    models = metrics['Model'].values
    values = metrics[metric].values

    bars = plt.bar(models, values, color='skyblue')

    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.title(f'{metric} Comparison for {country}')
    plt.ylim(0, 1.1)
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
