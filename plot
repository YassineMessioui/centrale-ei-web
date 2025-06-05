import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score

def plot_precision_recall_across_datasets(model, dataset_dict):
    """
    model: Trained classifier (e.g., XGBoost)
    dataset_dict: dict[str, tuple(X_test, y_test)]
    """
    dataset_names = []
    precisions = []
    recalls = []

    for name, (X, y) in dataset_dict.items():
        y_pred = model.predict(X)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)

        dataset_names.append(name)
        precisions.append(precision)
        recalls.append(recall)

    # Plot
    x = range(len(dataset_names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], recalls, width=width, label='Recall')
    plt.bar([i + width/2 for i in x], precisions, width=width, label='Precision')

    plt.xticks(ticks=x, labels=dataset_names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Precision and Recall per Dataset")
    plt.legend()
    plt.tight_layout()
    plt.show()
