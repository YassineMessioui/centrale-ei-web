import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve,
    confusion_matrix
)
import numpy as np

def evaluate_models(models, model_names, X_test, y_test):
    metrics = {
        'Model': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'ROC AUC': []
    }

    plt.figure(figsize=(14, 6))
    
    # ROC Curve
    plt.subplot(1, 2, 1)
    for model, name in zip(models, model_names):
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        metrics['Model'].append(name)
        metrics['ROC AUC'].append(roc_auc)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    for model, name in zip(models, model_names):
        y_probs = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.2f})")

    plt.title('Precision-Recall Curves')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Metrics scores and confusion matrices
    for model, name in zip(models, model_names):
        y_pred = model.predict(X_test)
        metrics['Precision'].append(precision_score(y_test, y_pred))
        metrics['Recall'].append(recall_score(y_test, y_pred))
        metrics['F1 Score'].append(f1_score(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    # Bar plots for metrics comparison
    for metric in ['Precision', 'Recall', 'F1 Score', 'ROC AUC']:
        plt.figure(figsize=(6, 4))
        sns.barplot(x=metrics['Model'], y=metrics[metric])
        plt.title(f'{metric} Comparison')
        plt.ylim(0, 1)
        plt.ylabel(metric)
        plt.xlabel('Model')
        plt.show()
