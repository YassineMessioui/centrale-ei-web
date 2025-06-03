import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import DBSCAN
from xgboost import XGBClassifier

def remap_labels_dbscan(labels):
    return np.where(labels == -1, 1, 0)

def remap_labels_isolation(labels):
    return np.where(labels == -1, 1, 0)

def evaluate_all_models(models, model_names, X_test, y_test):
    metrics = {
        'Model': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'ROC AUC': []
    }

    plt.figure(figsize=(14, 6))

    # ROC CURVES
    plt.subplot(1, 2, 1)
    for model, name in zip(models, model_names):
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        elif isinstance(model, IsolationForest):
            scores = model.decision_function(X_test)
            y_probs = -scores  # Higher = more anomalous
            y_probs = (y_probs - y_probs.min()) / (y_probs.max() - y_probs.min())
        elif isinstance(model, DBSCAN):
            labels = model.fit_predict(X_test)
            y_probs = remap_labels_dbscan(labels)
        else:
            continue

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

    # PRECISION-RECALL CURVE
    plt.subplot(1, 2, 2)
    for model, name in zip(models, model_names):
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        elif isinstance(model, IsolationForest):
            scores = model.decision_function(X_test)
            y_probs = -scores
            y_probs = (y_probs - y_probs.min()) / (y_probs.max() - y_probs.min())
        elif isinstance(model, DBSCAN):
            labels = model.fit_predict(X_test)
            y_probs = remap_labels_dbscan(labels)
        else:
            continue

        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.2f})")

    plt.title('Precision-Recall Curves')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # METRICS + CONFUSION MATRICES
    for model, name in zip(models, model_names):
        if isinstance(model, DBSCAN):
            labels = model.fit_predict(X_test)
            y_pred = remap_labels_dbscan(labels)
        elif isinstance(model, IsolationForest):
            y_pred = remap_labels_isolation(model.predict(X_test))
        else:
            y_pred = model.predict(X_test)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metrics['Precision'].append(precision)
        metrics['Recall'].append(recall)
        metrics['F1 Score'].append(f1)

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    # BAR CHARTS
    for metric in ['Precision', 'Recall', 'F1 Score', 'ROC AUC']:
        plt.figure(figsize=(6, 4))
        sns.barplot(x=metrics['Model'], y=metrics[metric])
        plt.title(f'{metric} Comparison')
        plt.ylim(0, 1)
        plt.ylabel(metric)
        plt.xlabel('Model')
        plt.show()
