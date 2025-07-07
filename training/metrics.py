# training/metrics.py

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def classification_metrics(y_true, y_pred):
    """
    Computes classification metrics.

    Args:
        y_true (list[int])
        y_pred (list[int])

    Returns:
        dict with accuracy, precision, recall, f1, auc
    """
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    try:
        metrics["auc"] = roc_auc_score(y_true, y_pred)
    except ValueError:
        metrics["auc"] = None  # AUC not defined (e.g., only one class predicted)

    return metrics
