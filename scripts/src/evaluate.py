"""
Evaluation utilities for the toxicity probe.
"""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def evaluate_probe(probe, X_val, y_val, as_dict: bool = False):
    """
    Compute accuracy, a classification report, and a confusion matrix.

    Parameters
    ----------
    probe : sklearn.pipeline.Pipeline
        Fitted probe (StandardScaler + LogisticRegression).
    X_val, y_val : np.ndarray
        Held-out embeddings and labels.
    as_dict : bool
        If True, `report` is returned as a dict instead of a string.

    Returns
    -------
    dict with keys: accuracy, report, confusion_matrix, predictions
    """
    predictions = probe.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    report = classification_report(y_val, predictions, output_dict=as_dict)
    cm = confusion_matrix(y_val, predictions)

    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "predictions": predictions,
    }


def print_evaluation(results: dict):
    """Pretty-print the output of `evaluate_probe` to stdout."""
    print(f"Validation accuracy: {results['accuracy']:.4f}")
    print("\nClassification report:")
    print(results["report"])
    print("\nConfusion matrix:")
    print(results["confusion_matrix"])
