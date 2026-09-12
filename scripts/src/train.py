"""
Training and hyperparameter search for the linear toxicity probe.

The probe is a scikit-learn Pipeline (StandardScaler + LogisticRegression)
fit on frozen Gemma layer-14 embeddings.
"""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def build_probe(C: float = 1.0, class_weight="balanced", max_iter: int = 2000, seed: int = config.SEED) -> Pipeline:
    """Construct an (unfit) StandardScaler + LogisticRegression pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            solver="lbfgs",
            random_state=seed,
        )),
    ])


def hyperparameter_search(
    X_train, y_train, X_val, y_val,
    C_values=config.C_VALUES,
    class_weights=config.CLASS_WEIGHTS,
):
    """
    Grid-search over C and class_weight, scored by validation accuracy.

    Returns
    -------
    results_df : pd.DataFrame
        Columns: model, C, class_weight, accuracy - sorted best first.
    """
    results = []

    for class_weight in class_weights:
        for C in C_values:
            probe = build_probe(C=C, class_weight=class_weight)
            probe.fit(X_train, y_train)
            predictions = probe.predict(X_val)
            accuracy = accuracy_score(y_val, predictions)

            results.append({
                "model": "LogisticRegression",
                "C": C,
                "class_weight": str(class_weight),
                "accuracy": accuracy,
            })

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)
    return results_df


def train_final_probe(X_train, y_train, best_C: float, best_class_weight="balanced", max_iter: int = 3000):
    """Fit the final probe using the best hyperparameters found."""
    probe = build_probe(C=best_C, class_weight=best_class_weight, max_iter=max_iter)
    probe.fit(X_train, y_train)
    return probe


def save_probe(probe: Pipeline, path: str = config.MODEL_PATH):
    """Persist the fitted probe to disk with joblib."""
    joblib.dump(probe, path)
    return path


def load_probe(path: str = config.MODEL_PATH) -> Pipeline:
    """Load a previously saved probe."""
    return joblib.load(path)
