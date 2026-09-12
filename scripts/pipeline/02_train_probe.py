"""
Step 2 — Run hyperparameter search over the linear probe and fit the
final model on the full training embeddings.

Usage
-----
    python scripts/02_train_probe.py

Requires
--------
    Embeddings produced by scripts/01_extract_embeddings.py

Outputs
-------
    models/hyperparameter_search.csv
    models/classifier.joblib
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import config
from src.train import hyperparameter_search, train_final_probe, save_probe
from src.evaluate import evaluate_probe, print_evaluation


def main():
    config.ensure_dirs()

    print("Loading embeddings...")
    X_train = np.load(config.X_TRAIN_PATH)
    y_train = np.load(config.Y_TRAIN_PATH)
    X_val = np.load(config.X_VAL_PATH)
    y_val = np.load(config.Y_VAL_PATH)

    print("Running hyperparameter search (C x class_weight)...")
    results_df = hyperparameter_search(X_train, y_train, X_val, y_val)
    results_df.to_csv(config.RESULTS_PATH, index=False)
    print(results_df)

    best_row = results_df.iloc[0]
    best_C = float(best_row["C"])
    best_class_weight = None if best_row["class_weight"] == "None" else best_row["class_weight"]
    print(f"\nBest config: C={best_C}, class_weight={best_class_weight}")

    print("Training final probe on full training embeddings...")
    final_probe = train_final_probe(X_train, y_train, best_C=best_C, best_class_weight=best_class_weight)

    print("Evaluating final probe on validation embeddings...")
    results = evaluate_probe(final_probe, X_val, y_val)
    print_evaluation(results)

    save_probe(final_probe, config.MODEL_PATH)
    print("\nSaved probe to:", config.MODEL_PATH)


if __name__ == "__main__":
    main()
