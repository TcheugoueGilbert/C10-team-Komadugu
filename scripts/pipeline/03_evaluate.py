"""
Step 3 — Re-evaluate a saved probe on the validation embeddings
(useful for verifying results without retraining).

Usage
-----
    python scripts/03_evaluate.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import config
from src.train import load_probe
from src.evaluate import evaluate_probe, print_evaluation


def main():
    print("Loading validation embeddings...")
    X_val = np.load(config.X_VAL_PATH)
    y_val = np.load(config.Y_VAL_PATH)

    print("Loading trained probe from:", config.MODEL_PATH)
    probe = load_probe(config.MODEL_PATH)

    results = evaluate_probe(probe, X_val, y_val)
    print_evaluation(results)


if __name__ == "__main__":
    main()
