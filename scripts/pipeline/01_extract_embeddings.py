"""
Step 1 — Load Civil Comments, label it, and extract Gemma layer-14
mean-pooled embeddings for the train/validation subsets.

Usage
-----
    python scripts/01_extract_embeddings.py

Outputs
-------
    toxicity_probe/embeddings/X_train_layer14.npy
    toxicity_probe/embeddings/y_train.npy
    toxicity_probe/embeddings/X_val_layer14.npy
    toxicity_probe/embeddings/y_val.npy
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import numpy as np
import torch

from src import config
from src.data import load_civil_comments, prepare_splits, set_global_seed, class_balance
from src.embeddings import load_backbone, extract_embeddings


def main():
    config.ensure_dirs()
    set_global_seed(config.SEED)

    print("Loading Civil Comments...")
    dataset = load_civil_comments()

    print("Preparing train/validation splits...")
    train_dataset, validation_dataset = prepare_splits(dataset)
    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(validation_dataset))
    print("Train class balance:", class_balance(train_dataset))
    print("Val class balance:", class_balance(validation_dataset))

    print(f"Loading backbone: {config.MODEL_NAME}")
    tokenizer, model = load_backbone()

    print("Extracting training embeddings...")
    X_train, y_train = extract_embeddings(train_dataset, tokenizer, model)
    np.save(config.X_TRAIN_PATH, X_train)
    np.save(config.Y_TRAIN_PATH, y_train)
    print("X_train:", X_train.shape)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Extracting validation embeddings...")
    X_val, y_val = extract_embeddings(validation_dataset, tokenizer, model)
    np.save(config.X_VAL_PATH, X_val)
    np.save(config.Y_VAL_PATH, y_val)
    print("X_val:", X_val.shape)

    print("Done. Embeddings saved to:", config.EMBEDDING_DIR)


if __name__ == "__main__":
    main()
