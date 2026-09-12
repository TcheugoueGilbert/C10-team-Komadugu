"""
Central configuration for the toxicity linear-probe project.

All tunable constants live here so that every script/module
(embedding extraction, training, evaluation, inference, and the
Streamlit app) shares a single source of truth.
"""

import os

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
SEED = 42

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
MODEL_NAME = "google/gemma-2-2b"   # frozen backbone used to produce embeddings
LAYER = 14                         # hidden-state layer probed
MAX_LENGTH = 64                    # max tokens per comment
BATCH_SIZE = 16                    # embedding-extraction batch size

# ------------------------------------------------------------
# Labeling
# ------------------------------------------------------------
TOXICITY_THRESHOLD = 0.5           # civil_comments "toxicity" >= this -> label 1

# ------------------------------------------------------------
# Dataset subsampling (set to None to use the full split)
# ------------------------------------------------------------
TRAIN_SIZE = 15_000
VAL_SIZE = 5_000

# ------------------------------------------------------------
# Hyperparameter search space
# ------------------------------------------------------------
C_VALUES = [0.001, 10.0]
CLASS_WEIGHTS = [None, "balanced"]

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
# This file lives at <project_root>/scripts/src/config.py, so the
# project root is three directories up from here. All generated
# artifacts (embeddings, trained probe) are written under the
# top-level data/ directory, per the C10-team-kagera layout.
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../scripts
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)                                # repo root

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EMBEDDING_DIR = os.path.join(DATA_DIR, "embeddings")
MODEL_DIR = os.path.join(DATA_DIR, "models")

X_TRAIN_PATH = os.path.join(EMBEDDING_DIR, "X_train_layer14.npy")
Y_TRAIN_PATH = os.path.join(EMBEDDING_DIR, "y_train.npy")
X_VAL_PATH = os.path.join(EMBEDDING_DIR, "X_val_layer14.npy")
Y_VAL_PATH = os.path.join(EMBEDDING_DIR, "y_val.npy")

MODEL_PATH = os.path.join(MODEL_DIR, "classifier.joblib")
RESULTS_PATH = os.path.join(MODEL_DIR, "hyperparameter_search.csv")


def ensure_dirs():
    """Create all output directories used by the pipeline."""
    for d in (DATA_DIR, EMBEDDING_DIR, MODEL_DIR):
        os.makedirs(d, exist_ok=True)
