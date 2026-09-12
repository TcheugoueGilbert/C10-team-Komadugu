"""
Data loading, labeling, and subsampling for the Civil Comments dataset.
"""

import random

import numpy as np
from datasets import load_dataset

from . import config


def set_global_seed(seed: int = config.SEED):
    """Seed python/numpy/torch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_civil_comments():
    """
    Load the raw `google/civil_comments` dataset from Hugging Face.

    Returns
    -------
    datasets.DatasetDict
        Splits: "train", "validation", "test".
    """
    return load_dataset("google/civil_comments")


def create_binary_label(example, threshold: float = config.TOXICITY_THRESHOLD):
    """
    Convert a continuous `toxicity` score into a binary label.

    Parameters
    ----------
    example : dict
        Dataset example containing `toxicity`.
    threshold : float
        Score at/above which a comment is labeled toxic (1).

    Returns
    -------
    dict
        The example with an added `label` field (0 = safe, 1 = toxic).
    """
    example["label"] = int(example["toxicity"] >= threshold)
    return example


def prepare_splits(
    dataset,
    train_size: int = config.TRAIN_SIZE,
    val_size: int = config.VAL_SIZE,
    seed: int = config.SEED,
):
    """
    Shuffle, subsample, and label the train/validation splits.

    Parameters
    ----------
    dataset : datasets.DatasetDict
        Output of `load_civil_comments`.
    train_size, val_size : int or None
        Number of examples to keep. None keeps the full split.
    seed : int
        Shuffle seed.

    Returns
    -------
    (train_dataset, validation_dataset) : tuple of datasets.Dataset
        Labeled, subsampled splits.
    """
    train_dataset = dataset["train"].shuffle(seed=seed)
    validation_dataset = dataset["validation"].shuffle(seed=seed)

    if train_size is not None:
        train_dataset = train_dataset.select(range(min(train_size, len(train_dataset))))

    if val_size is not None:
        validation_dataset = validation_dataset.select(range(min(val_size, len(validation_dataset))))

    train_dataset = train_dataset.map(create_binary_label)
    validation_dataset = validation_dataset.map(create_binary_label)

    return train_dataset, validation_dataset


def class_balance(dataset):
    """Return a dict of label -> count for a labeled dataset."""
    from collections import Counter
    return dict(Counter(dataset["label"]))
