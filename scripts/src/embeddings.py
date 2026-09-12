"""
Gemma-2-2B loading and layer-14 mean-pooled embedding extraction.

This module wraps the frozen backbone used to turn raw comment text
into fixed-size feature vectors for the downstream linear probe.
"""

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from . import config


def load_backbone(model_name: str = config.MODEL_NAME):
    """
    Load the frozen Gemma tokenizer + model used to produce embeddings.

    Returns
    -------
    (tokenizer, model) : tuple
        `model` is set to eval mode with `output_hidden_states=True`.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    return tokenizer, model


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool token embeddings while ignoring padding tokens.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Shape (batch_size, sequence_length, hidden_size).
    attention_mask : torch.Tensor
        Shape (batch_size, sequence_length).

    Returns
    -------
    torch.Tensor
        Shape (batch_size, hidden_size).
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


@torch.no_grad()
def extract_embeddings(
    data,
    tokenizer,
    model,
    layer: int = config.LAYER,
    max_length: int = config.MAX_LENGTH,
    batch_size: int = config.BATCH_SIZE,
    text_field: str = "text",
    label_field: str = "label",
):
    """
    Extract layer-`layer` mean-pooled embeddings for a labeled dataset.

    Parameters
    ----------
    data : datasets.Dataset
        Must contain `text_field` and `label_field` columns.
    tokenizer, model : transformers objects
        As returned by `load_backbone`.
    layer : int
        Hidden-state index to probe (0 = embedding layer).
    max_length : int
        Truncation length for tokenization.
    batch_size : int
        Number of texts processed per forward pass.

    Returns
    -------
    X : np.ndarray, shape (n_samples, hidden_size)
    y : np.ndarray, shape (n_samples,)
    """
    all_embeddings = []
    all_labels = []

    for start in tqdm(range(0, len(data), batch_size), desc="Extracting embeddings"):
        end = min(start + batch_size, len(data))
        batch = data[start:end]

        texts = batch[text_field]
        labels = batch[label_field]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model(**inputs)
        layer_embeddings = outputs.hidden_states[layer]

        pooled = mean_pool(layer_embeddings, inputs["attention_mask"])
        pooled = pooled.float().cpu().numpy()

        all_embeddings.append(pooled)
        all_labels.extend(labels)

    X = np.concatenate(all_embeddings, axis=0)
    y = np.asarray(all_labels, dtype=np.int64)
    return X, y


@torch.no_grad()
def embed_texts(texts, tokenizer, model, layer: int = config.LAYER, max_length: int = config.MAX_LENGTH):
    """
    Embed a small list of raw strings (no labels required).

    Used by the Streamlit app for single/batch inference.

    Parameters
    ----------
    texts : list[str]

    Returns
    -------
    np.ndarray, shape (len(texts), hidden_size)
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model(**inputs)
    layer_embeddings = outputs.hidden_states[layer]
    pooled = mean_pool(layer_embeddings, inputs["attention_mask"])

    return pooled.float().cpu().numpy()
