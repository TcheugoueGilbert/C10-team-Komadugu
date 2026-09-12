"""
Inference helpers: turn raw text (or precomputed embeddings) into
toxicity predictions using the frozen Gemma backbone + trained probe.
"""

import numpy as np

from . import config
from .embeddings import embed_texts, load_backbone
from .train import load_probe


def predict_from_embeddings(embeddings, probe) -> np.ndarray:
    """
    Predict binary toxicity labels from precomputed embeddings.

    Parameters
    ----------
    embeddings : array-like, shape (n_samples, hidden_size) or (hidden_size,)
    probe : fitted sklearn Pipeline

    Returns
    -------
    np.ndarray of int64
        0 = safe, 1 = toxic.
    """
    X = np.asarray(embeddings, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    predictions = probe.predict(X)
    return np.asarray(predictions, dtype=np.int64)


def predict_proba_from_embeddings(embeddings, probe) -> np.ndarray:
    """Same as `predict_from_embeddings` but returns P(toxic)."""
    X = np.asarray(embeddings, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return probe.predict_proba(X)[:, 1]


class ToxicityClassifier:
    """
    End-to-end classifier: raw text -> Gemma layer-14 embedding -> label.

    Loads the backbone and probe once, then can be called repeatedly.
    Intended for use in the Streamlit app and for ad-hoc scripts.
    """

    def __init__(self, model_path: str = config.MODEL_PATH, backbone_name: str = config.MODEL_NAME):
        self.tokenizer, self.model = load_backbone(backbone_name)
        self.probe = load_probe(model_path)

    def predict(self, texts):
        """
        Parameters
        ----------
        texts : str or list[str]

        Returns
        -------
        list[dict] with keys: text, label, label_name, probability_toxic
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        embeddings = embed_texts(texts, self.tokenizer, self.model)
        labels = predict_from_embeddings(embeddings, self.probe)
        probs = predict_proba_from_embeddings(embeddings, self.probe)

        results = [
            {
                "text": t,
                "label": int(l),
                "label_name": "toxic" if l == 1 else "safe",
                "probability_toxic": float(p),
            }
            for t, l, p in zip(texts, labels, probs)
        ]

        return results[0] if single else results
