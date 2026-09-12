"""
Toxicity linear-probe package.

Modules
-------
config      : shared constants and paths
data        : Civil Comments loading, labeling, subsampling
embeddings  : Gemma-2-2B loading and layer-14 mean-pooled embeddings
train       : probe construction, hyperparameter search, fitting
evaluate    : accuracy / classification report / confusion matrix
inference   : end-to-end text -> label classifier (ToxicityClassifier)
"""
