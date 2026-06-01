"""Refit the best-k LightGBM per dataset and return its full gain-importance vector.

Reuses the benchmark's cached feature matrices and the identical fixed LightGBM
hyperparameters + seed, so the refit reproduces the benchmarked model exactly
(deterministic) and the gain importances are reproducible. Gain importance is
summed across all trees (and, for multiclass, across all per-class trees).
"""
import os

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

import featurize
import models


def feature_names(k):
    """Authoritative column-index -> k-mer mapping (the fixed vocabulary order)."""
    vec = CountVectorizer(analyzer="char", ngram_range=(k, k),
                          vocabulary=featurize.vocabulary(k), lowercase=False)
    return vec.get_feature_names_out()


def fit_and_gain(dataset, k, cache_dir, seed):
    """Return (gain, names) over all 4^k k-mers (vocabulary order), and n_used."""
    Xtr = sparse.load_npz(os.path.join(cache_dir, f"{dataset}__train__k{k}.npz"))
    ytr = np.load(os.path.join(cache_dir, f"{dataset}__train__y.npy"))
    model = models.build_model("lgbm", seed, n_classes=len(np.unique(ytr)))
    model.fit(Xtr, ytr)
    gain = model.booster_.feature_importance(importance_type="gain").astype(np.float64)
    names = feature_names(k)
    assert len(gain) == len(names) == featurize.feature_dim(k), "feature/vocab mismatch"
    return gain, names, int((gain > 0).sum())
