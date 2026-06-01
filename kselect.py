#!/usr/bin/env python3
"""kselect.py -- canonical k selection on a HELD-OUT VALIDATION split.

The benchmark featurizes each task at k in {3,4,5,6}. The hyperparameter k is
chosen WITHOUT touching the test split: we carve a stratified 80/20 train/val
split from the TRAIN set (seeded), fit LightGBM at each k on the 80%, and keep
the k that maximizes VALIDATION MCC. The selected k is then retrained on the
full train split and evaluated once on the test split (done by the caller).

This replaces the earlier selection-by-test-set-MCC, which was selection-on-test
and gave best-k metrics a mild optimistic bias. Ties are broken toward the
smaller k (fewer features). Seed-deterministic given the cached matrices.
"""
import os
import numpy as np
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import matthews_corrcoef

import models

KS = (3, 4, 5, 6)
VAL_FRAC = 0.20


def best_k_val(task, cache_dir, seed, ks=KS, val_frac=VAL_FRAC):
    """Return the k in ``ks`` with the highest validation MCC for ``task``.
    Uses the cached L1-normalized k-mer spectra written by run_benchmark.py."""
    y = np.load(os.path.join(cache_dir, f"{task}__train__y.npy"))
    n_classes = len(np.unique(y))
    tr, va = next(StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=seed).split(np.zeros(len(y)), y))
    best_k, best_mcc = ks[0], -2.0
    for k in ks:
        X = sparse.load_npz(os.path.join(cache_dir, f"{task}__train__k{k}.npz"))
        m = models.build_model("lgbm", seed, n_classes)
        m.fit(X[tr], y[tr])
        pred = m.classes_[np.argmax(m.predict_proba(X[va]), axis=1)]
        mcc = matthews_corrcoef(y[va], pred)
        if mcc > best_mcc:          # strict > => ties resolve to the smaller k
            best_mcc, best_k = mcc, k
    return best_k
