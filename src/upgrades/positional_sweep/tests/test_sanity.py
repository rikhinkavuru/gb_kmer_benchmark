#!/usr/bin/env python3
"""Sanity checks for Upgrade 4 (graded positional rescue). Fast, synthetic, no torch.
Run:  python src/upgrades/positional_sweep/tests/test_sanity.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import _common as C
import featurize
import models as gbmodels
import run_positional_sweep as R
from sklearn.metrics import matthews_corrcoef


def test_featurize_arm_dims():
    seqs = ["ACGT" * 30, "TTGGCCAA" * 15]
    k = 4
    assert R.featurize_arm(seqs, k, 1).shape[1] == featurize.feature_dim(k)
    assert R.featurize_arm(seqs, k, 8).shape[1] == 9 * featurize.feature_dim(k)
    assert R.featurize_arm(seqs, k, 4).shape[1] == 5 * featurize.feature_dim(k)


def test_route_labels():
    assert R.route_label("nt_splice_sites_all") == "POSITIONAL"
    assert R.route_label("nt_promoter_tata") == "fixed-position promoter"
    assert R.route_label("human_enhancers_cohn") == "non-positional (control)"


def test_positional_signal_only_recovered_with_bins():
    """A FIXED-position motif (positives carry 'GGGGG' always at the start; negatives carry it at a
    random position) is separable only when position is resolved. delta(B=8 - B=1) should be > 0."""
    rng = np.random.RandomState(0)
    n, Lseq, k = 300, 60, 4
    def bg():
        return "".join(rng.choice(list("ACGT"), size=Lseq))
    pos, neg = [], []
    for _ in range(n):
        s = list(bg()); s[0:5] = list("GGGGG"); pos.append("".join(s))           # fixed at start
        s = list(bg()); j = rng.randint(6, Lseq - 5); s[j:j + 5] = list("GGGGG"); neg.append("".join(s))
    seqs = pos + neg; y = np.array([1] * n + [0] * n)
    tr = np.r_[np.arange(0, n, 2), np.arange(n, 2 * n, 2)]
    te = np.r_[np.arange(1, n, 2), np.arange(n + 1, 2 * n, 2)]
    def mcc(B):
        m = gbmodels.build_model("lgbm", 42, 2)
        Xtr = R.featurize_arm([seqs[i] for i in tr], k, B)
        Xte = R.featurize_arm([seqs[i] for i in te], k, B)
        m.fit(Xtr, y[tr])
        return matthews_corrcoef(y[te], m.classes_[np.argmax(m.predict_proba(Xte), 1)])
    d = mcc(8) - mcc(1)
    assert d > 0.1, f"position bins should recover the fixed-position signal (delta={d:.3f})"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} positional_sweep sanity tests PASSED")
