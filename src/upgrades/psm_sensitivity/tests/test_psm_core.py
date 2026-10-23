#!/usr/bin/env python3
"""Sanity tests for psm_core (fast, synthetic, no torch).

Run:  python src/upgrades/psm_sensitivity/tests/test_psm_core.py   (prints PASS / raises)
Also discoverable by pytest (def test_*)."""
import os
import sys

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             matthews_corrcoef, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "composition_clean"))
import psm_core as P  # noqa: E402


def test_metrics_match_sklearn():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 700)
    p = np.clip(0.35 * y + rng.uniform(0, 0.65, 700), 0, 1).round(2)      # many ties
    w = rng.uniform(0.1, 3, 700)
    for ww in (None, w):
        out = P.evaluate(y, p, w=ww, B=0, n_bal=0)
        sw = ww
        assert abs(out["auroc"] - roc_auc_score(y, p, sample_weight=sw)) < 1e-10
        assert abs(out["ap_pos"] - average_precision_score(y, p, sample_weight=sw)) < 1e-10
        assert abs(out["ap_neg"] - average_precision_score(1 - y, -p, sample_weight=sw)) < 1e-10
        pred = (p > 0.5).astype(int)
        assert abs(out["mcc"] - matthews_corrcoef(y, pred, sample_weight=sw)) < 1e-10
        assert abs(out["bacc"] - balanced_accuracy_score(y, pred, sample_weight=sw)) < 1e-10
    # class-balance-weighted MCC == MCC with weights 1/n_class
    wb = np.where(y == 1, 1 / (y == 1).sum(), 1 / (y == 0).sum())
    out = P.evaluate(y, p, B=0, n_bal=0)
    assert abs(out["mcc_bw"] - matthews_corrcoef(y, (p > 0.5).astype(int), sample_weight=wb)) < 1e-10


def test_bootstrap_matches_paper_protocol():
    """Same rng protocol as _common.fit_eval_boot -> identical AUROC percentile CI."""
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, 300)
    p = np.clip(0.2 * y + rng.uniform(0, 0.8, 300), 0, 1)
    out = P.evaluate(y, p, B=200, seed=42, n_bal=0)
    r = np.random.RandomState(42)
    aa = []
    for _ in range(200):
        idx = r.randint(0, 300, 300)
        aa.append(roc_auc_score(y[idx], p[idx]))
    assert abs(out["auroc_lo"] - np.percentile(aa, 2.5)) < 1e-10
    assert abs(out["auroc_hi"] - np.percentile(aa, 97.5)) < 1e-10


def test_hist_temper_alpha1_equals_paper_counts_and_alpha0_keeps_all():
    import run_composition_clean as R
    rng = np.random.RandomState(2)
    pp = rng.beta(5, 3, 800)
    pn = rng.beta(3, 5, 3000)
    pri = rng.uniform(size=len(pn))
    k1 = P.hist_temper(pp, pn, 20, 1.0, pri)
    kr = R.match_proportional(pp, pn, 20, np.random.RandomState(3))
    bt, bn = P._bins(pp, pn, 20)
    assert np.array_equal(np.bincount(bn[k1], minlength=20), np.bincount(bn[kr], minlength=20))
    assert len(P.hist_temper(pp, pn, 20, 0.0, pri)) == len(pn)
    # nested + monotone
    prev = set(range(len(pn)))
    for a in np.linspace(0, 1, 11):
        k = set(P.hist_temper(pp, pn, 20, a, pri).tolist())
        assert k <= prev
        prev = k


def test_caliper_matching_is_valid():
    rng = np.random.RandomState(4)
    lp = rng.normal(1, 1, 400)
    ln = rng.normal(-1, 1, 900)
    cal = 0.2 * np.std(np.r_[lp, ln])
    pi, ni = P.caliper_1to1(lp, ln, cal, np.random.RandomState(42))
    assert len(pi) == len(ni) > 50
    assert len(set(pi)) == len(pi) and len(set(ni)) == len(ni), "without replacement"
    assert np.all(np.abs(lp[pi] - ln[ni]) <= cal + 1e-12)
    # greedy NN property: brute-force replay must give identical pairs
    avail = np.ones(len(ln), bool)
    bp, bn = [], []
    for i in np.random.RandomState(42).permutation(len(lp)):
        d = np.where(avail, np.abs(ln - lp[i]), np.inf)
        j = int(np.argmin(d))
        if d[j] <= cal:
            bp.append(i); bn.append(j); avail[j] = False
    assert np.array_equal(pi, np.array(bp)) and np.array_equal(ni, np.array(bn))
    # matched sets are balanced on the score
    assert abs(lp[pi].mean() - ln[ni].mean()) < 0.05


def test_overlap_weights_balance_logistic_covariates():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(5)
    X = rng.normal(size=(3000, 3))
    y = (rng.uniform(size=3000) < 1 / (1 + np.exp(-(X @ [1.0, -0.5, 0.3])))).astype(int)
    e = LogisticRegression(C=1e6, max_iter=2000).fit(X, y).predict_proba(X)[:, 1]
    w = P.overlap_weights(e, y)
    s = P.smd(X[y == 1], X[y == 0], w[y == 1], w[y == 0])
    assert np.nanmax(np.abs(s)) < 1e-3, f"overlap weights should exactly balance LR covariates: {s}"


def test_dust():
    lc = P.dust_lowcomplexity_frac(["A" * 200, "".join(np.random.RandomState(6).choice(list("ACGT"), 200))])
    assert lc[0] > 0.99 and lc[1] < 0.05, lc


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
