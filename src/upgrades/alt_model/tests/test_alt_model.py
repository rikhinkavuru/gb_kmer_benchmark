"""Cheap unit/sanity tests for the alt_model module (pytest or plain python).
Run: bibm/lowcpu.sh /Users/rikhinkavuru/Downloads/venv/bin/python -m pytest -q src/upgrades/alt_model/tests"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.abspath(os.path.join(HERE, ".."))
UP = os.path.abspath(os.path.join(MOD, ".."))
for d in (UP, MOD, os.path.join(UP, "psm_sensitivity")):
    if d not in sys.path:
        sys.path.insert(0, d)

import alt_models as AM                    # noqa: E402
import analyze_alt_model as A              # noqa: E402
import run_alt_model as W                  # noqa: E402


def _rand(n=600, nc=2, seed=0, ties=True):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, nc, n)
    p = rng.dirichlet(np.ones(nc), n)
    p[np.arange(n), y] += 0.3 * rng.rand(n)
    p /= p.sum(1, keepdims=True)
    if ties:
        p = np.round(p, 2); p /= p.sum(1, keepdims=True)
    return y, p


def test_metrics_match_sklearn():
    from sklearn.metrics import matthews_corrcoef, roc_auc_score
    for nc in (2, 3):
        y, p = _rand(nc=nc)
        sc = A.Scorer(y, p)
        pred = p.argmax(1)
        assert abs(sc.mcc() - matthews_corrcoef(y, pred)) < 1e-12
        assert abs(AM.mcc_from_pred(y, pred, nc) - matthews_corrcoef(y, pred)) < 1e-12
        ref = roc_auc_score(y, p[:, 1]) if nc == 2 else roc_auc_score(y, p, multi_class="ovr", average="macro")
        assert abs(sc.auroc() - ref) < 1e-12
        assert abs(AM.auroc(y, p) - ref) < 1e-12


def test_weighted_equals_indexing():
    from sklearn.metrics import matthews_corrcoef, roc_auc_score
    for nc in (2, 3):
        y, p = _rand(nc=nc, seed=1)
        sc = A.Scorer(y, p)
        rng = np.random.RandomState(42)
        for _ in range(5):
            idx = rng.randint(0, len(y), len(y))
            w = np.bincount(idx, minlength=len(y)).astype(float)
            assert abs(sc.mcc(w) - matthews_corrcoef(y[idx], p[idx].argmax(1))) < 1e-12
            ref = (roc_auc_score(y[idx], p[idx, 1]) if nc == 2 else
                   roc_auc_score(y[idx], p[idx], multi_class="ovr", average="macro"))
            assert abs(sc.auroc(w) - ref) < 1e-12


def test_random_arm_helpers_match_psm():
    import psm_core as P
    pri_psm = lambda seed, sid, n: np.random.RandomState([int(seed), int(sid)]).uniform(size=n)  # noqa: E731
    for seed in range(3):
        a = W.pri(seed, 1, 500)
        assert np.array_equal(a, pri_psm(seed, 1, 500))
        assert np.array_equal(W.random_take(500, 37, a), P.random_take(500, 37, a))


def test_eqr_arm_sizes_and_positives():
    """Random arm = all original positives + exactly the equalized split's negative counts."""
    from cache_seqs import load_cached
    task = "nt_enhancers"
    tr, ytr, te, yte, meta, te_idx = W.arm_data(task, "eqr3")
    n_neg = W.eq_counts(task)
    otr, oytr, ote, oyte = load_cached(task)
    assert int((np.asarray(ytr) == 0).sum()) == n_neg[0] and int((np.asarray(yte) == 0).sum()) == n_neg[1]
    assert int((np.asarray(ytr) == 1).sum()) == int((np.asarray(oytr) == 1).sum())
    assert [ote[i] for i in te_idx] == list(te)


def test_eq_mapping_consistent():
    task = "drosophila_enhancers_stark"
    from cache_seqs import load_cached
    tr, ytr, te, yte, meta, te_idx = W.arm_data(task, "eq")
    otr, oytr, ote, oyte = load_cached(task)
    assert all(ote[i].upper() == s.upper() and int(oyte[i]) == int(y) for i, s, y in zip(te_idx, te, yte))


def test_cache_counts():
    import cache_seqs as CS
    exp = CS.expected_counts()
    for t in ("dummy_mouse_enhancers_ensembl", "nt_promoter_tata"):
        tr, ytr, te, yte = CS.load_cached(t)
        assert (len(tr), len(te)) == exp[t]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)


def test_ebm_design_pos_is_superset_of_b1_design():
    """P9: the global block of the B=8 design ranks exactly like the B=1 spectrum, so the ebmpos design
    contains the pre-registered EBM B1 design (same global columns) plus binned columns."""
    from scipy import sparse
    rng = np.random.RandomState(3)
    n, g, nb = 300, 64, 8 * 64
    y = rng.randint(0, 2, n)
    G = sparse.csr_matrix(rng.rand(n, g) * (rng.rand(n, g) < 0.3) + 0.2 * y[:, None] * (np.arange(g) < 5))
    Bn = sparse.csr_matrix(rng.rand(n, nb) * (rng.rand(n, nb) < 0.1))
    X8 = sparse.hstack([G, Bn], format="csr")
    names = [f"k{i}" for i in range(g)] + [f"bin{i // g}:k{i % g}" for i in range(nb)]
    comp = rng.rand(n, 20)
    _, _, fn1 = AM.ebm_design(G, G, comp, comp, y, names[:g], m=16)
    _, _, fn8, idx = AM.ebm_design_pos(X8, X8, comp, comp, y, names, g, 16, 16)
    glob8 = [f for f in fn8[20:] if not f.startswith("bin")]
    assert glob8 == fn1[20:]
    assert sum(f.startswith("bin") for f in fn8) == 16 and len(idx) == 32


def test_two_level_arm_weights_equal_indexing():
    """size_controlled scores every arm under the ORIGINAL-test-set multiplicity counts restricted to the
    arm's rows (c[posn[te_idx]]): identical to resampling the original set and keeping the arm's members."""
    from sklearn.metrics import roc_auc_score
    y0, p0 = _rand(n=500, seed=5)
    rng = np.random.RandomState(1)
    te_idx = np.sort(rng.choice(500, 200, replace=False))
    sc = A.Scorer(y0[te_idx], p0[te_idx])
    r = np.random.RandomState(42)
    for _ in range(5):
        draw = r.randint(0, 500, 500)
        c = np.bincount(draw, minlength=500).astype(float)
        keep = np.isin(draw, te_idx)
        ref = roc_auc_score(y0[draw[keep]], p0[draw[keep], 1])
        assert abs(sc.auroc(c[te_idx]) - ref) < 1e-12


def test_overlap_unit_uses_uncapped_original_test():
    """P11a: the overlap design uses the UNCAPPED original-split model (bench_full where bench is capped),
    scored on the same original test set as the bench unit."""
    u, arm = A.overlap_unit("human_enhancers_ensembl", "lgbm", "kmer")
    b = A.unit("human_enhancers_ensembl", "bench", "kmer", "lgbm")
    assert arm == "bench_full" and not u["train_capped"] and b["train_capped"]
    assert np.array_equal(u["y"], b["y"]) and u.get("test_fp") == b.get("test_fp")
    u2, arm2 = A.overlap_unit("human_enhancers_cohn", "lgbm", "kmer")
    assert arm2 == "bench" and not u2["train_capped"]


def test_posthoc_overlap_rule_reads_leakfree_nt_and_parent():
    """P11b: nt_enhancers is judged on its leak-free row; variants inherit the parent's row; missing = None."""
    import pandas as pd
    oc = pd.DataFrame([dict(task="nt_enhancers", model="ebm", leakfree_test=False, diff_lo=-0.5),
                       dict(task="nt_enhancers", model="ebm", leakfree_test=True, diff_lo=0.2),
                       dict(task="human_enhancers_ensembl", model="lr", leakfree_test=False, diff_lo=-0.1)])
    assert A.overlap_contam("nt_enhancers", "ebm", oc) is True
    assert A.overlap_contam("nt_enhancers", "ebm512", oc) is True          # parent = ebm
    assert A.overlap_contam("human_enhancers_ensembl", "lrx", oc) is False  # parent = lr
    assert A.overlap_contam("human_enhancers_cohn", "lgbm", oc) is None
