"""Cheap sanity tests for the homology re-clustering module (no MMseqs2 run, no model fit except a
tiny one). Run:  bibm/lowcpu.sh <venv>/bin/python -m pytest -q src/upgrades/homology_recluster/tests
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import _hc  # noqa: E402
import run_cluster as RC  # noqa: E402
import run_eval as RE  # noqa: E402


def test_thread_cap_applied():
    assert _hc.C.gbmodels.LGBM_PARAMS["n_jobs"] == _hc.N_THREADS <= 3
    m = _hc.C.gbmodels.build_model("lgbm", 42, 2)
    assert m.get_params()["n_jobs"] == _hc.N_THREADS


def test_components_chain_and_isolates():
    comp = RC.components(6, [0, 1, 4], [1, 2, 4])
    assert comp[0] == comp[1] == comp[2]
    assert len({comp[3], comp[4], comp[5], comp[0]}) == 4


def test_cluster_resplit_never_splits_a_component_and_hits_targets():
    rng = np.random.RandomState(0)
    n = 5000
    comp = rng.randint(0, 1500, n)                 # families of varying size
    y = rng.randint(0, 2, n)
    is_test = _hc.cluster_resplit(comp, y, 0.25, seed=42)
    per = pd.DataFrame({"c": comp, "t": is_test}).groupby("c")["t"].nunique()
    assert per.max() == 1                          # whole components only
    for l in (0, 1):
        tgt = round(0.25 * np.sum(y == l))
        assert abs(np.sum(is_test & (y == l)) - tgt) / tgt < 0.05
    assert np.array_equal(is_test, _hc.cluster_resplit(comp, y, 0.25, seed=42))   # deterministic


def test_random_resplit_is_stratified():
    y = np.r_[np.zeros(700, int), np.ones(300, int)]
    t = _hc.random_resplit(y, 0.2, seed=42)
    assert t[y == 0].sum() == 140 and t[y == 1].sum() == 60


def test_nr_one_per_component_label():
    comp = np.array([0, 0, 0, 1, 1, 2]); y = np.array([1, 1, 0, 0, 0, 1])
    idx = _hc.nr_representatives(comp, y, seed=42)
    keys = {(comp[i], y[i]) for i in idx}
    assert len(idx) == len(keys) == 4


def test_exact_dup_both_strands():
    tr = ["AACCG", "TTTTG"]
    te = ["CGGTT",            # reverse complement of AACCG
          "AACCG",            # identical
          "GGGGG"]            # new
    frac_te, frac_tr = RC.exact_dup_stats(tr, te)
    assert abs(frac_te - 2 / 3) < 1e-9 and frac_tr == 0.0


def test_purge_block_null_matches_class_counts_and_perfect_model():
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, 400)
    proba = np.c_[1 - y, y].astype(float)          # perfect model
    keep = rng.rand(400) < 0.6
    full, pb = RE.purge_block(y, proba, np.array([0, 1]), keep, B=50)
    assert pb["status"] == "ok" and pb["mcc"] == 1.0 and abs(pb["delta_mcc"]) < 1e-12
    assert abs(pb["null_mean"] - 1.0) < 1e-12


def test_purge_block_flags_class_elimination():
    """Verifier addendum: a purge that removes (almost) a whole class is NOT evaluable."""
    rng = np.random.RandomState(2)
    y = np.r_[np.zeros(200, int), np.ones(100, int), np.full(100, 2)]
    proba = np.eye(3)[y].astype(float)
    keep = y != 2                                   # every class-2 test sequence purged
    keep[rng.choice(np.where(y == 0)[0], 5, replace=False)] = False
    _, pb = RE.purge_block(y, proba, np.array([0, 1, 2]), keep, B=20)
    assert pb["status"] == "class_eliminated" and "mcc" not in pb
    assert pb["kept_per_class"] == "0:195/200;1:100/100;2:0/100"
    assert pb["acc_removed"] == 1.0 and pb["n_removed"] == 105


def test_purge_block_subtype_null():
    """Subtype-matched null: purging a whole subtype of positives is matched by the stratified null."""
    rng = np.random.RandomState(3)
    sub = np.r_[np.zeros(200, int), np.ones(100, int), np.full(100, 2)]
    y = (sub > 0).astype(int)
    pred = y.copy(); flip = rng.rand(400) < 0.3; pred[flip & (sub == 1)] = 0   # subtype 1 is hard, 2 easy
    proba = np.c_[1 - pred, pred].astype(float)
    keep = sub != 2
    _, pb = RE.purge_block(y, proba, np.array([0, 1]), keep, B=20, strata=sub)
    assert pb["status"] == "ok"
    assert abs(pb["homology_effect_strat"]) < 1e-12        # subtype-matched null reproduces the purge exactly
    assert pb["homology_effect"] < -0.05                     # class-matched null does not


def test_std_mcc_reweights_to_reference():
    y = np.r_[np.ones(90, int), np.zeros(10, int)]
    pred = y.copy(); pred[:9] = 0                  # 10 % of positives wrong
    ref = {0: 0.5, 1: 0.5}
    assert RE.std_mcc(y, pred, ref) > 0            # finite, weighted


def test_zip_loader_matches_repo_loader_when_cache_complete():
    """_hc.load_task must return byte-identical sequences/labels/order to data.load_dataset.
    Only runs if the shared ~/.genomic_benchmarks copy is already complete (never downloads)."""
    import pytest
    import data as gbdata
    task = "dummy_mouse_enhancers_ensembl"
    base = os.path.join(gbdata.GB_ROOT, task)
    if not gbdata._gb_complete(task, base):
        pytest.skip("shared ~/.genomic_benchmarks copy incomplete; this test never downloads")
    tr, ytr, te, yte = _hc.load_task(task)
    d = gbdata.load_dataset(task)
    assert tr == d["train_seqs"] and te == d["test_seqs"]
    assert np.array_equal(ytr, d["y_train"]) and np.array_equal(yte, d["y_test"])
