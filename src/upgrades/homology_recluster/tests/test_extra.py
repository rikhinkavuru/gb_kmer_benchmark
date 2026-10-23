"""Sanity tests for run_extra.py / prepare_ft_hc_splits.py helpers (cheap; one small real task, no big fits).
Run:  bibm/lowcpu.sh <venv>/bin/python -m pytest -q -p no:cacheprovider src/upgrades/homology_recluster/tests
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import _hc  # noqa: E402
import run_extra as RX  # noqa: E402


def test_split_idx_matches_hc_data():
    task = "dummy_mouse_enhancers_ensembl"
    seqs, y, itr, ite = RX.split_idx(task, "id80", nr=True)
    d = _hc.hc_data(task, "id80", nr=True)
    assert [seqs[i] for i in itr] == d["tr"] and [seqs[i] for i in ite] == d["te"]
    assert np.array_equal(y[itr], d["ytr"]) and np.array_equal(y[ite], d["yte"])
    s2, y2, a, b = RX.split_idx(task, "original")
    tr, ytr, te, yte = _hc.load_task(task)
    assert [s2[i] for i in a] == tr and [s2[i] for i in b] == te


def test_clean_excess_matches_run_cleaning():
    sys.path.insert(0, _hc.ROOT)
    import run_cleaning as RC
    rng0 = np.random.RandomState(0)
    y = rng0.randint(0, 2, 500); fl = rng0.rand(500) < 0.5
    keep, _ = RC.clean_indices(y, fl, 0.2, np.random.RandomState(42))
    m = RX.clean_excess(y, fl, 0.2, np.random.RandomState(42))
    assert np.array_equal(np.where(m)[0], keep)


def test_arm_masks_keep_positives_and_placebos_are_size_matched():
    rng = np.random.RandomState(1)
    mk = lambda n, gc: ["".join(rng.choice(list("ACGT"), 200, p=[(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]))
                        for _ in range(n)]
    tr = mk(150, 0.6) + mk(150, 0.4); ytr = np.r_[np.ones(150, int), np.zeros(150, int)]
    te = mk(50, 0.6) + mk(50, 0.4); yte = np.r_[np.ones(50, int), np.zeros(50, int)]
    masks, _ = RX.arm_masks(tr, ytr, te, yte, _hc.C.load_tbp())
    for arm, (mtr, mte) in masks.items():
        assert mtr[ytr == 1].all() and mte[yte == 1].all(), arm
    for a in RX.BASE_ARMS:
        for j, yy in ((0, ytr), (1, yte)):
            assert (masks[a][j] & (yy == 0)).sum() == (masks[f"placebo_{a}"][j] & (yy == 0)).sum()


def test_ft_hc_splits_format_if_written():
    d = os.path.join(_hc.EXT, "ft_splits")
    f = os.path.join(d, "human_enhancers_cohn__test.csv.gz")
    if not os.path.exists(f):
        pytest.skip("HC fine-tune splits not written yet (run prepare_ft_hc_splits.py)")
    df = pd.read_csv(f)
    assert {"idx", "label", "sequence", "original_pos", "gc_match_pos", "comp_equalized_pos", "random_pos"} <= set(df.columns)
    for arm in ("gc_match", "comp_equalized", "random"):
        assert (df.loc[df.label == 1, f"{arm}_pos"] >= 0).all()
    assert ((df.comp_equalized_pos >= 0) & (df.label == 0)).sum() == ((df.random_pos >= 0) & (df.label == 0)).sum()
