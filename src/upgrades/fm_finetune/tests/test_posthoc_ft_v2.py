#!/usr/bin/env python3
"""Cheap torch-free checks for the post hoc fine-tuning analyses (posthoc_ft_v2.py,
baselines_ft_v2.py, summarize_ft_v2.py labels). No model is trained here.

  1. the leak-free nt_enhancers set: exactly the 100 verbatim train->test duplicates of
     nt_enhancers_duplication.csv, all positives, and each eval arm loses exactly those rows;
  2. the leak-free transform keeps predictions aligned with labels (a perfect synthetic scorer
     stays perfect; a scorer that is perfect only on the duplicates falls to chance);
  3. paired bootstrap: two models with identical predictions have a difference of exactly 0 in
     every resample (so the fine-tuned-minus-baseline CIs are paired, not independent);
  4. every model the pipeline can produce has a display label and a family;
  5. if the baseline refits exist: they reproduce the published k-mer and frozen-probe numbers
     (fm_finetune_v2_baselines_repro.csv, all |diff| < 1e-3).
Run: bibm/lowcpu.sh <torch-free python> src/upgrades/fm_finetune/tests/test_posthoc_ft_v2.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.dirname(HERE)
sys.path.insert(0, MOD)
import eval_ft_v2 as ev  # noqa: E402
import posthoc_ft_v2 as ph  # noqa: E402
import summarize_ft_v2 as sm  # noqa: E402


def test_leakfree_set():
    dup, n_copies = ph.duplicated_test()
    mem = ev.load_membership()
    y = mem["nt_enhancers"]["y"]
    assert dup.sum() == 100 and (y[dup] == 1).all()
    assert n_copies == 107, n_copies           # psm_sensitivity_leakage_check.csv n_train_rows_equal_a_test_seq
    for a, m in mem["nt_enhancers"]["mask"].items():
        assert (m & dup).sum() == 100, a        # every arm keeps all positives, so all 100 duplicates
    print("ok  leak-free nt_enhancers: 100 duplicated positives, present in every eval arm, 107 train copies")
    return dup, mem


def test_leakfree_alignment(dup, mem):
    y = mem["nt_enhancers"]["y"]
    perfect = dict(meta=dict(model="x", dataset="nt_enhancers"), p=y.astype(float),
                   idx=np.arange(len(y)), y=y)
    only_dups = dict(meta=dict(model="z", dataset="nt_enhancers"),
                     p=np.where(dup, 1.0, 0.5), idx=np.arange(len(y)), y=y)
    runs, mem2 = ph.leakfree([perfect, only_dups], mem, dup)
    yl = mem2[ph.NTL]["y"]
    assert len(yl) == 300 and yl.sum() == 100
    assert ev.auroc(yl, runs[0]["p"]) == 1.0
    assert ev.auroc(yl, runs[1]["p"]) == 0.5
    assert ev.auroc(y, only_dups["p"]) > 0.6       # ... while the released test rewards it
    for a in ev.ARMS:
        assert mem2[ph.NTL]["mask"][a].sum() == mem["nt_enhancers"]["mask"][a].sum() - 100
    print("ok  leak-free transform keeps predictions aligned; a duplicate-only scorer drops to 0.5")


def test_paired_bootstrap(mem):
    ds = "human_enhancers_cohn"
    y = mem[ds]["y"]
    rng = np.random.RandomState(0)
    p = rng.rand(len(y)) + 0.3 * y
    g = {(m, ds, ta): {42: p} for m in ("m1", "m2") for ta in ("original",)}
    pairs = [("original", "original"), ("original", "comp_equalized")]
    b1, _ = ev.joint_boot_auc(g, mem, "m1", ds, pairs)
    b2, _ = ev.joint_boot_auc(g, mem, "m2", ds, pairs)
    for pr in pairs:
        assert np.array_equal(b1[pr], b2[pr])
    print("ok  joint bootstrap is paired across models (identical resamples)")


def test_labels():
    for m in list(sm.FT_MODELS) + ["kmer_lgbm", "hyenadna_frozen_mean", "hyenadna_frozen_sep",
                                   "dnabert2_frozen_mean", "dnabert2_frozen_cls"]:
        assert m in sm.MODEL_INFO, m
        assert sm.model_family(m) in ("fine-tuned", "frozen probe", "k-mer"), m
    print("ok  every model has a label and a family")


def test_repro():
    p = os.path.join(ev.OUT, "fm_finetune_v2_baselines_repro.csv")
    if not os.path.exists(p):
        print("skip baseline reproduction (run baselines_ft_v2.py first)")
        return
    q = pd.read_csv(p)
    assert len(q) >= 6 and q["match_1e-3"].all(), q[~q["match_1e-3"]]
    print(f"ok  baseline refits reproduce {len(q)} published numbers to 1e-3")


if __name__ == "__main__":
    dup, mem = test_leakfree_set()
    test_leakfree_alignment(dup, mem)
    test_paired_bootstrap(mem)
    test_labels()
    test_repro()
    print("all post hoc checks passed")
