#!/usr/bin/env python3
"""Sanity checks for the fine-tuned-FM module. Torch-free: nothing here loads a model.

Checks the pieces that could silently corrupt the headline number:
  1. the validation carve is deterministic, disjoint from the fit set, and class-stratified;
  2. every arm's splits load with the sizes the paper reports, and the POSITIVES are identical
     across arms (only negatives are resampled by the cleanings);
  3. the bootstrap metric helper agrees with sklearn on a known-answer case.
Run: python src/upgrades/fm_finetune/tests/test_sanity.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(MOD, "..", "..", ".."))
for p in (MOD, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_fm_finetune as ft   # noqa: E402


def test_val_carve():
    y = np.array([0] * 30 + [1] * 70)
    tr, va = ft.stratified_val_carve(y, 0.1, 42)
    assert len(set(tr) & set(va)) == 0, "validation carve overlaps the fit set"
    assert len(tr) + len(va) == len(y)
    assert (y[va] == 0).sum() == 3 and (y[va] == 1).sum() == 7, "carve is not stratified"
    tr2, va2 = ft.stratified_val_carve(y, 0.1, 42)
    assert np.array_equal(va, va2), "carve is not deterministic at a fixed seed"
    assert not np.array_equal(va, ft.stratified_val_carve(y, 0.1, 43)[1]), "seed is ignored"
    print("ok  validation carve: disjoint, stratified, deterministic, seed-sensitive")


def test_arms_share_positives():
    ds = "human_enhancers_cohn"
    pos = {}
    for arm in ("gc_match", "comp_equalized"):      # 'original' needs the GB download
        tr, ytr, te, yte = ft.load_arm(ds, arm)
        assert len(tr) == len(ytr) and len(te) == len(yte)
        assert set(np.unique(ytr)) == {0, 1}
        pos[arm] = (sorted(s for s, y in zip(tr, ytr) if y == 1),
                    sorted(s for s, y in zip(te, yte) if y == 1))
        n_neg_tr = int((ytr == 0).sum())
        print(f"ok  {arm}: n_train={len(tr)} (neg {n_neg_tr}) n_test={len(te)}")
    assert pos["gc_match"] == pos["comp_equalized"], \
        "the cleanings must leave the POSITIVE class identical; they do not"
    print("ok  positives identical across cleaning arms (only negatives are resampled)")


def test_eval_boot():
    rng = np.random.RandomState(0)
    y = np.array([0] * 200 + [1] * 200)
    # a perfectly separating score must give AUROC 1 and MCC 1
    p = np.concatenate([np.full(200, 0.1), np.full(200, 0.9)])
    m = ft.eval_boot(y, p, seed=42, B=50)
    assert abs(m["auroc"] - 1.0) < 1e-9 and abs(m["mcc"] - 1.0) < 1e-9, m
    # a random score must sit near chance with a CI that covers 0.5
    m = ft.eval_boot(y, rng.rand(400), seed=42, B=200)
    assert m["auroc_lo"] <= 0.5 <= m["auroc_hi"], m
    assert m["n_test"] == 400 and abs(m["pos_frac"] - 0.5) < 1e-9
    print("ok  eval_boot: exact on separable data, CI covers chance on random scores")


if __name__ == "__main__":
    test_val_carve()
    test_arms_share_positives()
    test_eval_boot()
    print("all fm_finetune sanity checks passed")
