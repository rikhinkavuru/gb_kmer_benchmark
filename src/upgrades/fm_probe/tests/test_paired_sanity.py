#!/usr/bin/env python3
"""Sanity checks for the paired FM-probe (Fix A/B). LightGBM-side only -- imports
run_fm_paired (which imports _common/LightGBM) but NOT torch, so it is safe to run in its
own process (kept separate from test_sanity.py, which uses torch). Run:
  python src/upgrades/fm_probe/tests/test_paired_sanity.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))           # fm_probe/ (run_fm_paired, extract_embeddings)
sys.path.insert(0, os.path.join(HERE, "..", ".."))     # src/upgrades (_common)
import run_fm_paired as R


def test_membership_mask_duplicate_safe():
    orig = ["A", "B", "B", "C", "A"]
    arm = ["B", "A"]                                    # wants one B and one A
    mask = R.membership_mask(orig, arm)
    assert mask.tolist() == [True, True, False, False, False], mask.tolist()
    # if the arm keeps both B's, both are marked
    assert R.membership_mask(orig, ["B", "B"]).tolist() == [False, True, True, False, False]


def test_membership_mask_full_and_empty():
    orig = ["X", "Y", "Z"]
    assert R.membership_mask(orig, ["X", "Y", "Z"]).all()
    assert not R.membership_mask(orig, []).any()


def test_excl0_ci_parsing():
    assert R._excl0("[0.043,0.053]") and R._excl0("[-0.107,-0.056]")
    assert not R._excl0("[-0.003,0.014]") and not R._excl0("[0.000,0.000]")


def test_metric_helpers_single_class_safe():
    y = np.array([1, 1, 1]); pred = np.array([1, 1, 1])
    assert np.isnan(R._mcc(y, pred)) and np.isnan(R._auc(y, np.zeros((3, 2)), 2))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} paired sanity tests PASSED")
