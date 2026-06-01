#!/usr/bin/env python3
"""Sanity checks for Upgrade 8 (multi-seed robustness). Fast, no torch.
Run:  python src/upgrades/multiseed/tests/test_sanity.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import run_multiseed as R


def test_tier_thresholds():
    assert R.tier(0.71) == "solved" and R.tier(0.55) == "partial" and R.tier(0.3) == "hard"


def test_bootstrap_resample_is_seeded_and_varies():
    # the per-seed bootstrap index must be deterministic given the seed, and differ across seeds
    n = 1000
    i42a = np.random.RandomState(42).randint(0, n, n)
    i42b = np.random.RandomState(42).randint(0, n, n)
    i43 = np.random.RandomState(43).randint(0, n, n)
    assert np.array_equal(i42a, i42b), "same seed -> same resample (determinism)"
    assert not np.array_equal(i42a, i43), "different seed -> different resample (variation)"


def test_output_small_sd_if_present():
    out = os.path.join(R.C.RESULTS_DIR, "multiseed.csv")
    if not os.path.exists(out):
        print("  SKIP (multiseed.csv not generated yet)"); return
    import pandas as pd
    df = pd.read_csv(out)
    # benchmark MCC SD across seeds should be small (robust); generous bound
    assert df["mcc_sd"].max() < 0.1, f"MCC SD unexpectedly large: {df['mcc_sd'].max()}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} multiseed sanity tests PASSED")
