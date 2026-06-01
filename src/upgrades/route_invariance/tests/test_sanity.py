#!/usr/bin/env python3
"""Sanity checks for Upgrade 9 (route invariance). Fast, no torch/LightGBM.
Run:  python src/upgrades/route_invariance/tests/test_sanity.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import run_route_invariance as R


def test_tier_thresholds():
    assert R.tier(0.9) == "solved" and R.tier(0.7) == "solved"
    assert R.tier(0.6) == "partial" and R.tier(0.5) == "partial"
    assert R.tier(0.49) == "hard" and R.tier(0.0) == "hard"
    assert R.tier(float("nan")) is None


def test_output_exists_and_consistent():
    out = os.path.join(R.C.RESULTS_DIR, "route_invariance.csv")
    if not os.path.exists(out):
        print("  SKIP (route_invariance.csv not generated yet)"); return
    import pandas as pd
    df = pd.read_csv(out, keep_default_na=False)
    assert len(df) == 15, "expected 15 tasks"
    # tier must be consistent with mcc
    for _, r in df.iterrows():
        assert R.tier(float(r["lgbm_mcc"])) == r["lgbm_tier"], f"{r['task']} lgbm tier mismatch"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} route_invariance sanity tests PASSED")
