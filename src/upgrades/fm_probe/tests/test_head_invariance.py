#!/usr/bin/env python3
"""Sanity checks for DEPTH 4 (FM-head invariance). Pure-pandas/CSV, NO torch, NO LightGBM.
Run:  python src/upgrades/fm_probe/tests/test_head_invariance.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))            # fm_probe/ (run_fm_head_invariance)
sys.path.insert(0, os.path.join(HERE, "..", ".."))      # src/upgrades (_common)
import run_fm_head_invariance as R

CSV = os.path.join(HERE, "..", "..", "..", "..", "results", "upgrades", "fm_head_invariance.csv")


def test_parse_ci():
    assert R.parse_ci("[-0.063,-0.052]") == (-0.063, -0.052)
    assert R.parse_ci("[0.002,0.016]") == (0.002, 0.016)
    assert R.parse_ci(" [-0.003,0.014] ") == (-0.003, 0.014)


def test_ci_excludes_zero():
    # both negative -> excludes 0
    assert R.ci_excludes_zero("[-0.107,-0.056]")
    # both positive -> excludes 0
    assert R.ci_excludes_zero("[0.043,0.053]")
    # straddles 0 -> includes 0
    assert not R.ci_excludes_zero("[-0.003,0.014]")
    assert not R.ci_excludes_zero("[0.000,0.000]")


def test_sign_str():
    assert R.sign_str(-0.05) == "-" and R.sign_str(0.05) == "+" and R.sign_str(0.0) == "+"


def test_sign_significance_consistency():
    # a clearly-negative effect with a CI fully below 0 must read sign '-' AND significant True
    eff, ci = -0.0481, "[-0.053,-0.043]"
    assert R.sign_str(eff) == "-" and R.ci_excludes_zero(ci)
    # a flat control effect with a CI crossing 0 must read NOT significant
    eff2, ci2 = 0.0055, "[-0.003,0.014]"
    assert not R.ci_excludes_zero(ci2)


def test_real_csv_cohn_nt_negative_both_heads():
    """If the real fm_head_invariance.csv exists, cohn & nt must be NEGATIVE for BOTH heads
    in the locked sep/gc_match cell (the head-invariance claim)."""
    if not os.path.exists(CSV):
        print("  SKIP test_real_csv_cohn_nt_negative_both_heads (csv not yet written)")
        return
    import pandas as pd
    df = pd.read_csv(CSV)
    locked = df[(df["pooling"] == "sep") & (df["arm"] == "gc_match")]
    for ds in ["human_enhancers_cohn", "nt_enhancers"]:
        for head in ["lgbm", "lr"]:
            r = locked[(locked["dataset"] == ds) & (locked["head"] == head)]
            assert len(r) == 1, f"missing locked row {ds}/{head}"
            eff = float(r["test_effect_auroc"].iloc[0])
            assert eff < 0, f"{ds}/{head} test_effect_auroc should be negative, got {eff}"
            assert bool(r["significant"].iloc[0]), f"{ds}/{head} should be significant"
            assert r["sign"].iloc[0] == "-", f"{ds}/{head} sign column should be '-'"


def test_real_csv_control_flat_both_heads():
    """Control (drosophila) must be FLAT (not significant) for BOTH heads in the locked cell."""
    if not os.path.exists(CSV):
        print("  SKIP test_real_csv_control_flat_both_heads (csv not yet written)")
        return
    import pandas as pd
    df = pd.read_csv(CSV)
    locked = df[(df["pooling"] == "sep") & (df["arm"] == "gc_match")
                & (df["dataset"] == "drosophila_enhancers_stark")]
    for head in ["lgbm", "lr"]:
        r = locked[locked["head"] == head]
        assert len(r) == 1, f"missing control locked row /{head}"
        assert not bool(r["significant"].iloc[0]), f"control/{head} should be flat (CI includes 0)"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} head-invariance sanity tests PASSED")
