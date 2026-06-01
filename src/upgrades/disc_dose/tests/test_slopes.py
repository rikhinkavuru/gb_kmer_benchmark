#!/usr/bin/env python3
"""Sanity checks for DEPTH 5 (per-motif dose-response slopes). Pure numpy/pandas, NO torch,
NO LightGBM. Run:  python src/upgrades/disc_dose/tests/test_slopes.py"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))            # disc_dose/ (run_disc_slopes)
sys.path.insert(0, os.path.join(HERE, "..", ".."))      # src/upgrades (_common)
import run_disc_slopes as R

CSV = os.path.join(HERE, "..", "..", "..", "..", "results", "upgrades", "disc_dose.csv")


def test_ols_slope_recovers_exact_linear():
    """On exact-linear y = a*x + b the OLS slope must equal the planted slope a."""
    x = np.linspace(0.5, 1.0, 9)
    for a, b in [(2.0, -1.0), (0.5, 0.1), (-1.3, 0.4)]:
        y = a * x + b
        assert abs(R.ols_slope(x, y) - a) < 1e-9, (a, R.ols_slope(x, y))


def test_boot_ci_brackets_exact_slope_and_is_tight():
    """On exact-linear data every bootstrap resample yields the same slope (no noise), so the
    CI collapses onto the planted slope."""
    x = np.linspace(0.5, 1.0, 9)
    a = 2.0
    y = a * x + 3.0
    lo, hi = R.boot_slope_ci(x, y, seed=42, B=200)
    assert abs(lo - a) < 1e-6 and abs(hi - a) < 1e-6, (lo, hi)


def test_per_motif_recovers_planted_slopes_synthetic():
    """Build a synthetic disc_dose-like frame with a DIFFERENT exact slope planted per 'motif';
    fit_group must recover each planted slope (and its CI brackets it)."""
    rng = np.random.RandomState(0)
    planted = {"A": 2.0, "B": 1.5, "C": 2.5}
    frames = []
    for m, a in planted.items():
        x = np.linspace(0.5, 1.0, 45)
        y = a * x + 0.2
        frames.append(pd.DataFrame(dict(motif=m, motif_ic=10.0,
                                        motif_auroc_d=x, bench_mcc=y)))
    df = pd.concat(frames, ignore_index=True)
    for m, a in planted.items():
        slope, lo, hi, rho, n = R.fit_group(df[df["motif"] == m])
        assert abs(slope - a) < 1e-9, f"{m}: got {slope}, planted {a}"
        assert lo - 1e-6 <= a <= hi + 1e-6, f"{m}: CI [{lo},{hi}] should bracket {a}"
        assert n == 45 and rho > 0.99


def test_overlap_helper_logic():
    """Replicate the overlap predicate used in the script and check edge cases."""
    def overlap(a, b):
        return not (a[1] < b[0] or b[1] < a[0])
    assert overlap((1.0, 3.0), (2.0, 4.0))       # partial overlap
    assert overlap((1.0, 3.0), (1.5, 2.0))       # nested
    assert not overlap((1.0, 2.0), (2.5, 3.0))   # disjoint
    assert overlap((1.0, 2.0), (2.0, 3.0))       # touch at endpoint counts as overlap


def test_real_csv_per_motif_cis_overlap_pooled():
    """On the real disc_dose.csv (if present) the 3 per-motif slope CIs must all overlap the
    pooled slope CI (the motif-invariance claim)."""
    if not os.path.exists(CSV):
        print("  SKIP test_real_csv_per_motif_cis_overlap_pooled (csv not present)")
        return
    df = pd.read_csv(CSV)
    motifs = sorted(set(df["motif"]))
    assert len(motifs) == 3, f"expected 3 motifs, got {motifs}"

    def overlap(a, b):
        return not (a[1] < b[0] or b[1] < a[0])

    _, plo, phi, _, _ = R.fit_group(df)
    pooled = (plo, phi)
    cis = {}
    for m in motifs:
        slope, lo, hi, rho, n = R.fit_group(df[df["motif"] == m])
        cis[m] = (lo, hi)
        assert overlap((lo, hi), pooled), \
            f"{m} CI [{lo:.3f},{hi:.3f}] must overlap pooled [{plo:.3f},{phi:.3f}]"
    # and pairwise overlap among the three motifs
    ms = list(cis)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            assert overlap(cis[ms[i]], cis[ms[j]]), \
                f"{ms[i]} vs {ms[j]} CIs must overlap: {cis[ms[i]]} {cis[ms[j]]}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} slope sanity tests PASSED")
