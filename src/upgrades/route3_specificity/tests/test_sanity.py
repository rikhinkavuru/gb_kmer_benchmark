#!/usr/bin/env python3
"""Sanity checks for the Route 3 specificity module (negative-control motif panel).
Fast, CPU-only, no torch/LightGBM. Verifies the control-PWM construction is correct
BEFORE trusting the measured AUROCs:

  1. Column-scrambling preserves per-column IC and total IC EXACTLY (permutation
     only reorders columns), within fp tolerance.
  2. Random IC-matched PWMs hit the target total IC within the stated tolerance.
  3. A column-scrambled-TBP scan on a small synthetic pos/neg set (no real signal)
     gives motif-only AUROC near 0.5 -- i.e. the control assay does not manufacture
     a skew by itself.

Run:  python src/upgrades/route3_specificity/tests/test_sanity.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))         # run_route3_specificity
sys.path.insert(0, os.path.join(HERE, "..", ".."))   # _common
import run_route3_specificity as R
import _common as C
import motif_match


def test_column_scramble_preserves_ic():
    tbp = C.load_tbp()
    pssm = tbp["pssm"]
    base_ppm = R.pssm_to_ppm(pssm)
    base_col_ic = np.sort(R.column_ic(base_ppm))          # multiset of per-column ICs
    base_total = R.total_ic(base_ppm)
    panel = R.scrambled_tbp_panel(pssm, n_perm=20, seed=42)
    assert len(panel) == 20, "expected 20 scrambled controls"
    for m in panel:
        ppm = R.pssm_to_ppm(m["pssm"])
        # per-column IC multiset identical (permutation only reorders)
        assert np.allclose(np.sort(R.column_ic(ppm)), base_col_ic, atol=1e-6), \
            "column-scrambling must preserve the per-column IC multiset"
        # total IC identical
        assert abs(R.total_ic(ppm) - base_total) < 1e-6, \
            "column-scrambling must preserve total IC exactly"
    # and at least one permutation actually reorders the columns (not all identity)
    reordered = any(not np.array_equal(R.pssm_to_ppm(m["pssm"]), base_ppm) for m in panel)
    assert reordered, "with 20 permutations at least one must differ from the original order"


def test_ppm_pssm_roundtrip():
    # the IC machinery relies on PSSM<->PPM inversion being exact for motif_jaspar PSSMs
    tbp = C.load_tbp()
    ppm = R.pssm_to_ppm(tbp["pssm"])
    assert np.allclose(ppm.sum(0), 1.0, atol=1e-6), "recovered PPM columns must sum to 1"
    re_pssm = R.ppm_to_pssm(ppm)
    assert np.allclose(re_pssm, tbp["pssm"], atol=1e-4), "PSSM->PPM->PSSM must round-trip"


def test_random_panel_matches_target_ic():
    tbp = C.load_tbp()
    L = tbp["pssm"].shape[1]
    target = R.total_ic(R.pssm_to_ppm(tbp["pssm"]))
    tol = 0.5
    panel = R.random_ic_panel(target, L, n_pwm=10, seed=42, tol_bits=tol)
    assert len(panel) == 10, f"expected 10 random IC-matched PWMs, got {len(panel)}"
    for m in panel:
        ppm = R.pssm_to_ppm(m["pssm"])
        assert ppm.shape == (4, L), "random PWM must be (4, L)"
        assert np.allclose(ppm.sum(0), 1.0, atol=1e-6), "random PPM columns must sum to 1"
        assert abs(R.total_ic(ppm) - target) <= tol + 1e-9, \
            f"random PWM total IC {R.total_ic(ppm):.3f} must be within {tol} of target {target:.3f}"


def test_scrambled_scan_on_synthetic_is_near_half():
    # Synthetic pos/neg with NO class-specific TATA structure: i.i.d. random ACGT for
    # both classes (same composition) => a scrambled-TBP scan must give AUROC ~ 0.5.
    rng = np.random.RandomState(0)
    bases = np.array(list("ACGT"))
    def rand_seqs(n, L=200):
        return ["".join(bases[rng.randint(0, 4, L)]) for _ in range(n)]
    pos = rand_seqs(400)
    neg = rand_seqs(400)
    pos_codes = motif_match.encode_sequences(pos)
    neg_codes = motif_match.encode_sequences(neg)
    tbp = C.load_tbp()
    panel = R.scrambled_tbp_panel(tbp["pssm"], n_perm=20, seed=42)
    boot_rng = np.random.RandomState(42)
    mean, sd, lo, hi, members = R.panel_auroc_stats(pos_codes, neg_codes, panel, 1.3, 200, boot_rng)
    assert abs(mean - 0.5) < 0.05, \
        f"scrambled-TBP scan on structureless synthetic data should give AUROC~0.5, got {mean:.3f}"
    # the CI on the panel mean should bracket 0.5 (no manufactured skew)
    assert lo <= 0.5 <= hi, f"panel-mean CI [{lo:.3f},{hi:.3f}] should contain 0.5 on null data"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} route3_specificity sanity tests PASSED")
