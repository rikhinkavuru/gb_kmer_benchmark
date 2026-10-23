#!/usr/bin/env python3
"""Sanity checks for Upgrade E2 (motif-presence null, n=9 -> n=15). Fast, no torch.

Synthetic checks exercise the statistics helpers; the CSV checks assert the shape of the
written outputs (run run_presence_null.py first).

Run:  python src/upgrades/presence_null/tests/test_sanity.py   (prints PASS / raises)
Also discoverable by pytest (def test_*)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))           # the module dir
sys.path.insert(0, os.path.join(HERE, "..", ".."))     # src/upgrades (for _common)
import _common as C
import run_presence_null as R
import pandas as pd

TABLE = os.path.join(C.RESULTS_DIR, "presence_null.csv")
STATS = os.path.join(C.RESULTS_DIR, "presence_null_stats.csv")


# ------------------------------------------------------------ written CSVs
def test_table_has_15_tasks():
    df = pd.read_csv(TABLE)
    assert len(df) == 15, f"presence_null.csv must hold all 15 tasks, got {len(df)}"
    assert df["task"].nunique() == 15, "task names must be unique"
    assert set(df["suite"]) == {"GB", "NT"}, "suites must be GB and NT"
    assert (df["suite"] == "GB").sum() == 9 and (df["suite"] == "NT").sum() == 6
    assert (df["n_enriched_TFs"] >= 0).all(), "enrichment counts cannot be negative"
    assert (df["n_enriched_TFs"] == 0).sum() == 2, "exactly two tasks have zero enriched TFs"
    assert (df["test_MCC"].between(-1, 1)).all(), "MCC out of range"
    assert np.allclose(df["test_MCC"], df["table1_MCC"], atol=R.TABLE1_TOL), \
        "every test_MCC must match the paper's Table 1"


def test_rho_in_range_and_ci_brackets_it():
    s = pd.read_csv(STATS)
    assert len(s) == 3, f"expected 3 subsets (GB/NT/pooled), got {len(s)}"
    assert list(s["n"]) == [9, 6, 15], f"subset sizes must be 9,6,15; got {list(s['n'])}"
    for _, r in s.iterrows():
        assert -1.0 <= r["rho"] <= 1.0, f"{r['subset']}: rho={r['rho']} outside [-1,1]"
        assert 0.0 <= r["p"] <= 1.0 and 0.0 <= r["p_perm"] <= 1.0, f"{r['subset']}: bad p-value"
        for lo, hi in [("fisher_lo", "fisher_hi"), ("boot_lo", "boot_hi")]:
            assert -1.0 <= r[lo] <= r["rho"] + 1e-9, f"{r['subset']}: {lo} above rho"
            assert r["rho"] - 1e-9 <= r[hi] <= 1.0, f"{r['subset']}: {hi} below rho"


# ------------------------------------------------------------- statistics
def test_rho_matches_scipy_and_hits_endpoints():
    from scipy.stats import spearmanr
    rng = np.random.RandomState(0)
    x = rng.normal(size=30)
    y = 2 * x + rng.normal(scale=0.3, size=30)
    assert abs(R._rho(x, y) - float(spearmanr(x, y).statistic)) < 1e-12, "must match scipy"
    assert abs(R._rho(x, np.exp(x)) - 1.0) < 1e-12, "monotone increasing -> rho=+1"
    assert abs(R._rho(x, -x) + 1.0) < 1e-12, "monotone decreasing -> rho=-1"
    assert not np.isfinite(R._rho(x, np.ones(30))), "no rank variance -> nan"


def test_exact_permutation_p_is_exact_and_calibrated():
    x = np.arange(6, dtype=float)
    p, meth, n = R.perm_p(x, x, 1.0, n_mc=10, seed=42)
    assert meth == "exact_enumeration" and n == 720, "n=6 must enumerate all 720 relabelings"
    assert abs(p - 2.0 / 720) < 1e-12, f"only the identity and the reversal give |rho|=1, got p={p}"
    # a null pairing should not be significant
    y = np.array([3.0, 1.0, 5.0, 0.0, 4.0, 2.0])
    p0, _, _ = R.perm_p(x, y, R._rho(x, y), n_mc=10, seed=42)
    assert p0 > 0.1, f"scrambled pairing should be n.s., got p={p0}"


def test_fisher_ci_widens_as_n_shrinks_and_brackets_rho():
    rho = 0.3
    lo9, hi9, _, _ = R.fisher_ci(rho, 9)
    lo50, hi50, _, _ = R.fisher_ci(rho, 50)
    assert lo9 < rho < hi9 and lo50 < rho < hi50, "CI must bracket the point estimate"
    assert (hi9 - lo9) > (hi50 - lo50), "n=9 CI must be wider than n=50"
    assert lo9 < 0 < hi9, "rho=+0.30 at n=9 must not exclude zero"


def test_bc_bootstrap_brackets_and_reports_degenerates():
    rng = np.random.RandomState(1)
    x = rng.normal(size=12)
    y = x + rng.normal(scale=0.05, size=12)          # near-perfectly monotone
    r = R._rho(x, y)
    out = R.bc_bootstrap_ci(x, y, r, B=2000, seed=C.SEED)
    assert out["n_valid"] + out["n_degenerate"] == 2000, "every resample must be accounted for"
    assert out["pct_lo"] <= out["pct_hi"] and out["boot_lo"] <= out["boot_hi"]
    assert out["boot_hi"] > 0.8, f"near-perfect monotone pair should bootstrap high, got {out}"
    # identical seed must give identical limits (determinism)
    out2 = R.bc_bootstrap_ci(x, y, r, B=2000, seed=C.SEED)
    assert out == out2, "bootstrap must be deterministic at a fixed seed"


def test_zero_enrichment_tasks_are_kept_not_dropped():
    """The two tasks with no rows in the enrichment CSVs must still appear, with a 0."""
    df, zeros = R.build_table()
    assert len(df) == 15, "build_table must return all 15 tasks"
    assert set(zeros) == {"dummy_mouse_enhancers_ensembl", "nt_promoter_tata"}, \
        f"unexpected zero-enrichment task set: {sorted(zeros)}"
    for t in zeros:
        assert int(df.loc[df["task"] == t, "n_enriched_TFs"].iloc[0]) == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} presence_null sanity tests PASSED")
