"""Sanity tests for substance-pass-2: clean-subset correlation (Task 1) + exact duplication (Task 2).
Torch-free; rank statistics and exact-string set logic."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import clean_subset_correlation as T1  # noqa: E402
import characterize_duplication as T2  # noqa: E402


# ---------- Task 1 ----------
def test_partial_spearman_equals_pooled_when_covariate_independent():
    rng = np.random.RandomState(0)
    x = rng.rand(40)
    y = x + 0.1 * rng.rand(40)                 # strong x->y
    suite = (rng.rand(40) < 0.5)               # suite independent of x,y
    from scipy.stats import spearmanr
    pooled = spearmanr(x, y)[0]
    rp, _ = T1.partial_spearman_suite(x, y, suite)
    assert abs(rp - pooled) < 0.06             # controlling for an independent covariate ~ no change


def test_jackknife_returns_n_rows_and_brackets_full():
    from scipy.stats import spearmanr
    x = np.array([1., 2, 3, 4, 5, 6]); y = np.array([1., 2, 3, 4, 5, 6])
    d = T1.jackknife(x, y, np.array([f"t{i}" for i in range(6)]))
    assert len(d) == 6 and (d.rho > 0.9).all()


def test_n11_reproduces_committed_stats():
    src = os.path.join(T1.C.RESULTS_DIR, "route1_predict.csv")
    df = pd.read_csv(src)
    row, loo = T1.stats_for(df, "n11")
    assert abs(row["pooled_rho"] - 0.7364) < 1e-3, row["pooled_rho"]      # paper +0.74
    assert abs(row["gb_rho"] - 0.8857) < 1e-3, row["gb_rho"]              # paper GB +0.89
    assert abs(row["jack_rho_lo"] - 0.685) < 5e-3 and abs(row["jack_rho_hi"] - 0.830) < 5e-3  # paper [+0.68,+0.83]
    assert row["jack_all_sig"] is True


# ---------- Task 2 ----------
def test_categorize_separates_train_test_from_within_test():
    train = ["AAAA", "CCCC", "GGGG"]
    test = ["AAAA",   # train->test verbatim (A only)
            "TTTT",   # clean
            "ACAC", "ACAC",  # within-test dup pair (B only) -- two test entries, same string
            "CCCC"]   # train->test verbatim (A only)
    c = T2._categorize(train, test)
    assert c["n_test"] == 5
    assert c["train_test_verbatim"] == 2          # AAAA, CCCC
    assert c["within_test_dup"] == 2              # the two ACAC entries
    assert c["both_A_and_B"] == 0
    assert c["neither_clean"] == 1                # TTTT
    assert c["n_test_unique"] == 4                # AAAA, TTTT, ACAC, CCCC


def test_categorize_overlap_counts_both():
    train = ["AAAA"]
    test = ["AAAA", "AAAA"]                        # in train AND duplicated within test
    c = T2._categorize(train, test)
    assert c["train_test_verbatim"] == 2 and c["within_test_dup"] == 2 and c["both_A_and_B"] == 2
    assert c["neither_clean"] == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("ALL PASS-2 SANITY TESTS PASSED")
