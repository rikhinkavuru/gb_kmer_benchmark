#!/usr/bin/env python3
"""Brute-force checks for the interval helpers used by the genomic (cCRE / RepeatMasker)
annotation in run_psm_sensitivity.py, and for the arm-building invariants that do not need data.

Run:  python src/upgrades/psm_sensitivity/tests/test_genomic_helpers.py   (prints PASS / raises)"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import run_psm_sensitivity as R  # noqa: E402
import psm_core as P  # noqa: E402


def test_covered_matches_bruteforce():
    rng = np.random.RandomState(7)
    L = 5000
    s = rng.randint(0, L - 50, 300)
    e = s + rng.randint(1, 120, 300)
    paint = np.zeros(L + 200, bool)
    for a, b in zip(s, e):
        paint[a:b] = True
    merged = R._merge(s, e)
    ms, me, _ = merged
    assert np.all(ms[1:] > me[:-1]), "merged intervals must be disjoint and sorted"
    qs = rng.randint(0, L, 500)
    qe = qs + rng.randint(1, 600, 500)
    got = R._covered(merged, qs, qe)
    want = np.array([paint[a:b].sum() for a, b in zip(qs, qe)], float)
    assert np.array_equal(got, want), np.flatnonzero(got != want)[:5]
    assert np.all(R._covered(R._merge(np.array([], np.int64), np.array([], np.int64)), qs, qe) == 0)


def test_random_take_nested_and_uniform():
    pri = R.pri(10003, 0, 1000)
    a = set(P.random_take(1000, 100, pri).tolist())
    b = set(P.random_take(1000, 300, pri).tolist())
    assert a <= b and len(a) == 100 and len(b) == 300


def test_trim_band_is_nested():
    rng = np.random.RandomState(3)
    lp, ln = rng.normal(1, 1, 500), rng.normal(-1, 1, 2000)
    prev = set(range(2000))
    for q in (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45):
        k = set(P.trim_band(lp, ln, q).tolist())
        assert k <= prev
        prev = k


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
