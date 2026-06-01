#!/usr/bin/env python3
"""Sanity checks for Upgrade 7 (predictive LOTO). Fast, no torch/LightGBM.
Run:  python src/upgrades/route1_predict/tests/test_sanity.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import run_route1_predict as R


def test_loto_recovers_linear_signal():
    # exact linear data -> LOTO predictions should be near-perfect
    x = np.linspace(0.5, 1.0, 11)
    y = 2.0 * x - 0.7
    pred = R.loto_predictions(x, y)
    assert np.max(np.abs(pred - y)) < 1e-6, "LOTO must recover an exact line"


def test_loto_shape_and_no_leak():
    rng = np.random.RandomState(0)
    x = rng.rand(8); y = rng.rand(8)
    pred = R.loto_predictions(x, y)
    assert pred.shape == (8,)
    # held-out prediction for point i must NOT equal y[i] in general (no leakage)
    assert not np.allclose(pred, y)


def test_r2_of_noise_is_low():
    rng = np.random.RandomState(1)
    x = rng.rand(11); y = rng.rand(11)                 # unrelated
    pred = R.loto_predictions(x, y)
    ss_res = np.sum((y - pred) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert r2 < 0.5, "unrelated data should not yield high held-out R^2"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} route1_predict sanity tests PASSED")
