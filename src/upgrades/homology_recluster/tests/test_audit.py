"""Sanity tests for the verification-audit scripts (run_audit.py, run_covmode_sensitivity.py,
run_linclust_calibration.py). No MMseqs2 run, no model fit.
Run: bibm/lowcpu.sh <venv>/bin/python -m pytest -q -p no:cacheprovider src/upgrades/homology_recluster/tests
"""
import os
import sys

import numpy as np
from sklearn.metrics import matthews_corrcoef

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import run_audit as RA  # noqa: E402


def test_fast_mcc_equals_sklearn_binary_and_multiclass():
    rng = np.random.RandomState(3)
    for K in (2, 3):
        for _ in range(5):
            y = rng.randint(0, K, 500); p = np.where(rng.rand(500) < 0.7, y, rng.randint(0, K, 500))
            assert abs(RA.fast_mcc(y, p, K) - matthews_corrcoef(y, p)) < 1e-12
    y = np.r_[np.ones(99, int), 0]; p = np.ones(100, int)          # predict-all-positive corner
    assert RA.fast_mcc(y, p, 2) == 0.0 == matthews_corrcoef(y, p)


def test_component_bootstrap_resamples_whole_components():
    comp = np.array([5, 5, 5, 7, 9, 9])
    idx = np.arange(6)
    order, starts, sizes = RA.comp_groups(idx, comp)
    assert list(sizes) == [3, 1, 2] and list(comp[order]) == [5, 5, 5, 7, 9, 9]
    # a model that is right on component 5 and wrong elsewhere: every resample is a union of whole components
    y = np.array([1, 1, 1, 0, 0, 1]); p = np.array([1, 1, 1, 1, 1, 0])
    bt = RA.comp_boot(y[order], p[order], 2, (order, starts, sizes), np.random.RandomState(0), B=200)
    assert np.isfinite(bt).any()
    assert np.nanmax(bt) <= 1.0 + 1e-9 and np.nanmin(bt) >= -1.0 - 1e-9


def test_canonical_strand():
    assert RA.canon("AACCG") == RA.canon("CGGTT") == "AACCG"


def test_covmode_args_switch_only_the_coverage_mode():
    import run_covmode_sensitivity as CV
    import run_cluster as RC
    diff = [(a, b) for a, b in zip(RC.SEARCH_ARGS, CV.ARGS) if a != b]
    assert diff == [("0", "1")] and CV.ARGS[CV.ARGS.index("--cov-mode") + 1] == "1"


def test_linclust_calibration_uses_the_same_synthetic_set_size():
    import run_linclust_calibration as LC
    seqs, meta = LC.synthetic_set()
    assert len(seqs) == 2 * 40 * 11 + 400
    assert sum(1 for m in meta if m[1] == -1) == 400
