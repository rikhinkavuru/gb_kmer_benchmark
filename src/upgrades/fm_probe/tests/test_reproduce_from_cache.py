#!/usr/bin/env python3
"""Sanity check for Upgrade-1 cache reproduction: assert reproduce_from_cache regenerates the
headline FM deltas from the cached embeddings, TORCH-FREE, matching the canonical RESULTS numbers.

Fast: it recomputes the (rng-free) test_effect POINT estimate for the headline cells directly from
cache and checks they equal the values in fm_paired.csv / fm_paired_dnabert.csv -- the full
bit-identical CI reproduction is reproduce_from_cache.py itself. Run:
  python src/upgrades/fm_probe/tests/test_reproduce_from_cache.py"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import reproduce_from_cache as R
import extract_embeddings as XH
import extract_embeddings_dnabert as XD

CACHE = os.path.join(R.C.ROOT, "cache", "fm_embeddings")
CANON = R.C.RESULTS_DIR


def _canon(csv, **filt):
    df = pd.read_csv(os.path.join(CANON, csv))
    for k, v in filt.items():
        df = df[df[k] == v]
    return float(df.iloc[0]["test_effect_auroc"])


def test_imports_are_torch_free():
    assert "torch" not in sys.modules, "reproduce_from_cache must import torch-free"


def test_hyenadna_headline_point_matches_canonical():
    # locked cell: sep / lgbm / gc_match
    got = R.point_test_effect("human_enhancers_cohn", "sep", "lgbm", "gc_match", CACHE, XH.load_cached)
    want = _canon("fm_paired.csv", dataset="human_enhancers_cohn", pooling="sep", head="lgbm", arm="gc_match")
    assert abs(got - want) < 1e-9, f"HyenaDNA cohn locked-cell point {got} != canonical {want}"


def test_dnabert_headline_point_matches_canonical():
    got = R.point_test_effect("human_enhancers_cohn", "mean", "lgbm", "gc_match", CACHE, XD.load_cached)
    want = _canon("fm_paired_dnabert.csv", dataset="human_enhancers_cohn", pooling="mean", head="lgbm", arm="gc_match")
    assert abs(got - want) < 1e-9, f"DNABERT-2 cohn point {got} != canonical {want}"


def test_still_torch_free_after_reproduction_calls():
    assert "torch" not in sys.modules, "reproducing from cache must NOT have loaded torch"


def test_manifest_covers_all_caches():
    import glob
    out = os.path.join(CANON, "fm_embeddings_manifest.csv")
    R.write_manifest(out)
    man = pd.read_csv(out)
    n_npz = len(glob.glob(os.path.join(CACHE, "*.npz")))
    assert len(man) == n_npz and man["sha256"].str.len().eq(64).all(), "manifest must list every .npz with a sha256"


if __name__ == "__main__":
    fns = [test_imports_are_torch_free, test_hyenadna_headline_point_matches_canonical,
           test_dnabert_headline_point_matches_canonical, test_still_torch_free_after_reproduction_calls,
           test_manifest_covers_all_caches]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} reproduce-from-cache sanity tests PASSED")
