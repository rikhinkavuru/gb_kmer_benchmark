#!/usr/bin/env python3
"""Sanity checks for Upgrade 1 (frozen FM-probe) -- the torch-SIDE logic only.

Deliberately does NOT import _common / LightGBM: torch and LightGBM each load their own
OpenMP and segfault if co-resident (the very reason embedding is a separate process). The
head-training side (fit_eval_boot) is covered by the composition_clean sanity test.

Tests the .npz cache round-trip (numpy only, always runs) and the pooling math
(constructs a fake hidden state -- no model download -- skipped if torch is absent).

Run:  python src/upgrades/fm_probe/tests/test_sanity.py
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))           # the module dir (extract_embeddings)
import extract_embeddings as X                          # torch-free at import time


def test_cache_roundtrip():
    mean = np.arange(2 * X.HIDDEN_DIM, dtype=np.float32).reshape(2, X.HIDDEN_DIM)
    sep = mean[::-1].copy()
    y = np.array([0, 1])
    with tempfile.TemporaryDirectory() as d:
        X.save_cached(d, "toytask", "orig", "train", mean, sep, y)
        got = X.load_cached(d, "toytask", "orig", "train")
        assert got is not None and np.array_equal(got["mean"], mean)
        assert np.array_equal(got["sep"], sep) and np.array_equal(got["y"], y)
        assert X.load_cached(d, "toytask", "orig", "test") is None, "missing split -> None"


def test_pool_batch_excludes_specials_and_picks_sep():
    if not X.available():
        print("  SKIP test_pool_batch (torch not installed)"); return
    import torch
    # batch of 2; H=3. seq0: 3 nucleotides + SEP + PAD ; seq1: 2 nucleotides + SEP + 2 PAD
    H = 3
    last = torch.tensor([
        [[1., 1, 1], [2, 2, 2], [3, 3, 3], [9, 9, 9], [0, 0, 0]],     # nuc,nuc,nuc,SEP,PAD
        [[4., 4, 4], [6, 6, 6], [8, 8, 8], [0, 0, 0], [0, 0, 0]],     # nuc,nuc,SEP,PAD,PAD
    ])
    # special_tokens_mask: True at SEP and PAD
    special = torch.tensor([[False, False, False, True, True],
                            [False, False, True, True, True]])
    pad = torch.tensor([[True, True, True, True, False],             # non-pad incl SEP
                        [True, True, True, False, False]])
    mean, sep = X._pool_batch(last, special, pad)
    # mean over nucleotide tokens only
    assert np.allclose(mean[0], [2, 2, 2]), f"seq0 mean over 3 nuc = 2, got {mean[0]}"
    assert np.allclose(mean[1], [5, 5, 5]), f"seq1 mean over (4,6)/(... ) = 5, got {mean[1]}"
    # sep = last non-pad position (the SEP row)
    assert np.allclose(sep[0], [9, 9, 9]), f"seq0 SEP row, got {sep[0]}"
    assert np.allclose(sep[1], [8, 8, 8]), f"seq1 SEP row, got {sep[1]}"


def test_constants_pinned():
    assert X.MODEL_REVISION and len(X.MODEL_REVISION) == 40, "model revision must be a pinned 40-char sha"
    assert X.HIDDEN_DIM == 128 and X.POOLINGS == ("mean", "sep")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} fm_probe sanity tests PASSED")
