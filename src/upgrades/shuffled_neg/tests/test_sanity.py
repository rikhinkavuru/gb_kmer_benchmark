#!/usr/bin/env python3
"""Sanity checks for Upgrade 5 (shuffled-negative control). Tests the dinucleotide-preserving
shuffle directly (pure Python; no torch, no LightGBM). Run:
  python src/upgrades/shuffled_neg/tests/test_sanity.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import dinuc_shuffle as D


def test_dinuc_exactly_preserved_many():
    rng = np.random.RandomState(42)
    n_ok = 0
    seqs = ["".join(rng.choice(list("ACGT"), size=rng.randint(40, 300))) for _ in range(300)]
    seqs += ["AT" * 80, "GC" * 80, "AAATTTGGGCCC" * 10, "TATAAAAGGGG" * 12]   # skewed compositions
    for s in seqs:
        sh = D.dinuc_shuffle(s, rng)
        assert D.verify_dinuc_preserved(s, sh), f"dinuc not preserved for len {len(s)}"
        n_ok += 1
    print(f"    ({n_ok} sequences, all dinucleotide-preserved)")


def test_gc_and_mono_identical():
    rng = np.random.RandomState(1)
    s = "GGGCCCATATATGCGCAAATTT" * 6
    sh = D.dinuc_shuffle(s, rng)
    for b in "ACGT":
        assert s.count(b) == sh.count(b), f"mononucleotide {b} count changed"
    gc = lambda x: (x.count("G") + x.count("C")) / len(x)
    assert abs(gc(s) - gc(sh)) < 1e-12


def test_shuffle_actually_permutes():
    rng = np.random.RandomState(7)
    s = "".join(rng.choice(list("ACGT"), size=400))
    diffs = sum(D.dinuc_shuffle(s, rng) != s for _ in range(20))
    assert diffs >= 18, "shuffle should almost always produce a different ordering"


def test_dinuc_counts_helper():
    assert D.dinuc_counts("ACGT") == {"AC": 1, "CG": 1, "GT": 1}
    assert D.dinuc_counts("AAAA") == {"AA": 3}


def test_degenerate_inputs_safe():
    rng = np.random.RandomState(0)
    assert D.dinuc_shuffle("A", rng) == "A"
    assert D.dinuc_shuffle("", rng) == ""
    assert D.verify_dinuc_preserved("AAAA", D.dinuc_shuffle("AAAA", rng))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} shuffled_neg sanity tests PASSED")
