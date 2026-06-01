#!/usr/bin/env python3
"""Sanity checks for Upgrade 2 (composition-equalized cleaning). Fast, synthetic, no torch.

Run:  python src/upgrades/composition_clean/tests/test_sanity.py   (prints PASS / raises)
Also discoverable by pytest (def test_*)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))           # the module dir
sys.path.insert(0, os.path.join(HERE, "..", ".."))     # src/upgrades (for _common)
import _common as C
import run_composition_clean as R


def _rand_seq(gc, length, rng):
    """Random ACGT sequence with expected GC fraction ``gc``."""
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]    # A,C,G,T
    return "".join(rng.choice(list("ACGT"), size=length, p=p))


def test_match_proportional_balances_propensity():
    rng = np.random.RandomState(0)
    p_pos = rng.uniform(0.55, 0.9, 400)                 # positives high-propensity
    p_neg = rng.uniform(0.1, 0.9, 6000)                 # negatives spread
    keep = R.match_proportional(p_pos, p_neg, 20, rng)
    assert 0 < len(keep) < len(p_neg), "match should keep a strict, non-empty subset"
    before = abs(p_neg.mean() - p_pos.mean())
    after = abs(p_neg[keep].mean() - p_pos.mean())
    assert after < before, f"kept-neg propensity should move toward positives ({after:.3f} !< {before:.3f})"


def test_gc_match_collapses_gc_separation():
    rng = np.random.RandomState(1)
    pos = [_rand_seq(g, 120, rng) for g in rng.uniform(0.45, 0.70, 300)]
    neg = [_rand_seq(g, 120, rng) for g in rng.uniform(0.25, 0.70, 1500)]   # overlapping, shifted low
    gp, gn = C.gc_content(pos), C.gc_content(neg)
    before = C.auc_float(gp, gn)
    keep = R.gc_match(gp, gn, 25, rng)
    after = C.auc_float(gp, gn[keep])
    assert before > 0.6, f"synthetic pos should start GC-separated ({before:.3f})"
    assert abs(after - 0.5) < abs(before - 0.5), f"GC AUROC should move toward 0.5 ({before:.3f}->{after:.3f})"


def test_smd_zero_for_identical_and_positive_for_shifted():
    rng = np.random.RandomState(2)
    a = rng.normal(0, 1, (500, 4))
    assert np.nanmax(np.abs(R.smd(a, a.copy()))) < 1e-9, "SMD of identical samples must be ~0"
    b = a + np.array([1.0, 0, 0, 0])
    s = R.smd(b, a)
    assert s[0] > 0.5 and abs(s[1]) < 0.2, "SMD should flag the shifted feature only"


def test_comp_signature_and_gc_consistency():
    rng = np.random.RandomState(3)
    seqs = [_rand_seq(0.8, 200, rng) for _ in range(20)]
    sig = C.comp_signature(seqs, ks=(1, 2))
    assert sig.shape == (20, 20), "4 mono + 16 dinuc = 20 features"
    # GC from mononucleotide block (cols C=1,G=2) ~ GC from gc_content
    gc_from_sig = sig[:, 1] + sig[:, 2]
    assert np.allclose(gc_from_sig, C.gc_content(seqs), atol=1e-5), "GC must agree across featurizers"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} composition_clean sanity tests PASSED")
