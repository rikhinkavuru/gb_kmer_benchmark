#!/usr/bin/env python3
"""Sanity checks for Upgrade 3 (discriminability dose-response). Fast, synthetic, no torch.
Run:  python src/upgrades/disc_dose/tests/test_sanity.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import _common as C
import run_disc_dose as R
import motif_jaspar
import motif_match


def test_ppm_from_pssm_normalized():
    m = next(x for x in motif_jaspar.load_pssms("vertebrates") if x["name"].upper() == "SP1")
    ppm = R.ppm_from_pssm(m["pssm"])
    assert np.allclose(ppm.sum(0), 1.0, atol=1e-5), "PPM columns must sum to 1"
    assert (ppm >= 0).all() and R.motif_ic(ppm) > 5.0, "SP1 should carry several bits of IC"


def test_background_composition_matches():
    rng = np.random.RandomState(0)
    freqs = np.array([0.1, 0.2, 0.3, 0.4])
    seqs = R.make_background(400, 200, freqs, rng)
    codes = motif_match.encode_sequences(seqs); valid = codes < 4
    got = np.array([((codes == b) & valid).sum() / valid.sum() for b in range(4)])
    assert np.allclose(got, freqs, atol=0.03), f"background base freqs should match target, got {got}"


def test_implant_raises_motif_only_auroc():
    rng = np.random.RandomState(1)
    m = next(x for x in motif_jaspar.load_pssms("vertebrates") if x["name"].upper() == "SP1")
    ppm = R.ppm_from_pssm(m["pssm"])
    freqs = np.array([0.25, 0.25, 0.25, 0.25])
    pos = R.implant(R.make_background(300, 150, freqs, rng), ppm, 1.0, rng)   # all positives carry it
    neg = R.make_background(300, 150, freqs, rng)                              # none of the negatives
    hp = motif_match.count_hits_batch(motif_match.encode_sequences(pos), m["pssm"], 1.0)
    hn = motif_match.count_hits_batch(motif_match.encode_sequences(neg), m["pssm"], 1.0)
    d = C.auc_counts(hp.astype(np.int64), hn.astype(np.int64))
    assert d > 0.8, f"implanting SP1 only in positives should drive motif-only AUROC high, got {d:.3f}"


def test_synth_dataset_shapes():
    rng = np.random.RandomState(2)
    m = next(x for x in motif_jaspar.load_pssms("vertebrates") if x["name"].upper() == "TBP")
    ppm = R.ppm_from_pssm(m["pssm"])
    seqs, y = R.synth_dataset(ppm, 50, 120, np.array([.25, .25, .25, .25]), 0.5, 0.0, rng)
    assert len(seqs) == 100 and y.sum() == 50 and all(len(s) == 120 for s in seqs)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} disc_dose sanity tests PASSED")
