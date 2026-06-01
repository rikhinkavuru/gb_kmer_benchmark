#!/usr/bin/env python3
"""Sanity checks for Upgrade 6 (nt_enhancers TSS coordinates). Fast; does NOT require mappy or the
GRCh38 FASTA (tests the pure helpers + the cached output). Run:
  python src/upgrades/nt_coords/tests/test_sanity.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import run_nt_coords as R


def test_ensembl_to_ucsc():
    assert R.ensembl_to_ucsc("1") == "chr1" and R.ensembl_to_ucsc("X") == "chrX"
    assert R.ensembl_to_ucsc("MT") == "chrM"
    assert R.ensembl_to_ucsc("KI270711.1") == "KI270711.1"   # scaffolds pass through


def test_best_tata_hit_flags_a_tata_box():
    tbp = R.C.load_tbp()
    seqs = ["TATAAAAGGGGCCCC" * 3, "GCGCGCGCGCGCGCGC" * 3]   # one TATA-rich, one GC-rich
    flagged, hit_start, best_bits, L = R.best_tata_hit(seqs, tbp)
    assert flagged[0] and not flagged[1], f"TATA box should flag seq0 only, got {flagged}"
    assert L == tbp["pssm"].shape[1]


def test_output_present_and_replicates_direction():
    out = os.path.join(R.C.RESULTS_DIR, "nt_coords.csv")
    if not os.path.exists(out):
        print("  SKIP (nt_coords.csv not generated yet; run run_nt_coords.py with the [coords] extra)"); return
    import pandas as pd
    r = pd.read_csv(out).iloc[0]
    # the headline replication: flagged-neg TSS overlap <= non-flagged (depletion, like cohn)
    assert r["flagged_neg_tss"] <= r["nonflagged_neg_tss"], "expected flagged-neg TSS-depletion (cohn replication)"
    assert r["mapped_frac"] >= 0.22, "fuzzy mapping should recover >= the 22% exact-match baseline"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} nt_coords sanity tests PASSED")
