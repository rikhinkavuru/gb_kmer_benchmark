"""Sanity tests for F003 homology measurement -- exact Jaccard on tiny controlled cases."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import measure_homology as H  # noqa: E402


def _jaccard_bruteforce(a, b, k=8):
    def kset(s):
        return {s[i:i + k] for i in range(len(s) - k + 1) if set(s[i:i + k]) <= set("ACGT")}
    A, B = kset(a), kset(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0


def test_kmer_ids_distinct_and_acgt_only():
    ids = H.kmer_ids("ACGTACGTACGT")
    assert ids.dtype == np.int64 and len(ids) == len(set(ids.tolist()))   # distinct
    # a non-ACGT base drops every window touching it
    assert H.kmer_ids("N" * 3) .size == 0
    assert H.kmer_ids("ACGTACGT").size == 1          # exactly one 8-mer


def test_identical_sequence_jaccard_is_one():
    s = "ACGTACGTTTGGCCAACGTACGT"
    Mtr = H.presence_matrix([s, "TTTTTTTTTTTTTTTT"])
    Mte = H.presence_matrix([s])
    mj = H.max_jaccard_to_train(Mte, Mtr)
    assert abs(mj[0] - 1.0) < 1e-9          # exact duplicate present in train -> Jaccard 1.0


def test_matches_bruteforce_on_random_pairs():
    rng = np.random.RandomState(0)
    seqs = ["".join(rng.choice(list("ACGT"), size=60)) for _ in range(6)]
    train = seqs[:4]
    test = seqs[4:] + [train[0]]            # last test seq duplicates a train seq
    Mtr = H.presence_matrix(train)
    Mte = H.presence_matrix(test)
    mj = H.max_jaccard_to_train(Mte, Mtr)
    for i, ts in enumerate(test):
        expect = max(_jaccard_bruteforce(ts, tr) for tr in train)
        assert abs(mj[i] - expect) < 1e-9, (i, mj[i], expect)
    assert abs(mj[-1] - 1.0) < 1e-9         # the planted duplicate


def test_disjoint_sequences_low_jaccard():
    Mtr = H.presence_matrix(["A" * 50 + "CGCGCGCGCGCG"])
    Mte = H.presence_matrix(["TATATATATATATATATATATATA"])
    mj = H.max_jaccard_to_train(Mte, Mtr)
    assert mj[0] < 0.1                       # share essentially no 8-mers


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("ALL F003 SANITY TESTS PASSED")
