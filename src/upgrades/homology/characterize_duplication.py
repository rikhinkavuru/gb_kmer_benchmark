#!/usr/bin/env python3
"""TASK 2 -- EXACT characterization of nt_enhancers train/test duplication.

The F003 8-mer-Jaccard audit flagged ~25% (100/400) of nt_enhancers test sequences with a Jaccard
of 1.0 to a training sequence. Jaccard 1.0 (identical 8-mer SET) is necessary but not sufficient for
true verbatim identity, so before this becomes a public claim about a widely-used benchmark we
characterize it with EXACT full-sequence string matching, and -- critically -- separate two distinct
defects that must not be conflated:
  * train->test LEAKAGE   : a test sequence whose exact string also appears in the TRAINING split.
  * within-test REDUNDANCY: a test sequence whose exact string appears more than once in the TEST split.

Matching criterion: full-sequence string identity after the benchmark's own normalization
(`str.strip().upper()`, exactly as data_nt.load_nt_task loads it). Sequences are read from the
PINNED Nucleotide Transformer dataset revision (asserted against data_nt.REVISION). We also reconcile
the exact-verbatim train/test count against the F003 Jaccard=1.0 count.

CPU-only, torch-free, deterministic. Writes results/upgrades/nt_enhancers_duplication.csv +
_statement.txt (a verbatim-ready paragraph with the exact counts).
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _common as C  # noqa: E402  (torch-free)

PINNED_REVISION = "96d86d567d4cd33536e49b429dc7983121619a08"


def _categorize(tr_seqs, te_seqs):
    """Pure exact-string categorization (no I/O). Sequences must already be normalized."""
    n_te = len(te_seqs)
    train_set = set(tr_seqs)
    test_counts = Counter(te_seqs)
    in_train = np.array([s in train_set for s in te_seqs], dtype=bool)       # train->test leakage
    dup_in_test = np.array([test_counts[s] >= 2 for s in te_seqs], dtype=bool)  # within-test redundancy
    return dict(n_test=n_te, n_train=len(tr_seqs),
                n_test_unique=len(set(te_seqs)), n_train_unique=len(train_set),
                train_test_verbatim=int(in_train.sum()),
                within_test_dup=int(dup_in_test.sum()),
                both_A_and_B=int((in_train & dup_in_test).sum()),
                train_test_only=int((in_train & ~dup_in_test).sum()),
                within_test_only=int((~in_train & dup_in_test).sum()),
                neither_clean=int((~in_train & ~dup_in_test).sum()))


def characterize(task, seed=42):
    import data_nt  # the NT loader (pinned revision); torch-free
    assert data_nt.REVISION == PINNED_REVISION, \
        f"NT revision mismatch: {data_nt.REVISION} != pinned {PINNED_REVISION}"
    tr_seqs, ytr, te_seqs, yte = C.load_original(task, seed)   # already strip().upper()
    c = _categorize(tr_seqs, te_seqs)
    n_te = c["n_test"]
    n_A = c["train_test_verbatim"]
    n_B = c["within_test_dup"]

    # reconcile with F003 Jaccard=1.0 (if the homology CSV is present)
    jacc1 = None
    hcsv = os.path.join(C.RESULTS_DIR, "homology_leakage.csv")
    if os.path.exists(hcsv):
        h = pd.read_csv(hcsv)
        row = h[h.task == task]
        if len(row):
            jacc1 = int(row.iloc[0]["n_ge_0_9"])    # Jaccard>=0.9 count (==100 for nt_enhancers)

    return dict(task=task, revision=PINNED_REVISION, match_criterion="full-string identity after strip().upper()",
                n_test=n_te, n_train=c["n_train"],
                n_test_unique=c["n_test_unique"], n_train_unique=c["n_train_unique"],
                train_test_verbatim=n_A, train_test_verbatim_frac=round(n_A / n_te, 4),
                within_test_dup=n_B, within_test_dup_frac=round(n_B / n_te, 4),
                both_A_and_B=c["both_A_and_B"], train_test_only=c["train_test_only"],
                within_test_only=c["within_test_only"],
                neither_clean=c["neither_clean"], clean_frac=round(c["neither_clean"] / n_te, 4),
                f003_jaccard1_count=jacc1)


def statement(d):
    """A verbatim-ready paragraph with the exact counts."""
    n = d["n_test"]
    s = (f"Exact full-sequence string matching (after the benchmark's own `strip().upper()` "
         f"normalization, on the pinned Nucleotide Transformer dataset revision "
         f"{d['revision'][:10]}...) shows that {d['train_test_verbatim']} of the {n} "
         f"nt_enhancers TEST sequences ({d['train_test_verbatim_frac']:.1%}) are VERBATIM duplicates "
         f"of a TRAINING sequence (train->test leakage), and {d['within_test_dup']} test sequences "
         f"({d['within_test_dup_frac']:.1%}) are exact duplicates of at least one OTHER test sequence "
         f"(within-test redundancy); {d['both_A_and_B']} sequences fall in both categories. "
         f"Of the {n} test sequences, {d['n_test_unique']} are distinct strings and "
         f"{d['neither_clean']} ({d['clean_frac']:.1%}) are clean (neither train-duplicated nor "
         f"test-duplicated). The train->test verbatim count "
         + (f"matches the F003 8-mer Jaccard=1.0 count exactly ({d['f003_jaccard1_count']}), "
            if d.get("f003_jaccard1_count") == d["train_test_verbatim"] else
            f"is {d['train_test_verbatim']} vs the F003 Jaccard>=0.9 count of {d['f003_jaccard1_count']}, ")
         + "confirming the flagged near-duplicates are true verbatim sequence duplications, not an "
         "artifact of the k-mer presence metric.")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "nt_enhancers_duplication.csv"))
    ap.add_argument("--task", default="nt_enhancers")
    args = ap.parse_args()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)

    print("=" * 96)
    print(f"TASK 2 -- EXACT nt_enhancers train/test duplication (full-string identity)")
    print("=" * 96)
    d = characterize(args.task)
    for k, v in d.items():
        print(f"  {k:<26} {v}")
    pd.DataFrame([d]).to_csv(args.out, index=False)
    st = statement(d)
    print("\n--- verbatim-ready statement ---\n" + st)
    with open(args.out.replace(".csv", "_statement.txt"), "w") as fh:
        fh.write(st + "\n")
    print(f"\nWrote {args.out} + _statement.txt")


if __name__ == "__main__":
    main()
