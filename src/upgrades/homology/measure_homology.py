#!/usr/bin/env python3
"""F003 -- TARGETED train/test homology (near-duplicate) measurement.

Defends Route 3 against the homology-leakage objection WITHOUT a suite-wide re-split: we only
MEASURE redundancy, we do not re-fit anything. For each task we compute, for every TEST sequence,
the maximum exact k-mer Jaccard (k=8, presence sets) against the WHOLE training split, and report
the fraction of test sequences whose nearest training neighbour exceeds 0.5 / 0.7 / 0.9. A high
fraction = a homology-leaky split (absolute MCC may be inflated by memorised near-duplicates); a
low fraction = the split is redundancy-clean.

Exact Jaccard (NOT MinHash/LSH): each sequence -> the SET of its distinct 8-mers (2-bit encoded,
windows containing a non-ACGT base are dropped, matching featurize.kmer_spectrum's vocabulary).
J(a,b) = |A n B| / |A u B| computed at FULL dataset scale via a sparse presence-matrix product
(no subsampling -- subsampling would mask leakage, per the spec).

Scope (the contamination flagship + the Route-1/Route-3 anchor promoters):
  human_enhancers_cohn   -- PRIMARY contamination dataset (must be clean for the magnitude claim)
  nt_enhancers           -- second contaminated enhancer set
  nt_promoter_tata       -- Route-1 anchor / Route-3 positive-control promoter
  human_nontata_promoters-- Route-1 anchor (the clean Sp1/GC-box promoter win)

CPU-only, torch-free, seed 42 (only used for the deterministic tie-free reporting; the Jaccard is
exact and seed-independent). Writes results/upgrades/homology_leakage.csv + _interpretation.txt.
"""
import argparse
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import sparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _common as C  # noqa: E402  (adds repo root to path; torch-free)

K = 8
THRESHOLDS = (0.5, 0.7, 0.9)
TASKS = [
    ("human_enhancers_cohn", "contamination flagship"),
    ("nt_enhancers", "contaminated enhancer"),
    ("nt_promoter_tata", "promoter anchor"),
    ("human_nontata_promoters", "promoter anchor"),
]
# ASCII codes for A,C,G,T -> 2-bit; everything else -> -1 (window dropped).
_LUT = np.full(256, -1, dtype=np.int64)
for _b, _v in zip(b"ACGT", range(4)):
    _LUT[_b] = _v
_POW = (4 ** np.arange(K - 1, -1, -1)).astype(np.int64)   # [4^7 .. 4^0]


def kmer_ids(seq):
    """Return the array of DISTINCT 8-mer ids in `seq` (windows with non-ACGT dropped)."""
    codes = _LUT[np.frombuffer(seq.upper().encode("ascii", "ignore"), dtype=np.uint8)]
    if codes.size < K:
        return np.empty(0, dtype=np.int64)
    W = sliding_window_view(codes, K)                 # (L-K+1, K)
    valid = (W >= 0).all(axis=1)
    if not valid.any():
        return np.empty(0, dtype=np.int64)
    ids = W[valid] @ _POW
    return np.unique(ids)


def presence_matrix(seqs):
    """Binary CSR (n, 4**K): row i has 1 at column = 8-mer id for each distinct 8-mer in seq i."""
    rows, cols = [], []
    for i, s in enumerate(seqs):
        ids = kmer_ids(s)
        rows.append(np.full(ids.size, i, dtype=np.int64))
        cols.append(ids)
    r = np.concatenate(rows) if rows else np.empty(0, np.int64)
    c = np.concatenate(cols) if cols else np.empty(0, np.int64)
    M = sparse.csr_matrix((np.ones(r.size, dtype=np.float32), (r, c)),
                          shape=(len(seqs), 4 ** K))
    return M


def max_jaccard_to_train(test_M, train_M, batch=256):
    """For each test row, the MAX exact Jaccard against any train row. Batched sparse product."""
    train_sz = np.asarray(train_M.sum(axis=1)).ravel()        # |B| per train seq
    test_sz = np.asarray(test_M.sum(axis=1)).ravel()          # |A| per test seq
    trainT = train_M.T.tocsc()                                # (V, n_train)
    out = np.zeros(test_M.shape[0], dtype=np.float64)
    for s in range(0, test_M.shape[0], batch):
        e = min(s + batch, test_M.shape[0])
        shared = (test_M[s:e] @ trainT).toarray().astype(np.float64)   # (b, n_train) = |A n B|
        union = test_sz[s:e, None] + train_sz[None, :] - shared        # |A u B|
        with np.errstate(divide="ignore", invalid="ignore"):
            jac = np.where(union > 0, shared / union, 0.0)
        out[s:e] = jac.max(axis=1) if jac.shape[1] else 0.0
    return out


def verdict(frac):
    """Descriptive flag from the >=0.7 near-duplicate fraction."""
    f7 = frac[0.7]
    if f7 >= 0.05:
        return "LEAKY"
    if f7 >= 0.01:
        return "borderline"
    return "clean"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "homology_leakage.csv"))
    ap.add_argument("--datasets", default="")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    tasks = TASKS if not args.datasets else [(t, r) for t, r in TASKS if t in set(args.datasets.split(","))]

    print("=" * 100)
    print(f"F003 -- TRAIN/TEST HOMOLOGY (exact {K}-mer Jaccard, FULL scale, no subsampling)")
    print("=" * 100)
    rows = []
    for task, role in tasks:
        tr, ytr, te, yte = C.load_original(task)
        print(f"\n[{task}] ({role}) train={len(tr)} test={len(te)} -- building {K}-mer presence sets ...",
              flush=True)
        Mtr = presence_matrix(tr)
        Mte = presence_matrix(te)
        mj = max_jaccard_to_train(Mte, Mtr, batch=args.batch)
        frac = {t: float((mj >= t).mean()) for t in THRESHOLDS}
        v = verdict(frac)
        row = dict(task=task, role=role, n_train=len(tr), n_test=len(te),
                   frac_ge_0_5=round(frac[0.5], 5), frac_ge_0_7=round(frac[0.7], 5),
                   frac_ge_0_9=round(frac[0.9], 5),
                   n_ge_0_9=int((mj >= 0.9).sum()),
                   mean_max_jaccard=round(float(mj.mean()), 5),
                   median_max_jaccard=round(float(np.median(mj)), 5),
                   p95_max_jaccard=round(float(np.percentile(mj, 95)), 5),
                   p99_max_jaccard=round(float(np.percentile(mj, 99)), 5),
                   max_max_jaccard=round(float(mj.max()), 5), verdict=v)
        rows.append(row)
        print(f"   near-dup fraction  >=0.5: {frac[0.5]:.4f}   >=0.7: {frac[0.7]:.4f}   "
              f">=0.9: {frac[0.9]:.4f}  (n>=0.9 = {row['n_ge_0_9']})")
        print(f"   max-Jaccard  median={row['median_max_jaccard']:.3f}  p95={row['p95_max_jaccard']:.3f}  "
              f"p99={row['p99_max_jaccard']:.3f}  max={row['max_max_jaccard']:.3f}  ->  {v}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    report = build_report(df)
    print("\n" + report)
    with open(args.out.replace(".csv", "_interpretation.txt"), "w") as fh:
        fh.write(report)
    print(f"\nWrote {args.out} ({len(df)} rows) + interpretation.")


def build_report(df):
    L = []; add = L.append
    add("=" * 100)
    add(f"F003 -- TRAIN/TEST HOMOLOGY: exact {K}-mer Jaccard, full scale, fraction of test sequences")
    add("with a training near-duplicate above each threshold.")
    add("=" * 100)
    hdr = f"{'task':<26}{'role':<24}{'>=0.5':>9}{'>=0.7':>9}{'>=0.9':>9}{'p99 maxJ':>10}{'verdict':>12}"
    add(hdr); add("-" * len(hdr))
    for _, r in df.iterrows():
        add(f"{r['task']:<26}{r['role']:<24}{r['frac_ge_0_5']:>9.4f}{r['frac_ge_0_7']:>9.4f}"
            f"{r['frac_ge_0_9']:>9.4f}{r['p99_max_jaccard']:>10.3f}{r['verdict']:>12}")
    add("")
    flag = df[df["task"] == "human_enhancers_cohn"]
    if len(flag):
        f = flag.iloc[0]
        if f["verdict"] == "clean":
            add(f"FLAGSHIP cohn is redundancy-CLEAN ({f['frac_ge_0_7']:.4f} of test seqs have a >=0.7 train")
            add("near-duplicate). Route 3's contamination magnitude on cohn is therefore NOT a homology")
            add("artifact: the composition shortcut exists independent of train/test redundancy.")
        else:
            add(f"*** WARNING: flagship cohn is NOT clean (verdict={f['verdict']}, >=0.7 frac "
                f"{f['frac_ge_0_7']:.4f}). This SHIFTS FRAMING -- the cohn magnitude could be partly a")
            add("homology artifact and must be addressed before the contamination claim stands. ***")
    add("")
    add("Reading: this MEASURES redundancy only (no re-split, no re-fit). A leaky task's ABSOLUTE MCC")
    add("(Table 1) may be inflated by memorised near-duplicates; but the route assignments and the")
    add("within-TEST composition interventions (gc_match / comp_equalized / shuffled-negative) are")
    add("computed inside a single split and do not depend on train/test redundancy. Exact 8-mer")
    add("Jaccard, full scale, no subsampling.")
    return "\n".join(L)


if __name__ == "__main__":
    main()
