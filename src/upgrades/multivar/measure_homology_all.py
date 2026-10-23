#!/usr/bin/env python3
"""Train/test homology (near-duplicate fraction) for the 11 tasks not in homology_leakage.csv.

Same METRIC as src/upgrades/homology/measure_homology.py: for a test sequence, its max EXACT 8-mer
Jaccard (presence sets, windows with non-ACGT dropped) against EVERY sequence of the training split;
the task statistic is the fraction of test sequences whose max is >= 0.5 / 0.7 / 0.9 (0.7 = the
paper's near-duplicate threshold).  Presence matrices come from measure_homology.presence_matrix
(imported, not re-implemented).  Two implementation changes, neither of which changes any value:
  * the TRAINING split is processed in chunks of 20,000 sequences with a running max (the max over
    the union of chunks is the max over the whole split) -- bounds RAM on a shared machine;
  * the per-row max is taken over the non-zero entries of the sparse shared-count product instead of
    a dense (batch x n_train) array (pairs sharing no 8-mer have Jaccard 0 and never change a max).
  Sanity check (pre-registered): nt_promoter_tata is recomputed and must reproduce
  homology_leakage.csv (frac_ge_0_5/0_7/0_9) exactly.

QUERY-SIDE SUBSAMPLING RULE (fixed before running): if n_test * n_train > 1e9 pairs, the statistic is
estimated on a uniform random sample of 5,000 TEST sequences (numpy RandomState(42), without
replacement), each still compared against the FULL training split (the reference side is never
subsampled -- that is what would hide leakage).  The fraction is then an unbiased binomial estimate;
its Wilson 95% CI is reported.  Applies to demo_human_or_worm, demo_coding_vs_intergenomic_seqs,
human_ensembl_regulatory, human_enhancers_ensembl, human_ocr_ensembl (1.9e9 - 1.3e10 pairs, i.e.
~6 CPU-hours at full scale on this shared machine).  All other tasks are full scale.

Checkpoints one JSON per task under external/multivar/homology/ (resume = skip existing);
--merge writes results/upgrades/multivar_homology.csv.  CPU-only, torch-free, single-threaded.
Usage:
  python src/upgrades/multivar/measure_homology_all.py --tasks t1,t2,...   # compute (resumable)
  python src/upgrades/multivar/measure_homology_all.py --merge            # write the CSV
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _mv  # noqa: E402
sys.path.insert(0, os.path.join(_mv.UPG, "homology"))
import measure_homology as H  # noqa: E402

NEW_TASKS = ["demo_human_or_worm", "human_ensembl_regulatory", "demo_coding_vs_intergenomic_seqs",
             "human_enhancers_ensembl", "dummy_mouse_enhancers_ensembl", "human_ocr_ensembl",
             "drosophila_enhancers_stark", "nt_promoter_all", "nt_promoter_no_tata",
             "nt_enhancers_types", "nt_splice_sites_all"]
CHECK_TASK = "nt_promoter_tata"          # must reproduce homology_leakage.csv
PAIR_CAP = 1e9
N_QUERY = 5000
TRAIN_CHUNK = 20000
OUT_DIR = os.path.join(_mv.EXT_DIR, "homology")
OUT_CSV = os.path.join(_mv.RESULTS_DIR, "multivar_homology.csv")


def max_jaccard_sparse(Q, T, batch=256):
    """Row-wise max exact Jaccard of Q (queries) against all rows of T. Q, T: binary CSR presence."""
    q_sz = np.asarray(Q.sum(axis=1)).ravel().astype(np.float64)
    t_sz = np.asarray(T.sum(axis=1)).ravel().astype(np.float64)
    TT = T.T.tocsr()
    out = np.zeros(Q.shape[0], dtype=np.float64)
    for s in range(0, Q.shape[0], batch):
        e = min(s + batch, Q.shape[0])
        S = (Q[s:e] @ TT).tocsr()
        if S.nnz == 0:
            continue
        rows = np.repeat(np.arange(e - s), np.diff(S.indptr))
        shared = S.data.astype(np.float64)
        union = q_sz[s + rows] + t_sz[S.indices] - shared
        jac = np.where(union > 0, shared / union, 0.0)
        nonempty = np.diff(S.indptr) > 0
        mx = np.zeros(e - s)
        mx[nonempty] = np.maximum.reduceat(jac, S.indptr[:-1][nonempty])
        out[s:e] = mx
    return out


def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def run_task(task):
    fp = os.path.join(OUT_DIR, f"{task}.json")
    if os.path.exists(fp):
        print(f"[{task}] done already -- skip", flush=True)
        return
    t0 = time.time()
    tr, ytr, te, yte = _mv.load_task_cached(task)
    n_tr, n_te = len(tr), len(te)
    if n_tr * n_te > PAIR_CAP:
        rng = np.random.RandomState(_mv.C.SEED)
        qidx = np.sort(rng.choice(n_te, N_QUERY, replace=False))
        mode = f"subsample{N_QUERY}"
    else:
        qidx = np.arange(n_te)
        mode = "full"
    Q = H.presence_matrix([te[i] for i in qidx])
    mj = np.zeros(len(qidx))
    for c0 in range(0, n_tr, TRAIN_CHUNK):
        T = H.presence_matrix(tr[c0:c0 + TRAIN_CHUNK])
        mj = np.maximum(mj, max_jaccard_sparse(Q, T))
        del T
        print(f"   [{task}] train chunk {c0 // TRAIN_CHUNK + 1}/{-(-n_tr // TRAIN_CHUNK)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    frac = {t: float((mj >= t).mean()) for t in H.THRESHOLDS}
    k7 = int((mj >= 0.7).sum())
    lo, hi = wilson(k7, len(mj))
    row = dict(task=task, n_train=n_tr, n_test=n_te, n_query=int(len(qidx)), query_mode=mode,
               frac_ge_0_5=round(frac[0.5], 5), frac_ge_0_7=round(frac[0.7], 5),
               frac_ge_0_7_lo=round(lo, 5), frac_ge_0_7_hi=round(hi, 5),
               frac_ge_0_9=round(frac[0.9], 5), n_ge_0_7=k7,
               mean_max_jaccard=round(float(mj.mean()), 5),
               median_max_jaccard=round(float(np.median(mj)), 5),
               p95_max_jaccard=round(float(np.percentile(mj, 95)), 5),
               p99_max_jaccard=round(float(np.percentile(mj, 99)), 5),
               max_max_jaccard=round(float(mj.max()), 5), verdict=H.verdict(frac),
               seconds=round(time.time() - t0, 1))
    np.save(os.path.join(OUT_DIR, f"{task}_maxjac.npy"), np.vstack([qidx, mj]))
    with open(fp + ".tmp", "w") as fh:
        json.dump(row, fh)
    os.replace(fp + ".tmp", fp)
    print(f"[{task}] {mode}: >=0.5 {frac[0.5]:.4f}  >=0.7 {frac[0.7]:.4f} [{lo:.4f},{hi:.4f}]  "
          f">=0.9 {frac[0.9]:.4f}  -> {row['verdict']}  ({row['seconds']}s)", flush=True)


def merge():
    import pandas as pd
    rows = []
    for t in [CHECK_TASK] + NEW_TASKS:
        fp = os.path.join(OUT_DIR, f"{t}.json")
        if os.path.exists(fp):
            with open(fp) as fh:
                rows.append(json.load(fh))
    df = pd.DataFrame(rows)
    old = pd.read_csv(os.path.join(_mv.RESULTS_DIR, "homology_leakage.csv")).set_index("task")
    chk = df[df["task"] == CHECK_TASK]
    if len(chk):
        c = chk.iloc[0]
        for col in ("frac_ge_0_5", "frac_ge_0_7", "frac_ge_0_9"):
            if abs(c[col] - old.loc[CHECK_TASK, col]) > 1e-9:
                sys.exit(f"SANITY FAIL: {CHECK_TASK} {col} {c[col]} != homology_leakage.csv "
                         f"{old.loc[CHECK_TASK, col]}")
        print(f"sanity: {CHECK_TASK} reproduces homology_leakage.csv exactly (0.5/0.7/0.9 fractions)")
    df["reproduces_homology_leakage"] = df["task"].eq(CHECK_TASK)
    df.to_csv(OUT_CSV, index=False)
    print(df[["task", "query_mode", "n_query", "frac_ge_0_7", "frac_ge_0_7_lo", "frac_ge_0_7_hi",
              "verdict"]].to_string(index=False))
    print(f"wrote {OUT_CSV} ({len(df)} rows; {len(NEW_TASKS) + 1 - len(df)} task(s) still missing)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=",".join([CHECK_TASK] + NEW_TASKS))
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    peak = _mv.start_memguard(tag="homology")
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.merge:
        merge()
        return
    for t in a.tasks.split(","):
        run_task(t)
        print(f"   peak footprint so far {peak():.0f} MB", flush=True)


if __name__ == "__main__":
    main()
