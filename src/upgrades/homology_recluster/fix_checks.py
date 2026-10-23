#!/usr/bin/env python3
"""Step 9 -- POST-HOC VERIFIER CHECK (2026-09-24 ~11:40). Written after an independent code/claims audit
of the results; DESCRIPTIVE, it replaces no pre-registered result. Numbers are recomputed from the stored
MMseqs2 edge lists and the sequences (no model is refit, MMseqs2 is not re-run).

C2 WHAT KIND OF HOMOLOGY (all 15 tasks). From the stored MMseqs2 edges (fident >= 0.5, >= 80 % coverage of
   both sequences): train-test edges at id80/id50, their median identity and fraction >= 0.99; per test sequence
   with a direct train edge, its best train identity (fraction >= 0.99 / >= 0.95); sequence-length summary
   (min, median, number of test sequences < 50 bp). Distinguishes near-copies of the same locus (identity ~1)
   from family-level homology (identity 0.8-0.9).

Checks C1 (exact-duplicate purge), C4 (equalized-arm draw sensitivity) and C5 (containment coverage) that an
earlier draft of this file contained were REMOVED in fix round 1 because they duplicated the verification-audit
scripts, which were run instead: C1 -> run_audit.py Q3 (homology_recluster_audit_task.csv), C4 ->
run_e1_reconcile.py (homology_recluster_e1_{reconcile,draws,draws_summary}.csv), C5 ->
run_covmode_sensitivity.py (homology_recluster_covmode.csv; all 5 variable-length tasks, not only the two small ones).

Output: results/upgrades/homology_recluster_fix_edges.csv. Checkpoints: external/homology_recluster/ckpt/fix/.
Usage: python fix_checks.py
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _hc  # noqa: E402  (caps LightGBM threads for this process)
from run_cluster import SIZE_ORDER  # noqa: E402

CK = os.path.join(_hc.CKPT_DIR, "fix")
TH = {"id80": 800, "id50": 500}


def out_csv(name):
    return os.path.join(_hc.RES, f"{_hc.PREFIX}_fix_{name}.csv")


def cached(name, fn):
    os.makedirs(CK, exist_ok=True)
    p = os.path.join(CK, name + ".csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    t0 = time.time()
    df = fn()
    df.to_csv(p + ".tmp", index=False)
    os.replace(p + ".tmp", p)
    print(f"  [{name}] {time.time() - t0:.0f}s", flush=True)
    return df


# ------------------------------------------------------------------ C2
def c2_task(task):
    tr, ytr, te, yte = _hc.load_task(task)
    n_tr, n = len(tr), len(tr) + len(te)
    z = np.load(os.path.join(_hc.EXT, "edges", f"{task}.npz"))
    assert int(z["n"]) == n and int(z["n_train"]) == n_tr
    a, b, f = z["a"].astype(np.int64), z["b"].astype(np.int64), z["fid"].astype(np.int64)
    lens_all = np.array([len(s) for s in tr + te])
    rows = []
    for tag, thr in TH.items():
        m = f >= thr
        tt = m & ((a < n_tr) != (b < n_tr))
        te_node = np.where(a[tt] >= n_tr, a[tt], b[tt])
        best = np.zeros(n, np.int64)
        np.maximum.at(best, te_node, f[tt])
        bt = best[n_tr:]
        hit = bt > 0
        rows.append(dict(task=task, threshold=tag, n_edges=int(m.sum()), n_train_test_edges=int(tt.sum()),
                         tt_edge_fid_median=float(np.median(f[tt]) / 1000) if tt.any() else np.nan,
                         tt_edge_frac_ge099=round(float(np.mean(f[tt] >= 990)), 4) if tt.any() else np.nan,
                         frac_test_direct_train_edge=round(float(hit.mean()), 5),
                         direct_best_fid_median=float(np.median(bt[hit]) / 1000) if hit.any() else np.nan,
                         direct_best_frac_ge099=round(float(np.mean(bt[hit] >= 990)), 4) if hit.any() else np.nan,
                         direct_best_frac_ge095=round(float(np.mean(bt[hit] >= 950)), 4) if hit.any() else np.nan,
                         len_min=int(lens_all.min()), len_median=float(np.median(lens_all)), len_max=int(lens_all.max()),
                         n_test_lt50bp=int((lens_all[n_tr:] < 50).sum())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checks", default="C2", help="only C2 exists (C1/C4/C5 were removed as duplicates)")
    a = ap.parse_args()
    todo = [c for c in a.checks.split(",") if c]
    if "C2" in todo:
        pd.concat([cached(f"C2_{t}", lambda t=t: c2_task(t)) for t in SIZE_ORDER], ignore_index=True).to_csv(
            out_csv("edges"), index=False)
    unknown = [c for c in todo if c != "C2"]
    if unknown:
        sys.exit(f"removed/unknown checks {unknown}: see the docstring (superseded by the audit scripts)")
    print("fix_checks done:", todo)


if __name__ == "__main__":
    main()
