#!/usr/bin/env python3
"""Upgrade 7 -- predictive held-out test for Route 1 (discriminability -> learnability).

Makes Route 1 PREDICTIVE rather than merely correlational: fit the relationship
benchmark MCC ~ max positive-direction motif-only AUROC (discriminability) in a LEAVE-ONE-TASK-OUT
manner across the 11 TF-motif tasks (both suites), predict each held-out task's MCC from a model
trained on the other 10, and report R^2 / MAE on the held-out predictions plus a predicted-vs-actual
table for the figure.

Reads results/cross_suite_summary.csv (tf_motif_task == True) -- no heavy recompute. The predictor
is an ordinary least-squares line (1 feature); LOTO with n=11 and a single predictor is the honest,
low-variance choice. Nulls reported straight. Seed 42 (only used for the bootstrap CI on R^2/MAE).
Writes results/upgrades/route1_predict.csv + _interpretation.txt.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _common as C


def loto_predictions(x, y):
    """Leave-one-task-out OLS predictions of y from x (1-D). Returns array of held-out preds."""
    n = len(x)
    pred = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        reg = LinearRegression().fit(x[m].reshape(-1, 1), y[m])
        pred[i] = reg.predict(x[i].reshape(1, -1))[0]
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default=os.path.join(C.ROOT, "results", "cross_suite_summary.csv"))
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "route1_predict.csv"))
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(args.summary)
    tf = df[df["tf_motif_task"] == True].dropna(subset=["max_pos_motif_auroc", "bench_mcc"]).reset_index(drop=True)
    x = tf["max_pos_motif_auroc"].to_numpy(float)
    y = tf["bench_mcc"].to_numpy(float)
    n = len(x)
    pred = loto_predictions(x, y)
    resid = y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    mae = float(np.mean(np.abs(resid)))
    rho, prho = spearmanr(x, y)
    full = LinearRegression().fit(x.reshape(-1, 1), y)
    slope, intercept = float(full.coef_[0]), float(full.intercept_)

    # bootstrap CI on LOTO R^2 and MAE (resample tasks)
    rng = np.random.RandomState(args.seed)
    r2s, maes = [], []
    for _ in range(1000):
        idx = rng.randint(0, n, n)
        rr = (y[idx] - pred[idx])
        sr = np.sum(rr ** 2); st = np.sum((y[idx] - y[idx].mean()) ** 2)
        if st > 0:
            r2s.append(1 - sr / st); maes.append(np.mean(np.abs(rr)))
    r2_lo, r2_hi = np.percentile(r2s, [2.5, 97.5])
    mae_lo, mae_hi = np.percentile(maes, [2.5, 97.5])

    rows = [dict(task=tf["task"].iloc[i], suite=tf["suite"].iloc[i],
                 max_pos_motif_auroc=round(x[i], 4), actual_mcc=round(y[i], 4),
                 loto_pred_mcc=round(pred[i], 4), abs_error=round(abs(resid[i]), 4)) for i in range(n)]
    out = pd.DataFrame(rows).sort_values("max_pos_motif_auroc")
    out.to_csv(args.out, index=False)

    L = ["=" * 96,
         "ROUTE 1 PREDICTIVE TEST -- leave-one-task-out: predict benchmark MCC from motif discriminability",
         "=" * 96,
         f"n = {n} TF-motif tasks (both suites). Predictor: OLS line MCC ~ max positive motif-only AUROC.",
         f"Full-data fit: MCC = {slope:+.3f} * d + {intercept:+.3f}   Spearman(d,MCC)={rho:+.3f} (p={prho:.2g})",
         f"LEAVE-ONE-TASK-OUT held-out:  R^2 = {r2:+.3f}  [{r2_lo:+.3f},{r2_hi:+.3f}]   "
         f"MAE = {mae:.3f}  [{mae_lo:.3f},{mae_hi:.3f}]",
         "",
         f"{'task':<28}{'suite':<14}{'d':>7}{'actual MCC':>12}{'LOTO pred':>11}{'|err|':>8}",
         "-" * 80]
    for _, r in out.iterrows():
        L.append(f"{r['task']:<28}{r['suite']:<14}{r['max_pos_motif_auroc']:>7.3f}{r['actual_mcc']:>12.3f}"
                 f"{r['loto_pred_mcc']:>11.3f}{r['abs_error']:>8.3f}")
    L += ["",
          "Reading: a positive held-out R^2 means motif discriminability PREDICTS learnability on tasks",
          "the line never saw -- Route 1 is predictive, not merely correlational. A modest R^2 with low",
          "MAE is reported straight; outliers (e.g. multiclass enhancer-types, or a highly-solved task",
          "with moderate single-motif AUROC) are visible in the per-task errors. Seed %d." % args.seed]
    rep = "\n".join(L)
    print(rep)
    with open(args.out.replace(".csv", "_interpretation.txt"), "w") as fh:
        fh.write(rep)
    print(f"\nWrote {args.out} ({n} rows) + interpretation.")


if __name__ == "__main__":
    main()
