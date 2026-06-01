#!/usr/bin/env python3
"""Figure for Upgrade 7: predicted-vs-actual MCC (leave-one-task-out). Reads
results/upgrades/route1_predict.csv ONLY. Writes results/upgrades/figures/route1_predict.{pdf,png}."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CSV = os.path.join(ROOT, "results", "upgrades", "route1_predict.csv")
OUTDIR = os.path.join(ROOT, "results", "upgrades", "figures")


def main():
    df = pd.read_csv(CSV)
    a = df["actual_mcc"].to_numpy(); p = df["loto_pred_mcc"].to_numpy()
    ss_res = np.sum((a - p) ** 2); ss_tot = np.sum((a - a.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot; mae = np.mean(np.abs(a - p))
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    lim = [min(a.min(), p.min()) - 0.05, max(a.max(), p.max()) + 0.05]
    ax.plot(lim, lim, color="0.6", lw=1, ls="--", zorder=0, label="y = x")
    col = {"NucTransformer": "#0072B2", "GenomicBench": "#D55E00"}
    for suite, g in df.groupby("suite"):
        ax.scatter(g["actual_mcc"], g["loto_pred_mcc"], s=30, color=col.get(suite, "0.4"), label=suite, zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r["task"].replace("human_", "").replace("_enhancers", "_enh").replace("nt_", "nt_")[:16],
                    (r["actual_mcc"], r["loto_pred_mcc"]), fontsize=5.2, xytext=(3, 2),
                    textcoords="offset points", color="0.3")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("actual benchmark MCC", fontsize=9)
    ax.set_ylabel("LOTO predicted MCC (from motif discriminability)", fontsize=9)
    ax.set_title(f"Route 1 is predictive: leave-one-task-out\nR² = {r2:+.3f},  MAE = {mae:.3f}  (n={len(df)})", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"route1_predict.{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUTDIR}/route1_predict.pdf and .png")


if __name__ == "__main__":
    main()
