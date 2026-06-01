#!/usr/bin/env python3
"""Figure for Upgrade 3: discriminability dose-response. Reads results/upgrades/disc_dose.csv ONLY.
Scatter of benchmark MCC vs dialed motif-only AUROC d, colored by motif (information content), with a
pooled linear fit + 95% bootstrap band and the Spearman annotation. Writes
results/upgrades/figures/disc_dose.{pdf,png}."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CSV = os.path.join(ROOT, "results", "upgrades", "disc_dose.csv")
OUTDIR = os.path.join(ROOT, "results", "upgrades", "figures")
COL = {"SP1": "#0072B2", "MYC": "#E69F00", "TBP": "#D55E00"}


def main():
    df = pd.read_csv(CSV)
    d = df["motif_auroc_d"].to_numpy(); mcc = df["bench_mcc"].to_numpy()
    rho, p = spearmanr(d, mcc)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for mname, g in df.groupby("motif"):
        ax.scatter(g["motif_auroc_d"], g["bench_mcc"], s=16, alpha=0.7,
                   color=COL.get(mname, "0.4"), label=f"{mname} (IC={g['motif_ic'].iloc[0]:.1f})", edgecolors="none")
    # pooled linear fit + bootstrap band
    xs = np.linspace(d.min(), d.max(), 100)
    rng = np.random.RandomState(42); fits = []
    for _ in range(1000):
        idx = rng.randint(0, len(d), len(d))
        a, b = np.polyfit(d[idx], mcc[idx], 1); fits.append(a * xs + b)
    fits = np.array(fits)
    a, b = np.polyfit(d, mcc, 1)
    ax.plot(xs, a * xs + b, color="0.2", lw=1.6, label=f"fit: slope={a:+.2f}")
    ax.fill_between(xs, np.percentile(fits, 2.5, 0), np.percentile(fits, 97.5, 0), color="0.6", alpha=0.25, lw=0)
    ax.axhline(0, color="0.7", lw=0.7, ls="--", zorder=0)
    ax.set_xlabel("motif-only AUROC  d  (dialed discriminability)", fontsize=9)
    ax.set_ylabel("benchmark test MCC (k-mer + LightGBM)", fontsize=9)
    ax.set_title(f"Discriminability dose-response (synthetic)\nSpearman(d, MCC) = {rho:+.3f}, "
                 f"n={len(df)}", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"disc_dose.{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUTDIR}/disc_dose.pdf and .png")


if __name__ == "__main__":
    main()
