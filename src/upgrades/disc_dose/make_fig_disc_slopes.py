#!/usr/bin/env python3
"""DEPTH 5 figure: per-motif dose-response slopes. Reads results/upgrades/disc_dose.csv (raw
points) and results/upgrades/disc_dose_slopes.csv (fitted slopes + CIs) ONLY.

Per-motif scatter of benchmark MCC vs dialed motif-only AUROC d, with each motif's own OLS
line, annotated with its slope + 95% bootstrap CI. The pooled slope (~+2.0) is overlaid as a
dashed reference. The three per-motif lines lie on top of one another and on the pooled line --
the slope is motif-invariant. Writes results/upgrades/figures/disc_dose_slopes.{pdf,png}."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PTS_CSV = os.path.join(ROOT, "results", "upgrades", "disc_dose.csv")
SLOPE_CSV = os.path.join(ROOT, "results", "upgrades", "disc_dose_slopes.csv")
OUTDIR = os.path.join(ROOT, "results", "upgrades", "figures")
COL = {"SP1": "#0072B2", "MYC": "#E69F00", "TBP": "#D55E00"}   # matches disc_dose figure
MOTIF_ORDER = ["SP1", "MYC", "TBP"]


def main():
    pts = pd.read_csv(PTS_CSV)
    slopes = pd.read_csv(SLOPE_CSV).set_index("motif")
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    dmin, dmax = pts["motif_auroc_d"].min(), pts["motif_auroc_d"].max()
    xs = np.linspace(dmin, dmax, 100)

    for m in MOTIF_ORDER:
        if m not in set(pts["motif"]):
            continue
        g = pts[pts["motif"] == m]
        c = COL.get(m, "0.4")
        ax.scatter(g["motif_auroc_d"], g["bench_mcc"], s=14, alpha=0.55, color=c,
                   edgecolors="none", zorder=2)
        # per-motif OLS line (re-derive intercept from this motif's points; slope from CSV)
        x = g["motif_auroc_d"].to_numpy(); y = g["bench_mcc"].to_numpy()
        a, b = np.polyfit(x, y, 1)
        ax.plot(xs, a * xs + b, color=c, lw=1.8, zorder=3,
                label=f"{m} (IC={slopes.loc[m, 'motif_ic']:.1f}): "
                      f"slope {slopes.loc[m, 'slope']:+.2f} "
                      f"[{slopes.loc[m, 'slope_lo']:+.2f},{slopes.loc[m, 'slope_hi']:+.2f}]")

    # pooled reference line
    px = pts["motif_auroc_d"].to_numpy(); py = pts["bench_mcc"].to_numpy()
    pa, pb = np.polyfit(px, py, 1)
    ax.plot(xs, pa * xs + pb, color="0.2", lw=1.4, ls="--", zorder=4,
            label=f"POOLED: slope {slopes.loc['POOLED', 'slope']:+.2f} "
                  f"[{slopes.loc['POOLED', 'slope_lo']:+.2f},{slopes.loc['POOLED', 'slope_hi']:+.2f}]")
    ax.axhline(0, color="0.7", lw=0.7, ls=":", zorder=0)

    ax.set_xlabel("motif-only AUROC  d  (dialed discriminability)", fontsize=9)
    ax.set_ylabel("benchmark test MCC (k-mer + LightGBM)", fontsize=9)
    ax.set_title("DEPTH 5: dose-response slope is MOTIF-INVARIANT\n"
                 "three TFs (IC 8.8-14.2) share one slope (~+2.0), CIs overlap", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"disc_dose_slopes.{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUTDIR}/disc_dose_slopes.pdf and .png")


if __name__ == "__main__":
    main()
