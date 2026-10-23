#!/usr/bin/env python3
"""Task-landscape figure (paper Figure 2): the two secondary routes on one row.

(a) Route 1, discriminability dose-response -- semi-synthetic, n=135 settings
    (3 motifs x 5 seeds x 9 doses), with the three per-motif slope fits.
    Source: results/upgrades/disc_dose.csv, results/upgrades/disc_dose_slopes.csv
(b) Route 2, positional rescue -- delta MCC against bin resolution B for the one
    responding task (splice) and the fixed-position TATA promoter, against the
    envelope of the thirteen non-responding tasks at B=8.
    Source: results/upgrades/positional_Bsweep.csv, results/upgrades/positional_sweep.csv

This replaces the old full-width single-panel dose-response figure and moves the
bin-resolution sweep out of the Results prose. Drawn at 5.5 x 2.2 in, included at
\\textwidth, so all tick and axis labels are >= 8 pt on the printed page.

Writes results/upgrades/figures/task_landscape.{pdf,png}.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
U = os.path.join(ROOT, "results", "upgrades")
OUTDIR = os.path.join(U, "figures")

MOTIF_C = {"SP1": "#0072B2", "MYC": "#E69F00", "TBP": "#D55E00"}
SPLICE_C = "#009E73"
TATA_C = "#0072B2"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8.5,
    "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.6, "axes.linewidth": 0.7, "xtick.major.width": 0.7,
    "ytick.major.width": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def panel_dose(ax):
    df = pd.read_csv(os.path.join(U, "disc_dose.csv"))
    sl = pd.read_csv(os.path.join(U, "disc_dose_slopes.csv")).set_index("motif")
    d = df["motif_auroc_d"].to_numpy()
    mcc = df["bench_mcc"].to_numpy()
    rho, _ = spearmanr(d, mcc)
    for m, g in df.groupby("motif"):
        ax.scatter(g["motif_auroc_d"], g["bench_mcc"], s=11, alpha=0.75, edgecolors="none",
                   color=MOTIF_C.get(m, "0.4"),
                   label=f"{m}  slope {sl.loc[m, 'slope']:+.2f}")
    xs = np.linspace(d.min(), d.max(), 100)
    rng = np.random.RandomState(42)
    fits = np.array([np.polyval(np.polyfit(d[i], mcc[i], 1), xs)
                     for i in (rng.randint(0, len(d), len(d)) for _ in range(1000))])
    a, b = np.polyfit(d, mcc, 1)
    ax.fill_between(xs, np.percentile(fits, 2.5, 0), np.percentile(fits, 97.5, 0),
                    color="0.55", alpha=0.25, lw=0)
    ax.plot(xs, a * xs + b, color="0.2", lw=1.4, zorder=3)
    ax.axhline(0, color="0.75", lw=0.6, ls="--", zorder=0)
    ax.set_xlabel("dialled motif-only AUROC $d$")
    ax.set_ylabel("benchmark test MCC")
    ax.set_title(f"(a) Route 1 dose-response, $n$={len(df)}\n"
                 f"Spearman $+{rho:.3f}$; pooled slope {a:+.2f}", pad=4)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.4, borderpad=0.2,
              labelspacing=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def panel_bins(ax):
    bs = pd.read_csv(os.path.join(U, "positional_Bsweep.csv"))
    ps = pd.read_csv(os.path.join(U, "positional_sweep.csv"))
    others = ps[~ps.task.isin(["nt_splice_sites_all", "nt_promoter_tata"])]["delta_mcc"]
    lo, hi = float(others.min()), float(others.max())
    ax.axhspan(lo, hi, color="0.75", alpha=0.45, lw=0, zorder=0)
    ax.text(16, hi + 0.02, f"13 other tasks: |$\\Delta$|$\\leq${max(abs(lo), hi):.3f}",
            fontsize=7.6, ha="right", va="bottom", color="0.35")
    for task, c, lab, mk in [("nt_splice_sites_all", SPLICE_C, "splice sites", "o"),
                             ("nt_promoter_tata", TATA_C, "TATA promoter", "s")]:
        g = bs[bs.task == task].sort_values("B")
        err = np.array([[d - float(ci.strip("[]").split(",")[0]),
                         float(ci.strip("[]").split(",")[1]) - d]
                        for d, ci in zip(g.delta_vs_B1, g.delta_ci)]).T
        ax.errorbar(g["B"], g["delta_vs_B1"], yerr=np.abs(err), color=c, marker=mk,
                    ms=4, lw=1.4, capsize=2, elinewidth=0.8, label=lab, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.set_xlabel("position bins $B$")
    ax.set_ylabel("$\\Delta$ MCC vs $B$=1")
    ax.set_title("(b) Route 2 rescue, 15 tasks:\nselective, and graded in $B$", pad=4)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.4, borderpad=0.2,
              labelspacing=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.18))
    panel_dose(axes[0])
    panel_bins(axes[1])
    fig.tight_layout(pad=0.35, w_pad=1.4)
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"task_landscape.{ext}"), dpi=400)
    print(f"wrote {OUTDIR}/task_landscape.pdf and .png")


if __name__ == "__main__":
    main()
