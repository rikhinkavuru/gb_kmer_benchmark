#!/usr/bin/env python3
"""Figure for Upgrade 5: shuffled-negative control. Reads results/upgrades/shuffled_neg.csv ONLY.
Grouped bars: TATA motif-only AUROC and GC AUROC, original-negatives vs dinucleotide-shuffled
negatives, per dataset. For the contaminated datasets the TATA artifact (AUROC far from 0.5 against
real negatives) collapses to ~0.5 against composition-matched shuffled negatives; GC -> 0.5 by
construction. Writes results/upgrades/figures/shuffled_neg.{pdf,png}."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CSV = os.path.join(ROOT, "results", "upgrades", "shuffled_neg.csv")
OUTDIR = os.path.join(ROOT, "results", "upgrades", "figures")


def _ci(s):
    lo, hi = str(s).strip("[]").split(","); return float(lo), float(hi)


def _panel(ax, df, metric, ci_col, title):
    datasets = list(dict.fromkeys(df["dataset"]))
    x = np.arange(len(datasets)); w = 0.36
    for off, arm, col in [(-w / 2, "original_neg", "#999999"), (w / 2, "shuffled_neg", "#0072B2")]:
        ys, los, his = [], [], []
        for ds in datasets:
            r = df[(df.dataset == ds) & (df.arm == arm)].iloc[0]
            ys.append(r[metric]); lo, hi = _ci(r[ci_col]); los.append(r[metric] - lo); his.append(hi - r[metric])
        ax.bar(x + off, ys, w, yerr=[los, his], label=arm, color=col, capsize=2, error_kw=dict(elinewidth=0.8))
    ax.axhline(0.5, color="0.4", lw=0.9, ls="--", zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([d.replace("_", "\n") for d in datasets], fontsize=6.5)
    ax.set_ylim(0.3, 0.85); ax.set_ylabel(title, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7)


def main():
    df = pd.read_csv(CSV)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.4))
    _panel(a1, df, "tata_auroc", "tata_ci", "TATA motif-only AUROC")
    _panel(a2, df, "gc_auroc", "gc_ci", "GC AUROC")
    a1.legend(fontsize=7, frameon=False, loc="upper right")
    a1.set_title("TATA artifact vs shuffled negatives", fontsize=9)
    a2.set_title("GC (→0.5 by construction)", fontsize=9)
    fig.suptitle("Dinucleotide-preserving shuffled negatives: the composition artifact is a property of the "
                 "negative-sampling protocol", fontsize=8, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"shuffled_neg.{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUTDIR}/shuffled_neg.pdf and .png")


if __name__ == "__main__":
    main()
