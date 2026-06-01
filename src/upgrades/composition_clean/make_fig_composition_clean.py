#!/usr/bin/env python3
"""Figure for Upgrade 2: composition-equalized cleaning. Reads results/upgrades/composition_clean.csv
ONLY (no recomputation). Small multiples (2 rows x 3 datasets):
  row 1 -- TATA & GC motif/composition AUROC across cleaning arms (collapse toward 0.5);
  row 2 -- k-mer benchmark MCC vs the composition-only baseline MCC (the gap = residual signal).
Writes results/upgrades/figures/composition_clean.{pdf,png}."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CSV = os.path.join(ROOT, "results", "upgrades", "composition_clean.csv")
OUTDIR = os.path.join(ROOT, "results", "upgrades", "figures")
ARM_ORDER = ["original", "tata_flag", "gc_match", "comp_equalized"]
ARM_LAB = {"original": "orig", "tata_flag": "TATA", "gc_match": "GC", "comp_equalized": "comp-eq"}
OKABE = {"tata": "#D55E00", "gc": "#0072B2", "kmer": "#009E73", "comp": "#CC79A7"}


def main():
    df = pd.read_csv(CSV)
    datasets = [d for d in ["human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark"]
                if d in set(df["dataset"])]
    fig, axes = plt.subplots(2, len(datasets), figsize=(2.5 * len(datasets) + 0.6, 5.0), squeeze=False)
    for j, ds in enumerate(datasets):
        g = df[df["dataset"] == ds].set_index("arm")
        arms = [a for a in ARM_ORDER if a in g.index]
        xs = range(len(arms))
        # row 1: AUROC collapse
        ax = axes[0][j]
        ax.axhline(0.5, color="0.6", lw=0.8, ls="--", zorder=0)
        ax.plot(xs, [g.loc[a, "tata_auroc"] for a in arms], "-o", color=OKABE["tata"], label="TATA AUROC", ms=4)
        ax.plot(xs, [g.loc[a, "gc_auroc"] for a in arms], "-s", color=OKABE["gc"], label="GC AUROC", ms=4)
        ax.set_xticks(list(xs)); ax.set_xticklabels([ARM_LAB[a] for a in arms], fontsize=7)
        ax.set_ylim(0.30, 0.90); ax.set_title(ds.replace("_", " "), fontsize=8)
        if j == 0:
            ax.set_ylabel("composition AUROC", fontsize=8); ax.legend(fontsize=6.5, loc="upper right", frameon=False)
        # row 2: kmer MCC vs composition-only baseline
        ax = axes[1][j]
        ax.plot(xs, [g.loc[a, "kmer_mcc"] for a in arms], "-o", color=OKABE["kmer"], label="k-mer MCC", ms=4)
        ax.plot(xs, [g.loc[a, "comp_only_mcc"] for a in arms], "-^", color=OKABE["comp"], label="composition-only MCC", ms=4)
        ax.set_xticks(list(xs)); ax.set_xticklabels([ARM_LAB[a] for a in arms], fontsize=7)
        ax.set_ylim(-0.05, max(0.55, df["kmer_mcc"].max() + 0.05)); ax.set_xlabel("cleaning arm", fontsize=8)
        if j == 0:
            ax.set_ylabel("test MCC", fontsize=8); ax.legend(fontsize=6.5, loc="upper right", frameon=False)
        for a in axes[:, j]:
            a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=7)
    fig.suptitle("Composition-equalized cleaning: composition AUROC collapses; residual k-mer signal vs composition baseline",
                 fontsize=8.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"composition_clean.{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUTDIR}/composition_clean.pdf and .png")


if __name__ == "__main__":
    main()
