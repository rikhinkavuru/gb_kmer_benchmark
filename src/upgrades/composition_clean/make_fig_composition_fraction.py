#!/usr/bin/env python3
"""Figure for Depth 2: composition-fraction decomposition. Reads
results/upgrades/composition_fraction.csv ONLY (no recomputation).

Grouped bar chart: composition_fraction per dataset, grouped by classifier (kmer vs fm),
with 95% bootstrap CI error bars and reference lines at 0 and 1. A fraction of 1 means the
above-chance enhancer AUROC is ENTIRELY composition; 0 means none of it is. Writes
results/upgrades/figures/composition_fraction.{pdf,png}."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CSV = os.path.join(ROOT, "results", "upgrades", "composition_fraction.csv")
OUTDIR = os.path.join(ROOT, "results", "upgrades", "figures")

DS_ORDER = ["human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark"]
CLF_ORDER = ["kmer", "fm"]
# Okabe-Ito (matches the rest of the figure suite)
CLF_COLOR = {"kmer": "#009E73", "fm": "#0072B2"}
CLF_LABEL = {"kmer": "k-mer LightGBM", "fm": "FM head (HyenaDNA)"}


def main():
    df = pd.read_csv(CSV)
    datasets = [d for d in DS_ORDER if d in set(df["dataset"])]
    clfs = [c for c in CLF_ORDER if c in set(df["classifier"])]

    x = np.arange(len(datasets), dtype=float)
    width = 0.8 / max(len(clfs), 1)

    fig, ax = plt.subplots(figsize=(1.9 * len(datasets) + 1.4, 4.0))
    ax.axhline(1.0, color="0.55", lw=0.9, ls="--", zorder=0)
    ax.axhline(0.0, color="0.55", lw=0.9, ls="-", zorder=0)

    for i, clf in enumerate(clfs):
        g = df[df["classifier"] == clf].set_index("dataset")
        vals = np.array([g.loc[d, "comp_fraction"] for d in datasets], dtype=float)
        lo = np.array([g.loc[d, "comp_fraction_lo"] for d in datasets], dtype=float)
        hi = np.array([g.loc[d, "comp_fraction_hi"] for d in datasets], dtype=float)
        yerr = np.vstack([vals - lo, hi - vals])
        offs = (i - (len(clfs) - 1) / 2.0) * width
        ax.bar(x + offs, vals, width=width * 0.92, color=CLF_COLOR.get(clf, None),
               label=CLF_LABEL.get(clf, clf), zorder=2, edgecolor="white", linewidth=0.5)
        ax.errorbar(x + offs, vals, yerr=yerr, fmt="none", ecolor="0.2",
                    elinewidth=1.0, capsize=2.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n", 1).replace("_", " ") for d in datasets], fontsize=8)
    ax.set_ylabel("composition fraction\n(AUROC$_{comp}$ - 0.5) / (AUROC$_{full}$ - 0.5)", fontsize=8.5)
    ax.set_title("Share of above-chance enhancer AUROC explained by composition (GC + 16 dinuc)",
                 fontsize=8.5)
    ax.set_ylim(0.0, max(1.18, float(df["comp_fraction_hi"].max()) + 0.06))
    ax.legend(fontsize=7.5, loc="lower left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    # annotate the reference lines
    ax.text(ax.get_xlim()[1], 1.0, " all composition", va="center", ha="left",
            fontsize=6.5, color="0.4", clip_on=False)
    fig.tight_layout()

    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"composition_fraction.{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUTDIR}/composition_fraction.pdf and .png")


if __name__ == "__main__":
    main()
