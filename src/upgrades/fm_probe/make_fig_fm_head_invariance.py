#!/usr/bin/env python3
"""DEPTH 4 figure: FM-head invariance. Reads results/upgrades/fm_head_invariance.csv ONLY.

For the LOCKED headline arm (pooling=sep, arm=gc_match), plot test_effect_auroc with 95%
bootstrap CI error bars, grouped by dataset (cohn, nt = contaminated; drosophila = clean
control), colored by probe head (lgbm vs lr). Horizontal line at 0. The two heads land on top
of each other on cohn/nt (both clearly below 0) and both hug 0 on the control -- i.e. the
composition-removal effect is a property of the frozen HyenaDNA embedding, not the read-out
head. Writes results/upgrades/figures/fm_head_invariance.{pdf,png}."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CSV = os.path.join(ROOT, "results", "upgrades", "fm_head_invariance.csv")
OUTDIR = os.path.join(ROOT, "results", "upgrades", "figures")
LOCKED_POOLING = "sep"
LOCKED_ARM = "gc_match"
DATASETS = ["human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark"]
DS_LAB = {"human_enhancers_cohn": "cohn\n(contaminated)",
          "nt_enhancers": "nt_enhancers\n(contaminated)",
          "drosophila_enhancers_stark": "drosophila\n(clean control)"}
HEAD_COL = {"lgbm": "#0072B2", "lr": "#E69F00"}   # Okabe-Ito blue / orange
HEAD_LAB = {"lgbm": "LightGBM head", "lr": "logistic-regression head"}


def _ci(s):
    lo, hi = str(s).strip().strip("[]").split(","); return float(lo), float(hi)


def main():
    df = pd.read_csv(CSV)
    df = df[(df["pooling"] == LOCKED_POOLING) & (df["arm"] == LOCKED_ARM)]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    offsets = {"lgbm": -0.12, "lr": +0.12}
    for head in ["lgbm", "lr"]:
        g = df[df["head"] == head].set_index("dataset")
        xs, ys, los, his = [], [], [], []
        for i, ds in enumerate(DATASETS):
            if ds not in g.index:
                continue
            eff = float(g.loc[ds, "test_effect_auroc"])
            lo, hi = _ci(g.loc[ds, "test_effect_auroc_ci"])
            xs.append(i + offsets[head]); ys.append(eff)
            los.append(eff - lo); his.append(hi - eff)
        ax.errorbar(xs, ys, yerr=[los, his], marker="o", ms=6, lw=0, elinewidth=1.6,
                    capsize=3, color=HEAD_COL[head], label=HEAD_LAB[head])
    ax.axhline(0, color="0.5", lw=0.9, ls="--", zorder=0)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DS_LAB[d] for d in DATASETS], fontsize=8)
    ax.set_ylabel("test_effect AUROC\n(same frozen HyenaDNA head, composition removed from test)",
                  fontsize=8)
    ax.set_title("DEPTH 4: composition-removal effect is HEAD-INVARIANT\n"
                 "locked cell sep/gc_match -- both probe heads agree on cohn/nt; control flat",
                 fontsize=8.5)
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"fm_head_invariance.{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUTDIR}/fm_head_invariance.pdf and .png")


if __name__ == "__main__":
    main()
