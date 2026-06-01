#!/usr/bin/env python3
"""Paired FM-probe figure (Fix A/B). Reads results/upgrades/fm_paired.csv ONLY.
Plots test_effect AUROC = (same frozen-FM head) score loss when composition-biased negatives are
removed from the TEST set, with 95% bootstrap CIs, for the 3 datasets across the 3 cleaning arms,
for each pooling (LightGBM head). The control (drosophila) is flat (CI crosses 0) under [SEP]
pooling -- which is why the locked headline cell is sep/gc_match -- while it moves under mean
pooling; cohn/nt drop significantly under both. Writes results/upgrades/figures/fm_paired.{pdf,png}."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CSV = os.path.join(ROOT, "results", "upgrades", "fm_paired.csv")
OUTDIR = os.path.join(ROOT, "results", "upgrades", "figures")
ARMS = ["tata_flag", "gc_match", "comp_equalized"]
ARM_LAB = {"tata_flag": "TATA", "gc_match": "GC\n(headline)", "comp_equalized": "comp-eq\n(robustness)"}
COL = {"human_enhancers_cohn": "#D55E00", "nt_enhancers": "#0072B2",
       "drosophila_enhancers_stark": "#009E73"}


def _ci(s):
    lo, hi = str(s).strip("[]").split(","); return float(lo), float(hi)


def main():
    df = pd.read_csv(CSV)
    df = df[df["head"] == "lgbm"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    for ax, pooling in zip(axes, ["mean", "sep"]):
        sub = df[df["pooling"] == pooling]
        for off, (ds, g) in zip([-0.12, 0.0, 0.12], sub.groupby("dataset")):
            g = g.set_index("arm")
            xs = [i + off for i, a in enumerate(ARMS) if a in g.index]
            ys = [g.loc[a, "test_effect_auroc"] for a in ARMS if a in g.index]
            los = [g.loc[a, "test_effect_auroc"] - _ci(g.loc[a, "test_effect_auroc_ci"])[0] for a in ARMS if a in g.index]
            his = [_ci(g.loc[a, "test_effect_auroc_ci"])[1] - g.loc[a, "test_effect_auroc"] for a in ARMS if a in g.index]
            ax.errorbar(xs, ys, yerr=[los, his], marker="o", ms=4, lw=0, elinewidth=1.3, capsize=2,
                        color=COL.get(ds, "0.3"), label=ds.replace("_", " "))
        ax.axhline(0, color="0.5", lw=0.9, ls="--", zorder=0)
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([ARM_LAB[a] for a in ARMS], fontsize=7)
        ax.set_title(f"{pooling}-pool", fontsize=9)
        ax.set_xlabel("cleaning arm", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7)
    axes[0].set_ylabel("test_effect AUROC\n(same frozen head, composition removed from test)", fontsize=7.5)
    axes[1].legend(fontsize=6.5, loc="lower left", frameon=False)
    fig.suptitle("Paired FM-probe: a FIXED frozen-FM head loses enhancer AUROC when composition is removed "
                 "from the test\n(control flat under [SEP] pooling = locked headline cell sep/gc_match)",
                 fontsize=8, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"fm_paired.{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUTDIR}/fm_paired.pdf and .png")


if __name__ == "__main__":
    main()
