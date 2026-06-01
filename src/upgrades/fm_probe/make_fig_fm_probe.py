#!/usr/bin/env python3
"""Figure for Upgrade 1: frozen FM-probe. Reads results/upgrades/fm_probe.csv ONLY.
2x2 (LightGBM head): rows = pooling (mean / [SEP]), cols = metric (MCC / AUROC); one line per
dataset with 95% bootstrap CIs across cleaning arms. cohn/nt_enhancers drop original->cleaned
(the frozen FM loses enhancer score when composition is removed); the clean control (drosophila)
is unchanged under TATA-flag cleaning and -- especially with [SEP] pooling -- stays flat across
arms, while it declines mildly under aggressive composition matching (test-set shrinkage + its
own mild composition component). Writes results/upgrades/figures/fm_probe.{pdf,png}."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CSV = os.path.join(ROOT, "results", "upgrades", "fm_probe.csv")
OUTDIR = os.path.join(ROOT, "results", "upgrades", "figures")
ARM_ORDER = ["original", "tata_flag", "gc_match", "comp_equalized"]
ARM_LAB = {"original": "orig", "tata_flag": "TATA", "gc_match": "GC", "comp_equalized": "comp-eq"}
COL = {"human_enhancers_cohn": "#D55E00", "nt_enhancers": "#0072B2",
       "drosophila_enhancers_stark": "#009E73"}


def _ci(s):
    lo, hi = s.strip("[]").split(","); return float(lo), float(hi)


def _panel(ax, df, metric, ci_col, ylabel):
    for ds, g in df.groupby("dataset"):
        g = g.set_index("arm")
        arms = [a for a in ARM_ORDER if a in g.index]
        xs = list(range(len(arms)))
        ys = [g.loc[a, metric] for a in arms]
        los = [_ci(g.loc[a, ci_col])[0] for a in arms]
        his = [_ci(g.loc[a, ci_col])[1] for a in arms]
        err = [[y - lo for y, lo in zip(ys, los)], [hi - y for y, hi in zip(ys, his)]]
        ax.errorbar(xs, ys, yerr=err, marker="o", ms=4, capsize=2, lw=1.4,
                    color=COL.get(ds, "0.3"), label=ds.replace("_", " "))
    ax.set_xticks(list(range(len(ARM_ORDER)))); ax.set_xticklabels([ARM_LAB[a] for a in ARM_ORDER], fontsize=7)
    ax.set_xlabel("cleaning arm", fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7)


def main():
    df = pd.read_csv(CSV)
    df = df[df["head"] == "lgbm"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), squeeze=False)
    for i, pooling in enumerate(["mean", "sep"]):
        sub = df[df["pooling"] == pooling]
        _panel(axes[i][0], sub, "mcc", "mcc_ci", f"{pooling}-pool MCC")
        _panel(axes[i][1], sub, "auroc", "auroc_ci", f"{pooling}-pool AUROC")
        axes[i][0].axhline(0, color="0.6", lw=0.7, ls="--", zorder=0)
        axes[i][1].axhline(0.5, color="0.6", lw=0.7, ls="--", zorder=0)
    axes[0][0].legend(fontsize=6.5, loc="best", frameon=False)
    fig.suptitle("Frozen genomic-FM (HyenaDNA-tiny) enhancer score vs composition cleaning "
                 "(LightGBM head; rows = pooling)", fontsize=8.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"fm_probe.{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUTDIR}/fm_probe.pdf and .png")


if __name__ == "__main__":
    main()
