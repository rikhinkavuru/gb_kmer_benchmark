#!/usr/bin/env python3
"""Figure for the Route 3 specificity module. Reads ONLY
  results/upgrades/route3_specificity.csv
  results/upgrades/route3_threshold_invariance.csv
(no recomputation) and writes results/upgrades/figures/route3_specificity.{pdf,png}.

LEFT panel  -- control-motif AUROC vs real TATA, per dataset. For each neg-skew
  dataset: the real-TATA point (with bootstrap-CI bar) far from 0.5, and the two
  IC-matched control panels drawn as mean +/- sd bands (panel spread) with their
  bootstrap-CI on the mean. A 0.5 reference line marks "no class skew". The random
  IC panel sits on 0.5; the column-scrambled panel sits near real TATA (it keeps
  TBP's AT-rich per-column composition), so the contrast that isolates the ordered
  motif is real-TATA-or-scrambled vs random.
RIGHT panel -- threshold sweep: real-TATA AUROC vs threshold (0.8-1.5 bits/pos),
  original split (neg-skewed across the whole range) vs comp_equalized split (flat
  on ~0.5 across the whole range), with bootstrap-CI bands.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SPEC_CSV = os.path.join(ROOT, "results", "upgrades", "route3_specificity.csv")
THR_CSV = os.path.join(ROOT, "results", "upgrades", "route3_threshold_invariance.csv")
OUTDIR = os.path.join(ROOT, "results", "upgrades", "figures")

NEG_SKEW = ["human_enhancers_cohn", "nt_enhancers"]
OKABE = {"real": "#D55E00", "scram": "#E69F00", "rand": "#0072B2",
         "orig": "#D55E00", "ce": "#009E73"}
DS_LAB = {"human_enhancers_cohn": "cohn", "nt_enhancers": "nt_enh"}


def _f(x):
    return float(x) if str(x) != "" else np.nan


def left_panel(ax, spec):
    ax.axhline(0.5, color="0.55", lw=0.9, ls="--", zorder=0, label="no skew (0.5)")
    datasets = [d for d in NEG_SKEW if d in set(spec["dataset"])]
    xpos = {ds: i for i, ds in enumerate(datasets)}
    dx = 0.22
    for ds in datasets:
        g = spec[spec["dataset"] == ds].set_index("motif_class")
        x = xpos[ds]
        # real TATA: point + bootstrap CI bar
        rt = g.loc["real_TATA"]
        ax.errorbar(x - dx, _f(rt["motif_only_auroc"]),
                    yerr=[[_f(rt["motif_only_auroc"]) - _f(rt["auroc_ci_lo"])],
                          [_f(rt["auroc_ci_hi"]) - _f(rt["motif_only_auroc"])]],
                    fmt="D", color=OKABE["real"], ms=7, capsize=3, lw=1.4, zorder=4,
                    label="real TATA" if ds == datasets[0] else None)
        # scrambled-TBP panel: mean +/- sd band (panel spread)
        sc = g.loc["scrambled_TBP_panel"]
        ax.errorbar(x, _f(sc["motif_only_auroc"]), yerr=_f(sc["auroc_sd"]),
                    fmt="o", color=OKABE["scram"], ms=6, capsize=3, lw=1.4, zorder=3,
                    label="scrambled-TBP panel (mean +/- sd)" if ds == datasets[0] else None)
        # random IC-matched panel: mean +/- sd band
        rd = g.loc["random_IC_panel"]
        ax.errorbar(x + dx, _f(rd["motif_only_auroc"]), yerr=_f(rd["auroc_sd"]),
                    fmt="s", color=OKABE["rand"], ms=6, capsize=3, lw=1.4, zorder=3,
                    label="random IC panel (mean +/- sd)" if ds == datasets[0] else None)
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels([DS_LAB.get(d, d) for d in datasets], fontsize=8)
    ax.set_ylim(0.30, 0.62)
    ax.set_ylabel("TATA motif-only AUROC (pos > neg)", fontsize=8)
    ax.set_title("Control panels vs real TATA\n(IC-matched controls; random panel on 0.5)", fontsize=8.5)
    ax.legend(fontsize=6.3, loc="upper left", frameon=False, ncol=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)


def right_panel(ax, thr):
    ax.axhline(0.5, color="0.55", lw=0.9, ls="--", zorder=0)
    style = {("human_enhancers_cohn", "original"): ("-", "o"),
             ("human_enhancers_cohn", "comp_equalized"): ("-", "s"),
             ("nt_enhancers", "original"): ("--", "o"),
             ("nt_enhancers", "comp_equalized"): ("--", "s")}
    for ds in [d for d in NEG_SKEW if d in set(thr["dataset"])]:
        for split in ["original", "comp_equalized"]:
            sub = thr[(thr["dataset"] == ds) & (thr["split"] == split)].sort_values("threshold_bits")
            if not len(sub):
                continue
            ls, mk = style[(ds, split)]
            color = OKABE["orig"] if split == "original" else OKABE["ce"]
            xs = sub["threshold_bits"].to_numpy()
            ys = sub["tata_auroc"].to_numpy()
            ax.fill_between(xs, sub["tata_ci_lo"].to_numpy(), sub["tata_ci_hi"].to_numpy(),
                            color=color, alpha=0.12, zorder=1)
            ax.plot(xs, ys, ls=ls, marker=mk, color=color, ms=4, lw=1.4, zorder=2,
                    label=f"{DS_LAB.get(ds, ds)} | {split}")
    ax.set_xlabel("TATA threshold (bits/pos)", fontsize=8)
    ax.set_ylabel("TATA motif-only AUROC (pos > neg)", fontsize=8)
    ax.set_ylim(0.25, 0.62)
    ax.set_title("Threshold invariance\noriginal stays neg-skewed; comp-equalized flat on 0.5", fontsize=8.5)
    ax.legend(fontsize=6.3, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)


def main():
    spec = pd.read_csv(SPEC_CSV)
    thr = pd.read_csv(THR_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    left_panel(axes[0], spec)
    right_panel(axes[1], thr)
    fig.suptitle("Route 3 specificity: the TATA/composition class-skew is specific to the real motif and threshold-robust",
                 fontsize=9.0, y=1.02)
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"route3_specificity.{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUTDIR}/route3_specificity.pdf and .png")


if __name__ == "__main__":
    main()
