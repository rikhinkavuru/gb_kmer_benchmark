#!/usr/bin/env python3
"""Route-3 evidence figure (paper Figure 3): magnitude, specificity and threshold-robustness.

(a) MAGNITUDE  -- TATA motif-only AUROC against the real negatives versus dinucleotide-preserving
    shuffles of the positives (no composition gap by construction). For the contaminated sets the
    skew vanishes; for the clean control, whose TATA is a genuine positive-class motif, it rises.
    Source: results/upgrades/shuffled_neg.csv
    (The companion GC panel of the old figure is omitted: GC -> 0.5 against a dinucleotide-preserving
    shuffle is true by construction and carries no information.)
(b) SPECIFICITY -- information-content-matched control PWMs. Random-IC controls land on chance, so
    the scan invents no skew; column-scrambled TBP (composition kept, order destroyed) stays skewed
    like real TATA, so the skew is compositional.
    Source: results/upgrades/route3_specificity.csv
(c) THRESHOLD  -- the same contrast swept over 0.8-1.5 bits/pos.
    Source: results/upgrades/route3_threshold_invariance.csv

Drawn at 5.5 in wide and included at \\textwidth, i.e. 1:1, so every label is >= 8 pt on the page.
Writes results/upgrades/figures/route3_evidence.{pdf,png}.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
U = os.path.join(ROOT, "results", "upgrades")
OUTDIR = os.path.join(U, "figures")

C_REAL = "#D55E00"      # real negatives / original split / real TATA
C_SHUF = "#0072B2"      # shuffled negatives / random-IC controls
C_SCRAM = "#E69F00"     # column-scrambled TBP
C_EQ = "#009E73"        # composition-equalized split
DS_LAB = {"human_enhancers_cohn": "cohn", "nt_enhancers": "nt_enh",
          "drosophila_enhancers_stark": "fly\n(control)"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8.5,
    "axes.titlesize": 8.8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _ci(s):
    lo, hi = str(s).strip("[]").split(",")
    return float(lo), float(hi)


def panel_magnitude(ax):
    df = pd.read_csv(os.path.join(U, "shuffled_neg.csv"))
    datasets = list(dict.fromkeys(df["dataset"]))
    x = np.arange(len(datasets))
    w = 0.36
    for off, arm, col, lab in [(-w / 2, "original_neg", C_REAL, "real negatives"),
                               (w / 2, "shuffled_neg", C_SHUF, "dinucleotide-shuffled")]:
        ys, los, his = [], [], []
        for ds in datasets:
            r = df[(df.dataset == ds) & (df.arm == arm)].iloc[0]
            lo, hi = _ci(r["tata_ci"])
            ys.append(r["tata_auroc"]); los.append(r["tata_auroc"] - lo); his.append(hi - r["tata_auroc"])
        ax.bar(x + off, ys, w, yerr=[los, his], color=col, label=lab, capsize=2,
               error_kw=dict(elinewidth=0.8))
    ax.axhline(0.5, color="0.4", lw=0.9, ls="--", zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([DS_LAB.get(d, d) for d in datasets], fontsize=7.4)
    ax.set_ylim(0.3, 0.85)
    ax.set_ylabel("TATA motif-only AUROC")
    ax.set_title("(a) Shuffled negatives")
    ax.legend(frameon=False, loc="upper left", handlelength=1.1, borderpad=0.15, labelspacing=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def panel_specificity(ax):
    spec = pd.read_csv(os.path.join(U, "route3_specificity.csv"))
    datasets = [d for d in ("human_enhancers_cohn", "nt_enhancers") if d in set(spec.dataset)]
    ax.axhline(0.5, color="0.55", lw=0.9, ls="--", zorder=0)
    for i, ds in enumerate(datasets):
        g = spec[spec.dataset == ds].set_index("motif_class")
        rt = g.loc["real_TATA"]
        ax.errorbar(i - 0.22, rt.motif_only_auroc,
                    yerr=[[rt.motif_only_auroc - rt.auroc_ci_lo],
                          [rt.auroc_ci_hi - rt.motif_only_auroc]],
                    fmt="D", color=C_REAL, ms=5, capsize=3, lw=1.3, zorder=4,
                    label="real TATA" if i == 0 else None)
        sc = g.loc["scrambled_TBP_panel"]
        ax.errorbar(i, sc.motif_only_auroc, yerr=sc.auroc_sd, fmt="o", color=C_SCRAM, ms=4.5,
                    capsize=3, lw=1.3, zorder=3,
                    label="scrambled TBP" if i == 0 else None)
        rd = g.loc["random_IC_panel"]
        ax.errorbar(i + 0.22, rd.motif_only_auroc, yerr=rd.auroc_sd, fmt="s", color=C_SHUF, ms=4.5,
                    capsize=3, lw=1.3, zorder=3,
                    label="random IC" if i == 0 else None)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([DS_LAB.get(d, d) for d in datasets])
    ax.set_xlim(-0.55, len(datasets) - 0.45)
    ax.set_ylim(0.30, 0.68)
    ax.set_title("(b) IC-matched controls")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.4, borderpad=0.15, labelspacing=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def panel_threshold(ax):
    thr = pd.read_csv(os.path.join(U, "route3_threshold_invariance.csv"))
    ax.axhline(0.5, color="0.55", lw=0.9, ls="--", zorder=0)
    for ds, ls in (("human_enhancers_cohn", "-"), ("nt_enhancers", "--")):
        for split, col, mk in (("original", C_REAL, "o"), ("comp_equalized", C_EQ, "s")):
            g = thr[(thr.dataset == ds) & (thr.split == split)].sort_values("threshold_bits")
            if not len(g):
                continue
            ax.fill_between(g.threshold_bits, g.tata_ci_lo, g.tata_ci_hi, color=col, alpha=0.12, lw=0)
            lab = ("original" if split == "original" else "equalized") if ds == "human_enhancers_cohn" else None
            ax.plot(g.threshold_bits, g.tata_auroc, ls=ls, marker=mk, color=col, ms=3.5, lw=1.3,
                    label=lab)
    ax.set_xlabel("TATA threshold (bits/pos)")
    ax.set_ylim(0.25, 0.62)
    ax.set_title("(c) Threshold sweep")
    ax.legend(frameon=False, loc="lower right", handlelength=1.4, handletextpad=0.4,
              borderpad=0.15, labelspacing=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.28),
                             gridspec_kw=dict(width_ratios=[1.1, 0.85, 1.05]))
    panel_magnitude(axes[0])
    panel_specificity(axes[1])
    panel_threshold(axes[2])
    fig.tight_layout(pad=0.35, w_pad=0.7)
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"route3_evidence.{ext}"), dpi=400)
    print(f"wrote {OUTDIR}/route3_evidence.pdf and .png")


if __name__ == "__main__":
    main()
