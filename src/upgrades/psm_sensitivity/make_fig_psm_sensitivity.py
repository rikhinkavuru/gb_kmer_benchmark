#!/usr/bin/env python3
"""Figure: k-mer AUROC and class-balanced MCC vs retained-negative fraction, composition-matched vs
size-matched random, per dataset (Reviewer 1, comment 4). IEEE single column (3.5 in), fonts >= 7 pt,
colorblind-safe pair (#0072B2 / #D55E00, Okabe-Ito; identity is also carried by line style and
filled-vs-hollow markers, never by color alone).

Curves: F1 tempered propensity-histogram matching sweep (alpha 0 -> 1; alpha = 1 applies the paper's
equalizer rule, i.e. the paper's per-bin counts, re-drawn with selection seed 42 -- it is NOT the published
split) vs uniform-random negatives at the same per-split counts. Matched band = 95% bootstrap CI of the
seed-42 selection; random band = min-max over the 10 random seeds. The paper's PUBLISHED equalized split is
drawn separately as a hollow star with its 95% bootstrap CI (cohn 0.613 vs the seed-42 re-draw 0.621).
Markers (removal-free / symmetric designs): the k-mer model RETRAINED with overlap weights on the ORIGINAL
split and scored on the overlap-weighted original test (square; x = the negatives' effective sample size; cohn
0.671) vs the permuted-weight placebo (retrained the same way), and two-sided 1:1 caliper-0.2 matching
(diamond; x = fraction of each class kept) vs random 1:1 subsets of the same size. The unretrained original
model on the overlap-weighted test (cohn 0.609 vs placebo 0.806) is NOT plotted. Markers are dodged in x so
they do not overlap: matched / weighted markers at x * 1.08, random / placebo markers at x / 1.08, the star at
x / 1.12; their x positions are therefore approximate.
With --with-f3 the full two-sided caliper sweep (calipers 2 -> 0.01 SD) is drawn as a thin line.
"Balanced MCC" = MCC with each class re-weighted to equal total weight, at the prior-corrected threshold
t = positive fraction of the TRAINING set (column mcc_bw_pc; e.g. t = 0.963 for cohn at alpha = 1).

nt_enhancers is drawn from the POST HOC leak-free re-analysis (nt_enhancers_leakfreeTest: the same models,
scored without the 100 test positives that occur verbatim in the training split); the uncorrected nt
numbers are inflated by that leakage (see run_posthoc.py P0/P1). Pass --nt-raw to draw the uncorrected nt.
Each column has its own x-range (the clean control's matching only goes down to ~43% retention).

Reads results/upgrades/psm_sensitivity_{summary,sweep,posthoc_sweep}.csv; writes
results/upgrades/figures/psm_sensitivity.{pdf,png}.

--caliper (fix round 3; supplementary figure psm_sensitivity_caliper.{pdf,png}): the literal sensitivity curve
across propensity-score matching thresholds. Two-sided 1:1 nearest-neighbour matching at calipers 2, 1, 0.5, 0.2 and
0.01 SD of the logit propensity (pre-registered part 4b, within design), drawn at equal spacing. Top: k-mer AUROC, matched (solid; 95%
bootstrap CI band) vs random 1:1 subsets of the same size (dashed; 10-seed min-max band). Bottom: balance of the
matched set (max |SMD| over the 20 composition features; dotted line = 0.1) vs the random subsets. Reads
results/upgrades/psm_sensitivity_caliper_sweep.csv (written by summarize_psm.py).
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import _common as C  # noqa: E402

MATCH, CTRL = "#0072B2", "#D55E00"
DATASETS = [("human_enhancers_cohn", "cohn"), ("nt_enhancers_leakfreeTest", "nt_enh. (leak-free)"),
            ("drosophila_enhancers_stark", "drosophila (ctrl)")]
XAXIS = {"drosophila_enhancers_stark": ((1.12, 0.36), [1, 0.7, 0.5, 0.4], ["100", "70", "50", "40"])}
XDEFAULT = ((1.15, 0.035), [1, 0.3, 0.1, 0.05], ["100", "30", "10", "5"])
METRICS = [("auroc", "k-mer AUROC", 0.5), ("mcc_bw_pc", "Balanced MCC", 0.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=C.RESULTS_DIR)
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "figures", "psm_sensitivity"))
    ap.add_argument("--with-f3", action="store_true", help="also draw the two-sided caliper sweep")
    ap.add_argument("--nt-raw", action="store_true", help="draw the uncorrected (leaky) nt_enhancers numbers")
    ap.add_argument("--caliper", action="store_true", help="supplementary caliper-sweep figure instead")
    args = ap.parse_args()
    global DATASETS
    if args.nt_raw:
        DATASETS = [DATASETS[0], ("nt_enhancers", "nt_enhancers (raw)"), DATASETS[2]]
    if args.caliper:
        if args.out == os.path.join(C.RESULTS_DIR, "figures", "psm_sensitivity"):
            args.out += "_caliper"
        return caliper_fig(args)
    summ = pd.read_csv(os.path.join(args.dir, "psm_sensitivity_summary.csv"))
    sw = pd.read_csv(os.path.join(args.dir, "psm_sensitivity_sweep.csv"))
    php = os.path.join(args.dir, "psm_sensitivity_posthoc_sweep.csv")
    if os.path.exists(php):
        sw = pd.concat([sw, pd.read_csv(php)], ignore_index=True)
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "font.family": "sans-serif",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5})
    fig, axes = plt.subplots(2, 3, figsize=(3.5, 3.0), sharex="col", sharey="row")
    ylims = {m: [np.inf, -np.inf] for m, _, _ in METRICS}
    for j, (ds, title) in enumerate(DATASETS):
        s = summ[(summ["dataset"] == ds) & (summ["model"] == "kmer") & (summ["design"] == "within")]
        o = sw[(sw["dataset"] == ds) & (sw["family"] == "original") & (sw["model"] == "kmer")
               & (sw["design"] == "within")].iloc[0]
        f1 = s[s["family"] == "F1_hist"].sort_values("retained_neg_frac", ascending=False)
        f3all = s[s["family"] == "F3_1to1"].sort_values("retained_neg_frac", ascending=False)
        f3 = f3all[f3all["level_label"] == "c0.2"].iloc[0]
        wo = s[s["family"] == "W_overlap"].iloc[0]
        pa = s[s["family"] == "paper_anchor"]
        pa = pa.iloc[0] if len(pa) else None
        for i, (mk, ylab, chance) in enumerate(METRICS):
            ax = axes[i, j]
            x = np.r_[1.0, f1["retained_neg_frac"].to_numpy()]
            ym = np.r_[o[mk], f1[f"{mk}_matched"].to_numpy()]
            lo = np.r_[o[f"{mk}_lo"], f1[f"{mk}_matched_lo"].to_numpy()]
            hi = np.r_[o[f"{mk}_hi"], f1[f"{mk}_matched_hi"].to_numpy()]
            yr = np.r_[o[mk], f1[f"{mk}_random_mean"].to_numpy()]
            rlo = np.r_[o[mk], f1[f"{mk}_random_min"].to_numpy()]
            rhi = np.r_[o[mk], f1[f"{mk}_random_max"].to_numpy()]
            ax.axhline(chance, color="0.6", lw=0.5, ls=":", zorder=0)
            ax.fill_between(x, lo, hi, color=MATCH, alpha=0.18, lw=0, zorder=1)
            ax.fill_between(x, rlo, rhi, color=CTRL, alpha=0.22, lw=0, zorder=1)
            ax.plot(x, yr, color=CTRL, lw=1.0, ls="--", marker="o", ms=2.0, mfc="white", mew=0.6, zorder=3)
            ax.plot(x, ym, color=MATCH, lw=1.0, marker="o", ms=2.0, zorder=4)
            if args.with_f3:
                xf = f3all["retained_neg_frac"].to_numpy()
                ax.plot(xf, f3all[f"{mk}_matched"], color=MATCH, lw=0.6, ls=":", zorder=2)
                ax.plot(xf, f3all[f"{mk}_random_mean"], color=CTRL, lw=0.6, ls=":", zorder=2)
            # removal-free / symmetric designs (x = retained fraction per class, or effective sample size)
            for row, xv, mkr in ((f3, f3["retained_neg_frac"], "D"), (wo, wo["ess_neg_frac"], "s")):
                ax.errorbar(xv * 1.08, row[f"{mk}_matched"],
                            yerr=[[row[f"{mk}_matched"] - row[f"{mk}_matched_lo"]],
                                  [row[f"{mk}_matched_hi"] - row[f"{mk}_matched"]]],
                            fmt=mkr, ms=3.0, color=MATCH, mec="white", mew=0.4, elinewidth=0.7, capsize=0,
                            zorder=6)
                ax.errorbar(xv / 1.08, row[f"{mk}_random_mean"],
                            yerr=[[row[f"{mk}_random_mean"] - row[f"{mk}_random_min"]],
                                  [row[f"{mk}_random_max"] - row[f"{mk}_random_mean"]]],
                            fmt=mkr, ms=3.0, color=CTRL, mfc="white", mew=0.7, elinewidth=0.7, capsize=0,
                            zorder=5)
                ylims[mk][0] = min(ylims[mk][0], row[f"{mk}_matched_lo"], row[f"{mk}_random_min"])
                ylims[mk][1] = max(ylims[mk][1], row[f"{mk}_matched_hi"], row[f"{mk}_random_max"])
            if pa is not None:          # the paper's published equalized split (not the seed-42 re-draw)
                ax.errorbar(pa["retained_neg_frac"] / 1.12, pa[f"{mk}_matched"],
                            yerr=[[pa[f"{mk}_matched"] - pa[f"{mk}_matched_lo"]],
                                  [pa[f"{mk}_matched_hi"] - pa[f"{mk}_matched"]]],
                            fmt="*", ms=4.0, color=MATCH, mfc="white", mew=0.6, elinewidth=0.6, capsize=0, zorder=6)
                ylims[mk][0] = min(ylims[mk][0], pa[f"{mk}_matched_lo"])
                ylims[mk][1] = max(ylims[mk][1], pa[f"{mk}_matched_hi"])
            ylims[mk][0] = min(ylims[mk][0], np.nanmin(lo), np.nanmin(rlo))
            ylims[mk][1] = max(ylims[mk][1], np.nanmax(hi), np.nanmax(rhi))
            ax.set_xscale("log")
            (xl, xt, xtl) = XAXIS.get(ds, XDEFAULT)
            ax.set_xlim(*xl)
            ax.set_xticks(xt)
            ax.set_xticklabels(xtl)
            ax.minorticks_off()
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.grid(axis="y", color="0.9", lw=0.4, zorder=0)
            if i == 0:
                ax.set_title(title, pad=2)
            if j == 0:
                ax.set_ylabel(ylab)
    for i, (mk, _, chance) in enumerate(METRICS):
        lo_, hi_ = min(ylims[mk][0], chance), ylims[mk][1]
        pad = 0.04 * (hi_ - lo_)
        axes[i, 0].set_ylim(lo_ - pad, hi_ + pad)
    fig.text(0.56, 0.008, "Negatives retained, % (log; weighting: effective sample size)",
             ha="center", va="bottom", fontsize=7)
    handles = [Line2D([], [], color=MATCH, lw=1.0, marker="o", ms=2.5, label="composition-matched"),
               Line2D([], [], color=CTRL, lw=1.0, ls="--", marker="o", ms=2.5, mfc="white",
                      label="size-matched random"),
               Line2D([], [], color=MATCH, lw=0, marker="*", ms=4.0, mfc="white", mew=0.6,
                      label="published equalized split"),
               Line2D([], [], color="0.3", lw=0, marker="D", ms=3.0, label="two-sided 1:1 matching"),
               Line2D([], [], color="0.3", lw=0, marker="s", ms=3.0, label="overlap-wt. retraining")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, handlelength=1.8,
               columnspacing=1.0, handletextpad=0.4, borderaxespad=0.1, bbox_to_anchor=(0.54, 1.0))
    fig.subplots_adjust(left=0.14, right=0.985, top=0.815, bottom=0.115, wspace=0.14, hspace=0.14)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out + ".pdf")
    fig.savefig(args.out + ".png", dpi=300)
    print("wrote", args.out + ".{pdf,png}")


def caliper_fig(args):
    cal = pd.read_csv(os.path.join(args.dir, "psm_sensitivity_caliper_sweep.csv"))
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "font.family": "sans-serif",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5})
    fig, axes = plt.subplots(2, 3, figsize=(3.5, 2.55), sharex=True, sharey="row",
                             gridspec_kw=dict(height_ratios=[1.6, 1.0]))
    xt = [2, 1, 0.5, 0.2, 0.01]
    for j, (ds, title) in enumerate(DATASETS):
        c = cal[cal["dataset"] == ds].sort_values("caliper_sd", ascending=False)
        assert c["caliper_sd"].tolist() == xt, c["caliper_sd"].tolist()
        x = np.arange(len(c))                      # calipers at equal spacing (categorical axis)
        ax = axes[0, j]
        ax.axhline(0.5, color="0.6", lw=0.5, ls=":", zorder=0)
        ax.fill_between(x, c["kmer_matched_lo"], c["kmer_matched_hi"], color=MATCH, alpha=0.18, lw=0, zorder=1)
        ax.fill_between(x, c["kmer_random_min"], c["kmer_random_max"], color=CTRL, alpha=0.22, lw=0, zorder=1)
        ax.plot(x, c["kmer_random_mean"], color=CTRL, lw=1.0, ls="--", marker="o", ms=2.0, mfc="white", mew=0.6, zorder=3)
        ax.plot(x, c["kmer_matched"], color=MATCH, lw=1.0, marker="o", ms=2.0, zorder=4)
        ax.set_title(title, pad=2)
        ax2 = axes[1, j]
        ax2.axhline(0.1, color="0.4", lw=0.5, ls=":", zorder=0)
        ax2.plot(x, c["max_abs_smd_random_mean"], color=CTRL, lw=1.0, ls="--", marker="o", ms=2.0, mfc="white", mew=0.6)
        ax2.plot(x, c["max_abs_smd_matched"], color=MATCH, lw=1.0, marker="o", ms=2.0)
        for a in (ax, ax2):
            a.set_xlim(-0.35, len(xt) - 0.65)
            a.set_xticks(np.arange(len(xt)))
            a.set_xticklabels(["2", "1", ".5", ".2", ".01"])
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            a.grid(axis="y", color="0.9", lw=0.4, zorder=0)
    axes[0, 0].set_ylabel("k-mer AUROC")
    axes[1, 0].set_ylabel("max |SMD|")
    lo = min(cal["kmer_matched_lo"].min(), cal["kmer_random_min"].min(), 0.5)
    hi = max(cal["kmer_matched_hi"].max(), cal["kmer_random_max"].max())
    axes[0, 0].set_ylim(lo - 0.02, hi + 0.02)
    axes[1, 0].set_ylim(0, max(cal["max_abs_smd_matched"].max(), cal["max_abs_smd_random_mean"].max()) * 1.08)
    fig.text(0.56, 0.008, "Caliper, SD of logit propensity (stricter to the right)", ha="center", va="bottom", fontsize=7)
    handles = [Line2D([], [], color=MATCH, lw=1.0, marker="o", ms=2.5, label="1:1 caliper-matched"),
               Line2D([], [], color=CTRL, lw=1.0, ls="--", marker="o", ms=2.5, mfc="white",
                      label="random 1:1, same size")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, handlelength=1.8,
               columnspacing=1.0, handletextpad=0.4, borderaxespad=0.1, bbox_to_anchor=(0.54, 1.0))
    fig.subplots_adjust(left=0.15, right=0.985, top=0.87, bottom=0.13, wspace=0.14, hspace=0.18)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out + ".pdf")
    fig.savefig(args.out + ".png", dpi=300)
    print("wrote", args.out + ".{pdf,png}")


if __name__ == "__main__":
    main()
