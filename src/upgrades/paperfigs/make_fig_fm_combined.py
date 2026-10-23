#!/usr/bin/env python3
"""Foundation-model figure (paper Figure 5): frozen probes and fine-tuning on one row.

(a) Frozen probes -- the enhancer AUROC a FIXED head loses when composition-biased negatives
    are removed from the test set (test_effect), for two architecturally-unrelated FMs, each
    against its own flat control.
    Source: results/upgrades/fm_cross_architecture.csv, results/upgrades/dnabert_control.csv
(b) Fine-tuned HyenaDNA-tiny -- test AUROC per arm (trained and evaluated within the arm,
    three seeds), against the composition-only floor and the k-mer classifier on the same
    splits. The right-most bar is the original-trained model evaluated, without retraining,
    on the composition-equalized test set.
    Source: results/upgrades/fm_finetune_summary.csv, results/upgrades/composition_clean.csv

Drawn at 5.5 in wide and included at \\textwidth, so all labels are >= 8 pt on the page.
Writes results/upgrades/figures/fm_combined.{pdf,png}.
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

HYENA_C = "#0072B2"
DNABERT_C = "#D55E00"
CTRL_C = "#999999"
FLOOR_C = "#009E73"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8.5,
    "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.6, "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _ci(s):
    lo, hi = str(s).strip("[]").split(",")
    return float(lo), float(hi)


def panel_frozen(ax):
    cx = pd.read_csv(os.path.join(U, "fm_cross_architecture.csv"))
    cx = cx[cx["head"] == "lgbm"]
    rows = []
    for arch, c in [("HyenaDNA-tiny-16k", HYENA_C), ("DNABERT-2-117M", DNABERT_C)]:
        g = cx[cx.architecture == arch]
        for ds, lab in [("human_enhancers_cohn", "cohn"), ("nt_enhancers", "nt_enh"),
                        ("drosophila_enhancers_stark", "fly (ctrl)")]:
            r = g[g.dataset == ds]
            if not len(r):
                continue
            r = r.iloc[0]
            lo, hi = _ci(r["ci"])
            rows.append((lab, arch, float(r.test_effect_auroc), lo, hi,
                         CTRL_C if "ctrl" in lab else c))
    # DNABERT-2's own flat control: the composition-representative placebo
    pl = pd.read_csv(os.path.join(U, "dnabert_control.csv"))
    pl = pl[(pl.arm == "placebo") & (pl["head"] == "lgbm") & (pl.pooling == "mean")].iloc[0]
    lo, hi = _ci(pl["test_effect_auroc_ci"])
    rows.append(("placebo (ctrl)", "DNABERT-2-117M",
                 float(pl.test_effect_auroc), lo, hi, CTRL_C))

    y = np.arange(len(rows))[::-1]
    for yi, (lab, arch, v, lo, hi, c) in zip(y, rows):
        ax.barh(yi, v, height=0.62, color=c,
                xerr=[[v - lo], [hi - v]], capsize=2, error_kw=dict(elinewidth=0.8))
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab}\n{arch.split('-')[0]}" for lab, arch, *_ in rows], fontsize=7.0)
    ax.axvline(0, color="0.35", lw=0.9)
    ax.set_xlabel("$\\Delta$ AUROC, composition removed")
    ax.set_title("(a) Frozen probes")
    ax.spines[["top", "right"]].set_visible(False)


def panel_finetune(ax):
    ft = pd.read_csv(os.path.join(U, "fm_finetune_summary.csv"))
    ft = ft[ft.dataset == "human_enhancers_cohn"]
    raw = pd.read_csv(os.path.join(U, "fm_finetune.csv"))
    raw = raw[raw.dataset == "human_enhancers_cohn"]
    cc = pd.read_csv(os.path.join(U, "composition_clean.csv"))
    cc = cc[cc.dataset == "human_enhancers_cohn"].set_index("arm")

    bars = [("original", "original", "released"),
            ("gc_match", "gc_match", "GC-\nmatched"),
            ("comp_equalized", "comp_equalized", "equalized"),
            ("original", "comp_equalized", "released\u2192\nequalized")]
    x = np.arange(len(bars))
    for i, (tr, ev, _) in enumerate(bars):
        row = ft[(ft.train_arm == tr) & (ft.eval_arm == ev)]
        if not len(row):
            continue
        m = float(row.auroc_mean.iloc[0])
        hatch = "//" if tr != ev else None
        ax.bar(i, m, 0.62, color=HYENA_C, alpha=0.9 if tr == ev else 0.45, hatch=hatch,
               edgecolor=HYENA_C, lw=0.8,
               label="fine-tuned HyenaDNA-tiny" if i == 0 else None, zorder=2)
        pts = raw[(raw.train_arm == tr) & (raw.eval_arm == ev)]["auroc"]
        ax.scatter([i] * len(pts), pts, s=8, color="0.15", zorder=4)
    # composition-only floor and the k-mer classifier on the SAME splits
    for i, (tr, ev, _) in enumerate(bars):
        arm = ev
        ax.plot([i - 0.34, i + 0.34], [cc.loc[arm, "comp_only_auroc"]] * 2, color=FLOOR_C,
                lw=1.8, zorder=5, label="composition-only floor" if i == 0 else None)
        ax.plot([i - 0.34, i + 0.34], [cc.loc[arm, "kmer_auroc"]] * 2, color="0.25", lw=1.2,
                ls=":", zorder=5, label="$k$-mer + LightGBM" if i == 0 else None)
    ax.axhline(0.5, color="0.6", lw=0.8, ls="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([b[2] for b in bars], fontsize=7.4)
    ax.set_ylim(0.47, 0.93)
    ax.set_ylabel("enhancer test AUROC")
    ax.set_title("(b) Fine-tuned, 3 seeds (cohn)")
    h, l = ax.get_legend_handles_labels()
    order = [l.index(k) for k in ("fine-tuned HyenaDNA-tiny", "composition-only floor",
                                 "$k$-mer + LightGBM") if k in l]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=1,
              frameon=False, handlelength=1.3, handletextpad=0.4, borderpad=0.1,
              labelspacing=0.2, fontsize=7.2)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.38), gridspec_kw=dict(width_ratios=[1, 1.15]))
    panel_frozen(axes[0])
    panel_finetune(axes[1])
    fig.tight_layout(pad=0.35, w_pad=0.8)
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"fm_combined.{ext}"), dpi=400)
    print(f"wrote {OUTDIR}/fm_combined.pdf and .png")


if __name__ == "__main__":
    main()
