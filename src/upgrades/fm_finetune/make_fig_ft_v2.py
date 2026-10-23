#!/usr/bin/env python3
"""Compact IEEE single-column figures for the end-to-end fine-tuning result (R1 comment 2).

(1) results/upgrades/figures/fm_finetune_v2.{pdf,png} -- fine-tuned models only.
One row per fine-tuned model x dataset. x = test AUROC, seed mean with its 95% bootstrap CI (1000
resamples of the test sequences, seeds averaged within a resample); small ticks = individual seeds.
Encoding:
  colour = the TEST set's negatives   black: released   blue: random subsample (size-matched)
                                      vermillion: composition-equalized
  marker = the TRAINING negatives     circle: released  square: the same arm as the test set
So black/blue/vermillion circles are ONE set of weights scored on three test sets, and the squares
are models retrained on the size-matched random or composition-equalized negatives.
nt_enhancers is shown on its LEAK-FREE test (the 100 test positives that are verbatim training
sequences removed; post hoc, posthoc_ft_v2.py); the released-test numbers are in the CSVs.

(2) results/upgrades/figures/fm_finetune_v2_compare.{pdf,png} -- the candidate paper figure
(post hoc). Share of the released-split model's above-chance AUROC that the SAME weights lose when
the test negatives are composition-equalized, (A-B)/(A-0.5) (vermillion), versus when they are a
random subsample of the same size and class ratio, (A-C)/(A-0.5) (blue), with 95% bootstrap CIs, for
the k-mer LightGBM, the frozen-FM probes (the paper's headline cells: HyenaDNA [SEP], DNABERT-2
mean-pool; LightGBM heads) and the fine-tuned FMs, on cohn, leak-free nt_enhancers and the
composition-aligned drosophila control. The pooling cells were fixed before the post hoc numbers
were computed. AFTER seeing them (2026-09-26), a HyenaDNA frozen MEAN-pool row was added: the fine-tuned
models mean-pool, and the paper's flat drosophila control holds for [SEP] pooling only (numbers
ledger C2), so the [SEP] row alone would flatter the frozen control. Every other cell is in
fm_finetune_v2_compare.csv. The x-axis is clipped to [-0.5, 1.5]; CIs that run past it (only the
leak-free nt_enhancers rows, whose equalized arm has 36 negatives) end in an arrowhead.
Fix round 1 (2026-09-26, labels only): drosophila is labelled "GC-balanced control", not
"composition-aligned": it keeps a dinucleotide gap (composition-only LR AUROC 0.689 on its released
test, fm_finetune_v2_split_manifest.csv), and pre-registered rule 5 was not met. Everything on the
x-axis of figure (2) (the share lost) and the leak-free nt rows is post hoc; the pre-registered
released-test nt_enhancers rows are in the CSVs, not in the figure (say so in the caption).

Okabe-Ito colours; exactly 3.5 in wide (no tight-bbox growth); every font >= 7 pt (asserted).
Inputs : results/upgrades/fm_finetune_v2_{summary,metrics,posthoc_summary,posthoc_metrics,compare}.csv
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES = os.path.join(ROOT, "results", "upgrades")
FIG = os.path.join(RES, "figures")

CONDS = [  # (train_arm, eval_arm, label, colour, marker, filled)
    ("original", "original", "released → released", "#000000", "o", True),
    ("original", "random", "released → random neg.", "#0072B2", "o", True),
    ("original", "comp_equalized", "released → equalized neg.", "#D55E00", "o", True),
    ("random", "random", "random → random", "#0072B2", "s", False),
    ("comp_equalized", "comp_equalized", "equalized → equalized", "#D55E00", "s", False),
]
MODEL_LABEL = {"hyenadna": "HyenaDNA (full FT)", "dnabert2": "DNABERT-2 (LoRA)",
               "dnabert2full": "DNABERT-2 (full FT)", "ntv2_500m": "NT-v2-500M (LoRA)"}
DS_LABEL = {"human_enhancers_cohn": "cohn", "nt_enhancers": "nt_enhancers",
            "nt_enhancers_leakfree": "nt_enh. (leak-free)",
            "drosophila_enhancers_stark": "drosophila (GC-bal. ctl)"}
WIDTH = 3.5          # IEEE column width, inches; the saved figure is exactly this wide
MIN_PT = 7.0
FT_MODELS = ("hyenadna", "dnabert2", "dnabert2full", "ntv2_500m")
DATASETS = ("human_enhancers_cohn", "nt_enhancers_leakfree", "drosophila_enhancers_stark")


def style():
    plt.rcParams.update({"font.size": MIN_PT, "axes.labelsize": MIN_PT, "xtick.labelsize": MIN_PT,
                         "ytick.labelsize": MIN_PT, "legend.fontsize": MIN_PT, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "font.family": "sans-serif"})


def guard(fig, ax):
    """Everything inside the 3.5 in canvas, legend above the axes, no text below MIN_PT."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    axbb = ax.get_window_extent(rend)
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_text() or not t.get_visible():
            continue
        e = t.get_window_extent(rend)
        assert e.x0 >= -1 and e.x1 <= W + 1 and e.y0 >= -1 and e.y1 <= H + 1, (t.get_text(), e, W, H)
    if fig.legends:
        leg = fig.legends[0].get_window_extent(rend)
        assert leg.y0 >= axbb.y1 - 1, ("legend overlaps the axes", leg, axbb)
    sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text()]
    assert min(sizes) >= MIN_PT, min(sizes)
    assert abs(fig.get_size_inches()[0] - WIDTH) < 1e-9


def fig_auroc():
    summ = pd.concat([pd.read_csv(os.path.join(RES, "fm_finetune_v2_summary.csv")),
                      pd.read_csv(os.path.join(RES, "fm_finetune_v2_posthoc_summary.csv"))])
    met = pd.concat([pd.read_csv(os.path.join(RES, "fm_finetune_v2_metrics.csv")),
                     pd.read_csv(os.path.join(RES, "fm_finetune_v2_posthoc_metrics.csv"))])
    order = [(m, d) for d in DATASETS for m in FT_MODELS]
    groups = [g for g in order if ((summ.model == g[0]) & (summ.dataset == g[1])).any()]
    style()
    leg_h, bot_h = 0.68, 0.40                      # inches: legend band on top, x-axis label band
    h = 0.62 * len(groups) + leg_h + bot_h
    fig, ax = plt.subplots(figsize=(WIDTH, h))
    fig.subplots_adjust(left=0.34, right=0.95, top=1 - leg_h / h, bottom=bot_h / h)
    offs = np.linspace(0.3, -0.3, len(CONDS))
    yt, yl = [], []
    lo_all, hi_all = [], []
    for gi, (model, ds) in enumerate(groups):
        y0 = len(groups) - 1 - gi
        yt.append(y0); yl.append(f"{DS_LABEL[ds]}\n{MODEL_LABEL.get(model, model)}")
        if gi % 2 == 0:
            ax.axhspan(y0 - 0.5, y0 + 0.5, color="#f2f2f2", zorder=0, lw=0)
        for (ta, ea, lab, col, mk, filled), dy in zip(CONDS, offs):
            r = summ[(summ.model == model) & (summ.dataset == ds) & (summ.train_arm == ta)
                     & (summ.eval_arm == ea)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = y0 + dy
            lo_all.append(r.auroc_seedmean_lo); hi_all.append(r.auroc_seedmean_hi)
            ax.plot([r.auroc_seedmean_lo, r.auroc_seedmean_hi], [y, y], color=col, lw=1.0,
                    zorder=2, solid_capstyle="butt")
            ax.plot(r.auroc_mean, y, marker=mk, ms=4.2, color=col, zorder=3,
                    mfc=col if filled else "white", mew=1.0, ls="none")
            s = met[(met.model == model) & (met.dataset == ds) & (met.train_arm == ta)
                    & (met.eval_arm == ea)]
            ax.plot(s.auroc, np.full(len(s), y), marker="|", ms=4.5, color=col, alpha=0.55,
                    ls="none", zorder=2, mew=0.8)
    ax.axvline(0.5, color="#777777", lw=0.7, ls="--", zorder=1)
    ax.set_yticks(yt); ax.set_yticklabels(yl)
    ax.set_ylim(-0.5, len(groups) - 0.5)
    ax.set_xlabel("test AUROC (seed mean, 95% CI; ticks = seeds)")
    ax.set_xlim(min(0.4, float(np.nanmin(lo_all)) - 0.01), max(0.9, float(np.nanmax(hi_all)) + 0.01))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], marker=mk, color=col, mfc=col if filled else "white", ls="-",
                          lw=1.0, ms=4.2, label=lab) for _, _, lab, col, mk, filled in CONDS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
               frameon=False, handlelength=1.4, columnspacing=0.9, borderaxespad=0.1,
               title="train negatives → test negatives (same positives)",
               title_fontsize=MIN_PT)
    guard(fig, ax)
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fm_finetune_v2.{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", os.path.join(FIG, "fm_finetune_v2.{pdf,png}"), "groups:", groups)


CMP_MODELS = [  # (model, row label, is fine-tuned)
    ("kmer_lgbm", "k-mer LightGBM", False),
    ("hyenadna_frozen_sep", "HyenaDNA frozen, [SEP]", False),
    # 2026-09-26, added after the post hoc numbers were seen: the fine-tuned HyenaDNA mean-pools, and
    # the paper's flat drosophila control holds for [SEP] pooling only (numbers_ledger.md C2,
    # fm_head_invariance.csv), so showing [SEP] alone would flatter the frozen control.
    ("hyenadna_frozen_mean", "HyenaDNA frozen, mean", False),
    ("hyenadna", "HyenaDNA full FT", True),
    ("dnabert2_frozen_mean", "DNABERT-2 frozen, mean", False),
    ("dnabert2", "DNABERT-2 LoRA FT", True),
    ("dnabert2full", "DNABERT-2 full FT", True),
    ("ntv2_500m", "NT-v2-500M LoRA FT", True),
]
XLIM = (-0.5, 1.5)   # leak-free nt CIs reach 2.2 (n = 136 on the equalized arm); clipped with arrows
CMP_DS = [("human_enhancers_cohn", "cohn"),
          ("nt_enhancers_leakfree", "nt_enhancers (leak-free test)"),
          ("drosophila_enhancers_stark", "drosophila (GC-balanced control)")]


def fig_compare():
    comp = pd.read_csv(os.path.join(RES, "fm_finetune_v2_compare.csv"))
    rows = []                                           # (y, label, fine-tuned, record) / headers
    y = 0.0
    heads = []
    for ds, dlab in CMP_DS:
        sub = [(m, lab, ft) for m, lab, ft in CMP_MODELS
               if ((comp.model == m) & (comp.dataset == ds) & comp.frac_lost.notna()).any()]
        if not sub:
            continue
        heads.append((y, dlab)); y += 0.85
        for m, lab, ft in sub:
            rows.append((y, lab, ft, comp[(comp.model == m) & (comp.dataset == ds)].iloc[0]))
            y += 1.0
        y += 0.2
    n = y
    style()
    leg_h, bot_h, row_h = 0.30, 0.36, 0.15
    h = row_h * n + leg_h + bot_h
    fig, ax = plt.subplots(figsize=(WIDTH, h))
    fig.subplots_adjust(left=0.37, right=0.97, top=1 - leg_h / h, bottom=bot_h / h)
    ax.set_ylim(n - 0.2, -0.7)
    x_head = -(0.37 * WIDTH - 0.04) / ((0.97 - 0.37) * WIDTH)   # 0.04 in from the canvas edge
    for yy, lab in heads:                    # header rows carry no data, so text may run into the axes
        ax.text(x_head, yy, lab, transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=MIN_PT, fontweight="bold", zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4))   # keeps the x=0 line off the text
    lo_x, hi_x = XLIM
    for yy, lab, ft, r in rows:
        if ft:
            ax.axhspan(yy - 0.5, yy + 0.5, color="#f0f0f0", zorder=0, lw=0)
        for key, col, dy, mfc in (("frac_lost_random", "#0072B2", -0.18, "white"),
                                  ("frac_lost", "#D55E00", 0.18, "#D55E00")):
            v, lo, hi = r[key], r[key + "_lo"], r[key + "_hi"]
            a, b = max(lo, lo_x), min(hi, hi_x)
            ax.plot([a, b], [yy + dy, yy + dy], color=col, lw=1.0, zorder=2, solid_capstyle="butt")
            if hi > hi_x:                     # CI runs past the axis: arrowhead at the edge
                ax.plot(hi_x, yy + dy, marker=">", ms=3.0, color=col, zorder=3, ls="none", clip_on=False)
            if lo < lo_x:
                ax.plot(lo_x, yy + dy, marker="<", ms=3.0, color=col, zorder=3, ls="none", clip_on=False)
            assert lo_x <= v <= hi_x, (r.model, r.dataset, key, v)
            ax.plot(v, yy + dy, marker="o", ms=3.6, color=col, mfc=mfc, mew=1.0, zorder=3, ls="none")
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows])
    for t, (_, _, ft, _) in zip(ax.get_yticklabels(), rows):
        if ft:
            t.set_fontweight("bold")
    ax.axvline(0.0, color="#777777", lw=0.7, ls="--", zorder=1)
    ax.axvline(1.0, color="#bbbbbb", lw=0.6, ls=":", zorder=1)
    ax.set_xlim(*XLIM)
    ax.set_xlabel("share of above-chance AUROC lost")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], marker="o", color="#D55E00", mfc="#D55E00", ls="-", lw=1.0, ms=3.6,
                          label="equalized neg. (A−B)/(A−.5)"),
               plt.Line2D([], [], marker="o", color="#0072B2", mfc="white", ls="-", lw=1.0, ms=3.6,
                          label="random neg. (A−C)/(A−.5)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
               frameon=False, handlelength=1.4, columnspacing=0.9, borderaxespad=0.1)
    guard(fig, ax)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fm_finetune_v2_compare.{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", os.path.join(FIG, "fm_finetune_v2_compare.{pdf,png}"), f"{len(rows)} rows")


if __name__ == "__main__":
    fig_auroc()
    if os.path.exists(os.path.join(RES, "fm_finetune_v2_compare.csv")):
        fig_compare()
