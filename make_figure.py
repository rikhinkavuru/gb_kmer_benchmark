#!/usr/bin/env python3
"""Main paper figure: 3-panel summary. Reads existing result CSVs only (no recompute).
CPU-only matplotlib. Saves vector PDF + 300-dpi PNG to results/figures/.

Layout: full text width (two-column journal), 1 row x 3 panels, ~7.2 x 2.9 in.
Colorblind-safe Okabe-Ito palette; route = hue (consistent across panels),
condition = shade (lighter = baseline, darker = enhanced/cleaned).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
FIG = os.path.join(R, "figures"); os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 7.5, "axes.labelsize": 8,
    "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.6, "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Okabe-Ito colorblind-safe palette, one hue per failure route
ROUTE_COLOR = {
    "solved (motif-local)": "#0072B2",   # blue
    "shared-motif":         "#E69F00",   # orange
    "contamination":        "#D55E00",   # vermillion
    "positional":           "#009E73",   # green
    "composition (non-TF)": "#999999",   # grey
}
ROUTE = {
    "human_nontata_promoters": "solved (motif-local)",
    "human_ensembl_regulatory": "solved (motif-local)",
    "nt_promoter_all": "solved (motif-local)",
    "nt_promoter_tata": "solved (motif-local)",
    "nt_promoter_no_tata": "solved (motif-local)",
    "human_enhancers_cohn": "contamination",
    "human_enhancers_ensembl": "contamination",
    "nt_enhancers": "contamination",
    "human_ocr_ensembl": "shared-motif",
    "drosophila_enhancers_stark": "shared-motif",
    "nt_enhancers_types": "shared-motif",
    "nt_splice_sites_all": "positional",
    "demo_human_or_worm": "composition (non-TF)",
    "demo_coding_vs_intergenomic_seqs": "composition (non-TF)",
    "dummy_mouse_enhancers_ensembl": "composition (non-TF)",
}
SHORT = {
    "human_nontata_promoters": "nontata", "human_ensembl_regulatory": "regulatory",
    "human_enhancers_cohn": "cohn", "human_enhancers_ensembl": "enh_ensembl",
    "human_ocr_ensembl": "ocr", "drosophila_enhancers_stark": "drosophila",
    "dummy_mouse_enhancers_ensembl": "dummy_mouse",
    "demo_coding_vs_intergenomic_seqs": "coding/interg", "demo_human_or_worm": "human/worm",
    "nt_promoter_all": "promoter_all", "nt_promoter_tata": "promoter_TATA",
    "nt_promoter_no_tata": "prom-noTATA", "nt_enhancers": "enhancers",
    "nt_enhancers_types": "enhancers_types", "nt_splice_sites_all": "splice",
}
SUITE_MARK = {"GenomicBench": "o", "NucTransformer": "^"}


def lighten(c, f=0.58):
    r, g, b = mcolors.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def parse_ci(s):
    if not isinstance(s, str) or "[" not in s:
        return (np.nan, np.nan)
    lo, hi = s.strip().strip("[]").split(",")
    return float(lo), float(hi)


def check_nan(df, cols, name):
    miss = []
    for _, r in df.iterrows():
        for c in cols:
            v = r[c]
            if (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == ""):
                miss.append((r.get("task", r.get("dataset", "?")), c))
    if miss:
        print(f"  !! MISSING/NaN in {name}: {miss}")
    else:
        print(f"  (no missing/NaN in {name} for plotted columns)")


fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.2, 2.9))

# ============================ Panel A ============================
print("\n=== PANEL A: cross_suite_summary.csv  (x=max_pos_motif_auroc, y=bench_mcc) ===")
a = pd.read_csv(os.path.join(R, "cross_suite_summary.csv"))
check_nan(a, ["bench_mcc", "max_pos_motif_auroc"], "Panel A")
a["route"] = a["task"].map(ROUTE)
for _, r in a.iterrows():
    xlo, xhi = parse_ci(r["motif_ci"]); ylo, yhi = parse_ci(r["mcc_ci"])
    print(f"  {r['suite']:<14} {r['task']:<32} x={r['max_pos_motif_auroc']:.3f} "
          f"[{xlo:.3f},{xhi:.3f}]  y={r['bench_mcc']:.3f} [{ylo:.3f},{yhi:.3f}]  route={r['route']}")
    # x-CIs (motif AUROC) come from a separate run/subsample and are omitted to avoid
    # mismatched bars; y error bars are the consistent 95% MCC bootstrap CI.
    axA.plot([r["max_pos_motif_auroc"]] * 2, [ylo, yhi], color="0.6", lw=0.6, alpha=0.5, zorder=1)
for suite, mk in SUITE_MARK.items():
    sub = a[a["suite"] == suite]
    axA.scatter(sub["max_pos_motif_auroc"], sub["bench_mcc"], marker=mk,
                c=[ROUTE_COLOR[r] for r in sub["route"]], s=34, zorder=3,
                edgecolors="black", linewidths=0.4)
# label key points
KEY = {"human_nontata_promoters": (5, 5), "human_ensembl_regulatory": (-3, -11),
       "human_enhancers_cohn": (6, -2), "nt_enhancers": (6, 2), "nt_splice_sites_all": (6, 0)}
for task, (dx, dy) in KEY.items():
    rr = a[a["task"] == task].iloc[0]
    axA.annotate(SHORT[task], (rr["max_pos_motif_auroc"], rr["bench_mcc"]),
                 textcoords="offset points", xytext=(dx, dy), fontsize=6.2, color="0.15")
# pooled Spearman (TF-motif tasks) -- computed from the CSV to verify
tf = a[a["tf_motif_task"] == True]
rho_tf, p_tf = spearmanr(tf["bench_mcc"], tf["max_pos_motif_auroc"])
rho_all, p_all = spearmanr(a["bench_mcc"], a["max_pos_motif_auroc"])
print(f"  Spearman pooled TF-motif (n={len(tf)}): rho={rho_tf:+.2f} p={p_tf:.2f} "
      f"| all-15: rho={rho_all:+.2f} p={p_all:.2f}")
axA.text(0.04, 0.97, f"pooled TF-motif tasks (n={len(tf)}):\n"
         rf"Spearman $\rho$={rho_tf:+.2f}, p={p_tf:.2f}",
         transform=axA.transAxes, va="top", ha="left", fontsize=6.4,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))
axA.text(0.97, 0.05, "○ GB   △ NT", transform=axA.transAxes, ha="right", va="bottom", fontsize=6.4)
axA.set_xlabel("max positive-direction motif-only AUROC")
axA.set_ylabel("benchmark LightGBM MCC")
axA.set_xlim(0.5, 0.8); axA.set_ylim(0.2, 1.0)
axA.spines[["top", "right"]].set_visible(False)

# ============================ Panel B ============================
print("\n=== PANEL B: positional_features.csv  (spectrum vs +positional MCC) ===")
b = pd.read_csv(os.path.join(R, "positional_features.csv"))
check_nan(b, ["spectrum_mcc", "positional_mcc", "delta_mcc"], "Panel B")
b["route"] = b["task"].map(ROUTE)
b = b.sort_values("delta_mcc", ascending=False).reset_index(drop=True)
x = np.arange(len(b)); w = 0.38
for i, r in b.iterrows():
    col = ROUTE_COLOR[r["route"]]
    slo, shi = parse_ci(r["spectrum_ci"]); plo, phi = parse_ci(r["positional_ci"])
    print(f"  {r['task']:<24} spectrum={r['spectrum_mcc']:.3f} [{slo:.3f},{shi:.3f}]  "
          f"+positional={r['positional_mcc']:.3f} [{plo:.3f},{phi:.3f}]  delta={r['delta_mcc']:+.3f}")
    axB.bar(i - w/2, r["spectrum_mcc"], w, color=lighten(col), edgecolor="black", lw=0.4,
            yerr=[[r["spectrum_mcc"]-slo], [shi-r["spectrum_mcc"]]], error_kw=dict(lw=0.7, capsize=1.6))
    axB.bar(i + w/2, r["positional_mcc"], w, color=col, edgecolor="black", lw=0.4,
            yerr=[[r["positional_mcc"]-plo], [phi-r["positional_mcc"]]], error_kw=dict(lw=0.7, capsize=1.6))
sp = b[b["task"] == "nt_splice_sites_all"].index[0]
axB.annotate(r"$\Delta$=+0.624", (sp, b.loc[sp, "positional_mcc"]), textcoords="offset points",
             xytext=(0, 7), ha="center", fontsize=6.6, fontweight="bold")
axB.text(0.97, 0.55, "controls\n|Δ| ≤ 0.013", transform=axB.transAxes, ha="right", va="center",
         fontsize=6.3, color="0.25")
axB.set_xticks(x); axB.set_xticklabels([SHORT[t] for t in b["task"]], rotation=40, ha="right", fontsize=6.0)
axB.set_ylabel("test MCC"); axB.set_ylim(0, 1.10)
axB.legend(handles=[Patch(fc="0.7", ec="black", lw=0.4, label="k-mer spectrum"),
                    Patch(fc="0.3", ec="black", lw=0.4, label="+ positional bins")],
           loc="upper right", frameon=False, fontsize=5.8, bbox_to_anchor=(1.0, 1.0), borderpad=0.2)
axB.spines[["top", "right"]].set_visible(False)

# ============================ Panel C ============================
print("\n=== PANEL C: contamination_cleaning.csv  (TATA AUROC original vs cleaned) ===")
c = pd.read_csv(os.path.join(R, "contamination_cleaning.csv"))
check_nan(c, ["orig_TATA_auroc", "cleaned_TATA_auroc", "frac_neg_removed"], "Panel C")
c["route"] = c["dataset"].map(ROUTE)
xc = np.arange(len(c))
for i, r in c.iterrows():
    col = ROUTE_COLOR[r["route"]]
    olo, ohi = parse_ci(r["orig_TATA_ci"]); clo, chi = parse_ci(r["cleaned_TATA_ci"])
    print(f"  {r['dataset']:<30} removed={r['frac_neg_removed']:.1%}  "
          f"TATA AUROC orig={r['orig_TATA_auroc']:.3f} [{olo:.3f},{ohi:.3f}] -> "
          f"cleaned={r['cleaned_TATA_auroc']:.3f} [{clo:.3f},{chi:.3f}]")
    axC.bar(i - w/2, r["orig_TATA_auroc"], w, color=lighten(col), edgecolor="black", lw=0.4,
            yerr=[[r["orig_TATA_auroc"]-olo], [ohi-r["orig_TATA_auroc"]]], error_kw=dict(lw=0.7, capsize=1.6))
    axC.bar(i + w/2, r["cleaned_TATA_auroc"], w, color=col, edgecolor="black", lw=0.4,
            yerr=[[r["cleaned_TATA_auroc"]-clo], [chi-r["cleaned_TATA_auroc"]]], error_kw=dict(lw=0.7, capsize=1.6))
    axC.text(i, 0.02, f"−{r['frac_neg_removed']*100:.0f}%" if r['frac_neg_removed']>0 else "0%",
             ha="center", va="bottom", fontsize=6.0, color="0.2")
axC.axhline(0.5, ls="--", lw=0.8, color="0.35")
axC.text(-0.40, 0.515, "no signal (0.5)", ha="left", va="bottom", fontsize=6.0, color="0.35")
lblC = {"human_enhancers_cohn": "cohn", "nt_enhancers": "nt_enhancers",
        "drosophila_enhancers_stark": "drosophila\n(control)"}
axC.set_xticks(xc); axC.set_xticklabels([lblC[d] for d in c["dataset"]], rotation=0, fontsize=6.6)
axC.set_ylabel("TATA motif-only AUROC"); axC.set_ylim(0, 0.66)
axC.legend(handles=[Patch(fc="0.7", ec="black", lw=0.4, label="original"),
                    Patch(fc="0.3", ec="black", lw=0.4, label="cleaned")],
           loc="upper left", frameon=False, fontsize=6.0)
axC.spines[["top", "right"]].set_visible(False)

# panel labels
for ax, lab in [(axA, "A"), (axB, "B"), (axC, "C")]:
    ax.text(-0.16, 1.06, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

# shared route legend at bottom
handles = [Patch(fc=ROUTE_COLOR[k], ec="black", lw=0.4, label=k) for k in ROUTE_COLOR]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=6.4,
           bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.30, wspace=0.34)

pdf = os.path.join(FIG, "main_figure.pdf"); png = os.path.join(FIG, "main_figure.png")
fig.savefig(pdf, bbox_inches="tight"); fig.savefig(png, dpi=300, bbox_inches="tight")
print(f"\nSaved: {pdf}\n       {png}")
print("Figure width: full text width (7.2 in, two-column journal); 1x3 panels.")

CAP = (
"Figure 1. CPU-only k-mer/LightGBM analysis of two DNA sequence-classification suites "
"(Genomic Benchmarks, GB; Nucleotide Transformer downstream tasks, NT). "
"(A) Benchmark learnability (LightGBM MCC) versus the strongest positive-direction "
"motif-only AUROC per task, pooled across suites and coloured by failure route; the "
"discriminability-learnability relationship is significant across TF-motif tasks "
f"(pooled Spearman rho={rho_tf:+.2f}, p={p_tf:.2f}, n={len(tf)}), while single-suite "
"correlations are underpowered and the composition/non-TF tasks (grey) fall off-trend "
"as expected. (B) Adding coarse positional information (position-binned k-mers, B=8) "
"recovers only the positional task (splice, MCC 0.24->0.87, delta +0.62) while "
"motif-local and contamination controls barely move (|delta|<=0.013), a selective "
"dissociation establishing position-discarding as a distinct, causal failure route. "
"(C) Removing the excess TATA-bearing negatives from the contaminated enhancer sets "
"collapses the negative-direction TATA motif-only AUROC toward 0.5 (cohn 0.37->0.44, "
"nt_enhancers 0.39->0.49) after dropping ~20-22% of negatives, whereas the clean "
"control (drosophila) loses no sequences and is unchanged, confirming the TATA/AT-rich "
"excess in the negatives is the causal driver of the spurious signal. Error bars are 95% bootstrap CIs (on MCC in A; on the plotted statistic in B and C); seed 42."
)
print("\n=== FIGURE CAPTION DRAFT ===\n" + CAP)
