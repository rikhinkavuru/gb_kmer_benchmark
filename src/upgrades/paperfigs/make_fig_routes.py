#!/usr/bin/env python3
"""Schematic of the three failure routes (paper Figure 1).

A diagram, not a plot -- but every number annotated on it is READ FROM the result CSVs
(no literals), so the figure cannot drift from the tables:

  shared-motif   results/motif_discriminability.csv  (human_ocr_ensembl, E-box TFAP4)
  positional     results/upgrades/positional_sweep.csv (nt_splice_sites_all, B=8)
  contamination  results/contamination_analysis.csv + results/upgrades/composition_clean.csv
                 (human_enhancers_cohn TATA-box fractions; k-mer MCC original vs equalized)

Sizing: drawn at 5.4 x 2.15 in and included at \\textwidth (5.6 in), i.e. essentially 1:1,
so every label below is >= 8 pt on the printed page.

Writes results/upgrades/figures/routes_schematic.{pdf,png}.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
R = os.path.join(ROOT, "results")
OUTDIR = os.path.join(R, "upgrades", "figures")

POS_C = "#0072B2"       # positive class (Okabe-Ito blue), consistent with the other figures
NEG_C = "#7f7f7f"       # negative class
ROUTE_C = {"shared-motif": "#E69F00", "positional": "#009E73", "contamination": "#D55E00"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _numbers():
    """Pull every annotated value out of the result CSVs."""
    md = pd.read_csv(os.path.join(R, "motif_discriminability.csv"))
    ocr = md[(md.dataset == "human_ocr_ensembl") & (md.TF == "TFAP4")].iloc[0]
    ps = pd.read_csv(os.path.join(R, "upgrades", "positional_sweep.csv"))
    spl = ps[ps.task == "nt_splice_sites_all"].iloc[0]
    ca = pd.read_csv(os.path.join(R, "contamination_analysis.csv"))
    tata = ca[(ca.dataset == "human_enhancers_cohn") &
              (ca.measure == "TATA_box_fraction(>=1)")].iloc[0]
    cc = pd.read_csv(os.path.join(R, "upgrades", "composition_clean.csv"))
    cc = cc[cc.dataset == "human_enhancers_cohn"].set_index("arm")
    return dict(
        ocr_auroc=float(ocr.motif_auroc), ocr_pos=float(ocr.pos_mean_hits),
        ocr_neg=float(ocr.neg_mean_hits),
        spl_lo=float(spl.spectrum_mcc), spl_hi=float(spl.positional_mcc),
        tata_pos=float(tata.pos_mean), tata_neg=float(tata.neg_mean),
        cohn_lo=float(cc.loc["comp_equalized", "kmer_mcc"]),
        cohn_hi=float(cc.loc["original", "kmer_mcc"]),
    )


LANE_X0, LANE_X1 = 0.105, 0.700


def _lane(ax, y, label, color, marks, mark_color, hatch=None, tag=None):
    """One class as a sequence lane with motif tick marks."""
    ax.add_patch(Rectangle((LANE_X0, y), LANE_X1 - LANE_X0, 0.10, facecolor=color,
                           alpha=0.18, edgecolor=color, lw=0.7, hatch=hatch))
    ax.text(LANE_X0 - 0.02, y + 0.05, label, ha="right", va="center", fontsize=8,
            color=color)
    for f in marks:
        x = LANE_X0 + f * (LANE_X1 - LANE_X0)
        ax.plot([x, x], [y + 0.011, y + 0.089], color=mark_color, lw=1.7,
                solid_capstyle="butt")
    if tag:
        ax.text(LANE_X1 + 0.025, y + 0.05, tag, ha="left", va="center", fontsize=7.8,
                color=mark_color, style="italic")


def _panel(ax, title, route, pos_marks, neg_marks, diagnosis, readout,
           neg_hatch=None, pos_tag=None, neg_tag=None):
    c = ROUTE_C[route]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.99, title, fontsize=9, fontweight="bold", color=c, va="top")
    _lane(ax, 0.665, "pos", POS_C, pos_marks, c, tag=pos_tag)
    _lane(ax, 0.505, "neg", NEG_C, neg_marks, c, hatch=neg_hatch, tag=neg_tag)
    ax.text(0.0, 0.435, diagnosis, fontsize=8, va="top", color="0.15", linespacing=1.35)
    ax.add_patch(FancyArrowPatch((0.015, 0.20), (0.115, 0.20), color=c, lw=1.0,
                                 arrowstyle="-|>", mutation_scale=7))
    ax.text(0.135, 0.20, "intervention", fontsize=8, color=c, va="center")
    ax.text(0.0, 0.135, readout, fontsize=8, va="top", color="0.15", linespacing=1.35)


def main():
    n = _numbers()
    fig, axes = plt.subplots(1, 3, figsize=(5.55, 1.82))
    fig.subplots_adjust(left=0.045, right=0.995, top=0.985, bottom=0.015, wspace=0.17)

    # --- Route 1: shared motif -- same motif, same density, both classes -------------
    _panel(axes[0], "1  Shared motif", "shared-motif",
           [0.11, 0.34, 0.61, 0.88], [0.19, 0.42, 0.68, 0.95],
           "the same motif appears in\n$\\it{both}$ classes: no separation",
           f"motif-only AUROC {n['ocr_auroc']:.2f}\n(E-box, human_ocr_ensembl)")

    # --- Route 2: positional -- same count, but positives align at a fixed offset -----
    _panel(axes[1], "2  Positional", "positional",
           [0.46, 0.50, 0.54, 0.58], [0.08, 0.35, 0.66, 0.93],
           "same motif count: the signal\nis $\\it{where}$, not $\\it{what}$",
           f"position-binned $k$-mers,\nMCC {n['spl_lo']:.2f} $\\rightarrow$ {n['spl_hi']:.2f} (splice)",
           pos_tag="fixed", neg_tag="scattered")

    # --- Route 3: contamination -- negatives are a different compositional draw -------
    _panel(axes[2], "3  Contamination", "contamination",
           [0.28, 0.79], [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95],
           "the $\\it{negatives}$ are an\nAT-rich compositional draw",
           f"equalize composition,\nMCC {n['cohn_hi']:.2f} $\\rightarrow$ {n['cohn_lo']:.2f} (cohn)",
           neg_hatch="////", pos_tag="GC-rich", neg_tag="AT-rich")

    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"routes_schematic.{ext}"), dpi=400)
    print(f"wrote {OUTDIR}/routes_schematic.pdf and .png")
    print("annotated values:", {k: round(v, 4) for k, v in n.items()})


if __name__ == "__main__":
    main()
