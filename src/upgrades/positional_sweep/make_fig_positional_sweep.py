#!/usr/bin/env python3
"""Figure for Upgrade 4: graded positional rescue. Reads positional_sweep.csv + positional_Bsweep.csv
ONLY. Panel A: per-task delta-MCC from B=8 position bins (sorted), with 95% paired-bootstrap CIs,
colored by route. Panel B: resolution sweep B in {1,2,4,8,16} for the splice + TATA-promoter tasks.
Writes results/upgrades/figures/positional_sweep.{pdf,png}."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CSV = os.path.join(ROOT, "results", "upgrades", "positional_sweep.csv")
BCSV = os.path.join(ROOT, "results", "upgrades", "positional_Bsweep.csv")
OUTDIR = os.path.join(ROOT, "results", "upgrades", "figures")
RCOL = {"POSITIONAL": "#D55E00", "fixed-position promoter": "#E69F00", "non-positional (control)": "#0072B2"}


def _ci(s):
    lo, hi = str(s).strip("[]").split(","); return float(lo), float(hi)


def main():
    df = pd.read_csv(CSV).sort_values("delta_mcc")
    bdf = pd.read_csv(BCSV)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
    ys = range(len(df))
    los = [r["delta_mcc"] - _ci(r["delta_ci"])[0] for _, r in df.iterrows()]
    his = [_ci(r["delta_ci"])[1] - r["delta_mcc"] for _, r in df.iterrows()]
    a1.barh(list(ys), df["delta_mcc"], xerr=[los, his], color=[RCOL.get(r, "0.4") for r in df["route"]],
            error_kw=dict(elinewidth=0.9, capsize=2), height=0.7)
    a1.set_yticks(list(ys)); a1.set_yticklabels(df["task"], fontsize=6.5)
    a1.axvline(0, color="0.5", lw=0.8); a1.set_xlabel("delta-MCC (B=8 positional - global)", fontsize=8)
    a1.set_title("Per-task positional rescue", fontsize=9)
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=r) for r, c in RCOL.items()]
    a1.legend(handles=handles, fontsize=6.3, loc="lower right", frameon=False)
    for task, g in bdf.groupby("task"):
        a2.plot(g["B"], g["delta_vs_B1"], "-o", ms=4, label=task)
    a2.set_xscale("log", base=2); a2.set_xticks(sorted(bdf["B"].unique()))
    a2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a2.axhline(0, color="0.6", lw=0.7, ls="--")
    a2.set_xlabel("position bins B", fontsize=8); a2.set_ylabel("delta-MCC vs B=1", fontsize=8)
    a2.set_title("Graded in resolution", fontsize=9); a2.legend(fontsize=6.5, frameon=False)
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7)
    fig.suptitle("Position helps exactly where fixed-position signal exists (large: splice; modest: TATA-promoter; ~0: shared-motif)",
                 fontsize=8.2, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"positional_sweep.{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUTDIR}/positional_sweep.pdf and .png")


if __name__ == "__main__":
    main()
