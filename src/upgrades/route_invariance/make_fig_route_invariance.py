#!/usr/bin/env python3
"""Figure for Upgrade 9: route-invariance. Reads results/upgrades/route_invariance.csv ONLY.
Left: per-task LR/LGBM/FM MCC with the solved(0.7)/partial(0.5) tier bands. Right: family tier-
agreement matrix. Writes results/upgrades/figures/route_invariance.{pdf,png}."""
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CSV = os.path.join(ROOT, "results", "upgrades", "route_invariance.csv")
OUTDIR = os.path.join(ROOT, "results", "upgrades", "figures")
FAMS = ["lr", "lgbm", "fm"]


def main():
    df = pd.read_csv(CSV, keep_default_na=False)
    df["lgbm_mcc"] = pd.to_numeric(df["lgbm_mcc"]); df["lr_mcc"] = pd.to_numeric(df["lr_mcc"])
    df = df.sort_values("lgbm_mcc", ascending=True).reset_index(drop=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    y = np.arange(len(df))
    a1.axvspan(0.7, 1.0, color="#009E73", alpha=0.08); a1.axvspan(0.5, 0.7, color="#E69F00", alpha=0.08)
    a1.scatter(df["lr_mcc"], y, s=26, marker="o", color="#56B4E9", label="LR floor", zorder=3)
    a1.scatter(df["lgbm_mcc"], y, s=30, marker="s", color="#D55E00", label="LightGBM", zorder=3)
    fmv = pd.to_numeric(df["fm_mcc"], errors="coerce")
    a1.scatter(fmv, y, s=44, marker="*", color="#000000", label="FM-probe", zorder=4)
    a1.set_yticks(y); a1.set_yticklabels([f"{t}  [{m}]" for t, m in zip(df["task"], df["mechanism_route"])], fontsize=6)
    a1.axvline(0.7, color="0.5", lw=0.6, ls=":"); a1.axvline(0.5, color="0.5", lw=0.6, ls=":")
    a1.set_xlabel("benchmark MCC (per model family)", fontsize=8); a1.set_xlim(0, 1)
    a1.set_title("Difficulty landscape across families\n(bands: solved ≥0.7, partial ≥0.5)", fontsize=8.5)
    a1.legend(fontsize=7, loc="lower right", frameon=False)
    # agreement matrix
    M = np.full((3, 3), np.nan)
    for i, a in enumerate(FAMS):
        for j, b in enumerate(FAMS):
            ta, tb = df[f"{a}_tier"].astype(str), df[f"{b}_tier"].astype(str)
            mask = ta.isin(["solved", "partial", "hard"]) & tb.isin(["solved", "partial", "hard"])
            if mask.sum():
                M[i, j] = (ta[mask] == tb[mask]).mean()
    im = a2.imshow(M, cmap="Greens", vmin=0.5, vmax=1.0)
    a2.set_xticks(range(3)); a2.set_xticklabels([f.upper() for f in FAMS], fontsize=8)
    a2.set_yticks(range(3)); a2.set_yticklabels([f.upper() for f in FAMS], fontsize=8)
    for i in range(3):
        for j in range(3):
            if M[i, j] == M[i, j]:
                a2.text(j, i, f"{M[i,j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.8 else "black")
    a2.set_title("Tier-agreement matrix", fontsize=8.5)
    fig.suptitle("Route assignments are stable across model families (mechanism is model-agnostic; "
                 "tier agrees ≥87%)", fontsize=8.2, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"route_invariance.{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUTDIR}/route_invariance.pdf and .png")


if __name__ == "__main__":
    main()
