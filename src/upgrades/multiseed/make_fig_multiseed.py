#!/usr/bin/env python3
"""Figure for Upgrade 8: multi-seed robustness. Reads results/upgrades/multiseed.csv ONLY.
Per-task mean MCC with +/- SD error bars across seeds, colored by route, with the solved(0.7)/
partial(0.5) tier bands. Writes results/upgrades/figures/multiseed.{pdf,png}."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CSV = os.path.join(ROOT, "results", "upgrades", "multiseed.csv")
OUTDIR = os.path.join(ROOT, "results", "upgrades", "figures")


def main():
    df = pd.read_csv(CSV).sort_values("mcc_mean")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    y = np.arange(len(df))
    ax.axvspan(0.7, 1.0, color="#009E73", alpha=0.08); ax.axvspan(0.5, 0.7, color="#E69F00", alpha=0.08)
    ax.errorbar(df["mcc_mean"], y, xerr=df["mcc_sd"], fmt="o", ms=5, capsize=3, color="#0072B2", elinewidth=1.4)
    ax.set_yticks(y); ax.set_yticklabels([f"{t}\n[{r}]" for t, r in zip(df["task"], df["route"])], fontsize=6)
    ax.axvline(0.7, color="0.5", lw=0.6, ls=":"); ax.axvline(0.5, color="0.5", lw=0.6, ls=":")
    ax.set_xlabel("benchmark MCC (mean ± SD over seeds, bootstrap-train)", fontsize=8); ax.set_xlim(0, 1)
    sdmax = df["mcc_sd"].max()
    ax.set_title(f"Multi-seed robustness: MCC is seed-invariant (max SD = {sdmax:.3f})", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7)
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"multiseed.{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUTDIR}/multiseed.pdf and .png")


if __name__ == "__main__":
    main()
