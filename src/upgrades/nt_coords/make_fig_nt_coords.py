#!/usr/bin/env python3
"""Figure for Upgrade 6: nt_enhancers TSS coordinate analysis. Reads results/upgrades/nt_coords.csv
and localization/coordinate_analysis.csv (cohn) ONLY. Side-by-side: in BOTH datasets the TATA-flagged
negatives are TSS-DEPLETED relative to non-flagged negatives (0.46x / 0.55x), replicating the
AT-rich-not-core-promoter conclusion. Writes results/upgrades/figures/nt_coords.{pdf,png}."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
NT = os.path.join(ROOT, "results", "upgrades", "nt_coords.csv")
COHN = os.path.join(ROOT, "localization", "coordinate_analysis.csv")
OUTDIR = os.path.join(ROOT, "results", "upgrades", "figures")


def cohn_rates():
    d = pd.read_csv(COHN)
    neg = d[d["klass"] == "negative"]
    fl = neg[neg["flagged"] == True]["tss_overlap"].mean()
    nfl = neg[neg["flagged"] == False]["tss_overlap"].mean()
    return float(fl), float(nfl)


def main():
    nt = pd.read_csv(NT).iloc[0]
    cohn_fl, cohn_nfl = cohn_rates()
    data = {
        "human_enhancers_cohn\n(100% mapped)": (cohn_fl, cohn_nfl),
        f"nt_enhancers\n({nt['mapped_frac']:.0%} mapped, fuzzy)": (nt["flagged_neg_tss"], nt["nonflagged_neg_tss"]),
    }
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.arange(len(data)); w = 0.36
    fl = [v[0] for v in data.values()]; nfl = [v[1] for v in data.values()]
    ax.bar(x - w / 2, fl, w, label="TATA-flagged negatives", color="#D55E00")
    ax.bar(x + w / 2, nfl, w, label="non-flagged negatives", color="#0072B2")
    for i, (a, b) in enumerate(zip(fl, nfl)):
        ax.annotate(f"{a/b:.2f}×", (i, max(a, b) + 0.001), ha="center", fontsize=8, color="0.2")
    ax.set_xticks(x); ax.set_xticklabels(list(data.keys()), fontsize=7.5)
    ax.set_ylabel("fraction overlapping a GENCODE TSS (±500 bp)", fontsize=8.5)
    ax.set_title("TATA-flagged negatives are TSS-DEPLETED in BOTH datasets\n"
                 "(refutes core-promoter leakage → AT-rich compositional bias)", fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"nt_coords.{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUTDIR}/nt_coords.pdf and .png")


if __name__ == "__main__":
    main()
