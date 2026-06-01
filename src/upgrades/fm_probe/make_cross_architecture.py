#!/usr/bin/env python3
"""Depth 1 -- cross-architecture table + figure. Reads results/upgrades/fm_paired.csv (HyenaDNA) and
results/upgrades/fm_paired_dnabert.csv (DNABERT-2) ONLY. Tests whether the composition-removal effect
(the locked-arm gc_match `test_effect_auroc`: the same frozen-FM head's AUROC loss when composition-
biased negatives are removed from the test) has the SAME SIGN and is SIGNIFICANT across BOTH pretrained
architectures on the short contaminated sets (cohn, nt_enhancers). HyenaDNA cell = [SEP]/gc_match;
DNABERT-2 cells = mean/gc_match and [CLS]/gc_match (its mean/[SEP] analogues). Both heads (lgbm, lr) are
reported so the cross-architecture result doubles as DNABERT-2's Depth-4 head-invariance. drosophila
(HyenaDNA only) is the flat-control reference. Writes results/upgrades/fm_cross_architecture.csv and
results/upgrades/figures/fm_cross_architecture.{pdf,png}."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES = os.path.join(ROOT, "results", "upgrades")
HYENA = os.path.join(RES, "fm_paired.csv")
DNABERT = os.path.join(RES, "fm_paired_dnabert.csv")
OUTCSV = os.path.join(RES, "fm_cross_architecture.csv")
OUTDIR = os.path.join(RES, "figures")


def _excl0(ci):
    lo, hi = [float(x) for x in str(ci).strip("[]").split(",")]
    return not (lo <= 0 <= hi)


def main():
    hy = pd.read_csv(HYENA)
    rows = []
    # HyenaDNA: locked cell sep/gc_match, both heads; + drosophila control reference
    sub = hy[(hy.pooling == "sep") & (hy.arm == "gc_match")]
    for _, r in sub.iterrows():
        rows.append(dict(architecture="HyenaDNA-tiny-16k", dataset=r["dataset"], role=r["role"],
                         pooling="sep", head=r["head"], arm="gc_match",
                         test_effect_auroc=r["test_effect_auroc"], ci=r["test_effect_auroc_ci"],
                         significant=_excl0(r["test_effect_auroc_ci"])))
    # DNABERT-2: mean + cls / gc_match, both heads (cohn + nt only)
    if os.path.exists(DNABERT):
        db = pd.read_csv(DNABERT)
        sub = db[(db.pooling.isin(["mean", "cls"])) & (db.arm == "gc_match")]
        for _, r in sub.iterrows():
            rows.append(dict(architecture="DNABERT-2-117M", dataset=r["dataset"], role=r["role"],
                             pooling=r["pooling"], head=r["head"], arm="gc_match",
                             test_effect_auroc=r["test_effect_auroc"], ci=r["test_effect_auroc_ci"],
                             significant=_excl0(r["test_effect_auroc_ci"])))
    df = pd.DataFrame(rows)
    df.to_csv(OUTCSV, index=False)

    # cross-architecture verdict (lgbm head; contaminated sets): same sign + significant on BOTH?
    prim = df[df["head"] == "lgbm"]
    contam = ["human_enhancers_cohn", "nt_enhancers"]
    verdict_lines = []
    for ds in contam:
        cells = prim[prim.dataset == ds]
        signs = {(r.architecture, r.pooling): (r.test_effect_auroc, r.significant) for _, r in cells.iterrows()}
        all_neg_sig = all(v[0] < 0 and v[1] for v in signs.values())
        verdict_lines.append(f"  {ds}: " + "; ".join(
            f"{a}/{p}={v[0]:+.3f}{'*' if v[1] else ''}" for (a, p), v in signs.items())
            + f"  -> {'consistent (all negative + significant)' if all_neg_sig else 'NOT all neg+sig (report straight)'}")

    # figure: test_effect_auroc by dataset, grouped by architecture/pooling (lgbm head)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    groups = [("HyenaDNA-tiny-16k", "sep", "#D55E00"), ("DNABERT-2-117M", "mean", "#0072B2"),
              ("DNABERT-2-117M", "cls", "#56B4E9")]
    datasets = ["human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark"]
    x = np.arange(len(datasets)); w = 0.25
    for gi, (arch, pool, col) in enumerate(groups):
        ys, los, his = [], [], []
        for ds in datasets:
            c = prim[(prim.architecture == arch) & (prim.pooling == pool) & (prim.dataset == ds)]
            if len(c):
                r = c.iloc[0]; lo, hi = [float(v) for v in str(r["ci"]).strip("[]").split(",")]
                ys.append(r["test_effect_auroc"]); los.append(r["test_effect_auroc"] - lo); his.append(hi - r["test_effect_auroc"])
            else:
                ys.append(np.nan); los.append(0); his.append(0)
        ax.bar(x + (gi - 1) * w, ys, w, yerr=[los, his], label=f"{arch.split('-')[0]} ({pool})",
               color=col, capsize=2, error_kw=dict(elinewidth=0.8))
    ax.axhline(0, color="0.4", lw=0.9, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([d.replace("_", "\n") for d in datasets], fontsize=7)
    ax.set_ylabel("test_effect AUROC (composition removed from test)", fontsize=8)
    ax.set_title("Cross-architecture: BOTH frozen FMs lose enhancer AUROC when composition is removed\n"
                 "(gc_match arm, LightGBM head; drosophila = HyenaDNA-only flat control)", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7)
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"fm_cross_architecture.{ext}"), dpi=300, bbox_inches="tight")

    print("CROSS-ARCHITECTURE test_effect AUROC (gc_match, lgbm head; * = CI excludes 0):")
    print("\n".join(verdict_lines))
    print(f"\nWrote {OUTCSV} ({len(df)} rows) + figures/fm_cross_architecture.{{pdf,png}}")


if __name__ == "__main__":
    main()
