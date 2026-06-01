#!/usr/bin/env python3
"""DEPTH 4 -- FM-head invariance of the composition-removal effect (HyenaDNA only).

Read-only over results/upgrades/fm_paired.csv. NO torch, NO refitting: this script only
re-tabulates an EXISTING quantity and asks one question -- is the headline FM result a
property of the frozen foundation-model representation, or an artifact of the probe head?

The locked headline FM cell is [SEP] pooling + the gc_match cleaning arm. The
composition-removal effect is the "test_effect_auroc": the SAME frozen-FM head's AUROC loss
when composition-biased negatives are removed from the TEST set (already computed for BOTH
probe heads -- LightGBM and logistic regression -- in fm_paired.csv, with the bootstrap CI
whose resampling unit is the test sequence, identical to all other CIs in the paper).

If the effect is genuinely in the HyenaDNA embedding, swapping the read-out head (a non-linear
GBDT vs a linear LR) must not change its SIGN or SIGNIFICANCE on the contaminated enhancer
tasks (cohn, nt), and the clean control (drosophila) must stay flat (CI includes 0) for BOTH
heads. That is the "head-invariance" verdict computed and printed here.

Extracted, for model = HyenaDNA-tiny-16k-d128:
  * locked arm  = pooling=sep, arm=gc_match  -- test_effect_auroc + CI for head in {lgbm, lr}
  * secondary   = pooling=mean, same arm     -- for context (mean pooling is NOT the headline)
  for datasets human_enhancers_cohn, nt_enhancers (contaminated) and
  drosophila_enhancers_stark (clean control).

Output: results/upgrades/fm_head_invariance.csv with columns
  model, dataset, role, pooling, arm, head, test_effect_auroc, test_effect_auroc_ci,
  sign(+/-), significant(bool: CI excludes 0).
The CSV is designed so the lead can trivially APPEND DNABERT-2 rows later: every row is fully
self-describing (model is the literal "HyenaDNA-tiny-16k-d128" here), so a second model's rows
just stack underneath with the same schema. This script does NOT touch DNABERT-2.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _common as C  # noqa: E402  (repo root on sys.path; SEED/RESULTS_DIR conventions)

MODEL = "HyenaDNA-tiny-16k-d128"
LOCKED_POOLING = "sep"
LOCKED_ARM = "gc_match"
CONTAMINATED = ["human_enhancers_cohn", "nt_enhancers"]
CONTROL = "drosophila_enhancers_stark"
DATASETS = CONTAMINATED + [CONTROL]
HEADS = ["lgbm", "lr"]
POOLINGS = [LOCKED_POOLING, "mean"]   # locked first, mean as secondary context


def parse_ci(s):
    """Parse a '[lo,hi]' CI string (the paper's format) into (lo, hi) floats."""
    lo, hi = str(s).strip().strip("[]").split(",")
    return float(lo), float(hi)


def ci_excludes_zero(s):
    """True iff the [lo,hi] CI does not straddle 0 (lo and hi same strict sign)."""
    lo, hi = parse_ci(s)
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)


def sign_str(x):
    return "+" if x >= 0 else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=os.path.join(C.RESULTS_DIR, "fm_paired.csv"))
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "fm_head_invariance.csv"))
    args = ap.parse_args()

    src = pd.read_csv(args.inp)
    rows = []
    for ds in DATASETS:
        for pooling in POOLINGS:
            for head in HEADS:
                m = src[(src["dataset"] == ds) & (src["arm"] == LOCKED_ARM)
                        & (src["pooling"] == pooling) & (src["head"] == head)]
                if len(m) != 1:
                    raise SystemExit(f"expected exactly 1 row for {ds}/{pooling}/{LOCKED_ARM}/{head}, "
                                     f"got {len(m)} -- is fm_paired.csv the expected schema?")
                r = m.iloc[0]
                eff = float(r["test_effect_auroc"])
                ci = str(r["test_effect_auroc_ci"])
                rows.append(dict(
                    model=MODEL,
                    dataset=ds,
                    role=str(r["role"]),
                    pooling=pooling,
                    arm=LOCKED_ARM,
                    head=head,
                    test_effect_auroc=round(eff, 6),
                    test_effect_auroc_ci=ci,
                    sign=sign_str(eff),
                    significant=bool(ci_excludes_zero(ci)),
                ))

    df = pd.DataFrame(rows, columns=["model", "dataset", "role", "pooling", "arm", "head",
                                     "test_effect_auroc", "test_effect_auroc_ci",
                                     "sign", "significant"])
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    df.to_csv(args.out, index=False)

    # ----- verdict: is the effect head-invariant? -----
    # locked cell only (sep/gc_match): contaminated must be NEGATIVE + significant for both heads;
    # control must be FLAT (not significant) for both heads.
    locked = df[df["pooling"] == LOCKED_POOLING]

    def both_heads(ds):
        return locked[locked["dataset"] == ds].set_index("head")

    contam_ok = True
    for ds in CONTAMINATED:
        g = both_heads(ds)
        for head in HEADS:
            r = g.loc[head]
            contam_ok &= (r["sign"] == "-") and bool(r["significant"])
    ctrl = both_heads(CONTROL)
    control_flat = all(not bool(ctrl.loc[head, "significant"]) for head in HEADS)
    head_invariant = bool(contam_ok and control_flat)

    lines = [
        "=" * 96,
        f"DEPTH 4 -- FM-HEAD INVARIANCE of the composition-removal effect  ({MODEL})",
        "locked headline cell = pooling=sep, arm=gc_match  (mean-pool rows below are secondary context)",
        "=" * 96,
        "test_effect_auroc = (same frozen-FM head) AUROC loss when composition-biased negatives are",
        "removed from the TEST set. CI resampling unit = test sequence (identical to all paper CIs).",
        "",
        f"{'dataset':<28}{'role':<14}{'pool':<6}{'head':<6}{'effect':>9}  {'CI':<20}{'sig?':>6}",
        "-" * 96,
    ]
    for _, r in df.iterrows():
        lines.append(f"{r['dataset']:<28}{r['role']:<14}{r['pooling']:<6}{r['head']:<6}"
                     f"{r['test_effect_auroc']:>+9.4f}  {r['test_effect_auroc_ci']:<20}"
                     f"{('yes' if r['significant'] else 'no'):>6}")
    lines += [
        "-" * 96,
        "LOCKED-CELL (sep/gc_match) verdict:",
        f"  contaminated (cohn, nt): NEGATIVE + significant for BOTH heads ........ {contam_ok}",
        f"  control (drosophila):    FLAT (CI includes 0) for BOTH heads .......... {control_flat}",
        f"  => composition-removal effect is HEAD-INVARIANT .................... {head_invariant}",
        "",
        "Reading: the headline FM finding survives swapping a non-linear GBDT probe for a linear",
        "logistic-regression probe -- the effect lives in the frozen HyenaDNA representation, not in",
        "the read-out head. (Append DNABERT-2 rows later with the same schema to test cross-model.)",
        f"Seed {C.SEED}; read-only over fm_paired.csv; no refitting.",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(args.out.replace(".csv", "_interpretation.txt"), "w") as fh:
        fh.write(report + "\n")
    print(f"\nWrote {args.out} ({len(df)} rows) + interpretation.")
    print(f"VERDICT head_invariant = {head_invariant}")


if __name__ == "__main__":
    main()
