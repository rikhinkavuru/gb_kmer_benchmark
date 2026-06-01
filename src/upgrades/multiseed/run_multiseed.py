#!/usr/bin/env python3
"""Upgrade 8 -- multi-seed robustness of the LightGBM benchmark + route assignment.

The cleaning/discriminability subsets are already seed-swept; this wraps the LightGBM TRAINING itself
in a >=5-seed loop on the key tasks and confirms the benchmark MCC -- and hence the coarse route tier
(solved / hard) -- is seed-invariant.

Because LightGBM with subsample=colsample=1.0 is near-deterministic given a fixed training set (the
random_state only breaks ties), genuine training variation is injected by BOOTSTRAP-RESAMPLING the
training set per seed (resample train rows with replacement, seed-controlled) and fitting with
random_state=seed; the model is evaluated on the FIXED test set. The SD across seeds is therefore a
real training-robustness estimate, not ~0. k is the validation-selected k (kselect, seed 42), held
fixed across seeds. Reuses the cached k-mer matrices. CPU-only. Writes results/upgrades/multiseed.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import matthews_corrcoef, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _common as C
import models as gbmodels
import kselect

# representative key tasks across the routes (giants regulatory/enhancers_ensembl excluded for speed)
KEY = ["human_nontata_promoters", "nt_promoter_tata", "demo_human_or_worm",
       "human_enhancers_cohn", "nt_enhancers", "human_ocr_ensembl",
       "drosophila_enhancers_stark", "nt_splice_sites_all"]
ROUTE = {"human_nontata_promoters": "solved (promoter)", "nt_promoter_tata": "solved (TATA-promoter)",
         "demo_human_or_worm": "solved (composition)", "human_enhancers_cohn": "hard (contamination)",
         "nt_enhancers": "hard (contamination)", "human_ocr_ensembl": "hard (shared-motif)",
         "drosophila_enhancers_stark": "hard (control)", "nt_splice_sites_all": "hard (positional)"}


def tier(mcc):
    return "solved" if mcc >= 0.7 else ("partial" if mcc >= 0.5 else "hard")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "multiseed.csv"))
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--tasks", default=",".join(KEY))
    args = ap.parse_args()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    cache = os.path.join(C.ROOT, "cache")
    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = [t for t in args.tasks.split(",") if t]

    print("=" * 92)
    print(f"UPGRADE 8 -- MULTI-SEED ROBUSTNESS (bootstrap-train) | seeds={seeds}")
    print("=" * 92)
    rows = []
    for task in tasks:
        k = kselect.best_k_val(task, cache, 42)            # canonical k, fixed across seeds
        Xtr = sparse.load_npz(os.path.join(cache, f"{task}__train__k{k}.npz"))
        ytr = np.load(os.path.join(cache, f"{task}__train__y.npy"))
        Xte = sparse.load_npz(os.path.join(cache, f"{task}__test__k{k}.npz"))
        yte = np.load(os.path.join(cache, f"{task}__test__y.npy"))
        nc = len(np.unique(ytr)); n = len(ytr)
        mccs, aucs, tiers = [], [], []
        for seed in seeds:
            rng = np.random.RandomState(seed)
            idx = rng.randint(0, n, n)                     # bootstrap-resample the training set
            # guard: ensure all classes present in the resample
            if len(np.unique(ytr[idx])) < nc:
                idx = np.concatenate([idx, [np.where(ytr == c)[0][0] for c in range(nc)]])
            m = gbmodels.build_model("lgbm", seed, nc)
            m.fit(Xtr[idx], ytr[idx])
            proba = m.predict_proba(Xte); pred = m.classes_[np.argmax(proba, axis=1)]
            mcc = float(matthews_corrcoef(yte, pred))
            auc = float(roc_auc_score(yte, proba[:, 1]) if nc == 2 else
                        roc_auc_score(yte, proba, multi_class="ovr", average="macro"))
            mccs.append(mcc); aucs.append(auc); tiers.append(tier(mcc))
        mccs = np.array(mccs); aucs = np.array(aucs)
        tier_stable = len(set(tiers)) == 1
        rows.append(dict(task=task, route=ROUTE.get(task, ""), k=k, n_seeds=len(seeds),
            mcc_mean=round(float(mccs.mean()), 4), mcc_sd=round(float(mccs.std(ddof=1)), 4),
            mcc_min=round(float(mccs.min()), 4), mcc_max=round(float(mccs.max()), 4),
            auroc_mean=round(float(aucs.mean()), 4), auroc_sd=round(float(aucs.std(ddof=1)), 4),
            tier=tiers[0] if tier_stable else "VARIES", tier_stable=tier_stable))
        print(f"  {task:<30} [{ROUTE.get(task,''):<22}] k={k} | MCC {mccs.mean():.3f} +/- {mccs.std(ddof=1):.3f} "
              f"(min {mccs.min():.3f}, max {mccs.max():.3f}) | tier={'STABLE:'+tiers[0] if tier_stable else 'VARIES'}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    rep = ["=" * 92, "MULTI-SEED ROBUSTNESS (bootstrap-train; LightGBM)", "=" * 92,
           f"{'task':<30}{'route':<24}{'MCC mean+/-SD':>16}{'[min,max]':>16}  tier"]
    rep.append("-" * 92)
    for _, r in df.iterrows():
        rep.append(f"{r['task']:<30}{r['route']:<24}{r['mcc_mean']:>8.3f}+/-{r['mcc_sd']:<6.3f}"
                   f"[{r['mcc_min']:.3f},{r['mcc_max']:.3f}]  {'STABLE:'+r['tier'] if r['tier_stable'] else 'VARIES'}")
    n_stable = int(df["tier_stable"].sum())
    rep += ["", f"Tier (solved>=0.7 / partial>=0.5 / hard) stable across {len(seeds)} seeds for "
            f"{n_stable}/{len(df)} tasks; max MCC SD = {df['mcc_sd'].max():.3f}.",
            "Reading: small SD and a fully-stable tier assignment mean the benchmark numbers and the",
            "solved-vs-hard route classification are seed-invariant; training randomness (bootstrap",
            "resampling) does not move tasks across routes. Seeds %s." % seeds]
    rep = "\n".join(rep)
    print("\n" + rep)
    with open(args.out.replace(".csv", "_interpretation.txt"), "w") as fh:
        fh.write(rep)
    print(f"\nWrote {args.out} ({len(df)} rows) + interpretation.")


if __name__ == "__main__":
    main()
