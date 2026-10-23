#!/usr/bin/env python3
"""What do the two redundancy-inflated tasks score once the leaked test sequences are removed?

`measure_homology.py` reports that 25.0% of the 400 `nt_enhancers` test sequences are verbatim
duplicates of a training sequence and that 40.6% of `human_nontata_promoters` test sequences have a
>=0.7 8-mer-Jaccard near-duplicate in training. It stops there. This module answers the obvious next
question directly: hold the model fixed, drop the leaked test sequences, and re-score.

Three evaluations per task, all with the SAME trained model (so any change comes from the test set):
  all            the released test split, which must reproduce Table 1;
  no_verbatim    test sequences with an exact (Jaccard = 1) training duplicate removed;
  no_near_dup    test sequences with a >=0.7 8-mer-Jaccard training neighbour removed.

This is the cheap, exact complement to a full homology-clean re-split: it cannot say what the task
would score if the split were rebuilt (the training set still contains the redundancy), but it needs
no clustering heuristic and no retraining, so nothing about it is approximate.

Reuses measure_homology.py's 8-mer presence matrices and exact batched Jaccard, the
validation-selected k recorded in results/val_selection.csv (the same source Table 1 uses), and the
standard 1000-resample bootstrap. Seed 42, CPU-only.

Usage:  python src/upgrades/homology/run_purged_eval.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (os.path.join(HERE, ".."), HERE):
    _d = os.path.abspath(_d)
    if _d not in sys.path:
        sys.path.insert(0, _d)

import _common as C                     # noqa: E402
import featurize                        # noqa: E402
import measure_homology as H            # noqa: E402
from sklearn.metrics import matthews_corrcoef, roc_auc_score   # noqa: E402

TASKS = ["nt_enhancers", "human_nontata_promoters"]
NEAR_DUP = 0.7


def boot_metrics(y, pred, proba, seed, B):
    """Point MCC/AUROC + 1000-resample percentile bootstrap CIs (test sequence = unit)."""
    rng = np.random.RandomState(seed)
    n = len(y)
    nc = proba.shape[1]

    def auroc(yy, pp):
        if len(np.unique(yy)) < 2:
            return np.nan
        return (roc_auc_score(yy, pp[:, 1]) if nc == 2
                else roc_auc_score(yy, pp, multi_class="ovr", average="macro"))

    mm = np.full(B, np.nan)
    aa = np.full(B, np.nan)
    for b in range(B):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        mm[b] = matthews_corrcoef(y[idx], pred[idx])
        aa[b] = auroc(y[idx], proba[idx])
    q = lambda a, p: round(float(np.nanpercentile(a, p)), 4)
    return dict(mcc=round(float(matthews_corrcoef(y, pred)), 4),
                mcc_lo=q(mm, 2.5), mcc_hi=q(mm, 97.5),
                auroc=round(float(auroc(y, proba)), 4),
                auroc_lo=q(aa, 2.5), auroc_hi=q(aa, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "homology_purged.csv"))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--boot", type=int, default=C.BOOT)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)

    # k comes from the authoritative validation-selection table (the same source as Table 1),
    # not from the optional feature cache, so this runs on a fresh checkout.
    kmap = pd.read_csv(os.path.join(C.ROOT, "results", "val_selection.csv")).set_index("dataset")
    rows = []
    print("=" * 96)
    print("PURGED-TEST EVALUATION -- same trained model, leaked test sequences removed")
    print("=" * 96)
    for task in a.tasks.split(","):
        k = int(kmap.loc[task, "new_k"])
        tr_seqs, ytr, te_seqs, yte = C.load_original(task, seed=a.seed)
        print(f"\n[{task}] best_k={k}  n_train={len(ytr)}  n_test={len(yte)}", flush=True)

        m = C.gbmodels.build_model("lgbm", a.seed, len(np.unique(ytr)))
        m.fit(featurize.kmer_spectrum(tr_seqs, k), ytr)
        proba = m.predict_proba(featurize.kmer_spectrum(te_seqs, k))
        pred = m.classes_[np.argmax(proba, axis=1)]

        jac = H.max_jaccard_to_train(H.presence_matrix(te_seqs), H.presence_matrix(tr_seqs))
        masks = [("all", np.ones(len(yte), bool)),
                 ("no_verbatim", jac < 0.999),
                 ("no_near_dup", jac < NEAR_DUP)]
        base = None
        for name, keep in masks:
            if keep.sum() < 10 or len(np.unique(yte[keep])) < 2:
                print(f"   {name:12s} skipped (n={int(keep.sum())})")
                continue
            r = boot_metrics(yte[keep], pred[keep], proba[keep], a.seed, a.boot)
            if base is None:
                base = r["mcc"]
            r.update(task=task, protocol=name, k=int(k), n_train=len(ytr),
                     n_test=int(keep.sum()), frac_test_kept=round(float(keep.mean()), 4),
                     removed_frac=round(float(1 - keep.mean()), 4),
                     delta_mcc_vs_all=round(r["mcc"] - base, 4),
                     pos_frac=round(float((yte[keep] == 1).mean()), 4))
            rows.append(r)
            print(f"   {name:12s} n={r['n_test']:6d} ({r['frac_test_kept']:.1%} kept)  "
                  f"MCC={r['mcc']:.3f} [{r['mcc_lo']:.3f},{r['mcc_hi']:.3f}]  "
                  f"AUROC={r['auroc']:.3f}  delta={r['delta_mcc_vs_all']:+.3f}", flush=True)

    cols = ["task", "protocol", "k", "n_train", "n_test", "frac_test_kept", "removed_frac",
            "pos_frac", "mcc", "mcc_lo", "mcc_hi", "auroc", "auroc_lo", "auroc_hi",
            "delta_mcc_vs_all"]
    df = pd.DataFrame(rows)[cols]
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
