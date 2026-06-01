#!/usr/bin/env python3
"""run_cleaning.py -- make the contamination finding CAUSAL by cleaning, and release
the cleaned splits as a reusable artifact.

The negative sets of human_enhancers_cohn (+22.7% neg-TATA excess) and nt_enhancers
(+20.9%) are contaminated with TATA/TSS-core sequence. This intervenes:

  1. Flag contaminating negatives: negative-class sequences carrying a high-confidence
     TATA box (TBP PSSM >= 1.3 bits/pos -- the specificity-controlled threshold from
     the stats layer).
  2. Cleaned split = remove the EXCESS flagged negatives so the negative class's
     TATA-box rate matches the positive class's (a seeded random subset of flagged
     negatives is dropped). This eliminates the contamination while keeping
     background-rate TATA negatives; a dataset with no excess loses ~nothing. Removing
     ALL TATA-bearing negatives would be non-specific (TBP is low-information, so at
     1.3 bits it also flags AT-rich non-contaminant sequence -- e.g. ~94% of the
     AT-rich drosophila negatives), which is why we target the excess. Positives are
     left identical.
  3. Causal test: re-run k-mer+LightGBM on original vs cleaned, bootstrap CIs. The
     prediction is that the TATA motif-only AUROC moves to ~0.5 (the neg-direction
     artifact is gone); overall MCC/AUROC are reported honestly (the test set changes,
     so MCC is not a paired comparison -- the TATA-AUROC collapse is the causal readout).
  4. Control: identical procedure on drosophila_enhancers_stark (clean) should remove
     ~nothing and barely change -- proving the cleaning targets the artifact, not just
     'removing sequences helps'.
  5. Cleaned splits written to cleaned_splits/<task>_{train,test}.csv (sequence,label).

CPU-only; reuses motif_jaspar/motif_match (TATA scan), featurize/models, run_stats
bootstrap helpers. Seed 42.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

import data as gbdata
import featurize
import models
import kselect
import motif_jaspar
import motif_match
from run_stats import auc_counts, boot_ci

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = [("human_enhancers_cohn", "contaminated"),
           ("nt_enhancers", "contaminated"),
           ("drosophila_enhancers_stark", "clean control")]


def best_k(task, cache_dir, seed):
    # k is selected on a held-out validation split carved from TRAIN (kselect),
    # never on the test split. (Former gb_csv/nt_csv args are no longer needed.)
    return kselect.best_k_val(task, cache_dir, seed)


def tata_counts(seqs, tbp, thr):
    return motif_match.count_hits_batch(motif_match.encode_sequences(seqs), tbp["pssm"], thr)


def clean_indices(y, flagged, target_rate, rng):
    """Keep all positives; drop the excess flagged negatives so the negative flagged
    rate ~ target_rate. Returns kept indices (sorted) and #removed."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    neg_fl = neg[flagged[neg]]
    neg_un = neg[~flagged[neg]]
    keep_fl = int(round(target_rate * len(neg)))
    if len(neg_fl) <= keep_fl:
        keep_neg = neg
    else:
        sel = rng.permutation(len(neg_fl))[:keep_fl]
        keep_neg = np.concatenate([neg_un, neg_fl[sel]])
    keep = np.sort(np.concatenate([pos, keep_neg]))
    return keep, len(y) - len(keep)


def fit_eval_boot(tr_seqs, ytr, te_seqs, yte, k, seed, B, rng):
    Xtr = featurize.kmer_spectrum(tr_seqs, k)
    Xte = featurize.kmer_spectrum(te_seqs, k)
    nc = len(np.unique(ytr))
    m = models.build_model("lgbm", seed, nc)
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)
    pred = m.classes_[np.argmax(proba, axis=1)]
    mcc = matthews_corrcoef(yte, pred)
    auc = (roc_auc_score(yte, proba[:, 1]) if nc == 2 else
           roc_auc_score(yte, proba, multi_class="ovr", average="macro"))
    n = len(yte)
    mm = np.empty(B); aa = np.full(B, np.nan)
    for b in range(B):
        idx = rng.randint(0, n, n)
        if len(np.unique(yte[idx])) < 2:
            mm[b] = np.nan; continue
        mm[b] = matthews_corrcoef(yte[idx], pred[idx])
        try:
            aa[b] = (roc_auc_score(yte[idx], proba[idx, 1]) if nc == 2 else
                     roc_auc_score(yte[idx], proba[idx], multi_class="ovr", average="macro"))
        except ValueError:
            pass
    return (mcc, np.nanpercentile(mm, 2.5), np.nanpercentile(mm, 97.5),
            auc, np.nanpercentile(aa, 2.5), np.nanpercentile(aa, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gb-results", default=os.path.join(HERE, "results", "results.csv"))
    ap.add_argument("--nt-results", default=os.path.join(HERE, "results", "results_nt.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "contamination_cleaning.csv"))
    ap.add_argument("--splits-dir", default=os.path.join(HERE, "cleaned_splits"))
    ap.add_argument("--tata-bits", type=float, default=1.3)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets", default="")
    args = ap.parse_args()

    os.makedirs(args.splits_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    tbp = next(m for m in motif_jaspar.load_pssms("vertebrates") if m["name"].upper() == "TBP")
    tasks = DEFAULT
    if args.datasets:
        want = set(args.datasets.split(","))
        tasks = [(t, r) for t, r in DEFAULT if t in want]

    print("=" * 96)
    print("CONTAMINATION CLEANING -- remove excess TATA-bearing negatives; causal re-benchmark")
    print(f"TATA threshold {args.tata_bits} bits/pos | boot={args.boot} | seed={args.seed}")
    print("=" * 96)

    rows = []
    for task, role in tasks:
        k = best_k(task, os.path.join(HERE, "cache"), args.seed)
        d = gbdata.load_dataset(task, seed=args.seed)
        tr_seqs, ytr = d["train_seqs"], d["y_train"]
        te_seqs, yte = d["test_seqs"], d["y_test"]
        ctr = tata_counts(tr_seqs, tbp, args.tata_bits)
        cte = tata_counts(te_seqs, tbp, args.tata_bits)
        ftr, fte = ctr >= 1, cte >= 1
        pos_rate = (ftr[ytr == 1].sum() + fte[yte == 1].sum()) / max((ytr == 1).sum() + (yte == 1).sum(), 1)

        ktr, rem_tr = clean_indices(ytr, ftr, pos_rate, rng)
        kte, rem_te = clean_indices(yte, fte, pos_rate, rng)
        n_neg = int((ytr == 0).sum() + (yte == 0).sum())
        frac_neg_removed = (rem_tr + rem_te) / max(n_neg, 1)

        ctr_tr = [tr_seqs[i] for i in ktr]; cytr = ytr[ktr]
        cte_te = [te_seqs[i] for i in kte]; cyte = yte[kte]

        def tata_auc(ca, ya, cb, yb):
            cnt = np.concatenate([ca, cb]); yy = np.concatenate([ya, yb])
            pos = cnt[yy == 1]; neg = cnt[yy == 0]
            lo, hi = boot_ci(pos, neg, auc_counts, args.boot, rng)
            return auc_counts(pos, neg), lo, hi
        o_ta, o_ta_lo, o_ta_hi = tata_auc(ctr, ytr, cte, yte)
        c_ta, c_ta_lo, c_ta_hi = tata_auc(ctr[ktr], cytr, cte[kte], cyte)

        o_mcc, o_ml, o_mh, o_au, o_al, o_ah = fit_eval_boot(tr_seqs, ytr, te_seqs, yte, k, args.seed, args.boot, rng)
        c_mcc, c_ml, c_mh, c_au, c_al, c_ah = fit_eval_boot(ctr_tr, cytr, cte_te, cyte, k, args.seed, args.boot, rng)

        for split, seqs_s, y_s in [("train", ctr_tr, cytr), ("test", cte_te, cyte)]:
            pd.DataFrame({"sequence": seqs_s, "label": y_s}).to_csv(
                os.path.join(args.splits_dir, f"{task}_{split}.csv"), index=False)

        rows.append(dict(dataset=task, role=role, best_k=k, tata_bits=args.tata_bits,
            pos_TATA_rate=round(float(pos_rate), 4), frac_neg_removed=round(float(frac_neg_removed), 4),
            n_removed=int(rem_tr + rem_te), n_total_orig=len(ytr) + len(yte),
            n_total_cleaned=len(cytr) + len(cyte),
            orig_TATA_auroc=round(o_ta, 4), orig_TATA_ci=f"[{o_ta_lo:.3f},{o_ta_hi:.3f}]",
            cleaned_TATA_auroc=round(c_ta, 4), cleaned_TATA_ci=f"[{c_ta_lo:.3f},{c_ta_hi:.3f}]",
            orig_mcc=round(o_mcc, 4), orig_mcc_ci=f"[{o_ml:.3f},{o_mh:.3f}]",
            cleaned_mcc=round(c_mcc, 4), cleaned_mcc_ci=f"[{c_ml:.3f},{c_mh:.3f}]",
            orig_auroc=round(o_au, 4), orig_auroc_ci=f"[{o_al:.3f},{o_ah:.3f}]",
            cleaned_auroc=round(c_au, 4), cleaned_auroc_ci=f"[{c_al:.3f},{c_ah:.3f}]"))
        print(f"  {task:<30} [{role:<13}] k={k} | removed {frac_neg_removed:.1%} of negatives | "
              f"TATA AUROC {o_ta:.3f}->{c_ta:.3f} | MCC {o_mcc:.3f}->{c_mcc:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    report = build_report(df, args)
    print("\n" + report)
    with open(args.out.replace(".csv", "_interpretation.txt"), "w") as fh:
        fh.write(report)
    print(f"\nWrote {args.out}, cleaned_splits/<task>_{{train,test}}.csv, + interpretation.")


def build_report(df, args):
    L = []; add = L.append
    add("=" * 96)
    add("CONTAMINATION CLEANING -- causal test (remove excess TATA-bearing negatives)")
    add("=" * 96)
    hdr = (f"{'dataset':<30}{'role':<14}{'neg removed':>12}{'TATA AUROC orig->clean':>26}"
           f"{'MCC orig->clean':>20}")
    add(hdr); add("-" * len(hdr))
    for _, r in df.iterrows():
        add(f"{r['dataset']:<30}{r['role']:<14}{r['frac_neg_removed']:>11.1%} "
            f"{r['orig_TATA_auroc']:>10.3f} -> {r['cleaned_TATA_auroc']:<10.3f}"
            f"{r['orig_mcc']:>9.3f} -> {r['cleaned_mcc']:<8.3f}")
    add("")
    add("TATA AUROC = does TATA-hit-count separate the classes (pos>neg). <0.5 = neg-enriched")
    add("(the contamination artifact); ~0.5 after cleaning = artifact removed. Full CIs in the CSV.")
    add("\nVerdict:")
    for _, r in df.iterrows():
        toward = abs(r["cleaned_TATA_auroc"] - 0.5) < abs(r["orig_TATA_auroc"] - 0.5)
        if r["role"] == "contaminated":
            add(f"  {r['dataset']}: removed {r['frac_neg_removed']:.1%} of negatives; TATA AUROC "
                f"{r['orig_TATA_auroc']:.3f}->{r['cleaned_TATA_auroc']:.3f} "
                f"({'moved toward 0.5 -- artifact removed' if toward else 'did NOT move toward 0.5'}); "
                f"MCC {r['orig_mcc']:.3f}->{r['cleaned_mcc']:.3f}.")
        else:
            add(f"  {r['dataset']} (CONTROL): removed {r['frac_neg_removed']:.1%} of negatives; "
                f"TATA AUROC {r['orig_TATA_auroc']:.3f}->{r['cleaned_TATA_auroc']:.3f}, "
                f"MCC {r['orig_mcc']:.3f}->{r['cleaned_mcc']:.3f} -- should barely change.")
    add("\nReading: if the contaminated datasets' TATA AUROC collapses toward 0.5 after removing a")
    add("modest fraction of negatives, while the clean control loses ~nothing and barely moves, the")
    add("TATA enrichment WAS the contaminant (causal). MCC is reported straight (test set changes, so")
    add("it is not a paired comparison). Cleaned splits are in cleaned_splits/. Seed=%d." % args.seed)
    return "\n".join(L)


if __name__ == "__main__":
    main()
