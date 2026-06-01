#!/usr/bin/env python3
"""Depth 2 -- COMPOSITION-FRACTION decomposition of the enhancer classification signal.

Deepens the Upgrade-2 / Upgrade-5 finding ("cohn collapses hardest under composition
equalization, nt_enhancers retains residual signal, drosophila is the clean control") into
ONE interpretable number per dataset x classifier:

    composition_fraction = (AUROC_composition_only - 0.5) / (AUROC_full - 0.5)

i.e. the share of ABOVE-CHANCE enhancer-classification AUROC attributable to composition
(GC + 16 dinucleotide frequencies, the existing 20-d `comp_signature`) ALONE. AUROC has a
clean 0.5 chance floor, so the ratio is well-defined. It is applied IDENTICALLY across the
three datasets and both classifiers. Values can exceed 1 (composition beats the full model)
or go below 0 (full model below chance) -- we report whatever we measure.

Two classifiers, evaluated on the ORIGINAL split (`_common.load_original`):
  * "kmer": full = LightGBM on the standard k-mer spectrum at the VALIDATION-selected best k
            (read from results/val_selection.csv column `new_k`: cohn 5, nt_enhancers 4,
            drosophila 4); comp_only = LightGBM on `_common.comp_signature(seqs, ks=(1,2))`.
  * "fm":   full = LightGBM head on the cached frozen HyenaDNA MEAN-pooled embeddings
            (cache/fm_embeddings/emb__{task}__orig__{split}__hyenadna-tiny-16k-d128.npz,
            key "mean"; cached "y" is verified against _common.load_original); comp_only =
            the SAME composition LightGBM. The .npz is read with numpy only -- NO torch.

Method (per dataset x classifier): fit the full model and the comp-only model ONCE on train,
get test-set class-1 probabilities for both, then bootstrap 1000x: resample test indices
(np.random.RandomState(42), randint(0,n,n)); on EACH resample compute AUROC_full and
AUROC_comp (roc_auc_score on column 1; skip resamples with <2 classes -> nan) and the fraction;
report the point fraction (from the full-data AUROCs) and the 2.5/97.5 percentile CI of the
bootstrap fractions. CPU-only, seed 42. Writes results/upgrades/composition_fraction.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _common as C
import featurize

FM_DIR = os.path.join(C.ROOT, "cache", "fm_embeddings")
FM_TAG = "hyenadna-tiny-16k-d128"
VAL_SEL = os.path.join(C.ROOT, "results", "val_selection.csv")

DEFAULT = [("human_enhancers_cohn", "contaminated"),
           ("nt_enhancers", "contaminated"),
           ("drosophila_enhancers_stark", "clean control")]


# ------------------------------------------------------------------ helpers
def read_best_k(task, val_csv=VAL_SEL):
    """Validation-selected best k for `task` from results/val_selection.csv column new_k."""
    df = pd.read_csv(val_csv)
    row = df[df["dataset"] == task]
    if len(row) != 1:
        raise KeyError(f"{task} not uniquely in {val_csv} (found {len(row)} rows)")
    return int(row["new_k"].iloc[0])


def load_fm_mean(task, split):
    """(X_mean, y) from the cached frozen HyenaDNA MEAN-pooled embeddings (numpy only, no torch)."""
    p = os.path.join(FM_DIR, f"emb__{task}__orig__{split}__{FM_TAG}.npz")
    d = np.load(p, allow_pickle=True)
    return np.asarray(d["mean"], dtype=np.float32), np.asarray(d["y"], dtype=np.int64)


def _proba1(model_name, Xtr, ytr, Xte, seed):
    """Fit `model_name` once on (Xtr,ytr); return P(class==1) on Xte (binary tasks)."""
    nc = len(np.unique(ytr))
    m = C.gbmodels.build_model(model_name, seed, nc)
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)
    # locate the column for class label 1 (robust to class ordering)
    col = int(np.where(m.classes_ == 1)[0][0])
    return proba[:, col]


def _auroc1(y, p1):
    """Binary AUROC on class-1 scores; nan if <2 classes present."""
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, p1))
    except ValueError:
        return float("nan")


def paired_fraction(yte, p_full, p_comp, seed, boot):
    """Point composition_fraction (full-data) + percentile-bootstrap CI, paired on the SAME
    resample. Resampling unit = individual test sequence (rng.randint(0,n,n)), seed 42, B=boot.

    fraction = (AUROC_comp - 0.5) / (AUROC_full - 0.5)."""
    yte = np.asarray(yte, dtype=np.int64)
    full_pt = _auroc1(yte, p_full)
    comp_pt = _auroc1(yte, p_comp)
    denom = full_pt - 0.5
    frac_pt = (comp_pt - 0.5) / denom if denom != 0 else float("nan")

    n = len(yte)
    rng = np.random.RandomState(seed)
    af = np.full(boot, np.nan)            # full AUROC per resample
    ac = np.full(boot, np.nan)            # comp AUROC per resample
    fr = np.full(boot, np.nan)            # fraction per resample
    for b in range(boot):
        idx = rng.randint(0, n, n)
        yb = yte[idx]
        if len(np.unique(yb)) < 2:
            continue                      # leave nan
        fa = _auroc1(yb, p_full[idx])
        ca = _auroc1(yb, p_comp[idx])
        af[b] = fa
        ac[b] = ca
        d = fa - 0.5
        if d != 0:
            fr[b] = (ca - 0.5) / d
    return dict(
        full_auroc=full_pt, comp_auroc=comp_pt, frac=frac_pt,
        full_lo=float(np.nanpercentile(af, 2.5)), full_hi=float(np.nanpercentile(af, 97.5)),
        comp_lo=float(np.nanpercentile(ac, 2.5)), comp_hi=float(np.nanpercentile(ac, 97.5)),
        frac_lo=float(np.nanpercentile(fr, 2.5)), frac_hi=float(np.nanpercentile(fr, 97.5)),
        n_test=int(n))


def _row(dataset, role, classifier, res):
    return dict(
        dataset=dataset, role=role, classifier=classifier,
        full_auroc=round(res["full_auroc"], 6),
        full_auroc_ci=f"[{res['full_lo']:.4f},{res['full_hi']:.4f}]",
        comp_only_auroc=round(res["comp_auroc"], 6),
        comp_only_auroc_ci=f"[{res['comp_lo']:.4f},{res['comp_hi']:.4f}]",
        comp_fraction=round(res["frac"], 6),
        comp_fraction_lo=round(res["frac_lo"], 6),
        comp_fraction_hi=round(res["frac_hi"], 6),
        n_test=res["n_test"])


# ------------------------------------------------------------------ per dataset
def run_dataset(task, role, args):
    rows = []
    tr_seqs, ytr, te_seqs, yte = C.load_original(task)
    k = read_best_k(task)
    print(f"\n[{task}] role={role} best_k={k} | n_train={len(ytr)} n_test={len(yte)}")

    # composition-only features (shared by both classifiers; train/test, proper)
    Ctr = C.comp_signature(tr_seqs, ks=(1, 2))
    Cte = C.comp_signature(te_seqs, ks=(1, 2))
    p_comp = _proba1("lgbm", Ctr, ytr, Cte, args.seed)

    # ---- classifier "kmer": full = LightGBM on k-mer spectrum at best k ----
    Xtr = featurize.kmer_spectrum(tr_seqs, k)
    Xte = featurize.kmer_spectrum(te_seqs, k)
    p_full_kmer = _proba1("lgbm", Xtr, ytr, Xte, args.seed)
    res_k = paired_fraction(yte, p_full_kmer, p_comp, args.seed, args.boot)
    rows.append(_row(task, role, "kmer", res_k))
    print(f"   kmer  full AUROC={res_k['full_auroc']:.4f} comp AUROC={res_k['comp_auroc']:.4f}"
          f"  comp_fraction={res_k['frac']:.4f} [{res_k['frac_lo']:.4f},{res_k['frac_hi']:.4f}]")

    # ---- classifier "fm": full = LightGBM head on frozen HyenaDNA mean-pool embeddings ----
    Xtr_fm, ytr_fm = load_fm_mean(task, "train")
    Xte_fm, yte_fm = load_fm_mean(task, "test")
    # verify cached labels match the canonical split (order + values)
    assert np.array_equal(ytr_fm, ytr), f"{task}: cached FM train y != load_original y"
    assert np.array_equal(yte_fm, yte), f"{task}: cached FM test y != load_original y"
    p_full_fm = _proba1("lgbm", Xtr_fm, ytr_fm, Xte_fm, args.seed)
    res_f = paired_fraction(yte, p_full_fm, p_comp, args.seed, args.boot)
    rows.append(_row(task, role, "fm", res_f))
    print(f"   fm    full AUROC={res_f['full_auroc']:.4f} comp AUROC={res_f['comp_auroc']:.4f}"
          f"  comp_fraction={res_f['frac']:.4f} [{res_f['frac_lo']:.4f},{res_f['frac_hi']:.4f}]")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "composition_fraction.csv"))
    ap.add_argument("--boot", type=int, default=C.BOOT)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--datasets", default="")
    args = ap.parse_args()

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    tasks = (DEFAULT if not args.datasets
             else [(t, r) for t, r in DEFAULT if t in set(args.datasets.split(","))])

    print("=" * 100)
    print("DEPTH 2 -- COMPOSITION-FRACTION decomposition  (AUROC_comp_only-0.5)/(AUROC_full-0.5)")
    print(f"boot={args.boot} | seed={args.seed} | comp = GC + 16 dinuc (20-d comp_signature)")
    print("=" * 100)

    rows = []
    for task, role in tasks:
        rows.extend(run_dataset(task, role, args))

    cols = ["dataset", "role", "classifier", "full_auroc", "full_auroc_ci",
            "comp_only_auroc", "comp_only_auroc_ci", "comp_fraction", "comp_fraction_lo",
            "comp_fraction_hi", "n_test"]
    df = pd.DataFrame(rows)[cols]
    df.to_csv(args.out, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nWrote {args.out} ({len(df)} rows).")


if __name__ == "__main__":
    main()
