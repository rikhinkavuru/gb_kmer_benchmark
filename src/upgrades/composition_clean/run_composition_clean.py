#!/usr/bin/env python3
"""Upgrade 2 -- FULLY COMPOSITION-EQUALIZED cleaning of the enhancer negative sets.

Closes the paper's limitation (iii): the existing cleanings remove only a TATA-flagged
subset (run_cleaning.py) or match a single GC histogram (localization/jackknife_and_gcmatch.py).
Here we resample the NEGATIVE set (positives left identical, seed 42) to match the POSITIVE
class's JOINT distribution over GC + all 16 dinucleotide frequencies, via propensity-score
stratified resampling:

  1. Composition signature per sequence = [4 mononucleotide + 16 dinucleotide] L1-normalized
     frequencies (GC is implied by the mononucleotide block). Reuses featurize.kmer_spectrum.
  2. Fit ONE logistic-regression propensity model p = P(positive | composition) on the pooled
     (train+test) sequences (seed 42). By Rosenbaum-Rubin, matching on this scalar score
     balances every covariate that entered it (GC + all dinucleotides), in expectation.
  3. Within each split (train, test) separately: bin the propensity score and keep negatives
     in numbers proportional to the positive histogram (exact-shape match over supported bins),
     so the kept-negative composition distribution matches the positives'. Report retained frac.
  4. Decisive readouts on the equalized split: neg-direction TATA motif-only AUROC (-> ~0.5),
     GC AUROC (-> ~0.5), the k-mer+LightGBM benchmark MCC, and a COMPOSITION-ONLY BASELINE
     (LightGBM/LR on the 20-d composition features alone). "Residual enhancer signal" is then
     quantified as benchmark MCC minus the composition-only baseline MCC, not asserted.
  5. Balance is VERIFIED empirically (post-match GC AUROC and per-feature standardized mean
     differences), since propensity balancing is only asymptotic.

Comparison arms kept as-is: ORIGINAL, TATA-flag-cleaned (loaded from cleaned_splits/), and
25-bin GC-matched (re-derived with the existing histogram-match logic). Control dataset
drosophila_enhancers_stark should barely change on every arm.

CAVEAT (reproduced in the README + interpretation): equalized cleaning changes the test set,
so MCC deltas are UNPAIRED; the TATA/GC AUROC collapse is the causal readout, not the MCC.

CPU-only, seed 42, 1000-resample percentile bootstrap. No torch. Writes:
  results/upgrades/composition_clean.csv          (one row per dataset x arm)
  results/upgrades/composition_clean_balance.csv  (SMD balance diagnostics, comp arm)
  cleaned_splits_v2/<task>_{comp_equalized,gcmatch}_{train,test}.csv
  results/upgrades/composition_clean_interpretation.txt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _common as C
import kselect

DEFAULT = [("human_enhancers_cohn", "contaminated"),
           ("nt_enhancers", "contaminated"),
           ("drosophila_enhancers_stark", "clean control")]


# ----------------------------------------------------------- matching core
def fit_propensity(comp, y, seed):
    """LR propensity p=P(positive|composition) on pooled composition features (seed 42)."""
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=1000, random_state=seed))
    lr.fit(comp, y)
    return lr


def match_proportional(p_pos, p_neg, n_bins, rng):
    """Keep negatives so their propensity histogram is proportional to the positives'
    (exact-shape match over supported bins). Returns kept-negative indices (into the neg
    array) and the per-bin retained counts. Positives are never dropped."""
    lo = float(min(p_pos.min(), p_neg.min()))
    hi = float(max(p_pos.max(), p_neg.max()))
    edges = np.linspace(lo, hi, n_bins + 1)
    pb = np.clip(np.digitize(p_pos, edges) - 1, 0, n_bins - 1)
    nb = np.clip(np.digitize(p_neg, edges) - 1, 0, n_bins - 1)
    npos = np.array([(pb == b).sum() for b in range(n_bins)])
    avail = [np.where(nb == b)[0] for b in range(n_bins)]
    ratios = [len(avail[b]) / npos[b] for b in range(n_bins)
              if npos[b] > 0 and len(avail[b]) > 0]
    f = min(ratios) if ratios else 0.0          # largest proportional factor that fits
    keep = []
    for b in range(n_bins):
        if npos[b] == 0:
            continue
        take = min(len(avail[b]), int(round(f * npos[b])))
        if take > 0:
            keep.extend(avail[b][rng.permutation(len(avail[b]))[:take]])
    return np.array(sorted(keep), dtype=int)


def gc_match(pos_gc, neg_gc, n_bins, rng):
    """25-bin GC histogram match (per-bin min), the existing localization/ convention."""
    edges = np.linspace(0, 1, n_bins + 1)
    pb = np.clip(np.digitize(pos_gc, edges) - 1, 0, n_bins - 1)
    nb = np.clip(np.digitize(neg_gc, edges) - 1, 0, n_bins - 1)
    keep = []
    for b in range(n_bins):
        pn = int((pb == b).sum())
        av = np.where(nb == b)[0]
        if pn == 0 or len(av) == 0:
            continue
        take = min(len(av), pn)
        keep.extend(av[rng.permutation(len(av))[:take]])
    return np.array(sorted(keep), dtype=int)


def smd(pos_feat, neg_feat):
    """Per-feature standardized mean difference (pos vs neg)."""
    mp, mn = pos_feat.mean(0), neg_feat.mean(0)
    sd = np.sqrt((pos_feat.var(0) + neg_feat.var(0)) / 2.0)
    sd = np.where(sd < 1e-12, np.nan, sd)
    return (mp - mn) / sd


# --------------------------------------------------------- split assembly
def negatives_kept_split(seqs, y, keep_neg_local):
    """Build (kept_seqs, kept_y) keeping ALL positives + the selected negatives."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    kept_neg = neg[keep_neg_local]
    idx = np.sort(np.concatenate([pos, kept_neg]))
    return [seqs[i] for i in idx], y[idx]


def auroc_ci(pos_vals, neg_vals, aucfn, boot, rng):
    pt = aucfn(pos_vals, neg_vals)
    lo, hi = C.boot_ci(np.asarray(pos_vals), np.asarray(neg_vals), aucfn, boot, rng)
    return pt, lo, hi


def arm_metrics(tag, tr_seqs, ytr, te_seqs, yte, k, tbp, args, rng):
    """Compute the full metric block for one cleaning arm on one dataset."""
    # composition (train+test) for GC + comp-baseline
    gc_tr, gc_te = C.gc_content(tr_seqs), C.gc_content(te_seqs)
    tata_tr = C.tata_hit_counts(tr_seqs, tbp, args.tata_bits)
    tata_te = C.tata_hit_counts(te_seqs, tbp, args.tata_bits)
    # pooled pos/neg arrays for the discriminability AUROCs
    gc_pos = np.concatenate([gc_tr[ytr == 1], gc_te[yte == 1]])
    gc_neg = np.concatenate([gc_tr[ytr == 0], gc_te[yte == 0]])
    ta_pos = np.concatenate([tata_tr[ytr == 1], tata_te[yte == 1]]).astype(np.int64)
    ta_neg = np.concatenate([tata_tr[ytr == 0], tata_te[yte == 0]]).astype(np.int64)
    gc_a, gc_lo, gc_hi = auroc_ci(gc_pos, gc_neg, C.auc_float, args.boot, rng)
    ta_a, ta_lo, ta_hi = auroc_ci(ta_pos, ta_neg, C.auc_counts, args.boot, rng)

    # composition-only baseline (LGBM + LR) on the 20-d signature, proper train/test
    Ctr = C.comp_signature(tr_seqs, ks=(1, 2))
    Cte = C.comp_signature(te_seqs, ks=(1, 2))
    comp_lgbm = C.fit_eval_boot(Ctr, ytr, Cte, yte, "lgbm", args.seed, args.boot, rng)
    comp_lr = C.fit_eval_boot(Ctr, ytr, Cte, yte, "lr", args.seed, args.boot, rng)

    # k-mer benchmark (LGBM at best k)
    bench = C.kmer_fit_eval_boot(tr_seqs, ytr, te_seqs, yte, k, "lgbm", args.seed, args.boot, rng)

    n_pos = int((ytr == 1).sum() + (yte == 1).sum())
    n_neg = int((ytr == 0).sum() + (yte == 0).sum())
    return dict(arm=tag, best_k=k, n_total=n_pos + n_neg, n_pos=n_pos, n_neg=n_neg,
        tata_auroc=round(ta_a, 4), tata_ci=f"[{ta_lo:.3f},{ta_hi:.3f}]",
        gc_auroc=round(gc_a, 4), gc_ci=f"[{gc_lo:.3f},{gc_hi:.3f}]",
        comp_only_mcc=comp_lgbm["mcc"], comp_only_mcc_ci=f"[{comp_lgbm['mcc_lo']:.3f},{comp_lgbm['mcc_hi']:.3f}]",
        comp_only_auroc=comp_lgbm["auroc"], comp_only_auroc_ci=f"[{comp_lgbm['auroc_lo']:.3f},{comp_lgbm['auroc_hi']:.3f}]",
        comp_only_lr_mcc=comp_lr["mcc"], comp_only_lr_auroc=comp_lr["auroc"],
        kmer_mcc=bench["mcc"], kmer_mcc_ci=f"[{bench['mcc_lo']:.3f},{bench['mcc_hi']:.3f}]",
        kmer_auroc=bench["auroc"], kmer_auroc_ci=f"[{bench['auroc_lo']:.3f},{bench['auroc_hi']:.3f}]",
        residual_mcc=round(bench["mcc"] - comp_lgbm["mcc"], 4), n_test=bench["n_test"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "composition_clean.csv"))
    ap.add_argument("--balance-out", default=os.path.join(C.RESULTS_DIR, "composition_clean_balance.csv"))
    ap.add_argument("--splits-dir", default=C.SPLITS_V2_DIR)
    ap.add_argument("--tata-flag-dir", default=os.path.join(C.ROOT, "cleaned_splits"))
    ap.add_argument("--n-bins", type=int, default=20, help="propensity histogram bins (comp arm)")
    ap.add_argument("--gc-bins", type=int, default=25, help="GC histogram bins (gc_match arm)")
    ap.add_argument("--tata-bits", type=float, default=1.3)
    ap.add_argument("--boot", type=int, default=C.BOOT)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--datasets", default="")
    args = ap.parse_args()

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    os.makedirs(args.splits_dir, exist_ok=True)
    tbp = C.load_tbp()
    tasks = DEFAULT if not args.datasets else [(t, r) for t, r in DEFAULT if t in set(args.datasets.split(","))]

    print("=" * 100)
    print("UPGRADE 2 -- COMPOSITION-EQUALIZED CLEANING (GC + 16 dinucleotide joint match)")
    print(f"propensity bins={args.n_bins} | gc bins={args.gc_bins} | tata={args.tata_bits} bits | "
          f"boot={args.boot} | seed={args.seed}")
    print("=" * 100)

    rows, balance_rows = [], []
    for task, role in tasks:
        rng = np.random.RandomState(args.seed)          # per-dataset deterministic stream
        k = kselect.best_k_val(task, os.path.join(C.ROOT, "cache"), args.seed)
        tr_seqs, ytr, te_seqs, yte = C.load_original(task)
        print(f"\n[{task}] role={role} best_k={k} | n_train={len(ytr)} n_test={len(yte)}")

        # ---- ORIGINAL arm ----
        r = arm_metrics("original", tr_seqs, ytr, te_seqs, yte, k, tbp, args, rng)
        r.update(dataset=task, role=role, retained_neg_frac=1.0); rows.append(r)
        print(f"   original     TATA={r['tata_auroc']:.3f} GC={r['gc_auroc']:.3f} "
              f"kmerMCC={r['kmer_mcc']:.3f} compMCC={r['comp_only_mcc']:.3f}")

        # ---- composition signatures + propensity (pooled) ----
        comp_tr = C.comp_signature(tr_seqs, ks=(1, 2))
        comp_te = C.comp_signature(te_seqs, ks=(1, 2))
        comp_all = np.vstack([comp_tr, comp_te])
        y_all = np.concatenate([ytr, yte])
        prop = fit_propensity(comp_all, y_all, args.seed)
        p_tr = prop.predict_proba(comp_tr)[:, 1]
        p_te = prop.predict_proba(comp_te)[:, 1]

        # ---- COMPOSITION-EQUALIZED arm (propensity match per split) ----
        keep_tr = match_proportional(p_tr[ytr == 1], p_tr[ytr == 0], args.n_bins, rng)
        keep_te = match_proportional(p_te[yte == 1], p_te[yte == 0], args.n_bins, rng)
        ce_tr_seqs, ce_ytr = negatives_kept_split(tr_seqs, ytr, keep_tr)
        ce_te_seqs, ce_yte = negatives_kept_split(te_seqs, yte, keep_te)
        n_neg_orig = int((ytr == 0).sum() + (yte == 0).sum())
        retained = (len(keep_tr) + len(keep_te)) / max(n_neg_orig, 1)
        r = arm_metrics("comp_equalized", ce_tr_seqs, ce_ytr, ce_te_seqs, ce_yte, k, tbp, args, rng)
        r.update(dataset=task, role=role, retained_neg_frac=round(float(retained), 4)); rows.append(r)
        _write_split(args.splits_dir, task, "comp_equalized", ce_tr_seqs, ce_ytr, ce_te_seqs, ce_yte)
        print(f"   comp_equal   TATA={r['tata_auroc']:.3f} GC={r['gc_auroc']:.3f} "
              f"kmerMCC={r['kmer_mcc']:.3f} compMCC={r['comp_only_mcc']:.3f} | kept {retained:.1%} of negatives")

        # balance diagnostics (comp arm): SMD on the 20-d signature, before vs after
        comp_neg_tr = comp_tr[ytr == 0]; comp_pos_tr = comp_tr[ytr == 1]
        comp_neg_te = comp_te[yte == 0]; comp_pos_te = comp_te[yte == 1]
        pos_all_c = np.vstack([comp_pos_tr, comp_pos_te])
        neg_before = np.vstack([comp_neg_tr, comp_neg_te])
        neg_after = np.vstack([comp_neg_tr[keep_tr], comp_neg_te[keep_te]])
        smd_before = np.abs(smd(pos_all_c, neg_before))
        smd_after = np.abs(smd(pos_all_c, neg_after))
        balance_rows.append(dict(dataset=task, n_features=comp_all.shape[1],
            max_abs_smd_before=round(float(np.nanmax(smd_before)), 4),
            mean_abs_smd_before=round(float(np.nanmean(smd_before)), 4),
            max_abs_smd_after=round(float(np.nanmax(smd_after)), 4),
            mean_abs_smd_after=round(float(np.nanmean(smd_after)), 4)))
        print(f"   balance      max|SMD| {np.nanmax(smd_before):.3f}->{np.nanmax(smd_after):.3f} "
              f"mean|SMD| {np.nanmean(smd_before):.3f}->{np.nanmean(smd_after):.3f}")

        # ---- GC-MATCHED arm (25-bin GC histogram, existing convention) ----
        gc_tr = C.gc_content(tr_seqs); gc_te = C.gc_content(te_seqs)
        gk_tr = gc_match(gc_tr[ytr == 1], gc_tr[ytr == 0], args.gc_bins, rng)
        gk_te = gc_match(gc_te[yte == 1], gc_te[yte == 0], args.gc_bins, rng)
        gm_tr_seqs, gm_ytr = negatives_kept_split(tr_seqs, ytr, gk_tr)
        gm_te_seqs, gm_yte = negatives_kept_split(te_seqs, yte, gk_te)
        retained_gc = (len(gk_tr) + len(gk_te)) / max(n_neg_orig, 1)
        r = arm_metrics("gc_match", gm_tr_seqs, gm_ytr, gm_te_seqs, gm_yte, k, tbp, args, rng)
        r.update(dataset=task, role=role, retained_neg_frac=round(float(retained_gc), 4)); rows.append(r)
        _write_split(args.splits_dir, task, "gcmatch", gm_tr_seqs, gm_ytr, gm_te_seqs, gm_yte)
        print(f"   gc_match     TATA={r['tata_auroc']:.3f} GC={r['gc_auroc']:.3f} "
              f"kmerMCC={r['kmer_mcc']:.3f} compMCC={r['comp_only_mcc']:.3f} | kept {retained_gc:.1%} of negatives")

        # ---- TATA-FLAG arm (load existing cleaned_splits/) ----
        tf_tr = os.path.join(args.tata_flag_dir, f"{task}_train.csv")
        tf_te = os.path.join(args.tata_flag_dir, f"{task}_test.csv")
        if os.path.exists(tf_tr) and os.path.exists(tf_te):
            s_tr, y_tr2, s_te, y_te2 = C.load_csv_pair(tf_tr, tf_te)
            retained_tf = (int((y_tr2 == 0).sum() + (y_te2 == 0).sum())) / max(n_neg_orig, 1)
            r = arm_metrics("tata_flag", s_tr, y_tr2, s_te, y_te2, k, tbp, args, rng)
            r.update(dataset=task, role=role, retained_neg_frac=round(float(retained_tf), 4)); rows.append(r)
            print(f"   tata_flag    TATA={r['tata_auroc']:.3f} GC={r['gc_auroc']:.3f} "
                  f"kmerMCC={r['kmer_mcc']:.3f} compMCC={r['comp_only_mcc']:.3f} | kept {retained_tf:.1%} of negatives")
        else:
            print(f"   tata_flag    SKIPPED (no {task} in {args.tata_flag_dir}; run run_cleaning.py first)")

    cols = ["dataset", "role", "arm", "best_k", "retained_neg_frac", "n_total", "n_pos", "n_neg",
            "n_test", "tata_auroc", "tata_ci", "gc_auroc", "gc_ci", "comp_only_mcc", "comp_only_mcc_ci",
            "comp_only_auroc", "comp_only_auroc_ci", "comp_only_lr_mcc", "comp_only_lr_auroc",
            "kmer_mcc", "kmer_mcc_ci", "kmer_auroc", "kmer_auroc_ci", "residual_mcc"]
    df = pd.DataFrame(rows)[cols]
    df.to_csv(args.out, index=False)
    bdf = pd.DataFrame(balance_rows)
    bdf.to_csv(args.balance_out, index=False)
    report = build_report(df, bdf, args)
    print("\n" + report)
    with open(args.out.replace(".csv", "_interpretation.txt"), "w") as fh:
        fh.write(report)
    print(f"\nWrote {args.out} ({len(df)} rows), {args.balance_out}, cleaned_splits_v2/*, + interpretation.")


def _write_split(d, task, tag, tr_seqs, ytr, te_seqs, yte):
    for split, seqs, y in [("train", tr_seqs, ytr), ("test", te_seqs, yte)]:
        pd.DataFrame({"sequence": seqs, "label": y}).to_csv(
            os.path.join(d, f"{task}_{tag}_{split}.csv"), index=False)


def build_report(df, bdf, args):
    L = []; add = L.append
    add("=" * 100)
    add("UPGRADE 2 -- COMPOSITION-EQUALIZED CLEANING: causal test of the compositional shortcut")
    add("=" * 100)
    hdr = (f"{'dataset':<28}{'arm':<16}{'kept%':>7}{'TATA AUROC':>12}{'GC AUROC':>10}"
           f"{'compMCC':>9}{'kmerMCC':>9}{'residual':>9}")
    add(hdr); add("-" * len(hdr))
    for _, r in df.iterrows():
        add(f"{r['dataset']:<28}{r['arm']:<16}{r['retained_neg_frac']:>6.1%} "
            f"{r['tata_auroc']:>11.3f}{r['gc_auroc']:>10.3f}{r['comp_only_mcc']:>9.3f}"
            f"{r['kmer_mcc']:>9.3f}{r['residual_mcc']:>+9.3f}")
    add("")
    add("TATA/GC AUROC = does the motif/GC measure separate pos>neg (0.5 = no separation).")
    add("compMCC = composition-only baseline (LightGBM on 4 mono + 16 dinuc freqs); kmerMCC = the")
    add("k-mer+LightGBM benchmark; residual = kmerMCC - compMCC (signal beyond composition).")
    add("\nBalance (comp_equalized arm), standardized mean diff over the 20 composition features:")
    for _, b in bdf.iterrows():
        add(f"  {b['dataset']:<28} max|SMD| {b['max_abs_smd_before']:.3f} -> {b['max_abs_smd_after']:.3f}   "
            f"mean|SMD| {b['mean_abs_smd_before']:.3f} -> {b['mean_abs_smd_after']:.3f}")
    add("\nVerdict (per contaminated dataset):")
    for ds in df["dataset"].unique():
        g = df[df["dataset"] == ds].set_index("arm")
        if "comp_equalized" not in g.index or "original" not in g.index:
            continue
        o, c = g.loc["original"], g.loc["comp_equalized"]
        tata_collapse = abs(c["tata_auroc"] - 0.5) < abs(o["tata_auroc"] - 0.5)
        gc_collapse = abs(c["gc_auroc"] - 0.5) < abs(o["gc_auroc"] - 0.5)
        residual_pos = c["residual_mcc"] > 0
        add(f"  {ds} [{g.loc['comp_equalized','role'] if 'role' in g.columns else ''}]: "
            f"TATA {o['tata_auroc']:.3f}->{c['tata_auroc']:.3f} ({'collapses' if tata_collapse else 'no collapse'}), "
            f"GC {o['gc_auroc']:.3f}->{c['gc_auroc']:.3f} ({'collapses' if gc_collapse else 'no collapse'}); "
            f"residual MCC on equalized split = {c['residual_mcc']:+.3f} "
            f"({'signal survives composition' if residual_pos else 'no residual above composition'}).")
    add("\nCAVEAT: equalized cleaning changes the test set, so the benchmark MCC is NOT a paired")
    add("comparison across arms; the TATA/GC AUROC collapse (and the residual-vs-composition gap on")
    add("the equalized split) are the causal readouts. Positives are identical across all arms.")
    add(f"Method: LR propensity on 20-d composition, {args.n_bins}-bin proportional match, seed {args.seed}.")
    return "\n".join(L)


if __name__ == "__main__":
    main()
