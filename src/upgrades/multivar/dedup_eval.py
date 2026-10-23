#!/usr/bin/env python3
"""POST-HOC (verification round, 2026-09-24): how much of each homology-flagged TF-motif task's
Table-1 score comes from test sequences that have a near-duplicate in train?

Plan, fixed before this script was first run (descriptive; no decision threshold, nothing tuned):
  Tasks   = the S11 (TF-motif) tasks whose homology covariate is >= 0.05, i.e. verdict "LEAKY" under
            the pre-existing threshold of src/upgrades/homology/measure_homology.verdict:
            human_nontata_promoters, human_enhancers_ensembl, nt_enhancers, nt_enhancers_types.
  Flags   = for EVERY test sequence (full test split, no subsampling), its max exact 8-mer Jaccard to
            the whole train split (same functions as measure_homology_all.py); near-duplicate = >= 0.7
            (the paper's threshold).  Also the verbatim flag (exact string, either strand) from
            audit_exact_dups.py.  Sanity: the per-sequence values must equal the values already stored
            for the same test indices (5,000-query subsample for human_enhancers_ensembl; full for
            nt_enhancers_types), and the >=0.7 fraction must equal homology_leakage.csv for
            human_nontata_promoters and nt_enhancers.
  Model   = the Table-1 model: validation-selected k (results/val_selection.csv new_k), L1 k-mer
            spectrum (featurize.kmer_spectrum), models.LGBM_PARAMS (seed 42) with n_jobs=N_THREADS instead
            of -1 for machine sharing (thread count can move LightGBM results in the last digits, so the
            full-test MCC is compared with Table 1 and ASSERTED equal to its 4 printed decimals, |d| < 5e-5).  N_THREADS is 2 (the
            laptop CPU policy); the cached predictions under external/multivar/dedup/ were produced on
            2026-09-24 12:27-12:57 with n_jobs=3, before the 2-thread rule, and reproduce Table 1 exactly
            (|d| < 5e-5 on all four tasks), so they were kept.
  Readout = MCC, AUROC (macro one-vs-rest for 3 classes) and per-class recall on: all test sequences;
            the near-duplicate-free subset (max Jaccard < 0.7 and not verbatim); the near-duplicate
            subset; and the same split by the verbatim flag alone.  For binary tasks also the AUROC of
            the subset's positives against ALL test negatives (duplicates are almost all positives, so a
            subset can lose one class).  MCC depends on class prevalence, which the subsets change;
            per-class recall and AUROC are the prevalence-free comparisons, and mcc_prevalence_std is
            the subset's MCC with every class re-weighted to its full-test count (added after the first
            task finished, before any subset result of the other three was seen; it re-uses the cached
            predictions, so it changes no other column).
Checkpoints: external/multivar/dedup/<task>_maxjac_full.npy, <task>_proba.npy (resume = skip).
Output: results/upgrades/multivar_posthoc_dedup_eval.csv.  Imports lightgbm (never torch).
Run: bibm/lowcpu.sh /Users/rikhinkavuru/Downloads/venv/bin/python src/upgrades/multivar/dedup_eval.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _mv  # noqa: E402
import measure_homology_all as MH  # noqa: E402  (imports homology/measure_homology as MH.H)

sys.path.insert(0, _mv.ROOT)
import featurize  # noqa: E402
import metrics as gbmetrics  # noqa: E402

RD = _mv.RESULTS_DIR
OUT_DIR = os.path.join(_mv.EXT_DIR, "dedup")
FEAT_CHUNK = 20000
N_THREADS = 2                   # CPU policy: LightGBM num_threads <= 2 (see docstring)


def full_maxjac(task, tr, te):
    fp = os.path.join(OUT_DIR, f"{task}_maxjac_full.npy")
    if os.path.exists(fp):
        return np.load(fp)
    t0 = time.time()
    Q = MH.H.presence_matrix(te)
    mj = np.zeros(len(te))
    for c0 in range(0, len(tr), MH.TRAIN_CHUNK):
        T = MH.H.presence_matrix(tr[c0:c0 + MH.TRAIN_CHUNK])
        mj = np.maximum(mj, MH.max_jaccard_sparse(Q, T))
        del T
        print(f"   [{task}] jaccard train chunk {c0 // MH.TRAIN_CHUNK + 1}/{-(-len(tr) // MH.TRAIN_CHUNK)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    np.save(fp + ".tmp.npy", mj)
    os.replace(fp + ".tmp.npy", fp)
    return mj


def features(seqs, k):
    return sparse.vstack([featurize.kmer_spectrum(seqs[i:i + FEAT_CHUNK], k)
                          for i in range(0, len(seqs), FEAT_CHUNK)]).tocsr()


def fit_predict(task, k, tr, ytr, te):
    fp = os.path.join(OUT_DIR, f"{task}_proba.npy")
    if os.path.exists(fp):
        return np.load(fp)
    import models as gbmodels                       # lightgbm (never torch in this process)
    from lightgbm import LGBMClassifier
    t0 = time.time()
    Xtr, Xte = features(tr, k), features(te, k)
    print(f"   [{task}] features k={k} {Xtr.shape} ({time.time() - t0:.0f}s)", flush=True)
    params = dict(gbmodels.LGBM_PARAMS)
    params["n_jobs"] = N_THREADS
    model = LGBMClassifier(random_state=_mv.C.SEED, **params)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    assert list(model.classes_) == list(range(proba.shape[1]))
    print(f"   [{task}] LightGBM fit+predict ({time.time() - t0:.0f}s)", flush=True)
    np.save(fp + ".tmp.npy", proba)
    os.replace(fp + ".tmp.npy", fp)
    return proba


def subset_metrics(y, proba, m, ncls):
    """MCC / AUROC exactly as metrics.compute_metrics (binary: P(class 1); 3-class: macro OvR, defined
    only when every class is present in the subset), plus per-class recall."""
    from sklearn.metrics import matthews_corrcoef, roc_auc_score
    ys, ps = y[m], proba[m]
    pred = ps.argmax(axis=1)                        # == model.classes_[argmax]; classes_ == 0..K-1 asserted
    row = dict(n=int(m.sum()), mcc=np.nan, auroc=np.nan)
    for c in range(3):
        row[f"n_class{c}"] = int((ys == c).sum()) if c < ncls else np.nan
        row[f"recall_class{c}"] = float((pred[ys == c] == c).mean()) if c < ncls and (ys == c).any() else np.nan
    present = len(np.unique(ys))
    row["mcc_prevalence_std"] = np.nan
    if present >= 2:
        row["mcc"] = float(matthews_corrcoef(ys, pred))
        if ncls == 2 or present == ncls:
            row["auroc"] = gbmetrics.compute_metrics(ys, pred, ps, ncls)["auroc"]
        if present == ncls:
            # MCC with each class re-weighted to its FULL-test count (removes the prevalence shift that
            # dropping near-duplicates causes; per-class confusion rates are the subset's own)
            w = np.array([(y == c).sum() / (ys == c).sum() for c in range(ncls)])[ys]
            row["mcc_prevalence_std"] = float(matthews_corrcoef(ys, pred, sample_weight=w))
    if ncls == 2:
        pos = m & (y == 1)
        neg = y == 0
        if pos.any() and neg.any():
            sel = pos | neg
            row["auroc_subset_pos_vs_all_neg"] = float(roc_auc_score(y[sel], proba[sel, 1]))
    return row


def main():
    _mv.start_memguard(tag="dedup_eval")
    os.makedirs(OUT_DIR, exist_ok=True)
    cov = pd.read_csv(os.path.join(RD, "multivar_covariates.csv")).set_index("task")
    tasks = [t for t in cov.index if cov.loc[t, "S11"] and cov.loc[t, "homology"] >= 0.05]
    assert sorted(tasks) == sorted(["human_nontata_promoters", "human_enhancers_ensembl", "nt_enhancers",
                                    "nt_enhancers_types"]), tasks
    vs = pd.read_csv(os.path.join(_mv.ROOT, "results", "val_selection.csv")).set_index("dataset")
    leak = pd.read_csv(os.path.join(RD, "homology_leakage.csv")).set_index("task")
    rows = []
    for task in tasks:
        t0 = time.time()
        tr, ytr, te, yte = _mv.load_task_cached(task)
        ncls = int(max(ytr.max(), yte.max()) + 1)
        k = int(vs.loc[task, "new_k"])
        mj = full_maxjac(task, tr, te)
        # sanity checks against stored values
        chk = ""
        fp_sub = os.path.join(_mv.EXT_DIR, "homology", f"{task}_maxjac.npy")
        if os.path.exists(fp_sub):
            qidx, v = np.load(fp_sub)
            dmax = float(np.max(np.abs(mj[qidx.astype(int)] - v)))
            assert dmax < 1e-12, (task, dmax)
            chk += f"matches stored per-sequence values at {len(qidx)} indices (max|d| {dmax:.1e}); "
        if task in leak.index:
            f7 = float((mj >= 0.7).mean())
            assert abs(round(f7, 5) - leak.loc[task, "frac_ge_0_7"]) < 1e-9, (task, f7)
            chk += f"frac>=0.7 {f7:.5f} == homology_leakage.csv; "
        dz = np.load(os.path.join(_mv.EXT_DIR, "dups", f"{task}.npz"))
        assert np.array_equal(dz["yte"], yte)
        verb = dz["any"]
        near = (mj >= 0.7) | verb
        proba = fit_predict(task, k, tr, ytr, te)
        table1 = float(cov.loc[task, "bench_mcc"])
        from sklearn.metrics import matthews_corrcoef
        full_mcc = float(matthews_corrcoef(yte, proba.argmax(axis=1)))
        # reproduces Table 1 to its 4 printed decimals (fix round 2: tolerance tightened from 5e-4 to 5e-5,
        # which is what 'equal to 4 decimals' means; observed |d| <= 3.9e-5 on all four tasks)
        assert abs(full_mcc - table1) < 5e-5, (task, full_mcc, table1)
        for name, m in (("all", np.ones(len(yte), bool)), ("clean_j07", ~near), ("neardup_j07", near),
                        ("not_verbatim", ~verb), ("verbatim", verb)):
            r = dict(task=task, k=k, n_classes=ncls, subset=name, frac_of_test=float(m.mean()),
                     table1_mcc=table1, frac_ge_0_7_full=float((mj >= 0.7).mean()),
                     frac_verbatim=float(verb.mean()), sanity=chk.strip())
            r.update(subset_metrics(yte, proba, m, ncls))
            rows.append(r)
        a = [r for r in rows if r["task"] == task]
        print(f"[{task}] k={k}  full-test MCC {a[0]['mcc']:.4f} (Table 1 {table1:.4f}, d {a[0]['mcc'] - table1:+.4f})  "
              f"near-dup {a[0]['frac_ge_0_7_full']:.4f} verbatim {a[0]['frac_verbatim']:.4f} | clean MCC "
              f"{a[1]['mcc']:.4f} AUROC {a[1]['auroc']:.4f} | recall per class all "
              + "/".join(f"{a[0][f'recall_class{c}']:.3f}" for c in range(ncls)) + "  clean "
              + "/".join(f"{a[1][f'recall_class{c}']:.3f}" for c in range(ncls)) + "  near-dup "
              + "/".join(f"{a[2][f'recall_class{c}']:.3f}" for c in range(ncls))
              + f"  ({time.time() - t0:.0f}s; peak {_mv._PEAK[0]:.0f} MB)", flush=True)
        del tr, te
    df = pd.DataFrame(rows)
    tmp = os.path.join(RD, "multivar_posthoc_dedup_eval.csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(RD, "multivar_posthoc_dedup_eval.csv"))
    print(f"wrote results/upgrades/multivar_posthoc_dedup_eval.csv ({len(df)} rows); peak footprint "
          f"{_mv._PEAK[0]:.0f} MB")


if __name__ == "__main__":
    main()
