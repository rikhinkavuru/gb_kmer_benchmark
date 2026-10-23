#!/usr/bin/env python3
"""Per-task covariate table for the multivariate learnability analysis (all 15 tasks).

Step 1 (per task, checkpointed to external/multivar/taskstats/<task>.json, resumable):
  * sequence length mean / SD / min / max over train+test (bp, all characters);
  * per-sequence GC fraction over A/C/G/T (_mv.gc_fraction == _common.gc_content): task mean,
    total SD, within-class pooled SD, |Cohen's d| between classes (max over pairs for 3-class);
  * composition-only baseline: 20-d mono+dinucleotide signature (_common.comp_signature), the fixed
    LightGBM config of models.py (seed 42) with num_threads=N_THREADS (2, the laptop CPU policy; the 15
    taskstats checkpoints were computed on 2026-09-24 11:13-11:20 with 3 threads, before the 2-thread rule --
    the --merge reproduction check below still gives |diff| = 0 on all five published values), fit on
    the original train split,
    scored once on the original test split, 1000-resample percentile bootstrap CIs (test sequence =
    unit; fresh RandomState(42) per task).  This is the exact recipe of
    composition_clean/run_composition_clean.py's `original` arm, so its five published values
    (cohn, nt_enhancers, drosophila, human_ocr_ensembl, human_enhancers_ensembl) must reproduce --
    checked in --merge and reported.
Step 2 (--merge): joins these with the benchmark outcome (cross_suite_summary.csv: bench_mcc,
  bench_auroc, CIs, max_pos_motif_auroc, tf_motif_task), homology (homology_leakage.csv for the 4
  measured tasks; multivar_homology.csv for the other 11), positional rescue (positional_sweep.csv
  delta_mcc) and n_train (results.csv / results_nt.csv, selected_by_val=1) and writes
  results/upgrades/multivar_covariates.csv.
Sequences come from _mv.load_task_cached (see _mv.py for why the GB loader is bypassed).
Usage:
  bibm/lowcpu.sh python src/upgrades/multivar/build_covariates.py [--tasks a,b]      # step 1
  python src/upgrades/multivar/build_covariates.py --merge                           # step 2
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _mv  # noqa: E402

C = _mv.C
OUT_DIR = os.path.join(_mv.EXT_DIR, "taskstats")
OUT_CSV = os.path.join(_mv.RESULTS_DIR, "multivar_covariates.csv")
N_THREADS = 2          # CPU policy: LightGBM num_threads <= 2


def fit_eval_boot_2threads(Xtr, ytr, Xte, yte, seed, B, rng):
    """Copy of _common.fit_eval_boot for LightGBM with num_threads=N_THREADS (machine-sharing limit).
    LightGBM runs with deterministic=True, force_row_wise=True (models.LGBM_PARAMS)."""
    from sklearn.metrics import matthews_corrcoef
    nc = len(np.unique(ytr))
    m = C.gbmodels.build_model("lgbm", seed, nc)
    m.set_params(n_jobs=N_THREADS)
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)
    pred = m.classes_[np.argmax(proba, axis=1)]
    mcc_pt = float(matthews_corrcoef(yte, pred))
    auc_pt = float(C._auroc(yte, proba, nc))
    n = len(yte)
    mm = np.full(B, np.nan)
    aa = np.full(B, np.nan)
    for b in range(B):
        idx = rng.randint(0, n, n)
        if len(np.unique(yte[idx])) < 2:
            continue
        mm[b] = matthews_corrcoef(yte[idx], pred[idx])
        try:
            aa[b] = C._auroc(yte[idx], proba[idx], nc)
        except ValueError:
            pass
    return dict(comp_mcc=round(mcc_pt, 6),
                comp_mcc_lo=round(float(np.nanpercentile(mm, 2.5)), 6),
                comp_mcc_hi=round(float(np.nanpercentile(mm, 97.5)), 6),
                comp_auroc=round(auc_pt, 6),
                comp_auroc_lo=round(float(np.nanpercentile(aa, 2.5)), 6),
                comp_auroc_hi=round(float(np.nanpercentile(aa, 97.5)), 6))


def task_stats(task, boot):
    fp = os.path.join(OUT_DIR, f"{task}.json")
    if os.path.exists(fp):
        print(f"[{task}] done already -- skip", flush=True)
        return
    t0 = time.time()
    tr, ytr, te, yte = _mv.load_task_cached(task)
    seqs = tr + te
    y = np.concatenate([ytr, yte])
    L = np.array([len(s) for s in seqs], dtype=np.float64)
    gc = _mv.gc_fraction(seqs)
    row = dict(task=task, n_train_loaded=len(tr), n_test_loaded=len(te),
               n_classes_loaded=int(len(np.unique(y))),
               len_mean=float(L.mean()), len_sd=float(L.std(ddof=1)), len_min=int(L.min()),
               len_max=int(L.max()),
               gc_mean=float(gc.mean()), gc_sd_total=float(gc.std(ddof=1)),
               gc_sd_within=_mv.pooled_within_sd(gc, y), gc_gap_d=_mv.max_pairwise_cohens_d(gc, y))
    print(f"[{task}] len {row['len_mean']:.1f}+-{row['len_sd']:.1f}  GC {row['gc_mean']:.3f} "
          f"sd_within {row['gc_sd_within']:.3f} |d| {row['gc_gap_d']:.3f}  ({time.time() - t0:.0f}s) "
          f"-- fitting composition-only LightGBM ...", flush=True)
    Ctr = C.comp_signature(tr, ks=(1, 2))
    Cte = C.comp_signature(te, ks=(1, 2))
    rng = np.random.RandomState(C.SEED)
    row.update(fit_eval_boot_2threads(Ctr, ytr, Cte, yte, C.SEED, boot, rng))
    row["seconds"] = round(time.time() - t0, 1)
    with open(fp + ".tmp", "w") as fh:
        json.dump(row, fh)
    os.replace(fp + ".tmp", fp)
    print(f"[{task}] comp-only MCC {row['comp_mcc']:.4f} [{row['comp_mcc_lo']:.3f},{row['comp_mcc_hi']:.3f}] "
          f"AUROC {row['comp_auroc']:.4f}  ({row['seconds']}s)", flush=True)


def _ci(s):
    a, b = s.strip("[]").split(",")
    return float(a), float(b)


def merge():
    cs = pd.read_csv(os.path.join(_mv.ROOT, "results", "cross_suite_summary.csv"))
    res = pd.concat([pd.read_csv(os.path.join(_mv.ROOT, "results", f))
                     for f in ("results.csv", "results_nt.csv")])
    res = res[(res["model"] == "lgbm") & (res["selected_by_val"] == 1)].set_index("dataset")
    hom_old = pd.read_csv(os.path.join(_mv.RESULTS_DIR, "homology_leakage.csv")).set_index("task")
    hom_new = pd.read_csv(os.path.join(_mv.RESULTS_DIR, "multivar_homology.csv")).set_index("task")
    pos = pd.read_csv(os.path.join(_mv.RESULTS_DIR, "positional_sweep.csv")).set_index("task")
    rows = []
    for _, r in cs.iterrows():
        t = r["task"]
        fp = os.path.join(OUT_DIR, f"{t}.json")
        if not os.path.exists(fp):
            sys.exit(f"missing task stats for {t}; run step 1 first")
        with open(fp) as fh:
            st = json.load(fh)
        if t in hom_old.index:
            hom, hsrc, hlo, hhi = hom_old.loc[t, "frac_ge_0_7"], "homology_leakage.csv(full)", np.nan, np.nan
        elif t in hom_new.index:
            h = hom_new.loc[t]
            hom, hsrc = h["frac_ge_0_7"], f"multivar_homology.csv({h['query_mode']})"
            hlo, hhi = h["frac_ge_0_7_lo"], h["frac_ge_0_7_hi"]
        else:
            hom, hsrc, hlo, hhi = np.nan, "MISSING", np.nan, np.nan
        n_train = int(res.loc[t, "n_train"])
        assert n_train == st["n_train_loaded"], f"{t}: n_train {n_train} vs loaded {st['n_train_loaded']}"
        assert int(res.loc[t, "n_test"]) == st["n_test_loaded"], t
        assert int(r["n_classes"]) == st["n_classes_loaded"], t
        mlo, mhi = _ci(r["mcc_ci"])
        alo, ahi = _ci(r["auroc_ci"])
        rows.append(dict(
            task=t, suite=r["suite"], S11=bool(r["tf_motif_task"]), n_classes=int(r["n_classes"]),
            bench_mcc=r["bench_mcc"], mcc_lo=mlo, mcc_hi=mhi,
            bench_auroc=r["bench_auroc"], auroc_lo=alo, auroc_hi=ahi,
            best_k=int(res.loc[t, "k"]), n_train=n_train, n_test=int(res.loc[t, "n_test"]),
            # ---- pre-registered covariates (transformed) ----
            disc=r["max_pos_motif_auroc"],
            len_log=np.log10(st["len_mean"]), len_sd_log=np.log10(1.0 + st["len_sd"]),
            gc_mean=st["gc_mean"], gc_sd=st["gc_sd_within"], gc_gap=st["gc_gap_d"],
            homology=hom, ntrain_log=np.log10(n_train), multiclass=int(r["n_classes"] > 2),
            pos_delta=pos.loc[t, "delta_mcc"],
            comp_mcc=st["comp_mcc"], comp_auroc=st["comp_auroc"],
            # ---- raw / provenance columns ----
            len_mean=st["len_mean"], len_sd=st["len_sd"], len_min=st["len_min"], len_max=st["len_max"],
            gc_sd_total=st["gc_sd_total"], homology_lo=hlo, homology_hi=hhi, homology_source=hsrc,
            comp_mcc_lo=st["comp_mcc_lo"], comp_mcc_hi=st["comp_mcc_hi"],
            comp_auroc_lo=st["comp_auroc_lo"], comp_auroc_hi=st["comp_auroc_hi"],
            best_pos_TF=r["best_pos_TF"]))
    df = pd.DataFrame(rows)
    # ---- reproduction check of the composition-only baseline against composition_clean ----
    chk = []
    for f in ("composition_clean.csv", "composition_clean_specificity.csv"):
        cc = pd.read_csv(os.path.join(_mv.RESULTS_DIR, f))
        cc = cc[cc["arm"] == "original"]
        for _, c in cc.iterrows():
            mine = df.loc[df["task"] == c["dataset"]].iloc[0]
            chk.append(dict(task=c["dataset"], published_comp_mcc=c["comp_only_mcc"],
                            this_comp_mcc=mine["comp_mcc"],
                            abs_diff_mcc=abs(c["comp_only_mcc"] - mine["comp_mcc"]),
                            published_comp_auroc=c["comp_only_auroc"], this_comp_auroc=mine["comp_auroc"],
                            abs_diff_auroc=abs(c["comp_only_auroc"] - mine["comp_auroc"]), source=f))
    chk = pd.DataFrame(chk)
    chk.to_csv(os.path.join(_mv.RESULTS_DIR, "multivar_comp_reproduction.csv"), index=False)
    print(chk.to_string(index=False))
    df.to_csv(OUT_CSV, index=False)
    print(df[["task", "S11", "bench_mcc", "disc", "len_log", "gc_sd", "gc_gap", "homology",
              "pos_delta", "comp_mcc"]].round(3).to_string(index=False))
    print(f"wrote {OUT_CSV} ({len(df)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=",".join(_mv.TASKS))
    ap.add_argument("--boot", type=int, default=C.BOOT)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    _mv.start_memguard(tag="build_covariates")
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.merge:
        merge()
        return
    for t in a.tasks.split(","):
        task_stats(t, a.boot)


if __name__ == "__main__":
    main()
