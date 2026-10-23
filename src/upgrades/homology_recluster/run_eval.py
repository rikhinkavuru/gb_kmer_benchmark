#!/usr/bin/env python3
"""Step 3 -- homology-clean re-evaluation of the benchmark baseline on ALL 15 tasks.

PRE-REGISTERED ANALYSIS PLAN (written 2026-09-24, before any evaluation in this file was run;
clusters come from run_cluster.py: connected components of the MMseqs2 both-strand homology graph,
id80 = >=80 % identity / >=80 % coverage PRIMARY, id50 = strict)
-----------------------------------------------------------------------------------------------
Model. The paper's fixed LightGBM (models.LGBM_PARAMS, seed 42) on the L1-normalised k-mer spectrum
at the task's validation-selected k (results/val_selection.csv new_k). k is NOT re-selected: only the
split changes. Threads are capped at 3 for every arm (same thread count in every comparison).

U0  ORIGINAL. Fit on the provided train split, predict the provided test split. Sanity check: the
    MCC must reproduce Table I (val_selection.csv new_test_mcc) to within 0.005 (LightGBM's
    deterministic mode can differ in the last digits across thread counts). All deltas are computed
    against OUR reproduction so every arm shares one thread count; the published value is carried
    alongside for reference.

(a) TEST-PURGE (same trained model, no refit). Drop every test sequence whose component at threshold
    T contains >=1 training sequence. Report MCC and AUROC with 95 % percentile-bootstrap CIs (1000
    resamples, test sequence = unit, seed 42) and
      * the PAIRED delta vs the full test set (one bootstrap of the full test set; both MCCs
        recomputed on every resample; purged = resampled members that are purged-kept);
      * a prevalence-standardised MCC (sample weights restoring the original per-class test
        proportions), because purging can shift the class balance (e.g. if near-duplicates are
        concentrated in one class), which by itself moves MCC;
      * a RANDOM-REMOVAL null: 1000 random subsets of the full test set with exactly the purged set's
        per-class counts. homology_effect = MCC_purged - mean(null); p_null = P(null <= MCC_purged).
        This separates "the removed sequences were homologous to training data" from "the test set
        simply got smaller / changed class mix".

(b) CLUSTER-AWARE RE-SPLIT. Pool train+test; assign WHOLE components to train or test with
    _hc.cluster_resplit (seeded greedy, label-stratified: per-label test targets
    round(test_frac * n_label), test_frac = original n_test / (n_train + n_test)), seed 42; retrain;
    evaluate on the new test split. CONTROL: a label-stratified RANDOM re-split with the same per-label
    targets (_hc.random_resplit, seed 42), also retrained -> isolates "homology-aware" from "any
    re-split". delta_vs_random = MCC_cluster - MCC_random, CI from the difference of independent
    bootstrap resamples of the two test sets. Both are also compared with U0.
    The re-split is flagged INFEASIBLE (not reported as a result) if the achieved test fraction is
    < 50 % of the target for any label (giant components can make whole-cluster assignment impossible;
    this mostly concerns id50).

Decision rules (fixed in advance):
  * A task's published MCC is called HOMOLOGY-INFLATED at threshold T if EITHER
      - the purge delta is negative with its paired CI excluding 0 AND MCC_purged is below the 2.5th
        percentile of the random-removal null, OR
      - the cluster re-split MCC is below the random re-split MCC with the delta CI excluding 0.
  * Tier (solved >= 0.70, partial >= 0.50, hard < 0.50) is reported for the original and each
    homology-clean MCC; route-label preservation is decided in run_routes.py.
Nulls and adverse results are reported as they come out.

VERIFIER ADDENDUM (2026-09-24 ~11:30, POST HOC -- added after the results above had been seen, in
response to an independent code/claims audit; the pre-registered rules above are unchanged):
  * CLASS ELIMINATION. A purge that leaves < MIN_CLASS_KEPT (10) test sequences of a class that is
    present in the full test set scores a k-class model on a (k-1)-class test set; the prevalence-
    standardised MCC cannot restore an absent class. Such a purge is now status 'class_eliminated'
    (NOT evaluable; the degenerate MCC is carried only as mcc_degenerate) instead of 'ok'. The old guard
    only required >= 2 surviving classes. Affected: nt_enhancers_types (all 100 class-2 'strong' test
    enhancers have a train homolog; id80 and id50).
  * MEMORISATION READOUT (descriptive). acc_removed / acc_kept = accuracy of the SAME model on the
    purged-away test sequences (those with a train homolog) vs the kept ones, with per-class counts.
  * SUBTYPE-MATCHED NULL. nt_enhancers is the binary collapse of nt_enhancers_types (identical sequences
    and order, asserted). Its purge removes every 'strong' positive, so the binary-class-matched null
    confounds "homologous to train" with "loses the strong subclass". A second random-removal null
    matching the kept counts per SUBTYPE (NULL_STRATA) is reported as homology_effect_strat / p_null_strat
    (separate RandomState(seed + 11), so the pre-registered null is unchanged).
  * Featurisation is now lazy: a run whose predictions are all cached re-scores without re-featurising
    (the numbers are recomputed from the cached predictions with the same seeds).

Checkpointing: per (task, unit) predictions -> external/homology_recluster/preds/<task>__<unit>.npz;
metrics are recomputed from predictions. Tasks run sequentially, smallest first.
Outputs: results/upgrades/homology_recluster_eval.csv (one row per task x protocol x threshold).

Usage: python run_eval.py [--tasks a,b] [--thresholds id80,id50] [--seed 42]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _hc  # noqa: E402
from run_cluster import SIZE_ORDER  # noqa: E402

PRED_DIR = os.path.join(_hc.EXT, "preds")
N_NULL = 1000
MIN_CLASS_KEPT = 10                       # verifier addendum: fewer kept => class eliminated, not evaluable
NULL_STRATA = {"nt_enhancers": "nt_enhancers_types"}   # binary task -> task holding its subtype labels


def null_strata(task, n_tr, n):
    """Subtype labels of the TEST split for tasks listed in NULL_STRATA (else None). Asserts the two
    tasks hold identical sequences in identical order and that the subtypes refine the binary labels."""
    if task not in NULL_STRATA:
        return None
    tr, ytr, te, yte = _hc.load_task(task)
    tr2, ytr2, te2, yte2 = _hc.load_task(NULL_STRATA[task])
    assert tr == tr2 and te == te2, f"{task} and {NULL_STRATA[task]} do not hold identical sequences"
    ct = pd.crosstab(yte, yte2)
    assert int((ct > 0).sum(axis=0).max()) == 1, "subtype labels do not refine the binary labels"
    assert len(yte2) == n - n_tr
    return np.asarray(yte2)


def tier(m):
    return "solved" if m >= 0.7 else ("partial" if m >= 0.5 else "hard")


# ------------------------------------------------------------------ fitting (checkpointed)
def fit_unit(task, unit, X, y, tr_idx, te_idx, k, seed):
    """Fit LightGBM on rows tr_idx, predict te_idx; cache (te_idx, proba, classes).
    X may be the feature matrix or a zero-argument callable returning it (lazy: only called on a cache miss)."""
    f = os.path.join(PRED_DIR, f"{task}__{unit}.npz")
    if os.path.exists(f):
        z = np.load(f)
        if np.array_equal(z["te_idx"], te_idx) and np.array_equal(z["tr_idx"], tr_idx) and int(z["k"]) == k:
            return z["proba"], z["classes"]
        print(f"   [{unit}] cached split differs -> refit", flush=True)
    if callable(X):
        X = X()
    t0 = time.time()
    m = _hc.build_lgbm(len(np.unique(y[tr_idx])), seed)
    m.fit(X[tr_idx], y[tr_idx])
    proba = m.predict_proba(X[te_idx])
    np.savez_compressed(f + ".tmp.npz", te_idx=te_idx, tr_idx=tr_idx, proba=proba.astype(np.float32),
                        classes=m.classes_, k=k, seed=seed)
    os.replace(f + ".tmp.npz", f)
    print(f"   [{unit}] fit n_train={len(tr_idx)} n_test={len(te_idx)} in {time.time()-t0:.0f}s", flush=True)
    return proba, m.classes_


def pred_of(proba, classes):
    return classes[np.argmax(proba, axis=1)]


# ------------------------------------------------------------------ metrics
def std_mcc(y, pred, ref_prop):
    """Prevalence-standardised MCC: weights so each class has its reference proportion."""
    labs, cnt = np.unique(y, return_counts=True)
    w = np.zeros(len(y))
    for l, c in zip(labs, cnt):
        w[y == l] = ref_prop.get(int(l), 0.0) / (c / len(y))
    if w.sum() == 0:
        return float("nan")
    return float(matthews_corrcoef(y, pred, sample_weight=w))


def purge_block(y, proba, classes, keep, seed=_hc.SEED, B=_hc.BOOT, strata=None):
    """Metrics for the purged subset `keep` (bool over the full test set) of the SAME predictions.
    strata (optional, same length as y): subtype labels for the subtype-matched null (verifier addendum)."""
    pred = pred_of(proba, classes)
    n = len(y)
    ref = {int(l): float(np.mean(y == l)) for l in np.unique(y)}
    full = _hc.boot_metrics(y, pred, proba, seed, B)
    kept_cls = {int(l): int(np.sum(y[keep] == l)) for l in np.unique(y)}
    full_cls = {int(l): int(np.sum(y == l)) for l in np.unique(y)}
    rem = ~keep
    out = dict(n_test_eval=int(keep.sum()), frac_test_kept=round(float(keep.mean()), 5),
               kept_per_class=";".join(f"{l}:{kept_cls[l]}/{full_cls[l]}" for l in kept_cls),
               min_class_kept_frac=round(min(kept_cls[l] / full_cls[l] for l in kept_cls), 5),
               n_removed=int(rem.sum()),
               acc_removed=round(float(np.mean(pred[rem] == y[rem])), 5) if rem.any() else np.nan,
               acc_kept=round(float(np.mean(pred[keep] == y[keep])), 5) if keep.any() else np.nan,
               acc_removed_per_class=";".join(
                   f"{l}:{np.mean(pred[rem & (y == l)] == l):.3f}(n={int(np.sum(rem & (y == l)))})"
                   for l in kept_cls if np.any(rem & (y == l))))
    if keep.sum() < 10 or len(np.unique(y[keep])) < 2:
        out.update(status="too_few_kept")
        return full, out
    if min(kept_cls.values()) < MIN_CLASS_KEPT:
        out.update(status="class_eliminated", mcc_degenerate=float(matthews_corrcoef(y[keep], pred[keep])))
        return full, out
    pk = _hc.boot_metrics(y[keep], pred[keep], proba[keep], seed, B)
    # paired delta + standardised MCC bootstrap (resample the FULL test set)
    rng = np.random.RandomState(seed)
    dd = np.full(B, np.nan); ss = np.full(B, np.nan)
    for b in range(B):
        idx = rng.randint(0, n, n)
        kb = idx[keep[idx]]
        if len(np.unique(y[idx])) < 2 or len(np.unique(y[kb])) < 2:
            continue
        dd[b] = matthews_corrcoef(y[kb], pred[kb]) - matthews_corrcoef(y[idx], pred[idx])
        ss[b] = std_mcc(y[kb], pred[kb], ref)
    # random-removal null with identical per-class counts
    rng2 = np.random.RandomState(seed + 7)
    counts = {int(l): int(np.sum(y[keep] == l)) for l in np.unique(y)}
    null = np.empty(N_NULL)
    by = {l: np.where(y == l)[0] for l in counts}
    for r in range(N_NULL):
        sel = np.concatenate([rng2.choice(by[l], counts[l], replace=False) for l in counts if counts[l] > 0])
        null[r] = matthews_corrcoef(y[sel], pred[sel])
    q = lambda a, p: float(np.nanpercentile(a, p))
    out.update(status="ok", mcc=pk["mcc"], mcc_lo=pk["mcc_lo"], mcc_hi=pk["mcc_hi"],
               auroc=pk["auroc"], auroc_lo=pk["auroc_lo"], auroc_hi=pk["auroc_hi"],
               pos_frac=pk["pos_frac"],
               delta_mcc=round(pk["mcc"] - full["mcc"], 5), delta_lo=q(dd, 2.5), delta_hi=q(dd, 97.5),
               delta_auroc=round(pk["auroc"] - full["auroc"], 5) if pk["auroc"] == pk["auroc"] else np.nan,
               std_mcc=std_mcc(y[keep], pred[keep], ref), std_mcc_lo=q(ss, 2.5), std_mcc_hi=q(ss, 97.5),
               null_mean=float(null.mean()), null_lo=q(null, 2.5), null_hi=q(null, 97.5),
               homology_effect=round(pk["mcc"] - float(null.mean()), 5),
               p_null=float(np.mean(null <= pk["mcc"])))
    if strata is not None:
        # subtype-matched random-removal null (verifier addendum; separate stream -> pre-registered null unchanged)
        rng3 = np.random.RandomState(seed + 11)
        sc = {int(s): int(np.sum(strata[keep] == s)) for s in np.unique(strata)}
        bys = {s: np.where(strata == s)[0] for s in sc}
        nul2 = np.empty(N_NULL)
        for r in range(N_NULL):
            sel = np.concatenate([rng3.choice(bys[s], sc[s], replace=False) for s in sc if sc[s] > 0])
            nul2[r] = matthews_corrcoef(y[sel], pred[sel])
        out.update(kept_per_stratum=";".join(f"{s}:{sc[s]}/{len(bys[s])}" for s in sc),
                   null_strat_mean=float(nul2.mean()), null_strat_lo=q(nul2, 2.5), null_strat_hi=q(nul2, 97.5),
                   homology_effect_strat=round(pk["mcc"] - float(nul2.mean()), 5),
                   p_null_strat=float(np.mean(nul2 <= pk["mcc"])))
    return full, out


def boot_diff(y1, p1, y2, p2, seed=_hc.SEED, B=_hc.BOOT):
    """CI of MCC(split1) - MCC(split2) from independent bootstrap resamples of the two test sets."""
    r1 = np.random.RandomState(seed); r2 = np.random.RandomState(seed + 1)
    d = np.full(B, np.nan)
    for b in range(B):
        i = r1.randint(0, len(y1), len(y1)); j = r2.randint(0, len(y2), len(y2))
        if len(np.unique(y1[i])) < 2 or len(np.unique(y2[j])) < 2:
            continue
        d[b] = matthews_corrcoef(y1[i], p1[i]) - matthews_corrcoef(y2[j], p2[j])
    return float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def split_balance(y, is_test, test_frac):
    """Achieved per-label test fraction relative to target; min ratio over labels."""
    ratios = {}
    for l in np.unique(y):
        tgt = round(test_frac * np.sum(y == l))
        ratios[int(l)] = float(np.sum(is_test & (y == l)) / tgt) if tgt > 0 else float("nan")
    return ratios


# ------------------------------------------------------------------ per task
def do_task(task, thresholds, seed):
    import featurize
    t0 = time.time()
    k = _hc.best_k(task)
    tr, ytr, te, yte = _hc.load_task(task)
    seqs = tr + te
    y = np.concatenate([ytr, yte])
    n_tr, n = len(tr), len(tr) + len(te)
    orig_is_test = np.r_[np.zeros(n_tr, bool), np.ones(len(te), bool)]
    test_frac = len(te) / n
    pub = _hc.table1(task)
    print(f"\n[{task}] k={k} n_train={n_tr} n_test={len(te)} test_frac={test_frac:.4f} "
          f"published MCC={pub['mcc']:.4f}", flush=True)
    _X = {}

    def X():   # lazy: only featurise when a unit's predictions are not cached
        if "m" not in _X:
            _X["m"] = featurize.kmer_spectrum(seqs, k)
        return _X["m"]
    strata = null_strata(task, n_tr, n)
    rows = []
    base = dict(task=task, k=k, n_train_orig=n_tr, n_test_orig=len(te),
                published_mcc=pub["mcc"], published_auroc=pub["auroc"])

    # ---- U0 original ----
    tr_idx = np.arange(n_tr); te_idx = np.arange(n_tr, n)
    proba0, cls0 = fit_unit(task, "original", X, y, tr_idx, te_idx, k, seed)
    pred0 = pred_of(proba0, cls0)
    orig = _hc.boot_metrics(yte, pred0, proba0, seed)
    repro_ok = abs(orig["mcc"] - pub["mcc"]) <= 0.005
    print(f"   original MCC={orig['mcc']:.4f} [{orig['mcc_lo']:.3f},{orig['mcc_hi']:.3f}] "
          f"AUROC={orig['auroc']:.4f} | reproduces Table I: {repro_ok} (|d|={abs(orig['mcc']-pub['mcc']):.4f})",
          flush=True)
    rows.append(dict(base, protocol="original", threshold="", status="ok", n_train_eval=n_tr,
                     n_test_eval=len(te), frac_test_kept=1.0, mcc=orig["mcc"], mcc_lo=orig["mcc_lo"],
                     mcc_hi=orig["mcc_hi"], auroc=orig["auroc"], auroc_lo=orig["auroc_lo"],
                     auroc_hi=orig["auroc_hi"], pos_frac=orig["pos_frac"], reproduces_table1=repro_ok,
                     tier=tier(orig["mcc"])))

    # ---- random re-split control (threshold-independent) ----
    rnd_is_test = _hc.random_resplit(y, test_frac, seed)
    pr_r, cl_r = fit_unit(task, "resplit_random", X, y, np.where(~rnd_is_test)[0], np.where(rnd_is_test)[0],
                          k, seed)
    y_r = y[rnd_is_test]; pred_r = pred_of(pr_r, cl_r)
    rnd = _hc.boot_metrics(y_r, pred_r, pr_r, seed)
    print(f"   random re-split MCC={rnd['mcc']:.4f} [{rnd['mcc_lo']:.3f},{rnd['mcc_hi']:.3f}]", flush=True)
    rows.append(dict(base, protocol="resplit_random", threshold="", status="ok",
                     n_train_eval=int((~rnd_is_test).sum()), n_test_eval=int(rnd_is_test.sum()),
                     mcc=rnd["mcc"], mcc_lo=rnd["mcc_lo"], mcc_hi=rnd["mcc_hi"], auroc=rnd["auroc"],
                     auroc_lo=rnd["auroc_lo"], auroc_hi=rnd["auroc_hi"], pos_frac=rnd["pos_frac"],
                     delta_vs_original=round(rnd["mcc"] - orig["mcc"], 5), tier=tier(rnd["mcc"])))

    for tag in thresholds:
        comp, is_test_saved = _hc.load_clusters(task, tag)
        assert np.array_equal(is_test_saved, orig_is_test) and len(comp) == n
        # ---- (a) test purge ----
        has_tr = np.zeros(comp.max() + 1, bool); has_tr[comp[:n_tr]] = True
        keep = ~has_tr[comp[n_tr:]]
        _, pb = purge_block(yte, proba0, cls0, keep, seed, strata=strata)
        r = dict(base, protocol="test_purge", threshold=tag, n_train_eval=n_tr, **pb)
        if pb.get("status") == "ok":
            r["tier"] = tier(pb["mcc"])
            r["homology_inflated"] = bool((pb["delta_hi"] < 0 and pb["mcc"] < pb["null_lo"]))
            print(f"   purge {tag}: kept {pb['frac_test_kept']:.3f} MCC={pb['mcc']:.4f} "
                  f"[{pb['mcc_lo']:.3f},{pb['mcc_hi']:.3f}] delta={pb['delta_mcc']:+.4f} "
                  f"[{pb['delta_lo']:+.3f},{pb['delta_hi']:+.3f}] null={pb['null_mean']:.4f} "
                  f"[{pb['null_lo']:.3f},{pb['null_hi']:.3f}] stdMCC={pb['std_mcc']:.4f} "
                  f"acc removed/kept={pb['acc_removed']:.3f}/{pb['acc_kept']:.3f}"
                  + (f" | subtype-matched null {pb['null_strat_mean']:.4f} effect {pb['homology_effect_strat']:+.4f} "
                     f"p={pb['p_null_strat']:.3f}" if "null_strat_mean" in pb else ""), flush=True)
        else:
            print(f"   purge {tag}: {pb.get('status')} (kept {pb['frac_test_kept']:.3f}; per class "
                  f"{pb['kept_per_class']})", flush=True)
        rows.append(r)

        # ---- (b) cluster-aware re-split ----
        cl_is_test = _hc.cluster_resplit(comp, y, test_frac, seed)
        # invariant: no component straddles the split
        ct = pd.DataFrame({"c": comp, "t": cl_is_test}).groupby("c")["t"].nunique()
        assert int(ct.max()) == 1, "a component straddles the cluster-aware split"
        bal = split_balance(y, cl_is_test, test_frac)
        feasible = all(v >= 0.5 for v in bal.values() if v == v)
        base_rs = dict(base, protocol="resplit_cluster", threshold=tag,
                       n_train_eval=int((~cl_is_test).sum()), n_test_eval=int(cl_is_test.sum()),
                       achieved_test_frac=round(float(cl_is_test.mean()), 5),
                       min_label_target_ratio=round(min(bal.values()), 4))
        if not feasible:
            print(f"   re-split {tag}: INFEASIBLE (label target ratios {bal})", flush=True)
            rows.append(dict(base_rs, status="infeasible"))
            continue
        pr_c, cl_c = fit_unit(task, f"resplit_{tag}", X, y, np.where(~cl_is_test)[0], np.where(cl_is_test)[0],
                              k, seed)
        y_c = y[cl_is_test]; pred_c = pred_of(pr_c, cl_c)
        cm = _hc.boot_metrics(y_c, pred_c, pr_c, seed)
        dlo, dhi = boot_diff(y_c, pred_c, y_r, pred_r, seed)
        dlo0, dhi0 = boot_diff(y_c, pred_c, yte, pred0, seed)
        # how many of the re-split test sequences were in the ORIGINAL test split
        infl = bool(cm["mcc"] < rnd["mcc"] and dhi < 0)
        print(f"   re-split {tag}: test_frac={cl_is_test.mean():.4f} MCC={cm['mcc']:.4f} "
              f"[{cm['mcc_lo']:.3f},{cm['mcc_hi']:.3f}] vs random {rnd['mcc']:.4f}: "
              f"d={cm['mcc']-rnd['mcc']:+.4f} [{dlo:+.3f},{dhi:+.3f}] | vs original d={cm['mcc']-orig['mcc']:+.4f}",
              flush=True)
        rows.append(dict(base_rs, status="ok", mcc=cm["mcc"], mcc_lo=cm["mcc_lo"], mcc_hi=cm["mcc_hi"],
                         auroc=cm["auroc"], auroc_lo=cm["auroc_lo"], auroc_hi=cm["auroc_hi"],
                         pos_frac=cm["pos_frac"], delta_vs_random=round(cm["mcc"] - rnd["mcc"], 5),
                         delta_vs_random_lo=dlo, delta_vs_random_hi=dhi,
                         delta_vs_original=round(cm["mcc"] - orig["mcc"], 5),
                         delta_vs_original_lo=dlo0, delta_vs_original_hi=dhi0,
                         frac_test_from_orig_test=round(float(orig_is_test[cl_is_test].mean()), 5),
                         homology_inflated=infl, tier=tier(cm["mcc"])))
    _X.clear()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(_hc.CKPT_DIR, f"eval_{task}.csv"), index=False)
    print(f"[{task}] done in {time.time()-t0:.0f}s", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=",".join(SIZE_ORDER))
    ap.add_argument("--thresholds", default="id80,id50")
    ap.add_argument("--seed", type=int, default=_hc.SEED)
    ap.add_argument("--out", default=os.path.join(_hc.RES, f"{_hc.PREFIX}_eval.csv"))
    a = ap.parse_args()
    os.makedirs(PRED_DIR, exist_ok=True); os.makedirs(_hc.CKPT_DIR, exist_ok=True)
    ths = [t for t in a.thresholds.split(",") if t]
    for t in [t for t in a.tasks.split(",") if t]:
        if not all(os.path.exists(os.path.join(_hc.CLU_DIR, f"{t}_{th}.npz")) for th in ths):
            print(f"[{t}] clusters missing -> skipped (run run_cluster.py first)", flush=True)
            continue
        do_task(t, ths, a.seed)
    parts = [pd.read_csv(os.path.join(_hc.CKPT_DIR, f"eval_{t}.csv")) for t in _hc.TASKS
             if os.path.exists(os.path.join(_hc.CKPT_DIR, f"eval_{t}.csv"))]
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(a.out + ".tmp", index=False)
        os.replace(a.out + ".tmp", a.out)          # atomic: other scripts read this file while we run
        print(f"wrote {a.out} ({len(parts)} tasks)")


if __name__ == "__main__":
    main()
