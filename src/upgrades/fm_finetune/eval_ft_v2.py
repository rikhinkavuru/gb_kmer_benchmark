#!/usr/bin/env python3
"""Imbalance-robust scoring + pre-registered contrasts for the end-to-end fine-tuning runs
(torch-free; reads the per-sequence predictions written by run_ft_v2.py).

PRE-REGISTERED ANALYSIS PLAN (written before any v2 run finished)
Notation, per model x dataset, seed-mean test AUROC:
  A = orig->orig      B = orig->comp_eq     C = orig->random
  D = comp_eq->comp_eq                      E = random->random
  (train arm -> eval arm; every eval set is a subset of the original test set)
Primary quantities
  S  (shortcut worth)      = A - B
  T  (test-imbalance ctl)  = C - B     C is the SAME weights on a test set with the comp_equalized
                                       arm's exact size and class imbalance but random negatives
  U  (train-size ctl)      = E - D     E is trained on exactly as many (and as outnumbered)
                                       negatives as D, but random ones
  R  (signal retention)    = (E - 0.5) / (A - 0.5)
CIs: every contrast gets a 95% percentile CI from 1000 bootstrap resamples of the ORIGINAL test
sequences (seed 42); an eval arm's resampled set = the resampled sequences that belong to it (with
multiplicity), so contrasts across arms are resampled jointly; within a resample, the metric is
averaged over the seeds (same indices for every seed). Seed SD of the per-seed contrast is also
reported (seeds are paired by seed number).
Decision rules (fixed in advance)
  1. "Fine-tuned model rides the shortcut" if S > 0 with CI excluding 0, AND B's CI covers 0.5 or
     B < 0.6 (near chance on composition-equalized negatives).
  2. "The orig->comp_eq drop is NOT a test-size/imbalance artifact" if T > 0 with CI excluding 0
     and C's CI overlaps A's.
  3. "The comp_eq->comp_eq collapse is due to composition removal, not fewer/imbalanced negatives"
     if U > 0 with CI excluding 0 AND R >= 0.5. If U > 0 (CI excl. 0) but R < 0.5: size/imbalance
     contributes; both effects are reported. If U's CI includes 0: size/imbalance cannot be
     excluded as the cause of D (rules 1-2 do not depend on D).
  4. Secondary: random->comp_eq AUROC (does a model trained on random negatives also rely on
     composition?) and, for HyenaDNA, the gc_match arms.
  5. Composition-ALIGNED control (drosophila_enhancers_stark, HyenaDNA only; added 2026-09-23
     22:30 before any drosophila unit was trained): equalization is "not a generic
     signal-destroyer" if B_dros's CI lies above 0.5 AND D_dros's CI lies above 0.5 AND
     S_dros < S of each composition-unaligned dataset with non-overlapping CIs. Any other
     outcome is reported as it is (drosophila also has a composition gap: 20-d composition-only
     LR AUROC 0.689 on its released test split, manifest column comp_only_lr_auroc_test).
Secondary, descriptive (also fixed before results): score-composition coupling. For every run,
on the ORIGINAL test split, the model's logit score is regressed on the standardized 20-d
mono+dinucleotide signature (OLS, 5-fold CV, KFold seed 42). Reported: CV R^2, the AUROC of the
CV composition projection of the score, and the fraction of the model's above-chance AUROC that
this projection recovers, (AUROC_proj - 0.5) / (AUROC_model - 0.5). Expectation: high for
original-trained models on the unaligned datasets (the score is largely a composition readout).
Imbalance-robust metrics for EVERY (run, eval arm): AUROC (prevalence-invariant), balanced
accuracy and MCC at p=0.5 (class-weighted training targets a balanced prior), PR-AUC (average
precision) with its prevalence baseline, each with a 1000-resample CI on the eval arm itself; and
the same four metrics on 20 class-balanced resamples of the eval arm (majority class subsampled
without replacement to the minority count; draws seeded 42.., identical for every model so they
are paired), reported as mean and SD over the 20 draws.

Outputs
  results/upgrades/fm_finetune_v2_metrics.csv    one row per model x dataset x train arm x seed x eval arm
  results/upgrades/fm_finetune_v2_summary.csv    seed-aggregated (mean, SD, seed-mean bootstrap CI)
  results/upgrades/fm_finetune_v2_contrasts.csv  S, T, U, R, ... with CIs and the rule verdicts
  results/upgrades/fm_finetune_v2_runs.csv       training metadata (params, best epoch, time)
  results/upgrades/fm_finetune_v2_verdicts.csv   decision-rule outcomes
  results/upgrades/fm_finetune_v2_score_composition{,_summary}.csv   score-composition coupling

Usage: python src/upgrades/fm_finetune/eval_ft_v2.py   (torch-free venv)
       python src/upgrades/fm_finetune/eval_ft_v2.py --pred-dir <smoke dir> --out-prefix <path>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PRED_DIR = os.path.join(ROOT, "results", "upgrades", "fm_finetune_v2_preds")
SPLIT_DIR = os.path.join(ROOT, "external", "fm_finetune", "splits")
OUT = os.path.join(ROOT, "results", "upgrades")
ARMS = ("original", "gc_match", "comp_equalized", "random")
B = 1000
N_BAL = 20
SEED = 42


# ------------------------------------------------------------- fast metrics
def auroc(y, p):
    """Mann-Whitney AUROC with average ranks for ties (== sklearn roc_auc_score)."""
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def avg_precision(y, p):
    """Step-wise average precision over distinct thresholds (== sklearn average_precision_score)."""
    n1 = int((y == 1).sum())
    if n1 == 0:
        return np.nan
    o = np.argsort(-p, kind="mergesort")
    ps, ys = p[o], y[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    last = np.r_[np.where(np.diff(ps) != 0)[0], len(ps) - 1]   # last index of each tie block
    tp, fp = tp[last], fp[last]
    prec = tp / (tp + fp); rec = tp / n1
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def thr_metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(np.int64)
    tp = float(((pred == 1) & (y == 1)).sum()); tn = float(((pred == 0) & (y == 0)).sum())
    fp = float(((pred == 1) & (y == 0)).sum()); fn = float(((pred == 0) & (y == 1)).sum())
    tpr = tp / (tp + fn) if tp + fn else np.nan
    tnr = tn / (tn + fp) if tn + fp else np.nan
    ba = 0.5 * (tpr + tnr)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return ba, mcc


def all_metrics(y, p):
    ba, mcc = thr_metrics(y, p)
    return dict(auroc=auroc(y, p), ap=avg_precision(y, p), bal_acc=ba, mcc=mcc)


METRICS = ("auroc", "ap", "bal_acc", "mcc")


# --------------------------------------------------------------- resamplers
def boot_indices(n, seed=SEED, B=B):
    rng = np.random.RandomState(seed)
    return [rng.randint(0, n, n) for _ in range(B)]


def balanced_draws(y, seed=SEED, k=N_BAL):
    """k class-balanced subsets: the majority class subsampled without replacement."""
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    lo = min(len(pos), len(neg))
    out = []
    for d in range(k):
        rng = np.random.RandomState(seed + d)
        out.append(np.sort(np.r_[rng.choice(pos, lo, replace=False),
                                 rng.choice(neg, lo, replace=False)]))
    return out


# ------------------------------------------------------------------- loading
def load_membership():
    mem = {}
    for f in glob.glob(os.path.join(SPLIT_DIR, "*__test.csv.gz")):
        ds = os.path.basename(f).split("__")[0]
        df = pd.read_csv(f, usecols=["idx", "label"] + [f"{a}_pos" for a in ARMS])
        mem[ds] = dict(y=df["label"].to_numpy(np.int64),
                       mask={a: df[f"{a}_pos"].to_numpy() >= 0 for a in ARMS})
    return mem


def load_runs(pred_dir=PRED_DIR):
    runs = []
    for js in sorted(glob.glob(os.path.join(pred_dir, "*.json"))):
        csv = js[:-5] + ".csv.gz"
        if not os.path.exists(csv):
            continue
        meta = json.load(open(js))
        df = pd.read_csv(csv)
        te = df[df["split"] == "test"].sort_values("idx")
        runs.append(dict(meta=meta, p=te["prob"].to_numpy(np.float64),
                         idx=te["idx"].to_numpy(np.int64), y=te["label"].to_numpy(np.int64)))
    return runs


def eval_arms(model):
    return ARMS


# ---------------------------------------------------------------- per run
def per_run_rows(runs, mem):
    rows = []
    cache_boot, cache_bal = {}, {}
    for r in runs:
        mt = r["meta"]; ds = mt["dataset"]; y = mem[ds]["y"]
        assert np.array_equal(r["idx"], np.arange(len(y))) and np.array_equal(r["y"], y)
        for ea in ARMS:
            msk = mem[ds]["mask"][ea]
            yy, pp = y[msk], r["p"][msk]
            key = (ds, ea)
            if key not in cache_boot:
                cache_boot[key] = boot_indices(len(yy))
                cache_bal[key] = balanced_draws(yy)
            point = all_metrics(yy, pp)
            bs = np.array([[all_metrics(yy[i], pp[i])[m] for m in METRICS]
                           for i in cache_boot[key]])
            bal = np.array([[all_metrics(yy[i], pp[i])[m] for m in METRICS]
                            for i in cache_bal[key]])
            row = dict(model=mt["model"], model_tag=mt["model_tag"], dataset=ds,
                       train_arm=mt["train_arm"], eval_arm=ea, seed=mt["seed"],
                       n_test=int(msk.sum()), n_pos=int((yy == 1).sum()),
                       n_neg=int((yy == 0).sum()), prevalence=round(float(yy.mean()), 6))
            for j, m in enumerate(METRICS):
                row[m] = point[m]
                row[f"{m}_lo"] = np.nanpercentile(bs[:, j], 2.5)
                row[f"{m}_hi"] = np.nanpercentile(bs[:, j], 97.5)
            row["ap_baseline"] = row["prevalence"]
            row["ap_lift"] = point["ap"] - row["prevalence"]
            for j, m in enumerate(METRICS):
                row[f"bal_{m}_mean"] = float(np.mean(bal[:, j]))
                row[f"bal_{m}_sd"] = float(np.std(bal[:, j], ddof=1))
            row["bal_n_per_class"] = int(min(row["n_pos"], row["n_neg"]))
            row["best_epoch"] = mt["best_epoch"]; row["best_val_auroc"] = mt["best_val_auroc"]
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------- seed-aggregated + contrasts
def group_runs(runs):
    g = {}
    for r in runs:
        mt = r["meta"]
        g.setdefault((mt["model"], mt["dataset"], mt["train_arm"]), {})[mt["seed"]] = r["p"]
    return g


def joint_boot_auc(g, mem, model, ds, pairs, seeds=None):
    """For each bootstrap resample of the ORIGINAL test set, seed-mean AUROC of every
    (train_arm, eval_arm) pair in `pairs`. Returns dict pair -> (B,) array, and the seed list."""
    y = mem[ds]["y"]; n = len(y)
    avail = [set(g.get((model, ds, ta), {}).keys()) for ta, _ in pairs]
    common = sorted(set.intersection(*avail)) if avail else []
    if seeds is not None:
        common = [s for s in common if s in seeds]
    out = {pr: np.full(B, np.nan) for pr in pairs}
    if not common:
        return out, common
    rng = np.random.RandomState(SEED)
    for b in range(B):
        idx = rng.randint(0, n, n)
        for ta, ea in pairs:
            sub = idx[mem[ds]["mask"][ea][idx]]
            out[(ta, ea)][b] = np.mean([auroc(y[sub], g[(model, ds, ta)][s][sub]) for s in common])
    return out, common


def point_auc(g, mem, model, ds, ta, ea, seeds):
    y = mem[ds]["y"]; msk = mem[ds]["mask"][ea]
    return np.array([auroc(y[msk], g[(model, ds, ta)][s][msk]) for s in seeds])


def summarize(per, runs, mem):
    g = group_runs(runs)
    rows = []
    for (model, ds, ta), sub in per.groupby(["model", "dataset", "train_arm"]):
        pairs = [(ta, ea) for ea in ARMS]
        bt, seeds = joint_boot_auc(g, mem, model, ds, pairs)
        for ea in ARMS:
            s = sub[sub["eval_arm"] == ea]
            row = dict(model=model, dataset=ds, train_arm=ta, eval_arm=ea, n_seeds=len(s),
                       seeds=",".join(map(str, sorted(s["seed"]))), n_test=int(s["n_test"].iloc[0]),
                       prevalence=float(s["prevalence"].iloc[0]))
            for m in METRICS + ("ap_lift",):
                row[f"{m}_mean"] = s[m].mean(); row[f"{m}_sd"] = s[m].std(ddof=1)
            row["auroc_seedmean_lo"] = np.nanpercentile(bt[(ta, ea)], 2.5)
            row["auroc_seedmean_hi"] = np.nanpercentile(bt[(ta, ea)], 97.5)
            row["auroc_min"] = s["auroc"].min(); row["auroc_max"] = s["auroc"].max()
            for m in METRICS:
                row[f"bal_{m}_mean"] = s[f"bal_{m}_mean"].mean()
                row[f"bal_{m}_seed_sd"] = s[f"bal_{m}_mean"].std(ddof=1)
            row["best_epoch_mean"] = s["best_epoch"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def contrasts(runs, mem):
    g = group_runs(runs)
    rows = []
    for model, ds in sorted({(k[0], k[1]) for k in g}):
        have = {k[2] for k in g if k[0] == model and k[1] == ds}
        pairs = [("original", "original"), ("original", "comp_equalized"),
                 ("original", "random"), ("comp_equalized", "comp_equalized"),
                 ("random", "random"), ("random", "comp_equalized"),
                 ("comp_equalized", "original"), ("random", "original"),
                 ("original", "gc_match"), ("gc_match", "gc_match")]
        pairs = [pr for pr in pairs if pr[0] in have]
        if not pairs:
            continue
        seeds_all = sorted(set.intersection(*[set(g[(model, ds, ta)]) for ta, _ in pairs]))
        bt, seeds = joint_boot_auc(g, mem, model, ds, pairs)
        pt = {pr: point_auc(g, mem, model, ds, pr[0], pr[1], seeds) for pr in pairs}

        def add(name, defn, a, b=None, kind="diff"):
            if a not in pt or (b is not None and b not in pt):
                return
            if kind == "diff":
                per_seed = pt[a] - pt[b]; bb = bt[a] - bt[b]
            elif kind == "level":
                per_seed = pt[a]; bb = bt[a]
            else:                                     # retention ratio (E-0.5)/(A-0.5)
                per_seed = (pt[a] - 0.5) / (pt[b] - 0.5)
                bb = (bt[a] - 0.5) / (bt[b] - 0.5)
            rows.append(dict(model=model, dataset=ds, contrast=name, definition=defn,
                             n_seeds=len(seeds), estimate=float(np.mean(per_seed)),
                             seed_sd=float(np.std(per_seed, ddof=1)) if len(seeds) > 1 else np.nan,
                             ci_lo=float(np.nanpercentile(bb, 2.5)),
                             ci_hi=float(np.nanpercentile(bb, 97.5)),
                             per_seed=";".join(f"{v:.4f}" for v in per_seed)))
        O, CE, RD, GC = "original", "comp_equalized", "random", "gc_match"
        add("A_orig_orig", "AUROC orig->orig", (O, O), kind="level")
        add("B_orig_compeq", "AUROC orig->comp_eq", (O, CE), kind="level")
        add("C_orig_random", "AUROC orig->random", (O, RD), kind="level")
        add("D_compeq_compeq", "AUROC comp_eq->comp_eq", (CE, CE), kind="level")
        add("E_random_random", "AUROC random->random", (RD, RD), kind="level")
        add("F_random_compeq", "AUROC random->comp_eq", (RD, CE), kind="level")
        add("S_shortcut_worth", "A - B", (O, O), (O, CE))
        add("T_test_imbalance_ctl", "C - B", (O, RD), (O, CE))
        add("C_minus_A", "C - A (should be ~0)", (O, RD), (O, O))
        add("U_train_size_ctl", "E - D", (RD, RD), (CE, CE))
        add("R_retention", "(E-0.5)/(A-0.5)", (RD, RD), (O, O), kind="ratio")
        add("E_minus_F", "E - F (random-trained: random vs comp_eq test)", (RD, RD), (RD, CE))
        add("G_orig_gcmatch", "AUROC orig->gc_match", (O, GC), kind="level")
        add("H_gcmatch_gcmatch", "AUROC gc_match->gc_match", (GC, GC), kind="level")
        add("S_gc", "A - AUROC(orig->gc_match)", (O, O), (O, GC))
    ct = pd.DataFrame(rows)
    return ct, verdicts(ct)


def verdicts(ct):
    out = []
    for (model, ds), c in ct.groupby(["model", "dataset"]):
        get = lambda k: c[c["contrast"] == k].iloc[0] if (c["contrast"] == k).any() else None
        S, Bq, T, Cq, A, U, R = (get("S_shortcut_worth"), get("B_orig_compeq"),
                                 get("T_test_imbalance_ctl"), get("C_orig_random"),
                                 get("A_orig_orig"), get("U_train_size_ctl"), get("R_retention"))
        v = dict(model=model, dataset=ds)
        if S is not None and Bq is not None:
            v["rule1_rides_shortcut"] = bool(S.ci_lo > 0 and (Bq.ci_lo <= 0.5 <= Bq.ci_hi
                                                              or Bq.estimate < 0.6))
        if T is not None and Cq is not None and A is not None:
            v["rule2_drop_not_test_imbalance"] = bool(T.ci_lo > 0 and Cq.ci_lo <= A.ci_hi
                                                      and A.ci_lo <= Cq.ci_hi)
        if U is not None and R is not None:
            if U.ci_lo > 0 and R.estimate >= 0.5:
                v["rule3_collapse_cause"] = "composition removal (random arm preserves signal)"
            elif U.ci_lo > 0:
                v["rule3_collapse_cause"] = "both: composition removal + size/imbalance (R<0.5)"
            else:
                v["rule3_collapse_cause"] = "size/imbalance NOT excluded (U CI includes 0)"
        if ds in ALIGNED:
            D = get("D_compeq_compeq")
            others = ct[(ct["model"] == model) & (~ct["dataset"].isin(ALIGNED))
                        & (ct["contrast"] == "S_shortcut_worth")]
            if S is not None and Bq is not None and D is not None and len(others):
                v["rule5_aligned_control_not_destroyed"] = bool(
                    Bq.ci_lo > 0.5 and D.ci_lo > 0.5 and (S.ci_hi < others["ci_lo"]).all())
                v["rule5_compared_to"] = ",".join(others["dataset"])
        out.append(v)
    return pd.DataFrame(out)


ALIGNED = ("drosophila_enhancers_stark",)


def score_composition(runs, mem):
    """Per run: how much of the model's logit score on the ORIGINAL test split is a linear
    function of the 20-d mono+dinucleotide composition signature (5-fold CV OLS)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    sys.path.insert(0, os.path.join(ROOT, "src", "upgrades")); sys.path.insert(0, ROOT)
    import _common as C
    sig, rows = {}, []
    for r in runs:
        mt = r["meta"]; ds = mt["dataset"]; y = mem[ds]["y"]
        if ds not in sig:
            te = pd.read_csv(os.path.join(SPLIT_DIR, f"{ds}__test.csv.gz"))
            X = C.comp_signature(te["sequence"].astype(str).tolist()).astype(np.float64)
            sig[ds] = (X - X.mean(0)) / (X.std(0) + 1e-12)
        X = sig[ds]
        p = np.clip(r["p"], 1e-6, 1 - 1e-6)
        z = np.log(p / (1.0 - p))
        zhat = np.zeros_like(z)
        for tr_i, te_i in KFold(5, shuffle=True, random_state=SEED).split(X):
            zhat[te_i] = LinearRegression().fit(X[tr_i], z[tr_i]).predict(X[te_i])
        r2 = 1.0 - np.sum((z - zhat) ** 2) / np.sum((z - z.mean()) ** 2)
        a_m, a_p = auroc(y, z), auroc(y, zhat)
        rows.append(dict(model=mt["model"], dataset=ds, train_arm=mt["train_arm"], seed=mt["seed"],
                         eval_split="original_test", n=len(y), r2_cv=r2, auroc_model=a_m,
                         auroc_comp_projection=a_p,
                         frac_above_chance_recovered=(a_p - 0.5) / (a_m - 0.5)
                         if a_m > 0.5 else np.nan))
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df
    num = ["r2_cv", "auroc_model", "auroc_comp_projection", "frac_above_chance_recovered"]
    g = df.groupby(["model", "dataset", "train_arm"])[num]
    summ = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd"))
    summ = summ.join(df.groupby(["model", "dataset", "train_arm"])["seed"].size().rename("n_seeds"))
    return df, summ.reset_index()


def run_table(runs):
    rows = []
    for r in runs:
        mt = r["meta"]
        rows.append(dict(model=mt["model"], model_tag=mt["model_tag"], dataset=mt["dataset"],
                         train_arm=mt["train_arm"], seed=mt["seed"], n_fit=mt["n_fit"],
                         n_fit_neg=mt["n_fit_neg"], n_val=mt["n_val"],
                         n_params_total=mt["n_params_total"],
                         n_params_trainable=mt["n_params_trainable"],
                         best_epoch=mt["best_epoch"], epochs_run=mt["epochs_run"],
                         best_val_auroc=mt["best_val_auroc"], train_secs=mt["train_secs"],
                         val_auroc_by_epoch=";".join(str(h["val_auroc"]) for h in mt["history"]),
                         device=mt["device"],
                         # memory settings (numerical no-ops; added after the 2026-09-24 incident)
                         pad_multiple=mt.get("pad_multiple", 0) or 0,
                         micro_batch=mt.get("micro_batch", 0) or 0,
                         eval_batch=mt.get("eval_batch_used", mt["recipe"]["eval_batch"]),
                         mps_high_watermark=(mt.get("env") or {}).get(
                             "PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
                         peak_footprint_mb=mt.get("peak_footprint_mb", np.nan)))
    return pd.DataFrame(rows).sort_values(["model", "dataset", "train_arm", "seed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=PRED_DIR)
    ap.add_argument("--out-prefix", default=os.path.join(OUT, "fm_finetune_v2"))
    a = ap.parse_args()
    pre = a.out_prefix
    mem = load_membership()
    runs = load_runs(a.pred_dir)
    print(f"[eval_ft_v2] {len(runs)} runs from {a.pred_dir}")
    rt = run_table(runs)
    rt.to_csv(pre + "_runs.csv", index=False)
    per = per_run_rows(runs, mem)
    per.round(6).to_csv(pre + "_metrics.csv", index=False)
    summ = summarize(per, runs, mem)
    summ.round(6).to_csv(pre + "_summary.csv", index=False)
    ct, vd = contrasts(runs, mem)
    ct.round(6).to_csv(pre + "_contrasts.csv", index=False)
    vd.to_csv(pre + "_verdicts.csv", index=False)
    sc, scs = score_composition(runs, mem)
    sc.round(6).to_csv(pre + "_score_composition.csv", index=False)
    scs.round(6).to_csv(pre + "_score_composition_summary.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["model", "dataset", "train_arm", "eval_arm", "n_seeds", "auroc_mean", "auroc_sd",
            "auroc_seedmean_lo", "auroc_seedmean_hi", "bal_bal_acc_mean", "bal_mcc_mean",
            "ap_mean", "prevalence"]
    print(summ[cols].round(3).to_string(index=False))
    print(ct[["model", "dataset", "contrast", "n_seeds", "estimate", "seed_sd", "ci_lo",
              "ci_hi"]].round(3).to_string(index=False))
    print(vd.to_string(index=False))


if __name__ == "__main__":
    main()
