#!/usr/bin/env python3
"""Independent cross-check of the WRITTEN v2 result tables against the stored per-sequence
predictions, using sklearn instead of eval_ft_v2's own metric code. Torch-free, well under a minute.
Added 2026-09-26 (post hoc verification; changes no result).

  1. inventory: 54 fine-tuning units = HyenaDNA 3 datasets x 4 arms x seeds 42-44 + DNABERT-2 LoRA
     2 datasets x 3 arms x seeds 42-44; 52 baseline units (k-mer 3x4, frozen HyenaDNA 2 poolings x
     3x4, frozen DNABERT-2 2 poolings x 2x4); every prediction file covers the whole released test
     split with the verified labels.
  2. every point metric in fm_finetune_v2_metrics.csv and fm_finetune_v2_posthoc_metrics.csv
     (AUROC, average precision, balanced accuracy and MCC at p=0.5, n_test/n_pos/n_neg, prevalence)
     equals sklearn on the stored probabilities restricted to the eval arm (|diff| < 1e-6);
     nt_enhancers_leakfree rows use the released test minus the 100 duplicated positives.
  3. the seed-aggregated summaries are the seed mean / SD of those rows.
  4. contrasts: level estimates are seed-mean AUROCs; S=A-B, T=C-B, U=E-D, C-A, E-F, A-G are
     differences of them (per seed, averaged); R=(E-.5)/(A-.5) per seed, averaged; lo <= hi.
  5. paper tables: S_est = A - B, frac_lost = (A-B)/(A-.5), frac_lost_random = (A-C)/(A-.5)
     (point estimates, from the seed-mean AUROCs).
  6. fm_finetune_v2_compare.csv carries the pre-registered fine-tuned rows unchanged and the post
     hoc rows from the post hoc table; compare_paired differences = finetuned_value - baseline_value.
Run: bibm/lowcpu.sh <torch-free python> src/upgrades/fm_finetune/tests/test_outputs_ft_v2.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, matthews_corrcoef,
                             roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RES = os.path.join(ROOT, "results", "upgrades")
PRE = os.path.join(RES, "fm_finetune_v2")
SPLITS = os.path.join(ROOT, "external", "fm_finetune", "splits")
PREDS = {"ft": os.path.join(RES, "fm_finetune_v2_preds"),
         "base": os.path.join(ROOT, "external", "fm_finetune", "baseline_preds")}
ARMS = ("original", "gc_match", "comp_equalized", "random")
NTL = "nt_enhancers_leakfree"
TOL = 1e-6


def membership():
    mem = {}
    for f in glob.glob(os.path.join(SPLITS, "*__test.csv.gz")):
        ds = os.path.basename(f).split("__")[0]
        df = pd.read_csv(f, usecols=["idx", "label"] + [f"{a}_pos" for a in ARMS])
        assert np.array_equal(df["idx"].to_numpy(), np.arange(len(df)))
        mem[ds] = dict(y=df["label"].to_numpy(np.int64),
                       mask={a: df[f"{a}_pos"].to_numpy() >= 0 for a in ARMS})
    dups = pd.read_csv(PRE + "_leakfree_dups.csv")
    keep = np.ones(len(mem["nt_enhancers"]["y"]), bool)
    keep[dups["test_idx"].to_numpy()] = False
    mem[NTL] = dict(y=mem["nt_enhancers"]["y"][keep],
                    mask={a: m[keep] for a, m in mem["nt_enhancers"]["mask"].items()}, keep=keep)
    return mem


def load_preds(mem):
    P = {}
    for fam, d in PREDS.items():
        for js in sorted(glob.glob(os.path.join(d, "*.json"))):
            mt = json.load(open(js))
            df = pd.read_csv(js[:-5] + ".csv.gz")
            te = df[df["split"] == "test"].sort_values("idx")
            y = mem[mt["dataset"]]["y"]
            assert np.array_equal(te["idx"].to_numpy(), np.arange(len(y))), js
            assert np.array_equal(te["label"].to_numpy(np.int64), y), js
            P[(mt["model"], mt["dataset"], mt["train_arm"], int(mt["seed"]))] = (
                fam, te["prob"].to_numpy(np.float64))
            if mt["dataset"] == "nt_enhancers":
                P[(mt["model"], NTL, mt["train_arm"], int(mt["seed"]))] = (
                    fam, te["prob"].to_numpy(np.float64)[mem[NTL]["keep"]])
    return P


def test_inventory(P):
    ft = {k for k, v in P.items() if v[0] == "ft" and k[1] != NTL}
    base = {k for k, v in P.items() if v[0] == "base" and k[1] != NTL}
    exp = set()
    for ds in ("human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark"):
        exp |= {("hyenadna", ds, a, s) for a in ARMS for s in (42, 43, 44)}
    for ds in ("human_enhancers_cohn", "nt_enhancers"):
        exp |= {("dnabert2", ds, a, s) for a in ("original", "comp_equalized", "random")
                for s in (42, 43, 44)}
    assert exp <= ft, sorted(exp - ft)
    extra = sorted(ft - exp)            # GPU hand-back units (dnabert2full, ntv2_500m) are allowed
    assert all(k[0] in ("dnabert2full", "ntv2_500m") for k in extra), extra
    assert len(base) == 52, len(base)
    print(f"ok  inventory: {len(exp)} local fine-tuning units (expected grid) + {len(extra)} GPU units, "
          f"{len(base)} baseline units; every file covers the released test split with the verified labels")


def sk(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(auroc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                bal_acc=balanced_accuracy_score(y, pred), mcc=matthews_corrcoef(y, pred))


def test_metrics(P, mem):
    n = 0
    worst = 0.0
    for f in (PRE + "_metrics.csv", PRE + "_posthoc_metrics.csv"):
        m = pd.read_csv(f)
        for r in m.itertuples():
            fam, p = P[(r.model, r.dataset, r.train_arm, int(r.seed))]
            y = mem[r.dataset]["y"]; msk = mem[r.dataset]["mask"][r.eval_arm]
            yy, pp = y[msk], p[msk]
            assert (r.n_test, r.n_pos, r.n_neg) == (len(yy), int(yy.sum()), int((yy == 0).sum())), r
            assert abs(r.prevalence - yy.mean()) < TOL
            ref = sk(yy, pp)
            for k, v in ref.items():
                d = abs(getattr(r, k) - v)
                worst = max(worst, d)
                assert d < TOL, (f, r.model, r.dataset, r.train_arm, r.seed, r.eval_arm, k, getattr(r, k), v)
            n += 1
    lf = pd.read_csv(PRE + "_posthoc_metrics.csv")
    lf = lf[(lf.dataset == NTL)]
    o = lf[lf.eval_arm == "original"]
    assert (o.n_test == 300).all() and (o.n_pos == 100).all(), o[["n_test", "n_pos"]].drop_duplicates()
    ce = lf[lf.eval_arm == "comp_equalized"]
    assert (ce.n_pos == 100).all() and (ce.n_neg == 36).all()
    print(f"ok  {n} metric rows (pre-registered + post hoc) == sklearn on the stored predictions "
          f"(max |diff| {worst:.1e}); leak-free rows = 300 (100+/200-), equalized arm 100+/36-")


def test_summary():
    for tag in ("", "_posthoc"):
        m = pd.read_csv(PRE + f"{tag}_metrics.csv")
        s = pd.read_csv(PRE + f"{tag}_summary.csv")
        g = m.groupby(["model", "dataset", "train_arm", "eval_arm"])
        for r in s.itertuples():
            q = g.get_group((r.model, r.dataset, r.train_arm, r.eval_arm))
            assert r.n_seeds == len(q)
            for k in ("auroc", "ap", "bal_acc", "mcc"):
                assert abs(getattr(r, f"{k}_mean") - q[k].mean()) < 2e-6, (r, k)
                if len(q) > 1:
                    assert abs(getattr(r, f"{k}_sd") - q[k].std(ddof=1)) < 2e-6, (r, k)
            assert r.auroc_seedmean_lo <= r.auroc_mean + 1e-9 or r.auroc_seedmean_lo <= r.auroc_seedmean_hi
            assert r.auroc_seedmean_lo <= r.auroc_seedmean_hi
        print(f"ok  {'post hoc' if tag else 'pre-registered'} summary: {len(s)} rows = seed mean/SD of the metric rows")


LEV = {"A_orig_orig": ("original", "original"), "B_orig_compeq": ("original", "comp_equalized"),
       "C_orig_random": ("original", "random"), "D_compeq_compeq": ("comp_equalized", "comp_equalized"),
       "E_random_random": ("random", "random"), "F_random_compeq": ("random", "comp_equalized"),
       "G_orig_gcmatch": ("original", "gc_match"), "H_gcmatch_gcmatch": ("gc_match", "gc_match")}
DIF = {"S_shortcut_worth": ("A_orig_orig", "B_orig_compeq"),
       "T_test_imbalance_ctl": ("C_orig_random", "B_orig_compeq"),
       "C_minus_A": ("C_orig_random", "A_orig_orig"),
       "U_train_size_ctl": ("E_random_random", "D_compeq_compeq"),
       "E_minus_F": ("E_random_random", "F_random_compeq"),
       "S_gc": ("A_orig_orig", "G_orig_gcmatch")}


def test_contrasts():
    for tag in ("", "_posthoc"):
        c = pd.read_csv(PRE + f"{tag}_contrasts.csv")
        m = pd.read_csv(PRE + f"{tag}_metrics.csv")
        n = 0
        for (model, ds), q in c.groupby(["model", "dataset"]):
            per = {}
            for name, (ta, ea) in LEV.items():
                x = m[(m.model == model) & (m.dataset == ds) & (m.train_arm == ta) & (m.eval_arm == ea)]
                if len(x):
                    per[name] = x.sort_values("seed").set_index("seed")["auroc"]
            est = q.set_index("contrast")
            for name in LEV:
                if name in est.index:
                    assert abs(est.loc[name, "estimate"] - per[name].mean()) < 2e-6, (model, ds, name)
                    n += 1
            for name, (a, b) in DIF.items():
                if name in est.index:
                    assert abs(est.loc[name, "estimate"] - (per[a] - per[b]).mean()) < 2e-6, (model, ds, name)
                    n += 1
            if "R_retention" in est.index:
                r = ((per["E_random_random"] - 0.5) / (per["A_orig_orig"] - 0.5)).mean()
                # the metric rows are stored to 6 decimals and the ratio divides by (A-0.5) >= ~0.15,
                # so rounding alone can move it by a few 1e-6 (observed max 2.7e-6)
                assert abs(est.loc["R_retention", "estimate"] - r) < 1e-5, (model, ds)
                n += 1
            assert (q.ci_lo <= q.ci_hi + 1e-12).all()
        print(f"ok  {'post hoc' if tag else 'pre-registered'} contrasts: {n} estimates recomputed from the metric rows")


def test_tables():
    for tag in ("", "_posthoc"):
        t = pd.read_csv(PRE + f"{tag}_paper_table.csv")
        for r in t.itertuples():
            if pd.notna(getattr(r, "S_est", np.nan)):
                assert abs(r.S_est - (r.A_auroc_mean - r.B_auroc_mean)) < 2e-6, (r.model, r.dataset)
            if pd.notna(getattr(r, "frac_lost", np.nan)):
                assert abs(r.frac_lost - (r.A_auroc_mean - r.B_auroc_mean) / (r.A_auroc_mean - 0.5)) < 2e-5
                assert abs(r.frac_lost_random - (r.A_auroc_mean - r.C_auroc_mean) / (r.A_auroc_mean - 0.5)) < 2e-5
                assert r.frac_lost_lo <= r.frac_lost_hi
        print(f"ok  {'post hoc' if tag else 'pre-registered'} paper table: S = A-B, shares lost = (A-X)/(A-.5)")
    pre = pd.read_csv(PRE + "_paper_table.csv").set_index(["model", "dataset"])
    ph = pd.read_csv(PRE + "_posthoc_paper_table.csv").set_index(["model", "dataset"])
    comp = pd.read_csv(PRE + "_compare.csv")
    for r in comp.itertuples():
        src = ph if bool(r.posthoc) else pre
        s = src.loc[(r.model, r.dataset)]
        for k in ("A_auroc_mean", "B_auroc_mean", "C_auroc_mean", "S_est", "frac_lost"):
            if k in comp.columns and pd.notna(getattr(r, k)):
                assert abs(getattr(r, k) - s[k]) < 2e-6, (r.model, r.dataset, k)
    pp = pd.read_csv(PRE + "_compare_paired.csv")
    assert (abs(pp.difference - (pp.finetuned_value - pp.baseline_value)) < 2e-6).all()
    assert (pp.ci_lo <= pp.ci_hi).all()
    print(f"ok  compare.csv ({len(comp)} rows) matches its source tables; {len(pp)} paired differences consistent")


if __name__ == "__main__":
    mem = membership()
    P = load_preds(mem)
    test_inventory(P)
    test_metrics(P, mem)
    test_summary()
    test_contrasts()
    test_tables()
    print("all output cross-checks passed")
