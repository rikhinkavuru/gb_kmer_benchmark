#!/usr/bin/env python3
"""Paper-facing tables for the end-to-end fine-tuning result (R1 comment 2). Torch-free.
Reads the outputs of eval_ft_v2.py (run it first) and the per-sequence predictions.

Nothing here changes the pre-registered analysis in eval_ft_v2.py; it re-arranges its numbers
into one row per model x dataset and adds two DESCRIPTIVE (post hoc, not decision-rule) quantities:
  frac_lost  = (A - B) / (A - 0.5)   share of the released-split model's above-chance AUROC that
               disappears when only the negatives are swapped for composition-equalized ones
               (same weights, same positives). 95% CI from the same joint bootstrap as
               eval_ft_v2 (1000 resamples of the original test sequences, seed 42, seed-mean
               within each resample).
  frac_lost_random = (A - C) / (A - 0.5)   the same for the size-matched random-negative test
               (the imbalance control; expected ~0).

Outputs
  results/upgrades/fm_finetune_v2_paper_table.csv   one row per model x dataset
  results/upgrades/fm_finetune_v2_model_contrast.csv   between fine-tuned models on the same dataset
      (HyenaDNA minus DNABERT-2, and any GPU-deferred model once its units exist) for S and
      (A-B)/(A-0.5), paired bootstrap CI (post hoc, descriptive; added 2026-09-24 after the first
      DNABERT-2 unit finished, before any further DNABERT-2 unit)
  results/upgrades/fm_finetune_v2_v1_vs_v2.csv      the superseded v1 run (fm_finetune_summary.csv,
      single-class-batch defect, no saved predictions) next to the v2 re-run of the same arms

2026-09-26 (presentation only, no analysis change): the model -> fine-tuning-mode labels cover every
model the pipeline can produce (the two local fine-tunes, the GPU-deferred dnabert2full / ntv2_500m,
and the post hoc frozen / k-mer baselines of posthoc_ft_v2.py); `paper_table()` is a function so
posthoc_ft_v2.py builds its tables with the identical code.

Usage: python src/upgrades/fm_finetune/summarize_ft_v2.py   (torch-free venv)
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_ft_v2 as ev  # noqa: E402

RES = ev.OUT
PRE = os.path.join(RES, "fm_finetune_v2")
O, CE, RD, GC = "original", "comp_equalized", "random", "gc_match"
LEVELS = [("A", O, O), ("B", O, CE), ("C", O, RD), ("D", CE, CE), ("E", RD, RD), ("F", RD, CE),
          ("G", O, GC), ("H", GC, GC), ("I", CE, O), ("J", RD, O)]
DIFFS = [("S", "S_shortcut_worth"), ("T", "T_test_imbalance_ctl"), ("U", "U_train_size_ctl"),
         ("R", "R_retention"), ("CmA", "C_minus_A"), ("Sgc", "S_gc")]
# model -> (display name, how it is trained). Fine-tuned models first, in display order.
MODEL_INFO = {
    "hyenadna": ("HyenaDNA-tiny-16k-d128", "full fine-tune (all 436,354 params)"),
    "dnabert2": ("DNABERT-2-117M", "LoRA r=8 + head"),
    "dnabert2full": ("DNABERT-2-117M", "full fine-tune (GPU-deferred)"),
    "ntv2_500m": ("NT-v2-500M", "LoRA r=8 + head (GPU-deferred)"),
    "hyenadna_frozen_mean": ("HyenaDNA-tiny-16k-d128", "frozen, mean-pool + LightGBM head"),
    "hyenadna_frozen_sep": ("HyenaDNA-tiny-16k-d128", "frozen, [SEP] + LightGBM head"),
    "dnabert2_frozen_mean": ("DNABERT-2-117M", "frozen, mean-pool + LightGBM head"),
    "dnabert2_frozen_cls": ("DNABERT-2-117M", "frozen, [CLS] + LightGBM head"),
    "kmer_lgbm": ("k-mer spectrum", "LightGBM (paper classifier)"),
}
FT_MODELS = ("hyenadna", "dnabert2", "dnabert2full", "ntv2_500m")


def model_family(m):
    if m in FT_MODELS:
        return "fine-tuned"
    if "_frozen_" in m:
        return "frozen probe"
    if m == "kmer_lgbm":
        return "k-mer"
    return "other"


def ratio_boot(g, mem, model, ds):
    pairs = [(O, O), (O, CE), (O, RD)]
    if not all((model, ds, ta) in g for ta, _ in pairs):
        return {}
    bt, seeds = ev.joint_boot_auc(g, mem, model, ds, pairs)
    pt = {pr: ev.point_auc(g, mem, model, ds, pr[0], pr[1], seeds) for pr in pairs}
    out = {}
    for name, other in (("frac_lost", (O, CE)), ("frac_lost_random", (O, RD))):
        a_pt, o_pt = pt[(O, O)].mean(), pt[other].mean()
        bb = (bt[(O, O)] - bt[other]) / (bt[(O, O)] - 0.5)
        out[name] = (a_pt - o_pt) / (a_pt - 0.5)
        out[name + "_lo"] = float(np.nanpercentile(bb, 2.5))
        out[name + "_hi"] = float(np.nanpercentile(bb, 97.5))
    return out


def model_contrast(g, mem, models=FT_MODELS):
    """Post hoc, descriptive: fine-tuned model X minus fine-tuned model Y on the same dataset, for
    S = A - B and the share lost (A-B)/(A-0.5). Paired bootstrap: joint_boot_auc draws its
    resamples from RandomState(42) over the same n test sequences, so resample b is identical for
    both models. The original pair (hyenadna minus dnabert2) keeps its contrast labels."""
    rows = []
    for ds in sorted({k[1] for k in g}):
        have = [m for m in models if all((m, ds, ta) in g for ta in (O, CE))]
        if len(have) < 2:
            continue
        res = {}
        for m in have:
            bt, seeds = ev.joint_boot_auc(g, mem, m, ds, [(O, O), (O, CE)])
            pt = {pr: ev.point_auc(g, mem, m, ds, pr[0], pr[1], seeds) for pr in [(O, O), (O, CE)]}
            a, b = pt[(O, O)].mean(), pt[(O, CE)].mean()
            res[m] = dict(S=a - b, F=(a - b) / (a - 0.5), bS=bt[(O, O)] - bt[(O, CE)],
                          bF=(bt[(O, O)] - bt[(O, CE)]) / (bt[(O, O)] - 0.5), seeds=seeds)
        for m1, m2 in itertools.combinations(have, 2):
            h, d = res[m1], res[m2]
            if (m1, m2) == ("hyenadna", "dnabert2"):
                labs = (("S", "S_hyena_minus_S_dnabert2"), ("F", "fraclost_hyena_minus_dnabert2"))
            else:
                labs = (("S", f"S_{m1}_minus_S_{m2}"), ("F", f"fraclost_{m1}_minus_{m2}"))
            for q, lab in labs:
                bb = h["b" + q] - d["b" + q]
                row = dict(dataset=ds, contrast=lab, estimate=h[q] - d[q],
                           ci_lo=float(np.nanpercentile(bb, 2.5)),
                           ci_hi=float(np.nanpercentile(bb, 97.5)),
                           model_1=m1, model_2=m2, value_1=h[q], value_2=d[q],
                           seeds_1=",".join(map(str, h["seeds"])), seeds_2=",".join(map(str, d["seeds"])))
                if (m1, m2) == ("hyenadna", "dnabert2"):      # legacy column names
                    row.update(hyenadna=h[q], dnabert2=d[q], seeds_hyenadna=row["seeds_1"],
                               seeds_dnabert2=row["seeds_2"])
                rows.append(row)
    return pd.DataFrame(rows)


def paper_table(summ, ct, vd, rt, g, mem):
    rows = []
    for (model, ds), c in ct.groupby(["model", "dataset"], sort=False):
        r = dict(model=model, dataset=ds)
        rr = rt[(rt.model == model) & (rt.dataset == ds)]
        name, how = MODEL_INFO.get(model, (model, "?"))
        r["model_name"] = name
        r["family"] = model_family(model)
        r["finetune"] = how
        tp = rr["n_params_trainable"].iloc[0] if len(rr) else np.nan
        tt = rr["n_params_total"].iloc[0] if len(rr) else np.nan
        r["n_params_trainable"] = int(tp) if pd.notna(tp) else np.nan
        r["n_params_total"] = int(tt) if pd.notna(tt) else np.nan
        r["train_arms_seeds"] = ";".join(f"{ta}:{','.join(map(str, sorted(s.seed)))}"
                                         for ta, s in rr.groupby("train_arm"))
        for key, ta, ea in LEVELS:
            s = summ[(summ.model == model) & (summ.dataset == ds) & (summ.train_arm == ta)
                     & (summ.eval_arm == ea)]
            if s.empty:
                continue
            s = s.iloc[0]
            r[f"{key}_auroc_mean"] = s.auroc_mean
            r[f"{key}_auroc_sd"] = s.auroc_sd
            r[f"{key}_auroc_lo"] = s.auroc_seedmean_lo
            r[f"{key}_auroc_hi"] = s.auroc_seedmean_hi
            r[f"{key}_n_seeds"] = int(s.n_seeds)
            r[f"{key}_balres_auroc"] = s.bal_auroc_mean
            r[f"{key}_balres_mcc"] = s.bal_mcc_mean
            r[f"{key}_balres_bal_acc"] = s.bal_bal_acc_mean
            r[f"{key}_bal_acc"] = s.bal_acc_mean
            r[f"{key}_mcc"] = s.mcc_mean
            r[f"{key}_ap"] = s.ap_mean
            r[f"{key}_ap_baseline"] = s.prevalence
            r[f"{key}_n_test"] = int(s.n_test)
            r[f"{key}_fmt"] = (f"{s.auroc_mean:.3f}" + (f"±{s.auroc_sd:.3f}" if s.n_seeds > 1 else "")
                               + f" [{s.auroc_seedmean_lo:.3f},{s.auroc_seedmean_hi:.3f}]")
        for key, cname in DIFFS:
            x = c[c.contrast == cname]
            if x.empty:
                continue
            x = x.iloc[0]
            r[f"{key}_est"] = x.estimate
            r[f"{key}_seed_sd"] = x.seed_sd
            r[f"{key}_lo"] = x.ci_lo
            r[f"{key}_hi"] = x.ci_hi
            r[f"{key}_fmt"] = f"{x.estimate:.3f} [{x.ci_lo:.3f},{x.ci_hi:.3f}]"
        r.update(ratio_boot(g, mem, model, ds))
        v = vd[(vd.model == model) & (vd.dataset == ds)]
        if len(v):
            for col in v.columns:
                if col.startswith("rule"):
                    r[col] = v.iloc[0][col]
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=PRE, help="eval_ft_v2 --out-prefix (inputs AND outputs)")
    ap.add_argument("--pred-dir", default=ev.PRED_DIR)
    a = ap.parse_args()
    PRE_ = a.prefix
    summ = pd.read_csv(PRE_ + "_summary.csv")
    ct = pd.read_csv(PRE_ + "_contrasts.csv")
    vd = pd.read_csv(PRE_ + "_verdicts.csv")
    rt = pd.read_csv(PRE_ + "_runs.csv")
    mem = ev.load_membership()
    g = ev.group_runs(ev.load_runs(a.pred_dir))
    tab = paper_table(summ, ct, vd, rt, g, mem)
    tab.round(6).to_csv(PRE_ + "_paper_table.csv", index=False)
    show = ["model", "dataset", "n_params_trainable", "A_fmt", "B_fmt", "C_fmt", "D_fmt", "E_fmt",
            "F_fmt", "S_fmt", "T_fmt", "U_fmt", "R_fmt", "frac_lost"]
    pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 40)
    print(tab[[c for c in show if c in tab.columns]].to_string(index=False))
    mc = model_contrast(g, mem)
    if len(mc):
        mc.round(6).to_csv(PRE_ + "_model_contrast.csv", index=False)
        print(mc.drop(columns=[c for c in ("hyenadna", "dnabert2", "seeds_hyenadna", "seeds_dnabert2")
                               if c in mc.columns]).round(3).to_string(index=False))

    # ---- v1 (superseded) vs v2, HyenaDNA on cohn, same (train, eval) arms
    v1p = os.path.join(RES, "fm_finetune_summary.csv")
    if os.path.exists(v1p):
        v1 = pd.read_csv(v1p)
        out = []
        for _, a_ in v1.iterrows():
            s = summ[(summ.model == "hyenadna") & (summ.dataset == a_.dataset)
                     & (summ.train_arm == a_.train_arm) & (summ.eval_arm == a_.eval_arm)]
            out.append(dict(dataset=a_.dataset, train_arm=a_.train_arm, eval_arm=a_.eval_arm,
                            v1_auroc_mean=a_.auroc_mean, v1_auroc_sd=a_.auroc_sd, v1_n_seeds=a_.n_seeds,
                            v2_auroc_mean=s.auroc_mean.iloc[0] if len(s) else np.nan,
                            v2_auroc_sd=s.auroc_sd.iloc[0] if len(s) else np.nan,
                            v2_n_seeds=int(s.n_seeds.iloc[0]) if len(s) else 0))
        vv = pd.DataFrame(out)
        vv["note"] = ("v1 = run_fm_finetune.py (class-sorted lists + stable length sort -> nearly "
                      "single-class mini-batches; no per-sequence predictions saved); v2 = run_ft_v2.py "
                      "(batches permuted before the length sort); otherwise identical recipe")
        vv.round(6).to_csv(PRE_ + "_v1_vs_v2.csv", index=False)
        print(vv.drop(columns="note").round(3).to_string(index=False))
    print("wrote", PRE_ + "_paper_table.csv", "and", PRE_ + "_v1_vs_v2.csv")


if __name__ == "__main__":
    main()
