#!/usr/bin/env python3
"""POST HOC analyses of the end-to-end fine-tuning runs (torch-free). Added 2026-09-26, after every
one of the 54 fine-tuning units had been trained and scored by the pre-registered eval_ft_v2.py.
Nothing here alters the pre-registered outputs (results/upgrades/fm_finetune_v2_{metrics,summary,
contrasts,verdicts,...}.csv); every quantity below is written to separate *_posthoc_* / *_compare*
files and is labelled post hoc wherever it is reported.

P1  Leak-free nt_enhancers. 100 of its 400 released test sequences are verbatim copies of training
    sequences, all positives (results/upgrades/nt_enhancers_duplication.csv, re-derived here by exact
    string match against the verified split files). A model that memorizes them gets them right for
    free, which inflates A (the released-split AUROC) and every eval arm (all arms share the
    positives). The derived dataset `nt_enhancers_leakfree` drops those 100 test sequences from
    every eval arm (300 test sequences: 100 positives / 200 negatives; comp_eq and random arms 100 /
    36) and re-runs the IDENTICAL pre-registered computations (per-run metrics, seed-aggregated
    summary, contrasts S/T/U/R, rules 1-3) with the bootstrap resampling the 300 leak-free
    sequences. Training is unchanged (the 107 train copies stay in; only evaluation is leak-free:
    a memorized copy can inflate only its own test row, and those rows are gone).
P2  Baselines on the identical arms (baselines_ft_v2.py): the k-mer LightGBM and frozen-FM
    LightGBM heads, fit on each fine-tuning train arm and scored on every original test sequence,
    run through the same eval functions, on all three datasets and on nt_enhancers_leakfree.
P3  Model-family comparison (fm_finetune_v2_compare.csv): for every model x dataset, the share of
    above-chance AUROC the SAME weights lose when only the test negatives change,
        frac_lost        = (A - B) / (A - 0.5)   composition-equalized negatives
        frac_lost_random = (A - C) / (A - 0.5)   size-matched random negatives (control)
    plus the pre-registered S, T, U, R. Paired differences (fine-tuned minus frozen probe / k-mer
    on the same dataset) use the joint bootstrap of eval_ft_v2 (RandomState(42) over the same test
    sequences, so resample b is identical for every model); 95% percentile CIs, 1000 resamples.
    Descriptive; no decision rule is attached post hoc.

Outputs (results/upgrades/)
  fm_finetune_v2_leakfree_dups.csv                        the duplicated test indices (P1)
  fm_finetune_v2_posthoc_{runs,metrics,summary,contrasts,verdicts,paper_table}.csv
        P1 + P2: baselines on cohn / nt_enhancers / drosophila / nt_enhancers_leakfree and the
        fine-tuned models on nt_enhancers_leakfree
  fm_finetune_v2_posthoc_score_composition{,_summary}.csv score-composition coupling of the baselines
  fm_finetune_v2_compare.csv            P3, one row per model x dataset (fine-tuned rows taken from
                                        the pre-registered paper table; leak-free and baseline rows
                                        from the post hoc one)
  fm_finetune_v2_compare_paired.csv     P3 paired differences, fine-tuned minus each baseline
Usage: bibm/lowcpu.sh <torch-free python> src/upgrades/fm_finetune/posthoc_ft_v2.py
       (run eval_ft_v2.py, summarize_ft_v2.py and baselines_ft_v2.py first)
"""
import copy
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import baselines_ft_v2 as bl  # noqa: E402
import eval_ft_v2 as ev  # noqa: E402
import summarize_ft_v2 as sm  # noqa: E402

PRE = os.path.join(ev.OUT, "fm_finetune_v2")
PH = PRE + "_posthoc"
NT, NTL = "nt_enhancers", "nt_enhancers_leakfree"
O, CE, RD, GC = "original", "comp_equalized", "random", "gc_match"


def duplicated_test(ds=NT):
    tr = pd.read_csv(os.path.join(ev.SPLIT_DIR, f"{ds}__train.csv.gz"))
    te = pd.read_csv(os.path.join(ev.SPLIT_DIR, f"{ds}__test.csv.gz"))
    norm = lambda s: s.astype(str).str.strip().str.upper()   # the benchmark's own normalization
    trs = set(norm(tr["sequence"]))
    dup = norm(te["sequence"]).isin(trs).to_numpy()
    ref = pd.read_csv(os.path.join(ev.OUT, "nt_enhancers_duplication.csv")).iloc[0]
    assert int(dup.sum()) == int(ref["train_test_verbatim"]) == 100, int(dup.sum())
    assert (te["label"].to_numpy()[dup] == 1).all(), "duplicates are expected to be positives only"
    n_train_copies = int(norm(tr["sequence"]).isin(set(norm(te["sequence"])[dup])).sum())
    out = pd.DataFrame(dict(dataset=ds, test_idx=np.where(dup)[0], label=te["label"].to_numpy()[dup]))
    out.to_csv(PRE + "_leakfree_dups.csv", index=False)
    print(f"[P1] {ds}: {int(dup.sum())} test sequences are verbatim training sequences "
          f"(labels: {sorted(set(out.label))}); {n_train_copies} training rows equal one of them")
    return dup, n_train_copies


def leakfree(runs, mem, dup):
    keep = ~dup
    y = mem[NT]["y"]
    mem2 = dict(mem)
    mem2[NTL] = dict(y=y[keep], mask={a: m[keep] for a, m in mem[NT]["mask"].items()})
    out = []
    for r in runs:
        if r["meta"]["dataset"] != NT:
            continue
        rr = dict(meta=copy.deepcopy(r["meta"]), p=r["p"][keep], idx=np.arange(int(keep.sum())),
                  y=r["y"][keep])
        rr["meta"]["dataset"] = NTL
        out.append(rr)
    return out, mem2


def compare(g, mem, tab_all):
    """P3: per model x dataset shares lost, and paired differences fine-tuned minus baselines."""
    pairs = [(O, O), (O, CE), (O, RD)]
    boots, rows, prow = {}, [], []
    for (model, ds) in sorted({(k[0], k[1]) for k in g}):
        if not all((model, ds, ta) in g for ta, _ in pairs):
            continue
        bt, seeds = ev.joint_boot_auc(g, mem, model, ds, pairs)
        fl = (bt[(O, O)] - bt[(O, CE)]) / (bt[(O, O)] - 0.5)
        flr = (bt[(O, O)] - bt[(O, RD)]) / (bt[(O, O)] - 0.5)
        boots[(model, ds)] = dict(fl=fl, flr=flr, S=bt[(O, O)] - bt[(O, CE)], seeds=seeds)
    for ds in sorted({k[1] for k in boots}):
        fts = [m for m in sm.FT_MODELS if (m, ds) in boots]
        bls = [m for (m, d) in boots if d == ds and m not in sm.FT_MODELS]
        for f in fts:
            for b in sorted(bls):
                x, z = boots[(f, ds)], boots[(b, ds)]
                tf = tab_all[(tab_all.model == f) & (tab_all.dataset == ds)].iloc[0]
                tb = tab_all[(tab_all.model == b) & (tab_all.dataset == ds)].iloc[0]
                for q, lab, pf, pb in (("fl", "frac_lost", tf.frac_lost, tb.frac_lost),
                                       ("S", "S_shortcut_worth", tf.S_est, tb.S_est)):
                    bb = x[q] - z[q]
                    prow.append(dict(dataset=ds, finetuned=f, baseline=b, quantity=lab,
                                     finetuned_value=pf, baseline_value=pb, difference=pf - pb,
                                     ci_lo=float(np.nanpercentile(bb, 2.5)),
                                     ci_hi=float(np.nanpercentile(bb, 97.5)),
                                     note="post hoc, descriptive; paired joint bootstrap"))
    return pd.DataFrame(prow)


def main():
    mem = ev.load_membership()
    ft = ev.load_runs(ev.PRED_DIR)
    base = ev.load_runs(bl.OUT_DIR)
    print(f"[posthoc] {len(ft)} fine-tuning runs, {len(base)} baseline runs")
    # >= : the 54 local units, plus any GPU hand-back units (dnabert2full, ntv2_500m). Was `== 54`
    # for the 2026-09-26 run (54 units present); relaxed afterwards so the hand-back needs no edit.
    assert len(ft) >= 54, len(ft)
    dup, n_copies = duplicated_test()
    lf, mem2 = leakfree(ft + base, mem, dup)
    runs = base + lf
    print(f"[posthoc] scoring {len(runs)} runs (baselines x 3 datasets + everything on {NTL})")
    rt = ev.run_table(runs)
    rt.to_csv(PH + "_runs.csv", index=False)
    per = ev.per_run_rows(runs, mem2)
    per.round(6).to_csv(PH + "_metrics.csv", index=False)
    summ = ev.summarize(per, runs, mem2)
    summ.round(6).to_csv(PH + "_summary.csv", index=False)
    ct, vd = ev.contrasts(runs, mem2)
    ct.round(6).to_csv(PH + "_contrasts.csv", index=False)
    vd.to_csv(PH + "_verdicts.csv", index=False)
    sc, scs = ev.score_composition(base, mem)
    sc.round(6).to_csv(PH + "_score_composition.csv", index=False)
    scs.round(6).to_csv(PH + "_score_composition_summary.csv", index=False)
    g = ev.group_runs(runs)
    tab = sm.paper_table(summ, ct, vd, rt, g, mem2)
    tab["posthoc"] = True
    tab.round(6).to_csv(PH + "_paper_table.csv", index=False)

    # ---- P3: one comparison table over the pre-registered fine-tuned rows + post hoc rows
    pre = pd.read_csv(PRE + "_paper_table.csv")
    pre["posthoc"] = False
    for c in ("model_name", "family", "finetune"):
        if c not in pre.columns:
            pre[c] = np.nan
    pre["model_name"] = pre["model"].map(lambda m: sm.MODEL_INFO.get(m, (m, "?"))[0])
    pre["family"] = pre["model"].map(sm.model_family)
    pre["finetune"] = pre["model"].map(lambda m: sm.MODEL_INFO.get(m, (m, "?"))[1])
    allt = pd.concat([pre, tab], ignore_index=True, sort=False)
    ds_order = {"human_enhancers_cohn": 0, NT: 1, NTL: 2, "drosophila_enhancers_stark": 3}
    m_order = {m: i for i, m in enumerate(sm.MODEL_INFO)}
    allt["_d"] = allt.dataset.map(ds_order); allt["_m"] = allt.model.map(m_order)
    allt = allt.sort_values(["_d", "_m"]).drop(columns=["_d", "_m"])
    keep = ["dataset", "model", "model_name", "family", "finetune", "posthoc", "train_arms_seeds"]
    for k in ("A", "B", "C", "D", "E", "F", "G", "H"):
        keep += [f"{k}_auroc_mean", f"{k}_auroc_sd", f"{k}_auroc_lo", f"{k}_auroc_hi", f"{k}_n_test",
                 f"{k}_balres_auroc", f"{k}_balres_bal_acc", f"{k}_balres_mcc", f"{k}_ap",
                 f"{k}_ap_baseline"]
    for k in ("S", "T", "U", "R", "Sgc"):
        keep += [f"{k}_est", f"{k}_seed_sd", f"{k}_lo", f"{k}_hi"]
    keep += ["frac_lost", "frac_lost_lo", "frac_lost_hi", "frac_lost_random", "frac_lost_random_lo",
             "frac_lost_random_hi", "rule1_rides_shortcut", "rule2_drop_not_test_imbalance",
             "rule3_collapse_cause", "rule5_aligned_control_not_destroyed"]
    comp = allt[[c for c in keep if c in allt.columns]]
    comp.round(6).to_csv(PRE + "_compare.csv", index=False)
    # paired differences need the bootstrap arrays of every model on every dataset
    g_all = ev.group_runs(ft + base + lf)
    paired = compare(g_all, mem2, allt)
    paired.round(6).to_csv(PRE + "_compare_paired.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 30)
    show = ["dataset", "model", "A_auroc_mean", "B_auroc_mean", "C_auroc_mean", "D_auroc_mean",
            "E_auroc_mean", "S_est", "frac_lost", "frac_lost_lo", "frac_lost_hi", "frac_lost_random"]
    print(comp[show].round(3).to_string(index=False))
    print(paired[paired.quantity == "frac_lost"].round(3).to_string(index=False))
    print(vd.to_string(index=False))
    print("wrote", PH + "_*.csv", PRE + "_compare.csv", PRE + "_compare_paired.csv")


if __name__ == "__main__":
    main()
