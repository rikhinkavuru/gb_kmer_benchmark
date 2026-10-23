#!/usr/bin/env python3
"""Interpretation text for run_extra.py (X1-X10) and run_labels_addendum.py (A1-A4), generated ONLY from the CSVs.
Applies each pre-registered decision rule of run_extra.py mechanically and prints the verdict next to the numbers.
Writes results/upgrades/homology_recluster_extra_interpretation.txt. No computation beyond reading/formatting.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _hc  # noqa: E402

R, P = _hc.RES, _hc.PREFIX


def rd(name):
    p = os.path.join(R, f"{P}_{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def ci(lo, hi, f="{:.3f}"):
    return "" if lo != lo else ("[" + f.format(lo) + "," + f.format(hi) + "]")


ORDER = {"original_released": 0, "original": 1, "id80": 2, "id50": 3}


def srt(df, col="data"):
    """Stable display order: tasks as in _hc.TASKS, data original -> id80 -> id50."""
    if df is None or col not in df.columns:
        return df
    t = {k: i for i, k in enumerate(_hc.TASKS)}
    key = df["task"].map(t).fillna(99) * 10 + df[col].map(ORDER).fillna(9)
    return df.iloc[np.argsort(key.to_numpy(), kind="stable")]


FT_MIN_NEG = 10   # not-evaluable rule for an FT eval arm, fixed in run_stage_ft_hc.sh's header before any HC unit was trained
FT_DEP = {"A_orig_orig": {"original"}, "B_orig_compeq": {"comp_equalized"}, "C_orig_random": {"random"},
          "D_compeq_compeq": {"comp_equalized"}, "E_random_random": {"random"}, "F_random_compeq": {"comp_equalized"},
          "S_shortcut_worth": {"original", "comp_equalized"}, "T_test_imbalance_ctl": {"random", "comp_equalized"},
          "C_minus_A": {"random", "original"}, "U_train_size_ctl": {"random", "comp_equalized"},
          "R_retention": {"random", "original"}, "E_minus_F": {"random", "comp_equalized"},
          "G_orig_gcmatch": {"gc_match"}, "H_gcmatch_gcmatch": {"gc_match"}, "S_gc": {"original", "gc_match"}}


def _ft_rules(con, man, model, datasets):
    """fm_finetune/eval_ft_v2.py decision rules 1-3 and 5 (docstring, fixed in advance), re-applied to its contrasts
    with the not-evaluable guard: a contrast is n/e if any eval arm it uses keeps < FT_MIN_NEG test negatives."""
    nev = {}
    if man is not None:
        te = man[man.split == "test"]
        for _, r in te[te.n_neg < FT_MIN_NEG].iterrows():
            nev.setdefault(r.dataset, set()).add(r.arm)
    out = {}
    for ds in datasets:
        g = con[(con.model == model) & (con.dataset == ds)].set_index("contrast")
        if not len(g):
            continue
        ok = lambda c, g=g, ds=ds: c in g.index and not (FT_DEP.get(c, set()) & nev.get(ds, set()))   # bind per dataset
        v = {}
        if ok("S_shortcut_worth") and ok("B_orig_compeq"):
            S, B = g.loc["S_shortcut_worth"], g.loc["B_orig_compeq"]
            v["rule1"] = bool(S.estimate > 0 and S.ci_lo > 0 and ((B.ci_lo <= 0.5 <= B.ci_hi) or B.estimate < 0.6))
        else:
            v["rule1"] = "n/e"
        if ok("T_test_imbalance_ctl") and ok("C_orig_random") and ok("A_orig_orig"):
            T, C_, A = g.loc["T_test_imbalance_ctl"], g.loc["C_orig_random"], g.loc["A_orig_orig"]
            v["rule2"] = bool(T.estimate > 0 and T.ci_lo > 0 and C_.ci_lo <= A.ci_hi and A.ci_lo <= C_.ci_hi)
        else:
            v["rule2"] = "n/e"
        if ok("U_train_size_ctl") and ok("R_retention"):
            U, Rr = g.loc["U_train_size_ctl"], g.loc["R_retention"]
            v["rule3"] = ("composition removal" if U.ci_lo > 0 and Rr.estimate >= 0.5 else
                          "size/imbalance contributes" if U.ci_lo > 0 else "size/imbalance NOT excluded")
        else:
            v["rule3"] = "n/e"
        v["nev_arms"] = sorted(nev.get(ds, set()))
        v["g"] = g
        v["ok"] = ok
        out[ds] = v
    d = out.get("drosophila_enhancers_stark")
    if d is not None:
        g = d["g"]
        comp = [ds for ds in ("human_enhancers_cohn", "nt_enhancers") if ds in out and out[ds]["ok"]("S_shortcut_worth")]
        ne_ = [ds for ds in ("human_enhancers_cohn", "nt_enhancers") if ds in out and not out[ds]["ok"]("S_shortcut_worth")]
        if all(c in g.index for c in ("B_orig_compeq", "D_compeq_compeq", "S_shortcut_worth")) and comp:
            Sd = g.loc["S_shortcut_worth"]
            d["rule5"] = bool(g.loc["B_orig_compeq"].ci_lo > 0.5 and g.loc["D_compeq_compeq"].ci_lo > 0.5
                              and all(Sd.ci_hi < out[ds]["g"].loc["S_shortcut_worth"].ci_lo for ds in comp))
        else:
            d["rule5"] = "n/e"
        d["rule5_vs"] = comp; d["rule5_ne"] = ne_
    return out


def ft_hc_section():
    """End-to-end fine-tuning on the id80 homology-clean splits (run_stage_ft_hc.sh) vs the provided splits
    (fm_finetune v2), under fm_finetune/eval_ft_v2.py's pre-registered rules. Reads CSVs only."""
    L = []; add = L.append
    hc, man = rd("ft_contrasts"), rd("ft_split_manifest")
    if hc is None:
        return L
    pp = os.path.join(R, "fm_finetune_v2_contrasts.csv"); pm = os.path.join(R, "fm_finetune_v2_split_manifest.csv")
    pv = pd.read_csv(pp) if os.path.exists(pp) else None
    pman = pd.read_csv(pm) if os.path.exists(pm) else None
    vhc = rd("ft_verdicts")
    DS = ["human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark"]
    add("\nFT-HC END-TO-END FINE-TUNING ON THE id80 HOMOLOGY-CLEAN SPLITS (HyenaDNA-tiny, full fine-tune, seeds 42-44;")
    add("  run_stage_ft_hc.sh: recipe = fm_finetune/run_ft_v2.py unchanged, run_ft_hc.py only redirects paths; scoring and")
    add("  decision rules = fm_finetune/eval_ft_v2.py unchanged, via eval_ft_hc.py). HC splits: prepare_ft_hc_splits.py (id80")
    add("  cluster-aware re-split + one representative per component x label; cleaning arms rebuilt inside the HC splits).")
    add("  Contrasts = seed-mean test AUROC (train arm -> eval arm), 95% CI from 1000 resamples of the test sequences.")
    add(f"  Not-evaluable rule (fixed in run_stage_ft_hc.sh before training): a contrast whose eval arm keeps < {FT_MIN_NEG} test negatives.")
    if man is not None:
        for ds, g in man[man.split == "test"].groupby("dataset", sort=False):
            add(f"  {ds:<28} HC test arms (pos/neg): " + ", ".join(f"{r.arm} {int(r.n_pos)}/{int(r.n_neg)}" for _, r in g.iterrows()))
    rh = _ft_rules(hc, man, "hyenadna", DS)
    rp = _ft_rules(pv, pman, "hyenadna", DS) if pv is not None else {}
    show = ["A_orig_orig", "B_orig_compeq", "C_orig_random", "D_compeq_compeq", "E_random_random", "S_shortcut_worth",
            "T_test_imbalance_ctl", "U_train_size_ctl", "R_retention", "S_gc"]
    for ds in DS:
        if ds not in rh:
            continue
        add(f"  {ds}")
        for c in show:
            def cell(v):
                if v is None or c not in v["g"].index:
                    return "      --      "
                if not v["ok"](c):
                    return "n/e (<10 test negs)"
                r = v["g"].loc[c]
                return f"{r.estimate:.3f}{ci(r.ci_lo, r.ci_hi)}"
            add(f"    {c:<22} HC {cell(rh[ds]):<24} provided {cell(rp.get(ds))}")
        for k, lab in (("rule1", "rule 1 rides the shortcut"), ("rule2", "rule 2 drop not test-imbalance"),
                       ("rule3", "rule 3 collapse cause")):
            add(f"    => FT-HC {ds} {lab}: HC {rh[ds][k]} | provided {rp.get(ds, {}).get(k, 'n/a')}")
        if ds == "drosophila_enhancers_stark":
            add(f"    => FT-HC drosophila rule 5 (aligned control not destroyed): HC {rh[ds].get('rule5')} (vs "
                f"{', '.join(rh[ds].get('rule5_vs', [])) or 'none'}; not evaluable vs {', '.join(rh[ds].get('rule5_ne', [])) or 'none'}) | "
                f"provided {rp.get(ds, {}).get('rule5', 'n/a')} (vs {', '.join(rp.get(ds, {}).get('rule5_vs', [])) or 'none'})")
    if vhc is not None:
        add("  eval_ft_hc.py's own mechanical verdicts (no not-evaluable guard): " + "; ".join(
            f"{r.dataset}: r1 {r.rule1_rides_shortcut}, r2 {r.rule2_drop_not_test_imbalance}, r3 {r.rule3_collapse_cause}"
            + (f", r5 {r.rule5_aligned_control_not_destroyed}" if r.rule5_aligned_control_not_destroyed == r.rule5_aligned_control_not_destroyed else "")
            for _, r in vhc.iterrows()))
    sc = rd("ft_score_composition_summary")
    psc = os.path.join(R, "fm_finetune_v2_score_composition_summary.csv")
    psc = pd.read_csv(psc) if os.path.exists(psc) else None
    if sc is not None:
        o = sc[(sc.model == "hyenadna") & (sc.train_arm == "original")].set_index("dataset")
        po = psc[(psc.model == "hyenadna") & (psc.train_arm == "original")].set_index("dataset") if psc is not None else None
        add("  score-composition coupling (secondary, descriptive; original-trained model, original test split): fraction of the")
        add("  model's above-chance AUROC recovered by the CV composition projection of its score, HC vs provided: " + "; ".join(
            f"{ds} {o.loc[ds, 'frac_above_chance_recovered_mean']:.2f} vs "
            + (f"{po.loc[ds, 'frac_above_chance_recovered_mean']:.2f}" if po is not None and ds in po.index else "n/a")
            for ds in DS if ds in o.index))
    add("  DEFERRED (not dropped): DNABERT-2-117M LoRA fine-tuning on the same HC splits (cohn + nt_enhancers, arms original/")
    add("  comp_equalized/random, seeds 42-44 = 18 units; process peak 3.5-3.8 GB > the laptop's per-job budget). Package:")
    add("  src/upgrades/homology_recluster/deferred/ (README.md, run_deferred_dnabert_hc.sh).")
    return L


def build():
    L = []; add = L.append
    add("=" * 110)
    add("HOMOLOGY RE-CLUSTERING -- EXTRA RE-RUNS (run_extra.py X1-X10, FT-HC fine-tuning) + LABEL ADDENDA (run_labels_addendum.py)")
    add("Every number is read from results/upgrades/homology_recluster_{extra_*,labels_addendum,class_redundancy,")
    add("route1_loto,paper_table}.csv. HC = homology-clean (id80 primary / id50 strict; cluster-aware re-split seed 42")
    add("+ one representative per component x label unless stated). Verdicts apply the rules pre-registered in the")
    add("run_extra.py docstring; A1-A4 are POST HOC and flagged as such.")
    add("=" * 110)

    x1 = srt(rd("extra_cleaning"))
    if x1 is not None:
        add("\nX1 ROUTE-3 CLEANING (TATA-flag / GC-match / composition-equalized + size-matched random-removal placebos)")
        for (task, dv), g in x1.groupby(["task", "data"], sort=False):
            g = g.set_index("arm")
            u = g.loc["uncleaned"]
            add(f"  {task:<27}{dv:<9} uncleaned TATA {u.tata_auroc:.3f}{ci(u.tata_lo, u.tata_hi)} GC {u.gc_auroc:.3f} "
                f"kmerMCC {u.kmer_mcc:.3f}{ci(u.kmer_mcc_lo, u.kmer_mcc_hi)} compMCC {u.comp_only_mcc:.3f}")
            for arm in ["tata_flag", "gc_match", "comp_equalized", "placebo_tata_flag", "placebo_gc_match", "placebo_comp_equalized"]:
                if arm not in g.index:
                    continue
                r = g.loc[arm]
                mcc = (f"kmerMCC {r.kmer_mcc:.3f}{ci(r.kmer_mcc_lo, r.kmer_mcc_hi)} (d {r.d_kmer_mcc:+.3f}{ci(r.d_kmer_mcc_lo, r.d_kmer_mcc_hi)})"
                       if r.get("status") == "ok" and r.d_kmer_mcc == r.d_kmer_mcc else f"[{r.get('status')}]")
                add(f"  {'':<36}{arm:<23} -{r.frac_neg_removed:.1%} negs  TATA {r.tata_auroc:.3f}{ci(r.tata_lo, r.tata_hi)} "
                    f"GC {r.gc_auroc:.3f}  {mcc}")
            role = g.iloc[0].role
            if role == "contaminated":
                ok = all(abs(g.loc[a].tata_auroc - .5) < abs(u.tata_auroc - .5) and g.loc[a].tata_lo > u.tata_hi
                         for a in ["tata_flag", "gc_match"])
                spec = all(not (g.loc[a].tata_lo > u.tata_hi or g.loc[a].tata_hi < u.tata_lo)
                           for a in ["placebo_tata_flag", "placebo_gc_match"])
                add(f"  => X1 cleaning {task} [{dv}]: TATA-flag/GC-match effect {'PRESERVED' if ok else 'NOT preserved'}; placebo-specific: {spec}")
            else:
                r = g.loc["tata_flag"]
                ok = r.frac_neg_removed <= 0.01 and abs(r.tata_auroc - u.tata_auroc) <= 0.01
                add(f"  => X1 cleaning {task} [{dv}] (clean control): removes <=1% negs and |dTATA|<=0.01: {ok}")

    s, t = srt(rd("extra_specificity")), srt(rd("extra_threshold"))
    if s is not None:
        add("\nX2 ROUTE-3 SPECIFICITY (control-motif panels; 4000/class cap; TATA threshold sweep)")
        for (task, dv), g in s.groupby(["task", "data"], sort=False):
            g = g.set_index("motif_class")
            line = "  ".join(f"{c.replace('_panel', '')} {g.loc[c].motif_only_auroc:.3f}"
                             + (f"+/-{g.loc[c].auroc_sd:.3f}" if g.loc[c].auroc_sd == g.loc[c].auroc_sd else "")
                             + ci(g.loc[c].auroc_ci_lo, g.loc[c].auroc_ci_hi) for c in g.index)
            add(f"  {task:<27}{dv:<9} {line}")
        if t is not None:
            for (task, dv), g in t.groupby(["task", "data"], sort=False):
                o = g[g.arm == "uncleaned"].sort_values("threshold_bits"); e = g[g.arm == "comp_equalized"].sort_values("threshold_bits")
                add(f"  {task:<27}{dv:<9} sweep 0.8-1.5 bits: uncleaned {o.tata_auroc.min():.3f}-{o.tata_auroc.max():.3f} "
                    f"(CI upper max {o.tata_ci_hi.max():.3f}); equalized {e.tata_auroc.min():.3f}-{e.tata_auroc.max():.3f}")
        for dv in ["original", "id80", "id50"]:
            try:
                pc = s[(s.task == "nt_promoter_tata") & (s.data == dv)].iloc[0]
                verdicts = []
                for task in ["human_enhancers_cohn", "nt_enhancers"]:
                    g = s[(s.task == task) & (s.data == dv)].set_index("motif_class")
                    rt, sc, rn = g.loc["real_TATA"], g.loc["scrambled_TBP_panel"], g.loc["random_IC_panel"]
                    tt = t[(t.task == task) & (t.data == dv)]
                    o = tt[tt.arm == "uncleaned"].set_index("threshold_bits"); e = tt[tt.arm == "comp_equalized"].set_index("threshold_bits")
                    ok = (rt.auroc_ci_hi < 0.5 and (abs(rn.motif_only_auroc - .5) <= 0.02 or rn.auroc_ci_lo <= 0.5 <= rn.auroc_ci_hi)
                          and sc.auroc_ci_hi < 0.5 and pc.auroc_ci_lo > 0.5 and (o.tata_ci_hi < 0.5).all()
                          and all(abs(e.loc[x].tata_auroc - .5) < abs(o.loc[x].tata_auroc - .5) for x in o.index))
                    verdicts.append(f"{task} {'PRESERVED' if ok else 'NOT preserved'}")
                add(f"  => X2 specificity [{dv}]: " + "; ".join(verdicts) + f" (positive control nt_promoter_tata {pc.motif_only_auroc:.3f}"
                    f"{ci(pc.auroc_ci_lo, pc.auroc_ci_hi)})")
            except (IndexError, KeyError):
                pass

    c = rd("extra_composition")
    if c is not None:
        c = c.iloc[np.argsort(c.task.map({k: i for i, k in enumerate(_hc.TASKS)}).to_numpy(), kind="stable")]
    if c is not None:
        nt = c.task.nunique()
        add(f"\nX3 COMPOSITION-ONLY BASELINE (LightGBM on 4 mono + 16 dinucleotide frequencies; same units as run_eval)"
            + ("" if nt == 15 else f" -- PARTIAL: {nt}/15 tasks (run stopped by the CPU emergency; see README 'Pending CPU jobs')"))
        add(f"  {'task':<34}{'unit':<18}{'kmerMCC':>8}{'compMCC':>8}{'  resid [CI]':<22}{'comp fraction [CI]':<24}")
        for _, r in c.iterrows():
            add(f"  {r.task:<34}{r.unit:<18}{r.kmer_mcc:>8.3f}{r.comp_mcc:>8.3f}  {r.residual_mcc:+.3f}{ci(r.residual_lo, r.residual_hi):<15}"
                f"{r.comp_fraction:>6.3f}{ci(r.comp_fraction_lo, r.comp_fraction_hi)}")
        # descriptive summary per unit (X3 is descriptive in the pre-registration; no verdict is claimed)
        for u in ["original", "test_purge_id80", "resplit_id80", "resplit_random"]:
            g = c[c.unit == u]
            if not len(g):
                continue
            beat = g[g.residual_lo > 0]; nb = g[~(g.residual_lo > 0)]
            add(f"  => X3 composition-only [{u}] (descriptive): k-mer model beats composition-only (residual CI > 0) on "
                f"{len(beat)}/{len(g)} tasks; not beaten: " + (", ".join(
                    f"{r.task} {r.residual_mcc:+.3f}{ci(r.residual_lo, r.residual_hi)}" for _, r in nb.iterrows()) or "none"))

    f = rd("extra_fm")
    if f is not None:
        add("\nX4 FROZEN-FM PROBE test_effect AUROC (headline cell: gc_match, mean pooling, LightGBM head) and placebo")
        h = f[(f.pooling == "mean") & (f["head"] == "lgbm")]
        for (m, task), g in h.groupby(["model", "task"], sort=False):
            parts = []
            for dv in ["original_released", "original", "id80", "id50"]:
                gg = g[g.data == dv].set_index("arm")
                if "gc_match" not in gg.index:
                    continue
                a = gg.loc["gc_match"]
                s_ = f"{dv}: {a.test_effect_auroc:+.3f}{ci(a.test_effect_auroc_lo, a.test_effect_auroc_hi)}"
                if "placebo_gc_match" in gg.index:
                    p_ = gg.loc["placebo_gc_match"]
                    s_ += f" (placebo {p_.test_effect_auroc:+.3f}{ci(p_.test_effect_auroc_lo, p_.test_effect_auroc_hi)})"
                parts.append(s_)
            add(f"  {m:<15}{task:<28}" + " | ".join(parts))
        def x4_rule(dv, pool_of):
            """pre-registered X4 rule on data dv; pool_of[model] = pooling of the headline cell (LightGBM head, gc_match)."""
            cell = f[(f.data == dv) & (f["head"] == "lgbm") & (f.arm == "gc_match")]
            cell = cell[cell.apply(lambda r: r.pooling == pool_of[r.model], axis=1)].set_index(["model", "task"])
            pl = f[(f.data == dv) & (f.arm == "placebo_gc_match")]
            contam = all(cell.loc[(m, t_)].test_effect_auroc < 0 and cell.loc[(m, t_)].test_effect_auroc_hi < 0
                         for m in ["HyenaDNA-tiny", "DNABERT-2-117M"] for t_ in ["human_enhancers_cohn", "nt_enhancers"])
            ctl = cell.loc[("HyenaDNA-tiny", "drosophila_enhancers_stark")]
            ctl_ok = bool(ctl.test_effect_auroc_lo <= 0 <= ctl.test_effect_auroc_hi)
            cover = (pl.test_effect_auroc_lo <= 0) & (pl.test_effect_auroc_hi >= 0)
            bad = pl[~cover]
            return contam, ctl, ctl_ok, int(cover.sum()), len(pl), bad

        pre = {"HyenaDNA-tiny": "mean", "DNABERT-2-117M": "mean"}
        for dv in ["original", "id80", "id50"]:
            if not len(f[f.data == dv]):
                continue
            try:
                contam, ctl, ctl_ok, nc, npl, bad = x4_rule(dv, pre)
            except KeyError:
                continue
            add(f"  => X4 FM probe [{dv}] PRE-REGISTERED rule (gc_match, mean pooling, LightGBM head): contaminated test_effect <0 (CI) "
                f"for both models: {contam}; drosophila {ctl.test_effect_auroc:+.3f}{ci(ctl.test_effect_auroc_lo, ctl.test_effect_auroc_hi)} "
                f"CI covers 0: {ctl_ok}; placebo_gc_match CIs covering 0: {nc}/{npl} => "
                f"{'PRESERVED' if contam and ctl_ok and nc == npl else 'NOT fully preserved'}"
                + ("" if dv == "original" else " (compare the same rule on 'original' data above)"))
            if len(bad):
                add("       placebo CIs excluding 0: " + "; ".join(
                    f"{r.model} {r.task} {r.pooling}/{r['head']} {r.test_effect_auroc:+.3f}{ci(r.test_effect_auroc_lo, r.test_effect_auroc_hi)}"
                    for _, r in bad.iterrows()))
        paper = {"HyenaDNA-tiny": "sep", "DNABERT-2-117M": "mean"}
        add("  POST HOC: the submitted paper's HyenaDNA headline (-0.048 cohn, -0.079 nt_enhancers, +0.005 drosophila) is the SEP-pooling")
        add("  LightGBM cell (fm_paired.csv), not mean pooling as the X4 rule assumed; the same rule on the paper's cells (HyenaDNA sep, DNABERT mean):")
        for dv in ["original_released", "original", "id80", "id50"]:
            try:
                contam, ctl, ctl_ok, nc, npl, bad = x4_rule(dv, paper)
            except KeyError:
                continue
            cell = f[(f.data == dv) & (f["head"] == "lgbm") & (f.arm == "gc_match")]
            cell = cell[cell.apply(lambda r: r.pooling == paper[r.model], axis=1)]
            add(f"  => [{dv}] " + "; ".join(f"{r.model} {r.task} {r.test_effect_auroc:+.3f}{ci(r.test_effect_auroc_lo, r.test_effect_auroc_hi)}"
                                           for _, r in cell.iterrows())
                + (f" | placebo CIs covering 0: {nc}/{npl}" if npl else "")
                + f" => contaminated <0: {contam}; drosophila flat: {ctl_ok}")
        allc = f[(f.arm == "gc_match") & f.task.isin(["human_enhancers_cohn", "nt_enhancers"])]
        for dv in ["original", "id80", "id50"]:
            g = allc[allc.data == dv]
            if len(g):
                add(f"  contaminated-task gc_match cells (all models x poolings x heads) with test_effect CI upper < 0 [{dv}]: "
                    f"{int((g.test_effect_auroc_hi < 0).sum())}/{len(g)}")
        rep = f[f.data == "original_released"]
        if len(rep):
            ref = []
            for fn in ("fm_paired.csv", "fm_paired_dnabert.csv"):
                p = os.path.join(R, fn)
                if os.path.exists(p):
                    d = pd.read_csv(p)
                    d["model"] = d.get("model", "HyenaDNA-tiny")
                    d["model"] = d["model"].fillna("HyenaDNA-tiny")
                    ref.append(d)
            if ref:
                ref = pd.concat(ref).rename(columns={"dataset": "task"})
                mm = rep.merge(ref[["model", "task", "arm", "pooling", "head", "test_effect_auroc"]],
                               on=["model", "task", "arm", "pooling", "head"], suffixes=("", "_pub"))
                add(f"  reproduction vs fm_paired*.csv (original_released): {len(mm)} cells, max |test_effect diff| = "
                    f"{np.abs(mm.test_effect_auroc - mm.test_effect_auroc_pub).max():.4f}")

    g5 = srt(rd("extra_graded"))
    if g5 is not None:
        add("\nX5 GRADED POSITIONAL RESCUE (nt_splice_sites_all, dMCC(B) - B1, paired bootstrap)")
        for dv, g in g5.groupby("data", sort=False):
            g = g.sort_values("n_bins")
            mono = bool(np.all(np.diff(g.delta_mcc.to_numpy()) > 0)); pos = bool((g.delta_lo > 0).all())
            add(f"  {dv:<9} " + "  ".join(f"B{int(r.n_bins)} {r.delta_mcc:+.3f}{ci(r.delta_lo, r.delta_hi)}" for _, r in g.iterrows())
                + f"  => X5 graded [{dv}] {'PRESERVED' if mono and pos else 'NOT preserved'} (monotone {mono}, all CI>0 {pos})")

    pos = rd("positional")
    if pos is not None:
        add("\nX6 POSITIONAL RESCUE dMCC(B8-B1) AT id50 (all tasks now at both thresholds)")
        for task in _hc.TASKS:
            g = pos[pos.task == task].set_index("data")
            add(f"  {task:<34}" + "  ".join(f"{dv}: {g.loc[dv, 'delta_mcc']:+.3f}{ci(g.loc[dv, 'delta_lo'], g.loc[dv, 'delta_hi'])}"
                                             for dv in ["original", "id80", "id50"] if dv in g.index))

    x7 = rd("extra_lr")
    if x7 is not None:
        add("\nX7 LR FLOOR vs LightGBM, tier agreement (solved >= 0.7 / partial >= 0.5 / hard)")
        for u, g in x7.groupby("unit", sort=False):
            add(f"  {u:<14} agree {int(g.tier_agree.sum())}/{len(g)}; disagreements: "
                + ", ".join(f"{r.task} (LR {r.lr_mcc:.3f} {r.lr_tier} / LGBM {r.lgbm_mcc:.3f} {r.lgbm_tier})" for _, r in g[~g.tier_agree].iterrows()))
        add("  => X7 LR floor (descriptive): LR/LightGBM tier agreement " + "; ".join(
            f"{u} {int(g.tier_agree.sum())}/{len(g)}" for u, g in x7.groupby("unit", sort=False)) + " (paper: 13/15 on the provided split)")

    x8 = rd("extra_modelseeds")
    if x8 is not None:
        add("\nX8 MODEL-SEED ROBUSTNESS (5 seeds, bootstrap-resampled training rows, fixed test)")
        for u, g in x8.groupby("unit", sort=False):
            add(f"  {u:<14} tier stable {int(g.tier_stable.sum())}/{len(g)}; max SD {g.mcc_sd.max():.4f}; "
                + ", ".join(f"{r.task} {r.mcc_mean:.3f}+/-{r.mcc_sd:.3f} {r.tier}" for _, r in g.iterrows()))
        units = list(x8.unit.unique())
        if "original" in units and len(units) > 1:
            b = x8[x8.unit == "original"]; stb = set(b[b.tier_stable].task)
            for u in [x for x in units if x != "original"]:
                a = x8[x8.unit == u]; sta = set(a[a.tier_stable].task)
                same, sup, sd_ok = sta == stb, stb <= sta, bool(a.mcc_sd.max() <= 0.02)
                add(f"  => X8 model seeds [{u}] PRE-REGISTERED rule (tier seed-stable for the same tasks as before AND max SD <= 0.02): "
                    f"same stable set {same} (before {len(stb)}/{len(b)}, after {len(sta)}/{len(a)}; every task stable before is stable "
                    f"after: {sup}; newly stable: {', '.join(sorted(sta - stb)) or 'none'}); max SD {a.mcc_sd.max():.4f} <= 0.02: {sd_ok} "
                    f"=> {'PRESERVED' if same and sd_ok else ('PRESERVED except for the newly-stable task(s) (stability improved)' if sup and sd_ok else 'NOT preserved')}")

    x9 = rd("extra_presence")
    if x9 is not None:
        from scipy.stats import spearmanr
        ev = rd("eval")
        add("\nX9 MOTIF PRESENCE (#FDR-enriched TFs of the LightGBM gain vector) vs MCC, 9 GB tasks")
        for u, g in x9.groupby("unit", sort=False):
            mcc = []
            for t_ in g.task:
                e = ev[ev.task == t_]
                mcc.append(float(e[e.protocol == "original"].iloc[0].mcc) if u == "original"
                           else float(e[(e.protocol == "resplit_cluster") & (e.threshold == "id80")].iloc[0].mcc))
            rho, p = spearmanr(mcc, g.n_enriched_TF)
            add(f"  {u:<14} Spearman(MCC, #enriched) rho={rho:+.3f} p={p:.3f} => X9 presence null "
                f"{'PRESERVED' if p >= 0.05 else 'NOT preserved'}; " + ", ".join(f"{r.task} {int(r.n_enriched_TF)}" for _, r in g.iterrows()))

    x10 = rd("extra_coords")
    if x10 is not None:
        add("\nX10 COORDINATES (cohn negatives, GENCODE v44 TSS +/-500 bp)")
        for _, r in x10.iterrows():
            add(f"  {r.data:<9} flagged {r.flagged_tss_rate:.3f} vs unflagged {r.unflagged_tss_rate:.3f} (x{r.rate_ratio:.2f}, "
                f"Fisher p={r.fisher_p:.1e}; n_neg={int(r.n_neg)}); positives {r.pos_tss_rate:.3f} => X10 coordinates TSS depletion "
                f"{'PRESERVED' if r.rate_ratio < 1 and r.fisher_p < 0.05 else 'NOT preserved'}")

    L += ft_hc_section()

    a = rd("labels_addendum")
    if a is not None:
        add("\nA1 (POST HOC) RE-POOLING CONTROL FOR RELABELS (label rule on cluster vs RANDOM re-split MCC; seed-mean primary)")
        for th, g in a.groupby("threshold", sort=False):
            rel = g[g.relabel_seedmean]
            add(f"  {th}: relabels under cluster re-split (seed-mean): " + (", ".join(
                f"{r.task} ({r.label_before}->{r.label_cluster_seedmean}; random re-split -> {r.label_random_seedmean}; "
                f"{'HOMOLOGY-attributable' if r.relabel_homology_attributable_seedmean else 're-pooling'})" for _, r in rel.iterrows()) or "none"))
            s42 = g[g.relabel_s42 & ~g.relabel_seedmean]
            if len(s42):
                add(f"        split-seed-42-only relabels (not in the seed mean): " + ", ".join(f"{r.task} ({r.label_before}->{r.label_cluster_s42})" for _, r in s42.iterrows()))
        add("A2 (POST HOC) SUBMITTED-PAPER LABELS ('composition', 'contamination (weak)')")
        for _, r in a[a.published_submitted.isin(["composition", "contamination (weak)"])].iterrows():
            add(f"  {r.task:<34}{r.threshold:<5} {r.published_submitted:<22} {r.sub_preservation}")
            add(f"  {'':<39}before: {r.get('sub_diag_before', '')}")
            add(f"  {'':<39}after : {r.get('sub_diag_after', '')}")
    cr = rd("class_redundancy")
    if cr is not None:
        add("A3 (POST HOC) WHERE THE HOMOLOGY IS (id80; fraction of test sequences with a train homolog, by class)")
        for _, r in cr[cr.threshold == "id80"].iterrows():
            cls = "  ".join(f"class{j} {r[f'class{j}_frac_any']:.3f}" for j in range(3) if f"class{j}_frac_any" in r and r[f"class{j}_frac_any"] == r[f"class{j}_frac_any"])
            add(f"  {r.task:<34} any {r.frac_test_any_train_homolog:.3f}  same-label {r.frac_test_same_label_homolog:.3f}  "
                f"cross-label-only {r.frac_test_cross_label_only:.3f}  | {cls}")
    lo = rd("route1_loto")
    if lo is not None:
        add("A4 (POST HOC) ROUTE-1 with homology-clean MCCs (11 TF tasks; published discriminability on x)")
        for _, r in lo.iterrows():
            add(f"  y={r.y:<28} rho={r.spearman_rho:+.3f} p={r.spearman_p:.3f}  LOTO MAE {r.loto_mae:.3f}{ci(r.loto_mae_lo, r.loto_mae_hi)}  "
                f"R2 {r.loto_r2:+.2f}{ci(r.loto_r2_lo, r.loto_r2_hi, '{:+.2f}')}  slope {r.ols_slope:.2f}")
    return L


def main():
    L = build()
    out = os.path.join(R, f"{P}_extra_interpretation.txt")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
