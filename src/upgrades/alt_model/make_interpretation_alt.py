"""Writes results/upgrades/alt_model_interpretation.txt and alt_model_paper_table.csv from the CSVs
produced by analyze_alt_model.py. Every number printed here is read from those CSVs (file + row are
named next to each block). No statistics are computed here beyond formatting, counting and simple lookups.
(Fix rounds 1-2, 2026-09-27: wording aligned with the independent verifications -- see the README.)"""
import os

import numpy as np
import pandas as pd

MODELS = ["lgbm", "lr", "ebm"]
MN = {"lgbm": "LightGBM", "lr": "LR", "ebm": "EBM", "lrx": "LR-ext", "lra": "LR-auc", "ebm512": "EBM-512",
      "ebmpos": "EBM-pos"}
SHORT = {"demo_human_or_worm": "human_or_worm", "human_ensembl_regulatory": "regulatory",
         "nt_promoter_all": "nt_prom_all", "nt_promoter_no_tata": "nt_prom_notata",
         "human_nontata_promoters": "nontata_prom", "nt_promoter_tata": "nt_prom_tata",
         "demo_coding_vs_intergenomic_seqs": "coding_vs_interg", "human_enhancers_ensembl": "enh_ensembl",
         "dummy_mouse_enhancers_ensembl": "mouse_enh", "nt_enhancers": "nt_enhancers",
         "nt_enhancers_types": "nt_enh_types", "human_enhancers_cohn": "enh_cohn",
         "human_ocr_ensembl": "ocr_ensembl", "drosophila_enhancers_stark": "drosophila",
         "nt_splice_sites_all": "nt_splice"}
ORDER = list(SHORT)
SPLICE = "nt_splice_sites_all"
N9 = "route1_disc_vs_mcc_n9_homology_clean"
N11 = "route1_disc_vs_mcc_n11"


def _r(R, f):
    p = os.path.join(R, f)
    try:
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def ci(v, lo, hi, d=3, sign=True):
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    return f"{f.format(v)} [{f.format(lo)}, {f.format(hi)}]"


def reason(row, ref, sch):
    """Which rule component differs from the LightGBM reference (numbers from alt_model_labels.csv)."""
    out = []
    if (row["mcc"] >= 0.7) != (ref["mcc"] >= 0.7):
        out.append(f"benchmark MCC {row['mcc']:.3f} vs LightGBM {ref['mcc']:.3f} crosses the 0.70 solved cut")
    if bool(row["positional_rule"]) != bool(ref["positional_rule"]):
        out.append("positional rescue rule differs")
    if sch == "primary" and bool(row["eq_collapses_prereg"]) != bool(ref["eq_collapses_prereg"]):
        out.append("pre-registered equalized MCC-margin collapse differs")
    if sch == "sensitivity":
        a, b = row["comp_fraction"], ref["comp_fraction"]
        if (a == a and b == b) and ((a >= 0.8) != (b >= 0.8)):
            out.append(f"composition fraction {a:.3f} vs LightGBM {b:.3f} crosses 0.80")
    if sch == "posthoc_overlap" and str(row.get("overlap_contrast_ci_lo_gt_0")) != str(ref.get("overlap_contrast_ci_lo_gt_0")):
        out.append(f"overlap contrast CI>0: {row.get('overlap_contrast_ci_lo_gt_0')} vs LightGBM {ref.get('overlap_contrast_ci_lo_gt_0')}")
    if sch == "bibm" and bool(row.get("model_free_gate")) and (row["mcc"] >= 0.7) != (ref["mcc"] >= 0.7):
        out.append("gate-positive task: failing under this model -> bibm contamination rule applies")
    return "; ".join(out) if out else "same rule components as LightGBM"


def r2_summary(p, m):
    """Route-2 summary numbers for one model from alt_model_route2.csv (arm=pos)."""
    g = p[p.model == m]
    if g.empty:
        return None
    sp = g[g.task == SPLICE]
    oth = g[g.task != SPLICE]
    sig = oth[oth.delta_ci_excludes_0.astype(bool)]
    i_all = oth.delta_mcc.abs().idxmax()
    out = dict(splice=sp.iloc[0] if not sp.empty else None,
               max_all=float(abs(oth.loc[i_all, "delta_mcc"])), max_all_task=oth.loc[i_all, "task"],
               max_all_lo=float(oth.loc[i_all, "delta_lo"]), max_all_hi=float(oth.loc[i_all, "delta_hi"]),
               max_sig=np.nan, max_sig_task="", n_other=len(oth),
               rule_tasks=list(g[g.positional_rule.astype(bool)].task),
               zero_bin=list(g[g.b8_equals_b1_by_construction.astype(bool)].task) if "b8_equals_b1_by_construction" in g else [],
               few_bin=list(g[(g.n_bin_features_B8 > 0) & (g.n_bin_features_B8 < 16)].task) if "n_bin_features_B8" in g else [])
    if not sig.empty:
        i = sig.delta_mcc.abs().idxmax()
        out.update(max_sig=float(abs(sig.loc[i, "delta_mcc"])), max_sig_task=sig.loc[i, "task"])
    return out


def write(R):
    L = []
    add = L.append
    bench = _r(R, "alt_model_bench.csv")
    r2 = _r(R, "alt_model_route2.csv")
    r3p = _r(R, "alt_model_route3_prereg.csv")
    r3s = _r(R, "alt_model_route3_sizecontrolled.csv")
    r3c = _r(R, "alt_model_route3_contrast.csv")
    ov = _r(R, "alt_model_route3_overlap.csv")
    ovc = _r(R, "alt_model_route3_overlap_contrast.csv")
    shuf = _r(R, "alt_model_route3_shuffled.csv")
    r1 = _r(R, "alt_model_route1.csv")
    dose = _r(R, "alt_model_dose.csv")
    attr = _r(R, "alt_model_attribution.csv")
    lab = _r(R, "alt_model_labels.csv")
    agr = _r(R, "alt_model_agreement.csv")
    chk = _r(R, "alt_model_checks.csv")
    units = _r(R, "alt_model_units.csv")
    ri = _r(R, "route_invariance.csv")
    bar = "=" * 100
    add(bar)
    add("ALTERNATIVE INTERPRETABLE MODELS (Reviewer 2, comment 3): do the route diagnostics and labels depend on")
    add("LightGBM? Same k-mer spectrum (LightGBM's validation-selected k), same splits, same caps, seed 42.")
    add("  lgbm = the paper's LightGBM (reference, fixed hyperparameters), all 4^k k-mer columns.")
    add("  lr   = L1/L2 logistic regression (liblinear, one-vs-rest for the 3 multiclass tasks) on all 4^k")
    add("         standardized k-mer columns; penalty and C tuned on an 80/20 carve of train by validation MCC.")
    add("  ebm  = Explainable Boosting Machine (GA2M; 10 pairwise terms) on the 20-d composition signature + the")
    add("         128 k-mers with the largest train-only ANOVA F -- a filtered SUBSET of the spectrum, not all of it.")
    add("  Post hoc variants: lrx = LR with C grid extended to 10; lra = LR tuned by validation AUROC; ebm512 = EBM")
    add("  with 512 k-mers; ebmpos (P9, fix round 1) = EBM B=8 arm with 128 global + 128 position-binned k-mers")
    add("  ranked separately (the pre-registered filter admitted no binned k-mer on 3 tasks).")
    add("  CIs: 1000-resample percentile bootstrap, test sequence = unit (paired where arms share a test set);")
    add("  Route-3 size-controlled readouts: two-level paired bootstrap (test sequences x random seeds, P10b).")
    add("  Pre-registered plan + post hoc addenda (P1-P11): run_alt_model.py docstring.")
    add(bar)
    if not chk.empty:
        bad = chk[~chk["ok"].astype(bool)]
        add(f"Sanity checks (alt_model_checks.csv): {int(chk['ok'].astype(bool).sum())}/{len(chk)} pass."
            + ("" if bad.empty else " FAILING: " + "; ".join(f"{r.check}={r.value} (exp {r.expected})" for _, r in bad.iterrows())))
    # ---------------------------------------------------------------- benchmark
    if not bench.empty:
        add("")
        add("[1] Benchmark MCC [95% CI] on the original split (alt_model_bench.csv, arm=bench; train capped at 50,000")
        add("    for 6 tasks as pre-registered -- uncapped rows arm=bench_full). Tier: solved >=0.70, partial >=0.50.")
        add(f"{'task':<18}" + "".join(f"{MN[m]:>26}" for m in MODELS))
        b = bench[bench.arm == "bench"]
        for t in ORDER:
            s = f"{SHORT[t]:<18}"
            for m in MODELS:
                r = b[(b.task == t) & (b.model == m)]
                s += (f"{r.mcc.iloc[0]:>8.3f} [{r.mcc_lo.iloc[0]:.3f},{r.mcc_hi.iloc[0]:.3f}] {r.tier.iloc[0][:4]:>4}"
                      if not r.empty else f"{'-':>26}")
            add(s)
        for m in ("lr", "ebm"):
            g = b[b.model == m]
            if "dmcc_vs_lgbm" in g:
                add(f"  {MN[m]} minus LightGBM MCC: median {g.dmcc_vs_lgbm.median():+.3f}, range "
                    f"[{g.dmcc_vs_lgbm.min():+.3f}, {g.dmcc_vs_lgbm.max():+.3f}] over {g.dmcc_vs_lgbm.notna().sum()} tasks")
        bf = bench[bench.arm == "bench_full"]
        if not bf.empty:
            add("  Uncapped (post hoc P2, arm=bench_full): " + "; ".join(
                f"{SHORT[r.task]} {MN.get(r.model, r.model)} {r.mcc:.3f}" for _, r in bf.iterrows() if r.model in MODELS))
        # tier agreement vs the paper's route_invariance LR floor
        bl, bll = b[b.model == "lr"].set_index("task"), b[b.model == "lgbm"].set_index("task")
        diff = [t for t in bl.index if t in bll.index and bl.loc[t, "tier"] != bll.loc[t, "tier"]]
        add(f"  LR vs LightGBM tier agreement here: {len(bl) - len(diff)}/{len(bl)} (differ: "
            f"{', '.join(f'{SHORT[t]} {bl.loc[t, 'tier']}/{bll.loc[t, 'tier']}' for t in diff)}).")
        if not ri.empty:
            rd = ri[ri.lr_tier != ri.lgbm_tier]
            reg = ri[ri.task == "human_ensembl_regulatory"]
            add(f"  The paper's Robustness count ({(ri.lr_tier == ri.lgbm_tier).sum()}/{len(ri)}, route_invariance.csv) uses the"
                f" paper's LR FLOOR (multinomial, L2-normalized rows, uncapped; differs on {', '.join(SHORT[t] for t in rd.task)}),")
            add("  not this module's tuned one-vs-rest L1/L2 LR at the 50k cap; the two counts are different LR models and"
                " should not be quoted side by side.")
            if not reg.empty and "human_ensembl_regulatory" in bl.index:
                add(f"  regulatory (3-class): tuned OvR L1/L2 LR {bl.loc['human_ensembl_regulatory', 'mcc']:.3f}; paper's"
                    f" multinomial LR floor {reg.lr_mcc.iloc[0]:.4f}; LightGBM {bll.loc['human_ensembl_regulatory', 'mcc']:.3f}"
                    f" (capped) / {reg.lgbm_mcc.iloc[0]:.4f} (uncapped, route_invariance.csv).")
    # ---------------------------------------------------------------- Route 2
    if not r2.empty:
        add("")
        add("[2] Route 2 -- dMCC(B8 - B1) [paired 95% CI] (alt_model_route2.csv, arm=pos; both splits capped at 20,000).")
        add("    #bin = position-binned k-mer columns the B=8 model received (LightGBM/LR: all 8*4^k).")
        p = r2[r2.arm == "pos"]
        cols = MODELS + ["ebmpos"]
        add(f"{'task':<18}" + "".join(f"{MN[m]:>26}" for m in cols) + "   #bin EBM/EBM-pos")
        for t in ORDER:
            s = f"{SHORT[t]:<18}"
            for m in cols:
                r = p[(p.task == t) & (p.model == m)]
                s += (f"{r.delta_mcc.iloc[0]:>+9.3f} [{r.delta_lo.iloc[0]:+.3f},{r.delta_hi.iloc[0]:+.3f}]"
                      if not r.empty else f"{'-':>26}")
            ne = p[(p.task == t) & (p.model == "ebm")]; np_ = p[(p.task == t) & (p.model == "ebmpos")]
            s += (f"   {int(ne.n_bin_features_B8.iloc[0]) if not ne.empty else '-':>4}/"
                  f"{int(np_.n_bin_features_B8.iloc[0]) if not np_.empty else '-'}")
            add(s)
        for m in cols:
            sm = r2_summary(p, m)
            if sm is None:
                continue
            spl = sm["splice"]
            add(f"  {MN[m]}: splice dMCC {spl.delta_mcc:+.3f}; other {sm['n_other']} tasks: largest |dMCC| "
                f"{sm['max_all']:.3f} ({SHORT[sm['max_all_task']]}, CI [{sm['max_all_lo']:+.3f},{sm['max_all_hi']:+.3f}]), "
                f"largest CI-significant |dMCC| "
                + (f"{sm['max_sig']:.3f} ({SHORT[sm['max_sig_task']]})" if sm["max_sig_task"] else "none")
                + f"; tasks meeting the positional rule: {', '.join(SHORT[t] for t in sm['rule_tasks']) or 'none'}")
            if m in ("ebm", "ebmpos") and (sm["zero_bin"] or sm["few_bin"]):
                add(f"      {MN[m]} B=8 design received NO binned k-mer on {', '.join(SHORT[t] for t in sm['zero_bin']) or 'no task'}"
                    f" (dMCC = 0 by construction: test uninformative) and <16 on {', '.join(SHORT[t] for t in sm['few_bin']) or 'no task'}.")
        pf = r2[r2.arm == "pos_full"]
        if not pf.empty:
            add("  Uncapped splice (post hoc P3, arm=pos_full): " + "; ".join(
                f"{MN.get(r.model, r.model)} {r.mcc_B1:.3f}->{r.mcc_B8:.3f} (d={r.delta_mcc:+.3f} [{r.delta_lo:+.3f},{r.delta_hi:+.3f}])"
                for _, r in pf.iterrows()))
    # ---------------------------------------------------------------- Route 3
    if not r3p.empty:
        add("")
        add("[3a] Route 3, PRE-REGISTERED readouts on the composition-equalized split (alt_model_route3_prereg.csv).")
        add("     margin = k-mer MCC - composition-only MCC on the equalized test set (paired); 'collapses' = CI lower <= 0.")
        add("     CAVEAT: equalized test sets are 80-94% positive, so default-threshold MCC collapses largely for")
        add("     size/imbalance reasons (see [3b] MCC columns); the collapse also fires on the clean control under some models.")
        add("     For LR/LR-ext the composition-only fit on every equalized split is DEGENERATE (all-zero, MCC 0), so for LR")
        add("     'collapses onto the composition baseline' only means 'equalized k-mer MCC CI reaches 0'.")
        for _, r in r3p.iterrows():
            if r.model not in MODELS + ["lrx"]:
                continue
            add(f"  {SHORT[r.task]:<13}{MN[r.model]:<9} origMCC {r.orig_mcc:.3f} -> eq k-mer {r.eq_kmer_mcc:.3f} / comp "
                f"{r.eq_comp_mcc:.3f}; margin {r.eq_margin_mcc:+.3f} [{r.eq_margin_lo:+.3f},{r.eq_margin_hi:+.3f}] "
                f"collapses={bool(r.collapses_prereg)}; eq-orig MCC {r.eq_minus_orig_mcc:+.3f} "
                f"[{r.eq_minus_orig_mcc_lo:+.3f},{r.eq_minus_orig_mcc_hi:+.3f}]")
        ctl = r3p[(r3p.task == "drosophila_enhancers_stark") & r3p.collapses_prereg.astype(bool)]
        add("  Clean control drosophila: pre-registered collapse fires under " +
            (", ".join(MN.get(m, m) for m in ctl.model) if not ctl.empty else "no model") +
            " -> the model-dependent half of rule 3 does not separate the control there; the model-free TATA gate prevents"
            " a false contamination label on it. Among GATE-POSITIVE failing tasks the model-dependent half decides (see [7]).")
    if not r3s.empty:
        add("")
        add("[3b] Route 3, SIZE-CONTROLLED readouts (post hoc P1/P10; alt_model_route3_sizecontrolled.csv, feat=kmer).")
        add("     A0 = original; Ar = mean of 10 size-matched random-negative arms (min-max); Am = composition-matched")
        add("     (published equalized split); comp effect = Ar - Am; share = (Ar-Am)/(Ar-0.5); fixed = original-split")
        add("     model (no retraining) scored on the random vs matched TEST subsets. CI: two-level paired bootstrap.")
        for _, r in r3s[r3s.feat == "kmer"].iterrows():
            tag = " (leak-free test)" if bool(r.leakfree_test) else ""
            add(f"  {SHORT[r.task]:<13}{MN.get(r.model, r.model):<8}{tag:<17} A0 {r.auroc_orig:.3f}  Ar {r.auroc_random_mean:.3f} "
                f"({r.auroc_random_min:.3f}-{r.auroc_random_max:.3f})  Am {r.auroc_matched:.3f}  effect "
                f"{ci(r.comp_effect, r.comp_effect_lo, r.comp_effect_hi)}  share {r.share_lost:.2f} "
                f"[{r.share_lost_lo:.2f},{r.share_lost_hi:.2f}]  fixed {ci(r.fixed_comp_effect, r.fixed_comp_effect_lo, r.fixed_comp_effect_hi)}"
                f"  n_neg_test={r.n_neg_test_matched}")
        add("  Default-threshold MCC on the same arms (P10c/P11c): how much of the MCC drop orig -> matched do size-matched")
        add("  RANDOM negatives already produce? size share = (M0 - mean Mr)/(M0 - Mm) [two-level CI]; ~1 = the whole MCC")
        add("  collapse is a size/imbalance effect. (The random-minus-matched MCC CI alone cannot show reproduction.)")
        for _, r in r3s[r3s.feat == "kmer"].iterrows():
            tag = " (leak-free)" if bool(r.leakfree_test) else ""
            add(f"  {SHORT[r.task]:<13}{MN.get(r.model, r.model):<8}{tag:<12} MCC orig {r.mcc_orig:.3f} -> random "
                f"{r.mcc_random_mean:.3f} ({r.mcc_random_min:.3f}-{r.mcc_random_max:.3f}) vs matched {r.mcc_matched:.3f}; "
                f"random-matched {ci(r.mcc_comp_effect, r.mcc_comp_effect_lo, r.mcc_comp_effect_hi)}; size share of the MCC drop "
                f"{r.mcc_share_of_drop_from_size:.2f} [{r.mcc_share_of_drop_from_size_lo:.2f},{r.mcc_share_of_drop_from_size_hi:.2f}]")
        if not units.empty:
            lu = units[(units.model == "lr") & (units.feat == "comp")]
            parts = []
            for lab_, msk in (("eq", lu.arm == "eq"), ("eqr", lu.arm.astype(str).str.startswith("eqr")), ("shuf", lu.arm == "shuf")):
                g_ = lu[msk]
                nd_ = int((g_.lr_degenerate.astype(str) == "True").sum())
                bytask = g_[g_.lr_degenerate.astype(str) == "True"].task.value_counts()
                parts.append(f"{lab_} {nd_}/{len(g_)}" + (f" ({', '.join(f'{SHORT.get(t, t)} {n}' for t, n in bytask.items())})"
                                                         if lab_ == "eqr" and nd_ else ""))
            add("  MCC-tuned LR composition-only fits that are degenerate (all-zero; alt_model_units.csv lr_degenerate): "
                + "; ".join(parts) + ".")
        cmp_ = r3s[r3s.feat == "comp"]
        if not cmp_.empty:
            add("  Composition-only arm, random vs matched (balance check; feat=comp). LR rows are UNINFORMATIVE where the")
            add("  MCC-tuned LR degenerates to an all-zero model (AUROC exactly 0.5; alt_model_units.csv lr_degenerate):")
            for _, r in cmp_.iterrows():
                if r.model not in MODELS + ["lra"]:
                    continue
                tag = " (leak-free)" if bool(r.leakfree_test) else ""
                deg = ""
                if r.model in ("lr", "lra") and not units.empty:
                    u = units[(units.task == r.task) & (units.model == r.model) & (units.feat == "comp")
                              & (units.arm.astype(str).str.startswith("eq"))]
                    nd = int((u.lr_degenerate.astype(str) == "True").sum())
                    deg = f"  [{nd}/{len(u)} eq/eqr fits degenerate]" if nd else ""
                add(f"    {SHORT[r.task]:<13}{MN.get(r.model, r.model):<8}{tag:<12} comp-only AUROC random {r.auroc_random_mean:.3f} "
                    f"-> matched {r.auroc_matched:.3f}{deg}")
    if not r3c.empty:
        add("")
        add("[3c] Random-negative design: share of above-chance AUROC lost, task minus the clean control (drosophila)")
        add("     (alt_model_route3_contrast.csv; point estimate [independent two-level bootstraps of the two datasets]).")
        for _, r in r3c.iterrows():
            tag = " (leak-free)" if bool(r.leakfree_test) else ""
            add(f"  {SHORT[r.task]:<13}{MN.get(r.model, r.model):<8}{tag:<12} share {r.share_task:.3f} vs control {r.share_control:.3f}; "
                f"diff {ci(r['diff'], r.diff_lo, r.diff_hi)} -> loses more than control: {bool(r.task_loses_more)}")
    if not ov.empty:
        add("")
        add("[3c'] Route 3, REMOVAL-FREE overlap-weighting design (post hoc P8 = psm_sensitivity's primary between-dataset")
        add("      design, reproduced exactly for LightGBM; P11a extends it to enh_ensembl, ocr_ensembl (uncapped bench_full")
        add("      units) and nontata_prom): the ORIGINAL model on the original test set, overlap-weighted vs 10")
        add("      permuted-weight placebos; share = (placebo - overlap)/(placebo - 0.5) (alt_model_route3_overlap*.csv).")
        for _, r in ov[ov.feat == "kmer"].iterrows():
            tag = " (leak-free)" if bool(r.leakfree_test) else ""
            add(f"  {SHORT[r.task]:<13}{MN.get(r.model, r.model):<8}{tag:<12} [{r.source_arm}] overlap {r.auroc_overlap:.3f} vs placebo "
                f"{r.auroc_placebo_mean:.3f}; drop {ci(r['drop'], r.drop_lo, r.drop_hi)}; share {r.share_lost:.2f} "
                f"[{r.share_lo:.2f},{r.share_hi:.2f}]" + ("  (overlap AUROC < 0.5: over-correction)" if bool(r.overlap_auroc_below_chance) else ""))
        for _, r in ovc.iterrows():
            tag = " (leak-free)" if bool(r.leakfree_test) else ""
            verdict = "MORE" if bool(r.task_loses_more) else ("LESS" if bool(r.task_loses_less) else "n.s.")
            add(f"  contrast {SHORT[r.task]:<13}- drosophila {MN.get(r.model, r.model):<8}{tag:<12} share diff "
                f"{ci(r['diff'], r.diff_lo, r.diff_hi)} -> task loses {verdict} than the clean control")
        add("  nt_enhancers shares > 1 mean the overlap-weighted AUROC falls BELOW 0.5: the linear-propensity overlap")
        add("  weights over-correct there (they reverse the ranking), so only the direction of its contrast is read.")
    if not shuf.empty:
        add("")
        add("[3d] Dinucleotide-shuffled negatives (alt_model_route3_shuffled.csv). SANITY CHECK, not model-robustness")
        add("     evidence: every shuffle has exactly its positive twin's mono+dinucleotide composition, so ANY classifier")
        add("     on the 20-d composition signature has AUROC exactly 0.5 by construction (LR composition-only fits are")
        add("     additionally all-zero). The model-dependent part is the k-mer AUROC and the composition fraction.")
        for _, r in shuf.iterrows():
            add(f"  {SHORT[r.task]:<13}{MN.get(r.model, r.model):<8} comp-only AUROC {r.shuf_comp_auroc:.3f}; k-mer AUROC "
                f"{r.shuf_kmer_auroc:.3f} (MCC {r.shuf_kmer_mcc:.3f}); comp fraction "
                f"{r.comp_fraction_orig:.2f} [{r.comp_fraction_orig_lo:.2f},{r.comp_fraction_orig_hi:.2f}] -> "
                f"{r.comp_fraction_shuf:.2f} [{r.comp_fraction_shuf_lo:.2f},{r.comp_fraction_shuf_hi:.2f}]")
    hs = _r(R, "alt_model_shuffle_hashseed.csv")
    if not hs.empty:
        add("")
        add("[3e] REPRODUCIBILITY FINDING (alt_model_shuffle_hashseed.csv; check_shuffle_hashseed.py): dinuc_shuffle.py")
        add("     iterates set(seq), so the shuffled negatives depend on PYTHONHASHSEED. Over "
            f"{hs.hashseed.nunique()} hash seeds ({hs.set_order.nunique()} distinct 'ACGT' set orders of 24; "
            f"{hs.groupby('task').shuffle_sha1.nunique().min()}+ distinct draws per task):")
        for t, g in hs.groupby("task", sort=False):
            add(f"  {SHORT.get(t, t):<13} TATA AUROC {g.tata_auroc.min():.4f}-{g.tata_auroc.max():.4f} (paper {g.paper_tata_auroc.iloc[0]:.4f}); "
                f"GC AUROC {g.gc_auroc.min():.3f}-{g.gc_auroc.max():.3f}; k-mer LightGBM MCC {g.kmer_mcc.min():.3f}-{g.kmer_mcc.max():.3f} "
                f"(paper {g.paper_kmer_mcc.iloc[0]:.3f}); draws reproducing the paper CSV: {int(g.reproduces_paper_csv.sum())}")
        add("     The paper's draw was NOT RECOVERED by this scan (the scan is not exhaustive: the draw depends on each")
        add("     sequence's own set order, and the committed shuffled_neg.csv, dated 2026-05-31, predates the 2026-09-23")
        add("     data-integrity fix). The alt-model shuffled arms all use ONE pinned draw (PYTHONHASHSEED=0), paired across models.")

    # ---------------------------------------------------------------- Route 1
    if not r1.empty:
        add("")
        add("[4] Route 1 (alt_model_route1.csv): Spearman(model-free discriminability, model MCC), BC bootstrap over tasks")
        add("    (10,000, seed 42) + permutation p + jackknife (task whose removal gives the min/max rho); LOTO OLS MAE;")
        add("    presence null = Spearman(#enriched TFs, MCC), GB n=9. Rows without suffix = the pre-registered capped MCCs;")
        add("    '_uncapped' = bench_full MCCs where available (P10a; = the paper's Table corr MCCs for LightGBM); written only")
        add("    for models with >= 1 uncapped refit in the set (P11d). p_t = scipy Spearman t-approximation (the method")
        add("    that gives the paper's p=0.003); p_perm = exact enumeration (n=9) / Monte-Carlo (n=11) permutation p.")
        for _, r in r1.iterrows():
            if r.model not in MODELS + ["ebm512"]:
                continue
            if str(r.analysis).startswith("route1_loto"):
                add(f"  {r.analysis:<50}{MN.get(r.model, r.model):<9} MAE {r.mae:.3f} [{r.mae_lo:.3f},{r.mae_hi:.3f}]")
            else:
                jt = ""
                if isinstance(r.get("jackknife_max_drop_task"), str):
                    jt = f" (max w/o {SHORT.get(r.jackknife_max_drop_task, r.jackknife_max_drop_task)}, min w/o " \
                         f"{SHORT.get(r.jackknife_min_drop_task, r.jackknife_min_drop_task)})"
                nc = r.get("n_tasks_capped_mcc")
                add(f"  {r.analysis:<50}{MN.get(r.model, r.model):<9} rho {r.rho:+.3f} [{r.boot_lo:+.3f},{r.boot_hi:+.3f}] "
                    f"p_perm={r.p_perm:.3g} ({r.p_perm_method}) p_t={r.p_spearman_t_approx:.3g} "
                    f"jackknife [{r.jackknife_lo:+.2f},{r.jackknife_hi:+.2f}]{jt}"
                    + (f" capped-MCC tasks {int(nc)}" if nc == nc and nc is not None and "_uncapped" in str(r.analysis) else "")
                    + (f"  null preserved (CI includes 0): {r.presence_null_preserved}"
                       if str(r.analysis).startswith("presence") else ""))
    if not dose.empty:
        add("")
        add("[5] Route 1 semi-synthetic dose-response (alt_model_dose.csv): OLS slope dMCC/dd [row bootstrap CI], Spearman.")
        for _, r in dose.iterrows():
            add(f"  {MN.get(r.model, r.model):<9}{r.motif:<7} n={r.n_points:<4} slope {r.slope:+.2f} [{r.slope_lo:+.2f},{r.slope_hi:+.2f}] "
                f"Spearman {r.spearman:+.3f}")
    if not attr.empty:
        add("")
        add("[6] Attribution check (post hoc P5; alt_model_attribution.csv): per-TF gain-weighted enrichment test")
        add("    (paper's run_interpretability.enrichment_z, BH-FDR<0.05) on each model's own k-mer importances, same k.")
        add("    CAVEAT: the EBM importances are nonzero only on its 128 filter-selected k-mers, so its profile partly")
        add("    reflects the univariate ANOVA filter, not the EBM itself.")
        for t in ORDER:
            g = attr[attr.task == t]
            if g.empty:
                continue
            s = f"  {SHORT[t]:<16}"
            for m in MODELS:
                r = g[g.model == m]
                if r.empty:
                    continue
                r = r.iloc[0]
                extra = "" if m == "lgbm" else f" J={r.jaccard_vs_lgbm:.2f} zrho={r.z_spearman_vs_lgbm:+.2f}"
                nm = r.get("paper_named_tf_rank", "")
                nm = f" ranks[{nm}]" if isinstance(nm, str) and nm else ""
                s += f" | {MN[m]} n={int(r.n_enriched)} top={r.top1 if isinstance(r.top1, str) and r.top1 else '-'}{extra}{nm}"
            add(s)
        for m in ("lr", "ebm"):
            g = attr[attr.model == m]
            if g.empty:
                continue
            add(f"  {MN[m]} vs LightGBM over {len(g)} tasks: median z-profile Spearman {g.z_spearman_vs_lgbm.median():+.2f} "
                f"(min {g.z_spearman_vs_lgbm.min():+.2f}, {SHORT[g.loc[g.z_spearman_vs_lgbm.idxmin(), 'task']]}); median Jaccard of "
                f"enriched sets {g.jaccard_vs_lgbm.median():.2f}; top TF identical on {int(g.top1_same_as_lgbm.astype(bool).sum())}/{len(g)}"
                f" tasks, in LightGBM's top 5 on {int(g.top1_in_lgbm_top5.astype(bool).sum())}/{len(g)}; median #features with"
                f" nonzero importance {int(g.n_features_used.median())}")
    # ---------------------------------------------------------------- labels
    gate_df = _r(R, "alt_model_gate.csv")
    gpos = list(gate_df[gate_df.gate.astype(bool)].task) if not gate_df.empty else []
    if not agr.empty:
        add("")
        add("[7] ROUTE LABELS (alt_model_labels.csv / alt_model_agreement.csv).")
        add("    primary = pre-registered rules (target: camera-ready v2 labels); sensitivity = pre-registered rule 3 with")
        add("    composition fraction >= 0.80; bibm = post hoc map to the SUBMITTED BIBM Table-1 vocabulary (contamination =")
        add("    failing + gate, graded by TATA excess); posthoc_overlap (P11b) = rule 3 with the overlap-design contrast vs the")
        add("    control (CI lower > 0; leak-free nt_enhancers) replacing the equalized-MCC collapse.")
        add("    'preserved' = same label as the paper AND its label-defining effect keeps sign with a 95% CI excluding")
        add("    0/the cut. Post hoc criteria (P10d) replace ONLY the contamination effect: 'auroc' = size-controlled")
        add("    AUROC effect (leak-free for nt_enhancers); 'overlap' = overlap-design share lost minus the control.")
        for _, r in agr.iterrows():
            add(f"  [{r.bench_arm}] {MN.get(r.model, r.model):<9}{r.scheme:<12} same label {r.same_label_as_paper}/{r.n_tasks}"
                f" (failure labels {r.failure_labels_same_as_paper}/{r.n_failure_labels_paper}); preserved pre-reg "
                f"{r.preserved_prereg_criterion}, post hoc auroc {r.preserved_posthoc_auroc_criterion}, overlap "
                f"{r.preserved_posthoc_overlap_criterion}; same as LightGBM {r.same_label_as_lgbm_reference}/{r.n_tasks}"
                + (f"; differs: {r.disagreements_vs_paper}" if isinstance(r.disagreements_vs_paper, str) and r.disagreements_vs_paper else "")
                + (f"; pre-reg effect-CI failures: {r.effect_ci_failures}" if isinstance(r.effect_ci_failures, str) and r.effect_ci_failures else "")
                + (f"; post hoc auroc failures: {r.posthoc_auroc_failures}" if isinstance(r.posthoc_auroc_failures, str) and r.posthoc_auroc_failures else "")
                + (f"; post hoc overlap failures: {r.posthoc_overlap_failures}" if isinstance(r.posthoc_overlap_failures, str) and r.posthoc_overlap_failures else "")
                + (f"; post hoc readout not computed (not assessed): {r.posthoc_not_assessed}" if isinstance(r.posthoc_not_assessed, str) and r.posthoc_not_assessed else "")
                + (f"; eq collapse fires without gate: {r.eq_collapse_fires_without_gate}"
                   if isinstance(r.eq_collapse_fires_without_gate, str) and r.eq_collapse_fires_without_gate else "")
                + (f"; gate-positive failing tasks -> {r.gate_positive_failing_labels}"
                   if isinstance(r.gate_positive_failing_labels, str) and r.gate_positive_failing_labels else ""))
        add("")
        add("  What the agreement test can detect: a weaker model keeps a shared-motif (residual) label automatically")
        add("  unless it falsely fires the positional rule or gate+contamination rule; a failing label is lost only if the")
        add("  model SOLVES the task. The informative model-dependent checks are the positional rule on the 14 non-splice")
        add("  tasks, the splice rescue, and the Route-3 readouts.")
        l0_ = lab[lab.bench_arm == "bench"]
        ctl_fire = sorted(set(l0_[(l0_.task == "drosophila_enhancers_stark") & l0_.eq_collapse_fires_without_gate.astype(bool)].model))

        def _who(sch, task, pref="contamination"):
            g_ = l0_[(l0_.task == task) & l0_[f"{sch}_label"].astype(str).str.startswith(pref)]
            return ", ".join(MN.get(m, m) for m in g_.model) or "no model"
        add("  How the contamination label is decided: contamination requires BOTH the model-free TATA gate AND a model-")
        add(f"  dependent Route-3 effect. The gate is positive on {len(gpos)} binary tasks (alt_model_gate.csv: "
            + (", ".join(SHORT[t] for t in gpos) if gpos else "-") + ").")
        add("  On the gate-negative control the gate blocks a false label where the equalized-MCC collapse also fires ("
            + (", ".join(MN.get(m, m) for m in ctl_fire) or "no model") + "; EBM-512/EBM-pos inherit the EBM's equalized arms).")
        add("  Among gate-positive FAILING tasks the model-dependent half decides contamination vs shared-motif; the rule-3")
        add("  variants differ there. enh_ensembl labelled contamination under: primary (equalized collapse) -> "
            + _who("primary", "human_enhancers_ensembl") + "; sensitivity (composition fraction) -> "
            + _who("sensitivity", "human_enhancers_ensembl") + "; post hoc overlap -> "
            + _who("posthoc_overlap", "human_enhancers_ensembl") + ".")
        add("  nontata_prom (gate-positive; failing only where the model does not solve it) labelled contamination under:"
            " primary -> " + _who("primary", "human_nontata_promoters") + "; sensitivity -> "
            + _who("sensitivity", "human_nontata_promoters") + "; bibm -> " + _who("bibm", "human_nontata_promoters")
            + "; post hoc overlap -> " + _who("posthoc_overlap", "human_nontata_promoters") + ".")
        add("  Under the bibm map contamination = failing + gate, so a BIBM label changes whenever a weaker model fails a")
        add("  gate-positive solved task: BIBM labels are NOT classifier-free.")
        add("  CAVEAT for cross-model overlap shares: the EBM receives the 20-d composition signature as explicit inputs, so a")
        add("  larger EBM share (e.g. nontata, [3c']) partly reflects that design, not only the dataset.")
        add("")
        add("  Every label that differs from LightGBM's (automatic component diff; numbers from alt_model_labels.csv):")
        l0 = lab[lab.bench_arm == "bench"]
        for sch in ("primary", "sensitivity", "bibm", "posthoc_overlap"):
            for _, r in l0.iterrows():
                if r.model == "lgbm":
                    continue
                ref = l0[(l0.task == r.task) & (l0.model == "lgbm")]
                if ref.empty:
                    continue
                ref = ref.iloc[0]
                if r[f"{sch}_label"] != ref[f"{sch}_label"]:
                    add(f"    {sch:<16}{SHORT[r.task]:<15}{MN.get(r.model, r.model):<8} {ref[f'{sch}_label']} -> {r[f'{sch}_label']}: "
                        f"{reason(r, ref, sch)}")
        add("")
        add("  Contamination-label effects per model (primary scheme; alt_model_labels.csv):")
        for _, r in l0[l0.primary_label == "contamination"].iterrows():
            add(f"    {SHORT[r.task]:<13}{MN.get(r.model, r.model):<8} pre-reg: {r.primary_effect} -> ok={bool(r.primary_effect_ci_ok)};"
                f" post hoc: {r.get('primary_posthoc_auroc_effect', '-')} -> ok={r.get('primary_posthoc_auroc_effect_ok', '-')};"
                f" {r.get('primary_posthoc_overlap_effect', '-')} -> ok={r.get('primary_posthoc_overlap_effect_ok', '-')}")
    # ---------------------------------------------------------------- bottom line (numbers read above)
    try:
        add("")
        add("[8] BOTTOM LINE (every number from the CSVs cited in [1]-[7]).")
        a = agr[(agr.bench_arm == "bench") & (agr.scheme == "primary")].set_index("model")
        add("  Route labels (pre-registered rules, same label as the paper): " + "; ".join(
            f"{MN[m]} {a.loc[m, 'same_label_as_paper']}/{a.loc[m, 'n_tasks']}" for m in ["lgbm", "lr", "ebm", "ebm512", "ebmpos"]
            if m in a.index) + ". The only changes are solved tasks the weaker model does not solve ([7]); under the")
        add("  PRIMARY rules all 8 failure labels keep their label (for the 5 shared-motif labels this is nearly automatic).")
        sa = agr[(agr.bench_arm == "bench") & (agr.scheme == "sensitivity")].set_index("model")
        add("  Pre-registered SENSITIVITY rule (composition fraction >= 0.80): failure labels kept " + "; ".join(
            f"{MN[m]} {sa.loc[m, 'failure_labels_same_as_paper']}/{sa.loc[m, 'n_failure_labels_paper']}"
            for m in ["lgbm", "lr", "ebm", "ebm512", "ebmpos"] if m in sa.index) + " (changes: " + "; ".join(
            f"{MN[m]} {sa.loc[m, 'disagreements_vs_paper']}" for m in ["lr", "ebm", "ebm512", "ebmpos"]
            if m in sa.index and isinstance(sa.loc[m, 'disagreements_vs_paper'], str)) + ").")
        oa = agr[(agr.bench_arm == "bench") & (agr.scheme == "posthoc_overlap")].set_index("model")
        if not oa.empty:
            add("  Post hoc OVERLAP labelling rule (P11b): same label " + "; ".join(
                f"{MN[m]} {oa.loc[m, 'same_label_as_paper']}/{oa.loc[m, 'n_tasks']} (failure {oa.loc[m, 'failure_labels_same_as_paper']}/"
                f"{oa.loc[m, 'n_failure_labels_paper']})" for m in ["lgbm", "lr", "ebm", "ebm512", "ebmpos"] if m in oa.index)
                + "; gate-positive failing tasks: " + " | ".join(
                f"{MN[m]}: {oa.loc[m, 'gate_positive_failing_labels']}" for m in MODELS if m in oa.index) + ".")
        ba = agr[(agr.bench_arm == "bench") & (agr.scheme == "bibm")].set_index("model")
        add("  BIBM vocabulary: the submitted 'contamination (weak)' label of enh_ensembl fails the overlap criterion under "
            + ", ".join(MN[m] for m in ["lgbm", "lr", "ebm", "ebm512", "ebmpos"] if m in ba.index and
                        "human_enhancers_ensembl" in str(ba.loc[m, "posthoc_overlap_failures"]))
            + " (it loses LESS above-chance AUROC than the clean control, [3c']).")
        add("  Contamination needs the model-free gate AND a model-dependent effect: the gate blocks the control (where the")
        add("  equalized-MCC collapse also fires under EBM/LR-ext); among gate-positive failing tasks the model-dependent half")
        add("  decides, and the pre-registered MCC readout itself is non-diagnostic (random negatives produce most of it, [3b]).")
        p = r2[r2.arm == "pos"]
        s = {m: r2_summary(p, m) for m in MODELS + ["ebmpos"]}
        add("  Route 2: splice is the only task meeting the positional rule under LightGBM and LR on all 15 tasks, and under"
            " the EBM on the tasks where it received binned k-mers (" + ", ".join(
                f"{MN[m]} {s[m]['splice'].delta_mcc:+.3f}" for m in MODELS + ["ebmpos"] if s.get(m)) + "); EBM-pos (binned")
        add(f"  k-mers guaranteed) meets it on: {', '.join(SHORT[t] for t in s['ebmpos']['rule_tasks']) if s.get('ebmpos') else 'n/a'}.")
        dp = dose[dose.motif == "POOLED"].set_index("model")
        add("  Route 1: the dose-response slope holds under every model (" + ", ".join(
            f"{MN[m]} {dp.loc[m, 'slope']:+.2f} [{dp.loc[m, 'slope_lo']:+.2f},{dp.loc[m, 'slope_hi']:+.2f}]" for m in MODELS if m in dp.index)
            + "); the real-task Spearman (n=9, uncapped) is positive under every model but significant only with LightGBM (" + ", ".join(
            f"{MN[m]} {r.rho:+.2f} [{r.boot_lo:+.2f},{r.boot_hi:+.2f}]" for m in MODELS
            for _, r in r1[(r1.model == m) & (r1.analysis == N9 + "_uncapped")].iterrows()) + ").")
        c3 = r3s[(r3s.task == "human_enhancers_cohn") & (r3s.feat == "kmer")].set_index("model")
        d3 = r3s[(r3s.task == "drosophila_enhancers_stark") & (r3s.feat == "kmer")].set_index("model")
        pa = agr[(agr.bench_arm == "bench") & (agr.scheme == "primary")].set_index("model")
        add("  Labels preserved when the contamination effect is the post hoc AUROC criterion (size-controlled, leak-free nt): " + "; ".join(
            f"{MN[m]} {pa.loc[m, 'preserved_posthoc_auroc_criterion']}/{pa.loc[m, 'n_tasks']}"
            + (f" (fails: {pa.loc[m, 'posthoc_auroc_failures']})" if isinstance(pa.loc[m, 'posthoc_auroc_failures'], str) else "")
            for m in MODELS if m in pa.index) + "; under the overlap criterion: " + "; ".join(
            f"{MN[m]} {pa.loc[m, 'preserved_posthoc_overlap_criterion']}/{pa.loc[m, 'n_tasks']}" for m in MODELS if m in pa.index) + ".")
        add("  Route 3: random-negative design, AUROC lost beyond size-matched random negatives: cohn " + ", ".join(
            f"{MN[m]} {c3.loc[m, 'comp_effect']:+.3f} [{c3.loc[m, 'comp_effect_lo']:+.3f},{c3.loc[m, 'comp_effect_hi']:+.3f}]"
            for m in MODELS if m in c3.index) + "; the clean control ALSO loses " + ", ".join(
            f"{MN[m]} {d3.loc[m, 'comp_effect']:+.3f} [{d3.loc[m, 'comp_effect_lo']:+.3f},{d3.loc[m, 'comp_effect_hi']:+.3f}]"
            for m in MODELS if m in d3.index) + ".")
        add("  Contamination-specific contrast (removal-free overlap design, cohn minus control): " + ", ".join(
            f"{MN[r.model]} {r['diff']:+.3f} [{r.diff_lo:+.3f},{r.diff_hi:+.3f}]" for _, r in ovc.iterrows()
            if r.task == "human_enhancers_cohn" and r.model in MODELS) + "; random design: " + ", ".join(
            f"{MN[r.model]} {r['diff']:+.3f} [{r.diff_lo:+.3f},{r.diff_hi:+.3f}]" for _, r in r3c.iterrows()
            if r.task == "human_enhancers_cohn" and r.model in MODELS) + ".")
        add("  Gate-positive but uncontaminated enh_ensembl in the overlap design (enh_ensembl minus control): " + ", ".join(
            f"{MN[r.model]} {r['diff']:+.3f} [{r.diff_lo:+.3f},{r.diff_hi:+.3f}]" for _, r in ovc.iterrows()
            if r.task == "human_enhancers_ensembl" and r.model in MODELS) + "; ocr_ensembl (gate-negative): " + ", ".join(
            f"{MN[r.model]} {r['diff']:+.3f} [{r.diff_lo:+.3f},{r.diff_hi:+.3f}]" for _, r in ovc.iterrows()
            if r.task == "human_ocr_ensembl" and r.model in MODELS) + ".")
        ck = r3s[(r3s.task == "human_enhancers_cohn") & (r3s.feat == "kmer")].set_index("model")
        add("  Size share of the cohn MCC drop (random negatives alone): " + ", ".join(
            f"{MN[m]} {ck.loc[m, 'mcc_share_of_drop_from_size']:.2f} [{ck.loc[m, 'mcc_share_of_drop_from_size_lo']:.2f},"
            f"{ck.loc[m, 'mcc_share_of_drop_from_size_hi']:.2f}]" for m in MODELS if m in ck.index) + ".")
    except Exception as e:  # noqa: BLE001
        add(f"  (bottom line unavailable: {e})")
    with open(os.path.join(R, "alt_model_interpretation.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    # compact paper table (per model): the numbers a 7-page paper can carry
    rows = []
    for m in MODELS + ["lrx", "lra", "ebm512", "ebmpos"]:
        row = dict(model=m)
        if not agr.empty:
            a = agr[(agr.model == m) & (agr.bench_arm == "bench")]
            for _, x in a.iterrows():
                row[f"{x.scheme}_same_label"] = f"{x.same_label_as_paper}/{x.n_tasks}"
                row[f"{x.scheme}_preserved_prereg"] = f"{x.preserved_prereg_criterion}/{x.n_tasks}"
                row[f"{x.scheme}_preserved_posthoc_auroc"] = f"{x.preserved_posthoc_auroc_criterion}/{x.n_tasks}"
                row[f"{x.scheme}_preserved_posthoc_overlap"] = f"{x.preserved_posthoc_overlap_criterion}/{x.n_tasks}"
                row[f"{x.scheme}_failure_labels_same"] = f"{x.failure_labels_same_as_paper}/{x.n_failure_labels_paper}"
        if not dose.empty:
            d = dose[(dose.model == m) & (dose.motif == "POOLED")]
            if not d.empty:
                row["dose_slope"] = f"{d.slope.iloc[0]:+.2f} [{d.slope_lo.iloc[0]:+.2f},{d.slope_hi.iloc[0]:+.2f}]"
        if not r1.empty:
            for an, key in ((N9, "route1_rho_n9_capped_prereg"), (N11, "route1_rho_n11_capped_prereg"),
                            (N9 + "_uncapped", "route1_rho_n9_uncapped"), (N11 + "_uncapped", "route1_rho_n11_uncapped")):
                d = r1[(r1.model == m) & (r1.analysis == an)]
                if not d.empty:
                    row[key] = (f"{d.rho.iloc[0]:+.2f} [{d.boot_lo.iloc[0]:+.2f},{d.boot_hi.iloc[0]:+.2f}] "
                                f"p_perm={d.p_perm.iloc[0]:.2g} p_t={d.p_spearman_t_approx.iloc[0]:.2g}")
        if not r2.empty:
            sm = r2_summary(r2[r2.arm == "pos"], m)
            if sm is not None and sm["splice"] is not None:
                sp = sm["splice"]
                row["splice_dMCC"] = f"{sp.delta_mcc:+.3f} [{sp.delta_lo:+.3f},{sp.delta_hi:+.3f}]"
                row["max_abs_dMCC_other14_any"] = f"{sm['max_all']:.3f} ({SHORT[sm['max_all_task']]}, CI [{sm['max_all_lo']:+.3f},{sm['max_all_hi']:+.3f}])"
                row["max_abs_dMCC_other14_ci_significant"] = (f"{sm['max_sig']:.3f} ({SHORT[sm['max_sig_task']]})"
                                                             if sm["max_sig_task"] else "none")
                row["positional_rule_tasks"] = ";".join(SHORT[t] for t in sm["rule_tasks"])
                row["tasks_B8_without_binned_features"] = ";".join(SHORT[t] for t in sm["zero_bin"])
        if not r3s.empty:
            for t, lf, key in (("human_enhancers_cohn", False, "cohn"), ("nt_enhancers", True, "nt_leakfree"),
                               ("drosophila_enhancers_stark", False, "drosophila")):
                d = r3s[(r3s.model == m) & (r3s.task == t) & (r3s.feat == "kmer") & (r3s.leakfree_test.astype(bool) == lf)]
                if not d.empty:
                    d = d.iloc[0]
                    row[f"r3_{key}_Ar_Am"] = f"{d.auroc_random_mean:.3f}->{d.auroc_matched:.3f}"
                    row[f"r3_{key}_auroc_effect_random_design"] = f"{d.comp_effect:+.3f} [{d.comp_effect_lo:+.3f},{d.comp_effect_hi:+.3f}]"
                    row[f"r3_{key}_mcc_orig_random_matched"] = f"{d.mcc_orig:.3f}/{d.mcc_random_mean:.3f}/{d.mcc_matched:.3f}"
                    row[f"r3_{key}_mcc_drop_share_from_size"] = (f"{d.mcc_share_of_drop_from_size:.2f} "
                                                                 f"[{d.mcc_share_of_drop_from_size_lo:.2f},{d.mcc_share_of_drop_from_size_hi:.2f}]")
        if not r3c.empty:
            d = r3c[(r3c.model == m) & (r3c.task == "human_enhancers_cohn")]
            if not d.empty:
                row["r3_cohn_minus_control_random_design"] = f"{d['diff'].iloc[0]:+.3f} [{d.diff_lo.iloc[0]:+.3f},{d.diff_hi.iloc[0]:+.3f}]"
        if not ovc.empty:
            for t, lf, key in (("human_enhancers_cohn", False, "cohn"), ("nt_enhancers", True, "nt_leakfree"),
                               ("human_enhancers_ensembl", False, "enh_ensembl"), ("human_ocr_ensembl", False, "ocr")):
                d = ovc[(ovc.model == m) & (ovc.task == t) & (ovc.leakfree_test.astype(bool) == lf)]
                if not d.empty:
                    row[f"r3_{key}_minus_control_overlap_design"] = (f"{d['diff'].iloc[0]:+.3f} [{d.diff_lo.iloc[0]:+.3f},"
                                                                     f"{d.diff_hi.iloc[0]:+.3f}]")
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(R, "alt_model_paper_table.csv"), index=False)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
    import _common as C
    write(C.RESULTS_DIR)
