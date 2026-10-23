#!/usr/bin/env python3
"""Writes the human-readable reading of the multivar analysis FROM THE RESULT CSVs ONLY, so that every
number in the text is read from a file that run_multivar.py (and the upstream steps) regenerated.

Outputs (results/upgrades/):
  multivar_interpretation.txt   the reading, with a file/row/column citation after every number block
  multivar_paper_table.csv      compact table for the paper/rebuttal: statistic, value, source file + row/column
  multivar_paper_snippets.txt   LaTeX: main-text paragraph (<= 120 words, asserted), a rebuttal paragraph,
                                the table as LaTeX rows, corrected rows for the paper's Table II (a row may
                                say "homology-clean" only if its task set has no task >= 0.05 -- asserted),
                                a Limitations (i) sentence and a Table II footnote on task non-independence
Every qualitative statement in the text ("CI includes 0", "worse", "0/11 folds", ...) is guarded by an
assertion on the numbers it describes: if a re-run changes a result, this script fails instead of printing
a stale sentence.
Run: bibm/lowcpu.sh /Users/rikhinkavuru/Downloads/venv/bin/python src/upgrades/multivar/write_interpretation.py
"""
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RD = os.path.join(ROOT, "results", "upgrades")


def rd(name):
    return pd.read_csv(os.path.join(RD, name))


def one(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    assert m.sum() == 1, (kw, int(m.sum()))
    return df[m].iloc[0]


def s(x, d=3):
    """signed"""
    return f"{x:+.{d}f}"


def u(x, d=3):
    return f"{x:.{d}f}"


def ci(lo, hi, d=3, signed=True):
    f = s if signed else u
    return f"[{f(lo, d)},{f(hi, d)}]"


def excl0(lo, hi):
    return bool(lo > 0 or hi < 0)


def check(cond, msg):
    assert bool(cond), "stale sentence guard failed: " + msg


def main():
    cov = rd("multivar_covariates.csv")
    models = rd("multivar_models.csv")
    top = rd("multivar_top_models.csv")
    nested = rd("multivar_nested.csv")
    perm = rd("multivar_permutation.csv")
    part = rd("multivar_partial.csv")
    lmgd = rd("multivar_lmg.csv")
    resid = rd("multivar_residuals.csv")
    noise = rd("multivar_noise.csv")
    rev = rd("multivar_reviewer_model.csv")
    syn = rd("multivar_synthetic_transfer.csv")
    p1 = rd("multivar_posthoc_residual_null.csv")
    p2 = rd("multivar_posthoc_slope_transfer.csv")
    h_alt = rd("multivar_posthoc_homology_alt.csv")
    p3 = rd("multivar_posthoc_homology_models.csv")
    p4 = rd("multivar_posthoc_orientation.csv")
    p5 = rd("multivar_posthoc_homology_clean.csv")
    p6 = rd("multivar_posthoc_dedup_eval.csv")
    exd = rd("multivar_posthoc_exact_dup.csv")
    p7 = rd("multivar_posthoc_gc_vs_disc.csv")
    p8 = rd("multivar_posthoc_orientation_robustness.csv")
    p9 = rd("multivar_posthoc_bounded.csv")
    p10 = rd("multivar_posthoc_synthetic_range.csv").set_index("quantity")
    p11 = rd("multivar_posthoc_family.csv")
    p12 = rd("multivar_posthoc_distinct_datasets.csv")
    p13 = rd("multivar_posthoc_locus_independent.csv")
    lkl = rd("multivar_posthoc_task_locus_links.csv")
    lko = pd.read_csv(os.path.join(RD, "multivar_posthoc_task_locus_overlap.csv"), dtype={"ref_label": str})
    s7 = rd("multivar_posthoc_subsets7.csv")
    famt = rd("multivar_posthoc_family_tasks.csv").set_index("task")
    pt7 = rd("multivar_posthoc_gc_vs_disc_per_task.csv").set_index("task")
    tov = rd("multivar_posthoc_task_overlap.csv")
    tid = rd("multivar_posthoc_task_identity.csv").set_index(["task", "check"])["value"]
    tcl = rd("multivar_posthoc_task_overlap_classes.csv")
    hom = rd("multivar_homology.csv")
    comp = rd("multivar_comp_reproduction.csv")
    summ = rd("multivar_summary.csv").set_index("key")
    cid = rd("multivar_posthoc_class_identity.csv").set_index(["task", "check"])["value"]

    T = []           # interpretation lines
    TAB = []         # paper table rows

    def tab(stat, value, source):
        TAB.append(dict(statistic=stat, value=value, source=source))

    S11 = "S11/Y1_MCC"
    S15 = "S15/Y1_MCC"
    # ---------------------------------------------------------------- core numbers
    a0 = one(p4, config=S11, variant="as_published")                  # A0 disc-only LOTO + route1 bootstrap CI
    md = one(models, config=S11, family="poolA", model="disc")
    mg = one(models, config=S11, family="poolA", model="gc_mean")
    check(abs(a0["loto_R2"] - md["loto_R2"]) < 1e-9 and abs(a0["loto_MAE"] - md["loto_MAE"]) < 1e-9, "A0 sources agree")
    prim = nested[nested["in_decision_rule"]].iloc[0]
    assert len(nested[nested["in_decision_rule"]]) == 1
    a4a = one(perm, config=S11, test="A4a_global_bestLOTO_R2_poolA")
    a4b = one(perm, config=S11, test="A4b_incremental_bestLOTO_MAE_disc_anchored")
    dec = summ.loc["DECISION_covariates_improve_on_disc", "value"]
    check(dec == "NO", "decision rule verdict NO")
    check(prim["nested_loto_MAE"] >= prim["disc_only_loto_MAE"], "(a) fails")
    check(not excl0(prim["dMAE_lo"], prim["dMAE_hi"]), "(b) fails: dMAE CI includes 0")
    check(a4b["p"] >= 0.05, "(c) fails")
    n11 = int(prim["n"])
    check(n11 == 11, "S11 n")
    top_aicc = models[(models["config"] == S11) & (models["family"] == "poolA")].sort_values(["AICc", "p"]).iloc[0]
    top_mae = models[(models["config"] == S11) & (models["family"] == "poolA")].sort_values(["loto_MAE", "p"]).iloc[0]
    tg = one(top, config=S11, family="poolA", ranking="AICc", rank=1, predictor="gc_mean")
    g7 = one(p7, config=S11, variant="as_published")
    g7f = one(p7, config=S11, variant="P4_orientation_consistent")
    g7s = one(p7, config=S15, variant="as_published")
    pa = part[part["config"] == S11].set_index("covariate")
    pa15 = part[part["config"] == S15].set_index("covariate")
    pau = part[part["config"] == "S11/Y2_AUROC"].set_index("covariate")
    # reverse partial in A1 and P7 must be the same computation
    check(abs(pa.loc["gc_mean", "posthoc_rev_partial_rho_c_given_disc"] - g7["partial_gc_given_disc"]) < 1e-12 and
          abs(pa.loc["gc_mean", "posthoc_rev_partial_boot_lo"] - g7["partial_gc_given_disc_lo"]) < 1e-12 and
          abs(pa.loc["gc_mean", "posthoc_rev_partial_boot_hi"] - g7["partial_gc_given_disc_hi"]) < 1e-12,
          "A1 reverse partial == P7")
    check(abs(pa.loc["gc_mean", "partial_rho"] - g7["partial_disc_given_gc"]) < 1e-12 and
          abs(pa.loc["gc_mean", "partial_boot_lo"] - g7["partial_disc_given_gc_lo"]) < 1e-12, "A1 == P7 (disc|GC)")
    fam_b = one(p11, variant="as_published", analysis="between_family_S11")
    fam_bf = one(p11, variant="P4_orientation_consistent", analysis="between_family_S11")
    fam_s10 = one(p11, variant="as_published", analysis="between_family_S10_excl_regulatory")
    q8_ = one(p5, subset="paper9_minus_human_enhancers_ensembl")
    B_ = 1000
    fw = {(r["variant"], r["analysis"]): r for _, r in p11[p11["analysis"].str.startswith("within")].iterrows()}
    g7_ = one(p7, config=S11, variant="as_published")
    two7 = pt7[pt7["nested_n_covariates"] > 1]
    n_worse7 = int(two7["nested_worse_than_gc_only"].sum())
    n_gc_alone = int((pt7["nested_selected"] == "gc_mean").sum())
    sel_two7 = ", ".join(f"{t}: {r['nested_selected']}" for t, r in two7.iterrows())
    worse_two7 = ", ".join(f"{t} {u(r['abs_err_nested'])} vs GC-only {u(r['abs_err_gc_only'])}"
                           for t, r in two7[two7["nested_worse_than_gc_only"]].iterrows())

    # ================================================================ header
    T += ["MULTIVARIATE COVARIATE ANALYSIS OF REAL-TASK LEARNABILITY -- Reviewer 1, comment 1",
          "=" * 100,
          "Reviewer: \"Please provide a multivariate regression analysis on the real-world datasets controlling for",
          "sequence length, GC content variance, and homology redundancy to explain what unobserved factors drive",
          "performance deviations, addressing the exceptionally low goodness-of-fit and high fitting uncertainty",
          "observed on real tasks despite the strong synthetic dose-response.\"",
          "",
          "GENERATED by src/upgrades/multivar/write_interpretation.py from the result CSVs (every number below is",
          "read from the file cited in brackets; nothing is typed by hand).  Plan pre-registered before any new",
          "covariate existed: results/upgrades/multivar_preregistration.txt (docstring of run_multivar.py, sha256",
          "recorded; unchanged).  Full numeric log: multivar_analysis_log.txt.  Units = tasks.  S11 = the 11",
          "TF-motif tasks (PRIMARY, the Route-1 scope); S15 = all 15.  Outcome = Table-1 benchmark MCC (primary);",
          "AUROC secondary.  Seed 42; 1000 task-bootstrap resamples (percentile CIs); 1000 permutations.",
          "Everything labelled P1-P13 is POST HOC (not pre-registered); P4, P5-P13 were motivated by results or by",
          "the verification rounds and are reported as diagnoses, never as replacement headlines.  Every Spearman p of",
          "a task set (n <= 12) is an exact permutation p over all n! orderings ('perm p' / 'exact p'), shown next to",
          "scipy's t-approximation where that matters; residual-permutation p of partials is exact for n <= 9 and",
          "Monte Carlo (200,000 orderings) above; a degenerate |rho| = 1 has no p and no CI ('degenerate'), and no",
          "p is ever printed as zero.",
          ""]
    # ================================================================ covariates
    cr = float(comp["abs_diff_mcc"].max())
    check(cr == 0.0, "composition reproduction exact")
    sub = hom[hom["query_mode"] == "subsample5000"]
    hw = float(((sub["frac_ge_0_7_hi"] - sub["frac_ge_0_7_lo"]) / 2).max())
    T += ["COVARIATES (multivar_covariates.csv, 15 rows; train/test counts asserted against Table 1 on every load)",
          "  disc = max positive-direction motif-only AUROC (cross_suite_summary.csv); log10 mean length; log10(1 +",
          "  length SD); GC mean; within-class pooled GC SD; class GC gap |Cohen's d|; homology = fraction of test",
          "  sequences with a >= 0.7 exact 8-mer-Jaccard near-duplicate in train (4 tasks from homology_leakage.csv,",
          f"  11 new in multivar_homology.csv, same method; {len(sub)} large tasks use a pre-registered 5,000-test-",
          f"  sequence sample against the FULL train split, Wilson CI half-width <= {hw:.4f}); log10 n_train; 3-class",
          "  indicator; positional-rescue delta MCC; composition-only LightGBM MCC (reproduces the 5 published",
          f"  composition_clean values, max |diff| = {cr:.1f}, multivar_comp_reproduction.csv).",
          ""]

    # ================================================================ H1 decision rule
    sel = prim["selected_models"]
    nd, ng = int(prim["n_folds_selected_with_disc"]), int(prim["n_folds_selected_with_gc_mean"])
    T += ["H1  PRE-REGISTERED VERDICT: no measured covariate set improves on discriminability out of sample; at",
          "    n = 11 tasks the data cannot resolve covariate effects beyond discriminability.",
          f"  Disc-only LOTO (S11) reproduced exactly: R^2 {s(a0['loto_R2'])} {ci(a0['loto_R2_lo'], a0['loto_R2_hi'])}, "
          f"MAE {u(a0['loto_MAE'])} {ci(a0['loto_MAE_lo'], a0['loto_MAE_hi'], signed=False)}",
          "    [multivar_posthoc_orientation.csv row S11/as_published; = multivar_models.csv S11/poolA/model=disc].",
          "  Decision rule [multivar_nested.csv row in_decision_rule==True; multivar_permutation.csv]:",
          f"    (a) nested-LOTO MAE of the AICc-selected Pool-A model (175 subsets, <= 3 of 10 covariates, the search",
          f"        rerun inside every fold) {u(prim['nested_loto_MAE'])} vs disc-only {u(prim['disc_only_loto_MAE'])}  -> NO",
          f"    (b) paired task-bootstrap dMAE {s(prim['dMAE'])} {ci(prim['dMAE_lo'], prim['dMAE_hi'])} (includes 0)  -> NO",
          f"    (c) incremental permutation test A4b (disc-anchored family, 46 models; rows of the other 9 covariates",
          f"        permuted) p = {u(a4b['p'])}  -> NO",
          f"  What the nested search actually selected: {sel} -- a GC-mean model in {ng}/{n11} outer folds, a",
          f"  disc-containing model in {nd}/{n11}.  So the tie '{u(prim['nested_loto_MAE'])} vs {u(prim['disc_only_loto_MAE'])}' is GC-mean models vs",
          "  discriminability, not 'covariates add nothing to discriminability' (see H2).",
          f"  Global permutation A4a: the best of 175 in-sample-searched LOTO R^2 values ({u(a4a['observed'])}) is below the",
          f"  95th percentile of permuted outcomes ({u(a4a['null_critical'])}; p = {u(a4a['p'])}): subset search alone buys that much.",
          f"  Exploratory in-sample winners (multivar_models.csv S11/poolA): top AICc = {top_aicc['model']} (R^2 {u(top_aicc['R2'])},",
          f"  LOTO MAE {u(top_aicc['loto_MAE'])}; std beta of GC mean {s(tg['std_beta'])}, bootstrap {ci(tg['boot_lo'], tg['boot_hi'])},",
          f"  multivar_top_models.csv); top LOTO MAE = {top_mae['model']} ({u(top_mae['loto_MAE'])}).  These are",
          "  exploratory because of (a)-(c).",
          f"  Why the nested MAE ({u(prim['nested_loto_MAE'])}) sits far above the GC-only LOTO MAE ({u(g7_['gc_only_loto_MAE'])}) "
          "[multivar_posthoc_gc_vs_disc_per_task.csv]:",
          f"  GC-only is the in-sample AICc winner, so its LOTO MAE is NOT selection-honest (optimistic); inside the",
          f"  folds the search picked GC mean alone in {n_gc_alone}/{n11} folds (identical prediction) but added a second,",
          f"  unstable covariate in {len(two7)}/{n11} ({sel_two7}),",
          f"  which raised the error in {n_worse7} of them ({worse_two7}).",
          ""]
    check(nd == 0 and ng == n11, "nested selection: disc 0 folds, GC mean all folds")
    check(abs(pt7["abs_err_nested"].mean() - prim["nested_loto_MAE"]) < 1e-9 and
          abs(pt7["abs_err_gc_only"].mean() - g7_["gc_only_loto_MAE"]) < 1e-9, "per-task file == summary MAEs")
    check(n_gc_alone + len(two7) == n11 and (pt7.loc[pt7["nested_selected"] == "gc_mean", "abs_err_nested"]
                                             - pt7.loc[pt7["nested_selected"] == "gc_mean", "abs_err_gc_only"]).abs().max() < 1e-9,
          "GC-alone folds reproduce GC-only LOTO")
    check(top_aicc["model"] == "gc_mean", "top AICc model is GC mean")
    check(a4a["observed"] < a4a["null_critical"], "A4a observed below null 95th pct")
    tab("Disc-only LOTO, S11 (pre-registered reproduction)",
        f"MAE {u(a0['loto_MAE'])} {ci(a0['loto_MAE_lo'], a0['loto_MAE_hi'], signed=False)}; R^2 {s(a0['loto_R2'], 2)} "
        f"{ci(a0['loto_R2_lo'], a0['loto_R2_hi'], 2)}", "multivar_posthoc_orientation.csv S11/as_published")
    tab("Nested best-subset (<=3 of 10 covariates, AICc), S11",
        f"MAE {u(prim['nested_loto_MAE'])}; dMAE vs disc {s(prim['dMAE'])} {ci(prim['dMAE_lo'], prim['dMAE_hi'])}; "
        f"A4b p={u(a4b['p'], 2)}; GC mean selected in {ng}/{n11} folds, disc in {nd}/{n11}",
        "multivar_nested.csv in_decision_rule==True; multivar_permutation.csv A4b")

    # ================================================================ H2 disc vs GC mean / family
    check(not excl0(g7["dMAE_lo"], g7["dMAE_hi"]), "GC-only vs disc dMAE CI includes 0")
    check(not excl0(g7["partial_disc_given_gc_lo"], g7["partial_disc_given_gc_hi"]), "disc|GC CI includes 0")
    check(excl0(g7["partial_gc_given_disc_lo"], g7["partial_gc_given_disc_hi"]), "GC|disc CI excludes 0")
    check(bool(fam_b["gc_separates_promoter_from_enhancer"]), "GC separates families")
    for fb_ in (fam_b, fam_bf, fam_s10):
        check(fb_["dMAE_hi"] < 0, "family label beats disc (dMAE CI < 0)")
    check(not excl0(fam_b["partial_disc_given_indicator_lo"], fam_b["partial_disc_given_indicator_hi"]), "disc|family CI incl 0")
    check(excl0(fam_b["partial_indicator_given_disc_lo"], fam_b["partial_indicator_given_disc_hi"]), "family|disc CI excl 0")
    for (v_, a_), r_ in fw.items():
        if "homology_clean" not in a_:
            check(r_["spearman_disc_mcc_p_exact"] > 0.05, "within-family disc~MCC not significant (all S11)")
        else:
            check(r_["spearman_disc_mcc_p_exact"] > 0.05 and r_["n"] <= 4, "clean within-family: n<=4, exact p>0.05")
    # ---- the two-cluster structure (fix round 2): family label == Table I 'solved' == MCC dichotomy
    ind1, ind0 = famt[famt["family_indicator"] == 1], famt[famt["family_indicator"] == 0]
    n_pos1 = int((ind1["disc_only_loto_resid"] > 0).sum())
    n_neg0 = int((ind0["disc_only_loto_resid"] < 0).sum())
    check(bool(fam_b["indicator_equals_table1_solved"]) and bool(fam_b["indicator_is_mcc_dichotomy"]),
          "family label == Table I solved == MCC dichotomy")
    check(set(ind1.index) == set(famt.index[famt["table1_route"] == "solved"]), "solved set")
    d12 = p12.set_index("subset")
    hc6 = d12.loc["homology_clean7_minus_union"]
    check(bool(hc6["homology_clean"]) and int(hc6["within_promoter_family_n"]) == 3 and
          int(hc6["within_enhancer_family_n"]) == 3, "clean distinct within-cluster n = 3 / 3")
    fa, fn = famt.loc["nt_promoter_all"], famt.loc["nt_promoter_no_tata"]
    check(fa["mcc_lo"] <= fn["bench_mcc"] <= fa["mcc_hi"] and fn["mcc_lo"] <= fa["bench_mcc"] <= fn["mcc_hi"],
          "nt_promoter_all vs no_tata MCC within each other's CIs")
    T += ["H2  ON REAL TASKS DISCRIMINABILITY CANNOT BE SEPARATED FROM GC CONTENT OR FROM THE SOLVED/FAILING SPLIT",
          "    (post hoc P7, P11, P12; the reverse partials are a post-hoc column of the pre-registered A1).",
          f"  Collinearity: Spearman(disc, GC mean) {s(g7['spearman_disc_gc'])} on S11 [multivar_posthoc_gc_vs_disc.csv].",
          f"  GC mean alone predicts held-out MCC at least as well: LOTO MAE {u(g7['gc_only_loto_MAE'])} vs disc-only "
          f"{u(g7['disc_only_loto_MAE'])}, paired dMAE {s(g7['dMAE_gc_minus_disc'])} {ci(g7['dMAE_lo'], g7['dMAE_hi'])} (CI includes 0);",
          f"  LOTO R^2 {s(g7['gc_only_loto_R2'])} vs {s(g7['disc_only_loto_R2'])} (GC-only was chosen in-sample, so this is optimistic; H1).",
          f"  Mutual control: rho(disc, MCC | GC) {s(g7['partial_disc_given_gc'])} {ci(g7['partial_disc_given_gc_lo'], g7['partial_disc_given_gc_hi'])} "
          f"(t-test p = {u(g7['partial_disc_given_gc_p'])});",
          f"  rho(GC, MCC | disc) {s(g7['partial_gc_given_disc'])} {ci(g7['partial_gc_given_disc_lo'], g7['partial_gc_given_disc_hi'])} "
          f"(p = {u(g7['partial_gc_given_disc_p'])}).  On all 11 tasks GC mean survives control for disc and disc does not",
          "  survive control for GC (on the n = 8 set of H7 this reverses; below).  LMG share of disc in disc + GC mean: "
          f"{u(g7['lmg_disc_share'], 2)} {ci(g7['lmg_disc_share_lo'], g7['lmg_disc_share_hi'], 2, False)} of R^2 {u(g7['R2_disc_plus_gc'])}.",
          f"  With the post-hoc enhancer-oriented disc for nt_enhancers_types (P4): rho(disc|GC) {s(g7f['partial_disc_given_gc'])} "
          f"{ci(g7f['partial_disc_given_gc_lo'], g7f['partial_disc_given_gc_hi'])} (p = {u(g7f['partial_disc_given_gc_p'])}),",
          f"  rho(GC|disc) {s(g7f['partial_gc_given_disc'])} {ci(g7f['partial_gc_given_disc_lo'], g7f['partial_gc_given_disc_hi'])} "
          f"(p = {u(g7f['partial_gc_given_disc_p'])}); GC-only minus disc-only dMAE {s(g7f['dMAE_gc_minus_disc'])} "
          f"{ci(g7f['dMAE_lo'], g7f['dMAE_hi'])}.",
          "  THE TWO-CLUSTER STRUCTURE (P11; multivar_posthoc_family.csv, multivar_posthoc_family_tasks.csv).  The post-hoc",
          "  task-family label (promoters + the 3-class regulatory task = 1, enhancer/OCR = 0; read from the task names)",
          "  is EXACTLY Table I's 'solved' set (indicator_equals_table1_solved), and the two groups' MCC ranges do not",
          f"  overlap (solved min {u(fam_b['mcc_min_indicator1'])} > failing max {u(fam_b['mcc_max_indicator0'])}; the paper defines 'solved' by non-overlapping",
          "  MCC CIs).  The label is therefore a dichotomy of the outcome itself: the real-task MCCs form two clusters, and",
          f"  a two-level label predicts them almost by construction (LOTO MAE {u(fam_b['loto_MAE_indicator_only'])}, R^2 "
          f"{s(fam_b['loto_R2_indicator_only'])}, vs disc {u(fam_b['loto_MAE_disc_only'])};",
          f"  dMAE {s(fam_b['dMAE_indicator_minus_disc'])} {ci(fam_b['dMAE_lo'], fam_b['dMAE_hi'])}).  It identifies NO additional explanatory "
          "factor; it shows that the real-task",
          "  regression carries essentially ONE between-cluster contrast.  GC mean separates the clusters (promoter GC >= "
          f"{u(fam_b['gc_promoter_family_min'])} >",
          f"  enhancer/OCR GC <= {u(fam_b['gc_enhancer_family_max'])}), and so does disc, less cleanly: the disc-only LOTO residuals are "
          f"positive for {n_pos1}/{len(ind1)} solved tasks",
          f"  and negative for {n_neg0}/{len(ind0)} failing tasks (the disc line under-separates the clusters).  Given the label, disc adds",
          f"  nothing resolvable (disc + label {u(fam_b['loto_MAE_disc_plus_indicator'])}; rho(disc, MCC | label) "
          f"{s(fam_b['partial_disc_given_indicator'])} {ci(fam_b['partial_disc_given_indicator_lo'], fam_b['partial_disc_given_indicator_hi'])},",
          f"  t p = {u(fam_b['partial_disc_given_indicator_p'])}, residual-perm p = {u(fam_b['partial_disc_given_indicator_p_perm'])}), while "
          f"rho(label, MCC | disc) {s(fam_b['partial_indicator_given_disc'])} "
          f"{ci(fam_b['partial_indicator_given_disc_lo'], fam_b['partial_indicator_given_disc_hi'])}.",
          f"  Without the regulatory task (n = {int(fam_s10['n'])}): label {u(fam_s10['loto_MAE_indicator_only'])} vs disc "
          f"{u(fam_s10['loto_MAE_disc_only'])} (dMAE {ci(fam_s10['dMAE_lo'], fam_s10['dMAE_hi'])}); with the P4 disc: label "
          f"{u(fam_bf['loto_MAE_indicator_only'])} vs disc {u(fam_bf['loto_MAE_disc_only'])}",
          f"  (dMAE {ci(fam_bf['dMAE_lo'], fam_bf['dMAE_hi'])}).",
          "  Within a cluster the association is not resolvable (n = 3-6, narrow disc ranges; the promoter cluster's",
          "  nt_promoter_all is the exact union of two other members, H9):"]
    for (v, a), r in fw.items():
        T.append(f"    {v:<26} {a[len('within_'):]:<40} n={int(r['n'])}  disc {ci(r['disc_min'], r['disc_max'], signed=False)}  "
                 f"Spearman(disc, MCC) {s(r['spearman_disc_mcc'])} (exact p={u(r['spearman_disc_mcc_p_exact'])});  MCC range "
                 f"{u(r['mcc_range'])} vs mean CI half-width {u(r['mean_mcc_ci_halfwidth'])}")
    cp = fw[("as_published", "within_homology_clean_promoter_plus_regulatory")]
    ce = fw[("as_published", "within_homology_clean_enhancer_ocr")]
    fam_s7 = one(p11, variant="as_published", analysis="between_family_S7_homology_clean")
    # ---- fix round 3: GC mean's own rank correlation with MCC, and where disc / GC has the lower LOTO MAE
    h11, h7c = one(p5, subset="S11_all"), one(p5, subset="rule_clean_homology_lt_0.05")
    check(abs(h11["spearman_disc_mcc_p_perm"] - h11["spearman_p_perm"]) < 1e-12, "P5 disc p columns agree")
    check(h11["spearman_gc_mcc"] >= h11["spearman"] and not excl0(h11["rho_gc_minus_rho_disc_lo"], h11["rho_gc_minus_rho_disc_hi"]),
          "'GC tracks MCC at least as closely' (S11 rho_GC >= rho_disc; difference CI includes 0)")
    TIE = 0.005                                        # 'tied' = |MAE difference| < 0.005 MCC (descriptive)
    ords = []
    for src_, df_ in (("P5", p5), ("P12", p12)):
        for _, r in df_.iterrows():
            if src_ == "P12" and r["subset"] == "S11_all":
                continue                               # identical to P5 S11_all (asserted in run_multivar.py)
            md_ = r["loto_MAE_disc"] if src_ == "P5" else r["loto_MAE_disc_only"]
            mg_ = r["loto_MAE_gc"] if src_ == "P5" else r["loto_MAE_gc_only"]
            ords.append(dict(src=src_, subset=r["subset"], n=int(r["n"]), clean=bool(r["homology_clean"]),
                             reg=bool(r["contains_regulatory"]), disc=md_, gc=mg_,
                             rev=r["loto_MAE_order_reversed_by_dropping"] if isinstance(r["loto_MAE_order_reversed_by_dropping"], str) else "",
                             low_noreg=r["loto_MAE_lower_without_regulatory"],
                             e_d=r["abs_err_disc_regulatory"], e_g=r["abs_err_gc_regulatory"]))
    ords = pd.DataFrame(ords)
    dwin = ords[ords["gc"] - ords["disc"] > TIE]
    ties_ = ords[(ords["gc"] - ords["disc"]).abs() <= TIE]
    gwin = ords[ords["disc"] - ords["gc"] > TIE]
    for _, r in dwin.iterrows():                       # GUARD for every 'disc beats GC' statement below
        check((r["clean"] and r["reg"] and r["low_noreg"] == "gc" and "human_ensembl_regulatory" in r["rev"].split(";"))
              or r["subset"] == "suite_GB_only",
              f"disc beats GC on {r['subset']} only via the regulatory task or on GB-only")
    gbo = d12.loc["suite_GB_only"]
    check(gbo["loto_MAE_lower_without_regulatory"] == "disc", "GB-only: disc lower even without the regulatory task")
    h7nr = d12.loc["homology_clean7_minus_regulatory"]
    check(abs(h7c["loto_MAE_disc"] - fam_s7["loto_MAE_disc_only"]) < 1e-12 and abs(h7c["loto_MAE_gc"] - fam_s7["loto_MAE_gc_only"]) < 1e-12,
          "P5 clean-7 MAEs == P11 S7")
    check(abs(h7nr["loto_MAE_disc_only"] - h7c["loto_MAE_disc_without_regulatory"]) < 1e-12 and
          h7nr["loto_MAE_gc_only"] < h7nr["loto_MAE_disc_only"], "clean 7 minus regulatory: GC lower (P12 row)")
    reg_gc = float(famt.loc["human_ensembl_regulatory", "gc_mean"])
    at1 = s7[s7["spearman_disc_mcc"] > 1 - 1e-12]
    check(len(at1) >= 1, "subsets7: at least one 7-task subset with rho = 1")
    check(h7c["dropped"] in list(at1["dropped"]) or set(h7c["dropped"].split(";")) in [set(x.split(";")) for x in at1["dropped"]],
          "clean 7 is one of the rho = 1 subsets")
    T += ["  On all 11 tasks the within-cluster associations are weak and not significant.",
          "  GC CONTENT TRACKS MCC AT LEAST AS CLOSELY AS DISC (post hoc, fix round 3; descriptive, not a test of GC vs",
          f"  disc) [multivar_posthoc_homology_clean.csv row S11_all]: Spearman(GC mean, MCC) {s(h11['spearman_gc_mcc'])} (exact p = "
          f"{u(h11['spearman_gc_mcc_p_perm'], 4)}) vs",
          f"  Spearman(disc, MCC) {s(h11['spearman'])} (exact p = {u(h11['spearman_p_perm'], 4)}); paired task-bootstrap 95% CI of the "
          f"difference {ci(h11['rho_gc_minus_rho_disc_lo'], h11['rho_gc_minus_rho_disc_hi'], 2)}",
          "  (includes 0: the two rank correlations are not distinguishable at n = 11).",
          f"  SENSITIVITY OF THE DISC-vs-GC LOTO-MAE ORDERING TO THE TASK SET ({len(ords)} task sets of P5 and P12; 'tied' =",
          f"  |difference| < {TIE} MCC; columns loto_MAE_*, loto_MAE_order_reversed_by_dropping, *_without_regulatory):",
          f"  * GC mean has the lower held-out MAE on {len(gwin)} sets and ties on {len(ties_)} ("
          + ", ".join(f"{r['subset']}: disc {u(r['disc'])} vs GC {u(r['gc'])}" for _, r in ties_.iterrows()) + ").",
          f"  * disc has an MAE lower by > {TIE} on {len(dwin)} sets, and each is either a homology-clean set that contains the",
          "    3-class regulatory task -- where removing that one task reverses the ordering -- or the GB-only suite:"]
    for _, r in dwin.iterrows():
        T.append(f"      {r['subset']:<40} n={r['n']:>2}  disc {u(r['disc'])} vs GC {u(r['gc'])}"
                 + (f"  (regulatory task's own LOTO error: disc {u(r['e_d'])}, GC {u(r['e_g'])}; order reversed by dropping: "
                    f"{r['rev']})" if r["reg"] and r["subset"] != "suite_GB_only" else
                    f"  (disc lower also without the regulatory task)"))
    T += [f"    On the {int(fam_s7['n'])} homology-clean tasks disc {u(fam_s7['loto_MAE_disc_only'])} vs GC {u(fam_s7['loto_MAE_gc_only'])} (label "
          f"{u(fam_s7['loto_MAE_indicator_only'])}; label - disc {ci(fam_s7['dMAE_lo'], fam_s7['dMAE_hi'])}, includes 0) rests on",
          f"    human_ensembl_regulatory: its GC mean ({u(reg_gc)}) averages its enhancer, OCR and promoter classes, so GC",
          f"    under-predicts its MCC; without it (n = {int(h7nr['n'])}, P12 homology_clean7_minus_regulatory) disc "
          f"{u(h7nr['loto_MAE_disc_only'])} vs GC {u(h7nr['loto_MAE_gc_only'])}.",
          f"    In rank terms disc does order the clean 7 perfectly (Spearman {s(h7c['spearman'], 2)} vs GC {s(h7c['spearman_gc_mcc'], 2)}, "
          f"GC's exact p = {u(h7c['spearman_gc_mcc_p_perm'])}),",
          f"    and only {len(at1)} of the {len(s7)} seven-task subsets of S11 reach disc Spearman +1.00 (median "
          f"{s(s7['spearman_disc_mcc'].median(), 2)}; GC median {s(s7['spearman_gc_mcc'].median(), 2)}) [multivar_posthoc_subsets7.csv];",
          "    both drop " + ", ".join(sorted(set.intersection(*[set(x.split(";")) for x in at1["dropped"]]))) + ".",
          f"  * without the 3-class regulatory task (n = {int(fam_s10['n'])}): label {u(fam_s10['loto_MAE_indicator_only'])}, "
          f"GC {u(fam_s10['loto_MAE_gc_only'])}, disc {u(fam_s10['loto_MAE_disc_only'])} (label - disc "
          f"{ci(fam_s10['dMAE_lo'], fam_s10['dMAE_hi'])}).",
          f"  * inside both clusters of the clean 7 disc orders MCC perfectly (Spearman {s(cp['spearman_disc_mcc'], 2)}, n = "
          f"{int(cp['n'])}; {s(ce['spearman_disc_mcc'], 2)}, n = {int(ce['n'])}),",
          f"    but with the union nt_promoter_all dropped these are {int(hc6['within_promoter_family_n'])} and "
          f"{int(hc6['within_enhancer_family_n'])} exact-duplicate-free datasets (P12), exact p = "
          f"{u(hc6['within_promoter_family_p_exact'])} and {u(hc6['within_enhancer_family_p_exact'])}",
          f"    (the smallest attainable), and nt_promoter_all vs nt_promoter_no_tata MCC ({u(fa['bench_mcc'])} vs "
          f"{u(fn['bench_mcc'])}) lie inside each other's CIs.",
          f"  * on the n = 8 set of H7 (the paper's 9 minus human_enhancers_ensembl; it still contains nt_enhancers_types,",
          f"    25% verbatim duplicates, so it is NOT homology-clean) the mutual-partial tests REVERSE relative to S11:",
          f"    rho(disc | GC) {s(q8_['partial_given_gc_mean'], 2)} (t p = {u(q8_['partial_given_gc_mean_p'])}, residual-perm p = "
          f"{u(q8_['partial_given_gc_mean_p_perm'])}), rho(GC | disc) {s(q8_['partial_gc_given_disc'], 2)} (t p = "
          f"{u(q8_['partial_gc_given_disc_p'])}, perm p = {u(q8_['partial_gc_given_disc_p_perm'])});",
          f"    percentile bootstrap CIs are not usable at n = 8 (the partial is undefined on "
          f"{B_ - int(q8_['partial_given_gc_mean_n_boot_ok'])} and {B_ - int(q8_['partial_gc_given_disc_n_boot_ok'])} of {B_} resamples).",
          "  Reading: the real-task disc-MCC correlation is carried largely by one between-cluster contrast.  The solved",
          "  promoter/regulatory tasks are both more discriminable (GC-box motifs) and more learnable than the failing",
          "  enhancer/OCR tasks, and GC content indexes the same split at least as closely as disc does.  Disc has a",
          f"  held-out error lower by > {TIE} MCC only on homology-clean sets that contain the 3-class regulatory task (whose pooled GC",
          "  mean misleads a GC-only model) and on the GB-only suite; within clusters (<= 5 exact-duplicate-free",
          "  datasets) neither is resolvable.  The real tasks therefore cannot separate discriminability from GC content",
          "  or other properties of the two clusters; they are consistent with Route 1 but are not independent evidence",
          "  for it.  Only the synthetic dose-response (motifs implanted into composition-matched random backgrounds)",
          "  isolates discriminability, which is why Route 1 must rest on it.  The two-level label is post hoc, equals",
          "  the outcome split, and explains nothing mechanistically.",
          ""]
    check(fam_s7["loto_MAE_disc_only"] < fam_s7["loto_MAE_gc_only"] and not excl0(fam_s7["dMAE_lo"], fam_s7["dMAE_hi"]),
          "clean set: disc beats GC, ties family")
    check(q8_["partial_given_gc_mean_p"] < 0.05 and q8_["partial_gc_given_disc_p"] > 0.05 and
          g7["partial_disc_given_gc_p"] > 0.05 and g7["partial_gc_given_disc_p"] < 0.05, "n=8 partial t-tests reverse S11")
    tab("Homology-clean 7 tasks (every task <5% near-dup), LOTO MAE (post hoc)",
        f"disc {u(fam_s7['loto_MAE_disc_only'])}; family label {u(fam_s7['loto_MAE_indicator_only'])}; GC mean {u(fam_s7['loto_MAE_gc_only'])}; "
        f"without the 3-class regulatory task (n={int(h7nr['n'])}): disc {u(h7nr['loto_MAE_disc_only'])}, GC {u(h7nr['loto_MAE_gc_only'])}",
        "multivar_posthoc_family.csv as_published/between_family_S7_homology_clean; multivar_posthoc_distinct_datasets.csv "
        "homology_clean7_minus_regulatory")
    tab("Spearman with MCC on S11, GC mean vs discriminability (post hoc, descriptive)",
        f"GC mean {s(h11['spearman_gc_mcc'], 2)} (exact p={u(h11['spearman_gc_mcc_p_perm'], 4)}) vs disc {s(h11['spearman'], 2)} "
        f"(exact p={u(h11['spearman_p_perm'], 3)}); difference {ci(h11['rho_gc_minus_rho_disc_lo'], h11['rho_gc_minus_rho_disc_hi'], 2)}",
        "multivar_posthoc_homology_clean.csv S11_all (spearman_gc_mcc, spearman, rho_gc_minus_rho_disc_lo/hi)")
    tab("Disc-only vs GC-only LOTO MAE across task sets (post hoc)",
        f"GC lower on {len(gwin)}, tied (<{TIE}) on {len(ties_)}, disc lower on {len(dwin)} of {len(ords)} sets: "
        + "; ".join(f"{r['subset']}" for _, r in dwin.iterrows()),
        "multivar_posthoc_homology_clean.csv, multivar_posthoc_distinct_datasets.csv (loto_MAE_*)")
    tab("Promoter/enhancer label only, LOTO S11 (post hoc; the label IS Table I's solved/failing split, so it "
        "predicts MCC nearly by construction)",
        f"MAE {u(fam_b['loto_MAE_indicator_only'])} (R^2 {s(fam_b['loto_R2_indicator_only'], 2)}); dMAE vs disc "
        f"{s(fam_b['dMAE_indicator_minus_disc'])} {ci(fam_b['dMAE_lo'], fam_b['dMAE_hi'])}; rho(disc, MCC | family) "
        f"{s(fam_b['partial_disc_given_indicator'], 2)} {ci(fam_b['partial_disc_given_indicator_lo'], fam_b['partial_disc_given_indicator_hi'], 2)}",
        "multivar_posthoc_family.csv as_published/between_family_S11")
    tab("GC mean only, LOTO S11 (post hoc; in-sample AICc winner, so not selection-honest)",
        f"MAE {u(g7['gc_only_loto_MAE'])}; dMAE vs disc {s(g7['dMAE_gc_minus_disc'])} {ci(g7['dMAE_lo'], g7['dMAE_hi'])}",
        "multivar_posthoc_gc_vs_disc.csv S11/as_published")
    tab("rho(disc, MCC | GC mean) / rho(GC mean, MCC | disc), S11",
        f"{s(g7['partial_disc_given_gc'], 2)} {ci(g7['partial_disc_given_gc_lo'], g7['partial_disc_given_gc_hi'], 2)} / "
        f"{s(g7['partial_gc_given_disc'], 2)} {ci(g7['partial_gc_given_disc_lo'], g7['partial_gc_given_disc_hi'], 2)}",
        "multivar_posthoc_gc_vs_disc.csv S11/as_published (= multivar_partial.csv S11/Y1_MCC gc_mean)")

    # ================================================================ H3 partials per covariate
    cnt = summ.loc[f"A1_{S11}_n_covariates_partial_rho_CI_excl_0", "value"]
    cntt = summ.loc[f"A1_{S11}_n_covariates_partial_rho_t_p_lt_0.05", "value"]
    cnt15 = summ.loc[f"A1_{S15}_n_covariates_partial_rho_CI_excl_0", "value"]
    cntau = summ.loc["A1_S11/Y2_AUROC_n_covariates_partial_rho_CI_excl_0", "value"]
    cnth = summ.loc[f"POSTHOC_A1_{S11}_n_covariates_partial_rho_holm_p_lt_0.05", "value"]
    holm_ok = pa.index[pa["posthoc_partial_p_holm"] < 0.05].tolist()
    check(int(cnth.split("/")[0]) == len(holm_ok), "Holm count consistent")
    pos = pa.index[(pa["partial_boot_lo"] > 0)].tolist()
    notpos = pa.index[~(pa["partial_boot_lo"] > 0)].tolist()
    check(int(cnt.split("/")[0]) == len(pos), "count consistent")
    check((pa["partial_rho"] > 0).all(), "all S11 partial point estimates positive")
    hpo = pa.loc["homology"]
    check(excl0(hpo["partial_boot_lo"], hpo["partial_boot_hi"]), "homology partial CI excludes 0")
    p3s11 = p3[(p3["config"] == S11)].iloc[0]
    rsh, psh = stats.spearmanr(h_alt["homology_jaccard07"], h_alt["homology_mm80"])
    T += ["H3  PARTIAL ASSOCIATION OF DISCRIMINABILITY GIVEN EACH COVARIATE (pre-registered A1; multivar_partial.csv)",
          f"  S11/MCC: raw Spearman {s(pa['spearman_disc_y'].iloc[0])} (t-approximation p = {u(pa['spearman_p'].iloc[0])}; exact permutation p = "
          f"{u(h11['spearman_p_perm'])}, H7).  Partial rho(disc, MCC | c) point",
          f"  estimates are all positive ({s(pa['partial_rho'].min())} to {s(pa['partial_rho'].max())}), but the 95% bootstrap CI excludes 0 for",
          f"  only {cnt} covariates ({', '.join(pos)}) and includes 0 for {', '.join(notpos)};",
          f"  t-test p < 0.05 for {cntt} (uncorrected; percentile CIs at n = 11 reach 1.000, so both are shown), and for",
          f"  {cnth} after Holm correction over the 10 covariates ({', '.join(holm_ok) or 'none'}; post-hoc column "
          "posthoc_partial_p_holm).  S15: CI excludes 0",
          f"  for {cnt15}.  AUROC outcome (S11): {cntau}.",
          "  Per covariate (S11/MCC): rho(disc, MCC | c) [95% CI] t p (Holm p)   ||   post-hoc reverse rho(c, MCC | disc)",
          "  [95% CI] t p (valid bootstrap resamples forward / reverse)"]
    for c_, r in pa.iterrows():
        T.append(f"    {c_:<11} {s(r['partial_rho'])} {ci(r['partial_boot_lo'], r['partial_boot_hi'])} p={u(r['partial_p'])} "
                 f"(Holm {u(r['posthoc_partial_p_holm'])})   ||   "
                 f"{s(r['posthoc_rev_partial_rho_c_given_disc'])} {ci(r['posthoc_rev_partial_boot_lo'], r['posthoc_rev_partial_boot_hi'])} "
                 f"p={u(r['posthoc_rev_partial_p'])}   (Spearman disc~c {s(r['spearman_disc_c'], 2)}; "
                 f"n_boot {int(r['n_boot_ok'])}/{int(r['posthoc_rev_n_boot_ok'])})")
    T += [f"  Homology does not confound discriminability: rho(disc | homology) {s(hpo['partial_rho'])} "
          f"{ci(hpo['partial_boot_lo'], hpo['partial_boot_hi'])}; reverse rho(homology | disc)",
          f"  {s(hpo['posthoc_rev_partial_rho_c_given_disc'])} {ci(hpo['posthoc_rev_partial_boot_lo'], hpo['posthoc_rev_partial_boot_hi'])}.  "
          f"Same with the MMseqs2 both-strand id80 measure (P3; Spearman {s(rsh, 2)} with ours, n = {len(h_alt)}):",
          f"  {s(p3s11['partial_rho_disc_given_mm80'])} {ci(p3s11['partial_boot_lo'], p3s11['partial_boot_hi'])} "
          "[multivar_posthoc_homology_models.csv; that measure is another lane's",
          f"  homology_recluster_redundancy.csv, {h_alt['source'].iloc[0].split('mtime ')[-1]}].  This is a cross-task statement: it does not bear on",
          "  within-task inflation of absolute MCCs (H7).",
          ""]
    tab("rho(disc, MCC | homology), S11", f"{s(hpo['partial_rho'], 2)} {ci(hpo['partial_boot_lo'], hpo['partial_boot_hi'], 2)}",
        "multivar_partial.csv S11/Y1_MCC homology")
    tab("rho(disc, MCC | c): CI excludes 0", f"{cnt} covariates (S11); {cnt15} (S15); t p<0.05 {cntt} (S11, uncorrected), "
        f"{cnth} after Holm",
        "multivar_summary.csv A1_*_n_covariates_partial_rho_CI_excl_0, *_t_p_lt_0.05, POSTHOC_A1_*_holm_p_lt_0.05")

    # ================================================================ H4 reviewer model
    r11 = rev[rev["config"] == S11].set_index("predictor")
    r15 = rev[rev["config"] == S15].set_index("predictor")
    check((~r11["boot_excludes_0"]).all() and ((r11["t_lo"] <= 0) & (r11["t_hi"] >= 0)).all(), "S11: no coefficient CI excl 0")
    t15 = r15.index[(r15["t_lo"] > 0) | (r15["t_hi"] < 0)].tolist()
    b15 = r15.index[r15["boot_excludes_0"]].tolist()
    check(set(t15) == {"disc", "len_log"} and b15 == ["disc"], "S15 significance pattern")
    rr11, rr15 = r11.iloc[0], r15.iloc[0]
    check(rr11["loto_MAE"] > a0["loto_MAE"], "reviewer model worse S11")
    dso = one(models, config=S15, family="poolA", model="disc")
    check(rr15["loto_MAE"] > dso["loto_MAE"], "reviewer model worse S15")
    T += ["H4  THE REVIEWER'S MODEL, MCC ~ disc + log length + within-class GC SD + homology (pre-registered A8;",
          "    multivar_reviewer_model.csv; standardized betas, t-based and task-bootstrap 95% CIs)",
          f"  S11 (n = 11, residual df {int(rr11['resid_df'])}, n/p < 3: OVER-PARAMETERIZED): in-sample R^2 {u(rr11['R2'])}, adjusted "
          f"{u(rr11['adjR2'])} (disc alone: R^2 {u(md['R2'])}, adjusted {u(md['adjR2'])});",
          f"    out of sample WORSE than disc alone: LOTO R^2 {s(rr11['loto_R2'])}, MAE {u(rr11['loto_MAE'])} vs {u(a0['loto_MAE'])}.  No coefficient's",
          "    CI (t or bootstrap) excludes 0:"]
    for p_, r in r11.iterrows():
        T.append(f"      {p_:<9} {s(r['std_beta'])}  t-CI {ci(r['t_lo'], r['t_hi'])} p={u(r['t_p'])}  bootstrap {ci(r['boot_lo'], r['boot_hi'])}  VIF {u(r['vif'], 2)}")
    T += [f"  S15 (n = 15): R^2 {u(rr15['R2'])}, adjusted {u(rr15['adjR2'])}; LOTO R^2 {s(rr15['loto_R2'])}, MAE {u(rr15['loto_MAE'])} vs "
          f"disc-only {u(dso['loto_MAE'])} (worse).  t-CIs excluding 0: {', '.join(t15)};",
          f"    task-bootstrap CI excluding 0: {', '.join(b15)} only."]
    for p_, r in r15.iterrows():
        T.append(f"      {p_:<9} {s(r['std_beta'])}  t-CI {ci(r['t_lo'], r['t_hi'])} p={u(r['t_p'])}  bootstrap {ci(r['boot_lo'], r['boot_hi'])}  VIF {u(r['vif'], 2)}")
    ll = r15.loc["len_log"]
    for r_ in (r11, r15):
        h_ = r_.loc["homology"]
        check(h_["std_beta"] < 0 and h_["t_lo"] < 0 < h_["t_hi"] and h_["boot_lo"] < 0 < h_["boot_hi"], "homology beta <0, CIs incl 0")
    check(not excl0(r11.loc["len_log", "t_lo"], r11.loc["len_log", "t_hi"]), "len_log S11 CI incl 0")
    check(not excl0(ll["boot_lo"], ll["boot_hi"]) and excl0(ll["t_lo"], ll["t_hi"]), "len_log S15 t excl, boot incl")
    T += [f"  Length: on S15 longer-sequence tasks have lower MCC given the other terms ({s(ll['std_beta'], 2)}, t-CI "
          f"{ci(ll['t_lo'], ll['t_hi'], 2)}, p = {u(ll['t_p'])}),",
          f"    but the task bootstrap {ci(ll['boot_lo'], ll['boot_hi'], 2)} includes 0 and it is absent on S11 "
          f"({s(r11.loc['len_log', 'std_beta'], 2)} {ci(r11.loc['len_log', 't_lo'], r11.loc['len_log', 't_hi'], 2)}), i.e. it appears only when the 4",
          "    non-TF tasks are added.  Homology's coefficient is negative on both sets with t- and bootstrap CIs",
          f"    including 0 ({s(r11.loc['homology', 'std_beta'], 2)} / {s(r15.loc['homology', 'std_beta'], 2)}): no evidence of homology inflation of the",
          "    cross-task relation.",
          ""]
    tab("Reviewer model disc+length+GC-SD+homology, LOTO",
        f"S11 MAE {u(rr11['loto_MAE'])} (R^2 {s(rr11['loto_R2'], 2)}); S15 MAE {u(rr15['loto_MAE'])} (R^2 {s(rr15['loto_R2'], 2)})",
        "multivar_reviewer_model.csv loto_MAE/loto_R2")

    # ================================================================ H5 noise
    n11r, n15r = one(noise, config=S11), one(noise, config=S15)
    T += ["H5  THE MISFIT IS BETWEEN-TASK HETEROGENEITY, NOT TEST-SET SAMPLING NOISE (pre-registered A7; multivar_noise.csv)",
          f"  mean(test-set SE of MCC)^2 / disc-only LOTO MSE = {100 * n11r['noise_share']:.1f}% on S11 ({100 * n15r['noise_share']:.1f}% on S15).",
          "  Scope: only the test-set sampling error of each task's MCC enters; the sampling error of the disc",
          "  covariate itself (motif scan on finite sequences) is not included.",
          ""]
    tab("Test-set sampling noise share of disc-only LOTO MSE", f"{100 * n11r['noise_share']:.1f}% (S11); {100 * n15r['noise_share']:.1f}% (S15)",
        "multivar_noise.csv noise_share")

    # ================================================================ H6 per-task deviations
    rs11 = resid[resid["config"] == S11].set_index("task")
    rs15 = resid[resid["config"] == S15].set_index("task")
    nte = rs11.loc["nt_enhancers_types"]
    nte15 = rs15.loc["nt_enhancers_types"]
    npt = rs11.loc["nt_promoter_tata"]
    pn = p1.set_index(["config", "task"])
    check(bool(nte15["explained"]) and pn.loc[(S15, "nt_enhancers_types"), "p_max"] > 0.05, "S15 nt_enhancers_types explained, not beyond chance")
    u1 = one(p8, config=S11, variant="U1_functional_representative")
    u2_15 = one(p8, config=S15, variant="U2_max_over_functional_classes")
    alts = p8[(p8["config"] == S11) & (p8["variant"].isin(["P4_0_vs_rest_reversed", "1_vs_rest", "2_vs_rest"]))]
    fix = one(p4, config=S11, variant="orientation_consistent")
    check(not bool(nte["explained"]), "nt_enhancers_types not explained")
    check(bool(npt["explained"]) and pn.loc[(S11, "nt_promoter_tata"), "p_max"] > 0.05, "tata explained but not beyond chance")
    check(u1["changed"].startswith("nt_enhancers_types=") and ";" not in u1["changed"], "U1 changes only nt_enhancers_types on S11")
    check(cid[("nt_enhancers_types", "label0_set_equals_nt_enhancers_label0_test")] == 1, "class 0 = non-enhancer")
    nn9 = p9[(p9["config"] == S11) & (p9["task"] == "human_nontata_promoters")].set_index("variant")
    fl_ = famt[famt["family_indicator"] == 0]
    res_con = ", ".join(f"{t} {s(r['disc_only_loto_resid'], 2)}" for t, r in fl_.iterrows()
                        if r["table1_route"].startswith("contamination"))
    res_sm = ", ".join(f"{t} {s(r['disc_only_loto_resid'], 2)}" for t, r in fl_.iterrows() if r["table1_route"] == "shared-motif")
    check(all(abs(famt.loc[t, "disc_only_loto_resid"] - rs11.loc[t, "resid"]) < 1e-12 for t in famt.index), "family_tasks resid == A6")
    _sm = fl_[fl_["table1_route"] == "shared-motif"]["disc_only_loto_resid"]
    _co = fl_[fl_["table1_route"].str.startswith("contamination")]["disc_only_loto_resid"]
    check((_sm < 0).all() and (_co > 0).any() and (_co < 0).any(), "residual pointer wording")
    fz = p9[(p9["config"] == S11) & (p9["variant"] == "fisher_z")].iloc[0]
    T += ["H6  WHICH FACTOR DRIVES EACH LARGE DEVIATION (A6 pre-registered; P1, P4, P8, P9 post hoc;",
          f"    multivar_residuals.csv, multivar_posthoc_*.csv).  Flagged = |disc-only LOTO residual| > LOTO RMSE ({u(nte['loto_rmse'])} on S11).",
          f"  * nt_enhancers_types (S11 residual {s(nte['resid'])}, the largest): not explained by any measured covariate (best:",
          f"    {nte['best_covariate']} {100 * nte['frac_reduction']:.0f}% of |residual|; P1 permutation p = {u(pn.loc[(S11, 'nt_enhancers_types'), 'p_max'])}).",
          f"    (On S15 its residual is {s(nte15['resid'])} and the pre-registered 50% rule labels it 'explained' by "
          f"{nte15['best_covariate']} ({100 * nte15['frac_reduction']:.0f}%),",
          f"    but the best of 10 row-permuted covariates does as well with P1 p = {u(pn.loc[(S15, 'nt_enhancers_types'), 'p_max'])}: "
          "not distinguishable from chance.)",
          "    POST HOC, found BECAUSE of this residual: its disc is the best motif of the representative '0_vs_rest'",
          "    contrast, and class 0 is the NON-enhancer class (its label-0 set is identical to nt_enhancers' label 0,",
          "    multivar_posthoc_class_identity.csv), so disc = TBP 0.675 measures the TATA-in-negatives contamination",
          "    axis, oriented opposite to every binary task.  Uniform rule (disc = best motif enriched in a FUNCTIONAL",
          "    class; for a multiclass task keep the representative contrast if its class is functional, else take the",
          f"    best functional-class contrast): on S11 it changes only nt_enhancers_types ({u1['changed']}),",
          f"    giving LOTO R^2 {s(u1['loto_R2'])} {ci(u1['loto_R2_lo'], u1['loto_R2_hi'])}, MAE {u(u1['loto_MAE'])}; across all {len(alts)}",
          f"    enhancer-oriented readings LOTO R^2 is {s(alts['loto_R2'].min())} to {s(alts['loto_R2'].max())} and MAE {u(alts['loto_MAE'].min())}-{u(alts['loto_MAE'].max())}",
          f"    (multivar_posthoc_orientation_robustness.csv; nt_enhancers_types residual then {s(fix['resid_nt_enhancers_types'])}).",
          f"    On S15 the max-over-functional-classes rule also changes splice ({u2_15['changed'].split('; ')[-1]}); S15 LOTO R^2 {s(u2_15['loto_R2'])}.",
          f"    The pre-registered R^2 {s(a0['loto_R2'], 2)} stays the primary number; this is a diagnosis of one deviation.",
          f"  * nt_promoter_tata ({s(npt['resid'])}): composition-only MCC removes {100 * npt['frac_reduction']:.0f}% ('explained' under the pre-registered",
          f"    50% rule), but the best of 10 row-permuted covariates reaches >= 50% in {100 * pn.loc[(S11, 'nt_promoter_tata'), 'null_frac_ge_50pct']:.0f}% of permutations",
          f"    (P1 p = {u(pn.loc[(S11, 'nt_promoter_tata'), 'p_max'])}): not distinguishable from chance at n = 11.",
          f"  * human_nontata_promoters ({s(nn9.loc['linear_published', 'resid'])}; not flagged): the linear LOTO prediction is "
          f"{u(nn9.loc['linear_published', 'loto_pred'])}, above the MCC ceiling.",
          f"    Capping at 1 leaves {s(nn9.loc['linear_capped_at_1', 'resid'])} "
          f"({100 * (1 - nn9.loc['linear_capped_at_1', 'resid'] / nn9.loc['linear_published', 'resid']):.0f}% removed); a bounded Fisher-z model (post hoc P9, OLS on atanh MCC) gives "
          f"{s(nn9.loc['fisher_z', 'resid'])}",
          f"    (S11 LOTO MAE {u(fz['loto_MAE'])} vs {u(a0['loto_MAE'])}, dMAE CI {ci(fz['dMAE_lo'], fz['dMAE_hi'])}).  Homology cannot explain a NEGATIVE",
          "    residual (inflation would push actual MCC above the prediction); this says nothing about the within-task",
          "    homology inflation of its absolute MCC, which H7 measures directly.",
          "  * S15-only tasks: the composition-labelled tasks (Table 1 label 'composition', a non-failing label) are",
          f"    explained by composition-only MCC -- demo_human_or_worm {s(rs15.loc['demo_human_or_worm', 'resid'])}: "
          f"{100 * rs15.loc['demo_human_or_worm', 'frac_reduction']:.0f}% (P1 p = {u(pn.loc[(S15, 'demo_human_or_worm'), 'p_max'])});",
          f"    dummy_mouse_enhancers_ensembl {s(rs15.loc['dummy_mouse_enhancers_ensembl', 'resid'])}: "
          f"{100 * rs15.loc['dummy_mouse_enhancers_ensembl', 'frac_reduction']:.0f}% (p = {u(pn.loc[(S15, 'dummy_mouse_enhancers_ensembl'), 'p_max'])}).",
          f"    The Route-2 positional task nt_splice_sites_all ({s(rs15.loc['nt_splice_sites_all', 'resid'])}; composition "
          f"{100 * rs15.loc['nt_splice_sites_all', 'frac_reduction']:.0f}%, p = {u(pn.loc[(S15, 'nt_splice_sites_all'), 'p_max'])}) is the only",
          "    positional task, so a cross-task regression cannot learn its mechanism.  These tasks are outside Route 1's",
          "    scope, which is why S11 is primary.",
          "  * Pointer only (post hoc, n = 3 per label, NOT a test; multivar_posthoc_family_tasks.csv): among the failing",
          f"    S11 tasks, the disc-only LOTO residuals of Table I's contamination-labelled tasks are {res_con}, those of",
          f"    its shared-motif tasks {res_sm}: all shared-motif tasks sit below the disc line, the contamination-",
          "    labelled ones on both sides of it.",
          ""]

    # ================================================================ H7 homology-clean subset
    ex = exd.set_index("task")
    hh = hom.set_index("task").loc["human_enhancers_ensembl"]
    q9 = one(p5, subset="paper_homology_clean_9")
    q8 = one(p5, subset="paper9_minus_human_enhancers_ensembl")
    q7 = one(p5, subset="rule_clean_homology_lt_0.05")
    q11 = one(p5, subset="S11_all")
    de = p6.set_index(["task", "subset"])
    hee = ex.loc["human_enhancers_ensembl"]
    check(hh["frac_ge_0_7"] >= 0.05 and ex.loc["nt_enhancers_types", "frac_exact_any"] >= 0.05, "both LEAKY")
    check(set(q9["kept_homology_ge_0_05"].split(";")) == {"human_enhancers_ensembl", "nt_enhancers_types"}, "paper 9 keeps 2 leaky")
    # fix round 2 guard: only the n = 7 set may be called homology-clean
    check(list(p5.loc[p5["homology_clean"], "subset"]) == ["rule_clean_homology_lt_0.05"], "only n=7 is homology-clean")
    check(not bool(q8["homology_clean"]) and q8["kept_homology_ge_0_05"] == "nt_enhancers_types", "n=8 keeps nt_enhancers_types")
    check(bool(q7["jackknife_degenerate"]) and q7["spearman"] > 1 - 1e-12 and np.isnan(q7["partial_given_gc_mean_p"]),
          "n=7 degenerate: rho = 1, partial p undefined")
    check(abs(q7["spearman_p_perm"] - 2 / 5040) < 1e-12, "n=7 exact p = 2/7!")
    ntt_c2 = de.loc[("nt_enhancers_types", "clean_j07"), "n_class2"]
    check(ntt_c2 == 0, "all class-2 test sequences of nt_enhancers_types are near-duplicates")

    def pfmt(x):
        if x != x:
            return "undefined (|rho| = 1)"
        check(x > 0, "a p-value is never printed as 0")
        return f"{x:.2g}" if x >= 0.001 else f"{x:.1e}"

    T += ["H7  THE PAPER'S 'HOMOLOGY-CLEAN' 9-TASK SUBSET IS NOT HOMOLOGY-CLEAN -- FLAG FOR THE WRITER (post hoc P5, P6;",
          "    multivar_homology.csv, multivar_posthoc_exact_dup.csv, _homology_clean.csv, _dedup_eval.csv)",
          "  The paper drops only human_nontata_promoters and nt_enhancers.  This module's full homology table flags two",
          "  more S11 tasks (>= 0.05, the pre-existing LEAKY threshold of homology/measure_homology.py):",
          f"  * human_enhancers_ensembl: {100 * hh['frac_ge_0_7']:.1f}% of test sequences have a >= 0.7 8-mer near-duplicate in train "
          f"(Wilson {ci(100 * hh['frac_ge_0_7_lo'], 100 * hh['frac_ge_0_7_hi'], 1, False)}%, 5,000-query sample);",
          f"    on the FULL test split {100 * hee['frac_exact_any']:.1f}% are VERBATIM train copies, {100 * hee['dup_share_class1']:.1f}% of them positives "
          f"({100 * hee['dup_frac_class1']:.1f}% of all test positives);",
          f"    {100 * hee['frac_dup_label_concordant']:.2f}% of the duplicates have ALL their train copies carrying the test label "
          f"({100 * hee['frac_dup_label_any_copy_concordant']:.2f}% have at least one such copy).",
          f"  * nt_enhancers_types: {100 * ex.loc['nt_enhancers_types', 'frac_exact_any']:.1f}% verbatim -- the same sequences as nt_enhancers "
          "(identical train/test lists); ALL of its class-2",
          f"    test sequences are verbatim train copies (near-duplicate-free test subset has {int(ntt_c2)} class-2 sequences).",
          "  So removing BOTH gives the only homology-clean subset (every task < 5% near-duplicates, n = 7); the n = 8",
          "  set drops only human_enhancers_ensembl and still contains nt_enhancers_types (25% verbatim), so it is a",
          "  sensitivity row, NOT homology-clean.",
          "  Correlation by subset (Spearman of MCC vs disc, with the t-approximation p and the exact permutation p over",
          "  all n! orderings; jackknife range; disc-only LOTO; partial rho given GC mean with t p and residual-",
          "  permutation p; GC mean's own Spearman with MCC (fix round 3); exact-duplicate-free sequence sets, H9):"]
    for q in (q11, q9, q8, q7):
        T.append(f"    {q['subset']:<38} n={int(q['n']):>2} ({int(q['n_exact_distinct_sets'])} exact-dup-free)  rho {s(q['spearman'])} "
                 f"(t p={pfmt(q['spearman_p'])}; perm p={pfmt(q['spearman_p_perm'])})  jackknife "
                 + ("+1.000 on every leave-one-out (degenerate)" if q["jackknife_degenerate"] else
                    f"{ci(q['jackknife_min'], q['jackknife_max'])}")
                 + f"  LOTO MAE {u(q['loto_MAE'])} {ci(q['loto_MAE_lo'], q['loto_MAE_hi'], signed=False)} "
                 f"R^2 {s(q['loto_R2'])}  rho|GC " + ("degenerate (rho = 1)" if q['spearman'] > 1 - 1e-9 else
                                                    f"{s(q['partial_given_gc_mean'])} (t p={u(q['partial_given_gc_mean_p'])}, "
                                                    f"perm p={u(q['partial_given_gc_mean_p_perm'])})")
                 + f"  rho(GC, MCC) {s(q['spearman_gc_mcc'])} (perm p={pfmt(q['spearman_gc_mcc_p_perm'])})"
                 + f"  LEAKY kept: {q['kept_homology_ge_0_05'] if isinstance(q['kept_homology_ge_0_05'], str) else 'none'}")
    check(q8["partial_given_gc_mean_p"] < 0.05 and q9["partial_given_gc_mean_p"] > 0.05 and
          q11["partial_given_gc_mean_p"] > 0.05, "partial|GC t-test significant only in the 8")
    check(q9["spearman_p_perm"] > q9["spearman_p"] and q8["spearman_p_perm"] > q8["spearman_p"], "t-approx p optimistic at n<=9")
    check(np.isnan(q7["spearman_p"]), "n=7 t-approx p undefined (rho = 1), not 0")
    T += ["  So the corrected homology-clean correlation is still positive and larger (the paper's direction holds);",
          f"  the t-approximation understates p at n <= 9 (paper's 9: {pfmt(q9['spearman_p'])} vs exact {pfmt(q9['spearman_p_perm'])}; "
          f"n = 8: {pfmt(q8['spearman_p'])} vs {pfmt(q8['spearman_p_perm'])}).",
          f"  disc and GC mean stay collinear in every subset (Spearman disc~GC {s(q9['spearman_disc_gc_mean'], 2)} in the 9, "
          f"{s(q8['spearman_disc_gc_mean'], 2)} in the 8, {s(q7['spearman_disc_gc_mean'], 2)} in the 7).",
          f"  rho(disc, MCC | GC) has t-test p < 0.05 only in the n = 8 set ({s(q8['partial_given_gc_mean'], 2)}, p = "
          f"{u(q8['partial_given_gc_mean_p'])}; reverse rho(GC | disc) {s(q8['partial_gc_given_disc'], 2)},",
          f"  p = {u(q8['partial_gc_given_disc_p'])}; bootstrap CIs unusable at n <= 9, see H2); in the n = 7 set rho = "
          f"{s(q7['spearman'], 2)}, so the partials and the jackknife are degenerate.  Both sets are",
          f"  still ~half promoter/regulatory tasks ({int(q8['n_promoter_tasks'])} of {int(q8['n'])}; {int(q7['n_promoter_tasks'])} of "
          f"{int(q7['n'])}), i.e. still dominated by the two-cluster contrast (H2).",
          "  Within-task effect on the Table-1 model (P6; same model refit, full-test MCC asserted == Table 1 to 4 decimals;",
          "  MCC on a subset shifts with class prevalence, so the prevalence-standardized MCC and per-class recall are",
          "  the fair comparison):"]
    for t in ("human_enhancers_ensembl", "nt_enhancers_types", "human_nontata_promoters", "nt_enhancers"):
        a_, c_ = de.loc[(t, "all")], de.loc[(t, "clean_j07")]
        extra = (f", prevalence-std {u(c_['mcc_prevalence_std'])}" if c_["mcc_prevalence_std"] == c_["mcc_prevalence_std"]
                 else " (a class is absent from the clean subset; prevalence-std undefined)")
        T.append(f"    {t:<24} near-dup {100 * a_['frac_ge_0_7_full']:.1f}% (verbatim {100 * a_['frac_verbatim']:.1f}%)  MCC all "
                 f"{u(a_['mcc'])} -> near-dup-free {u(c_['mcc'])}{extra}")
    hc = de.loc[("human_enhancers_ensembl", "clean_j07")]
    T += [f"  human_enhancers_ensembl's Table-1 MCC {u(de.loc[('human_enhancers_ensembl', 'all'), 'mcc'])} is therefore partly homology-inflated "
          f"(near-dup-free prevalence-standardized {u(hc['mcc_prevalence_std'])}; AUROC",
          f"  {u(de.loc[('human_enhancers_ensembl', 'all'), 'auroc'])} -> {u(hc['auroc'])}), and nt_enhancers_types' MCC is mostly carried by its duplicated class 2",
          f"  (near-dup-free MCC {u(de.loc[('nt_enhancers_types', 'clean_j07'), 'mcc'])}).  This module does not re-derive route labels; the writer should",
          "  check whether Table 1's 'shared-motif' label for nt_enhancers_types (and the 'not the route assignments'",
          "  sentence of Limitations iv) still holds given that its MCC is carried by duplicated class-2 sequences.  The",
          "  paper's 'homology-clean' label, its n = 9 numbers and the list of homology-inflated tasks must be updated.",
          ""]
    tab("Homology-clean correlation (paper's 9 -> corrected)",
        f"paper's n=9 {s(q9['spearman'], 2)} (keeps 2 leaky tasks; exact p={pfmt(q9['spearman_p_perm'])}) -> homology-clean n=7 "
        f"(every task <5% near-dup) {s(q7['spearman'], 2)} (exact p={pfmt(q7['spearman_p_perm'])}); sensitivity n=8 "
        f"(nt_enhancers_types kept) {s(q8['spearman'], 2)} (exact p={pfmt(q8['spearman_p_perm'])})",
        "multivar_posthoc_homology_clean.csv (spearman, spearman_p_perm, homology_clean)")

    # ================================================================ H9 task non-independence (P12, P13)
    ov_ = tov[(tov["S11_a"]) & (tov["S11_b"]) & (tov["n_shared"] > 0)].sort_values("frac_a_in_b", ascending=False)
    rego = tcl[(tcl["task_a"] == "human_ensembl_regulatory") & (tcl["task_b"] == "human_ocr_ensembl")]
    check(abs(rego["n_shared"].sum() - rego[(rego["split_label_in_a"] == rego["split_label_in_b"]) &
                                            (rego["split_label_in_a"].str.endswith(":1"))]["n_shared"].sum()) == 0,
          "regulatory/OCR shared sequences are all label 1 in both, same split")
    csz = tcl[tcl["task_b"] == "(class sizes)"].set_index(["task_a", "split_label_in_a"])
    check("ocr" in csz.loc[("human_ensembl_regulatory", "train:1"), "split_label_in_b"] and
          "positive" in csz.loc[("human_ocr_ensembl", "train:1"), "split_label_in_b"], "class names")
    check(tid.min() == 1 and len(tid) == 8, "identity checks all hold (8)")
    tidn = rd("multivar_posthoc_task_identity.csv").set_index(["task", "check"])["note"]
    ocr_tr = tidn[("human_ensembl_regulatory", "train_ocr_class_multiset_equals_human_ocr_ensembl_positives")]
    ocr_te = tidn[("human_ensembl_regulatory", "test_ocr_class_multiset_equals_human_ocr_ensembl_positives")]
    m_tr = re.match(r"n (\d+) = (\d+) \((\d+) distinct\)", ocr_tr)
    m_te = re.match(r"n (\d+) = (\d+) \((\d+) distinct\)", ocr_te)
    check(m_tr and m_te, "identity note format")
    oth = ov_[~ov_.apply(lambda r: {r["task_a"], r["task_b"]} in (
        {"nt_promoter_all", "nt_promoter_tata"}, {"nt_promoter_all", "nt_promoter_no_tata"},
        {"nt_enhancers", "nt_enhancers_types"}, {"human_ensembl_regulatory", "human_ocr_ensembl"}), axis=1)]
    check(oth["frac_a_in_b"].max() < 0.001, "no other S11 pair shares >= 0.1% of sequences")
    oth1 = oth.iloc[0]
    rw = {k: d12.loc[k] for k in d12.index}
    T += ["H9  THE 11 TASKS ARE NOT 11 INDEPENDENT UNITS (post hoc P12, P13)",
          "  EXACT SEQUENCES (audit_task_overlap.py over all 210 ordered task pairs, canonical strand;",
          "  multivar_posthoc_task_overlap*.csv, _task_identity.csv):",
          "  * nt_promoter_all is EXACTLY the union of nt_promoter_tata and nt_promoter_no_tata (train and test, as",
          "    multisets of (sequence, label); the two parts are disjoint): it adds no sequence.",
          "  * nt_enhancers_types re-labels the nt_enhancers sequences (identical train/test lists).",
          "  * the OCR class of human_ensembl_regulatory IS human_ocr_ensembl's positive class: identical multisets of",
          f"    exact strings, split for split (train {int(m_tr.group(1)):,} = {int(m_tr.group(2)):,}, {int(m_tr.group(3)):,} distinct; "
          f"test {int(m_te.group(1)):,} = {int(m_te.group(2)):,}).",
          f"  * no other S11 pair shares >= 0.1% of its sequences (largest: {oth1['task_a']} / {oth1['task_b']}, "
          f"{int(oth1['n_shared'])} sequences, {100 * oth1['frac_a_in_b']:.3f}%).",
          f"  So S11 is {int(rw['S11_all']['n_exact_distinct_sets'])} exact-duplicate-free sequence sets, of which at most "
          f"{int(rw['S11_all']['n_no_exact_sharing'])} share no exact sequence;",
          f"  the n = 8 set of H7 is {int(q8['n_exact_distinct_sets'])}, the homology-clean 7 is {int(q7['n_exact_distinct_sets'])}, "
          f"and the paper's NT-only row (n = {int(rw['suite_NT_only']['n'])}) is "
          f"{int(rw['suite_NT_only']['n_exact_distinct_sets'])} exact-duplicate-free sets."]
    # ---- loci (P13)
    lk_ex = lkl[lkl["exact_shared_frac_max"] >= 0.001]
    lk_nx = lkl[lkl["exact_shared_frac_max"] < 0.001].sort_values("locus_link", ascending=False)
    check(len(lk_ex) == 4 and (lk_ex["locus_link"] > 0.99).all(), "k-mer audit recovers the 4 exact identities")
    whole = lko[lko["ref_label"] == "all"]
    ENH3 = ["human_enhancers_ensembl", "human_enhancers_cohn"]     # human enhancer tasks without exact sharing
    in_reg = whole[(whole["ref_task"] == "human_ensembl_regulatory") & whole["query_task"].isin(ENH3)]
    e_pos = in_reg[in_reg["query_label"] == 1]["frac_ge_0_5"]
    e_neg = in_reg[in_reg["query_label"] == 0]["frac_ge_0_5"]
    check(e_pos.max() < 0.2 and e_pos.min() > e_neg.max(), "enhancer positives in regulatory: modest, above own negatives")

    def cont(qt, ql, rt):
        return one(whole, query_task=qt, query_label=ql, ref_task=rt)
    c_np = cont("human_nontata_promoters", 1, "nt_promoter_all")
    c_nr = cont("human_nontata_promoters", 1, "human_ensembl_regulatory")
    c_ar = cont("nt_promoter_all", 1, "human_ensembl_regulatory")
    check(min(c_np["frac_ge_0_5"], c_nr["frac_ge_0_5"], c_ar["frac_ge_0_5"]) > 0.5, "promoter loci shared (> 50%)")
    pl_ = lk_nx[lk_nx["locus_link"] >= 0.25]["locus_link"]
    fam_map = famt["family"].to_dict()
    check(all(fam_map[a] != "enhancer_ocr" and fam_map[b] != "enhancer_ocr"
              for a, b in zip(lk_nx[lk_nx["locus_link"] >= 0.25]["task_a"], lk_nx[lk_nx["locus_link"] >= 0.25]["task_b"])),
          "every locus link >= 0.25 without exact sharing is between promoter/regulatory tasks")
    p13g = {tau: g for tau, g in p13.groupby("tau")}
    tmain = 0.25
    g25 = p13g[tmain]
    check(all((g["spearman_gc_mcc"] > g["spearman_disc_mcc"]).all() for g in p13g.values()),
          "GC rho > disc rho on every locus-independent set")
    check(not bool(g25["homology_clean"].any()), "no locus-independent set is homology-clean")
    T += ["  LOCI (audit_locus_overlap.py, fix round 3: canonical 31-mers, 1/4 hash-sampled, up to 2,000 query sequences",
          "  per class, train + test pooled; a query sequence is 'mostly contained' in a task when >= 50% of its sampled",
          "  31-mers occur there; multivar_posthoc_task_locus_overlap.csv, _task_locus_links.csv).  Exact matching misses",
          "  the same locus in a shifted or resized window:",
          f"  * {100 * c_np['frac_ge_0_5']:.1f}% of human_nontata_promoters' promoters are mostly contained in nt_promoter_all "
          f"(Wilson {ci(100 * c_np['frac_ge_0_5_lo'], 100 * c_np['frac_ge_0_5_hi'], 1, False)}%),",
          f"    {100 * c_nr['frac_ge_0_5']:.1f}% in human_ensembl_regulatory (its promoter class), and {100 * c_ar['frac_ge_0_5']:.1f}% of "
          "nt_promoter_all's promoters in human_ensembl_regulatory.",
          "  * task pairs WITHOUT exact sharing, by locus link (largest share of any class of one task mostly contained in",
          "    the other task, both directions):"]
    for _, r in lk_nx[lk_nx["locus_link"] >= 0.05].iterrows():
        T.append(f"      {r['task_a']:<26} ~ {r['task_b']:<26} {r['locus_link']:.3f} ({r['locus_link_from']})")
    T += [f"    The human enhancer tasks' positives are mostly contained in human_ensembl_regulatory at "
          f"{100 * e_pos.min():.1f}-{100 * e_pos.max():.1f}%, of the same order",
          f"    as their own NEGATIVE (random genomic) windows ({100 * e_neg.min():.1f}-{100 * e_neg.max():.1f}%): modest sharing, "
          f"unlike the promoter links ({100 * pl_.min():.0f}-{100 * pl_.max():.0f}%).",
          f"  * linked groups (union of links >= threshold; thresholds fixed before P13 ran): "
          + "; ".join(f"{tau:.2f}: {int(g['n_linked_groups'].iloc[0])} groups" for tau, g in sorted(p13g.items(), reverse=True))
          + f"; at {tmain:.2f}: {g25['linked_groups'].iloc[0]}."]
    T += ["  Recomputed without the redundancy (P12, exact sharing; the union is always dropped, both choices reported",
          "  for each exactly-overlapping pair; rho(GC, MCC) and the LOTO MAEs are post hoc, fix round 3):"]
    for k in ("S11_all", "S11_minus_union", "dupfree9_keep_nt_enhancers", "dupfree9_keep_nt_enhancers_types",
              "noshare8_drop_ocr", "noshare8_drop_regulatory", "n8_minus_union", "homology_clean7_minus_union",
              "homology_clean_noshare5_drop_ocr", "homology_clean_noshare5_drop_regulatory", "S11_minus_regulatory",
              "homology_clean7_minus_regulatory", "suite_GB_only", "suite_NT_only"):
        r = rw[k]
        pdg = ("degenerate" if not r["partial_disc_given_gc"] == r["partial_disc_given_gc"] or abs(r["partial_disc_given_gc"]) > 1 - 1e-12
               else f"{s(r['partial_disc_given_gc'], 2)} (t p {u(r['partial_disc_given_gc_p'])})")
        pgd = ("degenerate" if not r["partial_gc_given_disc"] == r["partial_gc_given_disc"] or abs(r["partial_gc_given_disc"]) > 1 - 1e-12
               else f"{s(r['partial_gc_given_disc'], 2)} (t p {u(r['partial_gc_given_disc_p'])})")
        T.append(f"    {k:<41} n={int(r['n']):>2}{' clean' if r['homology_clean'] else '      '}  rho(disc) {s(r['spearman'], 3)} "
                 f"(p {pfmt(r['spearman_p_perm'])})  rho(GC) {s(r['spearman_gc_mcc'], 3)} (p {pfmt(r['spearman_gc_mcc_p_perm'])})  "
                 f"MAE disc {u(r['loto_MAE_disc_only'])} GC {u(r['loto_MAE_gc_only'])} label {u(r['loto_MAE_family_only'])}  "
                 f"rho(disc|GC) {pdg}; rho(GC|disc) {pgd}")
    T += [f"  and on the LARGEST SETS OF TASKS THAT SHARE NO LOCI (P13; multivar_posthoc_locus_independent.csv; all maximum",
          "  sets of mutually unlinked tasks at each threshold; descriptive):"]
    def rng_(lo_, hi_, f_):
        return f_(lo_) if f_(lo_) == f_(hi_) else f"{f_(lo_)} to {f_(hi_)}"
    for tau, g in sorted(p13g.items(), reverse=True):
        nsl = rng_(int(g['n_solved'].min()), int(g['n_solved'].max()), str)
        T.append(f"    link < {tau:.2f}: n = {int(g['max_unlinked_size'].iloc[0])} ({len(g)} sets, {nsl} solved tasks each, "
                 + ("none homology-clean" if not bool(g["homology_clean"].any()) else "some homology-clean") + ")  "
                 f"rho(disc) {rng_(g['spearman_disc_mcc'].min(), g['spearman_disc_mcc'].max(), lambda v: s(v, 3))} (exact p "
                 f"{rng_(g['spearman_disc_mcc_p_perm'].min(), g['spearman_disc_mcc_p_perm'].max(), pfmt)})  rho(GC) "
                 f"{rng_(g['spearman_gc_mcc'].min(), g['spearman_gc_mcc'].max(), lambda v: s(v, 3))} (exact p <= "
                 f"{pfmt(g['spearman_gc_mcc_p_perm'].max())})")
    dis = [rw[k] for k in ("dupfree9_keep_nt_enhancers", "dupfree9_keep_nt_enhancers_types", "noshare8_drop_ocr",
                           "noshare8_drop_regulatory")]
    check(all(r["spearman_p_perm"] < 0.05 for r in dis) and all(r["spearman"] > 0.7 for r in dis),
          "disc-MCC holds on exact-duplicate-free sets")
    check(all(r["loto_MAE_gc_only"] < r["loto_MAE_disc_only"] for r in dis), "GC lower MAE on all four exact-dup-free sets")
    check(all(r["spearman_gc_mcc"] >= r["spearman"] for r in dis), "GC rho >= disc rho on all four exact-dup-free sets")
    check(all(r["loto_MAE_family_only"] < r["loto_MAE_disc_only"] for r in rw.values()
              if r["loto_MAE_family_only"] == r["loto_MAE_family_only"]), "label beats disc wherever defined")
    dmin = min(r["spearman"] for r in dis)
    dmax = max(r["spearman"] for r in dis)
    pmax = max(r["spearman_p_perm"] for r in dis)
    gmin = min(r["spearman_gc_mcc"] for r in dis)
    gmax = max(r["spearman_gc_mcc"] for r in dis)
    lmin, lmax = g25["spearman_disc_mcc"].min(), g25["spearman_disc_mcc"].max()
    lpmin, lpmax = g25["spearman_disc_mcc_p_perm"].min(), g25["spearman_disc_mcc_p_perm"].max()
    lg, lgp = g25["spearman_gc_mcc"].max(), g25["spearman_gc_mcc_p_perm"].max()
    check(abs(g25["spearman_gc_mcc"].min() - lg) < 1e-12, "GC rho identical on all locus-independent sets")
    check(lpmax > 0.05 and lpmin < 0.05, "locus-independent disc p straddles 0.05")
    nlk = int(g25["max_unlinked_size"].iloc[0])
    T += [f"  Reading: after collapsing EXACT duplicates the disc-MCC correlation stays positive (n = 8-9: rho {s(dmin, 2)} to "
          f"{s(dmax, 2)}, exact p <= {u(pmax)}),",
          f"  but GC mean's is at least as high on those sets ({s(gmin, 2)} to {s(gmax, 2)}) and GC-only has the lower LOTO MAE on",
          f"  all four.  On the largest sets of tasks that share no LOCI (n = {nlk}, link < {tmain:.2f}; only "
          f"{rng_(int(g25['n_solved'].min()), int(g25['n_solved'].max()), str)} solved promoter tasks each) disc's rho is",
          f"  {s(lmin, 2)} to {s(lmax, 2)} (exact p {pfmt(lpmin)}-{pfmt(lpmax)}) and GC's {s(lg, 2)} (p = {pfmt(lgp)}).  The n and "
          "p-values of the 11-task analysis",
          "  therefore overstate the independent evidence, the promoter side of the between-cluster contrast rests on",
          "  largely shared promoter loci, and on every locus-independent set GC content is the stronger rank correlate",
          "  of MCC.  The P12/P13 rows are sensitivity readouts of one correlation, not separate tests.",
          ""]
    tab("Exact-duplicate-free task sets (P12): Spearman with MCC, disc / GC mean",
        f"n=9 (union dropped, one of the relabelled pair) disc {s(rw['dupfree9_keep_nt_enhancers']['spearman'], 2)} / "
        f"{s(rw['dupfree9_keep_nt_enhancers_types']['spearman'], 2)}; no exact sharing n=8 "
        f"{s(rw['noshare8_drop_ocr']['spearman'], 2)} / {s(rw['noshare8_drop_regulatory']['spearman'], 2)} (exact p <= {u(pmax, 3)}); "
        f"GC mean {s(gmin, 2)} to {s(gmax, 2)}; homology-clean exact-dup-free n=6 disc {s(rw['homology_clean7_minus_union']['spearman'], 2)} "
        f"(exact p={pfmt(rw['homology_clean7_minus_union']['spearman_p_perm'])})",
        "multivar_posthoc_distinct_datasets.csv (spearman, spearman_p_perm, spearman_gc_mcc)")
    tab(f"Largest task sets sharing no loci (P13, link < {tmain:.2f}, n={nlk}): Spearman with MCC",
        f"disc {s(lmin, 2)} to {s(lmax, 2)} (exact p {pfmt(lpmin)}-{pfmt(lpmax)}); GC mean {s(lg, 2)} (p={pfmt(lgp)})",
        "multivar_posthoc_locus_independent.csv (tau=0.25)")
    tab("Promoter loci shared across tasks (31-mer containment >= 0.5)",
        f"human_nontata_promoters positives in nt_promoter_all {100 * c_np['frac_ge_0_5']:.1f}%, in human_ensembl_regulatory "
        f"{100 * c_nr['frac_ge_0_5']:.1f}%; nt_promoter_all positives in human_ensembl_regulatory {100 * c_ar['frac_ge_0_5']:.1f}%",
        "multivar_posthoc_task_locus_overlap.csv (ref_label=all, frac_ge_0_5)")

    # ================================================================ H8 synthetic vs real
    q2 = one(p2, config=S11)
    q2s = one(p2, config=S15)
    a9 = syn[syn["config"] == S11]
    check(bool(q2["synthetic_in_boot_ci"]), "synthetic slope inside real CI")
    g = lambda k: float(p10.loc[k, "value"])
    syn_raw = p10.loc["synthetic_n_in_range", "note"]
    m_syn = re.search(r"motifs: (.*); seeds (\d+) \(points clustered by motif and seed\)", syn_raw)
    check(m_syn, "synthetic clustering note")
    syn_note = f"{len(m_syn.group(1).split(', '))} motifs ({m_syn.group(1)}) x {m_syn.group(2)} seeds"
    T += ["H8  'STRONG SYNTHETIC, WEAK REAL' (A9 pre-registered; P2, P10 post hoc)",
          f"  Slope: real S11 OLS slope {s(q2['real_slope'], 2)} (task bootstrap {ci(q2['real_slope_boot_lo'], q2['real_slope_boot_hi'], 2)}) is "
          f"consistent with the synthetic pooled slope {s(q2['synthetic_slope'], 2)}",
          f"    (inside the CI; the CI spans a factor {q2['real_slope_boot_hi'] / q2['real_slope_boot_lo']:.1f}) [multivar_posthoc_slope_transfer.csv].",
          f"  Level: real MCC sits {s(q2['level_offset_at_mean_d'], 3)} above the synthetic line (A9: zero parameters fitted on real tasks, MAE "
          f"{u(np.mean(np.abs(a9['resid'])))}).",
          f"    The offset correlates with composition-only MCC (Spearman {s(g('spearman_level_offset_vs_comp_mcc'), 2)}, "
          f"{p10.loc['spearman_level_offset_vs_comp_mcc', 'note'].split(',')[0]}) and GC mean",
          f"    ({s(g('spearman_level_offset_vs_gc_mean'), 2)}, {p10.loc['spearman_level_offset_vs_gc_mean', 'note'].split(',')[0]}), not n_train "
          f"({s(g('spearman_level_offset_vs_ntrain_log'), 2)}) -- consistent with real sequences carrying",
          "    composition signal that random synthetic backgrounds lack, but correlational; the synthetic points also",
          f"    all used n_train/n_test {p10.loc['synthetic_n_train_n_test', 'value']} vs real n_train {p10.loc['real_S11_n_train_range', 'value']}.",
          f"  Fixing the slope at the synthetic value (1 free parameter) gives LOTO MAE {u(q2['slope_fixed_loto_MAE'])} vs {u(q2['free_loto_MAE'])} free "
          f"(dMAE {ci(q2['dMAE_lo'], q2['dMAE_hi'])}) on S11",
          f"    and {u(q2s['slope_fixed_loto_MAE'])} vs {u(q2s['free_loto_MAE'])} on S15 (dMAE {ci(q2s['dMAE_lo'], q2s['dMAE_hi'])}).",
          f"  Range restriction is not the reason the real fit is loose: the {int(g('synthetic_n_in_range'))} synthetic points inside the real",
          f"    disc range {p10.loc['real_S11_disc_range', 'value']} still give OLS R^2 {u(g('synthetic_in_range_ols_R2'))} (residual SD "
          f"{u(g('synthetic_in_range_resid_sd'))}) vs real S11 R^2 {u(g('real_S11_ols_R2'))}",
          f"    (residual SD {u(g('real_S11_resid_sd'))}, {g('resid_sd_ratio_real_over_synthetic'):.1f}x) [multivar_posthoc_synthetic_range.csv].",
          f"    Caveat: the {int(g('synthetic_n_in_range'))} synthetic points come from {syn_note} and are clustered by motif and",
          f"    seed, not {int(g('synthetic_n_in_range'))} independent units,",
          "    so this R^2 describes the fit tightness of three per-motif dose-response curves, not an n-point sample.",
          "    The looser real fit is between-task heterogeneity -- chiefly the solved/failing two-cluster contrast of",
          f"    H2 -- at n = 11 ({int(rw['S11_all']['n_exact_distinct_sets'])} exact-duplicate-free sequence sets that also share "
          "promoter loci, H9).",
          ""]

    # ================================================================ caveats + writer flags
    s11r = rw["S11_all"]
    T += ["CAVEATS",
          "  * n = 11 (15) tasks bounds every multivariate conclusion; that is the pre-registered verdict.",
          f"  * The 11 tasks are {int(s11r['n_exact_distinct_sets'])} exact-duplicate-free sequence sets "
          f"(at most {int(s11r['n_no_exact_sharing'])} without exact sharing), and the promoter tasks of",
          "    both suites and the regulatory task share most promoter loci (H9): n and p-values of task-level statistics",
          f"    overstate the independent evidence.  On the largest locus-independent sets (n = {nlk}) disc's rho is "
          f"{s(lmin, 2)} to {s(lmax, 2)}, GC's {s(lg, 2)}.",
          "  * Cross-task overlap is exact strings (P12) and 1/4-sampled canonical 31-mer containment (P13; 2,000",
          "    query sequences per class, Wilson CIs in the CSV).  The locus-link thresholds (0.50/0.25/0.10) are",
          "    descriptive choices; all three are reported.  Near-identical windows shorter than 31 bp are not seen.",
          "  * GC mean's own rank correlation with MCC (fix round 3) is descriptive: the difference to disc's is not",
          f"    resolvable (S11 CI {ci(h11['rho_gc_minus_rho_disc_lo'], h11['rho_gc_minus_rho_disc_hi'], 2)}).  The regulatory task's GC mean pools three classes.",
          "  * The promoter/enhancer label (P11) is post hoc and equals Table I's outcome-defined solved/failing split;",
          "    its predictive accuracy shows two MCC clusters, not an additional explanatory factor.",
          "  * P1-P13 are post hoc; P4/P8 were found through the residual they explain; P5-P13 respond to the",
          "    verification rounds.  None changes the pre-registered verdict.",
          f"  * The per-covariate partial t-tests are uncorrected ({cntt} at p < 0.05 on S11; {cnth} after Holm).  At n <= 9",
          "    the percentile bootstrap CIs of partials are unusable (undefined in many resamples); t and",
          "    residual-permutation p are reported, and |rho| = 1 is reported as degenerate.",
          "  * Pool-B models (with the composition-only baseline) predict well but DECOMPOSE rather than explain (a",
          f"    weaker classifier's score on the same test set); S15 nested-LOTO MAE "
          f"{u(one(nested, config=S15, family='poolB', selection='aicc')['nested_loto_MAE'])} vs {u(dso['loto_MAE'])} "
          "(multivar_nested.csv); never in the decision rule.",
          "  * Homology for the 5 largest tasks is a 5,000-test-query estimate; the 8-mer Jaccard is single-strand",
          "    (the verbatim audit, the task-overlap audit and the MMseqs2 alternative are both-strand).",
          "  * disc is taken as published (cross_suite_summary.csv).  Upstream quirk (not ours): run_cross_suite.py",
          "    looks up the motif CI by TF only, so for multiclass tasks the motif_ci column can come from another",
          "    contrast; the point values used here are correct (asserted in P8).",
          "  * The composition-only and dedup LightGBM predictions were computed with 3 threads before the 2-thread",
          "    CPU rule; both reproduce published values exactly (multivar_comp_reproduction.csv; P6 asserts Table 1",
          "    to 4 decimals).",
          "",
          "FLAGS FOR THE PAPER WRITER (statements in bibm/paper_bibm_submitted.tex these results change; ready-made",
          "LaTeX in multivar_paper_snippets.txt)",
          "  1. Sec. Route 1 'Real-task corroboration' + Table II: the 'homology-clean (n=9)' subset still contains",
          "     human_enhancers_ensembl and nt_enhancers_types (H7).  Only the n = 7 set (every task < 5% near-dup) is",
          f"     homology-clean ({s(q7['spearman'], 2)}, exact p = {pfmt(q7['spearman_p_perm'])}); the n = 8 set keeps nt_enhancers_types and",
          "     is a sensitivity row, not 'homology-clean'.  Snippets (4) and (7).",
          "  2. Sec. Route 1: 'On the real tasks, the synthetic law holds' -> on real tasks discriminability is collinear",
          f"     with GC content, GC tracks MCC at least as closely (rho {s(h11['spearman_gc_mcc'], 2)} vs {s(h11['spearman'], 2)}), and both separate",
          "     the solved from the failing tasks; the real tasks cannot isolate discriminability (H2, H9); the synthetic",
          "     dose-response is the evidence.  Snippet (7).",
          "  3. Limitations (i): add the multivariate result (H1, H4), the noise share (H5) and the two-cluster reading",
          "     (H2).  Snippet (5).",
          "  4. Limitations (iv) and Table I: human_enhancers_ensembl (and nt_enhancers_types) join the partly",
          "     homology-inflated tasks (H7).",
          "  5. Sec. Route 1: the LOTO 'MAE 0.14 [0.11, 0.18]' is the n = 11 fit, not the 9-task subset it follows in",
          "     the text (H1, H7).",
          "  6. Table 1 row nt_enhancers_types ('shared-motif', MCC 0.476): its MCC is mostly carried by class-2 test",
          "     sequences that are verbatim train copies (H7) -- check the label and footnote the MCC.",
          "  7. Table II / Route 1: the tasks are not independent (H9) -- footnote (snippet 6); the NT-only row rests on",
          f"     {int(rw['suite_NT_only']['n_exact_distinct_sets'])} exact-duplicate-free sequence sets; the promoter tasks share loci across suites;",
          f"     the p-values should be exact permutation p (paper's 9: {pfmt(q9['spearman_p_perm'])}, not {pfmt(q9['spearman_p'])}).  Table II",
          "     should carry GC mean's rank correlation next to discriminability's (snippet 4).",
          ""]
    out = "\n".join(T) + "\n"
    with open(os.path.join(RD, "multivar_interpretation.txt"), "w") as fh:
        fh.write(out)

    # ================================================================ paper table + snippets
    tab("Post hoc: enhancer-oriented disc for nt_enhancers_types (uniform functional-class rule), LOTO S11",
        f"R^2 {s(u1['loto_R2'], 2)} {ci(u1['loto_R2_lo'], u1['loto_R2_hi'], 2)} (range over readings {s(alts['loto_R2'].min(), 2)} to "
        f"{s(alts['loto_R2'].max(), 2)}); pre-registered {s(a0['loto_R2'], 2)} stays primary",
        "multivar_posthoc_orientation_robustness.csv S11/U1_functional_representative")
    pd.DataFrame(TAB).to_csv(os.path.join(RD, "multivar_paper_table.csv"), index=False)

    def pt(x):
        """p for LaTeX: 2 significant digits, never 0"""
        check(x > 0, "p must be > 0")
        return f"{x:.2g}" if x >= 0.001 else f"{x:.1g}"

    # (1) main-text paragraph (<= 120 words).  Fix round 3: the clean-7 'discriminability beats GC' clause is
    #     qualified by the one task it rests on, and GC mean's own rank correlation is reported.
    n9 = int(s11r["n_exact_distinct_sets"])
    para = (r"\textbf{Multivariate control.} On the 11 TF-motif tasks "
            f"({n9} exact-duplicate-free sequence sets) we regressed MCC on discriminability "
            r"plus nine pre-registered covariates, including length, within-class GC SD and train/test homology. "
            r"Selection-honest nested leave-one-task-out search did not beat discriminability alone (MAE "
            f"{u(prim['nested_loto_MAE'])} vs.\\ {u(prim['disc_only_loto_MAE'])}; "
            f"$\\Delta$ 95\\% CI $[{s(prim['dMAE_lo'], 2)},{s(prim['dMAE_hi'], 2)}]$); the discriminability+length+GC-SD+homology "
            f"model was worse ({u(rr11['loto_MAE'])}). Homology did not confound discriminability (partial "
            f"$\\rho{{=}}{s(hpo['partial_rho'], 2)}$), but discriminability is collinear with GC content "
            f"($\\rho{{=}}{s(g7['spearman_disc_gc'], 2)}$); both separate the solved promoter from the failing "
            f"enhancer/OCR tasks. Post hoc, GC alone tracks MCC at least as closely ($\\rho{{=}}{s(h11['spearman_gc_mcc'], 2)}$; "
            f"MAE {u(g7['gc_only_loto_MAE'])}); discriminability beats GC on the seven homology-clean tasks "
            f"({u(fam_s7['loto_MAE_disc_only'])} vs.\\ {u(fam_s7['loto_MAE_gc_only'])}) only via the three-class regulatory task "
            f"(without it: {u(h7nr['loto_MAE_disc_only'])} vs.\\ {u(h7nr['loto_MAE_gc_only'])}). "
            r"Real tasks cannot isolate discriminability; the synthetic dose-response does.")
    words = len(para.split())
    check(words <= 120, f"paragraph <= 120 words ({words})")
    check(int(fam_s7["n"]) == 7 and bool(q7["homology_clean"]), "paragraph's 'seven homology-clean tasks'")
    check(fam_s7["loto_MAE_disc_only"] < fam_s7["loto_MAE_gc_only"] and h7c["loto_MAE_lower"] == "disc" and
          "human_ensembl_regulatory" in h7c["loto_MAE_order_reversed_by_dropping"].split(";") and
          h7nr["loto_MAE_gc_only"] < h7nr["loto_MAE_disc_only"], "'beats GC on the clean 7 only via the regulatory task'")
    check(g7["gc_only_loto_MAE"] <= g7["disc_only_loto_MAE"] and not excl0(g7["dMAE_lo"], g7["dMAE_hi"]), "GC MAE <= disc")
    check(abs(h11["spearman"] - g7["spearman_disc_gc"]) > 0 and h11["spearman_gc_mcc"] >= h11["spearman"], "'at least as closely'")
    ntn = rw["suite_NT_only"]
    gb_ = rw["suite_GB_only"]
    c6 = rw["homology_clean7_minus_union"]
    rebut = (
        "We fitted the requested regression on the real tasks (plan pre-registered; nine covariates incl. length, "
        "within-class GC SD and 8-mer train/test homology; all <=3-covariate subsets; selection-honest nested "
        f"leave-one-task-out). At n=11 TF-motif tasks no covariate set improved on discriminability "
        f"(MAE {u(prim['nested_loto_MAE'])} vs {u(prim['disc_only_loto_MAE'])}, Delta CI {ci(prim['dMAE_lo'], prim['dMAE_hi'], 2)}, "
        f"incremental permutation p={u(a4b['p'], 2)}). The discriminability + length + GC-SD + homology model you asked for "
        f"overfits (LOTO MAE {u(rr11['loto_MAE'])}; with 6 residual df no coefficient's CI excludes 0; on all 15 tasks only "
        f"discriminability's bootstrap CI excludes 0, while length's t-CI {ci(ll['t_lo'], ll['t_hi'], 2)} excludes 0 but its "
        f"bootstrap CI does not). Test-set sampling noise explains {100 * n11r['noise_share']:.1f}% of the misfit; the rest is "
        "between-task heterogeneity. Post hoc, its main axis is that the real MCCs form two non-overlapping clusters, the "
        "solved promoter/regulatory and the failing enhancer/OCR tasks of Table I, which discriminability and GC content "
        f"(rho {s(g7['spearman_disc_gc'], 2)}) both separate. GC content tracks MCC at least as closely as discriminability "
        f"(rho {s(h11['spearman_gc_mcc'], 2)} vs {s(h11['spearman'], 2)}; difference CI "
        f"{ci(h11['rho_gc_minus_rho_disc_lo'], h11['rho_gc_minus_rho_disc_hi'], 2)}) and predicts held-out MCC at least as "
        f"well (within {TIE} MCC) on all 11 tasks (MAE {u(g7['gc_only_loto_MAE'])}, a model chosen in-sample) and on every other task set we "
        "examined, except the Genomic-Benchmarks suite alone and homology-clean sets that contain the three-class regulatory "
        f"task: on the seven homology-clean tasks discriminability wins ({u(fam_s7['loto_MAE_disc_only'])} vs "
        f"{u(fam_s7['loto_MAE_gc_only'])}) only through that task ({u(h7nr['loto_MAE_disc_only'])} vs "
        f"{u(h7nr['loto_MAE_gc_only'])} without it). Within clusters neither is resolvable. Real tasks therefore cannot "
        "isolate discriminability, which is why Route 1 rests on the synthetic dose-response; we now say so. "
        f"Homology does not confound the relation (partial rho {s(hpo['partial_rho'], 2)}), but the audit found two errors on "
        f"our side. Our 'homology-clean' subset still held two leaky tasks; dropping both gives rho = {s(q7['spearman'], 2)} "
        f"(n=7, exact p={pt(q7['spearman_p_perm'])}; {s(q8['spearman'], 2)}, n=8, exact p={pt(q8['spearman_p_perm'])}, if "
        f"nt_enhancers_types is kept). And the 11 tasks are not independent units: they are {n9} exact-duplicate-free "
        "sequence sets (nt_promoter_all is the union of the two other NT promoter tasks, nt_enhancers_types relabels "
        "nt_enhancers, and the regulatory task's OCR class is human_ocr_ensembl's positive class), and the promoter tasks "
        f"of both suites share most promoter loci ({100 * c_np['frac_ge_0_5']:.0f}% of human_nontata_promoters' promoters lie "
        "mostly within nt_promoter_all's). With exact duplicates collapsed discriminability's rho stays "
        f"{s(dmin, 2)} to {s(dmax, 2)} (exact p<={pt(pmax)}); on the largest task sets sharing no loci (n={nlk}) it is "
        f"{s(lmin, 2)} to {s(lmax, 2)} (exact p {pt(lpmin)}-{pt(lpmax)}), against {s(lg, 2)} for GC content.")
    rows_tex = "\n".join(f"{r['statistic']} & {r['value']} \\\\" for r in TAB)
    # (4) Table II rows, two columns: rho(disc, MCC) and rho(GC mean, MCC).  GUARD: a row may say "Homology-clean"
    #     only if its task set has no task >= 0.05.
    d9a, d9b = rw["dupfree9_keep_nt_enhancers"], rw["dupfree9_keep_nt_enhancers_types"]
    d8a, d8b = rw["noshare8_drop_ocr"], rw["noshare8_drop_regulatory"]

    def cell(r, col, pcol):
        return f"${s(r[col], 2)}$ ($p{{=}}{pt(r[pcol])}$)"

    def cell2(ra, rb, col, pcol):
        va, vb = s(ra[col], 2), s(rb[col], 2)
        pm = max(ra[pcol], rb[pcol])
        if va == vb:
            return f"${va}$ ($p{{\\le}}{pt(pm)}$)"
        return f"${va}$ / ${vb}$ ($p{{\\le}}{pt(pm)}$)"
    t2 = [(f"Full TF-motif set ($n{{=}}{int(q11['n'])}$)", q11, cell(q11, "spearman", "spearman_p_perm"),
           cell(q11, "spearman_gc_mcc", "spearman_gc_mcc_p_perm")),
          (f"Homology-clean: every task $<$5\\% near-duplicates ($n{{=}}{int(q7['n'])}$)", q7,
           cell(q7, "spearman", "spearman_p_perm"), cell(q7, "spearman_gc_mcc", "spearman_gc_mcc_p_perm")),
          (f"Leaky tasks dropped except \\texttt{{nt\\_enhancers\\_types}} ($n{{=}}{int(q8['n'])}$)", q8,
           cell(q8, "spearman", "spearman_p_perm"), cell(q8, "spearman_gc_mcc", "spearman_gc_mcc_p_perm")),
          (f"Paper's former `homology-clean' subset, keeps 2 leaky tasks ($n{{=}}{int(q9['n'])}$)", q9,
           cell(q9, "spearman", "spearman_p_perm"), cell(q9, "spearman_gc_mcc", "spearman_gc_mcc_p_perm")),
          (f"Exact duplicates collapsed ($n{{=}}{int(d9a['n'])}$, 2 choices)", d9a,
           cell2(d9a, d9b, "spearman", "spearman_p_perm"), cell2(d9a, d9b, "spearman_gc_mcc", "spearman_gc_mcc_p_perm")),
          (f"No exact sequence sharing ($n{{=}}{int(d8a['n'])}$, 2 choices)", d8a,
           cell2(d8a, d8b, "spearman", "spearman_p_perm"), cell2(d8a, d8b, "spearman_gc_mcc", "spearman_gc_mcc_p_perm")),
          (f"No shared loci ($n{{=}}{nlk}$, {len(g25)} choices)", g25.iloc[0],
           f"${s(lmin, 2)}$ to ${s(lmax, 2)}$ ($p{{\\le}}{pt(lpmax)}$)", f"${s(lg, 2)}$ ($p{{=}}{pt(lgp)}$)"),
          (f"GB-only ($n{{=}}{int(gb_['n'])}$)", gb_, cell(gb_, "spearman", "spearman_p_perm"),
           cell(gb_, "spearman_gc_mcc", "spearman_gc_mcc_p_perm")),
          (f"NT-only ($n{{=}}{int(ntn['n'])}$; {int(ntn['n_exact_distinct_sets'])} exact-duplicate-free sets)", ntn,
           cell(ntn, "spearman", "spearman_p_perm"), cell(ntn, "spearman_gc_mcc", "spearman_gc_mcc_p_perm"))]
    for lab_, src_, _, _ in t2:
        if "omology-clean" in lab_ and "former" not in lab_:
            check(bool(src_["homology_clean"]), f"Table II row '{lab_}' claims homology-clean")
    check(all(r["spearman_disc_mcc_p_perm"] > 0 for _, r in g25.iterrows()), "locus row p")
    tab2 = ("Subset & $\\rho$(discriminability, MCC) & $\\rho$(GC mean, MCC)\\\\\n\\midrule\n"
            + "\n".join(f"{lab_} & {vd_} & {vg_}\\\\" for lab_, _, vd_, vg_ in t2))
    # (5) Limitations (i) sentence
    lim = (f"A pre-registered multivariate analysis (nine covariates including length, within-class GC SD and train/test "
           f"homology; nested leave-one-task-out) finds no covariate set that improves on discriminability (MAE "
           f"{u(prim['nested_loto_MAE'])} vs.\\ {u(prim['disc_only_loto_MAE'])}, $\\Delta$ CI $[{s(prim['dMAE_lo'], 2)},"
           f"{s(prim['dMAE_hi'], 2)}]$); test-set noise explains {100 * n11r['noise_share']:.1f}\\% of the misfit, and the "
           "remaining heterogeneity is, post hoc, the split between the solved promoter and the failing enhancer/OCR tasks, "
           f"which GC content tracks at least as closely as discriminability ($\\rho{{=}}{s(h11['spearman_gc_mcc'], 2)}$ vs.\\ "
           f"${s(h11['spearman'], 2)}$).")
    # (6) Table II footnote on non-independence
    foot = (f"The {int(q11['n'])} tasks are {n9} exact-duplicate-free sequence sets: "
            r"\texttt{nt\_promoter\_all} is the union of the TATA and no-TATA tasks, \texttt{nt\_enhancers\_types} "
            r"relabels \texttt{nt\_enhancers}, and the regulatory task's OCR class equals \texttt{human\_ocr\_ensembl}'s "
            r"positives; beyond exact copies, the promoter tasks of both suites and the regulatory task share most promoter "
            f"loci (31-mer containment; `no shared loci': largest sets with link $<${tmain:.2f}); the $n{{=}}{int(q7['n'])}$ row has "
            f"$p{{=}}{pt(c6['spearman_p_perm'])}$ on its {int(c6['n'])} exact-duplicate-free sets. Rows are sensitivity "
            r"readouts of one correlation, not separate tests; $p$: exact permutation.")
    # (7) replacement for the Route-1 'Real-task corroboration' paragraph
    corr = (r"\textbf{Real-task consistency.} On the 11 TF-motif tasks discriminability tracks learnability "
            f"(Spearman $\\rho{{=}}{s(q11['spearman'], 2)}$, $p{{=}}{pt(q11['spearman_p_perm'])}$; leave-one-task-out MAE "
            f"{u(a0['loto_MAE'], 2)} MCC, 95\\% CI $[{u(a0['loto_MAE_lo'], 2)}, {u(a0['loto_MAE_hi'], 2)}]$), but GC content "
            f"tracks it at least as closely ($\\rho{{=}}{s(h11['spearman_gc_mcc'], 2)}$): the two are collinear "
            f"($\\rho{{=}}{s(g7['spearman_disc_gc'], 2)}$) and both separate the solved promoter from the failing "
            "enhancer/OCR tasks. Dropping the four tasks with $\\ge$5\\% train/test near-duplicates gives "
            f"$\\rho{{=}}{s(q7['spearman'], 2)}$ ($n{{=}}{int(q7['n'])}$; exact $p{{=}}{pt(c6['spearman_p_perm'])}$ on its "
            f"{int(c6['n'])} exact-duplicate-free sets; Table~\\ref{{tab:corr}}), but the tasks are not independent: they are "
            f"{n9} exact-duplicate-free sequence sets, the promoter tasks share most promoter loci across suites, and on the "
            f"largest sets sharing no loci ($n{{=}}{nlk}$) $\\rho$ is ${s(lmin, 2)}$ to ${s(lmax, 2)}$ (GC: ${s(lg, 2)}$). "
            "The real-task relation is therefore consistent with the synthetic dose-response but is not independent "
            "evidence for it.")
    check(len(corr.split()) <= 120, f"snippet (7) <= 120 words ({len(corr.split())})")
    check(len(set(q7["dropped"].split(";"))) == 4, "'dropping the four tasks'")
    check(abs(round(a0["loto_MAE"], 2) - 0.14) < 1e-9, "MAE 0.14")
    check(bool(c6["homology_clean"]) and int(c6["n"]) == int(c6["n_exact_distinct_sets"]) == 6, "clean 7 -> 6 exact-dup-free")
    # global wording guards (fix round 3)
    alltext = "\n".join([out, para, rebut, rows_tex, tab2, lim, foot, corr])
    check("flip" not in alltext.lower(), "the word 'flip' is not used (orderings are reported per set instead)")
    check("sequence-disjoint" not in alltext and "distinct sequence set" not in alltext.replace("exact-duplicate-free", ""),
          "exact-duplicate terminology only")
    check(not re.search(r"p\s?=\s?0(\.0+)?(?![.0-9])", alltext), "no p printed as 0")
    snip = ["% multivar_paper_snippets.txt -- GENERATED by src/upgrades/multivar/write_interpretation.py from the CSVs",
            f"% (1) main-text paragraph ({words} words, <= 120)", para, "",
            f"% (2) rebuttal paragraph (response letter, R1.1; {len(rebut.split())} words)", rebut, "",
            "% (3) table rows: statistic & value (sources in multivar_paper_table.csv)", rows_tex, "",
            "% (4) Table II (tab:corr) replacement body, two columns, from multivar_posthoc_homology_clean.csv,",
            "%     multivar_posthoc_distinct_datasets.csv and multivar_posthoc_locus_independent.csv (tau = 0.25);",
            "%     p = exact permutation p over all n! orderings.  Use \\begin{tabular}{@{}lcc@{}}.  Only the n = 7 row may be",
            "%     labelled homology-clean (asserted).  GC mean's column is post hoc and descriptive.",
            tab2, "",
            "% (5) Limitations (i): sentence to add after the R^2 = +0.44 [-0.27, +0.70] sentence", lim, "",
            "% (6) Table II footnote (task non-independence; multivar_posthoc_task_identity.csv, _task_locus_links.csv)",
            foot, "",
            f"% (7) replacement for the Sec. Route 1 'Real-task corroboration' paragraph ({len(corr.split())} words)", corr, ""]
    with open(os.path.join(RD, "multivar_paper_snippets.txt"), "w") as fh:
        fh.write("\n".join(snip) + "\n")
    print(out)
    print("\n".join(snip))
    print(f"wrote multivar_interpretation.txt, multivar_paper_table.csv, multivar_paper_snippets.txt (paragraph {words} words)")


if __name__ == "__main__":
    main()
