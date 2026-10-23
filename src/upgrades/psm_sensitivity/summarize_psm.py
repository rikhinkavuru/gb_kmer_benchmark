#!/usr/bin/env python3
"""Summarize the PSM-sensitivity run and apply the PRE-REGISTERED decision rule verbatim
(see run_psm_sensitivity.py docstring). Reads results/upgrades/psm_sensitivity_{sweep,balance,
selection_bias}.csv; writes psm_sensitivity_summary.csv, psm_sensitivity_decision.csv and
psm_sensitivity_interpretation.txt. No model fitting here (pure aggregation).

POST-RUN NOTE (2026-09-24, after the full run's outputs were read). Added, without touching the
pre-registered decision code (decide()): (a) psm_sensitivity_paper_table.csv, a compact table of
already-computed rows, including the pre-specified fixed_model design on the overlap-weighted test;
(b) a data-driven HEADLINE block; (c) DESCRIPTIVE, POST HOC dissociation outputs
(psm_sensitivity_dissociation{,_stats}.csv): equal-retention comparisons of F2/F3 vs F1 arms and
Spearman / OLS summaries of k-mer AUROC gap vs composition-only AUROC gap vs -log retention across
arms. These are labelled post hoc wherever they are printed and never enter a verdict.

POST-VERIFICATION NOTE (2026-09-24, after two independent verification reviews). decide() is still
unchanged. Added: (d) the POST HOC outputs of run_posthoc.py (psm_sensitivity_posthoc_{sweep,balance}.csv)
are appended to the pre-registered rows under their own dataset labels (nt_enhancers_leakfreeTest,
nt_enhancers_leakfreeRefit) or families (F1bw_hist / paperbw_anchor = class-balanced training), so the
SAME decision code and tables are applied to them as sensitivity analyses; (e) reported-only columns
(posthoc_columns: class-balanced-training decomposition, discarded-negative ELS rate, matched-set
balance, between-dataset contrast); (f) a data-driven headline: every qualitative sentence is chosen
by an explicit numeric rule printed next to it, never a fixed template; (g) Bonferroni-adjusted
p-values for the selection-bias tests.

SECOND FIX-UP NOTE (2026-09-24, reporting only; no new fits): (h) a weighting diagnostic, the
composition-only AUROC under the same overlap weights (posthoc columns *_comp_auroc,
*_weights_overcorrect), because for nt_enhancers the weighted k-mer AUROC falls BELOW 0.5 and a
'share of above-chance AUROC lost' above 1 cannot be read as a share; (i) the random-stream overlap
note and the limits of the dissociation pairs are printed with them.

THIRD FIX-UP NOTE (2026-09-24 ~11:00, reporting only; no new fits): (j) the label-noise estimate is also
computed for the leak-free nt rows (same retained negatives as raw nt, so the raw nt ELS rates are reused;
the variant's own random-control AUROC and composition effect enter the formula).

FIX ROUND 1 NOTE (2026-09-24 ~16:40, after the re-check; decide() still unchanged). (k) The label-noise lines
now print the Bonferroni p of the ELS enrichment behind rho and say, by rule, when the check says little (the
composition effect itself NOT SUPPORTED, or the ELS enrichment not significant). (l) The between-dataset block
prints run_posthoc.py P6: the composition-only share lost under the same weights and resamples, the difference
of contrasts and the ratio difference, read by the rule fixed in the run_posthoc.py docstring. (m) The weighting
diagnostic names the design(s) whose composition-only weighted AUROC is below 0.5. (n) The P0b interval-overlap
audit is printed. (o) Selection-bias lines carry Bonferroni p-values.

FIX ROUND 2 NOTE (2026-09-24 ~17:40, reporting only; decide() still unchanged; the only new numbers come from
run_posthoc.py P7, whose plan was fixed in its docstring before it ran). (p) Composition-only MCC@0.5 columns
(comp_mcc*) and the share of each model's published-split MCC collapse that size-matched random negatives reproduce
(read 'fully' >= 1, 'mostly' >= 0.5, else 'partly'). (q) P7 decomposition CIs (p7_* columns in decision.csv;
psm_sensitivity_decomposition_ci.csv). (r) Wording, all data-driven: the pre-registered 'shortcut' term is printed as
the 'matched-vs-random effect' (it measures what matching removes, whatever its cause), the size component is stated
with its share and CI, matching is said to remove most but not all composition-associated signal (with the
composition-only AUROC on the published split), the clean control's drop is not called a composition effect, the F2
adverse flag is scoped to the flagged (strictest) levels, the P6 scale-dependent READING prints the CI of the other
scale instead of asserting equality, label-noise lines lead with 'UNINFORMATIVE' when a qualifier applies, and caveat
(11) notes that MCC@0.5 is uninformative on the class-balanced rows as well.

FIX ROUND 3 NOTE (2026-09-24 ~18:20, reporting only; decide() still unchanged; the only new numbers come from
run_posthoc.py P7 (extended to the fixed_model design) and P8, whose plans were fixed in its FIX-ROUND-3 ADDENDUM
before they ran). (s) Every F2 / dissociation reading is reported per design (within = both models retrained on the
selected split; fixed_model = original models on the selected test; train_only = selected-split models on the
original test): f2_selection_difficulty.csv gains the composition-only gaps and the flag for each design, and
dissociation_stats.csv one row per design. The within-design F2 readings are no longer generalised: in the paired
designs the composition-only AUROC falls with the k-mer AUROC. (t) F2 is described by its construction (every
positive kept, negatives only inside the positives' central band; P8 = psm_sensitivity_f2_band.csv), which is why
its retrained composition-only AUROC can RISE; the phrases 'composition-preserving trimming' and 'retention alone
does not produce the drop' are withdrawn. (u) New reported-only columns: P7 fixed_model CIs (p7_fixed_*), the size
share of the total drop with its CI (p7_size_share_of_total*), R7b's own tau (p7_tau_R7b), composition-only F1 a=1
fixed_model / train_only values (comp_fixed_*, comp_trainonly_*), positive-side matching (f4_*), and the two-sided
caliper sweep (psm_sensitivity_caliper_sweep.csv). (v) The class-balanced-training line reads its size component
against whether the unweighted analysis had one."""
import argparse
import warnings
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import _common as C  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN balanced-draw MCC on weighted arms

KM = ["auroc", "mcc", "mcc_bw", "mcc_bw_pc", "bacc", "bacc_pc", "ap_pos", "ap_neg",
      "mcc_bal20", "mcc_bal20_pc"]
TOL_MIN = 0.03
LF_TEST, LF_REFIT = "nt_enhancers_leakfreeTest", "nt_enhancers_leakfreeRefit"
NT_RAW = "nt_enhancers"
ORDER = ["human_enhancers_cohn", "nt_enhancers", LF_TEST, LF_REFIT, "drosophila_enhancers_stark"]
POSTHOC_DS = {LF_TEST: "post hoc: nt scored without the 100 test positives that occur verbatim in train",
              LF_REFIT: "post hoc: nt refit without the 107 train rows equal to a test sequence; full test"}
# post hoc, descriptive thresholds for flagging selection-induced difficulty in F2 (printed with the flag)
F2_KGAP_MIN, F2_CGAP_MAX = 0.05, 0.02
DISSOC_MARGIN = 0.3


def ordered(df):
    k = {d: i for i, d in enumerate(ORDER)}
    return df.assign(_o=df["dataset"].map(lambda d: k.get(d, 99))).sort_values("_o", kind="stable").drop(columns="_o")


def _seed(v):
    return None if pd.isna(v) else int(float(v))


def summarize(sw, bal):
    """One row per (dataset, family, level, design, model): matched value (+CI), matched
    selection-seed range, random-control mean/sd/min/max/2.5-97.5 pct and the gap."""
    rows = []
    sw = sw.copy()
    sw["train_seed_i"] = sw["train_seed"].map(_seed)
    sw["test_seed_i"] = sw["test_seed"].map(_seed)
    balk = bal.set_index(["dataset", "arm"])          # arm names repeat across datasets
    assert balk.index.is_unique
    for (ds, fam, lev, design, model), g in sw.groupby(["dataset", "family", "level_label", "design", "model"],
                                                      sort=False):
        if design.startswith("cross") or fam == "original":
            continue
        vary = {"within": "train_kind", "fixed_model": "test_kind", "train_only": "train_kind",
                "within_ccrefreeTest": "train_kind", "fixed_model_ccrefreeTest": "test_kind"}[design]
        seedcol = "test_seed_i" if design.startswith("fixed_model") else "train_seed_i"
        m_all = g[g[vary] == "matched"]
        rnd = g[g[vary] == "random"]
        if len(m_all) == 0:
            continue
        m = m_all[m_all[seedcol].isin([42])] if (m_all[seedcol] == 42).any() else m_all.iloc[:1]
        m = m.iloc[0]
        arm_name = (m["test_arm"] if design != "train_only" else m["train_arm"]).split("|")[0]
        b = balk.loc[(ds, arm_name)]
        r = dict(dataset=ds, role=m["role"], family=fam, level_label=lev, level=m["level"], design=design,
                 model=model, retained_neg_frac=b["retained_neg_frac"], retained_pos_frac=b["retained_pos_frac"],
                 ess_neg_frac=b["ess_neg_frac"], max_abs_smd_matched=b["max_abs_smd"],
                 mean_abs_smd_matched=b["mean_abs_smd"], gc_auroc_matched=b["gc_auroc"],
                 tata_auroc_matched=b["tata_auroc"],
                 n_test_pos=m["n_pos"], n_test_neg=m["n_neg"], n_train_pos=m["n_train_pos"],
                 n_train_neg=m["n_train_neg"], underpowered=m["underpowered"], n_random=len(rnd),
                 n_matched_seeds=len(m_all))
        rb = balk.loc[[(ds, (x if design != "train_only" else y).split("|")[0])
                       for x, y in zip(rnd["test_arm"], rnd["train_arm"])]] if len(rnd) else None
        if rb is not None:
            r["max_abs_smd_random_mean"] = rb["max_abs_smd"].mean()
            r["gc_auroc_random_mean"] = rb["gc_auroc"].mean()
        for k in KM:
            r[f"{k}_matched"] = m[k]
            if f"{k}_lo" in m:
                r[f"{k}_matched_lo"], r[f"{k}_matched_hi"] = m[f"{k}_lo"], m[f"{k}_hi"]
            r[f"{k}_matched_seedmin"] = m_all[k].min()
            r[f"{k}_matched_seedmax"] = m_all[k].max()
            if len(rnd):
                v = rnd[k].to_numpy(float)
                r[f"{k}_random_mean"] = np.nanmean(v)
                r[f"{k}_random_sd"] = np.nanstd(v, ddof=1) if len(v) > 1 else np.nan
                r[f"{k}_random_min"] = np.nanmin(v)
                r[f"{k}_random_max"] = np.nanmax(v)
                r[f"{k}_random_lo_ci_mean"] = rnd[f"{k}_lo"].mean() if f"{k}_lo" in rnd else np.nan
                r[f"{k}_random_hi_ci_mean"] = rnd[f"{k}_hi"].mean() if f"{k}_hi" in rnd else np.nan
                r[f"{k}_gap_random_minus_matched"] = np.nanmean(v) - m[k]
        rows.append(r)
    return pd.DataFrame(rows)


def cross_table(sw, bal):
    """2x2 (model trained on matched|random) x (test matched|random), one row per MATCHED arm
    (every F1 level at seed 42, and the paper's published split), random = its 10 size-matched
    random arms. Keyed on the matched arm name (not family/level), so the paper-anchor model and
    the F1 seed-42 model -- which share the F1 a=1 random arms -- are never pooled."""
    rows = []
    x1 = sw[sw["design"] == "cross_matchedModel_randomTest"]
    x2 = sw[sw["design"] == "cross_randomModel_matchedTest"]
    w = sw[sw["design"] == "within"]
    balk = bal.set_index(["dataset", "arm"])
    for (ds, model, M), g1 in x1.groupby(["dataset", "model", "train_arm"], sort=False):
        g2 = x2[(x2["dataset"] == ds) & (x2["model"] == model) & (x2["test_arm"] == M)]
        rand_arms = g1["test_arm"].unique()
        mm = w[(w["dataset"] == ds) & (w["model"] == model) & (w["train_arm"] == M)]
        rr = w[(w["dataset"] == ds) & (w["model"] == model) & (w["train_arm"].isin(rand_arms))]
        r = dict(dataset=ds, model=model, matched_arm=M, level_label=g1["level_label"].iloc[0],
                 retained_neg_frac=balk.loc[(ds, M), "retained_neg_frac"], n_random=len(rand_arms))
        for k in ("auroc", "mcc_bw_pc"):
            r[f"{k}_matchedModel_matchedTest"] = mm[k].iloc[0]
            r[f"{k}_randomModel_randomTest"] = rr[k].mean()
            r[f"{k}_matchedModel_randomTest"] = g1[k].mean()
            r[f"{k}_randomModel_matchedTest"] = g2[k].mean()
            # decomposition of the within-arm gap (random-random minus matched-matched):
            r[f"{k}_test_effect_randomModel"] = r[f"{k}_randomModel_randomTest"] - r[f"{k}_randomModel_matchedTest"]
            r[f"{k}_train_effect_matchedTest"] = r[f"{k}_randomModel_matchedTest"] - r[f"{k}_matchedModel_matchedTest"]
        rows.append(r)
    return pd.DataFrame(rows)


def decide(sw, summ, sb=None):
    out = []
    for ds in sw["dataset"].unique():
        s = summ[(summ["dataset"] == ds) & (summ["model"] == "kmer")]
        o = sw[(sw["dataset"] == ds) & (sw["family"] == "original") & (sw["model"] == "kmer")].iloc[0]
        oc = sw[(sw["dataset"] == ds) & (sw["family"] == "original") & (sw["model"] == "comp")].iloc[0]
        A0, A0lo, A0hi = o["auroc"], o["auroc_lo"], o["auroc_hi"]
        tau = max(TOL_MIN, (A0hi - A0lo) / 2)

        def get(fam, lev, design):
            q = s[(s["family"] == fam) & (s["level_label"] == lev) & (s["design"] == design)]
            return q.iloc[0] if len(q) else None
        f1 = get("F1_hist", "a1", "within")
        ow = get("W_overlap", "w", "within")
        ts = get("F3_1to1", "c0.2", "within")
        fx = get("F1_hist", "a1", "fixed_model")
        tro = get("F1_hist", "a1", "train_only")
        pa = get("paper_anchor", "paper", "within")
        Am, Amhi = f1["auroc_matched"], f1["auroc_matched_hi"]
        Ar, Armin = f1["auroc_random_mean"], f1["auroc_random_min"]
        short = Ar - Am
        ia = (A0 - Ar) <= tau
        ib = (Am < Armin) and (Amhi < Ar)

        def removal_free(row):
            d = row["auroc_random_mean"] - row["auroc_matched"]
            ok = (row["auroc_matched"] < row["auroc_random_min"]) and (row["auroc_matched_hi"] < row["auroc_random_mean"]) \
                and (d >= 0.5 * short)
            return d, ok
        dw, iia = removal_free(ow)
        d2, iib = removal_free(ts)
        if ia and ib and iia and iib:
            verdict = "SUPPORTED"
        elif ib and iia and iib:
            verdict = "SUPPORTED-WITH-SIZE-COMPONENT"
        elif (not ib) or (not iia and not iib):
            verdict = "NOT SUPPORTED"
        else:
            verdict = "MIXED"
        total = A0 - Am
        S = short / total if total != 0 else np.nan
        r = dict(dataset=ds, role=o["role"], A0=A0, A0_lo=A0lo, A0_hi=A0hi, tau=tau,
                 retained_neg_frac=f1["retained_neg_frac"],
                 Am=Am, Am_lo=f1["auroc_matched_lo"], Am_hi=Amhi,
                 Am_seedmin=f1["auroc_matched_seedmin"], Am_seedmax=f1["auroc_matched_seedmax"],
                 Ar_mean=Ar, Ar_min=Armin, Ar_max=f1["auroc_random_max"],
                 drop_total=total, drop_size=A0 - Ar, drop_shortcut=short, shortcut_share=S,
                 crit_ia_size=ia, crit_ib_separation=ib,
                 Aw_overlap=ow["auroc_matched"], Aw_overlap_lo=ow["auroc_matched_lo"], Aw_overlap_hi=ow["auroc_matched_hi"],
                 Apw_placebo_mean=ow["auroc_random_mean"], Apw_placebo_min=ow["auroc_random_min"],
                 Apw_placebo_max=ow["auroc_random_max"], overlap_ess_neg_frac=ow["ess_neg_frac"],
                 drop_overlap=dw, ratio_overlap_to_shortcut=dw / short if short else np.nan, crit_iia_weighting=iia,
                 A2_1to1=ts["auroc_matched"], A2_1to1_lo=ts["auroc_matched_lo"], A2_1to1_hi=ts["auroc_matched_hi"],
                 A2r_mean=ts["auroc_random_mean"], A2r_min=ts["auroc_random_min"], A2r_max=ts["auroc_random_max"],
                 twosided_retained_frac=ts["retained_neg_frac"],
                 drop_twosided=d2, ratio_twosided_to_shortcut=d2 / short if short else np.nan, crit_iib_twosided=iib,
                 verdict=verdict,
                 # reported, not decision-driving
                 paper_anchor_auroc=pa["auroc_matched"] if pa is not None else np.nan,
                 paper_anchor_mcc=pa["mcc_matched"] if pa is not None else np.nan,
                 mcc0=o["mcc"], mcc_m=f1["mcc_matched"], mcc_r_mean=f1["mcc_random_mean"],
                 mcc_r_min=f1["mcc_random_min"], mcc_r_max=f1["mcc_random_max"],
                 mcc_drop_size_imbalance=o["mcc"] - f1["mcc_random_mean"],
                 mcc_drop_shortcut=f1["mcc_random_mean"] - f1["mcc_matched"],
                 balmcc0=o["mcc_bw_pc"], balmcc_m=f1["mcc_bw_pc_matched"], balmcc_m_lo=f1["mcc_bw_pc_matched_lo"],
                 balmcc_m_hi=f1["mcc_bw_pc_matched_hi"], balmcc_r_mean=f1["mcc_bw_pc_random_mean"],
                 balmcc_r_min=f1["mcc_bw_pc_random_min"], balmcc_r_max=f1["mcc_bw_pc_random_max"],
                 balmcc_drop_size=o["mcc_bw_pc"] - f1["mcc_bw_pc_random_mean"],
                 balmcc_drop_shortcut=f1["mcc_bw_pc_random_mean"] - f1["mcc_bw_pc_matched"],
                 fixed_A_matchedTest=fx["auroc_matched"], fixed_A_matchedTest_lo=fx["auroc_matched_lo"],
                 fixed_A_matchedTest_hi=fx["auroc_matched_hi"], fixed_A_randomTest_mean=fx["auroc_random_mean"],
                 fixed_A_randomTest_min=fx["auroc_random_min"], fixed_A_randomTest_max=fx["auroc_random_max"],
                 test_distortion_effect=A0 - fx["auroc_random_mean"],
                 test_shortcut_effect=fx["auroc_random_mean"] - fx["auroc_matched"],
                 trainonly_A_matchedTrain=tro["auroc_matched"], trainonly_A_randomTrain_mean=tro["auroc_random_mean"],
                 trainonly_A_randomTrain_min=tro["auroc_random_min"], trainonly_A_randomTrain_max=tro["auroc_random_max"],
                 training_size_effect=A0 - tro["auroc_random_mean"],
                 comp0=oc["auroc"])
        sc = summ[(summ["dataset"] == ds) & (summ["model"] == "comp")]
        c1 = sc[(sc["family"] == "F1_hist") & (sc["level_label"] == "a1") & (sc["design"] == "within")].iloc[0]
        r.update(comp_m=c1["auroc_matched"], comp_r_mean=c1["auroc_random_mean"])
        # ---- reported, not decision-driving: ATT stability, label-noise checks ----
        att = get("W_att", "w", "within")
        if att is not None:
            r.update(att_auroc=att["auroc_matched"], att_placebo_mean=att["auroc_random_mean"],
                     att_ess_neg_frac=att["ess_neg_frac"], att_unstable=bool(att["ess_neg_frac"] < 0.05))
        f5 = get("F5_ccrefree", "a1", "within")
        if f5 is not None:
            r.update(f5_retained_neg_frac=f5["retained_neg_frac"], f5_Am=f5["auroc_matched"],
                     f5_Am_lo=f5["auroc_matched_lo"], f5_Am_hi=f5["auroc_matched_hi"],
                     f5_Ar_mean=f5["auroc_random_mean"], f5_Ar_min=f5["auroc_random_min"],
                     f5_Ar_max=f5["auroc_random_max"], f5_drop=f5["auroc_random_mean"] - f5["auroc_matched"],
                     f5_max_abs_smd=f5["max_abs_smd_matched"])
        f5b = get("F5b_ccrefree_1to1", "c0.2", "within")
        if f5b is not None:
            r.update(f5b_retained_neg_frac=f5b["retained_neg_frac"], f5b_Am=f5b["auroc_matched"],
                     f5b_Am_lo=f5b["auroc_matched_lo"], f5b_Am_hi=f5b["auroc_matched_hi"],
                     f5b_Ar_mean=f5b["auroc_random_mean"], f5b_Ar_min=f5b["auroc_random_min"],
                     f5b_Ar_max=f5b["auroc_random_max"], f5b_drop=f5b["auroc_random_mean"] - f5b["auroc_matched"],
                     f5b_max_abs_smd=f5b["max_abs_smd_matched"])
        # descriptive, post hoc (see run script POST-SMOKE NOTE): share of above-chance AUROC lost
        r["frac_above_chance_lost_F1"] = short / (Ar - 0.5) if Ar > 0.5 else np.nan
        r["frac_above_chance_lost_overlap"] = dw / (ow["auroc_random_mean"] - 0.5)
        r["frac_above_chance_lost_twosided"] = d2 / (ts["auroc_random_mean"] - 0.5)
        cf = get("F1_hist", "a1", "within_ccrefreeTest")
        cfx = get("F1_hist", "a1", "fixed_model_ccrefreeTest")
        if cf is not None:
            o_cf = sw[(sw["dataset"] == ds) & (sw["family"] == "original") & (sw["model"] == "kmer")
                      & (sw["design"] == "within_ccrefreeTest")].iloc[0]
            r.update(ccrefreeTest_A0=o_cf["auroc"], ccrefreeTest_n_neg_orig=o_cf["n_neg"],
                     ccrefreeTest_Am=cf["auroc_matched"], ccrefreeTest_Am_lo=cf["auroc_matched_lo"],
                     ccrefreeTest_Am_hi=cf["auroc_matched_hi"], ccrefreeTest_n_neg_matched=cf["n_test_neg"],
                     ccrefreeTest_Ar_mean=cf["auroc_random_mean"], ccrefreeTest_Ar_min=cf["auroc_random_min"],
                     ccrefreeTest_Ar_max=cf["auroc_random_max"],
                     ccrefreeTest_drop=cf["auroc_random_mean"] - cf["auroc_matched"],
                     ccrefreeTest_fixed_Am=cfx["auroc_matched"], ccrefreeTest_fixed_Ar_mean=cfx["auroc_random_mean"],
                     ccrefreeTest_underpowered=bool(cf["underpowered"]))
        if sb is not None:
            q = sb[(sb["dataset"] == ds) & (sb["selection"] == "F1_hist_a1_seed42") & (sb["feature"] == "ccre_els")]
            if len(q):
                q = q.iloc[0]
                rp, rn, rr_ = q["mean_pos"], q["mean_neg_all"], q["mean_retained"]
                rho = min(1.0, max(0.0, (rr_ - rn) / (rp - rn))) if rp != rn else np.nan
                bound = rho * (Ar - 0.5)
                r.update(els_rate_pos=rp, els_rate_neg_all=rn, els_rate_retained=rr_,
                         els_n_retained=q["n_retained_nonmissing"], rho_hidden_pos=rho,
                         labelnoise_max_auroc_drop=bound,
                         labelnoise_bound_over_shortcut_drop=bound / short if short else np.nan)
        out.append(r)
    return pd.DataFrame(out)


def posthoc_columns(dec, summ, bal, sb=None, bt=None, sw=None, dc=None):
    """REPORTED-ONLY, POST HOC columns appended to the decision table (the verdict columns produced by
    the unchanged decide() are not touched)."""
    dec = dec.copy()
    balk = bal.set_index(["dataset", "arm"])
    for i, d in dec.iterrows():
        ds = d["dataset"]
        s = summ[(summ["dataset"] == ds) & (summ["model"] == "kmer")]

        def get(fam, lev, design):
            q = s[(s["family"] == fam) & (s["level_label"] == lev) & (s["design"] == design)]
            return q.iloc[0] if len(q) else None
        f1 = get("F1_hist", "a1", "within")
        upd = dict(posthoc_label=POSTHOC_DS.get(ds, ""),
                   f1_max_abs_smd=f1["max_abs_smd_matched"], f1_n_test_neg=f1["n_test_neg"],
                   f1_n_test_pos=f1["n_test_pos"])
        if (ds, "paper_equalized_matched") in balk.index:
            upd["paper_anchor_max_abs_smd"] = balk.loc[(ds, "paper_equalized_matched"), "max_abs_smd"]
        # class-balanced training weights (run_posthoc.py P3)
        bw, bwt = get("F1bw_hist", "a1", "within"), get("F1bw_hist", "a1", "train_only")
        if bw is not None:
            upd.update(bw_Am=bw["auroc_matched"], bw_Am_lo=bw["auroc_matched_lo"], bw_Am_hi=bw["auroc_matched_hi"],
                       bw_Am_seedmin=bw["auroc_matched_seedmin"], bw_Am_seedmax=bw["auroc_matched_seedmax"],
                       bw_Ar_mean=bw["auroc_random_mean"], bw_Ar_min=bw["auroc_random_min"],
                       bw_Ar_max=bw["auroc_random_max"],
                       bw_drop_size=d["A0"] - bw["auroc_random_mean"],
                       bw_drop_shortcut=bw["auroc_random_mean"] - bw["auroc_matched"],
                       bw_size_within_tau=bool(d["A0"] - bw["auroc_random_mean"] <= d["tau"]),
                       bw_mcc05_m=bw["mcc_matched"], bw_mcc05_r_mean=bw["mcc_random_mean"],
                       bw_mcc05_r_min=bw["mcc_random_min"], bw_mcc05_r_max=bw["mcc_random_max"],
                       bw_balmcc_m=bw["mcc_bw_pc_matched"], bw_balmcc_r_mean=bw["mcc_bw_pc_random_mean"],
                       bw_trainonly_Am=bwt["auroc_matched"] if bwt is not None else np.nan,
                       bw_trainonly_Ar_mean=bwt["auroc_random_mean"] if bwt is not None else np.nan)
            pb = get("paperbw_anchor", "paper", "within")
            if pb is not None:
                upd.update(bw_paper_anchor_auroc=pb["auroc_matched"], bw_paper_anchor_mcc05=pb["mcc_matched"])
        # weighting diagnostic (added at the second fix-up pass, post hoc): composition-only LightGBM AUROC
        # under the SAME overlap weights. > 0.5 -> weights leave residual (nonlinear) composition signal, so
        # the k-mer drop is conservative; < 0.5 -> the linear-propensity weights OVER-correct (they invert
        # the composition signal a nonlinear model uses), so the drop overstates what removing composition
        # alone would do and a 'share of above-chance AUROC lost' above 1 is not interpretable as a share.
        sc = summ[(summ["dataset"] == ds) & (summ["model"] == "comp") & (summ["family"] == "W_overlap")
                  & (summ["level_label"] == "w")]
        for design, tag in (("within", "overlap"), ("fixed_model", "fixedoverlap")):
            q = sc[sc["design"] == design]
            if len(q):
                cw = q.iloc[0]["auroc_matched"]
                upd.update({f"{tag}_comp_auroc": cw, f"{tag}_comp_auroc_lo": q.iloc[0]["auroc_matched_lo"],
                            f"{tag}_comp_auroc_hi": q.iloc[0]["auroc_matched_hi"],
                            f"{tag}_comp_placebo": q.iloc[0]["auroc_random_mean"],
                            f"{tag}_weights_overcorrect": bool(cw < 0.5)})
        # selection bias: ELS rate of DISCARDED negatives (the decide() columns hold all negatives)
        if sb is not None:
            q = sb[(sb["dataset"] == ds) & (sb["selection"] == "F1_hist_a1_seed42") & (sb["feature"] == "ccre_els")]
            if len(q):
                upd["els_rate_discarded"] = q.iloc[0]["mean_discarded"]
                upd["els_p_retained_vs_discarded"] = q.iloc[0]["p_retained_vs_discarded"]
                upd["els_p_bonferroni"] = q.iloc[0]["p_bonferroni"]
            # (j, third fix-up pass, post hoc) the leak-free nt variants keep EXACTLY the raw nt negatives (only test
            # positives / training positive copies differ), so the raw nt ELS rates give their hidden-positive
            # fraction rho; the label-noise AUROC estimate is recomputed with each variant's own random-control AUROC
            # and composition effect, using the same formula as decide(): rho * (Ar - 0.5).
            elif ds in (LF_TEST, LF_REFIT):
                q = sb[(sb["dataset"] == NT_RAW) & (sb["selection"] == "F1_hist_a1_seed42") & (sb["feature"] == "ccre_els")]
                if len(q):
                    q = q.iloc[0]
                    rp, rn, rr_ = q["mean_pos"], q["mean_neg_all"], q["mean_retained"]
                    rho = min(1.0, max(0.0, (rr_ - rn) / (rp - rn))) if rp != rn else np.nan
                    bound = rho * (d["Ar_mean"] - 0.5)
                    upd.update(els_rate_pos=rp, els_rate_neg_all=rn, els_rate_retained=rr_,
                               els_rate_discarded=q["mean_discarded"], els_n_retained=q["n_retained_nonmissing"],
                               els_p_retained_vs_discarded=q["p_retained_vs_discarded"],
                               els_p_bonferroni=q["p_bonferroni"],
                               rho_hidden_pos=rho, labelnoise_max_auroc_drop=bound,
                               labelnoise_bound_over_shortcut_drop=bound / d["drop_shortcut"] if d["drop_shortcut"] else np.nan,
                               labelnoise_source="nt_enhancers selection-bias rows (identical retained negatives)")
        # between-dataset contrast (run_posthoc.py P4): share of above-chance AUROC lost
        if bt is not None:
            for dz, tag in (("fixed_model_overlap", "fixedoverlap"), ("within_overlap", "withinoverlap")):
                q = bt[(bt["kind"] == "dataset") & (bt["dataset"] == ds) & (bt["design"] == dz)]
                if len(q):
                    q = q.iloc[0]
                    upd.update({f"{tag}_A": q["auroc_overlap"], f"{tag}_A_lo": q["auroc_overlap_lo"],
                                f"{tag}_A_hi": q["auroc_overlap_hi"], f"{tag}_placebo": q["auroc_placebo_mean"],
                                f"{tag}_share_lost": q["share_above_chance_lost"], f"{tag}_share_lo": q["share_lo"],
                                f"{tag}_share_hi": q["share_hi"]})
                    if "comp_share_lost" in q.index and pd.notna(q["comp_share_lost"]):   # run_posthoc.py P6
                        upd.update({f"{tag}_comp_share_lost": q["comp_share_lost"],
                                    f"{tag}_comp_share_lo": q["comp_share_lo"], f"{tag}_comp_share_hi": q["comp_share_hi"],
                                    f"{tag}_kmer_over_comp_share": q["kmer_over_comp_share"],
                                    f"{tag}_kmer_over_comp_share_lo": q["kmer_over_comp_share_lo"],
                                    f"{tag}_kmer_over_comp_share_hi": q["kmer_over_comp_share_hi"],
                                    f"{tag}_share_interpretable": bool(q["share_interpretable"])})
        # (p, fix round 2) composition-only MCC@0.5 along the same arms, and how much of each model's published-split
        # MCC collapse size-matched random negatives reproduce: (mcc0 - mcc_random) / (mcc0 - mcc_published)
        cs = summ[(summ["dataset"] == ds) & (summ["model"] == "comp")]
        c1 = cs[(cs["family"] == "F1_hist") & (cs["level_label"] == "a1") & (cs["design"] == "within")]
        cpa = cs[(cs["family"] == "paper_anchor") & (cs["design"] == "within")]
        if sw is not None and len(c1):
            oc = sw[(sw["dataset"] == ds) & (sw["family"] == "original") & (sw["model"] == "comp") & (sw["design"] == "within")]
            c1 = c1.iloc[0]
            upd.update(comp_mcc0=oc.iloc[0]["mcc"], comp_mcc_r_mean=c1["mcc_random_mean"],
                       comp_mcc_r_min=c1["mcc_random_min"], comp_mcc_r_max=c1["mcc_random_max"],
                       comp_mcc_m=c1["mcc_matched"],
                       comp_paper_anchor_mcc=cpa.iloc[0]["mcc_matched"] if len(cpa) else np.nan,
                       comp_paper_anchor_auroc=cpa.iloc[0]["auroc_matched"] if len(cpa) else np.nan,
                       comp_paper_anchor_auroc_lo=cpa.iloc[0]["auroc_matched_lo"] if len(cpa) else np.nan,
                       comp_paper_anchor_auroc_hi=cpa.iloc[0]["auroc_matched_hi"] if len(cpa) else np.nan,
                       comp_m_lo=c1["auroc_matched_lo"], comp_m_hi=c1["auroc_matched_hi"])
            with np.errstate(invalid="ignore", divide="ignore"):
                upd["mcc_collapse_share_reproduced_by_random"] = ((d["mcc0"] - d["mcc_r_mean"]) /
                                                                  (d["mcc0"] - d["paper_anchor_mcc"]))
                upd["comp_mcc_collapse_share_reproduced_by_random"] = ((upd["comp_mcc0"] - upd["comp_mcc_r_mean"]) /
                                                                       (upd["comp_mcc0"] - upd["comp_paper_anchor_mcc"]))
        # (q, fix round 2) run_posthoc.py P7: two-level paired-bootstrap CIs of the decomposition
        if dc is not None and (dc["dataset"] == ds).any():
            q = dc[dc["dataset"] == ds].set_index(["design", "model", "matched_seeds"])

            def g(design, model, ms, col):
                k = (design, model, ms)
                return q.loc[k, col] if k in q.index else np.nan
            for c_ in ("size_effect", "matched_vs_random", "rel_gap"):
                for sfx in ("", "_lo", "_hi"):
                    upd[f"p7_{c_}{sfx}"] = g("within", "kmer", "42", c_ + sfx)
                    upd[f"p7_comp_{c_}{sfx}"] = g("within", "comp", "42", c_ + sfx)
                    upd[f"p7_kmc_{c_}{sfx}"] = g("within", "kmer_minus_comp", "42", c_ + sfx)
                    upd[f"p7_s4244_{c_}{sfx}"] = g("within", "kmer", "42-44", c_ + sfx)
                    upd[f"p7_s4244_comp_{c_}{sfx}"] = g("within", "comp", "42-44", c_ + sfx)
                    upd[f"p7_s4244_kmc_{c_}{sfx}"] = g("within", "kmer_minus_comp", "42-44", c_ + sfx)
                    upd[f"p7_trainonly_{c_}{sfx}"] = g("train_only", "kmer", "42", c_ + sfx)
                    upd[f"p7_trainonly_comp_{c_}{sfx}"] = g("train_only", "comp", "42", c_ + sfx)
            for sfx in ("", "_lo", "_hi"):
                upd[f"p7_mvr_share_of_total{sfx}"] = g("within", "kmer", "42", "mvr_share_of_total" + sfx)
            upd["p7_size_vs_tau"] = g("within", "kmer", "42", "size_vs_tau")
            upd["p7_mvr_ci_excludes_zero"] = g("within", "kmer", "42", "mvr_ci_excludes_zero")
            upd["p7_comp_mvr_ci_excludes_zero"] = g("within", "comp", "42", "mvr_ci_excludes_zero")
            upd["p7_kmc_reading"] = g("within", "kmer_minus_comp", "42", "reading")
            upd["p7_s4244_kmc_reading"] = g("within", "kmer_minus_comp", "42-44", "reading")
            upd["p7_trainonly_kmc_reading"] = g("train_only", "kmer_minus_comp", "42", "reading")
            # (u, fix round 3) R7b's own tau, the size share of the total drop (1 - matched-vs-random share; the
            # percentile CI maps exactly under this monotone transform), train_only k-mer minus composition-only,
            # and the fixed_model design (original model on the arm's own test; P7 extended)
            upd["p7_tau_R7b"] = g("within", "kmer", "42", "tau_R7b")
            upd["p7_size_share_of_total"] = 1 - upd["p7_mvr_share_of_total"]
            upd["p7_size_share_of_total_lo"] = 1 - upd["p7_mvr_share_of_total_hi"]
            upd["p7_size_share_of_total_hi"] = 1 - upd["p7_mvr_share_of_total_lo"]
            for c_ in ("matched_vs_random", "rel_gap"):
                for sfx in ("", "_lo", "_hi"):
                    upd[f"p7_trainonly_kmc_{c_}{sfx}"] = g("train_only", "kmer_minus_comp", "42", c_ + sfx)
                    upd[f"p7_fixed_{c_}{sfx}"] = g("fixed_model", "kmer", "42", c_ + sfx)
                    upd[f"p7_fixed_comp_{c_}{sfx}"] = g("fixed_model", "comp", "42", c_ + sfx)
                    upd[f"p7_fixed_kmc_{c_}{sfx}"] = g("fixed_model", "kmer_minus_comp", "42", c_ + sfx)
            upd["p7_fixed_mvr_ci_excludes_zero"] = g("fixed_model", "kmer", "42", "mvr_ci_excludes_zero")
            upd["p7_fixed_comp_mvr_ci_excludes_zero"] = g("fixed_model", "comp", "42", "mvr_ci_excludes_zero")
            upd["p7_trainonly_comp_mvr_ci_excludes_zero"] = g("train_only", "comp", "42", "mvr_ci_excludes_zero")
            upd["p7_fixed_kmc_reading"] = g("fixed_model", "kmer_minus_comp", "42", "reading")
        # (u, fix round 3) composition-only model at F1 alpha=1 in the paired designs (summary.csv rows)
        for design, tag in (("fixed_model", "comp_fixed"), ("train_only", "comp_trainonly")):
            q = cs[(cs["family"] == "F1_hist") & (cs["level_label"] == "a1") & (cs["design"] == design)]
            if len(q):
                upd.update({f"{tag}_Am": q.iloc[0]["auroc_matched"], f"{tag}_Ar_mean": q.iloc[0]["auroc_random_mean"],
                            f"{tag}_mvr": q.iloc[0]["auroc_random_mean"] - q.iloc[0]["auroc_matched"]})
        # (u, fix round 3) positive-side matching F4: ALL negatives kept, positives subsampled; control = random
        # positive subsets of the same size (summary.csv rows)
        f4, f4c = get("F4_posside", "a1", "within"), cs[(cs["family"] == "F4_posside") & (cs["design"] == "within")]
        if f4 is not None:
            upd.update(f4_retained_pos_frac=f4["retained_pos_frac"], f4_retained_neg_frac=f4["retained_neg_frac"],
                       f4_Am=f4["auroc_matched"], f4_Am_lo=f4["auroc_matched_lo"], f4_Am_hi=f4["auroc_matched_hi"],
                       f4_Ar_mean=f4["auroc_random_mean"], f4_Ar_min=f4["auroc_random_min"],
                       f4_Ar_max=f4["auroc_random_max"], f4_mvr=f4["auroc_random_mean"] - f4["auroc_matched"],
                       f4_n_test_pos=f4["n_test_pos"], f4_n_test_neg=f4["n_test_neg"],
                       f4_underpowered=bool(f4["underpowered"]),
                       f4_comp_Am=f4c.iloc[0]["auroc_matched"] if len(f4c) else np.nan,
                       f4_comp_Ar_mean=f4c.iloc[0]["auroc_random_mean"] if len(f4c) else np.nan)
        for k, v in upd.items():
            dec.loc[i, k] = v
    return dec


def paper_table(dec, summ, sw):
    """Compact, paper-ready table (reported numbers only; no new analysis). One row per
    (dataset, condition). k-mer LightGBM unless the column says composition-only.
    For random / placebo conditions the value is the mean over the 10 seeds and *_min/*_max give the
    range; for a single matched arm *_lo/*_hi are its 95% bootstrap CI."""
    rows = []
    for _, d in dec.iterrows():
        ds = d["dataset"]
        s = summ[(summ["dataset"] == ds)]
        o = sw[(sw["dataset"] == ds) & (sw["family"] == "original") & (sw["design"] == "within")]
        ok, oc = o[o["model"] == "kmer"].iloc[0], o[o["model"] == "comp"].iloc[0]

        def g(fam, lev, design, model="kmer"):
            q = s[(s["family"] == fam) & (s["level_label"] == lev) & (s["design"] == design) & (s["model"] == model)]
            return q.iloc[0] if len(q) else None
        rows.append(dict(dataset=ds, condition="original (all negatives)", retained_neg_frac=1.0, retained_pos_frac=1.0,
                         n_test_pos=int(ok["n_pos"]), n_test_neg=int(ok["n_neg"]),
                         auroc=ok["auroc"], auroc_lo=ok["auroc_lo"], auroc_hi=ok["auroc_hi"],
                         mcc05=ok["mcc"], mcc05_lo=ok["mcc_lo"], mcc05_hi=ok["mcc_hi"],
                         bal_mcc=ok["mcc_bw_pc"], bal_mcc_lo=ok["mcc_bw_pc_lo"], bal_mcc_hi=ok["mcc_bw_pc_hi"],
                         mcc_bal20_pc=ok["mcc_bal20_pc"], ap_neg=ok["ap_neg"], neg_prevalence=1 - ok["prev_pos"],
                         comp_auroc=oc["auroc"]))
        spec = [("composition-matched (paper retention; F1 a=1)", "F1_hist", "a1", "within", "matched"),
                ("size-matched random negatives (10 seeds)", "F1_hist", "a1", "within", "random"),
                ("overlap weighting, nothing discarded", "W_overlap", "w", "within", "matched"),
                ("permuted-weight placebo (10 seeds)", "W_overlap", "w", "within", "random"),
                ("two-sided 1:1 caliper 0.2 SD", "F3_1to1", "c0.2", "within", "matched"),
                ("random 1:1 subsets, same n (10 seeds)", "F3_1to1", "c0.2", "within", "random"),
                ("positive-side matching (all negatives kept)", "F4_posside", "a1", "within", "matched"),
                ("random positive subsets, same n (10 seeds)", "F4_posside", "a1", "within", "random"),
                ("fixed original model -> overlap-weighted ORIGINAL test", "W_overlap", "w", "fixed_model", "matched"),
                ("fixed original model -> placebo-weighted ORIGINAL test", "W_overlap", "w", "fixed_model", "random"),
                ("fixed original model -> matched test", "F1_hist", "a1", "fixed_model", "matched"),
                ("fixed original model -> random test, same n/ratio", "F1_hist", "a1", "fixed_model", "random"),
                ("matched-trained model -> ORIGINAL test", "F1_hist", "a1", "train_only", "matched"),
                ("random-reduced-trained model -> ORIGINAL test", "F1_hist", "a1", "train_only", "random"),
                ("[post hoc] composition-matched, class-balanced training weights", "F1bw_hist", "a1", "within",
                 "matched"),
                ("[post hoc] size-matched random, class-balanced training weights (10 seeds)", "F1bw_hist", "a1",
                 "within", "random"),
                ("[post hoc] published split, class-balanced training weights", "paperbw_anchor", "paper", "within",
                 "matched"),
                ("published comp-equalized split (paper)", "paper_anchor", "paper", "within", "matched")]
        for label, fam, lev, design, kind in spec:
            r, c = g(fam, lev, design), g(fam, lev, design, "comp")
            if r is None:
                continue
            ret = r["ess_neg_frac"] if fam.startswith("W_") else r["retained_neg_frac"]
            row = dict(dataset=ds, condition=label, retained_neg_frac=ret,
                       retained_pos_frac=r["retained_pos_frac"],
                       n_test_pos=int(r["n_test_pos"]), n_test_neg=int(r["n_test_neg"]),
                       max_abs_smd=r["max_abs_smd_matched"] if kind == "matched" else r.get("max_abs_smd_random_mean", np.nan))
            if design == "train_only":
                row["n_test_pos"], row["n_test_neg"] = int(ok["n_pos"]), int(ok["n_neg"])
            if kind == "matched":
                for k, kk in (("auroc", "auroc"), ("mcc05", "mcc"), ("bal_mcc", "mcc_bw_pc")):
                    row[k], row[k + "_lo"], row[k + "_hi"] = r[f"{kk}_matched"], r[f"{kk}_matched_lo"], r[f"{kk}_matched_hi"]
                row.update(mcc_bal20_pc=r["mcc_bal20_pc_matched"], ap_neg=r["ap_neg_matched"],
                           comp_auroc=c["auroc_matched"] if c is not None else np.nan)
            else:
                for k, kk in (("auroc", "auroc"), ("mcc05", "mcc"), ("bal_mcc", "mcc_bw_pc")):
                    row[k], row[k + "_min"], row[k + "_max"] = (r[f"{kk}_random_mean"], r[f"{kk}_random_min"],
                                                                r[f"{kk}_random_max"])
                row.update(mcc_bal20_pc=r["mcc_bal20_pc_random_mean"], ap_neg=r["ap_neg_random_mean"],
                           comp_auroc=c["auroc_random_mean"] if c is not None else np.nan)
            # negative-class prevalence of the (possibly weighted) test set this row is scored on
            q = sw[(sw["dataset"] == ds) & (sw["family"] == fam) & (sw["level_label"] == lev) & (sw["design"] == design)
                   & (sw["model"] == "kmer") & (sw["train_kind" if design in ("within", "train_only") else "test_kind"] == kind)]
            row["neg_prevalence"] = float(1 - q["prev_pos"].mean()) if len(q) else np.nan
            rows.append(row)
    t = pd.DataFrame(rows)
    t["ap_neg_lift"] = t["ap_neg"] / t["neg_prevalence"]
    return t


def dissociation(summ):
    """DESCRIPTIVE, POST HOC (not part of the pre-registered rule; added after the full run was read).
    Size vs shortcut dissociation at (near-)equal retention: for every F2 (common-support trimming) and
    F3 (two-sided caliper) level, pair it with the F1 level whose retained-negative fraction is closest,
    and report k-mer AUROC next to the composition-only AUROC (how much composition shortcut is still
    available) and the size-matched random mean. If the collapse were a sample-size effect, arms with
    equal retention would have equal k-mer AUROC whatever their residual composition signal.
    NOTE: an F1 level can be the nearest partner of several F2/F3 levels, so this pair table repeats F1
    rows; it is a display table. The correlation summary (dissociation_stats) uses each arm once. The
    random controls of F1, F2, F3 and F5 draw negatives from the same priority stream (10000+r), so
    random arms of different families are nested/overlapping, not independent."""
    rows = []
    s = summ[summ["design"] == "within"]
    for ds in s["dataset"].unique():
        k = s[(s["dataset"] == ds) & (s["model"] == "kmer")]
        c = s[(s["dataset"] == ds) & (s["model"] == "comp")].set_index(["family", "level_label"])
        f1 = k[k["family"] == "F1_hist"]
        for fam in ("F2_trim", "F3_1to1"):
            for _, r in k[k["family"] == fam].iterrows():
                j = (f1["retained_neg_frac"] - r["retained_neg_frac"]).abs().idxmin()
                r1 = f1.loc[j]
                for tag, rr in ((fam, r), ("F1_hist", r1)):
                    cc = c.loc[(rr["family"], rr["level_label"])]
                    rows.append(dict(dataset=ds, pair=f"{fam}:{r['level_label']}", family=tag,
                                     level_label=rr["level_label"], retained_neg_frac=rr["retained_neg_frac"],
                                     retained_pos_frac=rr["retained_pos_frac"], n_test_neg=rr["n_test_neg"],
                                     max_abs_smd=rr["max_abs_smd_matched"], gc_auroc=rr["gc_auroc_matched"],
                                     comp_auroc=cc["auroc_matched"], comp_auroc_random=cc["auroc_random_mean"],
                                     kmer_auroc=rr["auroc_matched"], kmer_auroc_lo=rr["auroc_matched_lo"],
                                     kmer_auroc_hi=rr["auroc_matched_hi"], kmer_auroc_random=rr["auroc_random_mean"],
                                     kmer_gap_random_minus_matched=rr["auroc_random_mean"] - rr["auroc_matched"],
                                     underpowered=rr["underpowered"],
                                     retention_comparable=bool(abs(np.log(r["retained_neg_frac"] / r1["retained_neg_frac"]))
                                                               <= np.log(1.25))))
    return pd.DataFrame(rows)


def dissociation_stats(summ):
    """DESCRIPTIVE, POST HOC: across every non-underpowered selection arm of the three retention-based
    families (F1 seed-42 levels, F2, F3), is the k-mer AUROC loss vs its size-matched random control
    explained by how much composition signal was removed (composition-only AUROC loss vs its control)
    or by how many negatives were removed (-log retained fraction = selection depth)? Spearman correlations and
    an OLS fit on both standardized predictors. Arms are nested/overlapping, so no p-values are reported.
    Each (family, level) arm enters ONCE (asserted); all 12 F1 levels are used, not only the F1 levels
    that happen to be nearest partners in the dissociation() display table.
    Fix round 3: one row per design (within = both models retrained on the selected split; fixed_model = the
    original models on the selected test; train_only = the selected-split models on the original test)."""
    from scipy.stats import spearmanr
    rows = []
    for design in ("within", "fixed_model", "train_only"):
        s = summ[(summ["design"] == design) & (summ["family"].isin(["F1_hist", "F2_trim", "F3_1to1"]))]
        for ds in s["dataset"].unique():
            k = s[(s["dataset"] == ds) & (s["model"] == "kmer") & (~s["underpowered"].astype(bool))]
            assert not k.duplicated(["family", "level_label"]).any(), f"{ds}: repeated arm in dissociation_stats"
            if len(k) < 5:
                continue
            c = s[(s["dataset"] == ds) & (s["model"] == "comp")].set_index(["family", "level_label"])
            comp_gap = np.array([c.loc[(f, l), "auroc_random_mean"] - c.loc[(f, l), "auroc_matched"]
                                 for f, l in zip(k["family"], k["level_label"])])
            kgap = (k["auroc_random_mean"] - k["auroc_matched"]).to_numpy()
            nlr = -np.log(k["retained_neg_frac"].to_numpy())
            X = np.c_[(comp_gap - comp_gap.mean()) / comp_gap.std(), (nlr - nlr.mean()) / nlr.std()]
            A = np.c_[np.ones(len(kgap)), X]
            beta, *_ = np.linalg.lstsq(A, kgap, rcond=None)
            pred = A @ beta
            r2 = 1 - ((kgap - pred) ** 2).sum() / ((kgap - kgap.mean()) ** 2).sum()
            sc, sr = spearmanr(kgap, comp_gap)[0], spearmanr(kgap, nlr)[0]
            reading = ("gap tracks composition-signal removal more than selection depth" if sc - sr >= DISSOC_MARGIN else
                       "gap tracks selection depth (-log retention) more than composition-signal removal"
                       if sr - sc >= DISSOC_MARGIN else "no clear separation of composition-signal removal and selection depth")
            rows.append(dict(dataset=ds, design=design, n_arms=len(kgap), n_F1=int((k["family"] == "F1_hist").sum()),
                             n_F2=int((k["family"] == "F2_trim").sum()), n_F3=int((k["family"] == "F3_1to1").sum()),
                             reading=reading, reading_rule=f"|Spearman difference| >= {DISSOC_MARGIN}",
                             spearman_kmer_gap_vs_comp_gap=sc, spearman_kmer_gap_vs_neglog_retention=sr,
                             ols_beta_comp_gap_per_sd=beta[1], ols_beta_neglog_retention_per_sd=beta[2], ols_r2=r2))
    return pd.DataFrame(rows)


def dst_row(dst, ds, design="within"):
    """One dataset x design row of dissociation_stats (None if absent)."""
    if dst is None:
        return None
    q = dst[(dst["dataset"] == ds) & (dst["design"] == design)]
    return q.iloc[0] if len(q) else None


def report_dissociation(dz, dst=None, f2d=None):
    L = []
    if dst is not None:
        L.append("[DESCRIPTIVE, POST HOC] across all non-underpowered F1/F2/F3 arms (psm_sensitivity_dissociation_stats.csv):"
                 " k-mer AUROC gap (random - matched) vs composition-only AUROC gap and vs -log(retained fraction)"
                 " (selection depth), per design (within = retrained; fixed_model = original models on the selected"
                 " test; train_only = selected-split models on the original test)")
        for _, r in dst.iterrows():
            L.append(f"  {r['dataset']:<28} {r['design']:<12} n={int(r['n_arms'])}  Spearman(gap, comp gap) "
                     f"{r['spearman_kmer_gap_vs_comp_gap']:+.2f}  "
                     f"Spearman(gap, -log ret) {r['spearman_kmer_gap_vs_neglog_retention']:+.2f}  OLS per-SD: comp gap "
                     f"{r['ols_beta_comp_gap_per_sd']:+.3f}, -log ret {r['ols_beta_neglog_retention_per_sd']:+.3f} (R2 {r['ols_r2']:.2f})")
        L.append("")
    L += ["[DESCRIPTIVE, POST HOC] same retention, different residual composition signal "
          "(psm_sensitivity_dissociation.csv; k-mer AUROC matched | size-matched random; comp = composition-only AUROC).",
          "  The random controls of F1, F2, F3 and F5 draw negatives from the same priority stream (10000+r), so random",
          "  arms of different families are nested/overlapping, not independent; an F1 row can partner several F2/F3 rows.",
          "  Within design only (both models retrained on the selected split). Do NOT read these pairs as showing that",
          "  retention alone does or does not produce the drop: F2 keeps every positive but only negatives inside the",
          "  positives' central propensity band, so it can also CREATE composition signal for the positives outside the",
          "  band (psm_sensitivity_f2_band.csv), and in the paired designs (fixed_model, train_only) F2 trimming lowers",
          "  composition-only and k-mer AUROC together (psm_sensitivity_f2_selection_difficulty.csv)."]
    if f2d is not None and (f2d["dataset"] == "human_enhancers_cohn").any():
        q = f2d[(f2d["dataset"] == "human_enhancers_cohn") & (~f2d["underpowered"])]
        r = q.loc[q["retained_neg_frac"].idxmin()]
        L.append(f"  e.g. cohn F2 {r['level_label']}: retrained composition-only AUROC {r['comp_matched']:.3f} vs random "
                 f"{r['comp_random_mean']:.3f}"
                 + (f" ({r['frac_test_pos_outside_band']:.0%} of test positives outside the kept negatives' band)"
                    if pd.notna(r.get("frac_test_pos_outside_band", np.nan)) else "")
                 + f"; fixed_model composition-only {r['fixed_comp_matched']:.3f} vs {r['fixed_comp_random_mean']:.3f}.")
    for ds, g in dz.groupby("dataset", sort=False):
        L.append(f"  {ds}")
        for pair, h in g.groupby("pair", sort=False):
            a, b = h.iloc[0], h.iloc[1]
            L.append(f"    {a['family']:<8}{a['level_label']:<6} ret {a['retained_neg_frac']:6.1%} maxSMD {a['max_abs_smd']:.2f} "
                     f"comp {a['comp_auroc']:.3f} -> k-mer {a['kmer_auroc']:.3f}|{a['kmer_auroc_random']:.3f}   vs   "
                     f"F1 {b['level_label']:<6} ret {b['retained_neg_frac']:6.1%} maxSMD {b['max_abs_smd']:.2f} "
                     f"comp {b['comp_auroc']:.3f} -> k-mer {b['kmer_auroc']:.3f}|{b['kmer_auroc_random']:.3f}"
                     f"{'  UNDERPOWERED' if a['underpowered'] else ''}"
                     f"{'' if a['retention_comparable'] else '  (no F1 level within 25% retention: not a like-for-like pair)'}")
    return "\n".join(L)


def f2_selection_difficulty(summ, band=None):
    """POST HOC, DESCRIPTIVE. Per F2 (common-support trimming) level: k-mer AUROC gap vs size-matched random
    and composition-only AUROC gap vs its random control, for the retrained (within), test-only
    (fixed_model) and training-only (train_only) designs. A level is flagged (per design) when the k-mer gap
    is >= F2_KGAP_MIN while the composition-only gap is <= F2_CGAP_MAX: in that design the selection lowers
    k-mer AUROC without lowering composition-only AUROC. Fix round 3: the within-design flag (the column
    formerly called selection_induced_difficulty) is sid_within; sid_fixed_model / sid_train_only apply the same
    rule to the paired designs. F2 keeps every positive but only negatives inside the positives' central
    [q, 1-q] logit-propensity band, so ~2q of the positives lie outside the kept negatives' propensity range
    (run_posthoc.py P8, merged here when psm_sensitivity_f2_band.csv exists): a retrained composition-only model
    can then separate those positives, which is why its within-design AUROC can RISE under trimming."""
    rows = []
    s = summ[summ["family"] == "F2_trim"]
    for (ds, lev), g in s.groupby(["dataset", "level_label"], sort=False):
        def v(design, model, col):
            q = g[(g["design"] == design) & (g["model"] == model)]
            return q.iloc[0][col] if len(q) else np.nan

        def gap(design, model):
            return v(design, model, "auroc_random_mean") - v(design, model, "auroc_matched")
        up = bool(v("within", "kmer", "underpowered"))
        r = dict(dataset=ds, level_label=lev, retained_neg_frac=v("within", "kmer", "retained_neg_frac"),
                 max_abs_smd=v("within", "kmer", "max_abs_smd_matched"), underpowered=up,
                 kmer_matched=v("within", "kmer", "auroc_matched"),
                 kmer_random_mean=v("within", "kmer", "auroc_random_mean"), kmer_gap=gap("within", "kmer"),
                 comp_matched=v("within", "comp", "auroc_matched"),
                 comp_random_mean=v("within", "comp", "auroc_random_mean"), comp_gap=gap("within", "comp"))
        for design, tag in (("fixed_model", "fixed"), ("train_only", "trainonly")):
            r.update({f"{tag}_kmer_matched": v(design, "kmer", "auroc_matched"),
                      f"{tag}_kmer_random_mean": v(design, "kmer", "auroc_random_mean"),
                      f"{tag}_kmer_gap": gap(design, "kmer"),
                      f"{tag}_comp_matched": v(design, "comp", "auroc_matched"),
                      f"{tag}_comp_random_mean": v(design, "comp", "auroc_random_mean"),
                      f"{tag}_comp_gap": gap(design, "comp")})
        for design, kc, cc in (("within", "kmer_gap", "comp_gap"), ("fixed_model", "fixed_kmer_gap", "fixed_comp_gap"),
                               ("train_only", "trainonly_kmer_gap", "trainonly_comp_gap")):
            r[f"sid_{design}"] = bool((not up) and r[kc] >= F2_KGAP_MIN and r[cc] <= F2_CGAP_MAX)
        r["flag_rule"] = (f"per design: kmer gap >= {F2_KGAP_MIN} and comp gap <= {F2_CGAP_MAX} (post hoc; not "
                          f"underpowered)")
        if band is not None:
            b = band[(band["dataset"] == ds) & (band["level_label"] == lev)]
            if len(b):
                b = b.iloc[0]
                r.update(frac_test_pos_outside_band=b["frac_test_pos_outside_band"],
                         frac_train_pos_outside_band=b["frac_train_pos_outside_band"])
                for k in ("within_kmer", "within_comp", "fixed_model_kmer", "fixed_model_comp"):
                    for p_ in ("inband", "outband"):
                        for sfx in ("matched", "random_mean", "gap"):
                            r[f"{k}_{p_}_{sfx}"] = b[f"{k}_{p_}_{sfx}"]
                # rule R8 (run_posthoc.py FIX-ROUND-3 ADDENDUM)
                r["r8_comp_change_from_tails"] = bool(b["within_comp_outband_gap"] <= -0.05
                                                      and b["within_comp_inband_gap"] >= 0.05)
                r["r8_kmer_inband_loss_beyond_comp"] = bool(b["within_kmer_inband_gap"] - b["within_comp_inband_gap"]
                                                            >= 0.05)
        rows.append(r)
    return pd.DataFrame(rows)


def caliper_sweep(summ):
    """(u, fix round 3) Two-sided 1:1 caliper matching (F3) across calipers 2 -> 0.01 SD (within design): k-mer and
    composition-only AUROC, matched vs random 1:1 subsets of the same size, and the matched set's balance. This is
    the literal 'sensitivity across propensity-score matching thresholds' curve. Pre-registered rows (part 4b)."""
    s = summ[(summ["family"] == "F3_1to1") & (summ["design"] == "within")]
    rows = []
    for (ds, lev), g in s.groupby(["dataset", "level_label"], sort=False):
        k, c = g[g["model"] == "kmer"].iloc[0], g[g["model"] == "comp"].iloc[0]
        rows.append(dict(dataset=ds, caliper_sd=float(lev[1:]), level_label=lev,
                         retained_frac_each_class=k["retained_neg_frac"], max_abs_smd_matched=k["max_abs_smd_matched"],
                         max_abs_smd_random_mean=k.get("max_abs_smd_random_mean", np.nan),
                         n_test_pos=k["n_test_pos"], n_test_neg=k["n_test_neg"], underpowered=bool(k["underpowered"]),
                         kmer_matched=k["auroc_matched"], kmer_matched_lo=k["auroc_matched_lo"],
                         kmer_matched_hi=k["auroc_matched_hi"], kmer_random_mean=k["auroc_random_mean"],
                         kmer_random_min=k["auroc_random_min"], kmer_random_max=k["auroc_random_max"],
                         kmer_gap=k["auroc_random_mean"] - k["auroc_matched"],
                         comp_matched=c["auroc_matched"], comp_random_mean=c["auroc_random_mean"],
                         comp_gap=c["auroc_random_mean"] - c["auroc_matched"],
                         mcc05_matched=k["mcc_matched"], mcc05_random_mean=k["mcc_random_mean"]))
    return ordered(pd.DataFrame(rows).sort_values(["dataset", "caliper_sd"], ascending=[True, False], kind="stable"))


def bw_reading(d):
    """(v, fix round 3) Class-balanced-training size effect read against whether the unweighted analysis had a size
    component (pre-registered criterion i-a)."""
    had = not bool(d["crit_ia_size"])
    within = bool(d["bw_size_within_tau"])
    if had:
        return ("<= tau with class weights: the size component is imbalance handling" if within else
                "> tau: the size component persists with class weights")
    return ("<= tau: no size component with or without class weights" if within else
            "> tau with class weights only (no size component without them)")


def f2_lines(f2d, ds):
    """(s, t, fix round 3) F2 trimming, per design, with its construction (P8). Every number from f2d."""
    q = f2d[(f2d["dataset"] == ds) & (~f2d["underpowered"])]
    if not len(q):
        return "  [post hoc] common-support trimming (F2): every level underpowered"
    L = []
    has8 = "frac_test_pos_outside_band" in q.columns and q["frac_test_pos_outside_band"].notna().all()

    def rng(col, sub, fmt="+.3f"):
        a, b = sub[col].min(), sub[col].max()
        return f"{a:{fmt}}" if len(sub) == 1 or a == b else f"{a:{fmt}} to {b:{fmt}}"
    fl = q[q["sid_within"]]
    sel = fl if len(fl) else q.loc[[q["retained_neg_frac"].idxmin()]]
    lv = ", ".join(sel["level_label"])
    kept = f"{rng('retained_neg_frac', sel, '.0%')} of negatives kept"
    head = (f"  [post hoc] common-support trimming (F2) at {lv} ({kept}"
            + (f"; {rng('frac_test_pos_outside_band', sel, '.0%')} of test positives lie OUTSIDE the kept negatives' "
               f"propensity band, P8" if has8 else "") + "). ")
    head += (f"RETRAINED models (within): k-mer gap vs size-matched random {rng('kmer_gap', sel)}, composition-only gap "
             f"{rng('comp_gap', sel)}"
             + (f" -> flagged at {len(fl)} of {len(q)} levels (rule: {q.iloc[0]['flag_rule']})" if len(fl) else
                f" -> no level flagged in this design (rule: {q.iloc[0]['flag_rule']})")
             + f". ORIGINAL model on the trimmed test (fixed_model): k-mer {rng('fixed_kmer_gap', sel)}, composition-only "
             f"{rng('fixed_comp_gap', sel)} (flagged at {int(q['sid_fixed_model'].sum())} of {len(q)} levels). TRIMMED-split "
             f"model on the ORIGINAL test (train_only): k-mer {rng('trainonly_kmer_gap', sel)}, composition-only "
             f"{rng('trainonly_comp_gap', sel)} (flagged at {int(q['sid_train_only'].sum())} of {len(q)} levels).")
    L.append(head)
    if has8 and "within_comp_inband_gap" in sel.columns:
        L.append("    P8 split of the within design, gap (random mean - trimmed) for IN-band | OUT-of-band test positives vs the "
                 "arm's negatives: " + "; ".join(
                     f"{r['level_label']} k-mer {r['within_kmer_inband_gap']:+.3f} | {r['within_kmer_outband_gap']:+.3f}, "
                     f"composition-only {r['within_comp_inband_gap']:+.3f} | {r['within_comp_outband_gap']:+.3f} -> R8: "
                     f"{'composition change from the tails' if r['r8_comp_change_from_tails'] else 'not attributed to the tails'}"
                     f"; {'k-mer loses in-band signal beyond composition-only' if r['r8_kmer_inband_loss_beyond_comp'] else 'k-mer in-band loss not beyond composition-only'}"
                     for _, r in sel.iterrows()) + " (R8 rule in the run_posthoc.py FIX-ROUND-3 ADDENDUM; point estimates)")
    if len(fl):
        nf = q[~q["sid_within"]]
        if len(nf):
            L.append(f"    at the other levels ({', '.join(nf['level_label'])}; {rng('retained_neg_frac', nf, '.0%')} "
                     f"kept) the retrained composition-only AUROC also falls vs random "
                     f"(gap {rng('comp_gap', nf)}; k-mer {rng('kmer_gap', nf)})")
    return "\n".join(L)


def headline(dec, pt, sb, dst=None, bt=None, lk=None, f2d=None, sbf2=None, cal=None):
    """Short, data-driven headline block for the top of the interpretation file. Every number is read
    from the frames written by this script (decision / paper_table / selection_bias / dissociation_stats /
    f2_selection_difficulty) or by run_posthoc.py (between_dataset / leakage_check). Every qualitative
    phrase is selected by the numeric rule printed next to it."""
    L = []
    add = L.append
    add("HEADLINE (numbers from psm_sensitivity_{decision,paper_table,selection_bias,dissociation_stats,"
        "f2_selection_difficulty,between_dataset,leakage_check}.csv; [post hoc] = not pre-registered)")
    add("-" * 100)
    if lk is not None and len(lk):
        for _, r in lk.iterrows():
            if r["n_test_verbatim_in_train"] > 0:
                add(f"LEAKAGE: {r['dataset']}: {int(r['n_test_verbatim_in_train'])} of {int(r['n_test'])} test sequences occur "
                    f"verbatim in train ({int(r['n_test_verbatim_pos'])} positives, labels consistent: "
                    f"{bool(r['dup_labels_consistent'])}); original model mean score {r['orig_mean_score_dup_pos']:.3f} on them vs "
                    f"{r['orig_mean_score_other_pos']:.3f} on the other positives; original AUROC {r['orig_auroc_full_test']:.3f} -> "
                    f"{r['orig_auroc_leakfree_test']:.3f} without them. Rows {LF_TEST} / {LF_REFIT} below are the leak-free "
                    f"re-analyses ({LF_TEST} is the primary nt result).")
        clean = [r["dataset"] for _, r in lk.iterrows() if r["n_test_verbatim_in_train"] == 0]
        if clean:
            add(f"LEAKAGE: no verbatim train/test duplicates in {', '.join(clean)}.")
        add("")
    for _, d in dec.iterrows():
        ds = d["dataset"]
        t = pt[pt["dataset"] == ds].set_index("condition")

        def v(cond, k="auroc"):
            return t.loc[cond, k] if cond in t.index else np.nan
        tag = f"  [{POSTHOC_DS[ds]}]" if ds in POSTHOC_DS else ""
        add(f"{ds} [{d['role']}]  decision rule verdict: {d['verdict']}  "
            f"(i-a size {d['crit_ia_size']}, i-b separation {d['crit_ib_separation']}, "
            f"ii-a weighting {d['crit_iia_weighting']}, ii-b two-sided {d['crit_iib_twosided']}; tau {d['tau']:.3f}){tag}")
        add(f"  k-mer AUROC {d['A0']:.3f} -> random negatives at the paper's retention ({d['retained_neg_frac']:.1%}) "
            f"{d['Ar_mean']:.3f} ({d['Ar_min']:.3f}-{d['Ar_max']:.3f}) -> composition-matched {d['Am']:.3f} "
            f"[{d['Am_lo']:.3f},{d['Am_hi']:.3f}] (test {int(d['f1_n_test_pos'])} pos / {int(d['f1_n_test_neg'])} neg): "
            f"size effect {d['drop_size']:+.3f} ({'<=' if d['crit_ia_size'] else '>'} tau), matched-vs-random effect "
            f"{d['drop_shortcut']:+.3f} (share of total drop {d['shortcut_share']:.2f}; the pre-registered 'shortcut' term: "
            f"it measures what composition MATCHING removes, whatever its cause)")
        if pd.notna(d.get("p7_matched_vs_random", np.nan)):
            def ci3(p):
                return f"{d[p]:+.3f} [{d[p + '_lo']:+.3f},{d[p + '_hi']:+.3f}]"
            add(f"  [post hoc P7] two-level paired bootstrap (test sequences x random seeds): size {ci3('p7_size_effect')} "
                f"({d['p7_size_vs_tau']} {d['p7_tau_R7b']:.3f} (R7b's fixed tau; the pre-registered tau here is "
                f"{d['tau']:.3f}); share of the total drop {d['p7_size_share_of_total']:.2f} "
                f"[{d['p7_size_share_of_total_lo']:.2f},{d['p7_size_share_of_total_hi']:.2f}]); matched-vs-random "
                f"{ci3('p7_matched_vs_random')} -> "
                f"{'non-zero' if d['p7_mvr_ci_excludes_zero'] else 'not distinguishable from 0'} (R7a); its share of the total "
                f"drop {d['p7_mvr_share_of_total']:.2f} [{d['p7_mvr_share_of_total_lo']:.2f},{d['p7_mvr_share_of_total_hi']:.2f}]")
            add(f"  [post hoc P7] composition-only model on the same arms: matched-vs-random {ci3('p7_comp_matched_vs_random')} "
                f"({'non-zero' if d['p7_comp_mvr_ci_excludes_zero'] else 'not distinguishable from 0'}); k-mer minus "
                f"composition-only: absolute {ci3('p7_kmc_matched_vs_random')}, relative (share of the random arm's "
                f"above-chance AUROC) {ci3('p7_kmc_rel_gap')} -> {d['p7_kmc_reading']} (R7c). Over matched seeds 42-44: "
                f"k-mer {ci3('p7_s4244_matched_vs_random')}, composition-only {ci3('p7_s4244_comp_matched_vs_random')} -> "
                f"{d['p7_s4244_kmc_reading']}")
            if pd.notna(d.get("p7_fixed_matched_vs_random", np.nan)):
                add(f"  [post hoc P7] PAIRED designs, matched-vs-random effect, k-mer | composition-only | k-mer minus "
                    f"composition-only (absolute; relative) -> R7c: original model on the selected test (fixed_model) "
                    f"{ci3('p7_fixed_matched_vs_random')} | {ci3('p7_fixed_comp_matched_vs_random')} | "
                    f"{ci3('p7_fixed_kmc_matched_vs_random')}; {ci3('p7_fixed_kmc_rel_gap')} -> {d['p7_fixed_kmc_reading']}; "
                    f"selected-split model on the ORIGINAL test (train_only) {ci3('p7_trainonly_matched_vs_random')} | "
                    f"{ci3('p7_trainonly_comp_matched_vs_random')} | {ci3('p7_trainonly_kmc_matched_vs_random')}; "
                    f"{ci3('p7_trainonly_kmc_rel_gap')} -> {d['p7_trainonly_kmc_reading']}")
        tr_c = "matched-trained model -> ORIGINAL test"
        add(f"  training negatives changed, test undistorted (arm-trained models scored on the ORIGINAL test): random-trained "
            f"{d['trainonly_A_randomTrain_mean']:.3f} ({d['trainonly_A_randomTrain_min']:.3f}-{d['trainonly_A_randomTrain_max']:.3f}) "
            f"vs matched-trained {d['trainonly_A_matchedTrain']:.3f} [{v(tr_c, 'auroc_lo'):.3f},{v(tr_c, 'auroc_hi'):.3f}]"
            + (f"; [post hoc P7] matched-vs-random {d['p7_trainonly_matched_vs_random']:+.3f} "
               f"[{d['p7_trainonly_matched_vs_random_lo']:+.3f},{d['p7_trainonly_matched_vs_random_hi']:+.3f}], training-size "
               f"effect {d['p7_trainonly_size_effect']:+.3f} [{d['p7_trainonly_size_effect_lo']:+.3f},"
               f"{d['p7_trainonly_size_effect_hi']:+.3f}]" if pd.notna(d.get("p7_trainonly_matched_vs_random", np.nan)) else ""))
        add(f"  matched AUROC over selection seeds 42/43/44: {d['Am_seedmin']:.3f}-{d['Am_seedmax']:.3f}; published split "
            f"{d['paper_anchor_auroc']:.3f}. Balance of the matched set: max|SMD| {d['f1_max_abs_smd']:.3f} "
            f"({'balanced, <=0.1' if d['f1_max_abs_smd'] <= 0.1 else 'NOT balanced, >0.1'}); published split max|SMD| "
            f"{d.get('paper_anchor_max_abs_smd', np.nan):.3f}; composition-only AUROC {d['comp0']:.3f} -> matched "
            f"{d['comp_m']:.3f} (random {d['comp_r_mean']:.3f})")
        add(f"  overlap weighting, nothing discarded (neg ESS {d['overlap_ess_neg_frac']:.0%}): {d['Aw_overlap']:.3f} "
            f"[{d['Aw_overlap_lo']:.3f},{d['Aw_overlap_hi']:.3f}] vs permuted-weight placebo {d['Apw_placebo_mean']:.3f} "
            f"({d['drop_overlap']:+.3f})")
        add(f"  two-sided 1:1 caliper-0.2 matching (keeps {d['twosided_retained_frac']:.0%} of EACH class; discards the "
            f"rest): {d['A2_1to1']:.3f} [{d['A2_1to1_lo']:.3f},{d['A2_1to1_hi']:.3f}] vs random 1:1 of the same size "
            f"{d['A2r_mean']:.3f} ({d['drop_twosided']:+.3f})")
        add(f"  fixed original model (no retraining): class-ratio change alone {d['test_distortion_effect']:+.3f} (AUROC is "
            f"prevalence-invariant, so ~0 is expected); composition-matched test negatives instead of random ones "
            f"{d['test_shortcut_effect']:+.3f} ({d['fixed_A_matchedTest']:.3f} vs {d['fixed_A_randomTest_mean']:.3f})")
        w1, w2 = "fixed original model -> overlap-weighted ORIGINAL test", "fixed original model -> placebo-weighted ORIGINAL test"
        if w1 in t.index:
            add(f"  nothing removed AND nothing retrained: original model on the overlap-weighted original test "
                f"{v(w1):.3f} [{v(w1, 'auroc_lo'):.3f},{v(w1, 'auroc_hi'):.3f}] vs placebo weights {v(w2):.3f} "
                f"({v(w2, 'auroc_min'):.3f}-{v(w2, 'auroc_max'):.3f}); drop {v(w2) - v(w1):+.3f}"
                + (f"; share of above-chance AUROC lost {d['fixedoverlap_share_lost']:.2f} "
                   f"[{d['fixedoverlap_share_lo']:.2f},{d['fixedoverlap_share_hi']:.2f}]"
                   if pd.notna(d.get("fixedoverlap_share_lost", np.nan)) else ""))
        if pd.notna(d.get("fixedoverlap_comp_auroc", np.nan)):
            below = [nm for nm, col in (("retrained", "overlap_comp_auroc"), ("original-model", "fixedoverlap_comp_auroc"))
                     if pd.notna(d.get(col, np.nan)) and d[col] < 0.5]
            above = [nm for nm, col in (("retrained", "overlap_comp_auroc"), ("original-model", "fixedoverlap_comp_auroc"))
                     if pd.notna(d.get(col, np.nan)) and d[col] >= 0.5]
            add(f"  weighting diagnostic: composition-only AUROC under the same overlap weights {d['overlap_comp_auroc']:.3f} "
                f"(retrained) / {d['fixedoverlap_comp_auroc']:.3f} (original model) vs placebo {d['overlap_comp_placebo']:.3f} / "
                f"{d['fixedoverlap_comp_placebo']:.3f} -> "
                + (f"weights OVER-correct in the {' and '.join(below)} design{'s' if len(below) > 1 else ''} "
                   f"(composition-only < 0.5 there): the weighted k-mer drop there overstates the effect of removing "
                   f"composition and only its direction is interpretable" if below else "")
                + ("; " if below and above else "")
                + (f"weights leave residual composition signal in the {' and '.join(above)} design{'s' if len(above) > 1 else ''} "
                   f"(composition-only > 0.5): the weighted drop there is conservative" if above else "")
                + " (rule, per design: composition-only weighted AUROC < 0.5)")
        c1, c2 = "two-sided 1:1 caliper 0.2 SD", "random 1:1 subsets, same n (10 seeds)"
        mcc_mostly = ("mostly size/imbalance" if d["mcc_drop_size_imbalance"] > d["mcc_drop_shortcut"]
                      else "mostly matched-vs-random")
        bal_mostly = ("mostly size/imbalance" if d["balmcc_drop_size"] > d["balmcc_drop_shortcut"]
                      else "mostly matched-vs-random")
        add(f"  MCC@0.5 (paper metric) {d['mcc0']:.3f} -> size-matched random {d['mcc_r_mean']:.3f} -> matched {d['mcc_m']:.3f}: "
            f"size/imbalance part {d['mcc_drop_size_imbalance']:+.3f}, matched-vs-random part {d['mcc_drop_shortcut']:+.3f} -> "
            f"{mcc_mostly} (rule: larger part)")
        add(f"  balanced MCC (class-balance-weighted, prior-corrected threshold) {d['balmcc0']:.3f} -> random "
            f"{d['balmcc_r_mean']:.3f} -> matched {d['balmcc_m']:.3f}: size part {d['balmcc_drop_size']:+.3f}, matched-vs-random part "
            f"{d['balmcc_drop_shortcut']:+.3f} -> {bal_mostly}; 1:1 design MCC@0.5 matched {v(c1, 'mcc05'):.3f} vs random 1:1 "
            f"{v(c2, 'mcc05'):.3f} (a matched-vs-random contrast, not before/after)")
        if pd.notna(d.get("comp_mcc0", np.nan)):
            def rep(s):
                if not np.isfinite(s):
                    return "n/a (no collapse on the published split)"
                return f"{s:.0%} of the collapse -> " + ("fully" if s >= 1 else "mostly" if s >= 0.5 else
                                                         "partly" if s > 0 else "not reproduced")
            sk, sc_ = d["mcc_collapse_share_reproduced_by_random"], d["comp_mcc_collapse_share_reproduced_by_random"]
            sk = sk if (d["mcc0"] - d["paper_anchor_mcc"]) > 0 else np.nan
            sc_ = sc_ if (d["comp_mcc0"] - d["comp_paper_anchor_mcc"]) > 0 else np.nan
            add(f"  published-split MCC@0.5 collapse vs size-matched RANDOM negatives: k-mer {d['mcc0']:.3f} -> random "
                f"{d['mcc_r_mean']:.3f} ({d['mcc_r_min']:.3f}-{d['mcc_r_max']:.3f}) vs published {d['paper_anchor_mcc']:.3f} "
                f"({rep(sk)}); composition-only {d['comp_mcc0']:.3f} -> random {d['comp_mcc_r_mean']:.3f} "
                f"({d['comp_mcc_r_min']:.3f}-{d['comp_mcc_r_max']:.3f}) vs published {d['comp_paper_anchor_mcc']:.3f} "
                f"({rep(sc_)}) (rule: share (mcc0 - random) / (mcc0 - published) >= 1 fully, >= 0.5 mostly, > 0 partly, "
                f"else not reproduced)")
        if pd.notna(d.get("bw_Am", np.nan)):
            add(f"  [post hoc] class-balanced training weights: random {d['bw_Ar_mean']:.3f} ({d['bw_Ar_min']:.3f}-"
                f"{d['bw_Ar_max']:.3f}) vs matched {d['bw_Am']:.3f} [{d['bw_Am_lo']:.3f},{d['bw_Am_hi']:.3f}] (seeds "
                f"{d['bw_Am_seedmin']:.3f}-{d['bw_Am_seedmax']:.3f}): size effect {d['bw_drop_size']:+.3f} "
                f"({bw_reading(d)}), "
                f"matched-vs-random effect {d['bw_drop_shortcut']:+.3f}; MCC@0.5 random {d['bw_mcc05_r_mean']:.3f} vs matched "
                f"{d['bw_mcc05_m']:.3f} (MCC@0.5 is not informative on these rows either: caveat 11)")
        if pd.notna(d.get("rho_hidden_pos", np.nan)):
            ratio = d["labelnoise_bound_over_shortcut_drop"]
            verdict = ("label noise cannot explain the matched-vs-random effect" if d["drop_shortcut"] > 0 and ratio < 0.5 else
                       "label noise is not excluded as an explanation" if d["drop_shortcut"] > 0 else
                       "matched-vs-random effect <= 0; ratio undefined")
            pb = d.get("els_p_bonferroni", np.nan)
            qual = []
            if pd.notna(pb) and pb >= 0.05:
                qual.append(f"the ELS enrichment of kept over discarded negatives is not significant (Bonferroni p "
                            f"{pb:.2g}), so the hidden-positive fraction is not distinguishable from 0")
            if d["verdict"] == "NOT SUPPORTED":
                qual.append("the composition-matching effect itself is NOT SUPPORTED for this variant")
            lead = ("  label noise: UNINFORMATIVE here, because " + " and ".join(qual)
                    + " (rules: Bonferroni p >= 0.05; verdict NOT SUPPORTED); label noise is neither shown nor excluded. "
                    "Details: " if qual else "  label noise: ")
            add(lead + f"mixture-model ESTIMATE (assumes hidden positives overlap ENCODE ELS at the labelled-"
                f"positive rate): ELS overlap retained {d['els_rate_retained']:.3f} vs discarded "
                f"{d.get('els_rate_discarded', np.nan):.3f} (Bonferroni p {pb:.2g}; all negatives {d['els_rate_neg_all']:.3f}, "
                f"positives {d['els_rate_pos']:.3f}) -> hidden-positive fraction {d['rho_hidden_pos']:.3f}, AUROC cost "
                f"{d['labelnoise_max_auroc_drop']:.3f} = {ratio:.2f}x the matched-vs-random effect -> "
                + (f"the ratio rule alone would read '{verdict}'" if qual else verdict) + " (rule: ratio < 0.5)")
        if pd.notna(d.get("f5b_Am", np.nan)):
            add(f"  label-noise check, cCRE-purged pools: F5 re-match {d['f5_Am']:.3f} vs random {d['f5_Ar_mean']:.3f}; "
                f"F5b 1:1 {d['f5b_Am']:.3f} vs {d['f5b_Ar_mean']:.3f} (drop {d['f5b_drop']:+.3f} vs {d['drop_twosided']:+.3f} "
                f"unpurged)")
        if sb is not None:
            q = sb[(sb["dataset"] == ds) & (sb["selection"] == "F1_hist_a1_seed42")]
            if len(q):
                big = q.reindex(q["smd_retained_vs_discarded"].abs().sort_values(ascending=False).index).head(4)
                add("  retained vs discarded negatives, largest out-of-model SMDs: " + "; ".join(
                    f"{r['feature']} {r['smd_retained_vs_discarded']:+.2f} (ret {r['mean_retained']:.3g}, disc "
                    f"{r['mean_discarded']:.3g}, pos {r['mean_pos']:.3g}; p_bonf {r['p_bonferroni']:.2g})"
                    for _, r in big.iterrows()))
                ns = q[(q["p_bonferroni"] >= 0.05) & q["feature"].isin(["tss500_overlap", "ccre_els", "ccre_any", "tata_any"])]
                if len(ns):
                    add("  kept-vs-discarded differences NOT significant after Bonferroni (p_bonf >= 0.05): " + "; ".join(
                        f"{r['feature']} {r['mean_retained']:.3f} vs {r['mean_discarded']:.3f} (p_bonf {r['p_bonferroni']:.2g})"
                        for _, r in ns.iterrows()))
        if dst is not None and (dst["dataset"] == ds).any():
            parts_ = []
            for design in ("within", "fixed_model", "train_only"):
                q = dst_row(dst, ds, design)
                if q is not None:
                    parts_.append(f"{design} (n={int(q['n_arms'])}) {q['spearman_kmer_gap_vs_comp_gap']:+.2f} vs "
                                  f"{q['spearman_kmer_gap_vs_neglog_retention']:+.2f} -> {q['reading']}")
            add("  [post hoc, descriptive] across unique F1/F2/F3 arms, Spearman of the k-mer AUROC gap with the "
                "composition-only gap vs with selection depth (-log retained fraction), per design: " + "; ".join(parts_)
                + f" (rule: {DISSOC_MARGIN} difference)")
        if f2d is not None and (f2d["dataset"] == ds).any():
            add(f2_lines(f2d, ds))
        if cal is not None and (cal["dataset"] == ds).any():
            c_ = cal[cal["dataset"] == ds]
            add("  two-sided 1:1 caliper sweep (F3, pre-registered part 4b; within design; k-mer AUROC matched vs random 1:1 of the "
                "same size; max|SMD| of the matched set; % of each class kept): " + "; ".join(
                    f"{r['level_label']} {r['kmer_matched']:.3f} vs {r['kmer_random_mean']:.3f} (gap {r['kmer_gap']:+.3f}, "
                    f"SMD {r['max_abs_smd_matched']:.2f}, {r['retained_frac_each_class']:.0%})" for _, r in c_.iterrows()))
        if pd.notna(d.get("f4_Am", np.nan)):
            add(f"  positive-side matching (F4; ALL negatives kept, positives subsampled to the negatives' propensity histogram, "
                f"{d['f4_retained_pos_frac']:.1%} of positives kept, test {int(d['f4_n_test_pos'])} pos / "
                f"{int(d['f4_n_test_neg'])} neg{', UNDERPOWERED' if d['f4_underpowered'] else ''}): k-mer {d['f4_Am']:.3f} "
                f"[{d['f4_Am_lo']:.3f},{d['f4_Am_hi']:.3f}] vs random positive subsets {d['f4_Ar_mean']:.3f} "
                f"({d['f4_Ar_min']:.3f}-{d['f4_Ar_max']:.3f}), gap {d['f4_mvr']:+.3f}; composition-only {d['f4_comp_Am']:.3f} vs "
                f"{d['f4_comp_Ar_mean']:.3f}")
        add("")
    if bt is not None and len(bt):
        add("[post hoc] BETWEEN-DATASET CONTRAST (psm_sensitivity_between_dataset.csv): share of above-chance AUROC lost "
            "under overlap weighting, S = (placebo - overlap) / (placebo - 0.5), 1000 shared bootstrap resamples")
        has6 = "comp_share_lost" in bt.columns and bt["comp_share_lost"].notna().any()
        if has6:
            add("  per dataset: k-mer S | composition-only S under the SAME weights (= how much composition signal the weights "
                "remove) | k-mer S / composition-only S (run_posthoc.py P6)")
            for _, r in bt[bt["kind"] == "dataset"].iterrows():
                add(f"    {r['design']:<20} {r['dataset']:<28} k-mer {r['share_above_chance_lost']:.3f} [{r['share_lo']:.3f},"
                    f"{r['share_hi']:.3f}] | comp-only {r['comp_share_lost']:.3f} [{r['comp_share_lo']:.3f},"
                    f"{r['comp_share_hi']:.3f}] | ratio {r['kmer_over_comp_share']:.2f} [{r['kmer_over_comp_share_lo']:.2f},"
                    f"{r['kmer_over_comp_share_hi']:.2f}]{'' if r['share_interpretable'] else '  NOT INTERPRETABLE (weighted AUROC < 0.5)'}")
        for _, r in bt[bt["kind"] == "contrast"].iterrows():
            add(f"  {r['design']:<20} {r['contrast']:<55} k-mer share diff {r['share_above_chance_lost']:+.3f} "
                f"[{r['share_lo']:+.3f},{r['share_hi']:+.3f}]  drop diff {r['drop']:+.3f} [{r['drop_lo']:+.3f},"
                f"{r['drop_hi']:+.3f}]  -> {'CI excludes 0' if r['ci_excludes_zero'] else 'CI includes 0'}")
            if has6 and pd.notna(r.get("comp_share_diff", np.nan)):
                add(f"  {'':<20} {'':<55} comp-only share diff {r['comp_share_diff']:+.3f} [{r['comp_share_diff_lo']:+.3f},"
                    f"{r['comp_share_diff_hi']:+.3f}]; difference of contrasts {r['diff_of_contrasts']:+.3f} "
                    f"[{r['diff_of_contrasts_lo']:+.3f},{r['diff_of_contrasts_hi']:+.3f}]; ratio difference "
                    f"{r['ratio_diff']:+.3f} [{r['ratio_diff_lo']:+.3f},{r['ratio_diff_hi']:+.3f}]"
                    f"{'' if r['share_interpretable'] else '  NOT INTERPRETABLE (a weighted AUROC < 0.5)'}")
        if has6:
            pr = bt[(bt["kind"] == "contrast") & (bt["contrast"] == "human_enhancers_cohn - drosophila_enhancers_stark")]
            for _, r in pr.iterrows():
                add_ = r["diff_of_contrasts_lo"] > 0          # additive scale, cohn loses more per unit
                mul_ = r["ratio_diff_lo"] > 0                 # multiplicative scale
                dsr = bt[bt["kind"] == "dataset"].set_index(["dataset", "design"])
                a_ = dsr.loc[("human_enhancers_cohn", r["design"])]
                b_ = dsr.loc[("drosophila_enhancers_stark", r["design"])]
                prim = "PRIMARY" if r["design"] == "fixed_model_overlap" else "secondary"
                nums = (f"the weights remove {a_['comp_share_lost']:.0%} of cohn's vs {b_['comp_share_lost']:.0%} of the "
                        f"control's above-chance composition-only AUROC (contrast {r['comp_share_diff']:+.3f}); difference of "
                        f"contrasts {r['diff_of_contrasts']:+.3f} [{r['diff_of_contrasts_lo']:+.3f},{r['diff_of_contrasts_hi']:+.3f}]; "
                        f"k-mer/comp ratio {a_['kmer_over_comp_share']:.2f} vs {b_['kmer_over_comp_share']:.2f}, difference "
                        f"{r['ratio_diff']:+.3f} [{r['ratio_diff_lo']:+.3f},{r['ratio_diff_hi']:+.3f}]")
                if add_ and mul_:
                    txt_ = "MET: cohn loses more k-mer AUROC per unit of composition signal removed on both scales"
                elif add_ or mul_:
                    oth = ("ratio difference", "ratio_diff") if add_ else ("difference of contrasts", "diff_of_contrasts")
                    txt_ = (f"NOT met (scale-dependent): per unit of composition signal removed cohn loses more on the "
                            f"{'additive' if add_ else 'ratio'} scale only; on the other scale the {oth[0]} is "
                            f"{r[oth[1]]:+.3f} [{r[oth[1] + '_lo']:+.3f},{r[oth[1] + '_hi']:+.3f}] "
                            f"({'CI includes 0' if r[oth[1] + '_hi'] >= 0 else 'CI below 0'}). Most of "
                            f"the k-mer contrast reflects cohn's larger removable composition signal; evidence of greater "
                            f"per-unit reliance is weak")
                else:
                    txt_ = ("NOT met: the k-mer contrast is accounted for by how much composition-only signal the same "
                            "weights remove, on both scales; it reflects cohn's larger removable composition signal, not "
                            "a greater reliance per unit of composition")
                add(f"  READING cohn - drosophila, {r['design']} ({prim}; rule fixed in the run_posthoc.py P6 docstring, "
                    f"post hoc: both CIs must exclude 0, positive): {txt_} ({nums}). Either way this contrast does not "
                    f"separate artifact from biology.")
        bad = bt[(bt["kind"] == "dataset") & (bt["share_above_chance_lost"] > 1)]
        if len(bad):
            add("  NOTE: share > 1 (weighted AUROC below 0.5) for " + ", ".join(
                f"{r['dataset']}/{r['design']} ({r['share_above_chance_lost']:.2f})" for _, r in bad.iterrows())
                + ": the overlap weights over-correct there (see the weighting diagnostic), so contrasts involving these "
                "rows are not interpretable as shares; the cohn - drosophila contrast is the interpretable one.")
    add("-" * 100)
    return "\n".join(L)


def paired_reading(k, dst, ds):
    """(fix round 3, post hoc, descriptive) Is a dataset's matched-vs-random k-mer loss consistent with removing
    composition-associated signal in the PAIRED designs? Rule: in both fixed_model and train_only the P7 R7c reading
    is not 'k-mer loses MORE', and across arms Spearman(k-mer gap, composition-only gap) >= 0.9 in both."""
    rs = [dst_row(dst, ds, x) for x in ("fixed_model", "train_only")]
    if any(r is None for r in rs) or pd.isna(k.get("p7_fixed_kmc_reading", np.nan)):
        return None
    more = any("MORE" in str(k[c]) for c in ("p7_fixed_kmc_reading", "p7_trainonly_kmc_reading"))
    co = all(r["spearman_kmer_gap_vs_comp_gap"] >= 0.9 for r in rs)
    return bool((not more) and co)


def scope_block(dec, pt, bt=None, dst=None, f2d=None, cal=None):
    """What the experiment does and does not show, with every number read from the frames (cohn vs control).
    Fix round 2: the size component is stated with its share and CI, matching is said to remove MOST (not all)
    composition-associated signal, and the control's matched-vs-random drop is not called a composition effect.
    Fix round 3: F2 is no longer used as evidence about retention (its construction also creates composition
    signal; P8), every dissociation reading is given per design, the control's loss is read in the paired designs
    by an explicit rule (paired_reading), and positive-side matching and the caliper sweep are added to (A)."""
    L = []
    add = L.append
    dd = dec.set_index("dataset")
    if "human_enhancers_cohn" not in dd.index or "drosophila_enhancers_stark" not in dd.index:
        return ""
    c, k = dd.loc["human_enhancers_cohn"], dd.loc["drosophila_enhancers_stark"]
    has7 = pd.notna(c.get("p7_matched_vs_random", np.nan))

    def ci(r_, p):
        return f"[{r_[p + '_lo']:+.3f},{r_[p + '_hi']:+.3f}]" if has7 else ""

    def v3(r_, p):
        return f"{r_[p]:+.3f} {ci(r_, p)}"
    add("WHAT THIS EXPERIMENT SHOWS AND DOES NOT SHOW (numbers from decision / decomposition_ci / between_dataset /")
    add("dissociation_stats / f2_selection_difficulty / f2_band / caliper_sweep; the qualitative split follows the rules printed)")
    add(f"  (A) Most of cohn's drop comes from composition matching; a smaller part comes from sample loss; changing the")
    add(f"      test's class ratio contributes nothing (as expected: AUROC does not depend on prevalence). Total drop "
        f"{c['drop_total']:+.3f} = size {c['drop_size']:+.3f} ({1 - c['shortcut_share']:.0%}"
        + (f" [{c['p7_size_share_of_total_lo']:.0%},{c['p7_size_share_of_total_hi']:.0%}]; {ci(c, 'p7_size_effect')} "
           f"{c['p7_size_vs_tau']} {c['p7_tau_R7b']:.3f}" if has7 else "")
        + f"; it is why the pre-registered verdict is {c['verdict']}; with class-balanced training {c['bw_drop_size']:+.3f}) "
        f"+ matched-vs-random {c['drop_shortcut']:+.3f} ({c['shortcut_share']:.0%}; {ci(c, 'p7_matched_vs_random')}).")
    add(f"      Designs without the reduced, distorted test: class-ratio change alone {c['test_distortion_effect']:+.3f}; same "
        f"undistorted original test with only the training negatives changed "
        f"{c['trainonly_A_randomTrain_mean'] - c['trainonly_A_matchedTrain']:+.3f} {ci(c, 'p7_trainonly_matched_vs_random')}; "
        f"nothing removed and nothing retrained (original model, overlap-weighted test) "
        f"{c['fixedoverlap_placebo'] - c['fixedoverlap_A']:+.3f}; overlap-weighted retraining {c['drop_overlap']:+.3f}; "
        f"balanced 1:1 matching {c['drop_twosided']:+.3f}"
        + (f"; positive-side matching, which keeps EVERY negative, {c['f4_mvr']:+.3f} ({c['f4_Am']:.3f} vs random positive "
           f"subsets {c['f4_Ar_mean']:.3f})" if pd.notna(c.get("f4_mvr", np.nan)) else "") + ".")
    if cal is not None and (cal["dataset"] == "human_enhancers_cohn").any():
        q = cal[(cal["dataset"] == "human_enhancers_cohn") & (~cal["underpowered"])]
        add(f"      Across calipers {q['caliper_sd'].max():g} -> {q['caliper_sd'].min():g} SD (two-sided 1:1) random 1:1 subsets "
            f"stay at {q['kmer_random_mean'].min():.3f}-{q['kmer_random_mean'].max():.3f} while the matched gap goes "
            + " -> ".join(f"{r['kmer_gap']:+.3f} (max|SMD| {r['max_abs_smd_matched']:.2f})" for _, r in q.iterrows()) + ".")
    if dst is not None:
        rs = [(x, dst_row(dst, "human_enhancers_cohn", x)) for x in ("within", "fixed_model", "train_only")]
        rs = [(x, r) for x, r in rs if r is not None]
        if rs:
            def short(r):
                return ("composition" if r["reading"].startswith("gap tracks composition") else
                        "selection depth" if r["reading"].startswith("gap tracks selection") else "no clear separation")
            larger = all(r["spearman_kmer_gap_vs_comp_gap"] > r["spearman_kmer_gap_vs_neglog_retention"] for _, r in rs)
            add("      Across arms (each once) the k-mer gap's Spearman with the composition-only gap vs with selection depth: "
                + "; ".join(f"{x} {r['spearman_kmer_gap_vs_comp_gap']:+.2f} vs {r['spearman_kmer_gap_vs_neglog_retention']:+.2f}"
                            f" ({short(r)})" for x, r in rs)
                + f" (rule: difference >= {DISSOC_MARGIN})"
                + ("; the composition-only correlation is the larger in every design." if larger else "."))
    if has7:
        add(f"      At the paper's retention the k-mer loss does not exceed the composition-only loss: within "
            f"{c['p7_matched_vs_random']:+.3f} vs {c['p7_comp_matched_vs_random']:+.3f} (difference "
            f"{v3(c, 'p7_kmc_matched_vs_random')} absolute, {v3(c, 'p7_kmc_rel_gap')} relative -> {c['p7_kmc_reading']})"
            + (f"; original model on the matched test {c['p7_fixed_matched_vs_random']:+.3f} vs "
               f"{c['p7_fixed_comp_matched_vs_random']:+.3f} -> {c['p7_fixed_kmc_reading']}"
               if pd.notna(c.get("p7_fixed_matched_vs_random", np.nan)) else "")
            + f"; original test {c['p7_trainonly_matched_vs_random']:+.3f} vs {c['p7_trainonly_comp_matched_vs_random']:+.3f} "
            f"-> {c['p7_trainonly_kmc_reading']}.")
        lo = dec[dec["p7_trainonly_kmc_reading"].astype(str).str.contains("LESS")]["dataset"].tolist()
        if len(lo) == len(dec):
            add("      (In the train_only design the k-mer model loses less than the composition-only model in EVERY dataset "
                "variant, so that reading is a property of the design, not evidence about any one dataset.)")
    if f2d is not None and (f2d["dataset"] == "human_enhancers_cohn").any():
        q = f2d[(f2d["dataset"] == "human_enhancers_cohn") & (~f2d["underpowered"])]
        r = q.loc[q["retained_neg_frac"].idxmin()]
        add(f"      Common-support trimming (F2) is NOT used as evidence about retention: it keeps every positive but only "
            f"negatives inside the positives' central propensity band"
            + (f" ({r['frac_test_pos_outside_band']:.0%} of cohn test positives outside it at {r['level_label']})"
               if pd.notna(r.get("frac_test_pos_outside_band", np.nan)) else "")
            + f", so the retrained composition-only AUROC RISES ({r['comp_matched']:.3f} vs random {r['comp_random_mean']:.3f}) "
            f"while the retrained k-mer AUROC stays level ({r['kmer_matched']:.3f} vs {r['kmer_random_mean']:.3f}); with the "
            f"original model or the original test held fixed the same trimming costs k-mer {r['fixed_kmer_gap']:+.3f} / "
            f"{r['trainonly_kmer_gap']:+.3f} and composition-only {r['fixed_comp_gap']:+.3f} / {r['trainonly_comp_gap']:+.3f}"
            + (f". Split by band (P8), the retrained models lose alike among IN-band positives (k-mer "
               f"{r['within_kmer_inband_gap']:+.3f}, composition-only {r['within_comp_inband_gap']:+.3f} vs random) and the "
               f"composition-only model gains among OUT-of-band positives ({r['within_comp_outband_gap']:+.3f}; k-mer "
               f"{r['within_kmer_outband_gap']:+.3f}): where the kept negatives resemble the positives in composition, both "
               f"models lose, as under matching"
               if pd.notna(r.get("within_kmer_inband_gap", np.nan)) else "") + ".")
    above = pd.notna(c.get("comp_paper_anchor_auroc_lo", np.nan)) and c["comp_paper_anchor_auroc_lo"] > 0.5
    add(f"  (B) It does NOT show that the removed signal is an artifact (a 'shortcut') rather than biology. Matching removes")
    add(f"      composition-associated signal indiscriminately, artifactual or biological, and "
        + ("not completely" if above else "(on these numbers) essentially completely")
        + f": the composition-only AUROC falls {c['comp0']:.3f} -> {c['comp_paper_anchor_auroc']:.3f} "
        f"[{c['comp_paper_anchor_auroc_lo']:.3f},{c['comp_paper_anchor_auroc_hi']:.3f}] on the published cohn split "
        f"({c['comp_m']:.3f} [{c['comp_m_lo']:.3f},{c['comp_m_hi']:.3f}] on the seed-42 re-draw)"
        + ("; both CIs exclude 0.5 (rule: lower CI > 0.5)." if above and c["comp_m_lo"] > 0.5 else "."))
    add(f"      The clean control also loses AUROC under matching: matched-vs-random {k['drop_shortcut']:+.3f} "
        f"{ci(k, 'p7_matched_vs_random')} (verdict {k['verdict']}; cohn {c['verdict']}; its composition-only AUROC starts "
        f"at {k['comp0']:.3f}).")
    if has7 and dst is not None:
        rw, rf, rt = (dst_row(dst, "drosophila_enhancers_stark", x) for x in ("within", "fixed_model", "train_only"))
        pr = paired_reading(k, dst, "drosophila_enhancers_stark")
        add(f"      The control's k-mer vs composition-only matched-vs-random loss, per design (P7): retrained, seed 42 "
            f"{v3(k, 'p7_matched_vs_random')} vs {v3(k, 'p7_comp_matched_vs_random')} (seeds 42-44 "
            f"{k['p7_s4244_matched_vs_random']:+.3f} vs {k['p7_s4244_comp_matched_vs_random']:+.3f}) -> {k['p7_kmc_reading']}"
            + (f"; original model on the matched test {v3(k, 'p7_fixed_matched_vs_random')} vs "
               f"{v3(k, 'p7_fixed_comp_matched_vs_random')} -> {k['p7_fixed_kmc_reading']}"
               if pd.notna(k.get("p7_fixed_matched_vs_random", np.nan)) else "")
            + f"; original test {v3(k, 'p7_trainonly_matched_vs_random')} vs {v3(k, 'p7_trainonly_comp_matched_vs_random')} "
            f"-> {k['p7_trainonly_kmc_reading']}. Across arms, Spearman of the two gaps: "
            + ", ".join(f"{x} {r['spearman_kmer_gap_vs_comp_gap']:+.2f}" for x, r in
                        (("within", rw), ("fixed_model", rf), ("train_only", rt)) if r is not None) + ".")
        if pr is True:
            add("      Reading (post hoc; rule: in both paired designs R7c is not 'MORE' and the cross-arm Spearman is >= 0.9): "
                "the control's loss is consistent with matching removing composition-associated signal from the clean "
                "control too; only the retrained seed-42 comparison hints at an extra k-mer loss, and it is not "
                "distinguishable (R7c) and not stable across selection seeds. Selection-induced difficulty is not shown.")
        elif pr is False:
            add("      Reading (post hoc; same rule): the paired designs do not both show the k-mer loss tracking the "
                "composition-only loss, so what the control's loss reflects is NOT resolved.")
    if bt is not None and "diff_of_contrasts" in bt.columns:
        q = bt[(bt["kind"] == "contrast") & (bt["contrast"] == "human_enhancers_cohn - drosophila_enhancers_stark")]
        q = q.set_index("design")
        if {"fixed_model_overlap", "within_overlap"} <= set(q.index):
            f_, w_ = q.loc["fixed_model_overlap"], q.loc["within_overlap"]
            add(f"      The cohn-vs-control contrast in k-mer share lost ({f_['share_above_chance_lost']:+.3f}) matches the "
                f"contrast in composition-only share removed by the same weights ({f_['comp_share_diff']:+.3f}; difference "
                f"{f_['diff_of_contrasts']:+.3f} [{f_['diff_of_contrasts_lo']:+.3f},{f_['diff_of_contrasts_hi']:+.3f}]).")

            def ci_(r_, k_):
                return f"{r_[k_]:+.3f} [{r_[k_ + '_lo']:+.3f},{r_[k_ + '_hi']:+.3f}]"
            sig = [f"{sc} scale, {dn} design" for dn, r_ in (("original-model", f_), ("retrained", w_))
                   for sc, k_ in (("additive", "diff_of_contrasts"), ("ratio", "ratio_diff")) if r_[k_ + "_lo"] > 0]
            add(f"      Per unit of composition signal removed (cohn minus control): additive scale original-model "
                f"{ci_(f_, 'diff_of_contrasts')}, retrained {ci_(w_, 'diff_of_contrasts')}; ratio scale original-model "
                f"{ci_(f_, 'ratio_diff')}, retrained {ci_(w_, 'ratio_diff')}.")
            add(("      Cohn loses more per unit only on the " + " and ".join(sig) +
                 " -> weak, scale- and design-dependent evidence of greater per-unit reliance (the P6 rule needs both "
                 "scales in the primary design).") if sig else
                "      Cohn does not lose more per unit of composition signal removed on either scale in either design.")
            add(f"      So the contrast is not evidence of specificity; even a greater per-unit reliance would not separate "
                f"artifact from biology.")
    add(f"      The artifact reading rests on the paper's independent evidence (dinucleotide-shuffled negatives, the")
    add(f"      information-content-matched control-motif panel, coordinate analysis), which this module does not re-test.")
    if f2d is not None:
        qd = f2d[(f2d["dataset"] == "drosophila_enhancers_stark") & (~f2d["underpowered"])]
        fl = qd[qd["sid_within"]]
        if len(fl):
            add(f"  (C) CAUTION (post hoc): a one-sided propensity selection changes a task in ways mean balance does not show. "
                f"In the control, at {len(fl)} common-support trimming levels ({', '.join(fl['level_label'])}; "
                f"{fl['retained_neg_frac'].min():.0%}-{fl['retained_neg_frac'].max():.0%} of negatives kept) the RETRAINED "
                f"k-mer AUROC falls {fl['kmer_gap'].min():.3f}-{fl['kmer_gap'].max():.3f} against random while the retrained "
                f"composition-only AUROC moves by at most {fl['comp_gap'].abs().max():.3f}. This is not shown to be difficulty "
                f"unrelated to composition: with the original model on the trimmed test the composition-only AUROC falls "
                f"{fl['fixed_comp_gap'].min():.3f}-{fl['fixed_comp_gap'].max():.3f} (k-mer {fl['fixed_kmer_gap'].min():.3f}-"
                f"{fl['fixed_kmer_gap'].max():.3f}), with the trimmed-split model on the original test "
                f"{fl['trainonly_comp_gap'].min():.3f}-{fl['trainonly_comp_gap'].max():.3f} (k-mer "
                f"{fl['trainonly_kmer_gap'].min():.3f}-{fl['trainonly_kmer_gap'].max():.3f}), and no level is flagged in "
                f"those designs ({int(qd['sid_fixed_model'].sum())} and {int(qd['sid_train_only'].sum())} of {len(qd)})"
                + (f". Split by band (P8; {fl['frac_test_pos_outside_band'].min():.0%}-{fl['frac_test_pos_outside_band'].max():.0%} "
                   f"of test positives lie outside the kept negatives' band): among IN-band positives the retrained k-mer "
                   f"and composition-only models both lose (k-mer {fl['within_kmer_inband_gap'].min():+.3f} to "
                   f"{fl['within_kmer_inband_gap'].max():+.3f}, composition-only {fl['within_comp_inband_gap'].min():+.3f} to "
                   f"{fl['within_comp_inband_gap'].max():+.3f}; R8 'k-mer loses in-band beyond composition-only' at "
                   f"{int(fl['r8_kmer_inband_loss_beyond_comp'].sum())} of {len(fl)} levels); among OUT-of-band positives the "
                   f"composition-only gap is {fl['within_comp_outband_gap'].min():+.3f} to {fl['within_comp_outband_gap'].max():+.3f} "
                   f"(R8 'composition change from the tails' at {int(fl['r8_comp_change_from_tails'].sum())} of {len(fl)} levels) "
                   f"while the k-mer gap is {fl['within_kmer_outband_gap'].min():+.3f} to {fl['within_kmer_outband_gap'].max():+.3f}: "
                   f"the flag mixes composition removal in the band with a retrained k-mer model that does not gain"
                   + (" (here it loses)" if fl["within_kmer_outband_gap"].min() > 0 else "")
                   + " on the positives left outside the band; only that last part is specific to the k-mer model" if "within_kmer_inband_gap" in fl.columns and fl["within_kmer_inband_gap"].notna().all()
                   else "")
                + ". The point for practice: benchmark every cleaned split against size-matched random negatives, and read "
                "it in a paired design (original model or original test held fixed), not only after retraining.")
    add("-" * 100)
    return "\n".join(L)


def report_table(pt, summ):
    """Second part of the interpretation file: the compact paper table and the imbalance-robust
    metrics along the F1 sweep (reported, not decision-driving)."""
    L = []
    add = L.append
    add("=" * 100)
    add("COMPACT TABLE (psm_sensitivity_paper_table.csv). k-mer LightGBM. matched rows: value [95% CI];")
    add("random/placebo rows: mean over 10 seeds (min-max). bal-MCC = class-balance-weighted MCC at the prior-")
    add("corrected threshold; MCC@0.5 = the paper's metric; AP-neg lift = negative-class PR-AUC / negative prevalence.")
    add("For weighting rows 'retained' is the negatives' effective sample size.")
    add("=" * 100)
    for ds, g in pt.groupby("dataset", sort=False):
        add(f"  {ds}")
        for _, r in g.iterrows():
            def fmt(k):
                if pd.notna(r.get(k + "_lo", np.nan)):
                    return f"{r[k]:.3f} [{r[k + '_lo']:.3f},{r[k + '_hi']:.3f}]"
                if pd.notna(r.get(k + "_min", np.nan)):
                    return f"{r[k]:.3f} ({r[k + '_min']:.3f}-{r[k + '_max']:.3f})"
                return f"{r[k]:.3f}"
            add(f"    {r['condition']:<52} ret {r['retained_neg_frac']:6.1%} test {r['n_test_pos']}/{r['n_test_neg']}"
                f"  AUROC {fmt('auroc')}  bal-MCC {fmt('bal_mcc')}  MCC@0.5 {fmt('mcc05')}"
                f"  bal20-MCC {r['mcc_bal20_pc']:.3f}  AP-neg lift {r['ap_neg_lift']:.2f}  comp-AUROC {r['comp_auroc']:.3f}")
    add("")
    add("F1 sweep, imbalance-robust metrics (within design; matched | random mean). bal20 = MCC over 20 explicit")
    add("class-balanced test draws (majority subsampled) at the prior-corrected threshold; AP-neg with its prevalence:")
    s = summ[(summ["family"] == "F1_hist") & (summ["design"] == "within")]
    for ds in s["dataset"].unique():
        add(f"  {ds}")
        k = s[(s["dataset"] == ds) & (s["model"] == "kmer")].sort_values("retained_neg_frac", ascending=False)
        c = s[(s["dataset"] == ds) & (s["model"] == "comp")].set_index("level_label")
        for _, r in k.iterrows():
            cr = c.loc[r["level_label"]]
            prev = r["n_test_neg"] / (r["n_test_neg"] + r["n_test_pos"])
            add(f"    ret {r['retained_neg_frac']:6.1%} test neg {int(r['n_test_neg']):>4} (prev {prev:.3f})"
                f"  bacc_pc {r['bacc_pc_matched']:.3f}|{r['bacc_pc_random_mean']:.3f}"
                f"  bal20 {r['mcc_bal20_pc_matched']:.3f}|{r['mcc_bal20_pc_random_mean']:.3f}"
                f"  MCC@0.5 {r['mcc_matched']:.3f}|{r['mcc_random_mean']:.3f}"
                f"  AP-neg {r['ap_neg_matched']:.3f}|{r['ap_neg_random_mean']:.3f}"
                f"  comp-only AUROC {cr['auroc_matched']:.3f} [{cr['auroc_matched_lo']:.3f},{cr['auroc_matched_hi']:.3f}]"
                f"|{cr['auroc_random_mean']:.3f}  comp MCC@0.5 {cr['mcc_matched']:.3f}|{cr['mcc_random_mean']:.3f}")
    add("")
    add("F1 sweep, fixed original model scored on the arm's test subset (test-distribution change only) and the")
    add("arm-trained model scored on the ORIGINAL test (training change only); k-mer AUROC, matched | random mean:")
    for ds in s["dataset"].unique():
        add(f"  {ds}")
        for design in ("fixed_model", "train_only"):
            q = summ[(summ["dataset"] == ds) & (summ["family"] == "F1_hist") & (summ["design"] == design)
                     & (summ["model"] == "kmer")].sort_values("retained_neg_frac", ascending=False)
            add("    " + design + ": " + "  ".join(
                f"{r['retained_neg_frac']:.1%}:{r['auroc_matched']:.3f}|{r['auroc_random_mean']:.3f}" for _, r in q.iterrows()))
    return "\n".join(L)


def report(dec, summ, sb, ct, dst=None):
    L = []
    add = L.append
    add("=" * 100)
    add("PSM SENSITIVITY (Reviewer 1, comment 4): shortcut removal vs sample reduction vs test distortion")
    add("=" * 100)
    add("Primary: k-mer LightGBM AUROC; F1 = tempered propensity-histogram matching (alpha=1 == paper);")
    add("random = uniform negatives at the SAME per-split counts (10 seeds). 95% bootstrap CIs, 1000 resamples.")
    for _, d in dec.iterrows():
        add("")
        add(f"--- {d['dataset']} [{d['role']}]  retained negatives at alpha=1: {d['retained_neg_frac']:.1%} ---")
        add(f"  original AUROC A0 = {d['A0']:.3f} [{d['A0_lo']:.3f},{d['A0_hi']:.3f}]  tau = {d['tau']:.3f}")
        add(f"  matched  AUROC Am = {d['Am']:.3f} [{d['Am_lo']:.3f},{d['Am_hi']:.3f}] (selection seeds {d['Am_seedmin']:.3f}-{d['Am_seedmax']:.3f});"
            f" paper split {d['paper_anchor_auroc']:.3f}")
        add(f"  random   AUROC Ar = {d['Ar_mean']:.3f} (10 seeds {d['Ar_min']:.3f}-{d['Ar_max']:.3f})")
        add(f"  drop: total {d['drop_total']:+.3f} = size {d['drop_size']:+.3f} + matched-vs-random {d['drop_shortcut']:+.3f}"
            f"  -> matched-vs-random share {d['shortcut_share']:.2f} (named 'shortcut' in the pre-registration)")
        add(f"  (i-a) size <= tau: {d['crit_ia_size']}   (i-b) separation: {d['crit_ib_separation']}")
        add(f"  overlap weighting (nothing discarded; neg ESS {d['overlap_ess_neg_frac']:.1%}): Aw = {d['Aw_overlap']:.3f} "
            f"[{d['Aw_overlap_lo']:.3f},{d['Aw_overlap_hi']:.3f}] vs permuted-weight placebo {d['Apw_placebo_mean']:.3f} "
            f"({d['Apw_placebo_min']:.3f}-{d['Apw_placebo_max']:.3f}); drop {d['drop_overlap']:+.3f} "
            f"= {d['ratio_overlap_to_shortcut']:.2f}x the F1 shortcut drop -> (ii-a) {d['crit_iia_weighting']}")
        add(f"  two-sided 1:1 caliper 0.2 (keeps {d['twosided_retained_frac']:.1%} of each class, balanced): A2 = {d['A2_1to1']:.3f} "
            f"[{d['A2_1to1_lo']:.3f},{d['A2_1to1_hi']:.3f}] vs random 1:1 {d['A2r_mean']:.3f} ({d['A2r_min']:.3f}-{d['A2r_max']:.3f}); "
            f"drop {d['drop_twosided']:+.3f} = {d['ratio_twosided_to_shortcut']:.2f}x -> (ii-b) {d['crit_iib_twosided']}")
        add(f"  VERDICT (pre-registered rule): {d['verdict']}")
        add(f"  paper metric MCC(0.5): {d['mcc0']:.3f} -> matched {d['mcc_m']:.3f}; random {d['mcc_r_mean']:.3f} "
            f"({d['mcc_r_min']:.3f}-{d['mcc_r_max']:.3f}) => size/imbalance part {d['mcc_drop_size_imbalance']:+.3f}, "
            f"matched-vs-random part {d['mcc_drop_shortcut']:+.3f}")
        add(f"  balanced MCC (class-balanced, prior-corrected threshold): {d['balmcc0']:.3f} -> matched {d['balmcc_m']:.3f} "
            f"[{d['balmcc_m_lo']:.3f},{d['balmcc_m_hi']:.3f}]; random {d['balmcc_r_mean']:.3f} ({d['balmcc_r_min']:.3f}-{d['balmcc_r_max']:.3f})")
        add(f"  fixed model (trained once on original): matched test {d['fixed_A_matchedTest']:.3f} "
            f"[{d['fixed_A_matchedTest_lo']:.3f},{d['fixed_A_matchedTest_hi']:.3f}] vs size/ratio-matched random test "
            f"{d['fixed_A_randomTest_mean']:.3f} ({d['fixed_A_randomTest_min']:.3f}-{d['fixed_A_randomTest_max']:.3f}); "
            f"test-distortion effect {d['test_distortion_effect']:+.3f}, test-shortcut effect {d['test_shortcut_effect']:+.3f}")
        add(f"  training only (arm-trained model on ORIGINAL test): random-reduced train {d['trainonly_A_randomTrain_mean']:.3f} "
            f"({d['trainonly_A_randomTrain_min']:.3f}-{d['trainonly_A_randomTrain_max']:.3f}) -> training-size effect "
            f"{d['training_size_effect']:+.3f}; composition-matched train {d['trainonly_A_matchedTrain']:.3f}")
        add(f"  composition-only AUROC: original {d['comp0']:.3f}; matched {d['comp_m']:.3f}; random {d['comp_r_mean']:.3f}")
        if "att_auroc" in d and pd.notna(d.get("att_auroc")):
            add(f"  ATT odds weighting: AUROC {d['att_auroc']:.3f} vs placebo {d['att_placebo_mean']:.3f}; neg ESS "
                f"{d['att_ess_neg_frac']:.2%}{'  -> UNSTABLE (ESS<5%), not interpreted' if d['att_unstable'] else ''}")
        add("  label-noise (selection-bias) checks [reported, not decision-driving]:")
        if pd.notna(d.get("rho_hidden_pos", np.nan)):
            add(f"    cCRE-ELS overlap: positives {d['els_rate_pos']:.3f}, all negatives {d['els_rate_neg_all']:.3f}, "
                f"retained negatives {d['els_rate_retained']:.3f} (n={int(d['els_n_retained'])}) -> estimated hidden-positive "
                f"fraction rho={d['rho_hidden_pos']:.3f} (mixture-model estimate); label-noise AUROC cost {d['labelnoise_max_auroc_drop']:.3f} "
                + (f"= {d['labelnoise_bound_over_shortcut_drop']:.2f}x the measured shortcut drop" if d["drop_shortcut"] > 0
                   else "(measured shortcut drop <= 0: ratio undefined)"))
        if pd.notna(d.get("ccrefreeTest_Am", np.nan)):
            add(f"    cCRE-free test negatives only: original {d['ccrefreeTest_A0']:.3f}; matched {d['ccrefreeTest_Am']:.3f} "
                f"[{d['ccrefreeTest_Am_lo']:.3f},{d['ccrefreeTest_Am_hi']:.3f}] (n_neg={int(d['ccrefreeTest_n_neg_matched'])}) vs random "
                f"{d['ccrefreeTest_Ar_mean']:.3f} ({d['ccrefreeTest_Ar_min']:.3f}-{d['ccrefreeTest_Ar_max']:.3f}); drop "
                f"{d['ccrefreeTest_drop']:+.3f}{'  UNDERPOWERED' if d['ccrefreeTest_underpowered'] else ''}; fixed model "
                f"{d['ccrefreeTest_fixed_Am']:.3f} vs {d['ccrefreeTest_fixed_Ar_mean']:.3f}")
        if pd.notna(d.get("f5_Am", np.nan)):
            add(f"    F5 cCRE-purged re-match (every cCRE-overlapping negative removed first; keeps "
                f"{d['f5_retained_neg_frac']:.1%} of negatives, max|SMD| {d['f5_max_abs_smd']:.3f}): matched {d['f5_Am']:.3f} "
                f"[{d['f5_Am_lo']:.3f},{d['f5_Am_hi']:.3f}] vs random {d['f5_Ar_mean']:.3f} ({d['f5_Ar_min']:.3f}-{d['f5_Ar_max']:.3f}); "
                f"drop {d['f5_drop']:+.3f}")
        if pd.notna(d.get("f5b_Am", np.nan)):
            add(f"    F5b cCRE-purged two-sided 1:1 caliper 0.2 (keeps {d['f5b_retained_neg_frac']:.1%} of negatives, max|SMD| "
                f"{d['f5b_max_abs_smd']:.3f}): matched {d['f5b_Am']:.3f} [{d['f5b_Am_lo']:.3f},{d['f5b_Am_hi']:.3f}] vs random "
                f"{d['f5b_Ar_mean']:.3f} ({d['f5b_Ar_min']:.3f}-{d['f5b_Ar_max']:.3f}); drop {d['f5b_drop']:+.3f}")
        add(f"  [descriptive, post hoc] share of above-chance AUROC lost: F1 {d['frac_above_chance_lost_F1']:.2f}, "
            f"overlap weighting {d['frac_above_chance_lost_overlap']:.2f}, two-sided 1:1 {d['frac_above_chance_lost_twosided']:.2f}")
    add("")
    add("Sweep (F1, within design, k-mer AUROC matched | random mean [min,max] | gap), by retained-negative fraction:")
    s = summ[(summ["family"] == "F1_hist") & (summ["design"] == "within") & (summ["model"] == "kmer")]
    for ds in s["dataset"].unique():
        add(f"  {ds}")
        for _, r in s[s["dataset"] == ds].sort_values("retained_neg_frac", ascending=False).iterrows():
            add(f"    ret {r['retained_neg_frac']:6.1%}  maxSMD {r['max_abs_smd_matched']:.3f}  matched {r['auroc_matched']:.3f} "
                f"[{r['auroc_matched_lo']:.3f},{r['auroc_matched_hi']:.3f}]  random {r['auroc_random_mean']:.3f} "
                f"[{r['auroc_random_min']:.3f},{r['auroc_random_max']:.3f}]  gap {r['auroc_gap_random_minus_matched']:+.3f}"
                f"  | balMCC {r['mcc_bw_pc_matched']:.3f} vs {r['mcc_bw_pc_random_mean']:.3f}")
    add("")
    add("2x2 cross at the paper's retention (k-mer AUROC; model trained on matched|random x test matched|random):")
    c = ct[(ct["model"] == "kmer") & (ct["matched_arm"].isin(["F1_hist_a1_matched42", "paper_equalized_matched"]))]
    for _, r in c.iterrows():
        add(f"  {r['dataset']:<28} {r['matched_arm']:<24} MM {r['auroc_matchedModel_matchedTest']:.3f}  "
            f"RM(random model, matched test) {r['auroc_randomModel_matchedTest']:.3f}  MR {r['auroc_matchedModel_randomTest']:.3f}  "
            f"RR {r['auroc_randomModel_randomTest']:.3f}  | test effect {r['auroc_test_effect_randomModel']:+.3f}, "
            f"training effect on matched test {r['auroc_train_effect_matchedTest']:+.3f}")
    add("")
    add("Secondary family F2 (common-support trimming), within, k-mer AUROC:")
    s = summ[(summ["family"] == "F2_trim") & (summ["design"] == "within") & (summ["model"] == "kmer")]
    for ds in s["dataset"].unique():
        add(f"  {ds}")
        for _, r in s[s["dataset"] == ds].sort_values("retained_neg_frac", ascending=False).iterrows():
            add(f"    ret {r['retained_neg_frac']:6.1%}  maxSMD {r['max_abs_smd_matched']:.3f}  matched {r['auroc_matched']:.3f} "
                f"random {r['auroc_random_mean']:.3f} [{r['auroc_random_min']:.3f},{r['auroc_random_max']:.3f}]"
                f"{'  UNDERPOWERED' if r['underpowered'] else ''}")
    add("")
    add("Two-sided 1:1 caliper matching (F3) caliper-sensitivity, within, k-mer AUROC (retained = each class):")
    s = summ[(summ["family"] == "F3_1to1") & (summ["design"] == "within") & (summ["model"] == "kmer")]
    for ds in s["dataset"].unique():
        add(f"  {ds}")
        for _, r in s[s["dataset"] == ds].sort_values("retained_neg_frac", ascending=False).iterrows():
            add(f"    {r['level_label']:<6} ret {r['retained_neg_frac']:6.1%}  maxSMD {r['max_abs_smd_matched']:.3f}  matched "
                f"{r['auroc_matched']:.3f} [{r['auroc_matched_lo']:.3f},{r['auroc_matched_hi']:.3f}]  random "
                f"{r['auroc_random_mean']:.3f} [{r['auroc_random_min']:.3f},{r['auroc_random_max']:.3f}]  balMCC "
                f"{r['mcc_bw_pc_matched']:.3f} vs {r['mcc_bw_pc_random_mean']:.3f}")
    add("")
    add("Positive-side matching (F4) and ATT weighting, within, k-mer AUROC:")
    for fam, lev in (("F4_posside", "a1"), ("W_att", "w")):
        s = summ[(summ["family"] == fam) & (summ["design"] == "within") & (summ["model"] == "kmer")]
        for _, r in s.iterrows():
            add(f"  {r['dataset']:<28} {fam:<11} matched {r['auroc_matched']:.3f} [{r['auroc_matched_lo']:.3f},"
                f"{r['auroc_matched_hi']:.3f}] vs random/placebo {r['auroc_random_mean']:.3f} "
                f"[{r['auroc_random_min']:.3f},{r['auroc_random_max']:.3f}]; retained pos {r['retained_pos_frac']:.1%}, "
                f"neg ESS {r['ess_neg_frac']:.1%}")
    if sb is not None and len(sb):
        add("")
        add("Selection bias (features OUTSIDE the propensity model), retained vs discarded negatives "
            f"(p = uncorrected Fisher / Mann-Whitney; p_bonf = Bonferroni over all {int(sb['n_tests_bonferroni'].iloc[0])} tests):")
        for (ds, sel), g in sb.groupby(["dataset", "selection"], sort=False):
            if sel == "paper_equalized":
                continue
            add(f"  {ds} / {sel} (retained {int(g['n_retained'].iloc[0])}, discarded {int(g['n_discarded'].iloc[0])})")
            for _, r in g.iterrows():
                add(f"    {r['feature']:<28} pos {r['mean_pos']:.4g} | retained {r['mean_retained']:.4g} | discarded "
                    f"{r['mean_discarded']:.4g} | random {r['mean_random_sizematched']:.4g}   SMD ret-vs-disc "
                    f"{r['smd_retained_vs_discarded']:+.2f}, ret-vs-pos {r['smd_retained_vs_pos']:+.2f}  "
                    f"p={r['p_retained_vs_discarded']:.1e} p_bonf={r['p_bonferroni']:.1e}")
    add("")
    add("CAVEATS: (1) matched and random arms change the test set, so within-arm comparisons are unpaired; the")
    add("fixed-model and weighting designs are the paired/removal-free complements. (2) nt_enhancers' test split has")
    ntr = dec.set_index("dataset")
    if NT_RAW in ntr.index and LF_TEST in ntr.index:
        a_, b_ = ntr.loc[NT_RAW], ntr.loc[LF_TEST]
        add(f"200 negatives, so its alpha=1 test keeps only {int(a_['f1_n_test_neg'])} negatives (wide CIs), and at alpha=1 its "
            f"matched set is NOT")
        add(f"balanced (max|SMD| {a_['f1_max_abs_smd']:.3f}; composition-only AUROC on it {a_['comp_m']:.3f} uncorrected / "
            f"{b_['comp_m']:.3f} leak-free test; the published nt split max|SMD| {a_['paper_anchor_max_abs_smd']:.3f}).")
    add("(3) nt_enhancers LEAKAGE: 100 of its 200 test positives occur verbatim (same label) in the training split and")
    add("are always kept by negative-side selection, so every uncorrected nt number is inflated; the leak-free rows")
    add("(nt_enhancers_leakfreeTest = same models, those 100 test positives removed; nt_enhancers_leakfreeRefit = refit")
    add("without the 107 training copies) are the ones to report. (4) Weighting balances the 20 features only in")
    add("expectation (overlap weights: exact mean balance for the linear propensity terms); nonlinear composition")
    add("structure survives (composition-only LightGBM AUROC under weighting), so weighting does not 'equalize' composition.")
    add("(5) The within-arm verdict uses selection seed 42 only (pre-registered); the seed range and the published split")
    add("are printed next to it. (6) MCC at the 0.5 threshold is prevalence- and calibration-dependent: on a")
    add("~94%-positive test set a model trained on ~96% positives predicts almost everything positive, so its collapse")
    add("is reproduced by size-matched RANDOM negatives and is not evidence of shortcut removal; even class-balanced MCC")
    add("at the prior-corrected threshold falls mostly through the size/imbalance part. Use AUROC against the random")
    add("control, and MCC only from the balanced 1:1 design. (7) Composition matching removes composition-associated")
    add("signal indiscriminately, artifactual or biological, and not completely (the composition-only AUROC on the")
    dd_ = dec.set_index("dataset")
    if "human_enhancers_cohn" in dd_.index and pd.notna(dd_.loc["human_enhancers_cohn"].get("comp_paper_anchor_auroc", np.nan)):
        c_ = dd_.loc["human_enhancers_cohn"]
        add(f"published cohn split is {c_['comp_paper_anchor_auroc']:.3f} [{c_['comp_paper_anchor_auroc_lo']:.3f},"
            f"{c_['comp_paper_anchor_auroc_hi']:.3f}], from {c_['comp0']:.3f}); calling the removed signal a 'shortcut' rests on")
    else:
        add("published cohn split stays above 0.5); calling the removed signal a 'shortcut' rests on")
    add("the paper's independent evidence (dinucleotide-shuffled negatives, control-motif panel, genomic localization).")
    pr = (paired_reading(dd_.loc["drosophila_enhancers_stark"], dst, "drosophila_enhancers_stark")
          if dst is not None and "drosophila_enhancers_stark" in dd_.index else None)
    if pr is True:
        add("The clean control also loses AUROC under matching; in the paired designs (original model or original test")
        add("held fixed) that loss follows the control's own composition-only loss (rule and numbers in (B) above), so")
        add("matching removes composition-associated signal from the clean control too, and a matched-vs-random loss is")
        add("not by itself evidence of an artifact. It loses a smaller share under overlap weighting, but run_posthoc.py P6 shows that this")
    else:
        add("The clean control also loses AUROC under matching, and what that loss reflects is not resolved (see (B)")
        add("above), so its matched-vs-random effect is not called a composition effect. It loses a smaller share under")
        add("overlap weighting, but run_posthoc.py P6 shows that this")
    add("contrast is largely accounted for by how much composition-only signal the same weights remove (fully on the")
    add("additive scale; see the READING lines for the ratio scale), and even a greater per-unit reliance would not")
    add("separate artifact from biology, so it is NOT evidence of specificity to an artifact. (8) Selected negatives are")
    add("not a random sample (part (6)). Common-support trimming (F2) keeps every positive but only negatives inside the")
    add("positives' central propensity band, so it both removes composition signal (in the band) and creates it (for the")
    add("positives outside the band; P8). In the clean control at the strictest levels the RETRAINED k-mer AUROC falls while")
    add("the retrained composition-only AUROC does not, but with the original model or the original test held fixed both")
    add("fall together (f2_selection_difficulty, per design). So F2 does not isolate difficulty unrelated to composition;")
    add("it shows that a propensity-based selection can change a task in ways mean balance does not reveal, which is why")
    add("cleaned splits must be benchmarked against size-matched random negatives, in a paired design.")
    add("(9) The label-noise figure is a mixture-model estimate, not a worst case.")
    if "labelnoise_bound_over_shortcut_drop" in dec.columns:
        q = dec[dec["dataset"].isin([NT_RAW, LF_TEST, LF_REFIT]) & dec["labelnoise_bound_over_shortcut_drop"].notna()]
        if len(q):
            pb = q["els_p_bonferroni"].iloc[0] if "els_p_bonferroni" in q.columns else np.nan
            add("For nt_enhancers it is " + ", ".join(
                f"{r['labelnoise_bound_over_shortcut_drop']:.2f}x ({r['dataset']}, verdict {r['verdict']})"
                for _, r in q.iterrows()) + f" the matched-vs-random effect; the nt ELS enrichment behind it has Bonferroni p "
                f"{pb:.2g} and the nt composition-matching effect is not supported in any variant, so for nt the check says little "
                f"either way (label noise is neither shown nor excluded).")
    if "fixedoverlap_comp_auroc" in dec.columns:
        add("(10) Overlap weights come from a LINEAR logistic propensity model. Where the composition-only LightGBM")
        add("scored under the same weights falls below 0.5, the weights invert the nonlinear composition signal, so that")
        add("weighted k-mer AUROC overstates the effect of removing composition and only its direction is interpretable.")
        add("Composition-only AUROC under the weights, retrained / original model (flag: < 0.5):")
        for _, r in dec.iterrows():
            a, b = r.get("overlap_comp_auroc", np.nan), r.get("fixedoverlap_comp_auroc", np.nan)
            add(f"  {r['dataset']:<28} {a:.3f}{' (<0.5)' if a < 0.5 else ''} / {b:.3f}{' (<0.5)' if b < 0.5 else ''}")
    if "bw_paper_anchor_mcc05" in dec.columns:
        c_ = dec.set_index("dataset").loc["human_enhancers_cohn"] if (dec["dataset"] == "human_enhancers_cohn").any() else None
        add("(11) MCC@0.5 on the class-balanced-training rows (bw_*, post hoc P3) is not informative either: class-balanced")
        add("LightGBM still calls almost every test sequence of a ~94%-positive matched test positive, so its MCC@0.5 moves")
        if c_ is not None:
            add(f"with a handful of predictions (cohn published split: {c_['bw_paper_anchor_mcc05']:.3f} with class-balanced "
                f"weights vs {c_['paper_anchor_mcc']:.3f} without, while AUROC differs, {c_['bw_paper_anchor_auroc']:.3f} vs "
                f"{c_['paper_anchor_auroc']:.3f}). Use AUROC there too.")
        else:
            add("with a handful of predictions. Use AUROC there too.")
    return "\n".join(L)


def add_bonferroni(sb):
    sb = sb.drop(columns=[c for c in ("n_tests_bonferroni", "p_bonferroni") if c in sb.columns])
    m = int(sb["p_retained_vs_discarded"].notna().sum())
    sb["n_tests_bonferroni"] = m
    sb["p_bonferroni"] = np.minimum(1.0, sb["p_retained_vs_discarded"] * m)
    return sb


def report_posthoc(lk, bt, f2d, sbf2, lki=None, dc=None):
    L = []
    add = L.append
    add("=" * 100)
    add("POST HOC ANALYSES (run_posthoc.py; not pre-registered; added after verification reviews)")
    add("=" * 100)
    if lk is not None:
        add("Leakage audit (psm_sensitivity_leakage_check.csv):")
        for _, r in lk.iterrows():
            add(f"  {r['dataset']:<28} test in train {int(r['n_test_verbatim_in_train'])} ({int(r['n_test_verbatim_pos'])} pos / "
                f"{int(r['n_test_verbatim_neg'])} neg); train rows equal to a test seq {int(r['n_train_rows_equal_a_test_seq'])}; "
                f"reverse-complement-only matches {int(r['n_test_revcomp_in_train_not_verbatim'])}; original AUROC "
                f"{r['orig_auroc_full_test']:.3f} -> leak-free {r['orig_auroc_leakfree_test']:.3f}")
    if lki is not None:
        add("Interval-overlap audit, P0b (psm_sensitivity_leakage_interval.csv; hg38; descriptive, nothing removed elsewhere):")
        for _, r in lki.iterrows():
            add(f"  {r['dataset']:<28} mapped test {int(r['n_test_mapped'])}/{int(r['n_test'])}: overlap a training interval "
                f">=1 bp {int(r['n_test_overlap_any'])} ({int(r['n_test_overlap_any_pos'])} pos / "
                f"{int(r['n_test_overlap_any_neg'])} neg), >=50% {int(r['n_test_overlap_half'])}; original AUROC "
                f"{r['orig_auroc_full_test']:.3f} -> {r['orig_auroc_without_overlapping']:.3f} without them; in the F1 a=1 "
                f"matched (seed 42) test {int(r['n_overlap_any_neg_in_F1a1_matched42_test'])} of "
                f"{int(r['F1a1_matched42_test_n_neg'])} negatives")
    if bt is not None:
        add("")
        add("Between-dataset contrast (psm_sensitivity_between_dataset.csv), overlap weighting vs permuted-weight placebo:")
        for _, r in bt[bt["kind"] == "dataset"].iterrows():
            add(f"  {r['design']:<20} {r['dataset']:<28} overlap {r['auroc_overlap']:.3f} [{r['auroc_overlap_lo']:.3f},"
                f"{r['auroc_overlap_hi']:.3f}] placebo {r['auroc_placebo_mean']:.3f}; drop {r['drop']:+.3f} [{r['drop_lo']:+.3f},"
                f"{r['drop_hi']:+.3f}]; share of above-chance AUROC lost {r['share_above_chance_lost']:.3f} "
                f"[{r['share_lo']:.3f},{r['share_hi']:.3f}]"
                + (f"; composition-only: overlap {r['comp_auroc_overlap']:.3f} vs placebo {r['comp_auroc_placebo_mean']:.3f}, "
                   f"share lost {r['comp_share_lost']:.3f} [{r['comp_share_lo']:.3f},{r['comp_share_hi']:.3f}]; k-mer minus "
                   f"comp share {r['kmer_minus_comp_share']:+.3f} [{r['kmer_minus_comp_share_lo']:+.3f},"
                   f"{r['kmer_minus_comp_share_hi']:+.3f}]"
                   if pd.notna(r.get("comp_share_lost", np.nan)) else ""))
    if f2d is not None:
        add("")
        add("Common-support trimming (F2): k-mer vs composition-only gap to size-matched random, per design; P8 = in-band / "
            "out-of-band test positives (psm_sensitivity_f2_selection_difficulty.csv, psm_sensitivity_f2_band.csv):")
        for _, r in f2d.iterrows():
            add(f"  {r['dataset']:<28} {r['level_label']:<6} kept {r['retained_neg_frac']:6.1%}"
                + (f" pos outside band {r['frac_test_pos_outside_band']:4.0%}" if pd.notna(r.get('frac_test_pos_outside_band', np.nan)) else "")
                + f"  gap k-mer | comp-only: within {r['kmer_gap']:+.3f} | {r['comp_gap']:+.3f}; fixed model "
                f"{r['fixed_kmer_gap']:+.3f} | {r['fixed_comp_gap']:+.3f}; train only {r['trainonly_kmer_gap']:+.3f} | "
                f"{r['trainonly_comp_gap']:+.3f}"
                + (f"; P8 within in-band {r['within_kmer_inband_gap']:+.3f} | {r['within_comp_inband_gap']:+.3f}, out-of-band "
                   f"{r['within_kmer_outband_gap']:+.3f} | {r['within_comp_outband_gap']:+.3f}"
                   if pd.notna(r.get('within_kmer_inband_gap', np.nan)) else "")
                + f"{'  FLAG(within)' if r['sid_within'] else ''}{'  FLAG(fixed_model)' if r['sid_fixed_model'] else ''}"
                f"{'  FLAG(train_only)' if r['sid_train_only'] else ''}{'  UNDERPOWERED' if r['underpowered'] else ''}")
    if dc is not None and len(dc):
        add("")
        add("P7 decomposition CIs at F1 alpha=1 (psm_sensitivity_decomposition_ci.csv; two-level paired bootstrap: test "
            "sequences x random [x matched] seeds, B=1000; post hoc):")
        for _, r in ordered(dc).iterrows():
            head = f"  {r['dataset']:<28} {r['design']:<10} {r['model']:<15} seeds {r['matched_seeds']:<5}"
            if r["model"] == "kmer_minus_comp":
                add(head + f" matched-vs-random diff {r['matched_vs_random']:+.3f} [{r['matched_vs_random_lo']:+.3f},"
                    f"{r['matched_vs_random_hi']:+.3f}]  relative diff {r['rel_gap']:+.3f} [{r['rel_gap_lo']:+.3f},"
                    f"{r['rel_gap_hi']:+.3f}] -> {r['reading']}")
            else:
                add(head + f" A0 {r['A0']:.3f} Ar {r['Ar_mean']:.3f} Am {r['Am']:.3f}; size {r['size_effect']:+.3f} "
                    f"[{r['size_effect_lo']:+.3f},{r['size_effect_hi']:+.3f}] ({r['size_vs_tau']}); matched-vs-random "
                    f"{r['matched_vs_random']:+.3f} [{r['matched_vs_random_lo']:+.3f},{r['matched_vs_random_hi']:+.3f}]; "
                    f"relative {r['rel_gap']:.3f} [{r['rel_gap_lo']:.3f},{r['rel_gap_hi']:.3f}]")
    if sbf2 is not None:
        add("")
        add("F2 selection bias (psm_sensitivity_selection_bias_f2.csv): retained vs discarded negatives vs positives "
            "(|SMD ret-vs-disc| >= 0.2 shown):")
        for (ds, sel), g in sbf2.groupby(["dataset", "selection"], sort=False):
            g = g[g["smd_retained_vs_discarded"].abs() >= 0.2]
            add(f"  {ds} / {sel} (kept {g['retained_neg_frac'].iloc[0]:.1%})" if len(g) else f"  {ds} / {sel}: none")
            for _, r in g.iterrows():
                add(f"    {r['feature']:<28} pos {r['mean_pos']:.4g} | retained {r['mean_retained']:.4g} | discarded "
                    f"{r['mean_discarded']:.4g} | random {r['mean_random_sizematched']:.4g}  SMD ret-vs-disc "
                    f"{r['smd_retained_vs_discarded']:+.2f}, ret-vs-pos {r['smd_retained_vs_pos']:+.2f}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=C.RESULTS_DIR)
    args = ap.parse_args()

    def opt(name):
        p = os.path.join(args.dir, name)
        return pd.read_csv(p) if os.path.exists(p) else None
    sw = pd.read_csv(os.path.join(args.dir, "psm_sensitivity_sweep.csv"))
    bal = pd.read_csv(os.path.join(args.dir, "psm_sensitivity_balance.csv"))
    psw, pbal = opt("psm_sensitivity_posthoc_sweep.csv"), opt("psm_sensitivity_posthoc_balance.csv")
    if psw is not None:                  # post hoc rows, under their own dataset labels / families
        sw = pd.concat([sw, psw], ignore_index=True)
        bal = pd.concat([bal, pbal], ignore_index=True)
    sbp = os.path.join(args.dir, "psm_sensitivity_selection_bias.csv")
    sb = None
    if os.path.exists(sbp):
        sb = add_bonferroni(pd.read_csv(sbp))
        sb.to_csv(sbp, index=False)
    bt, lk, sbf2 = (opt("psm_sensitivity_between_dataset.csv"), opt("psm_sensitivity_leakage_check.csv"),
                    opt("psm_sensitivity_selection_bias_f2.csv"))
    summ = summarize(sw, bal)
    summ.to_csv(os.path.join(args.dir, "psm_sensitivity_summary.csv"), index=False)
    ct = cross_table(sw, bal)
    ct.to_csv(os.path.join(args.dir, "psm_sensitivity_cross2x2.csv"), index=False)
    dec = decide(sw, summ, sb)                       # pre-registered, unchanged
    dc = opt("psm_sensitivity_decomposition_ci.csv")      # run_posthoc.py P7
    dec = posthoc_columns(ordered(dec), summ, bal, sb, bt, sw, dc)
    dec.to_csv(os.path.join(args.dir, "psm_sensitivity_decision.csv"), index=False)
    pt = paper_table(dec, summ, sw)
    pt.to_csv(os.path.join(args.dir, "psm_sensitivity_paper_table.csv"), index=False)
    dz = dissociation(summ)
    dz.to_csv(os.path.join(args.dir, "psm_sensitivity_dissociation.csv"), index=False)
    dst = ordered(dissociation_stats(summ))
    dst.to_csv(os.path.join(args.dir, "psm_sensitivity_dissociation_stats.csv"), index=False)
    f2d = ordered(f2_selection_difficulty(summ, opt("psm_sensitivity_f2_band.csv")))     # P8 merged when present
    f2d.to_csv(os.path.join(args.dir, "psm_sensitivity_f2_selection_difficulty.csv"), index=False)
    cal = caliper_sweep(summ)
    cal.to_csv(os.path.join(args.dir, "psm_sensitivity_caliper_sweep.csv"), index=False)
    lki = opt("psm_sensitivity_leakage_interval.csv")
    txt = (headline(dec, pt, sb, dst, bt, lk, f2d, sbf2, cal) + "\n" + scope_block(dec, pt, bt, dst, f2d, cal) + "\n\n"
           + report(dec, summ, sb, ct, dst) + "\n\n"
           + report_table(pt, summ) + "\n\n" + report_dissociation(dz, dst, f2d) + "\n\n"
           + report_posthoc(lk, bt, f2d, sbf2, lki, dc))
    with open(os.path.join(args.dir, "psm_sensitivity_interpretation.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(txt)



if __name__ == "__main__":
    main()
