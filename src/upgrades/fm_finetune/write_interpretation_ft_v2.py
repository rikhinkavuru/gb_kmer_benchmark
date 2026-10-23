#!/usr/bin/env python3
"""Write results/upgrades/fm_finetune_v2_interpretation.txt from the v2 result CSVs (torch-free).
Every number in the text is read from a CSV written by eval_ft_v2.py / summarize_ft_v2.py /
baselines_ft_v2.py / posthoc_ft_v2.py (or, for the reference numbers the paper already reports,
from the CSVs of the modules that produced them), so the file can be regenerated verbatim whenever
more fine-tuning units finish (e.g. the GPU-deferred units). Run order:
  verify_preds_ft_v2.py -> eval_ft_v2.py -> summarize_ft_v2.py -> baselines_ft_v2.py ->
  posthoc_ft_v2.py -> sensitivity_ft_v2.py -> make_fig_ft_v2.py -> write_interpretation_ft_v2.py
2026-09-26: labels for every model/dataset the pipeline can produce (DNABERT-2 LoRA on both datasets,
drosophila, gc_match arms, seed 44, GPU-deferred models), and the post hoc sections (leak-free
nt_enhancers, baselines on identical arms, reference numbers) are added; the pre-registered block
(Part 1) comes before every post hoc section and is unchanged in content. A short writer-facing
SUMMARY sits above both. It only re-arranges numbers from fm_finetune_v2_compare{,_paired}.csv and
tags each one [pre] or [post hoc]. Caveat numbers are read from the manifest, runs and v1-vs-v2 CSVs.
Fix round 1 (2026-09-26, after two independent verifiers): wording is now driven by the data where
the verifiers found over-statements. (a) The share lost is tagged [post hoc, descriptive] and follows
the pre-registered S and B. (b) A retrained-on-equalized AUROC (D) is not read as "learnability" when
sensitivity_ft_v2.py flags the units degenerate (one-class, near-constant); the same arm's D of the
other models is printed instead. (c) Rule 1's clause is named (B's CI covers 0.5 vs only B < 0.6).
(d) Leak-free nt_enhancers lines carry n and the post hoc label; the pre-registered rules are not
re-stated as verdicts there. (e) drosophila is a "GC-balanced control" (rule 5 not met; dinucleotide
gap). (f) New Part 3 (sensitivity_ft_v2.py): unit diagnostics, cCRE latent-positive sensitivity,
random-draw reference for B, within-class score-composition, G-A paired vs frozen probes,
Bonferroni CIs, and the psm_sensitivity comparison for every dataset (not cohn only).
"""
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES = os.path.join(ROOT, "results", "upgrades")
PRE = os.path.join(RES, "fm_finetune_v2")
OUT = PRE + "_interpretation.txt"
PLAN = {  # the queues (run_queue_v2.sh, then run_queue_v3.sh; GPU bundle), for the status block
    ("hyenadna", "human_enhancers_cohn"): (("original", "comp_equalized", "random", "gc_match"), 3, "local"),
    ("hyenadna", "nt_enhancers"): (("original", "comp_equalized", "random", "gc_match"), 3, "local"),
    ("hyenadna", "drosophila_enhancers_stark"): (("original", "comp_equalized", "random", "gc_match"), 3, "local"),
    ("dnabert2", "human_enhancers_cohn"): (("original", "comp_equalized", "random"), 3, "local"),
    ("dnabert2", "nt_enhancers"): (("original", "comp_equalized", "random"), 3, "local"),
    ("dnabert2full", "human_enhancers_cohn"): (("original", "comp_equalized", "random"), 3, "GPU job 2"),
    ("ntv2_500m", "human_enhancers_cohn"): (("original", "comp_equalized", "random"), 3, "GPU job 3"),
    ("ntv2_500m", "drosophila_enhancers_stark"): (("original", "comp_equalized", "random"), 3, "GPU job 5"),
    ("ntv2_500m", "nt_enhancers"): (("original", "comp_equalized", "random"), 3, "GPU job 6"),
    ("dnabert2full", "nt_enhancers"): (("original", "comp_equalized", "random"), 3, "GPU job 7"),
}
MNAME = {"hyenadna": "HyenaDNA-tiny-16k-d128, FULL fine-tune",
         "dnabert2": "DNABERT-2-117M, LoRA (r=8, alpha=16) + linear head",
         "dnabert2full": "DNABERT-2-117M, FULL-parameter fine-tune (GPU-deferred)",
         "ntv2_500m": "Nucleotide Transformer v2 500M, LoRA r=8 (GPU-deferred)",
         "kmer_lgbm": "k-mer spectrum + LightGBM (the paper's classifier)",
         "hyenadna_frozen_mean": "frozen HyenaDNA, mean-pool + LightGBM head",
         "hyenadna_frozen_sep": "frozen HyenaDNA, [SEP] + LightGBM head",
         "dnabert2_frozen_mean": "frozen DNABERT-2, mean-pool + LightGBM head",
         "dnabert2_frozen_cls": "frozen DNABERT-2, [CLS] + LightGBM head"}
SHORT = {"hyenadna": "HyenaDNA full-FT", "dnabert2": "DNABERT-2 LoRA-FT",
         "dnabert2full": "DNABERT-2 full-FT", "ntv2_500m": "NT-v2-500M LoRA-FT",
         "kmer_lgbm": "k-mer LightGBM", "hyenadna_frozen_mean": "HyenaDNA frozen mean",
         "hyenadna_frozen_sep": "HyenaDNA frozen [SEP]", "dnabert2_frozen_mean": "DNABERT-2 frozen mean",
         "dnabert2_frozen_cls": "DNABERT-2 frozen [CLS]"}
DSN = {"human_enhancers_cohn": "cohn", "nt_enhancers": "nt_enhancers (released test, 100 leaked)",
       "nt_enhancers_leakfree": "nt_enhancers LEAK-FREE test (100+/36- on B)", "drosophila_enhancers_stark":
       "drosophila (GC-balanced control)"}
UNIT = PRE + "_unit_diagnostics.csv"
DSHORT = {"human_enhancers_cohn": "cohn", "nt_enhancers": "nt_enh", "nt_enhancers_leakfree": "nt_enh leak-free",
          "drosophila_enhancers_stark": "dros"}


def dshort(ds):
    return DSHORT.get(ds, ds)


def load_opt(suffix):
    p = PRE + suffix
    return pd.read_csv(p) if os.path.exists(p) else None


def degenerate_seeds(model, ds, arm):
    """(n degenerate, n units) for a fine-tuned model x dataset x train arm (sensitivity_ft_v2.py Q1)."""
    ud = load_opt("_unit_diagnostics.csv")
    if ud is None:
        return 0, 0
    q = ud[(ud.model == model) & (ud.dataset == ds.replace("_leakfree", "")) & (ud.train_arm == arm)]
    return int(q.degenerate.sum()), len(q)


def others_D(model, ds):
    """D (retrained + tested on equalized negatives) of every OTHER model on the same arm."""
    p = PRE + "_compare.csv"
    if not os.path.exists(p):
        return ""
    c = pd.read_csv(p)
    c = c[(c.dataset == ds) & (c.model != model) & c.D_auroc_mean.notna()]
    return "; ".join(f"{SHORT.get(x.model, x.model)} {x.D_auroc_mean:.3f} [{x.D_auroc_lo:.3f},{x.D_auroc_hi:.3f}]"
                     for x in c.itertuples())
LEVEL_LABELS = (("A", "orig->orig"), ("B", "orig->comp_eq"), ("C", "orig->random"),
                ("D", "comp_eq->comp_eq"), ("E", "random->random"), ("F", "random->comp_eq"),
                ("G", "orig->gc_match"), ("H", "gc_match->gc_match"), ("I", "comp_eq->orig"),
                ("J", "random->orig"))


def f3(x):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def is_true(v):
    return v is True or str(v) == "True"


def model_block(w, r, sc):
    """Per model x dataset block (the pre-registered quantities + post hoc share lost)."""
    model, ds = r.model, r.dataset
    w("=" * 100)
    npar = (f"trainable {int(r.n_params_trainable):,} of {int(r.n_params_total):,} params"
            if not pd.isna(r.get("n_params_trainable")) else "no gradient training of the FM")
    w(f"{MNAME.get(model, model)} | {ds} | {npar} | seeds per train arm: {r.train_arms_seeds}")
    for key, lab in LEVEL_LABELS:
        if f"{key}_auroc_mean" in r and not pd.isna(r.get(f"{key}_auroc_mean")):
            w(f"  {key} {lab:18s} AUROC {r[f'{key}_fmt']:32s} | balanced-resample AUROC "
              f"{f3(r[f'{key}_balres_auroc'])}, bal.acc {f3(r[f'{key}_balres_bal_acc'])}, "
              f"MCC {f3(r[f'{key}_balres_mcc'])} | AP {f3(r[f'{key}_ap'])} vs prevalence "
              f"{f3(r[f'{key}_ap_baseline'])}")
    for key, lab in (("S", "S = A-B shortcut worth"), ("T", "T = C-B test-imbalance ctl"),
                     ("CmA", "C-A (should be ~0)"), ("U", "U = E-D train-size ctl"),
                     ("R", "R = (E-.5)/(A-.5)"), ("Sgc", "A-G (GC-matched test)")):
        if f"{key}_est" in r and not pd.isna(r.get(f"{key}_est")):
            w(f"  {lab:28s} {r[f'{key}_fmt']}   (seed SD {f3(r[f'{key}_seed_sd'])})")
    if "frac_lost" in r and not pd.isna(r.get("frac_lost")):
        w(f"  share of above-chance AUROC lost on comp_eq negatives, (A-B)/(A-.5) [post hoc]: "
          f"{r.frac_lost:.3f} [{r.frac_lost_lo:.3f},{r.frac_lost_hi:.3f}]; on size-matched "
          f"random negatives (A-C)/(A-.5): {r.frac_lost_random:.3f} "
          f"[{r.frac_lost_random_lo:.3f},{r.frac_lost_random_hi:.3f}]")
    for col in [c for c in r.index if str(c).startswith("rule")]:
        if not pd.isna(r.get(col)):
            w(f"  {col}: {r[col]}")
    if sc is not None:
        s = sc[(sc.model == model) & (sc.dataset == ds)]
        for _, q in s.iterrows():
            w(f"  score-composition coupling, train arm {q.train_arm:14s}: CV R^2 {q.r2_cv_mean:.3f}; "
              f"composition projection AUROC {q.auroc_comp_projection_mean:.3f} vs model "
              f"{q.auroc_model_mean:.3f} on the released test split"
              + (f" (recovers {q.frac_above_chance_recovered_mean:.2f} of above-chance AUROC)"
                 if q.auroc_model_mean > 0.55 else " (model near chance; ratio not interpretable)"))


def how_to_read(w, r, posthoc=False):
    if pd.isna(r.get("S_est")):
        return
    n = int(r.A_n_seeds) if not pd.isna(r.get("A_n_seeds")) else 0
    txt = (f"  {r.model}/{r.dataset} ({n} seed{'s' if n != 1 else ''}"
           + (f"; B arm {int(r.B_n_test)} sequences" if not pd.isna(r.get("B_n_test")) else "")
           + (", POST HOC derived test set" if posthoc else "") + "): the released-split model "
           f"scores {r.A_auroc_mean:.3f}; the same weights score {r.B_auroc_mean:.3f} "
           f"[{r.B_auroc_lo:.3f},{r.B_auroc_hi:.3f}] against composition-equalized negatives "
           f"(S = {r.S_est:.3f} [{r.S_lo:.3f},{r.S_hi:.3f}])")
    if not pd.isna(r.get("frac_lost")):
        txt += (f", i.e. {100 * r.frac_lost:.0f}% [{100 * r.frac_lost_lo:.0f},"
                f"{100 * r.frac_lost_hi:.0f}] of its above-chance AUROC [share lost: post hoc, descriptive],")
    if not pd.isna(r.get("C_auroc_mean")):
        txt += (f" but {r.C_auroc_mean:.3f} against an equally small, equally outnumbered random "
                f"subset of the released negatives (T = {r.T_est:.3f} [{r.T_lo:.3f},{r.T_hi:.3f}])")
        r2 = "rule 2 re-applied post hoc" if posthoc else "rule 2"
        txt += (f"; {r2} is met: the drop is not a test-size/imbalance artifact."
                if is_true(r.get("rule2_drop_not_test_imbalance"))
                else f"; {r2} is NOT met, so a test-size/imbalance contribution is not excluded.")
    else:
        txt += "."
    r1 = str(r.get("rule1_rides_shortcut"))
    covers = r.B_auroc_lo <= 0.5 <= r.B_auroc_hi
    lab1 = "Rule 1 re-applied post hoc (not a pre-registered verdict)" if posthoc else "Pre-registered rule 1"
    if r1 == "True":
        txt += (f" {lab1} is met" + (": B's CI covers 0.5." if covers else
                f" only through its 'B < 0.6' clause: B's CI [{r.B_auroc_lo:.3f},{r.B_auroc_hi:.3f}] excludes "
                "0.5, so quote B with its CI and do not call it 'at chance'."))
        if posthoc and covers:
            txt += (" With so few equalized negatives a CI covering 0.5 is weak evidence of chance by itself; "
                    "see the random-draw reference (Part 3, Q4) and the baselines' B on the same set.")
    elif r1 == "False" and r.S_lo <= 0:
        txt += (f" {lab1} is NOT met: S's CI includes 0, so no shortcut reliance is detected.")
    elif r1 == "False" and r.dataset == "drosophila_enhancers_stark":
        txt += (f" {lab1} is NOT met (B's CI lies above 0.5), as expected on the GC-balanced control; "
                "its pre-registered test is rule 5.")
    elif r1 == "False":
        txt += (f" {lab1} is NOT met: B >= 0.6 with a CI above 0.5, so this model relies on "
                "composition for part of its score but not for all of it ('fine-tuning reduces but does "
                "not remove reliance on composition').")
    if not pd.isna(r.get("U_est")):
        nd, nu = degenerate_seeds(r.model, r.dataset, "comp_equalized")
        txt += (f" Retrained under the same budget, random negatives give {r.E_auroc_mean:.3f} and "
                f"equalized negatives {r.D_auroc_mean:.3f} [{r.D_auroc_lo:.3f},{r.D_auroc_hi:.3f}] "
                f"(U = {r.U_est:.3f} [{r.U_lo:.3f},{r.U_hi:.3f}]).")
        if nu and nd == nu:
            txt += (f" D is NOT a learnability measure here: all {nu} equalized-arm units are degenerate "
                    "fits (every test probability on one side of 0.5, near-constant scores, validation AUROC "
                    "< 0.6 at every epoch; fm_finetune_v2_unit_diagnostics.csv), i.e. this recipe collapsed "
                    "on the heavily imbalanced equalized train arm. Other models learn the same arm: "
                    f"{others_D(r.model, r.dataset)} (fm_finetune_v2_compare.csv). U and rule 3 therefore "
                    "partly reflect this optimization collapse.")
        elif nd:
            txt += (f" {nd} of {nu} equalized-arm units are degenerate fits (fm_finetune_v2_unit_diagnostics.csv); "
                    "read D with care.")
        elif r.D_auroc_lo > 0.5:
            txt += (" D's CI excludes 0.5: part of the signal is learnable without the "
                    "composition gap on this dataset.")
        else:
            txt += " D's CI includes 0.5: no learnable signal is detected once composition is equalized."
        if not pd.isna(r.get("F_auroc_mean")):
            share = (r.E_auroc_mean - r.F_auroc_mean) / (r.E_auroc_mean - 0.5)
            txt += (f" The random-trained model scores {r.F_auroc_mean:.3f} "
                    f"[{r.F_auroc_lo:.3f},{r.F_auroc_hi:.3f}] on equalized negatives vs "
                    f"{r.E_auroc_mean:.3f} on its own test, i.e. {100 * share:.0f}% of what it "
                    f"learned from a few hundred released-composition negatives does not transfer "
                    f"to composition-equalized ones.")
    w(txt)


def cmp_word(lo, hi, less, more, same="not distinguishable from"):
    return less if hi < 0 else (more if lo > 0 else same)


def summary_block(w):
    """Writer-facing answer to R1.2. Pure re-arrangement of numbers read from fm_finetune_v2_compare.csv
    (pre-registered rows for the released test splits, post hoc rows for the leak-free nt_enhancers test
    and the baselines), fm_finetune_v2_compare_paired.csv and, from fix round 1 on, the post hoc
    sensitivity CSVs of sensitivity_ft_v2.py. Every item is tagged [pre] or [post hoc]."""
    comp = pd.read_csv(PRE + "_compare.csv")
    paired = pd.read_csv(PRE + "_compare_paired.csv")
    ftm = [m for m in ("hyenadna", "dnabert2", "dnabert2full", "ntv2_500m") if (comp.model == m).any()]
    rd, cs, en = load_opt("_randdraw.csv"), load_opt("_ccre_sensitivity.csv"), load_opt("_ccre_enrichment.csv")
    gp, mu, wc = load_opt("_gc_paired.csv"), load_opt("_multiplicity.csv"), load_opt("_score_composition_withinclass_summary.csv")

    def row(m, ds):
        x = comp[(comp.model == m) & (comp.dataset == ds)]
        return None if x.empty else x.iloc[0]

    def pct(v):
        return f"{100 * v:.0f}%"

    def ci(r, k):
        return f"[{r[k + '_lo']:.3f},{r[k + '_hi']:.3f}]"

    w("SUMMARY FOR THE WRITER -- does end-to-end fine-tuning remove the composition shortcut? "
      "[pre] = pre-registered (Part 1), [post hoc] = Parts 2-3 (no decision rule attached). A = released->"
      "released AUROC, B = same weights on composition-equalized negatives, C = same weights on "
      "size-matched random negatives, D/E = retrained on equalized/random negatives. S = A-B and T = C-B "
      "are pre-registered; the share lost, (A-X)/(A-0.5), is a post hoc descriptive re-expression of them.")
    w(" (1) Same weights, three test sets:")
    for m in ftm:
        for ds, tag in (("human_enhancers_cohn", "[pre]"), ("nt_enhancers", "[pre]"),
                        ("nt_enhancers_leakfree", "[post hoc]"), ("drosophila_enhancers_stark", "[pre]")):
            r = row(m, ds)
            if r is None or pd.isna(r.get("S_est")):
                continue
            covers = r.B_auroc_lo <= 0.5 <= r.B_auroc_hi
            txt = (f"  {SHORT[m]:18s} {DSN.get(ds, ds):44s} {tag:10s} A {r.A_auroc_mean:.3f} -> B "
                   f"{r.B_auroc_mean:.3f} {ci(r, 'B_auroc')} (n={int(r.B_n_test)}), S {r.S_est:.3f} {ci(r, 'S')}; "
                   f"C {r.C_auroc_mean:.3f}, T {r.T_est:.3f} {ci(r, 'T')}; share lost [post hoc] "
                   f"{pct(r.frac_lost)} [{pct(r.frac_lost_lo)},{pct(r.frac_lost_hi)}] vs random "
                   f"{pct(r.frac_lost_random)} [{pct(r.frac_lost_random_lo)},{pct(r.frac_lost_random_hi)}]")
            if tag == "[pre]":
                r1 = str(r.get("rule1_rides_shortcut"))
                txt += (f"; rule 1 {r1}" + ("" if r1 != "True" else
                        (" (B's CI covers 0.5)" if covers else " (only via 'B < 0.6'; B's CI excludes 0.5)"))
                        + f", rule 2 {r.get('rule2_drop_not_test_imbalance')}")
                if ds == "drosophila_enhancers_stark":
                    txt += f", rule 5 {r.get('rule5_aligned_control_not_destroyed')}"
            else:
                pre_r = row(m, "nt_enhancers")
                bl = comp[(comp.dataset == ds) & (~comp.model.isin(ftm)) & comp.B_auroc_lo.notna()]
                ncov = int(((bl.B_auroc_lo <= 0.5) & (bl.B_auroc_hi >= 0.5)).sum())
                txt += (f"; no verdict (post hoc set; the pre-registered rule-1 verdict on the released split is "
                        f"{pre_r.get('rule1_rides_shortcut') if pre_r is not None else 'n/a'}); B's CI covers 0.5 "
                        f"for {ncov}/{len(bl)} baselines on this set too, so a covering CI is weak evidence here")
            if rd is not None:
                q = rd[(rd.model == m) & (rd.dataset == ds)]
                if len(q):
                    q = q.iloc[0]
                    txt += (f"; B vs {int(q.n_draws)} random {int(q.k_neg)}-negative draws: 2.5-97.5% "
                            f"{q.draw_p2_5:.3f}-{q.draw_p97_5:.3f}, min {q.draw_min:.3f}, "
                            + (f"B below all draws (p = {q.p_one_sided_B:.3f})" if int(q.n_draws_le_B) == 0
                               else f"one-sided p {q.p_one_sided_B:.3f}") + " [post hoc]")
            w(txt)
    w(" (2) Retrained on equalized (D) vs random (E) negatives [pre]:")
    for m in ftm:
        for ds in ("human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark"):
            r = row(m, ds)
            if r is None or pd.isna(r.get("D_auroc_mean")):
                continue
            nd, nu = degenerate_seeds(m, ds, "comp_equalized")
            txt = (f"  {SHORT[m]:18s} {DSN.get(ds, ds):44s} D {r.D_auroc_mean:.3f} {ci(r, 'D_auroc')} vs E "
                   f"{r.E_auroc_mean:.3f}; U {r.U_est:.3f} {ci(r, 'U')}; rule 3: {r.get('rule3_collapse_cause')}")
            if nu and nd == nu:
                txt += (f". DEGENERATE: {nd}/{nu} equalized-arm units are one-class, near-constant fits "
                        "(unit_diagnostics); D is an optimization collapse of this recipe, not evidence that the arm "
                        f"is unlearnable (same arm: {others_D(m, ds)})")
            w(txt)
    psm = os.path.join(RES, "psm_sensitivity_decision.csv")
    if os.path.exists(psm):
        q = pd.read_csv(psm).set_index("dataset")
        if "nt_enhancers" in q.index:
            x = q.loc["nt_enhancers"]
            w(f"     Retrain-side caveat for nt_enhancers: psm_sensitivity's own matching gives verdict "
              f"'{x.verdict}' for the retrained k-mer model (matched {x.Am:.3f} vs size-matched random "
              f"{x.Ar_mean:.3f}), while its fixed-model test agrees with B-vs-C here (matched test "
              f"{x.fixed_A_matchedTest:.3f} vs random test {x.fixed_A_randomTest_mean:.3f}). So rest the "
              "nt_enhancers fine-tuning claim on the same-weights B vs C contrast, not on D/E, U or rule 3.")
    if en is not None and cs is not None:
        w(" (3) Latent positives (ENCODE cCRE v4) [post hoc]: the equalizer keeps negatives that overlap "
          "regulatory elements more often than the size-matched random control does:")
        for ds in ("human_enhancers_cohn", "nt_enhancers"):
            for fl in ("ccre_any", "ccre_els"):
                e = en[(en.dataset == ds) & (en.split == "test") & (en.feature == fl) & (en.arm == "comp_equalized")]
                if e.empty:
                    continue
                e = e.iloc[0]
                w(f"  {ds:22s} test {fl}: equalized negatives {e.rate_neg:.3f} ({int(e.n_neg_overlap)}/"
                  f"{int(e.n_neg_annotated)}) vs random {e.rate_vs_random:.3f} (Fisher p {e.fisher_p_vs_random:.3g}) "
                  f"vs released negatives not kept {e.rate_vs_released_not_kept:.3f} (p "
                  f"{e.fisher_p_vs_released_not_kept:.3g}); positives {e.rate_pos:.3f}")
        w("  Re-scored with every cCRE-overlapping (or ELS-overlapping) test negative removed (training unchanged):")
        for m in ftm:
            for ds in ("human_enhancers_cohn", "nt_enhancers", "nt_enhancers_leakfree"):
                for fl in ("ccre_any_free", "ccre_els_free"):
                    q = cs[(cs.model == m) & (cs.dataset == ds) & (cs["filter"] == fl)]
                    if q.empty:
                        continue
                    q = q.iloc[0]
                    w(f"    {SHORT[m]:18s} {DSN.get(ds, ds):44s} {fl:14s} A {q.A_est:.3f} -> B {q.B_est:.3f} "
                      f"[{q.B_lo:.3f},{q.B_hi:.3f}] (neg {int(q.B_n_neg)}), C {q.C_est:.3f}; S {q.S_est:.3f} "
                      f"[{q.S_lo:.3f},{q.S_hi:.3f}]; share lost {pct(q.frac_lost)} [{pct(q.frac_lost_lo)},"
                      f"{pct(q.frac_lost_hi)}] (unfiltered {pct(q.unfiltered_frac_lost)}), random "
                      f"{pct(q.frac_lost_random)} [{pct(q.frac_lost_random_lo)},{pct(q.frac_lost_random_hi)}]")
        k = cs[(cs.dataset == "human_enhancers_cohn") & (cs["filter"] == "ccre_any_free")]
        w("    cohn, cCRE-free, every model (share lost; unfiltered in brackets): " + "; ".join(
            f"{SHORT.get(x.model, x.model)} {pct(x.frac_lost)} [{pct(x.frac_lost_lo)},{pct(x.frac_lost_hi)}] "
            f"({pct(x.unfiltered_frac_lost)})" for x in k.itertuples()))
    w(" (4) Same shortcut as the paper's frozen probes and k-mer model? Paired differences in share lost, "
      "fine-tuned minus baseline on identical arms and resamples [post hoc]:")
    for m in ftm:
        for ds in ("human_enhancers_cohn", "nt_enhancers_leakfree", "drosophila_enhancers_stark"):
            p = paired[(paired.finetuned == m) & (paired.dataset == ds) & (paired.quantity == "frac_lost")]
            if p.empty:
                continue
            w(f"    {SHORT[m]:18s} {DSN.get(ds, ds):44s} " + "; ".join(
                f"vs {SHORT.get(x.baseline, x.baseline)} {x.difference:+.2f} [{x.ci_lo:+.2f},{x.ci_hi:+.2f}] "
                f"({cmp_word(x.ci_lo, x.ci_hi, 'less', 'more')})" for x in p.itertuples()))
    if gp is not None:
        w(" (5) GC-matched test effect G-A (the paper's frozen test_effect, same weights) [post hoc paired]:")
        lv = gp[gp.kind == "level"]
        for m in ftm:
            for ds in ("human_enhancers_cohn", "nt_enhancers", "nt_enhancers_leakfree", "drosophila_enhancers_stark"):
                x = lv[(lv.model == m) & (lv.dataset == ds)]
                if x.empty:
                    continue
                x = x.iloc[0]
                pp = gp[(gp.kind == "paired_diff") & (gp.model == m) & (gp.dataset == ds)]
                w(f"    {SHORT[m]:18s} {DSN.get(ds, ds):44s} G-A {x.G_minus_A:+.3f} [{x.ci_lo:+.3f},{x.ci_hi:+.3f}]; "
                  + "; ".join(f"minus {SHORT.get(y.baseline, y.baseline)} ({y.baseline_value:+.3f}) "
                              f"{y.G_minus_A:+.3f} [{y.ci_lo:+.3f},{y.ci_hi:+.3f}] "
                              f"({cmp_word(y.ci_lo, y.ci_hi, 'larger drop', 'smaller drop', 'similar')})"
                              for y in pp.itertuples()))
    if wc is not None:
        q = wc[(wc.train_arm == "original") & wc.model.isin(ftm)]
        w(" (6) Score-composition coupling beyond the label [post hoc; CV R^2 of the logit on 20-d composition, "
          "released-trained models, released test]: " + "; ".join(
              f"{SHORT[x.model]}/{dshort(x.dataset)} pooled {x.r2_pooled_mean:.2f}, within-pos "
              f"{x.r2_within_pos_mean:.2f}, within-neg {x.r2_within_neg_mean:.2f}, gain over label "
              f"{x.partial_gain_mean:.2f}" for x in q.itertuples()))
    if mu is not None:
        s = mu[mu.quantity == "S"]
        w(f" (7) Multiplicity [post hoc]: {int(s.n_cells.iloc[0])} pre-registered fine-tuned cells; S's "
          f"Bonferroni ({100 * s.bonferroni_level.iloc[0]:.0f}%) CI excludes 0 in {int(s.excludes_0_bonf.sum())}/"
          f"{len(s)}: " + "; ".join(f"{SHORT[x.model]}/{dshort(x.dataset)} [{x.ci_bonf_lo:.3f},"
                                     f"{x.ci_bonf_hi:.3f}]" for x in s.itertuples())
          + "; T: " + "; ".join(f"{SHORT[x.model]}/{dshort(x.dataset)} [{x.ci_bonf_lo:.3f},{x.ci_bonf_hi:.3f}]"
                                for x in mu[mu.quantity == "T"].itertuples()))
    w("  Reading rules fixed before the results (gpu_deferred/README.md): rule 1 met -> 'the fine-tuned "
      "model rides the shortcut'; rule 1 not met but S > 0 -> 'fine-tuning reduces but does not remove "
      "reliance on composition'. Leak-free nt_enhancers numbers, the baseline comparison and Part 3 are post hoc.")
    w("")


def main():
    tab = pd.read_csv(PRE + "_paper_table.csv")
    rt = pd.read_csv(PRE + "_runs.csv")
    sc = pd.read_csv(PRE + "_score_composition_summary.csv")
    vv = pd.read_csv(PRE + "_v1_vs_v2.csv") if os.path.exists(PRE + "_v1_vs_v2.csv") else None
    man = pd.read_csv(PRE + "_split_manifest.csv")
    have_ph = os.path.exists(PRE + "_compare.csv")
    L = []
    w = L.append
    w("fm_finetune v2 -- END-TO-END FINE-TUNING vs THE NEGATIVE-SET COMPOSITION SHORTCUT")
    w("(camera-ready response to Reviewer 1, comment 2)")
    w(f"generated {time.strftime('%Y-%m-%d %H:%M')} by src/upgrades/fm_finetune/"
      "write_interpretation_ft_v2.py; every number below is read from results/upgrades/"
      "fm_finetune_v2_{paper_table,summary,runs,score_composition_summary,v1_vs_v2,split_manifest,"
      "preds_check,compare,compare_paired,posthoc_*,baselines_repro}.csv or the cited module CSVs")
    w("")
    w("STATUS (trained units; per-sequence predictions in results/upgrades/fm_finetune_v2_preds/)")
    for (model, ds), (arms, nseed, where) in PLAN.items():
        sub = rt[(rt.model == model) & (rt.dataset == ds)]
        parts = []
        for a in arms:
            s = sorted(sub[sub.train_arm == a].seed.tolist())
            parts.append(f"{a}: {len(s)}/{nseed} seeds" + (f" ({','.join(map(str, s))})" if s else ""))
        tag = "" if where == "local" or len(sub) else f"   [PENDING: {where}, gpu_deferred/]"
        w(f"  {model:12s} {ds:27s} " + "; ".join(parts) + tag)
    pc = PRE + "_preds_check.csv"
    if os.path.exists(pc):
        q = pd.read_csv(pc)
        w(f"  integrity (verify_preds_ft_v2.py): {int(q.ok.sum())}/{len(q)} units pass (test rows = every "
          f"released test sequence, labels = verified split files, validation rows inside the arm's "
          f"train set, best epoch = arg-max validation AUROC, stored probabilities reproduce the "
          f"trainer's logged test AUROC to <= {q.max_abs_diff_vs_logged_test_auroc.max():.1e})")
    w("")
    if have_ph:
        summary_block(w)
    w("DESIGN. Same positives in every arm; only the negatives change. Arms: released split; "
      "composition-equalized (GC + 16-dinucleotide propensity-matched negatives, "
      "cleaned_splits_v2; the paper's published equalized split); RANDOM control = negatives drawn "
      "uniformly at random (seeded) to EXACTLY the equalized arm's negative count in train and test, "
      "i.e. the same size and class imbalance but the released composition gap; gc_match (HyenaDNA "
      "only) = the paper's GC-matched split. Every model scores every sequence of the released test "
      "split, so 'orig->X' is the same weights on arm X's test subset. Notation (seed-mean test "
      "AUROC): A orig->orig, B orig->comp_eq, C orig->random, D comp_eq->comp_eq, E random->random, "
      "F random->comp_eq, G orig->gc_match. S=A-B (shortcut worth), T=C-B (test size/imbalance "
      "control), U=E-D (train size/imbalance control), R=(E-.5)/(A-.5). CIs: 95% percentile, 1000 "
      "bootstrap resamples of the released test sequences (seed 42), seeds averaged inside each "
      "resample; +- is the SD over seeds 42/43/44. Decision rules 1-4 were written in "
      "eval_ft_v2.py at 22:00 EDT 2026-09-23, rule 5 and the score-composition analysis at 22:26:57 "
      "(agent transcript; the docstring's own label says '22:30'); the first unit finished at 22:33:24 "
      "(fm_finetune_v2_run.log) and the first drosophila unit started at 08:53 the next day. The share "
      "lost, the leak-free analysis, the baselines and everything in Part 3 are post hoc. Imbalance-robust metrics for every "
      "cell: AUROC (prevalence-free), 20 class-balanced resamples (AUROC, balanced accuracy, MCC at "
      "p=0.5), and PR-AUC (AP) next to its prevalence baseline; all columns in "
      "fm_finetune_v2_summary.csv / _metrics.csv.")
    w("")
    w("Composition-only readout of each arm (20-d mono+dinucleotide LR fit on the arm's train, "
      "AUROC on its test; manifest comp_only_lr_auroc_test):")
    for ds in man.dataset.unique():
        m = man[(man.dataset == ds) & (man.split == "test")].set_index("arm")
        w(f"  {ds:27s} " + ", ".join(f"{a} {m.loc[a, 'comp_only_lr_auroc_test']:.3f} "
                                      f"(test {int(m.loc[a, 'n_pos'])}+/{int(m.loc[a, 'n_neg'])}-)"
                                      for a in ("original", "random", "comp_equalized")))
    w("")
    w("#" * 100)
    w("PART 1 -- PRE-REGISTERED ANALYSIS (eval_ft_v2.py, unchanged), released test splits")
    w("#" * 100)
    for _, r in tab.iterrows():
        model_block(w, r, sc)
    w("=" * 100)
    w("")
    if os.path.exists(PRE + "_model_contrast.csv"):
        mc = pd.read_csv(PRE + "_model_contrast.csv")
        w("BETWEEN FINE-TUNED MODELS (post hoc, descriptive; same test sequences, paired bootstrap):")
        for _, q in mc.iterrows():
            w(f"  {q.dataset}: {q.contrast} = {q.estimate:.3f} [{q.ci_lo:.3f},{q.ci_hi:.3f}] "
              f"({q.model_1} {q.value_1:.3f}, seeds {q.seeds_1}; {q.model_2} {q.value_2:.3f}, "
              f"seeds {q.seeds_2})")
        w("")
    if vv is not None:
        w("v1 (run_fm_finetune.py, cited in the June/v2 manuscript) vs v2 re-run, HyenaDNA on cohn. "
          "v1 fed nearly single-class mini-batches (class-sorted lists + stable length sort on "
          "equal-length 500 bp sequences) and saved no predictions; v2 permutes before the "
          "length sort, otherwise the identical recipe:")
        for _, q in vv.iterrows():
            w(f"  {q.train_arm:>14s} -> {q.eval_arm:14s} v1 {q.v1_auroc_mean:.3f} (SD {q.v1_auroc_sd:.3f}, "
              f"n={int(q.v1_n_seeds)})  v2 " + (f"{q.v2_auroc_mean:.3f} (SD {q.v2_auroc_sd:.3f}, "
                                                 f"n={int(q.v2_n_seeds)})" if q.v2_n_seeds else "pending"))
        w("")
    w("HOW TO READ IT (pre-registered quantities)")
    for _, r in tab.iterrows():
        how_to_read(w, r)
    w("  Score-composition coupling: CV R^2 is the reliance measure (share of the logit's variance "
      "that is a linear function of 20-d composition). The projection-AUROC ratio is bounded by how "
      "discriminative composition itself is on the released test split (composition-only AUROC "
      "listed above), so it can be high even at low R^2 (e.g. a model trained on equalized "
      "negatives); ratios >1 mean the projection separates the classes better than the logit.")
    w("")

    if have_ph:
        comp = pd.read_csv(PRE + "_compare.csv")
        paired = pd.read_csv(PRE + "_compare_paired.csv")
        ph = pd.read_csv(PRE + "_posthoc_paper_table.csv")
        w("#" * 100)
        w("PART 2 -- POST HOC (added 2026-09-26 after all 54 units were scored; posthoc_ft_v2.py)")
        w("#" * 100)
        dups = pd.read_csv(PRE + "_leakfree_dups.csv")
        w(f"P1. LEAK-FREE nt_enhancers. {len(dups)} of the 400 released test sequences (all label "
          f"{sorted(set(dups.label))}) are verbatim training sequences (nt_enhancers_duplication.csv; "
          f"re-derived by exact match, fm_finetune_v2_leakfree_dups.csv). Dropping them leaves 300 "
          f"test sequences (100+/200-; comp_eq and random arms 100+/36-). Same computations, "
          f"bootstrap over the 300 leak-free sequences:")
        for _, r in ph[(ph.dataset == "nt_enhancers_leakfree") & ph.model.isin(
                ["hyenadna", "dnabert2", "dnabert2full", "ntv2_500m"])].iterrows():
            model_block(w, r, None)
            how_to_read(w, r, posthoc=True)
        w("=" * 100)
        w("")
        w("P2/P3. SAME ARMS, THREE MODEL FAMILIES. The paper's k-mer LightGBM and frozen-FM LightGBM "
          "heads re-fit on exactly these train arms (baselines_ft_v2.py; one deterministic seed, 42) and "
          "scored by the same code on the same test sequences and bootstrap resamples.")
        rp = PRE + "_baselines_repro.csv"
        if os.path.exists(rp):
            q = pd.read_csv(rp)
            w(f"  Reproduction check (fm_finetune_v2_baselines_repro.csv): {int(q['match_1e-3'].sum())}/"
              f"{len(q)} published numbers reproduced to 1e-3 (max |diff| {q.abs_diff.max():.4f}); "
              "e.g. " + "; ".join(f"{x.check.split(' vs ')[0]} {dshort(x.dataset)} "
                                   f"{x.ours:.4f} vs {x.published:.4f}" for x in q.head(4).itertuples()))
        w("  Columns: A released->released | C same weights, size-matched random negatives | B same "
          "weights, composition-equalized negatives | D retrained+tested on equalized | E retrained+"
          "tested on random | share lost (A-B)/(A-.5) [CI] | share lost on random negatives (A-C)/(A-.5).")
        for ds in comp.dataset.unique():
            w(f"  -- {DSN.get(ds, ds)}")
            for _, r in comp[comp.dataset == ds].iterrows():
                if pd.isna(r.get("frac_lost")):
                    continue
                pm = "" if pd.isna(r.A_auroc_sd) else f"±{r.A_auroc_sd:.3f}"
                w(f"    {SHORT.get(r.model, r.model):24s} A {r.A_auroc_mean:.3f}{pm:7s} C {r.C_auroc_mean:.3f} "
                  f"B {r.B_auroc_mean:.3f} [{r.B_auroc_lo:.3f},{r.B_auroc_hi:.3f}] "
                  f"D {r.D_auroc_mean:.3f} E {r.E_auroc_mean:.3f} | lost {r.frac_lost:.2f} "
                  f"[{r.frac_lost_lo:.2f},{r.frac_lost_hi:.2f}] vs random {r.frac_lost_random:.2f} "
                  f"[{r.frac_lost_random_lo:.2f},{r.frac_lost_random_hi:.2f}]"
                  + ("" if pd.isna(r.get("Sgc_est")) else f" | A-G {r.Sgc_est:.3f}"))
        w("  Paired differences in share lost, fine-tuned minus baseline (same resamples; post hoc, "
          "descriptive; fm_finetune_v2_compare_paired.csv):")
        pf = paired[paired.quantity == "frac_lost"]
        for (ds, f), s in pf.groupby(["dataset", "finetuned"], sort=False):
            w(f"    {DSN.get(ds, ds):40s} {SHORT.get(f, f):18s} " + "; ".join(
                f"- {SHORT.get(x.baseline, x.baseline)} {x.difference:+.2f} [{x.ci_lo:+.2f},{x.ci_hi:+.2f}]"
                for x in s.itertuples()))
        phs = PRE + "_posthoc_score_composition_summary.csv"
        if os.path.exists(phs):
            q = pd.read_csv(phs)
            q = q[q.train_arm == "original"]
            w("  Score-composition coupling of the released-split baselines (CV R^2 of logit on 20-d "
              "composition; fine-tuned models in Part 1): " + "; ".join(
                  f"{SHORT.get(x.model, x.model)}/{dshort(x.dataset)} {x.r2_cv_mean:.2f}"
                  for x in q.itertuples()))
        w("")
    # ---- Part 3: fix-round-1 sensitivity analyses (sensitivity_ft_v2.py), full tables
    ud, gp, en, cs = (load_opt("_unit_diagnostics.csv"), load_opt("_gc_paired.csv"),
                      load_opt("_ccre_enrichment.csv"), load_opt("_ccre_sensitivity.csv"))
    rd, wc, mu = (load_opt("_randdraw.csv"), load_opt("_score_composition_withinclass_summary.csv"),
                  load_opt("_multiplicity.csv"))
    if ud is not None:
        w("#" * 100)
        w("PART 3 -- POST HOC SENSITIVITY ANALYSES (fix round 1, 2026-09-26; sensitivity_ft_v2.py). Scratch "
          "versions of Q2-Q5 had been computed by the verifiers before the script was written; the definitions "
          "are theirs, extended to every model and dataset; no decision rule is attached.")
        w("#" * 100)
        w("Q1. Unit diagnostics (fm_finetune_v2_unit_diagnostics.csv). Degenerate = every test probability on "
          "one side of 0.5 AND test-probability SD < 0.01 AND max validation AUROC < 0.6:")
        dg = ud[ud.degenerate]
        w(f"  {len(dg)} of {len(ud)} fine-tuning units are degenerate: " + ("; ".join(
            f"{x.model}/{x.dataset}/{x.train_arm}/s{x.seed} (p>=0.5 for {100 * x.frac_test_p_ge_0_5:.0f}%, "
            f"SD {x.test_prob_sd:.4f}, max val AUROC {x.max_val_auroc:.3f})" for x in dg.itertuples()) or "none"))
        oc = ud[ud.one_class & ~ud.degenerate]
        if len(oc):
            w(f"  one-class but not degenerate (scores still rank; AUROC informative, thresholded metrics not): "
              + "; ".join(f"{x.model}/{x.dataset}/{x.train_arm}/s{x.seed} (SD {x.test_prob_sd:.3f}, own-arm "
                          f"AUROC {x.own_arm_test_auroc:.3f})" for x in oc.itertuples()))
        v = ud.groupby(["model", "dataset"]).agg(n_val=("n_val", "mean"), leak=("frac_val_verbatim_in_fit", "mean"),
                                                 val=("best_val_auroc", "mean"), test=("own_arm_test_auroc", "mean"))
        w("  validation-carve leakage (share of validation rows whose exact sequence is also a fitting row; mean "
          "over units) and best validation vs own-arm test AUROC: " + "; ".join(
              f"{m}/{dshort(d)} {x.leak:.3f} (val {x.val:.3f} vs test {x.test:.3f})"
              for (m, d), x in v.iterrows()))
        w("")
    if gp is not None:
        w("Q2. GC-matched test effect G - A (same weights; the paper's test_effect sign) and fine-tuned minus "
          "baseline paired differences (fm_finetune_v2_gc_paired.csv):")
        for ds in gp.dataset.unique():
            lv = gp[(gp.kind == "level") & (gp.dataset == ds)]
            w(f"  -- {DSN.get(ds, ds)}: " + "; ".join(f"{SHORT.get(x.model, x.model)} {x.G_minus_A:+.3f} "
                                                   f"[{x.ci_lo:+.3f},{x.ci_hi:+.3f}]" for x in lv.itertuples()))
            pd_ = gp[(gp.kind == "paired_diff") & (gp.dataset == ds)]
            for f_, q in pd_.groupby("model", sort=False):
                w(f"     {SHORT.get(f_, f_)} minus: " + "; ".join(
                    f"{SHORT.get(x.baseline, x.baseline)} {x.G_minus_A:+.3f} [{x.ci_lo:+.3f},{x.ci_hi:+.3f}]"
                    for x in q.itertuples()))
        w("")
    if en is not None and cs is not None:
        w("Q3. Latent positives: ENCODE SCREEN cCRE v4 overlap (psm_sensitivity annotation cache, read-only; "
          "fm_finetune_v2_ccre_enrichment.csv). Share of negatives overlapping, per arm (released / GC-matched / "
          "equalized / random) and positives; Fisher exact p, equalized vs random and vs released-not-kept:")
        for (ds, sp, fl), q in en.groupby(["dataset", "split", "feature"], sort=False):
            q = q.set_index("arm")
            e = q.loc["comp_equalized"]
            w(f"  {ds:22s} {sp:5s} {fl:9s} released {q.loc['original', 'rate_neg']:.3f} (n {int(q.loc['original', 'n_neg_annotated'])}), "
              f"GC-matched {q.loc['gc_match', 'rate_neg']:.3f}, equalized {e.rate_neg:.3f} (n {int(e.n_neg_annotated)}), "
              f"random {q.loc['random', 'rate_neg']:.3f} (n {int(q.loc['random', 'n_neg_annotated'])}); positives "
              f"{e.rate_pos:.3f}; p(eq vs random) {e.fisher_p_vs_random:.3g}, p(eq vs not kept) "
              f"{e.fisher_p_vs_released_not_kept:.3g}")
        w("  Re-scoring with the cCRE- (ELS-) overlapping test negatives removed from every eval arm "
          "(fm_finetune_v2_ccre_sensitivity.csv; rule columns there are re-applications, not verdicts):")
        for (ds, fl), q in cs.groupby(["dataset", "filter"], sort=False):
            w(f"  -- {DSN.get(ds, ds)}, {fl} (A/B/C negatives {int(q.A_n_neg.iloc[0])}/{int(q.B_n_neg.iloc[0])}/"
              f"{int(q.C_n_neg.iloc[0])})")
            for x in q.itertuples():
                w(f"    {SHORT.get(x.model, x.model):24s} A {x.A_est:.3f} B {x.B_est:.3f} [{x.B_lo:.3f},{x.B_hi:.3f}] "
                  f"C {x.C_est:.3f} | S {x.S_est:.3f} [{x.S_lo:.3f},{x.S_hi:.3f}] T {x.T_est:.3f} [{x.T_lo:.3f},"
                  f"{x.T_hi:.3f}] | lost {x.frac_lost:.2f} [{x.frac_lost_lo:.2f},{x.frac_lost_hi:.2f}] (unfiltered "
                  f"{x.unfiltered_frac_lost:.2f}) vs random {x.frac_lost_random:.2f} [{x.frac_lost_random_lo:.2f},"
                  f"{x.frac_lost_random_hi:.2f}]")
        w("")
    if rd is not None:
        w("Q4. Random-draw reference for B (fm_finetune_v2_randdraw.csv): released-split weights on 1000 random "
          "draws of as many test negatives as the equalized arm holds (+ all positives; draws identical for every "
          "model); p = (1 + #draws <= B)/1001:")
        for x in rd.itertuples():
            w(f"  {DSN.get(x.dataset, x.dataset):44s} {SHORT.get(x.model, x.model):24s} k={x.k_neg:4d} draws "
              f"{x.draw_p2_5:.3f}-{x.draw_p97_5:.3f} (min {x.draw_min:.3f}); B {x.B:.3f} (p {x.p_one_sided_B:.4f}); "
              f"C {x.C:.3f} at the {100 * x.C_percentile_in_draws:.0f}th percentile of the draws")
        w("")
    if wc is not None:
        w("Q5. Within-class score-composition coupling, released test split, train arm original "
          "(fm_finetune_v2_score_composition_withinclass_summary.csv; pooled = the pre-registered R^2):")
        for x in wc[wc.train_arm == "original"].itertuples():
            w(f"  {SHORT.get(x.model, x.model):24s} {x.dataset:27s} pooled {x.r2_pooled_mean:.3f}, within-pos "
              f"{x.r2_within_pos_mean:.3f}, within-neg {x.r2_within_neg_mean:.3f}, label only "
              f"{x.r2_label_only_mean:.3f}, label+comp {x.r2_label_plus_comp_mean:.3f} (gain {x.partial_gain_mean:.3f})")
        w("")
    if mu is not None:
        w(f"Q6. Multiplicity (fm_finetune_v2_multiplicity.csv): Bonferroni over the {int(mu.n_cells.iloc[0])} "
          f"pre-registered fine-tuned cells ({100 * mu.bonferroni_level.iloc[0]:.0f}% percentile CIs, same 1000 "
          "resamples; a 99% bound rests on 5 tail draws):")
        for x in mu.itertuples():
            w(f"  {x.model:9s} {x.dataset:27s} {x.quantity:9s} {x.estimate:.3f} 95% [{x.ci95_lo:.3f},{x.ci95_hi:.3f}] "
              f"Bonferroni [{x.ci_bonf_lo:.3f},{x.ci_bonf_hi:.3f}]")
        w("")
    # ---- numbers the submitted paper reports, for the writer
    w("REFERENCE NUMBERS ALREADY IN THE PAPER / OTHER MODULES (read from their CSVs):")
    fca = os.path.join(RES, "fm_cross_architecture.csv")
    if os.path.exists(fca):
        q = pd.read_csv(fca)
        q = q[(q["head"] == "lgbm") & (((q.architecture.str.startswith("Hyena")) & (q.pooling == "sep"))
                                       | ((q.architecture.str.startswith("DNABERT")) & (q.pooling == "mean")))]
        w("  frozen-probe test_effect on the GC-matched test (paper Sec. FM probe; fm_cross_architecture.csv): "
          + "; ".join(f"{x.architecture.split('-')[0]}/{dshort(x.dataset)} {x.test_effect_auroc:+.3f} {x.ci}"
                      for x in q.itertuples()))
        for fm in ("hyenadna", "dnabert2", "dnabert2full", "ntv2_500m"):
            ft = tab[tab.model == fm]
            if not len(ft) or ft.Sgc_est.isna().all():
                continue
            w(f"  fine-tuned {SHORT[fm]} analog, G - A (same weights, same GC-matched test subset, "
              f"seed mean): " + "; ".join(
                  f"{dshort(x.dataset)} {-x.Sgc_est:+.3f} [{-x.Sgc_hi:+.3f},{-x.Sgc_lo:+.3f}]"
                  for x in ft.itertuples() if not pd.isna(x.Sgc_est)))
        if have_ph:
            pht = pd.read_csv(PRE + "_posthoc_paper_table.csv")
            b = pht[pht.model.isin(["hyenadna_frozen_mean", "hyenadna_frozen_sep", "dnabert2_frozen_mean",
                                    "dnabert2_frozen_cls"]) & pht.dataset.isin(
                ["human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark"])]
            w("  the same G - A for every frozen-probe cell (post hoc refit, reproduces fm_paired*.csv; "
              "the paper reports the [SEP] (HyenaDNA) and mean/[CLS] (DNABERT-2) cells): " + "; ".join(
                  f"{SHORT[x.model]}/{dshort(x.dataset)} {-x.Sgc_est:+.3f} "
                  f"[{-x.Sgc_hi:+.3f},{-x.Sgc_lo:+.3f}]" for x in b.itertuples() if not pd.isna(x.Sgc_est)))
    psm = os.path.join(RES, "psm_sensitivity_decision.csv")
    if os.path.exists(psm):
        q = pd.read_csv(psm).set_index("dataset")
        for ds in q.index:
            x = q.loc[ds]
            w(f"  psm_sensitivity k-mer LightGBM {ds}: A0 {x.A0:.3f}, size-matched random retrain "
              f"{x.Ar_mean:.3f} (10 draws {x.Ar_min:.3f}-{x.Ar_max:.3f}), matched retrain {x.Am:.3f} "
              f"[{x.Am_lo:.3f},{x.Am_hi:.3f}], published equalized split {x.paper_anchor_auroc:.3f}; "
              f"fixed model on matched test {x.fixed_A_matchedTest:.3f} vs random test "
              f"{x.fixed_A_randomTest_mean:.3f}; verdict {x.verdict}"
              + ("" if pd.isna(x.get("ccrefreeTest_fixed_Am", np.nan)) else
                 f"; cCRE-free test: fixed model matched {x.ccrefreeTest_fixed_Am:.3f} vs random "
                 f"{x.ccrefreeTest_fixed_Ar_mean:.3f}"))
    hc = os.path.join(RES, "homology_recluster_ft_contrasts.csv")
    if os.path.exists(hc):
        q = pd.read_csv(hc)
        s = q[q.contrast.isin(["A_orig_orig", "B_orig_compeq", "S_shortcut_worth"])]
        w("  homology_recluster (read-only; fine-tuned HyenaDNA on homology-clean re-splits, "
          "homology_recluster_ft_contrasts.csv): " + "; ".join(
              f"{dshort(x.dataset)} {x.contrast.split('_')[0]} {x.estimate:.3f} [{x.ci_lo:.3f},{x.ci_hi:.3f}]"
              for x in s.itertuples()))
    w("")
    w("CAVEATS")
    w("  * B is above 0.5 where its CI says so (a residual non-composition signal survives); the "
      "claim is the share of the score that composition carries, not that the model learns nothing "
      "else. Pre-registered rule 1 accepts B<0.6 as 'near chance'; read B's CI directly.")
    mt = man[(man.split == "test") & (man.arm == "comp_equalized")].set_index("dataset")
    mo = man[(man.split == "test") & (man.arm == "original")].set_index("dataset")
    w("  * Equalized test sets keep few negatives (" + ", ".join(
        f"{ {'human_enhancers_cohn': 'cohn', 'drosophila_enhancers_stark': 'drosophila'}.get(d, d)} "
        f"{int(mt.loc[d, 'n_neg'])}" for d in mt.index)
      + "; manifest n_neg), so their CIs are wide (nt_enhancers especially); AUROC is "
      "prevalence-invariant, and the balanced-resample metrics (20 draws, majority class subsampled) "
      "are given alongside.")
    if "nt_enhancers" in mt.index:
        w("  * nt_enhancers: 100 released test positives are verbatim training sequences. The released-"
          "split A is inflated for every model that memorizes them; Part 2 P1 gives leak-free numbers. "
          f"The nt equalized split is also not fully composition-balanced (composition-only AUROC "
          f"{mt.loc['nt_enhancers', 'comp_only_lr_auroc_test']:.3f} on its test, manifest), so nt is "
          "the weaker of the two unaligned datasets.")
    if "drosophila_enhancers_stark" in mo.index:
        d = "drosophila_enhancers_stark"
        w(f"  * The drosophila control is GC-balanced (GC-only AUROC {mo.loc[d, 'gc_auroc_test']:.3f} "
          f"on its released test) but not dinucleotide-balanced (20-d composition-only AUROC "
          f"{mo.loc[d, 'comp_only_lr_auroc_test']:.3f}; cohn {mo.loc['human_enhancers_cohn', 'comp_only_lr_auroc_test']:.3f}, "
          f"nt_enhancers {mo.loc['nt_enhancers', 'comp_only_lr_auroc_test']:.3f}; manifest), so call it a "
          "GC-balanced control, not composition-free. A model can therefore lose some AUROC on its equalized "
          "negatives too. What separates it from cohn is the smaller share the SAME weights lose on "
          "equalized negatives (see the summary), not D: D stays above chance on cohn too for every model "
          "except the degenerate fine-tuned HyenaDNA units (fm_finetune_v2_compare.csv D_auroc_mean). "
          "Pre-registered rule 5 was NOT met (S_dros's CI overlaps S on released nt_enhancers); its "
          "non-overlap criterion was fixed with this caveat stated in advance. The drosophila control is "
          "flat for the paper's frozen [SEP] probe only; under mean pooling (frozen or fine-tuned) its "
          "GC-matched test effect is negative (Part 3, Q2).")
    ud0 = load_opt("_unit_diagnostics.csv")
    n1c = int(ud0.one_class.sum()) if ud0 is not None else 0
    w("  * Thresholded metrics (balanced accuracy, MCC at p=0.5) depend on calibration"
      + (f": {n1c} fine-tuning unit(s) put every test sequence on one side of 0.5 (balanced accuracy "
         "0.500, MCC 0; unit_diagnostics one_class)" if n1c else "")
      + ". AUROC / balanced-resample AUROC are the informative numbers for such arms.")
    ud = load_opt("_unit_diagnostics.csv")
    if ud is not None and ud.degenerate.any():
        dg = ud[ud.degenerate].groupby(["model", "dataset", "train_arm"]).size()
        w("  * Degenerate fits (Part 3 Q1): " + "; ".join(f"{m}/{d}/{a} {k} unit(s)" for (m, d, a), k in dg.items())
          + ". Their D is an optimization collapse of the fixed recipe on the heavily imbalanced equalized "
          "train arm, not a measured absence of learnable signal; do not cite it as such. Rules 1-2 do not "
          "use D.")
    if ud is not None:
        v = ud[ud.dataset == "nt_enhancers"]
        if len(v):
            w(f"  * nt_enhancers epoch selection: {100 * v.frac_val_verbatim_in_fit.mean():.0f}% of validation rows "
              f"(mean over units) have an exact copy among the fitting rows, so validation AUROC (mean "
              f"{v.best_val_auroc.mean():.3f}) overstates test AUROC (mean {v.own_arm_test_auroc.mean():.3f}; "
              "unit_diagnostics). This is not test leakage (test rows never enter training or selection), but "
              "the epoch choice on nt_enhancers is made on a partly memorized carve.")
    cs = load_opt("_ccre_sensitivity.csv")
    if cs is not None:
        k = cs[(cs.dataset == "human_enhancers_cohn") & (cs["filter"] == "ccre_any_free") & (cs.model.isin(["hyenadna", "dnabert2"]))]
        w("  * Latent positives (Part 3 Q3): the equalizer keeps cCRE-overlapping negatives more often than the "
          "random control, so part of S may reflect enhancer-like negatives rather than composition. With every "
          "cCRE-overlapping test negative removed the fine-tuned drop survives but is smaller (cohn share lost: "
          + "; ".join(f"{SHORT[x.model]} {x.frac_lost:.2f} [{x.frac_lost_lo:.2f},{x.frac_lost_hi:.2f}] vs "
                      f"{x.unfiltered_frac_lost:.2f} unfiltered, B {x.B_est:.3f} [{x.B_lo:.3f},{x.B_hi:.3f}]"
                      for x in k.itertuples()) + ").")
    mu = load_opt("_multiplicity.csv")
    if mu is not None:
        s_ = mu[mu.quantity == "S"]
        w(f"  * Multiplicity: rules were evaluated on {int(s_.n_cells.iloc[0])} cells without correction; at the "
          f"Bonferroni level S still excludes 0 in {int(s_.excludes_0_bonf.sum())}/{len(s_)} (Part 3 Q6).")
    w("  * The random arm controls size and imbalance, not every property the equalizer selects on "
      "(it keeps negatives whose composition resembles positives; any correlate of that is also "
      "removed). The composition-only floor of each arm is listed above for that reason. One random "
      "draw (RandomState 42) is used for training and test; psm_sensitivity shows the k-mer spread "
      "over 10 draws.")
    pp = rt.groupby("model")[["n_params_trainable", "n_params_total"]].first()
    if {"hyenadna", "dnabert2"} <= set(pp.index):
        h, d = pp.loc["hyenadna"], pp.loc["dnabert2"]
        w(f"  * Two FMs of modest size ({h.n_params_total / 1e6:.2f}M; DNABERT-2 {d.n_params_total / 1e6:.1f}M "
          "incl. LoRA + head; fm_finetune_v2_runs.csv) with one fixed recipe each; absolute AUROC is not "
          f"tuned to a leaderboard. DNABERT-2 is LoRA (backbone frozen, "
          f"{100 * d.n_params_trainable / d.n_params_total:.2f}% of weights trainable), not "
          "full-parameter; its 512-token context excludes the 2-3 kb drosophila sequences, so the "
          "GC-balanced drosophila control is HyenaDNA only. DNABERT-2 full-parameter and NT-v2-500M LoRA "
          "are packaged for a CUDA GPU (gpu_deferred/, external/fm_finetune/gbkb_ft_gpu_bundle.tar.gz).")
    # Published reference points (literature constants, not computed here): Nguyen et al., HyenaDNA,
    # NeurIPS 2023, arXiv:2306.15794v2, Table 4.1 (GenomicBenchmarks top-1 accuracy, the same 2-layer
    # width-128 436k-parameter architecture after a hyperparameter sweep, 100 epochs) and Table 4.2
    # (NT benchmark enhancer MCC, a 2-layer width-256 1.6M-parameter HyenaDNA).
    hc = tab[(tab.model == "hyenadna") & (tab.dataset == "human_enhancers_cohn")]
    hn = tab[(tab.model == "hyenadna") & (tab.dataset == "nt_enhancers")]
    dc = tab[(tab.model == "dnabert2") & (tab.dataset == "human_enhancers_cohn")]
    if len(hc) and len(hn):
        w(f"  * Not under-trained, but not leaderboard-tuned (post hoc check against published numbers): on the "
          f"balanced cohn test, fine-tuned HyenaDNA-tiny reaches accuracy {hc.A_bal_acc.iloc[0]:.3f} vs 0.742 "
          "published for the same 436k-parameter HyenaDNA architecture after a hyperparameter sweep (HyenaDNA "
          "paper, arXiv:2306.15794, Table 4.1)"
          + (f"; DNABERT-2 LoRA reaches {dc.A_bal_acc.iloc[0]:.3f}" if len(dc) else "")
          + f". On nt_enhancers its MCC is {hn.A_mcc.iloc[0]:.3f} vs 0.626 published for the larger 1.6M-parameter "
          "HyenaDNA (Table 4.2), so the nt_enhancers HyenaDNA numbers are below a tuned model's.")
    w("  * The baseline (k-mer, frozen) rows are post hoc, single-seed, and trained on the full train "
      "arm; the fine-tuned models train on a 90% carve (10% validation for epoch selection).")
    if vv is not None and len(vv):
        a = vv[(vv.train_arm == "original") & (vv.eval_arm == "original")].iloc[0]
        b = vv[(vv.train_arm == "original") & (vv.eval_arm == "comp_equalized")].iloc[0]
        w(f"  * The v1 numbers cited in the June/v2 PSB manuscript and numbers_ledger.md "
          f"({a.v1_auroc_mean:.3f} -> {b.v1_auroc_mean:.3f}) came from the defective batcher; the "
          f"v2 re-run gives {a.v2_auroc_mean:.3f} -> {b.v2_auroc_mean:.3f} (fm_finetune_v2_v1_vs_v2.csv). "
          "Replace them with the v2 numbers.")
    if {"pad_multiple", "micro_batch", "peak_footprint_mb"} <= set(rt.columns):
        fx = rt[(rt.pad_multiple > 0) | (rt.micro_batch > 0)]
        w(f"  * Memory (2026-09-24): after a drosophila unit grew to 17.6 GB (per-batch tensor shapes "
          f"-> unbounded MPS graph cache), the remaining {len(fx)} unit(s) ran with fixed-shape "
          f"padding (--pad-multiple), exact micro-batch gradient accumulation for drosophila "
          f"(--micro-batch 4; optimizer batch still 32) and hard memory caps (memguard.py). These "
          f"are numerical no-ops (tests/check_pad_invariance.py: logits <= 9e-8, gradients <= "
          f"1.4e-4 relative); per-unit settings and peak footprint in fm_finetune_v2_runs.csv"
          + (f" (max peak {fx.peak_footprint_mb.max():.0f} MB)." if len(fx) and
             fx.peak_footprint_mb.notna().any() else "."))
    open(OUT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
