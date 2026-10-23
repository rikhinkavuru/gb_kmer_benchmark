#!/usr/bin/env python3
"""POST HOC sensitivity analyses for the PSM-sensitivity experiment (Reviewer 1, comment 4).

WHY / WHEN. Added 2026-09-24, after the pre-registered run (run_psm_sensitivity.py) had finished and
after two independent verification reviews of its outputs. Everything here is POST HOC and is labelled
so in every output. The pre-registered run, its decision code (summarize_psm.decide) and its outputs
are unchanged. This docstring was written BEFORE this module ran any of the analyses below. The
verifiers computed some of these quantities in their own scratch code; none of their numbers are read
here, and everything is recomputed from the prediction cache or from new fits.

ANALYSIS PLAN (fixed before running)
P0  Leakage audit, all three datasets. Count the test sequences that occur verbatim (after
    strip().upper()) in the training split, and the training rows that equal a test sequence.
    Reverse-complement matches are counted separately and reported, not removed (the k-mer features
    are not strand-collapsed). Also report label agreement and the original model's mean score on
    duplicated vs other test positives. -> psm_sensitivity_leakage_check.csv
P1  nt_enhancers, LEAK-FREE TEST (the primary leak-free nt analysis; no refit). Every arm of the
    pre-registered run is re-scored with the SAME cached models after removing, from every test set,
    the test sequences that occur verbatim in the training split (P0). All designs (within,
    fixed_model, train_only, 2x2 cross, cCRE-free test) are recomputed with the identical metric code
    and bootstrap (B=1000). Dataset label `nt_enhancers_leakfreeTest`.
P2  nt_enhancers, LEAK-FREE REFIT (robustness check). Every arm is refit after its training set loses
    the training rows that equal any test sequence; the selection is otherwise unchanged (same
    propensity model, same kept negatives). Scored on the full 400-sequence test set, which is then
    leak-free. Label `nt_enhancers_leakfreeRefit`; cache external/psm_sensitivity/preds/<label>/.
    For P1 and P2 the pre-registered decision rule (summarize_psm.decide, unchanged) is applied as a
    sensitivity analysis, with tau recomputed from each variant's own original-AUROC CI. If P1 and the
    uncorrected nt analysis disagree, the P1 (leak-free) result is the one reported as primary for nt.
P3  CLASS-BALANCED TRAINING, all three datasets. The reduced arms train on ~96%-positive sets with no
    class weights (the pre-registered protocol, as in the paper). Refit F1 alpha=1 (selection seeds 42,
    43 and 44), its 10 size-matched random arms, and the paper's published split with class-balanced
    sample weights (each class weighted n/(2 n_c); the test is unweighted, so the prior-corrected
    threshold becomes 0.5). Families `F1bw_hist` / `paperbw_anchor`. Reported: size effect
    A0 - mean Ar_bw, shortcut effect mean Ar_bw - Am_bw, MCC@0.5 and balanced MCC for matched vs
    random. Reading rule: if the cohn size effect under balanced weights is <= tau (0.03), the
    pre-registered size component is attributed to imbalance handling; otherwise it is reported as
    persisting. nt is also scored on the P1 leak-free test.
P4  BETWEEN-DATASET CONTRAST (paired within a dataset; no retraining). Share of above-chance AUROC lost,
    S = (A_placebo - A_overlap) / (A_placebo - 0.5), for (a) the ORIGINAL model scored on the
    overlap-weighted original test vs the mean over the 10 permuted-weight placebo tests (primary),
    and (b) the overlap-weighted retrained model vs the 10 placebo-weighted retrained models.
    1000 bootstrap resamples of test sequences (RandomState(42)). Within a dataset the overlap and
    placebo AUROCs share each resample; datasets are resampled independently, and a contrast is the
    difference of the bootstrap draws. The control is called "proportionally less affected" only if
    the 95% CI of S(contaminated) - S(control) excludes 0. The absolute drop is reported too.
    -> psm_sensitivity_between_dataset.csv
P5  SELECTION BIAS OF COMMON-SUPPORT TRIMMING (F2). The verification review observed that in the clean
    control, F2 trimming lowers k-mer AUROC against size-matched random negatives while the
    composition-only AUROC stays the same. Characterize the negatives kept by F2 at q=0.3 and q=0.45
    exactly like part (6) of the pre-registered run: features outside the propensity model, retained
    vs discarded vs positives, each selection against its own random size-matched negatives.
    Descriptive. -> psm_sensitivity_selection_bias_f2.csv
Everything runs under bibm/lowcpu.sh with LightGBM on 2 threads (as in the main run) and one dataset in
memory at a time. Fits are checkpointed per arm, and every part writes a checkpoint under
external/psm_sensitivity/parts_posthoc/ (delete it to recompute).

FIX-ROUND ADDENDUM (2026-09-24, written BEFORE P0b/P6 were run; they ran 16:43-16:45:14 per
psm_sensitivity_posthoc_run.log. The '~17:00' first written here was wrong and was corrected to this wording in
fix round 2; nothing else in this addendum changed; still POST HOC). A re-check of the
fix pass showed, in its own scratch code, that the cohn-minus-drosophila contrast of P4 (k-mer share lost
0.353) equals the contrast in the share of COMPOSITION-ONLY AUROC that the same overlap weights remove
(0.352). The additive version of P6 below is therefore not blind: its outcome for the fixed-model design was
known when P6 was specified. Nothing here reuses the re-check's numbers; everything is recomputed.
P0b Interval-overlap audit (hg38 coordinates; cohn = every sequence, nt = the genomic-mapped subset). Count
    the test sequences whose interval overlaps the union of training intervals by >= 1 bp and by >= 50% of
    the test interval, per class, and the original model's AUROC without them. Descriptive; nothing is
    removed from any other analysis. -> psm_sensitivity_leakage_interval.csv
P6  COMPOSITION-NORMALISED between-dataset contrast. Same designs, same overlap / placebo weights and the SAME
    1000 bootstrap resamples as P4 (identical RandomState(42) draws; P6 recomputes the k-mer draws and asserts
    they are bit-identical to the saved P4 draws), now also for the composition-only LightGBM that was fit on
    the same arms. Per dataset: S_comp = share of above-chance composition-only AUROC lost under the weights
    (how much composition signal the weights remove), S_kmer - S_comp, and S_kmer / S_comp (k-mer share lost
    per unit of composition-only share lost). Per contrast (a - b): the composition-only contrast, the
    difference of contrasts (S_kmer,a - S_kmer,b) - (S_comp,a - S_comp,b) (additive scale) and the ratio
    difference S_kmer,a/S_comp,a - S_kmer,b/S_comp,b (multiplicative scale), each with a percentile CI.
    Reading rule (fixed now): the k-mer contrast is read as the contaminated set relying MORE on composition
    per unit of composition signal removed only if, for the primary fixed-model design, the CIs of BOTH the
    difference of contrasts and the ratio difference exclude 0 with a positive sign. Otherwise it is reported
    as accounted for by how much composition signal the weights remove, which says nothing about artifact vs
    biology. Rows whose weighted k-mer or composition-only AUROC falls below 0.5 (share > 1; over-correcting
    weights, i.e. nt) are flagged not interpretable. -> extra columns in psm_sensitivity_between_dataset.csv

FIX-ROUND-2 ADDENDUM (2026-09-24, written BEFORE P7 was first run: the file was last saved at 17:27:17 and P7 ran
17:27:25-17:30:28 per psm_sensitivity_posthoc_run.log; the '17:30' first written here was wrong and was corrected
to this wording in fix round 3; POST HOC; no refits, cached predictions only).
The second re-check asked for a CI on the cohn composition component (Ar_mean - Am) and pointed out that in the
clean control the matched-vs-random k-mer drop (0.081) is about twice the composition-only drop at the same arm
(0.041). All POINT estimates below are therefore already known (they are decision.csv columns A0 / Ar_mean / Am /
comp0 / comp_r_mean / comp_m / trainonly_*); P7 adds only their uncertainty and a relative scale.
P7  DECOMPOSITION CIs, two-level PAIRED bootstrap, primary arm F1 alpha=1 (the paper's retention). Every arm's
    test set is a subset of the dataset's test set, so level 1 resamples the dataset's test sequences (B = 1000,
    RandomState(C.SEED), identical draws to P4) and every arm is scored on its own members of that resample
    (shared positives stay paired); level 2 resamples the 10 random seeds with replacement (and, in the
    secondary 'matched 42-44' rows, the 3 matched selection seeds) with RandomState(C.SEED + 1). Per dataset
    label (cohn; nt raw; nt leak-free test; nt leak-free refit; drosophila), design (within = arm-trained model on
    its own test; train_only = arm-trained model on the full, undistorted test), model (kmer, comp) and matched
    seeds (42 = pre-registered; 42-44): A0, Ar_mean, Am; size effect A0 - Ar_mean; matched-vs-random effect
    Ar_mean - Am; its share of the total drop; and the relative gap (Ar_mean - Am) / (Ar_mean - 0.5) (share of
    the random arm's above-chance AUROC lost). Cross-model rows (model = kmer_minus_comp, same resamples): the
    k-mer minus composition-only matched-vs-random effect on the absolute and on the relative scale.
    Reading rules (fixed now; descriptive; the pre-registered verdicts do not change):
    R7a  the matched-vs-random effect is called non-zero if its 95% CI excludes 0.
    R7b  the size effect is reported against tau = 0.03 as 'CI above tau', 'CI below tau' or 'CI straddles tau'.
    R7c  within design, matched seed 42: the k-mer model is said to lose MORE than the composition-only model
         (part of the matched-vs-random drop is not explained by the composition-only signal lost, consistent
         with selection-induced difficulty) only if the kmer_minus_comp CIs on BOTH scales exclude 0 with a
         positive sign; LESS if both exclude 0 with a negative sign; otherwise 'not distinguishable or
         scale-dependent'. The two models start from different AUROCs, which is why both scales are required.
    -> psm_sensitivity_decomposition_ci.csv

FIX-ROUND-3 ADDENDUM (2026-09-24; the plan and rule R8 below were written BEFORE the first --parts P7,P8 run,
18:14:35-18:17:08 per psm_sensitivity_posthoc_run.log; only this timestamp phrase was edited afterwards, and P7/P8 were
then re-run from the final file (second log entry) with output identical to the first run; POST HOC; no refits, cached
predictions only). The third re-check showed (from summary.csv) that two within-design readings do not carry over
to the paired designs: (i) in the clean control, with the original model or the original test held fixed, the
composition-only AUROC falls as much as the k-mer AUROC (F1 alpha=1 fixed_model 0.076 vs 0.072; F2 q0.3-q0.45
0.15-0.19 vs 0.13-0.15), and (ii) at cohn F2 q0.45 the retrained composition-only AUROC RISES above its random
control (0.919 vs 0.740). Those point estimates are known; the two additions below add uncertainty and a
mechanism check, and no verdict changes.
P7 (extended)  Design 'fixed_model' is added to P7: the ORIGINAL model (trained once on the full split) scored on
    each F1 alpha=1 arm's own test subset (matched vs size- and class-ratio-matched random subsets; no training
    change). Same two-level paired bootstrap, same draws, same rules R7a and R7c (R7b is not applied: there is no
    training-size change, and size_vs_tau reads 'n/a (fixed_model)'). Its point estimates equal decision.csv
    fixed_A_* (k-mer) and summary.csv (composition-only). The within and train_only rows are unchanged (identical
    draws; verified against the fix-round-2 file). Column n_test_neg_matched42 (which on train_only rows held the
    full-test count) is renamed n_test_neg_scored: negatives in the test set the seed-42 matched arm is scored on
    (its own test for within and fixed_model; the full test for train_only). Column tau_R7b records R7b's tau.
P8  F2 CONSTRUCTION CHECK (descriptive). F2 keeps every positive but only the negatives whose logit-propensity lies
    inside the positives' [q, 1-q] quantile band of the same split, so about 2q of the positives lie outside the
    kept negatives' propensity range. Per dataset label (cohn, nt raw, nt leak-free test, drosophila) and F2 level:
    (a) the fraction of test positives below / above the band (and of training positives); (b) for the matched
    seed-42 arm and the mean over its 10 random arms, the AUROC of IN-BAND test positives vs the arm's test
    negatives and of OUT-OF-BAND test positives vs the same negatives, for the retrained ('within') k-mer and
    composition-only models and for the original ('fixed_model') models. Point estimates only.
    Reading rule R8 (fixed now): at a level, the within-design composition-only AUROC change is said to come from
    the band's tails if the retrained composition-only AUROC of out-of-band positives exceeds the random-arm mean
    by >= 0.05 while that of in-band positives is below the random-arm mean by >= 0.05; the k-mer model is said to
    lose in-band signal beyond the composition-only model if its in-band gap (random mean - matched) exceeds the
    composition-only in-band gap by >= 0.05. -> psm_sensitivity_f2_band.csv
"""
import argparse
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import run_psm_sensitivity as R  # noqa: E402
import psm_core as P  # noqa: E402
import _common as C  # noqa: E402

PARTS = os.path.join(R.SNAP, "parts_posthoc")
OUT = C.RESULTS_DIR
NT = "nt_enhancers"
LF_TEST = "nt_enhancers_leakfreeTest"
LF_REFIT = "nt_enhancers_leakfreeRefit"
ROLE = {LF_TEST: "contaminated; leak-free test (post hoc)",
        LF_REFIT: "contaminated; leak-free refit (post hoc)"}
F2_LEVELS = ("q0.3", "q0.45")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def part(name):
    return os.path.join(PARTS, name)


def clone(a, tr_idx=None, te_idx=None):
    return R.Arm(a.name, a.family, a.level, a.kind, a.seed,
                 a.tr_idx if tr_idx is None else tr_idx, a.te_idx if te_idx is None else te_idx,
                 w_tr=a.w_tr, w_te=a.w_te, level_label=a.level_label)


def dup_masks(ctx):
    """(test rows whose sequence occurs in train, train rows whose sequence occurs in test)."""
    trs, tes = set(ctx.tr), set(ctx.te)
    return (np.array([s in trs for s in ctx.te], bool), np.array([s in tes for s in ctx.tr], bool))


def main_cache(task):
    return os.path.join(R.SNAP, "preds", task)


def load_pred(task, name):
    d = np.load(os.path.join(main_cache(task), f"{name}.npz"))
    return dict(kmer=d["kmer"], comp=d["comp"], pi=float(d["pi"]))


# ================================================================ consistency with the main run
def check_against_main(ctx, arms, main_sw):
    """The rebuilt arms + cached predictions must reproduce the main run's rows exactly."""
    by = {a.name: a for a in arms}
    q = main_sw[(main_sw["dataset"] == ctx.task) & (main_sw["design"] == "within") & (main_sw["model"] == "kmer")]
    for nm in ("original", "F1_hist_a1_matched42", "F3_1to1_c0.2_matched42", "F2_trim_q0.3_matched42"):
        a = by[nm]
        p = load_pred(ctx.task, nm)["kmer"]
        auc = roc_auc_score(ctx.yte[a.te_idx], p[a.te_idx])
        row = q[q["test_arm"] == nm].iloc[0]
        assert abs(auc - row["auroc"]) < 1e-9, f"{ctx.task}/{nm}: {auc} != main {row['auroc']}"
        assert int((ctx.yte[a.te_idx] == 0).sum()) == int(row["n_neg"]), f"{ctx.task}/{nm}: n_neg differs"
    log(f"[{ctx.task}] rebuilt arms reproduce the main run (4 arms checked)")


# ================================================================ P0 leakage audit
def part_leakage(ctx):
    path = part(f"leakage_{ctx.task}.csv")
    if os.path.exists(path):
        return
    tr_ = str.maketrans("ACGTN", "TGCAN")
    trs = set(ctx.tr)
    dup_te, dup_tr = dup_masks(ctx)
    rc_te = np.array([s.translate(tr_)[::-1] in trs for s in ctx.te], bool) & ~dup_te
    labels = {}
    for s, y in zip(ctx.tr, ctx.ytr):
        labels.setdefault(s, set()).add(int(y))
    consistent = all(labels[s] == {int(y)} for s, y, d in zip(ctx.te, ctx.yte, dup_te) if d)
    p = load_pred(ctx.task, "original")["kmer"]
    pos = ctx.yte == 1
    keep = ~dup_te
    row = dict(dataset=ctx.task, n_train=len(ctx.ytr), n_test=len(ctx.yte),
               n_test_verbatim_in_train=int(dup_te.sum()),
               n_test_verbatim_pos=int((dup_te & pos).sum()), n_test_verbatim_neg=int((dup_te & ~pos).sum()),
               n_train_rows_equal_a_test_seq=int(dup_tr.sum()),
               n_train_rows_equal_a_test_seq_pos=int((dup_tr & (ctx.ytr == 1)).sum()),
               n_test_revcomp_in_train_not_verbatim=int(rc_te.sum()),
               dup_labels_consistent=bool(consistent),
               orig_mean_score_dup_pos=float(p[dup_te & pos].mean()) if (dup_te & pos).any() else np.nan,
               orig_mean_score_other_pos=float(p[~dup_te & pos].mean()),
               orig_auroc_full_test=float(roc_auc_score(ctx.yte, p)),
               orig_auroc_leakfree_test=float(roc_auc_score(ctx.yte[keep], p[keep])),
               n_test_pos_leakfree=int((keep & pos).sum()), n_test_neg_leakfree=int((keep & ~pos).sum()))
    pd.DataFrame([row]).to_csv(path, index=False)
    log(f"[{ctx.task}] P0 leakage: {row['n_test_verbatim_in_train']} test seqs in train "
        f"({row['n_test_verbatim_pos']} pos), {row['n_train_rows_equal_a_test_seq']} train rows")


# ================================================================ P1 leak-free test (no refit)
def part_leakfree_test(ctx, arms, B):
    ps, pb = part("sweep_leakfreeTest.csv"), part("balance_leakfreeTest.csv")
    if os.path.exists(ps) and os.path.exists(pb):
        return
    dup_te, _ = dup_masks(ctx)
    keep = ~dup_te
    missing = [a.name for a in arms if not os.path.exists(os.path.join(main_cache(ctx.task), a.name + ".npz"))]
    assert not missing, f"P1 needs the full cache; missing {missing[:5]}"   # never fits into the main cache
    arms2 = [clone(a, te_idx=a.te_idx[keep[a.te_idx]]) for a in arms]
    log(f"[P1] re-scoring {len(arms2)} cached nt arms on the leak-free test ({int(keep.sum())} sequences)")
    _, _, _, bal, ev = R.run_dataset(ctx, arms2, B)
    for df in (bal, ev):
        df["dataset"], df["role"] = LF_TEST, ROLE[LF_TEST]
    bal.to_csv(pb, index=False)
    ev.to_csv(ps, index=False)


# ================================================================ P2 leak-free refit
def part_leakfree_refit(ctx, arms, B):
    ps, pb = part("sweep_leakfreeRefit.csv"), part("balance_leakfreeRefit.csv")
    if os.path.exists(ps) and os.path.exists(pb):
        return
    _, dup_tr = dup_masks(ctx)
    keep = ~dup_tr
    arms3 = [clone(a, tr_idx=a.tr_idx[keep[a.tr_idx]]) for a in arms]
    task0 = ctx.task
    ctx.task = LF_REFIT          # -> separate prediction cache + dataset label in eval rows
    try:
        cache = os.path.join(R.SNAP, "preds", LF_REFIT)
        os.makedirs(cache, exist_ok=True)
        order = sorted(arms3, key=R.tier)
        t0 = time.time()
        for i, a in enumerate(order):
            ta = time.time()
            R.fit_arm(ctx, a, cache)
            if time.time() - ta > 1 and (i % 10 == 0 or i == len(order) - 1):
                log(f"  [P2] fit {i + 1}/{len(order)} {a.name} ({time.time() - ta:.0f}s; elapsed "
                    f"{(time.time() - t0) / 60:.1f} min)")
        _, _, _, bal, ev = R.run_dataset(ctx, arms3, B)
    finally:
        ctx.task = task0
    for df in (bal, ev):
        df["dataset"], df["role"] = LF_REFIT, ROLE[LF_REFIT]
    bal.to_csv(pb, index=False)
    ev.to_csv(ps, index=False)


# ================================================================ P3 class-balanced training
def bw_arms(ctx, arms):
    by = {a.name: a for a in arms}
    names = ([f"F1_hist_a1_matched{s}" for s in R.MATCH_SEEDS] + [f"F1_hist_a1_random{r}" for r in R.RAND_SEEDS]
             + ["paper_equalized_matched"])
    out = []
    for nm in names:
        a = by[nm]
        y = ctx.ytr[a.tr_idx]
        w = np.ones(len(ctx.ytr))
        for c in (0, 1):
            w[a.tr_idx[y == c]] = len(y) / (2.0 * (y == c).sum())
        paper = nm.startswith("paper")
        out.append(R.Arm(nm.replace("F1_hist", "F1bw_hist").replace("paper_equalized", "paperbw_equalized"),
                         "paperbw_anchor" if paper else "F1bw_hist", a.level, a.kind, a.seed, a.tr_idx, a.te_idx,
                         w_tr=w, w_te=None, level_label=a.level_label))
    return out


def part_balw(ctx, arms, B):
    ps, pb = part(f"sweep_balw_{ctx.task}.csv"), part(f"balance_balw_{ctx.task}.csv")
    if os.path.exists(ps) and os.path.exists(pb):
        return
    cache = os.path.join(R.SNAP, "preds", ctx.task + "_balw")
    os.makedirs(cache, exist_ok=True)
    bws = bw_arms(ctx, arms)
    preds = {}
    t0 = time.time()
    for a in bws:
        preds[a.name] = R.fit_arm(ctx, a, cache)
        assert abs(preds[a.name]["pi"] - 0.5) < 1e-9
    log(f"[P3] {ctx.task}: {len(bws)} class-balanced fits in {(time.time() - t0) / 60:.1f} min")
    orig = [a for a in arms if a.name == "original"][0]
    variants = [(ctx.task, None, ctx.role)]
    if ctx.task == NT:
        variants.append((LF_TEST, ~dup_masks(ctx)[0], ROLE[LF_TEST]))
    rows, bals = [], []
    for label, keep, role in variants:
        o = orig if keep is None else clone(orig, te_idx=orig.te_idx[keep[orig.te_idx]])
        ev = []
        for a in bws:
            at = a if keep is None else clone(a, te_idx=a.te_idx[keep[a.te_idx]])
            for mk in ("kmer", "comp"):
                ev.append(R.eval_row(ctx, at, preds[a.name], at, "within", mk, B))
                ev.append(R.eval_row(ctx, at, preds[a.name], o, "train_only", mk, B))
            b = R.balance_row(ctx, at)
            b.update(dataset=label, role=role)
            bals.append(b)
        ev = pd.DataFrame(ev)
        ev["dataset"], ev["role"] = label, role
        rows.append(ev)
    bal = pd.DataFrame(bals)
    ev = pd.concat(rows, ignore_index=True)
    ev = ev.merge(bal[["dataset", "arm", "retained_neg_frac", "retained_pos_frac", "ess_neg_frac", "max_abs_smd",
                       "mean_abs_smd", "gc_auroc", "tata_auroc"]].rename(columns={"arm": "test_arm"}),
                  on=["dataset", "test_arm"], how="left")
    bal.to_csv(pb, index=False)
    ev.to_csv(ps, index=False)


# ================================================================ P4 between-dataset contrast
def overlap_draws(ctx, arms, B, keep=None, model="kmer"):
    """Point estimates and bootstrap draws of the overlap-weighted vs placebo-weighted AUROC for the two
    designs (fixed original model; overlap-weighted retrained model), for one model's cached predictions.
    The resamples depend only on (C.SEED, n test sequences, B), so every call on the same test set uses
    the SAME resamples (P6 relies on this and asserts it)."""
    by = {a.name: a for a in arms}
    idx = np.arange(len(ctx.yte)) if keep is None else np.flatnonzero(keep)
    y = ctx.yte[idx]
    n = len(y)
    orig = load_pred(ctx.task, "original")[model][idx]
    wm = by["W_overlap_matched"].w_te[idx]
    pm = load_pred(ctx.task, "W_overlap_matched")[model][idx]
    wr = [by[f"W_overlap_random{r}"].w_te[idx] for r in R.RAND_SEEDS]
    pr = [load_pred(ctx.task, f"W_overlap_random{r}")[model][idx] for r in R.RAND_SEEDS]
    designs = {"fixed_model_overlap": ((orig, wm), [(orig, w) for w in wr]),
               "within_overlap": ((pm, wm), list(zip(pr, wr)))}
    rng = np.random.RandomState(C.SEED)
    counts = np.vstack([np.bincount(rng.randint(0, n, n), minlength=n) for _ in range(B)]).astype(np.float64)
    out = {}
    for dz, ((sm, wmm), plac) in designs.items():
        aw_pt = P.auc_rows(wmm[None, :], y, sm)[0]
        ap_pt = np.mean([P.auc_rows(w[None, :], y, s)[0] for s, w in plac])
        aw, ap = [], []
        for c0 in range(0, B, 250):
            Wc = counts[c0:c0 + 250]
            aw.append(P.auc_rows(Wc * wmm[None, :], y, sm))
            ap.append(np.mean([P.auc_rows(Wc * w[None, :], y, s) for s, w in plac], axis=0))
        out[f"{dz}_aw_pt"], out[f"{dz}_ap_pt"] = aw_pt, ap_pt
        out[f"{dz}_aw"], out[f"{dz}_ap"] = np.concatenate(aw), np.concatenate(ap)
    out["n_test"], out["n_neg"] = n, int((y == 0).sum())
    return out


def part_between_boot(ctx, arms, B, label=None, keep=None):
    """Bootstrap draws of the overlap vs placebo AUROC for two designs, saved per dataset label (k-mer)."""
    label = label or ctx.task
    path = part(f"between_{label}.npz")
    if os.path.exists(path):
        return
    np.savez(path, **overlap_draws(ctx, arms, B, keep, "kmer"))
    log(f"[P4] {label}: bootstrap draws saved")


# ================================================================ P6 composition-normalised contrast
def part_between_comp_boot(ctx, arms, B, label=None, keep=None):
    """P6: the composition-only model's overlap vs placebo draws on the SAME resamples as P4. The k-mer draws
    are recomputed and must be bit-identical to the saved P4 draws (proves the resamples are shared)."""
    label = label or ctx.task
    path = part(f"between_comp_{label}.npz")
    if os.path.exists(path):
        return
    p4 = np.load(part(f"between_{label}.npz"))
    km = overlap_draws(ctx, arms, B, keep, "kmer")
    for k, v in km.items():
        assert np.array_equal(np.asarray(v), np.asarray(p4[k])), f"P6 {label}: k-mer draws differ from P4 at {k}"
    np.savez(path, **overlap_draws(ctx, arms, B, keep, "comp"))
    log(f"[P6] {label}: composition-only draws saved (k-mer draws re-derived and identical to P4)")


# ================================================================ P0b interval-overlap audit
def part_interval_leakage(ctx, arms):
    """P0b: test sequences whose hg38 interval overlaps the union of training intervals (descriptive)."""
    path = part(f"leakage_interval_{ctx.task}.csv")
    if os.path.exists(path) or ctx.task not in R.GENOMIC_TASKS:
        return
    ch, st, en = R._coords(ctx)
    ntr = len(ctx.ytr)
    ch_tr, st_tr, en_tr = ch[:ntr], st[:ntr], en[:ntr]
    ch_te, st_te, en_te = ch[ntr:], st[ntr:], en[ntr:]
    mapped_te = np.array([c is not None for c in ch_te])
    ov = np.zeros(len(ctx.yte))
    for c in sorted(set(ch_te[mapped_te])):
        mt = np.flatnonzero(ch_te == c)
        mr = np.flatnonzero(ch_tr == c)
        ov[mt] = R._covered(R._merge(st_tr[mr], en_tr[mr]), st_te[mt], en_te[mt])
    L = np.where(mapped_te, en_te - st_te, 1).astype(float)
    any_ = mapped_te & (ov >= 1)
    half = mapped_te & (ov >= 0.5 * L)
    pos = ctx.yte == 1
    p = load_pred(ctx.task, "original")["kmer"]
    by = {a.name: a for a in arms}
    mt42 = np.zeros(len(ctx.yte), bool)
    mt42[by["F1_hist_a1_matched42"].te_idx] = True
    keep = ~any_
    row = dict(dataset=ctx.task, n_test=len(ctx.yte), n_test_mapped=int(mapped_te.sum()),
               n_train_mapped=int(sum(c is not None for c in ch_tr)),
               n_test_overlap_any=int(any_.sum()), n_test_overlap_any_pos=int((any_ & pos).sum()),
               n_test_overlap_any_neg=int((any_ & ~pos).sum()),
               n_test_overlap_half=int(half.sum()), n_test_overlap_half_pos=int((half & pos).sum()),
               n_test_overlap_half_neg=int((half & ~pos).sum()),
               orig_auroc_full_test=float(roc_auc_score(ctx.yte, p)),
               orig_auroc_without_overlapping=float(roc_auc_score(ctx.yte[keep], p[keep])),
               n_overlap_any_in_F1a1_matched42_test=int((any_ & mt42).sum()),
               n_overlap_any_neg_in_F1a1_matched42_test=int((any_ & mt42 & ~pos).sum()),
               F1a1_matched42_test_n_neg=int((mt42 & ~pos).sum()))
    pd.DataFrame([row]).to_csv(path, index=False)
    log(f"[P0b] {ctx.task}: {row['n_test_overlap_any']} test intervals overlap a training interval "
        f"({row['n_test_overlap_any_pos']} pos / {row['n_test_overlap_any_neg']} neg; >=50%: {row['n_test_overlap_half']})")


# ================================================================ P7 decomposition CIs (two-level paired bootstrap)
P7_TAU = 0.03
P7_Q = ("size_effect", "matched_vs_random", "total_drop", "mvr_share_of_total", "rel_gap")


def _p7_quantities(a0, ar, am):
    """size, matched-vs-random, total, share of total, relative gap (works on scalars and on draw arrays)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(size_effect=a0 - ar, matched_vs_random=ar - am, total_drop=a0 - am,
                    mvr_share_of_total=(ar - am) / (a0 - am), rel_gap=(ar - am) / (ar - 0.5))


def part_decomp(ctx, arms, B, label=None, keep=None, pred_task=None):
    """P7 (see the FIX-ROUND-2 ADDENDUM): size / matched-vs-random decomposition at F1 alpha=1 with a two-level
    paired bootstrap (level 1: the dataset's test sequences, shared by every arm; level 2: random seeds and, for
    the 42-44 rows, matched selection seeds). Point estimates are the plain (unbootstrapped) values."""
    label = label or ctx.task
    pred_task = pred_task or ctx.task
    path = part(f"decomp_{label}.csv")
    if os.path.exists(path):
        return
    assert R.MATCH_SEEDS[0] == 42
    by = {a.name: a for a in arms}
    nfull = len(ctx.yte)
    idx = np.arange(nfull) if keep is None else np.flatnonzero(keep)
    n = len(idx)
    posn = -np.ones(nfull, int)
    posn[idx] = np.arange(n)
    y = ctx.yte[idx]
    rng = np.random.RandomState(C.SEED)                     # identical level-1 draws to P4 / P6
    counts = np.vstack([np.bincount(rng.randint(0, n, n), minlength=n) for _ in range(B)]).astype(np.float64)
    rng2 = np.random.RandomState(C.SEED + 1)                # level 2
    rsel = rng2.randint(0, len(R.RAND_SEEDS), size=(B, len(R.RAND_SEEDS)))
    msel = rng2.randint(0, len(R.MATCH_SEEDS), size=(B, len(R.MATCH_SEEDS)))
    cols = np.arange(B)[:, None]

    def arm_auc(name, model, design):
        a = by[name]
        # fixed_model (fix round 3): the ORIGINAL model scored on the arm's own test subset
        s_all = load_pred(pred_task, "original" if design == "fixed_model" else name)[model][idx]
        if design in ("within", "fixed_model"):
            sub = posn[a.te_idx]
            sub = sub[sub >= 0]
        else:                                               # train_only: the full (undistorted) test
            sub = np.arange(n)
        s, yy = s_all[sub], y[sub]
        bs = np.concatenate([P.auc_rows(counts[c0:c0 + 250][:, sub], yy, s) for c0 in range(0, B, 250)])
        return float(roc_auc_score(yy, s)), bs, int((yy == 0).sum())

    rows, draws = [], {}
    designs = ("within", "train_only", "fixed_model")
    for design in designs:
        for model in ("kmer", "comp"):
            a0_pt, a0_b, _ = arm_auc("original", model, design)
            am = [arm_auc(f"F1_hist_a1_matched{s}", model, design) for s in R.MATCH_SEEDS]
            ar = [arm_auc(f"F1_hist_a1_random{r}", model, design) for r in R.RAND_SEEDS]
            AR, AM = np.vstack([x[1] for x in ar]), np.vstack([x[1] for x in am])
            ar_pt, ar_b = float(np.mean([x[0] for x in ar])), np.nanmean(AR.T[cols, rsel], axis=1)
            for ms, am_pt, am_b in (("42", am[0][0], AM[0]),
                                    ("42-44", float(np.mean([x[0] for x in am])), np.nanmean(AM.T[cols, msel], axis=1))):
                pt, bq = _p7_quantities(a0_pt, ar_pt, am_pt), _p7_quantities(a0_b, ar_b, am_b)
                draws[(design, model, ms)] = (pt, bq)
                row = dict(dataset=label, design=design, model=model, matched_seeds=ms, B=B, n_test=n,
                           n_test_neg_scored=am[0][2], tau_R7b=P7_TAU, A0=a0_pt, Ar_mean=ar_pt, Am=am_pt)
                for q in P7_Q:
                    row[q] = pt[q]
                    row[f"{q}_lo"], row[f"{q}_hi"] = (float(np.nanpercentile(bq[q], 2.5)),
                                                      float(np.nanpercentile(bq[q], 97.5)))
                row["n_boot_nan"] = int(np.isnan(bq["matched_vs_random"]).sum())
                row["mvr_frac_draws_le0"] = float(np.nanmean(bq["matched_vs_random"] <= 0))
                row["mvr_ci_excludes_zero"] = bool(row["matched_vs_random_lo"] > 0 or row["matched_vs_random_hi"] < 0)
                row["size_vs_tau"] = ("n/a (fixed_model)" if design == "fixed_model" else
                                      "CI above tau" if row["size_effect_lo"] > P7_TAU else
                                      "CI below tau" if row["size_effect_hi"] < P7_TAU else "CI straddles tau")
                rows.append(row)
    for design in designs:
        for ms in ("42", "42-44"):
            (pk, bk), (pc, bc) = draws[(design, "kmer", ms)], draws[(design, "comp", ms)]
            row = dict(dataset=label, design=design, model="kmer_minus_comp", matched_seeds=ms, B=B, n_test=n)
            for q in ("size_effect", "matched_vs_random", "rel_gap"):
                row[q] = pk[q] - pc[q]
                d_ = bk[q] - bc[q]
                row[f"{q}_lo"], row[f"{q}_hi"] = float(np.nanpercentile(d_, 2.5)), float(np.nanpercentile(d_, 97.5))
            pa = row["matched_vs_random_lo"] > 0 and row["rel_gap_lo"] > 0
            na = row["matched_vs_random_hi"] < 0 and row["rel_gap_hi"] < 0
            row["reading"] = ("k-mer loses MORE than composition-only on both scales" if pa else
                              "k-mer loses LESS than composition-only on both scales" if na else
                              "not distinguishable or scale-dependent")
            row["reading_rule"] = "R7c: both the absolute and the relative CI exclude 0 with the same sign"
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    r0 = [r for r in rows if r["design"] == "within" and r["model"] == "kmer" and r["matched_seeds"] == "42"][0]
    log(f"[P7] {label}: size {r0['size_effect']:+.3f} [{r0['size_effect_lo']:+.3f},{r0['size_effect_hi']:+.3f}], "
        f"matched-vs-random {r0['matched_vs_random']:+.3f} [{r0['matched_vs_random_lo']:+.3f},"
        f"{r0['matched_vs_random_hi']:+.3f}]")


def part_f2band(ctx, arms, label=None, keep=None):
    """P8 (FIX-ROUND-3 ADDENDUM): F2 construction check. The band is the one the selection used: the [q, 1-q]
    quantiles of the logit-propensity of ALL positives of the split (for the leak-free label the duplicated test
    positives are then left out of the evaluation, not of the band). Point estimates, descriptive."""
    label = label or ctx.task
    path = part(f"f2band_{label}.csv")
    if os.path.exists(path):
        return
    by = {a.name: a for a in arms}
    nfull = len(ctx.yte)
    keep = np.ones(nfull, bool) if keep is None else keep
    rows = []
    ltr_pos = ctx.ltr[ctx.ytr == 1]
    lte_pos = ctx.lte[ctx.yte == 1]
    preds = {}

    def pr(name):
        if name not in preds:
            preds[name] = load_pred(ctx.task, name)
        return preds[name]

    for q in R.TRIM_QS:
        lo_te, hi_te = np.quantile(lte_pos, q), np.quantile(lte_pos, 1.0 - q)
        lo_tr, hi_tr = np.quantile(ltr_pos, q), np.quantile(ltr_pos, 1.0 - q)
        inband = (ctx.lte >= lo_te) & (ctx.lte <= hi_te)
        posk = (ctx.yte == 1) & keep
        row = dict(dataset=label, level_label=f"q{q:g}", q=q,
                   frac_train_pos_outside_band=float(np.mean((ltr_pos < lo_tr) | (ltr_pos > hi_tr))),
                   frac_test_pos_below_band=float(np.mean(ctx.lte[posk] < lo_te)),
                   frac_test_pos_above_band=float(np.mean(ctx.lte[posk] > hi_te)),
                   n_test_pos_inband=int((posk & inband).sum()), n_test_pos_outband=int((posk & ~inband).sum()))
        row["frac_test_pos_outside_band"] = row["frac_test_pos_below_band"] + row["frac_test_pos_above_band"]
        mname = f"F2_trim_q{q:g}_matched42"
        rnames = [f"F2_trim_q{q:g}_random{r}" for r in R.RAND_SEEDS]
        am = by[mname]
        tm = np.zeros(nfull, bool)
        tm[am.te_idx] = True
        negm = tm & (ctx.yte == 0)
        assert np.all(inband[negm]), "F2 kept a test negative outside the positives' band"
        row["n_test_neg_matched42"] = int(negm.sum())
        for design in ("within", "fixed_model"):
            for model in ("kmer", "comp"):
                for part_, pmask in (("inband", posk & inband), ("outband", posk & ~inband)):
                    vals = []
                    for name in [mname] + rnames:
                        a = by[name]
                        t = np.zeros(nfull, bool)
                        t[a.te_idx] = True
                        sel = t & ((ctx.yte == 0) | pmask)
                        s = pr("original" if design == "fixed_model" else name)[model]
                        yy = ctx.yte[sel]
                        vals.append(float(roc_auc_score(yy, s[sel])) if 0 < yy.sum() < len(yy) else np.nan)
                    key = f"{design}_{model}_{part_}"
                    row[f"{key}_matched"] = vals[0]
                    row[f"{key}_random_mean"] = float(np.nanmean(vals[1:]))
                    row[f"{key}_gap"] = row[f"{key}_random_mean"] - vals[0]
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    log(f"[P8] {label}: F2 band rows written")


def _share(d, dz, pt=False):
    """Share of above-chance AUROC lost, (placebo - overlap) / (placebo - 0.5): point or bootstrap draws."""
    if pt:
        ap, aw = float(d[f"{dz}_ap_pt"]), float(d[f"{dz}_aw_pt"])
    else:
        ap, aw = d[f"{dz}_ap"], d[f"{dz}_aw"]
    return (ap - aw) / (ap - 0.5)


def _drop(d, dz, pt=False):
    return (float(d[f"{dz}_ap_pt"]) - float(d[f"{dz}_aw_pt"])) if pt else (d[f"{dz}_ap"] - d[f"{dz}_aw"])


def _ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def between_table(labels, B):
    """Per-dataset share/drop with CIs, and contrasts between datasets (P4). P6 adds, where the composition-only
    draws exist, the composition-only share lost under the same weights and resamples, S_kmer - S_comp,
    S_kmer / S_comp, and per contrast the composition-only contrast, the difference of contrasts and the ratio
    difference, each with a 95% percentile CI over the shared resamples."""
    boots, cboots, rows = {}, {}, []
    for lab in labels:
        p = part(f"between_{lab}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        boots[lab] = d
        pc = part(f"between_comp_{lab}.npz")
        c = np.load(pc) if os.path.exists(pc) else None
        if c is not None:
            assert int(c["n_test"]) == int(d["n_test"])
            cboots[lab] = c
        for dz in ("fixed_model_overlap", "within_overlap"):
            aw = d[f"{dz}_aw"]
            awp, app = float(d[f"{dz}_aw_pt"]), float(d[f"{dz}_ap_pt"])
            sh, dr = _share(d, dz), _drop(d, dz)
            row = dict(kind="dataset", dataset=lab, contrast="", design=dz, n_test=int(d["n_test"]),
                       n_test_neg=int(d["n_neg"]), auroc_overlap=awp,
                       auroc_overlap_lo=np.percentile(aw, 2.5), auroc_overlap_hi=np.percentile(aw, 97.5),
                       auroc_placebo_mean=app, drop=app - awp,
                       drop_lo=np.percentile(dr, 2.5), drop_hi=np.percentile(dr, 97.5),
                       share_above_chance_lost=(app - awp) / (app - 0.5),
                       share_lo=np.percentile(sh, 2.5), share_hi=np.percentile(sh, 97.5), B=len(aw))
            if c is not None:                       # P6
                csh, csh_pt = _share(c, dz), _share(c, dz, True)
                sh_pt = row["share_above_chance_lost"]
                caw = float(c[f"{dz}_aw_pt"])
                row.update(comp_auroc_overlap=caw, comp_auroc_placebo_mean=float(c[f"{dz}_ap_pt"]),
                           comp_share_lost=csh_pt, comp_share_lo=_ci(csh)[0], comp_share_hi=_ci(csh)[1],
                           kmer_minus_comp_share=sh_pt - csh_pt,
                           kmer_minus_comp_share_lo=_ci(sh - csh)[0], kmer_minus_comp_share_hi=_ci(sh - csh)[1],
                           kmer_over_comp_share=sh_pt / csh_pt,
                           kmer_over_comp_share_lo=_ci(sh / csh)[0], kmer_over_comp_share_hi=_ci(sh / csh)[1],
                           share_interpretable=bool(awp >= 0.5 and caw >= 0.5))
            rows.append(row)
    pairs = [("human_enhancers_cohn", "drosophila_enhancers_stark"), (LF_TEST, "drosophila_enhancers_stark"),
             (NT, "drosophila_enhancers_stark"), ("human_enhancers_cohn", LF_TEST)]
    for a, b in pairs:
        if a not in boots or b not in boots:
            continue
        for dz in ("fixed_model_overlap", "within_overlap"):
            da, db = boots[a], boots[b]
            ds_, dd_ = _share(da, dz) - _share(db, dz), _drop(da, dz) - _drop(db, dz)
            lo, hi = np.percentile(ds_, 2.5), np.percentile(ds_, 97.5)
            row = dict(kind="contrast", dataset="", contrast=f"{a} - {b}", design=dz,
                       drop=_drop(da, dz, True) - _drop(db, dz, True), drop_lo=np.percentile(dd_, 2.5),
                       drop_hi=np.percentile(dd_, 97.5),
                       share_above_chance_lost=_share(da, dz, True) - _share(db, dz, True), share_lo=lo, share_hi=hi,
                       ci_excludes_zero=bool(lo > 0 or hi < 0), B=len(ds_))
            if a in cboots and b in cboots:         # P6
                ca, cb = cboots[a], cboots[b]
                cs_ = _share(ca, dz) - _share(cb, dz)
                cs_pt = _share(ca, dz, True) - _share(cb, dz, True)
                doc = ds_ - cs_
                rat = _share(da, dz) / _share(ca, dz) - _share(db, dz) / _share(cb, dz)
                rat_pt = (_share(da, dz, True) / _share(ca, dz, True) - _share(db, dz, True) / _share(cb, dz, True))
                interp = all(float(x[f"{dz}_aw_pt"]) >= 0.5 for x in (da, db, ca, cb))
                row.update(comp_share_diff=cs_pt, comp_share_diff_lo=_ci(cs_)[0], comp_share_diff_hi=_ci(cs_)[1],
                           comp_ci_excludes_zero=bool(_ci(cs_)[0] > 0 or _ci(cs_)[1] < 0),
                           diff_of_contrasts=row["share_above_chance_lost"] - cs_pt,
                           diff_of_contrasts_lo=_ci(doc)[0], diff_of_contrasts_hi=_ci(doc)[1],
                           doc_ci_excludes_zero=bool(_ci(doc)[0] > 0 or _ci(doc)[1] < 0),
                           ratio_diff=rat_pt, ratio_diff_lo=_ci(rat)[0], ratio_diff_hi=_ci(rat)[1],
                           ratio_ci_excludes_zero=bool(_ci(rat)[0] > 0 or _ci(rat)[1] < 0),
                           share_interpretable=bool(interp))
            rows.append(row)
    return pd.DataFrame(rows)


# ================================================================ P5 selection bias of F2 trimming
def part_selbias_f2(ctx, arms):
    path = part(f"selbias_f2_{ctx.task}.csv")
    if os.path.exists(path):
        return
    by = {a.name: a for a in arms}
    seqs = list(ctx.tr) + list(ctx.te)
    y = np.r_[ctx.ytr, ctx.yte]
    ntr = len(ctx.ytr)
    f3, resid = R.trimer_features(seqs)
    centroid = f3[y == 1].mean(0)
    feats = pd.DataFrame(dict(      # identical definitions to run_psm_sensitivity.selection_bias
        length_acgt=[sum(ch in "ACGT" for ch in s) for s in seqs],
        trimer_l1_to_pos_centroid=np.abs(f3 - centroid).sum(1),
        trimer_markov1_residual=resid,
        tata_any=(np.r_[ctx.tata_tr, ctx.tata_te] > 0).astype(float),
        sp1_gcbox_any=(np.r_[ctx.sp1_tr, ctx.sp1_te] > 0).astype(float),
        lowcomplexity_frac=P.dust_lowcomplexity_frac(seqs),
        gc=np.r_[ctx.gc_tr, ctx.gc_te],
        in_model_logit_propensity=np.r_[ctx.ltr, ctx.lte]))
    if ctx.gann is not None:
        for c in ("genomic_mapped", "tss500_overlap", "repeatmasker_frac", "ccre_any", "ccre_els"):
            if c == "genomic_mapped" and ctx.gann[c].min() == 1:
                continue
            feats[c] = ctx.gann[c].to_numpy(float)

    def neg_mask(arm):
        m = np.zeros(len(y), bool)
        m[arm.tr_idx] = True
        m[ntr + arm.te_idx] = True
        return m & (y == 0)

    def sd2(a, b):
        s = np.sqrt((np.var(a) + np.var(b)) / 2) if len(a) and len(b) else np.nan
        return s if s > 0 else np.nan
    rows = []
    neg_all, pos_all = y == 0, y == 1
    sels = [(f"F2_trim_{q}", f"F2_trim_{q}_matched42", f"F2_trim_{q}_random{{}}") for q in F2_LEVELS]
    sels.append(("F1_hist_a1_seed42", "F1_hist_a1_matched42", "F1_hist_a1_random{}"))   # reference
    for sel, mname, rpat in sels:
        kept = neg_mask(by[mname])
        disc = neg_all & ~kept
        rms = [neg_mask(by[rpat.format(r)]) for r in R.RAND_SEEDS]
        for f in feats.columns:
            v = feats[f].to_numpy(float)
            ok = np.isfinite(v)
            vk, vd, vp = v[kept & ok], v[disc & ok], v[pos_all & ok]
            rmeans = [np.mean(v[m & ok]) for m in rms if (m & ok).any()]
            rows.append(dict(dataset=ctx.task, role=ctx.role, selection=sel, feature=f,
                             n_retained=int(kept.sum()), n_discarded=int(disc.sum()),
                             retained_neg_frac=kept.sum() / neg_all.sum(),
                             mean_pos=np.mean(vp), mean_neg_all=np.mean(v[neg_all & ok]),
                             mean_retained=np.mean(vk) if len(vk) else np.nan,
                             mean_discarded=np.mean(vd) if len(vd) else np.nan,
                             mean_random_sizematched=float(np.mean(rmeans)),
                             smd_retained_vs_discarded=(np.mean(vk) - np.mean(vd)) / sd2(vk, vd) if len(vk) and len(vd) else np.nan,
                             smd_retained_vs_pos=(np.mean(vk) - np.mean(vp)) / sd2(vk, vp) if len(vk) else np.nan))
    pd.DataFrame(rows).to_csv(path, index=False)
    log(f"[P5] {ctx.task}: F2 selection-bias rows written")


# ================================================================ main
def combine(B):
    def cat(prefix, out, exclude=None):
        fs = sorted(f for f in os.listdir(PARTS) if f.startswith(prefix) and f.endswith(".csv")
                    and not (exclude and f.startswith(exclude)))
        if fs:
            pd.concat([pd.read_csv(os.path.join(PARTS, f)) for f in fs], ignore_index=True).to_csv(
                os.path.join(OUT, out), index=False)
            log("wrote", out, f"({len(fs)} parts)")
    cat("sweep_", "psm_sensitivity_posthoc_sweep.csv")
    cat("balance_", "psm_sensitivity_posthoc_balance.csv")
    cat("leakage_", "psm_sensitivity_leakage_check.csv", exclude="leakage_interval_")   # P0
    cat("leakage_interval_", "psm_sensitivity_leakage_interval.csv")                   # P0b
    cat("selbias_f2_", "psm_sensitivity_selection_bias_f2.csv")
    cat("decomp_", "psm_sensitivity_decomposition_ci.csv")                            # P7
    cat("f2band_", "psm_sensitivity_f2_band.csv")                                     # P8
    bt = between_table(["human_enhancers_cohn", NT, LF_TEST, "drosophila_enhancers_stark"], B)
    if len(bt):
        bt.to_csv(os.path.join(OUT, "psm_sensitivity_between_dataset.csv"), index=False)
        log("wrote psm_sensitivity_between_dataset.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=C.BOOT)
    ap.add_argument("--parts", default="P0,P0b,P1,P2,P3,P4,P5,P6,P7,P8")
    ap.add_argument("--datasets", default="nt_enhancers,human_enhancers_cohn,drosophila_enhancers_stark")
    args = ap.parse_args()
    todo = set(args.parts.split(","))
    os.makedirs(PARTS, exist_ok=True)
    main_sw = pd.read_csv(os.path.join(OUT, "psm_sensitivity_sweep.csv"),
                          usecols=["dataset", "design", "model", "test_arm", "auroc", "n_neg"])
    roles = dict(R.DATASETS)
    tasks = args.datasets.split(",")
    for task in tasks:              # everything except the long P2 refit
        ctx = R.Ctx(task, roles[task])
        arms = R.build_arms(ctx)
        check_against_main(ctx, arms, main_sw)
        if "P0" in todo:
            part_leakage(ctx)
        if "P1" in todo and task == NT:
            part_leakfree_test(ctx, arms, args.boot)
        if "P3" in todo:
            part_balw(ctx, arms, args.boot)
        if "P4" in todo:
            part_between_boot(ctx, arms, args.boot)
            if task == NT:
                part_between_boot(ctx, arms, args.boot, label=LF_TEST, keep=~dup_masks(ctx)[0])
        if "P5" in todo:
            part_selbias_f2(ctx, arms)
        if "P0b" in todo:
            part_interval_leakage(ctx, arms)
        if "P6" in todo:            # needs the P4 draws of the same label
            part_between_boot(ctx, arms, args.boot)
            part_between_comp_boot(ctx, arms, args.boot)
            if task == NT:
                part_between_boot(ctx, arms, args.boot, label=LF_TEST, keep=~dup_masks(ctx)[0])
                part_between_comp_boot(ctx, arms, args.boot, label=LF_TEST, keep=~dup_masks(ctx)[0])
        if "P7" in todo:
            part_decomp(ctx, arms, args.boot)
            if task == NT:
                part_decomp(ctx, arms, args.boot, label=LF_TEST, keep=~dup_masks(ctx)[0])
                if os.path.exists(os.path.join(main_cache(LF_REFIT), "F1_hist_a1_matched44.npz")):
                    part_decomp(ctx, arms, args.boot, label=LF_REFIT, pred_task=LF_REFIT)
        if "P8" in todo:
            part_f2band(ctx, arms)
            if task == NT:
                part_f2band(ctx, arms, label=LF_TEST, keep=~dup_masks(ctx)[0])
        combine(args.boot)
        del ctx, arms
    if "P2" in todo and NT in tasks:
        ctx = R.Ctx(NT, roles[NT])
        arms = R.build_arms(ctx)
        part_leakfree_refit(ctx, arms, args.boot)
        combine(args.boot)
    log("done")


if __name__ == "__main__":
    main()
