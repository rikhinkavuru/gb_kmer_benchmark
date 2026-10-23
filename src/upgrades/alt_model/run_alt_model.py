#!/usr/bin/env python3
"""Reviewer 2, comment 3 -- do the route labels depend on k-mer spectra + LightGBM?

Re-derives every MODEL-DEPENDENT route diagnostic with two alternative interpretable models on the
SAME k-mer features (alt_models.py): (a) L1/L2 logistic regression with penalty and C tuned on a
validation carve of train, (b) an Explainable Boosting Machine (glass-box GA2M) on the 20-d
composition signature + the 128 most label-associated k-mers (train-only ANOVA-F filter). The
paper's LightGBM is refit through the identical code path as the reference (fixed hyperparameters,
1 thread). This script is the WORKER: it fits one "unit" = (task, arm, feature set, model) at a
time and checkpoints the test-set class probabilities to external/alt_model/units/ (resumable; an
existing unit is skipped). analyze_alt_model.py turns the units into statistics + route labels.

PRE-REGISTERED ANALYSIS PLAN (written before any alternative model was fit)
---------------------------------------------------------------------------
Which diagnostics are model-dependent. Route labels rest on two kinds of evidence. MODEL-FREE
sequence statistics (motif-only AUROCs, TATA negative-excess, GC AUROC, the TATA/GC collapse under
equalization/shuffling, the control-motif panel, coordinate analyses) are identical under any
classifier by construction and are NOT recomputed. MODEL-DEPENDENT readouts are recomputed here:
  bench  benchmark MCC/AUROC, all 15 tasks, original split (train capped at 50,000 sequences,
         stratified, seed 42, for the 6 tasks above that size -- regulatory, ocr, enhancers_ensembl,
         both demo tasks, nt_promoter_all -- identically for all three models; test sets never
         capped). Composition-only (20-d) arm on the 12 binary tasks.
  pos    Route 2: B=1 vs B=8 position-binned spectrum (global spectrum hstacked with the 8-bin
         spectrum, exactly positional_sweep.featurize_arm) on the positional_sweep split
         (both splits capped at 20,000 with positional_sweep.cap_split, seeds 42/43), all 15 tasks.
  eq     Route 3: the paper's composition-equalized splits (cleaned_splits_v2 for cohn,
         nt_enhancers, drosophila; cleaned_splits_specificity for human_enhancers_ensembl,
         human_ocr_ensembl), k-mer and composition-only arms (train capped at 50,000).
  shuf   Route 3: dinucleotide-shuffled negatives built exactly as run_shuffled_neg.py (seed 42)
         for cohn, nt_enhancers, drosophila; k-mer and composition-only arms.
  dose   Route 1 (run_dose_alt.py): the 135-setting semi-synthetic dose-response, same data.
k = the task's LightGBM validation-selected k (results/val_selection.csv:new_k) for every model.
Statistics: 95% percentile bootstrap, 1000 resamples, test sequence = unit, seed 42 (paired where
two arms share a test set); task-level Spearman CIs = bias-corrected bootstrap over tasks, 10,000
resamples, seed 42 (as presence_null).

Route-label decision rules (fixed; they reproduce the camera-ready (v2) Table-1 routes from the
paper's LightGBM numbers -- verified from the released CSVs BEFORE any alternative model was fit --
and are then applied verbatim to each model):
  1. SOLVED (non-failing)  if benchmark MCC >= 0.70 (Table-1 tier cut; diagnose.py SOLVED rule).
  2. POSITIONAL            else if Route-2 delta-MCC(B=8 - B=1) > 0.20 and its 95% CI excludes 0
                           (diagnose.py rule).
  3. CONTAMINATION         else if the MODEL-FREE gate holds (TATA-box negative excess >= 0.10 and
                           TATA motif-only AUROC < 0.5; diagnose.py rule, values from
                           results/{,nt/}contamination_analysis.csv or alt_model_gate.csv) AND the
                           task is composition-solved: on its composition-equalized split the
                           k-mer model's MCC margin over the composition-only model (paired
                           bootstrap) has a 95% CI lower bound <= 0 ("collapses onto the
                           composition-only baseline", the v2 criterion that separates cohn and
                           nt_enhancers from human_enhancers_ensembl).
  4. SHARED-MOTIF          otherwise (a failing task with none of the above).
  Sensitivity (reported, not primary): rule 3 with "composition fraction
  (AUROC_comp-0.5)/(AUROC_full-0.5) >= 0.80 on the original split" replacing the equalized margin.
Pre-registered preservation criterion: a task's label is PRESERVED under a model if (i) the rules
give the same label as the paper, and (ii) each label-defining model-dependent effect keeps its sign
with a 95% CI excluding 0: solved -> MCC-0.70 > 0; any failing label -> 0.70-MCC > 0; positional ->
delta-MCC(B8-B1) > 0; contamination -> equalized-split MCC minus original MCC < 0 (unpaired
bootstrap). Suite-level effects: Route 1 Spearman(discriminability, MCC) over the 11 TF-motif tasks
> 0 with CI excluding 0 (and on the n=9 homology-clean subset); dose-response slope dMCC/dd > 0 with
CI excluding 0; the presence null (Spearman with #enriched TFs, GB n=9) is preserved if its CI
includes 0. Every disagreement is reported with its explanation; nothing is tuned toward agreement.

POST HOC ADDENDUM (2026-09-26, written after the relaunch and BEFORE any unit of the relaunched run
was fit; the pre-registered text above is unchanged). Everything below is labelled post hoc in all
outputs. What had been seen before writing it: (a) nine units of the interrupted 2026-09-23 run
(dummy_mouse bench x3 models, three LR bench units) -- DISCARDED (built from pre-data-fix GB loads;
moved to external/alt_model/stale_20260923/); (b) one timing smoke test on cohn (lgbm reproduced
results.csv exactly: MCC 0.461223 / AUROC 0.804649; LR MCC 0.453, EBM 0.423); (c) the released
psm_sensitivity results (R1.4), which showed that the default-threshold MCC collapse on the
composition-equalized split is reproduced by size-matched RANDOM negatives (the cleaned cohn test set
is 94% positive), so an MCC-based equalization readout is not evidence of shortcut removal.
  P1  Size-controlled Route-3 equalization (psm_sensitivity design): arm eqr<s>, s = 0..9 = size-
      matched RANDOM negatives -- all positives kept, negatives drawn uniformly at random from the
      ORIGINAL split's negatives to exactly the per-split negative counts of the published equalized
      split (random_take/pri of psm_sensitivity: nested, RandomState([seed, split_id]), split_id 0 =
      train, 1 = test); train capped at 50,000 like the eq arm; k-mer and composition-only arms, all
      three models. Tasks: cohn, nt_enhancers, drosophila (tier 6), human_enhancers_ensembl,
      human_ocr_ensembl (tier 7). Readouts (AUROC, analyze_alt_model.py): composition effect =
      mean_s AUROC(eqr<s>) - AUROC(eq) with a two-level bootstrap CI; share of above-chance AUROC
      lost (mean Ar - Am)/(mean Ar - 0.5); contaminated-minus-control difference in that share;
      and the FIXED-MODEL design (the original-split model, no retraining, scored on the equalized
      test subset vs the random test subsets). For nt_enhancers every readout is also given on the
      LEAK-FREE test (the 100 test positives that occur verbatim in training removed), which the
      psm analysis made primary for that task.
  P2  bench_full: the 6 capped tasks refit on their FULL training split (no 50k cap), all three
      models, so the LightGBM reference equals Table 1 and the cap is shown not to drive any label.
  P3  pos_full: nt_splice_sites_all B=1 vs B=8 on its full split (27,000/3,000), the setting of the
      paper's headline Route-2 number (0.244 -> 0.868).
  P4  lrx: LR with the grid extended upward (C up to 10, alt_models.LR_C_GRID_EXT), run on every
      bench/pos/eq/shuf unit where the pre-registered LR chose the upper grid edge C = 0.1; reported
      as a sensitivity, never replacing the pre-registered LR.
  P4b lra: LR with the pre-registered grid but tuned by validation AUROC instead of validation MCC.
      Reason (seen in the smoke test above): on the ~95%-positive equalized splits every grid point
      has validation MCC 0 at the default threshold, so MCC tuning falls back to the most-regularized
      entry (L1, C=3e-4) = an all-zero, constant-score model (AUROC exactly 0.5). The AUROC-based
      Route-3 readouts (eq, eqr, composition fraction, shuffled) are therefore also reported with lra;
      the pre-registered lr stays primary and degenerate (all-zero) fits are flagged.
  P5  Attribution check: every bench k-mer unit stores the model's per-k-mer importance (lgbm gain,
      LR |coef|, EBM term importance); analyze_alt_model.py runs the paper's per-TF gain-weighted
      enrichment test (run_interpretability.enrichment_z + BH-FDR 0.05, JASPAR 2024 CORE) on each
      model's importances at the SAME k and compares the enriched-TF sets (top TF, overlap) and the
      presence null computed with each model's own enriched-TF count.
  P7  ebm512 (added 2026-09-26 ~16:20, after the pre-registered bench units showed the 128-k-mer EBM
      below LightGBM on k=6 tasks, e.g. human_nontata_promoters MCC 0.673 vs 0.874): the same EBM given
      the 512 (instead of 128) train-ANOVA-top k-mers, on every bench k-mer unit (all 15 tasks) --
      a capacity sensitivity for the pre-registered feature filter, never replacing the primary EBM.
  P8  (analysis only; added 2026-09-26 ~20:20 after seeing the P1 contrast) psm_sensitivity's removal-free
      overlap-weighting design for every model: the ORIGINAL-split model scored on the original test set
      under overlap weights (LR propensity on the 20-d composition, pos 1-e / neg e) vs 10 permuted-weight
      placebos (psm's seeds); share of above-chance AUROC lost and the cohn/nt minus drosophila contrast.
      Reason: the psm module's "control affected less" (0.29 vs 0.64) comes from THIS design, while the
      size-matched random-negative design (P1) has only ~220 test negatives per task and is much noisier.
  P6  A label map for the SUBMITTED BIBM Table-1 vocabulary (composition / solved / contamination
      (weak) / shared-motif / positional), in addition to the pre-registered camera-ready (v2) rules.
  Clarification of P4 (2026-09-27): run_all.sh fits lrx on EVERY tier-1..5 unit (a superset of the
      upper-edge units P4 names; run_post.sh completes the edge units of tiers 8/9 with --only-lr-edge).
      All fitted lrx units are used; on a non-edge unit lrx can still pick a larger C if its validation
      MCC is higher, so lrx is a sensitivity everywhere, never a replacement for lr.

FIX-ROUND-1 ADDENDUM (2026-09-27, written after independent verification of the first report and
BEFORE any of the units/analyses below were run; everything here is post hoc and labelled so; nothing
above is changed). What had been seen: all outputs of the first report, including that the EBM's
pre-registered train-ANOVA filter admitted ZERO position-binned columns into the B=8 design for
demo_coding_vs_intergenomic_seqs, human_enhancers_cohn and nt_enhancers_types (B8 == B1 by
construction) and only 2-13 of 128 for several others.
  P9  ebmpos: the EBM's Route-2 B=8 arm refit with the global and the position-binned blocks ranked
      SEPARATELY (alt_models.ebm_design_pos: 20-d composition + top-128 global + top-128 binned k-mers by
      train-only ANOVA F), on the pos arm of all 15 tasks and the pos_full arm of splice. The global
      selection equals the pre-registered EBM's B=1 selection, so the B=1 comparator is the existing ebm
      B1 unit and the B=8 design strictly contains the B=1 design. Rule (fixed now, before fitting): the
      pre-registered positional rule (dMCC(B8-B1) > 0.20 with paired 95% CI lower bound > 0) applied to
      ebmpos; reported as a sensitivity of the EBM Route-2 test (the pre-registered ebm rows stay primary),
      and the label scheme is re-run with ebmpos's positional rule (all other components from ebm).
  P10 analysis-only additions (analyze_alt_model.py): (a) Route-1 Spearman / LOTO / presence null also
      on the bench_full-where-available MCCs (uncapped; = the MCCs behind the paper's Table corr for
      LightGBM), the capped set staying the pre-registered one; jackknife argmin/argmax tasks reported;
      (b) the P1 bootstrap implemented as the two-level PAIRED bootstrap of psm_sensitivity run_posthoc P7
      (level 1: the original test sequences, shared by the original, matched and random arms, which are
      all subsets of it -- RandomState(42); level 2: the 10 random-negative seeds resampled with
      replacement -- RandomState(43)); previously level 1 was drawn independently per arm and level 2 was
      missing; (c) default-threshold MCC of the random-negative arms (mean, min, max) next to the
      matched and original MCC, with the same two-level CI, to test per model whether size-matched random
      negatives reproduce the pre-registered MCC collapse; (d) label bookkeeping under two post hoc
      contamination criteria (size-controlled AUROC effect, leak-free test for nt_enhancers as P1 made
      primary; overlap-design contrast vs the drosophila control, P8), and a flag for the clean control
      when the model-dependent half of rule 3 (equalized-margin collapse) fires on it.

FIX-ROUND-2 ADDENDUM (2026-09-27, analysis only -- no new units; written after the second independent
verification and BEFORE the analyses below were run; post hoc and labelled so; nothing above is
changed). What had been seen: all fix-round-1 outputs, and a verifier's scratch recompute of the P8
overlap design on human_enhancers_ensembl and human_ocr_ensembl (both lose a SMALLER share than the
drosophila control under LightGBM/LR/EBM; the EBM enh_ensembl CI upper bound +0.004).
  P11 (a) The P8 overlap design (unchanged: propensity, weights, placebos, seeds) extended to every
      gate-positive binary task and to the two P1 specificity tasks: human_enhancers_ensembl,
      human_ocr_ensembl (their `bench` units are train-capped, so the uncapped `bench_full` units are
      used -- same original test set) and human_nontata_promoters (gate-positive; `bench` is uncapped).
      Contrast = share lost, task minus the drosophila control, for every task (same paired-draw
      construction as P8).
      (b) Post hoc LABELLING scheme `posthoc_overlap`: the pre-registered rules with the model-dependent
      half of rule 3 replaced by "overlap-design contrast vs drosophila has 95% CI lower bound > 0"
      (leak-free test for nt_enhancers, as in P10d); a gate-positive failing task without an overlap
      row is 'not assessed'. Rule fixed now, before running. Variants without their own units inherit
      the parent model's rows exactly as for every other arm (ebm512/ebmpos <- ebm, lrx <- lr).
      Reported next to primary/sensitivity/bibm; never replaces them.
      (c) P10c flag `random_reproduces_mcc_collapse` renamed `mcc_comp_effect_ci_includes_0` (it tested
      only whether the random-minus-matched MCC CI includes 0, i.e. absence of evidence); the share of
      the MCC drop that size-matched random negatives already produce, (M0 - mean Mr)/(M0 - Mm), gets the
      same two-level bootstrap CI and becomes the reported MCC readout.
      (d) Route-1 `_uncapped` rows are written only for models with >= 1 uncapped substitution in the
      task set (lra/ebm512 have none), with the substitution count; the scipy Spearman t-approximation
      p-value is reported next to the permutation p (to identify the method behind the paper's p).


CLI:  python run_alt_model.py --models lgbm,lr,ebm --tiers 1,2,3,4,5      (pre-registered units)
      python run_alt_model.py --models lgbm,lr,ebm --tiers 6,7,8,9          (post hoc P1-P3)
      python run_alt_model.py --models lrx --tiers 1,2,3,4,5,8,9            (post hoc P4, every unit)
      python run_alt_model.py --models lra --tiers 1,3,5,6,7                (post hoc P4b)
      python run_alt_model.py --models ebm512 --tiers 1 --feats kmer        (post hoc P7)
      python run_alt_model.py --models ebmpos --tiers 2,4,9 --feats B8      (post hoc P9)
      (the full ordered queue is run_all.sh + run_post.sh + run_fix1.sh; fix round 2 = run_fix2.sh,
      analysis only)
Run everything through bibm/lowcpu.sh (3 threads, nice 20).
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.abspath(os.path.join(HERE, ".."))
for _d in (UP, os.path.join(UP, "positional_sweep"), os.path.join(UP, "shuffled_neg"), HERE):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import _common as C                      # noqa: E402
import featurize                         # noqa: E402
import alt_models as AM                  # noqa: E402
from cache_seqs import load_cached       # noqa: E402

UNIT_DIR = os.path.join(C.ROOT, "external", "alt_model", "units")
CAP_BENCH = 50000
CAP_POS = 20000
N_BINS = 8
N_RAND = 10                               # P1: random size-matched seeds 0..9 (psm_sensitivity)
TASKS = ["dummy_mouse_enhancers_ensembl", "drosophila_enhancers_stark", "nt_promoter_tata",
         "nt_enhancers", "nt_enhancers_types", "human_enhancers_cohn", "human_nontata_promoters",
         "nt_splice_sites_all", "nt_promoter_no_tata", "nt_promoter_all", "demo_human_or_worm",
         "demo_coding_vs_intergenomic_seqs", "human_enhancers_ensembl", "human_ocr_ensembl",
         "human_ensembl_regulatory"]                     # ascending training-set size
MULTICLASS = {"human_ensembl_regulatory", "nt_enhancers_types", "nt_splice_sites_all"}
POS_REQUIRED = ["nt_splice_sites_all", "nt_promoter_tata", "human_nontata_promoters",
                "drosophila_enhancers_stark", "nt_enhancers", "human_enhancers_cohn"]
EQ_SPLITS = {
    "human_enhancers_cohn": "cleaned_splits_v2",
    "nt_enhancers": "cleaned_splits_v2",
    "drosophila_enhancers_stark": "cleaned_splits_v2",
    "human_enhancers_ensembl": "cleaned_splits_specificity",
    "human_ocr_ensembl": "cleaned_splits_specificity",
}
SHUF_TASKS = ["human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark"]
EQR_A = ["human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark"]      # tier 6
EQR_B = ["human_enhancers_ensembl", "human_ocr_ensembl"]                             # tier 7
FULL_TASKS = ["nt_promoter_all", "demo_human_or_worm", "demo_coding_vs_intergenomic_seqs",
              "human_enhancers_ensembl", "human_ocr_ensembl", "human_ensembl_regulatory"]  # tier 8
POSFULL_TASKS = ["nt_splice_sites_all"]                                              # tier 9
POSTHOC_ARMS = ("eqr", "bench_full", "pos_full")
LR_BIG_NNZ = 30_000_000                   # above this many nonzeros the LR grid runs serially
EBM_M_POSTHOC = 512                       # P7: EBM given 512 (not 128) train-ANOVA-top k-mers


def best_k_map():
    import pandas as pd
    vs = pd.read_csv(os.path.join(C.ROOT, "results", "val_selection.csv"))
    return {r["dataset"]: int(r["new_k"]) for _, r in vs.iterrows()}


def unit_list(tiers, tasks_filter=None):
    """Priority-ordered list of (task, arm, feat)."""
    U = []
    if 1 in tiers:
        U += [(t, "bench", "kmer") for t in TASKS]
        U += [(t, "bench", "comp") for t in TASKS if t not in MULTICLASS]
    if 2 in tiers:
        for t in POS_REQUIRED:
            U += [(t, "pos", "B1"), (t, "pos", "B8")]
    if 3 in tiers:
        for t in EQ_SPLITS:
            U += [(t, "eq", "kmer"), (t, "eq", "comp")]
    if 4 in tiers:
        for t in TASKS:
            if t not in POS_REQUIRED:
                U += [(t, "pos", "B1"), (t, "pos", "B8")]
    if 5 in tiers:
        for t in SHUF_TASKS:
            U += [(t, "shuf", "kmer"), (t, "shuf", "comp")]
    if 6 in tiers:
        for t in EQR_A:
            U += [(t, f"eqr{s}", f) for s in range(N_RAND) for f in ("kmer", "comp")]
    if 7 in tiers:
        for t in EQR_B:
            U += [(t, f"eqr{s}", f) for s in range(N_RAND) for f in ("kmer", "comp")]
    if 8 in tiers:
        for t in FULL_TASKS:
            U += [(t, "bench_full", "kmer")]
            if t not in MULTICLASS:
                U += [(t, "bench_full", "comp")]
    if 9 in tiers:
        for t in POSFULL_TASKS:
            U += [(t, "pos_full", "B1"), (t, "pos_full", "B8")]
    if tasks_filter:
        U = [u for u in U if u[0] in tasks_filter]
    return U


def unit_path(task, arm, feat, model):
    return os.path.join(UNIT_DIR, f"{task}__{arm}__{feat}__{model}")


def fingerprint(seqs, y):
    h = hashlib.sha1("\n".join(seqs).encode())
    h.update(np.asarray(y, np.int64).tobytes())
    return h.hexdigest()[:16]


def pri(seed, split_id, n):
    """== psm_sensitivity.run_psm_sensitivity.pri: fixed random priority per sequence."""
    return np.random.RandomState([int(seed), int(split_id)]).uniform(size=n)


def random_take(n_pool, k, priority):
    """== psm_sensitivity.psm_core.random_take: the k members with the smallest priority."""
    return np.sort(np.argsort(priority, kind="stable")[:k]) if k > 0 else np.array([], dtype=int)


def eq_counts(task):
    """Per-split negative counts of the published equalized split (P1 size targets)."""
    import pandas as pd
    d = os.path.join(C.ROOT, EQ_SPLITS[task])
    out = []
    for sp in ("train", "test"):
        y = pd.read_csv(os.path.join(d, f"{task}_comp_equalized_{sp}.csv"), usecols=["label"])["label"]
        out.append(int((y == 0).sum()))
    return out


def map_to_original(te, yte, te_orig, yte_orig):
    """Index of each (sequence, label) of `te` in the ORIGINAL test split (equal pairs share an index;
    their features and hence predictions are identical)."""
    look = {}
    for i, (s, y) in enumerate(zip(te_orig, yte_orig)):
        look.setdefault((s.upper(), int(y)), i)
    return np.array([look[(s.upper(), int(y))] for s, y in zip(te, yte)], np.int64)


# ------------------------------------------------------------------ data per arm
_DATA_CACHE = {}


def arm_data(task, arm):
    """(tr, ytr, te, yte, meta) for one (task, arm); memoized (small LRU)."""
    key = (task, arm)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    if len(_DATA_CACHE) > 3:
        _DATA_CACHE.clear()
    from run_positional_sweep import cap_split
    meta = {"posthoc": arm.startswith(POSTHOC_ARMS)}
    tr, ytr, te, yte = load_cached(task)
    ytr = np.asarray(ytr, np.int64); yte = np.asarray(yte, np.int64)
    te_orig, yte_orig = list(te), yte
    meta.update(n_train_orig=len(ytr), n_test_orig=len(yte))
    te_idx = np.arange(len(yte), dtype=np.int64)          # indices into the ORIGINAL test split
    if arm == "bench":
        tr, ytr, capped = cap_split(tr, ytr, CAP_BENCH, C.SEED)
        meta.update(train_capped=bool(capped), test_capped=False)
    elif arm == "bench_full":
        meta.update(train_capped=False, test_capped=False)
    elif arm == "pos":
        tr, ytr, c1 = cap_split(tr, ytr, CAP_POS, C.SEED)
        te, yte, c2 = cap_split(te, yte, CAP_POS, C.SEED + 1)
        meta.update(train_capped=bool(c1), test_capped=bool(c2))
        te_idx = None
    elif arm == "pos_full":
        meta.update(train_capped=False, test_capped=False)
    elif arm == "eq":
        d = os.path.join(C.ROOT, EQ_SPLITS[task])
        tr, ytr, te, yte = C.load_csv_pair(os.path.join(d, f"{task}_comp_equalized_train.csv"),
                                           os.path.join(d, f"{task}_comp_equalized_test.csv"))
        meta.update(n_train_eq=len(ytr), n_test_eq=len(yte), eq_source=EQ_SPLITS[task])
        te_idx = map_to_original(te, yte, te_orig, yte_orig)
        tr, ytr, capped = cap_split(tr, ytr, CAP_BENCH, C.SEED)
        meta.update(train_capped=bool(capped), test_capped=False)
    elif arm.startswith("eqr"):
        seed = int(arm[3:])
        n_neg = eq_counts(task)
        sel = []
        for sid, y in enumerate((ytr, yte)):
            pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
            keep = neg[random_take(len(neg), n_neg[sid], pri(seed, sid, len(neg)))]
            sel.append(np.sort(np.concatenate([pos, keep])))
        tr = [tr[i] for i in sel[0]]; ytr = ytr[sel[0]]
        te = [te[i] for i in sel[1]]; yte = yte[sel[1]]; te_idx = sel[1].astype(np.int64)
        meta.update(random_seed=seed, n_train_eqr=len(ytr), n_test_eqr=len(yte))
        tr, ytr, capped = cap_split(tr, ytr, CAP_BENCH, C.SEED)
        meta.update(train_capped=bool(capped), test_capped=False)
    elif arm == "shuf":
        import run_shuffled_neg as SN
        rng = np.random.RandomState(C.SEED)
        pos_tr = [s for s, y in zip(tr, ytr) if y == 1]; pos_te = [s for s, y in zip(te, yte) if y == 1]
        shuf_tr, p1 = SN.shuffled_negatives(pos_tr, rng)
        shuf_te, p2 = SN.shuffled_negatives(pos_te, rng)
        tr = list(pos_tr) + list(shuf_tr); ytr = np.array([1] * len(pos_tr) + [0] * len(shuf_tr), np.int64)
        te = list(pos_te) + list(shuf_te); yte = np.array([1] * len(pos_te) + [0] * len(shuf_te), np.int64)
        meta.update(dinuc_preserved=round((p1 * len(pos_tr) + p2 * len(pos_te)) / (len(pos_tr) + len(pos_te)), 4),
                    train_capped=False, test_capped=False,
                    # dinuc_shuffle iterates set(seq): the draw depends on the string-hash salt
                    pythonhashseed=os.environ.get("PYTHONHASHSEED", "unset"), set_order="".join(set("ACGT")),
                    shuffle_sha1=hashlib.sha1("".join(list(shuf_tr) + list(shuf_te)).encode()).hexdigest()[:12])
        te_idx = None
    else:
        raise ValueError(arm)
    meta.update(n_train=len(ytr), n_test=len(yte), test_fp=fingerprint(te, yte),
                train_class_counts=np.bincount(ytr).tolist(), test_class_counts=np.bincount(yte).tolist())
    _DATA_CACHE[key] = (list(tr), ytr, list(te), yte, meta, te_idx)
    return _DATA_CACHE[key]


def features(tr, te, feat, k):
    """(Xtr, Xte, names) for a feature set. comp = 20-d dense composition signature."""
    if feat == "comp":
        return C.comp_signature(tr), C.comp_signature(te), list(AM.COMP_NAMES)
    voc = featurize.vocabulary(k)
    if feat in ("kmer", "B1"):
        return featurize.kmer_spectrum(tr, k), featurize.kmer_spectrum(te, k), voc
    if feat == "B8":
        from run_positional_sweep import featurize_arm
        names = voc + [f"bin{b}:{v}" for b in range(N_BINS) for v in voc]
        return featurize_arm(tr, k, N_BINS), featurize_arm(te, k, N_BINS), names
    raise ValueError(feat)


# ------------------------------------------------------------------ one model on one unit
def fit_one(model, Xtr, ytr, Xte, tr, te, names, feat):
    """Returns (proba, classes, info, importance-or-None)."""
    nnz = Xtr.nnz if sparse.issparse(Xtr) else Xtr.size
    imp = None
    if model == "lgbm":
        m, info = AM.fit_lgbm(Xtr, ytr, n_jobs=3)
        proba = m.predict_proba(Xte); classes = m.classes_
        imp = AM.lgbm_importance(m)
    elif model in ("lr", "lrx", "lra"):
        grid = AM.LR_C_GRID_EXT if model == "lrx" else AM.LR_C_GRID
        crit = "auroc" if model == "lra" else "mcc"
        pp, info, m, coefs = AM.fit_lr_tuned(Xtr, ytr, grid_c=grid, n_jobs=(3 if nnz < LR_BIG_NNZ else 1),
                                             criterion=crit)
        proba = pp(Xte); classes = m.classes_
        imp = AM.lr_importance(coefs)
        if coefs.shape[0] == 1:
            w = coefs[0]; o = np.argsort(w)
            info["top_pos_features"] = [(names[i], round(float(w[i]), 4)) for i in o[::-1][:10]]
            info["top_neg_features"] = [(names[i], round(float(w[i]), 4)) for i in o[:10]]
    elif model in ("ebm", "ebm512", "ebmpos"):
        mk = AM.EBM_M if model in ("ebm", "ebmpos") else EBM_M_POSTHOC
        if model == "ebmpos":                                   # post hoc P9: B8 only
            if feat != "B8":
                raise ValueError("ebmpos is defined for the B8 arm only (its B1 comparator is the ebm B1 unit)")
            n_global = sum(1 for n_ in names if not n_.startswith("bin"))
            Etr, Ete, fn, sel = AM.ebm_design_pos(Xtr, Xte, C.comp_signature(tr), C.comp_signature(te), ytr,
                                                  names, n_global, mk, mk)
        elif feat == "comp":
            Etr, Ete, fn = np.asarray(Xtr, np.float64), np.asarray(Xte, np.float64), list(names)
            sel = None
        else:
            idx = AM.ebm_select(Xtr, ytr, mk)
            Etr, Ete, fn = AM.ebm_design(Xtr, Xte, C.comp_signature(tr), C.comp_signature(te), ytr, names, m=mk)
            sel = idx
        m, info = AM.fit_ebm(Etr, ytr, n_jobs=3, feature_names=fn)
        proba = m.predict_proba(Ete); classes = m.classes_
        dimp = AM.ebm_importance(m, Etr.shape[1])
        if sel is None:
            imp = dimp
        else:
            imp = np.zeros(len(names)); imp[sel] = dimp[len(AM.COMP_NAMES):]
            info["ebm_comp_importance"] = [round(float(v), 6) for v in dimp[:len(AM.COMP_NAMES)]]
            info["ebm_kmer_features"] = [fn[j] for j in range(len(AM.COMP_NAMES), len(fn))]
    else:
        raise ValueError(model)
    return np.asarray(proba, np.float64), np.asarray(classes), info, imp


def save_unit(path, out, proba, classes, yte, te_idx, imp):
    arrs = dict(proba=np.asarray(proba, np.float64), classes=classes, y=np.asarray(yte, np.int64))
    if te_idx is not None:
        arrs["te_idx"] = np.asarray(te_idx, np.int64)
    if imp is not None:
        arrs["importance"] = np.asarray(imp, np.float32)
    np.savez_compressed(path + ".tmp.npz", **arrs)
    os.replace(path + ".tmp.npz", path + ".npz")
    with open(path + ".json.tmp", "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(path + ".json.tmp", path + ".json")


def done(task, arm, feat, model):
    p = unit_path(task, arm, feat, model)
    return os.path.exists(p + ".json") and os.path.exists(p + ".npz")


def maybe_alias_b1(task, model):
    """pos/B1 is byte-identical to bench/kmer when neither positional split is capped (both use the
    full original split at the same k) -- reuse it instead of refitting (recorded in the JSON)."""
    tr, ytr, te, yte, meta, _ = arm_data(task, "pos")
    if meta["train_capped"] or meta["test_capped"]:
        return False
    src = unit_path(task, "bench", "kmer", model)
    if not (os.path.exists(src + ".json") and os.path.exists(src + ".npz")):
        return False
    with open(src + ".json") as fh:
        out = json.load(fh)
    if out.get("test_fp") != meta["test_fp"] or out.get("n_train") != meta["n_train"]:
        return False
    d = np.load(src + ".npz", allow_pickle=False)
    out.update(arm="pos", feat="B1", aliased_from="bench/kmer")
    save_unit(unit_path(task, "pos", "B1", model), out, d["proba"].astype(np.float64), d["classes"],
              d["y"], None, None)
    return True


def lr_edge(task, arm, feat):
    p = unit_path(task, arm, feat, "lr") + ".json"
    if not os.path.exists(p):
        return False
    return bool(json.load(open(p))["info"].get("C") == AM.LR_C_GRID[-1])


def versions():
    import lightgbm, sklearn, interpret, scipy
    return dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                lightgbm=lightgbm.__version__, interpret=interpret.__version__, python=sys.version.split()[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="lgbm,lr,ebm")
    ap.add_argument("--tiers", default="1,2,3,4,5")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--only-lr-edge", action="store_true", help="P4: run only units where lr chose C=0.1")
    ap.add_argument("--feats", default="", help="restrict to these feature sets, e.g. kmer (P7 ebm512)")
    args = ap.parse_args()
    os.makedirs(UNIT_DIR, exist_ok=True)
    models = [m for m in args.models.split(",") if m]
    tiers = [int(t) for t in args.tiers.split(",") if t]
    kmap = best_k_map()
    units = []
    for t in tiers:                                   # tier order = priority order
        units += [u for u in unit_list({t}, set(args.tasks.split(",")) - {""}) if u not in units]
    if args.feats:
        units = [u for u in units if u[2] in set(args.feats.split(","))]
    ver = versions()
    print(f"[alt_model] {len(units)} units x models={models} | tiers={tiers} | {ver}", flush=True)
    for (task, arm, feat) in units:
        todo = [m for m in models if not done(task, arm, feat, m)]
        if args.only_lr_edge:
            todo = [m for m in todo if lr_edge(task, arm, feat)]
        if not todo:
            continue
        try:
            if arm == "pos" and feat == "B1":
                for m in list(todo):
                    if maybe_alias_b1(task, m):
                        print(f"  {task:<32}{arm:<11}{feat:<5}{m:<5} aliased to bench/kmer", flush=True)
                        todo.remove(m)
                if not todo:
                    continue
            k = kmap[task]
            tr, ytr, te, yte, meta, te_idx = arm_data(task, arm)
            t0 = time.time()
            Xtr, Xte, names = features(tr, te, feat, k)
            feat_s = round(time.time() - t0, 1)
            for model in todo:
                t1 = time.time()
                proba, classes, info, imp = fit_one(model, Xtr, ytr, Xte, tr, te, names, feat)
                nc = proba.shape[1]
                pred = classes[np.argmax(proba, axis=1)]
                mcc = AM.mcc_from_pred(yte, pred, nc); auc = AM.auroc(yte, proba)
                info["featurize_s"] = feat_s
                out = dict(task=task, arm=arm, feat=feat, model=model, k=k, n_features=int(Xtr.shape[1]),
                           mcc=round(mcc, 6), auroc=round(auc, 6), seconds=round(time.time() - t1, 1),
                           posthoc=bool(meta["posthoc"] or model in ("lrx", "lra", "ebm512", "ebmpos")), versions=ver,
                           written=time.strftime("%Y-%m-%d %H:%M:%S"),
                           **{kk: vv for kk, vv in meta.items() if kk != "posthoc"}, info=info)
                keep_imp = imp if (arm in ("bench", "bench_full") and feat == "kmer") else None
                save_unit(unit_path(task, arm, feat, model), out, proba, classes, yte, te_idx, keep_imp)
                extra = ""
                if model in ("lr", "lrx", "lra"):
                    extra = f" [{info['penalty']} C={info['C']} nnz={info['n_nonzero_coef']}]"
                print(f"  {task:<32}{arm:<11}{feat:<5}{model:<5} n_tr={out['n_train']:<7} "
                      f"MCC={out['mcc']:.4f} AUROC={out['auroc']:.4f} ({out['seconds']:.0f}s){extra}", flush=True)
            del Xtr, Xte
        except Exception:  # noqa: BLE001
            with open(os.path.join(UNIT_DIR, "errors.log"), "a") as fh:
                fh.write(f"{time.ctime()} {task} {arm} {feat} {todo}\n{traceback.format_exc()}\n")
            print(f"  ERROR {task} {arm} {feat} {todo} (see errors.log)", flush=True)
    print("[alt_model] done", flush=True)


if __name__ == "__main__":
    main()
