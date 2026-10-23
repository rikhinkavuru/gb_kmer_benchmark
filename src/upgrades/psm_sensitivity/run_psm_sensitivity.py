#!/usr/bin/env python3
"""PSM SENSITIVITY -- is the composition-equalized collapse caused by shortcut removal, or by
negative-set reduction / test-set distortion?  (Reviewer 1, comment 4; BIBM 2026 camera-ready)

PRE-REGISTERED ANALYSIS PLAN (written before any outcome metric was computed; the only things
inspected beforehand were propensity/retention counts per family, which involve no labels-vs-
prediction outcome, and a reproduction of the paper's original-split numbers)
=============================================================================================
Datasets: human_enhancers_cohn, nt_enhancers (contaminated), drosophila_enhancers_stark (clean
control). Data read from a frozen, count-verified snapshot (psm_data.py) in the exact
data.load_dataset order. k = validation-selected best k (results/val_selection.csv new_k:
5 / 4 / 4). Classifiers: the paper's k-mer LightGBM (models.LGBM_PARAMS, n_jobs=2 -- verified to
reproduce the paper's original-split AUROC/MCC to 6 decimals) and the composition-only
LightGBM on the 20-d signature (4 mono + 16 dinucleotide frequencies). Propensity model:
IDENTICAL to the paper (run_composition_clean.fit_propensity: standardized LR on the pooled
train+test 20-d signature, seed 42); e = P(pos | composition), logit l = log(e/(1-e)).
All selection is done WITHIN each split (train, test) separately, as in the paper.

(1) Matching-strictness sweep, primary family F1 "tempered histogram matching": 20 equal-width
    propensity bins; kept_b(alpha) = min(N_b, round(N_b*(f*T_b/N_b)^alpha)), f = min_b N_b/T_b.
    alpha=0 keeps all negatives (= original); alpha=1 IS the paper's proportional match (same
    per-bin counts; verified in tests). Grid alpha in {0.05,.1,.2,...,.9,1}. Negatives in bins
    with no positives (outside common support) are dropped for alpha>0. Within-bin choice uses a
    fixed random priority per sequence (nested subsets across alpha); 3 selection seeds
    {42 (primary), 43, 44}. Secondary family F2 "common-support trimming": keep negatives whose
    logit lies inside the positives' [q, 1-q] quantile band, q in {.05,.1,.2,.3,.4,.45}.
    At each point: retained n per split, balance (max & mean |SMD| over the 20 features on the
    pooled kept set, logit-propensity SMD, GC AUROC, TATA motif-only AUROC), and k-mer and
    composition-only metrics with 95% bootstrap CIs.
(2) Size-matched random controls: at every point of every family, negatives (F4: positives;
    F3: both classes) are subsampled uniformly at random to the SAME per-split counts, 10 seeds
    (0..9; nested across levels), identical metrics.
(3) Metrics everywhere: AUROC; MCC at the default 0.5 threshold (the paper's metric); balanced
    accuracy; class-balance-weighted MCC at the default threshold (mcc_bw) and at the
    prior-corrected threshold t = training positive fraction (mcc_bw_pc; uses training info
    only); PR-AUC of the positive AND the negative class with their prevalences; MCC mean/sd
    over 20 explicit class-balanced test draws (majority subsampled to minority size).
    CIs: 1000-resample percentile bootstrap, test sequence = resampling unit, rng
    RandomState(42).randint(0,n,n) per resample (the paper's protocol), weights carried.
(4) Removal-free / symmetric alternatives:
    (a) propensity WEIGHTING of the ORIGINAL train and test sets, nothing discarded: overlap
        weights (pos 1-e, neg e; primary) and ATT odds weights (pos 1, neg e/(1-e)); LightGBM
        trained with sample_weight (normalized to mean 1), weighted test metrics. Placebo
        control: the same weights randomly permuted within class (identical weight
        distribution/ESS, uncorrelated with composition), 10 seeds. ATT is flagged unstable
        when the negative-class ESS < 5% of the negatives.
    (b) two-sided greedy 1:1 nearest-neighbour matching on the logit propensity, without
        replacement, positives in random order (seed 42), caliper c*SD(logit of the split),
        c in {0.2 (Austin 2011; primary), 0.01}; random control = the same number of random
        positives and random negatives per split.
    (c) positive-side matching: all negatives kept, positives subsampled to the negatives'
        propensity histogram (F1 alpha=1 with roles swapped); random control = same number of
        random positives.
(5) Fixed-model analysis: for every arm, evaluate (within) the arm-trained model on the arm's
    test subset [= what the paper did]; (fixed_model) the model trained ONCE on the original
    train split on the arm's test subset [test-distribution change only]; (train_only) the
    arm-trained model on the ORIGINAL test set [training change only]; and for F1/paper
    anchor the 2x2 cross (matched-trained on random test, random-trained on matched test).
(6) Selection-bias characterization on features OUTSIDE the propensity model: length, N
    fraction, trinucleotide L1 distance to the positive-class centroid, trinucleotide Markov-1
    residual (|f(xyz) - f(xy)f(yz)/f(y)|, higher-order structure not implied by dinucleotides),
    TATA (TBP) and GC-box (SP1) hit rate and count at 1.3 bits/pos, DUST-like low-complexity
    fraction; for cohn also GENCODE v44 TSS +-500 bp overlap, distance to nearest TSS and
    RepeatMasker coverage (GRCh38 coordinates via localization/cohn). Groups: positives,
    retained vs discarded negatives (F1 alpha=1 seed 42, the paper anchor, and F3 c=0.2), and
    random size-matched negatives. SMDs + Mann-Whitney / Fisher tests; selection drift of each
    feature along the F1 sweep.

DECISION RULE (per contaminated dataset; primary family F1 at alpha=1 = the paper's retention;
primary metric = k-mer LightGBM AUROC, within-arm design):
  A0 = original AUROC; Am = matched (seed 42); Ar(s) = random size-matched, s=0..9, mean Ar.
  tau = max(0.03, half-width of the original AUROC 95% CI).
  (i-a) SIZE:        A0 - mean Ar <= tau  (random size-matched subsets keep AUROC close to A0)
  (i-b) SEPARATION:  Am < min_s Ar(s)  AND  upper 95% CI of Am < mean Ar
  (ii-a) WEIGHTING:  overlap-weighted AUROC Aw vs its permuted-weight placebo mean Apw:
                     Aw < min placebo AND upper CI of Aw < mean Apw AND
                     (mean Apw - Aw) >= 0.5 * (mean Ar - Am)
  (ii-b) TWO-SIDED:  1:1 caliper-0.2 AUROC A2 vs its random 1:1 control mean A2r: same three
                     conditions with (mean A2r - A2) >= 0.5 * (mean Ar - Am)
  Verdict: SUPPORTED if (i-a),(i-b),(ii-a),(ii-b) all hold; SUPPORTED-WITH-SIZE-COMPONENT if
  all but (i-a) hold (report shortcut share S = (mean Ar - Am)/(A0 - Am)); NOT SUPPORTED if
  (i-b) fails or both (ii-a) and (ii-b) fail; otherwise MIXED (state which criterion failed).
  The same decomposition is REPORTED (not decision-driving) for the paper's default-threshold
  MCC (size/imbalance part = MCC0 - mean MCCr; shortcut part = mean MCCr - MCCm) and for
  mcc_bw_pc. The control is reported with the same quantities; the expectation (not a
  criterion) is a smaller shortcut drop. Rows with < 20 test negatives are flagged
  `underpowered` and never enter a decision. Nulls/adverse results are reported as measured.

REVISION 2 (2026-09-23 22:10-22:20, after the 22:05 interruption; STILL BEFORE ANY OUTCOME METRIC WAS
COMPUTED -- no model had been fit on any arm; inspected so far: per-level retention/balance counts
(labels + composition only) and a reproduction of the paper's original-split cohn AUROC 0.804649 /
MCC 0.461223, which matched to 6 decimals). Changes, none of which alter the decision rule above:
  * data now come from data.load_dataset (lock-protected, count-checked) and the frozen snapshot is
    used only for file ids after an element-wise identity check (psm_sensitivity_data_check.csv).
  * grids widened so every family spans ~100% -> its floor: F1 alpha adds 0.02 (cohn 93% retained);
    F2 adds q=0.02; F3 calipers {2.0, 1.0, 0.5, 0.2 (primary), 0.01} (caliper-sensitivity curve for
    the symmetric design).
  * weighted arms: test weights normalized to mean 1 per split (metrics are scale-invariant; only
    the pooled train+test balance SMD was affected).
  * (6) selection bias: genomic annotation now also includes ENCODE SCREEN cCRE v4 overlap (any /
    ELS = pELS|dELS / PLS), for cohn (all sequences) and nt_enhancers (genomic subset; ~80% of its
    training set is synthetic RankGAN sequence with no genomic origin -> "genomic_mapped" is itself
    reported as a selection feature). Annotation tables = neg_provenance caches (read-only).
    Each selection is compared with ITS OWN size-matched random negatives.
  * LABEL-NOISE alternative (the main selection-bias threat: composition matching may preferentially
    keep negatives that are unlabelled regulatory elements, making the matched task hard by label
    noise rather than by shortcut removal). Three pre-specified, reported (not decision-driving)
    checks: (a) F5 "cCRE-purged re-match" (cohn): drop every negative overlapping ANY cCRE from both
    splits' pools, then F1 alpha=1 (seed 42) vs 10 random size-matched subsets of the same purged pool;
    (b) cCRE-free test re-scoring: same trained models (original, paper split, F1 a=1, F3 c=0.2, and
    their random controls) scored on test sets whose cCRE-overlapping negatives are removed (cohn and
    nt test); (c) worst-case bound: hidden-positive fraction rho = max(0,(r_ret-r_neg)/(r_pos-r_neg))
    with r = ELS-overlap rate of retained negatives / all negatives / positives; if hidden positives
    scored exactly like positives, label noise could lower AUROC by at most rho*(Ar-0.5); compared with
    the measured shortcut drop (Ar - Am).

POST-SMOKE NOTE (22:25-22:40). A pipeline smoke test (tiny grid: alpha=1 only, 2 random seeds,
B=50; scratch directory, never reported) was run to validate the code end to end, so smoke-level
outcomes were seen after this point. The decision rule, grids, seeds and metrics were NOT changed
afterwards. Changes after the smoke test: (a) bug fixes -- train_only rows are now filed under the
TRAINING arm's family (they were filed under "original" and silently dropped); summarize keys the
balance table on (dataset, arm) (arm names repeat across datasets); the 2x2 cross table is keyed on
the matched arm, so the paper-split model and the F1 seed-42 model are never pooled; (b) F5b added
(two-sided 1:1 caliper 0.2 on the cCRE-purged pool) because F5 retains only 128 negatives -- a
label-free, pre-outcome-knowable property; (c) one DESCRIPTIVE, post hoc quantity is reported in
addition (never decision-driving): the fraction of above-chance AUROC lost, (Ar-Am)/(Ar-0.5).

POST-RUN NOTE (2026-09-24 05:39): the full run completed with this file unchanged since the post-smoke
note. Descriptive, post hoc summaries added afterwards live only in summarize_psm.py (see its docstring).

Outputs (results/upgrades/): psm_sensitivity_sweep.csv (every evaluation row),
psm_sensitivity_balance.csv (every arm), psm_sensitivity_selection_bias.csv,
psm_sensitivity_selection_drift.csv; summarize_psm.py then writes _summary.csv, _decision.csv
and _interpretation.txt. Predictions are checkpointed per arm under
external/psm_sensitivity/preds/<task>/ so the run resumes after interruption.
CPU only, <=2 threads, no torch.
"""
import argparse
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "composition_clean"))
import psm_core as P  # noqa: E402
import psm_data as D  # noqa: E402
import _common as C  # noqa: E402
import featurize  # noqa: E402
import run_composition_clean as RCC  # noqa: E402  (fit_propensity: the paper's propensity model)

DATASETS = [("human_enhancers_cohn", "contaminated"),
            ("nt_enhancers", "contaminated"),
            ("drosophila_enhancers_stark", "clean control")]
ALPHAS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MATCH_SEEDS = [42, 43, 44]
RAND_SEEDS = list(range(10))
TRIM_QS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45]
CALIPERS = [2.0, 1.0, 0.5, 0.2, 0.01]
GENOMIC_TASKS = ("human_enhancers_cohn", "nt_enhancers")      # hg38 coordinates available
NP_CACHE = os.path.join(C.ROOT, "external", "neg_provenance", "cache")   # read-only reuse
NT_MAP = os.path.join(C.ROOT, "results", "upgrades", "neg_provenance_nt_map.csv")
N_BINS = 20
TATA_BITS = 1.3
SP1_BITS = 1.3
N_THREADS = 2
SNAP = D.SNAP_DIR
OUT = C.RESULTS_DIR
VAL_SEL = os.path.join(C.ROOT, "results", "val_selection.csv")
PAPER_SPLITS = os.path.join(C.ROOT, "cleaned_splits_v2")


def best_k(task):
    df = pd.read_csv(VAL_SEL)
    return int(df.loc[df["dataset"] == task, "new_k"].iloc[0])


def lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=C.SEED, **{**C.gbmodels.LGBM_PARAMS, "n_jobs": N_THREADS})


def pri(seed, split_id, n):
    """Fixed random priority per sequence (nested subsets)."""
    return np.random.RandomState([int(seed), int(split_id)]).uniform(size=n)


# ================================================================ context
class Ctx:
    def __init__(self, task, role):
        self.task, self.role = task, role
        self.k = best_k(task)
        (self.tr, self.ytr, self.te, self.yte, self.tr_ids, self.te_ids) = D.load_task(task)
        t0 = time.time()
        self.Xtr = featurize.kmer_spectrum(self.tr, self.k)
        self.Xte = featurize.kmer_spectrum(self.te, self.k)
        self.Ctr = C.comp_signature(self.tr, ks=(1, 2)).astype(np.float64)
        self.Cte = C.comp_signature(self.te, ks=(1, 2)).astype(np.float64)
        prop = RCC.fit_propensity(np.vstack([self.Ctr, self.Cte]), np.r_[self.ytr, self.yte], C.SEED)
        self.etr = prop.predict_proba(self.Ctr)[:, 1]
        self.ete = prop.predict_proba(self.Cte)[:, 1]
        self.ltr = np.log(self.etr / (1 - self.etr))
        self.lte = np.log(self.ete / (1 - self.ete))
        tbp, sp1 = C.load_tbp(), C.load_sp1()
        self.tata_tr = C.tata_hit_counts(self.tr, tbp, TATA_BITS)
        self.tata_te = C.tata_hit_counts(self.te, tbp, TATA_BITS)
        self.sp1_tr = C.tata_hit_counts(self.tr, sp1, SP1_BITS)
        self.sp1_te = C.tata_hit_counts(self.te, sp1, SP1_BITS)
        self.gc_tr, self.gc_te = C.gc_content(self.tr), C.gc_content(self.te)
        self.gann = genomic_annotation(self) if task in GENOMIC_TASKS else None
        if self.gann is not None:
            ntr = len(self.ytr)
            cc = self.gann["ccre_any"].to_numpy(float)          # NaN = unmapped
            self.ccre_tr, self.ccre_te = cc[:ntr], cc[ntr:]
        print(f"[{task}] k={self.k} n_train={len(self.ytr)} n_test={len(self.yte)} "
              f"features+propensity+motifs in {time.time() - t0:.0f}s", flush=True)

    def split(self, s):
        return ((self.ytr, self.etr, self.ltr) if s == 0 else (self.yte, self.ete, self.lte))


class Arm:
    def __init__(self, name, family, level, kind, seed, tr_idx, te_idx, w_tr=None, w_te=None,
                 level_label=""):
        self.name, self.family, self.level, self.kind, self.seed = name, family, level, kind, seed
        self.tr_idx, self.te_idx = np.asarray(tr_idx, int), np.asarray(te_idx, int)
        self.w_tr, self.w_te = w_tr, w_te
        self.level_label = level_label or str(level)


# ================================================================ arm construction
def _neg_side(ctx, fam, level, kind, seed, keep_fn, label=""):
    """Arm keeping all positives + selected negatives in each split. keep_fn(split_id) returns
    kept indices INTO that split's negative array."""
    idx = []
    for s in (0, 1):
        y = ctx.split(s)[0]
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        idx.append(np.sort(np.r_[pos, neg[keep_fn(s)]]))
    return Arm(f"{fam}_{label or level}_{kind}{seed if seed is not None else ''}", fam, level, kind,
               seed, idx[0], idx[1], level_label=label)


def build_arms(ctx):
    arms = []
    all_tr, all_te = np.arange(len(ctx.ytr)), np.arange(len(ctx.yte))
    arms.append(Arm("original", "original", 0.0, "orig", None, all_tr, all_te, level_label="0"))

    def masks(s):
        y, e, l = ctx.split(s)
        return y, e, l, np.where(y == 1)[0], np.where(y == 0)[0]

    # ---- F1 tempered histogram matching + random controls
    for a in ALPHAS:
        counts = {}
        for ms in MATCH_SEEDS:
            def kf(s, a=a, ms=ms):
                y, e, l, pos, neg = masks(s)
                k = P.hist_temper(e[pos], e[neg], N_BINS, a, pri(ms, s, len(neg)))
                counts[s] = len(k)
                return k
            arms.append(_neg_side(ctx, "F1_hist", a, "matched", ms, kf, label=f"a{a:g}"))
        for r in RAND_SEEDS:
            def kf(s, r=r):
                y, e, l, pos, neg = masks(s)
                return P.random_take(len(neg), counts[s], pri(10000 + r, s, len(neg)))
            arms.append(_neg_side(ctx, "F1_hist", a, "random", r, kf, label=f"a{a:g}"))

    # ---- paper anchor: the exact published comp_equalized split (cleaned_splits_v2)
    arms.append(paper_anchor(ctx))

    # ---- F2 common-support trimming + random controls
    for q in TRIM_QS:
        counts = {}

        def kf(s, q=q):
            y, e, l, pos, neg = masks(s)
            k = P.trim_band(l[pos], l[neg], q)
            counts[s] = len(k)
            return k
        arms.append(_neg_side(ctx, "F2_trim", q, "matched", 42, kf, label=f"q{q:g}"))
        for r in RAND_SEEDS:
            def kf(s, r=r):
                y, e, l, pos, neg = masks(s)
                return P.random_take(len(neg), counts[s], pri(10000 + r, s, len(neg)))
            arms.append(_neg_side(ctx, "F2_trim", q, "random", r, kf, label=f"q{q:g}"))

    # ---- F3 two-sided 1:1 caliper matching + random 1:1 controls
    for c in CALIPERS:
        sel, npairs = [], {}
        for s in (0, 1):
            y, e, l, pos, neg = masks(s)
            pi, ni = P.caliper_1to1(l[pos], l[neg], c * np.std(l), np.random.RandomState(42))
            npairs[s] = len(pi)
            sel.append(np.sort(np.r_[pos[pi], neg[ni]]))
        arms.append(Arm(f"F3_1to1_c{c:g}_matched42", "F3_1to1", c, "matched", 42, sel[0], sel[1],
                        level_label=f"c{c:g}"))
        for r in RAND_SEEDS:
            sel = []
            for s in (0, 1):
                y, e, l, pos, neg = masks(s)
                kp = P.random_take(len(pos), npairs[s], pri(20000 + r, s, len(pos)))
                kn = P.random_take(len(neg), npairs[s], pri(10000 + r, s, len(neg)))
                sel.append(np.sort(np.r_[pos[kp], neg[kn]]))
            arms.append(Arm(f"F3_1to1_c{c:g}_random{r}", "F3_1to1", c, "random", r, sel[0], sel[1],
                            level_label=f"c{c:g}"))

    # ---- F4 positive-side matching + random positive subsamples
    sel, npos = [], {}
    for s in (0, 1):
        y, e, l, pos, neg = masks(s)
        kp = P.hist_temper(e[neg], e[pos], N_BINS, 1.0, pri(42, s, len(pos)))
        npos[s] = len(kp)
        sel.append(np.sort(np.r_[pos[kp], neg]))
    arms.append(Arm("F4_posside_matched42", "F4_posside", 1.0, "matched", 42, sel[0], sel[1],
                    level_label="a1"))
    for r in RAND_SEEDS:
        sel = []
        for s in (0, 1):
            y, e, l, pos, neg = masks(s)
            kp = P.random_take(len(pos), npos[s], pri(20000 + r, s, len(pos)))
            sel.append(np.sort(np.r_[pos[kp], neg]))
        arms.append(Arm(f"F4_posside_random{r}", "F4_posside", 1.0, "random", r, sel[0], sel[1],
                        level_label="a1"))

    # ---- F5 label-noise-purged re-match (only where EVERY negative has hg38 coordinates, i.e. cohn):
    # negatives overlapping ANY ENCODE cCRE are removed from both splits' negative pools first, then
    # F1 alpha=1 matching (seed 42) vs uniform-random negatives from the same purged pool (same n).
    if ctx.gann is not None and np.all(np.isfinite(np.r_[ctx.ccre_tr, ctx.ccre_te][np.r_[ctx.ytr, ctx.yte] == 0])):
        counts, pools = {}, {}
        for s in (0, 1):
            y, e, l, pos, neg = masks(s)
            cc = ctx.ccre_tr if s == 0 else ctx.ccre_te
            pools[s] = np.flatnonzero(cc[neg] == 0)          # indices into this split's negatives

        def kf(s):
            y, e, l, pos, neg = masks(s)
            pool = pools[s]
            k = P.hist_temper(e[pos], e[neg[pool]], N_BINS, 1.0, pri(42, s, len(pool)))
            counts[s] = len(k)
            return pool[k]
        arms.append(_neg_side(ctx, "F5_ccrefree", 1.0, "matched", 42, kf, label="a1"))
        for r in RAND_SEEDS:
            def kf(s, r=r):
                pool = pools[s]
                return pool[P.random_take(len(pool), counts[s], pri(10000 + r, s, len(pool)))]
            arms.append(_neg_side(ctx, "F5_ccrefree", 1.0, "random", r, kf, label="a1"))
        # F5b (added after the pipeline smoke test; motivated ONLY by F5's retained count -- 128
        # negatives in total, a label-free property): two-sided 1:1 caliper-0.2 matching on the same
        # cCRE-purged negative pool; control = the same number of random positives + random purged
        # negatives per split.
        sel, npairs = [], {}
        for s in (0, 1):
            y, e, l, pos, neg = masks(s)
            pool = neg[pools[s]]
            pi, ni = P.caliper_1to1(l[pos], l[pool], 0.2 * np.std(l), np.random.RandomState(42))
            npairs[s] = len(pi)
            sel.append(np.sort(np.r_[pos[pi], pool[ni]]))
        arms.append(Arm("F5b_ccrefree_1to1_c0.2_matched42", "F5b_ccrefree_1to1", 0.2, "matched", 42,
                        sel[0], sel[1], level_label="c0.2"))
        for r in RAND_SEEDS:
            sel = []
            for s in (0, 1):
                y, e, l, pos, neg = masks(s)
                pool = neg[pools[s]]
                kp = P.random_take(len(pos), npairs[s], pri(20000 + r, s, len(pos)))
                kn = P.random_take(len(pool), npairs[s], pri(10000 + r, s, len(pool)))
                sel.append(np.sort(np.r_[pos[kp], pool[kn]]))
            arms.append(Arm(f"F5b_ccrefree_1to1_c0.2_random{r}", "F5b_ccrefree_1to1", 0.2, "random", r,
                            sel[0], sel[1], level_label="c0.2"))

    # ---- W weighting of the ORIGINAL splits (nothing discarded) + permuted-weight placebo
    for wname, wfn in (("W_overlap", P.overlap_weights), ("W_att", P.att_weights)):
        wtr, wte = wfn(ctx.etr, ctx.ytr), wfn(ctx.ete, ctx.yte)
        # each split's weights normalized to mean 1 (all metrics are scale-invariant; this only
        # keeps train and test on the same scale when balance is pooled over both splits)
        wtr, wte = wtr / wtr.mean(), wte / wte.mean()
        arms.append(Arm(f"{wname}_matched", wname, 1.0, "matched", 42, all_tr, all_te,
                        w_tr=wtr, w_te=wte, level_label="w"))
        for r in RAND_SEEDS:
            pw = []
            for s, (w, y) in enumerate(((wtr, ctx.ytr), (wte, ctx.yte))):
                w2 = w.copy()
                for cl in (0, 1):
                    ii = np.where(y == cl)[0]
                    w2[ii] = w[ii][np.random.RandomState([30000 + r, s, cl]).permutation(len(ii))]
                pw.append(w2)
            arms.append(Arm(f"{wname}_random{r}", wname, 1.0, "random", r, all_tr, all_te,
                            w_tr=pw[0], w_te=pw[1], level_label="w"))
    return arms


def paper_anchor(ctx):
    """Map the published cleaned_splits_v2/<task>_comp_equalized_{train,test}.csv onto indices of
    the original split (multiset match on sequence; positives must all be present)."""
    idx = []
    for s, (seqs, y, tag) in enumerate(((ctx.tr, ctx.ytr, "train"), (ctx.te, ctx.yte, "test"))):
        df = pd.read_csv(os.path.join(PAPER_SPLITS, f"{ctx.task}_comp_equalized_{tag}.csv"))
        pool = {}
        for i, (sq, lab) in enumerate(zip(seqs, y)):
            pool.setdefault((sq, int(lab)), []).append(i)
        got = []
        for sq, lab in zip(df["sequence"].astype(str).str.upper(), df["label"].astype(int)):
            got.append(pool[(sq, lab)].pop())
        got = np.sort(np.array(got))
        assert (y[got] == 1).sum() == (y == 1).sum(), "paper anchor must keep every positive"
        idx.append(got)
    return Arm("paper_equalized_matched", "paper_anchor", 1.0, "matched", None, idx[0], idx[1],
               level_label="paper")


# ================================================================ fitting (checkpointed)
def fit_arm(ctx, arm, cache_dir):
    path = os.path.join(cache_dir, f"{arm.name}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return dict(kmer=d["kmer"], comp=d["comp"], pi=float(d["pi"]))
    out = {}
    y = ctx.ytr[arm.tr_idx]
    w = None if arm.w_tr is None else arm.w_tr[arm.tr_idx]
    for name, Xtr, Xte in (("kmer", ctx.Xtr, ctx.Xte), ("comp", ctx.Ctr, ctx.Cte)):
        m = lgbm()
        m.fit(Xtr[arm.tr_idx], y, sample_weight=w)
        col = int(np.where(m.classes_ == 1)[0][0])
        out[name] = m.predict_proba(Xte)[:, col].astype(np.float64)
    ww = np.ones(len(y)) if w is None else w
    out["pi"] = float(ww[y == 1].sum() / ww.sum())
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, kmer=out["kmer"], comp=out["comp"], pi=out["pi"])
    os.replace(tmp, path)
    return out


# ================================================================ balance
def balance_row(ctx, arm):
    wtr = np.ones(len(ctx.ytr)) if arm.w_tr is None else arm.w_tr
    wte = np.ones(len(ctx.yte)) if arm.w_te is None else arm.w_te
    y = np.r_[ctx.ytr[arm.tr_idx], ctx.yte[arm.te_idx]]
    w = np.r_[wtr[arm.tr_idx], wte[arm.te_idx]]
    comp = np.vstack([ctx.Ctr[arm.tr_idx], ctx.Cte[arm.te_idx]])
    lg = np.r_[ctx.ltr[arm.tr_idx], ctx.lte[arm.te_idx]]
    gc = np.r_[ctx.gc_tr[arm.tr_idx], ctx.gc_te[arm.te_idx]]
    ta = np.r_[ctx.tata_tr[arm.tr_idx], ctx.tata_te[arm.te_idx]].astype(float)
    ps, ng = y == 1, y == 0
    s = np.abs(P.smd(comp[ps], comp[ng], w[ps], w[ng]))
    sl = P.smd(lg[ps, None], lg[ng, None], w[ps], w[ng])[0]
    n_neg_all = int((ctx.ytr == 0).sum() + (ctx.yte == 0).sum())
    n_pos_all = int((ctx.ytr == 1).sum() + (ctx.yte == 1).sum())
    ntrn = int((ctx.ytr[arm.tr_idx] == 0).sum()); nten = int((ctx.yte[arm.te_idx] == 0).sum())
    ntrp = int((ctx.ytr[arm.tr_idx] == 1).sum()); ntep = int((ctx.yte[arm.te_idx] == 1).sum())
    ess_neg = P.ess(w[ng]); ess_pos = P.ess(w[ps])
    return dict(dataset=ctx.task, role=ctx.role, arm=arm.name, family=arm.family, level=arm.level,
                level_label=arm.level_label, kind=arm.kind, seed=arm.seed,
                n_train_pos=ntrp, n_train_neg=ntrn, n_test_pos=ntep, n_test_neg=nten,
                retained_neg_frac=(ntrn + nten) / n_neg_all, retained_pos_frac=(ntrp + ntep) / n_pos_all,
                ess_neg_frac=ess_neg / n_neg_all, ess_pos_frac=ess_pos / n_pos_all,
                max_abs_smd=float(np.nanmax(s)), mean_abs_smd=float(np.nanmean(s)),
                logit_smd=float(sl),
                gc_auroc=float(P.auc_rows(w[None, :], y, gc)[0]),
                tata_auroc=float(P.auc_rows(w[None, :], y, ta)[0]))


# ================================================================ evaluation
def eval_row(ctx, arm_model, preds_model, arm_test, design, model_kind, B):
    te = arm_test.te_idx
    w = None if arm_test.w_te is None else arm_test.w_te[te]
    yv = ctx.yte[te]
    pv = preds_model[model_kind][te]
    r = P.evaluate(yv, pv, w=w, thr_pc=preds_model["pi"], B=B, seed=C.SEED)
    # the row is filed under the arm that VARIES in this design: the training arm for train_only
    # (its test set is always the original one), the test arm otherwise
    lab = arm_model if design == "train_only" else arm_test
    r.update(dataset=ctx.task, role=ctx.role, family=lab.family, level=lab.level,
             level_label=lab.level_label, design=design, model=model_kind,
             train_arm=arm_model.name, test_arm=arm_test.name,
             train_kind=arm_model.kind, test_kind=arm_test.kind,
             train_seed=arm_model.seed, test_seed=arm_test.seed,
             n_train=int(len(arm_model.tr_idx)),
             n_train_pos=int((ctx.ytr[arm_model.tr_idx] == 1).sum()),
             n_train_neg=int((ctx.ytr[arm_model.tr_idx] == 0).sum()),
             pi_train=preds_model["pi"], weighted_test=w is not None,
             underpowered=bool(r["n_neg"] < 20 or r["n_pos"] < 20))
    return r


def tier(arm):
    """Fit priority: 0 = everything the pre-registered decision rule needs; 1 = rest of the F1
    sweep (seed 42 + random); 2 = secondary families; 3 = extra F1 selection seeds."""
    if arm.family == "original" or arm.family == "paper_anchor" or arm.family == "W_overlap":
        return 0
    if arm.family == "F1_hist":
        if arm.kind == "matched" and arm.seed != 42:
            return 3
        return 0 if arm.level == 1.0 else 1
    if arm.family == "F3_1to1" and arm.level == 0.2:
        return 0
    if arm.family in ("F5_ccrefree", "F5b_ccrefree_1to1"):
        return 1
    return 2


def fit_tiers(tasks, ctxs, armsets):
    t0 = time.time()
    for t in range(4):
        for task, role in tasks:
            ctx, arms = ctxs[task], armsets[task]
            cache = os.path.join(SNAP, "preds", task)
            os.makedirs(cache, exist_ok=True)
            todo = [a for a in arms if tier(a) == t]
            for i, a in enumerate(todo):
                ta = time.time()
                fit_arm(ctx, a, cache)
                if time.time() - ta > 1:
                    print(f"  [tier {t}] {task} fit {i + 1}/{len(todo)} {a.name} ({time.time() - ta:.0f}s; "
                          f"elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
        print(f"== tier {t} complete for all datasets ({(time.time() - t0) / 60:.1f} min)", flush=True)


def run_dataset(ctx, arms, B):
    task = ctx.task
    cache = os.path.join(SNAP, "preds", task)
    os.makedirs(cache, exist_ok=True)
    by_name = {a.name: a for a in arms}
    print(f"[{task}] evaluating {len(arms)} arms", flush=True)
    preds = {}
    t0 = time.time()
    for a in arms:
        preds[a.name] = fit_arm(ctx, a, cache)          # loads from cache (fits only if missing)

    # --- balance ---
    bal = pd.DataFrame([balance_row(ctx, a) for a in arms])

    # --- evaluations ---
    rows = []
    orig = by_name["original"]
    rows.append(eval_row(ctx, orig, preds["original"], orig, "within", "kmer", B))
    rows.append(eval_row(ctx, orig, preds["original"], orig, "within", "comp", B))
    for a in arms:
        if a.name == "original":
            continue
        for mk in ("kmer", "comp"):
            rows.append(eval_row(ctx, a, preds[a.name], a, "within", mk, B))
            rows.append(eval_row(ctx, orig, preds["original"], a, "fixed_model", mk, B))
            rows.append(eval_row(ctx, a, preds[a.name], orig, "train_only", mk, B))
    # 2x2 cross at every F1 level (matched seed 42 vs each random seed) and for the paper anchor
    for a in ALPHAS:
        m = by_name[f"F1_hist_a{a:g}_matched42"]
        for r in RAND_SEEDS:
            rr = by_name[f"F1_hist_a{a:g}_random{r}"]
            for mk in ("kmer", "comp"):
                rows.append(eval_row(ctx, m, preds[m.name], rr, "cross_matchedModel_randomTest", mk, B))
                rows.append(eval_row(ctx, rr, preds[rr.name], m, "cross_randomModel_matchedTest", mk, B))
    pa = by_name["paper_equalized_matched"]
    for r in RAND_SEEDS:
        rr = by_name[f"F1_hist_a1_random{r}"]
        for mk in ("kmer", "comp"):
            rows.append(eval_row(ctx, pa, preds[pa.name], rr, "cross_matchedModel_randomTest", mk, B))
            rows.append(eval_row(ctx, rr, preds[rr.name], pa, "cross_randomModel_matchedTest", mk, B))
    # label-noise exclusion: same models, test negatives overlapping ANY ENCODE cCRE removed
    # (positives all kept). Needs hg38 coordinates for every test sequence (cohn; nt test).
    if ctx.gann is not None and np.all(np.isfinite(ctx.ccre_te)):
        keep_te = (ctx.yte == 1) | (ctx.ccre_te == 0)
        sel = ["original", "paper_equalized_matched", "F1_hist_a1_matched42", "F3_1to1_c0.2_matched42"] + \
              [f"F1_hist_a1_random{r}" for r in RAND_SEEDS] + [f"F3_1to1_c0.2_random{r}" for r in RAND_SEEDS]
        for nm in sel:
            a = by_name[nm]
            af = Arm(a.name + "|ccrefreeTest", a.family, a.level, a.kind, a.seed, a.tr_idx,
                     a.te_idx[keep_te[a.te_idx]], level_label=a.level_label)
            for mk in ("kmer", "comp"):
                rows.append(eval_row(ctx, a, preds[a.name], af, "within_ccrefreeTest", mk, B))
                if nm != "original":
                    rows.append(eval_row(ctx, orig, preds["original"], af, "fixed_model_ccrefreeTest", mk, B))
    ev = pd.DataFrame(rows)
    ev = ev.merge(bal[["arm", "retained_neg_frac", "retained_pos_frac", "ess_neg_frac", "max_abs_smd",
                       "mean_abs_smd", "gc_auroc", "tata_auroc"]].rename(columns={"arm": "test_arm"}),
                  on="test_arm", how="left")
    print(f"[{task}] evaluations done ({len(ev)} rows) in {time.time() - t0:.0f}s total", flush=True)
    return ctx, arms, preds, bal, ev


# ================================================================ selection bias (6)
def trimer_features(seqs, pos_centroid=None):
    f3 = featurize.kmer_spectrum(seqs, 3).toarray().astype(np.float64)
    f2 = featurize.kmer_spectrum(seqs, 2).toarray().astype(np.float64)
    f1 = featurize.kmer_spectrum(seqs, 1).toarray().astype(np.float64)
    # Markov-1 expectation f(xyz) ~ f(xy) f(yz) / f(y)
    exp = np.zeros_like(f3)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                exp[:, a * 16 + b * 4 + c] = f2[:, a * 4 + b] * f2[:, b * 4 + c] / np.maximum(f1[:, b], 1e-9)
    exp = exp / np.maximum(exp.sum(1, keepdims=True), 1e-12)
    resid = np.abs(f3 - exp).sum(1)
    return f3, resid


def _merge(s, e):
    """Merge half-open intervals -> sorted disjoint (starts, ends, cumulative-length-before)."""
    if len(s) == 0:
        z = np.array([], np.int64)
        return z, z, z
    o = np.argsort(s, kind="stable")
    s, e = np.asarray(s, np.int64)[o], np.asarray(e, np.int64)[o]
    emax = np.maximum.accumulate(e)
    new = np.r_[True, s[1:] > emax[:-1]]
    idx = np.flatnonzero(new)
    ms, me = s[idx], np.r_[emax[idx[1:] - 1], emax[-1]]
    cum = np.r_[0, np.cumsum(me - ms)[:-1]]
    return ms, me, cum


def _covered(merged, qs, qe):
    """Bases of each [qs,qe) covered by merged disjoint intervals (vectorised)."""
    ms, me, cum = merged
    if len(ms) == 0:
        return np.zeros(len(qs))

    def upto(x):
        i = np.searchsorted(ms, x, side="right") - 1
        ic = np.clip(i, 0, None)
        v = cum[ic] + np.clip(np.minimum(me[ic], x) - ms[ic], 0, None)
        return np.where(i >= 0, v, 0)
    return (upto(np.asarray(qe, np.int64)) - upto(np.asarray(qs, np.int64))).astype(float)


def _coords(ctx):
    """hg38 (chrom, start, end) per pooled [train; test] sequence; NaN/None where unavailable
    (nt_enhancers: synthetic RankGAN training sequences have no genomic origin)."""
    n = len(ctx.ytr) + len(ctx.yte)
    ch = np.array([None] * n, dtype=object)
    st = np.full(n, -1, np.int64)
    en = np.full(n, -1, np.int64)
    if ctx.task == "human_enhancers_cohn":
        rows = {}
        for tag in ("train", "test"):
            for cls in ("negative", "positive"):
                df = pd.read_csv(os.path.join(C.ROOT, "localization", "cohn", f"{tag}_{cls}.csv.gz"))
                rows.update({i: (c, int(a), int(b)) for i, c, a, b in zip(df["id"], df["region"], df["start"], df["end"])})
        for j, fid in enumerate(list(ctx.tr_ids) + list(ctx.te_ids)):
            ch[j], st[j], en[j] = rows[fid]
        assert np.all(en - st == 500), "cohn coordinates must be 500 bp"
    elif ctx.task == "nt_enhancers":
        m = pd.read_csv(NT_MAP)
        seqs = list(ctx.tr) + list(ctx.te)
        y = np.r_[ctx.ytr, ctx.yte]
        key = {(s, int(i)): r for s, i, r in zip(m["split"], m["idx"], range(len(m)))}
        for j, fid in enumerate(list(ctx.tr_ids) + list(ctx.te_ids)):
            sp, i = fid.split("_")
            r = m.iloc[key[(sp, int(i))]]
            assert int(r["label"]) == int(y[j]) and int(r["length"]) == len(seqs[j]), f"nt map misaligned at {fid}"
            if bool(r["genomic"]):
                ch[j], st[j], en[j] = r["chr"], int(r["r_st"]), int(r["r_en"])
    return ch, st, en


def genomic_annotation(ctx):
    """Per pooled sequence: mapped flag, GENCODE v44 TSS +-500 bp overlap and log10 distance,
    RepeatMasker (all / simple+low-complexity) coverage, ENCODE SCREEN cCRE v4 overlap (any, ELS =
    pELS|dELS, PLS). Annotation tables are the neg_provenance module's validated caches (read-only).
    Cached to external/psm_sensitivity/genomic_annot_<task>.csv.gz."""
    path = os.path.join(SNAP, f"genomic_annot_{ctx.task}.csv.gz")
    n = len(ctx.ytr) + len(ctx.yte)
    if os.path.exists(path):
        d = pd.read_csv(path)
        if len(d) == n:
            return d
    ch, st, en = _coords(ctx)
    mapped = np.array([c is not None for c in ch])
    out = pd.DataFrame(dict(genomic_mapped=mapped.astype(float)))
    for c in ("tss500_overlap", "log10_dist_tss", "repeatmasker_frac", "rmsk_simple_lowcomp_frac",
              "ccre_any", "ccre_els", "ccre_pls"):
        out[c] = np.nan
    z = np.load(os.path.join(NP_CACHE, "gencode_v44.npz"))
    t_chr, t_pos = z["t_chr"], z["t_pos"]
    r = np.load(os.path.join(NP_CACHE, "rmsk_hg38.npz"))
    r_chr, r_s, r_e, r_code = r["chr"], r["s"], r["e"], r["code"]
    q = np.load(os.path.join(NP_CACHE, "ccre_v4.npz"))
    q_chr, q_s, q_e, q_code = q["chr"], q["s"], q["e"], q["code"]
    assert len(q_s) > 2_000_000 and len(r_s) > 5_000_000 and len(t_pos) > 200_000, "annotation cache incomplete"
    for c in sorted(set(ch[mapped])):
        m = np.flatnonzero(ch == c)
        s, e = st[m], en[m]
        L = (e - s).astype(float)
        tss = np.sort(t_pos[t_chr == c])
        mid = (s + e) // 2
        if len(tss):
            i = np.searchsorted(tss, mid)
            d = np.minimum(np.abs(mid - tss[np.clip(i - 1, 0, len(tss) - 1)]),
                           np.abs(tss[np.clip(i, 0, len(tss) - 1)] - mid))
            j = np.searchsorted(tss, s - 500)
            ov = (j < len(tss)) & (tss[np.clip(j, 0, len(tss) - 1)] < e + 500)
        else:
            d, ov = np.full(len(m), 10 ** 9), np.zeros(len(m), bool)
        out.loc[m, "tss500_overlap"] = ov.astype(float)
        out.loc[m, "log10_dist_tss"] = np.log10(d + 1.0)
        rc = r_chr == c
        out.loc[m, "repeatmasker_frac"] = _covered(_merge(r_s[rc], r_e[rc]), s, e) / L
        rs = rc & np.isin(r_code, [9, 10])                    # Simple_repeat, Low_complexity
        out.loc[m, "rmsk_simple_lowcomp_frac"] = _covered(_merge(r_s[rs], r_e[rs]), s, e) / L
        qc = q_chr == c
        out.loc[m, "ccre_any"] = (_covered(_merge(q_s[qc], q_e[qc]), s, e) > 0).astype(float)
        qe_ = qc & np.isin(q_code, [2, 3])                    # pELS, dELS
        out.loc[m, "ccre_els"] = (_covered(_merge(q_s[qe_], q_e[qe_]), s, e) > 0).astype(float)
        qp = qc & (q_code == 1)                                # PLS
        out.loc[m, "ccre_pls"] = (_covered(_merge(q_s[qp], q_e[qp]), s, e) > 0).astype(float)
    tmp = path[:-7] + ".tmp.csv.gz"
    out.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)
    return out


def selection_bias(ctx, arms, by_name):
    """Out-of-propensity-model features for retained vs discarded negatives (and positives).
    NaN-aware: genomic features exist only for sequences with hg38 coordinates (all of cohn; the
    genomic subset of nt_enhancers); each comparison uses the non-missing values and reports n."""
    seqs = list(ctx.tr) + list(ctx.te)
    y = np.r_[ctx.ytr, ctx.yte]
    ntr = len(ctx.ytr)
    f3, resid = trimer_features(seqs)
    centroid = f3[y == 1].mean(0)
    feats = pd.DataFrame(dict(
        length_acgt=[sum(ch in "ACGT" for ch in s) for s in seqs],
        n_frac=[1 - sum(ch in "ACGT" for ch in s) / max(len(s), 1) for s in seqs],
        trimer_l1_to_pos_centroid=np.abs(f3 - centroid).sum(1),
        trimer_markov1_residual=resid,
        tata_any=(np.r_[ctx.tata_tr, ctx.tata_te] > 0).astype(float),
        tata_count=np.r_[ctx.tata_tr, ctx.tata_te].astype(float),
        sp1_gcbox_any=(np.r_[ctx.sp1_tr, ctx.sp1_te] > 0).astype(float),
        sp1_gcbox_count=np.r_[ctx.sp1_tr, ctx.sp1_te].astype(float),
        lowcomplexity_frac=P.dust_lowcomplexity_frac(seqs)))
    if ctx.gann is not None:
        for c in ctx.gann.columns:
            if c == "genomic_mapped" and ctx.gann[c].min() == 1:
                continue                                   # constant (cohn: all mapped)
            feats[c] = ctx.gann[c].to_numpy(float)
    binary = {"tata_any", "sp1_gcbox_any", "tss500_overlap", "ccre_any", "ccre_els", "ccre_pls",
              "genomic_mapped"}

    def pooled_neg_mask(arm):
        m = np.zeros(len(y), bool)
        m[arm.tr_idx] = True
        m[ntr + arm.te_idx] = True
        return m & (y == 0)

    from scipy.stats import fisher_exact, mannwhitneyu
    rows, drift = [], []
    neg_all = y == 0
    pos_all = y == 1
    # each selection is compared with ITS OWN size-matched random negative subsets
    selections = [("F1_hist_a1_seed42", by_name["F1_hist_a1_matched42"], "F1_hist_a1_random{}"),
                  ("paper_equalized", by_name["paper_equalized_matched"], "F1_hist_a1_random{}"),
                  ("F3_1to1_c0.2", by_name["F3_1to1_c0.2_matched42"], "F3_1to1_c0.2_random{}")]
    if "F5_ccrefree_a1_matched42" in by_name:
        selections.append(("F5_ccrefree_a1_seed42", by_name["F5_ccrefree_a1_matched42"], "F5_ccrefree_a1_random{}"))
        selections.append(("F5b_ccrefree_1to1_c0.2", by_name["F5b_ccrefree_1to1_c0.2_matched42"],
                           "F5b_ccrefree_1to1_c0.2_random{}"))

    def sd2(a, b):
        s = np.sqrt((np.var(a) + np.var(b)) / 2) if len(a) and len(b) else np.nan
        return s if s > 0 else np.nan

    for sel_name, arm, rand_pat in selections:
        rand_masks = [pooled_neg_mask(by_name[rand_pat.format(r)]) for r in RAND_SEEDS]
        kept = pooled_neg_mask(arm)
        disc = neg_all & ~kept
        for f in feats.columns:
            v = feats[f].to_numpy(float)
            ok = np.isfinite(v)
            vk, vd, vp, vn = v[kept & ok], v[disc & ok], v[pos_all & ok], v[neg_all & ok]
            if f in binary:
                a_, b_ = int(vk.sum()), int(len(vk) - vk.sum())
                c_, d_ = int(vd.sum()), int(len(vd) - vd.sum())
                pval = float(fisher_exact([[a_, b_], [c_, d_]])[1]) if len(vk) and len(vd) else np.nan
                test = "fisher"
            else:
                pval = float(mannwhitneyu(vk, vd).pvalue) if len(vk) and len(vd) else np.nan
                test = "mannwhitney"
            rand_means = [np.nanmean(v[m & ok]) if (m & ok).any() else np.nan for m in rand_masks]
            rows.append(dict(dataset=ctx.task, role=ctx.role, selection=sel_name, feature=f,
                             n_retained=int(kept.sum()), n_discarded=int(disc.sum()), n_pos=int(pos_all.sum()),
                             n_retained_nonmissing=len(vk), n_discarded_nonmissing=len(vd),
                             n_pos_nonmissing=len(vp),
                             mean_pos=np.mean(vp) if len(vp) else np.nan,
                             mean_neg_all=np.mean(vn) if len(vn) else np.nan,
                             mean_retained=np.mean(vk) if len(vk) else np.nan,
                             mean_discarded=np.mean(vd) if len(vd) else np.nan,
                             mean_random_sizematched=float(np.nanmean(rand_means)),
                             random_sizematched_min=float(np.nanmin(rand_means)),
                             random_sizematched_max=float(np.nanmax(rand_means)),
                             smd_retained_vs_discarded=(np.mean(vk) - np.mean(vd)) / sd2(vk, vd) if len(vk) and len(vd) else np.nan,
                             smd_retained_vs_pos=(np.mean(vk) - np.mean(vp)) / sd2(vk, vp) if len(vk) and len(vp) else np.nan,
                             smd_discarded_vs_pos=(np.mean(vd) - np.mean(vp)) / sd2(vd, vp) if len(vd) and len(vp) else np.nan,
                             test=test, p_retained_vs_discarded=pval))
    for a in [0.0] + ALPHAS:
        arm = by_name["original"] if a == 0 else by_name[f"F1_hist_a{a:g}_matched42"]
        kept = pooled_neg_mask(arm)
        ret = kept.sum() / neg_all.sum()
        for f in feats.columns:
            v = feats[f].to_numpy(float)
            ok = np.isfinite(v)
            drift.append(dict(dataset=ctx.task, alpha=a, retained_neg_frac=ret, feature=f,
                              n_retained_nonmissing=int((kept & ok).sum()),
                              mean_retained_neg=np.nanmean(v[kept & ok]) if (kept & ok).any() else np.nan,
                              mean_pos=np.nanmean(v[pos_all & ok]), mean_neg_all=np.nanmean(v[neg_all & ok])))
    return pd.DataFrame(rows), pd.DataFrame(drift)


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="", help="comma list (default: all three)")
    ap.add_argument("--boot", type=int, default=C.BOOT)
    ap.add_argument("--skip-selection-bias", action="store_true")
    ap.add_argument("--smoke", default="", help="scratch dir: tiny grid, separate cache + outputs")
    ap.add_argument("--phase", choices=["fit", "eval", "all"], default="all",
                    help="fit = prioritized fitting across datasets (tiers 0-3); eval = metrics from cache")
    args = ap.parse_args()
    global SNAP, OUT, ALPHAS, MATCH_SEEDS, RAND_SEEDS, TRIM_QS, CALIPERS
    if args.smoke:
        SNAP = OUT = args.smoke
        ALPHAS, MATCH_SEEDS, RAND_SEEDS, TRIM_QS, CALIPERS = [1.0], [42], [0, 1], [0.3], [0.2]
    tasks = [(t, r) for t, r in DATASETS if not args.datasets or t in args.datasets.split(",")]
    parts = os.path.join(SNAP, "parts")
    os.makedirs(parts, exist_ok=True)
    ctxs, armsets = {}, {}
    if args.phase in ("fit", "all"):
        for task, role in tasks:
            ctxs[task] = Ctx(task, role)
            armsets[task] = build_arms(ctxs[task])
            print(f"[{task}] {len(armsets[task])} arms "
                  f"(tiers: {pd.Series([tier(a) for a in armsets[task]]).value_counts().sort_index().to_dict()})",
                  flush=True)
        fit_tiers(tasks, ctxs, armsets)
        if args.phase == "fit":
            return
    for task, role in tasks:
        tdone = os.path.join(parts, f"{task}.done")
        if os.path.exists(tdone):
            print(f"[{task}] already complete (delete {tdone} to recompute)", flush=True)
            continue
        ctx = ctxs.pop(task, None) or Ctx(task, role)
        arms = armsets.pop(task, None) or build_arms(ctx)
        ctx, arms, preds, bal, ev = run_dataset(ctx, arms, args.boot)
        bal.to_csv(os.path.join(parts, f"{task}_balance.csv"), index=False)
        ev.to_csv(os.path.join(parts, f"{task}_sweep.csv"), index=False)
        if not args.skip_selection_bias:
            t0 = time.time()
            sb, dr = selection_bias(ctx, arms, {a.name: a for a in arms})
            sb.to_csv(os.path.join(parts, f"{task}_selection_bias.csv"), index=False)
            dr.to_csv(os.path.join(parts, f"{task}_selection_drift.csv"), index=False)
            print(f"[{task}] selection-bias features in {time.time() - t0:.0f}s", flush=True)
        open(tdone, "w").write("ok\n")
    # combine whatever parts exist, in DATASETS order
    for suffix in ("sweep", "balance", "selection_bias", "selection_drift"):
        frames = [pd.read_csv(os.path.join(parts, f"{t}_{suffix}.csv")) for t, _ in DATASETS
                  if os.path.exists(os.path.join(parts, f"{t}_{suffix}.csv"))]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(os.path.join(OUT, f"psm_sensitivity_{suffix}.csv"),
                                                        index=False)
    print("wrote results/upgrades/psm_sensitivity_{sweep,balance,selection_bias,selection_drift}.csv", flush=True)


if __name__ == "__main__":
    main()
