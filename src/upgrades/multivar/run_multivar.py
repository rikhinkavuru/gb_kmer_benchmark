#!/usr/bin/env python3
"""Multivariate covariate analysis of real-task learnability (Reviewer 1, comment 1).

Question: the synthetic discriminability dose-response is tight (Spearman +0.924, n=135), but on
the real TF-motif tasks the univariate fit MCC ~ discriminability is loose (Spearman +0.74, n=11;
leave-one-task-out R^2 +0.44 [-0.27,+0.70], MAE 0.14). Do sequence length, GC content variance,
homology redundancy, or other measurable task properties explain the deviations, and does
discriminability survive controlling for them?

=====================================================================================================
PRE-REGISTERED ANALYSIS PLAN  (written 2026-09-23, BEFORE any covariate below was computed and
BEFORE any multivariate model was fit. The univariate discriminability LOTO result was already
public in results/upgrades/route1_predict.csv; it is REPRODUCED here, not re-derived.)
=====================================================================================================
Units = tasks.  S11 = the 11 TF-motif tasks (results/cross_suite_summary.csv tf_motif_task==True),
PRIMARY.  S15 = all 15 tasks, SECONDARY.

Outcomes.  Y1 = benchmark MCC (cross_suite_summary.csv bench_mcc == Table 1: validation-selected k,
LightGBM, seed 42) -- PRIMARY.  Y2 = benchmark AUROC (bench_auroc; macro one-vs-rest for 3-class).

Covariates (fixed list and fixed transforms; every model z-scores its predictors with the mean/SD
re-estimated inside each training fold / bootstrap resample):
  disc        max positive-direction motif-only AUROC (cross_suite_summary max_pos_motif_auroc)
  len_log     log10 mean sequence length (train+test, all classes)
  len_sd_log  log10(1 + SD of sequence length in bp)            (many tasks are fixed-length: SD=0)
  gc_mean     mean per-sequence GC fraction over A/C/G/T positions (train+test)
  gc_sd       within-class pooled SD of per-sequence GC  (the GC variance NOT due to the class gap)
  gc_gap      |Cohen's d| of per-sequence GC between classes (pooled SD; max over class pairs, 3-class)
  homology    fraction of TEST sequences whose max exact 8-mer Jaccard to the whole TRAIN split is
              >= 0.7 (method of src/upgrades/homology/measure_homology.py; its 4 existing rows are
              reused, the other 11 are computed by measure_homology_all.py with the same functions)
  ntrain_log  log10 n_train
  multiclass  1 if the task has 3 classes else 0 (n_classes is 2 or 3, so this == n_classes - 2)
  pos_delta   positional-rescue delta MCC at B=8 (results/upgrades/positional_sweep.csv delta_mcc)
  comp        composition-only baseline: 20-d mono+dinucleotide signature (_common.comp_signature),
              same fixed LightGBM config, original split. comp MCC for Y1, comp AUROC for Y2.
Pool A (PRIMARY, "descriptive") = the first ten.  Pool B = Pool A + comp.  comp is a performance-
derived covariate (a weaker classifier's score on the same test set), so a model containing it
DECOMPOSES rather than explains the outcome; it is reported separately and never used for the
decision rule.  Tasks lacking a covariate value would be dropped from models using it (none expected).

Analyses
  A0  Reproduce Y1 ~ disc on S11: LOTO R^2 and MAE must equal route1_predict.csv (0.439 / 0.140)
      -- a hard sanity check; the script aborts if it fails.
  A1  Robustness of disc to each covariate c (Pool B minus disc): (i) rank-based first-order partial
      Spearman of (disc, Y | c), p from t with df = n-3, 95% task-bootstrap percentile CI (1000
      resamples, seed 42); (ii) standardized OLS coefficient of disc in Y ~ disc + c, with task-
      bootstrap CI and t-based CI.  Read-out: count of covariates for which the partial rho stays
      > 0 with its bootstrap CI excluding 0.  Also done on S15 and for Y2 (secondary).
  A2  Best-subset OLS: every subset of size 1-3 of Pool A (175 models; Pool B: 231) on S11/Y1
      (primary; S11/Y2 and S15/Y1 secondary). Per model: in-sample R^2, adjusted R^2,
      AICc = n ln(RSS/n) + 2K + 2K(K+1)/(n-K-1) with K = p + 2, LOTO predictions (standardization refit
      inside each fold; a predictor constant within a training fold is dropped for that fold -- OLS
      predictions are invariant to the affine standardization otherwise), LOTO R^2 (SS_tot about the
      full-sample mean, exactly as route1_predict.py), LOTO MAE.  Ranked by AICc (primary) and by
      LOTO MAE (secondary).  Top 5 of each ranking reported with standardized coefficients (X and Y
      z-scored), raw per-SD coefficients, 95% task-bootstrap percentile CIs (resamples whose design
      is singular or has a constant column are skipped and counted), t-based CIs, partial
      correlations (Pearson, each predictor vs Y given the other predictors in the model), VIFs.
      The "disc-anchored" family (disc + 0-2 Pool-A covariates, 46 models) is ranked the same way.
  A3  Selection-honest performance (nested LOTO): for each held-out task, rerun the whole subset
      search on the other n-1 tasks (select by min AICc; separately by min inner-LOTO MAE), refit the
      selected model, predict the held-out task.  Nested-LOTO R^2 / MAE are the honest estimate of
      what "best covariate model" buys out of sample.
  A4  Selection-optimism nulls (1000 permutations each, seed 42):
      A4a global -- permute Y across tasks, rerun the Pool-A search, record best LOTO R^2.
      A4b incremental -- keep (Y, disc) fixed, permute the ROWS of the other nine Pool-A covariates
          jointly (keeps their mutual correlation, breaks their link to Y and disc), rerun the
          disc-anchored search, record the best LOTO MAE.  p = (1 + #{null best MAE <= observed})/1001.
  A5  LMG / Shapley R^2 decomposition (mean incremental in-sample R^2 over all predictor orderings)
      for the top-AICc model and the top disc-anchored model, with task-bootstrap CIs.
  A6  Residual attribution: from the disc-only LOTO fit, flag tasks with |LOTO residual| > LOTO RMSE.
      For each flagged task and each c in Pool B, refit Y ~ disc + c by LOTO and record the reduction
      in |residual|; name the covariate with the largest reduction; call the deviation "explained" if
      that reduction is >= 50% of the disc-only |residual|, else "not explained by any measured
      covariate".  Also Spearman(disc-only LOTO residual, c) over tasks for every c.
  A7  Noise ceiling: each task's test-set sampling SE from its Table-1 bootstrap CI,
      se = (hi - lo)/3.92 (cross_suite_summary mcc_ci); report mean(se^2) / (disc-only LOTO MSE) =
      the share of the misfit that test-set sampling noise could account for.
  A8  Reviewer-specified model Y1 ~ disc + len_log + gc_sd + homology (4 predictors).  On S11 it is
      over-parameterized (n/p < 3, residual df 6): coefficients + LOTO reported, flagged as such.
      Also on S15.
  A9  Synthetic-law transfer: predict real MCC from the pooled OLS line fitted to the synthetic
      dose-response points (results/upgrades/disc_dose.csv; zero parameters fitted on real tasks);
      report MAE, R^2 and mean signed residual on S11.

DECISION RULE (fixed now).  "Measured covariates improve on discriminability alone" is declared only
if ALL of: (a) the nested-LOTO MAE of the AICc-selected Pool-A model (A3) is below the disc-only LOTO
MAE; (b) the paired task-bootstrap 95% CI of dMAE = MAE(nested) - MAE(disc-only) excludes 0 (same
resample indices for both, 1000 resamples, seed 42); (c) the incremental permutation p (A4b) < 0.05.
If any fails, the report states that n is too small to resolve covariate effects beyond
discriminability -- whatever the in-sample AICc winner looks like -- and the in-sample winners are
labelled exploratory.  Nothing in this plan is changed after seeing results; any post-hoc analysis
is labelled POST-HOC in the output.

Inputs: results/upgrades/multivar_covariates.csv (built by build_covariates.py), route1_predict.csv,
cross_suite_summary.csv, disc_dose.csv.  CPU-only, torch-free, no model refitting here (OLS on <=15
rows).  Writes results/upgrades/multivar_*.csv + multivar_interpretation.txt.
"""
# -----------------------------------------------------------------------------------------------------
# IMPLEMENTATION NOTES (added 2026-09-24 while coding, BEFORE run_multivar.py was first executed and
# before multivar_covariates.csv existed).  None changes the plan or the decision rule above.
#  * LOTO predictions use the exact PRESS identity pred_i = y_i - e_i / (1 - h_ii) (OLS with intercept
#    is invariant to the per-fold affine standardization); wherever 1 - h_ii <= 1e-8 (the held-out
#    task is the only one carrying some direction of the design, e.g. a constant training column) the
#    fold is refit explicitly with constant training columns dropped, as the plan specifies.
#    tests/test_multivar.py checks this against a brute-force per-fold standardized refit.
#  * Supplementary (NOT in the decision rule, labelled "supplementary" in the outputs): nested LOTO
#    (A3) is also run for the disc-anchored family and for Pool B; A4 is also run for the two
#    secondary configurations; A6/A7/A9 are also reported on S15.
#  * The machine-sharing memory cap (_mv.start_memguard) is started first; OLS on <=15 rows needs
#    <200 MB.
# -----------------------------------------------------------------------------------------------------
import itertools  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _mv  # noqa: E402

RD = _mv.RESULTS_DIR
SEED = 42
B = 1000
NPERM = 1000
POOL_A = ["disc", "len_log", "len_sd_log", "gc_mean", "gc_sd", "gc_gap", "homology", "ntrain_log",
          "multiclass", "pos_delta"]
COMP = {"bench_mcc": "comp_mcc", "bench_auroc": "comp_auroc"}
YNAME = {"bench_mcc": "Y1_MCC", "bench_auroc": "Y2_AUROC"}
# (subset, outcome, role)
CONFIGS = [("S11", "bench_mcc", "PRIMARY"), ("S11", "bench_auroc", "secondary"),
           ("S15", "bench_mcc", "secondary")]
A1_CONFIGS = [("S11", "bench_mcc"), ("S15", "bench_mcc"), ("S11", "bench_auroc"), ("S15", "bench_auroc")]
EPS_LEV = 1e-8


# ============================================================================ OLS machinery
def _d(X):
    return np.column_stack([np.ones(X.shape[0]), X])


def zscore(X):
    X = np.asarray(X, float)
    sd = X.std(axis=0, ddof=1)
    return (X - X.mean(axis=0)) / sd


def fit(X, y):
    """OLS with intercept. Returns (beta, resid, rss)."""
    X1 = _d(X)
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    r = y - X1 @ beta
    return beta, r, float(r @ r)


def _fold_pred(X, y, i):
    """Explicit LOTO fold: per-fold z-scoring, constant training columns dropped."""
    m = np.ones(len(y), bool)
    m[i] = False
    Xt = X[m]
    mu, sd = Xt.mean(axis=0), Xt.std(axis=0, ddof=1)
    keep = sd > 1e-12
    if not keep.any():
        return float(y[m].mean())
    Zt = (Xt[:, keep] - mu[keep]) / sd[keep]
    zi = (X[i, keep] - mu[keep]) / sd[keep]
    beta = np.linalg.lstsq(_d(Zt), y[m], rcond=None)[0]
    return float(beta[0] + zi @ beta[1:])


def loto_pred(X, y):
    """Exact leave-one-task-out OLS predictions (see IMPLEMENTATION NOTES)."""
    n = len(y)
    X1 = _d(X)
    pred = np.empty(n)
    bad = np.ones(n, bool)
    if np.linalg.matrix_rank(X1) == X1.shape[1]:
        Q, R = np.linalg.qr(X1)
        beta = np.linalg.solve(R, Q.T @ y)
        e = y - X1 @ beta
        h = (Q ** 2).sum(axis=1)
        bad = (1.0 - h) <= EPS_LEV
        ok = ~bad
        pred[ok] = y[ok] - e[ok] / (1.0 - h[ok])
    for i in np.where(bad)[0]:
        pred[i] = _fold_pred(X, y, i)
    return pred


def loto_pred_bruteforce(X, y):
    """Reference implementation (every fold refit explicitly) -- used by the unit test."""
    return np.array([_fold_pred(X, y, i) for i in range(len(y))])


def loto_metrics(y, pred):
    r = y - pred
    return 1.0 - float(r @ r) / float(((y - y.mean()) ** 2).sum()), float(np.mean(np.abs(r)))


def aicc(rss, n, p):
    K = p + 2
    return n * np.log(rss / n) + 2 * K + 2 * K * (K + 1) / (n - K - 1)


def r2_in(X, y):
    if X.shape[1] == 0:
        return 0.0
    _, _, rss = fit(X, y)
    return 1.0 - rss / float(((y - y.mean()) ** 2).sum())


def _degenerate(X):
    if np.any(X.std(axis=0, ddof=1) < 1e-12):
        return True
    return np.linalg.matrix_rank(_d(X)) < X.shape[1] + 1


# ============================================================================ per-model detail
def model_detail(X, y, names):
    """Standardized coefficients (X and y z-scored), raw per-SD coefficients, t- and task-bootstrap
    CIs, Pearson partial correlations, VIFs."""
    n, p = X.shape
    Zx, zy = zscore(X), zscore(y)
    beta, r, rss = fit(Zx, zy)
    df = n - p - 1
    X1 = _d(Zx)
    cov = (rss / df) * np.linalg.pinv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))[1:]
    tcrit = stats.t.ppf(0.975, df)
    tval = beta[1:] / se
    pval = 2 * stats.t.sf(np.abs(tval), df)
    sdy = y.std(ddof=1)
    pcor, vif = [], []
    for j in range(p):
        oth = [k for k in range(p) if k != j]
        if oth:
            _, rx, _ = fit(X[:, oth], X[:, j])
            _, ry, _ = fit(X[:, oth], y)
            pcor.append(float(np.corrcoef(rx, ry)[0, 1]))
            r2j = r2_in(X[:, oth], X[:, j])
            vif.append(1.0 / (1.0 - r2j) if r2j < 1 else np.inf)
        else:
            pcor.append(float(np.corrcoef(X[:, j], y)[0, 1]))
            vif.append(1.0)
    rng = np.random.RandomState(SEED)
    bs, skipped = [], 0
    for _ in range(B):
        idx = rng.randint(0, n, n)
        Xb, yb = X[idx], y[idx]
        if _degenerate(Xb) or yb.std(ddof=1) < 1e-12:
            skipped += 1
            continue
        bs.append(fit(zscore(Xb), zscore(yb))[0][1:])
    bs = np.array(bs)
    lo = np.percentile(bs, 2.5, axis=0) if len(bs) else np.full(p, np.nan)
    hi = np.percentile(bs, 97.5, axis=0) if len(bs) else np.full(p, np.nan)
    rows = []
    for j in range(p):
        rows.append(dict(predictor=names[j], std_beta=beta[j + 1], t_lo=beta[j + 1] - tcrit * se[j],
                         t_hi=beta[j + 1] + tcrit * se[j], t_p=pval[j], boot_lo=lo[j], boot_hi=hi[j],
                         boot_excludes_0=bool(lo[j] > 0 or hi[j] < 0),
                         raw_per_sd=beta[j + 1] * sdy, partial_r=pcor[j], vif=vif[j],
                         n_boot_ok=len(bs), n_boot_skipped=skipped))
    return rows


def lmg(X, y):
    """LMG / Shapley decomposition of in-sample R^2 (mean incremental R^2 over all orderings)."""
    p = X.shape[1]
    cache = {}

    def R2(cols):
        key = tuple(sorted(cols))
        if key not in cache:
            cache[key] = r2_in(X[:, list(key)], y)
        return cache[key]

    perms = list(itertools.permutations(range(p)))
    out = np.zeros(p)
    for perm in perms:
        before = []
        for j in perm:
            out[j] += R2(before + [j]) - R2(before)
            before.append(j)
    return out / len(perms), R2(list(range(p)))


# ============================================================================ searches
def subsets(pool_idx, kmax=3, anchor=None):
    out = []
    if anchor is None:
        for k in range(1, kmax + 1):
            out += list(itertools.combinations(pool_idx, k))
    else:
        rest = [c for c in pool_idx if c != anchor]
        for k in range(0, kmax):
            out += [(anchor,) + c for c in itertools.combinations(rest, k)]
    return out


def search(Xall, y, subs, with_loto=True):
    """Every subset -> (AICc, in-sample R^2, adj R^2, LOTO R^2, LOTO MAE)."""
    n = len(y)
    tss = float(((y - y.mean()) ** 2).sum())
    res = []
    for s in subs:
        X = Xall[:, list(s)]
        _, _, rss = fit(X, y)
        p = len(s)
        r2 = 1 - rss / tss
        row = [aicc(rss, n, p), r2, 1 - (1 - r2) * (n - 1) / (n - p - 1)]
        if with_loto:
            row += list(loto_metrics(y, loto_pred(X, y)))
        res.append(row)
    return np.array(res)


def best_loto_r2(Xall, y, subs):
    return max(loto_metrics(y, loto_pred(Xall[:, list(s)], y))[0] for s in subs)


def best_loto_mae(Xall, y, subs):
    return min(loto_metrics(y, loto_pred(Xall[:, list(s)], y))[1] for s in subs)


def nested_loto(Xall, y, subs, rule):
    """Selection-honest LOTO: for each held-out task rerun the subset search on the other n-1
    (select by min AICc or by min inner-LOTO MAE), refit, predict the held-out task."""
    n = len(y)
    pred, chosen = np.empty(n), []
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        Xt, yt = Xall[m], y[m]
        sc = search(Xt, yt, subs, with_loto=(rule == "loto_mae"))
        k = int(np.argmin(sc[:, 0])) if rule == "aicc" else int(np.argmin(sc[:, 4]))
        s = list(subs[k])
        chosen.append(subs[k])
        # refit on n-1 with per-fold z-scoring (dropping constant training columns), predict i
        Xs = Xall[:, s]
        pred[i] = _fold_pred(Xs, y, i)
    return pred, chosen


def partial_spearman(x, y, z):
    # FIX ROUND 1 (2026-09-24 evening, after verification; an implementation fix, the plan is unchanged):
    # the partial is undefined when x or y is a perfect monotone function of z (|r| = 1).  np.corrcoef can
    # return 1 - 2e-16 for identical rank vectors, which slipped past the old `den > 1e-12` test and put
    # numerical noise (values near 0) into a few bootstrap resamples (<= 9 of 1000 per covariate; CI
    # endpoints moved by <= 0.013, no include/exclude-0 call changed).  Now |r| within 1e-10 of 1 -> NaN
    # (the resample is skipped, as the bootstrap loops already do for NaN).
    R = np.corrcoef(np.vstack([stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)]))
    rxy, rxz, ryz = R[0, 1], R[0, 2], R[1, 2]
    if 1 - rxz ** 2 < 1e-10 or 1 - ryz ** 2 < 1e-10:
        return np.nan
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def paired_boot_dmae(err_a, err_b):
    """95% percentile CI of mean|err_a| - mean|err_b| (same task resample for both)."""
    rng = np.random.RandomState(SEED)
    n = len(err_a)
    d = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        d.append(np.mean(np.abs(err_a[idx])) - np.mean(np.abs(err_b[idx])))
    return np.percentile(d, [2.5, 97.5])


def fmt_model(names):
    return " + ".join(names)


def loto_boot_ci(y, pred):
    """LOTO R^2 / MAE with the route1_predict task-bootstrap CI (seed 42, 1000 resamples of residuals)."""
    r2, mae = loto_metrics(y, pred)
    rng = np.random.RandomState(SEED)
    r2b, maeb = [], []
    for _ in range(1000):
        idx = rng.randint(0, len(y), len(y))
        rr = y[idx] - pred[idx]
        st = np.sum((y[idx] - y[idx].mean()) ** 2)
        if st > 0:
            r2b.append(1 - np.sum(rr ** 2) / st)
            maeb.append(np.mean(np.abs(rr)))
    return (r2, np.percentile(r2b, 2.5), np.percentile(r2b, 97.5),
            mae, np.percentile(maeb, 2.5), np.percentile(maeb, 97.5))


def partial_boot(x, y, z, return_n=False):
    """Partial Spearman rho(x, y | z): value, t-test p (df n-3), 95% task-bootstrap CI (seed 42; same
    resampling and skip rule as A1)."""
    n = len(y)
    v = partial_spearman(x, y, z)
    # FIX ROUND 2: |rho| = 1 (e.g. rho(disc, MCC) = 1 on a small subset makes rho(disc, MCC | z) = 1) has an
    # infinite t and no meaningful p; it used to be printed as p = 0.  Now p = NaN ("degenerate").
    if np.isnan(v) or abs(v) > 1 - 1e-12:  # undefined (y a perfect monotone function of z) or degenerate
        p = np.nan
    else:
        t = v * np.sqrt((n - 3) / (1 - v ** 2))
        p = 2 * stats.t.sf(abs(t), n - 3)
    rng = np.random.RandomState(SEED)
    bs = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        if min(x[idx].std(), y[idx].std(), z[idx].std()) < 1e-12:
            continue
        w = partial_spearman(x[idx], y[idx], z[idx])
        if not np.isnan(w):
            bs.append(w)
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    # FIX ROUND 3: a degenerate |rho| = 1 partial also gets no percentile CI (it used to be [+1, +1], which
    # a reader could copy out of the CSVs as if it were an interval estimate)
    if not np.isnan(v) and abs(v) > 1 - 1e-12:
        lo, hi = np.nan, np.nan
    if return_n:                          # number of the B resamples on which the partial is defined
        return v, p, lo, hi, len(bs)
    return v, p, lo, hi


_PERMS = {}


def _all_orderings(n):
    if n not in _PERMS:
        _PERMS[n] = np.array(list(itertools.permutations(range(n))), dtype=np.int8)
    return _PERMS[n]


def perm_corr_p(a, b, n_mc=200000, exact_max=9):
    """FIX ROUND 2 (post hoc companion of the t-approximation at small n): two-sided permutation p of the
    Pearson correlation of a and b -- used on ranks (= Spearman rho) and on rank residuals (= partial
    Spearman).  Exact over all n! orderings of b for n <= 9 (p = share of orderings with |r| >= |r_obs|,
    the observed ordering included); otherwise Monte Carlo with n_mc random orderings (seed 42,
    p = (1 + #)/(1 + n_mc)).  Returns (p, method)."""
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = np.sqrt((a @ a) * (b @ b))
    if den <= 0:
        return np.nan, "undefined"
    r0 = abs(a @ b) / den
    n = len(a)
    if n <= exact_max:
        P = _all_orderings(n)
        rs = (b[P] @ a) / den
        return float(np.mean(np.abs(rs) >= r0 - 1e-12)), f"exact ({len(P)} orderings)"
    rng = np.random.RandomState(SEED)
    cnt, done = 0, 0
    while done < n_mc:
        m = min(20000, n_mc - done)
        idx = np.argsort(rng.rand(m, n), axis=1)
        cnt += int(np.sum(np.abs((b[idx] @ a) / den) >= r0 - 1e-12))
        done += m
    return (1 + cnt) / (1 + n_mc), f"Monte Carlo ({n_mc} orderings)"


def _perm_counts_dp(a, b):
    """FIX ROUND 3: exact distribution, over all n! orderings pi, of T = sum_i a[i] * b[pi(i)] for
    non-negative integer vectors a, b (dynamic programming over the subset of b already assigned, one
    position of a per layer).  Returns counts[T] (int64, summing to n!)."""
    a = [int(v) for v in a]
    b = [int(v) for v in b]
    n = len(a)
    tmax = int(np.dot(sorted(a), sorted(b)))           # rearrangement inequality: the largest attainable T
    start = np.zeros(tmax + 1, np.int64)
    start[0] = 1
    layer = {0: start}
    for i in range(n):
        new = {}
        for mask, cnt in layer.items():
            for j in range(n):
                if mask >> j & 1:
                    continue
                sh = a[i] * b[j]
                m2 = mask | (1 << j)
                arr = new.get(m2)
                if arr is None:
                    arr = new[m2] = np.zeros(tmax + 1, np.int64)
                arr[sh:] += cnt[:tmax + 1 - sh]          # every reachable partial sum stays <= tmax
        layer = new
    out = layer[(1 << n) - 1]
    assert int(out.sum()) == int(np.prod(np.arange(1, n + 1, dtype=np.int64)))
    return out


def exact_rank_perm_p(x, y):
    """FIX ROUND 3: exact two-sided permutation p of Spearman rho over ALL n! orderings, for any n <= 12,
    by dynamic programming (identical, up to float rounding, to enumerating the orderings; tests check it
    against the brute-force loop, with and without ties).  Ranks are mid-ranks (multiples of 1/2), so
    2 x rank is an integer; rho is an increasing affine function of T = sum_i a_i b_pi(i), hence
    |rho| >= |rho_obs|  <=>  |T - n mean(a) mean(b)| >= |T_obs - n mean(a) mean(b)|."""
    a = np.rint(2 * stats.rankdata(x)).astype(np.int64)
    b = np.rint(2 * stats.rankdata(y)).astype(np.int64)
    if np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    a //= np.gcd.reduce(a)
    b //= np.gcd.reduce(b)
    cnt = _perm_counts_dp(a, b)
    c = len(a) * a.mean() * b.mean()
    d0 = abs(float(a @ b) - c)
    dev = np.abs(np.arange(len(cnt)) - c)
    return float(cnt[dev >= d0 - 1e-9].sum() / cnt.sum())


def perm_spearman_p(x, y):
    """two-sided permutation p of Spearman rho: exact over all n! orderings for n <= 12 (fix round 3: DP;
    fix round 2 enumerated n <= 9 and used Monte Carlo for n = 10-11), Monte Carlo above."""
    n = len(x)
    if n <= 12:
        return exact_rank_perm_p(x, y), f"exact (all {n}! orderings)"
    return perm_corr_p(stats.rankdata(x), stats.rankdata(y))


def spearman_t(x, y):
    """scipy Spearman rho with its t-approximation p, except that a degenerate |rho| = 1 gets p = NaN
    (scipy returns p = 0 there, which is not a p-value; use perm_spearman_p instead).  Fix round 2."""
    r, p = stats.spearmanr(x, y)
    return r, (np.nan if abs(r) > 1 - 1e-12 else p)


def rank_residuals(x, z):
    rx, rz = stats.rankdata(x), stats.rankdata(z)
    return rx - np.polyval(np.polyfit(rz, rx, 1), rz)


def perm_partial_p(x, y, z):
    """FIX ROUND 2 (post hoc): residual-permutation p of the partial Spearman rho(x, y | z).  The partial
    equals corr(e_x, e_y) of the rank residuals on rank(z) (tests/test_multivar.py); e_y is permuted
    against e_x (Freedman-Lane-type; approximate because residuals are not exactly exchangeable).
    (nan, 'degenerate') when the partial is undefined or |rho| = 1."""
    v = partial_spearman(x, y, z)
    if np.isnan(v) or abs(v) > 1 - 1e-12:
        return np.nan, "degenerate"
    return perm_corr_p(rank_residuals(x, z), rank_residuals(y, z))


def holm(p):
    """Holm step-down adjusted p-values (NaN-free input)."""
    p = np.asarray(p, float)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def table1_routes():
    """{task: route label} parsed from Table I (tab:land) of the submitted manuscript
    bibm/paper_bibm_submitted.tex (read-only; the route labels are the paper's, not re-derived here)."""
    import re
    fp = os.path.join(_mv.ROOT, "bibm", "paper_bibm_submitted.tex")
    out = {}
    with open(fp) as fh:
        for line in fh:
            m = re.match(r"^(GB|NT) & \\texttt\{([^}]*)\} & .*& ([^&]*?)\\\\\s*$", line.strip())
            if m:
                t = m.group(2).replace("\\_", "_")
                t = t if t in _mv.TASKS else t + "_seqs"          # Table I prints demo_coding_vs_intergenomic
                assert t in _mv.TASKS, t
                out[t] = m.group(3).strip().replace("contam.~(weak)", "contamination (weak)")
    assert len(out) == 15, out
    return out


# FIX ROUND 3: exact-sequence sharing among the S11 tasks (audit_task_overlap.py; P12 asserts that these are
# exactly the S11 pairs sharing >= 0.1% of either task's distinct sequences).  The other pairs share at most a
# handful of sequences (human_enhancers_ensembl / human_ocr_ensembl: 13, 0.011%).
EXACT_SHARING = [("nt_promoter_all", "nt_promoter_tata"), ("nt_promoter_all", "nt_promoter_no_tata"),
                 ("nt_enhancers", "nt_enhancers_types"), ("human_ensembl_regulatory", "human_ocr_ensembl")]


def max_independent_sets(nodes, edges):
    """all maximum-size subsets of `nodes` containing no edge (brute force; <= 11 nodes here)."""
    nodes = list(nodes)
    e = [(nodes.index(a), nodes.index(b)) for a, b in edges if a in nodes and b in nodes]
    best, sets = 0, []
    for m in range(1 << len(nodes)):
        if any((m >> i & 1) and (m >> j & 1) for i, j in e):
            continue
        k = bin(m).count("1")
        if k > best:
            best, sets = k, [m]
        elif k == best:
            sets.append(m)
    return [[nodes[i] for i in range(len(nodes)) if m >> i & 1] for m in sets]


def n_distinct_sets(tasks):
    """Exact-duplicate structure of a task set (audit_task_overlap.py).  FIX ROUND 3 (the round-2 helper
    miscounted a set holding nt_promoter_all and only ONE of its components; no reported set had that case):
    returns (n_exact_distinct, n_no_exact_sharing):
      n_exact_distinct   tasks that are not an exact copy of others in the set: nt_promoter_all is redundant
                         only when BOTH nt_promoter_tata and nt_promoter_no_tata are present (it is their exact
                         union), and nt_enhancers / nt_enhancers_types are one sequence collection;
      n_no_exact_sharing the largest number of tasks in the set that pairwise share no exact sequence
                         (maximum independent set of EXACT_SHARING; the regulatory task's OCR class IS
                         human_ocr_ensembl's positive class).
    Exact strings only: locus-level sharing through shifted windows is measured by audit_locus_overlap.py."""
    t = set(tasks)
    k = len(t) - int({"nt_promoter_all", "nt_promoter_tata", "nt_promoter_no_tata"} <= t) \
        - int({"nt_enhancers", "nt_enhancers_types"} <= t)
    return k, len(max_independent_sets(sorted(t), EXACT_SHARING)[0])


REG = "human_ensembl_regulatory"


def gc_disc_readout(tk, xd, gc, yv):
    """FIX ROUND 3 (post hoc, descriptive; claims verifier round 3): for a task set, GC mean's OWN Spearman
    correlation with MCC next to disc's (exact permutation p for both), the paired task-bootstrap 95% CI of
    rho(GC, MCC) - rho(disc, MCC) (seed 42, B resamples; resamples with a constant column skipped), disc-only
    vs GC-only LOTO MAE, which one is lower, the tasks whose single removal reverses that ordering, and both
    MAEs with the 3-class regulatory task removed (its GC mean pools three classes)."""
    tk = list(tk)
    n = len(yv)
    rdm, rgm = stats.spearmanr(xd, yv)[0], stats.spearmanr(gc, yv)[0]
    rng = np.random.RandomState(SEED)
    dif = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        if min(xd[idx].std(), gc[idx].std(), yv[idx].std()) < 1e-12:
            continue
        a_, b_ = stats.spearmanr(gc[idx], yv[idx])[0], stats.spearmanr(xd[idx], yv[idx])[0]
        if a_ == a_ and b_ == b_:
            dif.append(a_ - b_)
    dlo, dhi = np.percentile(dif, [2.5, 97.5])

    def maes(keep):
        return (loto_metrics(yv[keep], loto_pred(xd[keep][:, None], yv[keep]))[1],
                loto_metrics(yv[keep], loto_pred(gc[keep][:, None], yv[keep]))[1])

    md_, mg_ = maes(np.ones(n, bool))
    if REG in tk:                          # the regulatory task's own held-out absolute errors in this set
        i_ = tk.index(REG)
        ed_ = abs(yv[i_] - loto_pred(xd[:, None], yv)[i_])
        eg_ = abs(yv[i_] - loto_pred(gc[:, None], yv)[i_])
    else:
        ed_ = eg_ = np.nan
    order = "gc" if mg_ < md_ else "disc"
    flips = []
    for i in range(n):
        keep = np.arange(n) != i
        if keep.sum() >= 4:
            a_, b_ = maes(keep)
            if ("gc" if b_ < a_ else "disc") != order:
                flips.append(tk[i])
    if REG in tk:
        mdr, mgr = maes(np.array([t != REG for t in tk]))
    else:
        mdr, mgr = md_, mg_
    return dict(spearman_gc_mcc=rgm, spearman_gc_mcc_p_perm=perm_spearman_p(gc, yv)[0],
                spearman_disc_mcc_p_perm=perm_spearman_p(xd, yv)[0],
                rho_gc_minus_rho_disc=rgm - rdm, rho_gc_minus_rho_disc_lo=dlo, rho_gc_minus_rho_disc_hi=dhi,
                rho_diff_n_boot_ok=len(dif),
                loto_MAE_disc=md_, loto_MAE_gc=mg_, loto_MAE_lower=order,
                loto_MAE_order_reversed_by_dropping=";".join(flips),
                contains_regulatory=REG in tk, abs_err_disc_regulatory=ed_, abs_err_gc_regulatory=eg_,
                loto_MAE_disc_without_regulatory=mdr, loto_MAE_gc_without_regulatory=mgr,
                loto_MAE_lower_without_regulatory="gc" if mgr < mdr else "disc")


def fisher_z_loto(x, y):
    """POST-HOC bounded-outcome LOTO: OLS on atanh(MCC), back-transformed with tanh (predictions stay in
    (-1, 1)).  Same per-fold refit as loto_pred."""
    assert np.all(np.abs(y) < 1)
    return np.tanh(loto_pred(x, np.arctanh(y)))


# ============================================================================ main
def main():
    _mv.start_memguard(tag="run_multivar")
    t0 = time.time()
    cov = pd.read_csv(os.path.join(RD, "multivar_covariates.csv"))
    assert len(cov) == 15, len(cov)
    need = POOL_A + ["comp_mcc", "comp_auroc", "bench_mcc", "bench_auroc"]
    miss = cov[need].isna().sum()
    if miss.any():
        sys.exit(f"missing covariate values (plan expects none): {miss[miss > 0].to_dict()}")
    subsets_of = {"S11": cov[cov["S11"]].reset_index(drop=True), "S15": cov.reset_index(drop=True)}
    L = []            # interpretation lines
    S = []            # summary key/value rows

    def add(key, value, source, note=""):
        S.append(dict(key=key, value=value, source=source, note=note))

    # ------------------------------------------------------------------ A0 reproduce
    d11 = subsets_of["S11"]
    x = d11[["disc"]].to_numpy(float)
    y = d11["bench_mcc"].to_numpy(float)
    pr = loto_pred(x, y)
    r2, mae = loto_metrics(y, pr)
    ref = pd.read_csv(os.path.join(RD, "route1_predict.csv")).set_index("task")
    ref_pred = ref.loc[d11["task"], "loto_pred_mcc"].to_numpy(float)
    rng = np.random.RandomState(SEED)
    r2s, maes = [], []
    n = len(y)
    for _ in range(1000):
        idx = rng.randint(0, n, n)
        rr = y[idx] - pr[idx]
        st = np.sum((y[idx] - y[idx].mean()) ** 2)
        if st > 0:
            r2s.append(1 - np.sum(rr ** 2) / st)
            maes.append(np.mean(np.abs(rr)))
    a0_ok = abs(r2 - 0.439) < 5e-4 and abs(mae - 0.140) < 5e-4 and np.max(np.abs(pr - ref_pred)) < 1e-3
    L.append("A0 reproduction of route1_predict (S11, MCC ~ disc, LOTO): "
             f"R^2 {r2:+.4f} [{np.percentile(r2s, 2.5):+.3f},{np.percentile(r2s, 97.5):+.3f}]  "
             f"MAE {mae:.4f} [{np.percentile(maes, 2.5):.3f},{np.percentile(maes, 97.5):.3f}]  "
             f"max|pred - route1_predict.csv| {np.max(np.abs(pr - ref_pred)):.1e}  -> "
             f"{'REPRODUCED' if a0_ok else 'FAILED'}")
    if not a0_ok:
        sys.exit("A0 SANITY FAIL: " + L[-1])
    add("A0_disc_only_S11_MCC_loto_r2", round(r2, 4), "multivar_models.csv (S11/Y1_MCC poolA, model=disc)")
    add("A0_disc_only_S11_MCC_loto_mae", round(mae, 4), "multivar_models.csv (S11/Y1_MCC poolA, model=disc)")
    add("A0_disc_only_S11_MCC_loto_r2_ci", f"[{np.percentile(r2s, 2.5):+.3f},{np.percentile(r2s, 97.5):+.3f}]",
        "task bootstrap of LOTO residuals (route1_predict recipe)")

    # ------------------------------------------------------------------ A1 robustness (partials)
    a1 = []
    for sub, yc in A1_CONFIGS:
        dd = subsets_of[sub]
        yv = dd[yc].to_numpy(float)
        xd = dd["disc"].to_numpy(float)
        nn = len(yv)
        rho0, p0 = stats.spearmanr(xd, yv)
        others = [c for c in POOL_A if c != "disc"] + [COMP[yc]]
        for c in others:
            zc = dd[c].to_numpy(float)
            pr_ = partial_spearman(xd, yv, zc)
            tt = pr_ * np.sqrt((nn - 3) / (1 - pr_ ** 2)) if abs(pr_) < 1 else np.inf
            pp = 2 * stats.t.sf(abs(tt), nn - 3)
            # POST-HOC columns (verification round): the REVERSE partial rho(c, outcome | disc), so the
            # collinearity is visible (which of the two carries the signal given the other).  Computed
            # with partial_boot -- the SAME function (seed 42, same resample stream and skip rule) that
            # P7 uses, so the GC-mean row here and multivar_posthoc_gc_vs_disc.csv agree exactly.
            rpr_, rpp, rlo, rhi, rnok = partial_boot(zc, yv, xd, return_n=True)
            rng = np.random.RandomState(SEED)
            bs, sk = [], 0
            for _ in range(B):
                idx = rng.randint(0, nn, nn)
                a, b_, cc = xd[idx], yv[idx], zc[idx]
                if min(a.std(), b_.std(), cc.std()) < 1e-12:
                    sk += 1
                    continue
                v = partial_spearman(a, b_, cc)
                if np.isnan(v):
                    sk += 1
                    continue
                bs.append(v)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            det = model_detail(np.column_stack([xd, zc]), yv, ["disc", c])
            a1.append(dict(config=f"{sub}/{YNAME[yc]}", n=nn, covariate=c,
                           spearman_disc_y=rho0, spearman_p=p0,
                           spearman_c_y=stats.spearmanr(zc, yv)[0],
                           spearman_disc_c=stats.spearmanr(xd, zc)[0],
                           partial_rho=pr_, partial_p=pp, partial_boot_lo=lo, partial_boot_hi=hi,
                           partial_boot_excludes_0=bool(lo > 0 or hi < 0), n_boot_ok=len(bs),
                           disc_std_beta=det[0]["std_beta"], disc_beta_t_lo=det[0]["t_lo"],
                           disc_beta_t_hi=det[0]["t_hi"], disc_beta_t_p=det[0]["t_p"],
                           disc_beta_boot_lo=det[0]["boot_lo"], disc_beta_boot_hi=det[0]["boot_hi"],
                           cov_std_beta=det[1]["std_beta"], cov_beta_boot_lo=det[1]["boot_lo"],
                           cov_beta_boot_hi=det[1]["boot_hi"], vif=det[0]["vif"],
                           posthoc_rev_partial_rho_c_given_disc=rpr_, posthoc_rev_partial_p=rpp,
                           posthoc_rev_partial_boot_lo=rlo, posthoc_rev_partial_boot_hi=rhi,
                           posthoc_rev_partial_boot_excludes_0=bool(rlo > 0 or rhi < 0),
                           posthoc_rev_n_boot_ok=rnok))
    a1 = pd.DataFrame(a1)
    # POST-HOC column (fix round 2): Holm step-down adjustment of the forward t-test p over the covariates
    # of each configuration (the plan reports the partials one covariate at a time, uncorrected).
    a1["posthoc_partial_p_holm"] = a1.groupby("config")["partial_p"].transform(lambda v: holm(v.to_numpy()))
    a1.to_csv(os.path.join(RD, "multivar_partial.csv"), index=False)
    L.append("")
    L.append("A1 robustness of discriminability: partial Spearman(disc, outcome | c) for each covariate c")
    for cfg, g in a1.groupby("config", sort=False):
        pos = int(((g["partial_rho"] > 0) & (g["partial_boot_lo"] > 0)).sum())
        L.append(f"  {cfg}: raw Spearman {g['spearman_disc_y'].iloc[0]:+.3f} (p={g['spearman_p'].iloc[0]:.2g}); "
                 f"partial rho stays >0 with bootstrap CI excluding 0 for {pos}/{len(g)} covariates; "
                 f"partial rho range [{g['partial_rho'].min():+.3f}, {g['partial_rho'].max():+.3f}]")
        for _, r in g.iterrows():
            L.append(f"      | {r['covariate']:<11} partial {r['partial_rho']:+.3f} [{r['partial_boot_lo']:+.3f},"
                     f"{r['partial_boot_hi']:+.3f}] p={r['partial_p']:.2g}   disc beta(std) "
                     f"{r['disc_std_beta']:+.3f} [{r['disc_beta_boot_lo']:+.3f},{r['disc_beta_boot_hi']:+.3f}]"
                     f"   | POST-HOC reverse rho(c, Y | disc) {r['posthoc_rev_partial_rho_c_given_disc']:+.3f} "
                     f"[{r['posthoc_rev_partial_boot_lo']:+.3f},{r['posthoc_rev_partial_boot_hi']:+.3f}] "
                     f"p={r['posthoc_rev_partial_p']:.2g}")
        add(f"A1_{cfg}_n_covariates_partial_rho_CI_excl_0", f"{pos}/{len(g)}", "multivar_partial.csv")
        add(f"A1_{cfg}_partial_rho_min", round(g["partial_rho"].min(), 3), "multivar_partial.csv",
            f"covariate={g.loc[g['partial_rho'].idxmin(), 'covariate']}")
        add(f"A1_{cfg}_n_covariates_partial_rho_t_p_lt_0.05", f"{int((g['partial_p'] < 0.05).sum())}/{len(g)}",
            "multivar_partial.csv (partial_p, t with df=n-3)",
            "parametric companion to the bootstrap count (percentile CIs can hit 1.000 at n=11)")
        add(f"POSTHOC_A1_{cfg}_n_covariates_partial_rho_holm_p_lt_0.05",
            f"{int((g['posthoc_partial_p_holm'] < 0.05).sum())}/{len(g)}",
            "multivar_partial.csv posthoc_partial_p_holm",
            "Holm-adjusted over the covariates of this configuration: "
            + ", ".join(f"{c} {v:.3f}" for c, v in zip(g["covariate"], g["posthoc_partial_p_holm"])))
        add(f"POSTHOC_A1_{cfg}_n_covariates_reverse_partial_CI_excl_0",
            f"{int(g['posthoc_rev_partial_boot_excludes_0'].sum())}/{len(g)}",
            "multivar_partial.csv posthoc_rev_partial_*", "rho(c, outcome | disc)")

    # ------------------------------------------------------------------ A2 / A3 / A4 per config
    all_models, top_rows, nested_rows, perm_rows, lmg_rows = [], [], [], [], []
    nested_store = {}          # (cfg, family, rule) -> (nested LOTO predictions, selected model per fold)
    disc_only_err = {}
    for sub, yc, role in CONFIGS:
        cfg = f"{sub}/{YNAME[yc]}"
        dd = subsets_of[sub]
        yv = dd[yc].to_numpy(float)
        nn = len(yv)
        names_B = POOL_A + [COMP[yc]]
        XB = dd[names_B].to_numpy(float)
        idxA = list(range(len(POOL_A)))
        idxB = list(range(len(names_B)))
        fams = {"poolA": subsets(idxA), "poolB": subsets(idxB), "disc_anchored": subsets(idxA, anchor=0)}
        # disc-only reference
        pr_d = loto_pred(XB[:, [0]], yv)
        disc_only_err[cfg] = yv - pr_d
        r2_d, mae_d = loto_metrics(yv, pr_d)
        for fam, subs in fams.items():
            sc = search(XB, yv, subs)
            dfm = pd.DataFrame(dict(config=cfg, role=role, family=fam, n=nn,
                                    model=[fmt_model([names_B[j] for j in s]) for s in subs],
                                    p=[len(s) for s in subs],
                                    contains_disc=[0 in s for s in subs],
                                    contains_comp=[(len(names_B) - 1) in s for s in subs],
                                    AICc=sc[:, 0], R2=sc[:, 1], adjR2=sc[:, 2],
                                    loto_R2=sc[:, 3], loto_MAE=sc[:, 4]))
            dfm["dAICc"] = dfm["AICc"] - dfm["AICc"].min()
            w = np.exp(-0.5 * dfm["dAICc"])
            dfm["akaike_w"] = w / w.sum()
            dfm["rank_AICc"] = dfm["AICc"].rank(method="min").astype(int)
            dfm["rank_lotoMAE"] = dfm["loto_MAE"].rank(method="min").astype(int)
            dfm["disc_only_loto_MAE"] = mae_d
            dfm["disc_only_loto_R2"] = r2_d
            all_models.append(dfm)
            for ranking, col in (("AICc", "AICc"), ("lotoMAE", "loto_MAE")):
                top = dfm.sort_values([col, "p"]).head(5)
                for rk, (_, mrow) in enumerate(top.iterrows(), 1):
                    s = subs[mrow.name]
                    det = model_detail(XB[:, list(s)], yv, [names_B[j] for j in s])
                    for dr in det:
                        top_rows.append(dict(config=cfg, family=fam, ranking=ranking, rank=rk,
                                             model=mrow["model"], p=mrow["p"], AICc=mrow["AICc"],
                                             dAICc=mrow["dAICc"], R2=mrow["R2"], adjR2=mrow["adjR2"],
                                             loto_R2=mrow["loto_R2"], loto_MAE=mrow["loto_MAE"], **dr))
                if ranking == "AICc" and fam in ("poolA", "disc_anchored"):
                    s = subs[top.index[0]]
                    Xs = XB[:, list(s)]
                    sh, r2full = lmg(Xs, yv)
                    rng = np.random.RandomState(SEED)
                    bs, sk = [], 0
                    for _ in range(B):
                        idx = rng.randint(0, nn, nn)
                        if _degenerate(Xs[idx]) or yv[idx].std() < 1e-12:
                            sk += 1
                            continue
                        bs.append(lmg(Xs[idx], yv[idx])[0])
                    bs = np.array(bs)
                    for j, jj in enumerate(s):
                        lmg_rows.append(dict(config=cfg, family=fam, model=fmt_model([names_B[k] for k in s]),
                                             R2_model=r2full, predictor=names_B[jj], lmg_R2=sh[j],
                                             lmg_share=sh[j] / r2full if r2full > 0 else np.nan,
                                             boot_lo=np.percentile(bs[:, j], 2.5),
                                             boot_hi=np.percentile(bs[:, j], 97.5),
                                             n_boot_ok=len(bs), n_boot_skipped=sk))
            # A3 nested LOTO
            for rule in ("aicc", "loto_mae"):
                prn, chosen = nested_loto(XB, yv, subs, rule)
                nested_store[(cfg, fam, rule)] = (prn.copy(), [fmt_model([names_B[j] for j in s_]) for s_ in chosen])
                r2n, maen = loto_metrics(yv, prn)
                lo, hi = paired_boot_dmae(yv - prn, yv - pr_d)
                cnt = pd.Series([fmt_model([names_B[j] for j in s]) for s in chosen]).value_counts()
                nested_rows.append(dict(
                    config=cfg, role=role, family=fam, selection=rule,
                    in_decision_rule=bool(role == "PRIMARY" and fam == "poolA" and rule == "aicc"),
                    n=nn, nested_loto_R2=r2n, nested_loto_MAE=maen, disc_only_loto_R2=r2_d,
                    disc_only_loto_MAE=mae_d, dMAE=maen - mae_d, dMAE_lo=lo, dMAE_hi=hi,
                    dMAE_CI_excludes_0=bool(lo > 0 or hi < 0),
                    n_distinct_selected=len(cnt),
                    # (post-hoc readout columns, verification round) in how many of the n outer folds
                    # the selected model contains disc / gc_mean
                    n_folds_selected_with_disc=int(sum(0 in s for s in chosen)),
                    n_folds_selected_with_gc_mean=int(sum(names_B.index("gc_mean") in s for s in chosen)),
                    selected_models="; ".join(f"{k} x{v}" for k, v in cnt.items()),
                    per_task_nested_pred=";".join(f"{t}={v:.4f}" for t, v in zip(dd["task"], prn)),
                    # (post-hoc readout column, fix round 2) the model selected in the fold that held out
                    # each task
                    per_task_selected=";".join(f"{t}={fmt_model([names_B[j] for j in s_])}"
                                               for t, s_ in zip(dd["task"], chosen))))
        # A4 permutations (Pool A)
        rngp = np.random.RandomState(SEED)
        subsA, subsD = fams["poolA"], fams["disc_anchored"]
        obs_r2 = best_loto_r2(XB, yv, subsA)
        obs_mae = best_loto_mae(XB, yv, subsD)
        null_r2, null_mae = np.empty(NPERM), np.empty(NPERM)
        for b in range(NPERM):
            null_r2[b] = best_loto_r2(XB, rngp.permutation(yv), subsA)
        for b in range(NPERM):
            Xp = XB.copy()
            perm = rngp.permutation(nn)
            Xp[:, 1:len(POOL_A)] = XB[perm][:, 1:len(POOL_A)]
            null_mae[b] = best_loto_mae(Xp, yv, subsD)
        perm_rows.append(dict(config=cfg, role=role, test="A4a_global_bestLOTO_R2_poolA",
                              observed=obs_r2, null_mean=null_r2.mean(), null_critical=np.percentile(null_r2, 95),
                              p=(1 + np.sum(null_r2 >= obs_r2)) / (NPERM + 1), n_perm=NPERM,
                              note="null_critical = 95th percentile of the null best LOTO R2 (higher = better)"))
        perm_rows.append(dict(config=cfg, role=role, test="A4b_incremental_bestLOTO_MAE_disc_anchored",
                              observed=obs_mae, null_mean=null_mae.mean(), null_critical=np.percentile(null_mae, 5),
                              p=(1 + np.sum(null_mae <= obs_mae)) / (NPERM + 1), n_perm=NPERM,
                              note=f"disc-only LOTO MAE {mae_d:.4f}; null_critical = 5th percentile of the null "
                                   f"best MAE (lower = better)"))
        print(f"[{cfg}] searches + nested + permutations done ({time.time() - t0:.0f}s)", flush=True)

    models = pd.concat(all_models, ignore_index=True)
    models.to_csv(os.path.join(RD, "multivar_models.csv"), index=False)
    top = pd.DataFrame(top_rows)
    top.to_csv(os.path.join(RD, "multivar_top_models.csv"), index=False)
    nested = pd.DataFrame(nested_rows)
    nested.to_csv(os.path.join(RD, "multivar_nested.csv"), index=False)
    perm = pd.DataFrame(perm_rows)
    perm.to_csv(os.path.join(RD, "multivar_permutation.csv"), index=False)
    lmgd = pd.DataFrame(lmg_rows)
    lmgd.to_csv(os.path.join(RD, "multivar_lmg.csv"), index=False)

    # ------------------------------------------------------------------ decision rule (PRIMARY)
    prim = nested[nested["in_decision_rule"]].iloc[0]
    pb = perm[(perm["config"] == "S11/Y1_MCC") & (perm["test"].str.startswith("A4b"))].iloc[0]
    ca = bool(prim["nested_loto_MAE"] < prim["disc_only_loto_MAE"])
    cb = bool(prim["dMAE_CI_excludes_0"] and prim["dMAE"] < 0)
    cc = bool(pb["p"] < 0.05)
    improved = ca and cb and cc
    L.append("")
    L.append("DECISION RULE (pre-registered; S11 / benchmark MCC / Pool A / AICc selection):")
    L.append(f"  (a) nested-LOTO MAE {prim['nested_loto_MAE']:.4f} < disc-only LOTO MAE "
             f"{prim['disc_only_loto_MAE']:.4f}?  {'YES' if ca else 'NO'}")
    L.append(f"  (b) paired bootstrap dMAE {prim['dMAE']:+.4f} [{prim['dMAE_lo']:+.4f},{prim['dMAE_hi']:+.4f}] "
             f"excludes 0 (and < 0)?  {'YES' if cb else 'NO'}")
    L.append(f"  (c) incremental permutation p (A4b) = {pb['p']:.4f} < 0.05?  {'YES' if cc else 'NO'}")
    L.append("  VERDICT: " + ("measured covariates IMPROVE on discriminability alone." if improved else
                             "n is too small to resolve covariate effects beyond discriminability; the "
                             "in-sample AICc winners below are EXPLORATORY."))
    add("DECISION_covariates_improve_on_disc", "YES" if improved else "NO", "multivar_nested.csv + multivar_permutation.csv")
    add("A3_primary_nested_loto_MAE", round(prim["nested_loto_MAE"], 4), "multivar_nested.csv in_decision_rule==True")
    add("A3_primary_nested_loto_R2", round(prim["nested_loto_R2"], 4), "multivar_nested.csv in_decision_rule==True")
    add("A3_primary_dMAE_vs_disc_only", f"{prim['dMAE']:+.4f} [{prim['dMAE_lo']:+.4f},{prim['dMAE_hi']:+.4f}]",
        "multivar_nested.csv in_decision_rule==True")
    add("A3_primary_folds_selected_model_with_disc", f"{int(prim['n_folds_selected_with_disc'])}/{int(prim['n'])}",
        "multivar_nested.csv in_decision_rule==True (n_folds_selected_with_disc)",
        f"selected: {prim['selected_models']}")
    add("A3_primary_folds_selected_model_with_gc_mean",
        f"{int(prim['n_folds_selected_with_gc_mean'])}/{int(prim['n'])}",
        "multivar_nested.csv in_decision_rule==True (n_folds_selected_with_gc_mean)")
    add("A4b_primary_incremental_perm_p", round(pb["p"], 4), "multivar_permutation.csv",
        "A4b = incremental test: disc-anchored family (46 models), rows of the other 9 Pool-A covariates permuted")
    pa = perm[(perm["config"] == "S11/Y1_MCC") & (perm["test"].str.startswith("A4a"))].iloc[0]
    add("A4a_primary_global_perm_p", round(pa["p"], 4), "multivar_permutation.csv",
        f"observed best LOTO R2 {pa['observed']:.3f}; null mean {pa['null_mean']:.3f}, 95th pct {pa['null_critical']:.3f}")

    for cfg in [f"{s}/{YNAME[y]}" for s, y, _ in CONFIGS]:
        L.append("")
        L.append(f"A2 best-subset OLS -- {cfg}")
        for fam in ("poolA", "disc_anchored", "poolB"):
            g = models[(models["config"] == cfg) & (models["family"] == fam)]
            ga = g.sort_values(["AICc", "p"]).head(5)
            gm = g.sort_values(["loto_MAE", "p"]).head(3)
            L.append(f"  [{fam}, {len(g)} models; disc-only LOTO MAE {g['disc_only_loto_MAE'].iloc[0]:.4f}, "
                     f"R2 {g['disc_only_loto_R2'].iloc[0]:+.3f}]")
            for _, r in ga.iterrows():
                L.append(f"     AICc #{int(r['rank_AICc'])}: {r['model']:<42} AICc {r['AICc']:7.2f} (d {r['dAICc']:.2f}, "
                         f"w {r['akaike_w']:.2f})  R2 {r['R2']:.3f} adj {r['adjR2']:.3f}  LOTO R2 {r['loto_R2']:+.3f} "
                         f"MAE {r['loto_MAE']:.4f}")
            for _, r in gm.iterrows():
                L.append(f"     LOTO-MAE #{int(r['rank_lotoMAE'])}: {r['model']:<38} MAE {r['loto_MAE']:.4f}  "
                         f"LOTO R2 {r['loto_R2']:+.3f}  AICc {r['AICc']:.2f}")
            if cfg == "S11/Y1_MCC":
                b1 = g.sort_values(["AICc", "p"]).iloc[0]
                add(f"A2_S11_MCC_{fam}_top_AICc_model", b1["model"], "multivar_models.csv",
                    f"AICc {b1['AICc']:.2f}, R2 {b1['R2']:.3f}, LOTO R2 {b1['loto_R2']:+.3f}, LOTO MAE {b1['loto_MAE']:.4f}")
                b2 = g.sort_values(["loto_MAE", "p"]).iloc[0]
                add(f"A2_S11_MCC_{fam}_top_lotoMAE_model", b2["model"], "multivar_models.csv",
                    f"LOTO MAE {b2['loto_MAE']:.4f}, LOTO R2 {b2['loto_R2']:+.3f}, AICc {b2['AICc']:.2f}")
        g = nested[nested["config"] == cfg]
        for _, r in g.iterrows():
            L.append(f"  A3 nested LOTO [{r['family']}, select={r['selection']}]{' (DECISION)' if r['in_decision_rule'] else ''}: "
                     f"MAE {r['nested_loto_MAE']:.4f} vs disc-only {r['disc_only_loto_MAE']:.4f}  dMAE {r['dMAE']:+.4f} "
                     f"[{r['dMAE_lo']:+.4f},{r['dMAE_hi']:+.4f}]  R2 {r['nested_loto_R2']:+.3f}; "
                     f"{r['n_distinct_selected']} distinct models selected across folds (disc in "
                     f"{r['n_folds_selected_with_disc']}/{r['n']}, gc_mean in {r['n_folds_selected_with_gc_mean']}/{r['n']})")
        g = perm[perm["config"] == cfg]
        for _, r in g.iterrows():
            L.append(f"  {r['test']}: observed {r['observed']:.4f}  null mean {r['null_mean']:.4f}  "
                     f"null critical {r['null_critical']:.4f}  p = {r['p']:.4f}")

    L.append("")
    L.append("A5 LMG / Shapley R^2 decomposition (in-sample) of the top-AICc models")
    for (cfg, fam, mdl), g in lmgd.groupby(["config", "family", "model"], sort=False):
        L.append(f"  {cfg} [{fam}] {mdl}  (R2 {g['R2_model'].iloc[0]:.3f})")
        for _, r in g.iterrows():
            L.append(f"      {r['predictor']:<11} LMG R2 {r['lmg_R2']:.3f} [{r['boot_lo']:.3f},{r['boot_hi']:.3f}] "
                     f"share {r['lmg_share']:.2f}")

    # ------------------------------------------------------------------ A6 residual attribution
    res_rows, rc_rows = [], []
    for sub, yc in (("S11", "bench_mcc"), ("S15", "bench_mcc")):
        cfg = f"{sub}/{YNAME[yc]}"
        dd = subsets_of[sub]
        yv = dd[yc].to_numpy(float)
        xd = dd[["disc"]].to_numpy(float)
        prd = loto_pred(xd, yv)
        rd = yv - prd
        rmse = float(np.sqrt(np.mean(rd ** 2)))
        others = [c for c in POOL_A if c != "disc"] + [COMP[yc]]
        red = {}
        for c in others:
            prc = loto_pred(np.column_stack([xd[:, 0], dd[c].to_numpy(float)]), yv)
            red[c] = np.abs(rd) - np.abs(yv - prc)
            rho, p = stats.spearmanr(rd, dd[c].to_numpy(float))
            rc_rows.append(dict(config=cfg, covariate=c, spearman_resid_c=rho, p=p, n=len(yv),
                                loto_MAE_disc_plus_c=np.mean(np.abs(yv - prc)),
                                loto_MAE_disc_only=np.mean(np.abs(rd))))
        for i, t in enumerate(dd["task"]):
            flag = abs(rd[i]) > rmse
            best_c = max(others, key=lambda c: red[c][i])
            best_red = red[best_c][i]
            descr = [c for c in POOL_A if c != "disc"]
            best_d = max(descr, key=lambda c: red[c][i])
            res_rows.append(dict(
                config=cfg, task=t, disc=xd[i, 0], actual=yv[i], disc_only_loto_pred=prd[i], resid=rd[i],
                abs_resid=abs(rd[i]), loto_rmse=rmse, flagged=flag, best_covariate=best_c,
                abs_resid_reduction=best_red, frac_reduction=best_red / abs(rd[i]),
                explained=bool(flag and best_red >= 0.5 * abs(rd[i])),
                best_descriptive_covariate=best_d, descr_frac_reduction=red[best_d][i] / abs(rd[i]),
                **{f"red_{c}": red[c][i] for c in others}))
    resd = pd.DataFrame(res_rows)
    resd.to_csv(os.path.join(RD, "multivar_residuals.csv"), index=False)
    rcd = pd.DataFrame(rc_rows)
    rcd.to_csv(os.path.join(RD, "multivar_residual_corr.csv"), index=False)
    L.append("")
    L.append("A6 residual attribution: disc-only LOTO residuals; flagged = |resid| > LOTO RMSE; for each flagged task,")
    L.append("   the covariate c whose LOTO model (disc + c) shrinks that task's |resid| most ('explained' if >= 50%)")
    for cfg, g in resd.groupby("config", sort=False):
        L.append(f"  {cfg} (LOTO RMSE {g['loto_rmse'].iloc[0]:.4f})")
        for _, r in g.sort_values("abs_resid", ascending=False).iterrows():
            tag = "FLAG" if r["flagged"] else "    "
            L.append(f"    {tag} {r['task']:<32} actual {r['actual']:.3f} pred {r['disc_only_loto_pred']:.3f} "
                     f"resid {r['resid']:+.3f}  best c = {r['best_covariate']:<11} ({100 * r['frac_reduction']:+.0f}% |resid|)"
                     f"{'  EXPLAINED' if r['explained'] else ('  not explained by any measured covariate' if r['flagged'] else '')}"
                     f"   [best descriptive: {r['best_descriptive_covariate']} {100 * r['descr_frac_reduction']:+.0f}%]")
        gg = rcd[rcd["config"] == cfg].sort_values("p")
        L.append("    Spearman(disc-only LOTO residual, c): " + "; ".join(
            f"{r['covariate']} {r['spearman_resid_c']:+.2f} (p={r['p']:.2g})" for _, r in gg.iterrows()))
        for _, r in g[g["flagged"]].iterrows():
            add(f"A6_{cfg}_{r['task']}", f"resid {r['resid']:+.3f}; best {r['best_covariate']} "
                f"{100 * r['frac_reduction']:+.0f}%; {'explained' if r['explained'] else 'not explained'}",
                "multivar_residuals.csv")

    # ------------------------------------------------------------------ A7 noise ceiling
    L.append("")
    nrows = []
    for sub in ("S11", "S15"):
        dd = subsets_of[sub]
        se = (dd["mcc_hi"] - dd["mcc_lo"]).to_numpy(float) / 3.92
        mse = float(np.mean(disc_only_err[f"{sub}/Y1_MCC"] ** 2))
        share = float(np.mean(se ** 2)) / mse
        nrows.append(dict(config=f"{sub}/Y1_MCC", mean_test_se=float(np.mean(se)), mean_se2=float(np.mean(se ** 2)),
                          disc_only_loto_mse=mse, noise_share=share))
        L.append(f"A7 noise ceiling {sub}: mean test-set SE of MCC {np.mean(se):.4f}; mean(se^2)/disc-only LOTO MSE = "
                 f"{share:.3f} -> test-set sampling noise accounts for {100 * share:.1f}% of the misfit")
        add(f"A7_{sub}_noise_share_of_disc_only_misfit", round(share, 4), "multivar_noise.csv")
    pd.DataFrame(nrows).to_csv(os.path.join(RD, "multivar_noise.csv"), index=False)

    # ------------------------------------------------------------------ A8 reviewer-specified model
    L.append("")
    rv_rows = []
    rv = ["disc", "len_log", "gc_sd", "homology"]
    for sub in ("S11", "S15"):
        dd = subsets_of[sub]
        yv = dd["bench_mcc"].to_numpy(float)
        Xr = dd[rv].to_numpy(float)
        nn = len(yv)
        _, _, rss = fit(Xr, yv)
        r2i = 1 - rss / float(((yv - yv.mean()) ** 2).sum())
        lr2, lmae = loto_metrics(yv, loto_pred(Xr, yv))
        det = model_detail(Xr, yv, rv)
        for dr in det:
            rv_rows.append(dict(config=f"{sub}/Y1_MCC", n=nn, p=4, resid_df=nn - 5, AICc=aicc(rss, nn, 4),
                                R2=r2i, adjR2=1 - (1 - r2i) * (nn - 1) / (nn - 5), loto_R2=lr2, loto_MAE=lmae,
                                over_parameterized=bool(nn / 4 < 3), **dr))
        L.append(f"A8 reviewer model {sub}: MCC ~ disc + len_log + gc_sd + homology  (n={nn}, resid df {nn - 5}"
                 f"{', OVER-PARAMETERIZED n/p<3' if nn / 4 < 3 else ''}): R2 {r2i:.3f} adj {1 - (1 - r2i) * (nn - 1) / (nn - 5):.3f} "
                 f"LOTO R2 {lr2:+.3f} MAE {lmae:.4f}")
        for dr in det:
            L.append(f"      {dr['predictor']:<9} std beta {dr['std_beta']:+.3f}  t-CI [{dr['t_lo']:+.3f},{dr['t_hi']:+.3f}] "
                     f"p={dr['t_p']:.2g}  boot [{dr['boot_lo']:+.3f},{dr['boot_hi']:+.3f}]  partial r {dr['partial_r']:+.3f} "
                     f"VIF {dr['vif']:.2f}")
        add(f"A8_{sub}_reviewer_model_loto_MAE", round(lmae, 4), "multivar_reviewer_model.csv")
        add(f"A8_{sub}_reviewer_model_loto_R2", round(lr2, 4), "multivar_reviewer_model.csv")
        tsig = [f"{d_['predictor']} {d_['std_beta']:+.3f} t-CI [{d_['t_lo']:+.3f},{d_['t_hi']:+.3f}] p={d_['t_p']:.3f}"
                for d_ in det if d_["t_lo"] > 0 or d_["t_hi"] < 0]
        bsig = [d_["predictor"] for d_ in det if d_["boot_excludes_0"]]
        L.append(f"      -> coefficients whose t-CI excludes 0: {'; '.join(tsig) if tsig else 'none'}; "
                 f"whose task-bootstrap CI excludes 0: {', '.join(bsig) if bsig else 'none'}")
        add(f"A8_{sub}_reviewer_model_t_significant", "; ".join(tsig) if tsig else "none", "multivar_reviewer_model.csv")
        add(f"A8_{sub}_reviewer_model_boot_significant", ", ".join(bsig) if bsig else "none",
            "multivar_reviewer_model.csv")
        add(f"A8_{sub}_reviewer_model_in_sample_R2_adjR2", f"{r2i:.3f} / {1 - (1 - r2i) * (nn - 1) / (nn - 5):.3f}",
            "multivar_reviewer_model.csv (R2, adjR2)")
    pd.DataFrame(rv_rows).to_csv(os.path.join(RD, "multivar_reviewer_model.csv"), index=False)

    # ------------------------------------------------------------------ A9 synthetic-law transfer
    L.append("")
    dose = pd.read_csv(os.path.join(RD, "disc_dose.csv"))
    xs, ys = dose["motif_auroc_d"].to_numpy(float), dose["bench_mcc"].to_numpy(float)
    bsyn = np.polyfit(xs, ys, 1)
    tr_rows = []
    for sub in ("S11", "S15"):
        dd = subsets_of[sub]
        yv = dd["bench_mcc"].to_numpy(float)
        ps = np.polyval(bsyn, dd["disc"].to_numpy(float))
        r = yv - ps
        r2s_ = 1 - float(r @ r) / float(((yv - yv.mean()) ** 2).sum())
        for t, dv, a, p_ in zip(dd["task"], dd["disc"], yv, ps):
            tr_rows.append(dict(config=f"{sub}/Y1_MCC", task=t, disc=dv, actual=a, synthetic_law_pred=p_,
                                resid=a - p_, slope=bsyn[0], intercept=bsyn[1]))
        L.append(f"A9 synthetic-law transfer {sub} (pooled synthetic OLS MCC = {bsyn[0]:+.3f} d {bsyn[1]:+.3f}, n={len(xs)} "
                 f"synthetic points; 0 parameters fitted on real tasks): MAE {np.mean(np.abs(r)):.4f}  R2 {r2s_:+.3f}  "
                 f"mean signed resid {np.mean(r):+.4f}")
        add(f"A9_{sub}_synthetic_law_MAE", round(float(np.mean(np.abs(r))), 4), "multivar_synthetic_transfer.csv")
        add(f"A9_{sub}_synthetic_law_mean_signed_resid", round(float(np.mean(r)), 4), "multivar_synthetic_transfer.csv")
    pd.DataFrame(tr_rows).to_csv(os.path.join(RD, "multivar_synthetic_transfer.csv"), index=False)

    # ================================================================== POST-HOC (not pre-registered)
    # Committed 2026-09-24 after the mock-data smoke test and BEFORE the real covariate table existed:
    #  P1  A6 calibration: a random covariate can shrink one task's LOTO residual by >50% at n=11, so
    #      each flagged task's best reduction is compared with a permutation null (rows of the Pool-B
    #      covariate block permuted jointly across tasks, disc and Y fixed; max over covariates).
    #  P2  Synthetic-law transfer in SLOPE vs LEVEL: real-task OLS slope (task-bootstrap CI) vs the
    #      pooled synthetic slope; LOTO with the slope FIXED at the synthetic value (1 fitted parameter,
    #      the intercept, per fold).
    #  P3  Alternative homology measure: MMseqs2 both-strand id>=0.8 / cov>=0.8 fraction of test
    #      sequences in a train-containing cluster (homology_recluster_redundancy.csv, threshold id80;
    #      another lane's output, values copied into multivar_posthoc_homology_alt.csv), substituted
    #      for the 8-mer-Jaccard homology in A1 (partial Spearman) and A8 (reviewer model).
    L.append("")
    L.append("=" * 110)
    L.append("POST-HOC analyses (not pre-registered; see run_multivar.py comments)")
    L.append("=" * 110)
    # ---- P1
    p1 = []
    for sub in ("S11", "S15"):
        cfg = f"{sub}/Y1_MCC"
        dd = subsets_of[sub]
        yv = dd["bench_mcc"].to_numpy(float)
        xd = dd["disc"].to_numpy(float)
        others = [c for c in POOL_A if c != "disc"] + ["comp_mcc"]
        Xo = dd[others].to_numpy(float)
        rd = yv - loto_pred(xd[:, None], yv)
        g = resd[resd["config"] == cfg].set_index("task")
        flagged = [i for i, t in enumerate(dd["task"]) if g.loc[t, "flagged"]]
        rng = np.random.RandomState(SEED)
        null = np.empty((NPERM, len(flagged)))
        null_d = np.empty((NPERM, len(flagged)))
        for b in range(NPERM):
            Xp = Xo[rng.permutation(len(yv))]
            reds = np.empty((len(others), len(flagged)))
            for j in range(len(others)):
                prc = loto_pred(np.column_stack([xd, Xp[:, j]]), yv)
                reds[j] = (np.abs(rd) - np.abs(yv - prc))[flagged] / np.abs(rd[flagged])
            null[b] = reds.max(axis=0)
            null_d[b] = reds[:-1].max(axis=0)          # descriptive covariates only (no comp)
        for k, i in enumerate(flagged):
            t = dd["task"].iloc[i]
            obs = g.loc[t, "frac_reduction"]
            obs_d = g.loc[t, "descr_frac_reduction"]
            p1.append(dict(config=cfg, task=t, resid=rd[i], best_covariate=g.loc[t, "best_covariate"],
                           frac_reduction=obs, null_mean_max=null[:, k].mean(),
                           null_q95_max=np.percentile(null[:, k], 95),
                           p_max=(1 + np.sum(null[:, k] >= obs)) / (NPERM + 1),
                           null_frac_ge_50pct=float(np.mean(null[:, k] >= 0.5)),
                           best_descriptive=g.loc[t, "best_descriptive_covariate"], descr_frac_reduction=obs_d,
                           p_max_descriptive=(1 + np.sum(null_d[:, k] >= obs_d)) / (NPERM + 1)))
    p1 = pd.DataFrame(p1)
    p1.to_csv(os.path.join(RD, "multivar_posthoc_residual_null.csv"), index=False)
    L.append("P1 residual-attribution null (max over the 10 Pool-B covariates of the LOTO |resid| reduction, rows permuted):")
    for _, r in p1.iterrows():
        L.append(f"    {r['config']} {r['task']:<30} best {r['best_covariate']:<10} {100 * r['frac_reduction']:+.0f}%  "
                 f"null P(max>=50%) {r['null_frac_ge_50pct']:.2f}  p_max {r['p_max']:.3f} | best descriptive "
                 f"{r['best_descriptive']} {100 * r['descr_frac_reduction']:+.0f}% p {r['p_max_descriptive']:.3f}")
        add(f"P1_{r['config']}_{r['task']}_p_max", round(r["p_max"], 4), "multivar_posthoc_residual_null.csv",
            f"best {r['best_covariate']} {100 * r['frac_reduction']:+.0f}%; null P(max>=50%) {r['null_frac_ge_50pct']:.2f}")
    # ---- P2
    p2 = []
    syn_slope, syn_int = bsyn
    for sub in ("S11", "S15"):
        dd = subsets_of[sub]
        yv = dd["bench_mcc"].to_numpy(float)
        xd = dd["disc"].to_numpy(float)
        nn = len(yv)
        sl, ic = np.polyfit(xd, yv, 1)
        res_ = stats.linregress(xd, yv)
        tcrit = stats.t.ppf(0.975, nn - 2)
        rng = np.random.RandomState(SEED)
        bsl = []
        for _ in range(B):
            idx = rng.randint(0, nn, nn)
            if xd[idx].std() < 1e-12:
                continue
            bsl.append(np.polyfit(xd[idx], yv[idx], 1)[0])
        lo, hi = np.percentile(bsl, [2.5, 97.5])
        # LOTO with the synthetic slope fixed: intercept = mean(y - s*d) over the n-1 training tasks
        off = yv - syn_slope * xd
        pr_fix = np.array([syn_slope * xd[i] + np.mean(np.delete(off, i)) for i in range(nn)])
        r2f, maef = loto_metrics(yv, pr_fix)
        r2d, maed = loto_metrics(yv, loto_pred(xd[:, None], yv))
        lo_d, hi_d = paired_boot_dmae(yv - pr_fix, yv - loto_pred(xd[:, None], yv))
        p2.append(dict(config=f"{sub}/Y1_MCC", n=nn, real_slope=sl, real_slope_t_lo=sl - tcrit * res_.stderr,
                       real_slope_t_hi=sl + tcrit * res_.stderr, real_slope_boot_lo=lo, real_slope_boot_hi=hi,
                       synthetic_slope=syn_slope, synthetic_in_boot_ci=bool(lo <= syn_slope <= hi),
                       real_intercept=ic, synthetic_intercept=syn_int,
                       level_offset_at_mean_d=float(np.mean(yv - np.polyval(bsyn, xd))),
                       slope_fixed_loto_R2=r2f, slope_fixed_loto_MAE=maef, free_loto_R2=r2d, free_loto_MAE=maed,
                       dMAE_fixed_minus_free=maef - maed, dMAE_lo=lo_d, dMAE_hi=hi_d))
        L.append(f"P2 {sub}: real slope {sl:+.3f} (t-CI [{sl - tcrit * res_.stderr:+.3f},{sl + tcrit * res_.stderr:+.3f}], "
                 f"boot [{lo:+.3f},{hi:+.3f}]) vs synthetic {syn_slope:+.3f} -> synthetic slope "
                 f"{'INSIDE' if lo <= syn_slope <= hi else 'OUTSIDE'} the real-task CI; level offset (real - synthetic "
                 f"line) {np.mean(yv - np.polyval(bsyn, xd)):+.3f} MCC.  LOTO with slope fixed at synthetic: "
                 f"MAE {maef:.4f} R2 {r2f:+.3f}  vs free slope MAE {maed:.4f} R2 {r2d:+.3f} "
                 f"(dMAE {maef - maed:+.4f} [{lo_d:+.4f},{hi_d:+.4f}])")
        add(f"P2_{sub}_real_slope", f"{sl:+.3f} boot [{lo:+.3f},{hi:+.3f}]", "multivar_posthoc_slope_transfer.csv",
            f"synthetic pooled slope {syn_slope:+.3f}")
        add(f"P2_{sub}_slope_fixed_loto_MAE", round(maef, 4), "multivar_posthoc_slope_transfer.csv",
            f"free-slope LOTO MAE {maed:.4f}")
    pd.DataFrame(p2).to_csv(os.path.join(RD, "multivar_posthoc_slope_transfer.csv"), index=False)
    # ---- P3
    alt_fp = os.path.join(RD, "homology_recluster_redundancy.csv")
    if os.path.exists(alt_fp):
        alt = pd.read_csv(alt_fp)
        alt = alt[alt["threshold"] == "id80"].set_index("task")
        rows3 = []
        for t in cov["task"]:
            rows3.append(dict(task=t, homology_jaccard07=cov.set_index("task").loc[t, "homology"],
                              homology_mm80=alt.loc[t, "frac_test_in_train_cluster"] if t in alt.index else np.nan,
                              source=f"homology_recluster_redundancy.csv (id80) mtime "
                                     f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(alt_fp)))}"))
        h3 = pd.DataFrame(rows3)
        h3.to_csv(os.path.join(RD, "multivar_posthoc_homology_alt.csv"), index=False)
        if h3["homology_mm80"].notna().all():
            covh = cov.merge(h3[["task", "homology_mm80"]], on="task")
            rs, ps_ = stats.spearmanr(h3["homology_jaccard07"], h3["homology_mm80"])
            L.append(f"P3 alternative homology (MMseqs2 id80, both strands) vs 8-mer Jaccard>=0.7: Spearman {rs:+.3f} "
                     f"(p={ps_:.2g}, n=15)")
            p3 = []
            for sub in ("S11", "S15"):
                dd = covh[covh["S11"]].reset_index(drop=True) if sub == "S11" else covh
                yv = dd["bench_mcc"].to_numpy(float)
                xd = dd["disc"].to_numpy(float)
                zc = dd["homology_mm80"].to_numpy(float)
                pr_ = partial_spearman(xd, yv, zc)
                rng = np.random.RandomState(SEED)
                bs = []
                for _ in range(B):
                    idx = rng.randint(0, len(yv), len(yv))
                    if min(xd[idx].std(), yv[idx].std(), zc[idx].std()) < 1e-12:
                        continue
                    v = partial_spearman(xd[idx], yv[idx], zc[idx])
                    if not np.isnan(v):
                        bs.append(v)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                rvh = ["disc", "len_log", "gc_sd", "homology_mm80"]
                Xr = dd[rvh].to_numpy(float)
                lr2, lmae = loto_metrics(yv, loto_pred(Xr, yv))
                det = model_detail(Xr, yv, rvh)
                for dr in det:
                    p3.append(dict(config=f"{sub}/Y1_MCC", partial_rho_disc_given_mm80=pr_, partial_boot_lo=lo,
                                   partial_boot_hi=hi, reviewer_model_mm80_loto_R2=lr2,
                                   reviewer_model_mm80_loto_MAE=lmae, **dr))
                L.append(f"P3 {sub}: partial Spearman(disc, MCC | homology_mm80) {pr_:+.3f} [{lo:+.3f},{hi:+.3f}];  "
                         f"reviewer model with homology_mm80: LOTO R2 {lr2:+.3f} MAE {lmae:.4f}; "
                         + "; ".join(f"{d_['predictor']} {d_['std_beta']:+.2f} [{d_['boot_lo']:+.2f},{d_['boot_hi']:+.2f}]"
                                     for d_ in det))
            pd.DataFrame(p3).to_csv(os.path.join(RD, "multivar_posthoc_homology_models.csv"), index=False)
    else:
        L.append("P3 skipped: homology_recluster_redundancy.csv not found")

    # ---- P4 (added AFTER the first full run, motivated by the A6 residual of nt_enhancers_types; RESULTS-
    #      MOTIVATED, reported as a diagnosis of that deviation, never as a replacement headline).
    #      cross_suite_summary's disc for the 3-class nt_enhancers_types is the max positive-direction
    #      motif AUROC of the '0_vs_rest' contrast, whose 'positive' class 0 is the NON-enhancer class
    #      (verified: its labels coincide with nt_enhancers label 0).  That value (TBP 0.675) is the
    #      TATA-in-negatives Route-3 contamination axis (cf. nt_enhancers TBP neg-direction 0.680), i.e.
    #      opposite in orientation to every binary task, where disc = motif enriched in the functional
    #      class.  The orientation-consistent value = best motif enriched in the enhancer classes
    #      (0_vs_rest, max of 1 - motif_auroc) from results/nt/motif_discriminability.csv.
    md = pd.read_csv(os.path.join(_mv.ROOT, "results", "nt", "motif_discriminability.csv"))
    g4 = md[(md["dataset"] == "nt_enhancers_types") & (md["contrast"] == "0_vs_rest")]
    j4 = (1 - g4["motif_auroc"]).idxmax()
    d_fix, tf_fix = float(1 - g4.loc[j4, "motif_auroc"]), g4.loc[j4, "TF"]
    p4 = []
    for sub in ("S11", "S15"):
        dd = subsets_of[sub].copy()
        yv = dd["bench_mcc"].to_numpy(float)
        for variant in ("as_published", "orientation_consistent"):
            if variant == "orientation_consistent":
                dd.loc[dd["task"] == "nt_enhancers_types", "disc"] = d_fix
            xd = dd["disc"].to_numpy(float)
            prv = loto_pred(xd[:, None], yv)
            r2v, maev = loto_metrics(yv, prv)
            rng = np.random.RandomState(SEED)
            r2b, maeb = [], []
            for _ in range(1000):
                idx = rng.randint(0, len(yv), len(yv))
                rr = yv[idx] - prv[idx]
                st = np.sum((yv[idx] - yv[idx].mean()) ** 2)
                if st > 0:
                    r2b.append(1 - np.sum(rr ** 2) / st)
                    maeb.append(np.mean(np.abs(rr)))
            rho, prho = stats.spearmanr(xd, yv)
            k = int(np.where(dd["task"] == "nt_enhancers_types")[0][0])
            p4.append(dict(config=f"{sub}/Y1_MCC", variant=variant,
                           disc_nt_enhancers_types=xd[k],
                           disc_source=("cross_suite_summary max_pos_motif_auroc (0_vs_rest, class-0 = non-enhancer "
                                        "direction, TBP)" if variant == "as_published" else
                                        f"results/nt/motif_discriminability.csv 0_vs_rest, max(1-motif_auroc) = {tf_fix}"),
                           spearman=rho, spearman_p=prho, loto_R2=r2v, loto_R2_lo=np.percentile(r2b, 2.5),
                           loto_R2_hi=np.percentile(r2b, 97.5), loto_MAE=maev, loto_MAE_lo=np.percentile(maeb, 2.5),
                           loto_MAE_hi=np.percentile(maeb, 97.5), resid_nt_enhancers_types=yv[k] - prv[k]))
    p4 = pd.DataFrame(p4)
    p4.to_csv(os.path.join(RD, "multivar_posthoc_orientation.csv"), index=False)
    L.append(f"P4 (results-motivated) nt_enhancers_types disc re-read in the functional-class orientation: "
             f"{tf_fix} {d_fix:.4f} instead of TBP (non-enhancer direction) "
             f"{p4['disc_nt_enhancers_types'].iloc[0]:.4f}")
    for _, r in p4.iterrows():
        L.append(f"    {r['config']} {r['variant']:<23} Spearman {r['spearman']:+.3f} (p={r['spearman_p']:.2g})  LOTO R2 "
                 f"{r['loto_R2']:+.3f} [{r['loto_R2_lo']:+.3f},{r['loto_R2_hi']:+.3f}]  MAE {r['loto_MAE']:.4f} "
                 f"[{r['loto_MAE_lo']:.3f},{r['loto_MAE_hi']:.3f}]  nt_enhancers_types resid {r['resid_nt_enhancers_types']:+.3f}")
        add(f"P4_{r['config']}_{r['variant']}_loto", f"R2 {r['loto_R2']:+.3f} [{r['loto_R2_lo']:+.3f},{r['loto_R2_hi']:+.3f}]; "
            f"MAE {r['loto_MAE']:.4f} [{r['loto_MAE_lo']:.3f},{r['loto_MAE_hi']:.3f}]; Spearman {r['spearman']:+.3f}",
            "multivar_posthoc_orientation.csv")

    # ================================================================== POST-HOC round 2 (verification)
    # Added 2026-09-24 after two independent verifiers reviewed the first report, BEFORE these blocks were
    # run.  All are RESULTS-MOTIVATED and labelled post hoc; none changes the pre-registered verdict.
    #  P5  "Homology-clean" subset re-check.  The paper's 9-task homology-clean subset drops only
    #      human_nontata_promoters and nt_enhancers.  This module's homology covariate flags two more S11
    #      tasks as LEAKY (>= 0.05, the pre-existing threshold of homology/measure_homology.verdict):
    #      human_enhancers_ensembl and nt_enhancers_types.  Spearman, leave-one-out jackknife, partial
    #      given suite and disc-only LOTO are recomputed on: S11, the paper's 9, the paper's 9 minus
    #      human_enhancers_ensembl, and the rule-based clean set (all S11 tasks with homology < 0.05).
    #      Verbatim-duplicate fractions come from audit_exact_dups.py.
    #  P6  Readout of dedup_eval.py (Table-1 model scored on near-duplicate-free vs near-duplicate test
    #      sequences for the four LEAKY S11 tasks).
    #  P7  Discriminability vs GC mean, head to head: GC-only vs disc-only LOTO (paired dMAE), both
    #      partial Spearmans (disc | GC and GC | disc), LMG shares of disc + GC mean; as published and
    #      with the P4 enhancer-oriented disc.
    #  P8  Robustness of P4 and a UNIFORM rule for multiclass tasks.  Every alternative enhancer-oriented
    #      reading of nt_enhancers_types (0_vs_rest reversed, 1_vs_rest, 2_vs_rest), plus two rules
    #      applied to all three multiclass tasks: U1 keep the representative contrast when its positive
    #      class is functional, otherwise take the best functional-class contrast; U2 max over the
    #      functional-class contrasts.  Functional classes: human_ensembl_regulatory all three
    #      (enhancer / OCR / promoter); nt_enhancers_types classes 1, 2 (class 0 = the non-enhancer
    #      class); nt_splice_sites_all classes 0 (acceptor) and 1 (donor), class 2 = neither -- both
    #      read from the sequences in multivar_posthoc_class_identity.csv.
    #  P9  Bounded-outcome sensitivity for the MCC ceiling: disc-only LOTO predictions capped at 1, and
    #      a Fisher-z model (OLS on atanh MCC, tanh back-transform).
    #  P10 Synthetic range restriction: the synthetic dose-response points inside the real S11 disc
    #      range (fit tightness vs the real tasks), and what the real-minus-synthetic level offset
    #      correlates with (composition-only MCC, GC mean, n_train, length).
    #  FIX ROUND 2 (2026-09-24 late evening, after the second verification round; all post hoc):
    #      P5 gains permutation p-values (exact over all n! orderings at n <= 9) and an explicit
    #      homology_clean flag; degenerate |rho| = 1 partials get p = NaN instead of 0 (partial_boot);
    #      A1 gains Holm-adjusted p and the reverse partial's valid-resample count; P7 writes per-task
    #      nested vs GC-only predictions; P11 records that the family indicator equals Table I's
    #      'solved' split; P12 (below) recomputes the headline statistics on task sets with shared
    #      sequences collapsed (audit_task_overlap.py).
    #  FIX ROUND 3 (2026-09-24 night, after the third verification round; all post hoc, descriptive):
    #      Spearman permutation p is exact for every n <= 12 (dynamic programming over all n! orderings);
    #      degenerate |rho| = 1 partials get no percentile CI (NaN) as well as no p; P5 and P12 report GC mean's
    #      own Spearman with MCC next to disc's (exact p), the paired task-bootstrap CI of their difference,
    #      which of disc-only / GC-only has the lower LOTO MAE, the single tasks whose removal reverses that
    #      ordering, and both MAEs without the 3-class regulatory task; P5 adds all 330 seven-task subsets of
    #      S11; P12 is renamed to exact-duplicate terminology ('dupfree' = exact duplicates / unions collapsed;
    #      'noshare' = no exact sequence sharing >= 0.1%) and gains two regulatory-removed rows; n_distinct_sets
    #      no longer miscounts a set holding nt_promoter_all with only one of its components; P13 (below)
    #      recomputes the correlations on sets of tasks that share no LOCI (audit_locus_overlap.py).
    L.append("")
    L.append("-" * 110)
    L.append("POST-HOC round 2 (added after independent verification; results-motivated)")
    L.append("-" * 110)
    d11 = subsets_of["S11"]
    # ---- P5
    exd_fp = os.path.join(RD, "multivar_posthoc_exact_dup.csv")
    exd = pd.read_csv(exd_fp).set_index("task") if os.path.exists(exd_fp) else None
    leaky = [t for t, h in zip(d11["task"], d11["homology"]) if h >= 0.05]
    p5_defs = [("S11_all", []),
               ("paper_homology_clean_9", ["human_nontata_promoters", "nt_enhancers"]),
               ("paper9_minus_human_enhancers_ensembl",
                ["human_nontata_promoters", "nt_enhancers", "human_enhancers_ensembl"]),
               ("rule_clean_homology_lt_0.05", leaky)]
    p5 = []
    for name, drop in p5_defs:
        dd = d11[~d11["task"].isin(drop)].reset_index(drop=True)
        xd, yv = dd["disc"].to_numpy(float), dd["bench_mcc"].to_numpy(float)
        su = (dd["suite"] == "NucTransformer").to_numpy(float)
        nn = len(yv)
        rho, prho = spearman_t(xd, yv)
        # jackknife: (rho, p) per leave-one-out; a degenerate |rho| = 1 uses the exact permutation p
        jk = []
        for i in range(nn):
            a_, b_ = np.delete(xd, i), np.delete(yv, i)
            r_, p_ = spearman_t(a_, b_)
            jk.append((r_, perm_spearman_p(a_, b_)[0] if np.isnan(p_) else p_))
        r2l, r2lo, r2hi, mael, maelo, maehi = loto_boot_ci(yv, loto_pred(xd[:, None], yv))
        kept_leaky = [t for t in dd["task"] if t in leaky]
        gcv = dd["gc_mean"].to_numpy(float)
        pdg5, pgd5 = partial_boot(xd, yv, gcv, return_n=True), partial_boot(gcv, yv, xd, return_n=True)
        # fix round 2: permutation p (exact over all n! orderings at n <= 9) next to scipy's t-approximation,
        # and an explicit homology_clean flag (no kept task >= 0.05) so no text can call a leaky set clean
        sp_perm, sp_meth = perm_spearman_p(xd, yv)
        ppd5, ppg5 = perm_partial_p(xd, yv, gcv)[0], perm_partial_p(gcv, yv, xd)[0]
        p5.append(dict(subset=name, n=nn, dropped=";".join(drop), kept_homology_ge_0_05=";".join(kept_leaky),
                       homology_clean=bool(len(kept_leaky) == 0),
                       n_exact_distinct_sets=n_distinct_sets(dd["task"])[0],
                       n_no_exact_sharing=n_distinct_sets(dd["task"])[1],
                       spearman_p_perm=sp_perm, spearman_p_perm_method=sp_meth,
                       **gc_disc_readout(dd["task"], xd, gcv, yv),
                       jackknife_degenerate=bool(min(j[0] for j in jk) > 1 - 1e-12),
                       partial_given_gc_mean_p_perm=ppd5, partial_gc_given_disc_p_perm=ppg5,
                       max_homology_kept=float(dd["homology"].max()),
                       max_verbatim_frac_kept=(float(exd.loc[dd["task"], "frac_exact_any"].max())
                                               if exd is not None else np.nan),
                       spearman=rho, spearman_p=prho, jackknife_min=min(j[0] for j in jk),
                       jackknife_max=max(j[0] for j in jk), jackknife_n_p_lt_0_05=sum(j[1] < 0.05 for j in jk),
                       partial_given_suite=partial_spearman(xd, yv, su),
                       spearman_disc_gc_mean=stats.spearmanr(xd, dd["gc_mean"].to_numpy(float))[0],
                       partial_given_gc_mean=pdg5[0], partial_given_gc_mean_p=pdg5[1],
                       partial_given_gc_mean_lo=pdg5[2], partial_given_gc_mean_hi=pdg5[3],
                       partial_gc_given_disc=pgd5[0], partial_gc_given_disc_p=pgd5[1],
                       partial_gc_given_disc_lo=pgd5[2], partial_gc_given_disc_hi=pgd5[3],
                       partial_given_gc_mean_n_boot_ok=pdg5[4], partial_gc_given_disc_n_boot_ok=pgd5[4],
                       n_promoter_tasks=int(dd["task"].str.contains("promoter|regulatory").sum()),
                       loto_R2=r2l, loto_R2_lo=r2lo,
                       loto_R2_hi=r2hi, loto_MAE=mael, loto_MAE_lo=maelo, loto_MAE_hi=maehi))
    p5 = pd.DataFrame(p5)
    p5.to_csv(os.path.join(RD, "multivar_posthoc_homology_clean.csv"), index=False)
    # fix round 3 (post hoc, descriptive; claims verifier round 3): is rho = +1.00 on the homology-clean 7 an
    # artifact of dropping ANY four tasks?  Spearman(disc, MCC) and Spearman(GC, MCC) on all C(11, 7) = 330
    # seven-task subsets of S11.
    x11, y11, g11 = (d11[c].to_numpy(float) for c in ("disc", "bench_mcc", "gc_mean"))
    t11 = list(d11["task"])
    s7 = []
    for keep in itertools.combinations(range(len(t11)), 7):
        k_ = list(keep)
        s7.append(dict(dropped=";".join(t for i, t in enumerate(t11) if i not in keep),
                       homology_clean=bool((d11["homology"].to_numpy()[k_] < 0.05).all()),
                       spearman_disc_mcc=stats.spearmanr(x11[k_], y11[k_])[0],
                       spearman_gc_mcc=stats.spearmanr(g11[k_], y11[k_])[0]))
    s7 = pd.DataFrame(s7)
    s7.to_csv(os.path.join(RD, "multivar_posthoc_subsets7.csv"), index=False)
    at1 = s7[s7["spearman_disc_mcc"] > 1 - 1e-12]
    add("P5_subsets7_n_with_disc_rho_1", f"{len(at1)}/{len(s7)}", "multivar_posthoc_subsets7.csv",
        "7-task subsets of S11 with Spearman(disc, MCC) = +1.00; dropped: " + " | ".join(at1["dropped"]))
    add("P5_subsets7_median_disc_rho", round(float(s7["spearman_disc_mcc"].median()), 4), "multivar_posthoc_subsets7.csv",
        f"median over {len(s7)} subsets; GC median {s7['spearman_gc_mcc'].median():.4f}")
    L.append("P5 'homology-clean' re-check (S11, MCC ~ disc).  S11 tasks with homology >= 0.05 (LEAKY): "
             + ", ".join(f"{t} {h:.3f}" + (f" (verbatim {exd.loc[t, 'frac_exact_any']:.3f})" if exd is not None else "")
                         for t, h in zip(d11["task"], d11["homology"]) if h >= 0.05))
    for _, r in p5.iterrows():
        L.append(f"    {r['subset']:<38} n={r['n']:>2}  Spearman {r['spearman']:+.3f} (t-approx p={r['spearman_p']:.2g}; "
                 f"permutation p={r['spearman_p_perm']:.2g}, {r['spearman_p_perm_method']})  jackknife "
                 f"[{r['jackknife_min']:+.3f},{r['jackknife_max']:+.3f}] ({r['jackknife_n_p_lt_0_05']}/{r['n']} p<0.05)  "
                 f"partial|suite {r['partial_given_suite']:+.3f}  partial|GC mean {r['partial_given_gc_mean']:+.3f} "
                 f"[{r['partial_given_gc_mean_lo']:+.3f},{r['partial_given_gc_mean_hi']:+.3f}] p={r['partial_given_gc_mean_p']:.2g}  "
                 f"rho(GC,Y|disc) {r['partial_gc_given_disc']:+.3f} [{r['partial_gc_given_disc_lo']:+.3f},"
                 f"{r['partial_gc_given_disc_hi']:+.3f}] p={r['partial_gc_given_disc_p']:.2g} "
                 f"(Spearman disc~GC {r['spearman_disc_gc_mean']:+.3f}; Spearman GC~MCC {r['spearman_gc_mcc']:+.3f} "
                 f"(perm p={r['spearman_gc_mcc_p_perm']:.2g}); LOTO MAE disc {r['loto_MAE_disc']:.4f} / GC {r['loto_MAE_gc']:.4f}; "
                 f"{r['n_promoter_tasks']} promoter/regulatory tasks)  "
                 f"LOTO R2 {r['loto_R2']:+.3f} [{r['loto_R2_lo']:+.3f},"
                 f"{r['loto_R2_hi']:+.3f}] MAE {r['loto_MAE']:.4f}  still-LEAKY kept: {r['kept_homology_ge_0_05'] or 'none'}")
        add(f"P5_{r['subset']}", f"n={r['n']}; Spearman {r['spearman']:+.3f} (t-approx p={r['spearman_p']:.2g}; "
            f"permutation p={r['spearman_p_perm']:.2g}); homology_clean={r['homology_clean']}; jackknife "
            f"[{r['jackknife_min']:+.3f},{r['jackknife_max']:+.3f}]; partial|suite {r['partial_given_suite']:+.3f}; "
            f"partial|GC mean {r['partial_given_gc_mean']:+.3f} [{r['partial_given_gc_mean_lo']:+.3f},"
            f"{r['partial_given_gc_mean_hi']:+.3f}] p={r['partial_given_gc_mean_p']:.3f}; LOTO R2 {r['loto_R2']:+.3f} [{r['loto_R2_lo']:+.3f},"
            f"{r['loto_R2_hi']:+.3f}], MAE {r['loto_MAE']:.4f}; Spearman(GC mean, MCC) {r['spearman_gc_mcc']:+.3f} (perm p="
            f"{r['spearman_gc_mcc_p_perm']:.2g}); LOTO MAE GC-only {r['loto_MAE_gc']:.4f}", "multivar_posthoc_homology_clean.csv",
            f"LEAKY tasks kept: {r['kept_homology_ge_0_05'] or 'none'}")
    # ---- P6
    dd_fp = os.path.join(RD, "multivar_posthoc_dedup_eval.csv")
    if os.path.exists(dd_fp):
        de = pd.read_csv(dd_fp)
        L.append("P6 Table-1 model re-fit (val-selected k, models.LGBM_PARAMS, seed 42; cached predictions, full-test MCC "
                 "asserted == Table 1) scored on near-duplicate-free vs near-duplicate test sequences "
                 "(near-dup = max 8-mer Jaccard >= 0.7 or verbatim; full test split):")
        for t, g in de.groupby("task", sort=False):
            g = g.set_index("subset")
            a, c, n_ = g.loc["all"], g.loc["clean_j07"], g.loc["neardup_j07"]
            ncls = int(a["n_classes"])
            rec = lambda r: "/".join(f"{r[f'recall_class{k}']:.3f}" if r[f'recall_class{k}'] == r[f'recall_class{k}']
                                     else "-" for k in range(ncls))
            extra = (f"  AUROC(subset positives vs all negatives): clean {c['auroc_subset_pos_vs_all_neg']:.4f}, "
                     f"near-dup {n_['auroc_subset_pos_vs_all_neg']:.4f}" if ncls == 2 else "")
            L.append(f"    {t:<26} k={int(a['k'])} near-dup {a['frac_ge_0_7_full']:.3f} (verbatim {a['frac_verbatim']:.3f}) | "
                     f"all: MCC {a['mcc']:.4f} (Table 1 {a['table1_mcc']:.4f}) AUROC {a['auroc']:.4f} | clean (n={int(c['n'])}): "
                     f"MCC {c['mcc']:.4f}, prevalence-std MCC {c['mcc_prevalence_std']:.4f}, AUROC {c['auroc']:.4f} | "
                     f"recall per class all {rec(a)}, clean {rec(c)}, near-dup {rec(n_)}{extra}")
            add(f"P6_{t}", f"all MCC {a['mcc']:.4f} (Table 1 {a['table1_mcc']:.4f}); near-dup-free MCC {c['mcc']:.4f} "
                f"(prevalence-std {c['mcc_prevalence_std']:.4f}); AUROC {a['auroc']:.4f} -> {c['auroc']:.4f}; recall all "
                f"{rec(a)} clean {rec(c)} near-dup {rec(n_)}", "multivar_posthoc_dedup_eval.csv",
                f"near-dup {a['frac_ge_0_7_full']:.3f}, verbatim {a['frac_verbatim']:.3f} of test")
    else:
        L.append("P6 skipped: multivar_posthoc_dedup_eval.csv not found (run dedup_eval.py)")
    # ---- P7
    p7 = []
    for sub in ("S11", "S15"):
        for variant, dv in (("as_published", None), ("P4_orientation_consistent", d_fix)):
            dd = subsets_of[sub].copy()
            if dv is not None:
                dd.loc[dd["task"] == "nt_enhancers_types", "disc"] = dv
            yv, xd, gc = (dd[c].to_numpy(float) for c in ("bench_mcc", "disc", "gc_mean"))
            prd, prg = loto_pred(xd[:, None], yv), loto_pred(gc[:, None], yv)
            r2d, maed = loto_metrics(yv, prd)
            r2g, maeg = loto_metrics(yv, prg)
            lo, hi = paired_boot_dmae(yv - prg, yv - prd)
            pdg = partial_boot(xd, yv, gc)
            pgd = partial_boot(gc, yv, xd)
            X2 = np.column_stack([xd, gc])
            sh, r2full = lmg(X2, yv)
            rng = np.random.RandomState(SEED)
            bsh = []
            for _ in range(B):
                idx = rng.randint(0, len(yv), len(yv))
                if _degenerate(X2[idx]) or yv[idx].std() < 1e-12:
                    continue
                s_, r_ = lmg(X2[idx], yv[idx])
                if r_ > 0:
                    bsh.append(s_[0] / r_)
            p7.append(dict(config=f"{sub}/Y1_MCC", variant=variant, n=len(yv),
                           spearman_disc_gc=stats.spearmanr(xd, gc)[0],
                           disc_only_loto_R2=r2d, disc_only_loto_MAE=maed, gc_only_loto_R2=r2g, gc_only_loto_MAE=maeg,
                           dMAE_gc_minus_disc=maeg - maed, dMAE_lo=lo, dMAE_hi=hi,
                           partial_disc_given_gc=pdg[0], partial_disc_given_gc_p=pdg[1],
                           partial_disc_given_gc_lo=pdg[2], partial_disc_given_gc_hi=pdg[3],
                           partial_gc_given_disc=pgd[0], partial_gc_given_disc_p=pgd[1],
                           partial_gc_given_disc_lo=pgd[2], partial_gc_given_disc_hi=pgd[3],
                           R2_disc_plus_gc=r2full, lmg_disc=sh[0], lmg_gc=sh[1], lmg_disc_share=sh[0] / r2full,
                           lmg_disc_share_lo=np.percentile(bsh, 2.5), lmg_disc_share_hi=np.percentile(bsh, 97.5)))
    p7 = pd.DataFrame(p7)
    p7.to_csv(os.path.join(RD, "multivar_posthoc_gc_vs_disc.csv"), index=False)
    # fix round 2 (post-hoc readout): per-task S11 LOTO predictions of disc-only, GC-only and the
    # pre-registered nested (selection-honest) search, to show why the nested MAE sits far above the
    # GC-only LOTO MAE: GC-only is the in-sample AICc winner, so its LOTO MAE is not selection-honest,
    # and the nested search sometimes adds a second covariate inside a fold.
    prn_, sel_ = nested_store[("S11/Y1_MCC", "poolA", "aicc")]
    dd = subsets_of["S11"]
    yv, xd, gc = (dd[c].to_numpy(float) for c in ("bench_mcc", "disc", "gc_mean"))
    prd, prg = loto_pred(xd[:, None], yv), loto_pred(gc[:, None], yv)
    pt7 = pd.DataFrame(dict(task=dd["task"], actual=yv, disc_only_pred=prd, gc_only_pred=prg, nested_pred=prn_,
                            nested_selected=sel_, nested_n_covariates=[len(m_.split(" + ")) for m_ in sel_],
                            abs_err_disc_only=np.abs(yv - prd), abs_err_gc_only=np.abs(yv - prg),
                            abs_err_nested=np.abs(yv - prn_)))
    pt7["nested_worse_than_gc_only"] = pt7["abs_err_nested"] > pt7["abs_err_gc_only"] + 1e-12
    assert abs(pt7["abs_err_nested"].mean() - float(nested[nested["in_decision_rule"]]["nested_loto_MAE"].iloc[0])) < 1e-12
    assert abs(pt7["abs_err_gc_only"].mean() - float(p7.iloc[0]["gc_only_loto_MAE"])) < 1e-12
    pt7.to_csv(os.path.join(RD, "multivar_posthoc_gc_vs_disc_per_task.csv"), index=False)
    two = pt7[pt7["nested_n_covariates"] > 1]
    add("P7_S11_nested_vs_gc_only_per_fold",
        f"nested search added a 2nd covariate in {len(two)}/{len(pt7)} folds; worse than GC-only in "
        f"{int(two['nested_worse_than_gc_only'].sum())} of them; same model as GC-only in "
        f"{int((pt7['nested_selected'] == 'gc_mean').sum())} folds",
        "multivar_posthoc_gc_vs_disc_per_task.csv")
    L.append("P7 discriminability vs GC mean (head to head):")
    for _, r in p7.iterrows():
        L.append(f"    {r['config']} {r['variant']:<26} Spearman(disc,GC) {r['spearman_disc_gc']:+.3f} | LOTO MAE GC-only "
                 f"{r['gc_only_loto_MAE']:.4f} vs disc-only {r['disc_only_loto_MAE']:.4f} (dMAE {r['dMAE_gc_minus_disc']:+.4f} "
                 f"[{r['dMAE_lo']:+.4f},{r['dMAE_hi']:+.4f}]) | rho(disc,Y|GC) {r['partial_disc_given_gc']:+.3f} "
                 f"[{r['partial_disc_given_gc_lo']:+.3f},{r['partial_disc_given_gc_hi']:+.3f}] p={r['partial_disc_given_gc_p']:.2g}; "
                 f"rho(GC,Y|disc) {r['partial_gc_given_disc']:+.3f} [{r['partial_gc_given_disc_lo']:+.3f},"
                 f"{r['partial_gc_given_disc_hi']:+.3f}] p={r['partial_gc_given_disc_p']:.2g} | LMG share disc "
                 f"{r['lmg_disc_share']:.2f} [{r['lmg_disc_share_lo']:.2f},{r['lmg_disc_share_hi']:.2f}] of R2 {r['R2_disc_plus_gc']:.3f}")
        add(f"P7_{r['config']}_{r['variant']}",
            f"GC-only LOTO MAE {r['gc_only_loto_MAE']:.4f} vs disc {r['disc_only_loto_MAE']:.4f}, dMAE {r['dMAE_gc_minus_disc']:+.4f} "
            f"[{r['dMAE_lo']:+.4f},{r['dMAE_hi']:+.4f}]; rho(disc|GC) {r['partial_disc_given_gc']:+.3f} "
            f"[{r['partial_disc_given_gc_lo']:+.3f},{r['partial_disc_given_gc_hi']:+.3f}] p={r['partial_disc_given_gc_p']:.3f}; "
            f"rho(GC|disc) {r['partial_gc_given_disc']:+.3f} [{r['partial_gc_given_disc_lo']:+.3f},"
            f"{r['partial_gc_given_disc_hi']:+.3f}] p={r['partial_gc_given_disc_p']:.3f}; LMG disc share {r['lmg_disc_share']:.2f}",
            "multivar_posthoc_gc_vs_disc.csv")
    # ---- P8
    ci_fp = os.path.join(RD, "multivar_posthoc_class_identity.csv")
    if os.path.exists(ci_fp):
        cid = pd.read_csv(ci_fp).set_index(["task", "check"])["value"]
        assert cid[("nt_enhancers_types", "label0_set_equals_nt_enhancers_label0_train")] == 1
        assert cid[("nt_enhancers_types", "label0_set_equals_nt_enhancers_label0_test")] == 1
        assert cid[("nt_splice_sites_all", "class0_AG_peak_frequency")] > 0.9       # acceptor
        assert cid[("nt_splice_sites_all", "class1_GT_peak_frequency")] > 0.9       # donor
        assert max(cid[("nt_splice_sites_all", "class2_GT_peak_frequency")],
                   cid[("nt_splice_sites_all", "class2_AG_peak_frequency")]) < 0.2  # neither
        mdall = pd.concat([pd.read_csv(os.path.join(_mv.ROOT, "results", "motif_discriminability.csv")), md],
                          ignore_index=True)

        def maxpos(task, contrast):
            g_ = mdall[(mdall["dataset"] == task) & (mdall["contrast"] == contrast)]
            j_ = g_["motif_auroc"].idxmax()
            return float(g_.loc[j_, "motif_auroc"]), f"{contrast}:{g_.loc[j_, 'TF']}"

        FUNCTIONAL = {"human_ensembl_regulatory": ["enhancer_vs_rest", "ocr_vs_rest", "promoter_vs_rest"],
                      "nt_enhancers_types": ["1_vs_rest", "2_vs_rest"],
                      "nt_splice_sites_all": ["0_vs_rest", "1_vs_rest"]}
        rep = {t: mdall[(mdall["dataset"] == t) & mdall["representative"]]["contrast"].iloc[0] for t in FUNCTIONAL}
        cvi = cov.set_index("task")
        for t in FUNCTIONAL:                     # as-published disc == max over the representative contrast
            assert abs(maxpos(t, rep[t])[0] - cvi.loc[t, "disc"]) < 1e-4, t
        u1 = {t: (maxpos(t, rep[t]) if rep[t] in FUNCTIONAL[t] else max(maxpos(t, c) for c in FUNCTIONAL[t]))
              for t in FUNCTIONAL}
        u2 = {t: max(maxpos(t, c) for c in FUNCTIONAL[t]) for t in FUNCTIONAL}
        variants = [("as_published", {}),
                    ("P4_0_vs_rest_reversed", {"nt_enhancers_types": (d_fix, f"0_vs_rest reversed:{tf_fix}")}),
                    ("1_vs_rest", {"nt_enhancers_types": maxpos("nt_enhancers_types", "1_vs_rest")}),
                    ("2_vs_rest", {"nt_enhancers_types": maxpos("nt_enhancers_types", "2_vs_rest")}),
                    ("U1_functional_representative", {t: v for t, v in u1.items() if abs(v[0] - cvi.loc[t, "disc"]) > 1e-9}),
                    ("U2_max_over_functional_classes", {t: v for t, v in u2.items() if abs(v[0] - cvi.loc[t, "disc"]) > 1e-9})]
        p8 = []
        for vname, repl in variants:
            for sub in ("S11", "S15"):
                dd = subsets_of[sub].copy()
                ch = []
                for t, (v, src) in repl.items():
                    if t in set(dd["task"]):
                        dd.loc[dd["task"] == t, "disc"] = v
                        ch.append(f"{t}={v:.4f} ({src})")
                yv, xd = dd["bench_mcc"].to_numpy(float), dd["disc"].to_numpy(float)
                pr8 = loto_pred(xd[:, None], yv)
                r2v, r2lo, r2hi, maev, maelo, maehi = loto_boot_ci(yv, pr8)
                rho, prho = stats.spearmanr(xd, yv)
                p8.append(dict(config=f"{sub}/Y1_MCC", variant=vname, changed="; ".join(ch) or "none",
                               spearman=rho, spearman_p=prho, loto_R2=r2v, loto_R2_lo=r2lo, loto_R2_hi=r2hi,
                               loto_MAE=maev, loto_MAE_lo=maelo, loto_MAE_hi=maehi))
        p8 = pd.DataFrame(p8)
        p8.to_csv(os.path.join(RD, "multivar_posthoc_orientation_robustness.csv"), index=False)
        L.append("P8 robustness of the P4 re-reading + uniform functional-class rules (disc-only LOTO):")
        for _, r in p8.iterrows():
            L.append(f"    {r['config']} {r['variant']:<31} LOTO R2 {r['loto_R2']:+.3f} [{r['loto_R2_lo']:+.3f},{r['loto_R2_hi']:+.3f}] "
                     f"MAE {r['loto_MAE']:.4f}  Spearman {r['spearman']:+.3f}  changed: {r['changed']}")
            add(f"P8_{r['config']}_{r['variant']}", f"LOTO R2 {r['loto_R2']:+.3f} [{r['loto_R2_lo']:+.3f},{r['loto_R2_hi']:+.3f}]; "
                f"MAE {r['loto_MAE']:.4f}; Spearman {r['spearman']:+.3f}; changed: {r['changed']}",
                "multivar_posthoc_orientation_robustness.csv")
    else:
        L.append("P8 skipped: multivar_posthoc_class_identity.csv not found (run audit_exact_dups.py)")
    # ---- P9
    p9 = []
    for sub in ("S11", "S15"):
        dd = subsets_of[sub]
        yv, xd = dd["bench_mcc"].to_numpy(float), dd["disc"].to_numpy(float)
        lin = loto_pred(xd[:, None], yv)
        for vname, pr9 in (("linear_published", lin), ("linear_capped_at_1", np.clip(lin, -1, 1)),
                           ("fisher_z", fisher_z_loto(xd[:, None], yv))):
            r2v, maev = loto_metrics(yv, pr9)
            lo, hi = paired_boot_dmae(yv - pr9, yv - lin)
            for t, a, q in zip(dd["task"], yv, pr9):
                p9.append(dict(config=f"{sub}/Y1_MCC", variant=vname, task=t, actual=a, loto_pred=q, resid=a - q,
                               loto_R2=r2v, loto_MAE=maev, dMAE_vs_linear=maev - loto_metrics(yv, lin)[1],
                               dMAE_lo=lo, dMAE_hi=hi))
    p9 = pd.DataFrame(p9)
    p9.to_csv(os.path.join(RD, "multivar_posthoc_bounded.csv"), index=False)
    L.append("P9 bounded-outcome sensitivity (disc-only LOTO):")
    for (cfg, vname), g in p9.groupby(["config", "variant"], sort=False):
        nt = g.set_index("task").loc["human_nontata_promoters"]
        L.append(f"    {cfg} {vname:<19} LOTO R2 {g['loto_R2'].iloc[0]:+.3f} MAE {g['loto_MAE'].iloc[0]:.4f} "
                 f"(dMAE vs linear {g['dMAE_vs_linear'].iloc[0]:+.4f} [{g['dMAE_lo'].iloc[0]:+.4f},{g['dMAE_hi'].iloc[0]:+.4f}])  "
                 f"human_nontata pred {nt['loto_pred']:.3f} resid {nt['resid']:+.3f}")
        add(f"P9_{cfg}_{vname}", f"LOTO R2 {g['loto_R2'].iloc[0]:+.3f}, MAE {g['loto_MAE'].iloc[0]:.4f}; "
            f"human_nontata pred {nt['loto_pred']:.3f}, resid {nt['resid']:+.3f}", "multivar_posthoc_bounded.csv")
    # ---- P10
    p10 = []
    lo_d, hi_d = float(d11["disc"].min()), float(d11["disc"].max())
    msk = (xs >= lo_d) & (xs <= hi_d)
    xr, yr = xs[msk], ys[msk]
    rho_s, p_s = stats.spearmanr(xr, yr)
    b_in = np.polyfit(xr, yr, 1)
    res_in = yr - np.polyval(b_in, xr)
    yv11, xd11 = d11["bench_mcc"].to_numpy(float), d11["disc"].to_numpy(float)
    b_re = np.polyfit(xd11, yv11, 1)
    res_re = yv11 - np.polyval(b_re, xd11)

    def q(name, value, note=""):
        p10.append(dict(quantity=name, value=value, note=note))

    q("real_S11_disc_range", f"[{lo_d:.4f},{hi_d:.4f}]")
    q("synthetic_n_in_range", int(msk.sum()), f"of {len(xs)}; motifs: " + ", ".join(
        f"{m_} {c_}" for m_, c_ in dose.loc[msk, "motif"].value_counts().items())
      + f"; seeds {dose.loc[msk, 'seed'].nunique()} (points clustered by motif and seed)")
    q("synthetic_in_range_spearman", rho_s, f"p={p_s:.2g}")
    q("synthetic_in_range_ols_R2", 1 - float(res_in @ res_in) / float(((yr - yr.mean()) ** 2).sum()))
    q("synthetic_in_range_resid_sd", float(np.sqrt(res_in @ res_in / (len(yr) - 2))), "OLS residual SD, df n-2")
    q("real_S11_ols_R2", 1 - float(res_re @ res_re) / float(((yv11 - yv11.mean()) ** 2).sum()))
    q("real_S11_resid_sd", float(np.sqrt(res_re @ res_re / (len(yv11) - 2))), "OLS residual SD, df n-2")
    q("resid_sd_ratio_real_over_synthetic", float(np.sqrt(res_re @ res_re / (len(yv11) - 2))
                                                  / np.sqrt(res_in @ res_in / (len(yr) - 2))))
    q("synthetic_n_train_n_test", f"{sorted(int(v) for v in dose['n_train'].unique())} / "
                                  f"{sorted(int(v) for v in dose['n_test'].unique())}", "every synthetic point")
    q("real_S11_n_train_range", f"[{int(d11['n_train'].min())},{int(d11['n_train'].max())}]")
    off = yv11 - np.polyval(bsyn, xd11)
    q("level_offset_mean", float(off.mean()), "real MCC minus pooled synthetic line (as A9/P2)")
    for c in ("comp_mcc", "gc_mean", "ntrain_log", "len_log"):
        r_, p_ = stats.spearmanr(off, d11[c].to_numpy(float))
        q(f"spearman_level_offset_vs_{c}", r_, f"p={p_:.3g}, n={len(off)}")
    p10 = pd.DataFrame(p10)
    p10.to_csv(os.path.join(RD, "multivar_posthoc_synthetic_range.csv"), index=False)
    L.append("P10 synthetic range restriction + level-offset correlates:")
    for _, r in p10.iterrows():
        v_ = r["value"]
        L.append(f"    {r['quantity']:<40} {v_:+.4f}  {r['note']}" if isinstance(v_, float) else
                 f"    {r['quantity']:<40} {v_}  {r['note']}")
        add(f"P10_{r['quantity']}", round(v_, 4) if isinstance(v_, float) else v_, "multivar_posthoc_synthetic_range.csv",
            r["note"])
    # ---- P11 (fix round 1, 2026-09-24 evening; RESULTS-MOTIVATED, descriptive): is GC mean a task-family
    #      indicator, and does discriminability track MCC WITHIN a family?  Family = biological task type
    #      read from the task name, fixed here: promoter (4 tasks), enhancer/OCR (6 tasks), and
    #      human_ensembl_regulatory (3-class enhancer/OCR/promoter; its disc comes from promoter_vs_rest),
    #      reported both inside the promoter family and on its own.  Indicator for the between-family
    #      models: 1 = promoter family incl. regulatory, 0 = enhancer/OCR.
    FAM = {"human_nontata_promoters": "promoter", "nt_promoter_all": "promoter", "nt_promoter_no_tata": "promoter",
           "nt_promoter_tata": "promoter", "human_ensembl_regulatory": "regulatory_mixed",
           "human_enhancers_ensembl": "enhancer_ocr", "human_enhancers_cohn": "enhancer_ocr",
           "drosophila_enhancers_stark": "enhancer_ocr", "human_ocr_ensembl": "enhancer_ocr",
           "nt_enhancers": "enhancer_ocr", "nt_enhancers_types": "enhancer_ocr"}
    assert set(FAM) == set(d11["task"]), sorted(set(d11["task"]) ^ set(FAM))
    p11 = []
    fam = d11["task"].map(FAM).to_numpy()
    ind = np.isin(fam, ["promoter", "regulatory_mixed"]).astype(float)
    clean = (d11["homology"] < 0.05).to_numpy()          # the rule-based homology-clean set of P5 (n = 7)

    def exact_spearman_p(a, b):
        """two-sided exact permutation p of Spearman rho (all n! orderings; n <= 6 here).  Fix round 2: the
        vectorized perm_spearman_p (identical values; tests/test_multivar.py checks it against the old loop)."""
        p_, meth_ = perm_spearman_p(a, b)
        assert meth_.startswith("exact"), meth_
        return p_

    t1 = table1_routes()
    solved = np.array([t1[t] == "solved" for t in d11["task"]])

    for variant, dv in (("as_published", None), ("P4_orientation_consistent", d_fix)):
        dd = d11.copy()
        if dv is not None:
            dd.loc[dd["task"] == "nt_enhancers_types", "disc"] = dv
        yv, xd, gc = (dd[c].to_numpy(float) for c in ("bench_mcc", "disc", "gc_mean"))
        halfw = ((dd["mcc_hi"] - dd["mcc_lo"]) / 2).to_numpy(float)
        prom = np.isin(fam, ["promoter", "regulatory_mixed"])
        for gname, gmask in (("promoter_n4", fam == "promoter"),
                             ("promoter_plus_regulatory_n5", prom),
                             ("enhancer_ocr_n6", fam == "enhancer_ocr"),
                             ("homology_clean_promoter_plus_regulatory", prom & clean),
                             ("homology_clean_enhancer_ocr", (fam == "enhancer_ocr") & clean)):
            rd_, pd_ = spearman_t(xd[gmask], yv[gmask])
            rg_, pg_ = spearman_t(gc[gmask], yv[gmask])
            p11.append(dict(variant=variant, analysis=f"within_{gname}", n=int(gmask.sum()),
                            tasks=";".join(dd["task"][gmask]),
                            disc_min=xd[gmask].min(), disc_max=xd[gmask].max(), mcc_min=yv[gmask].min(),
                            mcc_max=yv[gmask].max(), gc_min=gc[gmask].min(), gc_max=gc[gmask].max(),
                            mcc_range=yv[gmask].max() - yv[gmask].min(), mean_mcc_ci_halfwidth=float(halfw[gmask].mean()),
                            spearman_disc_mcc=rd_, spearman_disc_mcc_p=pd_,
                            spearman_disc_mcc_p_exact=exact_spearman_p(xd[gmask], yv[gmask]),
                            spearman_gc_mcc=rg_, spearman_gc_mcc_p=pg_))
        # between-family: LOTO of family indicator alone, disc alone, GC alone, disc + indicator; partials.
        # Sensitivity: the 3-class regulatory task (coded 1) removed -> n = 10.
        for bname, keep in (("between_family_S11", np.ones(len(yv), bool)),
                            ("between_family_S10_excl_regulatory", fam != "regulatory_mixed"),
                            ("between_family_S7_homology_clean", clean)):
            yk, xk, gk, ik, fk = yv[keep], xd[keep], gc[keep], ind[keep], fam[keep]
            prf, prd_, prg_ = (loto_pred(v[:, None], yk) for v in (ik, xk, gk))
            prdf = loto_pred(np.column_stack([xk, ik]), yk)
            row = dict(variant=variant, analysis=bname, n=int(keep.sum()),
                       tasks="indicator=1: " + ";".join(dd["task"][keep][ik == 1]),
                       spearman_gc_indicator=stats.spearmanr(gk, ik)[0],
                       spearman_disc_indicator=stats.spearmanr(xk, ik)[0],
                       gc_promoter_family_min=float(gk[fk == "promoter"].min()),
                       gc_enhancer_family_max=float(gk[fk == "enhancer_ocr"].max()),
                       gc_separates_promoter_from_enhancer=bool(gk[fk == "promoter"].min() > gk[fk == "enhancer_ocr"].max()),
                       loto_MAE_indicator_only=loto_metrics(yk, prf)[1], loto_R2_indicator_only=loto_metrics(yk, prf)[0],
                       loto_MAE_disc_only=loto_metrics(yk, prd_)[1], loto_R2_disc_only=loto_metrics(yk, prd_)[0],
                       loto_MAE_gc_only=loto_metrics(yk, prg_)[1],
                       loto_MAE_disc_plus_indicator=loto_metrics(yk, prdf)[1],
                       # fix round 2: the family indicator is the SAME partition as Table I's outcome-defined
                       # 'solved' vs failing split, and a dichotomy of MCC itself (non-overlapping ranges), so
                       # its predictive accuracy is close to tautological (two MCC clusters)
                       indicator_equals_table1_solved=bool(np.array_equal(ik == 1, solved[keep])),
                       mcc_min_indicator1=float(yk[ik == 1].min()), mcc_max_indicator0=float(yk[ik == 0].max()),
                       indicator_is_mcc_dichotomy=bool(yk[ik == 1].min() > yk[ik == 0].max()))
            lo_, hi_ = paired_boot_dmae(yk - prf, yk - prd_)
            row.update(dMAE_indicator_minus_disc=row["loto_MAE_indicator_only"] - row["loto_MAE_disc_only"],
                       dMAE_lo=lo_, dMAE_hi=hi_)
            for key, (a_, b_, c_) in (("disc_given_indicator", (xk, yk, ik)), ("indicator_given_disc", (ik, yk, xk)),
                                      ("gc_given_indicator", (gk, yk, ik))):
                v_, p_, l_, h_ = partial_boot(a_, b_, c_)
                row.update({f"partial_{key}": v_, f"partial_{key}_p": p_, f"partial_{key}_lo": l_,
                            f"partial_{key}_hi": h_, f"partial_{key}_p_perm": perm_partial_p(a_, b_, c_)[0]})
            p11.append(row)
    p11 = pd.DataFrame(p11)
    p11.to_csv(os.path.join(RD, "multivar_posthoc_family.csv"), index=False)
    # fix round 2: per-task table (family, Table I route label, disc-only LOTO residual, homology flag)
    r11_ = resd[resd["config"] == "S11/Y1_MCC"].set_index("task")
    ft = pd.DataFrame(dict(task=d11["task"], family=fam, family_indicator=ind.astype(int),
                           table1_route=[t1[t] for t in d11["task"]], bench_mcc=d11["bench_mcc"],
                           mcc_lo=d11["mcc_lo"], mcc_hi=d11["mcc_hi"], disc=d11["disc"], gc_mean=d11["gc_mean"],
                           homology=d11["homology"], homology_clean=clean,
                           disc_only_loto_resid=r11_.loc[d11["task"], "resid"].to_numpy(float)))
    ft.to_csv(os.path.join(RD, "multivar_posthoc_family_tasks.csv"), index=False)
    fail_ = ft[ft["family_indicator"] == 0]
    for lab_ in ("contamination", "shared-motif"):
        g_ = fail_[fail_["table1_route"].str.startswith(lab_)]
        add(f"P11_S11_failing_{lab_}_disc_only_loto_resid",
            "; ".join(f"{t}={v:+.3f}" for t, v in zip(g_["task"], g_["disc_only_loto_resid"])),
            "multivar_posthoc_family_tasks.csv", "descriptive pointer only (n = 3 per label); not a test")
    L.append("P11 task family (results-motivated, descriptive): promoter (4) / enhancer-OCR (6) / regulatory (mixed)")
    for _, r in p11.iterrows():
        if r["analysis"].startswith("within"):
            L.append(f"    {r['variant']:<26} {r['analysis']:<36} n={r['n']}  disc [{r['disc_min']:.3f},{r['disc_max']:.3f}] "
                     f"MCC [{r['mcc_min']:.3f},{r['mcc_max']:.3f}] GC [{r['gc_min']:.3f},{r['gc_max']:.3f}]  "
                     f"Spearman(disc,MCC) {r['spearman_disc_mcc']:+.3f} (p={r['spearman_disc_mcc_p']:.2g}, exact "
                     f"{r['spearman_disc_mcc_p_exact']:.3f})  Spearman(GC,MCC) {r['spearman_gc_mcc']:+.3f} "
                     f"(p={r['spearman_gc_mcc_p']:.2g})  MCC range {r['mcc_range']:.3f} vs mean CI half-width "
                     f"{r['mean_mcc_ci_halfwidth']:.3f}")
            add(f"P11_{r['variant']}_{r['analysis']}",
                f"n={r['n']}; Spearman(disc,MCC) {r['spearman_disc_mcc']:+.3f} (exact p={r['spearman_disc_mcc_p_exact']:.3f}); "
                f"Spearman(GC,MCC) {r['spearman_gc_mcc']:+.3f} (p={r['spearman_gc_mcc_p']:.3f}); disc range "
                f"[{r['disc_min']:.3f},{r['disc_max']:.3f}]", "multivar_posthoc_family.csv")
        else:
            L.append(f"    {r['variant']:<26} {r['analysis']} (n={r['n']}): GC separates promoter (min {r['gc_promoter_family_min']:.3f}) from "
                     f"enhancer/OCR (max {r['gc_enhancer_family_max']:.3f}): {r['gc_separates_promoter_from_enhancer']}; "
                     f"Spearman(GC, indicator) {r['spearman_gc_indicator']:+.3f}, (disc, indicator) "
                     f"{r['spearman_disc_indicator']:+.3f} | LOTO MAE indicator-only {r['loto_MAE_indicator_only']:.4f} "
                     f"(R2 {r['loto_R2_indicator_only']:+.3f}) vs disc-only {r['loto_MAE_disc_only']:.4f} (dMAE "
                     f"{r['dMAE_indicator_minus_disc']:+.4f} [{r['dMAE_lo']:+.4f},{r['dMAE_hi']:+.4f}]), GC-only "
                     f"{r['loto_MAE_gc_only']:.4f}, disc+indicator {r['loto_MAE_disc_plus_indicator']:.4f} | rho(disc,Y|family) "
                     f"{r['partial_disc_given_indicator']:+.3f} [{r['partial_disc_given_indicator_lo']:+.3f},"
                     f"{r['partial_disc_given_indicator_hi']:+.3f}] p={r['partial_disc_given_indicator_p']:.2g}; "
                     f"rho(family,Y|disc) {r['partial_indicator_given_disc']:+.3f} [{r['partial_indicator_given_disc_lo']:+.3f},"
                     f"{r['partial_indicator_given_disc_hi']:+.3f}] p={r['partial_indicator_given_disc_p']:.2g}; "
                     f"rho(GC,Y|family) {r['partial_gc_given_indicator']:+.3f} [{r['partial_gc_given_indicator_lo']:+.3f},"
                     f"{r['partial_gc_given_indicator_hi']:+.3f}]")
            add(f"P11_{r['variant']}_{r['analysis']}",
                f"LOTO MAE family-indicator-only {r['loto_MAE_indicator_only']:.4f} vs disc-only {r['loto_MAE_disc_only']:.4f} "
                f"(dMAE {r['dMAE_indicator_minus_disc']:+.4f} [{r['dMAE_lo']:+.4f},{r['dMAE_hi']:+.4f}]); rho(disc,Y|family) "
                f"{r['partial_disc_given_indicator']:+.3f} [{r['partial_disc_given_indicator_lo']:+.3f},"
                f"{r['partial_disc_given_indicator_hi']:+.3f}]; rho(family,Y|disc) {r['partial_indicator_given_disc']:+.3f} "
                f"[{r['partial_indicator_given_disc_lo']:+.3f},{r['partial_indicator_given_disc_hi']:+.3f}]; GC separates "
                f"families: {r['gc_separates_promoter_from_enhancer']}", "multivar_posthoc_family.csv")
    # ---- P12 (fix round 2, 2026-09-24 late evening; RESULTS-MOTIVATED, descriptive): the 11 TF-motif tasks
    #      are not 11 distinct sequence collections (audit_task_overlap.py, all 210 ordered task pairs):
    #      nt_promoter_all is the exact union of nt_promoter_tata and nt_promoter_no_tata (sequence AND
    #      label); nt_enhancers_types re-labels the nt_enhancers sequences; the OCR class of
    #      human_ensembl_regulatory is human_ocr_ensembl's positive class (same train/test split).  Every
    #      cross-task statistic above treats tasks as exchangeable units, so the headline statistics are
    #      recomputed on task sets with that redundancy removed.  The sets below were fixed before any P12
    #      number was computed; the union is always dropped (it adds no sequence), and for each overlapping
    #      pair both choices are reported.
    idf = pd.read_csv(os.path.join(RD, "multivar_posthoc_task_identity.csv"))
    assert (idf["value"] == 1).all(), idf
    ovl = pd.read_csv(os.path.join(RD, "multivar_posthoc_task_overlap.csv"))
    # fix round 3: EXACT_SHARING (n_distinct_sets) must be exactly the S11 pairs sharing >= 0.1% of either task
    s11set = set(d11["task"])
    big_ = ovl[ovl["task_a"].isin(s11set) & ovl["task_b"].isin(s11set) & (ovl["frac_a_in_b"] >= 0.001)]
    assert {frozenset(p) for p in zip(big_["task_a"], big_["task_b"])} == {frozenset(p) for p in EXACT_SHARING}, big_

    def max_overlap(tasks):
        g_ = ovl[ovl["task_a"].isin(tasks) & ovl["task_b"].isin(tasks)]
        return float(g_["frac_a_in_b"].max()) if len(g_) else 0.0

    P5DROP = dict(p5_defs)
    U_, E_, ET_, O_, R_ = ("nt_promoter_all", "nt_enhancers", "nt_enhancers_types", "human_ocr_ensembl",
                           "human_ensembl_regulatory")
    p12_defs = [
        ("S11_all", "all 11 TF-motif tasks", []),
        ("S11_minus_union", "drop nt_promoter_all (= nt_promoter_tata + nt_promoter_no_tata)", [U_]),
        ("dupfree9_keep_nt_enhancers", "and drop nt_enhancers_types (re-labels the nt_enhancers sequences)", [U_, ET_]),
        ("dupfree9_keep_nt_enhancers_types", "and drop nt_enhancers instead", [U_, E_]),
        ("noshare8_drop_ocr", "dupfree9_keep_nt_enhancers and drop human_ocr_ensembl (its positives = the OCR "
                              "class of human_ensembl_regulatory): no exact sequence sharing >= 0.1%", [U_, ET_, O_]),
        ("noshare8_drop_regulatory", "dupfree9_keep_nt_enhancers and drop human_ensembl_regulatory instead",
         [U_, ET_, R_]),
        ("n8_minus_union", "paper's 9 minus human_enhancers_ensembl (P5 n=8), and drop nt_promoter_all",
         P5DROP["paper9_minus_human_enhancers_ensembl"] + [U_]),
        ("homology_clean7_minus_union", "homology-clean 7 (P5), and drop nt_promoter_all",
         P5DROP["rule_clean_homology_lt_0.05"] + [U_]),
        ("homology_clean_noshare5_drop_ocr", "homology-clean 7, drop nt_promoter_all and human_ocr_ensembl",
         P5DROP["rule_clean_homology_lt_0.05"] + [U_, O_]),
        ("homology_clean_noshare5_drop_regulatory", "homology-clean 7, drop nt_promoter_all and "
                                                    "human_ensembl_regulatory",
         P5DROP["rule_clean_homology_lt_0.05"] + [U_, R_])]
    # fix round 3 (claims verifier round 3): the clean-7 disc-beats-GC result rests on the 3-class regulatory
    # task (its GC mean pools three classes), so the regulatory-removed sets are written out as rows
    p12_defs += [("S11_minus_regulatory", "all 11 minus human_ensembl_regulatory (3-class; GC mean pools 3 classes)",
                  [R_]),
                 ("homology_clean7_minus_regulatory", "homology-clean 7 (P5) minus human_ensembl_regulatory",
                  P5DROP["rule_clean_homology_lt_0.05"] + [R_])]
    # addendum (fix round 2, same pass): the suite rows of the paper's Table II, which the overlap affects
    # most (the 5 NT TF-motif tasks are 3 distinct sequence collections)
    p12_defs += [("suite_GB_only", "Genomic-Benchmarks TF-motif tasks only (Table II row 'GB-only')",
                  list(d11.loc[d11["suite"] == "NucTransformer", "task"])),
                 ("suite_NT_only", "Nucleotide-Transformer TF-motif tasks only (Table II row 'NT-only')",
                  list(d11.loc[d11["suite"] != "NucTransformer", "task"]))]
    fam_of = dict(zip(d11["task"], fam))
    p12 = []
    for name, desc, drop in p12_defs:
        dd = d11[~d11["task"].isin(drop)].reset_index(drop=True)
        tk = list(dd["task"])
        yv, xd, gc = (dd[c].to_numpy(float) for c in ("bench_mcc", "disc", "gc_mean"))
        ik = np.array([fam_of[t] in ("promoter", "regulatory_mixed") for t in tk], float)
        nn = len(yv)
        rho, prho = spearman_t(xd, yv)
        pp_, pm_ = perm_spearman_p(xd, yv)
        jk = [stats.spearmanr(np.delete(xd, i), np.delete(yv, i))[0] for i in range(nn)]
        prd_, prg_ = loto_pred(xd[:, None], yv), loto_pred(gc[:, None], yv)
        r2d_, maed_ = loto_metrics(yv, prd_)
        r2g_, maeg_ = loto_metrics(yv, prg_)
        both = ik.sum() >= 2 and (1 - ik).sum() >= 2
        prf_ = loto_pred(ik[:, None], yv) if both else np.full(nn, np.nan)
        r2f_, maef_ = loto_metrics(yv, prf_) if both else (np.nan, np.nan)
        gdl, gdh = paired_boot_dmae(yv - prg_, yv - prd_)
        fdl, fdh = paired_boot_dmae(yv - prf_, yv - prd_) if both else (np.nan, np.nan)
        pdg_ = partial_boot(xd, yv, gc, return_n=True)
        pgd_ = partial_boot(gc, yv, xd, return_n=True)
        pdf_ = partial_boot(xd, yv, ik) if both else (np.nan,) * 4
        row = dict(subset=name, description=desc, n=nn, dropped=";".join(drop), tasks=";".join(tk),
                   max_pairwise_shared_frac=max_overlap(tk), n_exact_distinct_sets=n_distinct_sets(tk)[0],
                   n_no_exact_sharing=n_distinct_sets(tk)[1],
                   homology_clean=bool((dd["homology"] < 0.05).all()),
                   n_promoter_family=int(ik.sum()), n_enhancer_family=int((1 - ik).sum()),
                   spearman=rho, spearman_p_t_approx=prho, spearman_p_perm=pp_, spearman_p_perm_method=pm_,
                   jackknife_min=min(jk), jackknife_max=max(jk),
                   loto_MAE_disc_only=maed_, loto_R2_disc_only=r2d_, loto_MAE_gc_only=maeg_, loto_R2_gc_only=r2g_,
                   loto_MAE_family_only=maef_, loto_R2_family_only=r2f_,
                   dMAE_gc_minus_disc=maeg_ - maed_, dMAE_gc_minus_disc_lo=gdl, dMAE_gc_minus_disc_hi=gdh,
                   dMAE_family_minus_disc=maef_ - maed_, dMAE_family_minus_disc_lo=fdl, dMAE_family_minus_disc_hi=fdh,
                   partial_disc_given_gc=pdg_[0], partial_disc_given_gc_p=pdg_[1],
                   partial_disc_given_gc_p_perm=perm_partial_p(xd, yv, gc)[0],
                   partial_disc_given_gc_lo=pdg_[2], partial_disc_given_gc_hi=pdg_[3],
                   partial_disc_given_gc_n_boot_ok=pdg_[4],
                   partial_gc_given_disc=pgd_[0], partial_gc_given_disc_p=pgd_[1],
                   partial_gc_given_disc_p_perm=perm_partial_p(gc, yv, xd)[0],
                   partial_gc_given_disc_lo=pgd_[2], partial_gc_given_disc_hi=pgd_[3],
                   partial_gc_given_disc_n_boot_ok=pgd_[4],
                   partial_disc_given_family=pdf_[0], partial_disc_given_family_p=pdf_[1],
                   partial_disc_given_family_p_perm=(perm_partial_p(xd, yv, ik)[0] if both else np.nan))
        rdo_ = gc_disc_readout(tk, xd, gc, yv)
        assert abs(rdo_["loto_MAE_disc"] - maed_) < 1e-12 and abs(rdo_["loto_MAE_gc"] - maeg_) < 1e-12
        assert abs(rdo_["spearman_disc_mcc_p_perm"] - pp_) < 1e-12
        row.update({k_: v_ for k_, v_ in rdo_.items() if k_ not in ("loto_MAE_disc", "loto_MAE_gc", "spearman_disc_mcc_p_perm")})
        for gname, gm in (("promoter_family", ik == 1), ("enhancer_family", ik == 0)):
            row[f"within_{gname}_n"] = int(gm.sum())
            ok_ = gm.sum() >= 3
            row[f"within_{gname}_spearman"] = stats.spearmanr(xd[gm], yv[gm])[0] if ok_ else np.nan
            row[f"within_{gname}_p_exact"] = perm_spearman_p(xd[gm], yv[gm])[0] if ok_ else np.nan
        p12.append(row)
    p12 = pd.DataFrame(p12)
    q0_, q5_ = p12.iloc[0], p5.iloc[0]
    assert abs(q0_["spearman"] - q5_["spearman"]) < 1e-12 and abs(q0_["loto_MAE_disc_only"] - q5_["loto_MAE"]) < 1e-12
    p12.to_csv(os.path.join(RD, "multivar_posthoc_distinct_datasets.csv"), index=False)
    L.append("P12 exact-duplicate sensitivity (tasks sharing exact sequences collapsed; results-motivated, descriptive):")
    for _, r in p12.iterrows():
        L.append(f"    {r['subset']:<42} n={r['n']:>2} (max shared {r['max_pairwise_shared_frac']:.3f}; clean "
                 f"{r['homology_clean']})  rho {r['spearman']:+.3f} (perm p={r['spearman_p_perm']:.2g})  rho(GC,MCC) "
                 f"{r['spearman_gc_mcc']:+.3f} (perm p={r['spearman_gc_mcc_p_perm']:.2g}; GC-disc "
                 f"[{r['rho_gc_minus_rho_disc_lo']:+.2f},{r['rho_gc_minus_rho_disc_hi']:+.2f}])  lower MAE: {r['loto_MAE_lower']} "
                 f"(reversed by dropping: {r['loto_MAE_order_reversed_by_dropping'] or 'none'}; without regulatory: "
                 f"{r['loto_MAE_lower_without_regulatory']})  jackknife "
                 f"[{r['jackknife_min']:+.3f},{r['jackknife_max']:+.3f}]  LOTO MAE disc {r['loto_MAE_disc_only']:.4f} GC "
                 f"{r['loto_MAE_gc_only']:.4f} family {r['loto_MAE_family_only']:.4f}  rho(disc|GC) "
                 f"{r['partial_disc_given_gc']:+.3f} (t p={r['partial_disc_given_gc_p']:.2g}, perm "
                 f"{r['partial_disc_given_gc_p_perm']:.2g})  rho(GC|disc) {r['partial_gc_given_disc']:+.3f} (t p="
                 f"{r['partial_gc_given_disc_p']:.2g}, perm {r['partial_gc_given_disc_p_perm']:.2g})  within promoter "
                 f"{r['within_promoter_family_spearman']:+.2f} (n={r['within_promoter_family_n']}, exact p="
                 f"{r['within_promoter_family_p_exact']:.3f}), enhancer {r['within_enhancer_family_spearman']:+.2f} "
                 f"(n={r['within_enhancer_family_n']}, exact p={r['within_enhancer_family_p_exact']:.3f})")
        add(f"P12_{r['subset']}", f"n={r['n']}; Spearman {r['spearman']:+.3f} (perm p={r['spearman_p_perm']:.2g}); "
            f"Spearman(GC, MCC) {r['spearman_gc_mcc']:+.3f} (perm p={r['spearman_gc_mcc_p_perm']:.2g}); "
            f"LOTO MAE disc {r['loto_MAE_disc_only']:.4f} / GC {r['loto_MAE_gc_only']:.4f} / family "
            f"{r['loto_MAE_family_only']:.4f}; rho(disc|GC) {r['partial_disc_given_gc']:+.3f} (t p="
            f"{r['partial_disc_given_gc_p']:.3f}); rho(GC|disc) {r['partial_gc_given_disc']:+.3f} (t p="
            f"{r['partial_gc_given_disc_p']:.3f})", "multivar_posthoc_distinct_datasets.csv", r["description"])

    # ---- P13 (fix round 3, 2026-09-24 night; RESULTS-MOTIVATED, descriptive): LOCUS-level sharing.  The exact-
    #      string audit cannot see the same locus in a shifted or resized window; audit_locus_overlap.py measures
    #      31-mer containment for every pair of S11 classes.  Task-level link = the largest share of any class
    #      of one task whose sequences are 'mostly contained' (>= 0.5 of their sampled 31-mers) in the other
    #      task, over both directions.  Thresholds fixed before this block was run: 0.50, 0.25, 0.10.  At each,
    #      tasks linked at or above it are treated as non-independent, and the statistics are recomputed on
    #      every MAXIMUM set of mutually unlinked tasks (all such sets reported; descriptive, not a test).
    lk = pd.read_csv(os.path.join(RD, "multivar_posthoc_task_locus_links.csv"))
    assert set(lk["task_a"]) | set(lk["task_b"]) == s11set and len(lk) == 55, len(lk)
    ex_pairs = {frozenset(p) for p in EXACT_SHARING}
    lk_ex = lk[[frozenset((a, b)) in ex_pairs for a, b in zip(lk["task_a"], lk["task_b"])]]
    assert (lk_ex["locus_link"] > 0.9).all(), lk_ex        # the exact identities are recovered by the k-mer audit
    dix = d11.set_index("task")
    p13 = []
    for tau in (0.50, 0.25, 0.10):
        edges = [(a, b) for a, b, v in zip(lk["task_a"], lk["task_b"], lk["locus_link"]) if v >= tau]
        comp_ = {t: t for t in s11set}

        def root(t):
            while comp_[t] != t:
                t = comp_[t]
            return t
        for a, b in edges:
            comp_[root(a)] = root(b)
        groups = {}
        for t in sorted(s11set):
            groups.setdefault(root(t), []).append(t)
        gtxt = " | ".join("+".join(g) for g in sorted(groups.values(), key=lambda g: (-len(g), g)))
        mis = max_independent_sets(sorted(s11set), edges)
        for i, m in enumerate(mis):
            xd_, yv_, gc_ = (dix.loc[m, c].to_numpy(float) for c in ("disc", "bench_mcc", "gc_mean"))
            ok_ = len(m) >= 3 and xd_.std() > 0 and yv_.std() > 0
            p13.append(dict(tau=tau, n_links=len(edges), n_linked_groups=len(groups), linked_groups=gtxt,
                            max_unlinked_size=len(m), n_max_unlinked_sets=len(mis), set_id=i, tasks=";".join(m),
                            n_solved=int(sum(fam_of[t] in ("promoter", "regulatory_mixed") for t in m)),
                            homology_clean=bool((dix.loc[m, "homology"] < 0.05).all()),
                            spearman_disc_mcc=stats.spearmanr(xd_, yv_)[0] if ok_ else np.nan,
                            spearman_disc_mcc_p_perm=perm_spearman_p(xd_, yv_)[0] if ok_ else np.nan,
                            spearman_gc_mcc=stats.spearmanr(gc_, yv_)[0] if ok_ else np.nan,
                            spearman_gc_mcc_p_perm=perm_spearman_p(gc_, yv_)[0] if ok_ else np.nan))
    p13 = pd.DataFrame(p13)
    p13.to_csv(os.path.join(RD, "multivar_posthoc_locus_independent.csv"), index=False)
    L.append("P13 locus-level sharing (audit_locus_overlap.py; results-motivated, descriptive):")
    L.append("    task pairs with locus link >= 0.10: " + "; ".join(
        f"{a}~{b} {v:.2f}" for a, b, v in zip(lk["task_a"], lk["task_b"], lk["locus_link"]) if v >= 0.10))
    for tau, g in p13.groupby("tau", sort=False):
        r0 = g.iloc[0]
        L.append(f"    tau {tau:.2f}: {int(r0['n_links'])} links, {int(r0['n_linked_groups'])} linked groups "
                 f"({r0['linked_groups']}); largest mutually unlinked sets: n = {int(r0['max_unlinked_size'])} "
                 f"({len(g)} sets; {int(g['n_solved'].min())}-{int(g['n_solved'].max())} solved tasks each): rho(disc, MCC) "
                 f"{g['spearman_disc_mcc'].min():+.3f} to {g['spearman_disc_mcc'].max():+.3f} (perm p "
                 f"{g['spearman_disc_mcc_p_perm'].min():.2g}-{g['spearman_disc_mcc_p_perm'].max():.2g}); rho(GC, MCC) "
                 f"{g['spearman_gc_mcc'].min():+.3f} to {g['spearman_gc_mcc'].max():+.3f}")
        add(f"P13_tau_{tau:.2f}", f"{int(r0['n_linked_groups'])} linked groups; max unlinked n = {int(r0['max_unlinked_size'])} "
            f"({len(g)} sets); rho(disc, MCC) {g['spearman_disc_mcc'].min():+.3f} to {g['spearman_disc_mcc'].max():+.3f} "
            f"(perm p <= {g['spearman_disc_mcc_p_perm'].max():.2g}); rho(GC, MCC) {g['spearman_gc_mcc'].min():+.3f} to "
            f"{g['spearman_gc_mcc'].max():+.3f}", "multivar_posthoc_locus_independent.csv", r0["linked_groups"])

    mdisc = models[(models["config"] == "S11/Y1_MCC") & (models["family"] == "poolA") & (models["model"] == "disc")].iloc[0]
    add("A2_S11_MCC_disc_only_in_sample_R2_adjR2", f"{mdisc['R2']:.3f} / {mdisc['adjR2']:.3f}",
        "multivar_models.csv (S11/Y1_MCC, poolA, model=disc)")

    pd.DataFrame(S).to_csv(os.path.join(RD, "multivar_summary.csv"), index=False)
    hdr = ["=" * 110,
           "MULTIVARIATE COVARIATE ANALYSIS OF REAL-TASK LEARNABILITY (Reviewer 1, comment 1) -- run_multivar.py",
           "Pre-registered plan: results/upgrades/multivar_preregistration.txt.  Seed 42; 1000 task-bootstrap",
           "resamples; 1000 permutations.  Units = tasks (S11 = 11 TF-motif tasks PRIMARY, S15 = all 15).",
           "=" * 110]
    rep = "\n".join(hdr + L)
    with open(os.path.join(RD, "multivar_analysis_log.txt"), "w") as fh:
        fh.write(rep + "\n")
    print(rep)
    print(f"\nwrote results/upgrades/multivar_{{models,top_models,nested,permutation,lmg,partial,residuals,"
          f"residual_corr,noise,reviewer_model,synthetic_transfer,summary}}.csv + multivar_analysis_log.txt "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
