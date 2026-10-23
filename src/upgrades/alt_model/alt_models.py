"""Model library for the alternative-interpretable-model analysis (Reviewer 2, comment 3).

Three model families, all fit on the SAME k-mer spectrum features (featurize.kmer_spectrum at the
task's LightGBM-validation-selected k, results/val_selection.csv:new_k):

  lgbm  reference: the paper's LightGBM with its FIXED hyperparameters (models.LGBM_PARAMS), only
        the thread count changed (n_jobs=1 here; deterministic mode unchanged).
  lr    alternative (a): L1- or L2-regularised logistic regression on per-feature standardized
        k-mer frequencies (StandardScaler(with_mean=False), sparse-safe), liblinear solver
        (one-vs-rest for the 3 multiclass tasks). The penalty (L1 vs L2) and C are TUNED on a
        stratified 80/20 validation carve of the TRAINING split (seed 42) by validation MCC over
        C in LR_C_GRID x penalty in {L1, L2}; ties go to the earlier grid entry (L1 before L2,
        smaller C first = the sparser / more regularised model). The winner is refit on the whole
        training split. Coefficients are directly interpretable per-k-mer log-odds weights.
  ebm   alternative (b): Explainable Boosting Machine (interpret-core 0.7.8; a glass-box GA2M,
        Lou et al. KDD 2013) on [20-d composition signature (4 mono + 16 dinucleotide freqs,
        _common.comp_signature)] + [the EBM_M=128 k-mers of the SAME spectrum with the largest
        univariate ANOVA F-statistic, ranked on the TRAINING split only]. Fixed settings (no
        tuning; EBM early-stops on its own internal 15% validation carve of the training data):
        outer_bags=4, interactions=10 pairwise terms, max_bins=256, random_state=42. The feature
        filter keeps EBM tractable on 2 CPU threads (an EBM over all 4^6=4096 k-mers is not).

The composition-only arms use the 20-d composition signature alone with the same family.

ENGINEERING CHANGES (2026-09-26 relaunch; no model setting changed): the user raised the CPU
allowance to 3 threads, so (i) the paper's LightGBM runs with n_jobs=3 (deterministic mode ->
bit-identical to n_jobs=1; verified: cohn MCC 0.461223 / AUROC 0.804649 == results.csv), (ii) the
12 LR grid fits run in <=3 worker PROCESSES (loky; processes, not threads, because liblinear's
L1 solver draws from the C-library global rand() state -- each process has its own, so the fit is
identical to the serial one), (iii) the EBM uses n_jobs=3 (bags are seeded by random_state).
fit_* additionally return a per-feature IMPORTANCE vector (lgbm: total split gain; lr: sum over
classes of |coef| on standardized features; ebm: term importance of each main-effect term, 0 for
features the EBM did not receive) for the post hoc attribution analysis.
"""
import os
import sys
import time
import warnings

import numpy as np
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import _common as C  # noqa: E402

SEED = 42
LR_C_GRID = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
# POST HOC (2026-09-26, added after the pre-registration, used only by model "lrx"): the same
# grid extended upward, run where the pre-registered LR picked the upper grid edge C=0.1.
LR_C_GRID_EXT = LR_C_GRID + (3e-1, 1.0, 3.0, 10.0)
LR_L1_RATIOS = (1.0, 0.0)          # 1.0 = L1 (lasso), 0.0 = L2 (ridge); L1 listed first
VAL_FRAC = 0.20
EBM_M = 128
EBM_PARAMS = dict(outer_bags=4, interactions=10, max_bins=256, random_state=SEED)
COMP_NAMES = [f"mono_{b}" for b in "ACGT"] + [f"di_{a}{b}" for a in "ACGT" for b in "ACGT"]


# ------------------------------------------------------------------ metrics (fast, exact)
def mcc_from_pred(y, pred, n_classes):
    """Multiclass (Gorodkin) MCC from a confusion matrix; identical to sklearn's
    matthews_corrcoef (returns 0.0 when the denominator is 0)."""
    y = np.asarray(y, dtype=np.int64); pred = np.asarray(pred, dtype=np.int64)
    cm = np.bincount(y * n_classes + pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    t = cm.sum(1).astype(np.float64); p = cm.sum(0).astype(np.float64)
    c = float(np.trace(cm)); s = float(cm.sum())
    num = c * s - float(np.dot(t, p))
    den = np.sqrt((s * s - float(np.dot(p, p))) * (s * s - float(np.dot(t, t))))
    return float(num / den) if den > 0 else 0.0


def auroc_binary(y, score):
    """Rank-based (Mann-Whitney) AUROC with average ranks for ties == sklearn roc_auc_score."""
    from scipy.stats import rankdata
    y = np.asarray(y)
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auroc(y, proba):
    """Binary: AUROC of P(class 1). Multiclass: one-vs-rest macro (as the paper)."""
    if proba.shape[1] == 2:
        return auroc_binary(y, proba[:, 1])
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


# ------------------------------------------------------------------ features
def ebm_select(Xtr_sparse, ytr, m=EBM_M):
    """Indices of the m columns with the largest ANOVA F on the TRAINING split (stable ties)."""
    from sklearn.feature_selection import f_classif
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F, _ = f_classif(Xtr_sparse, ytr)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0)
    m = min(m, Xtr_sparse.shape[1])
    return np.sort(np.argsort(-F, kind="stable")[:m])


def ebm_design(Xtr_sparse, Xte_sparse, comp_tr, comp_te, ytr, names, m=EBM_M):
    """Dense EBM design = [20-d composition | top-m k-mer columns by train-only F]."""
    idx = ebm_select(Xtr_sparse, ytr, m)
    Xtr = np.hstack([comp_tr.astype(np.float64), Xtr_sparse[:, idx].toarray().astype(np.float64)])
    Xte = np.hstack([comp_te.astype(np.float64), Xte_sparse[:, idx].toarray().astype(np.float64)])
    fn = COMP_NAMES + [names[i] for i in idx]
    return Xtr, Xte, fn


def ebm_design_pos(Xtr_sparse, Xte_sparse, comp_tr, comp_te, ytr, names, n_global, m_global=EBM_M, m_bin=EBM_M):
    """POST HOC P9 (fix round 1, 2026-09-27; model "ebmpos", Route-2 B=8 arm only). The pre-registered EBM
    filter ranks the global and the position-binned columns of the B=8 design together, and on several
    tasks admitted no binned column at all (B8 design == B1 design). Here the two blocks are ranked
    SEPARATELY by train-only ANOVA F: dense design = [20-d composition | top-m_global GLOBAL k-mer columns |
    top-m_bin POSITION-BINNED columns]. The global block of positional_sweep.featurize_arm is exactly
    featurize.kmer_spectrum (the B=1 features) and f_classif is per-column, so the global selection equals
    the pre-registered EBM's B=1 selection and this design is a strict SUPERSET of the EBM B=1 design.
    Returns (Xtr, Xte, feature_names, selected column indices)."""
    Xtr_sparse = sparse.csr_matrix(Xtr_sparse); Xte_sparse = sparse.csr_matrix(Xte_sparse)
    ig = ebm_select(Xtr_sparse[:, :n_global], ytr, m_global)
    ib = n_global + ebm_select(Xtr_sparse[:, n_global:], ytr, m_bin)
    idx = np.concatenate([ig, ib])
    Xtr = np.hstack([comp_tr.astype(np.float64), Xtr_sparse[:, idx].toarray().astype(np.float64)])
    Xte = np.hstack([comp_te.astype(np.float64), Xte_sparse[:, idx].toarray().astype(np.float64)])
    fn = COMP_NAMES + [names[i] for i in idx]
    return Xtr, Xte, fn, idx


# ------------------------------------------------------------------ model fits
def fit_lgbm(Xtr, ytr, n_jobs=1, seed=SEED):
    m = C.gbmodels.build_model("lgbm", seed, len(np.unique(ytr)))
    m.set_params(n_jobs=n_jobs)
    t0 = time.time(); m.fit(Xtr, ytr)
    return m, dict(fit_s=round(time.time() - t0, 1))


def lgbm_importance(m):
    return m.booster_.feature_importance(importance_type="gain").astype(np.float64)


def lr_importance(coefs):
    return np.abs(np.asarray(coefs, np.float64)).sum(0)


def ebm_importance(m, n_design_features):
    """Importance per DESIGN column (main-effect terms only; interaction terms are not split)."""
    imp = np.zeros(n_design_features)
    try:
        ti = np.asarray(m.term_importances(), np.float64)
        for t, feats in enumerate(m.term_features_):
            if len(feats) == 1:
                imp[feats[0]] = ti[t]
    except Exception:  # noqa: BLE001
        pass
    return imp


def _grid_fit(args):
    """One (penalty, C) fit on the carve; returns the validation criterion (process-safe)."""
    from sklearn.exceptions import ConvergenceWarning
    Str, ytr_, Sva, yva, c, l1, nc, seed, criterion = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = _make_lr(c, l1, nc, seed).fit(Str, ytr_)
    proba = m.predict_proba(Sva)
    if criterion == "auroc":
        v = auroc(yva, proba)
        return 0.5 if v != v else v
    pred = m.classes_[np.argmax(proba, axis=1)]
    return mcc_from_pred(yva, pred, nc)


def _make_lr(C_, l1_ratio, n_classes, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    base = LogisticRegression(C=C_, l1_ratio=l1_ratio, solver="liblinear", max_iter=2000,
                              random_state=seed)
    return base if n_classes == 2 else OneVsRestClassifier(base)


def _scale(Xtr, Xte_list):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler(with_mean=not sparse.issparse(Xtr))
    Str = sc.fit_transform(Xtr)
    return sc, Str, [sc.transform(X) for X in Xte_list]


def fit_lr_tuned(Xtr, ytr, seed=SEED, grid_c=LR_C_GRID, l1_ratios=LR_L1_RATIOS, n_jobs=1,
                 criterion="mcc"):
    """Tune (penalty, C) on a stratified 80/20 carve of TRAIN by validation MCC, refit on all of
    TRAIN. Returns (predict_proba callable, info dict incl. the full validation grid, model, coefs).
    n_jobs>1 evaluates the grid in worker processes (identical results; see module docstring).
    criterion="auroc" (POST HOC, model "lra" only) tunes by validation AUROC instead of MCC."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import StratifiedShuffleSplit
    nc = len(np.unique(ytr))
    tr, va = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC, random_state=seed)
                  .split(np.zeros(len(ytr)), ytr))
    _, Str, (Sva,) = _scale(Xtr[tr], [Xtr[va]])
    combos = [(l1, c) for l1 in l1_ratios for c in grid_c]
    t0 = time.time()
    jobs = [(Str, ytr[tr], Sva, ytr[va], c, l1, nc, seed, criterion) for (l1, c) in combos]
    if n_jobs > 1:
        from joblib import Parallel, delayed
        vals = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_grid_fit)(j) for j in jobs)
    else:
        vals = [_grid_fit(j) for j in jobs]
    grid = []
    best = (-2.0, None)
    for (l1, c), v in zip(combos, vals):
        grid.append(dict(penalty="l1" if l1 == 1.0 else "l2", C=c, **{f"val_{criterion}": round(v, 4)}))
        if v > best[0]:                           # strict > : ties keep the earlier (sparser) entry
            best = (v, (l1, c))
    l1, c = best[1]
    sc, Sfull, _ = _scale(Xtr, [])
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always", ConvergenceWarning)
        m = _make_lr(c, l1, nc, seed).fit(Sfull, ytr)
    conv_warn = any(issubclass(w.category, ConvergenceWarning) for w in wlist)
    coefs = (m.coef_ if nc == 2 else np.vstack([e.coef_ for e in m.estimators_]))
    info = dict(penalty="l1" if l1 == 1.0 else "l2", C=c, criterion=criterion,
                **{f"val_{criterion}": round(best[0], 4)}, degenerate_all_zero=bool(not np.any(coefs)),
                c_at_grid_edge=bool(c in (grid_c[0], grid_c[-1])),
                c_at_upper_edge=bool(c == grid_c[-1]),
                n_nonzero_coef=int((np.abs(coefs).max(0) > 0).sum()), n_features=int(Xtr.shape[1]),
                refit_convergence_warning=bool(conv_warn), tune_s=round(time.time() - t0, 1),
                grid=grid)

    def predict_proba(Xte):
        return m.predict_proba(sc.transform(Xte))
    return predict_proba, info, m, coefs


def fit_ebm(Xtr, ytr, n_jobs=1, seed=SEED, feature_names=None):
    """Fixed-setting EBM. If pairwise interactions are unsupported (multiclass on some versions),
    fall back to interactions=0 and record it."""
    from interpret.glassbox import ExplainableBoostingClassifier
    params = dict(EBM_PARAMS, random_state=seed)
    t0 = time.time()
    try:
        m = ExplainableBoostingClassifier(feature_names=feature_names, n_jobs=n_jobs, **params)
        m.fit(Xtr, ytr)
        inter = params["interactions"]
    except Exception as e:  # noqa: BLE001
        if "interaction" not in str(e).lower():
            raise
        params["interactions"] = 0
        m = ExplainableBoostingClassifier(feature_names=feature_names, n_jobs=n_jobs, **params)
        m.fit(Xtr, ytr)
        inter = 0
    info = dict(fit_s=round(time.time() - t0, 1), interactions=inter, n_features=int(Xtr.shape[1]),
                n_terms=len(m.term_names_))
    try:
        imp = np.asarray(m.term_importances())
        order = np.argsort(-imp)[:15]
        info["top_terms"] = [(m.term_names_[i], round(float(imp[i]), 5)) for i in order]
    except Exception:  # noqa: BLE001
        pass
    return m, info
