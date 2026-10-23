#!/usr/bin/env python3
"""Reviewer 2, comment 3 -- statistics + route labels from the alt-model units (analysis half of
run_alt_model.py; the PRE-REGISTERED plan and the POST HOC addendum live in that file's docstring and
are implemented here verbatim; every post hoc quantity is marked `posthoc` in the outputs).

Inputs: external/alt_model/units/<task>__<arm>__<feat>__<model>.{json,npz} (run_alt_model.py),
external/alt_model/dose_<model>.csv (run_dose_alt.py), results/upgrades/alt_model_gate.csv
(run_gate.py; model-free), results/cross_suite_summary.csv (model-free discriminability and the
TF-motif task flag), results/upgrades/presence_null.csv (#enriched TFs used by the paper).

Fix round 1 (P9/P10 of the run_alt_model.py docstring): EBM-pos Route-2 rows (B1 comparator = the ebm B1
unit), per-task binned-feature counts, Route-1 on uncapped MCCs, the two-level paired bootstrap for the
size-controlled Route-3 readouts plus random-arm MCC, post hoc contamination criteria in the labels.
Fix round 2 (P11): overlap design on every gate-positive task + ocr (uncapped units), the post hoc overlap
labelling scheme, CIs for the size share of the MCC drop (flag renamed), Route-1 uncapped bookkeeping and
the Spearman t-approximation p.

Statistics (as in the paper): 95% percentile bootstrap, 1000 resamples, test sequence = unit,
np.random.RandomState(42).randint(0, n, n) per resample (implemented as multiplicity weights, which
is algebraically identical to indexing), paired where two arms share a test set, independent draws
otherwise. Task-level Spearman: BC bootstrap over tasks, 10,000 resamples, seed 42 + permutation p
(presence_null functions). Dose slopes: run_disc_slopes.boot_slope_ci (rows, 1000, seed 42).

Outputs (results/upgrades/): alt_model_units.csv, alt_model_bench.csv, alt_model_route2.csv,
alt_model_route3_prereg.csv, alt_model_route3_sizecontrolled.csv, alt_model_route3_contrast.csv,
alt_model_route3_shuffled.csv, alt_model_route3_overlap{,_contrast}.csv (post hoc P8), alt_model_route1.csv, alt_model_dose.csv, alt_model_attribution.csv,
alt_model_labels.csv, alt_model_agreement.csv, alt_model_checks.csv, alt_model_interpretation.txt.
CLI: python analyze_alt_model.py [--quick]   (--quick = 200 resamples, for development only)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.abspath(os.path.join(HERE, ".."))
for _d in (UP, HERE, os.path.join(UP, "presence_null"), os.path.join(UP, "disc_dose")):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import _common as C                                   # noqa: E402
import run_alt_model as W                             # noqa: E402
from cache_seqs import load_cached                    # noqa: E402

R = C.RESULTS_DIR
UNITS = W.UNIT_DIR
MODELS = ["lgbm", "lr", "ebm"]                        # lgbm = the paper's model (reference)
ALT = ["lr", "ebm"]
MODEL_NAME = {"lgbm": "LightGBM (paper)", "lr": "L1/L2 logistic regression", "ebm": "EBM (GA2M)",
              "lrx": "LR, extended C grid (post hoc)", "lra": "LR, AUROC-tuned (post hoc)",
              "ebm512": "EBM with 512 k-mers (post hoc)",
              "ebmpos": "EBM, B=8 arm with binned k-mers guaranteed (post hoc P9)"}
B = 1000
SEED = 42
SOLVED_CUT = 0.70
POS_CUT = 0.20
COMP_FRAC_CUT = 0.80
BINARY = [t for t in W.TASKS if t not in W.MULTICLASS]
HOMOLOGY_INFLATED = {"human_nontata_promoters", "nt_enhancers"}      # paper: dropped for n=9
CONTAMINATED = ["human_enhancers_cohn", "nt_enhancers"]
CONTROL = "drosophila_enhancers_stark"
# the paper's route labels
V2_LABEL = {  # camera-ready v2 Table 1 (paper_psb.tex working tree) = pre-registered target
    "demo_human_or_worm": "solved", "human_ensembl_regulatory": "solved", "nt_promoter_all": "solved",
    "nt_promoter_no_tata": "solved", "human_nontata_promoters": "solved", "nt_promoter_tata": "solved",
    "demo_coding_vs_intergenomic_seqs": "solved", "human_enhancers_ensembl": "shared-motif",
    "dummy_mouse_enhancers_ensembl": "shared-motif", "nt_enhancers": "contamination",
    "nt_enhancers_types": "shared-motif", "human_enhancers_cohn": "contamination",
    "human_ocr_ensembl": "shared-motif", "drosophila_enhancers_stark": "shared-motif",
    "nt_splice_sites_all": "positional"}
BIBM_LABEL = {  # submitted BIBM Table 1 (bibm/paper_bibm_submitted.tex; current camera-ready base)
    "demo_human_or_worm": "composition", "human_ensembl_regulatory": "solved", "nt_promoter_all": "solved",
    "nt_promoter_no_tata": "solved", "human_nontata_promoters": "solved", "nt_promoter_tata": "solved",
    "demo_coding_vs_intergenomic_seqs": "composition", "human_enhancers_ensembl": "contamination (weak)",
    "dummy_mouse_enhancers_ensembl": "composition", "nt_enhancers": "contamination",
    "nt_enhancers_types": "shared-motif", "human_enhancers_cohn": "contamination",
    "human_ocr_ensembl": "shared-motif", "drosophila_enhancers_stark": "shared-motif",
    "nt_splice_sites_all": "positional"}
STRONG_EXCESS = 0.15
PARENT = {"lrx": "lr", "lra": "lr", "ebm512": "ebm", "ebmpos": "ebm"}   # variant -> model supplying arms the variant was not run on
BENCH_PARENT = {"ebmpos": "ebm"}   # P9 variant has no benchmark units of its own (only the Route-2 B=8 arm)
B1_FROM = {"ebmpos": "ebm"}        # P9: the B=1 comparator is the pre-registered EBM B1 unit (identical design)
LABEL_MODELS = ["lgbm", "lr", "ebm", "lrx", "ebm512", "ebmpos"]   # lra is an AUROC-readout variant only (no labels)
# BIBM map: 'contamination' vs 'contamination (weak)' (homology_recluster rule)


# ======================================================================= unit I/O
_U = {}


def unit(task, arm, feat, model):
    """Dict(json..., proba, y, te_idx, importance) or None if the unit does not exist."""
    key = (task, arm, feat, model)
    if key in _U:
        return _U[key]
    p = W.unit_path(task, arm, feat, model)
    if not (os.path.exists(p + ".json") and os.path.exists(p + ".npz")):
        _U[key] = None
        return None
    d = json.load(open(p + ".json"))
    z = np.load(p + ".npz", allow_pickle=False)
    d["proba"] = z["proba"].astype(np.float64)
    d["y"] = z["y"].astype(np.int64)
    d["classes"] = z["classes"]
    d["te_idx"] = z["te_idx"] if "te_idx" in z.files else None
    d["importance"] = z["importance"].astype(np.float64) if "importance" in z.files else None
    assert np.array_equal(d["classes"], np.arange(d["proba"].shape[1])), f"{key}: classes {d['classes']}"
    _U[key] = d
    return d


# ======================================================================= fast weighted metrics
def mcc_w(y, pred, nc, w=None):
    cm = np.bincount(y * nc + pred, weights=w, minlength=nc * nc).reshape(nc, nc)
    t = cm.sum(1); p = cm.sum(0); c = float(np.trace(cm)); s = float(cm.sum())
    num = c * s - float(t @ p)
    den = np.sqrt((s * s - float(p @ p)) * (s * s - float(t @ t)))
    return float(num / den) if den > 0 else 0.0


class Scorer:
    """MCC (argmax) and AUROC (binary: P(class1); multiclass: OvR macro) under resampling weights.
    AUROC via tie-grouped Mann-Whitney sums (== sklearn roc_auc_score, average ties)."""

    def __init__(self, y, proba, mask=None):
        if mask is not None:
            y, proba = y[mask], proba[mask]
        self.y = y; self.n = len(y); self.nc = proba.shape[1]
        self.pred = np.argmax(proba, axis=1)
        cols = [(y == 1, proba[:, 1])] if self.nc == 2 else [(y == c, proba[:, c]) for c in range(self.nc)]
        self.parts = []
        for pos, s in cols:
            _, inv = np.unique(s, return_inverse=True)
            self.parts.append((pos, inv, int(inv.max()) + 1))

    def mcc(self, w=None):
        return mcc_w(self.y, self.pred, self.nc, w)

    def auroc(self, w=None):
        vals = []
        for pos, inv, G in self.parts:
            ww = np.ones(self.n) if w is None else w
            P = np.bincount(inv[pos], weights=ww[pos], minlength=G)
            N = np.bincount(inv[~pos], weights=ww[~pos], minlength=G)
            tp, tn = P.sum(), N.sum()
            if tp == 0 or tn == 0:
                return float("nan")
            below = np.cumsum(N) - N
            vals.append(float((P * (below + 0.5 * N)).sum() / (tp * tn)))
        return float(np.mean(vals))

    def classes_present(self, w):
        return len(np.unique(self.y[w > 0])) == self.nc


def weights(rng, n):
    return np.bincount(rng.randint(0, n, n), minlength=n).astype(np.float64)


def pct(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def boot_single(sc, nboot=None):
    nboot = nboot or B
    rng = np.random.RandomState(SEED)
    mm = np.full(nboot, np.nan); aa = np.full(nboot, np.nan)
    for b in range(nboot):
        w = weights(rng, sc.n)
        if not sc.classes_present(w):
            continue
        mm[b] = sc.mcc(w); aa[b] = sc.auroc(w)
    return pct(mm), pct(aa), mm, aa


def boot_paired(sa, sb, fn, nboot=None):
    """fn(sa, sb, w) on the SAME resample of a shared test set."""
    nboot = nboot or B
    assert sa.n == sb.n and np.array_equal(sa.y, sb.y), "paired arms must share the test set"
    rng = np.random.RandomState(SEED)
    out = np.full(nboot, np.nan)
    for b in range(nboot):
        w = weights(rng, sa.n)
        if not sa.classes_present(w):
            continue
        out[b] = fn(sa, sb, w)
    return out


def boot_unpaired(sa, sb, fn, nboot=None):
    """fn(sa, wa, sb, wb) with INDEPENDENT resamples of two different test sets."""
    nboot = nboot or B
    rng = np.random.RandomState(SEED)
    out = np.full(nboot, np.nan)
    for b in range(nboot):
        wa = weights(rng, sa.n); wb = weights(rng, sb.n)
        if not (sa.classes_present(wa) and sb.classes_present(wb)):
            continue
        out[b] = fn(sa, wa, sb, wb)
    return out


def ci_str(v, lo, hi, d=3):
    return f"{v:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


# ======================================================================= 0. unit table
def unit_table():
    rows = []
    for f in sorted(os.listdir(UNITS)):
        if not f.endswith(".json"):
            continue
        task, arm, feat, model = f[:-5].split("__")
        u = unit(task, arm, feat, model)
        if u is None:
            continue
        sc = Scorer(u["y"], u["proba"])
        (mlo, mhi), (alo, ahi), _, _ = boot_single(sc)
        info = u.get("info", {})
        rows.append(dict(task=task, arm=arm, feat=feat, model=model, k=u["k"], posthoc=u.get("posthoc", False),
                         n_train=u["n_train"], n_test=u["n_test"], train_capped=u.get("train_capped"),
                         test_capped=u.get("test_capped"), mcc=round(sc.mcc(), 6), mcc_lo=round(mlo, 6),
                         mcc_hi=round(mhi, 6), auroc=round(sc.auroc(), 6), auroc_lo=round(alo, 6),
                         auroc_hi=round(ahi, 6), mcc_json=u["mcc"], auroc_json=u["auroc"],
                         lr_penalty=info.get("penalty", ""), lr_C=info.get("C", ""),
                         lr_c_upper_edge=info.get("c_at_upper_edge", ""),
                         lr_c_at_grid_edge=info.get("c_at_grid_edge", ""),
                         lr_degenerate=info.get("degenerate_all_zero", ""),
                         lr_nnz=info.get("n_nonzero_coef", ""), ebm_interactions=info.get("interactions", ""),
                         aliased_from=u.get("aliased_from", ""), seconds=u.get("seconds"),
                         test_fp=u.get("test_fp")))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(R, "alt_model_units.csv"), index=False)
    return df


def U(ut, task, arm, feat, model):
    r = ut[(ut.task == task) & (ut.arm == arm) & (ut.feat == feat) & (ut.model == model)]
    return None if r.empty else r.iloc[0]


# ======================================================================= 1. benchmark + comp fraction
def bench_table(ut, models, arm="bench"):
    rows = []
    for t in W.TASKS:
        for m in models:
            r = U(ut, t, arm, "kmer", m)
            if r is None:
                continue
            row = dict(task=t, model=m, arm=arm, k=r.k, n_train=r.n_train, n_test=r.n_test,
                       train_capped=r.train_capped, mcc=r.mcc, mcc_lo=r.mcc_lo, mcc_hi=r.mcc_hi,
                       auroc=r.auroc, auroc_lo=r.auroc_lo, auroc_hi=r.auroc_hi,
                       tier=("solved" if r.mcc >= 0.7 else "partial" if r.mcc >= 0.5 else "hard"),
                       solved=bool(r.mcc >= SOLVED_CUT),
                       mcc_ci_excludes_cut=bool(r.mcc_lo > SOLVED_CUT or r.mcc_hi < SOLVED_CUT))
            uk = unit(t, arm, "kmer", m)
            sk = Scorer(uk["y"], uk["proba"])
            if m != "lgbm" and unit(t, arm, "kmer", "lgbm") is not None:
                ul = unit(t, arm, "kmer", "lgbm"); sl = Scorer(ul["y"], ul["proba"])
                d = boot_paired(sk, sl, lambda a, b, w: a.mcc(w) - b.mcc(w))
                row.update(dmcc_vs_lgbm=round(sk.mcc() - sl.mcc(), 6), dmcc_vs_lgbm_lo=round(pct(d)[0], 6),
                           dmcc_vs_lgbm_hi=round(pct(d)[1], 6))
            uc = unit(t, arm, "comp", m)
            if uc is None and m == "ebm512":          # the composition-only EBM has no k-mers: identical to ebm
                uc = unit(t, arm, "comp", "ebm")
            if uc is not None:
                sc = Scorer(uc["y"], uc["proba"])
                row.update(comp_mcc=round(sc.mcc(), 6), comp_auroc=round(sc.auroc(), 6))

                def frac(a, b, w):
                    den = a.auroc(w) - 0.5
                    return (b.auroc(w) - 0.5) / den if den > 0 else np.nan
                fr = boot_paired(sk, sc, frac)
                f0 = frac(sk, sc, None)
                row.update(comp_fraction=round(f0, 6), comp_fraction_lo=round(pct(fr)[0], 6),
                           comp_fraction_hi=round(pct(fr)[1], 6))
            rows.append(row)
    return pd.DataFrame(rows)


# ======================================================================= 2. Route 2
def _bin_counts(u):
    """(# position-binned k-mer columns the B=8 model received, # global k-mer columns) -- the EBM gets a
    train-ANOVA-filtered subset (info.ebm_kmer_features); LightGBM/LR get every column."""
    names = u.get("info", {}).get("ebm_kmer_features")
    if names is None:
        n_glob = 4 ** int(u["k"])
        return int(u["n_features"]) - n_glob, n_glob
    nb = sum(1 for n_ in names if str(n_).startswith("bin"))
    return nb, len(names) - nb


def route2_table(ut, models, arm="pos"):
    rows = []
    for t in W.TASKS:
        for m in models:
            a, b = unit(t, arm, "B1", B1_FROM.get(m, m)), unit(t, arm, "B8", m)
            if a is None or b is None:
                continue
            sa, sb = Scorer(a["y"], a["proba"]), Scorer(b["y"], b["proba"])
            d = boot_paired(sb, sa, lambda x, y_, w: x.mcc(w) - y_.mcc(w))
            lo, hi = pct(d)
            dd = sb.mcc() - sa.mcc()
            nbin, nglob = _bin_counts(b)
            rows.append(dict(task=t, model=m, arm=arm, k=a["k"], n_train=a["n_train"], n_test=a["n_test"],
                             mcc_B1=round(sa.mcc(), 6), mcc_B8=round(sb.mcc(), 6), delta_mcc=round(dd, 6),
                             delta_lo=round(lo, 6), delta_hi=round(hi, 6),
                             delta_ci_excludes_0=bool(lo > 0 or hi < 0),
                             positional_rule=bool(dd > POS_CUT and lo > 0),
                             b1_aliased=bool(a.get("aliased_from")),
                             b1_from_model=B1_FROM.get(m, m), posthoc=bool(m in B1_FROM or arm == "pos_full"),
                             n_bin_features_B8=nbin, n_global_kmer_features_B8=nglob,
                             b8_equals_b1_by_construction=bool(nbin == 0)))
    return pd.DataFrame(rows)


# ======================================================================= 3. Route 3 (pre-registered)
def route3_prereg(ut, models):
    rows = []
    for t in W.EQ_SPLITS:
        for m in models:
            ek, ec, ok = unit(t, "eq", "kmer", m), unit(t, "eq", "comp", m), unit(t, "bench", "kmer", m)
            if ek is None or ec is None or ok is None:
                continue
            sek, sec, sok = Scorer(ek["y"], ek["proba"]), Scorer(ec["y"], ec["proba"]), Scorer(ok["y"], ok["proba"])
            mg = boot_paired(sek, sec, lambda a, b, w: a.mcc(w) - b.mcc(w))
            dm = boot_unpaired(sek, sok, lambda a, wa, b, wb: a.mcc(wa) - b.mcc(wb))
            da = boot_unpaired(sek, sok, lambda a, wa, b, wb: a.auroc(wa) - b.auroc(wb))
            rows.append(dict(task=t, model=m, eq_n_test=sek.n, eq_pos_frac=round(float(sek.y.mean()), 4),
                             orig_mcc=round(sok.mcc(), 6), eq_kmer_mcc=round(sek.mcc(), 6),
                             eq_comp_mcc=round(sec.mcc(), 6),
                             eq_margin_mcc=round(sek.mcc() - sec.mcc(), 6), eq_margin_lo=round(pct(mg)[0], 6),
                             eq_margin_hi=round(pct(mg)[1], 6), collapses_prereg=bool(pct(mg)[0] <= 0),
                             eq_minus_orig_mcc=round(sek.mcc() - sok.mcc(), 6),
                             eq_minus_orig_mcc_lo=round(pct(dm)[0], 6), eq_minus_orig_mcc_hi=round(pct(dm)[1], 6),
                             orig_auroc=round(sok.auroc(), 6), eq_kmer_auroc=round(sek.auroc(), 6),
                             eq_comp_auroc=round(sec.auroc(), 6),
                             eq_minus_orig_auroc=round(sek.auroc() - sok.auroc(), 6),
                             eq_minus_orig_auroc_lo=round(pct(da)[0], 6), eq_minus_orig_auroc_hi=round(pct(da)[1], 6),
                             lr_eq_comp_degenerate=ec.get("info", {}).get("degenerate_all_zero", "")))
    return pd.DataFrame(rows)


# ======================================================================= 3b. Route 3 size-controlled (post hoc P1)
_DUP = {}


def nt_dup_mask():
    """Original nt_enhancers test sequences that occur verbatim in its training split."""
    if "nt" not in _DUP:
        tr, ytr, te, yte = load_cached("nt_enhancers")
        trs = set(tr)
        _DUP["nt"] = np.array([s in trs for s in te])
    return _DUP["nt"]


def _mask_for(task, u, leakfree):
    if not leakfree:
        return None
    assert task == "nt_enhancers" and u["te_idx"] is not None
    return ~nt_dup_mask()[u["te_idx"]]


def size_controlled(task, m, feat="kmer", leakfree=False, nboot=None):
    """Within-design matched (eq) vs size-matched random (eqr0..9) AUROC and default-threshold MCC, plus the
    fixed-model design. CIs: two-level PAIRED bootstrap (P10b = psm_sensitivity run_posthoc P7): level 1 =
    the ORIGINAL test sequences (RandomState(42) multiplicity counts; every arm's test set is a subset of the
    original test set, so all arms share each draw), level 2 = the 10 random-negative seeds resampled with
    replacement (RandomState(43)). Point estimates are the plain (unbootstrapped) values."""
    nboot = nboot or B
    eq = unit(task, "eq", feat, m)
    rs = [unit(task, f"eqr{s}", feat, m) for s in range(W.N_RAND)]
    bench = unit(task, "bench", feat, m)
    if eq is None or any(r is None for r in rs) or bench is None:
        return None, None
    y0 = bench["y"]; n0 = len(y0)
    keep0 = (~nt_dup_mask()) if leakfree else np.ones(n0, bool)
    kidx = np.flatnonzero(keep0)
    posn = -np.ones(n0, np.int64); posn[kidx] = np.arange(len(kidx))

    def arm(y, proba, idx):
        """Scorer over the arm's kept test rows + their level-1 positions."""
        idx = np.asarray(idx, np.int64)
        assert np.array_equal(np.asarray(y), y0[idx]), f"{task}/{m}: arm labels differ from the original test"
        sel = keep0[idx]
        return Scorer(np.asarray(y)[sel], np.asarray(proba)[sel]), posn[idx[sel]]

    S_0 = arm(y0, bench["proba"], np.arange(n0))
    S_m = arm(eq["y"], eq["proba"], eq["te_idx"])
    S_r = [arm(r["y"], r["proba"], r["te_idx"]) for r in rs]
    # fixed model: the original-split model scored on the matched / random TEST subsets
    F_m = arm(y0[eq["te_idx"]], bench["proba"][eq["te_idx"]], eq["te_idx"])
    F_r = [arm(y0[r["te_idx"]], bench["proba"][r["te_idx"]], r["te_idx"]) for r in rs]
    A0, Am = S_0[0].auroc(), S_m[0].auroc()
    M0, Mm = S_0[0].mcc(), S_m[0].mcc()
    Ar = np.array([s.auroc() for s, _ in S_r]); Mr = np.array([s.mcc() for s, _ in S_r])
    Fm = F_m[0].auroc(); Fr = np.array([s.auroc() for s, _ in F_r])
    n = len(kidx)
    rng = np.random.RandomState(SEED)
    rsel = np.random.RandomState(SEED + 1).randint(0, W.N_RAND, size=(nboot, W.N_RAND))
    a0b, amb, m0b, mmb, fmb = (np.full(nboot, np.nan) for _ in range(5))
    arb, mrb, frb = (np.full((nboot, W.N_RAND), np.nan) for _ in range(3))
    for b in range(nboot):
        c = np.bincount(rng.randint(0, n, n), minlength=n).astype(np.float64)
        s, p = S_0; a0b[b] = s.auroc(c[p]); m0b[b] = s.mcc(c[p])
        s, p = S_m; amb[b] = s.auroc(c[p]); mmb[b] = s.mcc(c[p])
        s, p = F_m; fmb[b] = s.auroc(c[p])
        for j, (s, p) in enumerate(S_r):
            arb[b, j] = s.auroc(c[p]); mrb[b, j] = s.mcc(c[p])
        for j, (s, p) in enumerate(F_r):
            frb[b, j] = s.auroc(c[p])
    cols = np.arange(nboot)[:, None]
    ar = np.nanmean(arb[cols, rsel], axis=1); mr = np.nanmean(mrb[cols, rsel], axis=1)
    fr = np.nanmean(frb[cols, rsel], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = ar - amb; share = np.where(ar > 0.5, (ar - amb) / (ar - 0.5), np.nan); feff = fr - fmb
        size_b = a0b - ar; mcc_size_b = m0b - mr; mcc_comp_b = mr - mmb
        # P11c: share of the MCC drop (orig -> matched) that size-matched random negatives already produce
        mtot_b = m0b - mmb
        mshare_b = np.where(np.abs(mtot_b) > 1e-9, mcc_size_b / mtot_b, np.nan)
    arm_ = float(Ar.mean()); mrm = float(Mr.mean())
    mtot = M0 - Mm
    mcl, mch = pct(mcc_comp_b)
    row = dict(task=task, model=m, feat=feat, leakfree_test=leakfree, posthoc=True,
               bootstrap="two-level paired (P10b): original test sequences x random seeds",
               n_test_matched=S_m[0].n, n_neg_test_matched=int((S_m[0].y == 0).sum()),
               n_test_random=S_r[0][0].n, auroc_orig=round(A0, 6), auroc_random_mean=round(arm_, 6),
               auroc_random_min=round(float(Ar.min()), 6), auroc_random_max=round(float(Ar.max()), 6),
               auroc_matched=round(Am, 6), size_effect=round(A0 - arm_, 6),
               size_effect_lo=round(pct(size_b)[0], 6), size_effect_hi=round(pct(size_b)[1], 6),
               comp_effect=round(arm_ - Am, 6), comp_effect_lo=round(pct(eff)[0], 6),
               comp_effect_hi=round(pct(eff)[1], 6),
               matched_below_all_random=bool(Am < Ar.min()),
               share_lost=round((arm_ - Am) / (arm_ - 0.5), 6) if arm_ > 0.5 else np.nan,
               share_lost_lo=round(pct(share)[0], 6), share_lost_hi=round(pct(share)[1], 6),
               fixed_auroc_random_mean=round(float(Fr.mean()), 6), fixed_auroc_matched=round(Fm, 6),
               fixed_comp_effect=round(float(Fr.mean()) - Fm, 6), fixed_comp_effect_lo=round(pct(feff)[0], 6),
               fixed_comp_effect_hi=round(pct(feff)[1], 6),
               # P10c: default-threshold MCC (the pre-registered Route-3 metric) on the same arms
               mcc_orig=round(M0, 6), mcc_random_mean=round(mrm, 6), mcc_random_min=round(float(Mr.min()), 6),
               mcc_random_max=round(float(Mr.max()), 6), mcc_matched=round(Mm, 6),
               mcc_size_drop=round(M0 - mrm, 6), mcc_size_drop_lo=round(pct(mcc_size_b)[0], 6),
               mcc_size_drop_hi=round(pct(mcc_size_b)[1], 6),
               mcc_comp_effect=round(mrm - Mm, 6), mcc_comp_effect_lo=round(mcl, 6),
               mcc_comp_effect_hi=round(mch, 6),
               mcc_total_drop=round(mtot, 6),
               mcc_share_of_drop_from_size=round((M0 - mrm) / mtot, 6) if abs(mtot) > 1e-9 else np.nan,
               mcc_share_of_drop_from_size_lo=round(pct(mshare_b)[0], 6),
               mcc_share_of_drop_from_size_hi=round(pct(mshare_b)[1], 6),
               # P11c (renamed from P10c's random_reproduces_mcc_collapse): only whether the CI includes 0
               mcc_comp_effect_ci_includes_0=bool(mcl <= 0 <= mch))
    return row, share


def route3_size_table(models):
    rows, shares = [], {}
    for t in W.EQ_SPLITS:
        for m in models:
            for feat in ("kmer", "comp"):
                for lf in ([False, True] if t == "nt_enhancers" else [False]):
                    row, sh = size_controlled(t, m, feat, lf)
                    if row is None:
                        continue
                    rows.append(row)
                    shares[(t, m, feat, lf)] = sh
    return pd.DataFrame(rows), shares


def route3_contrast(shares, models, r3s=None):
    """Share of above-chance AUROC lost: contaminated minus clean control. The two datasets have different
    test sets, so their (two-level) bootstrap draws are independent; both use seeds 42/43 per dataset.
    Because both datasets use the same RandomState(43) level-2 stream, draw b of the task and draw b of
    the control would share their seed indices; the fixed permutation (RandomState(49)) of the control's
    draws below pairs each task draw with a different control draw. The arms of the two datasets are
    unrelated random draws, so this pairing choice does not affect validity (verifier note, fix round 2)."""
    rows = []
    for m in models:
        c = shares.get((CONTROL, m, "kmer", False))
        if c is None:
            continue
        for t, lf in [("human_enhancers_cohn", False), ("nt_enhancers", False), ("nt_enhancers", True),
                      ("human_enhancers_ensembl", False), ("human_ocr_ensembl", False)]:
            s = shares.get((t, m, "kmer", lf))
            if s is None:
                continue
            # the two datasets' draws come from the same seeds: decouple them by a fixed permutation of the
            # control's draws (independent pairing of two independent bootstraps)
            d = s - c[np.random.RandomState(SEED + 7).permutation(len(c))]
            pt = pt_c = np.nan
            if r3s is not None and not r3s.empty:
                a = r3s[(r3s.task == t) & (r3s.model == m) & (r3s.feat == "kmer") & (r3s.leakfree_test == lf)]
                bq = r3s[(r3s.task == CONTROL) & (r3s.model == m) & (r3s.feat == "kmer")]
                if not a.empty and not bq.empty:
                    pt, pt_c = float(a.share_lost.iloc[0]), float(bq.share_lost.iloc[0])
            rows.append(dict(task=t, control=CONTROL, model=m, leakfree_test=lf, posthoc=True,
                             share_task=round(pt, 6), share_control=round(pt_c, 6), diff=round(pt - pt_c, 6),
                             share_task_boot_median=round(float(np.nanmedian(s)), 6),
                             share_control_boot_median=round(float(np.nanmedian(c)), 6),
                             diff_boot_median=round(float(np.nanmedian(d)), 6),
                             diff_lo=round(pct(d)[0], 6), diff_hi=round(pct(d)[1], 6),
                             task_loses_more=bool(pct(d)[0] > 0)))
    return pd.DataFrame(rows)


# ======================================================================= 3b'. overlap weighting (post hoc P8)
_PROP = {}


def test_overlap_weights(task):
    """psm_sensitivity's removal-free design, reproduced: LR propensity e = P(pos | 20-d composition)
    fit on the pooled ORIGINAL train+test (run_composition_clean.fit_propensity, seed 42); test overlap
    weights (pos 1-e, neg e) normalized to mean 1; 10 placebo weight vectors permuted within class with
    RandomState([30000 + r, 1, class]) (psm's test-split seeds)."""
    if task in _PROP:
        return _PROP[task]
    sys.path.insert(0, os.path.join(UP, "composition_clean")); sys.path.insert(0, os.path.join(UP, "psm_sensitivity"))
    import run_composition_clean as RCC
    import psm_core as P
    tr, ytr, te, yte = load_cached(task)
    ytr = np.asarray(ytr, np.int64); yte = np.asarray(yte, np.int64)
    Ctr = C.comp_signature(tr, ks=(1, 2)).astype(np.float64); Cte = C.comp_signature(te, ks=(1, 2)).astype(np.float64)
    prop = RCC.fit_propensity(np.vstack([Ctr, Cte]), np.r_[ytr, yte], C.SEED)
    e = prop.predict_proba(Cte)[:, 1]
    w = P.overlap_weights(e, yte); w = w / w.mean()
    plac = []
    for r in range(10):
        w2 = w.copy()
        for cl in (0, 1):
            ii = np.where(yte == cl)[0]
            w2[ii] = w[ii][np.random.RandomState([30000 + r, 1, cl]).permutation(len(ii))]
        plac.append(w2)
    _PROP[task] = (yte, w, plac)
    return _PROP[task]


def overlap_unit(task, m, feat="kmer"):
    """(unit, arm) of the ORIGINAL-split model with UNCAPPED training: `bench` where the task was not capped,
    else the post hoc P2 `bench_full` refit (P11a). Both are scored on the full original test set."""
    for arm in ("bench", "bench_full"):
        u = unit(task, arm, feat, m)
        if u is not None and not u.get("train_capped"):
            return u, arm
    return None, None


def overlap_draws(task, m, feat="kmer", leakfree=False, nboot=None):
    nboot = nboot or B
    u, arm = overlap_unit(task, m, feat)
    if u is None:
        return None
    yte, w, plac = test_overlap_weights(task)
    assert np.array_equal(yte, u["y"])
    keep = (~nt_dup_mask()) if leakfree else np.ones(len(yte), bool)
    sc = Scorer(u["y"][keep], u["proba"][keep])
    w = w[keep]; plac = [p[keep] for p in plac]
    aw = sc.auroc(w); ap = float(np.mean([sc.auroc(p) for p in plac]))
    rng = np.random.RandomState(SEED)
    AW = np.empty(nboot); AP = np.empty(nboot)
    for b in range(nboot):
        c = weights(rng, sc.n)
        AW[b] = sc.auroc(c * w); AP[b] = np.mean([sc.auroc(c * p) for p in plac])
    return dict(aw=aw, ap=ap, AW=AW, AP=AP, n=sc.n, arm=arm)


# P8 tasks + P11a extension (gate-positive human_enhancers_ensembl / human_nontata_promoters, and the second
# P1 specificity task human_ocr_ensembl, which is gate-negative)
OVERLAP_TASKS = [("human_enhancers_cohn", False), ("nt_enhancers", False), ("nt_enhancers", True),
                 ("drosophila_enhancers_stark", False), ("human_enhancers_ensembl", False),
                 ("human_ocr_ensembl", False), ("human_nontata_promoters", False)]
OVERLAP_P11 = {"human_enhancers_ensembl", "human_ocr_ensembl", "human_nontata_promoters"}


def route3_overlap_table(models):
    """P8 (+ P11a): overlap-weighted vs placebo AUROC of the ORIGINAL-split model; contrasts = share lost,
    task minus the drosophila control. The contrast pairs draw b of the task with draw b of the control;
    both come from RandomState(42) on DIFFERENT test sets, i.e. independent resamples of unrelated
    sequences (as in the verified P8 rows)."""
    rows, draws = [], {}
    for t, lf in OVERLAP_TASKS:
        for m in models:
            for feat in ("kmer", "comp"):
                d = overlap_draws(t, m, feat, lf)
                if d is None:
                    continue
                with np.errstate(invalid="ignore", divide="ignore"):
                    sh = (d["AP"] - d["AW"]) / (d["AP"] - 0.5)
                draws[(t, m, feat, lf)] = sh
                rows.append(dict(task=t, model=m, feat=feat, leakfree_test=lf, posthoc=True,
                                 posthoc_step="P11a" if t in OVERLAP_P11 else "P8", source_arm=d["arm"],
                                 n_test=d["n"],
                                 auroc_overlap=round(d["aw"], 6), auroc_placebo_mean=round(d["ap"], 6),
                                 drop=round(d["ap"] - d["aw"], 6), drop_lo=round(pct(d["AP"] - d["AW"])[0], 6),
                                 drop_hi=round(pct(d["AP"] - d["AW"])[1], 6),
                                 share_lost=round((d["ap"] - d["aw"]) / (d["ap"] - 0.5), 6),
                                 share_lo=round(pct(sh)[0], 6), share_hi=round(pct(sh)[1], 6),
                                 overlap_auroc_below_chance=bool(d["aw"] < 0.5)))
    df = pd.DataFrame(rows)
    con = []
    for m in models:
        c = draws.get((CONTROL, m, "kmer", False))
        if c is None:
            continue
        for t, lf in OVERLAP_TASKS:
            if t == CONTROL:
                continue
            s_ = draws.get((t, m, "kmer", lf))
            if s_ is None:
                continue
            dd = s_ - c
            r0 = df[(df.task == t) & (df.model == m) & (df.feat == "kmer") & (df.leakfree_test == lf)].iloc[0]
            r1 = df[(df.task == CONTROL) & (df.model == m) & (df.feat == "kmer")].iloc[0]
            con.append(dict(task=t, control=CONTROL, model=m, leakfree_test=lf, design="fixed_model_overlap",
                            posthoc=True, posthoc_step="P11a" if t in OVERLAP_P11 else "P8",
                            source_arm=r0.source_arm, share_task=r0.share_lost, share_control=r1.share_lost,
                            diff=round(r0.share_lost - r1.share_lost, 6), diff_lo=round(pct(dd)[0], 6),
                            diff_hi=round(pct(dd)[1], 6), task_loses_more=bool(pct(dd)[0] > 0),
                            task_loses_less=bool(pct(dd)[1] < 0)))
    return df, pd.DataFrame(con)


# ======================================================================= 3c. shuffled negatives
def shuffled_table(models):
    rows = []
    for t in W.SHUF_TASKS:
        for m in models:
            sk, scp = unit(t, "shuf", "kmer", m), unit(t, "shuf", "comp", m)
            ok, oc = unit(t, "bench", "kmer", m), unit(t, "bench", "comp", m)
            if sk is None or scp is None or ok is None or oc is None:
                continue
            Sk, Sc, Ok, Oc = (Scorer(u["y"], u["proba"]) for u in (sk, scp, ok, oc))
            (_, _), (clo, chi), _, _ = boot_single(Sc)

            def frac(a, b, w):
                den = a.auroc(w) - 0.5
                return (b.auroc(w) - 0.5) / den if den > 0 else np.nan
            fs = boot_paired(Sk, Sc, frac); fo = boot_paired(Ok, Oc, frac)
            rows.append(dict(task=t, model=m, shuf_kmer_mcc=round(Sk.mcc(), 6), shuf_kmer_auroc=round(Sk.auroc(), 6),
                             shuf_comp_mcc=round(Sc.mcc(), 6), shuf_comp_auroc=round(Sc.auroc(), 6),
                             shuf_comp_auroc_lo=round(clo, 6), shuf_comp_auroc_hi=round(chi, 6),
                             shuf_comp_at_chance=bool(clo <= 0.5 <= chi),
                             comp_fraction_orig=round(frac(Ok, Oc, None), 6),
                             comp_fraction_orig_lo=round(pct(fo)[0], 6), comp_fraction_orig_hi=round(pct(fo)[1], 6),
                             comp_fraction_shuf=round(frac(Sk, Sc, None), 6),
                             comp_fraction_shuf_lo=round(pct(fs)[0], 6), comp_fraction_shuf_hi=round(pct(fs)[1], 6),
                             lr_comp_degenerate=scp.get("info", {}).get("degenerate_all_zero", "")))
    return pd.DataFrame(rows)


# ======================================================================= 4. Route 1
def route1_table(bench, models, dose, pn_counts, attr, suffix="", mcc_source="bench (train capped at 50k; pre-registered)",
                 subst=None):
    """Task-level Route-1 statistics. suffix="" = the pre-registered capped MCCs; suffix="_uncapped" (P10a) =
    bench_full MCCs wherever an uncapped refit exists (= the MCCs behind the paper's Table corr for LightGBM).
    P11d: with `subst` (model -> tasks whose MCC was replaced by an uncapped refit) an _uncapped row is written
    only if >= 1 task of its set was substituted (otherwise it would duplicate the capped row)."""
    import run_presence_null as PN
    from scipy.stats import spearmanr as _spr
    cs = pd.read_csv(os.path.join(C.ROOT, "results", "cross_suite_summary.csv"))
    tf = cs[cs["tf_motif_task"] == True]  # noqa: E712
    disc = dict(zip(tf["task"], tf["max_pos_motif_auroc"]))
    rows = []

    def spear(label, xs, ys, m, extra, tasks=None, ncap=None):
        label = label + suffix
        nsub = None
        if subst is not None:
            nsub = int(sum(t_ in subst.get(m, set()) for t_ in (tasks or [])))
            if nsub == 0:
                return
        x = np.asarray(xs, float); y = np.asarray(ys, float)
        rho = PN._rho(x, y)
        jk = np.array([PN._rho(np.delete(x, i), np.delete(y, i)) for i in range(len(x))])
        extra = dict(extra, jackknife_lo=round(float(np.nanmin(jk)), 6), jackknife_hi=round(float(np.nanmax(jk)), 6))
        if tasks is not None:
            extra.update(jackknife_min_drop_task=tasks[int(np.nanargmin(jk))],
                         jackknife_max_drop_task=tasks[int(np.nanargmax(jk))])
        pp, meth, npm = PN.perm_p(x, y, rho, 200000, SEED)
        bs = PN.bc_bootstrap_ci(x, y, rho, 10000, SEED)
        rows.append(dict(analysis=label, model=m, n=len(x), rho=round(rho, 6), p_perm=round(pp, 6),
                         p_perm_method=meth, p_spearman_t_approx=round(float(_spr(x, y).pvalue), 6),
                         boot_lo=round(bs["boot_lo"], 6), boot_hi=round(bs["boot_hi"], 6),
                         ci_excludes_0=bool(bs["boot_lo"] > 0 or bs["boot_hi"] < 0), mcc_source=mcc_source,
                         n_tasks_capped_mcc=ncap, n_tasks_uncapped_substituted=nsub,
                         posthoc=bool(suffix) or "posthoc" in label, **extra))

    def ncapped(mb, ts):
        return int(sum(bool(mb.loc[t, "train_capped"]) for t in ts)) if "train_capped" in mb else None
    for m in models:
        mb = bench[bench.model == m].set_index("task")
        t11 = [t for t in disc if t in mb.index]
        if len(t11) == 11:
            spear("route1_disc_vs_mcc_n11", [disc[t] for t in t11], [mb.loc[t, "mcc"] for t in t11], m, {},
                  t11, ncapped(mb, t11))
            t9 = [t for t in t11 if t not in HOMOLOGY_INFLATED]
            spear("route1_disc_vs_mcc_n9_homology_clean", [disc[t] for t in t9], [mb.loc[t, "mcc"] for t in t9], m, {},
                  t9, ncapped(mb, t9))
            # LOTO OLS MAE (route1_predict procedure)
            from sklearn.linear_model import LinearRegression
            x = np.array([disc[t] for t in t11]); y = np.array([mb.loc[t, "mcc"] for t in t11])
            pred = np.empty(len(x))
            for i in range(len(x)):
                k = np.ones(len(x), bool); k[i] = False
                pred[i] = LinearRegression().fit(x[k].reshape(-1, 1), y[k]).predict(x[i].reshape(1, -1))[0]
            err = np.abs(y - pred); rng = np.random.RandomState(SEED)
            maes = [np.mean(err[rng.randint(0, 11, 11)]) for _ in range(1000)]
            nsub11 = None if subst is None else int(sum(t_ in subst.get(m, set()) for t_ in t11))
            if subst is None or nsub11 > 0:
                rows.append(dict(analysis="route1_loto_mae_n11" + suffix, model=m, n=11, rho=np.nan,
                                 mae=round(float(err.mean()), 6), mae_lo=round(pct(maes)[0], 6),
                                 mae_hi=round(pct(maes)[1], 6), mcc_source=mcc_source,
                                 n_tasks_capped_mcc=ncapped(mb, t11), n_tasks_uncapped_substituted=nsub11,
                                 posthoc=bool(suffix)))
        gb = [t for t in pn_counts if t in mb.index]
        if len(gb) == 9:
            spear("presence_null_gb_n9_paper_counts", [pn_counts[t] for t in gb], [mb.loc[t, "mcc"] for t in gb], m, {},
                  gb, ncapped(mb, gb))
            if attr is not None and not attr.empty:
                a = attr[attr.model == m].set_index("task")
                if all(t in a.index for t in gb):
                    spear("presence_null_gb_n9_own_counts_posthoc", [a.loc[t, "n_enriched"] for t in gb],
                          [mb.loc[t, "mcc"] for t in gb], m, {}, gb, ncapped(mb, gb))
    df = pd.DataFrame(rows)
    pm = df["analysis"].astype(str).str.startswith("presence_null")
    df["presence_null_preserved"] = [(not bool(c)) if p_ else "" for p_, c in zip(pm, df["ci_excludes_0"])]
    return df


def dose_table(models):
    from run_disc_slopes import ols_slope, boot_slope_ci
    from scipy.stats import spearmanr
    rel = pd.read_csv(os.path.join(R, "disc_dose.csv"))
    rows, checks = [], []
    for m in models:
        p = os.path.join(C.ROOT, "external", "alt_model", f"dose_{m}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p).drop_duplicates(["motif", "seed", "p_pos"], keep="last")
        checks.append(dict(check=f"dose_{m}_rows", value=len(d), expected=len(rel), ok=len(d) == len(rel)))
        checks.append(dict(check=f"dose_{m}_d_matches_released", value=int(d["d_matches_released"].sum()),
                           expected=len(d), ok=bool(d["d_matches_released"].all())))
        if m == "lgbm":
            mm = d.merge(rel, on=["motif", "seed", "p_pos"], suffixes=("", "_rel"))
            agree = int((np.round(mm["bench_mcc"], 2) == np.round(mm["bench_mcc_rel"], 2)).sum())
            checks.append(dict(check="dose_lgbm_reproduces_released_mcc_2dp", value=agree, expected=len(mm),
                               ok=agree == len(mm)))
        for motif in ["SP1", "MYC", "TBP", "POOLED"]:
            g = d if motif == "POOLED" else d[d.motif == motif]
            if len(g) < 3:
                continue
            x = g["motif_auroc_d"].to_numpy(float); y = g["bench_mcc"].to_numpy(float)
            lo, hi = boot_slope_ci(x, y, SEED, B)
            rows.append(dict(model=m, motif=motif, n_points=len(x), complete=bool(len(d) == len(rel)),
                             slope=round(ols_slope(x, y), 4),
                             slope_lo=round(lo, 4), slope_hi=round(hi, 4),
                             spearman=round(float(spearmanr(x, y).statistic), 4),
                             slope_ci_excludes_0=bool(lo > 0 or hi < 0),
                             n_lr_degenerate=int(g.get("lr_degenerate", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())))
    return pd.DataFrame(rows), checks


# ======================================================================= 5. attribution (post hoc P5)
def attribution_table(models):
    import motif_jaspar
    import motif_match
    import run_interpretability as RI
    rows = []
    cache = os.path.join(C.ROOT, "external", "alt_model", "motif_scores")
    pssm = {}
    zs = {}
    for t in W.TASKS:
        for m in models:
            u = unit(t, "bench", "kmer", m)
            if u is None or u.get("importance") is None:
                continue
            coll = RI.collection_for(t); k = u["k"]
            if coll not in pssm:
                pssm[coll] = motif_jaspar.load_pssms(coll)
            M, usable = motif_match.tf_score_matrix(k, pssm[coll], cache_dir=cache,
                                                    tag=f"{coll}_{motif_jaspar.RELEASE}_pc{motif_jaspar.PSEUDOCOUNT}")
            imp = u["importance"]
            assert len(imp) == M.shape[0], (t, m, len(imp), M.shape)
            z, p = RI.enrichment_z(imp, M)
            fdr = RI.bh_fdr(p)
            enr = np.where((fdr < 0.05) & (z > 0))[0]
            enr = enr[np.argsort(z[enr])[::-1]]
            names = [usable[j]["name"] for j in enr]
            zs[(t, m)] = z
            allrank = [usable[j]["name"].upper() for j in np.argsort(z)[::-1]]
            named = {n_: allrank.index(n_) + 1 for n_ in PAPER_TFS.get(t, []) if n_ in allrank}
            rows.append(dict(task=t, model=m, k=k, collection=coll, n_features_used=int((imp > 0).sum()),
                             n_enriched=len(enr), top5=";".join(names[:5]), top10=";".join(names[:10]),
                             top1=(names[0] if names else ""),
                             paper_named_tf_rank=";".join(f"{a}:{b}" for a, b in named.items()),
                             paper_named_tf_enriched=";".join(f"{a}:{a in [x.upper() for x in names]}" for a in named),
                             _set=set(names)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    from scipy.stats import spearmanr
    out = []
    for _, r in df.iterrows():
        ref = df[(df.task == r.task) & (df.model == "lgbm")]
        d = r.to_dict(); s = d.pop("_set")
        if not ref.empty and r.model != "lgbm":
            rs = ref.iloc[0]["_set"]
            d["jaccard_vs_lgbm"] = round(len(s & rs) / len(s | rs), 4) if (s | rs) else np.nan
            d["top1_in_lgbm_top5"] = bool(d["top1"] and d["top1"] in ref.iloc[0]["top5"].split(";"))
            d["top1_same_as_lgbm"] = bool(d["top1"] == ref.iloc[0]["top1"])
            d["z_spearman_vs_lgbm"] = round(float(spearmanr(zs[(r.task, r.model)], zs[(r.task, "lgbm")]).statistic), 4)
        out.append(d)
    return pd.DataFrame(out)


PAPER_TFS = {"human_nontata_promoters": ["SP1", "SP2", "KLF5"], "human_ocr_ensembl": ["TFAP4", "BATF3"]}


# ======================================================================= 6. labels
def sel(df, **kw):
    """Rows of df matching all column==value pairs (empty frame-safe)."""
    if df is None or df.empty:
        return pd.DataFrame()
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def load_gate():
    g = pd.read_csv(os.path.join(R, "alt_model_gate.csv")).set_index("task")
    return {t: (bool(g.loc[t, "gate"]), float(g.loc[t, "tata_neg_excess"])) if t in g.index else (False, np.nan)
            for t in W.TASKS}


def bench_row(bench, t, m, bench_arm):
    b = bench[(bench.task == t) & (bench.model == m) & (bench.arm == bench_arm)]
    if b.empty and m in BENCH_PARENT:
        b = bench[(bench.task == t) & (bench.model == BENCH_PARENT[m]) & (bench.arm == bench_arm)]
    return None if b.empty else b.iloc[0]


def _with_parent(df, m, **kw):
    s = sel(df, model=m, **kw)
    if s.empty and m in PARENT:
        s = sel(df, model=PARENT[m], **kw)
    return s


def overlap_contam(t, m, r3oc):
    """P11b: model-dependent half of rule 3 under the post hoc overlap scheme -- overlap-design share lost,
    task minus the drosophila control, 95% CI lower bound > 0 (leak-free test for nt_enhancers).
    None = no overlap row for this task/model (not assessed)."""
    if r3oc is None or r3oc.empty:
        return None
    o = _with_parent(r3oc, m, task=t)
    if o.empty:
        return None
    o = o[o.leakfree_test.astype(bool) == (t == "nt_enhancers")]
    return None if o.empty else bool(o.iloc[0]["diff_lo"] > 0)


def assign(t, m, bench, r2, r3p, gate, bench_arm="bench", r3oc=None):
    """Returns dict of labels under (primary pre-registered, pre-registered sensitivity, BIBM map, post hoc
    overlap scheme P11b)."""
    b = bench_row(bench, t, m, bench_arm)
    if b is None:
        return None
    p2 = _with_parent(r2, m, task=t)
    pos = bool(p2.iloc[0]["positional_rule"]) if not p2.empty else False
    g, excess = gate[t]
    e = _with_parent(r3p, m, task=t)
    collapses = bool(e.iloc[0]["collapses_prereg"]) if not e.empty else False
    cf = b.get("comp_fraction", np.nan)
    solved = bool(b["mcc"] >= SOLVED_CUT)

    def v2(r3):
        if solved:
            return "solved"
        if pos:
            return "positional"
        if g and r3:
            return "contamination"
        return "shared-motif"
    prim = v2(collapses)
    sens = v2(bool(cf == cf and cf >= COMP_FRAC_CUT))
    oc = overlap_contam(t, m, r3oc)
    povl = v2(bool(oc)) if (oc is not None or solved or pos or not g) else "not assessed"
    # BIBM map (post hoc P6): composition = a-priori non-TF-motif task (model-free), then the rules
    if t in NON_TF and not pos:
        bibm = "composition"
    elif solved:
        bibm = "solved"
    elif pos:
        bibm = "positional"
    elif g:
        bibm = "contamination" if excess >= STRONG_EXCESS else "contamination (weak)"
    else:
        bibm = "shared-motif"
    return dict(primary=prim, sensitivity=sens, bibm=bibm, posthoc_overlap=povl, mcc=b["mcc"], mcc_lo=b["mcc_lo"],
                mcc_hi=b["mcc_hi"], positional=pos, gate=g, collapses_prereg=collapses, has_eq_split=not e.empty,
                comp_fraction=cf, overlap_contam=oc, solved=solved)


def effect_ok(label, t, m, bench, r2, r3p, r3s, r3oc=None, bench_arm="bench"):
    """Pre-registered preservation effect(s) for a label: (ok, description, posthoc dict).
    solved -> MCC CI lower > 0.70; any failing label -> MCC CI upper < 0.70, AND additionally
    positional -> dMCC(B8-B1) CI lower > 0; contamination -> eq MCC - orig MCC CI upper < 0
    (pre-registered). Post hoc (P10d), contamination labels only: 'auroc' = size-controlled AUROC effect
    AUROC(random) - AUROC(matched) CI lower > 0 (leak-free test for nt_enhancers, P1's primary test there);
    'overlap' = overlap-design share lost, task minus the drosophila control, CI lower > 0 (P8; leak-free for
    nt_enhancers)."""
    b = bench_row(bench, t, m, bench_arm)
    fail_ok = bool(b["mcc_hi"] < SOLVED_CUT)
    fail_s = f"0.70-MCC={0.7 - b['mcc']:+.3f} (MCC CI [{b['mcc_lo']:.3f},{b['mcc_hi']:.3f}])"
    if label == "solved":
        return bool(b["mcc_lo"] > SOLVED_CUT), f"MCC-0.70={b['mcc'] - 0.7:+.3f} (MCC CI [{b['mcc_lo']:.3f},{b['mcc_hi']:.3f}])", {}
    if label == "composition":        # BIBM map: model-free task-type label
        return True, (f"model-free task-type label; MCC={b['mcc']:.3f} [{b['mcc_lo']:.3f},{b['mcc_hi']:.3f}], "
                      f"comp.frac={b.get('comp_fraction', np.nan):.3f}"), {}
    if label == "positional":
        p2 = _with_parent(r2, m, task=t)
        if p2.empty:
            return False, f"{fail_s}; positional arm missing", {}
        p = p2.iloc[0]
        return (fail_ok and bool(p["delta_lo"] > 0),
                f"{fail_s}; dMCC(B8-B1)={ci_str(p['delta_mcc'], p['delta_lo'], p['delta_hi'])}", {})
    if label.startswith("contamination"):
        e = _with_parent(r3p, m, task=t)
        lf = (t == "nt_enhancers")
        ph = {}
        s = _with_parent(r3s, m, task=t, feat="kmer")
        if not s.empty:
            s0 = s[s.leakfree_test.astype(bool) == lf].iloc[0]
            ph["auroc"] = (fail_ok and bool(s0["comp_effect_lo"] > 0),
                           f"AUROC(random)-AUROC(matched){' [leak-free test]' if lf else ''}="
                           f"{ci_str(s0['comp_effect'], s0['comp_effect_lo'], s0['comp_effect_hi'])}")
        o = _with_parent(r3oc, m, task=t) if r3oc is not None else pd.DataFrame()
        if not o.empty:
            o0 = o[o.leakfree_test.astype(bool) == lf].iloc[0]
            ph["overlap"] = (fail_ok and bool(o0["diff_lo"] > 0),
                             f"overlap-design share lost minus drosophila{' [leak-free test]' if lf else ''}="
                             f"{ci_str(o0['diff'], o0['diff_lo'], o0['diff_hi'])}")
        if e.empty:     # (fix round 2: the post hoc readouts are recorded even without an equalized split)
            return fail_ok, f"{fail_s}; no equalized split", ph
        e = e.iloc[0]
        ok = fail_ok and bool(e["eq_minus_orig_mcc_hi"] < 0)
        return ok, (f"{fail_s}; eqMCC-origMCC="
                    f"{ci_str(e['eq_minus_orig_mcc'], e['eq_minus_orig_mcc_lo'], e['eq_minus_orig_mcc_hi'])}"), ph
    return fail_ok, fail_s, {}


NON_TF = set()


def labels_table(bench, r2, r3p, r3s, gate, models, bench_arm="bench", r3oc=None):
    rows = []
    for t in W.TASKS:
        ref = assign(t, "lgbm", bench, r2, r3p, gate, bench_arm, r3oc)
        for m in models:
            a = assign(t, m, bench, r2, r3p, gate, bench_arm, r3oc)
            if a is None:
                continue
            row = dict(task=t, model=m, bench_arm=bench_arm, paper_v2=V2_LABEL[t], paper_bibm=BIBM_LABEL[t],
                       mcc=round(a["mcc"], 4), mcc_lo=round(a["mcc_lo"], 4), mcc_hi=round(a["mcc_hi"], 4),
                       positional_rule=a["positional"], model_free_gate=a["gate"],
                       eq_collapses_prereg=a["collapses_prereg"],
                       # P10d: the model-dependent half of rule 3 fires on a task the model-free gate rejects
                       eq_collapse_fires_without_gate=bool(a["has_eq_split"] and a["collapses_prereg"] and not a["gate"]),
                       # the tasks on which the MODEL-DEPENDENT half of rule 3 decides contamination vs shared-motif
                       gate_positive_failing=bool(a["gate"] and not a["solved"] and not a["positional"]),
                       comp_fraction_ge_cut=bool(a["comp_fraction"] == a["comp_fraction"] and a["comp_fraction"] >= COMP_FRAC_CUT),
                       overlap_contrast_ci_lo_gt_0=("" if a["overlap_contam"] is None else a["overlap_contam"]),
                       comp_fraction=round(a["comp_fraction"], 4)
                       if a["comp_fraction"] == a["comp_fraction"] else np.nan)
            for sch, paper in (("primary", V2_LABEL[t]), ("sensitivity", V2_LABEL[t]), ("bibm", BIBM_LABEL[t]),
                               ("posthoc_overlap", V2_LABEL[t])):
                lab = a[sch]
                ok, desc, ph = effect_ok(lab, t, m, bench, r2, r3p, r3s, r3oc, bench_arm)
                if sch == "posthoc_overlap" and lab.startswith("contamination") and "overlap" in ph:
                    ok, desc = ph["overlap"]      # P11b: the scheme's own contamination effect
                row[f"{sch}_label"] = lab
                row[f"{sch}_same_as_paper"] = bool(lab == paper)
                row[f"{sch}_same_as_lgbm"] = bool(ref is not None and lab == ref[sch])
                row[f"{sch}_effect_ci_ok"] = ok
                row[f"{sch}_preserved"] = bool(lab == paper and ok)
                row[f"{sch}_effect"] = desc
                for key in ("auroc", "overlap"):
                    if key in ph:
                        row[f"{sch}_posthoc_{key}_effect_ok"] = ph[key][0]
                        row[f"{sch}_posthoc_{key}_effect"] = ph[key][1]
                    # preserved under the post hoc criterion: contamination labels use the post hoc effect,
                    # every other label keeps its pre-registered effect
                    if lab.startswith("contamination"):   # None = readout not computed for this task (not assessed)
                        row[f"{sch}_preserved_posthoc_{key}"] = (bool(lab == paper and ph[key][0]) if key in ph else None)
                    else:
                        row[f"{sch}_preserved_posthoc_{key}"] = bool(lab == paper and ok)
            rows.append(row)
    return pd.DataFrame(rows)


def _is_na(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def _is_false(v):
    return (not _is_na(v)) and not bool(v)


def agreement_table(lab):
    rows = []
    for (arm, m), g in lab.groupby(["bench_arm", "model"], sort=False):
        fail = g[g.paper_v2 != "solved"]
        for sch in ("primary", "sensitivity", "bibm", "posthoc_overlap"):
            paper_col = "paper_bibm" if sch == "bibm" else "paper_v2"
            fl = g[~g[paper_col].isin(["solved", "composition"])]
            rows.append(dict(bench_arm=arm, model=m, scheme=sch, n_tasks=len(g),
                             same_label_as_paper=int(g[f"{sch}_same_as_paper"].sum()),
                             preserved_prereg_criterion=int(g[f"{sch}_preserved"].sum()),
                             preserved_posthoc_auroc_criterion=int((g[f"{sch}_preserved_posthoc_auroc"] == True).sum()),  # noqa: E712
                             preserved_posthoc_overlap_criterion=int((g[f"{sch}_preserved_posthoc_overlap"] == True).sum()),  # noqa: E712
                             posthoc_not_assessed=";".join(sorted({f"{r.task}({key})" for _, r in g.iterrows() for key in ("auroc", "overlap")
                                                                   if _is_na(r[f"{sch}_preserved_posthoc_{key}"])})),
                             same_label_as_lgbm_reference=int(g[f"{sch}_same_as_lgbm"].sum()),
                             n_failure_labels_paper=len(fl),
                             failure_labels_same_as_paper=int(fl[f"{sch}_same_as_paper"].sum()),
                             disagreements_vs_paper=";".join(f"{r.task}:{r[f'{sch}_label']}"
                                                            for _, r in g.iterrows() if not r[f"{sch}_same_as_paper"]),
                             effect_ci_failures=";".join(r.task for _, r in g.iterrows()
                                                         if r[f"{sch}_same_as_paper"] and not r[f"{sch}_effect_ci_ok"]),
                             posthoc_auroc_failures=";".join(r.task for _, r in g.iterrows()
                                                             if r[f"{sch}_same_as_paper"] and _is_false(r[f"{sch}_preserved_posthoc_auroc"])),
                             posthoc_overlap_failures=";".join(r.task for _, r in g.iterrows()
                                                               if r[f"{sch}_same_as_paper"] and _is_false(r[f"{sch}_preserved_posthoc_overlap"])),
                             eq_collapse_fires_without_gate=";".join(fail[fail.eq_collapse_fires_without_gate.astype(bool)].task)
                             if sch == "primary" else "",
                             # rule 3 on the gate-positive failing tasks: the model-dependent half decides here
                             gate_positive_failing_labels=";".join(f"{r.task}:{r[f'{sch}_label']}" for _, r in g.iterrows()
                                                                   if bool(r.gate_positive_failing)),
                             posthoc=bool(sch in ("bibm", "posthoc_overlap"))))
    return pd.DataFrame(rows)


# ======================================================================= main
def main():
    global B, NON_TF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-attribution", action="store_true")
    args = ap.parse_args()
    if args.quick:
        B = 200
    t0 = time.time()
    cs = pd.read_csv(os.path.join(C.ROOT, "results", "cross_suite_summary.csv"))
    NON_TF = set(cs.loc[cs["tf_motif_task"] == False, "task"])  # noqa: E712
    checks = []
    ut = unit_table()
    print(f"[analyze] {len(ut)} units ({time.time()-t0:.0f}s)", flush=True)
    # consistency: recomputed (float32 probas) vs worker (float64) metrics
    checks.append(dict(check="max_abs_mcc_recomputed_minus_json", value=float((ut.mcc - ut.mcc_json).abs().max()),
                       expected="<1e-3", ok=bool((ut.mcc - ut.mcc_json).abs().max() < 1e-3)))
    checks.append(dict(check="max_abs_auroc_recomputed_minus_json", value=float((ut.auroc - ut.auroc_json).abs().max()),
                       expected="<1e-4", ok=bool((ut.auroc - ut.auroc_json).abs().max() < 1e-4)))
    # LightGBM reference reproduces the released numbers
    rel = pd.concat([pd.read_csv(os.path.join(C.ROOT, "results", f)) for f in ("results.csv", "results_nt.csv")])
    rel = rel[(rel.model == "lgbm") & (rel.selected_by_val == 1)].set_index("dataset")
    for arm in ("bench", "bench_full"):
        for t in W.TASKS:
            r = U(ut, t, arm, "kmer", "lgbm")
            if r is None:
                continue
            capped = bool(r.train_capped)
            ok = abs(r.mcc_json - rel.loc[t, "mcc"]) < 1e-6 and abs(r.auroc_json - rel.loc[t, "auroc"]) < 1e-6
            checks.append(dict(check=f"lgbm_{arm}_reproduces_results_csv:{t}", value=f"{r.mcc_json:.6f}/{r.auroc_json:.6f}",
                               expected=f"{rel.loc[t, 'mcc']:.6f}/{rel.loc[t, 'auroc']:.6f}" + (" (capped: not expected)" if capped else ""),
                               ok=bool(ok or capped)))
    ps = pd.read_csv(os.path.join(R, "positional_sweep.csv")).set_index("task")
    for t in W.TASKS:
        a, b_ = U(ut, t, "pos", "B1", "lgbm"), U(ut, t, "pos", "B8", "lgbm")
        if a is not None and b_ is not None:
            ok = abs(a.mcc_json - ps.loc[t, "spectrum_mcc"]) < 6e-5 and abs(b_.mcc_json - ps.loc[t, "positional_mcc"]) < 6e-5
            checks.append(dict(check=f"lgbm_pos_reproduces_positional_sweep:{t}",
                               value=f"{a.mcc_json:.4f}->{b_.mcc_json:.4f}",
                               expected=f"{ps.loc[t, 'spectrum_mcc']:.4f}->{ps.loc[t, 'positional_mcc']:.4f}", ok=bool(ok)))
    cc = pd.concat([pd.read_csv(os.path.join(R, f)) for f in ("composition_clean.csv", "composition_clean_specificity.csv")])
    for t in W.EQ_SPLITS:
        r = U(ut, t, "eq", "kmer", "lgbm"); rc = U(ut, t, "eq", "comp", "lgbm")
        row = cc[(cc.dataset == t) & (cc.arm == "comp_equalized")]
        if r is not None and not row.empty and not bool(r.train_capped):
            ok = abs(r.mcc_json - row.kmer_mcc.iloc[0]) < 1e-5 and abs(r.auroc_json - row.kmer_auroc.iloc[0]) < 1e-5
            checks.append(dict(check=f"lgbm_eq_reproduces_composition_clean:{t}", value=f"{r.mcc_json:.6f}/{r.auroc_json:.6f}",
                               expected=f"{row.kmer_mcc.iloc[0]:.6f}/{row.kmer_auroc.iloc[0]:.6f}", ok=bool(ok)))
        elif r is not None and not row.empty:
            checks.append(dict(check=f"lgbm_eq_reproduces_composition_clean:{t}", value=f"{r.mcc_json:.6f}",
                               expected=f"{row.kmer_mcc.iloc[0]:.6f} (train capped at 50k: not expected)", ok=True))
    # shuffled arm: dinuc_shuffle draws depend on PYTHONHASHSEED (check_shuffle_hashseed.py); the
    # paper's draw is unrecoverable, so the LightGBM reference is checked against the scan row for the
    # unit's own pinned hash seed, and the paper's CSV value against the scan's range.
    sn = pd.read_csv(os.path.join(R, "shuffled_neg.csv"))
    hs_p = os.path.join(R, "alt_model_shuffle_hashseed.csv")
    hs = pd.read_csv(hs_p, dtype={"hashseed": str}) if os.path.exists(hs_p) else pd.DataFrame()
    for t in W.SHUF_TASKS:
        u = unit(t, "shuf", "kmer", "lgbm")
        if u is None:
            continue
        seed = str(u.get("pythonhashseed", "unset"))
        row = hs[(hs.task == t) & (hs.hashseed == seed)] if not hs.empty else hs
        if row is not None and not row.empty:
            ok = abs(u["mcc"] - row.kmer_mcc.iloc[0]) < 1e-6 and u.get("shuffle_sha1") == row.shuffle_sha1.iloc[0]
            checks.append(dict(check=f"lgbm_shuf_matches_hashseed_scan:{t}", value=f"{u['mcc']:.6f} sha={u.get('shuffle_sha1')}",
                               expected=f"{row.kmer_mcc.iloc[0]:.6f} sha={row.shuffle_sha1.iloc[0]} (PYTHONHASHSEED={seed})", ok=bool(ok)))
        else:
            checks.append(dict(check=f"lgbm_shuf_matches_hashseed_scan:{t}", value=f"PYTHONHASHSEED={seed}",
                               expected="pinned hash seed present in the scan", ok=False))
        if not hs.empty:
            g = hs[hs.task == t]; pr = sn[(sn.dataset == t) & (sn.arm == "shuffled_neg")].iloc[0]
            checks.append(dict(check=f"paper_shuffled_neg_within_hashseed_range:{t}",
                               value=f"paper TATA {pr.tata_auroc:.4f} MCC {pr.kmer_mcc:.4f}",
                               expected=f"TATA [{g.tata_auroc.min():.4f},{g.tata_auroc.max():.4f}] MCC [{g.kmer_mcc.min():.4f},{g.kmer_mcc.max():.4f}] over {len(g)} hash seeds",
                               ok=bool(g.tata_auroc.min() - 1e-4 <= pr.tata_auroc <= g.tata_auroc.max() + 1e-4
                                       and g.kmer_mcc.min() <= pr.kmer_mcc <= g.kmer_mcc.max())))
    # the nt leak set
    checks.append(dict(check="nt_enhancers_test_seqs_verbatim_in_train", value=int(nt_dup_mask().sum()), expected=100,
                       ok=int(nt_dup_mask().sum()) == 100))

    all_models = [m for m in MODELS + ["lrx", "lra", "ebm512", "ebmpos"] if (ut.model == m).any()]
    bench = pd.concat([bench_table(ut, all_models, "bench"), bench_table(ut, all_models, "bench_full")], ignore_index=True)
    bench.to_csv(os.path.join(R, "alt_model_bench.csv"), index=False)
    # post hoc P2: benchmark MCCs with uncapped training wherever an uncapped refit exists
    bfull = bench[bench.arm == "bench"].copy()
    subst = {m: set() for m in all_models}
    for t in W.FULL_TASKS:
        for m in all_models:
            f = bench[(bench.task == t) & (bench.model == m) & (bench.arm == "bench_full")]
            if not f.empty:
                subst[m].add(t)
                bfull = bfull[~((bfull.task == t) & (bfull.model == m))]
                g = f.copy(); g["arm"] = "bench"
                bfull = pd.concat([bfull, g], ignore_index=True)
    print(f"[analyze] bench ({time.time()-t0:.0f}s)", flush=True)
    r2 = route2_table(ut, all_models, "pos")
    r2f = route2_table(ut, all_models, "pos_full")
    pd.concat([r2, r2f], ignore_index=True).to_csv(os.path.join(R, "alt_model_route2.csv"), index=False)
    # P9 check: the ebmpos global k-mer selection equals the pre-registered EBM B1 selection (strict superset)
    for arm in ("pos", "pos_full"):
        for t in W.TASKS:
            a, b = unit(t, arm, "B1", "ebm"), unit(t, arm, "B8", "ebmpos")
            if a is None or b is None:
                continue
            g1 = [n_ for n_ in a["info"].get("ebm_kmer_features", [])]
            g8 = [n_ for n_ in b["info"].get("ebm_kmer_features", []) if not str(n_).startswith("bin")]
            checks.append(dict(check=f"ebmpos_B8_design_contains_ebm_B1_design:{arm}:{t}", value=f"{len(g8)} global, "
                               f"{sum(str(n_).startswith('bin') for n_ in b['info']['ebm_kmer_features'])} binned",
                               expected="global selection == ebm B1 selection", ok=bool(g1 == g8 and len(g1) > 0)))
    r3p = route3_prereg(ut, all_models)
    r3p.to_csv(os.path.join(R, "alt_model_route3_prereg.csv"), index=False)
    print(f"[analyze] route2/3-prereg ({time.time()-t0:.0f}s)", flush=True)
    r3s, shares = route3_size_table(all_models)
    r3s.to_csv(os.path.join(R, "alt_model_route3_sizecontrolled.csv"), index=False)
    r3c = route3_contrast(shares, all_models, r3s)
    r3c.to_csv(os.path.join(R, "alt_model_route3_contrast.csv"), index=False)
    # the random-negative arms are independent draws of psm_sensitivity's design (seeds 0-9 here, psm's
    # 10000+r): LightGBM's mean random-arm AUROC should agree with psm's within draw-to-draw noise
    pt_ = pd.read_csv(os.path.join(R, "psm_sensitivity_paper_table.csv"))
    psm_diffs = []
    for t in W.EQR_A:
        a = r3s[(r3s.task == t) & (r3s.model == "lgbm") & (r3s.feat == "kmer") & (~r3s.leakfree_test.astype(bool))]
        b = pt_[(pt_.dataset == t) & (pt_.condition == "size-matched random negatives (10 seeds)")]
        if not a.empty and not b.empty:
            d = float(a.auroc_random_mean.iloc[0] - b.auroc.iloc[0])
            psm_diffs.append(abs(d))
            checks.append(dict(check=f"lgbm_random_arm_auroc_vs_psm_independent_draws:{t}",
                               value=f"{a.auroc_random_mean.iloc[0]:.4f} (n_neg_test {int(a.n_neg_test_matched.iloc[0])})",
                               expected=f"psm {b.auroc.iloc[0]:.4f} (n_neg_test {int(b.n_test_neg.iloc[0])}); independent "
                                        f"draws, |diff| < 0.02", ok=bool(abs(d) < 0.02),
                               note="not expected to be equal: different random seeds and (for psm) its own matched-arm size"))
    if psm_diffs:
        checks.append(dict(check="lgbm_random_arm_auroc_vs_psm_max_abs_diff", value=round(max(psm_diffs), 4),
                           expected="< 0.02 (independent draws; reported as the observed maximum)",
                           ok=bool(max(psm_diffs) < 0.02)))
    r3o, r3oc = route3_overlap_table(all_models)
    r3o.to_csv(os.path.join(R, "alt_model_route3_overlap.csv"), index=False)
    r3oc.to_csv(os.path.join(R, "alt_model_route3_overlap_contrast.csv"), index=False)
    bt = pd.read_csv(os.path.join(R, "psm_sensitivity_between_dataset.csv"))
    for t in ("human_enhancers_cohn", "drosophila_enhancers_stark"):
        a = r3o[(r3o.task == t) & (r3o.model == "lgbm") & (r3o.feat == "kmer")]
        b = bt[(bt.dataset == t) & (bt.design == "fixed_model_overlap")]
        if not a.empty and not b.empty:
            ok = abs(a.auroc_overlap.iloc[0] - b.auroc_overlap.iloc[0]) < 1e-6 and \
                abs(a.share_lost.iloc[0] - b.share_above_chance_lost.iloc[0]) < 1e-6
            checks.append(dict(check=f"lgbm_overlap_reproduces_psm_between_dataset:{t}",
                               value=f"{a.auroc_overlap.iloc[0]:.6f}/{a.share_lost.iloc[0]:.6f}",
                               expected=f"{b.auroc_overlap.iloc[0]:.6f}/{b.share_above_chance_lost.iloc[0]:.6f}", ok=bool(ok)))
    shuf = shuffled_table(all_models)
    shuf.to_csv(os.path.join(R, "alt_model_route3_shuffled.csv"), index=False)
    print(f"[analyze] route3 ({time.time()-t0:.0f}s)", flush=True)
    attr = pd.DataFrame() if args.skip_attribution else attribution_table(MODELS)
    if not attr.empty:
        attr.to_csv(os.path.join(R, "alt_model_attribution.csv"), index=False)
    elif os.path.exists(os.path.join(R, "alt_model_attribution.csv")):
        attr = pd.read_csv(os.path.join(R, "alt_model_attribution.csv"))
    pnd = pd.read_csv(os.path.join(R, "presence_null.csv"))
    pn_counts = dict(zip(pnd[pnd.suite == "GB"]["task"], pnd[pnd.suite == "GB"]["n_enriched_TFs"]))
    dose, dchecks = dose_table(all_models)
    checks += dchecks
    dose.to_csv(os.path.join(R, "alt_model_dose.csv"), index=False)
    r1 = route1_table(bench[bench.arm == "bench"], all_models, dose, pn_counts, attr)
    r1u = route1_table(bfull, all_models, dose, pn_counts, attr, suffix="_uncapped",
                       mcc_source="bench_full where available (uncapped; post hoc P10a)", subst=subst)
    r1 = pd.concat([r1, r1u], ignore_index=True)
    r1.to_csv(os.path.join(R, "alt_model_route1.csv"), index=False)
    # the uncapped LightGBM reference reproduces the paper's Table corr (+0.87 n=9 / +0.74 n=11, 2 dp)
    for an, exp in (("route1_disc_vs_mcc_n9_homology_clean_uncapped", 0.87), ("route1_disc_vs_mcc_n11_uncapped", 0.74)):
        r = r1[(r1.analysis == an) & (r1.model == "lgbm")]
        if not r.empty:
            checks.append(dict(check=f"lgbm_{an}_reproduces_paper_table_corr", value=f"{r.rho.iloc[0]:.4f}",
                               expected=f"{exp:+.2f} (2 dp)", ok=bool(round(float(r.rho.iloc[0]), 2) == exp)))
    print(f"[analyze] route1/dose/attr ({time.time()-t0:.0f}s)", flush=True)
    gate = load_gate()
    # rules reproduce the paper's labels from the RELEASED LightGBM numbers (pre-registered check)
    lmods = [m for m in LABEL_MODELS if m in all_models]
    lab = labels_table(bench, r2, r3p, r3s, gate, lmods, "bench", r3oc)
    labf = labels_table(bfull, r2, r3p, r3s, gate, lmods, "bench", r3oc)
    labf["bench_arm"] = "bench_full_where_available"
    labs = pd.concat([lab, labf], ignore_index=True)
    labs.to_csv(os.path.join(R, "alt_model_labels.csv"), index=False)
    agr = agreement_table(labs)
    agr.to_csv(os.path.join(R, "alt_model_agreement.csv"), index=False)
    lg = lab[lab.model == "lgbm"]
    for sch, col in (("primary", "primary_same_as_paper"), ("sensitivity", "sensitivity_same_as_paper"),
                     ("bibm", "bibm_same_as_paper"), ("posthoc_overlap", "posthoc_overlap_same_as_paper")):
        checks.append(dict(check=f"rules_reproduce_paper_labels_with_lgbm_reference:{sch}",
                           value=f"{int(lg[col].sum())}/{len(lg)}", expected="15/15",
                           ok=bool(len(lg) == 15 and lg[col].all()),
                           note=";".join(f"{r.task}:{r[sch + '_label']}" for _, r in lg.iterrows() if not r[col])))
    pd.DataFrame(checks).to_csv(os.path.join(R, "alt_model_checks.csv"), index=False)
    import make_interpretation_alt as MI
    MI.write(R)
    print(f"[analyze] done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
