#!/usr/bin/env python3
"""Step 7 -- the REMAINING experiments of the paper, re-run on HOMOLOGY-CLEAN data (Reviewer 1, comment 3:
"re-run all baseline evaluations and intervention experiments").

run_eval.py / run_interventions.py / run_routes.py already cover the baseline, the composition-only baseline +
composition equalization (E1), dinucleotide-shuffled negatives (E2), the positional rescue (E3), the Route-1
correlation and the route labels. This file re-runs every other experiment of the submitted paper whose outcome
could be moved by train/test or within-split homology, with IDENTICAL code on the paper's data (BEFORE) and on
homology-clean data (AFTER).

PRE-REGISTERED ANALYSIS PLAN (written 2026-09-24 ~11:05, before any run of this file)
-----------------------------------------------------------------------------------
Data. BEFORE = the provided split, all sequences ('original'). AFTER = homology-clean (HC) data at T in
{id80 (primary), id50 (strict)} = run_eval.py's cluster-aware re-split (seed 42) followed by one representative
per (component, label) (_hc.hc_data; identical index construction in split_idx()). Exceptions, stated per
experiment: X3/X7/X8 use the re-split WITHOUT the NR step (the exact split of run_eval.py, so their MCCs are
comparable with homology_recluster_eval.csv). Model everywhere: the paper's fixed LightGBM (3 threads) at the
validation-selected k. Bootstrap: 1000 resamples, percentile 95 % CI, test sequence (or scanned sequence) = unit.

Cleaning arms (shared by X1/X2/X4, one fresh RandomState(42) per arm so every experiment sees the SAME cleaned
sets): tata_flag = run_cleaning.py's rule (flag negatives with >= 1 TBP hit at 1.3 bits/pos; within each split
drop a seeded subset of flagged negatives so the negative flag rate equals the pooled positive flag rate);
gc_match = run_composition_clean.gc_match (25-bin GC histogram, per split); comp_equalized =
run_composition_clean propensity match (LR on the 20-d composition signature, pooled fit, 20-bin proportional
match per split). NEW control arms placebo_<arm>: remove the SAME number of negatives per split as <arm>, but
chosen uniformly at random (RandomState(43)) -- separates "the removed negatives were the composition-biased
ones" from "fewer negatives". Positives are never removed.

X1 ROUTE-3 CLEANING (the paper's "interventional headline"; cohn, nt_enhancers = contaminated, drosophila =
   control) per data in {original, id80, id50}: per arm the fraction of negatives removed, pooled TATA and GC
   motif-only AUROC (CI), composition-only MCC, k-mer MCC/AUROC (CI) and dMCC vs the uncleaned arm (unpaired
   CI, as the paper states). Arms whose test split keeps < 10 negatives are MCC-infeasible (AUROCs still
   reported). Decision (per contaminated task, HC): cleaning effect PRESERVED iff for tata_flag AND gc_match the
   TATA AUROC moves toward 0.5 (point closer to 0.5 than the uncleaned arm AND cleaned CI lower bound > the
   uncleaned CI upper bound); SPECIFIC iff each placebo arm's TATA AUROC CI overlaps the uncleaned arm's CI.
   Control behaves iff tata_flag removes <= 1 % of drosophila negatives and |dTATA AUROC| <= 0.01.
X2 ROUTE-3 SPECIFICITY (run_route3_specificity.py procedure, its functions imported): on the pooled sequences
   (NR representatives for HC), capped 4000/class, seed 42: real-TATA AUROC, 20 column-scrambled TBP PWMs and
   10 random IC-matched PWMs (panel mean, SD, bootstrap CI of the mean) for cohn and nt_enhancers; the positive
   control real TATA on nt_promoter_tata; the TATA threshold sweep {0.8,...,1.5} bits/pos on the uncleaned
   and the comp_equalized arm. Decision (HC): specificity PRESERVED iff (a) real TATA CI upper < 0.5,
   (b) random-IC panel mean within 0.5 +/- 0.02 or its CI covers 0.5, (c) scrambled-TBP panel mean CI upper
   < 0.5 (composition-driven skew), (d) positive control CI lower > 0.5, (e) the uncleaned arm is neg-skewed
   (CI upper < 0.5) at all 7 thresholds and the equalized arm is closer to 0.5 than the uncleaned arm at all 7.
X3 COMPOSITION-ONLY BASELINE, ALL 15 TASKS (the paper's recommendation (ii); diagnose.py's composition fraction):
   LightGBM on the 20-d composition signature for the units of run_eval.py (original split; test-purge id80/id50
   = same comp model scored on the purged test; cluster re-split id80/id50 and random re-split, seed 42). k-mer
   probabilities are read from run_eval.py's cached predictions (same units). Readouts: comp-only MCC/AUROC
   (CI), residual MCC = kmer - comp (paired CI), composition fraction = (AUROC_comp - 0.5)/(AUROC_kmer - 0.5)
   (paired CI; binary class-1 AUROC as diagnose.py, macro one-vs-rest for the 3 multiclass tasks).
   Descriptive; used by run_labels_addendum.py for the submitted paper's 'composition' label.
X4 FROZEN-FM PROBE (run_fm_paired.py / run_fm_paired_dnabert.py procedure; embeddings from the released cache,
   sha256-verified by fetch_fm_cache.py; no torch in this process): HyenaDNA (cohn, nt_enhancers, drosophila;
   mean/sep) and DNABERT-2 (cohn, nt_enhancers; mean/cls); heads LightGBM and LR. Data: 'original_released'
   (the released cleaned splits -- reproduction check against fm_paired*.csv), 'original' (arms rebuilt by the
   shared arm code) and HC id80/id50 (head trained on HC train embeddings; arms applied within HC splits).
   test_effect = AUROC(H_orig on T_clean) - AUROC(H_orig on T_full) with the nested paired bootstrap; also
   paired_delta (H_clean vs H_orig on T_clean). Decision (HC): FM result PRESERVED iff for the paper's headline
   cell (gc_match, mean pooling, LightGBM head) both contaminated tasks have test_effect < 0 with CI upper < 0
   for both models, the drosophila (HyenaDNA) test_effect CI covers 0, and every placebo_gc_match CI covers 0.
X5 GRADED POSITIONAL RESCUE (nt_splice_sites_all; B in {2,4,8,16} vs B=1 at k=5, run_interventions.run_e3's
   cap/featurisation, paired bootstrap per B). Decision (HC): graded PRESERVED iff dMCC increases with B
   (monotone point estimates) and every CI lower bound > 0.
X6 POSITIONAL RESCUE AT id50 FOR THE 11 TIER-2 TASKS (run_interventions.run_e3 unchanged) so that every task's
   Route-2 diagnostic exists at both thresholds. Selectivity rule as in run_interventions.py.
X7 LR FLOOR, ALL 15 TASKS (models.py LR pipeline, 1 thread) on the original split and the cluster re-splits
   id80/id50 (run_eval.py splits, no NR). Tier (solved >= 0.7 / partial >= 0.5 / hard) agreement LR vs
   LightGBM before vs after (paper: 13/15). Descriptive.
X8 MODEL-SEED ROBUSTNESS (run_multiseed.py procedure: seeds 42-46, bootstrap-resampled training rows, fixed
   test) for its 8 key tasks on the original split and the id80 cluster re-split. Decision: preserved iff the
   tier is seed-stable for the same tasks as before and max SD stays <= 0.02.
X9 MOTIF PRESENCE (run_interpretability.py procedure: gain-weighted per-TF enrichment z vs the exact
   permutation moments, BH-FDR < 0.05, JASPAR CORE vertebrates / insects) for the 9 GB tasks, from the
   LightGBM refit on the original train split and on the id80 cluster re-split train split (val-selected k for
   both). Spearman(MCC, #FDR-enriched TFs) before vs after. Decision: the presence NULL is preserved iff the
   after-cleaning Spearman has p >= 0.05.
X10 COORDINATE ANALYSIS (localization/coordinate_analysis.csv; cohn negatives, GRCh38/GENCODE v44 TSS +/-500 bp):
   TSS-overlap rate of TATA-flagged vs non-flagged negatives, ratio and Fisher exact test, on all negatives and
   on the NR representatives at id80/id50. Decision: the TSS-DEPLETION of flagged negatives is preserved iff the
   ratio < 1 with Fisher p < 0.05 on the NR data.
Nulls and adverse results are reported as they come out.

Outputs: results/upgrades/homology_recluster_extra_{cleaning,specificity,threshold,composition,fm,graded,lr,
modelseeds,presence,coords}.csv (+ the id50 rows appended to homology_recluster_positional.csv by X6).
Checkpoints: external/homology_recluster/ckpt/extra/. Usage: python run_extra.py --exp X1,X2,...
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import fisher_exact, spearmanr
from sklearn.metrics import matthews_corrcoef

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _hc  # noqa: E402  (caps LightGBM threads for this process)
C = _hc.C
import run_interventions as RI  # noqa: E402  (imports run_composition_clean as RI.CC)
from run_eval import boot_diff, pred_of, PRED_DIR  # noqa: E402
from run_cluster import SIZE_ORDER  # noqa: E402
sys.path.insert(0, os.path.join(_hc.UP, "route3_specificity"))
import run_route3_specificity as R3S  # noqa: E402
import featurize  # noqa: E402

CC = RI.CC
CK = os.path.join(_hc.CKPT_DIR, "extra")
SEED = _hc.SEED
BOOT = _hc.BOOT
TATA_BITS = 1.3
GC_BINS = 25
PROP_BINS = 20
DVS = ["original", "id80", "id50"]
R3_TASKS = [("human_enhancers_cohn", "contaminated"), ("nt_enhancers", "contaminated"),
            ("drosophila_enhancers_stark", "clean control")]
BASE_ARMS = ["tata_flag", "gc_match", "comp_equalized"]
ALL_ARMS = BASE_ARMS + [f"placebo_{a}" for a in BASE_ARMS]


def out_csv(name):
    return os.path.join(_hc.RES, f"{_hc.PREFIX}_extra_{name}.csv")


def ck(name):
    os.makedirs(CK, exist_ok=True)
    return os.path.join(CK, name + ".csv")


def cached(name, fn):
    p = ck(name)
    if os.path.exists(p):
        return pd.read_csv(p)
    t0 = time.time()
    df = fn()
    df.to_csv(p + ".tmp", index=False)
    os.replace(p + ".tmp", p)
    print(f"  [{name}] {time.time() - t0:.0f}s", flush=True)
    return df


def aggregate(prefix, name):
    parts = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(CK, prefix + "*.csv"))) if os.path.getsize(p) > 1]
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(out_csv(name), index=False)
        print(f"wrote {out_csv(name)} ({sum(len(p) for p in parts)} rows)", flush=True)


# ------------------------------------------------------------------ data
def split_idx(task, dv, nr=True):
    """(pooled seqs, pooled y, train idx, test idx). 'original' = provided split, all sequences; id80/id50 =
    run_eval.py's cluster-aware re-split (seed 42) [+ one representative per (component, label) if nr] --
    identical index construction to _hc.hc_data."""
    tr, ytr, te, yte = _hc.load_task(task)
    seqs = tr + te
    y = np.concatenate([ytr, yte])
    if dv == "original":
        return seqs, y, np.arange(len(tr)), np.arange(len(tr), len(seqs))
    comp, _ = _hc.load_clusters(task, dv)
    is_test = _hc.cluster_resplit(comp, y, len(te) / len(seqs), SEED)
    keep = np.ones(len(seqs), bool)
    if nr:
        keep[:] = False
        keep[_hc.nr_representatives(comp, y, SEED)] = True
    return seqs, y, np.where(~is_test & keep)[0], np.where(is_test & keep)[0]


def neg_mask(y, keep_neg_local):
    """Boolean keep mask over a split: all positives + the negatives at local negative positions."""
    m = y == 1
    neg = np.where(y == 0)[0]
    m = m.copy()
    m[neg[np.asarray(keep_neg_local, int)]] = True
    return m


def clean_excess(y, flagged, target_rate, rng):
    """run_cleaning.clean_indices as a mask: keep positives; drop a seeded subset of flagged negatives so
    the negative flag rate ~ target_rate."""
    neg = np.where(y == 0)[0]
    neg_fl = neg[flagged[neg]]
    neg_un = neg[~flagged[neg]]
    keep_fl = int(round(target_rate * len(neg)))
    keep_neg = neg if len(neg_fl) <= keep_fl else np.concatenate([neg_un, neg_fl[rng.permutation(len(neg_fl))[:keep_fl]]])
    m = y == 1
    m = m.copy()
    m[keep_neg] = True
    return m


def random_neg_mask(y, n_keep_neg, rng):
    neg = np.where(y == 0)[0]
    m = (y == 1).copy()
    m[rng.choice(neg, int(n_keep_neg), replace=False)] = True
    return m


def arm_masks(tr, ytr, te, yte, tbp, feats=None):
    """Keep masks (train, test) for every cleaning arm (see docstring). `feats` caches per-split features."""
    f = feats if feats is not None else {}
    if "tata_tr" not in f:
        f["tata_tr"] = C.tata_hit_counts(tr, tbp, TATA_BITS).astype(np.int64)
        f["tata_te"] = C.tata_hit_counts(te, tbp, TATA_BITS).astype(np.int64)
        f["gc_tr"], f["gc_te"] = C.gc_content(tr), C.gc_content(te)
        f["comp_tr"], f["comp_te"] = C.comp_signature(tr, ks=(1, 2)), C.comp_signature(te, ks=(1, 2))
    out = {}
    ftr, fte = f["tata_tr"] >= 1, f["tata_te"] >= 1
    pos_rate = (ftr[ytr == 1].sum() + fte[yte == 1].sum()) / max(int((ytr == 1).sum() + (yte == 1).sum()), 1)
    rng = np.random.RandomState(SEED)
    out["tata_flag"] = (clean_excess(ytr, ftr, pos_rate, rng), clean_excess(yte, fte, pos_rate, rng))
    rng = np.random.RandomState(SEED)
    gk_tr = CC.gc_match(f["gc_tr"][ytr == 1], f["gc_tr"][ytr == 0], GC_BINS, rng)
    gk_te = CC.gc_match(f["gc_te"][yte == 1], f["gc_te"][yte == 0], GC_BINS, rng)
    out["gc_match"] = (neg_mask(ytr, gk_tr), neg_mask(yte, gk_te))
    rng = np.random.RandomState(SEED)
    prop = CC.fit_propensity(np.vstack([f["comp_tr"], f["comp_te"]]), np.concatenate([ytr, yte]), SEED)
    p_tr = prop.predict_proba(f["comp_tr"])[:, 1]; p_te = prop.predict_proba(f["comp_te"])[:, 1]
    k_tr = CC.match_proportional(p_tr[ytr == 1], p_tr[ytr == 0], PROP_BINS, rng)
    k_te = CC.match_proportional(p_te[yte == 1], p_te[yte == 0], PROP_BINS, rng)
    out["comp_equalized"] = (neg_mask(ytr, k_tr), neg_mask(yte, k_te))
    for a in BASE_ARMS:
        rng = np.random.RandomState(SEED + 1)
        mtr, mte = out[a]
        out[f"placebo_{a}"] = (random_neg_mask(ytr, int((mtr & (ytr == 0)).sum()), rng),
                               random_neg_mask(yte, int((mte & (yte == 0)).sum()), rng))
    return out, f


def auc_ci(pos, neg, fn, seed=SEED):
    rng = np.random.RandomState(seed)
    pt = fn(np.asarray(pos), np.asarray(neg))
    lo, hi = C.boot_ci(np.asarray(pos), np.asarray(neg), fn, BOOT, rng)
    return float(pt), float(lo), float(hi)


# ------------------------------------------------------------------ X1
def x1_unit(task, role, dv, tbp):
    seqs, y, itr, ite = split_idx(task, dv, nr=True)
    tr = [seqs[i] for i in itr]; te = [seqs[i] for i in ite]; ytr, yte = y[itr], y[ite]
    k = _hc.best_k(task)
    masks, f = arm_masks(tr, ytr, te, yte, tbp)
    Xtr_all = featurize.kmer_spectrum(tr, k); Xte_all = featurize.kmer_spectrum(te, k)
    rows = []
    base_pred = None
    n_neg0 = int((ytr == 0).sum() + (yte == 0).sum())
    for arm in ["uncleaned"] + ALL_ARMS:
        mtr, mte = (np.ones(len(ytr), bool), np.ones(len(yte), bool)) if arm == "uncleaned" else masks[arm]
        a_ytr, a_yte = ytr[mtr], yte[mte]
        tp = np.concatenate([f["tata_tr"][mtr & (ytr == 1)], f["tata_te"][mte & (yte == 1)]])
        tn = np.concatenate([f["tata_tr"][mtr & (ytr == 0)], f["tata_te"][mte & (yte == 0)]])
        gp = np.concatenate([f["gc_tr"][mtr & (ytr == 1)], f["gc_te"][mte & (yte == 1)]])
        gn = np.concatenate([f["gc_tr"][mtr & (ytr == 0)], f["gc_te"][mte & (yte == 0)]])
        ta = auc_ci(tp, tn, C.auc_counts); ga = auc_ci(gp, gn, C.auc_float)
        n_neg = int((a_ytr == 0).sum() + (a_yte == 0).sum())
        r = dict(experiment="X1_cleaning", task=task, role=role, data=dv, arm=arm, k=k,
                 n_pos=int((a_ytr == 1).sum() + (a_yte == 1).sum()), n_neg=n_neg,
                 frac_neg_removed=round(1 - n_neg / max(n_neg0, 1), 4),
                 n_train=int(mtr.sum()), n_test=int(mte.sum()), n_test_neg=int((a_yte == 0).sum()),
                 tata_pos_rate=round(float((tp >= 1).mean()), 4), tata_neg_rate=round(float((tn >= 1).mean()), 4),
                 tata_auroc=round(ta[0], 4), tata_lo=round(ta[1], 4), tata_hi=round(ta[2], 4),
                 gc_auroc=round(ga[0], 4), gc_lo=round(ga[1], 4), gc_hi=round(ga[2], 4))
        if (a_yte == 0).sum() >= 10 and (a_ytr == 0).sum() >= 10:
            pk, predk = RI.fit_pred(Xtr_all[np.where(mtr)[0]], a_ytr, Xte_all[np.where(mte)[0]])
            km = _hc.boot_metrics(a_yte, predk, pk)
            ctr_, cte_ = f["comp_tr"][mtr], f["comp_te"][mte]
            pc, predc = RI.fit_pred(ctr_, a_ytr, cte_)
            cm = _hc.boot_metrics(a_yte, predc, pc)
            r.update(kmer_mcc=round(km["mcc"], 4), kmer_mcc_lo=round(km["mcc_lo"], 4), kmer_mcc_hi=round(km["mcc_hi"], 4),
                     kmer_auroc=round(km["auroc"], 4), kmer_auroc_lo=round(km["auroc_lo"], 4),
                     kmer_auroc_hi=round(km["auroc_hi"], 4), comp_only_mcc=round(cm["mcc"], 4),
                     comp_only_mcc_lo=round(cm["mcc_lo"], 4), comp_only_mcc_hi=round(cm["mcc_hi"], 4),
                     status="ok")
            if arm == "uncleaned":
                base_pred = (a_yte, predk)
            elif base_pred is not None:
                lo, hi = boot_diff(a_yte, predk, base_pred[0], base_pred[1])
                r.update(d_kmer_mcc=round(km["mcc"] - rows[0]["kmer_mcc"], 4), d_kmer_mcc_lo=round(lo, 4),
                         d_kmer_mcc_hi=round(hi, 4))
        else:
            r.update(status="MCC infeasible (<10 negatives kept in a split); pooled AUROCs only")
        rows.append(r)
    return pd.DataFrame(rows)


def run_x1(a):
    tbp = C.load_tbp()
    for task, role in R3_TASKS:
        for dv in DVS:
            cached(f"X1_{task}_{dv}", lambda: x1_unit(task, role, dv, tbp))
    aggregate("X1_", "cleaning")


# ------------------------------------------------------------------ X2
def x2_unit(task, dv, tbp, scram, rand):
    seqs, y, itr, ite = split_idx(task, dv, nr=True)
    rows, thr_rows = [], []
    pool = np.sort(np.concatenate([itr, ite]))
    ps, ns = R3S.cap_per_class([seqs[i] for i in pool], y[pool], 4000, SEED)
    pc, nc = C.motif_match.encode_sequences(ps), C.motif_match.encode_sequences(ns)
    rng = np.random.RandomState(SEED)
    pt, lo, hi = R3S.real_tata_auroc_ci(pc, nc, tbp["pssm"], TATA_BITS, BOOT, rng)
    rows.append(dict(task=task, data=dv, motif_class="real_TATA", motif_only_auroc=round(pt, 4), auroc_sd=np.nan,
                     auroc_ci_lo=round(lo, 4), auroc_ci_hi=round(hi, 4), n_panel=1, n_pos=len(ps), n_neg=len(ns)))
    if task == R3S.POS_CONTROL:
        return pd.DataFrame(rows), pd.DataFrame(thr_rows)
    for cls, panel in [("scrambled_TBP_panel", scram), ("random_IC_panel", rand)]:
        m, sd, plo, phi, _ = R3S.panel_auroc_stats(pc, nc, panel, TATA_BITS, BOOT, rng)
        rows.append(dict(task=task, data=dv, motif_class=cls, motif_only_auroc=round(m, 4), auroc_sd=round(sd, 4),
                         auroc_ci_lo=round(plo, 4), auroc_ci_hi=round(phi, 4), n_panel=len(panel),
                         n_pos=len(ps), n_neg=len(ns)))
    # threshold sweep: uncleaned vs comp_equalized arm (shared arm code, per split)
    tr = [seqs[i] for i in itr]; te = [seqs[i] for i in ite]; ytr, yte = y[itr], y[ite]
    masks, _ = arm_masks(tr, ytr, te, yte, tbp)
    mtr, mte = masks["comp_equalized"]
    eq_seqs = [s for s, m in zip(tr, mtr) if m] + [s for s, m in zip(te, mte) if m]
    eq_y = np.concatenate([ytr[mtr], yte[mte]])
    for arm, (S, Y) in [("uncleaned", ([seqs[i] for i in pool], y[pool])), ("comp_equalized", (eq_seqs, eq_y))]:
        aps, ans = R3S.cap_per_class(S, Y, 4000, SEED)
        apc, anc = C.motif_match.encode_sequences(aps), C.motif_match.encode_sequences(ans)
        brng = np.random.RandomState(SEED)
        for thr in R3S.THRESHOLDS:
            t_pt, t_lo, t_hi = R3S.real_tata_auroc_ci(apc, anc, tbp["pssm"], thr, BOOT, brng)
            thr_rows.append(dict(task=task, data=dv, arm=arm, threshold_bits=thr, tata_auroc=round(t_pt, 4),
                                 tata_ci_lo=round(t_lo, 4), tata_ci_hi=round(t_hi, 4), n_pos=len(aps), n_neg=len(ans)))
    return pd.DataFrame(rows), pd.DataFrame(thr_rows)


def run_x2(a):
    tbp = C.load_tbp()
    L = tbp["pssm"].shape[1]
    tic = R3S.total_ic(R3S.pssm_to_ppm(tbp["pssm"]))
    scram = R3S.scrambled_tbp_panel(tbp["pssm"], 20, SEED)
    rand = R3S.random_ic_panel(tic, L, 10, SEED, tol_bits=0.5)
    for task in ["human_enhancers_cohn", "nt_enhancers", R3S.POS_CONTROL]:
        for dv in DVS:
            if os.path.exists(ck(f"X2spec_{task}_{dv}")):
                continue
            t0 = time.time()
            s, t = x2_unit(task, dv, tbp, scram, rand)
            t.to_csv(ck(f"X2thr_{task}_{dv}"), index=False)
            s.to_csv(ck(f"X2spec_{task}_{dv}"), index=False)
            print(f"  [X2 {task} {dv}] {time.time() - t0:.0f}s", flush=True)
    aggregate("X2spec_", "specificity")
    aggregate("X2thr_", "threshold")


# ------------------------------------------------------------------ X3
def _auc1(y, p):
    """binary class-1 AUROC (diagnose.py) or macro one-vs-rest for multiclass."""
    return _hc.auroc(y, p)


def comp_block(y, pk, ck_, pc, cc_, unit, task, base):
    predk = pred_of(pk, ck_); predc = pred_of(pc, cc_)
    km = _hc.boot_metrics(y, predk, pk); cm = _hc.boot_metrics(y, predc, pc)
    rng = np.random.RandomState(SEED); n = len(y)
    fr = np.full(BOOT, np.nan); rs = np.full(BOOT, np.nan)
    for b in range(BOOT):
        i = rng.randint(0, n, n)
        if len(np.unique(y[i])) < len(np.unique(y)):
            continue
        dk = _auc1(y[i], pk[i]) - 0.5
        if dk != 0:
            fr[b] = (_auc1(y[i], pc[i]) - 0.5) / dk
        rs[b] = matthews_corrcoef(y[i], predk[i]) - matthews_corrcoef(y[i], predc[i])
    frac = (cm["auroc"] - 0.5) / (km["auroc"] - 0.5) if km["auroc"] != 0.5 else np.nan
    q = lambda v, p: float(np.nanpercentile(v, p))
    return dict(base, unit=unit, n_test=n, kmer_mcc=km["mcc"], kmer_mcc_lo=km["mcc_lo"], kmer_mcc_hi=km["mcc_hi"],
                kmer_auroc=km["auroc"], comp_mcc=cm["mcc"], comp_mcc_lo=cm["mcc_lo"], comp_mcc_hi=cm["mcc_hi"],
                comp_auroc=cm["auroc"], comp_auroc_lo=cm["auroc_lo"], comp_auroc_hi=cm["auroc_hi"],
                residual_mcc=km["mcc"] - cm["mcc"], residual_lo=q(rs, 2.5), residual_hi=q(rs, 97.5),
                comp_fraction=frac, comp_fraction_lo=q(fr, 2.5), comp_fraction_hi=q(fr, 97.5))


def x3_task(task):
    tr, ytr, te, yte = _hc.load_task(task)
    seqs = tr + te; y = np.concatenate([ytr, yte]); n_tr = len(tr); n = len(seqs)
    Cm = C.comp_signature(seqs, ks=(1, 2))
    base = dict(experiment="X3_composition", task=task, k=_hc.best_k(task))
    rows = []

    def kpred(unit, tr_idx, te_idx):
        z = np.load(os.path.join(PRED_DIR, f"{task}__{unit}.npz"))
        assert np.array_equal(z["te_idx"], te_idx) and np.array_equal(z["tr_idx"], tr_idx), (task, unit)
        return z["proba"].astype(np.float64), z["classes"]

    def cfit(tr_idx, te_idx):
        m = _hc.build_lgbm(len(np.unique(y[tr_idx])), SEED)
        m.fit(Cm[tr_idx], y[tr_idx])
        return m.predict_proba(Cm[te_idx]), m.classes_

    tr_idx, te_idx = np.arange(n_tr), np.arange(n_tr, n)
    pk, kc = kpred("original", tr_idx, te_idx)
    pc, cc_ = cfit(tr_idx, te_idx)
    rows.append(comp_block(y[te_idx], pk, kc, pc, cc_, "original", task, base))
    for tag in ["id80", "id50"]:
        comp, _ = _hc.load_clusters(task, tag)
        has_tr = np.zeros(comp.max() + 1, bool); has_tr[comp[:n_tr]] = True
        keep = ~has_tr[comp[n_tr:]]
        if keep.sum() >= 10 and len(np.unique(y[te_idx][keep])) == len(np.unique(y)):
            rows.append(comp_block(y[te_idx][keep], pk[keep], kc, pc[keep], cc_, f"test_purge_{tag}", task, base))
    rn = _hc.random_resplit(y, (n - n_tr) / n, SEED)
    for unit, is_test in [("resplit_random", rn)] + [
            (f"resplit_{tag}", _hc.cluster_resplit(_hc.load_clusters(task, tag)[0], y, (n - n_tr) / n, SEED))
            for tag in ["id80", "id50"]]:
        a_tr, a_te = np.where(~is_test)[0], np.where(is_test)[0]
        if not os.path.exists(os.path.join(PRED_DIR, f"{task}__{unit}.npz")):
            continue          # re-split infeasible in run_eval.py
        pk2, kc2 = kpred(unit, a_tr, a_te)
        pc2, cc2 = cfit(a_tr, a_te)
        rows.append(comp_block(y[a_te], pk2, kc2, pc2, cc2, unit, task, base))
    return pd.DataFrame(rows)


def run_x3(a):
    for task in SIZE_ORDER:
        cached(f"X3_{task}", lambda: x3_task(task))
    aggregate("X3_", "composition")


# ------------------------------------------------------------------ X4
def run_x4(a):
    sys.path.insert(0, os.path.join(_hc.UP, "fm_probe"))
    import extract_embeddings as XH          # numpy cache loaders only (torch is lazy inside, never called)
    import extract_embeddings_dnabert as XD
    import run_fm_paired as PP
    import run_fm_probe as P
    cache_dir = os.path.join(_hc.EXT, "fm_embeddings")
    models = [("HyenaDNA-tiny", XH, ["mean", "sep"], ["human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark"]),
              ("DNABERT-2-117M", XD, ["mean", "cls"], ["human_enhancers_cohn", "nt_enhancers"])]
    tbp = C.load_tbp()
    for mname, XM, pools, tasks in models:
        for task in tasks:
            for dv in ["original_released", "original", "id80", "id50"]:
                name = f"X4_{mname}_{task}_{dv}"
                if os.path.exists(ck(name)):
                    continue
                t0 = time.time()
                e_tr = XM.load_cached(cache_dir, task, "orig", "train"); e_te = XM.load_cached(cache_dir, task, "orig", "test")
                assert e_tr is not None and e_te is not None, f"missing {mname} cache for {task}; run fetch_fm_cache.py"
                tr, ytr0, te, yte0 = _hc.load_task(task)
                assert np.array_equal(np.asarray(e_tr["y"]).astype(np.int64), ytr0), "train embedding order/labels differ"
                assert np.array_equal(np.asarray(e_te["y"]).astype(np.int64), yte0), "test embedding order/labels differ"
                if dv == "original_released":
                    itr, ite = np.arange(len(tr)), np.arange(len(tr), len(tr) + len(te))
                    arms = {}
                    for arm in BASE_ARMS:
                        atr, _, ate, _ = P.arm_split_seqs(task, arm)
                        arms[arm] = (PP.membership_mask(tr, atr), PP.membership_mask(te, ate))
                else:
                    seqs, y, itr, ite = split_idx(task, dv.replace("original", "original"), nr=True)
                    s_tr = [seqs[i] for i in itr]; s_te = [seqs[i] for i in ite]
                    arms, _ = arm_masks(s_tr, np.concatenate([ytr0, yte0])[itr], s_te,
                                        np.concatenate([ytr0, yte0])[ite], tbp)
                y_all = np.concatenate([ytr0, yte0])
                ytr, yte = y_all[itr], y_all[ite]
                nc = len(np.unique(ytr))
                rows = []
                for pooling in pools:
                    E = np.vstack([e_tr[pooling], e_te[pooling]])
                    Xtr_o, Xte_o = E[itr], E[ite]
                    for head in ["lgbm", "lr"]:
                        rng = np.random.RandomState(SEED)     # identical stream to run_fm_paired per (task,pool,head)
                        H = PP.fit_head(Xtr_o, ytr, head, SEED)
                        pred_of_, proba_of = PP.predict(H, Xte_o)
                        of_mcc = PP._mcc(yte, pred_of_); of_auc = PP._auc(yte, proba_of, nc)
                        arm_list = BASE_ARMS + ([f"placebo_{x}" for x in BASE_ARMS] if dv != "original_released" else [])
                        for arm in arm_list:
                            mtr, mask = arms[arm]
                            pos = np.where(mask)[0]; y_c = yte[pos]
                            ok_clean = (ytr[mtr] == 0).sum() >= 10 and len(np.unique(ytr[mtr])) == nc
                            if ok_clean:
                                Hc = PP.fit_head(Xtr_o[mtr], ytr[mtr], head, SEED)
                                pred_cf, proba_cf = PP.predict(Hc, Xte_o)
                            ooc_mcc = PP._mcc(y_c, pred_of_[pos]); ooc_auc = PP._auc(y_c, proba_of[pos], nc)
                            coc_mcc = PP._mcc(y_c, pred_cf[pos]) if ok_clean else np.nan
                            coc_auc = PP._auc(y_c, proba_cf[pos], nc) if ok_clean else np.nan
                            n = len(pos); no = len(yte)
                            pdm = np.full(BOOT, np.nan); pda = np.full(BOOT, np.nan)
                            tem = np.full(BOOT, np.nan); tea = np.full(BOOT, np.nan)
                            for b in range(BOOT):
                                r = rng.randint(0, n, n); sub = pos[r]
                                if ok_clean and len(np.unique(y_c[r])) > 1:
                                    pdm[b] = matthews_corrcoef(y_c[r], pred_cf[sub]) - matthews_corrcoef(y_c[r], pred_of_[sub])
                                    pda[b] = PP._auc(y_c[r], proba_cf[sub], nc) - PP._auc(y_c[r], proba_of[sub], nc)
                                rf = rng.randint(0, no, no); mfull = mask[rf]
                                if len(np.unique(yte[rf])) > 1 and mfull.sum() > 1 and len(np.unique(yte[rf][mfull])) > 1:
                                    tem[b] = matthews_corrcoef(yte[rf][mfull], pred_of_[rf][mfull]) - matthews_corrcoef(yte[rf], pred_of_[rf])
                                    tea[b] = PP._auc(yte[rf][mfull], proba_of[rf][mfull], nc) - PP._auc(yte[rf], proba_of[rf], nc)
                            q = lambda v, p: float(np.nanpercentile(v, p)) if np.isfinite(v).any() else np.nan
                            rows.append(dict(experiment="X4_fm", model=mname, task=task, data=dv, arm=arm,
                                             pooling=pooling, head=head, n_train=len(ytr), n_test=int(no),
                                             n_clean_test=int(n), n_test_neg_removed=int(((yte == 0) & ~mask).sum()),
                                             orig_full_mcc=round(of_mcc, 4), orig_full_auroc=round(of_auc, 4),
                                             orig_on_clean_auroc=round(ooc_auc, 4), clean_on_clean_auroc=(round(coc_auc, 4) if coc_auc == coc_auc else np.nan),
                                             paired_delta_auroc=(round(coc_auc - ooc_auc, 4) if coc_auc == coc_auc else np.nan),
                                             paired_delta_auroc_lo=q(pda, 2.5), paired_delta_auroc_hi=q(pda, 97.5),
                                             test_effect_mcc=round(ooc_mcc - of_mcc, 4), test_effect_mcc_lo=q(tem, 2.5),
                                             test_effect_mcc_hi=q(tem, 97.5),
                                             test_effect_auroc=round(ooc_auc - of_auc, 4), test_effect_auroc_lo=q(tea, 2.5),
                                             test_effect_auroc_hi=q(tea, 97.5)))
                df = pd.DataFrame(rows)
                df.to_csv(ck(name) + ".tmp", index=False); os.replace(ck(name) + ".tmp", ck(name))
                print(f"  [X4 {mname} {task} {dv}] {time.time() - t0:.0f}s", flush=True)
    assert "torch" not in sys.modules, "torch was imported in the LightGBM process"
    aggregate("X4_", "fm")


# ------------------------------------------------------------------ X5 / X6
def x5_unit(task, dv, Bs=(2, 4, 8, 16)):
    d = RI.get_data(task, dv); k = _hc.best_k(task)
    tr, ytr, c1 = RI.cap_split(d["tr"], d["ytr"], RI.CAP, SEED)
    te, yte, c2 = RI.cap_split(d["te"], d["yte"], RI.CAP, SEED + 1)
    Xtr_g = featurize.kmer_spectrum(tr, k); Xte_g = featurize.kmer_spectrum(te, k)
    _, pred_g = RI.fit_pred(Xtr_g, ytr, Xte_g, SEED)
    mg = matthews_corrcoef(yte, pred_g)
    rows = []
    for B in Bs:
        Xtr_p = sparse.hstack([Xtr_g, featurize.binned_kmer_spectrum(tr, k, B)], format="csr")
        Xte_p = sparse.hstack([Xte_g, featurize.binned_kmer_spectrum(te, k, B)], format="csr")
        _, pred_p = RI.fit_pred(Xtr_p, ytr, Xte_p, SEED)
        del Xtr_p, Xte_p
        mp = matthews_corrcoef(yte, pred_p)
        rng = np.random.RandomState(SEED); n = len(yte); dd = np.empty(BOOT)
        for b in range(BOOT):
            i = rng.randint(0, n, n)
            dd[b] = matthews_corrcoef(yte[i], pred_p[i]) - matthews_corrcoef(yte[i], pred_g[i])
        lo, hi = np.percentile(dd, [2.5, 97.5])
        rows.append(dict(experiment="X5_graded", task=task, data=dv, k=k, n_bins=B, n_train=len(ytr), n_test=n,
                         capped=bool(c1 or c2), spectrum_mcc=round(mg, 4), positional_mcc=round(mp, 4),
                         delta_mcc=round(mp - mg, 4), delta_lo=round(float(lo), 4), delta_hi=round(float(hi), 4)))
    return pd.DataFrame(rows)


def run_x5(a):
    for dv in DVS:
        cached(f"X5_nt_splice_sites_all_{dv}", lambda: x5_unit("nt_splice_sites_all", dv))
    aggregate("X5_", "graded")


def run_x6(a):
    for task in RI.E3_TIER2:
        RI.run_e3(task, "id50")
    e3r = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(_hc.CKPT_DIR, "E3_*.csv")))]
    pd.concat(e3r, ignore_index=True).to_csv(os.path.join(_hc.RES, f"{_hc.PREFIX}_positional.csv"), index=False)
    print("re-aggregated homology_recluster_positional.csv", flush=True)


# ------------------------------------------------------------------ X7
def tier(m):
    return "solved" if m >= 0.7 else ("partial" if m >= 0.5 else "hard")


def x7_task(task):
    tr, ytr, te, yte = _hc.load_task(task)
    seqs = tr + te; y = np.concatenate([ytr, yte]); n_tr = len(tr); n = len(seqs)
    k = _hc.best_k(task)
    units = [("original", np.arange(n_tr), np.arange(n_tr, n))]
    for tag in ["id80", "id50"]:
        is_test = _hc.cluster_resplit(_hc.load_clusters(task, tag)[0], y, (n - n_tr) / n, SEED)
        if os.path.exists(os.path.join(PRED_DIR, f"{task}__resplit_{tag}.npz")):
            units.append((f"resplit_{tag}", np.where(~is_test)[0], np.where(is_test)[0]))
    rows = []
    for unit, a_tr, a_te in units:
        t0 = time.time()
        Xtr = featurize.kmer_spectrum([seqs[i] for i in a_tr], k)
        m = _hc.build_lr(len(np.unique(y[a_tr])), SEED)
        m.fit(Xtr, y[a_tr]); del Xtr
        Xte = featurize.kmer_spectrum([seqs[i] for i in a_te], k)
        proba = m.predict_proba(Xte); del Xte
        pred = m.classes_[np.argmax(proba, 1)]
        bm = _hc.boot_metrics(y[a_te], pred, proba)
        z = np.load(os.path.join(PRED_DIR, f"{task}__{unit}.npz"))
        assert np.array_equal(z["te_idx"], a_te)
        lg = float(matthews_corrcoef(y[a_te], pred_of(z["proba"], z["classes"])))
        rows.append(dict(experiment="X7_lr", task=task, unit=unit, k=k, n_train=len(a_tr), n_test=len(a_te),
                         lr_mcc=bm["mcc"], lr_mcc_lo=bm["mcc_lo"], lr_mcc_hi=bm["mcc_hi"], lr_auroc=bm["auroc"],
                         lgbm_mcc=lg, lr_tier=tier(bm["mcc"]), lgbm_tier=tier(lg), tier_agree=tier(bm["mcc"]) == tier(lg),
                         seconds=round(time.time() - t0, 1)))
        print(f"   X7 {task} {unit}: LR {bm['mcc']:.4f} LGBM {lg:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def run_x7(a):
    for task in SIZE_ORDER:
        cached(f"X7_{task}", lambda: x7_task(task))
    aggregate("X7_", "lr")


# ------------------------------------------------------------------ X8
KEY8 = ["human_nontata_promoters", "nt_promoter_tata", "demo_human_or_worm", "human_enhancers_cohn",
        "nt_enhancers", "human_ocr_ensembl", "drosophila_enhancers_stark", "nt_splice_sites_all"]


def x8_task(task, seeds=(42, 43, 44, 45, 46)):
    from sklearn.metrics import roc_auc_score  # noqa: F401
    tr, ytr, te, yte = _hc.load_task(task)
    seqs = tr + te; y = np.concatenate([ytr, yte]); n_tr = len(tr); n = len(seqs)
    k = _hc.best_k(task)
    is80 = _hc.cluster_resplit(_hc.load_clusters(task, "id80")[0], y, (n - n_tr) / n, SEED)
    units = [("original", np.arange(n_tr), np.arange(n_tr, n)), ("resplit_id80", np.where(~is80)[0], np.where(is80)[0])]
    X = featurize.kmer_spectrum(seqs, k)
    rows = []
    for unit, a_tr, a_te in units:
        Xtr, Xte, yt, ye = X[a_tr], X[a_te], y[a_tr], y[a_te]
        nc = len(np.unique(yt)); mccs, aucs = [], []
        for s in seeds:
            rng = np.random.RandomState(s)
            idx = rng.randint(0, len(yt), len(yt))
            if len(np.unique(yt[idx])) < nc:
                idx = np.concatenate([idx, [np.where(yt == c)[0][0] for c in range(nc)]])
            m = _hc.build_lgbm(nc, s)
            m.fit(Xtr[idx], yt[idx])
            proba = m.predict_proba(Xte); pred = m.classes_[np.argmax(proba, 1)]
            mccs.append(float(matthews_corrcoef(ye, pred))); aucs.append(_hc.auroc(ye, proba))
        mccs = np.array(mccs); tiers = [tier(v) for v in mccs]
        rows.append(dict(experiment="X8_modelseeds", task=task, unit=unit, k=k, n_seeds=len(seeds),
                         mcc_mean=mccs.mean(), mcc_sd=mccs.std(ddof=1), mcc_min=mccs.min(), mcc_max=mccs.max(),
                         auroc_mean=float(np.mean(aucs)), tier=tiers[0] if len(set(tiers)) == 1 else "VARIES",
                         tier_stable=len(set(tiers)) == 1, mccs=";".join(f"{v:.4f}" for v in mccs)))
        print(f"   X8 {task} {unit}: {mccs.mean():.4f} +/- {mccs.std(ddof=1):.4f} tiers {tiers}", flush=True)
    return pd.DataFrame(rows)


def run_x8(a):
    for task in [t for t in SIZE_ORDER if t in KEY8]:
        cached(f"X8_{task}", lambda: x8_task(task))
    aggregate("X8_", "modelseeds")


# ------------------------------------------------------------------ X9
GB9 = ["human_nontata_promoters", "human_enhancers_cohn", "human_enhancers_ensembl", "human_ocr_ensembl",
       "human_ensembl_regulatory", "drosophila_enhancers_stark", "dummy_mouse_enhancers_ensembl",
       "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm"]


def x9_task(task, pssm_by_coll):
    sys.path.insert(0, _hc.ROOT)
    import motif_importance as MI
    import run_interpretability as RIp
    tr, ytr, te, yte = _hc.load_task(task)
    seqs = tr + te; y = np.concatenate([ytr, yte]); n_tr = len(tr); n = len(seqs)
    k = _hc.best_k(task)
    coll = RIp.collection_for(task)
    M, usable = C.motif_match.tf_score_matrix(k, pssm_by_coll[coll], cache_dir=os.path.join(_hc.EXT, "motif_scores"),
                                              tag=f"{coll}_{C.motif_jaspar.RELEASE}_pc{C.motif_jaspar.PSEUDOCOUNT}")
    is80 = _hc.cluster_resplit(_hc.load_clusters(task, "id80")[0], y, (n - n_tr) / n, SEED)
    rows = []
    for unit, a_tr in [("original", np.arange(n_tr)), ("resplit_id80", np.where(~is80)[0])]:
        X = featurize.kmer_spectrum([seqs[i] for i in a_tr], k)
        m = _hc.build_lgbm(len(np.unique(y[a_tr])), SEED)
        m.fit(X, y[a_tr]); del X
        gain = m.booster_.feature_importance(importance_type="gain").astype(np.float64)
        assert len(gain) == M.shape[0] == len(MI.feature_names(k))
        z, p = RIp.enrichment_z(gain, M)
        fdr = RIp.bh_fdr(p)
        enr = np.where((fdr < 0.05) & (z > 0))[0]
        enr = enr[np.argsort(z[enr])[::-1]]
        rows.append(dict(experiment="X9_presence", task=task, unit=unit, k=k, collection=coll, n_usable_TF=len(usable),
                         n_enriched_TF=int(len(enr)), max_enrich_z=float(np.nanmax(z)),
                         top_enriched=";".join(usable[j]["name"] for j in enr[:5])))
        print(f"   X9 {task} {unit}: {len(enr)} enriched TFs (top {rows[-1]['top_enriched']})", flush=True)
    return pd.DataFrame(rows)


def run_x9(a):
    pssm_by_coll = {c: C.motif_jaspar.load_pssms(c) for c in ("vertebrates", "insects")}
    for task in [t for t in SIZE_ORDER if t in GB9]:
        cached(f"X9_{task}", lambda: x9_task(task, pssm_by_coll))
    aggregate("X9_", "presence")


# ------------------------------------------------------------------ X10
def run_x10(a):
    import zipfile
    task = "human_enhancers_cohn"
    co = pd.read_csv(os.path.join(_hc.ROOT, "localization", "coordinate_analysis.csv"))
    # pooled index of every GB file id: same class/file ordering as _hc._read_split_zip
    idx, i = {}, 0
    with zipfile.ZipFile(_hc._fetch_gb_zip(task)) as zf:
        for split in ("train", "test"):
            by = {}
            for nm in zf.namelist():
                pre = f"{task}/{split}/"
                if nm.startswith(pre) and nm.endswith(".txt") and nm[len(pre):].count("/") == 1:
                    c, fn = nm[len(pre):].split("/")
                    by.setdefault(c, []).append(nm)
            for c in sorted(by):
                for nm in sorted(by[c]):
                    if zf.read(nm).decode().strip():
                        idx[os.path.basename(nm)[:-4]] = i
                        i += 1
    tr, ytr, te, yte = _hc.load_task(task)
    assert i == len(tr) + len(te)
    co["pooled_idx"] = co["id"].map(idx)
    assert co["pooled_idx"].notna().all(), "coordinate ids not all found in the archive"
    # analyze_cohn.py appended the POSITIVE rows with a different column order (id,chr,start,end,strand,
    # flagged,tss_overlap,klass,...), so under the file header a positive's TSS overlap sits in 'hit_start' and
    # its klass is empty. Parse it explicitly and check against the documented 2,168/13,895 (15.6 %).
    is_neg = co["klass"] == "negative"
    co.loc[~is_neg, "klass"] = "positive"
    co.loc[~is_neg, "tss_overlap"] = co.loc[~is_neg, "hit_start"].astype(str) == "True"
    co["tss_overlap"] = co["tss_overlap"].astype(str) == "True"
    co["flagged"] = co["flagged"].astype(str) == "True"
    assert int(co.loc[~is_neg, "tss_overlap"].sum()) == 2168 and int((~is_neg).sum()) == 13895
    y = np.concatenate([ytr, yte])
    lab_ok = np.where(co["klass"] == "negative", 0, 1) == y[co["pooled_idx"].astype(int)]
    rows = []
    for dv in DVS:
        if dv == "original":
            keep = np.ones(len(y), bool)
        else:
            comp, _ = _hc.load_clusters(task, dv)
            keep = np.zeros(len(y), bool); keep[_hc.nr_representatives(comp, y, SEED)] = True
        sub = co[keep[co["pooled_idx"].astype(int)]]
        neg = sub[sub.klass == "negative"]; pos = sub[sub.klass == "positive"]
        fl = neg[neg.flagged.astype(bool)]; un = neg[~neg.flagged.astype(bool)]
        a_, b_ = int(fl.tss_overlap.sum()), int(len(fl) - fl.tss_overlap.sum())
        c_, d_ = int(un.tss_overlap.sum()), int(len(un) - un.tss_overlap.sum())
        orr, p = fisher_exact([[a_, b_], [c_, d_]])
        rf, ru = a_ / max(len(fl), 1), c_ / max(len(un), 1)
        rows.append(dict(experiment="X10_coords", task=task, data=dv, n_neg=len(neg), n_flagged=len(fl),
                         flagged_tss_rate=round(rf, 4), unflagged_tss_rate=round(ru, 4),
                         rate_ratio=round(rf / ru, 4) if ru > 0 else np.nan, fisher_odds_ratio=orr, fisher_p=p,
                         n_pos=len(pos), pos_tss_rate=round(float(pos.tss_overlap.mean()), 4),
                         labels_consistent=bool(lab_ok.all())))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv("coords"), index=False)
    print(df.to_string(), flush=True)


# ------------------------------------------------------------------ main
EXPS = {"X1": run_x1, "X2": run_x2, "X3": run_x3, "X4": run_x4, "X5": run_x5, "X6": run_x6, "X7": run_x7,
        "X8": run_x8, "X9": run_x9, "X10": run_x10}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="X10,X1,X2,X5,X3,X4,X6,X8,X9,X7")
    a = ap.parse_args()
    for e in [x for x in a.exp.split(",") if x]:
        t0 = time.time()
        print(f"\n===== {e} start {time.strftime('%H:%M:%S')}", flush=True)
        EXPS[e](a)
        print(f"===== {e} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
