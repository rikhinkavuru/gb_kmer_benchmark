#!/usr/bin/env python3
"""GUE as a third, independently curated suite: the paper's pipeline + route labels, unchanged.

Responds to Reviewer 2, comment 2 ("two suites do not support 'any genomic sequence-classification
benchmark'"). Applies the SAME CPU-only pipeline as the paper to the 28 GUE datasets of DNABERT-2
(Zhou et al., ICLR 2024) and labels each with diagnose.py's documented, fixed decision rules.

=====================================================================================================
PRE-REGISTERED ANALYSIS PLAN (written before any model was fit on GUE; thresholds are the paper's,
NOT tuned on GUE)
=====================================================================================================
Data. Official GUE archive (DNABERT-2 README Google-Drive link; SHA-256 pinned in gue_data.py), the
official train / dev / test CSVs. The GUE dev split is NOT used for model selection so that the
protocol is identical to the paper's: validation is a stratified 80/20 carve of TRAIN, the test split
is touched exactly once. Row counts are checked against DNABERT-2 Table 12 (gue_rowcounts.csv).
Scope = the 28 GUE datasets; GUE+ (EPI sequence pairs, 5-10 kb species tasks) is out of scope.
Processing order (priority): human -> mouse -> yeast EMP -> covid.

Per task (seed 42 everywhere; 1000-resample percentile bootstrap, test sequence = resampling unit):
 1. k selection: k in {3,4,5,6}; LightGBM with models.LGBM_PARAMS (fixed hyper-parameters) on the
    L1-normalised k-mer spectrum (featurize.kmer_spectrum); fit on the 80 % carve, pick max
    validation MCC (strict '>' so ties go to the smaller k) -- identical to kselect.best_k_val /
    diagnose._select_k. The ONLY deviation from the paper's runs is n_jobs=2 (machine sharing);
    LightGBM runs in deterministic mode.
 2. Learnability: refit on the full train split at the selected k; test MCC [CI], AUROC [CI]
    (one-vs-rest macro for multiclass), macro-F1 [CI] (the DNABERT-2 metric for covid).
 3. Composition-only baseline: the same LightGBM on the 20-d composition signature
    (_common.comp_signature: 4 mono + 16 dinucleotide frequencies); MCC [CI], AUROC [CI].
    Composition fraction = (AUROC_comp - 0.5)/(AUROC_full - 0.5), paired bootstrap -- the exact
    definition and bootstrap of diagnose._composition_fraction (binary). For multiclass tasks the
    same ratio on macro-OvR AUROC is reported as a labelled EXTENSION (not in diagnose.py).
 4. Positional rescue (Route 2): global spectrum vs global (+) B=8 position-binned spectrum at the
    selected k, paired 1000-bootstrap delta-MCC. Train/test stratified-capped at 20,000 per split
    exactly as run_positional_sweep.py (cap_split, seeds 42 / 43), same capped set for both arms.
 5. Route 1 motif + Route 3 negatives (binary tasks): diagnose.py's code path on the pooled released
    sequences (train+dev+test, capped at 4,000 per class): max positive-direction motif-only AUROC over
    a curated TF panel at 1.0 bits/pos + motif richness; TATA (vertebrate TBP, 1.3 bits/pos, used for
    every species per the paper's convention) neg-excess, TATA AUROC, GC AUROC, and the TATA-flag
    cleaning effect. Panel: diagnose's CURATED_VERT for human and mouse. Yeast: a species-matched
    fungal panel fixed a priori below (the paper likewise used an insect panel for Drosophila);
    the vertebrate-panel result is also stored as a sensitivity (verdict_vertebrate_panel).
 6. diagnose.py extras (binary): shuffled-negative control, equalized-split delta (negatives'
    joint GC+16-dinucleotide distribution matched to the positives'), and (all tasks) train/test
    8-mer-Jaccard redundancy.
 7. Route label = diagnose._verdict(card), unchanged: SOLVED if MCC>=0.7; Route 2 if delta
    significant and >0.2; Route 3 if TATA neg-excess>=0.10 & TATA AUROC<0.5 & cleaning moves it to
    chance; Route 1 if richness>0.2 & max motif AUROC<0.6; else HARD. Tier (paper Table 1):
    solved >= 0.7, partial >= 0.5, hard < 0.5.
 8. Composition-dominance flag (secondary, NOT part of diagnose's verdict; the reviewer-facing
    "contamination-route signature" = composition-only close to the full model): flag when the
    composition fraction >= 0.80 AND full AUROC >= 0.60 (so the ratio's denominator is stable).
    0.80 is fixed from the paper's own numbers before seeing GUE: ~midpoint between the clean
    Drosophila control (0.695) and the contaminated cohn / nt_enhancers (0.915 / 0.928) in
    results/upgrades/composition_fraction.csv. A flag is a signature, not a verdict: whether the
    composition gap is a negative-sampling artifact or genuine biology is read against GUE's
    documented negative construction (gue_data.NEG_CONSTRUCTION) and the equalized-split delta.
 9. Pre-stated predictions (falsifiable checks of the instrument, stated before running):
    P1 mouse TF negatives are dinucleotide shuffles of the positives => composition-only AUROC ~0.5
       (within 0.05) and composition fraction ~0; the equalized split retains most negatives.
    P2 human TF negatives are GC-matched => GC AUROC within 0.05 of 0.5.
    P3 splice/reconstructed carries fixed-position donor/acceptor signal => a significant positive
       positional delta; it is the only GUE task expected to be able to meet the Route-2 rule
       (it can only be labelled Route 2 if its position-blind MCC is < 0.7).
    P4 no prediction on magnitudes for the promoter tasks (random-substitution negatives).
10. Published comparison: per-dataset DNABERT-2 / NT / DNABERT / HyenaDNA numbers transcribed from
    DNABERT-2 Tables 6 and 9 (gue_data.PUBLISHED), MCC x100 (covid: F1 x100).

Output: one JSON report card per task in results/upgrades/gue_cards/<task>.json (checkpoint; a task
whose card exists is skipped unless --force). summarize_gue.py assembles the CSV tables.
CPU only, no torch. <=2 threads (OMP_NUM_THREADS, LightGBM n_jobs).
"""
import os

N_THREADS = int(os.environ.get("GUE_THREADS", "2"))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(N_THREADS))

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.abspath(os.path.join(HERE, ".."))
for _d in (HERE, UP, os.path.join(UP, "diagnose"), os.path.join(UP, "positional_sweep"),
           os.path.join(UP, "composition_clean")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import _common as C  # noqa: E402
import featurize  # noqa: E402
import models  # noqa: E402

models.LGBM_PARAMS["n_jobs"] = N_THREADS          # in-process only; shared file untouched
import diagnose as DG  # noqa: E402
import gue_data as G  # noqa: E402
from run_discriminability import CURATED_VERT, resolve_curated  # noqa: E402
from run_positional_sweep import cap_split  # noqa: E402

SEED = 42
BOOT = 1000
KS = (3, 4, 5, 6)
N_BINS = 8
POS_CAP = 20000
MOTIF_CAP = 4000
COMP_FLAG_FRAC = 0.80
COMP_FLAG_MIN_AUROC = 0.60
CHUNK = 5000
CARD_DIR = os.path.join(C.RESULTS_DIR, "gue_cards")

# a-priori species-matched fungal panel for the yeast EMP tasks (JASPAR 2024 CORE fungi)
CURATED_FUNGI = {
    "GRF/nucleosome-depleting": ["REB1", "ABF1", "RAP1", "TBF1", "RSC3", "RSC30"],
    "bHLH/E-box": ["CBF1", "PHO4"],
    "MADS/cell-cycle": ["MCM1", "MBP1", "SWI4"],
    "bZIP/stress": ["GCN4", "MSN2", "YAP1"],
    "TATA": ["SPT15"],
}


# ----------------------------------------------------------------------------- featurization
def kspec(seqs, k):
    """featurize.kmer_spectrum in row chunks (identical output; bounded peak RAM)."""
    if len(seqs) <= CHUNK:
        return featurize.kmer_spectrum(seqs, k)
    return sparse.vstack([featurize.kmer_spectrum(seqs[i:i + CHUNK], k)
                          for i in range(0, len(seqs), CHUNK)], format="csr")


def pos_arm(seqs, k, B):
    """Global spectrum (+) B-bin positional spectrum (run_positional_sweep.featurize_arm), chunked."""
    def one(ss):
        g = featurize.kmer_spectrum(ss, k)
        return sparse.hstack([g, featurize.binned_kmer_spectrum(ss, k, B)], format="csr")
    if len(seqs) <= CHUNK:
        return one(seqs)
    return sparse.vstack([one(seqs[i:i + CHUNK]) for i in range(0, len(seqs), CHUNK)], format="csr")


# ----------------------------------------------------------------------------- model helpers
def fit_proba(Xtr, ytr, Xte, seed=SEED):
    m = models.build_model("lgbm", seed, len(np.unique(ytr)))
    m.fit(Xtr, ytr)
    return m.classes_, m.predict_proba(Xte)


def _auroc(y, proba, nc):
    return (roc_auc_score(y, proba[:, 1]) if nc == 2 else
            roc_auc_score(y, proba, multi_class="ovr", average="macro"))


def metrics_boot(yte, classes, proba, nc, seed=SEED, B=BOOT):
    """Point + percentile CIs; identical resampling to _common.fit_eval_boot (fresh RandomState(seed)),
    plus macro-F1 on the same resamples."""
    rng = np.random.RandomState(seed)
    pred = classes[np.argmax(proba, axis=1)]
    out = dict(acc=float((pred == yte).mean()), mcc=float(matthews_corrcoef(yte, pred)),
               f1_macro=float(f1_score(yte, pred, average="macro")))
    try:
        out["auroc"] = float(_auroc(yte, proba, nc))
    except ValueError:
        out["auroc"] = float("nan")
    n = len(yte)
    mm = np.full(B, np.nan); aa = np.full(B, np.nan); ff = np.full(B, np.nan)
    for b in range(B):
        idx = rng.randint(0, n, n)
        if len(np.unique(yte[idx])) < 2:
            continue
        mm[b] = matthews_corrcoef(yte[idx], pred[idx])
        ff[b] = f1_score(yte[idx], pred[idx], average="macro")
        try:
            aa[b] = _auroc(yte[idx], proba[idx], nc)
        except ValueError:
            pass
    for key, arr in (("mcc", mm), ("auroc", aa), ("f1_macro", ff)):
        out[f"{key}_ci"] = [float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))]
    out["n_test"] = int(n)
    return out, pred


def select_k(tr_seqs, ytr, seed=SEED, ks=KS):
    """Validation-selected k: stratified 80/20 carve of TRAIN, max val MCC, strict '>' (ties -> smaller k).
    Same logic as kselect.best_k_val and diagnose._select_k; also returns the per-k val MCCs."""
    tr, va = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=seed)
                  .split(np.zeros(len(ytr)), ytr))
    best_k, best, vals = ks[0], -2.0, {}
    for k in ks:
        X = kspec(tr_seqs, k)
        cls, pr = fit_proba(X[tr], ytr[tr], X[va], seed)
        mcc = float(matthews_corrcoef(ytr[va], cls[np.argmax(pr, axis=1)]))
        vals[k] = mcc
        if mcc > best:
            best, best_k = mcc, k
        del X
    return best_k, vals


def comp_fraction(yte, p_full, p_comp, nc, seed=SEED, B=BOOT):
    """diagnose._composition_fraction's formula and paired bootstrap (fresh RandomState(seed)).
    Binary: p_* are class-1 probabilities. Multiclass (EXTENSION): p_* are full proba matrices and the
    AUROC is one-vs-rest macro."""
    def au(yy, p):
        if len(np.unique(yy)) < (2 if nc == 2 else nc):
            return float("nan")
        try:
            return float(roc_auc_score(yy, p) if nc == 2 else
                         roc_auc_score(yy, p, multi_class="ovr", average="macro"))
        except ValueError:
            return float("nan")
    full_pt, comp_pt = au(yte, p_full), au(yte, p_comp)
    frac_pt = (comp_pt - 0.5) / (full_pt - 0.5) if (full_pt - 0.5) != 0 else float("nan")
    n = len(yte); rng = np.random.RandomState(seed); fr = np.full(B, np.nan)
    for b in range(B):
        idx = rng.randint(0, n, n)
        yb = yte[idx]
        if len(np.unique(yb)) < 2:
            continue
        d = au(yb, p_full[idx]) - 0.5
        if d != 0 and d == d:
            fr[b] = (au(yb, p_comp[idx]) - 0.5) / d
    return dict(value=round(float(frac_pt), 4),
                ci=[round(float(np.nanpercentile(fr, 2.5)), 4), round(float(np.nanpercentile(fr, 97.5)), 4)],
                full_auroc=round(float(full_pt), 4), comp_only_auroc=round(float(comp_pt), 4),
                definition="(AUROC_comp_only-0.5)/(AUROC_full-0.5)" +
                           ("" if nc == 2 else " on macro-OvR AUROC [EXTENSION: multiclass]"))


def equalized_delta(tr_seqs, ytr, te_seqs, yte, k, before, seed=SEED, boot=BOOT):
    """diagnose._equalized_delta with the 'before' fit reused from the learnability fit (identical
    deterministic model, so identical point estimates) instead of refitting it."""
    import run_composition_clean as CC
    rng = np.random.RandomState(seed)
    comp_tr = C.comp_signature(tr_seqs, ks=(1, 2)); comp_te = C.comp_signature(te_seqs, ks=(1, 2))
    prop = CC.fit_propensity(np.vstack([comp_tr, comp_te]), np.concatenate([ytr, yte]), seed)
    p_tr = prop.predict_proba(comp_tr)[:, 1]; p_te = prop.predict_proba(comp_te)[:, 1]
    keep_tr = CC.match_proportional(p_tr[ytr == 1], p_tr[ytr == 0], DG.PROP_BINS, rng)
    keep_te = CC.match_proportional(p_te[yte == 1], p_te[yte == 0], DG.PROP_BINS, rng)
    n_neg = max(int((ytr == 0).sum() + (yte == 0).sum()), 1)
    retained = round(float((len(keep_tr) + len(keep_te)) / n_neg), 4)
    if len(keep_tr) < 10 or len(keep_te) < 10:
        return dict(feasible=False, reason="fewer than 10 negatives survive equalization in a split",
                    retained_neg_frac=retained)
    ce_tr, ce_ytr = CC.negatives_kept_split(tr_seqs, ytr, keep_tr)
    ce_te, ce_yte = CC.negatives_kept_split(te_seqs, yte, keep_te)
    cls, pr = fit_proba(kspec(ce_tr, k), ce_ytr, kspec(ce_te, k), seed)
    after, _ = metrics_boot(ce_yte, cls, pr, 2, seed, boot)
    # composition-only on the equalized split (should fall to ~chance if equalization worked)
    ccls, cpr = fit_proba(C.comp_signature(ce_tr, ks=(1, 2)), ce_ytr, C.comp_signature(ce_te, ks=(1, 2)), seed)
    comp_after, _ = metrics_boot(ce_yte, ccls, cpr, 2, seed, boot)
    neg_before = np.vstack([comp_tr[ytr == 0], comp_te[yte == 0]])
    neg_after = np.vstack([comp_tr[ytr == 0][keep_tr], comp_te[yte == 0][keep_te]])
    pos_all = np.vstack([comp_tr[ytr == 1], comp_te[yte == 1]])
    return dict(feasible=True,
                mcc_before=round(before["mcc"], 6), mcc_after=round(after["mcc"], 6),
                mcc_after_ci=[round(x, 4) for x in after["mcc_ci"]],
                delta_mcc=round(after["mcc"] - before["mcc"], 4),
                auroc_before=round(before["auroc"], 6), auroc_after=round(after["auroc"], 6),
                comp_only_auroc_after=round(comp_after["auroc"], 6),
                comp_only_mcc_after=round(comp_after["mcc"], 6),
                n_test_after=int(len(ce_yte)), pos_frac_test_after=round(float(ce_yte.mean()), 4),
                retained_neg_frac=retained,
                max_abs_smd_before=round(float(np.nanmax(np.abs(CC.smd(pos_all, neg_before)))), 4),
                max_abs_smd_after=round(float(np.nanmax(np.abs(CC.smd(pos_all, neg_after)))), 4))


def motif_block(pos, neg, curated, rng):
    """diagnose.diagnose's Route-1 / Route-3 block, verbatim logic (pos/neg already capped)."""
    pcodes = C.motif_match.encode_sequences(pos); ncodes = C.motif_match.encode_sequences(neg)
    best_auc, best_tf, rich, per_tf = 0.5, "", [], {}
    for fam, name, m in curated:
        pc = C.motif_match.count_hits_batch(pcodes, m["pssm"], DG.THR_BITS)
        ncc = C.motif_match.count_hits_batch(ncodes, m["pssm"], DG.THR_BITS)
        auc = C.auc_counts(pc.astype(np.int64), ncc.astype(np.int64))
        per_tf[name] = round(float(auc), 4)
        rich.append((pc.mean() + ncc.mean()) / 2)
        if auc > best_auc:
            best_auc, best_tf = auc, name
    r1 = dict(max_pos_motif_auroc=round(float(best_auc), 4), best_TF=best_tf,
              motif_richness=round(float(np.mean(rich)), 3), per_TF_auroc=per_tf, n_TFs=len(curated))
    tbp = C.load_tbp()
    ptata = C.motif_match.count_hits_batch(pcodes, tbp["pssm"], DG.TATA_BITS)
    ntata = C.motif_match.count_hits_batch(ncodes, tbp["pssm"], DG.TATA_BITS)
    pos_rate = float((ptata >= 1).mean()); neg_rate = float((ntata >= 1).mean())
    tata_auc = C.auc_counts(ptata.astype(np.int64), ntata.astype(np.int64))
    gc_auc = C.auc_float(C.gc_content(pos), C.gc_content(neg))
    neg_idx = np.arange(len(neg)); flagged = ntata >= 1
    keep_fl = int(round(pos_rate * len(neg))); fl_idx = neg_idx[flagged]
    if len(fl_idx) > keep_fl:
        sel = rng.permutation(len(fl_idx))[:keep_fl]
        keep = np.sort(np.concatenate([neg_idx[~flagged], fl_idx[sel]]))
    else:
        keep = neg_idx
    cleaned = C.auc_counts(ptata.astype(np.int64), ntata[keep].astype(np.int64))
    r3 = dict(tata_pos_rate=round(pos_rate, 4), tata_neg_rate=round(neg_rate, 4),
              tata_neg_excess=round(neg_rate - pos_rate, 4),
              tata_motif_auroc=round(float(tata_auc), 4), gc_auroc=round(float(gc_auc), 4),
              cleaned_tata_auroc=round(float(cleaned), 4),
              cleaning_moves_to_chance=bool(abs(cleaned - 0.5) < abs(tata_auc - 0.5)),
              frac_neg_removed=round(1 - len(keep) / max(len(neg), 1), 4))
    return r1, r3, tbp


def tier(mcc):
    return "solved" if mcc >= 0.7 else ("partial" if mcc >= 0.5 else "hard")


# ----------------------------------------------------------------------------- per task
def run_task(task, boot=BOOT, seed=SEED, log=print):
    T0 = time.time(); timing = {}
    d = G.load_gue(task)
    tr_seqs, ytr, te_seqs, yte = d["train_seqs"], d["y_train"], d["test_seqs"], d["y_test"]
    nc = d["n_classes"]; binary = (nc == 2 and set(np.unique(ytr)) == {0, 1})
    species = G.TASKS[task][1]
    log(f"[{task}] species={species} classes={nc} n_train={len(ytr)} n_dev={len(d['y_dev'])} "
        f"n_test={len(yte)}")

    # 1. k selection
    t = time.time(); k, vals = select_k(tr_seqs, ytr, seed); timing["select_k"] = time.time() - t
    log(f"  k={k}  val MCC " + " ".join(f"k{kk}={v:.3f}" for kk, v in vals.items())
        + f"  ({timing['select_k']:.0f}s)")

    # 2. learnability
    t = time.time()
    Xtr = kspec(tr_seqs, k); Xte = kspec(te_seqs, k)
    cls, proba = fit_proba(Xtr, ytr, Xte, seed)
    learn, pred_g = metrics_boot(yte, cls, proba, nc, seed, boot)
    del Xtr, Xte
    timing["learn"] = time.time() - t
    log(f"  learnability MCC={learn['mcc']:.4f} [{learn['mcc_ci'][0]:.3f},{learn['mcc_ci'][1]:.3f}] "
        f"AUROC={learn['auroc']:.4f} F1={learn['f1_macro']:.4f} ({timing['learn']:.0f}s)")

    # 3. composition-only + fraction
    t = time.time()
    ccls, cproba = fit_proba(C.comp_signature(tr_seqs, ks=(1, 2)), ytr,
                             C.comp_signature(te_seqs, ks=(1, 2)), seed)
    comp, _ = metrics_boot(yte, ccls, cproba, nc, seed, boot)
    if binary:
        c1 = int(np.where(cls == 1)[0][0]); cc1 = int(np.where(ccls == 1)[0][0])
        cf = comp_fraction(yte, proba[:, c1], cproba[:, cc1], 2, seed, boot)
    else:
        cf = comp_fraction(yte, proba, cproba, nc, seed, boot)
    timing["comp"] = time.time() - t
    log(f"  comp-only MCC={comp['mcc']:.4f} AUROC={comp['auroc']:.4f} | comp fraction "
        f"{cf['value']:+.3f} {cf['ci']} ({timing['comp']:.0f}s)")

    card = dict(task=task, species=species, family=G.TASKS[task][2], neg_construction=
                G.NEG_CONSTRUCTION.get(G.neg_family(task), ""), n_classes=nc, binary=binary,
                n_train=len(ytr), n_dev=len(d["y_dev"]), n_test=len(yte),
                pos_frac_test=(round(float(yte.mean()), 4) if binary else None),
                k=k, val_mcc={str(kk): round(v, 6) for kk, v in vals.items()},
                learnability=dict(mcc=learn["mcc"], mcc_ci=learn["mcc_ci"], auroc=learn["auroc"],
                                  auroc_ci=learn["auroc_ci"], f1_macro=learn["f1_macro"],
                                  f1_macro_ci=learn["f1_macro_ci"], acc=learn["acc"],
                                  n_test=learn["n_test"]),
                composition_only=dict(mcc=comp["mcc"], mcc_ci=comp["mcc_ci"], auroc=comp["auroc"],
                                      auroc_ci=comp["auroc_ci"], f1_macro=comp["f1_macro"]),
                composition_fraction=cf, tier=tier(learn["mcc"]))

    # 4. positional (Route 2), capped exactly as run_positional_sweep
    t = time.time()
    ptr, pytr, cap1 = cap_split(tr_seqs, ytr, POS_CAP, seed)
    pte, pyte, cap2 = cap_split(te_seqs, yte, POS_CAP, seed + 1)
    if cap1 or cap2:
        c_, p_ = fit_proba(kspec(ptr, k), pytr, kspec(pte, k), seed)
        pg = c_[np.argmax(p_, axis=1)]
    else:
        pg = pred_g                                   # identical deterministic fit, reuse
    c_, p_ = fit_proba(pos_arm(ptr, k, N_BINS), pytr, pos_arm(pte, k, N_BINS), seed)
    pp = c_[np.argmax(p_, axis=1)]
    rng = np.random.RandomState(seed); n = len(pyte); dd = np.empty(boot)
    for b in range(boot):
        idx = rng.randint(0, n, n)
        dd[b] = matthews_corrcoef(pyte[idx], pp[idx]) - matthews_corrcoef(pyte[idx], pg[idx])
    mg, mp = float(matthews_corrcoef(pyte, pg)), float(matthews_corrcoef(pyte, pp))
    dlo, dhi = np.percentile(dd, [2.5, 97.5])
    card["route2_positional"] = dict(spectrum_mcc=round(mg, 4), positional_mcc=round(mp, 4),
                                     delta_mcc=round(mp - mg, 4),
                                     delta_ci=[round(float(dlo), 4), round(float(dhi), 4)],
                                     significant=bool(not (dlo <= 0 <= dhi)), n_bins=N_BINS,
                                     capped=bool(cap1 or cap2), n_train_used=len(pytr), n_test_used=n)
    timing["positional"] = time.time() - t
    log(f"  positional B=1 {mg:.4f} -> B=8 {mp:.4f}  delta={mp - mg:+.4f} [{dlo:+.3f},{dhi:+.3f}]"
        f"{' (capped)' if cap1 or cap2 else ''} ({timing['positional']:.0f}s)")

    # 5-6. motif / negatives / diagnose extras (binary only)
    if binary:
        t = time.time()
        seqs_all = tr_seqs + d["dev_seqs"] + te_seqs
        y_all = np.concatenate([ytr, d["y_dev"], yte])
        mrng = np.random.RandomState(seed)
        pos = [seqs_all[i] for i in np.where(y_all == 1)[0]]
        neg = [seqs_all[i] for i in np.where(y_all == 0)[0]]
        if len(pos) > MOTIF_CAP:
            pos = [pos[i] for i in mrng.choice(len(pos), MOTIF_CAP, replace=False)]
        if len(neg) > MOTIF_CAP:
            neg = [neg[i] for i in mrng.choice(len(neg), MOTIF_CAP, replace=False)]
        vert, miss_v = resolve_curated(CURATED_VERT, C.motif_jaspar.load_pssms("vertebrates"))
        if species == "yeast":
            fungi, miss_f = resolve_curated(CURATED_FUNGI, C.motif_jaspar.load_pssms("fungi"))
            r1, r3, tbp = motif_block(pos, neg, fungi, np.random.RandomState(seed))
            r1["panel"] = "curated fungi (JASPAR2024 CORE fungi)"; r1["missing"] = miss_f
            r1v, _, _ = motif_block(pos, neg, vert, np.random.RandomState(seed))
            r1v["panel"] = "curated vertebrate (diagnose default)"
            card["route1_motif_vertebrate_panel"] = r1v
        else:
            r1, r3, tbp = motif_block(pos, neg, vert, np.random.RandomState(seed))
            r1["panel"] = "curated vertebrate (diagnose default)"; r1["missing"] = miss_v
        card["route1_motif"] = r1
        card["route3_negatives"] = r3
        card["shuffled_negative_control"] = DG._shuffled_negative_control(list(pos), tbp, seed)
        timing["motif"] = time.time() - t
        log(f"  route1 max motif AUROC={r1['max_pos_motif_auroc']:.3f} ({r1['best_TF']}) richness="
            f"{r1['motif_richness']:.2f} | route3 TATA excess={r3['tata_neg_excess']:+.3f} TATA AUROC="
            f"{r3['tata_motif_auroc']:.3f} GC AUROC={r3['gc_auroc']:.3f} ({timing['motif']:.0f}s)")
        t = time.time()
        card["equalized_delta"] = equalized_delta(tr_seqs, ytr, te_seqs, yte, k, learn, seed, boot)
        timing["equalized"] = time.time() - t
        ed = card["equalized_delta"]
        if ed.get("feasible"):
            log(f"  equalized: MCC {ed['mcc_before']:.3f} -> {ed['mcc_after']:.3f} ({ed['delta_mcc']:+.3f}); "
                f"comp-only AUROC after {ed['comp_only_auroc_after']:.3f}; retained neg "
                f"{ed['retained_neg_frac']:.1%} ({timing['equalized']:.0f}s)")
        else:
            log(f"  equalized: infeasible ({ed['reason']})")

    t = time.time()
    card["redundancy"] = DG._redundancy(tr_seqs, te_seqs)
    timing["redundancy"] = time.time() - t
    log(f"  redundancy >=0.7 Jaccard: {card['redundancy']['frac_ge_0_7']:.4f} "
        f"(verbatim {card['redundancy']['frac_verbatim']:.4f}) ({timing['redundancy']:.0f}s)")

    # 7-8. verdict + composition flag
    card["verdict"] = DG._verdict(card)
    if "route1_motif_vertebrate_panel" in card:
        alt = dict(card); alt["route1_motif"] = card["route1_motif_vertebrate_panel"]
        card["verdict_vertebrate_panel"] = DG._verdict(alt)
    card["composition_dominated"] = bool(cf["full_auroc"] >= COMP_FLAG_MIN_AUROC and
                                         cf["value"] >= COMP_FLAG_FRAC)
    card["composition_flag_rule"] = (f"composition fraction >= {COMP_FLAG_FRAC} and full AUROC >= "
                                     f"{COMP_FLAG_MIN_AUROC}" + ("" if binary else " [multiclass extension]"))
    card["timing_s"] = {kk: round(v, 1) for kk, v in timing.items()}
    card["total_s"] = round(time.time() - T0, 1)
    card["n_jobs"] = N_THREADS
    import lightgbm
    card["lightgbm_version"] = lightgbm.__version__
    log(f"  VERDICT: {card['verdict']} | tier={card['tier']} | composition_dominated="
        f"{card['composition_dominated']} | total {card['total_s']:.0f}s")
    return card


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=",".join(G.ORDER))
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(CARD_DIR, exist_ok=True)
    for task in [t for t in args.tasks.split(",") if t]:
        out = os.path.join(CARD_DIR, f"{task}.json")
        if os.path.exists(out) and not args.force:
            print(f"[{task}] card exists, skipping", flush=True)
            continue
        card = run_task(task, args.boot, args.seed, log=lambda s: print(s, flush=True))
        tmp = out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(card, fh, indent=2)
        os.replace(tmp, out)
        print(f"[{task}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
