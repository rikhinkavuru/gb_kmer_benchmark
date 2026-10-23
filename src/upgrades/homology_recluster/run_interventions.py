#!/usr/bin/env python3
"""Step 4 -- the key intervention experiments re-run on HOMOLOGY-CLEAN data.

PRE-REGISTERED ANALYSIS PLAN (written 2026-09-24 before any run of this file)
----------------------------------------------------------------------------
Data. BEFORE = the paper's setting (provided split, all sequences). AFTER = homology-clean (HC) data
at threshold T (id80 primary, id50 strict) = _hc.hc_data(): the cluster-aware re-split of run_eval.py
(whole MMseqs2 components on one side, label-stratified, original test fraction, seed 42) followed by
non-redundant (NR) de-duplication -- one representative per (component, label). HC data therefore has
NO train/test homology and NO within-split near-duplicate families (no pseudo-replication in pooled
statistics). BEFORE and AFTER are computed by the identical code in this file, same thread count.
Model everywhere: the paper's LightGBM at the task's validation-selected k (not re-selected).

E1 COMPOSITION (human_enhancers_cohn, nt_enhancers = contaminated; drosophila_enhancers_stark = clean
   control), exactly run_composition_clean.py's procedure (its functions are imported):
     arm original       k-mer MCC; composition-only baseline (LightGBM + LR on the 20-d 4 mono +
                        16 dinucleotide signature); TATA (TBP >= 1.3 bits/pos, both strands) and GC
                        motif-only AUROC, pooled over train+test; 1000-resample bootstrap CIs.
     arm comp_equalized one LR propensity p(pos | 20-d composition) on the pooled data; negatives
                        resampled per split to the positives' 20-bin propensity histogram (positives
                        untouched); same metrics + max|SMD| balance.
   Readouts: comp_only_mcc (CI excluding 0 => a composition shortcut exists); the equalization drop
   dMCC_eq = kmer_mcc(eq) - kmer_mcc(orig) (unpaired; CI from independent bootstraps of the two test
   sets); TATA/GC AUROC moving to 0.5 on the equalized arm.
E2 DINUCLEOTIDE-SHUFFLED NEGATIVES (same three tasks), run_shuffled_neg.py's procedure: positives vs one
   Altschul-Erikson dinucleotide-preserving shuffle per positive (shuffles made per split);
   TATA/GC AUROC and k-mer MCC.
E3 POSITIONAL-BIN RESCUE: B=1 vs B=8 (global spectrum hstacked with the 8-bin spectrum) at the paper k,
   paired bootstrap on the test split, <=20k stratified cap per split exactly as positional_sweep.py.
   Tasks fixed in advance: nt_splice_sites_all (positional) and nt_promoter_tata (fixed-position
   promoter) + two non-positional controls, human_enhancers_cohn and nt_enhancers (the two Route-3
   tasks). Tier 2 (if time): the other 11 tasks, so every task's Route-2 diagnostic exists on HC data.

Decision rules (fixed in advance; applied per task, HC data, T = id80 primary):
  Route-3 magnitude PRESERVED iff (i) TATA motif-only AUROC (original arm) < 0.5 with CI upper < 0.5,
     (ii) comp_only_mcc > 0 with CI lower > 0, (iii) dMCC_eq < 0 with CI upper < 0, and (iv) both the
     equalized-arm and the shuffled-arm TATA AUROC are closer to 0.5 than the original arm's.
  Route-3 control behaves (drosophila): TATA AUROC >= 0.5 on the original arm and its shuffled-arm
     TATA AUROC moves AWAY from 0.5 (motif belongs to the positives).
  Route-2 PRESERVED iff splice dMCC(B8-B1) > 0 with CI lower > 0; SELECTIVITY preserved iff each
     non-positional control's dMCC CI includes 0 or |dMCC| <= 0.05.
Nulls / adverse results are reported as they come out.

Outputs: results/upgrades/homology_recluster_interventions.csv (E1+E2 rows),
         results/upgrades/homology_recluster_positional.csv (E3 rows);
checkpoints per (experiment, task, data) in external/homology_recluster/ckpt/.
Usage: python run_interventions.py [--exp E1,E2,E3] [--tiers 1] [--thresholds id80,id50]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _hc  # noqa: E402  (caps LightGBM threads for this process)
C = _hc.C
for _d in ("composition_clean", "shuffled_neg", "positional_sweep"):
    sys.path.insert(0, os.path.join(_hc.UP, _d))
import run_composition_clean as CC  # noqa: E402
import dinuc_shuffle as D  # noqa: E402
import featurize  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.metrics import matthews_corrcoef  # noqa: E402
from run_eval import boot_diff  # noqa: E402

TATA_BITS = 1.3
PROP_BINS = 20
CAP = 20000
E12_TASKS = [("human_enhancers_cohn", "contaminated"), ("nt_enhancers", "contaminated"),
             ("drosophila_enhancers_stark", "clean control")]
E3_TIER1 = ["nt_splice_sites_all", "nt_promoter_tata", "human_enhancers_cohn", "nt_enhancers"]
E3_TIER2 = [t for t in _hc.TASKS if t not in E3_TIER1]
POS_ROLE = {"nt_splice_sites_all": "positional", "nt_promoter_tata": "fixed-position promoter"}


def get_data(task, dv):
    return _hc.original_data(task) if dv == "original" else _hc.hc_data(task, dv, nr=True)


# ------------------------------------------------------------------ shared metric blocks
def fit_pred(Xtr, ytr, Xte, seed=_hc.SEED):
    m = _hc.build_lgbm(len(np.unique(ytr)), seed)
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)
    return proba, m.classes_[np.argmax(proba, axis=1)]


def auroc_ci(pos, neg, fn, rng, B=_hc.BOOT):
    pt = fn(pos, neg)
    lo, hi = C.boot_ci(np.asarray(pos), np.asarray(neg), fn, B, rng)
    return float(pt), lo, hi


def comp_arm(tag, tr, ytr, te, yte, k, tbp, rng):
    """run_composition_clean.arm_metrics with numeric CIs and kept predictions (for dMCC CIs)."""
    gc_tr, gc_te = C.gc_content(tr), C.gc_content(te)
    ta_tr = C.tata_hit_counts(tr, tbp, TATA_BITS); ta_te = C.tata_hit_counts(te, tbp, TATA_BITS)
    gc_pos = np.concatenate([gc_tr[ytr == 1], gc_te[yte == 1]]); gc_neg = np.concatenate([gc_tr[ytr == 0], gc_te[yte == 0]])
    ta_pos = np.concatenate([ta_tr[ytr == 1], ta_te[yte == 1]]).astype(np.int64)
    ta_neg = np.concatenate([ta_tr[ytr == 0], ta_te[yte == 0]]).astype(np.int64)
    gc_a, gc_lo, gc_hi = auroc_ci(gc_pos, gc_neg, C.auc_float, rng)
    ta_a, ta_lo, ta_hi = auroc_ci(ta_pos, ta_neg, C.auc_counts, rng)
    Ctr = C.comp_signature(tr, ks=(1, 2)); Cte = C.comp_signature(te, ks=(1, 2))
    pc, predc = fit_pred(Ctr, ytr, Cte)
    comp = _hc.boot_metrics(yte, predc, pc)
    plr_m = _hc.build_lr(2); plr_m.fit(Ctr, ytr)
    plr = plr_m.predict_proba(Cte); predlr = plr_m.classes_[np.argmax(plr, 1)]
    Xtr = featurize.kmer_spectrum(tr, k); Xte = featurize.kmer_spectrum(te, k)
    pk, predk = fit_pred(Xtr, ytr, Xte)
    km = _hc.boot_metrics(yte, predk, pk)
    rlo, rhi = _paired_resid_ci(yte, predk, predc)
    return dict(arm=tag, n_pos=int((ytr == 1).sum() + (yte == 1).sum()), n_neg=int((ytr == 0).sum() + (yte == 0).sum()),
                n_train=len(ytr), n_test=len(yte),
                tata_auroc=round(ta_a, 4), tata_lo=round(ta_lo, 4), tata_hi=round(ta_hi, 4),
                gc_auroc=round(gc_a, 4), gc_lo=round(gc_lo, 4), gc_hi=round(gc_hi, 4),
                comp_only_mcc=round(comp["mcc"], 4), comp_only_mcc_lo=round(comp["mcc_lo"], 4),
                comp_only_mcc_hi=round(comp["mcc_hi"], 4), comp_only_auroc=round(comp["auroc"], 4),
                comp_only_lr_mcc=round(float(matthews_corrcoef(yte, predlr)), 4),
                comp_only_lr_auroc=round(_hc.auroc(yte, plr), 4),
                kmer_mcc=round(km["mcc"], 4), kmer_mcc_lo=round(km["mcc_lo"], 4), kmer_mcc_hi=round(km["mcc_hi"], 4),
                kmer_auroc=round(km["auroc"], 4), kmer_auroc_lo=round(km["auroc_lo"], 4),
                kmer_auroc_hi=round(km["auroc_hi"], 4),
                residual_mcc=round(km["mcc"] - comp["mcc"], 4), residual_lo=rlo, residual_hi=rhi), (yte, predk)


def _paired_resid_ci(y, pk, pc, seed=_hc.SEED, B=_hc.BOOT):
    rng = np.random.RandomState(seed); d = np.full(B, np.nan)
    for b in range(B):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d[b] = matthews_corrcoef(y[i], pk[i]) - matthews_corrcoef(y[i], pc[i])
    return round(float(np.nanpercentile(d, 2.5)), 4), round(float(np.nanpercentile(d, 97.5)), 4)


# ------------------------------------------------------------------ E1 / E2
def run_e1(task, role, dv, tbp):
    f = os.path.join(_hc.CKPT_DIR, f"E1_{task}_{dv}.csv")
    if os.path.exists(f):
        return pd.read_csv(f)
    t0 = time.time()
    d = get_data(task, dv); k = _hc.best_k(task)
    tr, ytr, te, yte = d["tr"], d["ytr"], d["te"], d["yte"]
    rng = np.random.RandomState(_hc.SEED)
    r0, (y0, p0) = comp_arm("original", tr, ytr, te, yte, k, tbp, rng)
    # composition-equalized arm (run_composition_clean procedure)
    ctr = C.comp_signature(tr, ks=(1, 2)); cte = C.comp_signature(te, ks=(1, 2))
    prop = CC.fit_propensity(np.vstack([ctr, cte]), np.concatenate([ytr, yte]), _hc.SEED)
    p_tr = prop.predict_proba(ctr)[:, 1]; p_te = prop.predict_proba(cte)[:, 1]
    keep_tr = CC.match_proportional(p_tr[ytr == 1], p_tr[ytr == 0], PROP_BINS, rng)
    keep_te = CC.match_proportional(p_te[yte == 1], p_te[yte == 0], PROP_BINS, rng)
    ce_tr, ce_ytr = CC.negatives_kept_split(tr, ytr, keep_tr)
    ce_te, ce_yte = CC.negatives_kept_split(te, yte, keep_te)
    retained = (len(keep_tr) + len(keep_te)) / max(int((ytr == 0).sum() + (yte == 0).sum()), 1)
    pos_all = np.vstack([ctr[ytr == 1], cte[yte == 1]])
    smd_b = np.nanmax(np.abs(CC.smd(pos_all, np.vstack([ctr[ytr == 0], cte[yte == 0]]))))
    smd_a = np.nanmax(np.abs(CC.smd(pos_all, np.vstack([ctr[ytr == 0][keep_tr], cte[yte == 0][keep_te]]))))
    rows = []
    r0.update(retained_neg_frac=1.0, max_abs_smd=round(float(smd_b), 4))
    rows.append(r0)
    if len(keep_te) >= 10 and len(keep_tr) >= 10:
        r1, (y1, p1) = comp_arm("comp_equalized", ce_tr, ce_ytr, ce_te, ce_yte, k, tbp, rng)
        lo, hi = boot_diff(y1, p1, y0, p0)
        r1.update(retained_neg_frac=round(float(retained), 4), max_abs_smd=round(float(smd_a), 4),
                  d_kmer_mcc_vs_original=round(r1["kmer_mcc"] - r0["kmer_mcc"], 4),
                  d_kmer_mcc_lo=round(lo, 4), d_kmer_mcc_hi=round(hi, 4))
        rows.append(r1)
    else:
        # POST-HOC (added after the first run, flagged in the output): the pre-registered MCC arm needs >=10
        # kept negatives per split; the paper's causal readout, the pooled TATA/GC AUROC collapse, does not.
        # So when the MCC arm is infeasible we still report the pooled AUROCs of positives vs the kept negatives.
        pos_s = [s for s, v in zip(tr, ytr) if v == 1] + [s for s, v in zip(te, yte) if v == 1]
        neg_tr_s = [s for s, v in zip(tr, ytr) if v == 0]; neg_te_s = [s for s, v in zip(te, yte) if v == 0]
        kn = [neg_tr_s[i] for i in keep_tr] + [neg_te_s[i] for i in keep_te]
        ta, tlo, thi = auroc_ci(C.tata_hit_counts(pos_s, tbp, TATA_BITS).astype(np.int64),
                                C.tata_hit_counts(kn, tbp, TATA_BITS).astype(np.int64), C.auc_counts, rng)
        ga, glo, ghi = auroc_ci(C.gc_content(pos_s), C.gc_content(kn), C.auc_float, rng)
        rows.append(dict(arm="comp_equalized", retained_neg_frac=round(float(retained), 4),
                         max_abs_smd=round(float(smd_a), 4), n_pos=len(pos_s), n_neg=len(kn),
                         n_test_neg_kept=int(len(keep_te)),
                         tata_auroc=round(ta, 4), tata_lo=round(tlo, 4), tata_hi=round(thi, 4),
                         gc_auroc=round(ga, 4), gc_lo=round(glo, 4), gc_hi=round(ghi, 4),
                         status="MCC infeasible (<10 negatives kept in a split); pooled AUROCs post hoc"))
    df = pd.DataFrame(rows)
    df.insert(0, "experiment", "E1_composition"); df.insert(1, "task", task); df.insert(2, "role", role)
    df.insert(3, "data", dv); df.insert(4, "k", k)
    df["n_pooled"] = d["n_pooled"]; df["n_nr"] = d["n_nr"]
    df.to_csv(f, index=False)
    print(f"  E1 {task} [{dv}] {time.time()-t0:.0f}s", flush=True)
    return df


def run_e2(task, role, dv, tbp):
    f = os.path.join(_hc.CKPT_DIR, f"E2_{task}_{dv}.csv")
    if os.path.exists(f):
        return pd.read_csv(f)
    t0 = time.time()
    d = get_data(task, dv); k = _hc.best_k(task)
    tr, ytr, te, yte = d["tr"], d["ytr"], d["te"], d["yte"]
    rng = np.random.RandomState(_hc.SEED)
    pos_tr = [s for s, v in zip(tr, ytr) if v == 1]; pos_te = [s for s, v in zip(te, yte) if v == 1]
    neg_tr = [s for s, v in zip(tr, ytr) if v == 0]; neg_te = [s for s, v in zip(te, yte) if v == 0]
    sh_tr = [D.dinuc_shuffle(s, rng) for s in pos_tr]; sh_te = [D.dinuc_shuffle(s, rng) for s in pos_te]
    pres = float(np.mean([D.verify_dinuc_preserved(a, b) for a, b in zip(pos_tr + pos_te, sh_tr + sh_te)]))
    rows = []
    for arm, ntr, nte in [("original_neg", neg_tr, neg_te), ("shuffled_neg", sh_tr, sh_te)]:
        gp = np.concatenate([C.gc_content(pos_tr), C.gc_content(pos_te)])
        gn = np.concatenate([C.gc_content(ntr), C.gc_content(nte)])
        tp = np.concatenate([C.tata_hit_counts(pos_tr, tbp, TATA_BITS), C.tata_hit_counts(pos_te, tbp, TATA_BITS)]).astype(np.int64)
        tn = np.concatenate([C.tata_hit_counts(ntr, tbp, TATA_BITS), C.tata_hit_counts(nte, tbp, TATA_BITS)]).astype(np.int64)
        ga, glo, ghi = auroc_ci(gp, gn, C.auc_float, rng)
        ta, tlo, thi = auroc_ci(tp, tn, C.auc_counts, rng)
        Xtr = featurize.kmer_spectrum(pos_tr + list(ntr), k); Xte = featurize.kmer_spectrum(pos_te + list(nte), k)
        ytr2 = np.r_[np.ones(len(pos_tr), int), np.zeros(len(ntr), int)]
        yte2 = np.r_[np.ones(len(pos_te), int), np.zeros(len(nte), int)]
        pk, predk = fit_pred(Xtr, ytr2, Xte)
        km = _hc.boot_metrics(yte2, predk, pk)
        rows.append(dict(arm=arm, n_pos=len(pos_tr) + len(pos_te), n_neg=len(ntr) + len(nte),
                         dinuc_preserved=round(pres, 4), n_test=len(yte2),
                         tata_auroc=round(ta, 4), tata_lo=round(tlo, 4), tata_hi=round(thi, 4),
                         gc_auroc=round(ga, 4), gc_lo=round(glo, 4), gc_hi=round(ghi, 4),
                         kmer_mcc=round(km["mcc"], 4), kmer_mcc_lo=round(km["mcc_lo"], 4),
                         kmer_mcc_hi=round(km["mcc_hi"], 4), kmer_auroc=round(km["auroc"], 4)))
    df = pd.DataFrame(rows)
    df.insert(0, "experiment", "E2_shuffled_neg"); df.insert(1, "task", task); df.insert(2, "role", role)
    df.insert(3, "data", dv); df.insert(4, "k", k)
    df["n_pooled"] = d["n_pooled"]; df["n_nr"] = d["n_nr"]
    df.to_csv(f, index=False)
    print(f"  E2 {task} [{dv}] {time.time()-t0:.0f}s", flush=True)
    return df


# ------------------------------------------------------------------ E3
def cap_split(seqs, y, cap, seed):
    """positional_sweep.cap_split (copied verbatim in behaviour): stratified deterministic cap."""
    if cap is None or len(seqs) <= cap:
        return list(seqs), y, False
    rng = np.random.RandomState(seed)
    chunks = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        take = min(len(ci), max(1, int(round(cap * len(ci) / len(y)))))
        chunks.append(rng.choice(ci, size=take, replace=False))
    idx = np.sort(np.concatenate(chunks))
    return [seqs[i] for i in idx], y[idx], True


def run_e3(task, dv, B=8, seed=_hc.SEED):
    f = os.path.join(_hc.CKPT_DIR, f"E3_{task}_{dv}.csv")
    if os.path.exists(f):
        return pd.read_csv(f)
    t0 = time.time()
    d = get_data(task, dv); k = _hc.best_k(task)
    tr, ytr, capped1 = cap_split(d["tr"], d["ytr"], CAP, seed)
    te, yte, capped2 = cap_split(d["te"], d["yte"], CAP, seed + 1)
    Xtr_g = featurize.kmer_spectrum(tr, k); Xte_g = featurize.kmer_spectrum(te, k)
    _, pred_g = fit_pred(Xtr_g, ytr, Xte_g, seed)
    Xtr_p = sparse.hstack([Xtr_g, featurize.binned_kmer_spectrum(tr, k, B)], format="csr")
    Xte_p = sparse.hstack([Xte_g, featurize.binned_kmer_spectrum(te, k, B)], format="csr")
    del Xtr_g, Xte_g
    _, pred_p = fit_pred(Xtr_p, ytr, Xte_p, seed)
    mg = matthews_corrcoef(yte, pred_g); mp = matthews_corrcoef(yte, pred_p)
    rng = np.random.RandomState(seed); n = len(yte); dd = np.empty(_hc.BOOT)
    for b in range(_hc.BOOT):
        i = rng.randint(0, n, n)
        dd[b] = matthews_corrcoef(yte[i], pred_p[i]) - matthews_corrcoef(yte[i], pred_g[i])
    lo, hi = np.percentile(dd, [2.5, 97.5])
    df = pd.DataFrame([dict(experiment="E3_positional", task=task, data=dv, k=k, n_bins=B,
                            role=POS_ROLE.get(task, "non-positional (control)"),
                            n_train=len(ytr), n_test=n, capped=bool(capped1 or capped2),
                            spectrum_mcc=round(mg, 4), positional_mcc=round(mp, 4),
                            delta_mcc=round(mp - mg, 4), delta_lo=round(float(lo), 4), delta_hi=round(float(hi), 4),
                            delta_excludes_0=bool(not (lo <= 0 <= hi)))])
    df.to_csv(f, index=False)
    print(f"  E3 {task} [{dv}] MCC {mg:.3f}->{mp:.3f} d={mp-mg:+.3f} [{lo:+.3f},{hi:+.3f}] "
          f"{time.time()-t0:.0f}s", flush=True)
    return df


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="E1,E2,E3")
    ap.add_argument("--tiers", default="1", help="E3 tiers to run: 1 or 1,2")
    ap.add_argument("--thresholds", default="id80,id50")
    a = ap.parse_args()
    exps = a.exp.split(","); ths = [t for t in a.thresholds.split(",") if t]
    dvs = ["original"] + ths
    tbp = C.load_tbp()
    os.makedirs(_hc.CKPT_DIR, exist_ok=True)

    def have(t, th):
        return os.path.exists(os.path.join(_hc.CLU_DIR, f"{t}_{th}.npz"))

    for task, role in E12_TASKS:
        for dv in dvs:
            if dv != "original" and not have(task, dv):
                continue
            if "E1" in exps:
                run_e1(task, role, dv, tbp)
            if "E2" in exps:
                run_e2(task, role, dv, tbp)
    if "E3" in exps:
        e3 = E3_TIER1 + (E3_TIER2 if "2" in a.tiers.split(",") else [])
        for task in e3:
            # tier-1 tasks: BEFORE recomputed here + id80 + id50; tier-2 tasks: id80 only (their BEFORE is the
            # published positional_sweep.csv row -- dMCC is paired within a split, so this is like-for-like)
            dvs3 = (["original", "id80"] + (["id50"] if "id50" in ths else [])) if task in E3_TIER1 else ["id80"]
            for dv in dvs3:
                if dv != "original" and not have(task, dv):
                    continue
                run_e3(task, dv)
    # aggregate
    import glob
    e12 = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(_hc.CKPT_DIR, "E[12]_*.csv")))]
    if e12:
        pd.concat(e12, ignore_index=True).to_csv(os.path.join(_hc.RES, f"{_hc.PREFIX}_interventions.csv"), index=False)
    e3r = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(_hc.CKPT_DIR, "E3_*.csv")))]
    if e3r:
        pd.concat(e3r, ignore_index=True).to_csv(os.path.join(_hc.RES, f"{_hc.PREFIX}_positional.csv"), index=False)
    print("aggregated.")


if __name__ == "__main__":
    main()
