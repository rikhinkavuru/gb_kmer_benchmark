#!/usr/bin/env python3
"""Step 10 -- RECONCILE the composition-equalized arm (E1) with the published composition_clean.csv, and
measure how much its readouts depend on WHICH negatives the propensity match happens to keep.
POST HOC (written 2026-09-24 ~11:40 after an independent verifier noted that run_interventions.py's
original-data equalized arm for human_enhancers_cohn gives k-mer MCC 0.000 / comp-only 0.064 while
composition_clean.csv (the paper's source) gives k-mer 0.064 / comp-only 0.000).

Hypothesis fixed before running: this is NOT a code difference but a random-stream difference. Both scripts
keep the same NUMBER of negatives (4.5 %), but run_composition_clean.py draws the kept negatives from a
RandomState(42) stream that its own bootstraps (comp LightGBM, comp LR, k-mer) have already advanced, whereas
run_interventions.py's bootstraps use their own streams, so the two draws keep different negatives; with
~167 test negatives among ~3,700 test sequences the equalized k-mer model is at (or next to) the
predict-all-positive corner, where MCC is ~0 and unstable, whereas AUROC is not.

R1 REPLAY. run_composition_clean.arm_metrics / fit_propensity / match_proportional are imported and called
   in exactly run_composition_clean.main()'s order with one RandomState(42) stream (3 LightGBM threads;
   the module's thread cap). Pass iff the replayed original and comp_equalized rows equal
   composition_clean.csv (MCC/AUROC to 1e-4) -- then the published 0.064 is reproduced by the published code
   and the module's 0.000 is a different draw of the same procedure.
R2 DRAW SENSITIVITY. For data in {original, id80, id50} (id80/id50 = run_interventions.py's homology-clean
   data: cluster-aware re-split + one representative per component x label), 20 matching draws
   (RandomState(1000 + d), d = 0..19) of the same propensity-matched equalization; per draw: kept negatives
   (train/test), k-mer LightGBM (paper k) MCC and AUROC, fraction of test sequences predicted positive, and
   pooled TATA / GC motif-only AUROC (positives vs kept negatives). Summary: median and 2.5-97.5 % range over
   draws. Readout: an arm is called MCC-DEGENERATE if the median draw predicts >= 95 % of test sequences
   positive; AUROC readouts are called draw-stable if their draw range is within +/- 0.03 of the median.

Outputs: results/upgrades/homology_recluster_e1_reconcile.csv (R1), homology_recluster_e1_draws.csv (R2 per
draw) and homology_recluster_e1_draws_summary.csv. Checkpoints under external/homology_recluster/ckpt/e1rec/.
Usage: python run_e1_reconcile.py [--part R1,R2] [--draws 20]
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _hc  # noqa: E402  (caps LightGBM threads for this process)
C = _hc.C
import run_interventions as RI  # noqa: E402
CC = RI.CC
import featurize  # noqa: E402

TASK = "human_enhancers_cohn"
CK = os.path.join(_hc.CKPT_DIR, "e1rec")


def r1_replay():
    f = os.path.join(CK, "R1.csv")
    if os.path.exists(f):
        return pd.read_csv(f)
    t0 = time.time()
    args = SimpleNamespace(tata_bits=1.3, boot=_hc.BOOT, seed=_hc.SEED, n_bins=20)
    tbp = C.load_tbp()
    k = _hc.best_k(TASK)
    tr, ytr, te, yte = _hc.load_task(TASK)
    rng = np.random.RandomState(args.seed)
    assert k == int(pd.read_csv(os.path.join(_hc.RES, "composition_clean.csv")).set_index("dataset").loc[TASK, "best_k"].iloc[0])
    rows = [CC.arm_metrics("original", tr, ytr, te, yte, k, tbp, args, rng)]
    comp_tr = C.comp_signature(tr, ks=(1, 2)); comp_te = C.comp_signature(te, ks=(1, 2))
    prop = CC.fit_propensity(np.vstack([comp_tr, comp_te]), np.concatenate([ytr, yte]), args.seed)
    p_tr = prop.predict_proba(comp_tr)[:, 1]; p_te = prop.predict_proba(comp_te)[:, 1]
    keep_tr = CC.match_proportional(p_tr[ytr == 1], p_tr[ytr == 0], args.n_bins, rng)
    keep_te = CC.match_proportional(p_te[yte == 1], p_te[yte == 0], args.n_bins, rng)
    ce_tr, ce_ytr = CC.negatives_kept_split(tr, ytr, keep_tr)
    ce_te, ce_yte = CC.negatives_kept_split(te, yte, keep_te)
    r = CC.arm_metrics("comp_equalized", ce_tr, ce_ytr, ce_te, ce_yte, k, tbp, args, rng)
    r["retained_neg_frac"] = round((len(keep_tr) + len(keep_te)) / int((ytr == 0).sum() + (yte == 0).sum()), 4)
    r["n_test_neg_kept"] = int(len(keep_te))
    rows.append(r)
    pub = pd.read_csv(os.path.join(_hc.RES, "composition_clean.csv"))
    pub = pub[pub.dataset == TASK].set_index("arm")
    iv = pd.read_csv(os.path.join(_hc.RES, f"{_hc.PREFIX}_interventions.csv"))
    iv = iv[(iv.experiment == "E1_composition") & (iv.task == TASK) & (iv.data == "original")].set_index("arm")
    out = []
    for r in rows:
        a = r["arm"]
        d = dict(task=TASK, arm=a, replay_kmer_mcc=r["kmer_mcc"], published_kmer_mcc=pub.loc[a, "kmer_mcc"],
                 module_kmer_mcc=iv.loc[a, "kmer_mcc"], replay_comp_only_mcc=r["comp_only_mcc"],
                 published_comp_only_mcc=pub.loc[a, "comp_only_mcc"], module_comp_only_mcc=iv.loc[a, "comp_only_mcc"],
                 replay_kmer_auroc=r["kmer_auroc"], published_kmer_auroc=pub.loc[a, "kmer_auroc"],
                 module_kmer_auroc=iv.loc[a, "kmer_auroc"], replay_tata_auroc=r["tata_auroc"],
                 published_tata_auroc=pub.loc[a, "tata_auroc"], module_tata_auroc=iv.loc[a, "tata_auroc"],
                 replay_gc_auroc=r["gc_auroc"], published_gc_auroc=pub.loc[a, "gc_auroc"],
                 module_gc_auroc=iv.loc[a, "gc_auroc"], replay_retained_neg_frac=r.get("retained_neg_frac", 1.0),
                 published_retained_neg_frac=pub.loc[a, "retained_neg_frac"])
        d["replay_equals_published"] = bool(
            all(abs(d[f"replay_{m}"] - d[f"published_{m}"]) <= 1e-4
                for m in ("kmer_mcc", "comp_only_mcc", "kmer_auroc", "tata_auroc", "gc_auroc")))
        out.append(d)
    df = pd.DataFrame(out)
    os.makedirs(CK, exist_ok=True)
    df.to_csv(f, index=False)
    print(df.T.to_string(), f"\nR1 done in {time.time()-t0:.0f}s", flush=True)
    return df


def r2_draws(dv, n_draws, tbp):
    f = os.path.join(CK, f"R2_{dv}.csv")
    if os.path.exists(f):
        return pd.read_csv(f)
    t0 = time.time()
    d = RI.get_data(TASK, dv); k = _hc.best_k(TASK)
    tr, ytr, te, yte = d["tr"], d["ytr"], d["te"], d["yte"]
    ctr = C.comp_signature(tr, ks=(1, 2)); cte = C.comp_signature(te, ks=(1, 2))
    prop = CC.fit_propensity(np.vstack([ctr, cte]), np.concatenate([ytr, yte]), _hc.SEED)
    p_tr = prop.predict_proba(ctr)[:, 1]; p_te = prop.predict_proba(cte)[:, 1]
    Xtr = featurize.kmer_spectrum(tr, k); Xte = featurize.kmer_spectrum(te, k)
    ta_tr = C.tata_hit_counts(tr, tbp, 1.3).astype(np.int64); ta_te = C.tata_hit_counts(te, tbp, 1.3).astype(np.int64)
    gc_tr = C.gc_content(tr); gc_te = C.gc_content(te)
    pos_tr, neg_tr = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    pos_te, neg_te = np.where(yte == 1)[0], np.where(yte == 0)[0]
    ta_pos = np.r_[ta_tr[pos_tr], ta_te[pos_te]]; gc_pos = np.r_[gc_tr[pos_tr], gc_te[pos_te]]
    rows = []
    for dr in range(n_draws):
        rng = np.random.RandomState(1000 + dr)
        ktr = CC.match_proportional(p_tr[ytr == 1], p_tr[ytr == 0], 20, rng)
        kte = CC.match_proportional(p_te[yte == 1], p_te[yte == 0], 20, rng)
        itr = np.sort(np.r_[pos_tr, neg_tr[ktr]]); ite = np.sort(np.r_[pos_te, neg_te[kte]])
        row = dict(task=TASK, data=dv, draw=dr, n_train=len(itr), n_test=len(ite),
                   n_train_neg_kept=len(ktr), n_test_neg_kept=len(kte),
                   retained_neg_frac=round((len(ktr) + len(kte)) / (len(neg_tr) + len(neg_te)), 4),
                   tata_auroc=C.auc_counts(ta_pos, np.r_[ta_tr[neg_tr[ktr]], ta_te[neg_te[kte]]]),
                   gc_auroc=C.auc_float(gc_pos, np.r_[gc_tr[neg_tr[ktr]], gc_te[neg_te[kte]]]))
        if len(ktr) >= 10 and len(kte) >= 10:
            m = _hc.build_lgbm(2, _hc.SEED); m.fit(Xtr[itr], ytr[itr])
            pr = m.predict_proba(Xte[ite]); pred = m.classes_[np.argmax(pr, 1)]
            row.update(kmer_mcc=float(matthews_corrcoef(yte[ite], pred)), kmer_auroc=_hc.auroc(yte[ite], pr),
                       frac_pred_pos=float(np.mean(pred == 1)))
        rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(CK, exist_ok=True)
    df.to_csv(f, index=False)
    print(f"R2 {dv}: {n_draws} draws in {time.time()-t0:.0f}s", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="R1,R2")
    ap.add_argument("--draws", type=int, default=20)
    a = ap.parse_args()
    parts = a.part.split(",")
    if "R1" in parts:
        r1_replay().to_csv(os.path.join(_hc.RES, f"{_hc.PREFIX}_e1_reconcile.csv"), index=False)
    if "R2" in parts:
        tbp = C.load_tbp()
        dr = pd.concat([r2_draws(dv, a.draws, tbp) for dv in ("original", "id80", "id50")], ignore_index=True)
        dr.to_csv(os.path.join(_hc.RES, f"{_hc.PREFIX}_e1_draws.csv"), index=False)
        rows = []
        for dv, g in dr.groupby("data", sort=False):
            r = dict(task=TASK, data=dv, n_draws=len(g))
            for m in ("retained_neg_frac", "n_test_neg_kept", "kmer_mcc", "kmer_auroc", "frac_pred_pos",
                      "tata_auroc", "gc_auroc"):
                v = g[m].dropna()
                r.update({f"{m}_median": float(v.median()) if len(v) else np.nan,
                          f"{m}_p2.5": float(v.quantile(0.025)) if len(v) else np.nan,
                          f"{m}_p97.5": float(v.quantile(0.975)) if len(v) else np.nan,
                          f"{m}_min": float(v.min()) if len(v) else np.nan,
                          f"{m}_max": float(v.max()) if len(v) else np.nan})
            r["mcc_degenerate"] = bool(r["frac_pred_pos_median"] >= 0.95) if r["frac_pred_pos_median"] == r["frac_pred_pos_median"] else np.nan
            for m in ("kmer_auroc", "tata_auroc", "gc_auroc"):
                r[f"{m}_draw_stable"] = bool(max(r[f"{m}_median"] - r[f"{m}_min"], r[f"{m}_max"] - r[f"{m}_median"]) <= 0.03)
            rows.append(r)
        s = pd.DataFrame(rows)
        s.to_csv(os.path.join(_hc.RES, f"{_hc.PREFIX}_e1_draws_summary.csv"), index=False)
        print(s.T.to_string())


if __name__ == "__main__":
    main()
