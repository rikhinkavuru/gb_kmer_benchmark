#!/usr/bin/env python3
"""Step 3b -- split-seed robustness of the re-split comparison (cluster-aware vs random).

PRE-REGISTERED (written 2026-09-24 before this file was run). run_eval.py uses ONE split seed (42); its
bootstrap CIs resample test sequences but not the split assignment itself, which matters for small test
sets (nt_enhancers: 400). Here the cluster-aware (id80) and random re-splits are repeated with split
seeds 43 and 44 (model seed fixed at 42, paper k, same code as run_eval.py) for the ten tasks with
<= 60k pooled sequences (originally planned; see EXTENDED below).
EXTENDED before this file was first run: demo_coding_vs_intergenomic_seqs showed id80 re-split MCC 0.677 but
id50 re-split MCC 0.806 at split seed 42. Diagnosis (from cached predictions): three giant single-label
(all-coding) components of ~3,000 sequences each; at id80 two of them (23 % of the test set) landed in test
(accuracy 0.65 on them), at id50 a different one did (accuracy 0.93). Single-seed re-split MCCs can therefore
be dominated by where a few giant families fall, so seeds 43/44 are run for ALL 15 tasks (smallest first,
checkpointed), and the id50 re-split is repeated too (--thresholds id80,id50).
Readout per task: mean and range over seeds {42,43,44} of MCC_cluster, MCC_random and
delta = MCC_cluster - MCC_random. Decision: the seed-42 homology-inflation call of run_eval.py is
called SEED-ROBUST if delta has the same sign for all three seeds.
Output: results/upgrades/homology_recluster_seeds.csv. Predictions cached as
external/homology_recluster/preds/<task>__resplit_{id80,random}_s<seed>.npz (seed-42 files untouched).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _hc  # noqa: E402
from run_eval import fit_unit, pred_of  # noqa: E402
from run_cluster import SIZE_ORDER  # noqa: E402

SMALL = list(SIZE_ORDER)                 # all 15 tasks, smallest first (see EXTENDED note)


def main():
    import featurize
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=",".join(SMALL))
    ap.add_argument("--seeds", default="43,44")
    ap.add_argument("--thresholds", default="id80,id50")
    a = ap.parse_args()
    ev = pd.read_csv(os.path.join(_hc.RES, f"{_hc.PREFIX}_eval.csv"))
    ths = [t for t in a.thresholds.split(",") if t]
    rows = []
    for task in [t for t in a.tasks.split(",") if t]:
        k = _hc.best_k(task)
        tr, ytr, te, yte = _hc.load_task(task)
        y = np.concatenate([ytr, yte]); n = len(y); test_frac = len(te) / n
        e = ev[ev.task == task]
        r42r = e[e.protocol == "resplit_random"].iloc[0]
        comps = {th: _hc.load_clusters(task, th)[0] for th in ths}
        for th in ths:
            r42c = e[(e.protocol == "resplit_cluster") & (e.threshold == th)].iloc[0]
            rows.append(dict(task=task, threshold=th, split_seed=42, mcc_cluster=r42c.mcc, mcc_random=r42r.mcc,
                             delta=r42c.mcc - r42r.mcc))
        X = None
        for s in [int(x) for x in a.seeds.split(",")]:
            rn = _hc.random_resplit(y, test_frac, s)
            if X is None:
                X = featurize.kmer_spectrum(tr + te, k)
            pr, cls = fit_unit(task, f"resplit_random_s{s}", X, y, np.where(~rn)[0], np.where(rn)[0], k, _hc.SEED)
            m_r = float(matthews_corrcoef(y[rn], pred_of(pr, cls)))
            for th in ths:
                cl = _hc.cluster_resplit(comps[th], y, test_frac, s)
                pr, cls = fit_unit(task, f"resplit_{th}_s{s}", X, y, np.where(~cl)[0], np.where(cl)[0], k, _hc.SEED)
                m_c = float(matthews_corrcoef(y[cl], pred_of(pr, cls)))
                rows.append(dict(task=task, threshold=th, split_seed=s, mcc_cluster=m_c, mcc_random=m_r,
                                 delta=m_c - m_r))
                print(f"[{task}] {th} seed {s}: cluster {m_c:.4f} random {m_r:.4f} delta {m_c-m_r:+.4f}", flush=True)
        del X
        pd.DataFrame(rows).to_csv(os.path.join(_hc.RES, f"{_hc.PREFIX}_seeds.csv"), index=False)  # checkpoint
    df = pd.DataFrame(rows)
    summ = df.groupby(["task", "threshold"], sort=False).agg(
        mcc_cluster_mean=("mcc_cluster", "mean"), mcc_cluster_min=("mcc_cluster", "min"),
        mcc_cluster_max=("mcc_cluster", "max"), mcc_random_mean=("mcc_random", "mean"),
        delta_mean=("delta", "mean"), delta_min=("delta", "min"), delta_max=("delta", "max"),
        n_seeds=("delta", "size")).reset_index()
    summ["same_sign_all_seeds"] = (np.sign(summ.delta_min) == np.sign(summ.delta_max))
    df.to_csv(os.path.join(_hc.RES, f"{_hc.PREFIX}_seeds.csv"), index=False)
    summ.to_csv(os.path.join(_hc.RES, f"{_hc.PREFIX}_seeds_summary.csv"), index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
