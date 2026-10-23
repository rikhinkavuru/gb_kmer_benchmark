#!/usr/bin/env python3
"""Model-FREE Route-3 gate for every binary task (does not involve any classifier).

Gate (diagnose.py rule): TATA-box negative excess (fraction of negatives with >=1 TBP hit at
1.3 bits/pos, both strands, minus the positives' fraction) >= 0.10 AND TATA motif-only AUROC < 0.5.
Procedure replicates run_stats.py PART B exactly: pooled train+test, at most 4000 sequences per class
drawn with np.random.RandomState(42).choice (positives first, then negatives), vertebrate TBP PSSM.
The paper's released values (results/contamination_analysis.csv, results/nt/contamination_analysis.csv)
are used where they exist; this script recomputes them for all 12 binary tasks and reports both, so the
four tasks the paper never scanned (human_nontata_promoters, the two demo tasks,
dummy_mouse_enhancers_ensembl) get a gate value from the identical procedure.
Writes results/upgrades/alt_model_gate.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
import _common as C                     # noqa: E402
from cache_seqs import load_cached      # noqa: E402

BINARY = ["demo_human_or_worm", "human_nontata_promoters", "demo_coding_vs_intergenomic_seqs",
          "human_enhancers_ensembl", "dummy_mouse_enhancers_ensembl", "human_enhancers_cohn",
          "human_ocr_ensembl", "drosophila_enhancers_stark", "nt_promoter_all", "nt_promoter_tata",
          "nt_promoter_no_tata", "nt_enhancers"]
TATA_BITS = 1.3
CAP = 4000


def released(task):
    for p in (os.path.join(C.ROOT, "results", "contamination_analysis.csv"),
              os.path.join(C.ROOT, "results", "nt", "contamination_analysis.csv")):
        df = pd.read_csv(p)
        a = df[(df.dataset == task) & (df.measure == "TATA_box_fraction(>=1)")]
        b = df[(df.dataset == task) & (df.measure == "TATA_TBP_hits")]
        if len(a) and len(b):
            return float(-a["diff_pos_minus_neg"].iloc[0]), float(b["auroc_pos_gt_neg"].iloc[0])
    return None, None


def main():
    tbp = C.load_tbp()
    rows = []
    for t in BINARY:
        tr, ytr, te, yte = load_cached(t)
        seqs = list(tr) + list(te); y = np.concatenate([np.asarray(ytr), np.asarray(yte)])
        rng = np.random.RandomState(C.SEED)
        pi = np.where(y == 1)[0]; ni = np.where(y == 0)[0]
        if len(pi) > CAP:
            pi = rng.choice(pi, CAP, replace=False)
        if len(ni) > CAP:
            ni = rng.choice(ni, CAP, replace=False)
        pc = C.tata_hit_counts([seqs[i] for i in pi], tbp, TATA_BITS)
        nc = C.tata_hit_counts([seqs[i] for i in ni], tbp, TATA_BITS)
        excess = float((nc >= 1).mean() - (pc >= 1).mean())
        auc = float(C.auc_counts(pc.astype(np.int64), nc.astype(np.int64)))
        rel_ex, rel_auc = released(t)
        use_ex = rel_ex if rel_ex is not None else excess
        use_auc = rel_auc if rel_auc is not None else auc
        rows.append(dict(task=t, tata_neg_excess_recomputed=round(excess, 4), tata_auroc_recomputed=round(auc, 4),
                         tata_neg_excess_released=("" if rel_ex is None else round(rel_ex, 4)),
                         tata_auroc_released=("" if rel_auc is None else round(rel_auc, 4)),
                         tata_neg_excess=round(use_ex, 4), tata_auroc=round(use_auc, 4),
                         source=("released" if rel_ex is not None else "recomputed"),
                         gate=bool(use_ex >= 0.10 and use_auc < 0.5), n_pos=len(pi), n_neg=len(ni)))
        print(rows[-1], flush=True)
    out = os.path.join(C.RESULTS_DIR, "alt_model_gate.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
