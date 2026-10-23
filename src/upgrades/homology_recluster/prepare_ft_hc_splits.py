#!/usr/bin/env python3
"""DEFERRED-JOB INPUTS: homology-clean splits for the end-to-end fine-tuning experiment (fm_finetune/run_ft_v2.py).

Torch-free. Writes external/homology_recluster/ft_splits/<dataset>__{train,test}.csv.gz in EXACTLY the format of
fm_finetune/prepare_splits_v2.py (columns idx, label, sequence, <arm>_pos with <arm>_pos = position in the arm's
list, -1 = not a member), so run_ft_v2.py / eval_ft_v2.py run unchanged on them (see run_ft_hc.py / eval_ft_hc.py).

Homology-clean data = the id80 cluster-aware re-split (seed 42) + one representative per (component, label)
(run_extra.split_idx, identical to _hc.hc_data). Arms (positives identical in every arm):
  original        all HC sequences of the split
  gc_match        run_extra.arm_masks gc_match arm (25-bin GC histogram, per split, RandomState(42))
  comp_equalized  run_extra.arm_masks comp_equalized arm (20-d propensity match, per split, RandomState(42))
  random          prepare_splits_v2.random_negative_draw: all positives + exactly the comp_equalized negative
                  count, uniformly at random, RandomState(42), train draw first then test (its documented order)
Also writes results/upgrades/homology_recluster_ft_split_manifest.csv (sizes per arm/split).
Usage: python prepare_ft_hc_splits.py [--datasets a,b] [--threshold id80]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _hc  # noqa: E402
import run_extra as RX  # noqa: E402

OUT_DIR = os.path.join(_hc.EXT, "ft_splits")
DATASETS = ("human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark")
ARMS = ("original", "gc_match", "comp_equalized", "random")


def random_negative_draw(y, n_neg, rng):
    """fm_finetune/prepare_splits_v2.random_negative_draw (copied verbatim)."""
    neg = np.where(y == 0)[0]
    pick = np.sort(rng.choice(neg, size=n_neg, replace=False))
    return np.sort(np.concatenate([np.where(y == 1)[0], pick]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--threshold", default="id80")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    tbp = _hc.C.load_tbp()
    rows = []
    for ds in a.datasets.split(","):
        seqs, y, itr, ite = RX.split_idx(ds, a.threshold, nr=True)
        tr = [seqs[i] for i in itr]; te = [seqs[i] for i in ite]; ytr, yte = y[itr], y[ite]
        masks, _ = RX.arm_masks(tr, ytr, te, yte, tbp)
        members = {"train": {"original": np.arange(len(ytr)), "gc_match": np.where(masks["gc_match"][0])[0],
                             "comp_equalized": np.where(masks["comp_equalized"][0])[0]},
                   "test": {"original": np.arange(len(yte)), "gc_match": np.where(masks["gc_match"][1])[0],
                            "comp_equalized": np.where(masks["comp_equalized"][1])[0]}}
        rng = np.random.RandomState(42)
        for sp, yy in (("train", ytr), ("test", yte)):
            n_neg = int((yy[members[sp]["comp_equalized"]] == 0).sum())
            members[sp]["random"] = random_negative_draw(yy, n_neg, rng)
        for sp, S, yy, pooled in (("train", tr, ytr, itr), ("test", te, yte, ite)):
            df = pd.DataFrame(dict(idx=np.arange(len(yy)), label=yy, sequence=S))
            for arm in ARMS:
                pos = np.full(len(yy), -1, dtype=np.int64)
                pos[members[sp][arm]] = np.arange(len(members[sp][arm]))
                df[f"{arm}_pos"] = pos
                m = members[sp][arm]
                assert set(m[yy[m] == 1].tolist()) == set(np.where(yy == 1)[0].tolist()), "positives differ"
                rows.append(dict(dataset=ds, threshold=a.threshold, split=sp, arm=arm, n=len(m),
                                 n_pos=int((yy[m] == 1).sum()), n_neg=int((yy[m] == 0).sum())))
            df["pooled_idx"] = pooled            # extra column (ignored by run_ft_v2): index into train+test pool
            out = os.path.join(OUT_DIR, f"{ds}__{sp}.csv.gz")
            df.to_csv(out, index=False)
            print(f"wrote {out} ({len(df)} rows)", flush=True)
    man = pd.DataFrame(rows)
    man.to_csv(os.path.join(_hc.RES, f"{_hc.PREFIX}_ft_split_manifest.csv"), index=False)
    print(man.to_string(index=False))


if __name__ == "__main__":
    main()
