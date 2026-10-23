#!/usr/bin/env python3
"""Reproduction check (needs the prediction cache written by run_psm_sensitivity.py):
the cached `original` and `paper_equalized_matched` k-mer / composition-only predictions must
reproduce the paper's numbers in results/upgrades/composition_clean.csv (original + comp_equalized
rows: AUROC and MCC at 0.5), which were produced by a different script (run_composition_clean.py).

Tolerance: original rows must match to 1e-4 (same training rows, same order); the paper-anchor
rows are allowed 0.01 AUROC / 0.03 MCC because run_composition_clean trains on the cleaned CSV's row
order and LightGBM tie-breaking can differ with row order (the numbers are printed either way).

Run:  bibm/lowcpu.sh <python> src/upgrades/psm_sensitivity/tests/test_reproduces_paper.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import _common as C  # noqa: E402
import psm_data as D  # noqa: E402

TASKS = ["human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark"]
PREDS = os.path.join(D.SNAP_DIR, "preds")


def anchor_test_idx(task, te, yte):
    df = pd.read_csv(os.path.join(C.ROOT, "cleaned_splits_v2", f"{task}_comp_equalized_test.csv"))
    pool = {}
    for i, (sq, lab) in enumerate(zip(te, yte)):
        pool.setdefault((sq, int(lab)), []).append(i)
    got = [pool[(sq, int(lab))].pop() for sq, lab in zip(df["sequence"].astype(str).str.upper(), df["label"])]
    return np.sort(np.array(got))


def test_reproduces_composition_clean():
    ref = pd.read_csv(os.path.join(C.RESULTS_DIR, "composition_clean.csv"))
    bad = []
    for task in TASKS:
        tr, ytr, te, yte, _, _ = D.load_task(task)
        yte = np.asarray(yte)
        for arm, ref_arm, idx, tol_a, tol_m in (
                ("original", "original", np.arange(len(yte)), 1e-4, 1e-4),
                ("paper_equalized_matched", "comp_equalized", anchor_test_idx(task, te, yte), 0.01, 0.03)):
            f = os.path.join(PREDS, task, f"{arm}.npz")
            if not os.path.exists(f):
                print(f"SKIP {task} {arm}: no cache")
                continue
            d = np.load(f)
            r = ref[(ref["dataset"] == task) & (ref["arm"] == ref_arm)].iloc[0]
            assert len(idx) == int(r["n_test"]), (task, arm, len(idx), r["n_test"])
            for mk, col in (("kmer", "kmer"), ("comp", "comp_only")):
                p = d[mk][idx]
                a = roc_auc_score(yte[idx], p)
                m = matthews_corrcoef(yte[idx], (p > 0.5).astype(int))
                ra, rm = float(r[f"{col}_auroc"]), float(r[f"{col}_mcc"])
                ok = abs(a - ra) <= tol_a and abs(m - rm) <= tol_m
                print(f"{'OK ' if ok else 'BAD'} {task:<28} {arm:<24} {mk:<5} AUROC {a:.6f} (paper {ra:.6f})  "
                      f"MCC {m:.6f} (paper {rm:.6f})  n_test={len(idx)}")
                if not ok:
                    bad.append((task, arm, mk))
    assert not bad, f"not reproduced: {bad}"


if __name__ == "__main__":
    test_reproduces_composition_clean()
    print("PASS test_reproduces_composition_clean")
