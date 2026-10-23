#!/usr/bin/env python3
"""DEFERRED JOB, step 2 (torch-free): score the homology-clean fine-tuning runs of run_ft_hc.py with the UNCHANGED
pre-registered analysis of fm_finetune/eval_ft_v2.py (same metrics, contrasts S/T/U/R, decision rules), redirected
to the homology-clean splits and predictions. Outputs results/upgrades/homology_recluster_ft_{runs,metrics,summary,
contrasts,verdicts,score_composition,score_composition_summary}.csv.
Usage: python src/upgrades/homology_recluster/eval_ft_hc.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "upgrades", "fm_finetune"))
import eval_ft_v2 as E  # noqa: E402

E.SPLIT_DIR = os.path.join(ROOT, "external", "homology_recluster", "ft_splits")
E.PRED_DIR = os.path.join(ROOT, "results", "upgrades", "homology_recluster_ft_preds")
if "--pred-dir" not in sys.argv:
    sys.argv += ["--pred-dir", E.PRED_DIR]
if "--out-prefix" not in sys.argv:
    sys.argv += ["--out-prefix", os.path.join(ROOT, "results", "upgrades", "homology_recluster_ft")]

if __name__ == "__main__":
    E.main()
