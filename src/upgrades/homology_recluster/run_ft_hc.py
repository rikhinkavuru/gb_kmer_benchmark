#!/usr/bin/env python3
"""DEFERRED JOB (GPU / bigger machine): end-to-end fine-tuning (fm_finetune/run_ft_v2.py, unchanged recipe) on the
HOMOLOGY-CLEAN splits written by prepare_ft_hc_splits.py.

Torch process (run in .venv-ft; never import LightGBM here). This wrapper only redirects run_ft_v2's module-level
paths so nothing of the provided-split fine-tuning run is touched:
  splits       external/homology_recluster/ft_splits/
  predictions  results/upgrades/homology_recluster_ft_preds/      (via --pred-dir)
  checkpoints  external/homology_recluster/ft_checkpoints/
All other arguments are passed through to run_ft_v2.main (see its docstring for models / arms / seeds).
Example (same priority order as fm_finetune/run_queue_v2.sh):
  .venv-ft/bin/python src/upgrades/homology_recluster/run_ft_hc.py --model hyenadna \
      --dataset human_enhancers_cohn --arms original,comp_equalized,random --seeds 42,43,44
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "upgrades", "fm_finetune"))
import run_ft_v2 as R  # noqa: E402  (torch imported lazily inside)

R.SPLIT_DIR = os.path.join(ROOT, "external", "homology_recluster", "ft_splits")
R.CKPT_DIR = os.path.join(ROOT, "external", "homology_recluster", "ft_checkpoints")
PRED_DIR = os.path.join(ROOT, "results", "upgrades", "homology_recluster_ft_preds")
R.PRED_DIR = PRED_DIR
os.makedirs(R.CKPT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
if "--pred-dir" not in sys.argv:
    sys.argv += ["--pred-dir", PRED_DIR]

if __name__ == "__main__":
    R.main()
