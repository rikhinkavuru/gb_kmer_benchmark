#!/bin/bash
# Sequential queue for the alt_model module (one heavy job at a time; every step resumable:
# finished units / dose rows are skipped). Usage (from the repo root):
#   nohup bibm/lowcpu.sh src/upgrades/alt_model/run_all.sh > external/alt_model/logs/run_all.log 2>&1 &
set -u
cd "$(dirname "$0")/../../.."
PY=/Users/rikhinkavuru/Downloads/venv/bin/python
W=src/upgrades/alt_model/run_alt_model.py
echo "[run_all] start $(date)"
$PY src/upgrades/alt_model/cache_seqs.py                                   # validated sequence cache
$PY src/upgrades/alt_model/run_gate.py                                     # model-free Route-3 gate
$PY src/upgrades/alt_model/fix_float32.py                                  # drop float32-probability units
$PY $W --models lgbm,lr,ebm --tiers 1,2,3,4,5                              # pre-registered units
$PY src/upgrades/alt_model/run_dose_alt.py --models lgbm,lr,ebm            # pre-registered Route-1 dose
$PY $W --models lgbm,lr,ebm --tiers 6,7                                    # post hoc P1 (random negatives)
$PY $W --models lra --tiers 1,3,5,6,7                                      # post hoc P4b (AUROC-tuned LR)
$PY $W --models lrx --tiers 1,2,3,4,5                                      # post hoc P4 (extended C grid)
$PY $W --models ebm512 --tiers 1 --feats kmer                              # post hoc P7 (EBM capacity)
$PY $W --models lgbm,lr,ebm --tiers 9,8                                    # post hoc P3, P2 (uncapped)
$PY $W --models lrx --tiers 8,9                                            # post hoc P4 on uncapped arms
echo "[run_all] done $(date)"
