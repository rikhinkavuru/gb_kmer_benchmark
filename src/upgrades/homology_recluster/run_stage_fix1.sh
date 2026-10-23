#!/bin/bash
# Fix round 1 (2026-09-24 evening, after the power loss): finish the PENDING step-7 experiments of run_extra.py
# (pre-registered in its docstring; nothing new), then the one containment-coverage search that was skipped
# (human_ensembl_regulatory, --cov-mode 1). Strictly sequential, one process at a time, every process through
# bibm/lowcpu.sh (background QoS = efficiency cores, nice 20, 2 BLAS/OpenMP/LightGBM threads; mmseqs --threads 2,
# --split-memory-limit 2G). Every unit is checkpointed, so re-running this script resumes where it stopped.
# Order = reviewer priority: X4 (frozen-FM probe) and X7 (LR floor) were verifier must_fix items; then X3
# (composition-only baseline, 11 remaining tasks), X6, X8, X9; the ~3 h mmseqs search last.
# Usage: nohup src/upgrades/homology_recluster/run_stage_fix1.sh >> external/homology_recluster/logs/stage_fix1.log 2>&1 &
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT" || exit 1
PY=/Users/rikhinkavuru/Downloads/venv/bin/python
M=src/upgrades/homology_recluster
echo "=== $(date '+%F %T') stage_fix1 start (pid $$)"
$M/run_stage_extra.sh ${FIX1_EXPS:-X4 X7 X3 X6 X8 X9}
echo "=== $(date '+%F %T') covmode human_ensembl_regulatory"
/usr/bin/time -l bibm/lowcpu.sh $PY $M/run_covmode_sensitivity.py --tasks human_ensembl_regulatory 2>&1 \
  | grep -vE "^ +[0-9]+ +(instructions|cycles|page|voluntary|involuntary|block|messages|signals|swaps|average|maximum resident)" \
  || echo "!!! covmode FAILED"
echo "=== $(date '+%F %T') stage_fix1 finished"
