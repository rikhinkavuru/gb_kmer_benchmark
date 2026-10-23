#!/bin/bash
# Driver for run_extra.py: one experiment per process (memory is released between experiments; a failure in
# one does not block the others), strictly sequential, background QoS + nice 20 + 2 threads via bibm/lowcpu.sh, peak memory
# recorded by /usr/bin/time -l. Checkpointed per unit, so re-running resumes.
# Usage: nohup src/upgrades/homology_recluster/run_stage_extra.sh [X1 X2 ...] >> external/homology_recluster/logs/stage_extra.log 2>&1 &
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT" || exit 1
PY=/Users/rikhinkavuru/Downloads/venv/bin/python
EXPS="${*:-X1 X2 X5 X3 X4 X6 X8 X9 X7}"
for e in $EXPS; do
  if [ "$e" = "X4" ]; then   # needs the embedding cache
    bibm/lowcpu.sh $PY src/upgrades/homology_recluster/fetch_fm_cache.py || { echo "!!! fetch failed; skipping X4"; continue; }
  fi
  echo "=== $(date '+%F %T') $e"
  /usr/bin/time -l bibm/lowcpu.sh $PY src/upgrades/homology_recluster/run_extra.py --exp "$e" 2>&1 \
    | grep -vE "^ +[0-9]+ +(instructions|cycles|page|voluntary|involuntary|block|messages|signals|swaps|average|maximum resident)" \
    || echo "!!! $e FAILED"
done
echo "=== $(date '+%F %T') stage_extra finished"
