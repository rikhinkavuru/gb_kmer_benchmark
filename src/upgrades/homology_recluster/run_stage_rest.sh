#!/bin/bash
# Stage B: waits for the clustering run and stage A (PIDs as args), then runs everything else
# sequentially (one heavy job at a time), every command through bibm/lowcpu.sh (nice 20, <=3 threads).
cd /Users/rikhinkavuru/gbkb_release || exit 1
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 60; done; done
PY=/Users/rikhinkavuru/Downloads/venv/bin/python
W=bibm/lowcpu.sh
M=src/upgrades/homology_recluster
echo "== stage B start $(date)"
$W $PY -W ignore $M/run_cluster.py || exit 1          # no-op if every task is cached; writes the table
$W $PY -W ignore $M/run_eval.py || exit 1
$W $PY -W ignore $M/run_routes.py || exit 1
$W $PY -W ignore $M/run_interventions.py --exp E3 --tiers 1,2 || exit 1
$W $PY -W ignore $M/run_routes.py || exit 1
echo "== stage B done $(date)"
