#!/bin/bash
# Stage C: after stage B (PID arg) -> split-seed robustness, then the report + figure.
cd /Users/rikhinkavuru/gbkb_release || exit 1
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 60; done; done
PY=/Users/rikhinkavuru/Downloads/venv/bin/python
W=bibm/lowcpu.sh
M=src/upgrades/homology_recluster
echo "== stage C start $(date)"
$W $PY -W ignore $M/run_seeds.py || exit 1
$W $PY -W ignore $M/make_report.py || exit 1
echo "== stage C done $(date)"
