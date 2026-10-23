#!/bin/bash
# Fix round 2 (2026-09-27): analysis only (post hoc P11 of the run_alt_model.py docstring -- overlap design
# on every gate-positive task + ocr, post hoc overlap labelling scheme, MCC size-share CIs, Route-1 uncapped
# bookkeeping); no new units. Usage (from the repo root):
#   nohup bibm/lowcpu.sh src/upgrades/alt_model/run_fix2.sh > external/alt_model/logs/run_fix2.log 2>&1 &
set -u
cd "$(dirname "$0")/../../.."
PY=/Users/rikhinkavuru/Downloads/venv/bin/python
echo "[run_fix2] start $(date)"
$PY src/upgrades/alt_model/analyze_alt_model.py                                   # every CSV + interpretation
$PY -m pytest -q src/upgrades/alt_model/tests
echo "[run_fix2] done $(date)"
