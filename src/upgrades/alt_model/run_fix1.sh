#!/bin/bash
# Fix round 1 (2026-09-27): post hoc P9 units (EBM B=8 arm with position-binned k-mers guaranteed),
# then the full analysis (P10 additions live in analyze_alt_model.py). Resumable (finished units skipped).
# Usage (from the repo root):
#   nohup bibm/lowcpu.sh src/upgrades/alt_model/run_fix1.sh > external/alt_model/logs/run_fix1.log 2>&1 &
set -u
cd "$(dirname "$0")/../../.."
PY=/Users/rikhinkavuru/Downloads/venv/bin/python
W=src/upgrades/alt_model/run_alt_model.py
echo "[run_fix1] start $(date)"
$PY $W --models ebmpos --tiers 2,4,9 --feats B8                                  # post hoc P9
echo "[run_fix1] units done $(date)"
$PY src/upgrades/alt_model/analyze_alt_model.py                                   # all statistics + P10 additions
$PY -m pytest -q src/upgrades/alt_model/tests
echo "[run_fix1] done $(date)"
