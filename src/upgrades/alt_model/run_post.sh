#!/bin/bash
# Second queue (2026-09-27): shuffled arms on ONE pinned draw, the P4 lrx edge completions, analysis,
# then the single longest unit (uncapped multiclass EBM on human_ensembl_regulatory, ~6 h) last.
set -u
cd "$(dirname "$0")/../../.."
PY=/Users/rikhinkavuru/Downloads/venv/bin/python
W=src/upgrades/alt_model/run_alt_model.py
echo "[run_post] start $(date)"
rm -f external/alt_model/units/*__shuf__*
PYTHONHASHSEED=0 $PY $W --models lgbm,lr,ebm,lra,lrx --tiers 5                  # pinned shuffle draw
$PY $W --models lrx --tiers 1,2,3,4,5,9,8 --only-lr-edge                          # P4 as defined (edge units)
$PY src/upgrades/alt_model/analyze_alt_model.py
$PY $W --models ebm --tiers 8 --tasks human_ensembl_regulatory                    # longest unit, last
$PY src/upgrades/alt_model/analyze_alt_model.py
echo "[run_post] done $(date)"
