#!/bin/bash
# Run the pre-registered GUE pipeline on all 28 datasets (checkpointed: finished cards are skipped).
# Usage: nohup src/upgrades/gue_suite/run_all.sh > external/gue/logs/run_gue.log 2>&1 &
cd "$(dirname "$0")/../../.."
export GUE_THREADS="${GUE_THREADS:-3}" PYTHONWARNINGS="ignore::UserWarning"
PY=/Users/rikhinkavuru/Downloads/venv/bin/python
exec bibm/lowcpu.sh $PY src/upgrades/gue_suite/run_gue.py "$@"
