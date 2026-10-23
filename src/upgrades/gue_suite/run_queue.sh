#!/bin/bash
# Sequential queue (one heavy job at a time): [wait for run_gue.py] -> run_gue_posthoc.py -> paper15_reference.py
# Every step is checkpointed per task, so re-running the queue resumes where it stopped.
# Usage: nohup src/upgrades/gue_suite/run_queue.sh > external/gue/logs/queue.log 2>&1 &
cd "$(dirname "$0")/../../.."
export GUE_THREADS="${GUE_THREADS:-3}" PYTHONWARNINGS="ignore::UserWarning"
PY=/Users/rikhinkavuru/Downloads/venv/bin/python
while pgrep -f "src/upgrades/gue_suite/run_gue.py" > /dev/null; do sleep 30; done
echo "[queue] main run finished $(date)"
bibm/lowcpu.sh $PY src/upgrades/gue_suite/run_gue_posthoc.py > external/gue/logs/posthoc.log 2>&1
echo "[queue] posthoc exit=$? $(date)"
bibm/lowcpu.sh $PY src/upgrades/gue_suite/paper15_reference.py > external/gue/logs/paper15.log 2>&1
echo "[queue] paper15 exit=$? $(date)"
