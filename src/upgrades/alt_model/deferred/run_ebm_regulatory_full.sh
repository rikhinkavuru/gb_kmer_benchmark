#!/bin/bash
# DEFERRED UNIT (post hoc P2): uncapped multiclass EBM on human_ensembl_regulatory (231,348 train /
# 57,713 test, k=5, 3 classes). Everything else in the module is finished; this single unit was stopped
# after 3 h 45 min on the 16 GB MacBook Air (3 threads: the 4 outer bags run as 3 + 1, each bag > 3.7 h,
# i.e. ~9-10 h in total) because it cannot change any label (capped EBM MCC 0.879 [0.876, 0.883],
# far above the 0.70 cut) and was blocking the camera-ready workflow.
# Hardware: any CPU box; >=4 cores lets all 4 bags run at once (~4 h here-equivalent); RAM < 2 GB.
# Usage (from the repo root):  nohup src/upgrades/alt_model/deferred/run_ebm_regulatory_full.sh > external/alt_model/logs/deferred_ebm_full.log 2>&1 &
set -eu
cd "$(dirname "$0")/../../../.."
PY=${PY:-/Users/rikhinkavuru/Downloads/venv/bin/python}
bibm/lowcpu.sh $PY src/upgrades/alt_model/run_alt_model.py --models ebm --tiers 8 --tasks human_ensembl_regulatory
bibm/lowcpu.sh $PY src/upgrades/alt_model/analyze_alt_model.py      # folds the unit into alt_model_bench.csv / labels
