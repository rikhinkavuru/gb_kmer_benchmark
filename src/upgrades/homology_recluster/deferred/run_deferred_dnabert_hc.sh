#!/bin/bash
# DEFERRED (see README.md in this folder): DNABERT-2 LoRA fine-tuning on the id80 homology-clean splits.
# Needs >= 5 GB free RAM (process peak 3.5-3.8 GB on MPS) or a CUDA GPU (set DEVICE=cuda). Resumable per unit.
set -u
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT" || exit 1
PY="${PY:-$ROOT/.venv-ft/bin/python}"
PYL="${PYL:-/Users/rikhinkavuru/Downloads/venv/bin/python}"
M=src/upgrades/homology_recluster
DEV=${DEVICE:+--device $DEVICE}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.12
for DS in human_enhancers_cohn nt_enhancers; do
  echo "=== $(date '+%F %T') dnabert2 HC $DS"
  "$PY" -W ignore $M/run_ft_hc.py --model dnabert2 --dataset $DS --arms original,comp_equalized,random \
      --seeds 42,43,44 --pad-multiple 16 --mem-cap-mb 4000 $DEV || echo "!!! FAILED dnabert2 HC $DS"
done
"$PYL" $M/eval_ft_hc.py
