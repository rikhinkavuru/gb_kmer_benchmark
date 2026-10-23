#!/bin/bash
# Deferred GPU jobs for the fine-tuning experiments, ranked by value to the paper (see README.md).
# Run from the bundle root (the directory that contains src/ and external/). Resumable: a unit whose
# prediction file + JSON already exist in results/upgrades/fm_finetune_v2_preds/ is skipped, so
# units the laptop already produced are never re-run. Re-running the script continues where it
# stopped.
#
#   bash src/upgrades/fm_finetune/gpu_deferred/jobs.sh check     # env + split checksums + model download
#   bash src/upgrades/fm_finetune/gpu_deferred/jobs.sh bench     # 30-step timing of each new model
#   bash src/upgrades/fm_finetune/gpu_deferred/jobs.sh all       # every job, in rank order
#   bash src/upgrades/fm_finetune/gpu_deferred/jobs.sh 2 3       # only jobs ranked 2 and 3
#
# Log: results/upgrades/fm_finetune_gpu_run.log (append). One GPU, one job at a time.
set -u
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT" || exit 1
PY="${PY:-python}"
RUN=src/upgrades/fm_finetune/gpu_deferred/run_gpu.py
LOG=results/upgrades/fm_finetune_gpu_run.log
mkdir -p results/upgrades
export TOKENIZERS_PARALLELISM=false
MAIN=original,comp_equalized,random
SEEDS=42,43,44

r() { echo "=== $(date '+%F %T') run_gpu.py $*" | tee -a "$LOG"
      "$PY" -W ignore "$RUN" "$@" 2>&1 | tee -a "$LOG"
      local rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || echo "!!! FAILED ($rc): $*" | tee -a "$LOG"; }

job() {
  case "$1" in
    1) # HyenaDNA-tiny full FT, composition-ALIGNED control (pre-registered rule 5). FALLBACK:
       # skipped unit by unit wherever the laptop's memory-capped run already produced it.
       r --model hyenadna --dataset drosophila_enhancers_stark --arms $MAIN --seeds $SEEDS
       r --model hyenadna --dataset drosophila_enhancers_stark --arms gc_match --seeds $SEEDS ;;
    2) # DNABERT-2-117M FULL-parameter fine-tune (Reviewer 1: "full-parameter or LoRA")
       r --model dnabert2full --dataset human_enhancers_cohn --arms $MAIN --seeds $SEEDS ;;
    3) # Nucleotide Transformer v2 500M, LoRA: a larger mainstream FM
       r --model ntv2_500m --dataset human_enhancers_cohn --arms $MAIN --seeds $SEEDS ;;
    4) # DNABERT-2 LoRA on nt_enhancers (second unaligned dataset). FALLBACK: skipped where the
       # laptop queue produced it
       r --model dnabert2 --dataset nt_enhancers --arms $MAIN --seeds $SEEDS ;;
    5) # NT-v2 500M on the composition-aligned control (second FM for rule 5; no truncation)
       r --model ntv2_500m --dataset drosophila_enhancers_stark --arms $MAIN --seeds $SEEDS ;;
    6) r --model ntv2_500m --dataset nt_enhancers --arms $MAIN --seeds $SEEDS ;;
    7) r --model dnabert2full --dataset nt_enhancers --arms $MAIN --seeds $SEEDS ;;
    *) echo "unknown job $1"; return 1 ;;
  esac
}

case "${1:-all}" in
  check)
    "$PY" -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, '| transformers', transformers.__version__, '| peft', peft.__version__)"
    "$PY" -W ignore "$RUN" --prefetch 2>&1 | tee -a "$LOG"
    ( cd "$ROOT" && sha256sum -c <(grep -v '^#' src/upgrades/fm_finetune/gpu_deferred/SHA256SUMS) ) ;;
  bench)
    r --model dnabert2full --dataset human_enhancers_cohn --bench 30
    r --model ntv2_500m    --dataset human_enhancers_cohn --bench 30
    r --model ntv2_500m    --dataset drosophila_enhancers_stark --bench 30
    r --model hyenadna     --dataset drosophila_enhancers_stark --bench 30 ;;
  all)
    for j in 1 2 3 4 5 6 7; do job "$j"; done ;;
  *)
    for j in "$@"; do job "$j"; done ;;
esac
echo "=== $(date '+%F %T') jobs.sh $* finished" | tee -a "$LOG"
