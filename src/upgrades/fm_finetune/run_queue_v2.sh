#!/bin/bash
# Sequential GPU queue for the reviewer-response fine-tuning runs (R1 comment 2). Resumable: every
# (model, dataset, arm, seed) unit whose prediction file + metadata exist is skipped, so re-running
# this script continues where it stopped. One job at a time; the only MPS user on the machine.
#
# CPU policy: HyenaDNA runs under bibm/lowcpu.sh (background QoS + nice 20 + 2 threads; measured
# 0.17-0.2 s/step, i.e. not GPU-starved). DNABERT-2 runs under plain `nice -n 15` with 2 threads,
# because background QoS measurably starved its MPS training (bench, batch 16, same code:
# 1.09 s/step under lowcpu.sh vs 0.75 s/step under nice -n 15, +44%). MPS memory is capped with
# the watermark ratios below (0.2 x 12.7 GB recommended working set = 2.5 GB hard cap).
#
# Order = priority (task items 1 -> 3 -> 4, then additions), seed-major so an interrupted queue
# leaves whole seeds: HyenaDNA cohn (orig/comp_eq/random) -> HyenaDNA nt_enhancers (4 arms) ->
# DNABERT-2 cohn seeds 42, 43 -> HyenaDNA drosophila (composition-aligned control) -> DNABERT-2
# cohn seed 44 -> HyenaDNA cohn gc_match -> DNABERT-2 nt_enhancers -> HyenaDNA drosophila gc_match.
#
# Usage: nohup src/upgrades/fm_finetune/run_queue_v2.sh [--after PID] \
#            >> results/upgrades/fm_finetune_v2_run.log 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT" || exit 1
PY="$ROOT/.venv-ft/bin/python"
RUN="src/upgrades/fm_finetune/run_ft_v2.py"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

mkdir -p "$ROOT/external/fm_finetune"
PIDF="$ROOT/external/fm_finetune/queue.pid"   # refuse to start a second copy
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "queue already running (pid $(cat "$PIDF"))"; exit 1
fi
echo $$ > "$PIDF"
trap 'rm -f "$PIDF"' EXIT

if [ "${1:-}" = "--after" ] && [ -n "${2:-}" ]; then   # wait for an already-running unit job
  echo "=== $(date '+%F %T') waiting for pid $2 to finish"
  while kill -0 "$2" 2>/dev/null; do sleep 120; done
fi

H() {  # HyenaDNA-tiny full fine-tune
  echo "=== $(date '+%F %T') hyenadna $*"
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.12 \
    "$ROOT/bibm/lowcpu.sh" "$PY" -W ignore "$RUN" --model hyenadna "$@" || echo "!!! FAILED hyenadna $*"
}
D() {  # DNABERT-2 LoRA
  echo "=== $(date '+%F %T') dnabert2 $*"
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.12 \
    nice -n 15 "$PY" -W ignore "$RUN" --model dnabert2 "$@" || echo "!!! FAILED dnabert2 $*"
}

MAIN=original,comp_equalized,random
H --dataset human_enhancers_cohn       --arms $MAIN --seeds 42,43,44
H --dataset nt_enhancers               --arms $MAIN,gc_match --seeds 42,43,44
D --dataset human_enhancers_cohn       --arms $MAIN --seeds 42,43
# DISABLED 2026-09-24: HyenaDNA on drosophila_enhancers_stark (long sequences) ran ~7 s/step and
# grew to a 17.6 GB memory footprint (swap exhausted on a 16 GB machine); killed. The frozen-FM
# probe already covers the drosophila control. Re-enable only with a sequence crop + memory cap.
# H --dataset drosophila_enhancers_stark --arms $MAIN --seeds 42,43,44
D --dataset human_enhancers_cohn       --arms $MAIN --seeds 44
H --dataset human_enhancers_cohn       --arms gc_match --seeds 42,43,44
D --dataset nt_enhancers               --arms $MAIN --seeds 42,43,44
# H --dataset drosophila_enhancers_stark --arms gc_match --seeds 42,43,44
echo "=== $(date '+%F %T') queue finished"
