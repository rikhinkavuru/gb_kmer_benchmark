#!/bin/bash
# Root-cause check for the 17.6 GB incident (HyenaDNA on drosophila_enhancers_stark, 2026-09-24).
# 1. padding invariance of HyenaDNA logits (tests/check_pad_invariance.py)
# 2. the same 40 training steps (same batches, seed 0 batcher) with the OLD padding (each batch to
#    its own longest sequence) and with --pad-multiple 512, logging per-step time and footprint.
# Both benches run under the 4 GB in-process cap AND the external watcher, so the old-padding run
# cannot repeat the incident: if it leaks, it is stopped at 4 GB.
# Usage: nohup src/upgrades/fm_finetune/bench_memfix.sh > results/upgrades/fm_finetune_v2_memfix_bench.log 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT" || exit 1
PY="$ROOT/.venv-ft/bin/python"
PYL=/Users/rikhinkavuru/Downloads/venv/bin/python
RUN="src/upgrades/fm_finetune/run_ft_v2.py"
GUARD="src/upgrades/fm_finetune/memguard.py"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.12

guarded() {   # guarded <cap_mb> <cmd...> : run in background, external watcher at cap+10%, wait
  local cap="$1"; shift
  "$@" & local pid=$!
  nice -n 20 "$PYL" "$GUARD" --watch "$pid" --cap-mb "$(( cap * 11 / 10 ))" --interval 10 &
  local wpid=$!
  wait "$pid"; local rc=$?
  wait "$wpid" 2>/dev/null
  echo "=== exit code $rc"
  return $rc
}

echo "=== $(date '+%F %T') pad invariance"
guarded 3000 "$ROOT/bibm/lowcpu.sh" "$PY" -W ignore src/upgrades/fm_finetune/tests/check_pad_invariance.py
echo "=== $(date '+%F %T') bench drosophila --pad-multiple 512 (fixed shapes), 40 steps"
guarded 4000 "$ROOT/bibm/lowcpu.sh" "$PY" -W ignore "$RUN" --model hyenadna \
  --dataset drosophila_enhancers_stark --bench 40 --pad-multiple 512 --mem-cap-mb 4000
echo "=== $(date '+%F %T') bench drosophila OLD padding (per-batch longest), 40 steps"
guarded 4000 "$ROOT/bibm/lowcpu.sh" "$PY" -W ignore "$RUN" --model hyenadna \
  --dataset drosophila_enhancers_stark --bench 40 --pad-multiple 0 --mem-cap-mb 4000
echo "=== $(date '+%F %T') done"
