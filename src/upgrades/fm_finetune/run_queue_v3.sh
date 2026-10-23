#!/bin/bash
# GPU queue v3 (2026-09-24), replaces run_queue_v2.sh after the 17.6 GB memory incident.
#
# Root cause (memguard.py): mini-batches whose tensor shapes change from batch to batch make the
# MPS backend compile and cache new graphs without bound -- HyenaDNA on drosophila (535 distinct
# lengths; FFT of size 2L) reached 17.6 GB; DNABERT-2 (unpadding -> a new (total_tokens, 768)
# shape nearly every batch) grew ~175 MB/min. Fixes, applied to every remaining unit:
#   * --pad-multiple K: pad to a multiple of K tokens and, for DNABERT-2, keep pad rows instead of
#     unpadding -> a handful of shapes. Numerically a no-op for the real tokens
#     (tests/check_pad_invariance.py must PASS, else nothing is trained).
#   * --micro-batch 4 --eval-batch 8 for drosophila (2-3.5 kb): the optimizer batch stays 32; its
#     gradient is accumulated exactly over micro-batches (run_ft_v2.accumulate_batch), so only
#     memory changes (needed: 32 x 3.5 kb exceeds the 2.4 GB MPS allocator cap).
#   * hard memory caps: in-process (run_ft_v2.py --mem-cap-mb, exit 86) + external watcher at
#     +10% (memguard.py --watch: SIGTERM then SIGKILL). A unit that hits the cap is logged FAILED
#     and the queue moves on.
# Recipes, arms, seeds, validation carve and outputs are unchanged (run_ft_v2.py; resumable per unit,
# finished units are skipped). Remaining units, in priority order:
#   1. HyenaDNA  drosophila_enhancers_stark  original/comp_eq/random  s42-44  (rule-5 aligned control)
#   2. DNABERT-2 human_enhancers_cohn        original/comp_eq/random  s44     (third seed)
#   3. HyenaDNA  human_enhancers_cohn        gc_match                 s42-44  (secondary, rule 4)
#   4. DNABERT-2 nt_enhancers                original/comp_eq/random  s42-44
#   5. HyenaDNA  drosophila_enhancers_stark  gc_match                 s42-44
# One job at a time; waits for any other queue holding external/fm_finetune/queue.pid, then holds it.
# Usage: nohup src/upgrades/fm_finetune/run_queue_v3.sh >> results/upgrades/fm_finetune_v2_run.log 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT" || exit 1
PY="$ROOT/.venv-ft/bin/python"
PYL=/Users/rikhinkavuru/Downloads/venv/bin/python
RUN="src/upgrades/fm_finetune/run_ft_v2.py"
GUARD="src/upgrades/fm_finetune/memguard.py"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.12
CAP_H=3500   # MB in-process cap, HyenaDNA-tiny
CAP_D=4000   # MB in-process cap, DNABERT-2 (117M weights + LoRA + activations)

PIDF="$ROOT/external/fm_finetune/queue.pid"
while [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; do sleep 300; done
echo $$ > "$PIDF"
trap 'rm -f "$PIDF"' EXIT
echo "=== $(date '+%F %T') queue v3 (pid $$) starts"

guarded() {   # guarded <cap_mb> <cmd...>: run it with the external watcher at cap+10%
  local cap="$1"; shift
  "$@" & local pid=$!
  nice -n 20 "$PYL" "$GUARD" --watch "$pid" --cap-mb "$(( cap * 11 / 10 ))" --interval 10 \
    --report-every 1800 &
  local wpid=$!
  wait "$pid"; local rc=$?
  wait "$wpid" 2>/dev/null
  return $rc
}
H() {  # HyenaDNA-tiny full fine-tune (CPU side under bibm/lowcpu.sh: nice 20, 3 threads)
  echo "=== $(date '+%F %T') hyenadna $* (cap ${CAP_H} MB)"
  guarded $CAP_H "$ROOT/bibm/lowcpu.sh" "$PY" -W ignore "$RUN" --model hyenadna \
    --mem-cap-mb $CAP_H "$@" || echo "!!! FAILED hyenadna $* (exit $?)"
}
D() {  # DNABERT-2 LoRA (nice 15, 2 threads: background QoS starved its MPS feed, see v2 notes)
  echo "=== $(date '+%F %T') dnabert2 $* (cap ${CAP_D} MB)"
  guarded $CAP_D env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 nice -n 15 "$PY" -W ignore "$RUN" \
    --model dnabert2 --mem-cap-mb $CAP_D "$@" || echo "!!! FAILED dnabert2 $* (exit $?)"
}

if ! grep -q "^PASS" results/upgrades/fm_finetune_v2_pad_invariance.log 2>/dev/null; then
  echo "!!! tests/check_pad_invariance.py has not PASSED -- refusing to train"; exit 1
fi

# memory/speed evidence: identical batches, fixed shapes vs the old per-batch shapes (old runs
# are capped, so they cannot repeat the incident). Ran once (2026-09-24 10:58-11:07, tabulated by
# memfix_evidence.py); set RUN_BENCH=1 to repeat.
DROS="--pad-multiple 512 --micro-batch 4 --eval-batch 8"   # 3.5 kb x 32 exceeds the 2.4 GB MPS cap;
# micro-batch 8 OOMed the allocator cap at width 3584 (11:07 run); 4 and 6 measured OK (driver 1.3/1.4 GB)
if [ "${RUN_BENCH:-0}" = "1" ]; then
echo "=== $(date '+%F %T') bench: dnabert2 cohn 80 steps, fixed shapes (--pad-multiple 16)"
guarded $CAP_D nice -n 15 "$PY" -W ignore "$RUN" --model dnabert2 --dataset human_enhancers_cohn \
  --bench 80 --pad-multiple 16 --mem-cap-mb $CAP_D
echo "=== $(date '+%F %T') bench: dnabert2 cohn 80 steps, OLD shapes (unpadded)"
guarded $CAP_D nice -n 15 "$PY" -W ignore "$RUN" --model dnabert2 --dataset human_enhancers_cohn \
  --bench 80 --pad-multiple 0 --mem-cap-mb $CAP_D
echo "=== $(date '+%F %T') bench: hyenadna drosophila 30 steps, fixed shapes ($DROS)"
guarded $CAP_H "$ROOT/bibm/lowcpu.sh" "$PY" -W ignore "$RUN" --model hyenadna \
  --dataset drosophila_enhancers_stark --bench 30 $DROS --mem-cap-mb $CAP_H
echo "=== $(date '+%F %T') bench: hyenadna drosophila 30 steps, OLD per-batch padding (micro-batch 4)"
guarded $CAP_H "$ROOT/bibm/lowcpu.sh" "$PY" -W ignore "$RUN" --model hyenadna \
  --dataset drosophila_enhancers_stark --bench 30 --pad-multiple 0 --micro-batch 4 --eval-batch 8 \
  --mem-cap-mb $CAP_H
fi

MAIN=original,comp_equalized,random
H --dataset drosophila_enhancers_stark --arms $MAIN --seeds 42,43,44 $DROS
D --dataset human_enhancers_cohn       --arms $MAIN --seeds 44 --pad-multiple 16
H --dataset human_enhancers_cohn       --arms gc_match --seeds 42,43,44   # 500 bp: one shape already
D --dataset nt_enhancers               --arms $MAIN --seeds 42,43,44 --pad-multiple 16
H --dataset drosophila_enhancers_stark --arms gc_match --seeds 42,43,44 $DROS
echo "=== $(date '+%F %T') queue v3 finished"
