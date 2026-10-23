#!/bin/bash
# Homology-clean END-TO-END FINE-TUNING, local part (fix round 1, 2026-09-25). HyenaDNA-tiny only: its units peak at
# ~2.0-2.9 GB (memguard log of the provided-split queue), inside the laptop's per-job budget; DNABERT-2 LoRA units
# peak at 3.5-3.8 GB and stay DEFERRED (deferred/README.md). Recipe, arms, seeds, validation carve and the
# memory-safe settings are fm_finetune's queue v3 unchanged (run_ft_v2.py via run_ft_hc.py, which only redirects
# splits / predictions / checkpoints to this module). Splits: prepare_ft_hc_splits.py (id80 homology-clean,
# arms rebuilt inside the HC splits; manifest results/upgrades/homology_recluster_ft_split_manifest.csv).
# Order = priority: cohn main arms (rules 1-3), drosophila main arms (rule 5 control), nt_enhancers (its HC
# comp_equalized test arm keeps 3 negatives -> contrasts reported as not evaluable), then the gc_match arms.
# Starts only after run_stage_fix1.sh has finished (one heavy job at a time) and while no fm_finetune queue holds
# external/fm_finetune/queue.pid. Units are resumable (finished units are skipped by run_ft_v2.py).
# Usage: nohup src/upgrades/homology_recluster/run_stage_ft_hc.sh >> external/homology_recluster/logs/stage_ft_hc.log 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT" || exit 1
PY="$ROOT/.venv-ft/bin/python"
PYL=/Users/rikhinkavuru/Downloads/venv/bin/python
M=src/upgrades/homology_recluster
GUARD=src/upgrades/fm_finetune/memguard.py
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.12
CAP_H=${CAP_H:-3000}   # MB in-process cap (run_ft_v2 --mem-cap-mb, exit 86); external watcher at +10%
DROS="--pad-multiple 512 --micro-batch 4 --eval-batch 8"
echo "=== $(date '+%F %T') stage_ft_hc (pid $$) waiting for run_stage_fix1.sh and any fm_finetune queue"
while pgrep -f "run_stage_fix1.sh" >/dev/null || pgrep -f "run_extra.py --exp" >/dev/null || pgrep -f "run_covmode_sensitivity.py" >/dev/null \
      || { [ -f external/fm_finetune/queue.pid ] && kill -0 "$(cat external/fm_finetune/queue.pid)" 2>/dev/null; }; do
  sleep 300
done
if ! grep -q "^PASS" results/upgrades/fm_finetune_v2_pad_invariance.log 2>/dev/null; then
  echo "!!! fm_finetune tests/check_pad_invariance.py has not PASSED -- refusing to train"; exit 1
fi
echo "=== $(date '+%F %T') stage_ft_hc starts"
guarded() {   # guarded <cap_mb> <cmd...>: run it with the external watcher at cap+10%
  local cap="$1"; shift
  "$@" & local pid=$!
  nice -n 20 "$PYL" "$GUARD" --watch "$pid" --cap-mb "$(( cap * 11 / 10 ))" --interval 10 --report-every 1800 &
  local wpid=$!
  wait "$pid"; local rc=$?
  wait "$wpid" 2>/dev/null
  return $rc
}
H() {
  echo "=== $(date '+%F %T') hyenadna HC $* (cap ${CAP_H} MB)"
  guarded $CAP_H "$ROOT/bibm/lowcpu.sh" "$PY" -W ignore $M/run_ft_hc.py --model hyenadna --mem-cap-mb $CAP_H "$@" \
    || echo "!!! FAILED hyenadna HC $* (exit $?)"
}
MAIN=original,comp_equalized,random
H --dataset human_enhancers_cohn       --arms $MAIN --seeds 42,43,44
H --dataset drosophila_enhancers_stark --arms $MAIN --seeds 42,43,44 $DROS
H --dataset nt_enhancers               --arms $MAIN,gc_match --seeds 42,43,44
H --dataset human_enhancers_cohn       --arms gc_match --seeds 42,43,44
H --dataset drosophila_enhancers_stark --arms gc_match --seeds 42,43,44 $DROS
echo "=== $(date '+%F %T') eval (torch-free)"
/usr/bin/time -l bibm/lowcpu.sh $PYL $M/eval_ft_hc.py 2>&1 | grep -vE "^ +[0-9]+ +(instructions|cycles|page|voluntary|involuntary|block|messages|signals|swaps|average|maximum resident)" \
  || echo "!!! eval_ft_hc FAILED"
echo "=== $(date '+%F %T') stage_ft_hc finished"
