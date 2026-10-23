#!/bin/bash
# Local end-to-end wiring test of the GPU bundle, on the LAPTOP CPU (never MPS, which belongs to the
# local fine-tuning queue). Tiny on purpose: 3 optimizer steps on 16 train sequences per class per
# arm, sequences cropped (HyenaDNA 256 bp, the others 64 bp), DNABERT-2 cut to 2 encoder layers
# (LoRA: after the full pretrained model is built and checked, because run_ft_v2 asserts 48 LoRA
# modules; full FT: built with 2 layers, and every loaded tensor is checked), and an in-process hard
# cap of 1.1 GB.
# Each unit still predicts EVERY sequence of the original (cropped) test split, so the output files
# have exactly the format of real runs. eval_ft_v2.py then scores them, unchanged, from their own
# directory. Nothing is written to results/upgrades/fm_finetune_v2_preds.
#
# Usage: nohup src/upgrades/fm_finetune/gpu_deferred/smoke_local.sh \
#            > results/upgrades/fm_finetune_gpu_deferred_smoke.log 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT" || exit 1
FT="$ROOT/.venv-ft/bin/python"
PYL=/Users/rikhinkavuru/Downloads/venv/bin/python
RUN=src/upgrades/fm_finetune/gpu_deferred/run_gpu.py
OUT="$ROOT/external/fm_finetune/gpu_smoke"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
rm -rf "$OUT/preds" "$OUT/eval"; mkdir -p "$OUT/preds" "$OUT/eval"
COMMON="--device cpu --threads 2 --seeds 42 --smoke 3 --subset 16 --mem-cap-mb 1100 --pred-dir $OUT/preds"
fail=0
s() { echo "=== $(date '+%F %T') smoke: $*"; "$ROOT/bibm/lowcpu.sh" "$FT" -W ignore "$RUN" $COMMON "$@" \
        || { echo "!!! SMOKE FAILED ($?): $*"; fail=1; }; }
s --model hyenadna     --dataset drosophila_enhancers_stark --arms original,comp_equalized,random,gc_match --crop 256
for arm in original comp_equalized random; do   # one process per arm: the 12-layer LoRA model
  s --model dnabert2   --dataset nt_enhancers               --arms $arm --crop 64 --smoke-layers 2
done                                             # is built in full before truncation (~1 GB)
s --model dnabert2full --dataset human_enhancers_cohn       --arms original,comp_equalized,random --crop 64 --smoke-layers 2
s --model ntv2_50m     --dataset human_enhancers_cohn       --arms original,comp_equalized,random --crop 64
echo "=== $(date '+%F %T') eval_ft_v2.py (unchanged) on the smoke predictions"
"$ROOT/bibm/lowcpu.sh" "$PYL" src/upgrades/fm_finetune/eval_ft_v2.py --pred-dir "$OUT/preds" \
    --out-prefix "$OUT/eval/smoke" || { echo "!!! EVAL FAILED"; fail=1; }
"$PYL" - "$OUT" <<'EOF'
import glob, json, os, sys
import pandas as pd
out = sys.argv[1]
js = sorted(glob.glob(os.path.join(out, "preds", "*.json")))
rows = []
for f in js:
    m = json.load(open(f))
    rows.append(dict(unit=os.path.basename(f)[:-5], device=m["device"], steps=m["max_steps"],
                     trainable=m["n_params_trainable"], peak_footprint_mb=m.get("peak_footprint_mb"),
                     triton_blocked=m.get("triton_blocked"), runner=m.get("runner")))
df = pd.DataFrame(rows)
print(df.to_string(index=False))
met = pd.read_csv(os.path.join(out, "eval", "smoke_metrics.csv"))
ct = pd.read_csv(os.path.join(out, "eval", "smoke_contrasts.csv"))
print(f"eval_ft_v2 ingested {met.groupby(['model', 'dataset', 'train_arm', 'seed']).ngroups} runs "
      f"({len(js)} prediction files), {len(met)} metric rows, {len(ct)} contrast rows, models "
      f"{sorted(met.model.unique())}")
assert met.groupby(['model', 'dataset', 'train_arm', 'seed']).ngroups == len(js)
assert (df.peak_footprint_mb < 1100).all()
print("SMOKE OK: max peak footprint %.0f MB" % df.peak_footprint_mb.max())
EOF
[ $? -eq 0 ] || fail=1
echo "=== $(date '+%F %T') smoke finished, fail=$fail"
exit $fail
