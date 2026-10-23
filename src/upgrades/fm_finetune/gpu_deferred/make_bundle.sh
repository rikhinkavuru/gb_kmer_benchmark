#!/bin/bash
# Build the self-contained GPU bundle (code + verified split files + the laptop's finished prediction
# files, so the GPU never re-runs a unit that already exists) as
#   external/fm_finetune/gbkb_ft_gpu_bundle.tar.gz      (unpacks to gbkb_ft_gpu/)
# plus BUNDLE_MANIFEST.txt (sha256 of every file, the source commit, and the included units).
# Re-run it just before copying the bundle, so that it carries the latest local predictions.
#
# Usage: bash src/upgrades/fm_finetune/gpu_deferred/make_bundle.sh
set -eu
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
STAGE="$ROOT/external/fm_finetune/gpu_bundle_stage"
NAME=gbkb_ft_gpu
B="$STAGE/$NAME"
rm -rf "$STAGE"; mkdir -p "$B"
FT=src/upgrades/fm_finetune
GD=$FT/gpu_deferred

# 1. code (run_ft_v2.py + its two local imports, the runner, the job list, env pins, checksums)
mkdir -p "$B/$GD/tests"
cp $FT/run_ft_v2.py $FT/run_fm_finetune.py $FT/memguard.py "$B/$FT/"
cp $GD/run_gpu.py $GD/extra_models.py $GD/jobs.sh $GD/requirements-gpu.txt $GD/SHA256SUMS \
   $GD/README.md "$B/$GD/"
cp $GD/tests/check_extra_models.py $GD/tests/ntv2_tf4_reference.py "$B/$GD/tests/"
chmod +x "$B/$GD/jobs.sh"

# 2. split files, verified against SHA256SUMS before they are packed
mkdir -p "$B/external/fm_finetune/splits"
grep -v '^#' $GD/SHA256SUMS | while read -r digest rel; do
  got=$(shasum -a 256 "$rel" | awk '{print $1}')
  [ "$got" = "$digest" ] || { echo "checksum mismatch: $rel"; exit 1; }
  cp "$rel" "$B/$rel"
done

# 3. the laptop's finished units (prediction csv.gz + json pairs only; skip-if-done on the GPU)
P=results/upgrades/fm_finetune_v2_preds
mkdir -p "$B/$P"
n=0
for js in $P/*.json; do
  csv="${js%.json}.csv.gz"
  [ -f "$csv" ] || continue
  cp "$js" "$csv" "$B/$P/"; n=$((n + 1))
done

# 4. manifest
{
  echo "gbkb fine-tuning GPU bundle, built $(date '+%F %T %Z') on $(hostname -s)"
  echo "source repo HEAD: $(git rev-parse HEAD 2>/dev/null || echo n/a) (working tree; see sha256 below)"
  echo "laptop units included (skipped on the GPU): $n"
  ls $P/*.json 2>/dev/null | xargs -n1 basename | sed 's/\.json$//' | sed 's/^/  /'
  echo "sha256:"
  (cd "$B" && find . -type f ! -name BUNDLE_MANIFEST.txt | sort | xargs shasum -a 256)
} > "$B/BUNDLE_MANIFEST.txt"

tar -C "$STAGE" -czf "$ROOT/external/fm_finetune/${NAME}_bundle.tar.gz" "$NAME"
ls -l "$ROOT/external/fm_finetune/${NAME}_bundle.tar.gz"
echo "bundle: external/fm_finetune/${NAME}_bundle.tar.gz ($n laptop units included)"
