# `gpu_deferred`: fine-tuning units that need a CUDA GPU

Reviewer 1, comment 2 asks for end-to-end (full-parameter or LoRA) fine-tuning of mainstream genomic
FMs on composition-aligned and composition-unaligned datasets. The laptop (M4, 16 GB, shared with other
experiment agents) runs HyenaDNA-tiny (full FT) and DNABERT-2-117M (LoRA) through `../run_ft_v2.py`.
The units below either do not fit in its memory or would take days on MPS, so they are packaged here
as one self-contained bundle to run on a single CUDA GPU. Every unit is **pending a GPU run** until
its prediction file comes back.

> **Status 2026-09-26.** The laptop queue finished all 54 local units on 2026-09-25 at 02:35. That
> completes ranks **1** (HyenaDNA drosophila, 12/12 units) and **4** (DNABERT-2 LoRA nt_enhancers,
> 9/9) below, which were packaged only as fallbacks. The rebuilt bundle ships those prediction
> files, so `jobs.sh all` skips them. **Pending on a GPU:** ranks 2, 3, 5, 6 and 7 (45 units). The
> presentation scripts (`summarize_ft_v2.py`, `write_interpretation_ft_v2.py`, `make_fig_ft_v2.py`,
> `posthoc_ft_v2.py`) already carry labels for `dnabert2full` and `ntv2_500m`. After the hand-back, run
> the full scoring chain in `../README.md` (step 3) with no code change.

The runner (`run_gpu.py`) imports `../run_ft_v2.py` read-only and calls its `load_split` /
`run_unit` / `train_unit` / `predict`. Everything below is therefore identical to the laptop runs:
the split files (SHA-256 checked), the per-seed stratified 10% validation carve, the mixed-class
batcher, class-weighted CE, 10% warmup + cosine, clipping, best-epoch selection / early stopping,
seeds 42/43/44 in seed-major order, skip-if-done resumption, and the per-sequence prediction format.
What it adds:
device auto-selection (cuda > mps > cpu), two new pre-registered model recipes (`extra_models.py`),
forced PyTorch attention for DNABERT-2 (triton is blocked; on the laptop it is simply absent), and
GPU telemetry in every unit's JSON.

## Deferred jobs, ranked by value to the paper

| rank | job (`jobs.sh N`) | model, fine-tuning | dataset | arms x seeds | why it matters | why not on the laptop |
|---|---|---|---|---|---|---|
| 1 | HyenaDNA drosophila | HyenaDNA-tiny-16k-d128 @ `d79fa37e`, **full** (436,354 params) | drosophila_enhancers_stark (composition-ALIGNED control, pre-registered rule 5) | orig / comp_eq / random x 42-44, then gc_match x 42-44 (12 units) | the reviewer asked for aligned **and** unaligned datasets | **FALLBACK.** The laptop queue (`../run_queue_v3.sh`, fixed-shape padding + 4 GB cap) is running it; every unit it finishes is skipped here. Unpadded, it grew to 17.6 GB on MPS. |
| 2 | DNABERT-2 full FT, cohn | DNABERT-2-117M @ `7bce263b`, **full-parameter** (117,070,082 trainable of 117,070,082) | human_enhancers_cohn | orig / comp_eq / random x 42-44 (9) | answers "full-parameter" on a mainstream 117M model, next to the existing LoRA run | full-model AdamW state is about 2 GB on top of weights and activations: over the laptop's 2.5 GB MPS cap; about 3 s/step |
| 3 | NT-v2 500M LoRA, cohn | Nucleotide Transformer v2 500M multi-species @ `06615c16`, LoRA r=8 (5,228,546 trainable of 498,312,659) | human_enhancers_cohn | orig / comp_eq / random x 42-44 (9) | a larger (0.5B) mainstream FM, from a third model family | 2 GB of fp32 weights alone; days on MPS |
| 4 | DNABERT-2 LoRA, nt_enhancers | DNABERT-2-117M LoRA (1,476,098 of 118,544,642) | nt_enhancers | orig / comp_eq / random x 42-44 (9) | second unaligned dataset for the LoRA model | **FALLBACK.** It is in the laptop queue; finished units are skipped |
| 5 | NT-v2 500M, drosophila | as rank 3 | drosophila_enhancers_stark | orig / comp_eq / random x 42-44 (9) | rule 5 (aligned control) for a second FM; the 2048-token context takes 2-3 kb unmodified | as rank 3 |
| 6 | NT-v2 500M, nt_enhancers | as rank 3 | nt_enhancers | 9 units | second unaligned dataset for NT-v2 | as rank 3 |
| 7 | DNABERT-2 full FT, nt_enhancers | as rank 2 | nt_enhancers | 9 units | second unaligned dataset for full FT | as rank 2 |

Not packaged: Caduceus (its Mamba kernels need `mamba-ssm` built against the GPU's CUDA toolchain,
which cannot be pinned or tested from this laptop), and DNABERT-2 on drosophila (sequences of up to
3.2 kb exceed its 512-token training context; NT-v2 covers the aligned control instead).

### Pre-registered recipes of the two new models (fixed 2026-09-24, before any unit was trained)

- **dnabert2full**: every DNABERT-2 weight plus the masked-mean-pool linear head is trainable.
  AdamW lr **3e-5** (the DNABERT-2 authors' GUE fine-tuning rate), wd 0.01, 10% warmup + cosine,
  clip 1.0, batch 32, class-weighted CE, at most 4 epochs, early stopping with patience 2 on
  validation AUROC, per-layer activation checkpointing (saves memory only). Only the trainable set
  and the learning rate differ from the LoRA recipe.
- **ntv2_500m**: LoRA r=8 / alpha=16 / dropout 0.05 on query, key, value, attention-output dense and
  both SwiGLU FFN projections of all 29 layers (174 modules). Backbone frozen; linear head on the
  attention-mask mean-pool, pooled as for DNABERT-2. Optimizer, schedule, batch, epochs and early
  stopping are **identical** to the DNABERT-2 LoRA recipe (lr 3e-4), with no per-model tuning. The
  6-mer tokenizer allows 2048 tokens, so no sequence is truncated (the longest is about 540 tokens).

The analysis is the one pre-registered in `../eval_ft_v2.py`. The same rules 1-5 (S = A-B shortcut
worth, T test-imbalance control, U train-size control, R retention, and the aligned-control rule on
drosophila) apply to every model automatically.

## Environment (exact)

- Python **3.14** (the laptop runs 3.14.5; 3.12 also resolves).
- torch **2.12.0**. Install it first, from the CUDA index that matches the driver:
  `pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu126` (driver >= 525;
  A100/L4/H100), or `.../whl/cu130` (driver >= 580; also what plain PyPI gives).
- Then `pip install -r src/upgrades/fm_finetune/gpu_deferred/requirements-gpu.txt`, which is the laptop's
  `.venv-ft` frozen: transformers 5.9.0, peft 0.21.0, tokenizers 0.22.2, huggingface-hub 1.26.0,
  safetensors 0.8.0, accelerate 1.15.0, einops 0.8.2, numpy 2.5.1, pandas 3.0.5, scikit-learn 1.9.0,
  and all transitive dependencies. It was resolved with `uv pip compile` for
  x86_64-manylinux_2_28 on Python 3.14 and on 3.12.
- Model revisions (pinned in code): HyenaDNA `LongSafari/hyenadna-tiny-16k-seqlen-d128-hf@d79fa37e2cd62dd338103c630f95be8f90812d46`,
  DNABERT-2 `zhihan1996/DNABERT-2-117M@7bce263b15377fc15361f52cfab88f8b586abda0`,
  NT-v2 `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species@06615c1660c892fc199840c18123f8385b3542a8`
  (the laptop's wiring test used `...-50m-multi-species@81b29e5786726d891dbf929404ef20adca5b36f1`, whose modeling code is byte-identical).
- DNABERT-2 and NT-v2 download into `<bundle>/external/fm_finetune/hf_cache`, HyenaDNA into the
  default HF cache (`$HF_HOME`). If compute nodes have no internet, run `jobs.sh check`
  (`--prefetch`) on a login node that shares the filesystem, then set `HF_HUB_OFFLINE=1`.
- NT-v2's remote code was written for transformers 4.32. `extra_models.py` restores the two helpers and two
  config defaults that 5.x removed, as exact copies of the 4.x behaviour. `tests/check_extra_models.py`
  shows that NT-v2 under 5.9 + these shims gives the **same** hidden states and masked-LM logits
  (max |diff| = 0.0) as transformers 4.57.6 running the unmodified remote code on 16 real, padded
  sequences.

## Getting the bundle onto the GPU machine and running it

```bash
# laptop: rebuild right before copying, so it carries the newest local predictions (skip-if-done)
bash src/upgrades/fm_finetune/gpu_deferred/make_bundle.sh     # -> external/fm_finetune/gbkb_ft_gpu_bundle.tar.gz
scp external/fm_finetune/gbkb_ft_gpu_bundle.tar.gz gpu-host:
# GPU host
tar xzf gbkb_ft_gpu_bundle.tar.gz && cd gbkb_ft_gpu
python3.14 -m venv .venv && . .venv/bin/activate
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r src/upgrades/fm_finetune/gpu_deferred/requirements-gpu.txt
bash src/upgrades/fm_finetune/gpu_deferred/jobs.sh check    # versions, GPU, model download, split sha256
bash src/upgrades/fm_finetune/gpu_deferred/jobs.sh bench    # measured s/step, peak VRAM, time per unit/job
nohup bash src/upgrades/fm_finetune/gpu_deferred/jobs.sh all > jobs.out 2>&1 &   # ranks 1..7, resumable
# or a subset, e.g.:  bash src/upgrades/fm_finetune/gpu_deferred/jobs.sh 2 3
```

One unit by hand:
`python src/upgrades/fm_finetune/gpu_deferred/run_gpu.py --model dnabert2full --dataset human_enhancers_cohn --arms original,comp_equalized,random --seeds 42,43,44`.
Precision is plain fp32 (TF32 off), as on the laptop. `--tf32` speeds up Ampere+ GPUs about 2-3x. If it is
used, use it for **every** unit of that model (the flag is recorded in each JSON).

### Expected runtime and VRAM on one GPU (estimates; `jobs.sh bench` prints measured values)

These are FLOP estimates at fp32, calibrated against the laptop. DNABERT-2 LoRA on cohn measured
3.0 s/step on the M4 GPU, which is about 0.8 TFLOP/s effective for about 2.5 TFLOP/step. They are
upper bounds with no early stopping. An A100 at plain fp32 is assumed to sustain about 8-10 TFLOP/s
and an L4 about 5-8.

| job | est. step time A100 / L4 | est. wall time, all units, A100 / L4 | est. peak VRAM |
|---|---|---|---|
| 1 HyenaDNA drosophila (12 units) | 0.05-0.2 s / 0.1-0.3 s | 0.5-1 h / 1-2 h | 4-6 GB |
| 2 DNABERT-2 full FT, cohn (9) | 0.25-0.35 s / 0.35-0.5 s | 1-1.6 h / 1.5-2.5 h | 4-5 GB |
| 3 NT-v2 500M LoRA, cohn (9) | 0.8-1.0 s / 1.0-1.5 s | 3.5-5 h / 5-7 h | 4-5 GB |
| 4 DNABERT-2 LoRA, nt_enhancers (9) | about 0.15 s / 0.2 s | 0.3-0.5 h / 0.5-1 h | 2-3 GB |
| 5 NT-v2 500M, drosophila (9) | 3.5-5 s / 5-8 s | 4-6 h / 6-9 h | 8-12 GB |
| 6 NT-v2 500M, nt_enhancers (9) | 0.4-0.6 s / 0.6-0.9 s | 1-1.5 h / 1.5-2.5 h | 3-4 GB |
| 7 DNABERT-2 full FT, nt_enhancers (9) | 0.15-0.2 s / 0.2-0.3 s | 0.5-1 h / 1-1.5 h | 3-4 GB |

All seven jobs: about 11-16 h on one A100 at fp32, about 17-26 h on an L4, and roughly half that with `--tf32`.
Any 24 GB GPU suffices. Jobs 1-3 carry most of the value and take about 5-8 h on an A100.

## Hand-back: how the results reach the paper

```bash
# GPU host -> laptop; copy only NEW units, never overwrite the laptop's own
rsync -av --ignore-existing gpu-host:gbkb_ft_gpu/results/upgrades/fm_finetune_v2_preds/ \
      results/upgrades/fm_finetune_v2_preds/
rsync -av gpu-host:gbkb_ft_gpu/results/upgrades/fm_finetune_gpu_run.log results/upgrades/
# laptop, torch-free env, NO code change: the full scoring chain of ../README.md, step 3
PY=/Users/rikhinkavuru/Downloads/venv/bin/python
for s in verify_preds_ft_v2 eval_ft_v2 summarize_ft_v2 baselines_ft_v2 posthoc_ft_v2 \
         make_fig_ft_v2 write_interpretation_ft_v2 tests/test_posthoc_ft_v2 tests/test_outputs_ft_v2; do
  bibm/lowcpu.sh $PY src/upgrades/fm_finetune/$s.py || break
done
```

`eval_ft_v2.py` reads every `<model>__<dataset>__<arm>__s<seed>.{csv.gz,json}` pair in the prediction
directory and groups them by the JSON's `model` / `dataset` / `train_arm` / `seed`. A new model
(`dnabert2full`, `ntv2_500m`) therefore gets its own rows in `fm_finetune_v2_{runs,metrics,summary,
contrasts,verdicts,score_composition}.csv`. The drosophila units feed rule 5 against every
unaligned dataset of the same model. This was verified on the laptop: `smoke_local.sh` ran
`eval_ft_v2.py` unchanged on the smoke predictions of hyenadna, dnabert2, dnabert2full and ntv2_50m
(see below). The presentation scripts (`summarize_ft_v2.py`, `write_interpretation_ft_v2.py`,
`make_fig_ft_v2.py`, `posthoc_ft_v2.py`) list the models to display explicitly. Since 2026-09-26
they carry label entries for `dnabert2full` / `ntv2_500m`, so the new models appear in the paper
table, both figures, the comparison table and the interpretation without any edit.

## Local verification (laptop CPU, < ~1 GB, never MPS)

- `tests/check_extra_models.py` (`external/fm_finetune/gpu_smoke/check_extra_models.log`) checks:
  device selection; that blocking triton works; NT-v2-50M loads all 186 checkpoint tensors and wraps
  72 LoRA modules; LoRA is a no-op at init; the NT-v2 forward under transformers 5.9 equals 4.57.6
  (max |diff| 0.0); the pretrained weights are in use (input reconstruction 0.996 vs 0.000 at random
  init); and the NT-v2-**500M** checkpoint at the pinned revision matches the model built from its
  config (all 441 `esm.*` tensors by name and shape, read from the safetensors header with an HTTP
  range request). Peak footprint 967 MB.
- `tests/ntv2_tf4_reference.py` writes the transformers-4.57 reference (throwaway venv
  `external/fm_finetune/venv-tf4`).
- `smoke_local.sh` (2026-09-24, **passed**, `results/upgrades/fm_finetune_gpu_deferred_smoke.log`)
  runs every job type end to end through `run_gpu.py` on CPU: 3 steps, 16 train sequences per class
  per arm, sequences cropped to 256 or 64 bp, DNABERT-2 cut to 2 layers. That is 13 units: hyenadna x
  drosophila (4 arms), dnabert2 x nt_enhancers (3), dnabert2full x cohn (3), ntv2_50m x cohn (3).
  Each unit predicts the whole original test split. `eval_ft_v2.py`, unchanged, ingested all 13
  runs (52 metric rows, 57 contrast rows, verdicts for all 4 model x dataset pairs;
  `external/fm_finetune/gpu_smoke/eval/`). Peak footprint per process: 0.73-0.86 GB (HyenaDNA,
  NT-v2-50M) and 0.95-1.05 GB (DNABERT-2, which must hold the 117M checkpoint while loading; hard
  cap 1.1 GB). The smoke AUROCs (3 steps) mean nothing. Before truncation, the full models report
  117,070,082 trainable params (dnabert2full) and 1,476,098 of 118,544,642 (dnabert2 LoRA). The
  first figure is consistent with the laptop's LoRA JSONs: 118,544,642 total minus 1,474,560 LoRA
  parameters (1,476,098 trainable minus the 1,538-parameter head).
- `make_bundle.sh` was followed by a self-test (`results/upgrades/fm_finetune_gpu_deferred_bundle_selftest.log`,
  **passed**): the tarball unpacked in a scratch directory (1) skips a laptop unit shipped inside it,
  (2) runs an NT-v2 unit from its own bundled code and splits, and (3) refuses a split file with one
  byte appended.
- 2026-09-26 rebuild (same log, appended). The bundle is 11.0 MB, sha256 `7e6707d9…d914`, and ships
  **54** laptop units. The check found all 125 manifest hashes OK and the bundled code identical to
  the working tree. It then ran skip-if-done on two of the newly shipped units (HyenaDNA drosophila
  gc_match s44, DNABERT-2 nt_enhancers random s44): both were skipped and nothing was trained.

## How the paper should read, with and without the GPU units

The numbers come from `fm_finetune_v2_paper_table.csv` / `_contrasts.csv` / `_verdicts.csv` after
`eval_ft_v2.py` is re-run. Rules 1-3 decide the wording: rule 1 means "rides the shortcut", rule 2
means "not a test-size artifact", rule 3 means "caused by composition removal".

- **Without them** (laptop only): "We fine-tuned two genomic FMs end to end, HyenaDNA-tiny with all
  parameters and DNABERT-2-117M with LoRA, on the released split, on composition-equalized negatives,
  and on a size- and imbalance-matched random-negative control ... [A, B, C, D, E and S per model x
  dataset]; HyenaDNA also on the composition-aligned drosophila control (rule 5)." Limitation
  sentence: "Full-parameter fine-tuning of a >100M-parameter model and a 0.5B model were not run
  (compute)." It is honest but weaker on the reviewer's "full-parameter" wording.
- **With ranks 2-3**: the sentence becomes "... and, on human_enhancers_cohn, DNABERT-2 with **all
  117M parameters** trained and Nucleotide Transformer v2 (500M, LoRA) ...". The limitation sentence
  is dropped. Report S (shortcut worth) and B's CI for every model x dataset in one table row each,
  and state which rules hold per model. Pre-committed reading: if rule 1 fails for a model (B well
  above chance), write "fine-tuning reduces but does not remove reliance on composition (S = ...)".
  Do not write "still exploits". If rules 1-2 hold, write "the shortcut survives full-parameter
  fine-tuning / 0.5B scale".
- **With rank 5**: rule 5 (aligned control) is reported for two FMs instead of one.

## Files

`run_gpu.py` (runner), `extra_models.py` (new wrappers + recipes + shims), `jobs.sh` (ranked jobs),
`requirements-gpu.txt`, `SHA256SUMS` (split files, verified by `tests/test_ft_v2.py` against
`data.load_dataset()` and the manifest hashes on 2026-09-24), `make_bundle.sh`, `smoke_local.sh`,
`tests/`.
