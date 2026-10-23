# `fm_finetune` — how much of a *fine-tuned* FM score is base composition?

The frozen probe (`../fm_probe/`) shows a pretrained **representation** carries the negative-set
composition shortcut. It cannot say what the shortcut is worth on a reported score, because a
fine-tuned model is free to stop using it. This module answers that directly: the same model, the
same recipe, trained and evaluated on the released benchmark split versus the composition-cleaned
splits emitted by `../composition_clean/`.

> **Status (2026-09-26).** The v1 run described first (`run_fm_finetune.py`, `fm_finetune.csv`,
> `fm_finetune_summary.csv`) is **superseded**: its batcher fed nearly single-class mini-batches and
> it saved no predictions. Use the **v2** section below (`fm_finetune_v2_*`), which is the
> camera-ready answer to Reviewer 1, comment 2. The v1 numbers still quoted in older drafts
> (0.752 -> 0.514) sit next to their v2 re-run in `fm_finetune_v2_v1_vs_v2.csv`. The v1 files are kept
> unchanged for provenance.

## Arms (v1)

| arm | train on | evaluate on | reads |
|---|---|---|---|
| `original` | released split | released split | `data.load_dataset(<task>)` |
| `gc_match` | GC-matched | GC-matched | `cleaned_splits_v2/<task>_gcmatch_{train,test}.csv` |
| `comp_equalized` | composition-equalized | composition-equalized | `cleaned_splits_v2/<task>_comp_equalized_{train,test}.csv` |
| `original` → `X` | released split | arm `X`'s test split | the same fixed weights, a cleaner test set |

The last row is the tightest comparison: identical weights, so any AUROC change comes from the test
set alone. The `comp_equalized` row is the "what a clean benchmark would report" number.

## Protocol

`LongSafari/hyenadna-tiny-16k-seqlen-d128-hf` at the **same pinned revision the frozen probe uses**,
mean-pooled over nucleotide tokens (identical pooling to the frozen `mean` arm) with a linear head;
all weights trainable. 6 epochs, batch 32, AdamW `lr=6e-4, weight_decay=0.1`, 10% warmup then cosine
decay, grad-norm clip 1.0, class-weighted cross-entropy (the cleaned arms are imbalanced by
construction). Seeds 42/43/44. The epoch is selected by AUROC on a stratified **10% validation carve
of the training split** — the test split is never used for model selection. Metrics carry
1000-resample percentile bootstrap CIs with the test sequence as the resampling unit, matching
`run_stats.py`.

Absolute AUROC here is not tuned to match a published leaderboard entry; the claim is the
**relative** difference between arms trained under an identical recipe.

Torch is confined to this file, as in `../fm_probe/extract_embeddings.py`; the diagnostic pipeline
still imports zero torch.

## Run

```bash
make finetune                       # both datasets, all arms, 3 seeds
python src/upgrades/fm_finetune/run_fm_finetune.py --dataset human_enhancers_cohn
python src/upgrades/fm_finetune/run_fm_finetune.py --smoke     # ~2-min wiring check
python src/upgrades/fm_finetune/tests/test_sanity.py           # torch-free checks
```

## Outputs

- `results/upgrades/fm_finetune.csv` — one row per (train arm, eval arm, seed)
- `results/upgrades/fm_finetune_summary.csv` — seed-aggregated mean/SD/min/max
- `results/upgrades/fm_finetune_run.log` — per-epoch loss and validation AUROC

---

## v2 (camera-ready, Reviewer 1 comment 2): end-to-end fine-tuning of two FMs + controls

Reviewer 1 asked for full-parameter or LoRA fine-tuning of a mainstream genomic FM on
composition-aligned and unaligned data, to test whether *fine-tuned* models (not only frozen
embeddings + linear heads) still exploit the negative-set composition shortcut. v2 adds:

| piece | what it is |
|---|---|
| `random` arm (control) | negatives drawn uniformly at random (seeded, RandomState(42)) to **exactly** the `comp_equalized` arm's negative count in train and in test (cohn 399/228, nt_enhancers 343/36, drosophila 1265/217); no composition criterion. Same size and class imbalance as `comp_equalized`, original composition gap kept. Separates "shortcut removed" from "fewer / outnumbered negatives". |
| DNABERT-2-117M + LoRA | `zhihan1996/DNABERT-2-117M` @ `7bce263b` (the frozen-embedding revision), PyTorch attention (no triton on macOS), LoRA r=8/alpha=16/dropout 0.05 on every `Wqkv`, attention-output `dense`, GLU `gated_layers` and `wo` (48 modules, 1,474,560 LoRA params) + trainable linear head on the masked mean-pool (1,538) = **1,476,098 trainable of 118,544,642**. AdamW 3e-4, wd 0.01, 10% warmup + cosine, batch 32, <=4 epochs, early stop (patience 2) on validation AUROC. |
| HyenaDNA-tiny full FT | same recipe as `run_fm_finetune.py` (436,354 params, all trainable, 6 epochs, best epoch on validation AUROC) with one fix: the old batcher produced nearly single-class mini-batches (class-sorted lists + stable length sort); v2 permutes before the length sort. All HyenaDNA numbers are therefore re-generated. |
| datasets | `human_enhancers_cohn`, `nt_enhancers` (composition-unaligned) and `drosophila_enhancers_stark` (the paper's composition-aligned clean control; HyenaDNA only, its 2-3 kb sequences exceed DNABERT-2's 512-token context). |
| per-sequence predictions | every model scores every sequence of the ORIGINAL test split (each arm's test set is a subset of it), so any metric on any arm can be recomputed: `results/upgrades/fm_finetune_v2_preds/*.csv.gz` (+ `.json` run metadata). Best-epoch trainable weights: `external/fm_finetune/checkpoints/`. |
| imbalance-robust metrics | AUROC, PR-AUC (average precision) with its prevalence baseline, balanced accuracy and MCC at p=0.5, each with a 1000-resample CI; and all four on 20 class-balanced resamples (majority class subsampled to the minority count; draws paired across models). |
| pre-registered contrasts | see the docstring of `eval_ft_v2.py` (written before any v2 run finished): S = A - B shortcut worth, T = C - B test-imbalance control, U = E - D train-size control, R = (E-0.5)/(A-0.5) retention; rules 1-5. |

Validation carve, bootstrap convention (test sequence = resampling unit, 1000 resamples, seed 42)
and class-weighted CE are as in v1. Seeds 42/43/44; units run seed-major.

### Reproduce (v2)

```bash
PY=/Users/rikhinkavuru/Downloads/venv/bin/python        # torch-free env
FT=.venv-ft/bin/python                                   # torch 2.12 / transformers 5.9 / peft 0.21
# 1. materialize arms as index membership over the original splits (+ manifest, + sanity readout)
bibm/lowcpu.sh $PY src/upgrades/fm_finetune/prepare_splits_v2.py
bibm/lowcpu.sh $PY src/upgrades/fm_finetune/tests/test_ft_v2.py      # splits == loader, arms, metrics, batcher
HF_HUB_OFFLINE=1 nice -n 15 $FT src/upgrades/fm_finetune/tests/check_ckpt_grads.py   # ckpt is gradient-exact
# 2. train (sequential GPU queue, resumable; HyenaDNA under bibm/lowcpu.sh, DNABERT-2 under nice -n 15)
#    units trained up to 2026-09-24 08:53 came from run_queue_v2.sh; everything after the memory
#    incident runs from run_queue_v3.sh (fixed shapes, micro-batching, hard memory caps):
HF_HUB_OFFLINE=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2 bibm/lowcpu.sh $FT \
    src/upgrades/fm_finetune/tests/check_pad_invariance.py > results/upgrades/fm_finetune_v2_pad_invariance.log
nohup src/upgrades/fm_finetune/run_queue_v3.sh >> results/upgrades/fm_finetune_v2_run.log 2>&1 &
RUN_BENCH=1 ...run_queue_v3.sh   # also repeats the fixed-vs-old memory benches
bibm/lowcpu.sh $PY src/upgrades/fm_finetune/memfix_evidence.py      # tabulate them
#    single unit, e.g.:
#    bibm/lowcpu.sh $FT src/upgrades/fm_finetune/run_ft_v2.py --model hyenadna \
#        --dataset human_enhancers_cohn --arms random --seeds 42
# 3. score (torch-free), in this order; re-run the whole chain after more units finish (GPU hand-back).
#    Measured wall times on the M4 laptop under bibm/lowcpu.sh (2026-09-26) are in brackets
#    (the eval took 20 min on an efficiency core and 6 min once performance cores were allowed).
bibm/lowcpu.sh $PY src/upgrades/fm_finetune/verify_preds_ft_v2.py     # integrity of every unit [<1 min]
bibm/lowcpu.sh $PY src/upgrades/fm_finetune/eval_ft_v2.py              # PRE-REGISTERED scoring [6-20 min]
bibm/lowcpu.sh $PY src/upgrades/fm_finetune/summarize_ft_v2.py         # paper table, model contrast [~3 min]
bibm/lowcpu.sh $PY src/upgrades/fm_finetune/baselines_ft_v2.py         # POST HOC k-mer + frozen-FM refits [~33 min; skip-if-done]
bibm/lowcpu.sh $PY src/upgrades/fm_finetune/posthoc_ft_v2.py           # POST HOC leak-free nt + comparison [~21 min]
bibm/lowcpu.sh $PY src/upgrades/fm_finetune/sensitivity_ft_v2.py \
    > results/upgrades/fm_finetune_v2_sensitivity.log 2>&1              # POST HOC fix-round-1 sensitivity [~2.5 min]
bibm/lowcpu.sh $PY src/upgrades/fm_finetune/make_fig_ft_v2.py          # both figures [<1 min]
bibm/lowcpu.sh $PY src/upgrades/fm_finetune/write_interpretation_ft_v2.py
# 4. checks (torch-free, each < 1 min)
bibm/lowcpu.sh $PY src/upgrades/fm_finetune/tests/test_ft_v2.py        # splits, arms, metrics, batcher
bibm/lowcpu.sh $PY src/upgrades/fm_finetune/tests/test_posthoc_ft_v2.py   # leak-free set, pairing, labels, baseline repro
bibm/lowcpu.sh $PY src/upgrades/fm_finetune/tests/test_outputs_ft_v2.py   # every written number vs sklearn on the preds
bibm/lowcpu.sh $PY src/upgrades/fm_finetune/tests/test_sensitivity_ft_v2.py  # fix-round-1 outputs vs sklearn / sequence join
```

### Compute budget (measured, not estimated)

- HyenaDNA-tiny full FT on MPS: cohn ~11-15 min per original-arm unit, ~7-8 min per 399-negative
  arm; nt_enhancers ~5 min / ~2.5 min (`train_secs` in `fm_finetune_v2_runs.csv`).
- DNABERT-2 LoRA on MPS, batch 32 with activation checkpointing: **3.0 s/step** on cohn (steps 0-100
  of the first unit, 00:30:03 -> 00:35:02 in the run log) with the GPU saturated (IOAccelerator
  "Device Utilization" 95-100%), i.e. GPU-bound, not CPU-starved. That is ~29 min per epoch on the
  original arm (587 steps) and ~15 min on the 399-negative arms (305 steps), so ~4 h per seed (three
  arms, <=4 epochs, early stopping) and ~12 h for three seeds on cohn. This exceeded the original
  ~8 h budget, so the queue ran DNABERT-2 on cohn first (seeds 42 and 43, then seed 44 after the
  HyenaDNA drosophila control) and DNABERT-2 on nt_enhancers last. **All of it finished locally**:
  `run_queue_v3.sh` completed on 2026-09-25 at 02:35, and all 54 planned units exist. They are
  HyenaDNA x {cohn, nt_enhancers, drosophila} x {original, comp_equalized, random, gc_match} x seeds
  42-44 (36), plus DNABERT-2 LoRA x {cohn, nt_enhancers} x {original, comp_equalized, random} x seeds
  42-44 (18). `verify_preds_ft_v2.py` passes 54/54 (`fm_finetune_v2_preds_check.csv`).
- `tests/check_ckpt_grads.py` (MPS, 2026-09-24 00:02): checkpointed vs plain gradients identical
  (max relative difference 0.0; a different dropout seed changes them by 1.24, so the check is not
  vacuous); 98 trainable tensors = 48 LoRA A/B pairs + head, all non-zero
  (`results/upgrades/fm_finetune_v2_ckpt_check.log`).

### Memory incident (2026-09-24) and fix

What happened: the HyenaDNA fine-tune on `drosophila_enhancers_stark` (PID 62701, started 08:53 by
`run_queue_v2.sh`) grew to a **17.6 GB** footprint on the 16 GB laptop, and swap reached ~16 GB. The
process was stopped and the drosophila line was disabled in `run_queue_v2.sh`. When the queue was
restarted, the next unit (DNABERT-2 cohn s44, PID 76231) also grew without bound, by ~175 MB/min (3.72
to 4.43 GB in 5 min, `results/upgrades/fm_finetune_v2_memwatch_76231.log`), so that unit was stopped as well.

Root cause: the tensor shapes changed from batch to batch. drosophila has 535 distinct lengths
(236-3237 bp), and every batch was padded only to its own longest sequence. DNABERT-2 "unpads" each
batch into a `(total real tokens, 768)` tensor, and that token total is different for almost every
batch. The MPS backend compiles and caches graphs (plus their scratch memory) for each new shape. That
memory sits outside PyTorch's caching allocator, so the watermark cap does not bound it. The cohn and
nt_enhancers HyenaDNA units have only 1-2 lengths, so they never showed the problem.

Measured on identical mini-batches (`memfix_evidence.py` -> `results/upgrades/fm_finetune_v2_memfix_bench.csv`):

| bench | shapes seen | footprint, step 10 -> last | growth | s/step |
|---|---|---|---|---|
| DNABERT-2 cohn, old (unpadded), 80 steps | 13 widths | 3066 -> 3758 MB | +9.5 MB/step | 2.59 |
| DNABERT-2 cohn, `--pad-multiple 16`, 80 steps | 2 widths | 2898 -> 2890 MB | -0.1 MB/step | 2.28 |
| HyenaDNA drosophila, old padding, stopped at step 20 | 21 widths | 3102 -> 3716 MB (memguard abort at 3.5 GB cap) | +43.6 MB/step | 1.43 |
| HyenaDNA drosophila, `--pad-multiple 512`, 30 steps | 2 widths | 1901 -> 1936 MB | +2.6 MB/step (flat after step 20) | 0.80 |

Fixes. All of them are numerically no-ops, checked by `tests/check_pad_invariance.py` on CPU and MPS:
logits differ by at most 9e-8, and gradients by at most 1.4e-4 relative (criterion < 1e-3); log in
`results/upgrades/fm_finetune_v2_pad_invariance.log`.
- `--pad-multiple K` pads each batch up to a multiple of K tokens. For DNABERT-2 the encoder also keeps
  the pad rows instead of unpadding them (`set_dnabert_fixed_shapes`). Pad keys are still masked out of
  attention by the model's own -10000 bias, and the pooling is masked.
- `--micro-batch N` does exact gradient accumulation: each micro-batch adds its weighted loss *sum*
  divided by the whole batch's weight total, so the optimizer batch stays 32. drosophila needs
  `--micro-batch 4`: a 32 x 3.5 kb batch does not fit under the 2.4 GB MPS allocator cap, and
  micro-batch 8 hit that cap as a clean OOM error.
- Hard caps. `memguard.py` samples `phys_footprint` (the number in Activity Monitor's Memory column,
  which includes Metal buffers). The in-process cap `--mem-cap-mb` exits the job with code 86, and an
  external watcher at +10% sends SIGTERM and then SIGKILL. Caps in `run_queue_v3.sh`: HyenaDNA 3.5 GB,
  DNABERT-2 4.0 GB. The `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2` allocator cap stays on (the v2 HyenaDNA
  units that ran before the fix did not have it; see `env` in their metadata).

`run_queue_v3.sh` now runs the remaining units with these settings. Every unit records its settings
and its peak footprint in its metadata JSON. The units trained before the fix used no padding
multiple, no micro-batching and the recipe eval batch. Their numbers are unaffected, because only
memory differed.

### Post hoc analyses (added 2026-09-26, after all 54 units had been scored)

None of these changes the pre-registered outputs of `eval_ft_v2.py`. They write separate
`*_posthoc_*` / `*_compare*` / `*_baselines*` files, and every text built from them says "post hoc".

- **Baselines on the identical arms** (`baselines_ft_v2.py`). The paper's two non-fine-tuned model
  families are re-fit on exactly the fine-tuning train arms and score every released test sequence:
  the k-mer LightGBM at the validation-selected k (cohn 5, nt_enhancers 4, drosophila 4) and
  LightGBM heads on the released frozen embeddings (HyenaDNA mean / [SEP], DNABERT-2 mean / [CLS];
  sha256-checked against `fm_embeddings_manifest.csv`). That is 52 units under
  `external/fm_finetune/baseline_preds/`. The refits reproduce **66/66** published numbers to 1e-3:
  the k-mer A0 and equalized-split AUROC of `psm_sensitivity_decision.csv`, and every frozen-probe
  `orig_full` / `orig_on_clean` / `clean_on_clean` AUROC of `fm_paired*.csv`. See
  `fm_finetune_v2_baselines_repro.csv`. They are single-seed (LightGBM is deterministic given the data),
  as in the paper, and are fit on the full train arm (nothing is early-stopped).
- **P1, leak-free nt_enhancers** (`posthoc_ft_v2.py`). 100 of the 400 released test sequences are
  verbatim training sequences, all of them positives (`nt_enhancers_duplication.csv`; re-derived by
  exact match into `fm_finetune_v2_leakfree_dups.csv`; 107 training rows equal one of them). The
  derived dataset `nt_enhancers_leakfree` drops those 100 rows from every eval arm, leaving 300
  sequences (100+/200-; the equalized and random arms keep 100+/36-). It re-runs the identical
  pre-registered computations: metrics, summary, S/T/U/R and rules 1-3, with the bootstrap over the
  300 sequences. Training is unchanged.
- **P2/P3, one comparison over three model families** (`fm_finetune_v2_compare.csv`). For each
  model x dataset it gives A-H, S/T/U/R, the share of above-chance AUROC the same weights lose,
  `(A-B)/(A-0.5)`, and the random-negative control `(A-C)/(A-0.5)`. `fm_finetune_v2_compare_paired.csv`
  gives the fine-tuned-minus-baseline differences of the share lost and of S. These are paired: every
  model uses the same RandomState(42) resamples of the same test sequences. 95% percentile CIs,
  1000 resamples. Descriptive; no decision rule is attached.
- **Figures**. `figures/fm_finetune_v2.{pdf,png}` shows AUROC by arm for the fine-tuned models, with
  nt_enhancers on its leak-free test. `figures/fm_finetune_v2_compare.{pdf,png}` is the candidate
  paper figure: the share lost per model family on cohn, leak-free nt_enhancers and the drosophila
  control, for the k-mer LightGBM, the paper's two frozen-probe cells and the fine-tuned FMs.
- **Checks**. `tests/test_posthoc_ft_v2.py` covers the leak-free set, paired resampling, labels and
  baseline reproduction. `tests/test_outputs_ft_v2.py` recomputes every written metric with sklearn
  from the stored predictions and re-derives every contrast and table entry from the metric rows.

### Fix round 1 (2026-09-26): post hoc sensitivity analyses and wording

Two independent verifiers reviewed the v2 report. Their corrections were applied in
`sensitivity_ft_v2.py` (new; post hoc, torch-free, ~2.5 min), `write_interpretation_ft_v2.py`
(data-driven wording) and `make_fig_ft_v2.py` (labels only). No pre-registered output changed, and
neither did any output of `posthoc_ft_v2.py`: a re-run of the current posthoc code by the code verifier
reproduced all 12 post hoc CSVs byte for byte (`fm_finetune_v2_rerun_check.log`). The verifiers had
computed scratch versions of Q2-Q5 before the script was written, so those numbers were seen first.
The script uses their definitions unchanged, extends them to every model and dataset, and attaches no
decision rule.

- **Q1 unit diagnostics** (`fm_finetune_v2_unit_diagnostics.csv`). The three HyenaDNA cohn
  `comp_equalized` units are **degenerate**: every test probability is >= 0.5, the probability SD is
  <= 0.0064, and the maximum validation AUROC is <= 0.556. Their D = 0.522 is an optimization collapse
  of the fixed recipe on the 26:1 equalized train arm. It is not evidence that the arm is unlearnable:
  on the same arm, five of the six other models reach 0.61-0.70 (DNABERT-2 LoRA 0.703, the k-mer model
  0.613; only frozen HyenaDNA [SEP] stays near chance at 0.524; `fm_finetune_v2_compare.csv`). Also in this
  file: validation-carve leakage. About 17% of nt_enhancers validation rows have an exact copy among
  the fitting rows (the dataset has 1,876 training rows in duplicate groups), so nt validation AUROC
  overstates test AUROC.
- **Q2 GC-matched test effect** (`fm_finetune_v2_gc_paired.csv`). G - A for every model, plus
  fine-tuned minus frozen / k-mer paired differences. On cohn, DNABERT-2 LoRA's drop is slightly
  *smaller* than its frozen probe's; fine-tuned HyenaDNA's is larger than the frozen [SEP] probe's.
- **Q3 latent positives (ENCODE cCRE v4)** (`fm_finetune_v2_ccre_enrichment.csv`,
  `..._ccre_sensitivity.csv`, `..._ccre_contrasts.csv`). The equalizer keeps cCRE-overlapping
  negatives more often than the random control does (cohn test 50% vs 32%). The script re-scores
  every model with the cCRE- (or ELS-) overlapping test negatives removed; the drop survives but
  shrinks. The annotation is read-only from `external/psm_sensitivity/`, and its row alignment to
  the split files is asserted.
- **Q4 random-draw reference for B** (`fm_finetune_v2_randdraw.csv`): 1000 random draws of as many
  negatives as the equalized arm holds.
- **Q5 within-class score-composition R^2** (`fm_finetune_v2_score_composition_withinclass{,_summary}.csv`).
- **Q6 Bonferroni CIs** over the 5 pre-registered cells (`fm_finetune_v2_multiplicity.csv`).
- Wording. The share lost is tagged post hoc/descriptive. The rule-1 clause that fired is named:
  for HyenaDNA it fired only through `B < 0.6`, and B's CI excludes 0.5. Leak-free lines carry n and
  no verdicts. drosophila is called a "GC-balanced control": rule 5 was not met, and the dataset
  keeps a dinucleotide gap. psm_sensitivity is compared for every dataset. For nt_enhancers the
  retrain side is NOT SUPPORTED there, so the nt claim rests on B vs C.
- Timing, harmonized. The `eval_ft_v2.py` docstring labels rule 5 "22:30". The agent transcript
  shows the rule-5 text written at 22:26:57 EDT on 2026-09-23. Either way it precedes the first
  finished unit (22:33:24) and the first drosophila unit (08:53 the next day). The pre-registered
  docstring is left unchanged.

### Deferred GPU units

DNABERT-2 **full-parameter** fine-tuning, NT-v2-500M LoRA and the corresponding extra datasets do not
fit the laptop, so they are packaged for one CUDA GPU in `gpu_deferred/` (see its README;
bundle `external/fm_finetune/gbkb_ft_gpu_bundle.tar.gz`, rebuilt with `gpu_deferred/make_bundle.sh`
so that it carries all 54 local prediction units, which the GPU then skips). Ranks 1 (HyenaDNA
drosophila) and 4 (DNABERT-2 LoRA nt_enhancers) of that list have been **completed locally**; ranks 2,
3, 5, 6 and 7 are pending. After the hand-back (`rsync --ignore-existing` into
`fm_finetune_v2_preds/`), re-run step 3 above unchanged. Every script already carries labels for
`dnabert2full` and `ntv2_500m`.

### Outputs (v2)

- `results/upgrades/fm_finetune_v2_split_manifest.csv` — arm sizes, index hashes, composition-only LR and GC AUROC per arm
- `results/upgrades/fm_finetune_v2_preds/` — per-sequence val/test probabilities + run metadata (params, epochs, memory, env)
- `results/upgrades/fm_finetune_v2_runs.csv` — one row per trained model (trainable/total params, best epoch, val AUROC by epoch, time)
- `results/upgrades/fm_finetune_v2_metrics.csv` — model x dataset x train arm x seed x eval arm, all metrics + CIs + balanced-resample metrics
- `results/upgrades/fm_finetune_v2_summary.csv` — seed-aggregated (mean, SD, seed-mean bootstrap CI)
- `results/upgrades/fm_finetune_v2_contrasts.csv`, `..._verdicts.csv` — pre-registered contrasts and rule outcomes
- `results/upgrades/fm_finetune_v2_score_composition{,_summary}.csv` — how much of each model's score is a linear function of 20 composition features
- `results/upgrades/fm_finetune_v2_paper_table.csv` — one row per model x dataset: A-H AUROCs (mean, seed SD, CI), balanced-resample AUROC/bal.acc/MCC, AP vs prevalence, S/T/U/R with CIs, post hoc share of above-chance AUROC lost `(A-B)/(A-0.5)` with CI, rule verdicts, trainable-parameter counts (`summarize_ft_v2.py`)
- `results/upgrades/fm_finetune_v2_model_contrast.csv` — HyenaDNA minus DNABERT-2 shortcut worth and share lost on the same dataset, paired bootstrap CI (post hoc, descriptive)
- `results/upgrades/fm_finetune_v2_v1_vs_v2.csv` — the superseded v1 numbers (`fm_finetune_summary.csv`) next to the v2 re-run
- `results/upgrades/fm_finetune_v2_ckpt_check.log` — gradient-checkpointing exactness check on MPS
- `results/upgrades/fm_finetune_v2_interpretation.txt` — headline numbers and caveats, generated from the CSVs (`write_interpretation_ft_v2.py`)
- `results/upgrades/figures/fm_finetune_v2.{pdf,png}` — single-column summary figure (fine-tuned models, AUROC by arm)
- `results/upgrades/fm_finetune_v2_run.log` — training log
- `results/upgrades/fm_finetune_v2_preds_check.csv` — per-unit integrity check (`verify_preds_ft_v2.py`)
- `results/upgrades/fm_finetune_v2_eval.log`, `..._summarize.log` — console output of the scoring runs
- `results/upgrades/fm_finetune_v2_rerun_check.log` — determinism check: re-running `eval_ft_v2.py` + `summarize_ft_v2.py` reproduces all ten pre-registered CSVs byte for byte
- `results/upgrades/fm_finetune_v2_tests.log` — output of the four torch-free test suites (step 4)
- post hoc, fix round 1 (`sensitivity_ft_v2.py`, log `fm_finetune_v2_sensitivity.log`): `..._unit_diagnostics.csv`,
  `..._gc_paired.csv`, `..._ccre_enrichment.csv`, `..._ccre_sensitivity.csv`, `..._ccre_contrasts.csv`,
  `..._randdraw.csv`, `..._score_composition_withinclass{,_summary}.csv`, `..._multiplicity.csv`
- post hoc: `results/upgrades/fm_finetune_v2_baselines_repro.csv` (+ `_baselines.log`), `..._leakfree_dups.csv`,
  `..._posthoc_{runs,metrics,summary,contrasts,verdicts,paper_table,score_composition,score_composition_summary}.csv`
  (baselines on all three datasets + everything on `nt_enhancers_leakfree`), `..._compare.csv`,
  `..._compare_paired.csv`, `..._posthoc.log`, `figures/fm_finetune_v2_compare.{pdf,png}`
