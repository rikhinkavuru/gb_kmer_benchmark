# Reproducibility supplement

Companion to *Three Routes to Failure* (PSB 2027). The values below are the exact
configuration used; the code in this repository is the ground truth.

## Environment
- Python 3, **CPU-only** (the diagnostic pipeline imports no PyTorch/TensorFlow).
- Dependencies / pinned versions: `requirements.txt` (core), `requirements-fm.txt`
  (isolated FM embedding extraction), `requirements-coords.txt` (coordinate analysis),
  `requirements-upgrades.txt`.
- Single fixed seed everywhere: **42**.

## Featurization (`featurize.py`)
- $k$-mer spectra for **k ∈ {3, 4, 5, 6}** over the fixed, data-independent vocabulary of all
  $4^k$ ACGT $k$-mers, L1-normalized to relative frequencies (sparse CSR). Non-ACGT $k$-mers ignored.
- Per dataset, $k$ is chosen by highest **validation** MCC (stratified 80/20 carve of the
  training set); the test split is never used for model selection.

## Models (`models.py`)
- **LightGBM** (main): `n_estimators=300, learning_rate=0.05, num_leaves=63,
  min_child_samples=20, subsample=1.0, colsample_bytree=1.0, reg_lambda=0.0, n_jobs=-1`,
  `random_state=42`; run in **deterministic mode** (`deterministic=True, force_row_wise=True,
  verbose=-1`) so a fixed seed + fixed config reproduces results bit-for-bit.
- **Logistic regression** (linear floor): `Normalizer(norm="l2")` →
  `LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, n_jobs=-1)`, `random_state=42`.
- Metrics: accuracy, MCC, AUROC (one-vs-rest macro for multiclass); MCC emphasized.

## Motif scoring (`motif_jaspar.py`, `motif_match.py`)
- JASPAR **2024 CORE** PWMs via pyJASPAR (offline SQLite). Count → log-odds PSSM with
  **pseudocount = 0.5** and **uniform background = 0.25**:
  `PPM = (count + 0.5) / (Σ_b count + 4·0.5)`, `PSSM = log2(PPM / 0.25)` (bits).
- Sequence scoring: best absolute log-odds over all gapless windows on **both strands**,
  reported as **bits-per-position** (max S / k).
- Motif-hit thresholds: **TATA (TBP) = 1.3 bits/pos** (specificity-controlled, since TBP is a
  low-information AT-rich motif), **GC-box (Sp1/KLF) = 1.0 bits/pos**. The Route-3 conclusion is
  robust across a 0.8–1.5 bits/pos sweep.

## Statistics (`run_stats.py`)
- **Bootstrap = 1000** percentile resamples; resampling unit = the individual test sequence
  (within-class, with replacement); 95% CIs throughout.
- **Permutation = 1000** label shuffles; two-sided p = fraction with |AUROC − 0.5| ≥ observed.

## Contamination cleaning (`run_cleaning.py`, `src/upgrades/composition_clean/run_composition_clean.py`)
- **TATA-flag**: remove the *excess* negatives carrying a high-confidence TBP hit (≥ 1.3 bits/pos),
  equal to the neg-minus-pos TATA-box-rate gap. Positives left identical.
- **GC-matched**: subsample negatives to the positive-class GC histogram — **25 bins** (`--gc-bins 25`).
- **Composition-equalized**: one LR propensity score P(positive | composition) fit on a 20-dimensional
  composition signature (4 mononucleotide + 16 dinucleotide L1-normalized frequencies); negatives
  resampled within each split to match the positives' propensity histogram — **20 bins** (`--n-bins 20`),
  Rosenbaum–Rubin; balance verified by per-feature standardized mean difference.
- **Dinucleotide-shuffled negatives**: dinucleotide-preserving (Eulerian-path) shuffle of the
  positives (`src/upgrades/shuffled_neg/`), removing the mono/dinucleotide composition gap by construction.

## Foundation-model probes (`src/upgrades/fm_probe/`, `src/upgrades/fm_finetune/`)
- Model (both probes): `LongSafari/hyenadna-tiny-16k-seqlen-d128-hf`, revision
  `d79fa37e2cd62dd338103c630f95be8f90812d46` (pinned); second architecture
  `zhihan1996/DNABERT-2-117M`. Right padding is forced for HyenaDNA (it is causal).
- **Frozen**: CPU forward passes only, mean-pool over nucleotide tokens plus the sequence-summary
  token ([SEP] / [CLS]); LR and LightGBM heads; paired `test_effect` with a nested 1000-resample
  bootstrap. Embeddings are released as a cache, so the frozen result reproduces with no torch.
- **Fine-tuned** (`src/upgrades/fm_finetune/run_ft_v2.py`, `RECIPES`):
  - *HyenaDNA*: full fine-tune (all 0.44M weights trainable) + linear head on the masked mean-pool;
    `epochs=6, batch=32, AdamW lr=6e-4, weight_decay=0.1`, 10% linear warmup then cosine decay,
    grad-norm clip 1.0, class-weighted cross-entropy.
  - *DNABERT-2-117M* (revision `7bce263b15377fc15361f52cfab88f8b586abda0`): **LoRA** via `peft`
    (`r=8, lora_alpha=16, lora_dropout=0.05`, target modules `Wqkv`, `attention.output.dense`,
    `gated_layers`, `wo`; 1.25% of weights trainable), backbone frozen, trainable linear head on
    the masked mean-pool; `AdamW lr=3e-4, weight_decay=0.01`, 10% warmup then cosine, grad-norm
    clip 1.0, `batch=32`, class-weighted cross-entropy, up to 4 epochs with early stopping
    (patience 2 on validation AUROC); inputs truncated at 512 tokens
    (`results/upgrades/fm_finetune_v2_token_lengths.csv`: no `cohn` or `nt_enhancers` sequence and 2.3% of `drosophila` sequences
    exceed it);
    PyTorch attention (no triton); per-layer activation checkpointing (memory only).
  - Both: **seeds 42, 43, 44**; the epoch is chosen by AUROC on a stratified **10% validation
    carve of the training split** (the test split is never used for model selection). Apple MPS or
    CPU. Same 1000-resample percentile bootstrap for the reported CIs. Larger runs that need a CUDA
    GPU are packaged in `src/upgrades/fm_finetune/gpu_deferred/`.

## Which result file is authoritative
Some early result files were written before $k$ selection moved from the test split to a
validation carve, and they disagree with the paper on tasks whose selected $k$ changed:
- **Table 1 benchmark MCCs** — authoritative: `results/cross_suite_summary.csv` with
  `results/val_selection.csv`. *Superseded* (old selection-on-test $k$): the `kind=benchmark`
  rows of `results/bootstrap_cis.csv`, PART A.2 of `results/stats_summary.txt`, and the
  `bench_mcc` column of `results/{,nt/}discriminability_summary.csv` (these differ on 7 of 15
  tasks, e.g. `drosophila` 0.445 vs 0.398).
- **TATA-flag cleaning** — authoritative: `results/contamination_cleaning.csv` and
  `results/cleaning_rerun.log` (validation-selected $k$). *Superseded*: `results/cleaning.log`.

Reported values are **truncated** at the stated precision, not rounded half-up
(e.g. `0.4355 → 0.435`).

## Datasets
- Genomic Benchmarks (9 tasks) and the Nucleotide Transformer downstream tasks.
- NT tasks are read at a pinned revision (`96d86d56…`; see `data_nt.py`).

## Large artifacts on Zenodo (not in this repo)
- Frozen-FM embedding cache (HyenaDNA, DNABERT-2) + $k$-mer feature caches (`cache/`).
- GRCh38 primary assembly + GENCODE v44 annotation (used by `localization/`).
- DOIs: frozen-FM embedding cache + manifest — **doi:10.5281/zenodo.20494982**;
  cleaned train/test splits — **doi:10.5281/zenodo.20448617**.

## Reproduce
```bash
pip install -r requirements.txt
python run_benchmark.py        # benchmark landscape (Table 1)
python run_stats.py            # bootstrap CIs + permutation tests + contamination
python src/upgrades/diagnose/diagnose.py --help   # route-labeled report card
```
See the `Makefile` for the full set of targets.
