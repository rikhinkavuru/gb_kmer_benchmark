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
