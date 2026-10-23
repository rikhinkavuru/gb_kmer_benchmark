# Upgrade 10 — Ship the three-route diagnostic as a tool (`diagnose.py`)

A standalone CLI that takes `(sequences, labels)` [+ optional coordinates] and emits a route-labeled
**report card** (JSON + Markdown): learnability, Route-1 motif discriminability, Route-2 positional-
rescue delta, Route-3 negative-set TATA/GC excess + cleaning effect, and a single route verdict.
Packaging mirrors the companion homology-leakage splitter: numpy/scipy/scikit-learn/LightGBM +
pyjaspar core, **deterministic (seed 42)**, **CPU-only, no PyTorch**, documented fixed thresholds.

It also runs the four checks the paper recommends every regulatory benchmark ship with:

| check | card key | what it answers |
|---|---|---|
| composition-only floor | `composition_fraction` | how much of the above-chance AUROC a 20-d mono+dinucleotide signature reaches alone |
| shuffled-negative control | `shuffled_negative_control` | the motif/GC measures against dinucleotide-preserving shuffles of the positives, which have no composition gap by construction — the one control that behaves *differently* on clean and contaminated data |
| equalized-split delta | `equalized_delta` | the score lost when the negatives' joint GC + 16-dinucleotide distribution is matched to the positives' (positives untouched) |
| train/test redundancy | `redundancy` | the ≥0.7 8-mer-Jaccard near-duplicate rate across the split |

If equalization leaves too few negatives to score, the tool says so — that failure is itself a strong
contamination signal, since it means almost no negative matches any positive's composition.

## Usage
```
# CSV with columns: sequence,label[,chr,start,end,strand]
python src/upgrades/diagnose/diagnose.py --input my_dataset.csv --out report
#   -> report.json + report.md
```
```python
from diagnose import diagnose
card = diagnose(sequences, labels)        # dict report card; card["verdict"] is the route
```

## Worked example (bundled toy: positives carry an Sp1 GC-box, negatives do not)
```
python src/upgrades/diagnose/diagnose.py --input src/upgrades/diagnose/toy_dataset.csv --out example_report
```
produces (`example_report.md`):
> - **n** = 300 sequences, 2 classes, selected **k = 5**
> - **Learnability**: MCC = **1.000** [1.0, 1.0], AUROC = 1.0
> - **Route 2 (positional)**: delta-MCC(B=8 − B=1) = **+0.000** [0.0, 0.0] (ns)
> - **Route 1 (motif)**: max positive motif-only AUROC = **0.990** (best TF SP1), richness 0.23 hits/seq
> - **Route 3 (negatives)**: TATA neg-excess = **+0.013**, TATA AUROC 0.494 → cleaned 0.500, GC AUROC 0.708
>
> **VERDICT: SOLVED (class-specific motif discriminability)**

i.e. the tool correctly attributes the toy's solvability to a class-specific motif (Route 1), with no
positional (Route 2) or contamination (Route 3) signal.

## Decision rules (documented, fixed)
- **SOLVED** if learnability MCC ≥ 0.7 (annotated "motif discriminability" when max positive motif-only
  AUROC ≥ 0.65).
- **Route 2 POSITIONAL** if the B=8 positional delta-MCC is significant and > 0.2.
- **Route 3 NEGATIVE-SET CONTAMINATION** if TATA neg-excess ≥ 0.10, the TATA motif-only AUROC < 0.5
  (neg-enriched), and cleaning the excess TATA-negatives moves that AUROC toward 0.5.
- **Route 1 SHARED-MOTIF** if motifs are abundant (richness > 0.2 hits/seq) but max positive motif-only
  AUROC < 0.6 (present but non-discriminative).
- else **HARD** (no single route dominates).

Thresholds: TATA/TBP ≥ 1.3 bits/pos; curated-TF ≥ 1.0 bits/pos; B=8 position bins; k ∈ {3,4,5,6} by
validation MCC; 1000-resample bootstrap (test sequence = unit); seed 42.

## Test
```
python src/upgrades/diagnose/tests/test_sanity.py   # 3 end-to-end checks on the bundled toy
```
