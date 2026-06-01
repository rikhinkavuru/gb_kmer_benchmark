# Three-route diagnostic report card -- toy_dataset

- **n** = 300 sequences, 2 classes, selected **k = 5**
- **Learnability** (k-mer + LightGBM): MCC = **1.000** [1.0, 1.0], AUROC = 1.0
- **Route 2 (positional)**: delta-MCC(B=8 - B=1) = **+0.000** [0.0, 0.0] (ns)
- **Route 1 (motif)**: max positive motif-only AUROC = **0.990** (best TF SP1), motif richness = 0.23 hits/seq
- **Route 3 (negatives)**: TATA neg-excess = **+0.013**, TATA motif AUROC = 0.494 -> cleaned 0.500, GC AUROC = 0.708

## VERDICT: SOLVED (class-specific motif discriminability)

_Thresholds: TATA>=1.3 bits/pos, curated-TF>=1.0 bits/pos, B=8, k by validation MCC, 1000-resample bootstrap, seed 42. CPU-only; no PyTorch._