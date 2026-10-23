# Three-route diagnostic report card -- toy_dataset

- **n** = 300 sequences, 2 classes, selected **k = 5**
- **Learnability** (k-mer + LightGBM): MCC = **1.000** [1.0, 1.0], AUROC = 1.0
- **Route 2 (positional)**: delta-MCC(B=8 - B=1) = **+0.000** [0.0, 0.0] (ns)
- **Composition fraction** = **+0.834** [0.6984, 0.9402] (comp-only AUROC 0.917 / full AUROC 1.000; (AUROC_comp_only-0.5)/(AUROC_full-0.5); share of above-chance AUROC from GC+16 dinuc, can exceed 1 / go <0)
- **Route 1 (motif)**: max positive motif-only AUROC = **0.990** (best TF SP1), motif richness = 0.23 hits/seq
- **Route 3 (negatives)**: TATA neg-excess = **+0.013**, TATA motif AUROC = 0.494 -> cleaned 0.500, GC AUROC = 0.708
- **Shuffled-negative control** (dinucleotide-preserving shuffles of the positives, no composition gap by construction): TATA AUROC = **0.482**, GC AUROC = 0.500. Near 0.5 => the real-negative skew was compositional; away from 0.5 => the motif genuinely belongs to the positive class.
- **Equalized-split delta**: not computable (fewer than 10 negatives survive equalization in a split; 4.0% of negatives would survive) — itself a strong contamination signal, since almost no negative matches any positive's composition
- **Train/test redundancy**: 0.0% of test sequences have a >=0.7 8-mer-Jaccard near-duplicate in train (0.0% verbatim) — clean

## VERDICT: SOLVED (class-specific motif discriminability)

_Thresholds: TATA>=1.3 bits/pos, curated-TF>=1.0 bits/pos, B=8, k by validation MCC, 1000-resample bootstrap, seed 42. CPU-only; no PyTorch._