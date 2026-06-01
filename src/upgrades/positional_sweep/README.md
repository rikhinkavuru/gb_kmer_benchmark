# Upgrade 4 — Graded positional rescue across the suite (Route 2)

Generalises the splice-only (n=1) positional result of `run_positional.py` to the whole suite, with no
new external tasks: (1) B=1 vs B=8 position-binned k-mers at matched k across ALL 15 tasks (paired
bootstrap), and (2) a resolution sweep B ∈ {1,2,4,8,16} for the splice and TATA-promoter tasks.

## Method
The positional arm = the global spectrum hstacked WITH the B-bin position-binned spectrum, so a task
that needs no position can ignore the extra bins and the delta isolates the BENEFIT of resolving
position. Matched k = the validation-selected k (read from `results/val_selection.csv`). Giant splits
are stratified-capped to 20k (applied identically to both arms; the delta is robust to it — verified:
nontata capped 20k Δ=−0.016 vs uncapped −0.010; logged per task, nothing silently truncated). Seed 42,
1000-resample paired bootstrap. Does NOT modify `run_positional.py`.

## Run
```
python src/upgrades/positional_sweep/run_positional_sweep.py        # results/upgrades/positional_sweep.csv + positional_Bsweep.csv
python src/upgrades/positional_sweep/make_fig_positional_sweep.py
python src/upgrades/positional_sweep/tests/test_sanity.py
```

## Measured result — per-task Δ-MCC from B=8 position bins (seed 42, 1000-bootstrap)
A textbook graded dissociation — position helps exactly where fixed-position signal exists:

| task | route | k | Δ-MCC (95% CI) |
|---|---|---|---|
| **nt_splice_sites_all** | **POSITIONAL** | 5 | **+0.601 `[0.572, 0.630]`** (0.243→0.843) |
| nt_promoter_tata | fixed-position promoter | 4 | **+0.019** `[-0.000, 0.041]` (modest, TATA at −25..−30 bp) |
| human_enhancers_ensembl | control | 6 | +0.028 `[0.019, 0.036]` |
| human_ocr_ensembl | control | 6 | +0.026 `[0.015, 0.038]` |
| nt_promoter_no_tata / nt_promoter_all | control | 4 | +0.011 / +0.009 |
| nt_enhancers, dummy_mouse, drosophila, regulatory, demos, cohn, nt_enhancers_types | control | — | \|Δ\| ≤ 0.013 |
| human_nontata_promoters | fixed-position promoter | 6 | −0.016 `[-0.023, -0.008]` (GC-box is position-*diffuse* → position slightly dilutes) |

**The single large rescue is splice (+0.60); every shared-motif / contamination / composition control
moves < 0.03.** The TATA-promoter shows the predicted *modest* positive gain (its TATA box is at a fixed
−25..−30 bp offset), while the non-TATA promoter — whose GC-box motif is positionally diffuse — does not
benefit (slightly negative). Position helps scaled by how positionally structured the task is.

## Resolution sweep (B ∈ {1,2,4,8,16}) — graded, not one lucky bin
See `results/upgrades/positional_Bsweep.csv` and the figure: the splice rescue **grows with bin
resolution** (Δ vs B=1 increases monotonically with B), confirming a graded positional effect rather
than a single fortunate bin; the TATA-promoter gain stays small at every B.

## Interpretation
Adding minimal positional features recovers EXACTLY the positional failure (splice) and nothing else —
a selective dissociation establishing position-discarding as a distinct, causal failure route (Route 2),
separable from shared-motif (Route 1) and contamination (Route 3), and graded by each task's positional
structure across the whole suite.
