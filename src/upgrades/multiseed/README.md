# Upgrade 8 — Multi-seed robustness

Wraps the LightGBM TRAINING (not just the already-seed-swept cleaning subsets) in a 5-seed loop on the
key tasks and confirms the benchmark MCC — and hence the coarse route tier — is seed-invariant.

## Method
LightGBM with `subsample=colsample=1.0` is near-deterministic on a fixed training set (the
`random_state` only breaks ties), so genuine training variation is injected by **bootstrap-resampling
the training set per seed** (resample train rows with replacement, seed-controlled) and fitting with
`random_state=seed`; the model is evaluated on the FIXED test set. k is the validation-selected k
(seed 42), held fixed. Reuses the cached k-mer matrices. Seeds 42–46.

## Run
```
python src/upgrades/multiseed/run_multiseed.py            # results/upgrades/multiseed.csv
python src/upgrades/multiseed/make_fig_multiseed.py       # figures/multiseed.{pdf,png}
python src/upgrades/multiseed/tests/test_sanity.py
```

## Measured result (8 key tasks across the routes; 5 seeds; bootstrap-train)
| task | route | MCC mean ± SD | [min, max] | tier |
|---|---|---|---|---|
| human_nontata_promoters | solved (promoter) | 0.836 ± 0.004 | [0.833, 0.841] | STABLE: solved |
| nt_promoter_tata | solved (TATA-promoter) | 0.845 ± 0.011 | [0.834, 0.862] | STABLE: solved |
| demo_human_or_worm | solved (composition) | 0.911 ± 0.001 | [0.910, 0.912] | STABLE: solved |
| human_enhancers_cohn | hard (contamination) | 0.444 ± 0.007 | [0.439, 0.454] | STABLE: hard |
| nt_enhancers | hard (contamination) | 0.498 ± 0.013 | [0.477, 0.510] | VARIES (0.5-boundary) |
| human_ocr_ensembl | hard (shared-motif) | 0.411 ± 0.002 | [0.409, 0.414] | STABLE: hard |
| drosophila_enhancers_stark | hard (control) | 0.377 ± 0.013 | [0.357, 0.393] | STABLE: hard |
| nt_splice_sites_all | hard (positional) | 0.241 ± 0.006 | [0.238, 0.251] | STABLE: hard |

## Interpretation
- **The benchmark MCC is seed-invariant: max SD = 0.013** across all 8 key tasks (bootstrap-train).
- **Route tier is stable for 7/8 tasks.** The single exception, `nt_enhancers`, is NOT instability: its
  MCC (0.498 ± 0.013) sits exactly on the 0.5 partial/hard cutoff, so the *label* flips while the number
  barely moves. Reported straight — the SD is the robustness measure, and it is tiny everywhere.
- Route assignments do not move across seeds; training randomness does not reclassify tasks.
