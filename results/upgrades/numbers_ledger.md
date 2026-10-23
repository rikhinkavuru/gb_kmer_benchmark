# Numbers ledger — `paper_psb.tex` vs the released result files

Read-only audit. Every "value in source" below was read from a file under `results/`,
`results/nt/`, `results/upgrades/`, or `localization/`. No experiments were run and no
result file was created or modified by this audit.

Paths are relative to the repo root `/Users/rikhinkavuru/gbkb_release`.

**Tally over 190 distinct ledger rows: 183 MATCH · 3 MISMATCH · 3 NOT-FOUND · 1 mixed**
(the mixed row is tex line 426, where `|Δ|≤0.028` MATCHes but "the other twelve" does not).
DERIVED rows are counted inside the MATCH total when the arithmetic reproduces the tex value.
The three rows under KNOWN STALE and the three under New findings are restatements of rows
that also appear in their section tables, and are not double-counted above.

Six tex locations carry the seven problems: lines **93** and **426** ("twelve"), **376**
(0.428), **493** (0.423/0.458 *and* "seed-invariant"), **230** (61–100 %), **425** (−25..−30 bp).

Rounding convention observed throughout the manuscript: values are generally *truncated*
at the reported precision rather than rounded half-up (e.g. `0.4355 -> 0.435`,
`0.4945 -> 0.494`, `0.00115 -> 0.11%`, `-96.5 -> -96`). Where that is the only discrepancy
the row is scored MATCH and flagged in the note.

---

## KNOWN STALE — the three .tex values the newer PDF already corrects

All three were checked against the CSVs. **In every case the newer PDF's value is the one
the data supports and the .tex value is wrong.** These must be carried into the .tex.

| claim (short quote from the tex) | tex line | value in tex | value in source | source file:column/row | VERDICT |
|---|---|---|---|---|---|
| "GC-matched cleaning agrees, $\to$0.423/0.458" | 493 | 0.423 / 0.458 | 0.4265 / 0.4618 → **0.426 / 0.462** | `results/upgrades/composition_clean.csv`:`tata_auroc`, rows `human_enhancers_cohn/gc_match` and `nt_enhancers/gc_match`; echoed in `composition_clean_interpretation.txt` ("gc_match 0.426" / "0.462") | **MISMATCH** (PDF's 0.426/0.462 is correct) |
| "SP1 hit-count positive mean 1.134 vs negative 0.428" | 376 | 0.428 | 0.4228 → **0.423** | `results/nt/contamination_analysis.csv`: row `nt_promoter_no_tata,control,GCbox_SP1_hits`, col `neg_mean` = 0.4228 (the 0.428 in the tex is the `nt_promoter_all` value from the row above) | **MISMATCH** (PDF's 0.423 is correct) |
| "$\sim$0 for the other twelve" / "$|\Delta|\le0.028$ for the other twelve" | 93, 426 | twelve | **thirteen** (15 tasks − splice − nt_promoter_tata = 13) | `results/upgrades/positional_sweep.csv`: 15 task rows; excluding `nt_splice_sites_all` and `nt_promoter_tata` leaves 13 rows, max \|delta_mcc\| = 0.0277 | **MISMATCH** (PDF's "thirteen" is correct) |

Companion values that are **correct as printed** and should not be touched while fixing the above:
`1.134` (pos mean, `nt_promoter_no_tata` GCbox_SP1) and `0.639` (its AUROC 0.6391) both MATCH the same CSV row;
`$|\Delta|\le0.028$` MATCHes (max over the thirteen = 0.0277).

---

## New findings — claims the released data does not support

| claim | tex line | value in tex | value in source | source | VERDICT |
|---|---|---|---|---|---|
| "with $\sim$880 profiles it \emph{saturates}: 61--100\% of $k$-mers are the optimal substring of \emph{some} low-information window" | 230 | 61–100 % | no measurement released; the only in-repo statement of it says **60–100 %** | searched every `.csv`/`.json`/`.txt`/`.log` under `results/` and `localization/` — no saturation table exists. Source of the range is a code comment: `motif_match.py:15` ("60-100% of all 4^k k-mers ... (measured)") and `run_interpretability.py:21` ("60-100% of ALL k-mers") | **NOT-FOUND** (and tex disagrees with the code comment) |
| "\texttt{nt\_enhancers} $0.393\to0.485$; **seed-invariant**" | 493 | "seed-invariant" | not tested | `results/cleaning.log:7` and `results/cleaning_rerun.log:15` both print `seed=42`; the two runs differ only in `best_k`, not seed. `results/upgrades/multiseed.csv` sweeps seeds 42–46 for the **benchmark MCC** only, never for the TATA-flag cleaning | **NOT-FOUND** |
| "(\texttt{nt\_promoter\_tata} $\Delta{=}+0.019$, fixed $-25$ to $-30$\,bp offset)" | 425 | −25 to −30 bp | no offset measured for this dataset | `localization/cohn_offsets.npy` holds offsets for **cohn negatives only** (n=145). No TATA→TSS offset file exists for `nt_promoter_tata`; the −25..−30 window is the canonical literature value, asserted not measured | **NOT-FOUND** |

---

## Abstract (lines 76–107)

| claim | tex line | value in tex | value in source | source file:column/row | VERDICT |
|---|---|---|---|---|---|
| "across the nine Genomic Benchmarks tasks the number of enriched TF motifs is uncorrelated with learnability (Spearman $-0.03$)" | 83–84 | −0.03 | −0.033333 | `results/upgrades/presence_null_stats.csv`:`rho`, row `GB_n9` | MATCH (see caveat C1) |
| "drives the benchmark MCC monotonically (Spearman $+0.924$" | 86 | +0.924 | 0.9239 | `results/upgrades/disc_dose_slopes.csv`:`spearman`, row `POOLED` | MATCH |
| "$n{=}135$" | 87 | 135 | 135 | `results/upgrades/disc_dose_slopes.csv`:`n_points`, row `POOLED` | MATCH |
| "OLS slope $+1.99$" | 87 | +1.99 | 1.9939 | `results/upgrades/disc_dose_slopes.csv`:`slope`, row `POOLED` | MATCH |
| "three independent per-motif fits of $+1.96$ to $+2.02$" | 87 | 1.96 – 2.02 | TBP 1.9584, MYC 2.0193 | `results/upgrades/disc_dose_slopes.csv`:`slope`, rows `TBP`/`MYC` | MATCH |
| "$\rho{=}+0.74$, $n{=}11$" | 88 | +0.74, 11 | 0.7364, n=11 | `results/upgrades/clean_subset_correlation.csv`:`pooled_rho`,`n`, row `n11_full` | MATCH |
| "strengthening to $+0.87$ on the homology-clean subset" | 88–89 | +0.87 | 0.8667 | `results/upgrades/clean_subset_correlation.csv`:`pooled_rho`, row `n9_clean` | MATCH |
| "mean absolute error of $0.14$ MCC" | 89–90 | 0.14 | 0.1399 = mean(`abs_error`) over 11 rows | `results/upgrades/route1_predict.csv`:`abs_error` (all rows) — **DERIVED**, also printed as `MAE = 0.140` in `route1_predict_interpretation.txt` | DERIVED / MATCH |
| "$R^2{=}+0.44$" | 90 | +0.44 | 0.439 | `results/upgrades/route1_predict_interpretation.txt` ("R^2 = +0.439") | MATCH |
| "a wide CI $[-0.27,+0.70]$ that includes $0$ at $n{=}11$" | 90 | [−0.27, +0.70] | [−0.269, +0.701] | `results/upgrades/route1_predict_interpretation.txt` | MATCH |
| "splice sites, MCC $0.244\to0.868$" | 92 | 0.244 → 0.868 | 0.2435 → 0.8676 | `results/positional_features.csv`:`spectrum_mcc`,`positional_mcc`, row `nt_splice_sites_all` | MATCH |
| "$\sim$0 for the other twelve" | 93 | twelve | thirteen | see KNOWN STALE | **MISMATCH** |
| "an excess of TATA/AT-rich sequence ($+20.9$ to $+22.7\%$)" | 95 | +20.9 / +22.7 % | 0.2092 / 0.2270 | `results/nt/contamination_analysis.csv` and `results/contamination_analysis.csv`:`diff_pos_minus_neg`, rows `nt_enhancers` / `human_enhancers_cohn`, measure `TATA_box_fraction(>=1)` | MATCH |
| "collapses the \texttt{human\_enhancers\_cohn} $k$-mer MCC $0.461\to0.064$" | 98 | 0.461 → 0.064 | 0.461223 → 0.064163 | `results/upgrades/composition_clean.csv`:`kmer_mcc`, rows `human_enhancers_cohn/original` and `/comp_equalized` | MATCH |
| "the TATA/GC motif-only AUROC to chance" | 98–99 | ≈0.5 | TATA 0.4998, GC 0.5001 | `results/upgrades/composition_clean.csv`:`tata_auroc`,`gc_auroc`, row `human_enhancers_cohn/comp_equalized` | MATCH |
| "dinucleotide-shuffled negatives make it vanish" | 99 | ≈0.5 | TATA 0.5514, GC 0.5000 | `results/upgrades/shuffled_neg.csv`, row `human_enhancers_cohn/shuffled_neg` | MATCH |
| "two architecturally-unrelated frozen genomic FMs ... lose near-identical, significant enhancer AUROC" | 100–102 | — | Hyena −0.0481, DNABERT-2 −0.0467 on cohn, both `significant=True` | `results/upgrades/fm_cross_architecture.csv`:`test_effect_auroc`,`significant` | MATCH |

## Introduction (lines 121–166)

| claim | tex line | value in tex | value in source | source file:column/row | VERDICT |
|---|---|---|---|---|---|
| "a dose-response at arbitrary sample size ($n{=}135$)" | 154 | 135 | 135 | `results/upgrades/disc_dose_slopes.csv`:`n_points`, row `POOLED` | MATCH |
| "a selective rescue across all 15 tasks" | 155 | 15 | 15 rows | `results/upgrades/positional_sweep.csv` (row count) | MATCH |
| "nine datasets" (Genomic Benchmarks) | 173 | 9 | 9 datasets | `results/results.csv`:`dataset` (9 unique) | MATCH |

## Methods (lines 204–301)

| claim | tex line | value in tex | value in source | source file:column/row | VERDICT |
|---|---|---|---|---|---|
| "a single fixed random seed (42) everywhere" | 207 | 42 | 42 | `results/results.csv`:`seed` (all rows = 42); `seed=42` in `results/stats_summary.txt`, `cleaning.log`, all `results/upgrades/*_interpretation.txt` | MATCH |
| "95\% percentile-bootstrap confidence intervals ... individual test sequence as the resampling unit" | 208–209 | 95 %, seq unit | "Bootstrap: 1000 resamples (within-class, with replacement) ... the scanned sequence as the unit (percentile CI)" | `results/stats_summary.txt` (footer); `results/upgrades/route3_specificity_interpretation.txt` (footer) | MATCH |
| "$k\in\{3,4,5,6\}$ ... all $4^k$ ACGT $k$-mers" | 213–215 | k=3..6, dim 64/256/1024/4096 | 64, 256, 1024, 4096 | `results/results.csv`:`k`,`feature_dim` | MATCH |
| "stratified 80/20 carve of the training set ... highest \emph{validation} MCC" | 222–224 | — | per-k validation MCCs recorded, `new_k` = argmax | `results/val_selection.csv`:`val`,`new_k` | MATCH |
| "BH FDR at $q<0.05$" | 229 | q<0.05 | max `fdr` in file = 0.0499 | `results/enriched_tfs.csv`:`fdr` | MATCH |
| "$\sim$880 profiles ... 61--100\% of $k$-mers" | 230 | 61–100 % | not released; code says 60–100 % | see New findings | **NOT-FOUND** |
| "For the GC-box (Sp1/KLF) we use 1.0 bits/pos; for TATA a specificity-controlled 1.3 bits/pos" | 238–239 | 1.0 / 1.3 | "GC-box(SP1) >= 1.0 bits/pos; TATA(TBP) >= 1.3 bits/pos" | `results/stats_summary.txt` (verdict-logic footer); `results/contamination_cleaning.csv`:`tata_bits` = 1.3 | MATCH |
| "intact across a 0.8--1.5 bits/pos sweep" | 240–241 | 0.8–1.5 | thresholds 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5 (7 values) | `results/upgrades/route3_threshold_invariance.csv`:`threshold_bits` | MATCH |
| "we repeat over five seeds and three motifs of differing information content (SP1, MYC, TBP)" | 251–252 | 5 seeds, 3 motifs | 3 motifs × 45 points each = 135 | `results/upgrades/disc_dose_slopes.csv`:`motif`,`n_points`; `disc_dose.csv` | MATCH |
| "a sweep $B\in\{1,2,4,8,16\}$" | 257 | {1,2,4,8,16} | 1, 2, 4, 8, 16 | `results/upgrades/positional_Bsweep.csv`:`B` | MATCH |
| "e.g.\ 22.2\% of \texttt{cohn}'s negatives" | 262 | 22.2 % | 0.2223 | `results/contamination_cleaning.csv`:`frac_neg_removed`, row `human_enhancers_cohn` | MATCH |
| "flag negatives carrying a high-confidence TATA box (TBP $\ge$ 1.3 bits/pos)" | 260–261 | 1.3 | 1.3 | `results/contamination_cleaning.csv`:`tata_bits` | MATCH |
| "a 20-dimensional composition signature (4 mononucleotide + 16 dinucleotide ...)" | 266–267 | 20 | 20 | `results/upgrades/composition_clean_balance.csv`:`n_features` (all rows = 20) | MATCH |
| "20 column-permutations of the TBP PSSM" | 281 | 20 | 20 | `results/upgrades/route3_specificity.csv`:`n_panel`, rows `scrambled_TBP_panel` | MATCH |
| "10 random PWMs matched to TBP's total information content" | 282 | 10 | 10 | `results/upgrades/route3_specificity.csv`:`n_panel`, rows `random_IC_panel` | MATCH |
| "GENCODE v44 transcription start sites (TSS, $\pm$500\,bp)" | 288–289 | v44, ±500 bp | "GENCODE v44, TSS +-500 bp" | `localization/contamination_localization.txt` (STEP 3a header); `results/upgrades/nt_coords_interpretation.txt` | MATCH |
| "HyenaDNA-tiny (16k context ...) and DNABERT-2-117M (512-token context ...)" | 293–294 | 16k / 512 | `HyenaDNA-tiny-16k`, `DNABERT-2-117M` | `results/upgrades/fm_cross_architecture.csv`:`architecture` | MATCH |
| "the sequence-summary token ([SEP] for HyenaDNA, [CLS] for DNABERT-2)" | 295–296 | sep / cls | `sep` (Hyena), `cls` and `mean` (DNABERT-2) | `results/upgrades/fm_cross_architecture.csv`:`pooling` | MATCH |

## Table 1 (`tab:land`, lines 305–335)

Every row was checked on six values: `n_train`, `n_test`, `k`, MCC, MCC CI, AUROC.
Point estimates and `k` come from `results/results.csv` / `results/results_nt.csv`
(`model=lgbm` & `selected_by_val=1`); the CIs come from `results/cross_suite_summary.csv`
(`mcc_ci`), which is itself keyed to `results/val_selection.csv` (`new_mcc_lo`/`new_mcc_hi`).

| claim (task row) | tex line | value in tex | value in source | source file:column/row | VERDICT |
|---|---|---|---|---|---|
| `demo_human_or_worm` 75,000 / 25,000 / k5 / 0.914 [0.909, 0.919] / 0.992 | 317 | as printed | 75000, 25000, k=5, 0.91408, [0.909,0.919], 0.992301 | `results.csv` row `demo_human_or_worm,k5,lgbm,selected_by_val=1`; `cross_suite_summary.csv`:`mcc_ci` | MATCH |
| `human_ensembl_regulatory` 231,348 / 57,713 / k5 / 0.907 [0.904, 0.910] / 0.991 | 318 | as printed | 0.906674, [0.904,0.910], 0.990743 | same pair of files | MATCH |
| `nt_promoter_all` 53,276 / 5,920 / k4 / 0.888 [0.877, 0.899] / 0.985 | 319 | as printed | 0.887832, [0.877,0.899], 0.985386 | `results_nt.csv`; `cross_suite_summary.csv` | MATCH |
| `nt_promoter_no_tata` 47,767 / 5,299 / k4 / 0.887 [0.873, 0.899] / 0.985 | 320 | as printed | 0.886663, [0.873,0.899], 0.984534 | same | MATCH |
| `human_nontata_promoters` 27,097 / 9,034 / k6 / 0.874 [0.864, 0.883] / 0.989 | 321 | as printed | 0.873673, lo 0.8645, 0.98908 | `results.csv`; CI from `results/bootstrap_cis.csv` ("[0.864, 0.883]"). `cross_suite_summary.csv` prints the same bound rounded up ("[0.865, 0.883]") | MATCH (0.8645 truncated) |
| `nt_promoter_tata` 5,509 / 621 / k4 / 0.846 [0.803, 0.886] / 0.984 | 322 | as printed | 0.846348, [0.803,0.886], 0.983804 | `results_nt.csv`; `cross_suite_summary.csv` | MATCH |
| `demo_coding_vs_intergenomic` 75,000 / 25,000 / k5 / 0.800 [0.793, 0.808] / 0.965 | 323 | as printed | 0.800248, [0.793,0.808], 0.964622 | `results.csv`; `cross_suite_summary.csv` | MATCH |
| `human_enhancers_ensembl` 123,872 / 30,970 / k6 / 0.593 [0.585, 0.602] / 0.875 | 324 | as printed | 0.593439, [0.585,0.602], 0.874799 | same | MATCH |
| `dummy_mouse_enhancers_ensembl` 968 / 242 / k6 / 0.529 [0.422, 0.632] / 0.858 | 325 | as printed | 0.529088, [0.422,0.632], 0.857592 | `results.csv`; `cross_suite_summary.csv`; `val_selection.csv` (`old_k=3 new_k=6 CHANGED`) | MATCH |
| `nt_enhancers` 14,968 / 400 / k4 / 0.514 [0.432, 0.595] / 0.850 | 326 | as printed | 0.513712, lo 0.4315, 0.849675 | `results_nt.csv`; CI from `val_selection.csv`:`new_mcc_lo`=0.4315 (`cross_suite_summary.csv` prints 0.431) | MATCH |
| `nt_enhancers_types` 14,968 / 400 / k6 / 0.476 [0.415, 0.532] / 0.837 | 327 | as printed | 0.476102, [0.415,0.532], 0.836683 | `results_nt.csv`; `cross_suite_summary.csv` | MATCH |
| `human_enhancers_cohn` 20,843 / 6,948 / k5 / 0.461 [0.440, 0.482] / 0.805 | 328 | as printed | 0.461223, [0.440,0.482], 0.804649 | `results.csv`; `cross_suite_summary.csv` | MATCH |
| `human_ocr_ensembl` 139,804 / 34,952 / k6 / 0.421 [0.412, 0.430] / 0.785 | 329 | as printed | 0.421252, [0.412,0.430], 0.785095 | same | MATCH |
| `drosophila_enhancers_stark` 5,184 / 1,730 / k4 / 0.398 [0.352, 0.441] / 0.763 | 330 | as printed | 0.398321, [0.352,0.441], 0.763135 | `results.csv`; `cross_suite_summary.csv`; `val_selection.csv` (`old_k=5 new_k=4 CHANGED`) | MATCH |
| `nt_splice_sites_all` 27,000 / 3,000 / k5 / 0.244 [0.219, 0.271] / 0.704 | 331 | as printed | 0.243511, [0.219,0.271], 0.70346 | `results_nt.csv`; `cross_suite_summary.csv` | MATCH |
| caption: "\texttt{human\_enhancers\_ensembl} ... its excess is only $+10.8\%$" | 311 | +10.8 % | 0.1083 | `results/contamination_analysis.csv`:`diff_pos_minus_neg`, row `human_enhancers_ensembl,TATA_box_fraction(>=1)`; also `cross_suite_summary.csv`:`tata_neg_excess` = 0.1083 | MATCH |
| caption: "Every value is read directly from the result CSVs." | 312 | — | true for all 15 rows × 6 values | verified above | MATCH |

## Results — Benchmark landscape (lines 337–355)

| claim | tex line | value in tex | value in source | source file:column/row | VERDICT |
|---|---|---|---|---|---|
| "Achievable MCC spans 0.244 to 0.914" | 341 | 0.244 – 0.914 | min 0.243511 (`nt_splice_sites_all`), max 0.91408 (`demo_human_or_worm`) | **DERIVED** from `results_nt.csv` / `results.csv`:`mcc` over the 15 `selected_by_val=1` lgbm rows | DERIVED / MATCH |
| "$\sim$41\% of its test sequences have a $\ge$0.7-Jaccard near-duplicate in training" | 351–352 | ~41 % | 0.40635 | `results/upgrades/homology_leakage.csv`:`frac_ge_0_7`, row `human_nontata_promoters` | MATCH |
| "the contamination flagship \texttt{cohn} is redundancy-clean; \texttt{nt\_enhancers} has verbatim overlap" | 353–354 | — | cohn `verdict=clean`; nt_enhancers `verdict=LEAKY` | `results/upgrades/homology_leakage.csv`:`verdict` | MATCH |

## Results — Route 1 (lines 357–405)

| claim | tex line | value in tex | value in source | source file:column/row | VERDICT |
|---|---|---|---|---|---|
| "Spearman of MCC against \#enriched TFs across the nine GB tasks $\rho{=}-0.03$, n.s." | 361–362 | −0.03, n.s. | rho −0.033333, p 0.932157 | `results/upgrades/presence_null_stats.csv`:`rho`,`p`, row `GB_n9` | MATCH (see caveat C1) |
| "\texttt{human\_enhancers\_ensembl} 265 ... FDR-enriched TFs" | 363 | 265 | 265 rows | `results/enriched_tfs.csv`, count of rows with `dataset=human_enhancers_ensembl`; also `presence_null.csv`:`n_enriched_TFs` | MATCH |
| "\texttt{human\_ocr\_ensembl} 243" | 363–364 | 243 | 243 rows | same, `dataset=human_ocr_ensembl` | MATCH |
| "SP2 hit-count separates the classes with motif-only AUROC 0.751" | 370 | 0.751 | 0.7513 | `results/motif_discriminability.csv`:`motif_auroc`, row `human_nontata_promoters,...,SP2` | MATCH |
| "ranks SP1 first (gain-weighted $z=7.60$ ...)" | 370–371 | 7.60 | 7.595 (top row by `enrich_z`) | `results/enriched_tfs.csv`:`enrich_z`, row `human_nontata_promoters,SP1` | MATCH |
| "BH-FDR $1.4\times10^{-11}$" | 371 | 1.4e−11 | 1.35e−11 | `results/enriched_tfs.csv`:`fdr`, same row | MATCH (2 s.f.) |
| "followed by SP2, SP4, KLF7, KLF12" | 371 | that order | ranks 2–5 by `enrich_z`: SP2 7.240, SP4 7.171, KLF7 7.145, KLF12 7.135 | `results/enriched_tfs.csv`, `dataset=human_nontata_promoters` sorted by `enrich_z` desc | MATCH |
| "TFAP4 E-box positive mean 0.103 vs negative mean 0.046" | 373 | 0.103 / 0.046 | 0.1035 / 0.0457 | `results/motif_discriminability.csv`:`pos_mean_hits`,`neg_mean_hits`, row `human_ocr_ensembl,...,TFAP4` | MATCH |
| "motif-only AUROC 0.527" | 374 | 0.527 | 0.5268 | same row, `motif_auroc` | MATCH |
| "BATF3 AP-1 0.648 vs 0.423, AUROC 0.573" | 374 | 0.648 / 0.423 / 0.573 | 0.648 / 0.423 / 0.5728 | `results/motif_discriminability.csv`, row `human_ocr_ensembl,...,BATF3` | MATCH |
| "\texttt{nt\_promoter\_no\_tata} SP1 hit-count positive mean 1.134" | 376 | 1.134 | 1.134 | `results/nt/contamination_analysis.csv`:`pos_mean`, row `nt_promoter_no_tata,control,GCbox_SP1_hits` | MATCH |
| "vs negative 0.428" | 376 | 0.428 | 0.4228 → 0.423 | same row, `neg_mean` | **MISMATCH** (KNOWN STALE) |
| "AUROC 0.639" | 376 | 0.639 | 0.6391 | same row, `auroc_pos_gt_neg` | MATCH |
| "a pooled real-task correlation over $n{=}11$ tasks" | 379 | 11 | 11 | `results/upgrades/clean_subset_correlation.csv`:`n`, row `n11_full`; 11 rows in `route1_predict.csv` | MATCH |
| "Across $n{=}135$ settings (3 motifs $\times$ 5 seeds $\times$ 9 doses)" | 382–383 | 135 = 3×5×9 | 135 (45 per motif × 3) | **DERIVED** from `results/upgrades/disc_dose_slopes.csv`:`n_points`; `disc_dose.csv` row count | DERIVED / MATCH |
| "MCC rises monotonically with $d$ (Spearman $+0.924$)" | 383 | +0.924 | 0.9239 | `results/upgrades/disc_dose_slopes.csv`:`spearman`, row `POOLED` | MATCH |
| "$+1.97$ (SP1, 95\% CI $[1.87, 2.07]$)" | 385 | +1.97 [1.87, 2.07] | 1.9711 [1.8742, 2.0674] | `results/upgrades/disc_dose_slopes.csv`:`slope`,`slope_lo`,`slope_hi`, row `SP1` | MATCH |
| "$+2.02$ (MYC, $[1.90, 2.14]$)" | 385 | +2.02 [1.90, 2.14] | 2.0193 [1.8961, 2.1434] | same, row `MYC` | MATCH |
| "$+1.96$ (TBP, $[1.78, 2.13]$)" | 385–386 | +1.96 [1.78, 2.13] | 1.9584 [1.7801, 2.1323] | same, row `TBP` | MATCH |
| "All three CIs exclude 0 and overlap each other" | 386–387 | — | "all three per-motif CIs overlap EACH OTHER pairwise ... True" | `results/upgrades/disc_dose_slopes_interpretation.txt` (overlap check block) | MATCH |
| "the pooled OLS slope $+1.99$ (95\% CI $[1.92, 2.06]$)" | 387 | +1.99 [1.92, 2.06] | 1.9939 [1.9238, 2.0587] | `results/upgrades/disc_dose_slopes.csv`, row `POOLED` | MATCH |
| "Higher-information motifs reach higher $d$ and MCC" | 388 | — | SP1 IC 14.2 d≤0.96 MCC≤0.88; MYC 11.0 d≤0.87 MCC≤0.72; TBP 8.8 d≤0.81 MCC≤0.60 | `results/upgrades/disc_dose_interpretation.txt` (per-motif d/MCC ranges) | MATCH |
| "Spearman $\rho{=}+0.87$ ($p{=}0.003$ ...)" | 395 | +0.87, 0.003 | 0.8667, 0.0025 | `results/upgrades/clean_subset_correlation.csv`:`pooled_rho`,`pooled_p`, row `n9_clean` | MATCH |
| "partial controlling for suite $+0.83$" | 395 | +0.83 | 0.8332 | same row, `partial_rho` | MATCH |
| "leave-one-out jackknife $[+0.81, +0.98]$, every subset positive and significant" | 395–396 | [0.81, 0.98] | [0.8095, 0.9762], `jack_all_sig=True` | same row, `jack_rho_lo`,`jack_rho_hi`,`jack_all_sig` | MATCH |
| "nine TF-motif tasks" (clean subset) | 394 | 9 | n=9 | same row, `n`; task list in `clean_subset_correlation_interpretation.txt` | MATCH |
| "Dropping the two tasks ... (\texttt{human\_nontata\_promoters}, \texttt{nt\_enhancers})" | 392–393 | those two | "dropped (homology-inflated MCC): ['human_nontata_promoters', 'nt_enhancers']" | `results/upgrades/clean_subset_correlation_interpretation.txt` | MATCH |
| "On the full eleven-task set the correlation is $\rho{=}+0.74$ ($p{=}0.01$)" | 396–397 | +0.74, 0.01 | 0.7364, 0.0098 | `clean_subset_correlation.csv`, row `n11_full` | MATCH |
| "GB-only $+0.89$" | 397 | +0.89 | 0.8857 | same row, `gb_rho` | MATCH |
| "partial $+0.75$" | 397 | +0.75 | 0.753 | same row, `partial_rho` | MATCH |
| "jackknife $[+0.68, +0.83]$" | 397 | [0.68, 0.83] | [0.6848, 0.8303] | same row | MATCH |
| "underpowered ($n{=}5\text{--}6$; NT-only $\rho{=}+0.40$, $p{=}0.50$)" | 399 | 5–6, +0.40, 0.50 | n_gb 6, n_nt 5; nt_rho 0.4, nt_p 0.5046 | same row, `n_gb`,`n_nt`,`nt_rho`,`nt_p` | MATCH |
| "mean absolute error of $0.14$ MCC (95\% CI $[0.11, 0.18]$)" | 400–401 | 0.14 [0.11, 0.18] | 0.140 [0.108, 0.180] | mean of `results/upgrades/route1_predict.csv`:`abs_error` (**DERIVED** = 0.1399); CI in `route1_predict_interpretation.txt` | DERIVED / MATCH |
| "$R^2{=}+0.44$ has a wide CI $[-0.27, +0.70]$" | 401 | +0.44, [−0.27, +0.70] | +0.439, [−0.269, +0.701] | `results/upgrades/route1_predict_interpretation.txt` | MATCH |
| "Excluded composition/non-TF tasks: human-vs-worm, coding-vs-intergenomic, \texttt{dummy\_mouse...}, \texttt{nt\_splice\_sites\_all}" | 402–404 | those four | 15 − 11 = the same four absent from `route1_predict.csv` | **DERIVED** from `results/upgrades/route1_predict.csv`:`task` (11 rows) vs the 15-task list | DERIVED / MATCH |
| fig. `disc_dose` caption: "$n{=}135$: 3 motifs $\times$ 5 seeds $\times$ 9 doses ... Seed 42" | 409–412 | 135, seed 42 | 135, seed 42 | `results/upgrades/disc_dose_slopes.csv`; `disc_dose_slopes_interpretation.txt` footer | MATCH |

## Results — Route 2 (lines 416–429)

| claim | tex line | value in tex | value in source | source file:column/row | VERDICT |
|---|---|---|---|---|---|
| "\texttt{nt\_splice\_sites\_all} MCC 0.244" | 418 | 0.244 | 0.243511 | `results/results_nt.csv`:`mcc`, row `nt_splice_sites_all,k5,lgbm,selected_by_val=1` | MATCH |
| "motif-only AUROC 0.522, at chance" | 418 | 0.522 | 0.5216 | `results/nt/discriminability_summary.csv`:`max_motif_auroc`, row `nt_splice_sites_all` | MATCH |
| "Adding a position-binned spectrum ($B{=}8$)" | 419–420 | B=8 | n_bins = 8 | `results/positional_features.csv`:`n_bins` | MATCH |
| "MCC $0.244\to0.868$ ($\Delta{=}+0.624$, 95\% CI $[0.594, 0.651]$)" | 421 | 0.244→0.868, +0.624 [0.594,0.651] | 0.2435→0.8676, 0.6241, "[0.594,0.651]" | `results/positional_features.csv`:`spectrum_mcc`,`positional_mcc`,`delta_mcc`,`delta_ci`, row `nt_splice_sites_all` | MATCH |
| "every control barely moves ($|\Delta|\le0.013$" | 422 | ≤0.013 | max \|delta\| over the 4 controls = 0.0127 | `results/positional_features.csv`:`delta_mcc`, non-splice rows (0.0096, 0.0047, −0.0101, −0.0127) | MATCH |
| "$\sim$50$\times$ smaller" | 422 | ~50× | 0.6241 / 0.0127 = **49.1×** | **DERIVED** from the two `delta_mcc` values above | DERIVED / MATCH |
| "Extended to all 15 tasks" | 422 | 15 | 15 rows | `results/upgrades/positional_sweep.csv` | MATCH |
| "one large rescue (splice $\Delta{=}+0.601$, $[0.572, 0.630]$" | 423 | +0.601 [0.572, 0.630] | 0.6005, "[0.572,0.630]" | `results/upgrades/positional_sweep.csv`:`delta_mcc`,`delta_ci`, row `nt_splice_sites_all` | MATCH |
| "below the uncapped $+0.624$ because splice's training set is capped to 20k in this sweep" | 423–424 | 20k cap | "[cap] nt_splice_sites_all: train 27000->20001, test 3000->3000" | `results/upgrades/positional_sweep_run.log:83` | MATCH |
| "\texttt{nt\_promoter\_tata} $\Delta{=}+0.019$" | 425 | +0.019 | 0.0194 | `results/upgrades/positional_sweep.csv`:`delta_mcc`, row `nt_promoter_tata` | MATCH |
| "fixed $-25$ to $-30$\,bp offset" | 425 | −25..−30 bp | not measured for this dataset | see New findings | **NOT-FOUND** |
| "$|\Delta|\le0.028$ for the other twelve" | 426 | ≤0.028; twelve | max = 0.0277 (`human_enhancers_ensembl`); count = **13** | `results/upgrades/positional_sweep.csv` (15 rows − splice − nt_promoter_tata) | 0.028 MATCH; "twelve" **MISMATCH** (KNOWN STALE) |
| "$B\in\{2,4,8,16\}$ grows the splice rescue monotonically ($+0.46, +0.53, +0.60, +0.66$)" | 427–428 | 0.46/0.53/0.60/0.66 | 0.4564, 0.5255, 0.6005, 0.6590 | `results/upgrades/positional_Bsweep.csv`:`delta_vs_B1`, rows `nt_splice_sites_all` B=2,4,8,16 | MATCH |

## Results — Route 3 (lines 431–502)

| claim | tex line | value in tex | value in source | source file:column/row | VERDICT |
|---|---|---|---|---|---|
| "TBP hit-count positive mean 0.683 vs negative 1.641" | 436 | 0.683 / 1.641 | 0.6833 / 1.6413 | `results/contamination_analysis.csv`:`pos_mean`,`neg_mean`, row `human_enhancers_cohn,suspect,TATA_TBP_hits` | MATCH |
| "motif-only AUROC 0.371, 95\% CI $[0.365, 0.376]$" | 436–437 | 0.371 [0.365, 0.376] | 0.3708, "[0.365,0.376]" | `results/upgrades/composition_clean.csv`:`tata_auroc`,`tata_ci`, row `human_enhancers_cohn/original` (also `contamination_cleaning.csv`:`orig_TATA_auroc`) | MATCH |
| "TATA-box fraction positive 34.6\% vs negative 57.3\%" | 437 | 34.6 / 57.3 % | 0.3458 / 0.5727 | `results/contamination_analysis.csv`:`pos_mean`,`neg_mean`, row `human_enhancers_cohn,...,TATA_box_fraction(>=1)` | MATCH |
| "an \textbf{excess of $+22.7\%$}" | 437–438 | +22.7 % | 0.2270 | same row, `diff_pos_minus_neg` = −0.227 | MATCH |
| "\texttt{nt\_enhancers} (TBP positive 0.14 vs negative 0.58" | 438–439 | 0.14 / 0.58 | 0.14 / 0.58 | `results/nt/contamination_analysis.csv`, row `nt_enhancers,suspect,TATA_TBP_hits` | MATCH |
| "AUROC 0.393 $[0.387, 0.399]$" | 439 | 0.393 [0.387, 0.399] | 0.3927, "[0.387,0.399]" | `results/upgrades/composition_clean.csv`, row `nt_enhancers/original` | MATCH |
| "TATA-box fraction positive 10.6\% vs negative 31.5\%, an excess of $+20.9\%$" | 439–440 | 10.6 / 31.5 / +20.9 | 0.1057 / 0.3150 / 0.2092 | `results/nt/contamination_analysis.csv`, row `nt_enhancers,...,TATA_box_fraction(>=1)` | MATCH |
| "\texttt{cohn}/\texttt{nt\_enhancers} GC AUROC 0.737/0.822" | 442 | 0.737 / 0.822 | 0.7374 / 0.8224 | `results/upgrades/composition_clean.csv`:`gc_auroc`, `original` rows | MATCH |
| "100 of the 400 \texttt{nt\_enhancers} test sequences (25.0\%) are verbatim duplicates of a training sequence" | 444–445 | 100 / 400 / 25.0 % | 100, 400, 0.25 | `results/upgrades/nt_enhancers_duplication.csv`:`train_test_verbatim`,`n_test`,`train_test_verbatim_frac` | MATCH |
| "all 13{,}896 \texttt{cohn} negatives" | 450–451 | 13,896 | 13,896 | `localization/coordinate_analysis.csv` (13,896 rows with `klass=negative`); `composition_clean.csv`:`n_neg` = 13896 | MATCH |
| "TATA-flagged negatives overlap a TSS ($\pm$500\,bp) in only 1.8\% of cases" | 452–453 | 1.8 % | 145 / 7,977 = 1.82 % | `localization/coordinate_analysis.csv`: `flagged==True & klass==negative`, mean of `tss_overlap`; printed as "1.8% (145/7,977)" in `contamination_localization.txt` | MATCH |
| "\emph{below} the 3.3\% of non-flagged negatives" | 453 | 3.3 % | 195 / 5,919 = 3.29 % | same file, `flagged==False` | MATCH |
| "(0.55$\times$ depletion" | 453 | 0.55× | 1.82 / 3.29 = **0.552** | **DERIVED** from the two rates above; printed as "0.55x" in `contamination_localization.txt` | DERIVED / MATCH |
| "Fisher $p{=}3{\times}10^{-8}$" | 453–454 | 3e−8 | 3.2e−8 (Fisher OR=0.54) | `localization/contamination_localization.txt`, STEP 3a | MATCH |
| "far below the 15.6\% positive rate" | 454 | 15.6 % | 2,168 / 13,895 = 15.60 % | `localization/coordinate_analysis.csv` (positive rows; TSS flag sits in the shifted `hit_start` column — 2,168 True of 13,895); printed as "15.6% (2,168/13,895)" in `contamination_localization.txt` | MATCH |
| "no peak at the canonical $-25$ to $-30$\,bp (median $-96$)" | 455 | median −96 | −96.5 (n=145) | `localization/cohn_offsets.npy` (median); printed as "median = -96 bp" in `contamination_localization.txt` | MATCH (−96.5 truncated) |
| "mappability 22\%$\to$28.2\%" | 457 | 22 % → 28.2 % | exact-match baseline 21.9 %; fuzzy `mapped_frac` 0.2819 | `localization/contamination_localization.txt` STEP 1 (21.9 %); `results/upgrades/nt_coords.csv`:`mapped_frac` = 0.2819 | MATCH |
| "the flagged class is likewise TSS-\emph{depleted} (0.6\% vs 1.4\%, $0.46\times$" | 458 | 0.6 / 1.4 / 0.46× | 0.0063 / 0.0138 / 0.456 | `results/upgrades/nt_coords.csv`:`flagged_neg_tss`,`nonflagged_neg_tss`,`enrichment` | MATCH |
| "(28\% mappable; Fisher $p{=}0.17$)" | 459 | 28 %, 0.17 | 0.2819, 0.166 | `results/upgrades/nt_coords.csv`:`mapped_frac`,`fisher_p` | MATCH |
| "(SMD $1.007\to0.059$ for \texttt{cohn})" | 462–463 | 1.007 → 0.059 | 1.0074 → 0.0595 | `results/upgrades/composition_clean_balance.csv`:`max_abs_smd_before`,`max_abs_smd_after`, row `human_enhancers_cohn` | MATCH |
| "\texttt{cohn} TATA $0.371\to0.500$, GC $0.737\to0.500$" | 464 | 0.500 / 0.500 | 0.4998 / 0.5001 | `results/upgrades/composition_clean.csv`, row `human_enhancers_cohn/comp_equalized` | MATCH |
| "\texttt{nt\_enhancers} TATA $0.393\to0.494$, GC $0.822\to0.540$" | 464–465 | 0.494 / 0.540 | 0.4945 / 0.5398 | same file, row `nt_enhancers/comp_equalized` | MATCH (0.4945 truncated) |
| "(\texttt{cohn} $0.461\to0.064$, baseline $0.410\to0.000$)" | 466 | 0.461→0.064; 0.410→0.000 | `kmer_mcc` 0.461223→0.064163; `comp_only_mcc` 0.410478→0.0 | `results/upgrades/composition_clean.csv`, rows `human_enhancers_cohn/original` and `/comp_equalized` | MATCH |
| "the equalizer retains 42.9\% of its negatives" | 468 | 42.9 % | 0.4287 | `results/upgrades/composition_clean.csv`:`retained_neg_frac`, row `drosophila_enhancers_stark/comp_equalized` | MATCH |
| "vs only 4.5--4.9\% for the contaminated sets" | 468 | 4.5–4.9 % | 0.0451 (cohn), 0.0493 (nt_enhancers) | same column, `comp_equalized` rows | MATCH |
| "(MCC $0.398\to0.177$, residual $+0.128$)" | 469 | 0.398→0.177, +0.128 | 0.398321→0.176527; `residual_mcc` 0.1279 | `results/upgrades/composition_clean.csv`, `drosophila_enhancers_stark` rows | MATCH |
| "\texttt{cohn} TATA $0.371\to0.551$, GC $\to0.500$" | 472–473 | 0.551 / 0.500 | 0.5514 / 0.5 | `results/upgrades/shuffled_neg.csv`:`tata_auroc`,`gc_auroc`, row `human_enhancers_cohn/shuffled_neg` | MATCH |
| "\texttt{nt\_enhancers} TATA $0.393\to0.507$, GC $\to0.500$" | 473–474 | 0.507 / 0.500 | 0.5067 / 0.5 | same file, row `nt_enhancers/shuffled_neg` | MATCH |
| "for the control ... the TATA AUROC instead \emph{rises} to 0.749" | 474–475 | 0.749 | 0.7486 | same file, row `drosophila_enhancers_stark/shuffled_neg` | MATCH |
| "random control PWMs ... AUROC $0.502\pm0.033$ for \texttt{cohn}" | 480 | 0.502 ± 0.033 | 0.5023 ± 0.0333 | `results/upgrades/route3_specificity.csv`:`motif_only_auroc`,`auroc_sd`, row `human_enhancers_cohn/random_IC_panel` | MATCH |
| "$0.507\pm0.016$ for \texttt{nt\_enhancers}" | 480 | 0.507 ± 0.016 | 0.5069 ± 0.0162 | same file, row `nt_enhancers/random_IC_panel` | MATCH |
| "a \emph{column-scrambled} TBP ... stays skewed ($0.376\pm0.017$ and $0.396\pm0.027$)" | 482–483 | 0.376±0.017, 0.396±0.027 | 0.3761 ± 0.0168; 0.3960 ± 0.0271 | same file, rows `*/scrambled_TBP_panel` | MATCH |
| "essentially equal to real TATA" | 483 | — | real TATA 0.3671 (cohn) / 0.3919 (nt) vs scrambled 0.3761 / 0.3960 | same file, rows `*/real_TATA` | MATCH |
| "real TATA on \texttt{nt\_promoter\_tata} skews to the \emph{positives} (AUROC 0.589, $[0.577, 0.600]$)" | 484–485 | 0.589 [0.577, 0.600] | 0.5885, [0.5768, 0.6001] | `results/upgrades/route3_specificity.csv`, row `nt_promoter_tata/real_TATA` (also `results/nt/contamination_analysis.csv`) | MATCH |
| "sweeping the TATA threshold across 0.8--1.5\,bits/pos ... (\texttt{cohn} $[0.485, 0.503]$" | 486–487 | [0.485, 0.503] | min 0.4855, max 0.5030 over 7 thresholds | `results/upgrades/route3_threshold_invariance.csv`:`tata_auroc`, rows `human_enhancers_cohn/comp_equalized` | MATCH |
| "\texttt{nt\_enhancers} $[0.490, 0.501]$" | 487 | [0.490, 0.501] | min 0.4902, max 0.5009 | same file, rows `nt_enhancers/comp_equalized` | MATCH |
| "the original is neg-skewed at every threshold" | 487–488 | all < 0.5 | cohn 0.307–0.401; nt 0.282–0.431 | same file, `original` rows | MATCH |
| "drops 22.2\% of \texttt{cohn} ... negatives" | 491 | 22.2 % | 0.2223 | `results/contamination_cleaning.csv`:`frac_neg_removed`, row `human_enhancers_cohn` (= 1 − 0.7777 in `composition_clean.csv`) | MATCH |
| "and 20.8\% of \texttt{nt\_enhancers} negatives" | 491 | 20.8 % | 0.2078 | same column, row `nt_enhancers` | MATCH |
| "\texttt{cohn} $0.371\to0.435$" | 492 | 0.435 | 0.4355 | `results/contamination_cleaning.csv`:`cleaned_TATA_auroc` (also `composition_clean.csv` `tata_flag` arm) | MATCH (truncated) |
| "\texttt{nt\_enhancers} $0.393\to0.485$" | 492–493 | 0.485 | 0.4850 | same | MATCH |
| "seed-invariant" | 493 | — | only seed 42 was ever run for this cleaning | see New findings | **NOT-FOUND** |
| "GC-matched cleaning agrees, $\to$0.423/0.458" | 493 | 0.423 / 0.458 | 0.4265 / 0.4618 | `results/upgrades/composition_clean.csv`, `gc_match` rows | **MISMATCH** (KNOWN STALE) |
| "\texttt{cohn} $0.461\to0.407$" | 494 | 0.407 | 0.4069 (`contamination_cleaning.csv`) / 0.406922 (`composition_clean.csv`) | `results/contamination_cleaning.csv`:`cleaned_mcc` | MATCH |
| "\texttt{nt\_enhancers} $0.514\to0.459$" | 494 | 0.459 | 0.4594 / 0.459419 | same | MATCH |
| "the identical procedure removes 0.0\% of its negatives" | 498–499 | 0.0 % | 0.0 (`n_removed` = 0) | `results/contamination_cleaning.csv`, row `drosophila_enhancers_stark` | MATCH |
| "leaves its TATA AUROC (0.539) and MCC unchanged" | 499 | 0.539 | 0.539 → 0.539; MCC 0.3983 → 0.3983 | same row, `orig_TATA_auroc`/`cleaned_TATA_auroc`, `orig_mcc`/`cleaned_mcc` | MATCH |
| "\texttt{human\_enhancers\_ensembl} is weaker/suggestive (excess $+10.8\%$)" | 500–501 | +10.8 % | 0.1083 | `results/contamination_analysis.csv`:`diff_pos_minus_neg` | MATCH |
| fig. `shuffled_neg` caption: "Seed 42, 1000-bootstrap" | 509 | 42, 1000 | seed 42, boot 1000 | `results/upgrades/shuffled_neg_run.log`; `shuffled_neg_interpretation.txt` footer | MATCH |
| fig. `route3_specificity` caption: "Across 0.8--1.5\,bits/pos ... Seed 42" | 517–518 | 0.8–1.5, 42 | 7 thresholds 0.8–1.5, seed 42 | `results/upgrades/route3_threshold_invariance.csv`; `route3_specificity_interpretation.txt` footer | MATCH |

## Results — Cross-architecture frozen-FM probe (lines 522–571)

| claim | tex line | value in tex | value in source | source file:column/row | VERDICT |
|---|---|---|---|---|---|
| "nested paired bootstrap, 1000 resamples" | 532 | 1000 | 1000 | `results/upgrades/fm_paired_interpretation.txt`; `dnabert_control_interpretation.txt` | MATCH |
| "\texttt{test\_effect} $= -0.048$ (95\% CI $[-0.053, -0.043]$) on \texttt{cohn}" | 535 | −0.048 [−0.053, −0.043] | −0.0481, "[-0.053,-0.043]" | `results/upgrades/fm_cross_architecture.csv`:`test_effect_auroc`,`ci`, row `HyenaDNA-tiny-16k/human_enhancers_cohn/sep/lgbm` | MATCH |
| "$-0.079$ ($[-0.107, -0.056]$) on \texttt{nt\_enhancers}" | 536 | −0.079 [−0.107, −0.056] | −0.0793, "[-0.107,-0.056]" | same file, row `HyenaDNA.../nt_enhancers/sep/lgbm` | MATCH |
| "(both CIs exclude 0)" | 536 | — | `significant=True` for both | same file, `significant` | MATCH |
| "the clean control \texttt{drosophila} is flat, $+0.005$ ($[-0.003, +0.014]$, CI includes 0)" | 536–537 | +0.005 [−0.003, +0.014] | 0.0055, "[-0.003,0.014]", `significant=False` | same file, row `HyenaDNA.../drosophila_enhancers_stark/sep/lgbm` | MATCH (see caveat C2) |
| "an LR head gives the same signs and significance" | 538 | — | cohn LR −0.0502 sig; nt LR −0.0971 sig; dros LR +0.004 n.s. | `results/upgrades/fm_head_invariance.csv`:`sign`,`significant`, `sep` rows | MATCH |
| "$-0.047$ ($[-0.051, -0.042]$, mean-pool)" on cohn | 543–544 | −0.047 [−0.051, −0.042] | −0.0467, "[-0.051,-0.042]" | `results/upgrades/fm_cross_architecture.csv`, row `DNABERT-2-117M/human_enhancers_cohn/mean/lgbm` | MATCH |
| "$-0.046$ ($[-0.052, -0.042]$, [CLS]) on \texttt{cohn}" | 544 | −0.046 [−0.052, −0.042] | −0.0465, "[-0.052,-0.042]" | same file, `.../cls/lgbm` | MATCH |
| "$-0.072$ ($[-0.100, -0.049]$)" on nt_enhancers | 544–545 | −0.072 [−0.100, −0.049] | −0.0720, "[-0.100,-0.049]" | same file, `DNABERT-2.../nt_enhancers/mean/lgbm` | MATCH |
| "$-0.077$ ($[-0.106, -0.051]$)" on nt_enhancers | 545 | −0.077 [−0.106, −0.051] | −0.0771, "[-0.106,-0.051]" | same file, `.../cls/lgbm` | MATCH |
| "all CIs exclude 0, both poolings, both heads" | 545–546 | — | all 8 DNABERT-2 rows `significant=True` | same file | MATCH |
| "we remove the \emph{same number} of test negatives that the GC-matched cleaning removes (1148)" | 548–549 | 1148 | 1148 | `results/upgrades/dnabert_control.csv`:`n_removed` (all rows) | MATCH |
| "This placebo is flat across all four head/pooling combinations (every \texttt{test\_effect} CI includes 0)" | 550–551 | 4, all include 0 | −0.0006, +0.0001, +0.0004, −0.0002; all 4 CIs straddle 0 | `results/upgrades/dnabert_control.csv`, rows `arm=placebo` | MATCH |
| "whereas removing the composition-biased negatives costs $-0.047$" | 551–552 | −0.047 | −0.0467 | same file, row `arm=gc_match,mean,lgbm` | MATCH |
| "each shuffled negative has the same per-sequence GC as its positive, so GC-matched cleaning removes nothing" | 553–556 | — | degeneracy documented | `results/upgrades/dnabert_control_degeneracy.csv`; `dnabert_control_interpretation.txt` | MATCH |
| fig. `fm_cross_architecture` caption: "LightGBM head, 95\% bootstrap CIs ... Seed 42" | 566–569 | lgbm, 95 %, 42 | `head=lgbm` rows; seed 42 | `results/upgrades/fm_cross_architecture.csv`:`head`; run logs | MATCH |

## Results — Robustness (lines 573–581)

| claim | tex line | value in tex | value in source | source file:column/row | VERDICT |
|---|---|---|---|---|---|
| "A 5-seed LightGBM loop" | 575 | 5 seeds | `n_seeds`=5, seeds [42,43,44,45,46] | `results/upgrades/multiseed.csv`:`n_seeds`; `multiseed_interpretation.txt` footer | MATCH |
| "max SD 0.013 across eight key tasks" | 576 | 0.013, 8 tasks | max `mcc_sd` = 0.0130 (drosophila); 8 rows | `results/upgrades/multiseed.csv`:`mcc_sd`, `task` | MATCH |
| "the solved/hard tier stable for 7/8" | 576–577 | 7/8 | `tier_stable`: 7 True, 1 False | `results/upgrades/multiseed.csv`:`tier_stable` | MATCH |
| "the lone flip, \texttt{nt\_enhancers}, sits on the 0.5 cutoff" | 577 | — | `nt_enhancers` `tier=VARIES`, mcc_min 0.4767 / mcc_max 0.5097 | same file | MATCH |
| "agrees on 13/15 tasks for LR vs LightGBM" | 578 | 13/15 | "lr vs lgbm : 13/15 = 87% agree" | `results/upgrades/route_invariance_interpretation.txt`; derivable from `route_invariance.csv`:`lr_tier` vs `lgbm_tier` | MATCH |
| "and 3/3 for LightGBM vs the frozen FM-probe head" | 578 | 3/3 | "lgbm vs fm : 3/3 = 100% agree" | same | MATCH |
| "disagreements are capacity-driven or 0.5-boundary" | 579 | — | `human_ensembl_regulatory` (LR 0.429 vs LGBM 0.907), `nt_enhancers` (0.437 vs 0.514) | `results/upgrades/route_invariance.csv`:`lr_mcc`,`lgbm_mcc` | MATCH |
| "the FM-probe independently agrees the enhancer tasks are hard" | 579–580 | — | `fm_tier` = hard/hard/partial for cohn/drosophila/nt_enhancers | `results/upgrades/route_invariance.csv`:`fm_tier` | MATCH |

## Discussion / Limitations (lines 583–623)

| claim | tex line | value in tex | value in source | source file:column/row | VERDICT |
|---|---|---|---|---|---|
| "the latter's $R^2$ CI includes 0 at $n{=}11$" | 602–603 | [−0.27,+0.70] ∋ 0, n=11 | [−0.269, +0.701], 11 tasks | `results/upgrades/route1_predict_interpretation.txt`; `route1_predict.csv` (11 rows) | MATCH |
| "We fixed the flagging criterion, overlap test, and thresholds before obtaining the coordinates" | 598–599 | — | pre-registration file exists and states the CONFIRMED/REFUTED rule | `localization/PRECOMMITTED_RULES.txt` | MATCH |
| "the result refuted our initial core-promoter prior" | 599 | — | "VERDICT: REFUTED" | `localization/contamination_localization.txt` | MATCH |
| "an $8$-mer-Jaccard audit" | 609 | 8-mer | "exact 8-mer Jaccard, full scale" | `results/upgrades/homology_leakage_interpretation.txt` (F003 header) | MATCH |
| "the flagship \texttt{cohn} redundancy-clean (0.11\% $\ge$0.7 near-duplicate)" | 609–610 | 0.11 % | 0.00115 (= 0.115 %) | `results/upgrades/homology_leakage.csv`:`frac_ge_0_7`, row `human_enhancers_cohn` | MATCH (truncated) |
| "\texttt{human\_nontata\_promoters} (41\% $\ge$0.7)" | 610 | 41 % | 0.40635 | same column, row `human_nontata_promoters` | MATCH |
| "\texttt{nt\_enhancers} (25\% verbatim)" | 610–611 | 25 % | 0.25 | `results/upgrades/homology_leakage.csv`:`frac_ge_0_7`; `nt_enhancers_duplication.csv`:`train_test_verbatim_frac` | MATCH |
| "cleaned train/test splits ... (TATA-flag, GC-matched, and composition-equalized variants)" | 615–617 | 3 variants | `arm` ∈ {original, comp_equalized, gc_match, tata_flag} | `results/upgrades/composition_clean.csv`:`arm`; `cleaned_splits_v2/` | MATCH |
| "the frozen-FM embedding cache with manifest" | 617–618 | — | manifest exists | `results/upgrades/fm_embeddings_manifest.csv`; `reproduce_from_cache.log` | MATCH |

## Conclusion (lines 625–637)

No new numbers; all quantities restate values audited above (three routes, two enhancer
benchmarks, two frozen FMs). **MATCH** by reference.

---

## Caveats (not scored as MISMATCH, but material for the rewrite)

**C1 — "uncorrelated" overstates the presence null (lines 84, 362).** The value −0.03 is
correct (`presence_null_stats.csv`:`rho`, row `GB_n9`), but the same file gives its 95 % CI
as Fisher `[-0.682, +0.645]` / bootstrap `[-0.735, +0.684]`, and the repo's own
`presence_null_interpretation.txt` says: *"'uncorrelated' overstates what n=9 supports; the
defensible claim is that no association is DETECTABLE, not that none exists."* The same file
also notes the enrichment counts were computed at the **old selection-on-test best k**, which
differs from the validation-selected k on 5 of 13 tasks; a k-consistent pairing gives −0.23
(`results/motif_interpretation.txt`) and the pooled n=15 value is −0.245
(`presence_null_stats.csv`, row `pooled_n15`). All are near zero and non-significant, so the
qualitative claim survives — but the word "uncorrelated" does not trace to the data.

**C2 — the HyenaDNA "flat control" is pooling-conditional (line 537).** `+0.005 [-0.003,
+0.014]` is exactly right for the `sep` pooling reported in the paper. Under `mean` pooling
the *same* `drosophila` control is significantly negative:
−0.0385 `[-0.047,-0.030]` (LGBM) and −0.0188 `[-0.033,-0.006]` (LR)
(`results/upgrades/fm_head_invariance.csv`, rows `drosophila_enhancers_stark/mean`). The
paper claims head-invariance (true) but a reader checking the released head-invariance CSV
will find the control is not flat in both poolings. Worth a sentence.

**C3 — two result files carry superseded benchmark MCCs.** `results/bootstrap_cis.csv`
(`kind=benchmark`) and `results/stats_summary.txt` (PART A.2), plus
`results/discriminability_summary.csv` (`bench_mcc`) and `results/nt/discriminability_summary.csv`,
were computed at the **old selection-on-test k** and disagree with Table 1 on 7 of 15 tasks
(e.g. `drosophila` 0.445 vs 0.398, `dummy_mouse` 0.622 vs 0.529, `nt_splice_sites_all` 0.278
vs 0.244, `nt_enhancers` 0.533 vs 0.514, `nt_promoter_tata` 0.859 vs 0.846). **The paper is
correct** — Table 1 traces cleanly to `cross_suite_summary.csv` / `val_selection.csv` — but a
reviewer opening `bootstrap_cis.csv` first will think the table is wrong. Consider stating in
REPRODUCIBILITY.md which file is authoritative for Table 1.

**C4 — `results/cleaning.log` is superseded by `results/cleaning_rerun.log`.** The former
reports `nt_enhancers` at k=5 with MCC `0.533->0.435`; the paper's `0.514->0.459` comes from
the rerun at the validation-selected k=4 (and from `contamination_cleaning.csv`). Same hazard
as C3.

**C5 — rounding.** Several figures are truncated rather than rounded (`0.4355 -> 0.435`,
`0.4945 -> 0.494`, `0.00115 -> 0.11%`, `-96.5 -> -96`, `0.8645 -> 0.864`). Individually
harmless and internally consistent with the interpretation `.txt` files, but if the .tex is
regenerated with round-half-up these five values will shift by one unit in the last place.

---

# Addendum — numbers added by the PSB revision

Every value below is read from the CSV named beside it. Written after the audit above; the
audit's three NOT-FOUND findings (the unreleased 61--100% saturation range, the untested
"seed-invariant" claim for TATA-flag cleaning, and the asserted −25..−30 bp offset for
`nt_promoter_tata`) were removed from the manuscript rather than defended.

| claim | value | source |
|---|---|---|
| motif-presence null, all 15 tasks | rho −0.245, 95% CI [−0.673, +0.306], p 0.379 | `presence_null_stats.csv`, row `pooled_n15` (Fisher CI; BC bootstrap [−0.649, +0.302]) |
| motif-presence null, 9 GB tasks | rho −0.033, CI [−0.682, +0.645] | `presence_null_stats.csv`, row `GB_n9` (reproduces the published −0.03) |
| fine-tuned HyenaDNA, released split | AUROC 0.752 (seeds 0.741–0.760) | `fm_finetune_summary.csv`, `original`→`original` |
| same weights, composition-matched negatives | 0.514 | `fm_finetune_summary.csv`, `original`→`comp_equalized` |
| retrained on GC-matched / equalized | 0.662 / 0.505 | `fm_finetune_summary.csv`, in-arm rows |
| `nt_enhancers` on non-redundant test | MCC 0.514→0.311, AUROC 0.850→0.735, n 400→300 | `homology_purged.csv` |
| `human_nontata_promoters` on non-redundant test | MCC 0.874→0.587, AUROC 0.989→0.961, n 9034→5363 (91% positive) | `homology_purged.csv` |
| `human_ocr_ensembl` equalized | MCC 0.421→0.237, GC AUROC 0.569→0.496, 24.9% negatives retained, max\|SMD\| 0.228→0.033 | `composition_clean_specificity{,_balance}.csv` |
| `human_enhancers_ensembl` equalized | MCC 0.593→0.373, residual over composition +0.174 unchanged, 19.4% negatives retained | `composition_clean_specificity.csv` |
| HyenaDNA `drosophila` control under mean pooling | −0.0385 [−0.047, −0.030] | `fm_head_invariance.csv` — now stated in the paper, since the control is flat only under [SEP] pooling |
