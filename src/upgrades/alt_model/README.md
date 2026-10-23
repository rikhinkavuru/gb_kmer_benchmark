# alt_model — are the route labels an artefact of k-mer spectra + LightGBM?

Answers **Reviewer 2, comment 3** (BIBM 2026, paper B290): *"justify the choice of k-mer spectra and
LightGBM as the interpretable model, clarify whether the conclusions depend on this particular choice,
and report whether the route labels are preserved under an alternative interpretable model."*

Every **model-dependent** route diagnostic is re-derived with two alternative interpretable models
on the k-mer spectrum at LightGBM's validation-selected k, with the same splits, the same caps and
seed 42. The paper's LightGBM is refit through the identical code path as the reference. Only the
**classifier** is varied, not the representation (representation dependence is the FM-probe /
fine-tuning modules' job). Model-free diagnostics (motif-only AUROCs, TATA negative excess, GC AUROC,
coordinate analyses, control-motif panel) are identical under any classifier and are not recomputed.

| model | what it is | inputs | tuning |
|---|---|---|---|
| `lgbm` | the paper's LightGBM (reference), fixed hyperparameters (`models.LGBM_PARAMS`), deterministic | all 4^k k-mer columns | none |
| `lr` | L1- or L2-regularised logistic regression (liblinear; one-vs-rest for the 3 multiclass tasks) | all 4^k standardized k-mer columns | penalty and C over {3e-4 … 0.1} × {L1, L2}, chosen by validation MCC on a stratified 80/20 carve of train, refit on all of train |
| `ebm` | Explainable Boosting Machine (interpret 0.7.8; GA2M, Lou et al. 2013; InterpretML, Nori et al. 2019) | the 20-d composition signature + the **128** k-mers with the largest train-only ANOVA F (a filtered *subset* of the spectrum) | fixed: 4 outer bags, 10 pairwise terms, 256 bins; internal early stopping |

Post hoc variants (labelled `posthoc` everywhere): `lrx` = LR with the C grid extended to 10;
`lra` = LR tuned by validation AUROC (MCC tuning degenerates to an all-zero model on the ~95%-positive
equalized splits); `ebm512` = EBM given 512 k-mers (capacity check of the filter); `ebmpos`
(fix round 1, P9) = the EBM's Route-2 B=8 arm with 128 global + 128 position-binned k-mers ranked
separately (the pre-registered joint filter admitted **no** binned k-mer on 3 tasks). `ebmpos`'s
global selection equals the EBM B=1 selection (verified on all 16 units, `alt_model_checks.csv`), so
its B=8 design strictly contains the B=1 design.

The **pre-registered analysis plan** (diagnostics, caps, decision rules, preservation criterion) is
the docstring of `run_alt_model.py`, written before any alternative model was fit and unchanged;
post hoc addenda P1–P8, the fix-round-1 addendum P9–P10 and the fix-round-2 addendum P11 (analysis
only) follow it in the same docstring.

## Files

| file | role |
|---|---|
| `cache_seqs.py` | validated pickle cache of every original split (built through the patched `data.load_dataset`; counts asserted against `results/results{,_nt}.csv` on every load; manifest with sha1) |
| `run_gate.py` | model-free Route-3 gate (TATA negative excess ≥ 0.10 and TATA motif-only AUROC < 0.5) for the 12 binary tasks → `results/upgrades/alt_model_gate.csv` |
| `alt_models.py` | the three model families (+ variants, incl. `ebm_design_pos` for P9), fast exact MCC/AUROC |
| `run_alt_model.py` | worker: one unit = (task, arm, feature set, model); test-set probabilities checkpointed to `external/alt_model/units/` (resumable) |
| `run_dose_alt.py` | Route-1 semi-synthetic dose-response (135 settings) refit with each model → `external/alt_model/dose_<model>.csv` |
| `check_shuffle_hashseed.py` | reproducibility check of the dinucleotide-shuffled negatives across `PYTHONHASHSEED` values |
| `fix_float32.py` | housekeeping: removes units stored with float32 probabilities so they are refit in float64 |
| `analyze_alt_model.py` | all statistics, route labels, agreement → `results/upgrades/alt_model_*.csv` |
| `make_interpretation_alt.py` | `results/upgrades/alt_model_interpretation.txt` + `alt_model_paper_table.csv` |
| `run_all.sh`, `run_post.sh`, `run_fix1.sh`, `run_fix2.sh` | the ordered queues (first run, pinned-shuffle/lrx completion, fix round 1, fix round 2 = analysis only) |
| `tests/test_alt_model.py` | metric identities vs scikit-learn, bootstrap-weight identity, two-level arm-weight identity, random-arm construction = psm_sensitivity, equalized-split mapping, cache counts, P9 design superset, P11 overlap-unit choice and overlap-rule lookup (10 tests) |
| `deferred/` | the one deferred unit (uncapped multiclass EBM on regulatory) |

Outputs in `results/upgrades/`: `alt_model_units.csv` (every fitted unit with CIs),
`alt_model_bench.csv`, `alt_model_route2.csv`, `alt_model_route3_prereg.csv`,
`alt_model_route3_sizecontrolled.csv`, `alt_model_route3_contrast.csv`, `alt_model_route3_overlap.csv`,
`alt_model_route3_overlap_contrast.csv`, `alt_model_route3_shuffled.csv`, `alt_model_route1.csv`,
`alt_model_dose.csv`, `alt_model_attribution.csv`, `alt_model_labels.csv`, `alt_model_agreement.csv`,
`alt_model_checks.csv`, `alt_model_gate.csv`, `alt_model_shuffle_hashseed.csv`,
`alt_model_paper_table.csv`, `alt_model_interpretation.txt`.

## Reproduce

All compute goes through `bibm/lowcpu.sh` (nice 20, 3 BLAS/OpenMP threads). From the repo root:

```bash
PY=/Users/rikhinkavuru/Downloads/venv/bin/python
nohup bibm/lowcpu.sh src/upgrades/alt_model/run_all.sh > external/alt_model/logs/run_all.log 2>&1 &
# then: shuffled arms on one pinned draw, lrx edge completions, analysis (see "Reproducibility finding")
nohup bibm/lowcpu.sh src/upgrades/alt_model/run_post.sh > external/alt_model/logs/run_post.log 2>&1 &
# fix round 1: P9 units (EBM B=8 with binned k-mers guaranteed), full analysis (P10), tests
nohup bibm/lowcpu.sh src/upgrades/alt_model/run_fix1.sh > external/alt_model/logs/run_fix1.log 2>&1 &
# fix round 2: analysis only (P11), ~8.5 min, then the 10 tests
nohup bibm/lowcpu.sh src/upgrades/alt_model/run_fix2.sh > external/alt_model/logs/run_fix2.log 2>&1 &
# individual steps
bibm/lowcpu.sh $PY src/upgrades/alt_model/check_shuffle_hashseed.py 16
bibm/lowcpu.sh $PY src/upgrades/alt_model/analyze_alt_model.py          # ~8.5 min, writes every CSV + interpretation
bibm/lowcpu.sh $PY src/upgrades/alt_model/make_interpretation_alt.py    # seconds: interpretation + paper table from the CSVs
bibm/lowcpu.sh $PY -m pytest -q src/upgrades/alt_model/tests
```

`run_post.sh`'s last unit (uncapped EBM on regulatory) is deferred (`deferred/run_ebm_regulatory_full.sh`).
`run_all.sh` fits `lrx` on every tier-1..5 unit (a superset of the upper-edge units P4 names; all are used).
Runtime on the 16 GB MacBook Air (3 threads): pre-registered units ≈ 1.6 h (the multiclass EBM on
`human_ensembl_regulatory` alone ≈ 27 min), dose-response ≈ 25 min, random-negative arms ≈ 1.7 h,
LR variants ≈ 2 h, EBM-512 ≈ 25 min, uncapped refits ≈ 2.5 h, P9 units ≈ 11 min (plus the deferred
~9-10 h EBM unit); analysis ≈ 8.5 min. Peak RAM < 3 GB.

## Why k-mer spectra + gradient-boosted trees (for the paper)

* **k-mer spectra** are alignment-free and fixed-length, and every feature is a short oligomer that
  can be scored directly against a TF position-weight matrix — which is what lets the paper name the
  TFs a classifier relies on. k-mer and gapped-k-mer SVMs are the established interpretable models
  for enhancer/regulatory-sequence prediction and recover known TF motifs (Lee et al. 2011,
  doi:10.1101/gr.121905.111; Ghandi et al. 2014, doi:10.1371/journal.pcbi.1003711).
* **Gradient-boosted trees** capture non-linear k-mer effects on the same features at CPU cost and
  admit exact per-feature attributions (TreeSHAP; Lundberg et al. 2020, doi:10.1038/s42256-019-0138-9).
  The paper's per-TF enrichment uses split **gain**, not SHAP; its attribution profile holds up across
  models only partly (see Attribution below). Non-linearity matters in our data: on the 3-class
  `human_ensembl_regulatory` task a tuned one-vs-rest L1/L2 LR on identical features reaches MCC
  0.389 (0.391 with C up to 10; 0.402 uncapped; the paper's multinomial LR floor reaches 0.429,
  `route_invariance.csv`) versus 0.899 (capped) / 0.907 (uncapped) for LightGBM, while the mostly
  additive EBM (148 main effects + 10 pairwise terms) reaches 0.879 — so the gap is non-linear
  per-feature effects, not high-order interactions.
* An **Explainable Boosting Machine** (Lou et al. 2013, doi:10.1145/2487575.2487579; implementation
  InterpretML, Nori et al. 2019, arXiv:1909.09223) is the glass-box alternative used here.

## Results (numbers from `results/upgrades/alt_model_*.csv`; 85/85 sanity checks pass)

**Reference reproduced** (`alt_model_checks.csv`). The LightGBM reference refit through this pipeline
reproduces `results/results{,_nt}.csv` to 6 decimals on the 9 uncapped tasks and on all 6 uncapped
`bench_full` refits (e.g. regulatory 0.906674), `positional_sweep.csv` for all 15 tasks, the published
equalized-split values, the uncapped splice rescue 0.244 -> 0.868, psm_sensitivity's overlap-weighting
shares (cohn 0.6437, drosophila 0.2905) exactly, and — with uncapped MCCs — the paper's Table corr
(n=9 +0.867, n=11 +0.736).

**Labels (`alt_model_agreement.csv`, pre-registered rules, capped as pre-registered).**
LR and the EBM each give the paper's label on 14/15 tasks (EBM-512 15/15; EBM-pos 14/15). Under the
**primary** rules all 8 failure labels keep their label under every model; the single change per model is a
*solved* task the weaker model does not solve: `human_ensembl_regulatory` under LR (MCC 0.389 [0.383,
0.396]) and `human_nontata_promoters` under the EBM (0.673 [0.658, 0.686]; 0.717 [0.703, 0.730] with 512
k-mers). Same counts with uncapped training. **What this test can detect:** a weaker model keeps a
shared-motif (residual) label almost automatically — it would lose it only by solving the task or by
falsely firing the positional or contamination rule — so the informative model-dependent checks are the
positional rule (14 non-splice tasks), the splice rescue and the Route-3 readouts.

*How contamination is decided.* A contamination label needs **both** the model-free TATA gate **and** a
model-dependent Route-3 effect. The gate is positive on four binary tasks (`alt_model_gate.csv`: cohn,
nt_enhancers, human_enhancers_ensembl, human_nontata_promoters). On the gate-negative control drosophila
the gate blocks a false label where the pre-registered equalized-MCC collapse also fires (EBM and LR-ext;
EBM-512/EBM-pos inherit the EBM's equalized arms). Among the gate-positive *failing* tasks the
model-dependent half decides contamination vs shared-motif, and the rule-3 variants differ there
(`gate_positive_failing_labels`):
* primary (equalized-MCC collapse): cohn and nt_enhancers contamination, human_enhancers_ensembl
  shared-motif, under every model;
* pre-registered sensitivity (composition fraction >= 0.80): the EBM, EBM-512 and EBM-pos relabel
  human_enhancers_ensembl as contamination (fraction 0.864 / 0.830 vs 0.743 for LightGBM), so a failure
  label changes (7/8), and the EBM variants that fail nontata label it contamination;
* post hoc overlap rule (P11b; overlap-design contrast vs the control, below): cohn and nt_enhancers
  contamination and human_enhancers_ensembl shared-motif under every model (same label 15/14/14/15/14,
  failure labels 8/8); the EBM variants that fail nontata label it contamination (+0.201 [0.134, 0.260]);
* submitted-BIBM vocabulary (contamination = failing + gate, graded by excess): same counts as primary, but
  the EBM moves nontata from solved to "contamination (weak)", so BIBM labels are **not** classifier-free.

Replacing only the contamination *effect* by the post hoc AUROC criterion (size-matched random negatives,
leak-free test for nt_enhancers) preserves 14/15 (LightGBM), 13/15 (LR), 13/15 (EBM): `nt_enhancers` is
**not** supported under any model, LightGBM included (36 test negatives). Under the overlap criterion the
counts are 15/14/14. Under the BIBM vocabulary the submitted "contamination (weak)" label of
human_enhancers_ensembl fails the overlap criterion under every model (it loses *less* than the control).

**Route 2 (`alt_model_route2.csv`).** Splice is the only task meeting the positional rule under
LightGBM and LR on all 15 tasks: dMCC(B8-B1) = +0.601 [0.571, 0.629] (LightGBM), +0.681 [0.654, 0.711]
(LR); uncapped +0.624 / +0.700. Largest |dMCC| on the other 14: LightGBM 0.028 (enh_ensembl, CI
excludes 0); LR 0.099 (mouse, n=242, CI [-0.000, +0.199]) and 0.035 among CI-significant ones.
**EBM caveat:** its pre-registered filter gave the B=8 design **zero** binned k-mers on
demo_coding_vs_intergenomic_seqs, human_enhancers_cohn and nt_enhancers_types (dMCC = 0 by
construction — the test is uninformative there) and only 2 on nontata, mouse and enh_ensembl
(`n_bin_features_B8`). The EBM gives +0.440 [0.409, 0.470] on splice and meets the rule nowhere else.
With 128 binned k-mers guaranteed (EBM-pos, P9) the EBM test is informative on all 15 tasks: splice
+0.450 [0.418, 0.480] (uncapped +0.465), no other task meets the rule (largest |dMCC| 0.034, mouse, CI
includes 0; largest CI-significant 0.027, ocr).

**Route 3 (`alt_model_route3_*.csv`).** (a) *The pre-registered MCC readout is non-diagnostic.* The
equalized-split MCC collapses for cohn and nt_enhancers under every model, but it also collapses onto the
composition-only baseline for the clean control under the EBM and LR-ext. Size-matched random negatives
alone produce most of the cohn MCC drop under every model: share of the drop 1.06 [0.87, 1.26] (LightGBM;
0.461 -> 0.041 random vs 0.064 matched), 0.76 [0.66, 0.87] (LR), 0.82 [0.73, 0.92] (EBM); nt_enhancers
0.75 [0.57, 0.94] / 0.36 [0.04, 0.68] / 0.57 [0.31, 1.09] (full test). For the control the random arms
*raise* MCC (LightGBM 0.398 -> 0.445), so the same metric mixes a size effect on one task with none on
another. For LR the composition-only fits on every equalized split are degenerate (all-zero, MCC 0), so
for LR "collapses onto the composition baseline" only means "equalized k-mer MCC CI reaches 0".
(b) *Random-negative design* (two-level paired bootstrap): cohn loses AUROC beyond size-matched random
negatives under every model (+0.152 [0.110, 0.193] LightGBM, +0.170 [0.124, 0.212] LR, +0.132 [0.091,
0.172] EBM) — but so does the clean control drosophila (+0.101 [0.063, 0.142], +0.090 [0.047, 0.133],
+0.108 [0.065, 0.149]); the cohn-minus-control difference in this design (~220 test negatives each) is
significant only for LR (+0.305 [0.091, 0.529]; LightGBM +0.187 [-0.014, 0.390], EBM +0.074 [-0.128,
0.289]); leak-free nt_enhancers is n.s. under every model (+0.087, +0.062, +0.093; 36 test negatives).
(c) *Removal-free overlap-weighting design* (psm_sensitivity's primary design; P8, extended in P11a to
every gate-positive task and ocr with uncapped `bench_full` units): cohn loses a larger share of
above-chance AUROC than the control under every model (+0.353 [0.293, 0.413], +0.292 [0.228, 0.353],
+0.419 [0.350, 0.487]); nt_enhancers too (direction only: its overlap-weighted AUROC falls *below* 0.5,
shares 1.0-1.9 — the linear-propensity weights over-correct there). The gate-positive but uncontaminated
human_enhancers_ensembl loses *less* than the control (-0.122 [-0.169, -0.079], -0.137 [-0.195, -0.087],
-0.047 [-0.103, +0.004]), as does ocr (-0.112, -0.138, -0.074, all CIs < 0). This is the
contamination-specific evidence: it separates cohn/nt_enhancers from both the clean control and the
gate-positive human_enhancers_ensembl under every model (EBM borderline for enh_ensembl). Caveat: the EBM
receives the 20-d composition signature as explicit inputs, so its shares are larger by design (nontata:
0.547 EBM vs 0.061 LightGBM). (d) Dinucleotide-shuffled negatives drive the composition-only AUROC to
0.500 — a **sanity check** that holds for any classifier by construction (each shuffle has its positive
twin's exact composition), not model-robustness evidence.

**Route 1 (`alt_model_dose.csv`, `alt_model_route1.csv`).** The causal dose-response holds under every
model: pooled slope +1.99 [1.92, 2.06] (LightGBM), +1.95 [1.88, 2.01] (LR), +1.81 [1.73, 1.87] (EBM);
all 9 per-motif CIs exclude 0. The real-task correlation (uncapped MCCs, n=9 homology-clean) is
+0.87 [0.26, 1.00] for LightGBM (= the paper; Spearman t-approximation p = 0.0025, identical to
`clean_subset_correlation.csv` `pooled_p`, which the paper prints as 0.003; exact permutation p = 0.0045)
but positive and **not significant** for the alternatives: LR +0.13 [-0.59, 0.94] (p_t 0.73), EBM +0.47
[-0.28, 0.94] (p_t 0.21) (n=11: +0.74 / +0.27 / +0.47). At n=9 it hinges on how a few tasks are ordered:
LR fails regulatory, and for every model the largest jackknife change at n=9 comes from dropping
nt_enhancers_types, a task LightGBM also fails (0.476); for LR the regulatory drop is nearly as large, and
at n=11 it is LR's largest. (The pre-registered capped values are +0.83 / +0.25 / +0.47; `_uncapped` rows
are written only for models with an uncapped refit in the set — not for lra/ebm512.) The presence null holds under every model (all CIs include 0),
also with each model's own enriched-TF counts.

**Attribution (`alt_model_attribution.csv`, post hoc).** Per-TF enrichment z-profiles from each
model's own importances correlate with LightGBM's (median Spearman 0.76 LR, 0.83 EBM), but the
FDR-enriched sets overlap only partly (median Jaccard 0.25 / 0.37; top TF identical on 0/15 and 3/15
tasks). On nontata, SP1 ranks 1 / 6 / 4 and SP2 2 / 4 / 8 (LightGBM / LR / EBM). The EBM profile is
restricted to its 128 filter-selected k-mers, so it partly reflects the univariate filter.

**Reproducibility finding (`alt_model_shuffle_hashseed.csv`).** `shuffled_neg/dinuc_shuffle.py`
iterates `set(seq)`, so the shuffles depend on `PYTHONHASHSEED`. Across 16 hash seeds (13 of the 24
'ACGT' set orders) the paper's model-free numbers are stable (cohn TATA AUROC 0.549-0.554, paper
0.551; GC 0.500) but the k-mer MCC varies (nt_enhancers 0.52-0.63, paper 0.545), and the exact draw
behind `shuffled_neg.csv` (committed 2026-05-31, before the data-integrity fix) was **not recovered**.
Fix: iterate `sorted(set(s))` and pin the seed (shared code, not changed here).

## Deviations and limitations

* Everything marked post hoc (P1-P10 in the `run_alt_model.py` docstring) was added after the
  pre-registration; the pre-registered rules and criteria are unchanged and reported as primary.
* Only the classifier is varied; the EBM sees a 128-k-mer subset (EBM-512: 512); its nontata failure
  is this filter (recovered at 512).
* LR tuned by validation MCC degenerates to an all-zero model on every composition-only equalized (5/5)
  and shuffled (3/3) fit and on 9 of the 50 random-negative fits (all cohn); those LR composition-only
  readouts are uninformative (flagged `lr_degenerate`; AUROC readouts repeated with `lra`). The LR k-mer
  fits are never degenerate.
* The random-negative arms use seeds 0-9 (P1 as pre-registered), independent of psm_sensitivity's
  draws (10000+r): LightGBM's random-arm AUROC agrees with psm's to max |diff| 0.008 (check tolerance
  0.02; cohn 0.766 vs 0.765, nt 0.799 vs 0.807, drosophila 0.761 vs 0.754; `alt_model_checks.csv`) — do
  not quote both as the same number.
* Six tasks are capped at 50,000 training sequences as pre-registered; uncapped refits (`bench_full`)
  leave every label unchanged. **Deferred:** the single uncapped multiclass EBM on
  `human_ensembl_regulatory` (~9-10 h here) — see `deferred/README.md`; it cannot change a label
  (capped EBM 0.879 [0.876, 0.883]) and the uncapped rows use the capped EBM for it.
* `--quick` (200 resamples) is for development only; all reported numbers use 1000.

## Fix round 1 (2026-09-27) — what changed

Uncapped Route-1 statistics (P10a); P1's two-level bootstrap implemented as specified (paired over the
original test sequences, seeds resampled; P10b); random-arm MCC columns (P10c); post hoc
contamination criteria and the control-collapse flag in the label bookkeeping, leak-free readout for
nt_enhancers (P10d); EBM-pos units (P9); `n_bin_features_B8` in `alt_model_route2.csv`; point estimates
in `alt_model_route3_contrast.csv`; wording of every claim flagged by the verifiers.

## Fix round 2 (2026-09-27) — what changed (analysis only, P11)

The overlap design is extended to every gate-positive task and ocr (uncapped units; P11a) and applied as
a post hoc labelling rule (`posthoc_overlap`, P11b); the contamination-label logic is stated correctly
(gate AND model-dependent effect; the sensitivity rule changes a failure label under the EBM; BIBM labels
change when a weaker model fails a gate-positive task). The random-arm MCC flag is renamed
`mcc_comp_effect_ci_includes_0`, and the size share of the MCC drop gets a two-level CI (P11c). Route-1
`_uncapped` rows are written only where an uncapped refit exists, with the Spearman t-approximation p
next to the permutation p (P11d). `effect_ok` records the post hoc readouts for tasks without an equalized
split. Also: the LR-degeneracy wording, the psm tolerance wording, the jackknife n=9/n=11 wording, and
the nt_enhancers over-correction wording.
