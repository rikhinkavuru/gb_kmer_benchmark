# psm_sensitivity: is the composition-equalized collapse caused by shortcut removal?

This module answers Reviewer 1, comment 4 (BIBM 2026 camera-ready). The reviewer asks whether the
drop in k-mer performance after composition-equalized cleaning comes from **removing the composition
shortcut**, or from two other things:

- **extreme negative-set reduction**: only 4.5% of `cohn` and 4.9% of `nt_enhancers` negatives survive;
- **test-set distortion**: the cleaned test sets are about 94% positive.

The reviewer also asks for sensitivity curves across matching strictness and a check for selection bias.

Datasets: `human_enhancers_cohn` and `nt_enhancers` (contaminated), and `drosophila_enhancers_stark`
(clean control). The classifier is the paper's k-mer LightGBM at the validation-selected k (5 / 4 / 4).
A composition-only LightGBM on the 20-d mono+dinucleotide signature is fit alongside it. The
propensity model is identical to the paper's: `run_composition_clean.fit_propensity`, a logistic
regression on the pooled 20-d signature, seed 42. Selection is done within each split.

The pipeline reproduces the paper's numbers exactly before doing anything new:
`tests/test_reproduces_paper.py` checks that the cached `original` and published-split
(`paper_equalized_matched`) predictions give the AUROC and MCC of `results/upgrades/composition_clean.csv`
(12 of 12 values match to 6 decimals, e.g. cohn k-mer AUROC 0.804649 -> 0.613457, MCC 0.461223 -> 0.064163).

## What is computed

| Part | What | Arms / families |
|---|---|---|
| (1) strictness sweep | tempered propensity-histogram matching; alpha = 0 keeps every negative, alpha = 1 **is** the paper's 20-bin proportional match rule (verified bin-for-bin in tests), re-drawn with selection seeds 42-44; the paper's published split is scored separately (`paper_anchor`). 12 levels span 93% to 4.5% retention on cohn. 3 selection seeds. | `F1_hist` |
| (1) secondary sweep | common-support trimming to the positives' central [q, 1-q] logit-propensity band | `F2_trim` |
| (2) size-matched random | at **every** level of every family: the same per-split counts drawn uniformly at random, 10 seeds, nested across levels | `kind=random` |
| (3) imbalance-robust metrics | AUROC; MCC at 0.5 (the paper's metric); balanced accuracy; class-balance-weighted MCC at 0.5 (`mcc_bw`) and at the prior-corrected threshold t = training positive fraction (`mcc_bw_pc`, "balanced MCC"); PR-AUC of both classes with their prevalences; MCC mean/sd over 20 class-balanced test draws (`mcc_bal20`, `mcc_bal20_pc`) | all rows |
| (4a) removal-free | overlap weights (pos 1-e, neg e) and ATT odds weights on the ORIGINAL train and test sets: weighted LightGBM training, weighted metrics. Placebo: the same weights permuted within class (10 seeds) | `W_overlap`, `W_att` |
| (4b) two-sided | greedy 1:1 nearest-neighbour matching on the logit propensity without replacement. Calipers {2, 1, 0.5, **0.2**, 0.01} x SD. Control: the same number of random positives + random negatives | `F3_1to1` |
| (4c) positive-side | all negatives kept; positives subsampled to the negatives' propensity histogram. Control: random positives | `F4_posside` |
| (5) fixed model | `design=fixed_model`: the model trained once on the original split, scored on each arm's test subset (matched vs size- and class-ratio-matched random). `design=train_only`: the arm-trained model on the ORIGINAL test. A 2x2 cross (matched/random model x matched/random test) at every F1 level and for the paper's published split | `design` column |
| (6) selection bias | features OUTSIDE the propensity model, compared for retained vs discarded negatives (and positives, and size-matched random negatives): length, N fraction, trinucleotide L1 distance to the positive centroid, trinucleotide Markov-1 residual, TATA / SP1 GC-box hit rate and count (1.3 bits/pos), DUST-like low-complexity fraction. For cohn and the genomic subset of nt: GENCODE v44 TSS overlap / distance, RepeatMasker coverage, ENCODE cCRE v4 overlap (any / ELS / PLS) | `selection_bias`, `selection_drift` |
| label-noise threat | (a) `F5_ccrefree`: every cCRE-overlapping negative removed, then paper matching vs random; (b) `F5b_ccrefree_1to1`: two-sided 1:1 on the same purged pool; (c) `*_ccrefreeTest` designs: the same models scored with cCRE-overlapping test negatives removed; (d) a mixture-model ESTIMATE (not a worst case) of how much hidden positives could lower AUROC, assuming hidden positives overlap ENCODE ELS at the labelled-positive rate | cohn (a, b); cohn + nt test (c) |
| POST HOC P0 | verbatim train/test duplicate audit (all three datasets) | `psm_sensitivity_leakage_check.csv` |
| POST HOC P1 | nt_enhancers leak-free TEST: every arm re-scored with the same cached models after removing the 100 test positives that occur verbatim in train (primary nt result) | dataset `nt_enhancers_leakfreeTest` |
| POST HOC P2 | nt_enhancers leak-free REFIT: every arm refit without the 107 training rows equal to a test sequence; full test | dataset `nt_enhancers_leakfreeRefit` |
| POST HOC P3 | class-balanced training weights for F1 a=1 (3 selection seeds), its 10 random arms and the published split | families `F1bw_hist`, `paperbw_anchor` |
| POST HOC P4 | between-dataset contrast: share of above-chance AUROC lost under overlap weighting, 1000 bootstraps | `psm_sensitivity_between_dataset.csv` |
| POST HOC P5 | selection bias of common-support trimming (F2 q0.3, q0.45) | `psm_sensitivity_selection_bias_f2.csv` |
| POST HOC P0b (fix round 1) | hg38 interval-overlap audit: test intervals overlapping the union of training intervals (cohn: all sequences; nt: genomic-mapped subset). Descriptive | `psm_sensitivity_leakage_interval.csv` |
| POST HOC P7 (fix round 2) | decomposition CIs at the paper's retention: two-level PAIRED bootstrap (level 1: the dataset's test sequences, shared by every arm, identical draws to P4; level 2: the 10 random seeds, and in secondary rows the 3 matched seeds) of the size effect, the matched-vs-random effect, its share and the relative gap, for the k-mer and composition-only models, within and train-only (undistorted test) designs, plus k-mer minus composition-only differences. Reading rules R7a-R7c and the fact that all point estimates were already known are in the `run_posthoc.py` docstring | `psm_sensitivity_decomposition_ci.csv`; `p7_*` columns of `psm_sensitivity_decision.csv` |
| POST HOC P7 extended (fix round 3) | the same P7 bootstrap for design `fixed_model`: the ORIGINAL model scored on each F1 alpha=1 arm's own test (matched vs random test subsets; no training change), with R7a / R7c. Column `n_test_neg_scored` (was `n_test_neg_matched42`) and `tau_R7b` added | `design=fixed_model` rows of `psm_sensitivity_decomposition_ci.csv`; `p7_fixed_*` columns |
| POST HOC P8 (fix round 3) | F2 construction check: fraction of positives outside the kept negatives' propensity band, and AUROC of in-band vs out-of-band test positives against each arm's negatives (retrained and original models; matched seed 42 vs the mean of 10 random arms). Reading rule R8 fixed in the `run_posthoc.py` FIX-ROUND-3 ADDENDUM before it ran | `psm_sensitivity_f2_band.csv` (merged into `psm_sensitivity_f2_selection_difficulty.csv`) |
| POST HOC P6 (fix round 1) | composition-normalised between-dataset contrast: the composition-only LightGBM under the SAME overlap / placebo weights and the SAME 1000 resamples as P4 (k-mer draws re-derived and asserted bit-identical). Per dataset: composition-only share lost, k-mer minus composition-only share, k-mer / composition-only share; per contrast: difference of contrasts (additive scale) and ratio difference (multiplicative scale). Reading rule and the fact that the additive outcome was already known are in the `run_posthoc.py` docstring | extra columns in `psm_sensitivity_between_dataset.csv` |

Why overlap weights rather than plain inverse-propensity weights: the contaminated sets have very poor
overlap (cohn negatives' propensities pile up near 0), so IPW/ATT weights are dominated by a handful of
negatives. Overlap weights are bounded in [0, 1] and give exact mean balance on the logistic-model
covariates (Li, Morgan & Zaslavsky 2018; verified in `tests/test_psm_core.py`). ATT odds weights are
reported with their effective sample size and flagged unstable when the negatives' ESS < 5%.

## Pre-registered decision rule

The full plan, including its revision history, is in the `run_psm_sensitivity.py` docstring. The rule
was fixed before any outcome was computed.

Primary: F1 at alpha = 1 (the paper's retention), k-mer AUROC, within-arm design.

- **(i-a) size**: A0 - mean Ar <= tau, where tau = max(0.03, half-width of the original AUROC CI).
- **(i-b) separation**: Am < min over seeds of Ar, and the upper CI of Am < mean Ar.
- **(ii-a) weighting**: overlap-weighted AUROC below every permuted-weight placebo, its upper CI below
  the placebo mean, and the drop >= 0.5 x the F1 shortcut drop.
- **(ii-b) two-sided**: the same three conditions for 1:1 matching at caliper 0.2 vs its random 1:1 control.

Verdicts:

- SUPPORTED: all four hold.
- SUPPORTED-WITH-SIZE-COMPONENT: all but (i-a) hold.
- NOT SUPPORTED: (i-b) fails, or both (ii-a) and (ii-b) fail.
- MIXED: anything else.

The MCC decomposition and the label-noise checks are reported but do not drive the verdict.

## Results

The pre-registered run finished 2026-09-24 05:39 (all 991 arms, B = 1000). The post hoc analyses
(`run_posthoc.py`, written after two verification reviews and labelled post hoc everywhere) finished
07:21; the fix-round-1 additions P0b and P6 ran 16:43-16:45, the fix-round-2 addition P7 ran 17:27-17:30, and
the fix-round-3 additions (P7 extended to the fixed_model design, P8) ran 18:14:35-18:17:08 and, after a
docstring-only edit, again from the final file 18:30:05-18:32:07 with byte-identical output (no refits; cached
predictions only; the within / train_only P7 rows are bit-identical to fix round 2). Every number here is
read from `psm_sensitivity_{decision,paper_table,summary,between_dataset,decomposition_ci,leakage_check,
leakage_interval,f2_selection_difficulty,f2_band,caliper_sweep,dissociation_stats,cross2x2,selection_bias}.csv`. The full report, with a headline block and a "what this
experiment shows and does not show" block first, is `psm_sensitivity_interpretation.txt`.

Terminology: the pre-registration calls Ar_mean - Am the "shortcut" term. It measures what composition
MATCHING removes relative to size-matched random negatives, whatever the cause, so it is reported here as the
**matched-vs-random effect**.

**Data leakage in nt_enhancers (read this first).** 100 of the 400 nt test sequences (all positives,
labels consistent) occur verbatim in the training split. The original model scores them 0.958 on average,
against 0.624 for the other positives. Negative-side selection never drops a positive, so these copies sit
in every nt arm. Without them the original AUROC is 0.735 instead of 0.850. cohn and drosophila have no
train/test duplicates. The hg38 interval audit (P0b) finds the same 100 nt test sequences and no others,
but only within the genomic-mapped subset (400 of 400 nt test and 3,933 of 14,968 nt training sequences map
to hg38), so interval leakage against the other 74% of nt training sequences cannot be assessed. In cohn
(every sequence mapped), 47 test negatives (0 positives; 21 by >= 50%) overlap a training interval, which
moves the original AUROC from 0.805 to 0.804 and touches 1 of the 228 matched test negatives. The leak-free
rows (`nt_enhancers_leakfreeTest`, same models, primary; and `nt_enhancers_leakfreeRefit`) are the nt
numbers to report. The uncorrected `nt_enhancers` rows are kept for transparency only.

**Verdicts under the pre-registered rule** (`psm_sensitivity_decision.csv`, column `verdict`)

| dataset | verdict | criteria that failed |
|---|---|---|
| cohn | SUPPORTED-WITH-SIZE-COMPONENT | (i-a) only: random reduction costs 0.040 AUROC, above tau = 0.030 |
| nt_enhancers, leak-free test (primary nt) | NOT SUPPORTED | (i-b): matched 0.684 is below every random seed (min 0.689), but its CI upper bound 0.782 is above the random mean 0.723 (36 test negatives) |
| nt_enhancers, leak-free refit | NOT SUPPORTED | (i-b): matched 0.706, CI upper 0.788 vs random mean 0.786 |
| nt_enhancers, uncorrected | NOT SUPPORTED | (i-a) and (i-b); inflated by the leakage above |
| drosophila (clean control) | SUPPORTED | none |

**cohn, k-mer AUROC at the paper's retention (4.5% of negatives kept).**

- original 0.805 [0.795, 0.814]
- size-matched random negatives 0.765 (10 seeds, 0.738-0.793)
- composition-matched (F1 alpha = 1, selection seed 42, re-drawn) 0.621 [0.582, 0.658]; selection seeds
  42-44 give 0.613-0.637; the paper's published split gives 0.613 [0.577, 0.654]

The total drop, 0.184, splits into a size effect of 0.040 (22%) and a matched-vs-random effect of 0.144
(78%). With the post hoc two-level paired bootstrap (P7: test sequences x random seeds, B = 1000;
`psm_sensitivity_decomposition_ci.csv`) the size effect is 0.040 [0.022, 0.059] (22% of the drop, [12%, 33%])
and the matched-vs-random effect 0.144 [0.102, 0.184] (78% [67%, 88%]). So sample loss is a real but minor
part of the drop: its point estimate exceeds the pre-registered tolerance of 0.030 (hence "with size
component"), and its CI excludes 0 but straddles 0.030. It is not an artefact of imbalance handling: with
class-balanced training weights (post hoc) the size effect is 0.043 and the matched-vs-random effect 0.158
(random 0.761 vs matched 0.603 [0.564, 0.643]).

Along the F1 sweep the matched-vs-random gap grows as balance improves. The random curve falls from 0.806
(93% kept) to 0.765 (4.5% kept), i.e. by 0.041, which is the size effect; the matched curve falls to 0.621.

| retained | max\|SMD\| of matched set | gap (random - matched) |
|---|---|---|
| 93% | 0.99 | +0.001 |
| 18% | 0.51 | +0.082 |
| 4.5% | 0.07 | +0.144 |

**Caliper sweep (the literal "PSM threshold" curve; pre-registered part 4b, `psm_sensitivity_caliper_sweep.csv`,
`figures/psm_sensitivity_caliper.pdf`).** Two-sided 1:1 matching keeps the same fraction of each class, so the
test stays balanced. On cohn, random 1:1 subsets of the same size stay at 0.800-0.805 at every caliper, while the
matched gap grows as the caliper tightens and balance improves, then plateaus once the matched set is balanced:

| caliper (SD) | kept per class | max\|SMD\| matched | matched | random 1:1 | gap |
|---|---|---|---|---|---|
| 2 | 89% | 0.74 | 0.765 | 0.805 | +0.040 |
| 1 | 73% | 0.31 | 0.702 | 0.802 | +0.101 |
| 0.5 | 65% | 0.10 | 0.693 | 0.800 | +0.108 |
| 0.2 | 61% | 0.04 | 0.688 | 0.802 | +0.114 |
| 0.01 | 60% | 0.03 | 0.694 | 0.802 | +0.107 |

**Positive-side matching keeps every negative.** Subsampling the POSITIVES to the negatives' propensity
histogram (F4; all negatives kept; 7.7% of positives; test 225 positives / 3,474 negatives) gives 0.622
[0.585, 0.659] against 0.780 (0.749-0.798) for random positive subsets of the same size, a gap of 0.157, close
to the 0.144 of negative-side matching. No negatives are filtered in this design.

**cohn, designs that discard nothing, keep the test undistorted, or keep both classes balanced**

| design | composition-balanced | control |
|---|---|---|
| model retrained with overlap weights on the original split; nothing discarded (negatives' ESS 79%) | 0.671 [0.656, 0.685] | permuted-weight placebo 0.805 |
| original model, not retrained, on the overlap-weighted original test | 0.609 [0.593, 0.624] | placebo 0.806 |
| arm-trained model on the ORIGINAL, undistorted test (only the training negatives differ) | matched-trained 0.584 [0.571, 0.597] | random-trained 0.765 (0.758-0.773) |
| two-sided 1:1 caliper-0.2 matching (keeps 61% of EACH class, discards the rest) | 0.688 [0.673, 0.704] | random 1:1 subsets of the same size 0.802 |
| positive-side matching (keeps every negative) | 0.622 [0.585, 0.659] | random positive subsets 0.780 |

P7 for the undistorted-test (train-only) design: matched-vs-random 0.181 [0.166, 0.195], training-size effect
0.040 [0.034, 0.046].

Weighting diagnostic: the composition-only LightGBM still reaches 0.605 (retrained) / 0.554 (original
model) under the overlap weights, so the weights under-correct for cohn and the weighted drops are
conservative. Overlap weights balance only the linear propensity terms; they do not "equalize" composition.

**Matching removes most, not all, composition-associated signal, and the k-mer loss does not exceed the
composition-only loss.** The composition-only LightGBM falls from 0.779 to 0.581 [0.544, 0.624] on the
published cohn split and to 0.549 [0.508, 0.585] on the seed-42 re-draw; both CIs exclude 0.5. At the paper's
retention (P7, matched-vs-random effect, k-mer vs composition-only):

| design | k-mer | composition-only | k-mer minus composition-only (absolute; relative) | R7c |
|---|---|---|---|---|
| within (retrained, own test) | 0.144 [0.102, 0.184] | 0.184 [0.138, 0.223] | -0.039 [-0.086, +0.007]; -0.245 [-0.433, -0.068] | not distinguishable |
| fixed_model (original model, matched vs random test) | 0.209 [0.169, 0.249] | 0.229 [0.187, 0.270] | -0.020 [-0.047, +0.004]; -0.141 [-0.232, -0.052] | not distinguishable |
| train_only (retrained, original test) | 0.181 [0.166, 0.195] | 0.216 [0.199, 0.233] | -0.036 [-0.054, -0.017]; -0.247 [-0.318, -0.179] | k-mer loses LESS |

In the train_only design the k-mer model loses less than the composition-only model in every dataset variant,
so that row is a property of the design, not evidence about cohn. Across the 24 non-underpowered F1/F2/F3 arms
(each once), cohn's k-mer gap correlates with the composition-only gap more than with selection depth
(-log retained fraction) in every design: within +0.96 vs +0.39, fixed_model +0.98 vs +0.71, train_only
+0.92 vs +0.65 (only the within difference reaches the 0.3 rule; `psm_sensitivity_dissociation_stats.csv`).

**Test-set distortion (cohn).** Changing the class ratio alone does not move AUROC (+0.001 for the original
model on a random test subset of the same size and 94%-positive ratio). This is expected, because AUROC does
not depend on prevalence, so it is a sanity check rather than evidence. The substantive answer is the
undistorted-test design: a model trained on matched negatives loses 0.181 on the untouched original test
against one trained on random negatives. The original model also loses 0.209 on the composition-matched test
subset (0.595 vs 0.803) and 0.197 on the overlap-weighted full test, with nothing removed. The 2x2 cross agrees:
a model trained on random negatives scores 0.765 on the random test and 0.579 on the matched test.

**MCC (the reviewer's metric).** The paper's MCC@0.5 collapse (0.461 -> 0.064 on the published split) is
reproduced fully by size-matched RANDOM negatives for the k-mer model (0.022, range 0.000-0.091) and mostly
for the composition-only baseline (0.410 -> 0.144, range 0.089-0.203, vs 0.000 on the published split: 65% of
its collapse). So it is not evidence of shortcut removal: on a ~94%-positive test a model trained on ~96%
positives predicts almost everything positive. Even the class-balanced MCC at the prior-corrected threshold
falls mostly through the size/imbalance part (0.462 -> random 0.117 -> matched 0.034: size part 0.344,
matched-vs-random part 0.083). MCC isolates the matched-vs-random effect only in the balanced 1:1 design, where
it is a matched-vs-random contrast (not a before/after): 0.277 [0.250, 0.305] for matched vs 0.448 for random
1:1 (original 0.461). MCC@0.5 is uninformative on the class-balanced-training rows too: on the published cohn
split both the unweighted and the class-balanced model call 1 of 3702 test sequences negative and share MCC
0.064 (`tests/test_posthoc.py`, test 10), while their AUROCs differ (0.613 vs 0.587). Report AUROC against the
random control instead of the equalized MCC.

**Clean control (drosophila).** Matching also lowers the control's AUROC (original 0.763, random 0.754,
matched 0.674 [0.632, 0.719]; size effect 0.009 [-0.009, 0.028], matched-vs-random effect 0.081
[0.040, 0.123]; verdict SUPPORTED, while cohn's is SUPPORTED-WITH-SIZE-COMPONENT). The control has composition
signal of its own (composition-only AUROC 0.683), and in the paired designs matching removes it together with
the k-mer signal:

| design | k-mer | composition-only | R7c |
|---|---|---|---|
| within, selection seed 42 | 0.081 [0.040, 0.123] | 0.041 [-0.004, 0.090] | not distinguishable (+0.040 [-0.009, +0.092]; relative +0.064 [-0.189, +0.350]) |
| within, selection seeds 42-44 | 0.091 [0.058, 0.131] | 0.080 [0.031, 0.127] | not distinguishable |
| fixed_model | 0.072 [0.039, 0.111] | 0.076 [0.039, 0.116] | not distinguishable (-0.004 [-0.042, +0.033]) |
| train_only | 0.051 [0.033, 0.070] | 0.098 [0.075, 0.123] | k-mer loses LESS (design property, see above) |

Across the same 24 arms the control's k-mer gap correlates with its composition-only gap at +0.99 (fixed_model)
and +0.97 (train_only), but only +0.07 in the within design, where it tracks selection depth (+0.86). By the
post hoc reading rule printed in the interpretation file (in both paired designs R7c is not "k-mer loses MORE"
and the cross-arm Spearman is >= 0.9), the control's loss is consistent with matching removing
composition-associated signal from the clean control too. Only the retrained seed-42 comparison hints at an
extra k-mer loss, and it is neither distinguishable (R7c) nor stable across selection seeds. Selection-induced
difficulty is not shown. On the published split the control keeps signal above composition in AUROC: k-mer
0.661 [0.623, 0.702] vs composition-only 0.549 [0.508, 0.595]. Either way, the experiment shows that removing
composition-ASSOCIATED signal lowers AUROC, including in the control, not that the removed signal is an
artefact.

**Between-dataset contrast (post hoc P4 + P6, `psm_sensitivity_between_dataset.csv`).** Under overlap
weighting the unretrained original model loses 64% [61, 68] of its above-chance AUROC on cohn and
29% [25, 34] on drosophila (k-mer contrast 0.353 [0.293, 0.413]). But the same weights also remove much
more of cohn's composition signal: the composition-only LightGBM loses 81% [76, 86] of its above-chance
AUROC on cohn and 45% [38, 55] on drosophila (contrast 0.352 [0.247, 0.442]).

| design | dataset | k-mer share lost | composition-only share lost | k-mer / comp.-only |
|---|---|---|---|---|
| original model (primary) | cohn | 0.644 [0.606, 0.685] | 0.806 [0.760, 0.855] | 0.80 [0.77, 0.83] |
| original model (primary) | drosophila | 0.290 [0.248, 0.336] | 0.454 [0.376, 0.545] | 0.64 [0.52, 0.75] |
| retrained | cohn | 0.441 [0.403, 0.477] | 0.622 [0.577, 0.665] | 0.71 [0.64, 0.78] |
| retrained | drosophila | 0.257 [0.209, 0.310] | 0.364 [0.278, 0.460] | 0.71 [0.54, 0.95] |

Per unit of composition signal removed (cohn minus drosophila):

- additive scale (difference of contrasts): original model +0.002 [-0.070, +0.088]; retrained
  -0.074 [-0.186, +0.031];
- ratio scale: original model +0.159 [+0.040, +0.272]; retrained +0.002 [-0.263, +0.191].

The rule fixed in the `run_posthoc.py` P6 docstring (both scales must exclude 0 in the primary design) is
NOT met. The k-mer contrast is (on the additive scale fully) accounted for by cohn's larger removable
composition signal; the only sign of greater per-unit reliance is on the ratio scale in one design. So the
contrast is NOT evidence that the drop is specific to an artefact ("shortcut"), and even a greater per-unit
reliance would not separate artefact from biology.

**What this experiment shows and does not show.**

- Shows: most of cohn's drop (0.144 of 0.184, 78%) comes from composition matching; sample loss accounts for
  0.040 (22% [12%, 33%]; CI 0.022-0.059, above the 0.030 tolerance as a point estimate); changing the test's
  class ratio accounts for nothing (+0.001, as expected for AUROC). Evidence: size-matched random negatives;
  the removal-free, fixed-model and undistorted-test designs; positive-side matching, which keeps every
  negative (0.622 vs 0.780); the F1 strictness sweep and the caliper sweep (random 1:1 flat at 0.800-0.805);
  the k-mer gap following the composition-only gap across arms in every design; and a k-mer loss that does
  not exceed the composition-only loss.
- Does NOT show: that the removed composition signal is an artefact rather than biology. Matching removes
  composition-associated signal indiscriminately and not completely; the clean control loses AUROC too, in
  step with its own composition-only signal in the paired designs; and the between-dataset contrast is
  explained by how much composition signal is removed. The artefact reading rests on the paper's independent
  evidence (dinucleotide-shuffled negatives, the information-content-matched control-motif panel, coordinate
  analysis), which this module does not re-test.

**nt_enhancers, leak-free (primary nt result).** At 4.9% retention only 36 test negatives remain, and the
matched set is NOT composition-balanced (max|SMD| 0.305; the published nt split 0.291; composition-only
AUROC 0.564 vs random 0.698), so the primary comparison is weak by construction:

- original 0.735 [0.670, 0.790]; random 0.723 (0.689-0.751); size effect 0.011 [-0.029, 0.052] (the
  uncorrected analysis's size component, 0.043, is largely leakage)
- matched 0.684 [0.587, 0.782]; selection seeds 0.575-0.684; published split 0.628; matched-vs-random
  effect 0.039 [-0.077, 0.145] (P7; not distinguishable from 0); leak-free refit 0.079 [-0.012, 0.181]
- original model on the matched test 0.535 vs random test 0.735 (P7 fixed_model 0.200 [0.117, 0.294];
  composition-only 0.236 [0.149, 0.329])
- two-sided 1:1 (balanced, max|SMD| 0.074; 32% of each class): 0.594 [0.506, 0.679] vs 0.729
- overlap weighting 0.406 [0.319, 0.496] vs placebo 0.736, but the weights OVER-correct on the leak-free
  test (composition-only AUROC under the same weights 0.472 retrained / 0.285 original model), so the
  weighted AUROC is below chance and only its direction is interpretable; the "share lost" (>1) is not a
  share. Scope of the over-correction across nt variants (composition-only AUROC under the weights,
  retrained / original model): uncorrected 0.606 / 0.436, leak-free test 0.472 / 0.285, leak-free refit
  0.523 / 0.395; i.e. below 0.5 in every original-model diagnostic and in the leak-free-test retrained one.
- 2x2 cross: the matched-trained model reaches 0.684 on the matched test, the random-trained model 0.465.
  Because the matched set keeps composition imbalance, this is NOT evidence of learnable non-composition
  signal. The earlier claim to that effect is withdrawn.
- The published nt equalized split is not composition-balanced either (max|SMD| 0.286 uncorrected, 0.291
  leak-free; composition-only AUROC 0.637 [0.517, 0.745] / 0.566 [0.450, 0.693]).

Label noise (nt): the mixture-model estimate is 0.44x the matched-vs-random effect on the leak-free test and
0.28x on the leak-free refit (below the 0.5 rule), and 2.97x on the uncorrected data, whose matched-vs-random
effect is only 0.008. None of these is informative: the ELS enrichment of kept over discarded nt negatives
behind the estimate is not significant (0.131 vs 0.076, Bonferroni p = 1.0, 236 kept negatives with
coordinates), and the nt composition-matching effect itself is NOT SUPPORTED in every variant. So for nt
label noise is neither shown nor excluded, and it does not bear on any nt conclusion.

**Common-support trimming (F2): what it does and does not show (post hoc, descriptive; fix round 3).**
F2 keeps every positive but only the negatives whose logit-propensity lies inside the positives' central
[q, 1-q] band, so 2q of the positives lie outside the kept negatives' propensity range (P8,
`psm_sensitivity_f2_band.csv`: 60%, 80% and 90% of test positives at q0.3, q0.4, q0.45). It therefore removes
composition signal inside the band and creates it for the positives outside the band, and it cannot isolate
retention in either direction. The earlier readings "trimming that keeps the composition signal leaves AUROC at
the random level" (cohn) and "propensity selection alone lowers k-mer AUROC while composition-only AUROC stays
flat" (control) held only for models retrained on the trimmed split and are withdrawn as general statements:

- cohn, q0.45 (5.5% of negatives kept): retrained k-mer 0.772 vs random 0.774 and retrained composition-only
  0.919 vs 0.740 (it RISES). With the original model on the trimmed test the same trimming costs k-mer 0.181 and
  composition-only 0.235; with the trimmed-split model on the original test 0.266 and 0.280. Split by band
  (P8, within design), among IN-band positives both retrained models lose (k-mer 0.203, composition-only 0.226
  vs random); among OUT-of-band positives the composition-only model gains (-0.225) and the k-mer model is level
  (-0.021). Where the kept negatives resemble the positives in composition, both models lose, as under matching.
- drosophila, q0.3 / q0.4 / q0.45 (25% / 13% / 6% kept): retrained k-mer gap 0.132 / 0.091 / 0.167 vs
  composition-only 0.004 / -0.023 / 0.014 (flagged in the within design by the post hoc rule k-mer gap >= 0.05
  and composition-only gap <= 0.02). In the paired designs no level is flagged: original model on the trimmed
  test, k-mer 0.147 / 0.130 / 0.152 vs composition-only 0.148 / 0.180 / 0.193; trimmed-split model on the
  original test, 0.131 / 0.128 / 0.132 vs 0.183 / 0.149 / 0.122. P8 (within): among in-band positives both
  models lose (k-mer 0.164 / 0.119 / 0.142, composition-only 0.152 / 0.110 / 0.087; rule R8 "k-mer loses
  in-band beyond composition-only" at q0.45 only); among out-of-band positives the composition-only model gains
  at q0.3 and q0.4 (-0.093, -0.056; +0.005 at q0.45) while the retrained k-mer model loses (0.112 / 0.084 /
  0.170). So the within-design flag mixes composition removal inside the band with a retrained k-mer model
  that fails to gain on the positives left outside it; only that last part is specific to the k-mer model.
  The kept negatives are also closer to the positive trinucleotide centroid (SMD -0.45 vs discarded at q0.3).
- Point for practice (kept, and now better founded): a propensity-based negative selection changes a task in
  ways mean balance does not reveal, so every cleaned split should be benchmarked against size-matched random
  negatives and read in a paired design (original model or original test held fixed), not only after
  retraining.
- Across all non-underpowered F1/F2/F3 arms (each arm once), per design (`psm_sensitivity_dissociation_stats.csv`;
  Spearman of the k-mer gap with the composition-only gap vs with selection depth, -log retained fraction):
  cohn within +0.96 vs +0.39, fixed_model +0.98 vs +0.71, train_only +0.92 vs +0.65; nt leak-free +0.81 vs
  +0.18, +0.99 vs +0.63, +0.86 vs +0.63; drosophila +0.07 vs +0.86, +0.99 vs +0.76, +0.97 vs +0.81. The
  control's within-design dissociation does not carry over to the paired designs. Arms overlap and the random
  controls share one priority stream, so no p-values are given.

**Selection bias (part 6), F1 alpha = 1, retained vs discarded negatives.**

- cohn: TATA rate 0.346 retained vs 0.585 discarded (positives 0.352; Bonferroni p 1e-29); ENCODE ELS
  overlap 0.337 retained vs 0.236 discarded (all negatives 0.241, positives 0.907; Bonferroni p 5e-6);
  TSS +/-500 bp overlap 0.043 vs 0.024, NOT significant after Bonferroni (p 0.77).
- Label noise (cohn): the mixture-model ESTIMATE (not a worst case) of the AUROC cost of hidden positives
  is 0.038 = 0.26x the matched-vs-random effect. Empirical cCRE-purged checks keep the drop: F5 re-match
  0.686 vs random 0.779; F5b 1:1 on the purged pool 0.755 vs 0.862 (drop 0.107 vs 0.114 unpurged); cCRE-free
  test 0.646 vs 0.817.
- nt: kept negatives are enriched for genomic-origin (hg38-mappable) sequences, 0.623 vs 0.213 discarded
  (positives 0.330); in nt, sequence length encodes provenance (the two SMDs are identical). The nt ELS
  (0.131 vs 0.076) and TSS (0.025 vs 0.009) differences are NOT significant after Bonferroni (p = 1.0).
- p-values in `psm_sensitivity_selection_bias.csv` are given uncorrected and Bonferroni-adjusted.

**Secondary families.** Positive-side matching (all negatives kept): cohn 0.622 vs 0.780; nt leak-free
0.464 vs 0.746 (17 test positives, underpowered); drosophila 0.662 vs 0.727. ATT odds weighting: cohn 0.629 vs
0.805 (negatives' ESS 16.8%); nt is unstable (ESS 0.08%) and not interpreted.

**Caveats.** Within-arm comparisons change the test set (unpaired); the fixed-model, undistorted-test and
weighting designs are the paired complements, and readings that hold only after retraining are labelled so.
The verdict uses selection seed 42 only, as pre-registered. The size-matched random controls of different
families share one priority stream (10000 + r), so they are nested, not independent. Only k-mer and
composition-only LightGBM were tested. P7's CIs resample test sequences and random / matched seeds but not the
training sets (the models are fixed). P8 is point estimates only.

## Outputs (results/upgrades/)

- `psm_sensitivity_data_check.csv`: loader vs snapshot identity, counts vs results.csv, sha256.
- `psm_sensitivity_sweep.csv`: every pre-registered evaluation row (dataset x arm x design x model):
  point metrics + 95% bootstrap CIs (1000 resamples, test sequence = unit) + class-balanced draws.
- `psm_sensitivity_balance.csv`: per arm: retained counts / fractions, ESS, max / mean |SMD| over the
  20 features, logit-propensity SMD, GC AUROC, TATA AUROC.
- `psm_sensitivity_posthoc_{sweep,balance}.csv`: the same, for the post hoc rows (leak-free nt,
  class-balanced training).
- `psm_sensitivity_summary.csv`: per (dataset, family, level, design, model): matched vs random-control
  mean / sd / min / max and the gap (pre-registered and post hoc rows).
- `psm_sensitivity_decision.csv`: the pre-registered decision quantities and verdict per dataset variant,
  plus reported-only columns (MCC, fixed model, label noise, class-balanced training `bw_*`, discarded
  ELS rate and its Bonferroni p `els_p_bonferroni`, weighting diagnostic `*_comp_auroc` /
  `*_weights_overcorrect`, between-dataset shares incl. the P6 `*_comp_share_lost` / `*_kmer_over_comp_share`).
- `psm_sensitivity_paper_table.csv`: compact table (one row per dataset x condition).
- `psm_sensitivity_cross2x2.csv`: 2x2 model x test decomposition at every F1 level and for the paper split.
- `psm_sensitivity_between_dataset.csv`: share of above-chance AUROC lost under overlap weighting, per
  dataset and cohn/nt minus drosophila contrasts, with bootstrap CIs (P4); P6 columns `comp_*`,
  `kmer_minus_comp_share*`, `kmer_over_comp_share*`, `diff_of_contrasts*`, `ratio_diff*`, `share_interpretable`.
- `psm_sensitivity_decomposition_ci.csv`: P7 decomposition CIs (one row per dataset x design x model x
  matched seeds; `model=kmer_minus_comp` rows carry the R7c reading).
- `psm_sensitivity_leakage_check.csv`: train/test duplicate audit (P0);
  `psm_sensitivity_leakage_interval.csv`: hg38 interval-overlap audit (P0b).
- `psm_sensitivity_f2_selection_difficulty.csv` (per design: k-mer and composition-only gaps, flags `sid_within`,
  `sid_fixed_model`, `sid_train_only`; P8 columns and R8 readings), `psm_sensitivity_f2_band.csv` (P8),
  `psm_sensitivity_selection_bias_f2.csv`: F2 trimming.
- `psm_sensitivity_caliper_sweep.csv`: two-sided 1:1 matching across calipers 2 -> 0.01 SD (k-mer and
  composition-only AUROC, matched vs random 1:1, balance).
- `psm_sensitivity_dissociation{,_stats}.csv`: post hoc, descriptive (`_stats`: one row per dataset x design).
- `psm_sensitivity_selection_bias.csv`, `psm_sensitivity_selection_drift.csv`: part (6).
- `psm_sensitivity_interpretation.txt`: human-readable report, generated from the CSVs.
- `figures/psm_sensitivity.{pdf,png}`: k-mer AUROC and balanced MCC vs retained-negative fraction
  (nt column = leak-free test). The last point of each matched curve is the paper's equalizer RULE re-drawn
  with selection seed 42 (cohn 0.621); the hollow star is the paper's PUBLISHED equalized split (cohn 0.613).
  The square is the model RETRAINED with overlap weights (cohn 0.671 vs placebo 0.805); the unretrained
  original model on the overlap-weighted test (cohn 0.609) is not plotted. Markers are dodged in x (matched
  x 1.08, random / 1.08, star / 1.12), so their x positions are approximate.
- `figures/psm_sensitivity_ntraw.{pdf,png}`: the same with the uncorrected (leaky) nt column (supplementary).
- `figures/psm_sensitivity_caliper.{pdf,png}`: supplementary caliper-sweep figure (k-mer AUROC and max|SMD| of the
  matched set vs caliper, matched vs random 1:1 of the same size; calipers drawn at equal spacing).
- `psm_sensitivity_run.log`, `psm_sensitivity_posthoc_run.log`: run logs.

Predictions are checkpointed per arm under `external/psm_sensitivity/preds/<label>/*.npz` and post hoc
parts under `external/psm_sensitivity/parts_posthoc/`, so every run resumes after an interruption.

## Reproduce

Python is `/Users/rikhinkavuru/Downloads/venv/bin/python` (torch-free). Every command goes through the
low-CPU wrapper `bibm/lowcpu.sh` (background QoS = efficiency cores, nice 20, 2 BLAS/OpenMP threads);
LightGBM uses 2 threads and one dataset is held in memory at a time. Measured peaks (`/usr/bin/time -l`):
`summarize_psm.py` 0.28 GB resident (24 s), `run_posthoc.py --parts P0b,P6` 0.53 GB resident / 0.73 GB
footprint (112 s), `run_posthoc.py --parts P7` 0.41 GB resident / 0.72 GB footprint (184 s), `run_posthoc.py
--parts P7,P8` 0.59 GB resident / 0.72 GB footprint (123 s; first run 0.55 / 0.76 GB, 153 s), `test_posthoc.py`
0.54 GB (32 s), other tests 0.20-0.42 GB.

```bash
cd /Users/rikhinkavuru/gbkb_release
W=bibm/lowcpu.sh; PY=/Users/rikhinkavuru/Downloads/venv/bin/python
$W $PY src/upgrades/psm_sensitivity/psm_data.py                         # data identity check (~1 min)
$W $PY src/upgrades/psm_sensitivity/tests/test_psm_core.py              # metric / matching unit tests
$W $PY src/upgrades/psm_sensitivity/tests/test_genomic_helpers.py       # interval helpers vs brute force
$W $PY -W ignore src/upgrades/psm_sensitivity/run_psm_sensitivity.py \
      > results/upgrades/psm_sensitivity_run.log 2>&1                   # pre-registered run; hours; resumable
$W $PY -W ignore src/upgrades/psm_sensitivity/tests/test_reproduces_paper.py   # cache reproduces the paper
$W $PY -W ignore src/upgrades/psm_sensitivity/run_posthoc.py \
      >> results/upgrades/psm_sensitivity_posthoc_run.log 2>&1          # post hoc P0-P8 (~1 h, mostly P2); resumable
      # (fix round 1 ran only  --parts P0b,P6  (~2 min, needs the P4 checkpoints);
      #  fix round 2 ran only  --parts P7  (~3 min, needs the prediction cache incl. the P2 refit cache);
      #  fix round 3 deleted external/psm_sensitivity/parts_posthoc/{decomp,f2band}_*.csv and ran  --parts P7,P8  (~2-2.5 min))
$W $PY -W ignore src/upgrades/psm_sensitivity/summarize_psm.py          # decision + tables + interpretation
$W $PY -W ignore src/upgrades/psm_sensitivity/tests/test_posthoc.py     # post hoc sanity checks (11 tests)
$W $PY src/upgrades/psm_sensitivity/make_fig_psm_sensitivity.py         # main figure
$W $PY src/upgrades/psm_sensitivity/make_fig_psm_sensitivity.py --nt-raw \
      --out results/upgrades/figures/psm_sensitivity_ntraw              # supplementary figure (uncorrected nt)
$W $PY src/upgrades/psm_sensitivity/make_fig_psm_sensitivity.py --caliper  # supplementary caliper-sweep figure
```

Tests (all pass, re-run 2026-09-24 after fix round 3): `test_psm_core` 6/6, `test_genomic_helpers` 3/3,
`test_reproduces_paper` 1/1 (12 of 12 paper values to 6 decimals), `test_posthoc` 11/11 (test 9: P7 point
estimates, now including the fixed_model rows and the composition-only paired-design rows, equal decision.csv;
test 10: MCC@0.5 on the published cohn split, independent sklearn recompute at p > 0.5; test 11: per-design F2
gaps equal summary.csv, the P8 band construction (2q of positives outside the band) with an independent sklearn
recompute of one P8 AUROC, the caliper sweep, and the per-design dissociation rows).

`--smoke <scratch dir>` runs a tiny grid into a separate cache for pipeline validation.
`--phase fit|eval` splits fitting from scoring.

Inputs:

- the three datasets through `data.load_dataset`, verified against the frozen snapshot;
- `cleaned_splits_v2/*_comp_equalized_*.csv`, the paper's published split, mapped onto the original indices;
- `localization/cohn/*.csv.gz` (cohn hg38 coordinates);
- `results/upgrades/neg_provenance_nt_map.csv` (nt_enhancers hg38 mapping);
- `external/neg_provenance/cache/{gencode_v44,rmsk_hg38,ccre_v4}.npz`, read only.
