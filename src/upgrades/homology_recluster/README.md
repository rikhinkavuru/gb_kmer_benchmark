# Homology re-clustering of all 15 benchmark tasks (Reviewer 1, comment 3)

> "Please re-cluster all benchmark datasets to remove sequence homology redundancy and re-run all
> baseline evaluations and intervention experiments, ensuring that memorization of near-duplicate
> loci does not inflate performance metrics or bias the route-labeling diagnosis."

This module (1) clusters train+test of every task with MMseqs2 on both strands, (2) measures
train/test and within-train redundancy, (3) re-scores the paper's best-k LightGBM under two
homology-clean protocols with controls, (4) re-runs the interventions (composition-only
baseline, composition equalization, dinucleotide-shuffled negatives, positional-bin rescue, Route-1
correlation) on homology-clean data, (5) re-derives every route label under a pre-registered
preservation rule, (6) re-runs the remaining paper experiments on homology-clean data (Route-3
cleaning arms with placebos, specificity panel, composition-only baseline on all tasks, frozen-FM probe,
graded positional rescue, LR floor, model seeds, motif presence, coordinates) and (7) re-runs the
end-to-end fine-tuning experiment (HyenaDNA-tiny) on homology-clean splits; DNABERT-2 LoRA on those
splits is packaged as a deferred job (`deferred/`). The analysis plan and decision rules are written in
each script's docstring before the script was first run; everything added afterwards is labelled post hoc.
Seed 42; 1000-resample percentile bootstrap (test sequence = unit; a component-level bootstrap is in the
audit). Everything is CPU-only and torch-free except the fine-tuning stage (`run_stage_ft_hc.sh`, a separate
torch/MPS process that never imports LightGBM).

Numbers: `results/upgrades/homology_recluster_interpretation.txt` (section 0 = headline),
`homology_recluster_extra_interpretation.txt` (X1-X10, FT-HC, addenda A1-A4) and
`homology_recluster_audit_interpretation.txt` (verification audit). All three are generated only from the CSVs.

## Method (short)
* **Homology graph.** `mmseqs search` all-vs-all, nucleotide mode (`--search-type 3`), **both strands**
  (`--strand 2`), `-k 11 -s 7.5 --max-seqs 1000 -e 1e-3`, coverage >= 0.8 of *both* sequences
  (`-c 0.8 --cov-mode 0`). Edges with identity (fident) >= T. **Clusters = connected components**
  (single linkage), so near-duplicates chained through intermediates can never straddle a split.
  * T = 0.8 (primary): the conventional nucleotide near-duplicate level (CD-HIT-EST's DNA range starts
    at 0.8). Unrelated 200-500 bp DNA aligns at about 25-45 %.
  * T = 0.5 (strict): all MMseqs2-*detectable* homology. Seeding limits recall below about 70 %
    identity (calibration: id50 recall 0.95-1.0 at 75 %, 0.70-0.85 at 70 %, 0.25-0.38 at 65 %), and
    id80 recall at exactly 80 % nominal identity is 0.68.
  * Why search and not linclust: on a synthetic calibration (`homology_recluster_calibration.csv`;
    planted mutants at 50-95 % identity on both strands + 400 random decoys), all-vs-all search +
    connected components recovers 100 % of >= 85 %-identity pairs on both strands with 0 false
    merges, and it was affordable for every task (largest: 289k sequences). linclust was calibrated on
    the same synthetic set (`homology_recluster_calibration_linclust.csv`, `run_linclust_calibration.py`):
    at id80 it recovers 0.90-0.97 of 95 %-identity pairs, 0.78-0.82 at 90 % and 0.57-0.72 at 85 %
    (+/- strand; kmer-per-seq 21 and 100), vs 1.00 for search + components, with 0/400 false merges for
    both. So no task used linclust.
  * Threshold-free exact-duplicate rate (identical or reverse-complement) is also reported, and
    `run_audit.py` Q3 adds exact duplicates that MMseqs2 cannot see (sequences shorter than the 11-mer
    seed, or mostly N): 644 masked all-N test sequences in demo_human_or_worm and 316 short ones
    (median 15.5 bp) in human_enhancers_ensembl (`homology_recluster_audit_task.csv`). They are reported
    as a sensitivity (purge MCC with them also removed), not merged into the primary graph.
  * Containment-coverage sensitivity (post hoc, `run_covmode_sensitivity.py`, all five variable-length
    tasks): `--cov-mode 1` (only the shorter sequence >= 80 % covered). Rule fixed before running: robust iff
    |homology effect(cov1) - homology effect(cov0)| <= 0.02; 9/10 task x threshold cells pass
    (`homology_recluster_covmode.csv`).
* **Protocol (a) test-purge.** The *same* trained model is scored on the provided test split minus
  every test sequence whose component contains a training sequence. It reports a paired delta vs
  the full test set, a prevalence-standardised MCC, and a **random-removal null** (1000 random
  subsets with the purged set's per-class counts). The null separates "the removed sequences were
  homologous" from "the test set shrank or changed its class mix". A purge that keeps < 10 test
  sequences of a class is `class_eliminated` (not evaluable; nt_enhancers_types at both thresholds). For
  binary nt_enhancers a second, subtype-matched null is reported because the purge removes every "strong"
  enhancer.
* **Protocol (b) cluster-aware re-split.** Pool train+test and assign whole components to train or
  test (seeded greedy, label-stratified, original test fraction), then retrain. The **control** is a
  label-stratified *random* re-split with identical targets, which isolates homology from "any
  re-split". k stays at the paper's validation-selected value, so only the split changes.
  Split seeds 42 (pre-registered), 43 and 44 (`run_seeds.py`); headline re-split numbers are split-seed
  means, and a re-split inflation call is "seed-robust" only if cluster < random for all three seeds.
* **Homology-clean (HC) data for interventions/diagnostics** is the cluster-aware re-split plus one
  representative per (component, label). This removes all MMseqs2-detectable train/test homology and
  within-split near-duplicate families (no pseudo-replication in pooled AUROCs). It does not remove
  exact twins MMseqs2 cannot seed: in the seed-42 id80 re-split, 716 demo_human_or_worm test sequences
  (masked, mostly N) and 314 human_enhancers_ensembl test sequences (a few bp long) have an exact
  train twin (`homology_recluster_audit_task.csv`, `n_resplit_test_with_exact_twin_in_train`).
* **Label rule** (reconstructed from the paper's Table I; verified to reproduce all 15 published
  labels on the provided data, `rule_reproduces_published` in `homology_recluster_route_labels.csv`):
  solved if MCC >= 0.7; else positional if dMCC(B8-B1) > 0.2 with CI > 0; else contamination if binary
  non-promoter, TATA neg-excess >= 0.15 and TATA AUROC CI < 0.5; else shared-motif.
  **Preservation rule** (pre-registered): a label is preserved if the diagnostic that assigns it
  keeps its sign and its 95 % CI still excludes 0 on HC data. Because that rule does not re-apply the
  MCC >= 0.7 gate to 'contamination'/'positional', a **strict** count (label rule re-applied, label
  unchanged) is reported next to it; the strict count is the one to quote. The MCC-based rules are
  evaluated with the re-split MCC (primary) and the prevalence-standardised purge MCC (secondary, chosen
  after the first eval results; raw purged MCC is also reported).
* **Re-pooling control for relabels** (`run_labels_addendum.py` A1, post hoc): a relabel is attributed
  to homology only if the label rule on the cluster re-split MCC differs from the label on the random
  re-split MCC (split-seed means primary).

## Fix history (post hoc; pre-registered results are kept and still reported)
Fix round 1 (2026-09-24), after independent code/claims verifiers:
* **Class elimination** (`run_eval.py`): purges that keep < 10 test sequences of a class are
  `class_eliminated`; excluded from purge label counts and from the purge Route-1 correlation.
* **Memorisation readout and subtype-matched null** (`run_eval.py`): accuracy of the same model on the
  removed (train-homolog) vs kept test sequences; subtype-matched null for nt_enhancers.
* **Strict label count** (`run_routes.py`): `preservation_strict`; both counts are reported.
* **Route-1 controls** (`run_routes.py`): the primary test is named (published discriminability vs the id80
  cluster re-split MCC, all 11 TF tasks); controls: random re-split, split-seed means, subsets without
  nt_enhancers_types, without both NT enhancer tasks, and the submitted paper's 9-task subset.
* **Inflation calls** (`make_report.py`): the pre-registered HOMOLOGY-INFLATED rule is reported first; the
  0.05 "material" cut is labelled post hoc; re-split inflation is also given as a seed-robust call.
* **Route-3 readout**: equalized-arm MCC is degenerate (about 96 % positive test set); the TATA / GC /
  k-mer AUROC is the readout. `run_e1_reconcile.py` replays composition_clean's random stream and
  reproduces the published arm exactly (k-mer MCC 0.0642); over 20 equally valid matching draws the
  equalized k-mer MCC ranges 0.000-0.064 while AUROCs are stable (`homology_recluster_e1_*`).
* **Verification audit** (`run_audit.py`, `make_report_audit.py`): per-class purge content, class
  elimination, exact duplicates missed by MMseqs2, identity of train-test homologs, component-level
  bootstrap CIs; `run_linclust_calibration.py`; `run_covmode_sensitivity.py`; `fix_checks.py` C2 (edge
  identity; the duplicate checks C1/C4/C5 were removed because the audit covers them).
* **Remaining experiments** (`run_extra.py` X1-X10, pre-registered in its docstring) and **end-to-end
  fine-tuning on HC splits** (`run_stage_ft_hc.sh`), both run to completion on 2026-09-25.

Fix round 2 (2026-09-26): no model was refit and `run_eval.py`'s logic did not change, so
`homology_recluster_eval.csv` (regenerated 2026-09-24 with the round-1 logic) stands.
* `run_audit.py` re-run from scratch on all 15 tasks (old checkpoints moved to
  `external/homology_recluster/ckpt/audit_prev_20260924/`): all three audit CSVs are identical to the
  previous run (max numeric difference 0).
* `run_labels_addendum.py` re-run now that X3 is complete: the submitted-paper 'composition' label check
  (A2) is filled in (it read "pending" before); nothing else changed.
* `make_report_extra.py`: FT-HC section applying `fm_finetune/eval_ft_v2.py`'s pre-registered rules 1-3 and
  5 to the HC contrasts next to the provided-split contrasts, with the not-evaluable guard fixed in
  `run_stage_ft_hc.sh` before training (< 10 test negatives in an eval arm); descriptive X3/X7 summary lines;
  the pre-registered X8 verdict line.
* `make_report.py`: headline lines for the frozen-FM probe, the LR floor and the HC fine-tuning.
* Reports, figure and README regenerated; tests re-run.

## Files
| file | content |
|---|---|
| `_hc.py` | shared helpers: count-asserted private GB loader, model factory with the thread cap, bootstrap, cluster-aware / random re-split, NR representatives, HC data builder |
| `run_cluster.py` | step 1-2: MMseqs2 graph, components (id80/id50), redundancy table, calibration (`--calibrate`) |
| `run_eval.py` | step 3: original / test-purge / cluster re-split / random re-split for all 15 tasks |
| `run_seeds.py` | step 3b: split seeds 43/44 for the cluster-aware and random re-splits (all tasks, id80+id50) |
| `run_interventions.py` | step 4: E1 composition-only + equalization, E2 shuffled negatives, E3 positional rescue |
| `run_routes.py` | step 4d/4e/5: route diagnostics before/after, labels + preservation, Route-1 Spearman, final table |
| `run_extra.py`, `run_stage_extra.sh`, `run_stage_fix1.sh`, `fetch_fm_cache.py` | step 7: X1-X10 on homology-clean data (Route-3 cleaning arms with placebos, specificity, composition-only on all tasks, frozen-FM probe, graded positional, positional at id50, LR floor, model seeds, motif presence, coordinates) |
| `prepare_ft_hc_splits.py`, `run_ft_hc.py`, `eval_ft_hc.py`, `run_stage_ft_hc.sh` | step 7b: end-to-end fine-tuning on the id80 HC splits (fm_finetune recipe and pre-registered scoring unchanged; HyenaDNA run here) |
| `deferred/` | DNABERT-2 LoRA fine-tuning on the HC splits (README with command, runtime, hardware; `run_deferred_dnabert_hc.sh`) |
| `run_labels_addendum.py` | step 8 (post hoc): re-pooling control for relabels, submitted-paper labels, class-level redundancy, Route-1 LOTO, compact paper table |
| `run_audit.py`, `run_e1_reconcile.py`, `run_linclust_calibration.py`, `run_covmode_sensitivity.py`, `fix_checks.py`, `make_report_audit.py` | step 9-10 verification audit (post hoc) |
| `make_report.py` | main interpretation text + figure, generated only from the CSVs |
| `make_report_extra.py` | X1-X10 + FT-HC + A1-A4 interpretation; applies the pre-registered decision rules mechanically |
| `run_stage_small.sh`, `run_stage_rest.sh`, `run_stage_final.sh`, `run_stage_extra.sh`, `run_stage_fix1.sh`, `run_stage_ft_hc.sh` | the exact driver scripts used (sequential, one heavy process at a time, `bibm/lowcpu.sh`) |
| `tests/test_homology_recluster.py`, `tests/test_extra.py`, `tests/test_audit.py` | sanity tests (no component straddles; stratification; purge null; class elimination; subtype null; loader identity; thread cap; HC FT splits; audit helpers) |

Outputs (`results/upgrades/`): `homology_recluster_{calibration,calibration_linclust,redundancy,eval,seeds,
seeds_summary,interventions,positional,routes,route_labels,route1,table}.csv`, `homology_recluster_interpretation.txt`,
`figures/homology_recluster_mcc.{pdf,png}`; step 7: `homology_recluster_extra_{cleaning,specificity,threshold,
composition,fm,graded,lr,modelseeds,presence,coords}.csv`; step 7b: `homology_recluster_ft_{split_manifest,runs,
metrics,summary,contrasts,verdicts,score_composition,score_composition_summary}.csv` and `homology_recluster_ft_preds/`;
step 8: `homology_recluster_{labels_addendum,class_redundancy,route1_loto,paper_table}.csv`,
`homology_recluster_extra_interpretation.txt`; step 9-10: `homology_recluster_fix_edges.csv`,
`homology_recluster_{audit_class,audit_task,audit_clusterboot,e1_reconcile,e1_draws,e1_draws_summary,covmode}.csv`,
`homology_recluster_audit_interpretation.txt`. Large intermediates (component ids, edge lists, per-unit
predictions, checkpoints, HC fine-tuning splits and checkpoints) live in `external/homology_recluster/`.

## Reproduce
Every command runs through `bibm/lowcpu.sh` (nice 20, 3 BLAS/OpenMP/LightGBM threads, performance cores
allowed; `LOWCPU_BG=1` restricts to the efficiency cores), strictly one heavy process at a time; MMseqs2
`--threads 3 --split-memory-limit 2G`. From the repository root:
```
PY=/Users/rikhinkavuru/Downloads/venv/bin/python
W=bibm/lowcpu.sh; M=src/upgrades/homology_recluster
$W $PY $M/run_cluster.py --calibrate                 # synthetic recall calibration (seconds)
$W $PY $M/run_cluster.py                             # all 15 tasks, smallest first (~1.5 h, checkpointed)
$W $PY $M/run_eval.py                                # original / purge / cluster + random re-splits
$W $PY $M/run_seeds.py                               # split seeds 43,44 (cluster + random re-splits)
$W $PY $M/run_interventions.py --exp E1,E2,E3 --tiers 1,2
$W $PY $M/run_routes.py                              # needs seeds_summary.csv for the seed-mean columns
$W $PY $M/fix_checks.py                              # post-hoc edge-identity check (C2)
$W $PY $M/run_audit.py                               # verification audit (~2 min; reads eval.csv inputs, no refits)
$W $PY $M/run_e1_reconcile.py; $W $PY $M/run_linclust_calibration.py; $W $PY $M/run_covmode_sensitivity.py
nohup $M/run_stage_extra.sh X10 X1 X2 X5 >> external/homology_recluster/logs/stage_extra.log 2>&1 &   # step 7, part 1
nohup $M/run_stage_fix1.sh >> external/homology_recluster/logs/stage_fix1.log 2>&1 &     # step 7, part 2: X4 X7 X3 X6 X8 X9 + cov-mode-1 search of human_ensembl_regulatory
$W $PY $M/prepare_ft_hc_splits.py                    # HC fine-tuning splits (external/homology_recluster/ft_splits/)
nohup $M/run_stage_ft_hc.sh >> external/homology_recluster/logs/stage_ft_hc.log 2>&1 &   # HyenaDNA on HC splits (MPS) + eval_ft_hc.py
$W $PY $M/run_labels_addendum.py                     # reads CSVs only (needs extra_composition.csv)
$W $PY $M/make_report_audit.py
$W $PY $M/make_report_extra.py
$W $PY $M/make_report.py                             # interpretation + figure
$W $PY -m pytest -q -p no:cacheprovider $M/tests
```
Deferred (see `deferred/README.md`): `src/upgrades/homology_recluster/deferred/run_deferred_dnabert_hc.sh`
(DNABERT-2 LoRA on the HC splits, 18 units; ~14-15 h on M4 MPS with >= 5 GB free RAM, or ~1.5-3 h on one
CUDA GPU with >= 8 GB), then `$PY $M/eval_ft_hc.py` and the three report scripts.

Measured cost on the 16 GB M4 laptop (logs in `external/homology_recluster/logs/`): X4 5.7 h (peak 1.7 GB),
X7 42 min (3.1 GB), X3 1.9 h (0.7 GB), X6 73 min (4.1 GB), X8 2.2 h (5.0 GB footprint, at the watchdog
limit), X9 75 min (4.2 GB), cov-mode-1 regulatory 2.0 h (0.8 GB); HC fine-tuning (36 HyenaDNA units +
eval) 5.7 h, per-unit peak 2.4-2.8 GB. Re-running is idempotent: finished units are cached in
`external/homology_recluster/{clusters,edges,preds,ckpt}`; delete a checkpoint to recompute that unit.
GB tasks are read from a private copy of the same Zenodo archive (record 16605299, v0) that
`data.load_dataset` uses, with train/test counts asserted against `results/results.csv` on every load. A
test checks byte-identity with `data.load_dataset` when the shared cache is intact. NT tasks use `data_nt`
at the pinned revision.

## Caveats (see the interpretation files for numbers)
* The re-split pools the provided splits. For a benchmark whose provided test split is out of
  distribution by design (the NT enhancer tasks: only 3.5-4 % of the re-split test set comes from the
  provided test split), the random re-split control shows that the MCC change is a split-design effect,
  not a homology effect. Test-purge is the protocol that keeps each benchmark's intended test set.
* nt_enhancers: all 100 "strong" test enhancers (25 % of the test set) are verbatim train copies, so the
  purge also removes a whole subclass; against a subtype-matched null the homology effect is -0.008
  (p = 0.28), so memorisation and subclass loss cannot be separated. nt_enhancers_types' purge is not
  evaluable (class eliminated).
* The equalized-arm MCC is degenerate (almost no negatives survive), as in the paper; the TATA/GC
  (and k-mer) AUROC is the readout. On HC data the nt_enhancers equalized arm keeps 1-3 test negatives
  (MCC infeasible; pooled AUROCs only), and the HC fine-tuning contrasts that use its comp_equalized /
  random eval arms are not evaluable.
* MMseqs2 recall falls below about 70 % identity (calibration), so id50 means "all *detectable*
  homology", not all homology at 50 %.
* Sequence-level bootstrap CIs are somewhat narrow for tasks with large homologous families; the
  component-level bootstrap (`homology_recluster_audit_clusterboot.csv`) changes one call
  (demo_human_or_worm id80 cluster - random). The cluster - random CI is conditional on one split;
  split-seed variance is in `homology_recluster_seeds_summary.csv`.
* Many tests (15 tasks x 2 thresholds x 2 protocols; 3 x 11 x 4 Route-1 variants) are uncorrected
  for multiplicity; the pre-registered rules and the named primary Route-1 test are the calls.
* The X4 rule's headline cell was pre-registered as mean pooling, but the paper's HyenaDNA headline is
  the sep-pooling cell; the rule is reported both ways (the paper-cell version is post hoc).
* Not re-run, by design: the semi-synthetic Route-1 dose-response (random backgrounds with implanted
  motifs, so there is no natural homology to remove).
