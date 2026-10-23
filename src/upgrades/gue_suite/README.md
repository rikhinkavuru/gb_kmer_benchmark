# gue_suite: the paper's diagnostic on a third suite (GUE, 28 datasets)

Answers Reviewer 2, comment 2 (BIBM 2026, paper B290): *"two suites do not support the abstract's
'route-labels any genomic sequence-classification benchmark'; narrow the wording or add a further
independently curated suite."*

We ran the paper's unchanged CPU pipeline and `diagnose.py`'s fixed route rules on all 28 datasets of
**GUE** (Genome Understanding Evaluation; DNABERT-2, Zhou et al., ICLR 2024). GUE spans human, mouse,
yeast and virus data. We then audited, exactly and in both directions, how much of GUE is the same DNA
as the paper's GB and NT tasks and the rest of the NT suite. The findings, the recommended wording and
every number with its source are in `results/upgrades/gue_interpretation.txt`.

## Files

| file | role |
|---|---|
| `gue_data.py` | GUE loader, pinned archive SHA-256, DNABERT-2 Table 12 split sizes, hand-transcribed Tables 6/9 |
| `verify_gue.py` | integrity gate, run before any model: archive hash, zip-member CRC-32 of every analysed CSV, Table-12 row counts, per-split profile (incl. forward and strand-aware within-split duplicates), exact train/dev/test leakage; re-parses Tables 6/9 from the saved arXiv text and checks all 224 hand-transcribed values |
| `seqshare.py` | exact containment scan (32-mer anchor index + full string confirmation, fwd + rc); unit-tested against brute force |
| `gue_overlap.py` | independence audit: targeted identity/containment relations with label agreement (checks 1-5, 7) and the **uniform** audit of every GUE dataset against every task of the paper's 15 and the other 12 NT-suite tasks, both directions (check 6) |
| `run_gue.py` | **pre-registered** per-task pipeline (plan in the docstring, written before any GUE fit, unchanged) -> `results/upgrades/gue_cards/<task>.json` |
| `run_gue_posthoc.py` | **post hoc** PH1 equalization decomposition against 10 size-matched random arms; PH2 dinucleotide-twin audit |
| `run_gue_posthoc_r2.py` | **post hoc** PH3 (fix round 1): where the positional gain sits in the three GUE Route-2 tasks and nt_promoter_tata (single-bin additions / leave-one-bin-out, TBP/GC/CpG profiles per bin); reading rule in the docstring, fixed before running |
| `rule_sensitivity.py` | **post hoc** PH4 (fix round 1, no fits): re-applies `diagnose._verdict` to stored readouts: Route-3/1/2 signatures without precedence, why each 'hard' label is hard, length-normalised Route-1 richness (L_ref 200-500 bp), MCC-CI straddling 0.7 |
| `emp_provenance.py` | **post hoc** PH5 (fix round 2, no fits): provenance of the yeast EMP classes (GUE label counts vs the Pham et al. 2005 release; archived H3/H4 files), exact location of every EMP window in the sacCer3 genome (ORF vs intergenic, SGD ORFs), and the location-standardised TATA neg-excess of the Route-3 readout; reading rule in the docstring, fixed before running |
| `fetch_emp_sources.sh` | downloads PH5's sources: the Internet-Archive copy of the Pham release page and dataset01/02 (the only archived data files), UCSC sacCer3 + sgdGene; SHA-256 listed |
| `paper15_reference.py` | **post hoc**: the same driver code on the paper's 15 tasks (composition-flag frequency on GB/NT; in-sample consistency of `diagnose._verdict` with the submitted Table 1) |
| `summarize_gue.py` | assembles all CSV tables (no fitting) |
| `make_interpretation_gue.py` | writes `gue_interpretation.txt`; every number and every conditional sentence is derived from the CSVs |
| `run_all.sh`, `run_queue.sh` | background drivers of the first run (checkpointed per task) |
| `tests/test_gue.py` | archive/row counts; driver == `diagnose.py` (mouse_tf_3); reproduces the paper's nt_promoter_tata MCC 0.8463 at k=4; post-hoc helpers; `seqshare` == brute force; stored verdicts reproduced; PH5 helpers (genome location, location classes, standardised excess, reading rule, Pham counts) |

## Outputs (`results/upgrades/`)

- `gue_integrity.csv`: per task-split sizes vs Table 12, CRC check, lengths, non-ACGT rate, duplicates, exact leakage.
- `gue_published.csv`: the parsed DNABERT-2 Tables 6 and 9 (MCC x100; covid macro-F1 x100).
- `gue_results.csv`: one row per GUE dataset with every pre-registered readout and the route label. Post-hoc
  DERIVED columns (fix round 1, no refits): `independence_class`, `frac_in_paper_tasks[_label1]`,
  `frac_in_nt_suite`, `frac_in_any_gb_nt` (from the uniform audit); `route3_signature`, `route1_signature`,
  `route2_signature`, `hard_reasons`, `hard_only_because_richness`, `mcc_ci_contains_0.7` (from PH4);
  fix round 2: `emp_intergenic_frac_label1/label0`, `emp_tata_excess_std[_lo/_hi]`,
  `emp_location_verdict[_4strata]` on the ten EMP rows (from PH5).
- `gue_published_comparison.csv`: k-mer LightGBM and composition-only vs the published models.
- `gue_family_summary.csv`, `gue_paper_table.csv`: per task family (paper table drops the Table-9 HyenaDNA column).
- `gue_predictions.csv`: the pre-stated predictions P1-P4, scored.
- `gue_overlap.csv` (targeted relations, label agreement, core-window offset), `gue_overlap_matrix.csv`
  (28 GUE x 27 GB/NT tasks, both directions, label/split breakdowns), `gue_overlap_other_tasks.csv`
  (GB/NT task sizes and lengths), `gue_overlap_summary.csv` (per GUE task; nothing un-audited is written as 0).
- `gue_independence_routes.csv`: route labels and signatures per independence class.
- `gue_equalization_decomposition.csv`, `gue_twins.csv`: post hoc PH1 and PH2.
- `gue_positional_localisation.csv`: post hoc PH3 (per task x bin). `carrier`/`necessary` are left empty where
  the full B=8 delta CI includes 0 (the comparator nt_promoter_tata; `full_delta_significant`).
- `gue_rule_sensitivity.csv`, `gue_rule_sensitivity_summary.csv`: post hoc PH4 (GUE + paper 15).
- `gue_paper15_reference.csv`: post hoc reference on the paper's 15 tasks.
- `gue_emp_provenance.csv` (one row per yeast mark), `gue_emp_provenance_strata.csv` (mark x location stratum),
  `gue_emp_provenance_tests.csv` (provenance, label direction, cross-mark tests, SHA-256 manifest): post hoc PH5.
  Window coordinates: `external/gue/yeast_genome/emp_window_locations.csv.gz`.
- `gue_failure_route_audit.csv`: the 12 GUE failure-route labels against PH4/PH5 (fix round 2).
- `gue_suite_comparison.csv`: route, signature and flag counts for GB, NT and GUE.
- `gue_interpretation.txt`: the written answer.

## Reproduce (CPU only, no torch; about 2 h total on a MacBook Air at 3 threads)

```bash
cd ~/gbkb_release
PY=/Users/rikhinkavuru/Downloads/venv/bin/python
# 0. data: the GUE archive linked from the DNABERT-2 README (Google Drive id 1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-)
#    -> external/gue/GUE.zip, unzipped to external/gue/GUE/ ; e.g. `gdown 1uOrwlf07qGQuruXqGXWMpPn8avBoW7T- -O external/gue/GUE.zip`
#    DNABERT-2 arXiv:2306.15006v2 full text -> external/gue/refs/dnabert2_arxiv_2306.15006_fulltext.txt
#    NT downstream tasks are fetched by data_nt._read at the pinned revision (HF cache).
bibm/lowcpu.sh $PY src/upgrades/gue_suite/verify_gue.py                                  # integrity gate (~1 min)
LOWCPU_THREADS=1 bibm/lowcpu.sh $PY -W ignore src/upgrades/gue_suite/gue_overlap.py      # independence audit (~14 min, 1 thread, peak ~1.5 GB)
GUE_THREADS=3 bibm/lowcpu.sh $PY src/upgrades/gue_suite/run_gue.py                        # 28 cards, ~48 min (covid ~15 min)
GUE_THREADS=3 bibm/lowcpu.sh $PY src/upgrades/gue_suite/run_gue_posthoc.py                # post hoc PH1/PH2, ~21 min
GUE_THREADS=3 bibm/lowcpu.sh $PY src/upgrades/gue_suite/paper15_reference.py              # post hoc, ~27 min, peak ~2.1 GB
GUE_THREADS=3 bibm/lowcpu.sh $PY -W ignore src/upgrades/gue_suite/run_gue_posthoc_r2.py   # post hoc PH3, ~15 min
bibm/lowcpu.sh $PY -W ignore src/upgrades/gue_suite/rule_sensitivity.py                   # post hoc PH4, no fits, <1 min
src/upgrades/gue_suite/fetch_emp_sources.sh                                              # PH5 sources (network, ~12 MB)
bibm/lowcpu.sh $PY -W ignore src/upgrades/gue_suite/emp_provenance.py                     # post hoc PH5, no fits, ~2 min, <1 GB
$PY src/upgrades/gue_suite/summarize_gue.py && $PY src/upgrades/gue_suite/make_interpretation_gue.py
GUE_THREADS=3 bibm/lowcpu.sh $PY -W ignore src/upgrades/gue_suite/tests/test_gue.py      # 6 tests, ~1 min
# first-run background drivers:  nohup src/upgrades/gue_suite/run_all.sh > external/gue/logs/run_gue.log 2>&1 &
#                                nohup src/upgrades/gue_suite/run_queue.sh > external/gue/logs/queue.log 2>&1 &
```

Every model-fitting stage is checkpointed per task: existing cards and parts are skipped, and `--force`
recomputes them. `rule_sensitivity.py` reads `gue_overlap_other_tasks.csv`, so it runs after `gue_overlap.py`;
`emp_provenance.py` reads `gue_rule_sensitivity.csv`, and `summarize_gue.py` reads both.

## Protocol notes and deviations (all disclosed)

- **Pre-registered plan.** The plan is the `run_gue.py` docstring, written 2026-09-23 before any GUE
  model was fit, and it is unchanged. Thresholds are the paper's. Seed 42. 1000-resample percentile
  bootstrap with the test sequence as the unit. (The docstring's `gue_rowcounts.csv` is written as
  `gue_integrity.csv`.)
- **Threads.** The docstring states n_jobs=2. The run used `GUE_THREADS=3`, following the machine
  policy of 2026-09-26. `tests/test_gue.py` shows that at 3 threads the driver reproduces the paper's
  nt_promoter_tata numbers exactly (k=4, MCC 0.8463), and `paper15_reference.py` reproduces all 15
  paper MCCs.
- **covid.** The archive's split sizes are 73,335/9,166/9,168. DNABERT-2 Table 12 lists
  77,669/7,000/7,000. The total, 91,669, is the same. We use the archive's split, so the published covid
  numbers may come from a different test partition; covid also has 1,118 verbatim test-in-train sequences.
- **DNABERT-2 Table 9 vs Table 6.** Table 9 re-prints a DNABERT-2 row whose human-TF, mouse-TF and
  prom_core_all values differ from Table 6 (11 datasets). We compare against Table 6. The Table-9 HyenaDNA
  row (the DNABERT-2 authors' run) contains values such as 5.34 on prom_300_tata and is not used in the
  paper table. The Table 9 CNN row parses to 29 values for 28 columns and is excluded.
- **Post-hoc analyses** (`run_gue_posthoc.py`, `paper15_reference.py`, `run_gue_posthoc_r2.py`,
  `rule_sensitivity.py`, `emp_provenance.py`, `gue_overlap.py` checks 4-7) were written after the first cards and before they
  were run. Each carries its motivation and reading rule in its docstring. None changes a pre-registered
  readout or label.
- **Equalized-split MCC** is reported but not interpreted: it is exactly 0.000 on four tasks where only
  2.6-8.3 % of negatives survive, and the equalized test sets are 56-98 % positive. PH1's AUROC
  decomposition against size-matched random removal is the readout.
- **Independence.** The first version of `gue_overlap.py` audited only the pairs with an obvious
  counterpart and wrote 0.0 for human TF, mouse TF and covid. Fix round 1 audits every pair (both
  directions); a verifier-reported 20-45 % sharing of human-TF sequences with GB/NT is reproduced exactly.
- **Near-disjoint.** `gue_overlap.py` classes a dataset as near-disjoint when < 1 % of its sequences are
  shared with GB/NT. The six such datasets share at most 0.2 %: mouse TF 2-23 sequences per dataset,
  covid none. Only covid is sequence-disjoint.
- **EMP provenance (PH5).** GUE's yeast label-1/label-0 totals equal the #positive/#negative of the Pham
  et al. 2005 release on all ten marks, and the two archived Pham files (H3, H4) contain every GUE window
  with the same label. The jaist.ac.jp site is offline and the Internet Archive holds only dataset01/02
  (CDX listing in `external/gue/pham2005/cdx_listing.txt`). The ten marks and the 'relative to H3'
  normalisation match Pokholok et al. 2005, but the Pham abstract does not name it and its full text
  was not accessed, so Pokholok is cited only for where the marks sit, not as the data source.
- **PH5 run log.** The first PH5 run (2026-09-27 ~13:10) kept windows that were not located uniquely as
  a spurious stratum (weights summed to ~0.97). This was found on inspection, fixed (weights now asserted
  to sum to 1) and re-run before any number was used. The reading rules are unchanged.
- **Stale text in the pre-registered cards.** The `neg_construction` field of `gue_cards/emp_*.json`
  still carries the first run's EMP description ("Pokholok et al. yeast ChIP-chip windows ..."). The
  cards are not rewritten, so the pre-registered record stays intact. The tables use
  `gue_data.NEG_CONSTRUCTION`, and no number depends on this text.
- **`gue_data.py` edit timeline.** Only the `NEG_CONSTRUCTION` text was edited, at 12:26 on 2026-09-27
  (fix round 1, after `gue_overlap.py` ran and while PH3 ran) and again in fix round 2. The SHA-256 pin,
  split sizes and published tables are unchanged (tests), so no output is stale.
- **Multiplicity.** The 25 non-Route-2 positional-delta CIs are not corrected for multiplicity, so the
  8 that exclude 0 are called nominally significant.
- **GUE+.** The GUE+ datasets (EPI sequence pairs, 5-10 kb species tasks) ship in the same archive
  but are not part of the 28-dataset GUE. They are out of scope (`gue_data.SKIPPED`).
