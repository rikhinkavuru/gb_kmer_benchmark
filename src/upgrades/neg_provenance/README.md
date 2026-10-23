# neg_provenance: where the AT-rich negatives come from

This module answers Reviewer 2, comment 1 (BIBM 2026, paper B290): *"...the paper does not yet explain why the
negatives are AT-rich or which genomic regions they come from. Please characterize the provenance of the negative
sets of human_enhancers_cohn and nt_enhancers."*

The pre-registered analysis plan (decision rules D1 to D4) is in the docstring of `run_neg_provenance.py`.
Everything added later is labelled post hoc there or in the script's own docstring: the resume additions
(BG_rule_pc, D5, ChromHMM state composition, origin census, header concordance), `chromhmm_rule_counts.py`, and
the fix-round-1 scripts `nt_split_provenance.py` and `bg_seed_stability.py`.

## Result in one paragraph

- **cohn.** The negatives are random 500-bp windows at least 15 kb from genes and outside liver H3K27ac peaks
  (Cohn, Zuk & Kaplan 2018). On GRCh38, 94.7% lie at least 15 kb from every GENCODE v44 protein-coding gene; for
  random windows the figure is 46.2%. They are gene-distal but they are not gene deserts: 36% lie inside a GENCODE
  gene (mostly in introns of non-coding genes), 57.5% are ChromHMM state 13 in all nine ENCODE lines, and 14.3%
  carry an enhancer state in at least one line.
- **nt_enhancers.** Only 28.2% of the sequences are genomic, namely the verbatim iEnhancer originals. The rest are
  199-bp RankGAN-synthetic sequences: 76.6% of all negatives and 78.7% of the training negatives. The test split
  has no synthetic sequences.
  - Training negatives (iEnhancer-2L) are 200-bp fragments of ChromHMM state 13 in all nine lines. This state is
    "Heterochromatin; low signal": the segmentation used nine marks, none of them H3K9me3. The rule holds for
    99.8% of fragments (lifted midpoint) and for 1,484/1,484 hg19 source windows (no liftOver). "State 13 in all
    nine" also reproduces the published non-enhancer candidate count: 5,260,191 fragments against 5,257,994.
  - Test negatives (the iEnhancer-EL independent set) do not follow that rule: 64.0% [57.0, 70.0] are state 13 in
    all nine lines, and 12.5% touch an enhancer state. Their FASTA headers show why. They were cut from single
    ChromHMM state-13 segments (each identical in at least two of the nine lines) of a UCSC custom track, not from
    the nine-line intersection. The test weak enhancers come from single H1-hESC segments of the same track.
  - Train and test negatives are equally AT-rich: GC 0.375 vs 0.372, TATA 0.357 vs 0.345.

Gene-distal and unmarked DNA sits in AT-rich, L1-rich isochores (L1+L2: 71% and 84% of negatives, 60% of the
genome). Backgrounds re-drawn by the documented rules, with the unstated details inferred post hoc
(protein-coding genes; state 13), reproduce the negatives' GC and approximately their TATA rate. The six-draw
pooled cohn background has TATA +0.011 [+0.002, +0.021] relative to the negatives. Local GC alone approximately
predicts the negatives' TATA rate.

The sampling rules account for a minority of the positive-negative gap:

| gap | cohn | genomic nt_enhancers |
|---|---|---|
| GC | 20% | 31% (13.5% over the whole class) |
| TATA | 15% | 30% |

The rest is the positives' GC-richness and TATA depletion relative to random genome. All numbers are in
`results/upgrades/neg_provenance_interpretation.txt`.

## Files

| file | role |
|---|---|
| `verify_inputs.py` | Integrity and build checks for every input: gzip, GENCODE v44/GRCh38 header, chain sizes, ChromHMM hg19/15 states, rmsk soft-mask concordance, gap vs N content, cache row counts, all 27,791 cohn intervals reproduce the benchmark sequences, nt counts, GENOMIC call equals verbatim iEnhancer identity, and the cCRE BED md5. Writes `neg_provenance_input_checks.csv` (39 checks). |
| `map_nt_enhancers.py` | minimap2 (mappy, `sr`) mapping of all 15,368 nt_enhancers sequences to hg38 (first run, 2026-09-23; verified). Writes `neg_provenance_nt_map.csv`. |
| `genome_utils.py` | Low-RAM hg38 annotation: 1-kb GC bins, GENCODE v44 genes/TSS/exons, rmsk, SCREEN V4 cCREs, ChromHMM, liftOver. |
| `run_neg_provenance.py` | Main analysis: groups, backgrounds, annotation, bootstrap, D1 to D5, chromosomes (plus fix round 1: numeric CI columns, `_chrom_tests.csv` with exact log10 p), origin census. |
| `nt_split_provenance.py` | Post hoc (fix round 1): nt train/test/pooled metrics and contrasts, D3 per split, liftover-free hg19-header check, parent regions of the independent-test negatives and weak enhancers with shifted-region null and segment-sharing baseline. |
| `bg_seed_stability.py` | Post hoc (fix round 1): five more background draws (seeds 1 to 5), pooled with seed 42 (60,000 windows each), background minus NEG and f_rule with CIs. |
| `chromhmm_rule_counts.py` | Post hoc: which ChromHMM definition reproduces the iEnhancer-2L candidate counts. |
| `write_sources.py` | Primary-source table (25 rows) with verbatim quotes and DOIs/URLs. |
| `make_report.py` | Compact table (CSV and LaTeX, compiled with tectonic/IEEEtran), post hoc whole-class nt decomposition, typelabels side check, interpretation text. |
| `tests/test_sanity.py` | 13 unit and sanity tests, including the header parsers, exact-segment matching, and the log10 chi-square tail. |

Outputs are written to `results/upgrades/neg_provenance_*`:

- **Main analysis:** `summary`, `table`, `tests` (CI columns `*_ci_lo`/`*_ci_hi`), `chrom`, `chrom_tests`,
  `intervals.csv.gz`, `meta`, `nt_origin`, `sanity_paper_numbers`.
- **Fix round 1:** `nt_by_split`, `nt_by_split_tests`, `nt_hg19_check`, `nt_test_parent`, `bg_seeds`,
  `bg_pooled`, `bg_seeds_windows.csv.gz`.
- **Report:** `compact_table.{csv,tex}`, `nt_wholeclass`, `nt_typelabels`, `interpretation.txt`.
- **Inputs and sources:** `sources`, `input_checks`, `nt_map`, `chromhmm_rule_counts`.

No figure: the compact table carries the result more compactly than a figure would in a 7-page paper.

Large inputs live under `external/neg_provenance/`:

- GENCODE v44 GTF
- UCSC hg38 `rmsk.txt.gz`, `gap.txt.gz`, `hg38.chrom.sizes`, and the hg38↔hg19 chains
- `GRCh38-cCREs.bed`: SCREEN Registry V4, from https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed;
  2,348,854 records, md5 `60c14d9f4903a7bd13b3db4f290ace48`. The copy was taken from a local mirror, and
  `verify_inputs.py` checks it.
- `chromhmm_hg19/`: the nine Broad ChromHMM segmentations
- `ienhancer/`: iEnhancer-2L/-EL FASTA from two GitHub copies (commits in `SOURCES.txt`)
- `sources/`: saved primary-source texts, including the open-access isochore-boundary papers
- `cache/`: npz caches
- `nt_map/`: per-chromosome mapping checkpoints
- `logs/`

The hg38 FASTA is `~/.cache/glrb_hg38/hg38.fa` (UCSC, soft-masked; override with `HG38_FA`).

## Reproduce

Run these from the repo root. The steps are CPU-only, at most 3 threads, and use under 2 GB RAM. Runtimes were
measured on 2026-09-26 on a MacBook Air (M-series, performance cores allowed).

```bash
PY=/Users/rikhinkavuru/Downloads/venv/bin/python
L=bibm/lowcpu.sh
$L $PY src/upgrades/neg_provenance/map_nt_enhancers.py      # ~10 min; resumes from external/neg_provenance/nt_map/
$L $PY src/upgrades/neg_provenance/verify_inputs.py         # ~20 s with caches; exit code 1 if any check fails
$L $PY src/upgrades/neg_provenance/tests/test_sanity.py     # ~1 min, 13 tests
$L $PY src/upgrades/neg_provenance/run_neg_provenance.py    # ~70 s, 1.1 GB (--quick: smoke test, *_QUICKTEST outputs)
$L $PY src/upgrades/neg_provenance/nt_split_provenance.py   # ~1 min (needs run_neg_provenance outputs)
$L $PY src/upgrades/neg_provenance/bg_seed_stability.py     # ~70 s, 1.6 GB
$L $PY src/upgrades/neg_provenance/chromhmm_rule_counts.py  # ~1 min
$L $PY src/upgrades/neg_provenance/write_sources.py
$L $PY src/upgrades/neg_provenance/make_report.py           # ~1 min
```

- **Seeds:** seed 42 throughout, plus background seeds 1 to 5 in `bg_seed_stability.py`.
- **Bootstrap:** 1000 percentile resamples, with the window or sequence as the unit and groups resampled
  independently.
- **Determinism:** re-running `run_neg_provenance.py` in fix round 1 reproduced every previously written output
  byte-for-byte on the common columns (summary, intervals, tests, chrom, nt_origin, meta, table).
- **Dependencies:** `pyfaidx`, `pyliftover` and `mappy` in the venv.

## Fix round 1 (2026-09-26): what changed and why

1. **nt train/test split.** The pooled "95.5% state 13 in all nine" hid a split difference: 99.8% for train
   against 64.0% for test. The difference is real, not a liftOver artefact. Train is 1,484/1,484 by hg19 headers.
   Test has perfect mappings, and its negatives come from single-segment parent regions (see above). The earlier
   limitation that blamed the shortfall on "boundary or liftover effects" was wrong and has been removed.
2. **Synthetic share.** The synthetic share is 76.6% of *all* nt negatives and 78.7% of *training* negatives
   (positives: 67.0% of all, 68.8% of training).
3. **Wording.** State 13 is described as ChromHMM low-signal ("Heterochromatin; low signal", no H3K9me3 in the
   model), not as "constitutive heterochromatin". The cohn negatives are described as gene-distal, not as
   "gene deserts".
4. **Gap attribution.** The rules explain the negatives' departure from random genome, which is a minority of the
   positive-negative gap (see the table above).
5. **Seed stability.** The cohn protein-coding rule reproduces GC exactly, but pooled over six draws its TATA rate
   is +0.011 [+0.002, +0.021] above the negatives. It is therefore described as "approximately" matching.
6. **Chromosomes.** The nt chromosome skew is fully explained by the state-13 rule background (homogeneity
   p = 0.48). The cohn skew is not explained (p = 10^-331 against BG_rule_pc).
7. **Smaller fixes.**
   - D5 is worded "approximately": one of four NEG obs/exp CIs excludes 1.
   - The count reconstruction reproduces the non-enhancer count, but only weakly separates state 13 from
     states 13-15.
   - The isochore boundaries are confirmed from two open-access papers that cite Costantini et al. 2006.
   - The cCRE BED is now inside `external/` with its md5 recorded.
   - The CIs are stored as numeric columns (no `eval`).

## Side finding (not part of R2.1)

In `nt_enhancers_types`, every one of the 742 iEnhancer-2L weak-enhancer training originals also appears in the
NT training split under the *strong* label. All 100 test strong enhancers are verbatim iEnhancer-2L training
strong enhancers. See `neg_provenance_nt_typelabels.csv`. The binary `nt_enhancers` labels are consistent.

## Caveats

- **Cohn's gene set is not specified.** The paper cites only "Ensembl 2018". The protein-coding variant
  `BG_rule_pc` is post hoc. With the pre-registered all-biotype GENCODE v44 genes, the rule overshoots the AT shift.
- **Villar et al. H3K27ac peaks are proxied** by the benchmark positives.
- **The original Cohn archive could not be inspected.** The archive at cs.huji.ac.il returned a server error page,
  so the negatives' original build and coordinates come from Genomic Benchmarks' exact GRCh38 relocation.
- **The cohn negatives have an unexplained chromosome skew.** There are none on chr9 or chrY, and chr2 and chrX are
  depleted. The rule background does not predict this, but it does not drive the AT shift (see the
  chromosome-matched sensitivity).
- **The iEnhancer non-enhancer definition is inferred** from the exact count match and the header check. The
  paper does not list states. The pre-registered primary reading ("non-enhancer in all nine") is reported with
  its NOT EXPLAINED verdict.
- **The iEnhancer-EL construction is inferred.** It is read from the FASTA headers and the ChromHMM segment
  match, not from a documented protocol.
- **Multiplicity.** About 40 descriptive CIs are not multiplicity-corrected. The verdicts rest only on the
  pre-registered rules D1 to D4.
- **Pre-registration evidence.** The docstring can only be compared with a same-day snapshot
  (2026-09-26 13:15), and no copy from 2026-09-23 exists. In fix round 1, `verify_inputs.py` was re-run before
  the main script (logs in `external/neg_provenance/logs/`).
