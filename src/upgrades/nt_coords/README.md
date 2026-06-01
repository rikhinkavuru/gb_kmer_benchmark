# Upgrade 6 — Extend the TSS coordinate refutation to nt_enhancers (Route 3)

Closes the cohn-only asymmetry in the GRCh38/GENCODE coordinate analysis. The cohn analysis
(`localization/analyze_cohn.py`) **refuted** core-promoter leakage — cohn's TATA-flagged negatives are
TSS-*depleted* (not enriched), with no −25..−30 bp offset peak → AT-rich compositional bias. That test
mapped 100% of cohn but only ~22% of nt_enhancers by exact substring match. Here we use **fuzzy
alignment** (minimap2 via `mappy`, preset `sr`) to recover more nt_enhancers coordinates and repeat the
cohn analysis on the second dataset.

## Run (needs the optional `[coords]` extra + GRCh38 FASTA)
```
make setup-coords     # mappy
# download GRCh38 (Ensembl release-97, matches cohn) to localization/GRCh38.primary_assembly.fa.gz
python src/upgrades/nt_coords/run_nt_coords.py          # results/upgrades/nt_coords.csv (+ cached mapping)
python src/upgrades/nt_coords/make_fig_nt_coords.py
python src/upgrades/nt_coords/tests/test_sanity.py
```

## Measured result (seed 42; GRCh38 Ensembl release-97; GENCODE v44; mappy `sr`)
- **Mappability 28.2%** (4332/15368), up from the ~22% exact-match baseline; negatives 23.3% (1794/7684).
- **TSS overlap (±500 bp):** TATA-flagged negatives **0.6%** vs non-flagged negatives **1.4%**
  (**0.46×**, Fisher OR=0.45, p=0.17); positives 15.9% (assay positive control). Flagged negatives are
  **TSS-DEPLETED**, exactly as in cohn (flagged 1.82% vs non-flagged 3.29%, 0.55×).
- **TATA→TSS offset:** no enrichment in the canonical −25..−30 bp core-promoter window (frac = 0.000).

## Verdict (reported straight)
**REPLICATES cohn**: the flagged negatives are TSS-depleted, not TSS-proximal, with no core-promoter
offset peak → the nt_enhancers contamination is **AT-rich compositional bias, not TSS leakage** — the
same conclusion as cohn, now on the second dataset.

## Honest caveats
- **Underpowered.** Fuzzy mapping lifted mappability only to 28.2% (nt_enhancers' 200 bp fragments are
  hard to place uniquely on GRCh38 release-97; many likely derive from a different build/processing).
  With only 1794 mapped negatives (635 flagged), the Fisher test is **not significant (p=0.17)** — the
  result replicates cohn in **direction and effect size** (0.46× vs 0.55× depletion) but not in
  statistical power, and the offset analysis has only n=4 flagged-negs-on-TSS. The conclusion is
  "consistent with, and directionally replicates, cohn," not an independently-significant refutation.
- The mapping is cached to `localization/nt_coords_mappy.csv` so reruns skip the alignment.
