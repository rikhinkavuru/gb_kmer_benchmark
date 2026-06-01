# gb_kmer_benchmark — *Three Routes to Failure*

Interpretable, **CPU-only** diagnostics of what genomic sequence-classification benchmarks
actually measure. Companion code, cleaned splits, and diagnostic tool for:

> Rikhin Kavuru. *Three Routes to Failure: An Interpretable Diagnosis of Task Difficulty
> and Negative-Set Contamination in Genomic Sequence-Classification Benchmarks.*
> Submitted to the Pacific Symposium on Biocomputing (PSB) 2027.

The diagnostic pipeline imports **zero PyTorch**: $k$-mer spectra + gradient-boosted trees
(LightGBM), plus released frozen-FM *embedding caches*, let the entire analysis (including
the foundation-model probe) reproduce on CPU. Exact parameters are in **`REPRODUCIBILITY.md`**.

## What's in this repo
- **Code** — `*.py` (featurization, models, motif scoring, the `run_*` drivers) and `src/upgrades/`
- **`src/upgrades/diagnose/diagnose.py`** — the route-labeled diagnostic report-card CLI
- **`cleaned_splits/`, `cleaned_splits_v2/`** — contamination-cleaned train/test splits
  (TATA-flag, GC-matched, composition-equalized) for `human_enhancers_cohn`, `nt_enhancers`,
  and the `drosophila_enhancers_stark` control
- **`results/`** — result CSVs and figures
- **`localization/`** — Route-3 GRCh38/GENCODE coordinate-analysis code and outputs
- **`kavuru.pdf`, `paper_psb.tex`** — the manuscript
- **`REPRODUCIBILITY.md`** — versions, seeds, hyperparameters, pinned dataset revision

## What's on Zenodo (too large for GitHub)
- Frozen-FM embedding cache (HyenaDNA + DNABERT-2) and $k$-mer feature caches (`cache/`, ~1.4 GB)
- Reference data used by `localization/`: GRCh38 primary assembly + GENCODE v44 annotation

> **Zenodo DOI** (frozen-FM embedding cache, HyenaDNA + DNABERT-2): [10.5281/zenodo.20494982](https://doi.org/10.5281/zenodo.20494982)

## Quickstart
```bash
pip install -r requirements.txt           # core, CPU-only (no torch)
python src/upgrades/diagnose/diagnose.py --help
```
The frozen-FM probe reproduces from the released embedding cache with no torch;
`requirements-fm.txt` is only needed to re-extract embeddings from scratch.
`requirements-coords.txt` covers the `localization/` coordinate analysis (needs the Zenodo reference data).

## Building the paper
`paper_psb.tex` requires the World Scientific PSB proceedings class (`ws-procs11x85.cls`
and companions) from the PSB author kit; those third-party files are **not** redistributed here.

## License
To be added before public release.
