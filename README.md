# gb_kmer_benchmark — *Three Routes to Failure*

Interpretable, **CPU-only** diagnostics of what genomic sequence-classification benchmarks
actually measure. Companion code, cleaned splits, and diagnostic tool for:

> Rikhin Kavuru. *Three Routes to Failure: An Interpretable Diagnosis of Task Difficulty
> and Negative-Set Contamination in Genomic Sequence-Classification Benchmarks.*
> In *Proc. IEEE International Conference on Bioinformatics and Biomedicine (BIBM)*, 2026
> (regular paper B290).

**Online Supplementary Material (BIBM 2026):**
[`bibm/supplement/B290_SM.pdf`](bibm/supplement/B290_SM.pdf) (sections S1–S9, cited as "SM S*n*"
in the paper). Every SM table is regenerated from the result files by
`bibm/supplement/make_sm_tables.py`.

The diagnostic pipeline imports **zero PyTorch**: $k$-mer spectra + gradient-boosted trees
(LightGBM), plus released frozen-FM *embedding caches*, let the entire analysis (including
the foundation-model probe) reproduce on CPU. Exact parameters are in **`REPRODUCIBILITY.md`**.

## What's in this repo
- **Code** — `*.py` (featurization, models, motif scoring, the `run_*` drivers) and `src/upgrades/`
- **`src/upgrades/diagnose/diagnose.py`** — the route-labeled diagnostic report-card CLI
- **`src/upgrades/fm_finetune/`** — end-to-end fine-tuning of HyenaDNA-tiny on the released vs
  composition-cleaned splits (the only module that trains a foundation model; `make finetune`)
- **`src/upgrades/presence_null/`, `src/upgrades/paperfigs/`** — the motif-presence null with
  confidence intervals and the composite paper figures
- **Camera-ready revision modules** (each has a README; outputs in `results/upgrades/`):
  `psm_sensitivity/` (propensity-matching sensitivity sweeps), `multivar/` (multivariate
  regression of benchmark difficulty), `homology_recluster/` (MMseqs2 homology re-clustering and
  re-runs), `fm_finetune/` (HyenaDNA full and DNABERT-2 LoRA fine-tuning; CUDA-only jobs in
  `fm_finetune/gpu_deferred/`), `alt_model/` (route labels under alternative interpretable
  models), `neg_provenance/` (genomic provenance of the negative sets), `gue_suite/` (third
  benchmark suite, GUE)
- **`cleaned_splits/`, `cleaned_splits_v2/`** — contamination-cleaned train/test splits
  (TATA-flag, GC-matched, composition-equalized) for `human_enhancers_cohn`, `nt_enhancers`,
  and the `drosophila_enhancers_stark` control
- **`results/`** — result CSVs and figures
- **`localization/`** — Route-3 GRCh38/GENCODE coordinate-analysis code and outputs
- **`kavuru.pdf`, `paper_psb.tex`** — the earlier (PSB-format) manuscript
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
MIT; see [`LICENSE`](LICENSE). Third-party datasets keep their original licenses.
