# Upgrade 1 — Frozen FM-probe on original vs cleaned splits

Converts the paper's conditional claim ("a spurious signal exists and our k-mer classifier
exploits it") into a **measured** one about an independent, pretrained model family: *a frozen
genomic foundation model loses enhancer score when the composition artifact is removed.*

## Model (confirmed from the live HF card, not assumed)
`LongSafari/hyenadna-tiny-16k-seqlen-d128-hf` — HyenaDNA, a small pretrained genomic FM.
`d_model=128`, `n_layer=2`, **436,096 backbone params**, single-nucleotide char tokenizer,
max context 16,386 nt. Revision **pinned** to `d79fa37e2cd62dd338103c630f95be8f90812d46`.
The 16k-context tiny variant (not the 1k one) is used because `drosophila_enhancers_stark`
sequences reach 3,237 nt (median 2,142) — the 1k model would truncate ~97% of the control and
confound it; cohn (500 nt) and nt_enhancers (200 nt) fit either.

## Method
- Frozen forward passes only (CPU, eval mode, no fine-tuning, no GPU). **Right padding is
  forced** (HyenaDNA is causal; left padding — the tokenizer default — corrupts every real
  token; verified bit-safe). Two poolings: **mean** over nucleotide tokens (default) and the
  final **[SEP]** token (the causal CLS-equivalent).
- LightGBM head AND LogisticRegression head on the frozen embeddings (dense path), test
  accuracy/MCC/AUROC + 1000-resample bootstrap CIs.
- Arms: `original`, `tata_flag` (`cleaned_splits/`), `gc_match`, `comp_equalized`
  (`cleaned_splits_v2/`, from Upgrade 2). Datasets: cohn, nt_enhancers, drosophila (control).
- **Two-process design (required):** torch and LightGBM each load their own OpenMP and
  segfault if co-resident on macOS. `extract_embeddings.py` (torch only) caches embeddings;
  `run_fm_probe.py` (LightGBM only) reads the cache and trains heads. They never share a process.

## Run
```
make setup-fm                                                  # torch, transformers, einops
python src/upgrades/fm_probe/extract_embeddings.py --datasets … # torch process: cache embeddings
python src/upgrades/fm_probe/run_fm_probe.py                    # lightgbm process: heads + bootstrap
python src/upgrades/fm_probe/make_fig_fm_probe.py
python src/upgrades/fm_probe/tests/test_sanity.py
# or simply:  make upgrade1
```

## Measured result — primary head (mean-pool, LightGBM), seed 42, 1000-resample bootstrap
| dataset | arm | n_test | FM MCC (95% CI) | FM AUROC (95% CI) | ΔAUROC vs orig |
|---|---|---|---|---|---|
| human_enhancers_cohn | original | 6948 | 0.421 `[0.401,0.441]` | 0.790 `[0.780,0.800]` | — |
| human_enhancers_cohn | tata_flag | 6149 | 0.374 `[0.350,0.398]` | 0.767 `[0.755,0.780]` | −0.023 |
| human_enhancers_cohn | gc_match | 5800 | 0.338 `[0.314,0.363]` | 0.742 `[0.730,0.755]` | −0.048 |
| human_enhancers_cohn | comp_equalized | 3702 | 0.144 `[0.064,0.209]` | 0.638 `[0.600,0.677]` | **−0.152** |
| nt_enhancers | original | 400 | 0.546 `[0.465,0.629]` | 0.818 `[0.777,0.861]` | — |
| nt_enhancers | tata_flag | 351 | 0.407 `[0.314,0.495]` | 0.777 `[0.726,0.823]` | −0.041 |
| nt_enhancers | gc_match | 303 | 0.257 `[0.143,0.373]` | 0.717 `[0.658,0.776]` | **−0.101** |
| nt_enhancers | comp_equalized | 236 | 0.218 `[-0.025,0.372]` | 0.733 `[0.632,0.823]` | −0.085 |
| drosophila_enhancers_stark | original | 1730 | 0.345 `[0.302,0.385]` | 0.728 `[0.704,0.750]` | — |
| drosophila_enhancers_stark | tata_flag | 1730 | 0.345 `[0.302,0.389]` | 0.728 `[0.704,0.751]` | **0.000** |
| drosophila_enhancers_stark | gc_match | 1592 | 0.284 `[0.236,0.331]` | 0.686 `[0.661,0.713]` | −0.041 |
| drosophila_enhancers_stark | comp_equalized | 1082 | 0.208 `[0.134,0.280]` | 0.638 `[0.594,0.683]` | −0.090 |

**Headline:** the frozen pretrained FM loses enhancer score when the composition artifact is
removed — cohn AUROC 0.790→0.638 (MCC 0.421→0.144), nt_enhancers AUROC 0.818→0.717/0.733
(MCC 0.546→0.257/0.218). This is a *pretrained model*, not just our k-mer classifier, riding
the composition shortcut.

## Honest nuances (reported straight)
1. **The control is cleanest under TATA-flag and [SEP] pooling.** With TATA-flag cleaning
   (which removes 0 negatives from the control) drosophila is unchanged (Δ0.000), while cohn/nt
   drop — so the cleaning targets the artifact, not "removing data hurts." With the **[SEP]
   pooling** (LightGBM) the control is flat across *all* arms (AUROC 0.678→0.678→0.676→0.669;
   MCC 0.207→0.207→0.224→0.212) while cohn/nt collapse — a clean dissociation. Under the
   *mean-pool* aggressive arms the control also declines mildly (gc_match −0.041, comp_equalized
   −0.090 AUROC), because (a) the cleaned test sets are smaller (unpaired) and (b) drosophila has
   its own mild composition component (Upgrade-2 max|SMD| 0.26). It declines *less* than the
   contaminated sets and from a lower artifact baseline.
2. **Unpaired MCC.** Each arm has its own, smaller test set; MCC deltas are not paired and the
   `comp_equalized` arm retains few negatives (4.5–4.9% for the contaminated sets) so its MCC is
   underpowered/degenerate (nt MCC CI spans 0). Read alongside Upgrade 2's TATA/GC AUROC collapse.
3. All poolings × heads (48 rows) are in `results/upgrades/fm_probe.csv`. The LR head and
   [SEP] pooling tell the same qualitative story.

## Paired evaluation (Fix A / Fix B) — `run_fm_paired.py`
The unpaired table above confounds "composition removed" with "test set changed" (the cleaned arm
uses a smaller test set). `run_fm_paired.py` removes that confound by scoring, per arm, the SAME
fixed head on the identical retained subset `T_clean`, and reports:
- **`test_effect`** = the SAME frozen-FM head's AUROC change when composition-biased negatives are
  removed from the TEST set (fixed model; nested paired bootstrap) — the benchmark-score inflation;
- **`paired_delta`** = (cleaned-trained − original-trained head) on the identical `T_clean` (training
  effect) — for the control this flattens to ~0 if its earlier decline was just test-set shrinkage.

**LOCKED HEADLINE CELL (Fix B): [SEP] pooling, LightGBM head, `gc_match` arm** — selected because the
control's `test_effect` CI **includes 0** (provably flat) while both contaminated CIs exclude 0:

| dataset | orig AUROC | test_effect AUROC (95% CI) |
|---|---|---|
| human_enhancers_cohn | 0.724 | **−0.048 `[-0.053,-0.043]`** |
| nt_enhancers | 0.800 | **−0.079 `[-0.107,-0.056]`** |
| drosophila (control) | 0.677 | **+0.005 `[-0.003,+0.014]`** (flat) |

`comp_equalized` is reported as a ROBUSTNESS row (aggressive, low-retention), NOT the headline: under
mean pooling its control is not flat (drosophila −0.088 `[-0.124,-0.056]`); under [SEP] pooling the
control is flat across all arms. Figure: `results/upgrades/figures/fm_paired.{pdf,png}`. Full table
(2 poolings × 2 heads × 3 arms × 3 datasets): `results/upgrades/fm_paired.csv`.

## REQUIRED reproducibility note (not tribal knowledge)
**torch and LightGBM must run in SEPARATE processes.** On macOS/arm64 both load their own OpenMP
runtime and **hard-segfault (exit 139)** if co-resident, even with `KMP_DUPLICATE_LIB_OK=TRUE`. Hence
`extract_embeddings.py` (torch only; imports the `data` loaders, never LightGBM) caches embeddings to
`.npz`, and `run_fm_probe.py` / `run_fm_paired.py` (LightGBM only; never import torch) read the cache.
The Makefile `upgrade1` target runs them as separate `python` invocations — do NOT merge them.

## Honesty item for the manuscript
The claim **"no PyTorch"** must become **"no PyTorch in the diagnostic pipeline."** PyTorch is
used ONLY here, for frozen FM feature extraction, isolated behind the optional `[fm]` extra and
run as a separate process; the core diagnostic pipeline imports and runs with zero torch installed.

## Depth 1 — DNABERT-2 (second FM architecture; cross-architecture robustness of U1)
Shows the composition-riding result is not HyenaDNA-specific by repeating the frozen-FM probe with a second,
architecturally-unrelated pretrained genomic FM.

**Model (confirmed from the live HF card):** `zhihan1996/DNABERT-2-117M` — MosaicBERT bidirectional encoder,
BPE tokenizer (vocab 4096), hidden 768, ~117M params, max 512 tokens, revision `7bce263b`. Pooling: mean
over masked tokens (default) + `[CLS]` (alternative) — the BERT analogues of HyenaDNA's mean / `[SEP]`.

**Datasets — DELIBERATE EXCLUSION:** only the SHORT sets `human_enhancers_cohn` (500 bp → ~129 BPE tokens)
and `nt_enhancers` (200 bp → ~54 tokens). `drosophila_enhancers_stark` is **excluded by design**: its 3.2 kb
sequences exceed the 512-token context and would truncate ~97% of the clean control, reintroducing exactly
the confound the HyenaDNA-16k choice avoided. So DNABERT-2 has no flat-control arm; the cross-architecture
claim is the SIGN + SIGNIFICANCE of the composition-removal delta on the two contaminated sets, and the
flat-control evidence remains HyenaDNA's.

**Run (two processes — torch extract, then LightGBM probe):**
```
python src/upgrades/fm_probe/extract_embeddings_dnabert.py --datasets human_enhancers_cohn,nt_enhancers
python src/upgrades/fm_probe/run_fm_paired_dnabert.py        # paired Fix-A eval (LightGBM)
python src/upgrades/fm_probe/make_cross_architecture.py      # cross-arch table + figure
# or: make depth1
```

**CPU loading recipe (documented, reproducible; the MosaicBERT remote code predates transformers-5.x /
torch-2.x / Python-3.14):** (1) bypass transformers' STATIC `check_imports` for the GPU-only `triton`/
`flash_attn` — the model's own `try/except` (bert_layers.py) falls back to standard attention on CPU when
the kernel is absent (do NOT `pip install triton`, which has no macOS wheel); (2) build via
`AutoModel.from_config` on CPU (avoids the meta-device lazy init `from_pretrained` triggers); (3) load the
checkpoint with the `bert.` prefix stripped and the MLM `cls.` head dropped — only the UNUSED pooler is left
at init; (4) set `config.pad_token_id`. A load **guard** asserts the encoder/embeddings loaded (no silent
random-weight load). Embeddings are real frozen DNABERT-2 CPU forward passes.

**Measured cross-architecture result (gc_match arm, test_effect AUROC; * = CI excludes 0; seed 42, 1000-boot):**

| dataset | HyenaDNA ([SEP]) | DNABERT-2 (mean) | DNABERT-2 ([CLS]) |
|---|---|---|---|
| human_enhancers_cohn | −0.048 * | −0.047 * | −0.046 * |
| nt_enhancers | −0.079 * | −0.072 * | −0.077 * |

→ Both architectures lose nearly the same enhancer AUROC when composition is removed — negative + significant
for both, both poolings, both heads (LR & LightGBM). The U1 effect is architecture-robust.

## Release & reproduction (torch-free) — `reproduce_from_cache.py`
The frozen FM embeddings are **released as a standalone artifact** (10 `.npz` files: HyenaDNA mean/[SEP]
and DNABERT-2 mean/[CLS], for the original splits of each dataset; arm results are subsets selected by
sequence lookup, so only the original-split embeddings are cached). Manifest with per-file sha256:
`results/upgrades/fm_embeddings_manifest.csv`. Release alongside the existing Zenodo artifact (cleaned
splits + `cleaned_splits_v2/`).

**The entire downstream FM result reproduces from the cache with ZERO torch:**
```
python src/upgrades/fm_probe/reproduce_from_cache.py   # regenerates fm_probe.csv, fm_paired.csv,
                                                       # fm_paired_dnabert.csv -- byte-identical, torch-free
```
It asserts each reproduced CSV is byte-identical to the canonical one and that `torch` never entered
`sys.modules`. A replicator therefore **never loads a model** — both heads, all arms, all paired Fix-A
deltas, and all bootstrap CIs come from the released embeddings.

**The brittle model-load path is needed ONLY to regenerate embeddings from scratch.** Pinned stack for
that step: **Python 3.14.5, torch 2.12.0, transformers 5.9.0**. HyenaDNA loads cleanly (right-padding
only). DNABERT-2's MosaicBERT remote code predates this stack, so `extract_embeddings_dnabert.py` applies
the documented workaround above (static triton check bypass + runtime CPU fallback; `from_config` build;
`bert.`-strip checkpoint; `pad_token_id`; load guard). **HyenaDNA is the PRIMARY arm** (clean load, flat
control — it carries the flat-control evidence since drosophila is excluded from DNABERT-2 for the
512-token limit); **DNABERT-2 is the CORROBORATING arm** (cross-architecture robustness).
