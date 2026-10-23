# Deferred job: DNABERT-2 LoRA fine-tuning on the homology-clean splits

Reviewer 1, comment 3 asks for every experiment to be re-run on homology-clean data. The camera-ready also
reports end-to-end fine-tuning (Reviewer 1, comment 2; `src/upgrades/fm_finetune`). This module re-runs it on
the id80 homology-clean splits in two parts:

* **HyenaDNA-tiny (run locally):** `../run_stage_ft_hc.sh`. Its units peak at about 2.0-2.9 GB, which fits
  the laptop's per-job budget.
* **DNABERT-2-117M LoRA (DEFERRED, this folder):** its units peak at 3.5-3.8 GB of process memory. Source:
  the memguard lines in `results/upgrades/fm_finetune_v2_run.log`, e.g. "peak footprint seen 3757 MB" and
  "3659 MB". That is above the ~3 GB per-job limit of the fanless 16 GB laptop, which must also keep the
  rest of the pipeline running. No result is dropped: the job below is complete and needs no code changes.

## What it runs
The fm_finetune recipe is used unchanged (`run_ft_v2.py`: LoRA r=8, 4 epochs, batch 32, lr 3e-4, fixed-shape
padding `--pad-multiple 16`, seeds 42-44, 10 % validation carve). `../run_ft_hc.py` only redirects three paths:

* splits: `external/homology_recluster/ft_splits/`, written by `../prepare_ft_hc_splits.py`; manifest in
  `results/upgrades/homology_recluster_ft_split_manifest.csv`;
* predictions: `results/upgrades/homology_recluster_ft_preds/`;
* checkpoints: `external/homology_recluster/ft_checkpoints/`.

The provided-split fine-tuning outputs are never touched.

## Exact commands (from the repository root)
```
# torch env (never import LightGBM here); MPS or CUDA
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
src/upgrades/homology_recluster/deferred/run_deferred_dnabert_hc.sh     # the 2 lines below, memory-guarded
#   .venv-ft/bin/python src/upgrades/homology_recluster/run_ft_hc.py --model dnabert2 --dataset human_enhancers_cohn \
#       --arms original,comp_equalized,random --seeds 42,43,44 --pad-multiple 16 --mem-cap-mb 4000
#   .venv-ft/bin/python src/upgrades/homology_recluster/run_ft_hc.py --model dnabert2 --dataset nt_enhancers \
#       --arms original,comp_equalized,random --seeds 42,43,44 --pad-multiple 16 --mem-cap-mb 4000
# then, torch-free (scores HyenaDNA + DNABERT-2 together with eval_ft_v2's pre-registered contrasts S/T/U/R):
/Users/rikhinkavuru/Downloads/venv/bin/python src/upgrades/homology_recluster/eval_ft_hc.py
```
On a CUDA machine, add `--device cuda` to both run lines.

## Expected cost and hardware
| hardware | cohn: 9 units | nt_enhancers: 9 units | total |
|---|---|---|---|
| Apple M4 MPS (16 GB unified memory; timings from the provided-split runs in `fm_finetune_v2_run.log`, >= 5 GB free RAM needed) | ~80 min each, ~12 h | ~15-20 min each, ~2.5-3 h | about 14-15 h |
| single CUDA GPU with >= 8 GB (e.g. T4 or A10) | | | an estimated 1.5-3 h |

## Outputs and how they enter the paper
Running `eval_ft_hc.py` writes `results/upgrades/homology_recluster_ft_{runs,metrics,summary,contrasts,verdicts,score_composition,score_composition_summary}.csv`.

The informative dataset is cohn: its homology-clean comp_equalized test arm keeps 176 negatives. The nt_enhancers
homology-clean comp_equalized test arm keeps only 3 negatives, so its contrasts S and T are not evaluable. This
was fixed in advance and is stated in `../README.md`.
