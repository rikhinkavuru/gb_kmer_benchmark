# `paperfigs` — the composite figures used in the manuscript

Each script reads **only** result CSVs and writes a vector PDF + 400-dpi PNG to
`results/upgrades/figures/`. No script recomputes anything, and no number is typed in by hand: even
the annotations on the schematic are read from the CSVs, so a figure cannot drift from the tables.

| script | figure | reads |
|---|---|---|
| `make_fig_routes.py` | Fig 1, the three-route schematic | `motif_discriminability.csv`, `upgrades/positional_sweep.csv`, `contamination_analysis.csv`, `upgrades/composition_clean.csv` |
| `make_fig_task_landscape.py` | Fig 2, Route-1 dose-response + Route-2 bin sweep | `upgrades/disc_dose.csv`, `upgrades/disc_dose_slopes.csv`, `upgrades/positional_Bsweep.csv`, `upgrades/positional_sweep.csv` |
| `make_fig_route3_evidence.py` | Fig 3, shuffled negatives + IC-matched controls + threshold sweep | `upgrades/shuffled_neg.csv`, `upgrades/route3_specificity.csv`, `upgrades/route3_threshold_invariance.csv` |
| `make_fig_fm_combined.py` | Fig 4, frozen probes + fine-tuning | `upgrades/fm_cross_architecture.csv`, `upgrades/dnabert_control.csv`, `upgrades/fm_finetune{,_summary}.csv`, `upgrades/composition_clean.csv` |

## Print sizing

Every figure is drawn at **5.5 in wide** and included at `\textwidth` (5.61 in in the PSB
`ws-procs11x85` class), i.e. essentially 1:1, with a minimum font size of 8 pt. This matters: the
earlier single-module figures were drawn at 7.6--9.0 in and included at 0.5--0.7 `\textwidth`, which
shrank their labels to roughly 3 pt on the printed page. `make_fig_shuffled_neg.py` and
`make_fig_route3_specificity.py` in the sibling modules were re-sized for the same reason; they remain
as the per-module figures, while `make_fig_route3_evidence.py` merges their informative panels into
the single manuscript figure.

```bash
make paperfigs
```
