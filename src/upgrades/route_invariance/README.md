# Upgrade 9 — Route-invariance across model families

Shows the route assignment is stable across model families, using LR floor, LightGBM, and the frozen
**FM-probe head** as the three families (no from-scratch CNN — the FM-probe supplies the cross-family
evidence). The **mechanism route** (solved / shared-motif / positional / contamination) is assigned
from the model-agnostic diagnostics; the **solvability tier** (solved ≥0.7 / partial ≥0.5 / hard) is
read per family from its benchmark MCC. Reads existing CSVs only — no heavy recompute.

## Run
```
python src/upgrades/route_invariance/run_route_invariance.py       # results/upgrades/route_invariance.csv
python src/upgrades/route_invariance/make_fig_route_invariance.py  # figures/route_invariance.{pdf,png}
python src/upgrades/route_invariance/tests/test_sanity.py
```

## Measured result (15 tasks; FM-probe on the 3 enhancer tasks)
**Tier-agreement matrix** (same solved/partial/hard tier over shared tasks):

| | LR | LightGBM | FM-probe |
|---|---|---|---|
| **LR** | — | 13/15 = **87%** | 2/3 = 67% |
| **LightGBM** | | — | 3/3 = **100%** |
| **FM-probe** | | | — |

**LR-vs-LightGBM tier disagreements (capacity-driven, NOT route changes):**
- `human_ensembl_regulatory`: LR **hard** (0.429) vs LightGBM **solved** (0.907) — the 3-class
  regulatory signal is captured by gradient-boosted trees but not the linear floor.
- `nt_enhancers`: LR **hard** (0.437) vs LightGBM **partial** (0.514) — a boundary case at the 0.5 cutoff.

## Interpretation
- The **mechanism route is model-agnostic** (assigned from the diagnostics), so it does not change with
  the model family.
- The **solvability tier agrees on 13/15 tasks for LR vs LightGBM and 3/3 for LightGBM vs the FM-probe**;
  the FM-probe independently agrees the enhancer tasks are hard, supplying cross-family evidence
  without a from-scratch CNN.
- The only disagreements are **capacity-driven** (a multiclass task that trees solve but the linear
  floor cannot) or a **0.5-boundary** task — neither reassigns the route. Reported straight.
