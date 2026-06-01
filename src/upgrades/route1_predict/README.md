# Upgrade 7 — Predictive held-out test for Route 1

Makes Route 1 (discriminability → learnability) **predictive**, not merely correlational. Fits the
relationship `benchmark MCC ~ max positive-direction motif-only AUROC` in a **leave-one-task-out**
(LOTO) manner across the 11 TF-motif tasks (both suites): each task's MCC is predicted by an OLS line
trained on the other 10.

## Run
```
python src/upgrades/route1_predict/run_route1_predict.py      # results/upgrades/route1_predict.csv
python src/upgrades/route1_predict/make_fig_route1_predict.py # figures/route1_predict.{pdf,png}
python src/upgrades/route1_predict/tests/test_sanity.py       # 3 sanity checks
```
Reads `results/cross_suite_summary.csv` (`tf_motif_task == True`) — no heavy recompute.

## Measured result (n = 11 TF-motif tasks; bootstrap CIs over tasks, seed 42)
- **Held-out (LOTO) R² = +0.439, 95% CI [−0.269, +0.701]**
- **Held-out MAE = 0.140 MCC, 95% CI [0.108, 0.180]**
- Full-data fit: `MCC ≈ 4.0·d − 1.9`; Spearman(d, MCC) = +0.73.

**Interpretation:** motif discriminability **predicts** learnability on tasks the line never saw —
held-out R² is positive and MAE is ~0.14 MCC. Reported straight: with n=11 the R² CI includes
negative values, and the largest error is `nt_enhancers_types` (|err| 0.33), a 3-class enhancer-type
task whose single-motif AUROC overstates its (multiclass) difficulty — visible in the per-task table.
The result upgrades Route 1 from a correlation to a (modestly) predictive law.
