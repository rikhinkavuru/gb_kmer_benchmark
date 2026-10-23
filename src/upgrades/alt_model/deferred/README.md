# Deferred: one post hoc unit

| unit | why deferred | command | expected runtime / hardware | effect on results |
|---|---|---|---|---|
| `human_ensembl_regulatory__bench_full__kmer__ebm` (uncapped multiclass EBM, post hoc P2) | ~9-10 h on the 16 GB MacBook Air at 3 threads (4 outer bags run 3+1; each bag > 3.7 h); stopped after 3 h 45 min so it would not block the camera-ready workflow | `nohup src/upgrades/alt_model/deferred/run_ebm_regulatory_full.sh > external/alt_model/logs/deferred_ebm_full.log 2>&1 &` | ~9-10 h here; ~4 h with >= 4 free cores; RAM < 2 GB; CPU only | adds one `bench_full` row; cannot change a label: the capped EBM is solved at MCC 0.879 [0.876, 0.883] (`results/upgrades/alt_model_bench.csv`, arm=bench) and on this task uncapping moved LightGBM by +0.008 and LR by +0.013 MCC (across all six capped tasks uncapping changed MCC by at most +0.040, LR on ocr, and changed no label). Until it runs, the `bench_full_where_available` label rows use the capped EBM for this one task. |

All other units (~790), including every other uncapped refit, finished locally.
