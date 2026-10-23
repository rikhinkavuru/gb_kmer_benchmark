#!/bin/bash
# Full multivar pipeline, one heavy job at a time, every step at nice 20 / 2 threads / efficiency cores via
# bibm/lowcpu.sh (LightGBM num_threads=2),
# every python process under an in-process 2.8 GB footprint cap (_mv.start_memguard).
# Resumable: every step skips per-task work already checkpointed under external/multivar/.
# Usage (from repo root):  nohup bash src/upgrades/multivar/run_pipeline.sh > results/upgrades/multivar_run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=/Users/rikhinkavuru/Downloads/venv/bin/python
LOW=bibm/lowcpu.sh
M=src/upgrades/multivar
stamp() { echo "=== $(date '+%Y-%m-%d %H:%M:%S') $*"; }

stamp "1/6 prefetch + verify all 15 tasks (counts vs results.csv/results_nt.csv)"
$LOW $PY $M/prefetch_tasks.py
stamp "2/6 per-task length / GC / composition-only LightGBM (resumable)"
$LOW $PY $M/build_covariates.py
stamp "3/6 train/test 8-mer-Jaccard homology for the 11 unmeasured tasks + nt_promoter_tata check (resumable)"
$LOW $PY $M/measure_homology_all.py
$LOW $PY $M/measure_homology_all.py --merge
stamp "4/6 merge covariate table"
$LOW $PY $M/build_covariates.py --merge
stamp "5/6 POST-HOC sequence audits: verbatim duplicates + class identity; cross-task exact overlap; cross-task locus (31-mer) overlap; dedup re-scoring of the LEAKY S11 tasks (resumable)"
$LOW $PY $M/audit_exact_dups.py
$LOW $PY $M/audit_task_overlap.py
$LOW $PY $M/audit_locus_overlap.py
$LOW $PY $M/dedup_eval.py
if [ "${SKIP_ANALYSIS:-0}" = "1" ]; then stamp "SKIP_ANALYSIS=1 -> stop before step 6"; exit 0; fi
stamp "6/6 multivariate analysis + interpretation/paper snippets + figure + unit tests"
$LOW $PY $M/run_multivar.py
$LOW $PY $M/write_interpretation.py
$LOW $PY $M/make_fig_multivar.py
$LOW $PY $M/tests/test_multivar.py
stamp "done"
