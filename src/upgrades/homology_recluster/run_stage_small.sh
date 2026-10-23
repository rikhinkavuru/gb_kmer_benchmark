#!/bin/bash
# Stage A: evaluation + tier-1 interventions for the small tasks whose clusters already exist.
# Every command runs through bibm/lowcpu.sh (nice 20, <=3 BLAS/OpenMP threads).
cd /Users/rikhinkavuru/gbkb_release || exit 1
PY=/Users/rikhinkavuru/Downloads/venv/bin/python
W=bibm/lowcpu.sh
M=src/upgrades/homology_recluster
$W $PY -W ignore $M/run_eval.py --tasks drosophila_enhancers_stark,nt_enhancers,nt_enhancers_types,human_enhancers_cohn,nt_splice_sites_all
$W $PY -W ignore $M/run_interventions.py --exp E1,E2,E3 --tiers 1
