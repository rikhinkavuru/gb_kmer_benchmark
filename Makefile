# Makefile -- one target per upgrade module. CPU-only.
#
# The working interpreter is the venv that matches requirements.txt (see MAP.md). Override with
# `make PY=/path/to/python <target>`. The FM probe (Upgrade 1) needs the optional [fm] extra
# (torch/transformers); Upgrade 6 needs the [coords] extra (mappy) + the GRCh38 FASTA; everything
# else is torch-free.
#
# IMPORTANT (Upgrade 1): embedding (torch) and head-training (LightGBM) MUST run in separate
# processes -- both load their own OpenMP and segfault if co-resident on macOS. The upgrade1 target
# does this automatically (extract step, then probe + paired steps).

PY ?= /Users/rikhinkavuru/Downloads/venv/bin/python
THREADS ?= 6
FM_DATASETS ?= human_enhancers_cohn,nt_enhancers,drosophila_enhancers_stark

UP := src/upgrades

.PHONY: help test setup-fm setup-coords all \
        upgrade1 upgrade1-extract upgrade1-probe upgrade2 upgrade3 upgrade4 upgrade5 \
        upgrade6 upgrade7 upgrade8 upgrade9 upgrade10 \
        finetune presence-null route3-specificity-extra paperfigs

help:
	@echo "Setup:    setup-fm (torch/transformers), setup-coords (mappy)"
	@echo "Upgrades: upgrade1 (FM-probe) .. upgrade10 (diagnostic CLI); 'all' runs 2..10 (1 needs setup-fm)"
	@echo "Tests:    test (all module sanity checks)"

setup-fm:
	uv pip install --python $(PY) -r requirements-fm.txt
setup-coords:
	uv pip install --python $(PY) -r requirements-coords.txt

# ---- Upgrade 1: frozen FM-probe (two processes: torch extract, then lightgbm probe + paired) ----
upgrade1-extract:
	$(PY) $(UP)/fm_probe/extract_embeddings.py --datasets $(FM_DATASETS) --threads $(THREADS)
upgrade1-probe:
	$(PY) $(UP)/fm_probe/run_fm_probe.py --datasets $(FM_DATASETS)
	$(PY) $(UP)/fm_probe/run_fm_paired.py --datasets $(FM_DATASETS)
upgrade1: upgrade1-extract upgrade1-probe
	$(PY) $(UP)/fm_probe/make_fig_fm_probe.py
	$(PY) $(UP)/fm_probe/make_fig_fm_paired.py

# ---- Upgrade 2: composition-equalized cleaning (run BEFORE upgrade1; emits its splits) ----
upgrade2:
	$(PY) $(UP)/composition_clean/run_composition_clean.py
	$(PY) $(UP)/composition_clean/make_fig_composition_clean.py

upgrade3:
	$(PY) $(UP)/disc_dose/run_disc_dose.py
	$(PY) $(UP)/disc_dose/make_fig_disc_dose.py

upgrade4:
	$(PY) $(UP)/positional_sweep/run_positional_sweep.py
	$(PY) $(UP)/positional_sweep/make_fig_positional_sweep.py

upgrade5:
	$(PY) $(UP)/shuffled_neg/run_shuffled_neg.py
	$(PY) $(UP)/shuffled_neg/make_fig_shuffled_neg.py

upgrade6:   # needs setup-coords + localization/GRCh38.primary_assembly.fa.gz
	$(PY) $(UP)/nt_coords/run_nt_coords.py
	$(PY) $(UP)/nt_coords/make_fig_nt_coords.py

upgrade7:
	$(PY) $(UP)/route1_predict/run_route1_predict.py
	$(PY) $(UP)/route1_predict/make_fig_route1_predict.py

upgrade8:
	$(PY) $(UP)/multiseed/run_multiseed.py
	$(PY) $(UP)/multiseed/make_fig_multiseed.py

upgrade9:
	$(PY) $(UP)/route_invariance/run_route_invariance.py
	$(PY) $(UP)/route_invariance/make_fig_route_invariance.py

upgrade10:
	$(PY) $(UP)/diagnose/diagnose.py --input $(UP)/diagnose/toy_dataset.csv --out $(UP)/diagnose/example_report --name toy_dataset

# torch-free upgrades, in dependency order (2 emits splits consumed by 1)
all: upgrade2 upgrade3 upgrade4 upgrade5 upgrade7 upgrade8 upgrade9 upgrade10

test:
	$(PY) $(UP)/composition_clean/tests/test_sanity.py
	$(PY) $(UP)/fm_probe/tests/test_sanity.py
	$(PY) $(UP)/fm_probe/tests/test_paired_sanity.py
	$(PY) $(UP)/disc_dose/tests/test_sanity.py
	$(PY) $(UP)/positional_sweep/tests/test_sanity.py
	$(PY) $(UP)/shuffled_neg/tests/test_sanity.py
	$(PY) $(UP)/route1_predict/tests/test_sanity.py
	$(PY) $(UP)/multiseed/tests/test_sanity.py
	$(PY) $(UP)/route_invariance/tests/test_sanity.py
	$(PY) $(UP)/nt_coords/tests/test_sanity.py
	$(PY) $(UP)/diagnose/tests/test_sanity.py
	$(PY) $(UP)/composition_clean/tests/test_composition_fraction.py
	$(PY) $(UP)/route3_specificity/tests/test_sanity.py
	$(PY) $(UP)/fm_probe/tests/test_head_invariance.py
	$(PY) $(UP)/disc_dose/tests/test_slopes.py
	$(PY) $(UP)/fm_probe/tests/test_dnabert.py
	$(PY) $(UP)/fm_probe/tests/test_reproduce_from_cache.py
	$(PY) $(UP)/fm_finetune/tests/test_sanity.py
	$(PY) $(UP)/presence_null/tests/test_sanity.py

# ---- PSB revision additions ----
# E1: fine-tuned FM arm (needs setup-fm; MPS or CPU). Trains 3 arms x 3 seeds per dataset.
finetune:
	$(PY) -W ignore $(UP)/fm_finetune/run_fm_finetune.py --dataset human_enhancers_cohn
	$(PY) -W ignore $(UP)/fm_finetune/run_fm_finetune.py --dataset nt_enhancers \
		--out results/upgrades/fm_finetune_nt.csv
# E2: motif-presence null extended to all 15 tasks, with confidence intervals (torch-free)
presence-null:
	$(PY) $(UP)/presence_null/run_presence_null.py
# E3: Route-3 specificity on the two remaining large GB tasks (torch-free, slow)
route3-specificity-extra:
	$(PY) $(UP)/composition_clean/run_composition_clean.py \
		--datasets human_ocr_ensembl,human_enhancers_ensembl \
		--out results/upgrades/composition_clean_specificity.csv \
		--balance-out results/upgrades/composition_clean_specificity_balance.csv \
		--splits-dir cleaned_splits_specificity
	$(PY) $(UP)/shuffled_neg/run_shuffled_neg.py \
		--datasets human_ocr_ensembl,human_enhancers_ensembl \
		--out results/upgrades/shuffled_neg_specificity.csv
# paper figures (schematic + the two composite panels)
paperfigs:
	$(PY) $(UP)/paperfigs/make_fig_routes.py
	$(PY) $(UP)/paperfigs/make_fig_task_landscape.py
	$(PY) $(UP)/paperfigs/make_fig_fm_combined.py

# ---- DEPTH upgrades (deepen existing claims; reuse cached artifacts) ----
.PHONY: depth depth1-extract depth1-probe
depth2-compfrac:
	$(PY) $(UP)/composition_clean/run_composition_fraction.py
	$(PY) $(UP)/composition_clean/make_fig_composition_fraction.py
depth3-specificity:
	$(PY) $(UP)/route3_specificity/run_route3_specificity.py
	$(PY) $(UP)/route3_specificity/make_fig_route3_specificity.py
depth4-headinv:
	$(PY) $(UP)/fm_probe/run_fm_head_invariance.py
	$(PY) $(UP)/fm_probe/make_fig_fm_head_invariance.py
depth5-slopes:
	$(PY) $(UP)/disc_dose/run_disc_slopes.py
	$(PY) $(UP)/disc_dose/make_fig_disc_slopes.py
# Depth 1 (DNABERT-2): torch extract, then lightgbm paired probe + cross-architecture table (separate processes)
depth1-extract:   # needs setup-fm
	$(PY) $(UP)/fm_probe/extract_embeddings_dnabert.py --datasets human_enhancers_cohn,nt_enhancers --threads $(THREADS)
depth1-probe:
	$(PY) $(UP)/fm_probe/run_fm_paired_dnabert.py
	$(PY) $(UP)/fm_probe/make_cross_architecture.py
depth1: depth1-extract depth1-probe
depth: depth2-compfrac depth3-specificity depth4-headinv depth5-slopes   # torch-free depth items
