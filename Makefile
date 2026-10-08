PYTHON ?= python
export PYTHONPATH := $(CURDIR)/src:$(CURDIR)
OUTPUT_ROOT ?= outputs
PILOT_OUTPUT_ROOT ?= $(OUTPUT_ROOT)
SMOKE_CONFIG := configs/smoke/exp3.toml
PILOT_CONFIG := configs/pilot/exp3.toml
FINAL_CONFIG := configs/final/exp3.toml
EXP2_SMOKE_CONFIG := configs/smoke/exp2.toml
EXP2_PILOT_CONFIG := configs/pilot/exp2.toml
EXP2_FINAL_CONFIG := configs/final/exp2.toml
EXP2_INITIALIZATION_CONFIG := configs/postprocess/exp2_initialization.toml
EXP2_FINAL_RUN_DIR ?= $(OUTPUT_ROOT)/exp2-eaaf84c93847
EXP2_POSTPROCESS_OUTPUT ?= $(EXP2_FINAL_RUN_DIR)/postprocess
EXP1_SMOKE_CONFIG := configs/smoke/exp1.toml
EXP1_PILOT_CONFIG := configs/pilot/exp1.toml
EXP1_FINAL_CONFIG := configs/final/exp1.toml
EXP4_SMOKE_CONFIG := configs/smoke/exp4.toml
EXP4_FINAL_CONFIG := configs/final/exp4.toml
EXP4_SOURCE_RUN ?= $(OUTPUT_ROOT)/exp2-eaaf84c93847
ARTIFACT_FIGURE_OUTPUT ?= reproduced_figures

.PHONY: test lint artifact-smoke artifact-figures smoke-exp3 pilot-exp3 final-exp3 smoke-exp2 pilot-exp2 final-exp2 replot-exp2-initialization smoke-exp1 pilot-exp1 final-exp1 smoke-exp4 final-exp4

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

artifact-smoke:
	$(PYTHON) -m pytest -q tests/unit/test_random_and_stable.py tests/unit/test_integrator_and_theory.py tests/integration/test_exp1_pipeline.py tests/integration/test_exp2_pipeline.py tests/integration/test_exp3_pipeline.py tests/integration/test_exp4_pipeline.py

artifact-figures:
	$(PYTHON) scripts/replot_saved_results.py --data-root artifact_data --output-dir $(ARTIFACT_FIGURE_OUTPUT)

smoke-exp3:
	$(PYTHON) scripts/run_exp3.py --config $(SMOKE_CONFIG) --device cpu --output-dir $(OUTPUT_ROOT) --pilot-output-dir $(PILOT_OUTPUT_ROOT) --resume
	$(PYTHON) scripts/aggregate_exp3.py --config $(SMOKE_CONFIG) --device cpu --output-dir $(OUTPUT_ROOT) --pilot-output-dir $(PILOT_OUTPUT_ROOT) --resume

pilot-exp3:
	$(PYTHON) scripts/run_exp3.py --config $(PILOT_CONFIG) --device auto --output-dir $(OUTPUT_ROOT) --pilot-output-dir $(PILOT_OUTPUT_ROOT) --resume
	$(PYTHON) scripts/aggregate_exp3.py --config $(PILOT_CONFIG) --device cpu --output-dir $(OUTPUT_ROOT) --pilot-output-dir $(PILOT_OUTPUT_ROOT) --resume

final-exp3:
	$(PYTHON) scripts/run_exp3.py --config $(FINAL_CONFIG) --device gpu --output-dir $(OUTPUT_ROOT) --pilot-output-dir $(PILOT_OUTPUT_ROOT) --resume --allow-publication-scale
	$(PYTHON) scripts/aggregate_exp3.py --config $(FINAL_CONFIG) --device cpu --output-dir $(OUTPUT_ROOT) --pilot-output-dir $(PILOT_OUTPUT_ROOT) --resume --allow-publication-scale

smoke-exp2:
	$(PYTHON) -m experiments.exp2.run --config $(EXP2_SMOKE_CONFIG) --device cpu --precision float64 --output-dir $(OUTPUT_ROOT) --pilot-output-dir $(PILOT_OUTPUT_ROOT) --resume
	$(PYTHON) -m experiments.exp2.aggregate --config $(EXP2_SMOKE_CONFIG) --device cpu --precision float64 --output-dir $(OUTPUT_ROOT) --pilot-output-dir $(PILOT_OUTPUT_ROOT) --resume

pilot-exp2:
	$(PYTHON) -m experiments.exp2.run --config $(EXP2_PILOT_CONFIG) --device gpu --precision float64 --output-dir $(OUTPUT_ROOT) --pilot-output-dir $(PILOT_OUTPUT_ROOT) --resume
	$(PYTHON) -m experiments.exp2.aggregate --config $(EXP2_PILOT_CONFIG) --device cpu --precision float64 --output-dir $(OUTPUT_ROOT) --pilot-output-dir $(PILOT_OUTPUT_ROOT) --resume

final-exp2:
	$(PYTHON) -m experiments.exp2.run --config $(EXP2_FINAL_CONFIG) --device gpu --precision float64 --output-dir $(OUTPUT_ROOT) --pilot-output-dir $(PILOT_OUTPUT_ROOT) --resume --allow-publication-scale
	$(PYTHON) -m experiments.exp2.aggregate --config $(EXP2_FINAL_CONFIG) --device cpu --precision float64 --output-dir $(OUTPUT_ROOT) --pilot-output-dir $(PILOT_OUTPUT_ROOT) --resume --allow-publication-scale

replot-exp2-initialization:
	$(PYTHON) scripts/plot_exp2_initialization.py --run-dir "$(EXP2_FINAL_RUN_DIR)" --config $(EXP2_INITIALIZATION_CONFIG) --output-dir "$(EXP2_POSTPROCESS_OUTPUT)" --resume

smoke-exp1:
	$(PYTHON) -m experiments.exp1.run --config $(EXP1_SMOKE_CONFIG) --device cpu --precision float64 --output-dir $(OUTPUT_ROOT) --pilot-output-dir $(PILOT_OUTPUT_ROOT) --resume
	$(PYTHON) -m experiments.exp1.aggregate --config $(EXP1_SMOKE_CONFIG) --device cpu --precision float64 --output-dir $(OUTPUT_ROOT) --pilot-output-dir $(PILOT_OUTPUT_ROOT) --resume

pilot-exp1:
	$(PYTHON) -m experiments.exp1.run --config $(EXP1_PILOT_CONFIG) --device gpu --precision float64 --output-dir $(OUTPUT_ROOT) --pilot-output-dir $(PILOT_OUTPUT_ROOT) --resume
	$(PYTHON) -m experiments.exp1.aggregate --config $(EXP1_PILOT_CONFIG) --device cpu --precision float64 --output-dir $(OUTPUT_ROOT) --pilot-output-dir $(PILOT_OUTPUT_ROOT) --resume

final-exp1:
	$(PYTHON) -m experiments.exp1.run --config $(EXP1_FINAL_CONFIG) --device gpu --precision float64 --output-dir $(OUTPUT_ROOT) --pilot-output-dir $(PILOT_OUTPUT_ROOT) --resume --allow-publication-scale
	$(PYTHON) -m experiments.exp1.aggregate --config $(EXP1_FINAL_CONFIG) --device cpu --precision float64 --output-dir $(OUTPUT_ROOT) --pilot-output-dir $(PILOT_OUTPUT_ROOT) --resume --allow-publication-scale

smoke-exp4:
	$(PYTHON) -m experiments.exp4.run --config $(EXP4_SMOKE_CONFIG) --device cpu --precision float64 --output-dir $(OUTPUT_ROOT) --resume

final-exp4:
	$(PYTHON) -m experiments.exp4.run --config $(EXP4_FINAL_CONFIG) --source-run $(EXP4_SOURCE_RUN) --device cpu --precision float64 --output-dir $(OUTPUT_ROOT) --resume
