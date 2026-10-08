# Anonymous artifact: stable SP-SDE experiments

## Version-42 figure notation update

Quantile plots now use `kappa` for the quantile level and `1-kappa` for survival
mass, matching `mainaistats_42_notations (1).tex`. The two Experiment 1 PDFs
and PNG previews have been regenerated from the unchanged saved numerical
table. Filenames remain compatible with the manuscript's `figures/` references.
See `reports/NOTATION_V42.md` for the exact scope and validation.

To regenerate just these two figures without simulation, after the environment
setup below, use a fresh output directory:

```bash
python scripts/replot_saved_results.py --experiment 1 --data-root artifact_data --output-dir reproduced_figures_v42
```

This is a notation-only update, not a new mathematical audit of version 42.
The version-40 audit and historical execution receipts below remain historical.
Their figure-identity statements refer to the earlier releases. Simulation
equations, schedule `beta`, parameters, and saved numerical results are unchanged.
The MIT license and anonymous copyright designation are preserved.

This archive contains the CPU/GPU implementation, frozen configurations,
plot-ready seed summaries, and vector figures for the numerical experiments in
the accompanying anonymous manuscript. The published numerical values are not
recomputed during plotting.

The underlying implementation was audited against `mainaistats (40).tex` (source hash in
`src/levy_experiments/authority.py`). Start with `reports/VALIDATION_REPORT.md`
for those verified results and remaining manuscript actions. Numerical data
are unchanged; only the two quantile figures have updated annotations. Historical
formula reports describe execution-time provenance; `reports/MATH_V40.md`
records the version-40 equation-to-code check. No new publication-scale simulation was run.

Licensing revision (2026-10-05): the original supplementary materials are now
released under the MIT License in `LICENSE`. See `reports/LICENSING_RELEASE.md`
for the scope, checks, and replacement checklist answer. That licensing revision did
not change scientific code, configurations, numerical data, or figures.

The current manuscript includes four figure files:

| Manuscript file | Experiment | Archived source table |
| --- | --- | --- |
| `experiment4_ct.pdf` | deterministic denoiser modulus | `artifact_data/exp4/ct_curve.csv` |
| `experiment2_initialization_comparison_cropped.pdf` | initialization decay in empirical Wasserstein distance | `artifact_data/exp2/initialization_wp_summary.csv` |
| `experiment1_quantile_ratio.pdf` | target-specific extreme-quantile coverage | `artifact_data/exp1/tail_summary.csv` |
| `experiment1_main.pdf` | three-panel tail diagnostics in the appendix | `artifact_data/exp1/tail_summary.csv` |

Experiment 3 (same marginals, different path laws) remains implemented and its
seed-level summaries are included, but its figure is not referenced by the
current manuscript source. See `reports/TRACEABILITY.md` for the complete map.

## Contents

```text
artifact_data/     plot-ready final tables and seed-level summaries
configs/           smoke, pilot, final, and post-processing configurations
experiments/       experiment-specific simulation and analysis code
figures/           manuscript-ready PDF and PNG files
scripts/           execution, aggregation, verification, and replotting entry points
slurm/             one-GPU-per-task array workflows for Experiments 1--3
src/               shared stable generator, integrators, storage, and metrics
tests/             unit and end-to-end smoke tests
containers/        pinned CPU environment and CUDA-12/Apptainer specifications
reports/           formula, provenance, validation, and manuscript-action reports
```

Raw terminal particle arrays are not distributed: the executed final runs total
about 3.5 GiB, while the archived per-seed/aggregate tables are sufficient to
reproduce every displayed point, interval, and figure. The frozen configs and
commands below reproduce the raw arrays from scratch.

## CPU installation and quick verification

Python 3.12 is required. The saved final runs used Python 3.12.10 with JAX
0.7.2, jaxlib 0.7.2, NumPy 2.3.3, SciPy 1.16.2, and Matplotlib 3.10.6.

Linux/macOS shell:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r containers/requirements-cpu.lock
python -m pip install --no-build-isolation --no-deps .
export PYTHONPATH="$PWD/src:$PWD"
python -m pip check
make artifact-smoke
python scripts/verify_artifact.py
python scripts/verify_saved_statistics.py
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r containers\requirements-cpu.lock
& .\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps .
$env:PYTHONPATH = "$($PWD.Path)\src;$($PWD.Path)"
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -m pytest -q `
  tests\unit\test_random_and_stable.py `
  tests\unit\test_integrator_and_theory.py `
  tests\integration\test_exp1_pipeline.py `
  tests\integration\test_exp2_pipeline.py `
  tests\integration\test_exp3_pipeline.py `
  tests\integration\test_exp4_pipeline.py
& .\.venv\Scripts\python.exe scripts\verify_artifact.py
```

The smoke suite uses tiny, explicitly non-publication configurations.

Run from the archive root. Keep the displayed `PYTHONPATH` setting in each
new shell: simulation provenance hashes the source checkout, not a possibly
older installed package. The wheel install avoids Hatchling's optional
editable-build dependency; no unpinned editable installer is needed.

## Rebuild the manuscript figures without simulation

From a clean checkout after installation:

```bash
make artifact-figures
```

This writes into `reproduced_figures/` and leaves the distributed `figures/`
directory untouched. The equivalent command is:

```bash
python scripts/replot_saved_results.py \
  --data-root artifact_data \
  --output-dir reproduced_figures
```

The Experiment 2 dotted curves remain the exact--exact finite-sample Monte
Carlo references. All method markers are filled. The numerical resolution
diagnostic is retained in the CSV columns `mc_resolved_fraction` and
`marker_resolved`, but it has no graphical encoding and is not a legend item.

PDF bytes are deterministic in the validated local environment, but no
bitwise-identity claim is made across operating systems, Matplotlib/font
stacks, or accelerator backends. Scientific table hashes and displayed values
are the cross-platform invariants.

## Full publication-scale reproduction

Use Linux with one NVIDIA CUDA GPU per simulation task, or the validated CPU
wrapper. Final runs are guarded: the matching pilot must complete and pass its
scientific gates before the final command is accepted.

Use a new output root for this code revision. Do not resume a historical pilot
or simulation with changed source hashes: the simulation guards intentionally
reject that mixture. Read-only replotting still uses the frozen tables.

```bash
mkdir -p outputs/pilot outputs/final

# Experiment 3: pilot then final
make pilot-exp3 OUTPUT_ROOT=outputs/pilot PILOT_OUTPUT_ROOT=outputs/pilot
make final-exp3 OUTPUT_ROOT=outputs/final PILOT_OUTPUT_ROOT=outputs/pilot

# Experiment 2: pilot then final
make pilot-exp2 OUTPUT_ROOT=outputs/pilot PILOT_OUTPUT_ROOT=outputs/pilot
make final-exp2 OUTPUT_ROOT=outputs/final PILOT_OUTPUT_ROOT=outputs/pilot

# Experiment 2 manuscript post-processing (p=1.1 and p=1.4)
python scripts/plot_exp2_initialization.py \
  --run-dir outputs/final/exp2-eaaf84c93847 \
  --config configs/postprocess/exp2_initialization.toml \
  --output-dir outputs/final/exp2-eaaf84c93847/postprocess \
  --resume

# Experiment 1: pilot then final
make pilot-exp1 OUTPUT_ROOT=outputs/pilot PILOT_OUTPUT_ROOT=outputs/pilot
make final-exp1 OUTPUT_ROOT=outputs/final PILOT_OUTPUT_ROOT=outputs/pilot

# Experiment 4 is deterministic CPU/FFT analysis and reuses Experiment 2 tables
make final-exp4 OUTPUT_ROOT=outputs/final \
  EXP4_SOURCE_RUN=outputs/final/exp2-eaaf84c93847
```

The frozen final task counts are 288 (Experiment 3), 192 (Experiment 2),
and 132 (Experiment 1). All final propagation modes are `float64`. Scientific
seeds are listed in the TOML files; Slurm array indices never replace them.
The Slurm submission commands and resource requests are documented in
`slurm/README.md`.

For CPU reproduction on Windows or Linux, use the following sequence instead
of the GPU Make targets (replace `python` by your environment's interpreter).
Run these pairs in order, for experiment 3, then 2, then 1:

```bash
python scripts/run_cpu_parallel.py --experiment 3 --config configs/pilot/exp3.toml --output-dir outputs/pilot --jobs 4 --resume
python scripts/run_cpu_parallel.py --experiment 3 --config configs/final/exp3.toml --output-dir outputs/final --pilot-output-dir outputs/pilot --jobs 4 --resume --allow-publication-scale
python scripts/run_cpu_parallel.py --experiment 2 --config configs/pilot/exp2.toml --output-dir outputs/pilot --jobs 4 --resume
python scripts/run_cpu_parallel.py --experiment 2 --config configs/final/exp2.toml --output-dir outputs/final --pilot-output-dir outputs/pilot --jobs 4 --resume --allow-publication-scale
python scripts/plot_exp2_initialization.py --run-dir outputs/final/exp2-eaaf84c93847 --config configs/postprocess/exp2_initialization.toml --output-dir outputs/final/exp2-eaaf84c93847/postprocess --resume
python scripts/run_cpu_parallel.py --experiment 1 --config configs/pilot/exp1.toml --output-dir outputs/pilot --jobs 4 --resume
python scripts/run_cpu_parallel.py --experiment 1 --config configs/final/exp1.toml --output-dir outputs/final --pilot-output-dir outputs/pilot --jobs 4 --resume --allow-publication-scale
python -m experiments.exp4.run --config configs/final/exp4.toml --source-run outputs/final/exp2-eaaf84c93847 --device cpu --precision float64 --output-dir outputs/final --resume
```

The wrapper includes aggregation. Experiment 4 accepts either the frozen
Experiment 2 execution or one generated with the current distribution; it
still requires matching configuration, passed gates and exact content hashes
for all four score tables. Unrecognized historical code is rejected. Bitwise
table identity on another platform is not guaranteed; a failed hash comparison
requires a documented comparison, never bypassing the guard.

The archived finals were actually executed on a Windows 11 CPU workstation
with 16 logical AMD64 CPUs, not on a GPU. Observed file-timestamp spans were
approximately 46.1 minutes (Experiment 3), 22.5 minutes (Experiment 2), and
22.8 minutes (Experiment 1) with the documented four-worker CPU wrapper.
Experiment 4 took 95.366 seconds on the same class of machine. Timestamp spans
are observational wall-clock estimates; per-task timing metadata is the more
precise performance record. GPU parity tests exist but were skipped locally
because no NVIDIA GPU was available.

## Reproducibility and interpretation boundaries

- Stable variables use `E exp(i u Z) = exp(-|u|^alpha)`; no clipping or
  coordinate-wise approximation is used in these one-dimensional experiments.
- Experiment 2 reports empirical one-dimensional Wasserstein distances at
  `p=1.1` and `p=1.4`, both strictly below `alpha=1.5`; no stable `W_2` is
  reported.
- Exact--exact curves are sampling-resolution references, not lower bounds and
  are never subtracted.
- Experiment 2 orders 1.1/1.4 were selected during post-processing. Temporal
  controls at those orders have been recomputed from saved particles and are
  included as `artifact_data/exp2/temporal_refinement_recomputed.csv`.
  Score-table sensitivity was originally measured at orders 1/1.25; the saved
  runs do not retain the high-resolution arm's particles, so an order-1.4
  score-sensitivity gate cannot be certified retrospectively.
- Experiment 4 is deterministic numerical Fourier evaluation of a truncated
  modulus, not a closed-form population supremum; statistical error bars would
  be inappropriate.
- Equality of marginal flows in Experiment 3 is not equality of path laws.
- Numerical experiments illustrate the theory and do not prove its theorems.

## Anonymity, policy, and licensing

The distributed files contain no author name, affiliation, local user path,
Git history, virtual environment, cache, or raw log. The AISTATS 2027 policy
check and required author actions are in `reports/MANUSCRIPT_ACTIONS.md`.

No third-party package is vendored. `THIRD_PARTY.md` records the direct
dependencies and their upstream licenses. `LICENSE` provides the MIT License
for the original code and accompanying original repository material, including
configuration files, documentation, and generated numerical outputs and figures.
It permits reuse, modification, and redistribution, including commercial use,
subject to retaining its copyright and permission notices. Third-party software
retains its own licenses; the MIT grant does not relicense dependencies. The
manuscript itself is not included in or licensed by this code archive.

The copyright-holder designation is anonymous for peer review. Replace it with
the actual rights holders for an identified public release, after confirming
the relevant author and institutional rights. Licensing does not attest to
scientific validity or resolve the other manuscript actions.

`artifact_data/source_inventory.json` records the historical source/configuration
hashes, verified raw-task checksums, dependencies and score-table hashes.
These receipts are not assertions that simulations were executed with this
new code revision. Final-PDF layout and complete AI disclosure require the
author checks listed in `reports/MANUSCRIPT_ACTIONS.md`.
