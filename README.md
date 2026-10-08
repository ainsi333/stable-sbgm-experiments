# Stable SP-SDE Experiments

Research code for **From Global to Tail Convergence: Dynamics of
$\alpha$-Stable Score-Based Generative Models**.

This repository studies marginal dynamics, initialization error, and tail
coverage in stable score-based generative models, with a Brownian VP-SDE
baseline. It includes simulation code, frozen configurations, saved statistics,
vector figures, and tests. Propagation uses JAX; numerical references and
analysis use NumPy/SciPy.

**To reproduce the figures, no new simulations or GPU are required.** Install
the CPU environment and follow [Rebuild the figures](#rebuild-the-figures).

## Experiments

All distributed experiments are one-dimensional.

| Experiment | Question | Setting |
| --- | --- | --- |
| **1 — Tail coverage** | How do tail index and quantile coverage depend on the target? | Student targets with $\nu\in\{1.2,1.5,3\}$; stable model with $\alpha=1.5$, $\eta=0.5$; VP comparison at $\nu=1.5$. |
| **2 — Initialization** | How does initialization error evolve with the forward horizon? | Student-$t_4$ target; stable and VP models; exact-forward initialization controls and empirical $W_{1.1}$/$W_{1.4}$ comparisons. |
| **3 — Marginals and paths** | Can a backward family share marginal distributions while having different joint laws? | Analytic stationary stable transitions and a nonstationary three-atom target, varying $\eta$. |
| **4 — Denoiser modulus** | How does the numerical modulus $C_t$ compare between stable and Brownian models? | Student target; deterministic calculations from score tables, with $1/a(t)$ as a reference. |

Experiment 3 is a numerical control retained with its saved summaries. Its
implementation lives in `src/levy_experiments/`, not `experiments/exp3/`;
the four distributed manuscript PDFs cover Experiments 1, 2, and 4.

## Installation

Run all commands from the repository root. **Python 3.12** is the validated
reproduction environment. The pinned CPU stack includes JAX/jaxlib 0.7.2,
NumPy 2.3.3, SciPy 1.16.2, and Matplotlib 3.10.6. No CUDA installation is needed.

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r containers/requirements-cpu.lock
python -m pip install --no-build-isolation --no-deps .
export PYTHONPATH="$PWD/src:$PWD"
python -m pip check
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r containers\requirements-cpu.lock
& $py -m pip install --no-build-isolation --no-deps .
$env:PYTHONPATH = "$($PWD.Path)\src;$($PWD.Path)"
& $py -m pip check
```

In the remaining commands, replace `python` with `& $py` when using
PowerShell. Set `PYTHONPATH` again in each new shell: it ensures that execution
and provenance refer to this checkout rather than an older installed copy.
The platform used for local validation was Windows CPU; other platforms should
run the checks below before a full experiment.

## Verify the installation and saved results

```bash
python scripts/verify_artifact.py
python scripts/verify_saved_statistics.py
```

These commands check the packaged-file inventory, scientific data and figure
hashes, and statistics recomputed from saved per-seed tables.

Run the small CPU integration pipelines and core numerical tests:

```bash
python -m pytest -q tests/unit/test_random_and_stable.py tests/unit/test_integrator_and_theory.py tests/integration/test_exp1_pipeline.py tests/integration/test_exp2_pipeline.py tests/integration/test_exp3_pipeline.py tests/integration/test_exp4_pipeline.py
```

This is equivalent to `make artifact-smoke` where Make is available. These
are small test cases, not publication-scale runs. For the complete test suite
and lint checks:

```bash
python -m pytest -p no:cacheprovider -q
python -m ruff check . --no-cache
```

## Rebuild the figures

Regenerate all four manuscript figures directly from the saved statistics:

```bash
python scripts/replot_saved_results.py --data-root artifact_data --output-dir reproduced_figures
```

Equivalent Make target: `make artifact-figures`.

| Figure | Saved input |
| --- | --- |
| [Tail diagnostics](figures/experiment1_main.pdf) | `artifact_data/exp1/tail_summary.csv` |
| [Quantile coverage](figures/experiment1_quantile_ratio.pdf) | `artifact_data/exp1/tail_summary.csv` |
| [Initialization error](figures/experiment2_initialization_comparison_cropped.pdf) | `artifact_data/exp2/initialization_wp_summary.csv` |
| [Denoiser modulus](figures/experiment4_ct.pdf) | `artifact_data/exp4/ct_curve.csv` |

The command writes vector PDFs, PNG previews, and a figure manifest. It does
not modify `figures/`. Use a fresh output directory; overwriting an existing
destination requires `--force`.

For only the two tail figures:

```bash
python scripts/replot_saved_results.py --experiment 1 --data-root artifact_data --output-dir reproduced_figures/quantiles
```

Figure manifests link outputs to their source tables. PDF byte identity is not
guaranteed across operating systems or font stacks; numerical values and data
hashes remain the primary cross-platform checks.

## Run the experiments

Simulation is separate from plotting. The recommended order is **3 → 2 → 1 → 4**:
first validate the backward dynamics, then initialization and distributional
metrics, then tail coverage; Experiment 4 reuses Experiment 2 score tables.

The CPU runner handles task execution and aggregation. Each publication-scale
run requires its matching successful pilot and the explicit
`--allow-publication-scale` flag. Adjust `--jobs` to the available CPU and RAM;
it controls concurrency, not the scientific seeds or sample sizes.

<details>
<summary>Complete CPU reproduction commands, in order</summary>

Run each command only after the preceding command completes successfully.

```bash
# 1. Common marginals and different path laws
python scripts/run_cpu_parallel.py --experiment 3 --config configs/pilot/exp3.toml --output-dir outputs/pilot --jobs 4 --resume
python scripts/run_cpu_parallel.py --experiment 3 --config configs/final/exp3.toml --output-dir outputs/final --pilot-output-dir outputs/pilot --jobs 4 --resume --allow-publication-scale

# 2. Initialization error and the displayed Wasserstein comparisons
python scripts/run_cpu_parallel.py --experiment 2 --config configs/pilot/exp2.toml --output-dir outputs/pilot --jobs 4 --resume
python scripts/run_cpu_parallel.py --experiment 2 --config configs/final/exp2.toml --output-dir outputs/final --pilot-output-dir outputs/pilot --jobs 4 --resume --allow-publication-scale
python scripts/plot_exp2_initialization.py --run-dir outputs/final/exp2-eaaf84c93847 --config configs/postprocess/exp2_initialization.toml --output-dir outputs/final/exp2-eaaf84c93847/postprocess --resume

# 3. Tail coverage
python scripts/run_cpu_parallel.py --experiment 1 --config configs/pilot/exp1.toml --output-dir outputs/pilot --jobs 4 --resume
python scripts/run_cpu_parallel.py --experiment 1 --config configs/final/exp1.toml --output-dir outputs/final --pilot-output-dir outputs/pilot --jobs 4 --resume --allow-publication-scale

# 4. Denoiser modulus, reusing the completed Experiment 2 run
python -m experiments.exp4.run --config configs/final/exp4.toml --source-run outputs/final/exp2-eaaf84c93847 --device cpu --precision float64 --output-dir outputs/final --resume
```

</details>

The frozen final configurations use `float64` and define 288, 192, and 132
tasks for Experiments 3, 2, and 1, respectively. The Experiment 2 directory
above corresponds to the supplied configuration; a changed configuration may
produce a different run identifier.

Use a new output root when changing the scientific code or configurations.
`--resume` accepts compatible completed work; it is not a way to reuse results
from a different code version. Do not bypass a failed pilot or provenance gate.
Experiment 4 requires the explicit `--source-run` override shown above because
its frozen configuration records a historical path. It checks the source run,
passed gates, and score-table hashes.

### Optional NVIDIA GPU / Slurm execution

GPU propagation targets Linux with CUDA. Container recipes and environment
files are in [containers/](containers/); submission wrappers, resource requests,
and Docker/Apptainer instructions are in [slurm/README.md](slurm/README.md).
Use the submission wrappers rather than invoking individual `.sbatch` files:
they validate pilots and connect table construction, simulation, and aggregation.

Each array task owns one GPU and its own output directory. Run a pilot on the
target machine before choosing concurrency or wall-time limits. GPU parity,
container execution, and Slurm execution were not validated on the local CPU
machine. The CUDA requirements file pins direct requirements, not every
transitive NVIDIA package. Experiment 4 is CPU/FFT work.

## Data and reproducibility

The repository includes per-seed and aggregate statistics, numerical
diagnostics, frozen configurations and seeds, source/configuration hashes,
and the final PDFs. These are sufficient to regenerate the distributed figures
and verify the saved summaries.

Full terminal particle arrays, score-table caches, and raw execution logs are
not distributed. Historical terminal arrays alone occupied approximately
3.5 GiB; rerunning the experiments reconstructs the raw outputs. Simulation
outputs record configuration/code identities, task seeds, software/backend
metadata, and timings. Resume and aggregation checks reject incompatible runs.

[SHA256SUMS](SHA256SUMS) covers the distributed files except itself.
The integrity check also detects unlisted files outside designated runtime
directories. When intentionally editing repository files or adding release
files, update the manifest and rerun verification; do not disable the check.
Use `outputs/` and `reproduced_figures/` for generated results.

## Interpretation and limitations

- **Stable convention.** The scalar reference law satisfies
  $\mathbb{E}[e^{iuZ}]=e^{-|u|^\alpha}$. Numerical samplers use finite steps
  and stop at positive $\varepsilon$. Scores come from analytic expressions
  or numerical tables, not learned neural networks.
- **Marginals are not path laws.** Experiment 3 tests a family reproducing
  marginal flows; it does not identify every member with the true time-reversed
  process. Its intentionally incorrect hybrid is a negative control.
- **Wasserstein comparisons.** Experiment 2 displays $p=1.1$ and $p=1.4$,
  both below $\alpha=1.5$. These display orders were selected in post-processing.
  Exact–exact curves measure finite-sample Monte Carlo resolution; they are
  neither lower bounds nor quantities to subtract from measured errors.
- **Order-specific validation.** Temporal refinement at the displayed orders
  is saved. Score-table sensitivity was evaluated at $p=1$ and $p=1.25$;
  high-resolution-arm particles were not retained, so an order-$1.4$
  score-sensitivity gate cannot be certified retrospectively.
- **Uncertainty.** Experiment 2 uses 90% seed-bootstrap intervals. Tail bands
  are descriptive 16–84% inter-seed ranges, not confidence intervals.
  Experiment 4 reports a deterministic finite-domain numerical maximum, not
  an exact global supremum; it has no sampling-error bars.
- **Scope.** Finite-sample diagnostics do not prove theorems, exact tail
  equivalence, or continuous-time convergence rates. Inconclusive tail
  diagnostics are retained. No universal stable-over-Brownian superiority
  is claimed.

For detailed provenance and numerical checks, see the
[validation report](reports/VALIDATION_REPORT.md),
[formula audit](reports/MATH_V40.md), and
[traceability map](reports/TRACEABILITY.md).
These reports retain their historical scope. The subsequent
[quantile-notation update](reports/NOTATION_V42.md) changes labels to
$\kappa$, not scientific data, and is not a new mathematical audit.

## Repository layout

```text
.
├── artifact_data/       # Saved statistics, diagnostics, and provenance
├── configs/             # Smoke, pilot, final, and post-processing settings
├── containers/          # Dependency locks and container recipes
├── experiments/         # Experiment 1, 2, and 4 entry points and analysis
├── figures/             # Four vector PDFs and their source manifest
├── reports/             # Mathematical and numerical validation records
├── scripts/             # Run, verify, aggregate, and replot commands
├── slurm/               # Cluster wrappers and job definitions
├── src/levy_experiments/ # Shared numerics and Experiment 3
├── tests/               # Unit and integration tests
├── Makefile
├── pyproject.toml
├── SHA256SUMS
├── LICENSE
└── THIRD_PARTY.md
```

## License

Original code and accompanying original repository material are distributed
under the [MIT License](LICENSE). Dependencies retain their own licenses;
see [THIRD_PARTY.md](THIRD_PARTY.md). The manuscript is not included and is
not licensed by this repository.
