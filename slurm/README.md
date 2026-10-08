# Experiment 3 Slurm resources

The deterministic task counts are 36 (smoke), 96 (pilot), and 288 (final).
The scripts submit three stages:

1. a single CPU score-table build;
2. an array in which every task owns one GPU and one output directory;
3. an `afterok` CPU aggregation task.

Default simulation request per array element:

- 1 GPU;
- 4 CPU cores;
- 16 GiB host RAM;
- 2 hours wall time.

Default aggregation request:

- 4 CPU cores;
- 32 GiB RAM;
- 4 hours wall time.

The largest final nonstationary kernel processes 65,536 particles in batches
of 16,384 for 3,520 integrator steps. Its live scientific arrays are far below 1 GiB; the 16
GiB host and a 24 GiB GPU budget leave margin for XLA compilation and driver
allocations. JAX preallocation is disabled.

Nested refinement records both integrator work and generation of the shared
finest-grid primitives. The revised pilot map contains `6,920,601,600`
integrator updates and `9,227,468,800` primitives (`16,148,070,400` EI work
particle-steps; `16,164,847,616` including exact work). The final contains
`20,761,804,800` integrator updates and `27,682,406,400` primitives
(`48,444,211,200` EI work particle-steps; `48,494,542,848` including exact
work). The older
integrator-only CPU extrapolation is therefore retired. The corrected CPU
pilot completed 96/96 tasks, passed every scientific gate, and took 983.961
seconds with four workers. No GPU wall time is asserted until a pilot has
produced measured metadata on the target GPU.

Go/no-go rule before final:

- all pilot tasks complete with no numerical warning;
- the submitter verifies the matching configuration/code/task-count pilot under
  `PILOT_OUTPUT_ROOT` before allocating the first final job;
- observed GPU memory remains below 20 GiB on a 24 GiB device;
- extrapolated aggregate GPU time remains below 36 hours;
- the finest two `N` values give marginal ECF changes no larger than the
  predeclared Monte Carlo envelope;
- otherwise add a finer pilot level rather than selecting a favorable `N`.

On sites where the GPU partition or resource spelling differs, change only the
`#SBATCH --gres=gpu:1`, partition/account directives, and wall-time request; do
not change the scientific TOML in an sbatch script.

Submit the pilot (the wrapper default) and then, only after authorization and a
passed matching pilot, the final:

```bash
MAX_CONCURRENT=4 bash slurm/submit_exp3.sh
ALLOW_PUBLICATION_SCALE=1 PILOT_OUTPUT_ROOT="$PWD/outputs" MAX_CONCURRENT=4 \
  bash slurm/submit_exp3.sh configs/final/exp3.toml outputs
```

`PILOT_OUTPUT_ROOT` may instead be supplied as the third positional argument.
It is forwarded to all three jobs and bound into Apptainer when it lies outside
the project/output mounts. The submitter verifies the full pilot receipt before
the first final `sbatch`, not merely the existence of a directory.

## Experiment 2 Slurm workflow

Experiment 2 has 12 smoke, 128 pilot, and 192 final tasks. A task is one
`(model,T,N,seed)` tuple and contains three propagated arms (main exact-`p_T`,
stationary-reference, and exact-`p_T` hires-table sensitivity) plus all three
independent exact references. The submission order is fixed:

1. `exp2_table.sbatch` builds and validates four score tables (stable/VP,
   each at main and independent high resolution);
2. `exp2_array.sbatch` maps each Slurm array index deterministically to one task;
3. `exp2_aggregate.sbatch` runs only after the complete array succeeds.

The array has an explicit `%MAX_CONCURRENT` cap and every task requests exactly
one GPU on the `gpu` partition. Scientific seeds come from the immutable TOML,
not from Slurm job IDs. Each task writes to its own atomic directory.

The current pilot records `484,442,112` integrator particle-steps across the
three arms plus `204,472,320` extra finest-grid primitives, hence
`688,914,432` work units. The final records `1,453,326,336` integrator steps
plus `613,416,960` extra primitives, hence `2,066,743,296` work units. The
four-worker CPU pilot completed end to end in 731.313 seconds. This is a CPU
wall-time measurement, not a GPU allocation estimate; GPU resources must still
be calibrated from a GPU pilot on the target cluster.

Default requests are:

- score tables: 8 CPUs, 24 GiB RAM, 90 min;
- each simulation task: 1 GPU, 4 CPUs, 24 GiB RAM, 4 h;
- aggregation: 8 CPUs, 32 GiB RAM, 4 h.

The scientific arrays are small relative to a 24 GiB GPU; the large resource
margin is for XLA compilation and FFT table construction. Go/no-go before the
final run:

- every pilot task and all four tables complete without a numerical warning;
- observed device memory remains below 20 GiB;
- all exact-reference ECF checks are compatible with their Monte Carlo envelope;
- the finest refinement points are stable under sample-size diagnostics;
- the estimated aggregate final cost remains below 36 GPU-hours.

Ruche with the Apptainer environment:

```bash
module purge
module load apptainer/1.4.4/gcc-15.1.0
apptainer build levy-exp2-0.1.0.sif containers/exp2.def
EXP2_SIF="$PWD/levy-exp2-0.1.0.sif" MAX_CONCURRENT=4 \
  bash slurm/submit_exp2.sh configs/pilot/exp2.toml outputs
```

The SIF supplies dependencies while both Python package roots are loaded from
the same host checkout through an explicit `PYTHONPATH`. External config,
output, and pilot roots are bound separately when needed. `CUDA_VISIBLE_DEVICES`
and CPU thread limits are forwarded through `--cleanenv`. The submitter resolves
and hashes the SIF, and it computes the task count inside that image.

The final wrapper refuses to submit without explicit authorization:

```bash
ALLOW_PUBLICATION_SCALE=1 PILOT_OUTPUT_ROOT="$PWD/outputs" \
  EXP2_SIF="$PWD/levy-exp2-0.1.0.sif" \
  MAX_CONCURRENT=4 \
  bash slurm/submit_exp2.sh configs/final/exp2.toml outputs
```

Do not call the three `.sbatch` files directly: the wrapper creates
`slurm_logs/`, resolves paths, and, before any allocation, validates both the
explicit final guard and the matching current-code pilot gate. It then wires
the dependencies. If Ruche changes its module or partition names, update only
the environment/resource directives after consulting the site documentation;
do not edit the scientific TOML in a job script.

## Experiment 1 Slurm workflow

Experiment 1 has 22 smoke, 60 pilot, and 132 final tasks. A dynamic task is one
`(model,nu,N,seed)` tuple and contains paired main/hires stationary-reference
arms, the exact-`p_T` control arm, and all three exact checkpoint references.
Direct S-alpha-S, signed-Pareto, and Gaussian controls are separate tasks. The
order is:

1. `exp1_table.sbatch`: CPU construction and validation of eight score tables
   (four model/target pairs, each at main and independent high resolution);
2. `exp1_array.sbatch`: deterministic one-GPU-per-task simulation array;
3. `exp1_aggregate.sbatch`: CPU statistics and figures after the whole array.

Default requests are 4 CPUs/24 GiB/90 min for tables, one GPU plus 4 CPUs and
24 GiB/4 h for each simulation task, and 8 CPUs/32 GiB/4 h for aggregation.
Particle batches are at most 16,384 in the final TOML, so scientific live
arrays are well below a 24 GiB GPU; the remaining margin covers XLA and driver
allocations. JAX preallocation is disabled.

The final task map consists of 48 primary dynamic tasks, 48 independent
refinement tasks, and 36 direct controls. Every dynamic task propagates paired
main/hires stationary-reference states plus an exact-`p_T` marginal-control
state. Primary tasks use 262,144 particles for each main/hires arm and 65,536
control particles at `N=80`; refinement uses 65,536/65,536/16,384 particles at
`N={40,80,160}`. The final work counter is `2,925,527,040` backward
particle-steps: `1,300,234,240` main, `1,300,234,240` hires, and `325,058,560`
control. The pilot counter is `330,301,440`. No post-change elapsed time has
been measured on GPU. The current four-worker CPU pilot completed 60/60 tasks,
passed every gate, and took 483.762 seconds. A GPU pilot must still supply the
throughput used for a definitive Slurm wall-time estimate.

Go/no-go before final:

- all pilot tasks and the eight tables finish with zero non-finite values;
- score validation remains below the frozen absolute-relative tolerances;
- exact-`p_T` marginal discrepancies stay within twice the independent
  exact-exact envelope;
- changes from `N=80` to `N=160` are below the prespecified split-half or
  exact-exact resolution for every main metric;
- the primary tail threshold has enough exceedances and sensitivity estimators
  do not manufacture a plateau;
- observed GPU memory is below 20 GiB and projected aggregate use below 36
  GPU-hours.

Submit the pilot through the wrapper, never by calling an sbatch file directly:

```bash
EXP1_SIF="$PWD/levy-exp1-0.1.0.sif" MAX_CONCURRENT=4 \
  bash slurm/submit_exp1.sh configs/pilot/exp1.toml outputs
```

After explicit authorization only:

```bash
ALLOW_PUBLICATION_SCALE=1 PILOT_OUTPUT_ROOT="$PWD/outputs" \
  EXP1_SIF="$PWD/levy-exp1-0.1.0.sif" \
  MAX_CONCURRENT=4 \
  bash slurm/submit_exp1.sh configs/final/exp1.toml outputs
```

The wrapper defaults to pilot, resolves all paths, computes the task count in
the SIF when present, binds external config/output/pilot roots, forwards
CPU/CUDA environment variables through `--cleanenv`, prints the SIF SHA-256,
and refuses a final submission before any `sbatch` unless the explicit guard
is set and the matching current-code pilot gate passes.
