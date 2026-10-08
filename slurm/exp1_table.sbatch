#!/usr/bin/env bash
#SBATCH --job-name=exp1-table
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

: "${PROJECT_ROOT:?PROJECT_ROOT is required}"
: "${CONFIG:?CONFIG is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${PILOT_OUTPUT_ROOT:?PILOT_OUTPUT_ROOT is required}"
: "${PRECISION:=float64}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

EXTRA_ARGS=(--pilot-output-dir "$PILOT_OUTPUT_ROOT")
if [[ "${ALLOW_PUBLICATION_SCALE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--allow-publication-scale)
fi

if [[ -n "${EXP1_SIF:-}" ]]; then
  BIND_ARGS=(--bind "$PROJECT_ROOT:$PROJECT_ROOT")
  if [[ "$OUTPUT_ROOT" != "$PROJECT_ROOT" && "$OUTPUT_ROOT" != "$PROJECT_ROOT/"* ]]; then
    BIND_ARGS+=(--bind "$OUTPUT_ROOT:$OUTPUT_ROOT")
  fi
  if [[ "$PILOT_OUTPUT_ROOT" != "$PROJECT_ROOT" && "$PILOT_OUTPUT_ROOT" != "$PROJECT_ROOT/"* \
    && "$PILOT_OUTPUT_ROOT" != "$OUTPUT_ROOT" && "$PILOT_OUTPUT_ROOT" != "$OUTPUT_ROOT/"* ]]; then
    BIND_ARGS+=(--bind "$PILOT_OUTPUT_ROOT:$PILOT_OUTPUT_ROOT")
  fi
  CONFIG_ROOT="$(dirname "$CONFIG")"
  if [[ "$CONFIG_ROOT" != "$PROJECT_ROOT" && "$CONFIG_ROOT" != "$PROJECT_ROOT/"* ]]; then
    BIND_ARGS+=(--bind "$CONFIG_ROOT:$CONFIG_ROOT")
  fi
  srun apptainer exec --cleanenv \
    "${BIND_ARGS[@]}" --pwd "$PROJECT_ROOT" \
    --env "PYTHONPATH=$PROJECT_ROOT/src:$PROJECT_ROOT" \
    --env "OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK" \
    --env "MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK" \
    --env "OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK" \
    "$EXP1_SIF" python -m experiments.exp1.run \
    --config "$CONFIG" --device cpu --precision "$PRECISION" \
    --output-dir "$OUTPUT_ROOT" --resume --table-only "${EXTRA_ARGS[@]}"
else
  srun "$PYTHON_BIN" -m experiments.exp1.run \
    --config "$CONFIG" --device cpu --precision "$PRECISION" \
    --output-dir "$OUTPUT_ROOT" --resume --table-only "${EXTRA_ARGS[@]}"
fi
