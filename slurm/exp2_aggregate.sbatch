#!/usr/bin/env bash
#SBATCH --job-name=exp2-aggregate
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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

if [[ -n "${EXP2_SIF:-}" ]]; then
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
    "$EXP2_SIF" python -m experiments.exp2.aggregate \
    --config "$CONFIG" --device cpu --precision "$PRECISION" \
    --output-dir "$OUTPUT_ROOT" --resume "${EXTRA_ARGS[@]}"
else
  srun "$PYTHON_BIN" -m experiments.exp2.aggregate \
    --config "$CONFIG" --device cpu --precision "$PRECISION" \
    --output-dir "$OUTPUT_ROOT" --resume "${EXTRA_ARGS[@]}"
fi
