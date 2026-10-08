#!/usr/bin/env bash
#SBATCH --job-name=exp3-sim
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

: "${PROJECT_ROOT:?PROJECT_ROOT is required}"
: "${CONFIG:?CONFIG is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${PILOT_OUTPUT_ROOT:?PILOT_OUTPUT_ROOT is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
: "${PRECISION:=float64}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXTRA_ARGS=(--pilot-output-dir "$PILOT_OUTPUT_ROOT")
if [[ "${ALLOW_PUBLICATION_SCALE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--allow-publication-scale)
fi

if [[ -n "${EXP3_SIF:-}" ]]; then
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
  srun apptainer exec --nv --cleanenv \
    "${BIND_ARGS[@]}" --pwd "$PROJECT_ROOT" \
    --env "PYTHONPATH=$PROJECT_ROOT/src:$PROJECT_ROOT" \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
    --env "OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK" \
    --env "MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK" \
    --env "OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK" \
    "$EXP3_SIF" python scripts/run_exp3.py \
    --config "$CONFIG" --device gpu --precision "$PRECISION" \
    --output-dir "$OUTPUT_ROOT" --resume \
    --task-index "$SLURM_ARRAY_TASK_ID" "${EXTRA_ARGS[@]}"
else
  srun "$PYTHON_BIN" scripts/run_exp3.py \
    --config "$CONFIG" --device gpu --precision "$PRECISION" \
    --output-dir "$OUTPUT_ROOT" --resume \
    --task-index "$SLURM_ARRAY_TASK_ID" \
    "${EXTRA_ARGS[@]}"
fi
