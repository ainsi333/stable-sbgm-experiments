#!/usr/bin/env bash
#SBATCH --job-name=exp3-table
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
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
  srun apptainer exec --cleanenv \
    "${BIND_ARGS[@]}" --pwd "$PROJECT_ROOT" \
    --env "PYTHONPATH=$PROJECT_ROOT/src:$PROJECT_ROOT" \
    "$EXP3_SIF" python scripts/run_exp3.py \
    --config "$CONFIG" --device cpu --precision "$PRECISION" \
    --output-dir "$OUTPUT_ROOT" --resume --table-only "${EXTRA_ARGS[@]}"
else
  srun "$PYTHON_BIN" scripts/run_exp3.py \
    --config "$CONFIG" --device cpu --precision "$PRECISION" \
    --output-dir "$OUTPUT_ROOT" --resume --table-only "${EXTRA_ARGS[@]}"
fi
