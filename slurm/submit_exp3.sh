#!/usr/bin/env bash
set -euo pipefail

CONFIG="$(realpath "${1:-configs/pilot/exp3.toml}")"
mkdir -p "${2:-outputs}" slurm_logs
OUTPUT_ROOT="$(realpath "${2:-outputs}")"
PILOT_OUTPUT_INPUT="${PILOT_OUTPUT_ROOT:-${3:-$OUTPUT_ROOT}}"
if [[ ! -d "$PILOT_OUTPUT_INPUT" ]]; then
  echo "Pilot output root not found: $PILOT_OUTPUT_INPUT" >&2
  exit 2
fi
PILOT_OUTPUT_ROOT="$(realpath "$PILOT_OUTPUT_INPUT")"
PROJECT_ROOT="$(pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PRECISION="${PRECISION:-float64}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
ALLOW_PUBLICATION_SCALE="${ALLOW_PUBLICATION_SCALE:-0}"
EXP3_SIF="${EXP3_SIF:-}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"

PYTHON_RUNNER=("$PYTHON_BIN")
SIF_SHA256=""
if [[ -n "$EXP3_SIF" ]]; then
  if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is required when EXP3_SIF is set" >&2
    exit 2
  fi
  EXP3_SIF="$(realpath "$EXP3_SIF")"
  if [[ ! -f "$EXP3_SIF" ]]; then
    echo "Apptainer image not found: $EXP3_SIF" >&2
    exit 2
  fi
  SIF_SHA256="$(sha256sum "$EXP3_SIF" | cut -d' ' -f1)"
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
  PYTHON_RUNNER=(
    apptainer exec --cleanenv "${BIND_ARGS[@]}" --pwd "$PROJECT_ROOT"
    --env "PYTHONPATH=$PROJECT_ROOT/src:$PROJECT_ROOT"
    "$EXP3_SIF" python
  )
fi
RUNNER=("${PYTHON_RUNNER[@]}" scripts/run_exp3.py)

DRY_RUN_JSON="$("${RUNNER[@]}" \
  --config "$CONFIG" --device cpu --precision "$PRECISION" \
  --output-dir "$OUTPUT_ROOT" --pilot-output-dir "$PILOT_OUTPUT_ROOT" \
  --dry-run --task-index 0)"
if printf '%s\n' "$DRY_RUN_JSON" | grep -q '"publication_scale": true'; then
  if [[ "$ALLOW_PUBLICATION_SCALE" != "1" ]]; then
    echo "Publication-scale submission blocked; set ALLOW_PUBLICATION_SCALE=1 explicitly" >&2
    exit 2
  fi
  "${PYTHON_RUNNER[@]}" -c \
    'import sys; from levy_experiments.pilot_gate import verify_pilot_gate; receipt = verify_pilot_gate(experiment=int(sys.argv[1]), final_config=sys.argv[2], output_root=sys.argv[3], precision=sys.argv[4], pilot_output_root=sys.argv[5]); print(f"pilot_gate=passed run_dir={receipt.run_dir} code_hash={receipt.code_hash}")' \
    3 "$CONFIG" "$OUTPUT_ROOT" "$PRECISION" "$PILOT_OUTPUT_ROOT"
fi

TASK_COUNT="$("${RUNNER[@]}" \
  --config "$CONFIG" --device cpu --precision "$PRECISION" \
  --output-dir "$OUTPUT_ROOT" --pilot-output-dir "$PILOT_OUTPUT_ROOT" \
  --dry-run --print-task-count)"
if [[ "$TASK_COUNT" -le 0 ]]; then
  echo "No tasks were generated" >&2
  exit 2
fi

EXPORTS="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$CONFIG,OUTPUT_ROOT=$OUTPUT_ROOT"
EXPORTS+=",PILOT_OUTPUT_ROOT=$PILOT_OUTPUT_ROOT"
EXPORTS+=",PYTHON_BIN=$PYTHON_BIN,PRECISION=$PRECISION"
EXPORTS+=",ALLOW_PUBLICATION_SCALE=$ALLOW_PUBLICATION_SCALE,EXP3_SIF=$EXP3_SIF"
EXPORTS+=",EXP3_SIF_SHA256=$SIF_SHA256"

TABLE_JOB="$(sbatch --parsable --export="$EXPORTS" slurm/exp3_table.sbatch)"
ARRAY_JOB="$(sbatch --parsable --dependency="afterok:$TABLE_JOB" \
  --array="0-$((TASK_COUNT - 1))%$MAX_CONCURRENT" \
  --export="$EXPORTS" slurm/exp3_array.sbatch)"
AGGREGATE_JOB="$(sbatch --parsable --dependency="afterok:$ARRAY_JOB" \
  --export="$EXPORTS" slurm/exp3_aggregate.sbatch)"

printf 'table_job=%s\narray_job=%s\naggregate_job=%s\ntask_count=%s\npilot_output_root=%s\nsif_sha256=%s\n' \
  "$TABLE_JOB" "$ARRAY_JOB" "$AGGREGATE_JOB" "$TASK_COUNT" \
  "$PILOT_OUTPUT_ROOT" "$SIF_SHA256"
