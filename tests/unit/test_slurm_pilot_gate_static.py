from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLURM = ROOT / "slurm"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_all_sbatch_stages_forward_and_bind_pilot_root() -> None:
    stages = sorted(SLURM.glob("exp[123]_*.sbatch"))
    assert len(stages) == 9
    for path in stages:
        source = _read(path)
        assert '${PILOT_OUTPUT_ROOT:?PILOT_OUTPUT_ROOT is required}' in source
        assert 'EXTRA_ARGS=(--pilot-output-dir "$PILOT_OUTPUT_ROOT")' in source
        assert 'BIND_ARGS+=(--bind "$PILOT_OUTPUT_ROOT:$PILOT_OUTPUT_ROOT")' in source


def test_submitters_preflight_matching_pilot_before_any_sbatch() -> None:
    for experiment in (1, 2, 3):
        source = _read(SLURM / f"submit_exp{experiment}.sh")
        first_allocation = source.index('TABLE_JOB="$(sbatch')
        assert f"configs/pilot/exp{experiment}.toml" in source
        explicit_guard = 'if [[ "$ALLOW_PUBLICATION_SCALE" != "1" ]]'
        assert source.index(explicit_guard) < first_allocation
        assert source.index('"publication_scale": true') < first_allocation
        assert source.index("verify_pilot_gate") < first_allocation
        assert source.index("Pilot output root not found") < first_allocation
        assert 'EXPORTS+=",PILOT_OUTPUT_ROOT=$PILOT_OUTPUT_ROOT"' in source
        assert 'BIND_ARGS+=(--bind "$PILOT_OUTPUT_ROOT:$PILOT_OUTPUT_ROOT")' in source
        assert '--pilot-output-dir "$PILOT_OUTPUT_ROOT"' in source


def test_make_targets_expose_distinct_output_roots() -> None:
    source = _read(ROOT / "Makefile")
    assert "OUTPUT_ROOT ?= outputs" in source
    assert "PILOT_OUTPUT_ROOT ?= $(OUTPUT_ROOT)" in source
    for experiment in (1, 2, 3):
        target = source.split(f"final-exp{experiment}:", maxsplit=1)[1]
        assert "--output-dir $(OUTPUT_ROOT)" in target
        assert "--pilot-output-dir $(PILOT_OUTPUT_ROOT)" in target
