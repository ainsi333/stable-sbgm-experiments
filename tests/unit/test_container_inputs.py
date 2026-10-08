"""Static checks supplement, but do not replace, a Linux container build."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIRECTORIES = (
    "src", "experiments", "scripts", "configs", "tests", "slurm", "reports", "containers",
)


@pytest.mark.parametrize("filename", ["Dockerfile", "Dockerfile.exp1", "Dockerfile.exp2"])
def test_docker_includes_test_and_runtime_inputs(filename):
    text = (ROOT / "containers" / filename).read_text("utf-8")
    copied_files = {
        part
        for line in text.splitlines()
        if line.startswith("COPY ") and line.endswith(" ./")
        for part in line.split()[1:-1]
    }
    assert {"pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY.md"} <= copied_files
    for directory in DIRECTORIES:
        assert f"COPY {directory} ./{directory}" in text
    assert "PYTHONPATH=/opt/" in text
    assert "pip check" in text and 'pytest -q -m "not gpu"' in text


@pytest.mark.parametrize("experiment", [1, 2, 3])
def test_apptainer_includes_test_and_runtime_inputs(experiment):
    text = (ROOT / "containers" / f"exp{experiment}.def").read_text("utf-8")
    files = text.split("%files")[1].split("%post")[0]
    included = {line.split()[0] for line in files.splitlines() if line.strip()}
    assert set(DIRECTORIES) <= included
    assert {"pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY.md"} <= included
    assert "pip check" in text and 'pytest -q -m "not gpu"' in text


def test_exp3_all_stages_use_the_same_bound_source():
    for stage in ("array", "aggregate", "table"):
        text = (ROOT / "slurm" / f"exp3_{stage}.sbatch").read_text("utf-8")
        assert '--env "PYTHONPATH=$PROJECT_ROOT/src:$PROJECT_ROOT"' in text
        assert '"$EXP3_SIF" python scripts/' in text
    text = (ROOT / "slurm/exp3_array.sbatch").read_text("utf-8")
    assert '--env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"' in text
