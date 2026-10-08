from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from experiments.exp1.config import enumerate_tasks, load_config
from experiments.exp1.storage import (
    compute_code_hash,
    load_task_artifact,
    prepare_run,
    task_directory,
    write_task_artifact,
)
from levy_experiments.errors import ArtifactError

ROOT = Path(__file__).resolve().parents[2]


def _metadata(config_hash: str, code_hash: str) -> dict[str, object]:
    return {
        "config_hash": config_hash,
        "code_hash": code_hash,
        "warnings": [],
        "execution_seconds_excluding_compilation": 0.1,
    }


def test_atomic_artifact_resume_and_corruption_detection(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "smoke" / "exp1.toml", seed_override=0)
    layout = prepare_run(tmp_path, config)
    task = enumerate_tasks(config)[0]
    code_hash = compute_code_hash()
    arrays = {"terminal_sample": np.linspace(-3.0, 3.0, 32), "model": np.asarray("stable")}
    destination = write_task_artifact(
        layout, task, arrays, _metadata(config.resolved_hash, code_hash)
    )
    loaded, metadata = load_task_artifact(
        layout,
        task,
        expected_config_hash=config.resolved_hash,
        expected_code_hash=code_hash,
    )
    assert np.array_equal(loaded["terminal_sample"], arrays["terminal_sample"])
    assert metadata["task"]["task_id"] == task.task_id
    assert (
        write_task_artifact(
            layout,
            task,
            arrays,
            _metadata(config.resolved_hash, code_hash),
            resume=True,
        )
        == destination
    )
    with pytest.raises(ArtifactError, match="Resume content differs"):
        write_task_artifact(
            layout,
            task,
            {**arrays, "terminal_sample": arrays["terminal_sample"] + 1.0},
            _metadata(config.resolved_hash, code_hash),
            resume=True,
        )
    npz = task_directory(layout, task) / "arrays.npz"
    npz.write_bytes(npz.read_bytes() + b"corrupt")
    with pytest.raises(ArtifactError, match="NPZ checksum mismatch"):
        load_task_artifact(
            layout,
            task,
            expected_config_hash=config.resolved_hash,
            expected_code_hash=code_hash,
        )


def test_nonfinite_and_incomplete_artifacts_are_rejected(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "smoke" / "exp1.toml", seed_override=0)
    layout = prepare_run(tmp_path, config)
    task = enumerate_tasks(config)[0]
    with pytest.raises(ArtifactError, match="non-finite"):
        write_task_artifact(
            layout,
            task,
            {"terminal_sample": np.asarray([0.0, np.inf])},
            _metadata(config.resolved_hash, compute_code_hash()),
        )
    incomplete = task_directory(layout, task)
    incomplete.mkdir()
    (incomplete / "sentinel.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ArtifactError, match="absent or incomplete"):
        write_task_artifact(
            layout,
            task,
            {"terminal_sample": np.arange(8.0)},
            _metadata(config.resolved_hash, compute_code_hash()),
            resume=True,
        )
    assert (incomplete / "sentinel.txt").read_text(encoding="utf-8") == "keep"


def test_code_hash_covers_imported_characteristic_metric(tmp_path: Path) -> None:
    """A scientific-metric mutation must invalidate every Exp1 artifact."""

    sandbox = tmp_path / "project"
    shutil.copytree(ROOT / "experiments" / "exp1", sandbox / "experiments" / "exp1")
    shutil.copytree(ROOT / "src" / "levy_experiments", sandbox / "src" / "levy_experiments")
    shutil.copy2(ROOT / "pyproject.toml", sandbox / "pyproject.toml")

    before = compute_code_hash(sandbox)
    metric = sandbox / "src" / "levy_experiments" / "metrics" / "characteristic.py"
    metric.write_text(metric.read_text(encoding="utf-8") + "\n# injected scientific mutation\n")
    after = compute_code_hash(sandbox)

    assert after != before
