from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from levy_experiments.config import enumerate_tasks, load_config
from levy_experiments.errors import ArtifactError
from levy_experiments.storage import (
    compute_code_hash,
    load_task_artifact,
    prepare_run,
    task_directory,
    write_task_artifact,
)

ROOT = Path(__file__).resolve().parents[2]


def _metadata(config):
    return {
        "config_hash": config.resolved_hash,
        "code_hash": compute_code_hash(),
        "compile_seconds": 0.0,
        "execution_seconds_excluding_compilation": 0.0,
        "throughput_particle_steps_per_second": 0.0,
        "warnings": [],
    }


def test_atomic_storage_resume_and_corruption_detection(tmp_path: Path) -> None:
    config = load_config(ROOT / "tests/data/exp3_tiny.toml")
    layout = prepare_run(tmp_path, config)
    task = enumerate_tasks(config)[0]
    arrays = {"values": np.arange(8, dtype=np.float64)}
    write_task_artifact(layout, task, arrays, _metadata(config))
    loaded, _ = load_task_artifact(
        layout,
        task,
        expected_config_hash=config.resolved_hash,
        expected_code_hash=compute_code_hash(),
    )
    assert np.array_equal(loaded["values"], arrays["values"])
    with pytest.raises(ArtifactError, match="overwrite"):
        write_task_artifact(layout, task, arrays, _metadata(config))
    arrays_path = task_directory(layout, task) / "arrays.npz"
    arrays_path.write_bytes(arrays_path.read_bytes() + b"corruption")
    with pytest.raises(ArtifactError, match="checksum"):
        load_task_artifact(
            layout,
            task,
            expected_config_hash=config.resolved_hash,
            expected_code_hash=compute_code_hash(),
        )


def test_incompatible_configuration_is_rejected(tmp_path: Path) -> None:
    config = load_config(ROOT / "tests/data/exp3_tiny.toml")
    prepare_run(tmp_path, config)
    resolved = tmp_path / f"exp3-{config.resolved_hash[:12]}" / "resolved_config.json"
    resolved.write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactError, match="incompatible"):
        prepare_run(tmp_path, config)


def test_nonfinite_array_is_rejected(tmp_path: Path) -> None:
    config = load_config(ROOT / "tests/data/exp3_tiny.toml")
    layout = prepare_run(tmp_path, config)
    task = enumerate_tasks(config)[0]
    with pytest.raises(ArtifactError, match="non-finite"):
        write_task_artifact(
            layout,
            task,
            {"values": np.asarray([0.0, np.inf])},
            _metadata(config),
        )
