from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.exp2.config import enumerate_tasks, load_config
from experiments.exp2.storage import (
    EXP2_SHARED_SOURCE_FILES,
    compute_code_hash,
    load_task_artifact,
    prepare_run,
    task_directory,
    write_task_artifact,
)
from levy_experiments.errors import ArtifactError, ConfigurationError
from levy_experiments.storage import compute_code_hash as compute_exp3_code_hash

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = PROJECT_ROOT / "configs" / "smoke" / "exp2.toml"
FINAL_CONFIG = PROJECT_ROOT / "configs" / "final" / "exp2.toml"


def _metadata(config_hash: str, code_hash: str) -> dict[str, object]:
    return {
        "config_hash": config_hash,
        "code_hash": code_hash,
        "compile_seconds": 0.1,
        "execution_seconds_excluding_compilation": 0.2,
        "warnings": [],
    }


def test_prepare_write_load_and_exact_resume(tmp_path: Path) -> None:
    config = load_config(SMOKE_CONFIG, seed_override=0)
    layout = prepare_run(tmp_path / "outputs", config)
    assert prepare_run(tmp_path / "outputs", config) == layout
    task = enumerate_tasks(config)[0]
    code_hash = compute_code_hash()
    arrays = {
        "samples": np.linspace(-2.0, 2.0, 17, dtype=np.float64),
        "model": np.asarray("stable"),
    }
    metadata = _metadata(config.resolved_hash, code_hash)

    destination = write_task_artifact(layout, task, arrays, metadata)
    loaded, loaded_metadata = load_task_artifact(
        layout,
        task,
        expected_config_hash=config.resolved_hash,
        expected_code_hash=code_hash,
    )

    assert destination == task_directory(layout, task)
    assert np.array_equal(loaded["samples"], arrays["samples"])
    assert loaded["model"].item() == "stable"
    assert loaded_metadata["task"]["task_id"] == task.task_id
    assert write_task_artifact(layout, task, arrays, metadata, resume=True) == destination
    with pytest.raises(ArtifactError, match="Refusing to overwrite"):
        write_task_artifact(layout, task, arrays, metadata)
    with pytest.raises(ArtifactError, match="Resume content differs"):
        write_task_artifact(
            layout,
            task,
            {**arrays, "samples": arrays["samples"] + 1.0},
            metadata,
            resume=True,
        )


def test_corruption_and_hash_incompatibility_are_detected(tmp_path: Path) -> None:
    config = load_config(SMOKE_CONFIG, seed_override=0)
    layout = prepare_run(tmp_path / "outputs", config)
    task = enumerate_tasks(config)[0]
    code_hash = compute_code_hash()
    arrays = {"samples": np.arange(8, dtype=np.float64)}
    with pytest.raises(ArtifactError, match="Configuration mismatch"):
        write_task_artifact(layout, task, arrays, _metadata("wrong-config", code_hash))
    with pytest.raises(ArtifactError, match="Code-version mismatch"):
        write_task_artifact(
            layout, task, arrays, _metadata(config.resolved_hash, "wrong-code")
        )
    write_task_artifact(
        layout, task, arrays, _metadata(config.resolved_hash, code_hash)
    )

    with pytest.raises(ArtifactError, match="Configuration mismatch"):
        load_task_artifact(
            layout,
            task,
            expected_config_hash="wrong-config",
            expected_code_hash=code_hash,
        )
    with pytest.raises(ArtifactError, match="Code-version mismatch"):
        load_task_artifact(
            layout,
            task,
            expected_config_hash=config.resolved_hash,
            expected_code_hash="wrong-code",
        )

    arrays_path = task_directory(layout, task) / "arrays.npz"
    arrays_path.write_bytes(arrays_path.read_bytes() + b"corruption")
    with pytest.raises(ArtifactError, match="NPZ checksum mismatch"):
        load_task_artifact(
            layout,
            task,
            expected_config_hash=config.resolved_hash,
            expected_code_hash=code_hash,
        )


def test_incomplete_task_is_never_silently_replaced_on_resume(tmp_path: Path) -> None:
    config = load_config(SMOKE_CONFIG, seed_override=0)
    layout = prepare_run(tmp_path / "outputs", config)
    task = enumerate_tasks(config)[0]
    incomplete = task_directory(layout, task)
    incomplete.mkdir()
    sentinel = incomplete / "partial.txt"
    sentinel.write_text("partial", encoding="utf-8")

    with pytest.raises(ArtifactError, match="absent or incomplete"):
        write_task_artifact(
            layout,
            task,
            {"samples": np.arange(4, dtype=np.float64)},
            _metadata(config.resolved_hash, compute_code_hash()),
            resume=True,
        )
    assert sentinel.read_text(encoding="utf-8") == "partial"


def test_prepare_rejects_changed_source_and_incompatible_run_file(tmp_path: Path) -> None:
    copied = tmp_path / "exp2.toml"
    copied.write_bytes(SMOKE_CONFIG.read_bytes())
    config = load_config(copied, seed_override=0)
    copied.write_text(copied.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="source changed"):
        prepare_run(tmp_path / "changed-source", config)
    assert not (tmp_path / "changed-source").exists()

    clean = load_config(SMOKE_CONFIG, seed_override=0)
    layout = prepare_run(tmp_path / "outputs", clean)
    (layout.run_dir / "resolved_config.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="incompatible"):
        prepare_run(tmp_path / "outputs", clean)


def test_final_prepare_guard_runs_before_creating_outputs(tmp_path: Path) -> None:
    config = load_config(FINAL_CONFIG)
    output = tmp_path / "outputs"

    with pytest.raises(ConfigurationError, match="explicitly pass allow_final=True"):
        prepare_run(output, config)
    assert not output.exists()


def test_exp2_local_source_changes_only_exp2_code_hash(tmp_path: Path) -> None:
    for relative in EXP2_SHARED_SOURCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    local = tmp_path / "experiments" / "exp2" / "local.py"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("VALUE = 1\n", encoding="utf-8")

    exp2_before = compute_code_hash(tmp_path)
    exp3_before = compute_exp3_code_hash(tmp_path)
    local.write_text("VALUE = 2\n", encoding="utf-8")

    assert compute_code_hash(tmp_path) != exp2_before
    assert compute_exp3_code_hash(tmp_path) == exp3_before
