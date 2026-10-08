"""Historical/current provenance must not bypass immutable table validation."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.exp2.score_tables import _save_score_table, _validated_table
from experiments.exp4 import source
from experiments.exp4.config import load_config
from levy_experiments.errors import ArtifactError

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    config = load_config(ROOT / "configs/final/exp4.toml", source_run_override=tmp_path)
    monkeypatch.setattr(source, "compute_code_hash", lambda: "c" * 64)
    fields = {}
    for name in ("stable_main", "stable_hires", "vp_main", "vp_hires"):
        model = name.split("_")[0]
        table = _validated_table(
            model, np.array([0.05, 2.0]), np.array([0.0, 1.0, 2.0]),
            np.array([[0.0, -0.1, -0.2], [0.0, -0.2, -0.3]]),
            {"alpha": 1.5, "beta": 1.0, "target": "centered_student_t_df4"},
        )
        filename = f"{name}.npz"
        _save_score_table(table, tmp_path / "score_tables" / filename)
        fields[f"{name}_file"] = filename
        fields[f"{name}_hash"] = table.table_hash
    config = replace(config, source=replace(config.source, **fields))
    (tmp_path / "aggregates").mkdir()
    for filename, payload in {
        "provenance.json": {"experiment": 2, "config_hash": config.source.config_hash,
                            "code_hash": config.source.code_hash},
        "aggregates/aggregate_manifest.json": {
            "config_hash": config.source.config_hash, "code_hash": config.source.code_hash,
            "status": "complete", "publication_gate_pass": True, "publication_gate_failures": [],
        },
        "resolved_config.json": {
            "resolved_hash": config.source.config_hash, "experiment": {"beta": 1.0},
            "stable": {"alpha": 1.5}, "target": {"degrees_of_freedom": 4.0},
        },
    }.items():
        (tmp_path / filename).write_text(json.dumps(payload), encoding="utf-8")
    return config


def amend(config, filename, **changes):
    path = Path(config.source.run_dir) / filename
    payload = json.loads(path.read_text("utf-8"))
    payload.update(changes)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("current", [False, True])
def test_frozen_and_current_execution_are_distinguished(inventory, current):
    if current:
        for filename in ("provenance.json", "aggregates/aggregate_manifest.json"):
            amend(inventory, filename, code_hash="c" * 64)
    result = source.validate_source(inventory)
    assert result.available and len(result.tables) == 4
    assert result.provenance["source_lineage"] == (
        "current_distribution_execution" if current else "frozen_execution"
    )


@pytest.mark.parametrize(
    ("filename", "changes", "message"),
    [
        ("provenance.json", {"code_hash": "d" * 64}, "neither frozen nor current"),
        ("aggregates/aggregate_manifest.json", {"code_hash": "c" * 64}, "hashes disagree"),
        ("resolved_config.json", {"resolved_hash": "d" * 64}, "hashes disagree"),
        ("aggregates/aggregate_manifest.json", {"publication_gate_pass": False}, "gate"),
        ("resolved_config.json", {"target": {"degrees_of_freedom": 3.0}}, "parameters"),
    ],
)
def test_incompatible_source_is_rejected(inventory, filename, changes, message):
    amend(inventory, filename, **changes)
    with pytest.raises(ArtifactError, match=message):
        source.validate_source(inventory)


def test_valid_but_different_table_is_rejected(inventory):
    config = replace(inventory, source=replace(inventory.source, stable_main_hash="d" * 64))
    with pytest.raises(ArtifactError, match="table mismatch"):
        source.validate_source(config)


def test_corrupt_table_is_rejected(inventory):
    path = Path(inventory.source.run_dir) / "score_tables" / inventory.source.stable_main_file
    path.write_bytes(b"corrupt")
    with pytest.raises(ArtifactError, match="corrupt score table"):
        source.validate_source(inventory)
