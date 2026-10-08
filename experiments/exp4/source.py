"""Validation and reuse of immutable Experiment 2 score tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.exp2.score_tables import ScoreTable, load_score_table
from experiments.exp2.storage import compute_code_hash, sha256_file
from levy_experiments.errors import ArtifactError

from .config import Experiment4Config


@dataclass(frozen=True)
class SourceTables:
    """Hash-checked Experiment 2 source inventory."""

    available: bool
    run_dir: Path | None
    tables: dict[str, ScoreTable]
    provenance: dict[str, Any]


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"corrupt {description}: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"invalid {description}: {path}")
    return value


def validate_source(config: Experiment4Config) -> SourceTables:
    """Load the final Experiment 2 tables, or return an explicit smoke absence."""

    if not config.source.required:
        return SourceTables(
            available=False,
            run_dir=None,
            tables={},
            provenance={"required": False, "status": "not_requested"},
        )
    run_dir = Path(config.source.run_dir).resolve()
    if not run_dir.is_dir():
        raise ArtifactError(f"Experiment 2 source run not found: {run_dir}")
    provenance_path = run_dir / "provenance.json"
    manifest_path = run_dir / "aggregates" / "aggregate_manifest.json"
    resolved_path = run_dir / "resolved_config.json"
    provenance = _read_json(provenance_path, "Experiment 2 provenance")
    manifest = _read_json(manifest_path, "Experiment 2 aggregate manifest")
    resolved = _read_json(resolved_path, "Experiment 2 resolved configuration")
    expected_config_hash = config.source.config_hash
    expected_code_hash = config.source.code_hash
    current_code_hash = compute_code_hash()
    source_code_hash = provenance.get("code_hash")
    # Accept the frozen execution or a fresh execution of this distribution.
    # In both cases the configuration and all four table contents remain frozen.
    if source_code_hash not in {expected_code_hash, current_code_hash}:
        raise ArtifactError("Experiment 2 source code is neither frozen nor current")
    if (
        provenance.get("experiment") != 2
        or provenance.get("config_hash") != expected_config_hash
        or manifest.get("config_hash") != expected_config_hash
        or manifest.get("code_hash") != source_code_hash
        or resolved.get("resolved_hash") != expected_config_hash
    ):
        raise ArtifactError("Experiment 2 source hashes disagree with the frozen inventory")
    if (
        manifest.get("status") != "complete"
        or manifest.get("publication_gate_pass") is not True
        or manifest.get("publication_gate_failures") != []
    ):
        raise ArtifactError("Experiment 2 source did not pass its publication gate")
    experiment = resolved.get("experiment")
    stable = resolved.get("stable")
    target = resolved.get("target")
    if (
        not isinstance(experiment, dict)
        or not isinstance(stable, dict)
        or not isinstance(target, dict)
        or float(experiment.get("beta", float("nan"))) != config.beta
        or float(stable.get("alpha", float("nan"))) != config.alpha
        or float(target.get("degrees_of_freedom", float("nan"))) != config.target_df
    ):
        raise ArtifactError("Experiment 2 source parameters do not match Experiment 4")

    inventory = {
        "stable_main": (config.source.stable_main_file, config.source.stable_main_hash, "stable"),
        "stable_hires": (
            config.source.stable_hires_file,
            config.source.stable_hires_hash,
            "stable",
        ),
        "vp_main": (config.source.vp_main_file, config.source.vp_main_hash, "vp"),
        "vp_hires": (config.source.vp_hires_file, config.source.vp_hires_hash, "vp"),
    }
    tables: dict[str, ScoreTable] = {}
    table_records: dict[str, Any] = {}
    for name, (filename, expected_hash, expected_model) in inventory.items():
        path = run_dir / "score_tables" / filename
        table = load_score_table(path)
        if table.table_hash != expected_hash or table.model != expected_model:
            raise ArtifactError(f"source score table mismatch: {path}")
        if (
            float(table.metadata.get("alpha", float("nan"))) != config.alpha
            or float(table.metadata.get("beta", float("nan"))) != config.beta
            or table.metadata.get("target") != "centered_student_t_df4"
        ):
            raise ArtifactError(f"source score table metadata mismatch: {path}")
        tables[name] = table
        table_records[name] = {
            "file": filename,
            "table_hash": table.table_hash,
            "file_sha256": sha256_file(path),
            "shape": [int(table.time_nodes.size), int(table.space_nodes.size)],
            "time_range": [float(table.time_nodes[0]), float(table.time_nodes[-1])],
            "space_max": float(table.space_nodes[-1]),
        }
    return SourceTables(
        available=True,
        run_dir=run_dir,
        tables=tables,
        provenance={
            "required": True,
            "status": "validated",
            "run_dir": str(run_dir),
            "config_hash": expected_config_hash,
            "code_hash": source_code_hash,
            "frozen_code_hash": expected_code_hash,
            "validation_code_hash": current_code_hash,
            "source_lineage": (
                "frozen_execution" if source_code_hash == expected_code_hash
                else "current_distribution_execution"
            ),
            "aggregate_manifest_sha256": sha256_file(manifest_path),
            "provenance_sha256": sha256_file(provenance_path),
            "tables": table_records,
        },
    )


__all__ = ["SourceTables", "validate_source"]
