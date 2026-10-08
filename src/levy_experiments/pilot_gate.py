"""Verify that a fresh, intact pilot authorizes publication-scale execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ArtifactError


@dataclass(frozen=True)
class PilotGateReceipt:
    """Identity of one complete pilot whose scientific gate passed."""

    experiment: int
    run_dir: Path
    config_hash: str
    code_hash: str
    task_count: int


@dataclass(frozen=True)
class _ConfigIdentity:
    config: Any
    code_hash: str
    tasks: tuple[Any, ...]
    compatibility_payload: dict[str, Any]


def _load_identity(
    experiment: int,
    path: Path,
    precision: str | None,
    *,
    authorize_final: bool,
) -> _ConfigIdentity:
    if experiment == 1:
        from experiments.exp1.config import (
            enumerate_tasks,
            load_config,
            pilot_compatibility_payload,
        )
        from experiments.exp1.storage import compute_code_hash

        config = load_config(
            path,
            precision_override=precision,
            allow_final=authorize_final,
        )
    elif experiment == 2:
        from experiments.exp2.config import (
            enumerate_tasks,
            load_config,
            pilot_compatibility_payload,
        )
        from experiments.exp2.storage import compute_code_hash

        config = load_config(
            path,
            precision_override=precision,
            allow_final=authorize_final,
        )
    elif experiment == 3:
        from .config import enumerate_tasks, load_config, pilot_compatibility_payload
        from .storage import compute_code_hash

        config = load_config(path, precision_override=precision)
    else:
        raise ArtifactError(f"unsupported experiment for pilot gate: {experiment}")
    return _ConfigIdentity(
        config=config,
        code_hash=compute_code_hash(),
        tasks=tuple(enumerate_tasks(config)),
        compatibility_payload=pilot_compatibility_payload(config),
    )


def _require_at_least(actual: int | float, pilot: int | float, label: str) -> None:
    if actual < pilot:
        raise ArtifactError(
            f"final budget regression for {label}: pilot={pilot}, final={actual}"
        )


def _verify_budget_monotonicity(
    experiment: int,
    pilot: _ConfigIdentity,
    final: _ConfigIdentity,
) -> None:
    p = pilot.config
    f = final.config
    if f.experiment.publication_scale is not True:
        raise ArtifactError("the requested final configuration is not publication-scale")
    if p.experiment.publication_scale is not False:
        raise ArtifactError("the matching pilot configuration must not be publication-scale")
    if p.experiment.precision != "float64" or f.experiment.precision != "float64":
        raise ArtifactError("pilot and final publication protocol require float64")
    if pilot.compatibility_payload != final.compatibility_payload:
        raise ArtifactError("pilot/final scientific compatibility payload mismatch")

    if experiment == 1:
        for field in (
            "main_particles",
            "control_particles",
            "refinement_particles",
            "refinement_control_particles",
        ):
            _require_at_least(
                getattr(f.experiment, field),
                getattr(p.experiment, field),
                f"experiment.{field}",
            )
        _require_at_least(
            len(f.experiment.primary_seeds),
            len(p.experiment.primary_seeds),
            "experiment.primary_seeds",
        )
        _require_at_least(
            len(f.experiment.refinement_seeds),
            len(p.experiment.refinement_seeds),
            "experiment.refinement_seeds",
        )
        if f.design.primary_steps not in p.design.primary_seed_steps:
            raise ArtifactError("final primary N was not exercised by the pilot")
        for field in ("time_points", "space_points", "stable_fft_points", "vp_fft_points"):
            _require_at_least(
                getattr(f.score_table, field),
                getattr(p.score_table, field),
                f"score_table.{field}",
            )
        for field in (
            "bootstrap_replicates",
            "minimum_exceedances_per_seed",
            "minimum_pooled_exceedances",
            "minimum_contributing_seeds",
        ):
            _require_at_least(
                getattr(f.analysis, field),
                getattr(p.analysis, field),
                f"analysis.{field}",
            )
    elif experiment == 2:
        _require_at_least(
            len(f.experiment.seeds), len(p.experiment.seeds), "experiment.seeds"
        )
        _require_at_least(
            f.experiment.particles, p.experiment.particles, "experiment.particles"
        )
        if not set(f.design.refinement_steps).issubset(set(p.design.refinement_steps)):
            raise ArtifactError("final refinement levels were not all exercised by the pilot")
        for table_name in ("score_table", "score_table_hires"):
            p_table = getattr(p, table_name)
            f_table = getattr(f, table_name)
            for field in (
                "time_points",
                "space_points",
                "stable_fft_points",
                "stable_tail_l",
                "stable_x_max",
                "stable_blend_start",
                "stable_blend_end",
                "vp_fft_points",
                "vp_tail_l",
                "vp_x_max",
                "vp_blend_start",
                "vp_blend_end",
            ):
                _require_at_least(
                    getattr(f_table, field),
                    getattr(p_table, field),
                    f"{table_name}.{field}",
                )
    else:
        _require_at_least(
            len(f.experiment.seeds), len(p.experiment.seeds), "experiment.seeds"
        )
        for section, field in (
            ("stationary", "particles"),
            ("nonstationary", "particles"),
            ("score_table", "points"),
            ("score_table", "validation_points"),
        ):
            _require_at_least(
                getattr(getattr(f, section), field),
                getattr(getattr(p, section), field),
                f"{section}.{field}",
            )
        for field in ("validation_rtol", "validation_atol"):
            if getattr(f.score_table, field) > getattr(p.score_table, field):
                raise ArtifactError(
                    f"final score-table tolerance is weaker for score_table.{field}"
                )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{label} is absent or corrupt: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"{label} must contain one JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactError(f"pilot artifact is absent or unreadable: {path}") from exc
    return digest.hexdigest()


def _inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactError(f"{label} escapes the pilot run directory: {resolved}") from exc
    return resolved


def _verify_hash(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ArtifactError(f"invalid SHA-256 entry for {label}")
    if not path.is_file() or _sha256_file(path) != expected:
        raise ArtifactError(f"pilot artifact checksum mismatch: {label}")


def _verify_exp1_manifest(
    run_dir: Path,
    manifest: dict[str, Any],
    identity: _ConfigIdentity,
) -> None:
    if manifest.get("tier") != "pilot":
        raise ArtifactError("Experiment 1 pilot manifest has the wrong tier")
    expected_ids = {task.task_id for task in identity.tasks}
    expected_npz = {f"raw/{task_id}/arrays.npz" for task_id in expected_ids}
    expected_metadata = {f"raw/{task_id}/metadata.json" for task_id in expected_ids}
    for field, expected_paths in (
        ("task_npz_hashes", expected_npz),
        ("task_metadata_hashes", expected_metadata),
    ):
        raw_inventory = manifest.get(field)
        if not isinstance(raw_inventory, dict):
            raise ArtifactError(f"Experiment 1 {field} inventory mismatch")
        inventory = {str(path).replace("\\", "/"): value for path, value in raw_inventory.items()}
        if set(inventory) != expected_paths:
            raise ArtifactError(f"Experiment 1 {field} inventory mismatch")
        for relative, expected_hash in inventory.items():
            path = _inside(run_dir / relative, run_dir, field)
            _verify_hash(path, expected_hash, relative)
    raw_files = manifest.get("files")
    if not isinstance(raw_files, dict):
        raise ArtifactError("Experiment 1 aggregate file inventory is incomplete")
    files = {str(path).replace("\\", "/"): value for path, value in raw_files.items()}
    if "aggregates/diagnostics.json" not in files:
        raise ArtifactError("Experiment 1 aggregate file inventory is incomplete")
    for relative, expected_hash in files.items():
        path = _inside(run_dir / str(relative), run_dir, "aggregate file")
        _verify_hash(path, expected_hash, str(relative))


def _verify_exp2_manifest(
    run_dir: Path,
    manifest: dict[str, Any],
    identity: _ConfigIdentity,
) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or "diagnostics.json" not in artifacts:
        raise ArtifactError("Experiment 2 aggregate artifact inventory is incomplete")
    for name, details in artifacts.items():
        if not isinstance(details, dict):
            raise ArtifactError(f"invalid Experiment 2 artifact entry: {name}")
        path = _inside(Path(str(details.get("path"))), run_dir, f"artifact {name}")
        _verify_hash(path, details.get("sha256"), f"artifact {name}")
        if details.get("size_bytes") != path.stat().st_size:
            raise ArtifactError(f"pilot artifact size mismatch: {name}")
    source = manifest.get("source")
    tasks = source.get("tasks") if isinstance(source, dict) else None
    if not isinstance(tasks, list) or len(tasks) != len(identity.tasks):
        raise ArtifactError("Experiment 2 raw task inventory mismatch")
    expected_ids = {task.task_id for task in identity.tasks}
    observed_ids: set[str] = set()
    for entry in tasks:
        if not isinstance(entry, dict):
            raise ArtifactError("invalid Experiment 2 raw task entry")
        task_id = str(entry.get("task_id"))
        observed_ids.add(task_id)
        for path_field, hash_field, filename in (
            ("arrays_path", "arrays_sha256", "arrays.npz"),
            ("metadata_path", "metadata_sha256", "metadata.json"),
        ):
            path = _inside(Path(str(entry.get(path_field))), run_dir, task_id)
            if path.name != filename:
                raise ArtifactError(f"Experiment 2 task path has wrong filename: {path}")
            _verify_hash(path, entry.get(hash_field), f"{task_id}/{filename}")
    if observed_ids != expected_ids:
        raise ArtifactError("Experiment 2 raw task identifiers mismatch")


def _verify_exp3_manifest(
    run_dir: Path,
    manifest: dict[str, Any],
    identity: _ConfigIdentity,
) -> None:
    from .config import pilot_compatibility_hash

    if (
        manifest.get("pilot_compatibility_payload") != identity.compatibility_payload
        or manifest.get("pilot_compatibility_hash")
        != pilot_compatibility_hash(identity.config)
    ):
        raise ArtifactError("Experiment 3 manifest compatibility signature mismatch")
    artifacts = manifest.get("aggregate_artifacts")
    expected_names = {
        "metrics_per_seed.csv",
        "metrics_summary.csv",
        "refinement_per_seed.csv",
        "diagnostics.json",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_names:
        raise ArtifactError("Experiment 3 aggregate artifact inventory mismatch")
    for name, expected_hash in artifacts.items():
        _verify_hash(run_dir / "aggregates" / name, expected_hash, name)
    entries = manifest.get("task_artifacts")
    if not isinstance(entries, list) or len(entries) != len(identity.tasks):
        raise ArtifactError("Experiment 3 raw task inventory mismatch")
    by_id = {
        str(entry.get("task_id")): entry for entry in entries if isinstance(entry, dict)
    }
    expected_ids = {task.task_id for task in identity.tasks}
    if set(by_id) != expected_ids:
        raise ArtifactError("Experiment 3 raw task identifiers mismatch")
    for task_id, entry in by_id.items():
        directory = run_dir / "raw" / task_id
        _verify_hash(
            directory / "arrays.npz", entry.get("arrays_npz_sha256"), f"{task_id}/arrays.npz"
        )
        _verify_hash(
            directory / "metadata.json",
            entry.get("metadata_json_sha256"),
            f"{task_id}/metadata.json",
        )


def _diagnostic_identity(payload: dict[str, Any], name: str) -> str | None:
    direct = payload.get(name)
    if isinstance(direct, str):
        return direct
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        nested = provenance.get(name)
        if isinstance(nested, str):
            return nested
    return None


def _verify_manifest_and_diagnostics(
    experiment: int,
    run_dir: Path,
    identity: _ConfigIdentity,
) -> dict[str, Any]:
    manifest_path = run_dir / "aggregates" / "aggregate_manifest.json"
    manifest = _read_json(manifest_path, "pilot aggregate manifest")
    source = manifest.get("source")
    manifest_task_count = (
        source.get("task_count")
        if experiment == 2 and isinstance(source, dict)
        else manifest.get("task_count")
    )
    if (
        manifest.get("status") != "complete"
        or manifest.get("config_hash") != identity.config.resolved_hash
        or manifest.get("code_hash") != identity.code_hash
        or manifest_task_count != len(identity.tasks)
    ):
        raise ArtifactError("pilot aggregate manifest identity mismatch")
    if experiment == 1:
        _verify_exp1_manifest(run_dir, manifest, identity)
    elif experiment == 2:
        _verify_exp2_manifest(run_dir, manifest, identity)
    else:
        _verify_exp3_manifest(run_dir, manifest, identity)

    diagnostics = _read_json(
        run_dir / "aggregates" / "diagnostics.json", "pilot diagnostics"
    )
    observed_config_hash = _diagnostic_identity(diagnostics, "config_hash")
    observed_code_hash = _diagnostic_identity(diagnostics, "code_hash")
    if observed_config_hash != identity.config.resolved_hash:
        raise ArtifactError("pilot diagnostics configuration hash mismatch")
    if observed_code_hash != identity.code_hash:
        raise ArtifactError("pilot diagnostics code hash mismatch")
    if diagnostics.get("task_count") != len(identity.tasks):
        raise ArtifactError("pilot diagnostics task-count mismatch")
    failures = diagnostics.get("publication_gate_failures")
    if not isinstance(failures, list) or any(not isinstance(item, str) for item in failures):
        raise ArtifactError("pilot diagnostics have no valid publication_gate_failures list")
    if experiment in {2, 3} and (
        manifest.get("publication_gate_pass") is not diagnostics.get("publication_gate_pass")
        or manifest.get("publication_gate_failures") != failures
    ):
        raise ArtifactError("publication gate differs between pilot manifest and diagnostics")
    if diagnostics.get("publication_gate_pass") is not True or failures:
        reason = "; ".join(failures) if failures else "unspecified pilot-gate failure"
        raise ArtifactError(f"pilot did not authorize a final run: {reason}")
    return diagnostics


def verify_pilot_gate(
    *,
    experiment: int,
    final_config: str | Path,
    output_root: str | Path,
    precision: str | None,
    pilot_output_root: str | Path | None = None,
) -> PilotGateReceipt:
    """Fail closed unless a matching, intact current-code pilot passed every gate."""

    final_path = Path(final_config).resolve()
    pilot_path = final_path.parent.parent / "pilot" / final_path.name
    if not pilot_path.is_file():
        raise ArtifactError(f"pilot configuration not found: {pilot_path}")
    pilot = _load_identity(
        experiment, pilot_path, precision, authorize_final=False
    )
    final = _load_identity(
        experiment, final_path, precision, authorize_final=True
    )
    if pilot.code_hash != final.code_hash:
        raise ArtifactError("pilot/final code-hash computation is inconsistent")
    _verify_budget_monotonicity(experiment, pilot, final)

    root = Path(pilot_output_root or output_root).resolve()
    run_dir = root / f"exp{experiment}-{pilot.config.resolved_hash[:12]}"
    if not run_dir.is_dir():
        raise ArtifactError(
            "publication-scale execution requires a completed matching pilot; "
            f"missing {run_dir}"
        )
    _verify_manifest_and_diagnostics(experiment, run_dir, pilot)
    return PilotGateReceipt(
        experiment=experiment,
        run_dir=run_dir,
        config_hash=pilot.config.resolved_hash,
        code_hash=pilot.code_hash,
        task_count=len(pilot.tasks),
    )


__all__ = ["PilotGateReceipt", "verify_pilot_gate"]
