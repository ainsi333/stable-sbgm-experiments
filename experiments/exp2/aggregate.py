"""Command-line aggregation for already-computed Experiment 2 task NPZ files.

The command performs no simulation.  Its complete input contract is documented
in :mod:`experiments.exp2.analysis` and may be inspected with ``--help``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from levy_experiments.authority import authority_manifest
from levy_experiments.errors import ArtifactError, ExperimentError
from levy_experiments.pilot_gate import verify_pilot_gate
from levy_experiments.storage import write_or_verify_bytes

from .analysis import (
    AnalysisSettings,
    Exp2ArtifactError,
    analyze_tasks,
    diagnostics,
    fit_horizon_initialization_per_seed,
    fit_refinement_per_seed,
    load_tasks,
    summarize_rows,
)
from .plotting import render_figures
from .storage import atomic_write_json, sha256_file

METRIC_FIELDS = (
    "source",
    "model",
    "horizon",
    "steps",
    "h",
    "seed",
    "forward_time",
    "metric",
    "component",
    "order",
    "sample_kind",
    "sample_size",
    "replicate",
    "value",
    "floor",
    "score_sensitivity",
    "controlled",
    "hoeffding_radius",
)
SUMMARY_FIELDS = (
    "model",
    "horizon",
    "steps",
    "h",
    "forward_time",
    "metric",
    "component",
    "order",
    "sample_kind",
    "sample_size",
    "n_seeds",
    "q16",
    "median",
    "q84",
    "floor_median",
    "controlled_fraction",
    "hoeffding_radius",
)
FIT_FIELDS = (
    "model",
    "horizon",
    "seed",
    "forward_time",
    "order",
    "n_points",
    "steps",
    "h_min",
    "h_max",
    "slope",
    "intercept",
    "r_squared",
)
HORIZON_FIT_FIELDS = (
    "model",
    "seed",
    "order",
    "component",
    "n_points",
    "n_candidate_horizons",
    "n_excluded_unresolved",
    "n_excluded_D_confounded",
    "horizons",
    "steps",
    "step_size_min",
    "step_size_max",
    "step_size_cv",
    "log_error_slope",
    "log_error_intercept",
    "r_squared",
    "status",
)
AGGREGATE_FILENAMES = (
    "metrics_per_seed.csv",
    "metrics_summary.csv",
    "refinement_fits_per_seed.csv",
    "horizon_fits_per_seed.csv",
    "analysis_arrays.npz",
    "diagnostics.json",
    "aggregate_manifest.json",
)
FIGURE_FILENAMES = (
    "experiment2_wp.pdf",
    "experiment2_wp.png",
    "experiment2_horizon.pdf",
    "experiment2_horizon.png",
    "experiment2_diagnostics.pdf",
    "experiment2_diagnostics.png",
)
OUTPUT_FILENAMES = (*AGGREGATE_FILENAMES, *FIGURE_FILENAMES)


def _enforce_final_publication_gate(
    *,
    tier: str | None,
    aggregate_dir: Path,
    report: dict[str, Any],
) -> None:
    """Persist a refusal and stop a failed final before any publishable output."""

    if tier != "final" or report.get("publication_gate_pass") is True:
        return
    raw_failures = report.get("publication_gate_failures")
    failures = (
        [str(value) for value in raw_failures]
        if isinstance(raw_failures, list) and raw_failures
        else ["publication_gate_missing_or_false_without_diagnostic_reason"]
    )
    failure_path = aggregate_dir / "publication_gate_failure.json"
    atomic_write_json(
        failure_path,
        {
            "status": "refused",
            "experiment": 2,
            "tier": "final",
            "publication_gate_pass": False,
            "publication_gate_failures": failures,
            "diagnostics": report,
        },
    )
    raise ArtifactError(
        "final Experiment 2 rendering refused because publication gates failed: "
        + "; ".join(failures)
    )


def _csv_cell(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return "" if not math.isfinite(numeric) else format(numeric, ".17g")
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if value is None:
        return ""
    return str(value)


def _csv_bytes(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        missing = set(fields) - set(row)
        if missing:
            raise Exp2ArtifactError(f"CSV row is missing fields: {sorted(missing)}")
        writer.writerow({name: _csv_cell(row[name]) for name in fields})
    return buffer.getvalue().encode("utf-8")


def _column_array(values: Sequence[Any]) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=np.float64)
    if all(isinstance(value, (bool, np.bool_)) for value in values):
        return np.asarray(values, dtype=np.bool_)
    if all(
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        for value in values
    ):
        return np.asarray(values, dtype=np.int64)
    if all(isinstance(value, (int, float, np.integer, np.floating)) for value in values):
        return np.asarray(values, dtype=np.float64)
    strings = ["" if value is None else str(value) for value in values]
    width = max(1, max(len(value) for value in strings))
    return np.asarray(strings, dtype=f"<U{width}")


def _table_arrays(
    prefix: str, rows: Sequence[dict[str, Any]], fields: Sequence[str]
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__{field}": _column_array([row[field] for row in rows])
        for field in fields
    }


def _deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Serialize NPZ without wall-clock timestamps or object arrays."""

    payload = io.BytesIO()
    with zipfile.ZipFile(
        payload,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(arrays):
            values = np.asarray(arrays[name])
            if values.dtype.hasobject:
                raise Exp2ArtifactError(f"object dtype forbidden in aggregate NPZ: {name}")
            member = io.BytesIO()
            np.lib.format.write_array(member, values, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                member.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return payload.getvalue()


def _discover_tasks(input_dir: Path, output_dir: Path, pattern: str) -> list[Path]:
    if not input_dir.is_dir():
        raise Exp2ArtifactError(f"input directory does not exist: {input_dir}")
    candidates = [
        path.resolve()
        for path in input_dir.glob(pattern)
        if path.is_file() and not path.resolve().is_relative_to(output_dir)
    ]
    if not candidates:
        raise Exp2ArtifactError(f"no task NPZ matches {pattern!r} under {input_dir}")
    return sorted(set(candidates), key=lambda path: path.as_posix())


def aggregate_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = "*.npz",
    settings: AnalysisSettings | None = None,
    resume: bool = False,
) -> dict[str, str]:
    """Low-level adapter for standalone flat NPZ directories (mainly tests)."""

    source = Path(input_dir).resolve()
    destination = Path(output_dir).resolve()
    task_paths = _discover_tasks(source, destination, pattern)
    return aggregate_task_paths(
        task_paths,
        destination,
        destination,
        settings=settings,
        resume=resume,
    )


def aggregate_task_paths(
    task_paths: Sequence[str | Path],
    aggregate_dir: str | Path,
    figure_dir: str | Path,
    *,
    settings: AnalysisSettings | None = None,
    resume: bool = False,
    diagnostic_context: dict[str, Any] | None = None,
    tier: str | None = None,
    checkpoint_fractions: Sequence[float] | None = None,
    raw_task_inventory: Sequence[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Aggregate an explicit, already provenance-verified collection of task NPZs."""

    aggregate_destination = Path(aggregate_dir).resolve()
    figure_destination = Path(figure_dir).resolve()
    if not resume:
        expected = [aggregate_destination / name for name in AGGREGATE_FILENAMES]
        expected.extend(figure_destination / name for name in FIGURE_FILENAMES)
        existing = [path for path in expected if path.exists()]
        if existing:
            raise ArtifactError(f"Refusing to overwrite existing artifact: {existing[0]}")
    settings = AnalysisSettings() if settings is None else settings
    tasks = load_tasks(task_paths, checkpoint_fractions=checkpoint_fractions)
    resolved_task_paths = {str(task.path.resolve()) for task in tasks}
    if raw_task_inventory is None:
        source_tasks = [
            {
                "task_id": task.path.stem,
                "arrays_path": str(task.path.resolve()),
                "arrays_sha256": sha256_file(task.path),
                "metadata_path": None,
                "metadata_sha256": None,
            }
            for task in tasks
        ]
    else:
        source_tasks = [dict(entry) for entry in raw_task_inventory]
        inventory_paths = {str(entry.get("arrays_path")) for entry in source_tasks}
        if inventory_paths != resolved_task_paths:
            raise ArtifactError("raw task inventory does not match analyzed NPZ paths")
        for entry in source_tasks:
            arrays_path = Path(str(entry["arrays_path"]))
            metadata_path = Path(str(entry["metadata_path"]))
            if sha256_file(arrays_path) != entry.get("arrays_sha256"):
                raise ArtifactError(f"raw arrays changed after verification: {arrays_path}")
            if sha256_file(metadata_path) != entry.get("metadata_sha256"):
                raise ArtifactError(f"raw metadata changed after verification: {metadata_path}")
    source_tasks.sort(key=lambda entry: str(entry["task_id"]))
    rows = analyze_tasks(tasks, settings)
    summaries = summarize_rows(rows)
    fits = fit_refinement_per_seed(rows)
    horizon_fits = fit_horizon_initialization_per_seed(rows)
    report = diagnostics(tasks, rows, fits, horizon_fits)
    if diagnostic_context:
        report["provenance"] = diagnostic_context
    _enforce_final_publication_gate(
        tier=tier,
        aggregate_dir=aggregate_destination,
        report=report,
    )

    aggregate_destination.mkdir(parents=True, exist_ok=True)
    payloads = {
        "metrics_per_seed.csv": _csv_bytes(rows, METRIC_FIELDS),
        "metrics_summary.csv": _csv_bytes(summaries, SUMMARY_FIELDS),
        "refinement_fits_per_seed.csv": _csv_bytes(fits, FIT_FIELDS),
        "horizon_fits_per_seed.csv": _csv_bytes(horizon_fits, HORIZON_FIT_FIELDS),
        "diagnostics.json": (
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
    }
    npz_arrays = {}
    npz_arrays.update(_table_arrays("per_seed", rows, METRIC_FIELDS))
    npz_arrays.update(_table_arrays("summary", summaries, SUMMARY_FIELDS))
    npz_arrays.update(_table_arrays("fit", fits, FIT_FIELDS))
    npz_arrays.update(
        _table_arrays("horizon_fit", horizon_fits, HORIZON_FIT_FIELDS)
    )
    payloads["analysis_arrays.npz"] = _deterministic_npz_bytes(npz_arrays)

    outputs: dict[str, str] = {}
    for name, payload in payloads.items():
        path = aggregate_destination / name
        write_or_verify_bytes(path, payload, resume=resume)
        outputs[name] = str(path)
    outputs.update(
        render_figures(summaries, figure_destination, resume=resume, tier=tier)
    )
    context = {} if diagnostic_context is None else diagnostic_context
    artifacts = {}
    for name, raw_path in sorted(outputs.items()):
        path = Path(raw_path)
        artifacts[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "status": "complete",
        "experiment": 2,
        "authority": authority_manifest(),
        "config_hash": context.get("config_hash"),
        "code_hash": context.get("code_hash"),
        "publication_gate_pass": report["publication_gate_pass"],
        "publication_gate_failures": report["publication_gate_failures"],
        "source": {
            "task_count": len(tasks),
            "raw_paths": sorted(str(task.path) for task in tasks),
            "tasks": source_tasks,
        },
        "artifacts": artifacts,
    }
    manifest_path = aggregate_destination / "aggregate_manifest.json"
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    write_or_verify_bytes(manifest_path, manifest_bytes, resume=resume)
    outputs[manifest_path.name] = str(manifest_path)
    return outputs


def aggregate_configured_run(
    config: Any, output_root: str | Path, *, resume: bool
) -> dict[str, str]:
    """Verify a complete immutable run, then aggregate it without simulation."""

    # Imported lazily so the standalone NPZ adapter remains usable independently
    # of the run-orchestration layer.
    from .config import enumerate_tasks
    from .storage import (
        compute_code_hash,
        load_task_artifact,
        prepare_run,
        task_directory,
    )

    layout = prepare_run(output_root, config)
    expected_tasks = enumerate_tasks(config)
    expected_ids = {task.task_id for task in expected_tasks}
    actual_ids = {
        path.name
        for path in layout.raw_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        raise ArtifactError(
            f"task collection mismatch: missing={missing[:8]}, extra={extra[:8]}"
        )
    code_hash = compute_code_hash()
    paths: list[Path] = []
    raw_task_inventory: list[dict[str, Any]] = []
    score_hashes: dict[str, set[str]] = {"stable": set(), "vp": set()}
    hires_score_hashes: dict[str, set[str]] = {"stable": set(), "vp": set()}
    for task in expected_tasks:
        _, metadata = load_task_artifact(
            layout,
            task,
            expected_config_hash=config.resolved_hash,
            expected_code_hash=code_hash,
        )
        score_hash = metadata.get("score_table_hash")
        if not isinstance(score_hash, str) or not score_hash:
            raise ArtifactError(f"task {task.task_id} has no score_table_hash")
        hires_score_hash = metadata.get("score_table_hires_hash")
        if not isinstance(hires_score_hash, str) or not hires_score_hash:
            raise ArtifactError(f"task {task.task_id} has no score_table_hires_hash")
        if hires_score_hash == score_hash:
            raise ArtifactError(
                f"task {task.task_id} reused its main table as the sensitivity table"
            )
        score_hashes[task.model].add(score_hash)
        hires_score_hashes[task.model].add(hires_score_hash)
        directory = task_directory(layout, task)
        arrays_path = directory / "arrays.npz"
        metadata_path = directory / "metadata.json"
        paths.append(arrays_path)
        raw_task_inventory.append(
            {
                "task_id": task.task_id,
                "arrays_path": str(arrays_path.resolve()),
                "arrays_sha256": sha256_file(arrays_path),
                "metadata_path": str(metadata_path.resolve()),
                "metadata_sha256": sha256_file(metadata_path),
            }
        )
    inconsistent = {
        model: sorted(hashes) for model, hashes in score_hashes.items() if len(hashes) != 1
    }
    if inconsistent:
        raise ArtifactError(f"inconsistent score tables within model: {inconsistent}")
    inconsistent_hires = {
        model: sorted(hashes)
        for model, hashes in hires_score_hashes.items()
        if len(hashes) != 1
    }
    if inconsistent_hires:
        raise ArtifactError(
            f"inconsistent high-resolution score tables within model: {inconsistent_hires}"
        )
    resolved_score_hashes = {
        model: next(iter(hashes)) for model, hashes in score_hashes.items()
    }
    resolved_hires_score_hashes = {
        model: next(iter(hashes)) for model, hashes in hires_score_hashes.items()
    }
    fractions = tuple(
        value for value in config.analysis.subsample_fractions if value < 1.0
    )
    settings = AnalysisSettings(
        frequency_max=max(
            abs(config.analysis.ecf_frequency_min),
            abs(config.analysis.ecf_frequency_max),
        ),
        frequency_count=config.analysis.ecf_frequency_count,
        ecf_chunk_size=config.analysis.chunk_size,
        subsample_fractions=fractions,
        block_count=4,
    )
    return aggregate_task_paths(
        paths,
        layout.aggregate_dir,
        layout.figure_dir,
        settings=settings,
        resume=resume,
        diagnostic_context={
            "config_hash": config.resolved_hash,
            "code_hash": code_hash,
            "raw_directory": str(layout.raw_dir),
            "score_table_hashes": resolved_score_hashes,
            "score_table_hires_hashes": resolved_hires_score_hashes,
        },
        tier=config.experiment.tier,
        checkpoint_fractions=config.design.checkpoint_fractions,
        raw_task_inventory=raw_task_inventory,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify, aggregate and render a complete Experiment 2 run; no simulation "
            "is imported or executed."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default="auto")
    parser.add_argument("--precision", choices=("float32", "float64", "mixed"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pilot-output-dir",
        type=Path,
        default=None,
        help="output root containing the matching passed pilot (defaults to --output-dir)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="verify byte-identical existing outputs instead of overwriting them",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-publication-scale", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        from .config import enumerate_tasks, load_config
        from .storage import run_layout

        config = load_config(
            arguments.config,
            precision_override=arguments.precision,
            seed_override=arguments.seed,
            allow_final=arguments.allow_publication_scale,
        )
        layout = run_layout(arguments.output_dir, config)
        if arguments.dry_run:
            expected = enumerate_tasks(config)
            completed = (
                sum(
                    (layout.raw_dir / task.task_id / "arrays.npz").is_file()
                    for task in expected
                )
                if layout.raw_dir.is_dir()
                else 0
            )
            print(
                json.dumps(
                    {
                        "analysis_device": "cpu",
                        "config_hash": config.resolved_hash,
                        "expected_tasks": len(expected),
                        "completed_task_arrays": completed,
                        "requested_device_ignored_for_analysis": arguments.device,
                        "run_dir": str(layout.run_dir),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if config.experiment.publication_scale:
            verify_pilot_gate(
                experiment=2,
                final_config=arguments.config,
                output_root=arguments.output_dir,
                pilot_output_root=arguments.pilot_output_dir,
                precision=arguments.precision,
            )
        outputs = aggregate_configured_run(config, arguments.output_dir, resume=arguments.resume)
        print(json.dumps(outputs, indent=2, sort_keys=True))
        return 0
    except ExperimentError as exc:
        build_parser().exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
