"""Strict aggregation and figure generation for Experiment 1."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from levy_experiments.authority import authority_manifest
from levy_experiments.errors import ArtifactError, ExperimentError
from levy_experiments.pilot_gate import verify_pilot_gate

from .analysis import (
    AnalysisSettings,
    analyze_tasks,
    diagnostics,
    load_tasks,
    make_decisions,
    pool_mean_excess_rows,
    summarize_m_slopes,
    summarize_tail_metrics,
)
from .config import Experiment1Config, enumerate_tasks, load_config
from .score_tables import (
    load_score_table,
    score_table_path,
    sensitivity_score_table_path,
    table_key,
)
from .storage import (
    RunLayout,
    atomic_write_json,
    compute_code_hash,
    load_task_artifact,
    prepare_run,
    run_layout,
    sha256_file,
    task_directory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Experiment 1 raw tasks")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default="cpu")
    parser.add_argument("--precision", choices=("float32", "float64", "mixed"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--pilot-output-dir",
        type=Path,
        default=None,
        help="output root containing the matching passed pilot (defaults to --output-dir)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-publication-scale", action="store_true")
    return parser


def _analysis_settings(config: Experiment1Config) -> AnalysisSettings:
    subsamples = tuple(value for value in config.analysis.subsample_fractions if value < 1.0)
    settings = AnalysisSettings(
        tier=config.experiment.tier,
        bootstrap_replicates=config.analysis.bootstrap_replicates,
        subsample_fractions=subsamples,
        block_count=config.analysis.blocks,
        m_masses=config.analysis.survival_probabilities,
        m_primary_window=config.analysis.fit_primary,
        m_sensitivity_windows=config.analysis.fit_sensitivities,
        ecf_frequency_max=config.analysis.ecf_frequency_max,
        ecf_frequency_count=config.analysis.ecf_frequency_count,
        ecf_chunk_size=config.analysis.chunk_size,
        minimum_exceedances_per_seed=config.analysis.minimum_exceedances_per_seed,
        minimum_pooled_exceedances=config.analysis.minimum_pooled_exceedances,
        minimum_contributing_seeds=config.analysis.minimum_contributing_seeds,
        pt_ecf_familywise_alpha=config.analysis.pt_ecf_familywise_alpha,
        score_sensitivity_floor_fraction=config.analysis.score_sensitivity_floor_fraction,
        score_sensitivity_hill_tolerance=config.analysis.score_sensitivity_hill_tolerance,
        score_sensitivity_constant_relative_tolerance=(
            config.analysis.score_sensitivity_constant_relative_tolerance
        ),
        refinement_floor_fraction=config.analysis.refinement_floor_fraction,
        refinement_hill_tolerance=config.analysis.refinement_hill_tolerance,
        refinement_constant_relative_tolerance=(
            config.analysis.refinement_constant_relative_tolerance
        ),
        refinement_m_slope_tolerance=config.analysis.refinement_m_slope_tolerance,
        gaussian_hill_separation_margin=config.analysis.gaussian_hill_separation_margin,
    )
    settings.validate()
    return settings


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    if path.exists():
        raise ArtifactError(f"refusing to overwrite aggregate: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({name for row in rows for name in row})
    if not fields:
        fields = ["empty"]
        rows = ({"empty": ""},)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fields})
    payload = buffer.getvalue().encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise ArtifactError(f"concurrent writer created aggregate: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_analysis_npz(
    path: Path,
    summaries: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
) -> Path:
    if path.exists():
        raise ArtifactError(f"refusing to overwrite aggregate: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary_model": np.asarray([str(row["model"]) for row in summaries]),
        "summary_nu": np.asarray([float(row["target_nu"]) for row in summaries]),
        "summary_steps": np.asarray([int(row["steps"]) for row in summaries]),
        "summary_metric": np.asarray([str(row["metric"]) for row in summaries]),
        "summary_fraction": np.asarray([float(row["fraction"]) for row in summaries]),
        "summary_median": np.asarray([float(row["median"]) for row in summaries]),
        "summary_q16": np.asarray([float(row["q16"]) for row in summaries]),
        "summary_q84": np.asarray([float(row["q84"]) for row in summaries]),
        "decision_claim": np.asarray([str(row["claim"]) for row in decisions]),
        "decision_value": np.asarray([str(row["decision"]) for row in decisions]),
        "decision_reason": np.asarray([str(row["reason"]) for row in decisions]),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _load_verified_tasks(
    layout: RunLayout,
    config: Experiment1Config,
) -> tuple[list[Path], list[dict[str, Any]], dict[str, str]]:
    tasks = enumerate_tasks(config)
    code_hash = compute_code_hash()
    expected_directories = {task.task_id for task in tasks}
    actual_directories = {path.name for path in layout.raw_dir.iterdir() if path.is_dir()}
    extra = sorted(actual_directories - expected_directories)
    if extra:
        raise ArtifactError(f"unexpected raw task directories: {extra}")
    paths: list[Path] = []
    metadata_rows: list[dict[str, Any]] = []
    table_hashes: dict[str, str] = {}
    for model, nus in (
        ("stable", config.target.stable_nus),
        ("vp", config.target.vp_nus),
    ):
        for nu in nus:
            table = load_score_table(score_table_path(layout.run_dir, config, model, nu))
            key = table_key(model, nu)
            table_hashes[key] = table.table_hash
            sensitivity = load_score_table(
                sensitivity_score_table_path(layout.run_dir, config, model, nu)
            )
            table_hashes[f"{key}:hires"] = sensitivity.table_hash
    for task in tasks:
        _, metadata = load_task_artifact(
            layout,
            task,
            expected_config_hash=config.resolved_hash,
            expected_code_hash=code_hash,
        )
        if task.kind == "dynamic":
            assert task.nu is not None
            expected_hash = table_hashes[table_key(task.model, task.nu)]
            if metadata.get("score_table_hash") != expected_hash:
                raise ArtifactError(f"score-table hash mismatch in {task.task_id}")
            expected_sensitivity = table_hashes[f"{table_key(task.model, task.nu)}:hires"]
            if metadata.get("score_sensitivity_table_hash") != expected_sensitivity:
                raise ArtifactError(f"score-sensitivity table hash mismatch in {task.task_id}")
        paths.append(task_directory(layout, task) / "arrays.npz")
        metadata_rows.append(metadata)
    return paths, metadata_rows, table_hashes


def _verify_existing_manifest(layout: RunLayout, config: Experiment1Config) -> dict[str, Any]:
    manifest_path = layout.aggregate_dir / "aggregate_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot resume aggregate manifest: {manifest_path}") from exc
    if manifest.get("config_hash") != config.resolved_hash:
        raise ArtifactError("aggregate manifest configuration mismatch")
    if manifest.get("code_hash") != compute_code_hash():
        raise ArtifactError("aggregate manifest code-version mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ArtifactError("aggregate manifest lacks its file inventory")
    for relative, expected_hash in files.items():
        path = layout.run_dir / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ArtifactError(f"aggregate file is absent or corrupt: {path}")
    return manifest


def _enforce_final_publication_gate(
    config: Experiment1Config,
    layout: RunLayout,
    diagnostic_payload: dict[str, Any],
) -> None:
    """Persist a machine-readable refusal and prevent final figure rendering."""

    if config.experiment.tier != "final" or bool(diagnostic_payload["publication_gate_pass"]):
        return
    failure_path = layout.aggregate_dir / "publication_gate_failure.json"
    atomic_write_json(failure_path, diagnostic_payload)
    reasons = "; ".join(diagnostic_payload["publication_gate_failures"])
    raise ArtifactError(
        "final Experiment 1 rendering refused because publication gates failed: " + reasons
    )


def aggregate_run(
    config: Experiment1Config,
    layout: RunLayout,
    *,
    resume: bool,
) -> dict[str, Any]:
    from .plotting import plot_diagnostics, plot_main, plot_quantile_ratio, write_caption

    manifest_path = layout.aggregate_dir / "aggregate_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise ArtifactError("aggregate already exists; pass --resume to verify it")
        return _verify_existing_manifest(layout, config)

    task_paths, task_metadata, table_hashes = _load_verified_tasks(layout, config)
    tasks = load_tasks(task_paths)
    settings = _analysis_settings(config)
    metric_rows, mean_rows, slope_rows = analyze_tasks(tasks, settings)
    summaries = summarize_tail_metrics(metric_rows, settings)
    pooled_mean = pool_mean_excess_rows(mean_rows, settings)
    slope_summaries = summarize_m_slopes(slope_rows, settings)
    decisions = make_decisions(
        tasks,
        metric_rows,
        summaries,
        pooled_mean,
        slope_summaries,
        settings,
    )
    diagnostic_payload = diagnostics(
        tasks,
        metric_rows,
        mean_rows,
        slope_rows,
        decisions,
        settings,
    ) | {
        "config_hash": config.resolved_hash,
        "code_hash": compute_code_hash(),
        "score_table_hashes": table_hashes,
    }
    _enforce_final_publication_gate(config, layout, diagnostic_payload)

    outputs = [
        _write_csv(layout.aggregate_dir / "tail_metrics_per_seed.csv", metric_rows),
        _write_csv(layout.aggregate_dir / "mean_excess_sufficient.csv", mean_rows),
        _write_csv(layout.aggregate_dir / "m_slopes_per_seed.csv", slope_rows),
        _write_csv(layout.aggregate_dir / "tail_summary.csv", summaries),
        _write_csv(layout.aggregate_dir / "mean_excess_pooled.csv", pooled_mean),
        _write_csv(layout.aggregate_dir / "m_slope_summary.csv", slope_summaries),
        _write_csv(layout.aggregate_dir / "tail_decisions.csv", decisions),
        _write_analysis_npz(layout.aggregate_dir / "analysis_arrays.npz", summaries, decisions),
    ]
    diagnostics_path = layout.aggregate_dir / "diagnostics.json"
    atomic_write_json(diagnostics_path, diagnostic_payload)
    outputs.append(diagnostics_path)
    outputs.extend(
        plot_main(
            summaries,
            layout.figure_dir,
            tier=settings.tier,
            primary_steps=config.design.primary_steps,
        )
    )
    outputs.extend(
        plot_quantile_ratio(
            summaries,
            layout.figure_dir,
            tier=settings.tier,
            primary_steps=config.design.primary_steps,
        )
    )
    outputs.extend(
        plot_diagnostics(
            summaries,
            pooled_mean,
            slope_summaries,
            metric_rows,
            layout.figure_dir,
            tier=settings.tier,
            primary_steps=config.design.primary_steps,
        )
    )
    outputs.append(write_caption(layout.figure_dir / "caption.md", tier=settings.tier))

    task_hashes = {str(path.relative_to(layout.run_dir)): sha256_file(path) for path in task_paths}
    file_hashes = {str(path.relative_to(layout.run_dir)): sha256_file(path) for path in outputs}
    manifest = {
        "experiment": 1,
        "authority": authority_manifest(),
        "tier": config.experiment.tier,
        "config_hash": config.resolved_hash,
        "code_hash": compute_code_hash(),
        "task_count": len(tasks),
        "task_npz_hashes": task_hashes,
        "task_metadata_hashes": {
            str(path.relative_to(layout.run_dir).parent / "metadata.json"): sha256_file(
                path.parent / "metadata.json"
            )
            for path in task_paths
        },
        "score_table_hashes": table_hashes,
        "files": file_hashes,
        "status": "complete",
        "scientific_interpretation": (
            "finite-sample numerical illustration; never a proof of tail equivalence"
        ),
        "execution_seconds": [
            float(metadata["execution_seconds_excluding_compilation"]) for metadata in task_metadata
        ],
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(
            args.config,
            precision_override=args.precision,
            seed_override=args.seed,
            allow_final=args.allow_publication_scale,
        )
        layout = run_layout(args.output_dir, config)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "aggregation_device": "cpu",
                        "config_hash": config.resolved_hash,
                        "requested_device": args.device,
                        "run_dir": str(layout.run_dir),
                        "task_count": len(enumerate_tasks(config)),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if config.experiment.publication_scale:
            verify_pilot_gate(
                experiment=1,
                final_config=args.config,
                output_root=args.output_dir,
                pilot_output_root=args.pilot_output_dir,
                precision=args.precision,
            )
        if args.device not in {"cpu", "auto"}:
            print("warning: aggregation is CPU-only; --device is recorded but ignored")
        layout = prepare_run(args.output_dir, config)
        manifest = aggregate_run(config, layout, resume=args.resume)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except ExperimentError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
