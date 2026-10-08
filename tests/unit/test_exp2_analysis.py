from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

import experiments.exp2.aggregate as aggregate_module
import experiments.exp2.analysis as analysis_module
from experiments.exp2.aggregate import OUTPUT_FILENAMES
from experiments.exp2.analysis import (
    AnalysisSettings,
    Exp2ArtifactError,
    _empirical_wasserstein_sorted_1d,
    _publication_gate_by_model_order,
    analyze_tasks,
    diagnostics,
    exact_forward_cf,
    fit_horizon_initialization_per_seed,
    fit_refinement_per_seed,
    load_task_npz,
    load_tasks,
    student_t4_cf,
    summarize_rows,
    validate_task_collection,
)
from levy_experiments.errors import ArtifactError
from levy_experiments.metrics.wasserstein import empirical_wasserstein_1d


def _task_payload(
    model: str,
    steps: int,
    seed: int,
    *,
    sample_count: int = 32,
) -> dict[str, np.ndarray]:
    base = np.linspace(-2.0, 2.0, sample_count, dtype=np.float64)
    h = (2.0 - 0.05) / steps
    error = 0.1 * h
    return {
        "model": np.asarray(model),
        "steps": np.asarray(steps),
        "seed": np.asarray(seed),
        "T": np.asarray(2.0),
        "epsilon": np.asarray(0.05),
        "alpha": np.asarray(1.5),
        "eta": np.asarray(0.5 if model == "stable" else 1.0),
        "beta": np.asarray(1.0),
        "forward_times": np.asarray([0.05]),
        "numerical_exact_init": (base + error)[None, :],
        "numerical_reference_init": (base + 2.0 * error)[None, :],
        "exact_a": (base - 0.005)[None, :],
        "exact_b": (base + 0.005)[None, :],
        "exact_c": base[None, :],
        "wasserstein_orders": np.asarray([1.0, 1.25]),
        "score_sensitivity_wp": np.asarray([[0.001, 0.001]]),
        "d_controlled": np.asarray([[True, True]]),
    }


def _write_task(
    directory: Path,
    model: str,
    steps: int,
    seed: int,
    *,
    payload: dict[str, np.ndarray] | None = None,
) -> Path:
    path = directory / f"{model}-N{steps}-s{seed}.npz"
    np.savez_compressed(
        path,
        **(_task_payload(model, steps, seed) if payload is None else payload),
    )
    return path


def _task_family(directory: Path) -> list[Path]:
    paths = []
    for model in ("stable", "vp"):
        for steps in (8, 16, 32):
            for seed in (0, 1):
                payload = _task_payload(model, steps, seed)
                payload["exact_a"] = payload["exact_c"] - 0.0005
                payload["exact_b"] = payload["exact_c"] + 0.0005
                payload["score_sensitivity_wp"] = np.asarray([[0.0001, 0.0001]])
                paths.append(
                    _write_task(directory, model, steps, seed, payload=payload)
                )
    return paths


def test_loader_and_exact_wasserstein_decomposition(tmp_path: Path) -> None:
    source = _write_task(tmp_path, "stable", 8, 0)
    task = load_task_npz(source)
    rows = analyze_tasks(
        [task, load_task_npz(_write_task(tmp_path, "vp", 8, 0))],
        AnalysisSettings(subsample_fractions=(), block_count=2),
    )
    full = [
        row
        for row in rows
        if row["model"] == "stable"
        and row["metric"] == "wasserstein"
        and row["sample_kind"] == "full"
        and row["order"] == 1.25
    ]
    values = {row["component"]: row["value"] for row in full}
    expected_error = 0.1 * task.h
    assert values["D"] == pytest.approx(expected_error)
    assert values["I"] == pytest.approx(expected_error)
    assert values["E"] == pytest.approx(2.0 * expected_error)
    assert values["F"] == pytest.approx(0.01)
    triangle = next(
        row["value"]
        for row in rows
        if row["model"] == "stable"
        and row["metric"] == "triangle_slack"
        and row["sample_kind"] == "full"
        and row["order"] == 1.25
    )
    assert triangle >= -1.0e-12
    assert {
        row["order"]
        for row in rows
        if row["model"] == "stable" and row["metric"] == "wasserstein"
    } == {1.0, 1.25}


def test_loader_rejects_checkpoint_start_shifted_from_horizon(tmp_path: Path) -> None:
    payload = _task_payload("stable", 8, 0)
    payload["forward_times"] = np.asarray([1.9, 1.025, 0.05], dtype=np.float64)
    for name in analysis_module.REQUIRED_SAMPLE_KEYS:
        payload[name] = np.repeat(payload[name], 3, axis=0)
    payload["score_sensitivity_wp"] = np.repeat(
        payload["score_sensitivity_wp"], 3, axis=0
    )
    payload["d_controlled"] = np.repeat(payload["d_controlled"], 3, axis=0)
    path = _write_task(tmp_path, "stable", 8, 0, payload=payload)

    with pytest.raises(Exp2ArtifactError, match="first forward time must equal T"):
        load_task_npz(path)


def test_configured_checkpoint_fractions_reject_shifted_interior_time(tmp_path: Path) -> None:
    paths = []
    for model in ("stable", "vp"):
        payload = _task_payload(model, 8, 0)
        payload["forward_times"] = np.asarray(
            [2.0, 1.5125, 1.125, 0.5375, 0.05], dtype=np.float64
        )
        for name in analysis_module.REQUIRED_SAMPLE_KEYS:
            payload[name] = np.repeat(payload[name], 5, axis=0)
        payload["score_sensitivity_wp"] = np.repeat(
            payload["score_sensitivity_wp"], 5, axis=0
        )
        payload["d_controlled"] = np.repeat(payload["d_controlled"], 5, axis=0)
        paths.append(_write_task(tmp_path, model, 8, 0, payload=payload))
    tasks = [load_task_npz(path) for path in paths]

    with pytest.raises(Exp2ArtifactError, match="checkpoint times do not match"):
        validate_task_collection(tasks, checkpoint_fractions=(0.0, 0.25, 0.5, 0.75, 1.0))


def test_presorted_wasserstein_is_bitwise_identical_to_original_estimator() -> None:
    rng = np.random.default_rng(918273)
    first = rng.standard_t(4.0, size=4096)
    second = rng.standard_t(4.0, size=4096)
    sorted_first = np.sort(first)
    sorted_second = np.sort(second)
    for order in (1.0, 1.25):
        expected = empirical_wasserstein_1d(first, second, order=order)
        actual = _empirical_wasserstein_sorted_1d(
            sorted_first, sorted_second, order=order
        )
        assert actual == expected


def test_wasserstein_analysis_sorts_each_selected_sample_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = load_task_npz(_write_task(tmp_path, "stable", 8, 0))
    settings = AnalysisSettings(subsample_fractions=(0.25, 0.5), block_count=4)
    original_sort = np.sort
    calls = 0

    def counted_sort(values, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_sort(values, *args, **kwargs)

    monkeypatch.setattr(analysis_module.np, "sort", counted_sort)
    analysis_module._wp_rows(task, settings)
    # One full plan, two nested plans and four blocks, with five sample arrays.
    assert calls == 7 * 5


def _canonical_rows(rows: list[dict[str, object]]) -> list[tuple[tuple[str, object], ...]]:
    canonical: list[tuple[tuple[str, object], ...]] = []
    for row in rows:
        items: list[tuple[str, object]] = []
        for name, value in sorted(row.items()):
            normalized = "NaN" if isinstance(value, float) and math.isnan(value) else value
            items.append((name, normalized))
        canonical.append(tuple(items))
    return sorted(canonical, key=repr)


def test_exact_reference_ecf_cache_preserves_rows_and_reduces_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [
        _write_task(tmp_path, model, steps, 0)
        for model in ("stable", "vp")
        for steps in (8, 16)
    ]
    tasks = load_tasks(paths)
    settings = AnalysisSettings(
        frequency_max=2.0,
        frequency_count=9,
        ecf_chunk_size=16,
        subsample_fractions=(),
        block_count=2,
    )
    original_ecf = analysis_module.empirical_cf_1d
    calls = 0

    def counted_ecf(samples, frequencies, *, chunk_size):
        nonlocal calls
        calls += 1
        return original_ecf(samples, frequencies, chunk_size=chunk_size)

    monkeypatch.setattr(analysis_module, "empirical_cf_1d", counted_ecf)
    uncached: list[dict[str, object]] = []
    for task in sorted(tasks, key=lambda item: item.key):
        uncached.extend(analysis_module.analyze_task(task, settings))
    uncached_calls = calls

    calls = 0
    cached = analysis_module.analyze_tasks(tasks, settings)
    cached_calls = calls

    assert _canonical_rows(cached) == _canonical_rows(uncached)
    assert uncached_calls == 4 * 5
    # Two numerical arrays are always evaluated per task.  The three exact
    # arrays in this controlled fixture are byte-identical across all tasks.
    assert cached_calls == 4 * 2 + 3


def test_loader_rejects_wrong_eta_and_complex_samples(tmp_path: Path) -> None:
    wrong_eta = _task_payload("stable", 8, 0)
    wrong_eta["eta"] = np.asarray(1.0)
    with pytest.raises(Exp2ArtifactError, match=r"requires eta=0\.5"):
        load_task_npz(_write_task(tmp_path, "stable", 8, 0, payload=wrong_eta))

    complex_samples = _task_payload("stable", 8, 0)
    complex_samples["exact_c"] = complex_samples["exact_c"].astype(np.complex128)
    with pytest.raises(Exp2ArtifactError, match="real numerical"):
        load_task_npz(_write_task(tmp_path, "stable", 8, 0, payload=complex_samples))

    reused_exact = _task_payload("stable", 8, 0)
    reused_exact["exact_b"] = reused_exact["exact_a"].copy()
    with pytest.raises(Exp2ArtifactError, match="exact references must be independent"):
        load_task_npz(_write_task(tmp_path, "stable", 8, 0, payload=reused_exact))


def test_loader_fails_closed_without_paired_table_controls(tmp_path: Path) -> None:
    missing_sensitivity = _task_payload("stable", 8, 0)
    missing_sensitivity.pop("score_sensitivity_wp")
    with pytest.raises(Exp2ArtifactError, match="missing NPZ key: score_sensitivity_wp"):
        load_task_npz(
            _write_task(tmp_path, "stable", 8, 0, payload=missing_sensitivity)
        )

    missing_decision = _task_payload("stable", 8, 0)
    missing_decision.pop("d_controlled")
    with pytest.raises(Exp2ArtifactError, match="missing NPZ key: d_controlled"):
        load_task_npz(_write_task(tmp_path, "stable", 8, 0, payload=missing_decision))


def test_analysis_rejects_a_stale_or_forged_control_decision(tmp_path: Path) -> None:
    payload = _task_payload("stable", 8, 0)
    payload["d_controlled"] = np.asarray([[False, False]])
    task = load_task_npz(_write_task(tmp_path, "stable", 8, 0, payload=payload))

    with pytest.raises(Exp2ArtifactError, match="d_controlled disagrees"):
        analysis_module._wp_rows(
            task,
            AnalysisSettings(subsample_fractions=(), block_count=2),
        )


def test_publication_gate_reports_any_uncontrolled_D_point_even_when_coverage_is_low(
    tmp_path: Path,
) -> None:
    paths = [
        _write_task(tmp_path, model, 8, 0)
        for model in ("stable", "vp")
    ]
    tasks = load_tasks(paths)
    rows = analyze_tasks(tasks, AnalysisSettings(subsample_fractions=(), block_count=2))
    report = diagnostics(tasks, rows, (), ({"status": "eligible"},))
    assert report["publication_gate_pass"] is False
    assert not any(
        "score_table_sensitivity_exceeds" in failure
        for failure in report["publication_gate_failures"]
    )

    uncontrolled = [dict(row) for row in rows]
    changed = False
    for row in uncontrolled:
        if (
            not changed
            and row["metric"] == "wasserstein"
            and row["component"] == "D"
            and row["sample_kind"] == "full"
        ):
            row["controlled"] = False
            changed = True
    failed = diagnostics(tasks, uncontrolled, (), ({"status": "eligible"},))
    assert failed["publication_gate_pass"] is False
    assert (
        "score_table_sensitivity_exceeds_half_exact_exact_floor_for_1_of_4_D_points"
        in failed["publication_gate_failures"]
    )


def test_publication_gate_rejects_missing_horizon_fit_triangle_and_ecf_failures(
    tmp_path: Path,
) -> None:
    tasks = load_tasks(
        [_write_task(tmp_path, model, 8, 0) for model in ("stable", "vp")]
    )
    rows = analyze_tasks(tasks, AnalysisSettings(subsample_fractions=(), block_count=2))

    no_horizon = diagnostics(tasks, rows, (), ())
    assert no_horizon["horizon_claim_permitted"] is False
    assert any(
        "horizon_fit_seed_coverage_0_below_3" in failure
        for failure in no_horizon["publication_gate_failures"]
    )

    mutated = [dict(row) for row in rows]
    next(row for row in mutated if row["metric"] == "triangle_slack")["value"] = -1.0
    exact_row = next(
        row
        for row in mutated
        if row["metric"] == "ecf_max_component" and row["component"] == "A_exact"
    )
    exact_row["value"] = float(exact_row["hoeffding_radius"]) + 1.0
    failed = diagnostics(tasks, mutated, (), ({"status": "eligible"},))
    assert failed["publication_gate_pass"] is False
    assert "empirical_triangle_violated_for_1_rows" in failed["publication_gate_failures"]
    assert any(
        failure.startswith("exact_ECF_outside_simultaneous_radius_fraction_")
        for failure in failed["publication_gate_failures"]
    )


def test_model_order_gate_requires_seed_coverage_for_horizon_and_finest_pair() -> None:
    tasks = [
        type("Task", (), {"model": model, "seed": seed})()
        for model in ("stable", "vp")
        for seed in range(4)
    ]
    rows: list[dict[str, object]] = []
    for model in ("stable", "vp"):
        for order in (1.0, 1.25):
            for seed in range(4):
                for component in ("D", "I"):
                    for steps, h, value in ((312, 0.00625, 0.20), (624, 0.003125, 0.205)):
                        rows.append(
                            {
                                "model": model,
                                "order": order,
                                "seed": seed,
                                "metric": "wasserstein",
                                "component": component,
                                "sample_kind": "full",
                                "replicate": -1,
                                "horizon": 2.0,
                                "forward_time": 0.05,
                                "steps": steps,
                                "h": h,
                                "value": value,
                                "floor": 0.01,
                                "controlled": True,
                            }
                        )
    one_fit = [
        {"model": "stable", "order": 1.0, "seed": 0},
    ]
    insufficient = _publication_gate_by_model_order(tasks, rows, one_fit)
    assert len(insufficient) == 4
    assert all(detail["required_seed_count"] == 3 for detail in insufficient)
    assert all(detail["finest_pair_gate_pass"] is True for detail in insufficient)
    assert all(detail["horizon_fit_gate_pass"] is False for detail in insufficient)
    assert all(detail["gate_pass"] is False for detail in insufficient)

    covered_fits = [
        {"model": model, "order": order, "seed": seed}
        for model in ("stable", "vp")
        for order in (1.0, 1.25)
        for seed in range(3)
    ]
    covered = _publication_gate_by_model_order(tasks, rows, covered_fits)
    assert all(detail["horizon_fit_gate_pass"] is True for detail in covered)
    assert all(detail["finest_pair_passing_seeds"] == [0, 1, 2, 3] for detail in covered)
    assert all(detail["gate_pass"] is True for detail in covered)

    refinement_failed = [dict(row) for row in rows]
    for row in refinement_failed:
        if (
            row["model"] == "stable"
            and row["order"] == 1.0
            and row["seed"] in {0, 1}
            and row["steps"] == 624
        ):
            row["value"] = 0.5
    failed_details = _publication_gate_by_model_order(
        tasks, refinement_failed, covered_fits
    )
    stable_w1 = next(
        detail
        for detail in failed_details
        if detail["model"] == "stable" and detail["order"] == 1.0
    )
    assert stable_w1["finest_pair_passing_seeds"] == [2, 3]
    assert stable_w1["finest_pair_gate_pass"] is False
    assert "finest_pair_passing_seed_coverage_2_below_3" in stable_w1["failures"]


def test_collection_key_includes_horizon_and_accepts_horizon_sweep(tmp_path: Path) -> None:
    paths = []
    for model in ("stable", "vp"):
        for horizon in (2.0, 8.0):
            payload = _task_payload(model, 120, 0)
            payload["T"] = np.asarray(horizon)
            path = tmp_path / f"{model}-T{horizon:g}-N120-s0.npz"
            np.savez_compressed(path, **payload)
            paths.append(path)
    tasks = load_tasks(paths)
    assert len({task.key for task in tasks}) == 4
    assert {task.horizon for task in tasks} == {2.0, 8.0}


def test_exact_student_and_forward_characteristic_functions() -> None:
    frequencies = np.asarray([-2.0, -0.5, 0.0, 1.0e-10, 0.5, 2.0])
    target = student_t4_cf(frequencies)
    assert target[2] == 1.0
    assert np.all(np.isfinite(target))
    assert np.allclose(target, target[::-1], atol=1.0e-12)
    assert np.allclose(exact_forward_cf("stable", 0.0, frequencies), target)
    assert np.allclose(exact_forward_cf("vp", 0.0, frequencies), target)

    time = 0.7
    u = np.asarray([0.0, 0.4, 1.2])
    lost_mass = 1.0 - math.exp(-time)
    stable_expected = student_t4_cf(np.exp(-time / 1.5) * u) * np.exp(
        -lost_mass * np.abs(u) ** 1.5
    )
    vp_expected = student_t4_cf(np.exp(-time / 2.0) * u) * np.exp(
        -0.5 * lost_mass * u**2
    )
    assert np.allclose(exact_forward_cf("stable", time, u), stable_expected)
    assert np.allclose(exact_forward_cf("vp", time, u), vp_expected)


def test_blocks_are_reduced_within_seed_before_cross_seed_quantiles() -> None:
    rows = []
    for seed, values in ((0, (0.0, 2.0, 4.0, 6.0)), (1, (10.0, 12.0, 14.0, 16.0))):
        for replicate, value in enumerate(values):
            rows.append(
                {
                    "model": "stable",
                    "horizon": 8.0,
                    "steps": 32,
                    "h": (8.0 - 0.05) / 32,
                    "seed": seed,
                    "forward_time": 0.05,
                    "metric": "wasserstein",
                    "component": "D",
                    "order": 1.25,
                    "sample_kind": "block",
                    "sample_size": 8,
                    "replicate": replicate,
                    "value": value,
                    "floor": 0.1,
                    "score_sensitivity": 0.01,
                    "controlled": True,
                    "hoeffding_radius": math.nan,
                }
            )
    [summary] = summarize_rows(rows)
    assert summary["n_seeds"] == 2
    assert summary["median"] == pytest.approx(8.0)


def test_refinement_fits_are_per_seed_and_require_three_controlled_resolved_points(
    tmp_path: Path,
) -> None:
    rows = analyze_tasks(
        load_tasks(_task_family(tmp_path)),
        AnalysisSettings(subsample_fractions=(), block_count=2),
    )
    fits = fit_refinement_per_seed(rows)
    assert len(fits) == 8  # two models x two seeds x two Wasserstein orders
    assert all(fit["slope"] == pytest.approx(1.0, abs=1.0e-12) for fit in fits)
    assert all(fit["n_points"] == 3 for fit in fits)
    ecf_summaries = [
        row for row in summarize_rows(rows) if str(row["metric"]).startswith("ecf_")
    ]
    assert ecf_summaries
    assert all(summary["n_seeds"] == 2 for summary in ecf_summaries)

    restricted = [dict(row) for row in rows]
    for row in restricted:
        if (
            row["model"] == "stable"
            and row["seed"] == 0
            and row["order"] == 1.0
            and row["steps"] == 8
            and row["metric"] == "wasserstein"
            and row["component"] == "D"
            and row["sample_kind"] == "full"
        ):
            row["controlled"] = False
    restricted_fits = fit_refinement_per_seed(restricted)
    keys = {
        (fit["model"], fit["seed"], fit["forward_time"], fit["order"])
        for fit in restricted_fits
    }
    assert ("stable", 0, 0.05, 1.0) not in keys


def test_horizon_fit_uses_I_and_rejects_D_confounded_points() -> None:
    rows = []
    for horizon, steps in (
        (0.1, 4),
        (0.15, 8),
        (0.25, 16),
        (0.5, 36),
        (1.0, 76),
        (2.0, 156),
    ):
        initialization = math.exp(-0.3 * horizon)
        common = {
            "model": "stable",
            "horizon": horizon,
            "steps": steps,
            "h": 0.01 + horizon * 1.0e-5,
            "seed": 7,
            "forward_time": 0.05,
            "metric": "wasserstein",
            "order": 1.25,
            "sample_kind": "full",
            "sample_size": 1024,
            "replicate": -1,
            "floor": 1.0e-4,
        }
        rows.append(common | {"component": "I", "value": initialization})
        rows.append(common | {"component": "D", "value": 0.1 * initialization})
    # A coarser duplicate at T=2 must not enter the horizon fit.
    coarse = [row for row in rows if row["horizon"] == 2.0]
    rows.extend(row | {"steps": 78, "h": 0.025, "value": 10.0} for row in coarse)

    [fit] = fit_horizon_initialization_per_seed(rows)
    assert fit["component"] == "I"
    assert fit["n_points"] == 6
    assert fit["log_error_slope"] == pytest.approx(-0.3, abs=1.0e-12)
    assert fit["n_excluded_D_confounded"] == 0

    confounded = [
        row | {"value": 1.0}
        if row["component"] == "D" and row["horizon"] == 2.0
        else row
        for row in rows
    ]
    [controlled_fit] = fit_horizon_initialization_per_seed(confounded)
    assert controlled_fit["n_points"] == 5
    assert controlled_fit["n_excluded_D_confounded"] == 1

    unresolved = [
        row | {"floor": row["value"]}
            if row["component"] == "I" and row["horizon"] in {0.25, 0.5, 1.0, 2.0}
        else row
        for row in rows
    ]
    assert fit_horizon_initialization_per_seed(unresolved) == []


def test_aggregate_outputs_are_complete_loadable_and_byte_reproducible(tmp_path: Path) -> None:
    input_dir = tmp_path / "tasks"
    output_dir = tmp_path / "aggregates"
    input_dir.mkdir()
    task_paths = _task_family(input_dir)
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    raw_inventory = []
    for task_path in task_paths:
        metadata_path = metadata_dir / f"{task_path.stem}.json"
        metadata_path.write_text("{}\n", encoding="utf-8")
        raw_inventory.append(
            {
                "task_id": task_path.stem,
                "arrays_path": str(task_path.resolve()),
                "arrays_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
                "metadata_path": str(metadata_path.resolve()),
                "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            }
        )
    settings = AnalysisSettings(
        frequency_max=2.0,
        frequency_count=9,
        ecf_chunk_size=16,
        subsample_fractions=(0.25, 0.5),
        block_count=4,
    )
    aggregate_module.aggregate_task_paths(
        task_paths,
        output_dir,
        output_dir,
        settings=settings,
        raw_task_inventory=raw_inventory,
    )
    assert all((output_dir / name).is_file() for name in OUTPUT_FILENAMES)
    hashes_before = {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        for name in OUTPUT_FILENAMES
    }
    aggregate_module.aggregate_task_paths(
        task_paths,
        output_dir,
        output_dir,
        settings=settings,
        raw_task_inventory=raw_inventory,
        resume=True,
    )
    hashes_after = {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        for name in OUTPUT_FILENAMES
    }
    assert hashes_after == hashes_before
    manifest = json.loads((output_dir / "aggregate_manifest.json").read_text("utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["publication_gate_pass"] is False
    assert manifest["publication_gate_failures"]
    assert manifest["source"]["task_count"] == 12
    assert len(manifest["source"]["tasks"]) == 12
    for task_entry in manifest["source"]["tasks"]:
        arrays_path = Path(task_entry["arrays_path"])
        assert task_entry["arrays_sha256"] == hashlib.sha256(
            arrays_path.read_bytes()
        ).hexdigest()
        metadata_path = Path(task_entry["metadata_path"])
        assert task_entry["metadata_sha256"] == hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest()
    assert "aggregate_manifest.json" not in manifest["artifacts"]
    assert set(manifest["artifacts"]) == set(OUTPUT_FILENAMES) - {
        "aggregate_manifest.json"
    }
    with np.load(output_dir / "analysis_arrays.npz", allow_pickle=False) as archive:
        assert "per_seed__value" in archive.files
        assert "summary__median" in archive.files
        assert "fit__slope" in archive.files
        assert "horizon_fit__log_error_slope" in archive.files
    raw_source = Path(manifest["source"]["tasks"][0]["arrays_path"])
    raw_bytes = raw_source.read_bytes()
    raw_source.write_bytes(raw_source.read_bytes() + b"corrupt")
    with pytest.raises(ArtifactError, match="raw arrays changed after verification"):
        aggregate_module.aggregate_task_paths(
            task_paths,
            output_dir,
            output_dir,
            settings=settings,
            raw_task_inventory=raw_inventory,
            resume=True,
        )
    raw_source.write_bytes(raw_bytes)
    metadata_source = Path(manifest["source"]["tasks"][0]["metadata_path"])
    metadata_bytes = metadata_source.read_bytes()
    metadata_source.write_bytes(metadata_bytes + b"corrupt")
    with pytest.raises(ArtifactError, match="raw metadata changed after verification"):
        aggregate_module.aggregate_task_paths(
            task_paths,
            output_dir,
            output_dir,
            settings=settings,
            raw_task_inventory=raw_inventory,
            resume=True,
        )
    metadata_source.write_bytes(metadata_bytes)
    corrupted = output_dir / "metrics_per_seed.csv"
    corrupted.write_bytes(corrupted.read_bytes() + b"corrupt\n")
    with pytest.raises(ArtifactError, match="differs from existing output"):
        aggregate_module.aggregate_task_paths(
            task_paths,
            output_dir,
            output_dir,
            settings=settings,
            raw_task_inventory=raw_inventory,
            resume=True,
        )


def test_failed_final_gate_writes_refusal_before_rendering_or_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "tasks"
    aggregate_dir = tmp_path / "aggregates"
    figure_dir = tmp_path / "figures"
    input_dir.mkdir()
    task_paths = _task_family(input_dir)

    def forbidden_render(*args, **kwargs):
        raise AssertionError("render_figures must not run after a failed final gate")

    monkeypatch.setattr(aggregate_module, "render_figures", forbidden_render)
    with pytest.raises(ArtifactError, match="final Experiment 2 rendering refused"):
        aggregate_module.aggregate_task_paths(
            task_paths,
            aggregate_dir,
            figure_dir,
            settings=AnalysisSettings(
                frequency_max=2.0,
                frequency_count=9,
                ecf_chunk_size=16,
                subsample_fractions=(),
                block_count=2,
            ),
            tier="final",
        )

    refusal_path = aggregate_dir / "publication_gate_failure.json"
    refusal = json.loads(refusal_path.read_text(encoding="utf-8"))
    assert refusal["status"] == "refused"
    assert refusal["publication_gate_pass"] is False
    assert any(
        "horizon_fit_seed_coverage" in failure
        for failure in refusal["publication_gate_failures"]
    )
    assert not (aggregate_dir / "aggregate_manifest.json").exists()
    assert not figure_dir.exists()
