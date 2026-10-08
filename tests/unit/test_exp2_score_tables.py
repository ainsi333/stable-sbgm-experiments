from __future__ import annotations

import concurrent.futures
import copy
import math
import threading
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.exp2.score_tables as score_table_module
from experiments.exp2.config import load_config
from experiments.exp2.score_tables import (
    build_score_table,
    direct_score_reference,
    ensure_score_table,
    interpolate_score_jax,
    interpolate_score_numpy,
    load_score_table,
    score_table_path,
)
from experiments.exp2.theory import stable_tail_score, vp_tail_score
from levy_experiments.errors import ArtifactError

ROOT = Path(__file__).resolve().parents[2]


def _tiny_config(*, workers: int = 2) -> dict:
    return {
        "experiment": {
            "beta": 1.0,
            "epsilon": 0.05,
            "tier": "smoke",
        },
        "stable": {"alpha": 1.5},
        "design": {
            "horizon_sweep": ({"horizon": 0.5}, {"horizon": 2.0}),
            "refinement_horizon": 2.0,
        },
        "score_table": {
            "time_points": 25,
            "space_points": 513,
            "fft_points": 32768,
            "tail_l": 256.0,
            "x_max": 16.0,
            "stable_blend_start": 12.0,
            "stable_blend_end": 16.0,
            "vp_blend_start": 8.0,
            "vp_blend_end": 10.0,
            "validation_points": 6,
            "validation_x_max": {"stable": 5.0, "vp": 5.0},
            "validation_rtol": 6.0e-3,
            "validation_atol": 2.0e-6,
            "workers": workers,
        },
    }


@pytest.fixture(scope="module")
def stable_table():
    return build_score_table(_tiny_config(), "stable")


@pytest.fixture(scope="module")
def vp_table():
    return build_score_table(_tiny_config(), "vp")


def test_tables_are_model_specific_and_directly_validated_off_both_grids(
    stable_table, vp_table
) -> None:
    assert stable_table.model == "stable"
    assert vp_table.model == "vp"
    assert stable_table.metadata["score_definition"].startswith("fractional_multiplier")
    assert vp_table.metadata["score_definition"].startswith("ordinary_multiplier")
    assert stable_table.metadata["time_grid"] == "geometric_in_forward_time"
    for table in (stable_table, vp_table):
        validation = table.metadata["direct_off_grid_validation"]
        assert validation["axes"] == "space_and_time"
        assert validation["points"] == 6
        assert validation["max_tolerance_ratio"] <= 1.0
        assert table.metadata["interpolation_quantity"].startswith("R=x+")
        assert table.metadata["time_interpolation"] == ("four_point_lagrange_in_log_forward_time")
        ratios = table.time_nodes[1:] / table.time_nodes[:-1]
        assert np.max(ratios) - np.min(ratios) < 2e-14


def test_spectral_scores_match_independent_quadrature_in_the_bulk(stable_table, vp_table) -> None:
    probes = ((0.37, 3), (1.41, 11), (4.63, 19))
    for table in (stable_table, vp_table):
        for point, interval in probes:
            time = math.sqrt(table.time_nodes[interval] * table.time_nodes[interval + 1])
            estimate = float(interpolate_score_numpy(point, time, table))
            reference = direct_score_reference(
                table.model,
                point,
                time,
                alpha=float(table.metadata["alpha"]),
                beta=float(table.metadata["beta"]),
            )
            assert abs(estimate - reference) / (1.0 + abs(reference)) < 3.0e-3


def test_stationary_limits_detect_score_multiplier_or_normalization_errors(
    stable_table, vp_table
) -> None:
    # Build a single-row diagnostic directly at a much later time by extending
    # only the table horizon; the target contribution is then negligible.
    config = _tiny_config()
    config["design"]["horizon_sweep"] = ({"horizon": 12.0},)
    config["design"]["refinement_horizon"] = 12.0
    config["score_table"]["time_points"] = 33
    stable_late = build_score_table(config, "stable")
    vp_late = build_score_table(config, "vp")
    points = np.asarray([0.5, 1.0, 2.0])
    stable_score = interpolate_score_numpy(points, 12.0, stable_late)
    vp_score = interpolate_score_numpy(points, 12.0, vp_late)
    assert np.max(np.abs(stable_score + points / 1.5)) < 2.0e-4
    assert np.max(np.abs(vp_score + points)) < 3.0e-4


def test_blend_reaches_the_explicit_tail_exactly(stable_table, vp_table) -> None:
    for table, tail_function in (
        (stable_table, stable_tail_score),
        (vp_table, vp_tail_score),
    ):
        blend_end = float(table.metadata["blend_end"])
        tail_nodes = table.space_nodes >= blend_end
        for row, time in zip(table.scores, table.time_nodes, strict=True):
            if table.model == "stable":
                expected = tail_function(
                    table.space_nodes[tail_nodes],
                    float(time),
                    alpha=float(table.metadata["alpha"]),
                    beta=float(table.metadata["beta"]),
                )
            else:
                expected = tail_function(
                    table.space_nodes[tail_nodes],
                    float(time),
                    beta=float(table.metadata["beta"]),
                )
            assert np.array_equal(row[tail_nodes], expected)


def test_jax_interpolation_is_jittable_odd_shape_preserving_and_safe_at_zero(
    stable_table, vp_table
) -> None:
    points = np.asarray([0.0, 0.37, -1.41, 20.0, -20.0])
    time = 0.31
    for table in (stable_table, vp_table):
        expected = interpolate_score_numpy(points, time, table)
        compiled = jax.jit(
            lambda values, score_table=table: interpolate_score_jax(
                values, time, score_table, jnp.float64
            )
        )
        actual = np.asarray(compiled(jnp.asarray(points)))
        assert actual.shape == points.shape
        assert np.all(np.isfinite(actual))
        assert actual[0] == 0.0
        assert np.max(np.abs(actual - expected)) < 2.0e-12
        positive = np.asarray(
            interpolate_score_jax(jnp.asarray([0.2, 2.0, 20.0]), time, table, jnp.float64)
        )
        negative = np.asarray(
            interpolate_score_jax(jnp.asarray([-0.2, -2.0, -20.0]), time, table, jnp.float64)
        )
        assert np.array_equal(negative, -positive)
        outside_time = np.asarray(
            interpolate_score_jax(jnp.asarray([0.0, 1.0]), 0.01, table, jnp.float64)
        )
        assert np.all(np.isnan(outside_time))


def test_parallel_construction_is_numerically_deterministic() -> None:
    serial_config = _tiny_config(workers=1)
    serial_config["score_table"].update(
        time_points=7,
        space_points=129,
        fft_points=8192,
        tail_l=128.0,
        x_max=12.0,
        stable_blend_start=8.0,
        stable_blend_end=12.0,
        validation_points=3,
        validation_rtol=2.0e-2,
    )
    parallel_config = copy.deepcopy(serial_config)
    parallel_config["score_table"]["workers"] = 4
    serial = build_score_table(serial_config, "stable")
    parallel = build_score_table(parallel_config, "stable")
    assert np.array_equal(serial.time_nodes, parallel.time_nodes)
    assert np.array_equal(serial.space_nodes, parallel.space_nodes)
    assert np.array_equal(serial.scores, parallel.scores)
    assert serial.metadata["construction_workers"] == 1
    assert parallel.metadata["construction_workers"] == 4
    assert serial.metadata["config_signature"] == parallel.metadata["config_signature"]
    assert score_table_path("run", serial_config, "stable") == score_table_path(
        "run", parallel_config, "stable"
    )


def test_stable_smoke_blend_and_midtime_interpolation_meet_prefixed_accuracy() -> None:
    config = load_config(ROOT / "configs/smoke/exp2.toml")
    table = build_score_table(config, "stable")
    settings = config.score_table
    assert settings.stable_blend_start == 128.0
    assert settings.stable_blend_end == settings.stable_x_max == 192.0

    time_indices = (0, 3, 7, 11, 15)
    times = [table.time_nodes[0]] + [
        math.sqrt(table.time_nodes[index] * table.time_nodes[index + 1]) for index in time_indices
    ]
    blend_points = np.linspace(settings.stable_blend_start, settings.stable_blend_end, 17)
    bulk_points = np.asarray([0.0, 0.01, 0.05, 0.2, 1.0, 5.0, 10.0])
    worst = 0.0
    for time in times:
        for point in np.concatenate((bulk_points, blend_points)):
            estimate = float(interpolate_score_numpy(float(point), float(time), table))
            reference = direct_score_reference(
                "stable",
                float(point),
                float(time),
                alpha=config.stable.alpha,
                beta=config.experiment.beta,
            )
            worst = max(worst, abs(estimate - reference) / (1.0 + abs(reference)))
    assert worst < 2.0e-3

    # The cubic smoothstep used during construction has zero endpoint slopes;
    # the last stored node is exactly on the explicit asymptotic branch.
    for time, row in zip(table.time_nodes, table.scores, strict=True):
        expected = stable_tail_score(
            table.space_nodes[-1],
            float(time),
            alpha=config.stable.alpha,
            beta=config.experiment.beta,
        )
        assert row[-1] == expected


def test_model_addressed_persistence_round_trip(tmp_path) -> None:
    config = _tiny_config(workers=1)
    config["score_table"].update(
        time_points=5,
        space_points=129,
        fft_points=8192,
        tail_l=128.0,
        validation_points=3,
        validation_rtol=3.0e-2,
    )
    table = ensure_score_table(tmp_path, config, "vp", resume=True)
    path = score_table_path(tmp_path, config, "vp")
    assert "vp" in path.name
    loaded = load_score_table(path)
    resumed = ensure_score_table(tmp_path, config, "vp", resume=True)
    assert loaded.table_hash == table.table_hash == resumed.table_hash
    assert np.array_equal(loaded.scores, table.scores)


def test_exclusive_build_lock_rejects_concurrent_writer(tmp_path, monkeypatch) -> None:
    config = _tiny_config(workers=1)
    config["score_table"].update(
        time_points=5,
        space_points=129,
        fft_points=8192,
        tail_l=128.0,
        validation_points=3,
        validation_rtol=3.0e-2,
    )
    entered = threading.Event()
    release = threading.Event()
    original_build = score_table_module.build_score_table

    def blocked_build(build_config, model):
        entered.set()
        assert release.wait(timeout=20.0)
        return original_build(build_config, model)

    monkeypatch.setattr(score_table_module, "build_score_table", blocked_build)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(ensure_score_table, tmp_path, config, "vp", True)
        assert entered.wait(timeout=10.0)
        with pytest.raises(ArtifactError, match="scheduler dependency"):
            ensure_score_table(tmp_path, config, "vp", resume=True)
        release.set()
        completed = first.result(timeout=30.0)
    path = score_table_path(tmp_path, config, "vp")
    assert path.is_file()
    assert not path.with_name(f"{path.name}.lock").exists()
    assert ensure_score_table(tmp_path, config, "vp", resume=True).table_hash == (
        completed.table_hash
    )
    with pytest.raises(ArtifactError, match="resume is disabled"):
        ensure_score_table(tmp_path, config, "vp", resume=False)
