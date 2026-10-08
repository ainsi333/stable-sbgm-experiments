from __future__ import annotations

from pathlib import Path

import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.exp2.config import enumerate_tasks, load_config
from levy_experiments.errors import ArtifactError, ConfigurationError
from levy_experiments.exp2_initialization import (
    ReanalysisSettings,
    _log_axis_limits,
    joint_seed_bootstrap_intervals,
    render_initialization_figure,
    select_horizon_tasks,
    summarize_initialization_rows,
    wasserstein_from_sorted,
)
from levy_experiments.metrics.wasserstein import empirical_wasserstein_1d


def test_historical_source_exception_is_exactly_scoped(monkeypatch) -> None:
    from levy_experiments import exp2_initialization as module

    monkeypatch.setattr(module, "compute_code_hash", lambda: "c" * 64)
    historical_config = "eaaf84c9384771ba1526332eca3021d0ace1c02f32d9f1e5a7f52b062f7ef737"
    historical_code = "e160f3e2e59784d145a316519347656cf5217615e6e7ed6cc03712408d04fe52"
    module._validate_source_version(historical_config, historical_code)
    module._validate_source_version("any-checked-config", "c" * 64)
    with pytest.raises(ArtifactError, match="unsupported"):
        module._validate_source_version("changed-config", historical_code)
    with pytest.raises(ArtifactError, match="unsupported"):
        module._validate_source_version(historical_config, "d" * 64)


def _settings(*, orders: tuple[float, float] = (1.1, 1.4)) -> ReanalysisSettings:
    return ReanalysisSettings(
        orders=orders,
        confidence_level=0.9,
        bootstrap_replicates=2000,
        bootstrap_seed=1234,
        mc_resolution_multiplier=2.0,
        minimum_resolved_seed_fraction=0.75,
    )


def _synthetic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for order in (1.1, 1.4):
        for model_index, model in enumerate(("stable", "vp")):
            for horizon_index, (horizon, steps) in enumerate(
                ((0.1, 4), (0.25, 16), (0.5, 36), (1.0, 76), (2.0, 156))
            ):
                for seed in range(8):
                    initialization = (
                        (0.55 + 0.08 * model_index)
                        * np.exp(-0.7 * horizon)
                        * (1.0 + 0.01 * seed)
                        * order
                    )
                    mc_reference = 0.02 * order * (1.0 + 0.015 * seed)
                    rows.append(
                        {
                            "model": model,
                            "horizon": horizon,
                            "steps": steps,
                            "h": 0.0125,
                            "terminal_time": 0.05,
                            "sample_count": 1024,
                            "seed": seed,
                            "order": order,
                            "initialization_wp": initialization,
                            "discretization_wp": 0.1 * initialization,
                            "total_wp": 1.05 * initialization,
                            "mc_reference_wp": mc_reference,
                            "triangle_slack": 0.05 * initialization,
                            "mc_resolved": initialization > 2.0 * mc_reference,
                            "discretization_not_dominant": True,
                            "source_task_id": f"{model}-{horizon_index}-{seed}",
                            "source_arrays_sha256": "0" * 64,
                        }
                    )
    return rows


@pytest.mark.parametrize("orders", [(1.0, 1.4), (1.1, 1.5), (1.1, 2.0)])
def test_theorem_aligned_orders_reject_endpoints(
    orders: tuple[float, float],
) -> None:
    with pytest.raises(ConfigurationError, match=r"1 < p < alpha=1\.5"):
        _settings(orders=orders).validate(alpha=1.5)


def test_theorem_aligned_orders_accept_1p1_and_1p4() -> None:
    _settings().validate(alpha=1.5)


def test_sorted_wasserstein_translation_is_exact_for_both_orders() -> None:
    sample = np.sort(np.linspace(-3.0, 4.0, 4096, dtype=np.float64))
    shifted = sample + 0.375
    for order in (1.1, 1.4):
        actual = wasserstein_from_sorted(sample, shifted, order=order)
        expected = empirical_wasserstein_1d(sample, shifted, order=order)
        assert actual == expected
        assert actual == pytest.approx(0.375, abs=2.0e-15)


def test_horizon_selection_excludes_T2_refinement_meshes() -> None:
    config = load_config(Path("configs/final/exp2.toml"))
    selected = select_horizon_tasks(enumerate_tasks(config))
    at_two = {(task.model, task.steps) for task in selected if task.horizon == 2.0}
    assert at_two == {("stable", 156), ("vp", 156)}
    assert all("horizon_sweep" in task.purposes for task in selected)
    assert all(task.steps not in {312, 624} for task in selected)
    assert {
        round((task.horizon - config.experiment.epsilon) / task.steps, 14)
        for task in selected
    } == {0.0125}


def test_joint_seed_bootstrap_is_deterministic_and_paired_across_horizons() -> None:
    values = np.arange(60, dtype=np.float64).reshape(10, 6)
    first = joint_seed_bootstrap_intervals(
        values, confidence_level=0.9, replicates=2000, seed=712
    )
    second = joint_seed_bootstrap_intervals(
        values, confidence_level=0.9, replicates=2000, seed=712
    )
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    # Every column is the first column plus its fixed offset. Joint seed
    # resampling must preserve those offsets in both interval endpoints.
    assert np.allclose(first[0] - first[0][0], np.arange(6))
    assert np.allclose(first[1] - first[1][0], np.arange(6))


def test_summary_rejects_a_missing_seed_horizon_cell() -> None:
    rows = _synthetic_rows()
    rows.pop()
    with pytest.raises(ArtifactError, match="incomplete or duplicate"):
        summarize_initialization_rows(rows, _settings())


def test_two_panel_figure_has_no_filled_uncertainty_polygon_and_no_clipping() -> None:
    settings = _settings()
    summaries = summarize_initialization_rows(_synthetic_rows(), settings)
    figure = render_initialization_figure(summaries, settings)
    try:
        assert len(figure.axes) == 2
        assert not any(
            isinstance(collection, matplotlib.collections.PolyCollection)
            for axis in figure.axes
            for collection in axis.collections
        )
        legend_labels = {
            text.get_text() for legend in figure.legends for text in legend.get_texts()
        }
        assert "MC-resolution limited" not in legend_labels
        point_lines = [
            line
            for axis in figure.axes
            for line in axis.lines
            if line.get_marker() not in {"None", "none", "", None}
            and line.get_linestyle() in {"None", "none", "", None}
        ]
        assert point_lines
        assert all(
            matplotlib.colors.to_rgba(line.get_markerfacecolor())
            != matplotlib.colors.to_rgba("white")
            for line in point_lines
        )
        for axis, order in zip(figure.axes, settings.orders, strict=True):
            panel = [row for row in summaries if row["order"] == order]
            expected_limits = _log_axis_limits(panel)
            actual_limits = axis.get_ylim()
            assert actual_limits == pytest.approx(expected_limits)
            for row in panel:
                assert actual_limits[0] < row["initialization_ci_low"]
                assert actual_limits[1] > row["initialization_ci_high"]
                assert actual_limits[0] < row["mc_reference_median"] < actual_limits[1]
    finally:
        plt.close(figure)
