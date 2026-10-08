from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

import experiments.exp1.analysis as analysis_module
from experiments.exp1.aggregate import _analysis_settings
from experiments.exp1.analysis import (
    AnalysisSettings,
    Exp1ArtifactError,
    _plateau_summary,
    analyze_tasks,
    diagnostics,
    load_task_npz,
    load_tasks,
    make_decisions,
    pool_mean_excess_rows,
    summarize_m_slopes,
    summarize_tail_metrics,
)
from experiments.exp1.config import load_config
from experiments.exp1.plotting import plot_diagnostics, plot_main, plot_quantile_ratio
from levy_experiments.metrics.wasserstein import empirical_wasserstein_1d


def _signed_pareto(seed: int, size: int, alpha: float = 1.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    radius = rng.pareto(alpha, size) + 1.0
    return radius * rng.choice(np.asarray([-1.0, 1.0]), size=size)


def _write_task(
    path: Path,
    *,
    model: str,
    seed: int,
    control: str = "",
    dynamic: bool = False,
    permutation: np.ndarray | None = None,
) -> Path:
    sample = _signed_pareto(seed, 4096)
    if permutation is not None:
        sample = sample[permutation]
    arrays: dict[str, np.ndarray] = {
        "terminal_sample": sample,
        "task_kind": np.asarray("dynamic" if dynamic else "control"),
        "model": np.asarray(model),
        "control_distribution": np.asarray(control),
        "target": np.asarray("student_t" if dynamic else "direct_control"),
        "target_nu": np.asarray(1.5 if dynamic else 0.0),
        "alpha": np.asarray(1.5),
        "eta": np.asarray(0.5 if model == "stable" else 1.0),
        "beta": np.asarray(1.0),
        "T": np.asarray(2.0),
        "epsilon": np.asarray(0.5),
        "steps": np.asarray(8 if dynamic else 0),
        "seed": np.asarray(seed),
        "reference_scale": np.asarray(1.0),
        "tail_constant_reference": np.asarray(1.0),
    }
    if dynamic:
        rng = np.random.default_rng(seed + 10_000)
        exact = rng.standard_t(1.5, size=(3, 512))
        arrays.update(
            {
                "control_forward_times": np.asarray([0.5, 1.25, 2.0]),
                "control_numerical": exact.copy(),
                "control_exact_a": exact.copy(),
                "control_exact_b": rng.standard_t(1.5, size=(3, 512)),
                "control_exact_c": rng.standard_t(1.5, size=(3, 512)),
                "score_hires_terminal": sample.copy(),
                "score_sensitivity_forward_times": np.asarray([0.5, 1.25, 2.0]),
                "score_sensitivity_mean_abs": np.zeros(3),
            }
        )
    np.savez_compressed(path, **arrays)
    return path


def _analyze(paths: list[Path]):
    settings = AnalysisSettings(
        tier="smoke",
        bootstrap_replicates=0,
        subsample_fractions=(0.5,),
        block_count=2,
        minimum_exceedances_per_seed=5,
        minimum_pooled_exceedances=10,
        minimum_contributing_seeds=1,
    )
    tasks = load_tasks(paths)
    metrics, mean_rows, slopes = analyze_tasks(tasks, settings)
    summaries = summarize_tail_metrics(metrics, settings)
    pooled = pool_mean_excess_rows(mean_rows, settings)
    slope_summaries = summarize_m_slopes(slopes, settings)
    decisions = make_decisions(tasks, metrics, summaries, pooled, slope_summaries, settings)
    return (
        settings,
        tasks,
        metrics,
        mean_rows,
        slopes,
        summaries,
        pooled,
        slope_summaries,
        decisions,
    )


def test_loader_requires_all_three_independent_references(tmp_path: Path) -> None:
    path = _write_task(tmp_path / "dynamic.npz", model="stable", seed=1, dynamic=True)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name]) for name in archive.files if name != "control_exact_c"
        }
    np.savez_compressed(path, **arrays)
    with pytest.raises(Exp1ArtifactError, match=r"missing=.*control_exact_c"):
        load_task_npz(path)


def test_full_tail_metrics_are_permutation_invariant_and_smoke_never_concludes(
    tmp_path: Path,
) -> None:
    base = _signed_pareto(5, 4096)
    permutation = np.random.default_rng(8).permutation(base.size)
    first = _write_task(tmp_path / "first.npz", model="sas", seed=5, control="sas")
    second = _write_task(
        tmp_path / "second.npz",
        model="sas",
        seed=5,
        control="sas",
        permutation=permutation,
    )
    settings = AnalysisSettings(tier="smoke", bootstrap_replicates=0)
    first_rows = analyze_tasks([load_task_npz(first)], settings)[0]
    second_rows = analyze_tasks([load_task_npz(second)], settings)[0]

    def signature(rows):
        return sorted(
            (row["metric"], row["fraction"], row["value"])
            for row in rows
            if row["sample_kind"] == "full" and row["valid"]
        )

    assert signature(first_rows) == signature(second_rows)

    paths = [
        _write_task(tmp_path / "stable.npz", model="stable", seed=0, dynamic=True),
        _write_task(tmp_path / "sas.npz", model="sas", seed=0, control="sas"),
        _write_task(tmp_path / "pareto.npz", model="pareto", seed=0, control="pareto"),
        _write_task(tmp_path / "gaussian.npz", model="gaussian", seed=0, control="gaussian"),
    ]
    result = _analyze(paths)
    decisions = result[-1]
    assert decisions
    assert {row["decision"] for row in decisions} == {"nonconclusive"}
    assert any("smoke tier" in row["reason"] for row in decisions)


def test_plateau_diagnostic_detects_a_monotone_threshold_drift() -> None:
    key = ("dynamic", "stable", "", "student_t", 1.5, 1.5, 0.5, 2.0, 0.5, 80, "primary")
    rows = []
    for seed in range(8):
        for fraction, value in ((0.004, 1.2), (0.006, 1.5), (0.008, 1.8)):
            rows.append(
                {
                    "task_kind": key[0],
                    "model": key[1],
                    "control_distribution": key[2],
                    "target": key[3],
                    "target_nu": key[4],
                    "alpha_theory": key[5],
                    "eta": key[6],
                    "horizon": key[7],
                    "epsilon": key[8],
                    "steps": key[9],
                    "seed": seed,
                    "sample_kind": "full",
                    "sample_size": 262144,
                    "replicate": -1,
                    "metric": "hill_xi",
                    "fraction": fraction,
                    "value": value + seed * 1.0e-4,
                    "valid": True,
                }
            )
    plateau = _plateau_summary(
        rows,
        key,
        AnalysisSettings(tier="final", bootstrap_replicates=200),
    )
    assert plateau["boot90_low"] > 0.03


def test_seed_summary_never_groups_by_random_order_statistic() -> None:
    settings = AnalysisSettings(tier="smoke", bootstrap_replicates=0)
    rows = []
    for seed, threshold, value in ((0, 8.0, 1.4), (1, 12.0, 1.6)):
        rows.append(
            {
                "task_kind": "dynamic",
                "model": "stable",
                "control_distribution": "",
                "target": "student_t",
                "target_nu": 1.5,
                "alpha_theory": 1.5,
                "eta": 0.5,
                "horizon": 2.0,
                "epsilon": 0.5,
                "steps": 8,
                "sample_kind": "full",
                "sample_size": 4096,
                "metric": "hill_alpha",
                "fraction": 0.006,
                "k": 24,
                "threshold": threshold,
                "beta": math.nan,
                "expected": 1.5,
                "seed": seed,
                "value": value,
                "valid": True,
                "controlled": False,
            }
        )
    summary = summarize_tail_metrics(rows, settings)
    assert len(summary) == 1
    assert summary[0]["n_seeds"] == 2
    assert summary[0]["threshold"] == 10.0
    assert summary[0]["median"] == 1.5


def test_marginal_control_uses_prespecified_gaussian_ecf_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_task(tmp_path / "dynamic.npz", model="stable", seed=3, dynamic=True)
    captured: list[np.ndarray] = []
    original = analysis_module.weighted_cf_rmse

    def record_weights(empirical, reference, weights):
        captured.append(np.asarray(weights, dtype=np.float64))
        return original(empirical, reference, weights)

    monkeypatch.setattr(analysis_module, "weighted_cf_rmse", record_weights)
    settings = AnalysisSettings(tier="smoke", bootstrap_replicates=0)
    analyze_tasks([load_task_npz(path)], settings)
    frequencies = np.linspace(
        -settings.ecf_frequency_max, settings.ecf_frequency_max, settings.ecf_frequency_count
    )
    expected = np.exp(-0.5 * frequencies**2)
    assert captured
    assert all(np.array_equal(weights, expected) for weights in captured)


def test_marginal_control_sorts_each_checkpoint_sample_once_without_changing_w1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_task(tmp_path / "dynamic.npz", model="stable", seed=7, dynamic=True)
    task = load_task_npz(path)
    calls = 0
    original_sort = analysis_module._sort_w1_sample

    def counted_sort(sample):
        nonlocal calls
        calls += 1
        return original_sort(sample)

    monkeypatch.setattr(analysis_module, "_sort_w1_sample", counted_sort)
    settings = AnalysisSettings(tier="smoke", bootstrap_replicates=0)
    rows = analysis_module._analyze_dynamic_control(task, settings)

    assert task.control_forward_times is not None
    assert task.control_numerical is not None
    assert task.control_exact_a is not None
    assert task.control_exact_b is not None
    assert task.control_exact_c is not None
    assert calls == 4 * task.control_forward_times.size + 5
    for index, forward_time in enumerate(task.control_forward_times):
        numerical = task.control_numerical[index]
        exact = (
            task.control_exact_a[index],
            task.control_exact_b[index],
            task.control_exact_c[index],
        )
        expected = float(
            np.median([empirical_wasserstein_1d(numerical, sample) for sample in exact])
        )
        expected_floor = max(
            empirical_wasserstein_1d(first, second)
            for first, second in ((exact[0], exact[1]), (exact[0], exact[2]), (exact[1], exact[2]))
        )
        row = next(
            item
            for item in rows
            if item["metric"] == "control_w1" and item["threshold"] == float(forward_time)
        )
        assert row["value"] == expected
        assert row["auxiliary"] == expected_floor


def test_figure_generation_is_byte_deterministic(tmp_path: Path) -> None:
    paths = [
        _write_task(tmp_path / "stable.npz", model="stable", seed=0, dynamic=True),
        _write_task(tmp_path / "sas.npz", model="sas", seed=0, control="sas"),
        _write_task(tmp_path / "pareto.npz", model="pareto", seed=0, control="pareto"),
        _write_task(tmp_path / "gaussian.npz", model="gaussian", seed=0, control="gaussian"),
    ]
    result = _analyze(paths)
    settings, tasks, metrics, mean_rows, slopes, summaries, pooled, slope_summary, decisions = (
        result
    )
    diagnostic = diagnostics(tasks, metrics, mean_rows, slopes, decisions, settings)
    assert diagnostic["decision_counts"]["compatible"] == 0
    first = tmp_path / "figures_a"
    second = tmp_path / "figures_b"
    outputs_a = (
        *plot_main(summaries, first, tier="smoke"),
        *plot_quantile_ratio(summaries, first, tier="smoke"),
        *plot_diagnostics(summaries, pooled, slope_summary, metrics, first, tier="smoke"),
    )
    outputs_b = (
        *plot_main(summaries, second, tier="smoke"),
        *plot_quantile_ratio(summaries, second, tier="smoke"),
        *plot_diagnostics(summaries, pooled, slope_summary, metrics, second, tier="smoke"),
    )
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs_a] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs_b
    ]


def test_diagnostics_render_when_mean_excess_is_not_estimable(tmp_path: Path) -> None:
    invalid_pooled_row = {
        "model": "stable",
        "target_nu": 1.5,
        "sample_kind": "full",
        "steps": 8,
        "purpose": "primary",
        "threshold_fraction": 0.006,
        "pooled_ratio": float("nan"),
        "valid": False,
    }
    outputs = plot_diagnostics(
        [], [invalid_pooled_row], [], [], tmp_path, tier="smoke", primary_steps=8
    )

    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)


def test_configured_mass_ratio_windows_reach_the_analysis(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "final" / "exp1.toml", allow_final=True)
    settings = _analysis_settings(config)

    assert settings.m_windows == {
        "primary": config.analysis.fit_primary,
        "broad": config.analysis.fit_sensitivities[0],
        "extreme": config.analysis.fit_sensitivities[1],
    }


def test_primary_and_refinement_purposes_survive_loading_and_analysis(tmp_path: Path) -> None:
    primary = _write_task(tmp_path / "primary.npz", model="stable", seed=1, dynamic=True)
    refinement = _write_task(tmp_path / "refinement.npz", model="stable", seed=2, dynamic=True)
    for path, purpose in ((primary, "primary"), (refinement, "refinement")):
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        arrays["purpose"] = np.asarray(purpose)
        np.savez_compressed(path, **arrays)

    settings = AnalysisSettings(tier="smoke", bootstrap_replicates=0)
    tasks = load_tasks([primary, refinement])
    rows = analyze_tasks(tasks, settings)[0]

    assert {task.purpose for task in tasks} == {"primary", "refinement"}
    assert {row["purpose"] for row in rows} == {"primary", "refinement"}


def test_main_figure_selector_uses_primary_not_largest_refinement_steps() -> None:
    from experiments.exp1.plotting import _maximum_steps

    rows = [
        {"model": "stable", "target_nu": 1.5, "steps": 80, "purpose": "primary"},
        {"model": "stable", "target_nu": 1.5, "steps": 160, "purpose": "refinement"},
    ]
    assert _maximum_steps(rows, "stable", 1.5, purpose="primary") == 80


def test_diagnostic_selectors_keep_primary_and_refinement_branches_separate() -> None:
    from experiments.exp1.plotting import (
        _primary_control_rows,
        _primary_slope_candidate,
        _refinement_hill_series,
    )

    common = {"model": "stable", "target_nu": 1.5, "sample_kind": "full"}
    slope_rows = [
        common | {"window": "primary", "purpose": "primary", "steps": 80, "median": 0.0},
        common
        | {"window": "primary", "purpose": "refinement", "steps": 80, "median": 0.1},
        common
        | {"window": "primary", "purpose": "refinement", "steps": 160, "median": 0.2},
    ]
    selected_slope = _primary_slope_candidate(slope_rows, 1.5, 80)
    assert selected_slope is not None
    assert selected_slope["purpose"] == "primary"
    assert selected_slope["median"] == 0.0

    control_rows = [
        common
        | {
            "metric": "control_w1",
            "purpose": purpose,
            "steps": steps,
            "auxiliary": 1.0,
        }
        for purpose, steps in (("primary", 80), ("refinement", 80), ("refinement", 160))
    ]
    selected_controls = _primary_control_rows(control_rows, 80)
    assert [(row["purpose"], row["steps"]) for row in selected_controls] == [("primary", 80)]

    hill_rows = [
        common
        | {
            "metric": "hill_alpha",
            "purpose": purpose,
            "steps": steps,
            "fraction": 0.006,
            "median": 1.5,
        }
        for purpose, steps in (
            ("primary", 80),
            ("refinement", 40),
            ("refinement", 80),
            ("refinement", 160),
        )
    ]
    selected_refinement = _refinement_hill_series(hill_rows, 1.5)
    assert [row["steps"] for row in selected_refinement] == [40, 80, 160]
    assert all(row["purpose"] == "refinement" for row in selected_refinement)


def test_tail_index_incompatibility_respects_alpha_xi_inverse_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = {
        "task_kind": "dynamic",
        "model": "stable",
        "control_distribution": "",
        "target": "student_t",
        "target_nu": 1.5,
        "alpha_theory": 1.5,
        "eta": 0.5,
        "horizon": 2.0,
        "epsilon": 0.5,
        "steps": 80,
        "purpose": "primary",
        "sample_kind": "full",
        "sample_size": 262_144,
        "fraction": 0.006,
        "k": 1_572,
        "threshold": 1.0,
        "beta": math.nan,
        "expected": math.nan,
        "n_seeds": 12,
        "q16": math.nan,
        "median": math.nan,
        "q84": math.nan,
        "boot90_low": math.nan,
        "boot90_high": math.nan,
        "boot95_low": math.nan,
        "boot95_high": math.nan,
        "controlled_fraction": 1.0,
    }

    def summary(metric: str, low: float, high: float, median: float) -> dict[str, object]:
        return base | {
            "metric": metric,
            "median": median,
            "boot90_low": low,
            "boot90_high": high,
            "boot95_low": low,
            "boot95_high": high,
        }

    summaries = [
        summary("hill_alpha", 1.9, 2.1, 2.0),
        summary("pickands_xi", 0.35, 0.45, 0.4),
        summary("gpd_xi", 0.35, 0.45, 0.4),
    ]
    monkeypatch.setattr(analysis_module, "_control_gate", lambda *args: (True, []))
    monkeypatch.setattr(
        analysis_module,
        "_plateau_summary",
        lambda *args: {
            "median": 0.0,
            "boot90_low": -0.01,
            "boot90_high": 0.01,
            "boot95_low": -0.02,
            "boot95_high": 0.02,
        },
    )

    decisions = make_decisions(
        [],
        [],
        summaries,
        [],
        [],
        AnalysisSettings(tier="pilot", bootstrap_replicates=0),
    )
    tail = next(row for row in decisions if row["claim"] == "stable_tail_index")
    assert tail["decision"] == "incompatible"


def test_gaussian_negative_control_is_a_fail_closed_gate(tmp_path: Path) -> None:
    tasks = load_tasks(
        [
            _write_task(tmp_path / "dynamic.npz", model="stable", seed=0, dynamic=True),
            _write_task(tmp_path / "sas.npz", model="sas", seed=0, control="sas"),
            _write_task(tmp_path / "pareto.npz", model="pareto", seed=0, control="pareto"),
            _write_task(
                tmp_path / "gaussian.npz", model="gaussian", seed=0, control="gaussian"
            ),
        ]
    )
    common = {
        "purpose": "primary",
        "model": "stable",
        "target_nu": 1.5,
        "steps": 8,
        "sample_kind": "full",
        "n_seeds": 1,
        "controlled_fraction": 1.0,
    }
    summaries = [
        *[
            common | {"metric": metric, "threshold": checkpoint}
            for checkpoint in (0.5, 1.25, 2.0)
            for metric in (
                "pt_ecf_sup_numerical",
                "pt_ecf_sup_exact_a",
                "pt_ecf_sup_exact_b",
                "pt_ecf_sup_exact_c",
            )
        ],
        *[
            common | {"metric": metric}
            for metric in (
                "score_table_w1",
                "score_table_ecf_rmse",
                "score_table_hill_alpha_absdiff",
                "score_table_tail_constant_reldiff",
            )
        ],
        *[
            {
                "control_distribution": distribution,
                "sample_kind": "full",
                "metric": "hill_alpha",
                "fraction": 0.006,
                "n_seeds": 1,
                "boot90_low": 1.45,
                "boot90_high": 1.55,
            }
            for distribution in ("pareto", "sas")
        ],
        {
            "control_distribution": "gaussian",
            "sample_kind": "full",
            "metric": "hill_alpha",
            "fraction": 0.006,
            "boot95_low": 1.70,
            "n_seeds": 1,
        },
    ]
    settings = AnalysisSettings(tier="pilot", bootstrap_replicates=200)
    passed, failures = analysis_module._control_gate(tasks, summaries, settings)
    assert not passed
    assert any("Gaussian negative control" in failure for failure in failures)

    summaries[-1]["boot95_low"] = 1.76
    passed, failures = analysis_module._control_gate(tasks, summaries, settings)
    assert passed
    assert not failures


def test_control_gate_checks_every_primary_cell_not_only_global_metric_names(
    tmp_path: Path,
) -> None:
    first = _write_task(tmp_path / "first.npz", model="stable", seed=0, dynamic=True)
    second = _write_task(tmp_path / "second.npz", model="stable", seed=1, dynamic=True)
    with np.load(second, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["target_nu"] = np.asarray(3.0)
    np.savez_compressed(second, **arrays)
    tasks = load_tasks([first, second])
    summaries = []
    for metric in (
        "pt_ecf_sup_numerical",
        "pt_ecf_sup_exact_a",
        "pt_ecf_sup_exact_b",
        "pt_ecf_sup_exact_c",
    ):
        for checkpoint in (0.5, 1.25, 2.0):
            summaries.append(
                {
                    "purpose": "primary",
                    "model": "stable",
                    "target_nu": 1.5,
                    "steps": 8,
                    "sample_kind": "full",
                    "metric": metric,
                    "threshold": checkpoint,
                    "n_seeds": 1,
                    "controlled_fraction": 1.0,
                }
            )
    for metric in (
        "score_table_w1",
        "score_table_ecf_rmse",
        "score_table_hill_alpha_absdiff",
        "score_table_tail_constant_reldiff",
    ):
        summaries.append(
            {
                "purpose": "primary",
                "model": "stable",
                "target_nu": 1.5,
                "steps": 8,
                "sample_kind": "full",
                "metric": metric,
                "n_seeds": 1,
                "controlled_fraction": 1.0,
            }
        )
    passed, failures = analysis_module._control_gate(
        tasks, summaries, AnalysisSettings(tier="smoke", bootstrap_replicates=0)
    )
    assert not passed
    assert any("nu=3" in failure and "missing" in failure for failure in failures)


def test_paired_refinement_gate_detects_a_numerical_shift(tmp_path: Path) -> None:
    paths: list[Path] = []
    for seed in (0, 1):
        for steps in (4, 8):
            path = _write_task(
                tmp_path / f"seed{seed}-N{steps}.npz",
                model="stable",
                seed=seed,
                dynamic=True,
            )
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
            arrays["steps"] = np.asarray(steps)
            arrays["purpose"] = np.asarray("refinement")
            np.savez_compressed(path, **arrays)
            paths.append(path)
    settings = AnalysisSettings(tier="smoke", bootstrap_replicates=0)
    tasks = load_tasks(paths)
    passed, failures, details = analysis_module._refinement_gate(tasks, settings)
    assert passed
    assert not failures
    assert details and all(row["passed"] for row in details)

    shifted = paths[1]
    with np.load(shifted, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["terminal_sample"] = 2.0 * arrays["terminal_sample"]
    np.savez_compressed(shifted, **arrays)
    passed, failures, details = analysis_module._refinement_gate(load_tasks(paths), settings)
    assert not passed
    assert failures
    assert any(not row["passed"] for row in details)


def test_ecf_simultaneous_band_tightens_with_sample_size() -> None:
    small = analysis_module._ecf_simultaneous_radius(
        1_000, 81, familywise_alpha=0.01, familywise_test_count=100
    )
    large = analysis_module._ecf_simultaneous_radius(
        4_000, 81, familywise_alpha=0.01, familywise_test_count=100
    )
    assert large == pytest.approx(small / 2.0)
