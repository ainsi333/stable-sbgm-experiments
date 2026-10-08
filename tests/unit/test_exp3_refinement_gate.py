from __future__ import annotations

import dataclasses
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

import levy_experiments.experiment3 as experiment3_module
from levy_experiments.analysis_exp3 import (
    hybrid_rejection_diagnostic,
    refinement_convergence_diagnostic,
    stationary_truth_diagnostic,
)
from levy_experiments.config import (
    ExperimentTask,
    load_config,
    pilot_compatibility_hash,
    pilot_compatibility_payload,
    validate_config,
)
from levy_experiments.errors import ArtifactError, ConfigurationError
from levy_experiments.experiment3 import (
    _nonstationary_kernel,
    nested_ei_noise_weights,
    nonstationary_crn_stream_label,
)
from levy_experiments.integrators import stable_ei_coefficients
from levy_experiments.plotting import render_experiment3_figure
from levy_experiments.storage import run_layout

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("drift_eta,noise_eta", [(0.25, 0.25), (2.0, 2.0), (0.5, 1.0)])
@pytest.mark.parametrize("task_steps", [1760, 3520])
def test_nested_innovation_weights_reproduce_exact_coarse_ei_scale(
    drift_eta: float, noise_eta: float, task_steps: int
) -> None:
    alpha, beta, duration, finest_steps = 1.5, 1.0, 2.75, 3520
    weights = nested_ei_noise_weights(
        alpha=alpha,
        beta=beta,
        duration=duration,
        task_steps=task_steps,
        finest_steps=finest_steps,
        drift_eta=drift_eta,
        noise_eta=noise_eta,
    )
    coarse = stable_ei_coefficients(
        alpha=alpha,
        beta=beta,
        step=duration / task_steps,
        drift_eta=drift_eta,
        noise_eta=noise_eta,
    )

    assert len(weights) == finest_steps // task_steps
    assert math.isclose(
        sum(abs(weight) ** alpha for weight in weights),
        coarse.noise_scale**alpha,
        rel_tol=3.0e-14,
        abs_tol=2.0e-16,
    )


def test_refinement_stream_is_resolution_independent_and_kernel_uses_primitive_indices() -> None:
    config = load_config(ROOT / "configs/pilot/exp3.toml")
    assert nonstationary_crn_stream_label(config) == "nonstationary/nested-ei-fine-3520/batch"
    source = " ".join(inspect.getsource(_nonstationary_kernel).split())
    assert "first_primitive_index = step_index * refinement_ratio" in source
    assert "jax.random.fold_in(key_steps, primitive_index)" in source
    assert "jax.random.fold_in(key_steps, step_index)" not in source


def test_nested_kernels_reproduce_the_same_pure_ou_path_at_common_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    base = load_config(ROOT / "configs/smoke/exp3.toml")
    nonstationary = dataclasses.replace(
        base.nonstationary,
        steps=(11, 22),
        particles=16,
        batch_size=16,
    )
    config = dataclasses.replace(base, nonstationary=nonstationary)

    def zero_score(x, *_args, **_kwargs):
        return jnp.zeros_like(x)

    def zero_remainder(x, *_args, **_kwargs):
        return jnp.zeros_like(x)

    def deterministic_primitive(key, _alpha, shape, dtype):
        return jax.random.normal(key, shape=shape, dtype=dtype)

    monkeypatch.setattr(experiment3_module, "discrete_fractional_score", zero_score)
    monkeypatch.setattr(experiment3_module, "popov_remainder", zero_remainder)
    monkeypatch.setattr(
        experiment3_module, "sample_symmetric_stable_1d", deterministic_primitive
    )
    duration = config.experiment.horizon - config.experiment.epsilon
    coarse = ExperimentTask(0, "three_atoms", "popov_ei", duration, 0.5, 0.5, 11, 7)
    fine = dataclasses.replace(coarse, index=1, steps=22)
    key = jax.random.PRNGKey(123)

    coarse_path = _nonstationary_kernel(config, coarse, object(), jnp.float64)(key)
    fine_path = _nonstationary_kernel(config, fine, object(), jnp.float64)(key)

    assert np.allclose(np.asarray(coarse_path), np.asarray(fine_path), rtol=0.0, atol=2e-14)


def _diagnostic_rows(config, *, failed_cell: tuple[float, float] | None = None):
    threshold = config.analysis.refinement_tolerance_multiplier / math.sqrt(
        config.nonstationary.particles
    )
    rows = []
    for eta in config.experiment.etas:
        for seed in config.experiment.seeds:
            for forward_time in (
                *config.nonstationary.checkpoints_forward,
                config.experiment.epsilon,
            ):
                failed = failed_cell == (eta, forward_time)
                rows.append(
                    {
                        "coarse_steps": 1760,
                        "fine_steps": 3520,
                        "seed": seed,
                        "eta": eta,
                        "forward_time": forward_time,
                        "metric": "paired_marginal_ecf_max_component_change",
                        "value": threshold * (1.1 if failed else 0.9),
                        "threshold": threshold,
                        "passed": not failed,
                    }
                )
    return rows


def test_refinement_gate_passes_all_cells_and_fails_closed_on_one_cell() -> None:
    config = load_config(ROOT / "configs/pilot/exp3.toml")
    passing = refinement_convergence_diagnostic(config, _diagnostic_rows(config))
    assert passing["pass"] is True
    assert passing["status"] == "pass"
    assert passing["expected_cell_count"] == 16

    failed_cell = (config.experiment.etas[-1], config.experiment.epsilon)
    failing = refinement_convergence_diagnostic(
        config, _diagnostic_rows(config, failed_cell=failed_cell)
    )
    assert failing["pass"] is False
    assert failing["status"] == "fail"
    assert failing["failed_cell_count"] == 1


def test_refinement_gate_rejects_missing_seed_and_single_level() -> None:
    config = load_config(ROOT / "configs/pilot/exp3.toml")
    incomplete = _diagnostic_rows(config)[1:]
    diagnostic = refinement_convergence_diagnostic(config, incomplete)
    assert diagnostic["pass"] is False
    assert diagnostic["missing_cells"]

    smoke = load_config(ROOT / "configs/smoke/exp3.toml")
    smoke_diagnostic = refinement_convergence_diagnostic(smoke, [])
    assert smoke_diagnostic["pass"] is False
    assert smoke_diagnostic["status"] == "not_evaluable"


def test_non_nested_refinement_config_is_rejected() -> None:
    config = load_config(ROOT / "configs/pilot/exp3.toml")
    mutated = dataclasses.replace(config.nonstationary, steps=(220, 440, 660))
    with pytest.raises(ConfigurationError, match=r"divide the finest|factor-two"):
        validate_config(dataclasses.replace(config, nonstationary=mutated))


def test_publication_rendering_is_fail_closed(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/final/exp3.toml")
    diagnostics = {
        "publication_gate_failures": ["synthetic failed convergence gate"],
        "publication_gate_pass": False,
    }
    with pytest.raises(ArtifactError, match="publication figure rendering refused"):
        render_experiment3_figure(
            config,
            run_layout(tmp_path, config),
            [],
            diagnostics,
            resume=False,
        )


def _stationary_rows(config, *, component_error: float = 0.005, collapse_joint=False):
    rows = []
    low_eta, high_eta = min(config.experiment.etas), max(config.experiment.etas)
    component_metrics = (
        "marginal_ecf_max_component_error_initial",
        "marginal_ecf_max_component_error_terminal",
        "joint_ecf_max_component_error_to_analytic_popov",
        "joint_probe_abs_error_to_analytic_popov",
    )
    for lag in config.stationary.lags:
        for eta in config.experiment.etas:
            analytic = 0.1 + 0.04 * (eta - low_eta) / (high_eta - low_eta)
            empirical = 0.1 if collapse_joint else analytic
            for seed in config.experiment.seeds:
                for metric in component_metrics:
                    rows.append(
                        {
                            "section": "stationary",
                            "metric": metric,
                            "value": component_error,
                            "seed": seed,
                            "eta": eta,
                            "lag": lag,
                        }
                    )
                rows.append(
                    {
                        "section": "stationary",
                        "metric": "joint_probe_real",
                        "value": empirical,
                        "analytic_value": analytic,
                        "floor_value": analytic - 0.03,
                        "seed": seed,
                        "eta": eta,
                        "lag": lag,
                    }
                )
    return rows


def test_stationary_truth_gate_requires_analytic_accuracy_and_joint_separation() -> None:
    config = load_config(ROOT / "configs/pilot/exp3.toml")
    passing = stationary_truth_diagnostic(config, _stationary_rows(config))
    assert passing["pass"] is True
    assert passing["exact_cf_component_pass"] is True
    assert passing["joint_eta_separation_pass"] is True

    inaccurate = stationary_truth_diagnostic(
        config, _stationary_rows(config, component_error=0.0061)
    )
    assert inaccurate["pass"] is False
    collapsed = stationary_truth_diagnostic(
        config, _stationary_rows(config, collapse_joint=True)
    )
    assert collapsed["pass"] is False
    assert collapsed["joint_eta_separation_pass"] is False


def test_hybrid_gate_needs_seed_robust_simultaneous_rejection() -> None:
    config = load_config(ROOT / "configs/pilot/exp3.toml")
    template = hybrid_rejection_diagnostic(config, [])
    radius = float(template["simultaneous_radius"])
    times = (*config.nonstationary.checkpoints_forward, config.experiment.epsilon)

    def rows(rejected_seeds: set[int]):
        return [
            {
                "seed": seed,
                "forward_time": time,
                "value": radius * (1.1 if seed in rejected_seeds else 0.9),
            }
            for seed in config.experiment.seeds
            for time in times
        ]

    passing = hybrid_rejection_diagnostic(config, rows(set(config.experiment.seeds[:3])))
    assert passing["pass"] is True
    assert passing["required_rejected_seed_count"] == 3

    one_outlier = hybrid_rejection_diagnostic(config, rows({config.experiment.seeds[0]}))
    assert one_outlier["pass"] is False
    assert one_outlier["rejected_seed_count"] == 1

    incomplete = hybrid_rejection_diagnostic(config, rows(set(config.experiment.seeds))[:-1])
    assert incomplete["pass"] is False
    assert incomplete["complete"] is False


def test_pilot_and_final_share_one_scientific_compatibility_signature() -> None:
    pilot = load_config(ROOT / "configs/pilot/exp3.toml")
    final = load_config(ROOT / "configs/final/exp3.toml")
    assert pilot_compatibility_payload(pilot) == pilot_compatibility_payload(final)
    assert pilot_compatibility_hash(pilot) == pilot_compatibility_hash(final)

    mutated_analysis = dataclasses.replace(
        pilot.analysis, stationary_joint_separation_min=0.021
    )
    mutated = dataclasses.replace(pilot, analysis=mutated_analysis)
    assert pilot_compatibility_hash(mutated) != pilot_compatibility_hash(final)
