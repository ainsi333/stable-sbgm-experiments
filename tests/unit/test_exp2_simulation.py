from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import kv

from experiments.exp2.simulation import (
    aggregate_ei_primitives,
    checkpoint_indices,
    forward_coefficients,
    refinement_coupling_steps,
    score_sensitivity_from_checkpoints,
    stream_labels,
    vp_ei_coefficients,
)
from experiments.exp2.theory import sample_exact_forward_marginal, sample_student_t4
from levy_experiments.integrators import stable_ei_coefficients


def _student_t4_cf(frequencies: np.ndarray) -> np.ndarray:
    absolute = np.abs(frequencies)
    result = np.ones_like(absolute)
    nonzero = absolute > 0.0
    result[nonzero] = 2.0 * absolute[nonzero] ** 2 * kv(2.0, 2.0 * absolute[nonzero])
    return result


def test_student_t4_sampler_matches_known_scale() -> None:
    values = np.asarray(
        sample_student_t4(
            jax.random.PRNGKey(8),
            (100_000,),
            jnp.float64,
        )
    )
    assert np.all(np.isfinite(values))
    assert abs(np.median(values)) < 0.02
    assert abs(np.quantile(np.abs(values), 0.95) - 2.776445) < 0.05


def test_exact_forward_sampling_matches_analytic_cf() -> None:
    alpha = 1.5
    time = 0.05
    frequencies = np.asarray([0.25, 0.75, 1.5])
    for model in ("stable", "vp"):
        values = np.asarray(
            sample_exact_forward_marginal(
                jax.random.PRNGKey(91 if model == "stable" else 92),
                model,
                time,
                (120_000,),
                jnp.float64,
                alpha=alpha,
                beta=1.0,
            )
        )
        empirical = np.exp(1j * values[:, None] * frequencies[None, :]).mean(axis=0)
        a, gamma = forward_coefficients(model=model, time=time, alpha=alpha, beta=1.0)
        if model == "stable":
            reference = _student_t4_cf(a * frequencies) * np.exp(
                -(gamma**alpha) * np.abs(frequencies) ** alpha
            )
        else:
            reference = _student_t4_cf(a * frequencies) * np.exp(
                -0.5 * gamma**2 * frequencies**2
            )
        assert np.max(np.abs(empirical - reference)) < 0.009


def test_ei_factors_include_eta_beta_and_step() -> None:
    stable = stable_ei_coefficients(
        alpha=1.5, beta=1.2, step=0.075, drift_eta=0.5
    )
    assert np.isclose(stable.decay, np.exp(-0.5 * 1.2 * 0.075 / 1.5))
    assert np.isclose(stable.noise_scale**1.5, 1.0 - np.exp(-0.5 * 1.2 * 0.075))
    vp = vp_ei_coefficients(beta=1.2, step=0.075)
    assert np.isclose(vp.decay, np.exp(-1.2 * 0.075 / 2.0))
    assert np.isclose(vp.noise_scale**2, 1.0 - np.exp(-1.2 * 0.075))


def test_checkpoint_grid_and_refinement_random_streams_are_exactly_nested() -> None:
    assert checkpoint_indices((0.0, 0.25, 0.5, 0.75, 1.0), 120) == (0, 30, 60, 90, 120)
    config = SimpleNamespace(
        design=SimpleNamespace(
            refinement_horizon=2.0,
            refinement_steps=(156, 312, 624),
        )
    )
    coarse = SimpleNamespace(model="stable", horizon=2.0, steps=156)
    fine = SimpleNamespace(model="stable", horizon=2.0, steps=624)
    assert refinement_coupling_steps(config, coarse) == 624
    assert refinement_coupling_steps(config, fine) == 624
    assert stream_labels(coarse, coupling_steps=624) == stream_labels(
        fine, coupling_steps=624
    )

    horizon_only = SimpleNamespace(model="stable", horizon=1.0, steps=76)
    assert refinement_coupling_steps(config, horizon_only) == 76
    assert stream_labels(horizon_only, coupling_steps=76) != stream_labels(
        coarse, coupling_steps=624
    )


def test_nested_ei_primitive_aggregation_has_the_exact_coarse_scale() -> None:
    alpha = 1.5
    beta = 1.0
    eta = 0.5
    coarse_step = 0.0125
    ratio = 4
    fine_step = coarse_step / ratio
    stable_fine = stable_ei_coefficients(
        alpha=alpha, beta=beta, step=fine_step, drift_eta=eta
    )
    stable_coarse = stable_ei_coefficients(
        alpha=alpha, beta=beta, step=coarse_step, drift_eta=eta
    )
    exponents = np.arange(ratio - 1, -1, -1)
    stable_weights = stable_fine.noise_scale * stable_fine.decay**exponents
    assert np.isclose(
        np.sum(np.abs(stable_weights) ** alpha), stable_coarse.noise_scale**alpha
    )

    vp_fine = vp_ei_coefficients(beta=beta, step=fine_step)
    vp_coarse = vp_ei_coefficients(beta=beta, step=coarse_step)
    vp_weights = vp_fine.noise_scale * vp_fine.decay**exponents
    assert np.isclose(np.sum(vp_weights**2), vp_coarse.noise_scale**2)

    primitives = jnp.arange(ratio * 3, dtype=jnp.float64).reshape(ratio, 3)
    observed = np.asarray(
        aggregate_ei_primitives(
            primitives,
            decay=stable_fine.decay,
            scale=stable_fine.noise_scale,
        )
    )
    assert np.allclose(observed, stable_weights @ np.asarray(primitives))


def test_score_sensitivity_uses_wp_and_the_prespecified_half_floor_gate() -> None:
    main = np.asarray([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]])
    hires = main + np.asarray([[0.01], [0.20]])
    exact_a = np.asarray([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    exact_b = exact_a + 0.1

    sensitivity, controlled = score_sensitivity_from_checkpoints(
        main, hires, exact_a, exact_b, (1.0, 1.25)
    )

    assert sensitivity.shape == (2, 2)
    assert np.allclose(sensitivity[0], 0.01)
    assert np.allclose(sensitivity[1], 0.20)
    assert np.array_equal(controlled, np.asarray([[True, True], [False, False]]))
