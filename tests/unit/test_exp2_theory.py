from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from experiments.exp2.theory import (
    forward_coefficients,
    forward_marginal_cf,
    sample_exact_forward_marginal,
    sample_stationary_reference,
    stable_tail_score,
    student_t4_cf,
    student_t4_density,
    vp_tail_score,
)


def _empirical_cf(samples: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    return np.mean(np.exp(1j * samples[:, None] * frequencies[None, :]), axis=0)


def test_student_t4_density_and_characteristic_function_normalizations() -> None:
    mass = quad(student_t4_density, -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12)[0]
    variance = quad(
        lambda value: value * value * student_t4_density(value),
        -np.inf,
        np.inf,
        epsabs=1e-11,
        epsrel=1e-11,
    )[0]
    assert math.isclose(mass, 1.0, rel_tol=0.0, abs_tol=2e-13)
    assert math.isclose(variance, 2.0, rel_tol=0.0, abs_tol=2e-11)
    assert student_t4_density(0.0) == 3.0 / 8.0
    assert student_t4_cf(0.0) == 1.0

    for frequency in (0.2, 0.7, 1.5):
        numerical = (
            2.0
            * quad(
                lambda value, probe=frequency: student_t4_density(value) * math.cos(probe * value),
                0.0,
                np.inf,
                epsabs=2e-12,
                epsrel=2e-12,
                limit=500,
            )[0]
        )
        assert math.isclose(student_t4_cf(frequency), numerical, rel_tol=2e-10, abs_tol=2e-11)
    small = 1.0e-6
    assert math.isclose(student_t4_cf(small), 1.0 - small * small, abs_tol=1e-15)


def test_forward_coefficients_and_characteristic_functions_use_manuscript_conventions() -> None:
    alpha, beta, time = 1.5, 1.3, 0.7
    stable_a, stable_gamma = forward_coefficients("stable", time, alpha=alpha, beta=beta)
    vp_a, vp_gamma = forward_coefficients("vp", time, alpha=alpha, beta=beta)
    innovation_power = -math.expm1(-beta * time)
    assert math.isclose(stable_a, math.exp(-beta * time / alpha), rel_tol=1e-15)
    assert math.isclose(stable_gamma**alpha, innovation_power, rel_tol=2e-15)
    assert math.isclose(vp_a, math.exp(-beta * time / 2.0), rel_tol=1e-15)
    assert math.isclose(vp_gamma**2, innovation_power, rel_tol=2e-15)

    frequencies = np.asarray([0.0, 0.3, 1.2])
    stable_expected = student_t4_cf(stable_a * frequencies) * np.exp(
        -innovation_power * np.abs(frequencies) ** alpha
    )
    vp_expected = student_t4_cf(vp_a * frequencies) * np.exp(
        -0.5 * innovation_power * frequencies**2
    )
    assert np.array_equal(
        forward_marginal_cf(frequencies, time, "stable", alpha=alpha, beta=beta),
        stable_expected,
    )
    assert np.array_equal(
        forward_marginal_cf(frequencies, time, "vp", alpha=alpha, beta=beta), vp_expected
    )


def test_exact_forward_and_stationary_samplers_match_their_characteristic_functions() -> None:
    alpha, beta, time = 1.5, 1.0, 0.6
    frequencies = np.asarray([0.25, 0.75, 1.5])
    sample_count = 80_000
    for index, model in enumerate(("stable", "vp")):
        key = jax.random.PRNGKey(1200 + index)
        samples = np.asarray(
            sample_exact_forward_marginal(
                key,
                model,
                time,
                (sample_count,),
                jnp.float64,
                alpha=alpha,
                beta=beta,
            )
        )
        repeated = np.asarray(
            sample_exact_forward_marginal(
                key,
                model,
                time,
                (sample_count,),
                jnp.float64,
                alpha=alpha,
                beta=beta,
            )
        )
        assert np.array_equal(samples, repeated)
        empirical = _empirical_cf(samples, frequencies)
        expected = forward_marginal_cf(frequencies, time, model, alpha=alpha, beta=beta)
        assert np.max(np.abs(empirical - expected)) < 9.0e-3

        reference = np.asarray(
            sample_stationary_reference(
                jax.random.PRNGKey(2200 + index),
                model,
                (sample_count,),
                jnp.float64,
                alpha=alpha,
            )
        )
        empirical_reference = _empirical_cf(reference, frequencies)
        if model == "stable":
            expected_reference = np.exp(-(np.abs(frequencies) ** alpha))
        else:
            expected_reference = np.exp(-0.5 * frequencies**2)
        assert np.max(np.abs(empirical_reference - expected_reference)) < 9.0e-3


def test_explicit_tail_scores_are_odd_and_have_the_correct_leading_terms() -> None:
    time = 0.4
    points = np.asarray([1000.0, 2000.0])
    stable = stable_tail_score(points, time, alpha=1.5, beta=1.0)
    stable_negative = stable_tail_score(-points, time, alpha=1.5, beta=1.0)
    vp = vp_tail_score(points, time, beta=1.0)
    vp_negative = vp_tail_score(-points, time, beta=1.0)
    assert np.array_equal(stable_negative, -stable)
    assert np.array_equal(vp_negative, -vp)

    innovation_power = -math.expm1(-time)
    assert np.max(np.abs(stable / (-points / (1.5 * innovation_power)) - 1.0)) < 1e-5
    assert np.max(np.abs(vp / (-5.0 / points) - 1.0)) < 1e-4
