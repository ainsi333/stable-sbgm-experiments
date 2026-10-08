from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.stats import t as student_t_distribution

from experiments.exp1.theory import (
    continuous_stable_scale_power,
    discrete_stable_scale_power,
    forward_coefficients,
    forward_marginal_cf,
    stable_density_tail_constant,
    stable_radial_tail_constant,
    stable_score_tail_linear_coefficient,
    stable_tail_score,
    student_abs_quantile,
    student_cf,
    student_density,
    student_density_tail_constant,
    student_radial_tail_constant,
    vp_tail_coefficients,
    vp_tail_score,
)


def test_student_density_cf_tail_constants_and_quantiles() -> None:
    for nu in (1.2, 1.5, 3.0):
        mass = (
            2.0
            * quad(
                lambda value, index=nu: student_density(value, index),
                0.0,
                np.inf,
                epsabs=2.0e-12,
                epsrel=2.0e-12,
                limit=600,
            )[0]
        )
        assert math.isclose(mass, 1.0, rel_tol=0.0, abs_tol=3.0e-11)
        assert student_cf(0.0, nu) == 1.0
        probes = np.asarray([-1.1, -0.3, 0.3, 1.1])
        assert np.array_equal(student_cf(probes, nu), student_cf(-probes, nu))
        for frequency in (0.3, 1.1):
            direct = (
                2.0
                * quad(
                    lambda value, index=nu: student_density(value, index),
                    0.0,
                    np.inf,
                    weight="cos",
                    wvar=frequency,
                    epsabs=2.0e-11,
                    epsrel=2.0e-11,
                    limit=600,
                )[0]
            )
            assert math.isclose(student_cf(frequency, nu), direct, rel_tol=2e-9, abs_tol=2e-10)

        radius = 1.0e5
        density_limit = student_density(radius, nu) * radius ** (nu + 1.0)
        assert math.isclose(density_limit, student_density_tail_constant(nu), rel_tol=2.0e-9)
        assert math.isclose(
            student_radial_tail_constant(nu),
            2.0 * student_density_tail_constant(nu) / nu,
            rel_tol=2.0e-15,
        )
        probabilities = np.asarray([0.9, 0.99, 0.999])
        quantiles = student_abs_quantile(probabilities, nu)
        recovered = 2.0 * student_t_distribution.cdf(quantiles, nu) - 1.0
        assert np.allclose(recovered, probabilities, rtol=0.0, atol=2.0e-12)


def test_forward_characteristic_functions_use_stated_conventions() -> None:
    alpha, beta, nu, time = 1.5, 1.2, 1.2, 0.7
    frequencies = np.asarray([0.0, 0.25, 0.9])
    stable_a, stable_gamma = forward_coefficients("stable", time, alpha=alpha, beta=beta)
    vp_a, vp_gamma = forward_coefficients("vp", time, alpha=alpha, beta=beta)
    innovation_power = -math.expm1(-beta * time)
    assert math.isclose(stable_gamma**alpha, innovation_power, rel_tol=2.0e-15)
    assert math.isclose(vp_gamma**2, innovation_power, rel_tol=2.0e-15)
    assert np.array_equal(
        forward_marginal_cf(frequencies, time, "stable", nu, alpha=alpha, beta=beta),
        student_cf(stable_a * frequencies, nu)
        * np.exp(-innovation_power * np.abs(frequencies) ** alpha),
    )
    assert np.array_equal(
        forward_marginal_cf(frequencies, time, "vp", nu, alpha=alpha, beta=beta),
        student_cf(vp_a * frequencies, nu) * np.exp(-0.5 * innovation_power * frequencies**2),
    )


def test_stable_tail_formula_has_all_three_target_regimes() -> None:
    alpha, time = 1.5, 0.5
    points = np.asarray([1.0e5, 2.0e5])
    for nu in (1.2, 1.5, 3.0):
        score = stable_tail_score(points, time, nu, alpha=alpha)
        assert np.array_equal(stable_tail_score(-points, time, nu, alpha=alpha), -score)
        slope = np.log(abs(score[1] / score[0])) / math.log(2.0)
        expected_power = 1.0 + nu - alpha if nu < alpha else 1.0
        assert math.isclose(slope, expected_power, rel_tol=0.0, abs_tol=5.0e-3)

    assert stable_score_tail_linear_coefficient(time, 1.2, alpha=alpha) == 0.0
    matched = stable_score_tail_linear_coefficient(time, alpha, alpha=alpha)
    lighter_target = stable_score_tail_linear_coefficient(time, 3.0, alpha=alpha)
    assert matched < 0.0
    assert math.isclose(
        lighter_target,
        -1.0 / (alpha * -math.expm1(-time)),
        rel_tol=2.0e-15,
    )


def test_vp_tail_expansion_coefficients_and_score_sign() -> None:
    time = 0.4
    coefficient_a, coefficient_b = vp_tail_coefficients(time, 4.0)
    a2 = math.exp(-time)
    g2 = -math.expm1(-time)
    assert math.isclose(coefficient_a, 15.0 * g2 - 10.0 * a2, rel_tol=2e-15)
    assert math.isclose(
        coefficient_b,
        70.0 * a2**2 - 280.0 * a2 * g2 + 210.0 * g2**2,
        rel_tol=2e-15,
    )
    points = np.asarray([100.0, 200.0])
    score = vp_tail_score(points, time, 1.5)
    assert np.array_equal(vp_tail_score(-points, time, 1.5), -score)
    assert np.all(score < 0.0)
    assert np.max(np.abs(score / (-2.5 / points) - 1.0)) < 2.0e-3


def test_continuous_and_discrete_stable_tail_scale_constants() -> None:
    alpha, beta, eta = 1.5, 1.0, 0.5
    horizon, epsilon = 2.0, 0.5
    duration = horizon - epsilon
    # For nu < alpha the manuscript gives ell_nu=beta/alpha, so rho^alpha
    # solves r'=beta*r+eta*beta with r(0)=1.
    exact_heavy_target = (1.0 + eta) * math.exp(beta * duration) - eta
    computed = continuous_stable_scale_power(1.2, horizon, epsilon, alpha=alpha, beta=beta, eta=eta)
    assert math.isclose(computed, exact_heavy_target, rel_tol=2.0e-11)

    for nu in (1.2, 1.5, 3.0):
        continuous = continuous_stable_scale_power(
            nu, horizon, epsilon, alpha=alpha, beta=beta, eta=eta
        )
        coarse = discrete_stable_scale_power(
            nu, horizon, epsilon, 100, alpha=alpha, beta=beta, eta=eta
        )
        fine = discrete_stable_scale_power(
            nu, horizon, epsilon, 800, alpha=alpha, beta=beta, eta=eta
        )
        assert abs(fine - continuous) < abs(coarse - continuous)
        assert math.isclose(fine, continuous, rel_tol=4.0e-3)

    stable_density = stable_density_tail_constant(alpha)
    assert math.isclose(
        stable_radial_tail_constant(alpha), 2.0 * stable_density / alpha, rel_tol=2e-15
    )
