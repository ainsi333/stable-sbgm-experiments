from __future__ import annotations

import math

import numpy as np

from levy_experiments.integrators import (
    popov_remainder,
    stable_ei_coefficients,
    stationary_terminal_scale_power,
)
from levy_experiments.theory import (
    discrete_marginal_cf,
    discrete_true_reverse_joint_cf,
    stationary_joint_cf,
    stationary_true_reverse_joint_cf,
)


def test_ei_factors_include_alpha_eta_beta_and_attenuation() -> None:
    alpha, eta, beta, step = 1.5, 0.5, 1.3, 0.07
    coefficients = stable_ei_coefficients(alpha=alpha, beta=beta, step=step, drift_eta=eta)
    rate = eta * beta / alpha
    assert math.isclose(coefficients.decay, math.exp(-rate * step), rel_tol=1e-15)
    assert math.isclose(
        coefficients.remainder_multiplier,
        -math.expm1(-rate * step) / rate,
        rel_tol=1e-15,
    )
    assert math.isclose(
        coefficients.noise_scale**alpha,
        -math.expm1(-eta * beta * step),
        rel_tol=1e-14,
    )


def test_popov_remainder_contains_one_plus_eta_alpha_and_beta() -> None:
    x = np.asarray([-2.0, 0.5, 3.0])
    score = np.asarray([0.4, -0.2, 1.1])
    alpha, eta, beta = 1.5, 0.7, 1.3
    expected = ((1.0 + eta) * beta / alpha) * (x + alpha * score)

    assert np.array_equal(
        popov_remainder(x, score, alpha=alpha, beta=beta, eta=eta), expected
    )


def test_valid_ei_is_stationary_for_every_eta() -> None:
    for eta in (0.25, 0.5, 1.0, 2.0):
        assert math.isclose(
            stationary_terminal_scale_power(beta=1.0, lag=0.8, drift_eta=eta, noise_eta=eta),
            1.0,
            abs_tol=1e-15,
        )


def test_invalid_hybrid_fails_stationary_marginal_automatically() -> None:
    scale_power = stationary_terminal_scale_power(beta=1.0, lag=1.0, drift_eta=0.5, noise_eta=1.0)
    assert math.isclose(scale_power, 2.0 - math.exp(-0.5), rel_tol=1e-15)
    assert scale_power > 1.3


def test_stationary_joint_laws_depend_on_eta_and_are_not_time_reversal() -> None:
    u, v = np.asarray([1.0]), np.asarray([0.5])
    low_eta = stationary_joint_cf(u, v, alpha=1.5, beta=1.0, eta=0.25, lag=1.0)
    high_eta = stationary_joint_cf(u, v, alpha=1.5, beta=1.0, eta=2.0, lag=1.0)
    true_reverse = stationary_true_reverse_joint_cf(u, v, alpha=1.5, beta=1.0, lag=1.0)
    assert not np.allclose(low_eta, high_eta)
    eta_one = stationary_joint_cf(u, v, alpha=1.5, beta=1.0, eta=1.0, lag=1.0)
    assert not np.allclose(eta_one, true_reverse)


def test_nonstationary_true_reverse_cf_has_correct_marginals() -> None:
    u = np.linspace(-2.0, 2.0, 11)
    zeros = np.zeros_like(u)
    kwargs = {
        "later_forward_time": 2.5,
        "earlier_forward_time": 1.5,
        "alpha": 1.5,
        "beta": 1.0,
        "atoms": (-3.0, 0.5, 2.0),
        "weights": (0.25, 0.5, 0.25),
    }
    later = discrete_true_reverse_joint_cf(u, zeros, **kwargs)
    earlier = discrete_true_reverse_joint_cf(zeros, u, **kwargs)
    assert np.allclose(
        later,
        discrete_marginal_cf(
            u,
            2.5,
            alpha=1.5,
            beta=1.0,
            atoms=kwargs["atoms"],
            weights=kwargs["weights"],
        ),
    )
    assert np.allclose(
        earlier,
        discrete_marginal_cf(
            u,
            1.5,
            alpha=1.5,
            beta=1.0,
            atoms=kwargs["atoms"],
            weights=kwargs["weights"],
        ),
    )
