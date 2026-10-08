"""Exact one-dimensional theory used by Experiment 1.

The target is a centred, unit-scale Student distribution.  The stable model
uses the convention ``E exp(i u Z_alpha) = exp(-|u|**alpha)``; the VP model
uses a standard normal innovation.  All public numerical formulae are kept in
float64 because Experiment 1 explicitly probes extreme tails.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gamma, gammaln, kv
from scipy.stats import t as student_t_distribution

from levy_experiments.errors import ConfigurationError
from levy_experiments.integrators import stable_ei_coefficients
from levy_experiments.stable import sample_symmetric_stable_1d

Model = Literal["stable", "vp"]


def _check_model(model: str) -> Model:
    if model not in {"stable", "vp"}:
        raise ConfigurationError(f"unknown Experiment 1 model: {model!r}")
    return model  # type: ignore[return-value]


def _check_nu(nu: float) -> float:
    value = float(nu)
    if not math.isfinite(value) or value <= 1.0:
        raise ConfigurationError("Experiment 1 requires a Student index nu > 1")
    return value


def _check_parameters(model: str, *, alpha: float, beta: float, nu: float) -> Model:
    checked = _check_model(model)
    _check_nu(nu)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ConfigurationError("beta must be finite and strictly positive")
    if checked == "stable" and (not math.isfinite(alpha) or not 1.0 < alpha < 2.0):
        raise ConfigurationError("the stable model requires 1 < alpha < 2")
    return checked


def _shape_tuple(shape: int | tuple[int, ...]) -> tuple[int, ...]:
    result = (shape,) if isinstance(shape, int) else tuple(shape)
    if any(not isinstance(size, int) or size < 0 for size in result):
        raise ConfigurationError("sample shape must contain non-negative integers")
    return result


def student_density(x, nu: float):
    """Density of a centred, unit-scale Student distribution."""

    checked_nu = _check_nu(nu)
    values = np.asarray(x, dtype=np.float64)
    log_normalization = (
        gammaln((checked_nu + 1.0) / 2.0)
        - gammaln(checked_nu / 2.0)
        - 0.5 * math.log(checked_nu * math.pi)
    )
    result = np.exp(
        log_normalization - ((checked_nu + 1.0) / 2.0) * np.log1p(values * values / checked_nu)
    )
    return float(result) if result.ndim == 0 else result


def student_cf(u, nu: float):
    """Characteristic function of a centred, unit-scale Student law.

    For ``z=sqrt(nu)*abs(u)`` this is
    ``2**(1-nu/2) * z**(nu/2) * K_(nu/2)(z) / Gamma(nu/2)``.
    A local expansion avoids the indeterminate product at zero and overflow
    for pathologically small nonzero probes.
    """

    checked_nu = _check_nu(nu)
    frequencies = np.asarray(u, dtype=np.float64)
    radii = np.abs(frequencies)
    z = math.sqrt(checked_nu) * radii
    order = checked_nu / 2.0
    result = np.empty_like(z)
    zero = z == 0.0
    result[zero] = 1.0
    small = (z > 0.0) & (z < 1.0e-5)
    if checked_nu < 2.0:
        nonanalytic = 2.0 ** (-checked_nu) * gamma(-order) / gamma(order) * z[small] ** checked_nu
        analytic = z[small] ** 2 / (2.0 * (2.0 - checked_nu))
        result[small] = 1.0 + nonanalytic + analytic
    elif checked_nu > 2.0:
        result[small] = 1.0 - z[small] ** 2 / (2.0 * (checked_nu - 2.0))
    else:  # nu=2 has a logarithmic small-frequency term; use the Bessel form.
        small[:] = False
    regular = ~(zero | small)
    regular_z = z[regular]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        log_values = (
            (1.0 - order) * math.log(2.0)
            - gammaln(order)
            + order * np.log(regular_z)
            + np.log(kv(order, regular_z))
        )
        result[regular] = np.exp(log_values)
    # Underflow of K_nu at frequencies where the CF is numerically zero is benign.
    result[regular & ~np.isfinite(result)] = 0.0
    return float(result) if result.ndim == 0 else result


def student_density_tail_constant(nu: float) -> float:
    """Return ``c_nu`` in ``f_nu(x) ~ c_nu |x|**(-(nu+1))``."""

    checked_nu = _check_nu(nu)
    return float(
        gamma((checked_nu + 1.0) / 2.0)
        * checked_nu ** (checked_nu / 2.0)
        / (math.sqrt(math.pi) * gamma(checked_nu / 2.0))
    )


def student_radial_tail_constant(nu: float) -> float:
    """Return ``c`` in ``P(abs(T_nu)>r) ~ c r**(-nu)``."""

    checked_nu = _check_nu(nu)
    return 2.0 * student_density_tail_constant(checked_nu) / checked_nu


def student_abs_quantile(probability, nu: float):
    """Exact radial quantile ``q_kappa(abs(T_nu))`` for ``0 < kappa < 1``."""

    checked_nu = _check_nu(nu)
    probabilities = np.asarray(probability, dtype=np.float64)
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities <= 0.0) | (probabilities >= 1.0)
    ):
        raise ConfigurationError("radial Student probabilities must lie strictly in (0,1)")
    result = student_t_distribution.ppf((1.0 + probabilities) / 2.0, checked_nu)
    residual = 2.0 * student_t_distribution.cdf(result, checked_nu) - 1.0 - probabilities
    result = result - residual / (2.0 * student_t_distribution.pdf(result, checked_nu))
    return float(result) if result.ndim == 0 else np.asarray(result, dtype=np.float64)


def forward_coefficients(model: Model, t, *, alpha: float = 1.5, beta: float = 1.0):
    """Return the exact multiplicative and innovation scales ``(a, gamma)``."""

    checked = _check_model(model)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ConfigurationError("beta must be finite and strictly positive")
    if checked == "stable" and (not math.isfinite(alpha) or not 1.0 < alpha < 2.0):
        raise ConfigurationError("the stable model requires 1 < alpha < 2")
    times = np.asarray(t, dtype=np.float64)
    if np.any(~np.isfinite(times)) or np.any(times < 0.0):
        raise ConfigurationError("forward times must be finite and non-negative")
    exponent = alpha if checked == "stable" else 2.0
    a = np.exp(-beta * times / exponent)
    innovation_power = -np.expm1(-beta * times)
    gamma_scale = np.power(innovation_power, 1.0 / exponent)
    if times.ndim == 0:
        return float(a), float(gamma_scale)
    return a, gamma_scale


def forward_marginal_cf(
    u,
    t,
    model: Model,
    nu: float,
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
):
    """Characteristic function of the exact forward marginal ``p_t``."""

    checked = _check_parameters(model, alpha=alpha, beta=beta, nu=nu)
    frequencies = np.asarray(u, dtype=np.float64)
    times = np.asarray(t, dtype=np.float64)
    if np.any(~np.isfinite(times)) or np.any(times < 0.0):
        raise ConfigurationError("forward times must be finite and non-negative")
    exponent = alpha if checked == "stable" else 2.0
    a = np.exp(-beta * times / exponent)
    innovation_power = -np.expm1(-beta * times)
    target = student_cf(a * frequencies, nu)
    if checked == "stable":
        innovation = np.exp(-innovation_power * np.abs(frequencies) ** alpha)
    else:
        innovation = np.exp(-0.5 * innovation_power * frequencies * frequencies)
    result = np.asarray(target) * innovation
    return float(result) if result.ndim == 0 else result


def sample_student(key, nu: float, shape: int | tuple[int, ...], dtype):
    """Draw exact standard Student samples with a caller-owned JAX key."""

    checked_nu = _check_nu(nu)
    import jax

    return jax.random.t(key, checked_nu, shape=_shape_tuple(shape), dtype=dtype)


def sample_exact_forward_marginal(
    key,
    model: Model,
    nu: float,
    t: float,
    shape: int | tuple[int, ...],
    dtype,
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
):
    """Draw exactly from ``p_t`` using independent target and innovation keys."""

    checked = _check_parameters(model, alpha=alpha, beta=beta, nu=nu)
    if not math.isfinite(t) or t < 0.0:
        raise ConfigurationError("forward time must be finite and non-negative")
    import jax
    import jax.numpy as jnp

    sample_shape = _shape_tuple(shape)
    key_target, key_noise = jax.random.split(key)
    target = sample_student(key_target, nu, sample_shape, dtype)
    exponent = alpha if checked == "stable" else 2.0
    a = jnp.asarray(math.exp(-beta * t / exponent), dtype=dtype)
    innovation_power = -math.expm1(-beta * t)
    innovation_scale = jnp.asarray(innovation_power ** (1.0 / exponent), dtype=dtype)
    if checked == "stable":
        noise = sample_symmetric_stable_1d(key_noise, alpha, sample_shape, dtype)
    else:
        noise = jax.random.normal(key_noise, shape=sample_shape, dtype=dtype)
    return a * target + innovation_scale * noise


def sample_stationary_reference(
    key,
    model: Model,
    shape: int | tuple[int, ...],
    dtype,
    *,
    alpha: float = 1.5,
):
    """Draw the stable or VP stationary reference law."""

    checked = _check_model(model)
    sample_shape = _shape_tuple(shape)
    if checked == "stable":
        if not math.isfinite(alpha) or not 1.0 < alpha < 2.0:
            raise ConfigurationError("the stable reference requires 1 < alpha < 2")
        return sample_symmetric_stable_1d(key, alpha, sample_shape, dtype)
    import jax

    return jax.random.normal(key, shape=sample_shape, dtype=dtype)


def stable_density_tail_constant(alpha: float) -> float:
    """Density-tail coefficient of the standard one-dimensional S-alpha-S law."""

    if not math.isfinite(alpha) or not 1.0 < alpha < 2.0:
        raise ConfigurationError("stable tail constants require 1 < alpha < 2")
    return alpha * gamma(alpha) * math.sin(math.pi * alpha / 2.0) / math.pi


def stable_radial_tail_constant(alpha: float) -> float:
    """Return ``C`` in ``P(abs(Z_alpha)>r) ~ C r**(-alpha)``."""

    return 2.0 * stable_density_tail_constant(alpha) / alpha


def stable_tail_score(
    x,
    t: float,
    nu: float,
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
):
    """Leading regime-correct fractional-score tail for a Student target.

    The denominator retains both possible leading sources of marginal tail:
    the scaled Student target and the stable innovation.  The numerator is the
    universal leading tail of the fractional derivative.  This single formula
    therefore covers ``nu < alpha``, ``nu == alpha`` and ``nu > alpha``.
    """

    _check_parameters("stable", alpha=alpha, beta=beta, nu=nu)
    if not math.isfinite(t) or t <= 0.0:
        raise ConfigurationError("the stable tail expansion requires positive forward time")
    values = np.asarray(x, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values == 0.0):
        raise ConfigurationError("the stable tail expansion requires finite nonzero x")
    radii = np.abs(values)
    a = math.exp(-beta * t / alpha)
    innovation_power = -math.expm1(-beta * t)
    stable_constant = stable_density_tail_constant(alpha)
    denominator = student_density_tail_constant(nu) * a**nu * radii ** (
        -(nu + 1.0)
    ) + stable_constant * innovation_power * radii ** (-(alpha + 1.0))
    numerator = -np.sign(values) * (stable_constant / alpha) * radii ** (-alpha)
    result = numerator / denominator
    return float(result) if result.ndim == 0 else result


def vp_tail_coefficients(t: float, nu: float, *, beta: float = 1.0) -> tuple[float, float]:
    """Return ``(A,B)`` for the Gaussian-convolved Student density tail."""

    checked_nu = _check_nu(nu)
    if not math.isfinite(beta) or beta <= 0.0 or not math.isfinite(t) or t < 0.0:
        raise ConfigurationError("VP tail coefficients require beta > 0 and t >= 0")
    a2 = math.exp(-beta * t)
    g2 = -math.expm1(-beta * t)
    coefficient_a = (
        -0.5 * checked_nu * (checked_nu + 1.0) * a2
        + 0.5 * (checked_nu + 1.0) * (checked_nu + 2.0) * g2
    )
    coefficient_b = (
        checked_nu**2 * (checked_nu + 1.0) * (checked_nu + 3.0) * a2**2 / 8.0
        - checked_nu * (checked_nu + 1.0) * (checked_nu + 3.0) * (checked_nu + 4.0) * a2 * g2 / 4.0
        + (checked_nu + 1.0)
        * (checked_nu + 2.0)
        * (checked_nu + 3.0)
        * (checked_nu + 4.0)
        * g2**2
        / 8.0
    )
    return coefficient_a, coefficient_b


def vp_tail_score(x, t: float, nu: float, *, beta: float = 1.0):
    """Three-term ordinary-score tail through order ``abs(x)**-5``."""

    values = np.asarray(x, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values == 0.0):
        raise ConfigurationError("the VP tail expansion requires finite nonzero x")
    coefficient_a, coefficient_b = vp_tail_coefficients(t, nu, beta=beta)
    result = (
        -(nu + 1.0) / values
        - 2.0 * coefficient_a / values**3
        + (2.0 * coefficient_a**2 - 4.0 * coefficient_b) / values**5
    )
    return float(result) if result.ndim == 0 else result


def stable_score_tail_linear_coefficient(
    forward_time: float,
    nu: float,
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
) -> float:
    """Return the coefficient of ``x`` in the stable score at infinity."""

    checked_nu = _check_nu(nu)
    if not math.isfinite(forward_time) or forward_time <= 0.0:
        raise ConfigurationError("tail coefficients require positive forward time")
    a = math.exp(-beta * forward_time / alpha)
    innovation_power = -math.expm1(-beta * forward_time)
    if checked_nu < alpha and not math.isclose(checked_nu, alpha):
        return 0.0
    if math.isclose(checked_nu, alpha, rel_tol=0.0, abs_tol=1.0e-12):
        stable_constant = stable_density_tail_constant(alpha)
        denominator = (
            student_density_tail_constant(checked_nu) * a**alpha
            + stable_constant * innovation_power
        )
        return -stable_constant / (alpha * denominator)
    return -1.0 / (alpha * innovation_power)


def stable_backward_linear_coefficient(
    forward_time: float,
    nu: float,
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
    eta: float = 0.5,
) -> float:
    """Linear coefficient ``ell_nu`` of the exact backward drift at infinity."""

    if not math.isfinite(eta) or eta <= 0.0:
        raise ConfigurationError("eta must be finite and positive")
    score_slope = stable_score_tail_linear_coefficient(forward_time, nu, alpha=alpha, beta=beta)
    return beta / alpha + (1.0 + eta) * beta * score_slope


def continuous_stable_scale_power(
    nu: float,
    horizon: float,
    epsilon: float,
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
    eta: float = 0.5,
) -> float:
    """Return ``rho**alpha`` for the continuous reference-initialized sampler."""

    _check_parameters("stable", alpha=alpha, beta=beta, nu=nu)
    if not math.isfinite(eta) or eta <= 0.0:
        raise ConfigurationError("eta must be finite and positive")
    if not math.isfinite(horizon) or not math.isfinite(epsilon) or not 0.0 < epsilon < horizon:
        raise ConfigurationError("continuous tail scale requires 0 < epsilon < horizon")
    duration = horizon - epsilon

    def equation(backward_time: float, state: np.ndarray) -> np.ndarray:
        forward_time = horizon - backward_time
        coefficient = stable_backward_linear_coefficient(
            forward_time, nu, alpha=alpha, beta=beta, eta=eta
        )
        return np.asarray([alpha * coefficient * state[0] + eta * beta], dtype=np.float64)

    solution = solve_ivp(
        equation,
        (0.0, duration),
        np.asarray([1.0], dtype=np.float64),
        method="DOP853",
        rtol=2.0e-12,
        atol=2.0e-14,
    )
    if not solution.success or not np.isfinite(solution.y[0, -1]) or solution.y[0, -1] <= 0.0:
        raise RuntimeError(f"continuous stable tail-scale integration failed: {solution.message}")
    return float(solution.y[0, -1])


def continuous_stable_tail_constant(
    nu: float,
    horizon: float,
    epsilon: float,
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
    eta: float = 0.5,
) -> float:
    """Radial survival-tail constant of the continuous stable output."""

    return stable_radial_tail_constant(alpha) * continuous_stable_scale_power(
        nu, horizon, epsilon, alpha=alpha, beta=beta, eta=eta
    )


def ei_stable_step_tail_multiplier(
    forward_time: float,
    nu: float,
    step: float,
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
    eta: float = 0.5,
) -> float:
    """Linear multiplier of one frozen-score stable EI update at infinity."""

    coefficients = stable_ei_coefficients(alpha=alpha, beta=beta, step=step, drift_eta=eta)
    score_slope = stable_score_tail_linear_coefficient(forward_time, nu, alpha=alpha, beta=beta)
    remainder_slope = ((1.0 + eta) * beta / alpha) * (1.0 + alpha * score_slope)
    return coefficients.decay + coefficients.remainder_multiplier * remainder_slope


def discrete_stable_scale_power(
    nu: float,
    horizon: float,
    epsilon: float,
    steps: int,
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
    eta: float = 0.5,
) -> float:
    """Return the exact tail-scale recurrence of the implemented stable EI chain."""

    _check_parameters("stable", alpha=alpha, beta=beta, nu=nu)
    if not isinstance(steps, int) or steps <= 0:
        raise ConfigurationError("the EI tail recurrence requires a positive step count")
    if not math.isfinite(horizon) or not math.isfinite(epsilon) or not 0.0 < epsilon < horizon:
        raise ConfigurationError("the EI tail recurrence requires 0 < epsilon < horizon")
    step = (horizon - epsilon) / steps
    coefficients = stable_ei_coefficients(alpha=alpha, beta=beta, step=step, drift_eta=eta)
    scale_power = 1.0
    for index in range(steps):
        forward_time = horizon - index * step
        multiplier = ei_stable_step_tail_multiplier(
            forward_time,
            nu,
            step,
            alpha=alpha,
            beta=beta,
            eta=eta,
        )
        scale_power = abs(multiplier) ** alpha * scale_power + coefficients.noise_scale**alpha
    return float(scale_power)


def discrete_stable_tail_constant(
    nu: float,
    horizon: float,
    epsilon: float,
    steps: int,
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
    eta: float = 0.5,
) -> float:
    """Radial survival-tail constant of the stable EI output."""

    return stable_radial_tail_constant(alpha) * discrete_stable_scale_power(
        nu, horizon, epsilon, steps, alpha=alpha, beta=beta, eta=eta
    )


# Explicit aliases matching the notation used in the manuscript and aggregators.
sample_exact_p_t = sample_exact_forward_marginal
sample_exact_p_s = sample_exact_forward_marginal
continuous_rho_power = continuous_stable_scale_power
ei_discrete_rho_power = discrete_stable_scale_power


__all__ = [
    "Model",
    "continuous_rho_power",
    "continuous_stable_scale_power",
    "continuous_stable_tail_constant",
    "discrete_stable_scale_power",
    "discrete_stable_tail_constant",
    "ei_discrete_rho_power",
    "ei_stable_step_tail_multiplier",
    "forward_coefficients",
    "forward_marginal_cf",
    "sample_exact_forward_marginal",
    "sample_exact_p_s",
    "sample_exact_p_t",
    "sample_stationary_reference",
    "sample_student",
    "stable_backward_linear_coefficient",
    "stable_density_tail_constant",
    "stable_radial_tail_constant",
    "stable_score_tail_linear_coefficient",
    "stable_tail_score",
    "student_abs_quantile",
    "student_cf",
    "student_density",
    "student_density_tail_constant",
    "student_radial_tail_constant",
    "vp_tail_coefficients",
    "vp_tail_score",
]
