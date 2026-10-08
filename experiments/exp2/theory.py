"""Exact one-dimensional ingredients for Experiment 2.

The target is the centred Student distribution with four degrees of freedom.
The two forward models use the same constant VP schedule ``beta``:

* ``stable``: ``X_t = exp(-beta*t/alpha) X_0 + gamma_alpha(t) Z_alpha``;
* ``vp``: ``X_t = exp(-beta*t/2) X_0 + gamma_2(t) G``.

Here ``E exp(i u Z_alpha) = exp(-|u|**alpha)`` and ``G`` is standard
normal.  These conventions are deliberately stated next to the executable
formulae because a scale mismatch would invalidate the comparison.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.special import kv

from levy_experiments.errors import ConfigurationError
from levy_experiments.scores.stable_table import three_term_asymptotic_coefficients
from levy_experiments.stable import sample_symmetric_stable_1d

Model = Literal["stable", "vp"]
STUDENT_DF = 4.0
STUDENT_VARIANCE = 2.0


def _check_model(model: str) -> Model:
    if model not in {"stable", "vp"}:
        raise ConfigurationError(f"unknown Experiment 2 model: {model!r}")
    return model  # type: ignore[return-value]


def _check_parameters(model: str, alpha: float, beta: float) -> Model:
    checked = _check_model(model)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ConfigurationError("beta must be finite and strictly positive")
    if checked == "stable" and (not math.isfinite(alpha) or not 1.0 < alpha < 2.0):
        raise ConfigurationError("the stable Experiment 2 model requires 1 < alpha < 2")
    return checked


def student_t4_density(x):
    """Density ``3/8 * (1 + x**2/4)**(-5/2)`` of the target law."""

    values = np.asarray(x, dtype=np.float64)
    result = (3.0 / 8.0) * np.power(1.0 + values * values / 4.0, -2.5)
    return float(result) if result.ndim == 0 else result


def student_t4_cf(u):
    """Characteristic function of the centred Student-t(4) target.

    For nonzero ``u`` this is ``2*u**2*K_2(2*abs(u))``.  The small-frequency
    branch avoids the indeterminate numerical product ``0 * inf`` and uses
    the variance-two expansion ``1 - u**2 + o(u**2)``.
    """

    frequencies = np.asarray(u, dtype=np.float64)
    radii = np.abs(frequencies)
    result = np.empty_like(radii)
    small = radii < 1.0e-5
    result[small] = 1.0 - radii[small] * radii[small]
    regular = ~small
    z = 2.0 * radii[regular]
    result[regular] = 0.5 * z * z * kv(2.0, z)
    return float(result) if result.ndim == 0 else result


def forward_coefficients(model: Model, t, *, alpha: float = 1.5, beta: float = 1.0):
    """Return the exact multiplicative and innovation scales ``(a, gamma)``."""

    checked = _check_parameters(model, alpha, beta)
    times = np.asarray(t, dtype=np.float64)
    if np.any(~np.isfinite(times)) or np.any(times < 0.0):
        raise ConfigurationError("forward times must be finite and non-negative")
    exponent = alpha if checked == "stable" else 2.0
    a = np.exp(-beta * times / exponent)
    gamma_power = -np.expm1(-beta * times)
    gamma = np.power(gamma_power, 1.0 / exponent)
    if times.ndim == 0:
        return float(a), float(gamma)
    return a, gamma


def forward_marginal_cf(u, t, model: Model, *, alpha: float = 1.5, beta: float = 1.0):
    """Characteristic function of the exact forward marginal ``p_t``."""

    checked = _check_parameters(model, alpha, beta)
    frequencies = np.asarray(u, dtype=np.float64)
    times = np.asarray(t, dtype=np.float64)
    if np.any(~np.isfinite(times)) or np.any(times < 0.0):
        raise ConfigurationError("forward times must be finite and non-negative")
    exponent = alpha if checked == "stable" else 2.0
    a = np.exp(-beta * times / exponent)
    innovation_power = -np.expm1(-beta * times)
    target_part = student_t4_cf(a * frequencies)
    if checked == "stable":
        innovation_part = np.exp(-innovation_power * np.abs(frequencies) ** alpha)
    else:
        innovation_part = np.exp(-0.5 * innovation_power * frequencies * frequencies)
    result = np.asarray(target_part) * innovation_part
    return float(result) if result.ndim == 0 else result


def _shape_tuple(shape: int | tuple[int, ...]) -> tuple[int, ...]:
    result = (shape,) if isinstance(shape, int) else tuple(shape)
    if any(not isinstance(size, int) or size < 0 for size in result):
        raise ConfigurationError("sample shape must contain non-negative integers")
    return result


def sample_student_t4(key, shape: int | tuple[int, ...], dtype):
    """Draw exact Student-t(4) target samples with a caller-owned JAX key."""

    import jax

    return jax.random.t(key, STUDENT_DF, shape=_shape_tuple(shape), dtype=dtype)


def sample_exact_forward_marginal(
    key,
    model: Model,
    t: float,
    shape: int | tuple[int, ...],
    dtype,
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
):
    """Draw from ``p_t`` exactly, separating target and innovation subkeys."""

    checked = _check_parameters(model, alpha, beta)
    if not math.isfinite(t) or t < 0.0:
        raise ConfigurationError("forward time must be finite and non-negative")
    import jax
    import jax.numpy as jnp

    sample_shape = _shape_tuple(shape)
    key_target, key_noise = jax.random.split(key)
    target = sample_student_t4(key_target, sample_shape, dtype)
    exponent = alpha if checked == "stable" else 2.0
    a = jnp.asarray(math.exp(-beta * t / exponent), dtype=dtype)
    innovation_power = -math.expm1(-beta * t)
    gamma = jnp.asarray(innovation_power ** (1.0 / exponent), dtype=dtype)
    if checked == "stable":
        noise = sample_symmetric_stable_1d(key_noise, alpha, sample_shape, dtype)
    else:
        noise = jax.random.normal(key_noise, shape=sample_shape, dtype=dtype)
    return a * target + gamma * noise


def sample_stationary_reference(
    key,
    model: Model,
    shape: int | tuple[int, ...],
    dtype,
    *,
    alpha: float = 1.5,
):
    """Draw the asymptotic reference law, distinct from the exact ``p_T``."""

    checked = _check_model(model)
    sample_shape = _shape_tuple(shape)
    if checked == "stable":
        if not math.isfinite(alpha) or not 1.0 < alpha < 2.0:
            raise ConfigurationError("the stable reference requires 1 < alpha < 2")
        return sample_symmetric_stable_1d(key, alpha, sample_shape, dtype)
    import jax

    return jax.random.normal(key, shape=sample_shape, dtype=dtype)


def vp_tail_score(x, t: float, *, beta: float = 1.0):
    """Three-term large-|x| ordinary-score expansion for the VP marginal.

    This is an explicit asymptotic extrapolation, not an exact score.  Gaussian
    convolution of the scaled Student density gives the coefficients through
    order ``|x|**-5``.
    """

    _check_parameters("vp", 2.0, beta)
    if not math.isfinite(t) or t < 0.0:
        raise ConfigurationError("forward time must be finite and non-negative")
    values = np.asarray(x, dtype=np.float64)
    if np.any(values == 0.0) or np.any(~np.isfinite(values)):
        raise ConfigurationError("the VP tail expansion requires finite nonzero x")
    a2 = math.exp(-beta * t)
    g2 = -math.expm1(-beta * t)
    coefficient_a = 15.0 * g2 - 10.0 * a2
    coefficient_b = 70.0 * a2**2 - 280.0 * a2 * g2 + 210.0 * g2**2
    result = (
        -5.0 / values
        - 2.0 * coefficient_a / values**3
        + (2.0 * coefficient_a**2 - 4.0 * coefficient_b) / values**5
    )
    return float(result) if result.ndim == 0 else result


def stable_tail_score(x, t: float, *, alpha: float = 1.5, beta: float = 1.0):
    """Explicit large-|x| fractional-score extrapolation for ``p_t``.

    The stable innovation density is represented by its first three tail
    terms.  The leading finite-variance posterior correction for the Student
    target is then inserted into the exact fractional Tweedie identity.  The
    construction is validated only for the manuscript's ``alpha=1.5``.
    """

    _check_parameters("stable", alpha, beta)
    if not math.isclose(alpha, 1.5, rel_tol=0.0, abs_tol=1.0e-12):
        raise ConfigurationError("the stable tail extrapolation is validated only at alpha=1.5")
    if not math.isfinite(t) or t <= 0.0:
        raise ConfigurationError("the stable tail expansion requires a positive forward time")
    values = np.asarray(x, dtype=np.float64)
    if np.any(values == 0.0) or np.any(~np.isfinite(values)):
        raise ConfigurationError("the stable tail expansion requires finite nonzero x")
    radii = np.abs(values)
    signs = np.sign(values)
    a = math.exp(-beta * t / alpha)
    innovation_power = -math.expm1(-beta * t)
    coefficients = three_term_asymptotic_coefficients(alpha)
    density_tail = np.zeros_like(radii)
    radial_derivative = np.zeros_like(radii)
    for k, coefficient in enumerate(coefficients, start=1):
        exponent = k * alpha + 1.0
        scaled = coefficient * innovation_power**k
        density_tail += scaled * radii ** (-exponent)
        radial_derivative -= exponent * scaled * radii ** (-exponent - 1.0)
    log_derivative = signs * radial_derivative / density_tail
    # E[X_0 | X_t=x] ~ -a Var(X_0) d_x log q_gamma(x).
    tweedie_residual = values + a * a * STUDENT_VARIANCE * log_derivative
    result = -tweedie_residual / (alpha * innovation_power)
    return float(result) if result.ndim == 0 else result


__all__ = [
    "STUDENT_DF",
    "STUDENT_VARIANCE",
    "Model",
    "forward_coefficients",
    "forward_marginal_cf",
    "sample_exact_forward_marginal",
    "sample_stationary_reference",
    "sample_student_t4",
    "stable_tail_score",
    "student_t4_cf",
    "student_t4_density",
    "vp_tail_score",
]
