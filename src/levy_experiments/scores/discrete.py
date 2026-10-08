"""Exact finite-mixture denoiser and fractional score."""

from __future__ import annotations

import math

from .stable_table import StableDensityTable, interpolate_log_density_jax


def forward_scales(time: float, *, alpha: float, beta: float) -> tuple[float, float]:
    """Return a(t) and gamma(t)^alpha for the constant-beta SP-SDE."""

    a = math.exp(-beta * time / alpha)
    gamma_power = -math.expm1(-beta * time)
    return a, gamma_power


def discrete_fractional_score(
    values,
    time,
    *,
    alpha: float,
    beta: float,
    atoms,
    weights,
    table: StableDensityTable,
    dtype,
):
    """Evaluate the exact three-atom formula using the validated density table."""

    import jax.numpy as jnp

    alpha_value = jnp.asarray(alpha, dtype=dtype)
    beta_value = jnp.asarray(beta, dtype=dtype)
    time_value = jnp.asarray(time, dtype=dtype)
    a = jnp.exp(-beta_value * time_value / alpha_value)
    gamma_power = -jnp.expm1(-beta_value * time_value)
    gamma = gamma_power ** (1.0 / alpha_value)
    atom_values = jnp.asarray(atoms, dtype=dtype)
    mixture_weights = jnp.asarray(weights, dtype=dtype)
    standardized = (values[..., None] - a * atom_values) / gamma
    log_likelihood = interpolate_log_density_jax(table, standardized, dtype)
    logits = jnp.log(mixture_weights) + log_likelihood
    centered_logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    posterior = jnp.exp(centered_logits)
    posterior = posterior / jnp.sum(posterior, axis=-1, keepdims=True)
    denoiser = jnp.sum(posterior * atom_values, axis=-1)
    return -(values - a * denoiser) / (alpha_value * gamma_power)
