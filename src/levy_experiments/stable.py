"""Backend-native one-dimensional symmetric alpha-stable sampling."""

from __future__ import annotations

import math

from .errors import ConfigurationError


def _open_uniform(key, shape: tuple[int, ...], dtype):
    """Generate a discrete uniform variate strictly inside (0, 1)."""

    import jax.numpy as jnp
    from jax import random

    raw = random.uniform(key, shape=shape, minval=0.0, maxval=1.0, dtype=dtype)
    half_eps = jnp.asarray(jnp.finfo(dtype).eps / 2.0, dtype=dtype)
    return (raw + half_eps) / (jnp.asarray(1.0, dtype=dtype) + 2.0 * half_eps)


def sample_symmetric_stable_1d(key, alpha: float, shape: tuple[int, ...], dtype):
    """Sample a standard symmetric stable law using the CMS formula.

    The name deliberately states ``1d``: independent calls along coordinates do
    not define an isotropic multivariate stable law when alpha is below two.
    """

    if not 0.0 < alpha <= 2.0 or math.isclose(alpha, 1.0):
        raise ConfigurationError("CMS implementation requires alpha in (0,2], alpha != 1")
    import jax
    import jax.numpy as jnp

    key_u, key_w = jax.random.split(key)
    if math.isclose(alpha, 2.0, rel_tol=0.0, abs_tol=1e-14):
        return jnp.sqrt(jnp.asarray(2.0, dtype=dtype)) * jax.random.normal(
            key_u, shape=shape, dtype=dtype
        )
    open_u = _open_uniform(key_u, shape, dtype)
    open_w = _open_uniform(key_w, shape, dtype)
    angle = jnp.asarray(math.pi, dtype=dtype) * (open_u - jnp.asarray(0.5, dtype=dtype))
    exponential = -jnp.log(open_w)
    alpha_value = jnp.asarray(alpha, dtype=dtype)
    first = jnp.sin(alpha_value * angle) / jnp.power(jnp.cos(angle), 1.0 / alpha_value)
    second = jnp.power(
        jnp.cos((1.0 - alpha_value) * angle) / exponential,
        (1.0 - alpha_value) / alpha_value,
    )
    return first * second


def euler_stable_increment_scale(alpha: float, intensity: float, step: float) -> float:
    """Scale of a constant-coefficient stable increment over one Euler interval."""

    if alpha <= 0.0 or intensity < 0.0 or step < 0.0:
        raise ConfigurationError("alpha, intensity and step must define a valid increment")
    return (intensity * step) ** (1.0 / alpha)
