from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import ks_2samp

from levy_experiments.random import named_key
from levy_experiments.stable import (
    euler_stable_increment_scale,
    sample_symmetric_stable_1d,
)


def _sample(seed: int, label: str, alpha: float, count: int) -> np.ndarray:
    kernel = jax.jit(lambda key: sample_symmetric_stable_1d(key, alpha, (count,), jnp.float64))
    return np.asarray(kernel(named_key(seed, label)))


def test_seed_reproducibility_and_subkey_independence() -> None:
    first = _sample(19, "first", 1.5, 4096)
    repeated = _sample(19, "first", 1.5, 4096)
    second = _sample(19, "second", 1.5, 4096)
    different_seed = _sample(20, "first", 1.5, 4096)
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, second)
    assert not np.array_equal(first, different_seed)


def test_stable_empirical_characteristic_function_and_symmetry() -> None:
    count = 200_000
    values = _sample(31, "ecf", 1.5, count)
    frequencies = np.asarray([0.25, 0.5, 1.0, 2.0])
    empirical = np.mean(np.exp(1j * values[:, None] * frequencies[None, :]), axis=0)
    exact = np.exp(-(np.abs(frequencies) ** 1.5))
    tolerance = 5.0 / math.sqrt(count) + 2e-4
    assert np.max(np.abs(empirical - exact)) < tolerance
    assert abs(np.mean(values > 0.0) - 0.5) < 4.0 * math.sqrt(0.25 / count)
    assert np.isfinite(values).all()


def test_stability_by_addition_and_increment_scaling() -> None:
    count = 100_000
    first = _sample(41, "sum-a", 1.5, count)
    second = _sample(41, "sum-b", 1.5, count)
    direct = _sample(41, "sum-direct", 1.5, count)
    combined = (0.03 ** (1 / 1.5) * first + 0.08 ** (1 / 1.5) * second) / (0.11 ** (1 / 1.5))
    assert ks_2samp(combined, direct).statistic < 0.008
    assert math.isclose(
        euler_stable_increment_scale(1.5, 0.7, 0.125) ** 1.5,
        0.7 * 0.125,
        rel_tol=1e-14,
    )


def test_brownian_limit_has_manuscript_normalization() -> None:
    values = _sample(51, "alpha-two", 2.0, 150_000)
    assert abs(np.mean(values)) < 0.02
    assert abs(np.var(values) - 2.0) < 0.04
