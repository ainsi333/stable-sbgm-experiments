"""Memory-bounded empirical characteristic-function calculations."""

from __future__ import annotations

import math

import numpy as np

from ..errors import ConfigurationError


def empirical_cf_1d(samples: np.ndarray, frequencies: np.ndarray, *, chunk_size: int) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    u = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    if values.size == 0 or chunk_size <= 0:
        raise ConfigurationError("ECF requires samples and a positive chunk size")
    accumulator = np.zeros(u.size, dtype=np.complex128)
    for start in range(0, values.size, chunk_size):
        chunk = values[start : start + chunk_size]
        accumulator += np.exp(1j * chunk[:, None] * u[None, :]).sum(axis=0)
    return accumulator / values.size


def empirical_cf_joint(
    first: np.ndarray,
    second: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    u_values = np.asarray(u, dtype=np.float64).reshape(-1)
    v_values = np.asarray(v, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or u_values.shape != v_values.shape or x.size == 0:
        raise ConfigurationError("joint ECF inputs have incompatible shapes")
    accumulator = np.zeros(u_values.size, dtype=np.complex128)
    for start in range(0, x.size, chunk_size):
        x_chunk = x[start : start + chunk_size]
        y_chunk = y[start : start + chunk_size]
        phase = x_chunk[:, None] * u_values[None, :] + y_chunk[:, None] * v_values[None, :]
        accumulator += np.exp(1j * phase).sum(axis=0)
    return accumulator / x.size


def weighted_cf_rmse(empirical: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> float:
    empirical_values = np.asarray(empirical, dtype=np.complex128)
    reference_values = np.asarray(reference, dtype=np.complex128)
    weight_values = np.asarray(weights, dtype=np.float64)
    if (
        empirical_values.shape != reference_values.shape
        or empirical_values.shape != weight_values.shape
    ):
        raise ConfigurationError("CF arrays and weights must have identical shapes")
    if np.any(weight_values < 0.0) or not np.any(weight_values > 0.0):
        raise ConfigurationError("CF weights must be non-negative and nonzero")
    return float(
        np.sqrt(
            np.sum(weight_values * np.abs(empirical_values - reference_values) ** 2)
            / np.sum(weight_values)
        )
    )


def hoeffding_complex_radius(
    *, sample_count: int, frequency_count: int, family_error: float = 0.05
) -> float:
    """Simultaneous componentwise Hoeffding radius for a complex ECF grid."""

    if sample_count <= 0 or frequency_count <= 0 or not 0.0 < family_error < 1.0:
        raise ConfigurationError("invalid Hoeffding-bound arguments")
    return math.sqrt(2.0 * math.log(4.0 * frequency_count / family_error) / sample_count)
