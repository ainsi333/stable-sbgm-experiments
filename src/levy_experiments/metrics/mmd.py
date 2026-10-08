"""Permutation-invariant random-feature MMD for a bounded Gaussian kernel."""

from __future__ import annotations

import numpy as np

from ..errors import ConfigurationError

DEFAULT_FEATURE_COUNT = 64
DEFAULT_CHUNK_SIZE = 16_384


def _feature_sum(
    values: np.ndarray,
    frequencies: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    """Accumulate paired cosine/sine features without materializing an ``n x D`` array."""

    feature_count = frequencies.shape[0]
    scale = feature_count**-0.5
    result = np.zeros(2 * feature_count, dtype=np.float64)
    for start in range(0, values.shape[0], chunk_size):
        projection = values[start : start + chunk_size] @ frequencies.T
        result[:feature_count] += np.sum(np.cos(projection), axis=0) * scale
        result[feature_count:] += np.sum(np.sin(projection), axis=0) * scale
    return result


def linear_mmd2(
    first: np.ndarray,
    second: np.ndarray,
    *,
    bandwidth: float,
    pairing_seed: int = 0,
    feature_count: int = DEFAULT_FEATURE_COUNT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> float:
    """Return an unbiased MMD-squared U-statistic for a fixed RFF kernel.

    ``feature_count`` frequencies are drawn deterministically from the spectral
    law of the Gaussian kernel using ``pairing_seed``.  Paired cosine/sine
    features define the positive-definite bounded approximation

    ``k_D(x,y) = D**-1 sum_j cos(omega_j dot (x-y))``.

    For that fixed kernel, the returned two-sample U-statistic is unbiased and
    permutation-invariant up to floating-point summation.  It can be negative
    and is never clipped.  The historical function name is retained for API
    compatibility; the estimator no longer depends on arbitrary row pairings.
    """

    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[1] != y.shape[1]
        or x.shape[0] < 2
        or y.shape[0] < 2
        or bandwidth <= 0.0
        or not isinstance(pairing_seed, int)
        or pairing_seed < 0
        or not isinstance(feature_count, int)
        or feature_count <= 0
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ConfigurationError(
            "RFF MMD requires two 2D arrays with a common dimension, n,m>=2, "
            "positive bandwidth/feature_count/chunk_size and a non-negative integer seed"
        )
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ConfigurationError("RFF MMD inputs must be finite")

    rng = np.random.default_rng(pairing_seed)
    frequencies = rng.normal(
        loc=0.0,
        scale=1.0 / bandwidth,
        size=(feature_count, x.shape[1]),
    )
    sum_x = _feature_sum(x, frequencies, chunk_size=chunk_size)
    sum_y = _feature_sum(y, frequencies, chunk_size=chunk_size)
    n = x.shape[0]
    m = y.shape[0]

    # Paired sine/cosine features have squared norm exactly one in real
    # arithmetic, so the diagonal correction is n (respectively m).
    within_x = (float(sum_x @ sum_x) - n) / (n * (n - 1))
    within_y = (float(sum_y @ sum_y) - m) / (m * (m - 1))
    between = float(sum_x @ sum_y) / (n * m)
    return within_x + within_y - 2.0 * between
