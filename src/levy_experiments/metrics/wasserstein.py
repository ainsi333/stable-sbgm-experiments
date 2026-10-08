"""Exact empirical one-dimensional optimal transport by sorting."""

from __future__ import annotations

import numpy as np

from ..errors import ConfigurationError


def empirical_wasserstein_1d(first: np.ndarray, second: np.ndarray, *, order: float = 1.0) -> float:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    if x.size == 0 or x.size != y.size:
        raise ConfigurationError("empirical 1D Wasserstein requires equal non-empty sample sizes")
    if order < 1.0:
        raise ConfigurationError("Wasserstein order must be at least one")
    differences = np.abs(np.sort(x) - np.sort(y)) ** order
    return float(np.mean(differences) ** (1.0 / order))
