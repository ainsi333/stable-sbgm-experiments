from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import t

from experiments.exp1.metrics import (
    TailMetricError,
    gpd_at_k,
    hill_at_k,
    mean_excess_sufficient,
    pickands_at_k,
    pooled_mean_excess,
    power_law_fit,
    sorted_radial,
    student_abs_quantiles,
    tail_constant_at_k,
    tail_count,
)


def test_student_radial_quantiles_are_exact_and_tail_counts_are_pinned() -> None:
    probabilities = np.asarray([0.9, 0.99, 0.999])
    for nu in (1.2, 1.5, 3.0):
        values = student_abs_quantiles(probabilities, nu)
        assert np.allclose(values, t.ppf((1.0 + probabilities) / 2.0, nu))
        assert np.allclose(2.0 * t.cdf(values, nu) - 1.0, probabilities, atol=3.0e-12)
    assert tail_count(262_144, 0.006) == 1_572


def test_pareto_hill_pickands_gpd_and_constant_have_correct_orientation() -> None:
    rng = np.random.default_rng(7421)
    alpha = 1.5
    sample = rng.pareto(alpha, 300_000) + 1.0
    ordered = sorted_radial(sample)
    xi, alpha_hat, _ = hill_at_k(ordered, 3_000)
    pickands, _, _ = pickands_at_k(ordered, 3_000)
    gpd, _, _ = gpd_at_k(ordered, 3_000)
    constant, _ = tail_constant_at_k(ordered, 3_000, alpha)
    assert abs(xi - 1.0 / alpha) < 0.04
    assert abs(alpha_hat - alpha) < 0.10
    assert abs(pickands - 1.0 / alpha) < 0.12
    assert abs(gpd - 1.0 / alpha) < 0.10
    assert abs(constant - 1.0) < 0.12


def test_scale_permutation_pooling_and_synthetic_slopes() -> None:
    rng = np.random.default_rng(93)
    sample = rng.pareto(1.5, 20_000) + 1.0
    shuffled = sample[rng.permutation(sample.size)]
    assert hill_at_k(sorted_radial(sample), 500) == hill_at_k(sorted_radial(shuffled), 500)
    c1 = tail_constant_at_k(sorted_radial(sample), 500, 1.5)[0]
    c2 = tail_constant_at_k(sorted_radial(3.0 * sample), 500, 1.5)[0]
    assert math.isclose(c2 / c1, 3.0**1.5, rel_tol=2.0e-14)
    thresholds = np.asarray([2.0, 4.0])
    rows = mean_excess_sufficient(sample, thresholds)
    pooled = pooled_mean_excess(
        np.asarray([rows[0]["exceedance_sum"]]),
        np.asarray([rows[0]["exceedance_count"]]),
        thresholds[0],
    )
    direct = sample[sample > thresholds[0]].mean() / thresholds[0] - 1.0
    assert math.isclose(pooled, direct, rel_tol=1.0e-14)
    masses = np.geomspace(0.02, 0.001, 13)
    fit = power_law_fit(masses, 2.5 * masses ** (-1.0 / 3.0), lower=0.001, upper=0.01)
    assert math.isclose(fit.slope, -1.0 / 3.0, abs_tol=1.0e-13)


def test_invalid_extremes_are_never_clipped_or_silently_fitted() -> None:
    with pytest.raises(TailMetricError, match="non-finite"):
        sorted_radial(np.asarray([1.0, np.inf]))
    tied = np.ones(100)
    with pytest.raises(TailMetricError, match="spacings"):
        pickands_at_k(tied, 10)
    with pytest.raises(TailMetricError, match="strictly positive"):
        gpd_at_k(tied, 10)
    with pytest.raises(TailMetricError, match="too few"):
        power_law_fit(
            np.asarray([0.01, 0.005]),
            np.asarray([1.0, 2.0]),
            lower=0.001,
            upper=0.02,
        )
