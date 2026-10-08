from __future__ import annotations

import numpy as np

from levy_experiments.metrics import empirical_wasserstein_1d, linear_mmd2


def test_empirical_w1_is_exact_by_sorting() -> None:
    first = np.asarray([0.0, 2.0, 1.0])
    second = np.asarray([1.0, 4.0, 2.0])
    assert empirical_wasserstein_1d(first, second) == 4.0 / 3.0
    assert empirical_wasserstein_1d(first, first) == 0.0


def test_bounded_kernel_mmd_is_finite_and_unclipped() -> None:
    first = np.arange(20, dtype=np.float64)[:, None] / 10.0
    second = first + 0.3
    estimate = linear_mmd2(first, second, bandwidth=1.0)
    assert np.isfinite(estimate)
    assert -2.0 <= estimate <= 2.0


def test_rff_mmd_is_invariant_to_input_permutations_up_to_roundoff() -> None:
    rng = np.random.default_rng(20260823)
    first = rng.normal(size=(200, 2))
    second = rng.standard_t(4.0, size=(200, 2))
    expected = linear_mmd2(first, second, bandwidth=1.7)

    actual = linear_mmd2(
        first[rng.permutation(first.shape[0])],
        second[rng.permutation(second.shape[0])],
        bandwidth=1.7,
    )
    assert np.isclose(actual, expected, rtol=2.0e-13, atol=5.0e-14)


def test_rff_mmd_matches_direct_fixed_feature_u_statistic() -> None:
    first = np.asarray([[-1.0], [0.0], [0.5], [2.0]])
    second = np.asarray([[-0.5], [0.25], [1.0]])
    bandwidth = 1.3
    seed = 17
    feature_count = 11
    rng = np.random.default_rng(seed)
    frequencies = rng.normal(0.0, 1.0 / bandwidth, size=(feature_count, 1))

    def kernel(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(np.cos(frequencies[:, 0] * (x[0] - y[0]))))

    within_first = sum(
        kernel(first[i], first[j])
        for i in range(first.shape[0])
        for j in range(first.shape[0])
        if i != j
    ) / (first.shape[0] * (first.shape[0] - 1))
    within_second = sum(
        kernel(second[i], second[j])
        for i in range(second.shape[0])
        for j in range(second.shape[0])
        if i != j
    ) / (second.shape[0] * (second.shape[0] - 1))
    between = sum(
        kernel(x, y) for x in first for y in second
    ) / (first.shape[0] * second.shape[0])
    expected = within_first + within_second - 2.0 * between

    actual = linear_mmd2(
        first,
        second,
        bandwidth=bandwidth,
        pairing_seed=seed,
        feature_count=feature_count,
        chunk_size=2,
    )
    assert np.isclose(actual, expected, rtol=2.0e-13, atol=5.0e-14)
