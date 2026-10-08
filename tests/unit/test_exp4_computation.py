from __future__ import annotations

import math

import numpy as np

from experiments.exp4.computation import forward_scales, spectral_denoiser_profile
from experiments.exp4.config import SpectralSettings
from experiments.exp4.run import _figure_bytes, render_figure


def _settings() -> SpectralSettings:
    return SpectralSettings(
        fft_half_width=64.0,
        fft_points=16384,
        spatial_max=4.0,
        density_floor=1.0e-12,
    )


def test_forward_scales_use_distinct_alpha_and_brownian_attenuation() -> None:
    time = 0.7
    stable_a, stable_scale = forward_scales("stable", time, alpha=1.5, beta=1.0)
    vp_a, vp_scale = forward_scales("vp", time, alpha=1.5, beta=1.0)
    innovation = 1.0 - math.exp(-time)

    assert stable_a == math.exp(-time / 1.5)
    assert stable_scale == 1.5 * innovation
    assert vp_a == math.exp(-time / 2.0)
    assert vp_scale == innovation


def test_vp_stationary_gaussian_has_exact_linear_denoiser() -> None:
    time = 0.7
    profile = spectral_denoiser_profile(
        "vp",
        time,
        _settings(),
        alpha=1.5,
        beta=1.0,
        characteristic=lambda frequency: np.exp(-0.5 * frequency * frequency),
    )
    expected = math.exp(-time / 2.0)

    assert np.max(np.abs(profile.derivative - expected)) < 2.0e-6
    assert abs(profile.ct - expected) < 2.0e-6


def test_stationary_stable_case_has_analytic_denoiser_slope() -> None:
    alpha = 1.5
    time = 0.7
    profile = spectral_denoiser_profile(
        "stable",
        time,
        _settings(),
        alpha=alpha,
        beta=1.0,
        characteristic=lambda frequency: np.exp(-np.abs(frequency) ** alpha),
    )
    expected = math.exp(-time * (alpha - 1.0) / alpha)

    central = profile.x <= 1.0
    assert np.max(np.abs(profile.derivative[central] - expected)) < 2.0e-4
    assert abs(profile.derivative[0] - expected) < 2.0e-4


def test_exp4_figure_bytes_are_deterministic() -> None:
    rows = []
    for model, values in (("stable", (0.9, 0.5)), ("vp", (1.1, 2.0))):
        for time, ct in zip((0.1, 1.0), values, strict=True):
            rows.append(
                {
                    "model": model,
                    "time": time,
                    "ct_truncated": ct,
                    "brownian_reference": math.exp(time / 2.0),
                }
            )
    first = render_figure(rows)
    second = render_figure(rows)
    try:
        assert _figure_bytes(first, ".pdf") == _figure_bytes(second, ".pdf")
        assert _figure_bytes(first, ".png") == _figure_bytes(second, ".png")
    finally:
        import matplotlib.pyplot as plt

        plt.close(first)
        plt.close(second)
