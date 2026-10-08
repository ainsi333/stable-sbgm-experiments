from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.stats import levy_stable

from levy_experiments.config import load_config
from levy_experiments.scores.discrete import discrete_fractional_score
from levy_experiments.scores.stable_table import (
    build_stable_density_table,
    interpolate_log_density_numpy,
    three_term_asymptotic_density,
)

ROOT = Path(__file__).resolve().parents[2]


def _config_with_points(points: int):
    config = load_config(ROOT / "configs/smoke/exp3.toml")
    table = dataclasses.replace(
        config.score_table,
        points=points,
        workers=1,
        validation_rtol=0.02,
        validation_points=65,
    )
    return dataclasses.replace(config, score_table=table)


def test_scipy_density_matches_independent_fourier_quadrature() -> None:
    alpha = 1.5
    for point in (0.0, 0.1, 1.0, 5.0, 20.0):
        integral = (
            quad(
                lambda radius, location=point: math.exp(-(radius**alpha))
                * math.cos(radius * location),
                0.0,
                np.inf,
                epsabs=1e-12,
                epsrel=1e-12,
                limit=500,
            )[0]
            / math.pi
        )
        reference = levy_stable.pdf(point, alpha, 0.0)
        assert math.isclose(integral, reference, rel_tol=2e-10, abs_tol=2e-13)
    assert math.isclose(
        levy_stable.pdf(0.0, alpha, 0.0),
        math.gamma(1.0 / alpha) / (math.pi * alpha),
        rel_tol=1e-13,
    )


def test_table_converges_and_tail_seam_is_small() -> None:
    coarse = build_stable_density_table(_config_with_points(513))
    fine = build_stable_density_table(_config_with_points(2049))
    transformed = (np.arange(1000) + 0.37) / 1000 * math.log1p(60.0)
    points = np.expm1(transformed)
    exact = np.log(levy_stable.pdf(points, 1.5, 0.0))
    coarse_error = np.max(np.abs(interpolate_log_density_numpy(coarse, points) - exact))
    fine_error = np.max(np.abs(interpolate_log_density_numpy(fine, points) - exact))
    assert fine_error < coarse_error / 8.0
    reference_at_seam = levy_stable.pdf(60.0, 1.5, 0.0)
    tail_at_seam = three_term_asymptotic_density(np.asarray([60.0]), 1.5)[0]
    assert abs(tail_at_seam / reference_at_seam - 1.0) < 1e-7


def test_three_atom_score_matches_direct_scipy_weights() -> None:
    config = _config_with_points(2049)
    table = build_stable_density_table(config)
    values = np.asarray([-20.0, -2.0, 0.0, 1.0, 13.0], dtype=np.float64)
    time = 0.5
    actual = np.asarray(
        discrete_fractional_score(
            jnp.asarray(values),
            time,
            alpha=1.5,
            beta=1.0,
            atoms=config.nonstationary.atoms,
            weights=config.nonstationary.weights,
            table=table,
            dtype=jnp.float64,
        )
    )
    a = math.exp(-time / 1.5)
    gamma_power = -math.expm1(-time)
    gamma = gamma_power ** (1 / 1.5)
    atoms = np.asarray(config.nonstationary.atoms)
    weights = np.asarray(config.nonstationary.weights)
    likelihood = levy_stable.pdf((values[:, None] - a * atoms[None, :]) / gamma, 1.5, 0.0)
    posterior = likelihood * weights[None, :]
    posterior /= posterior.sum(axis=1, keepdims=True)
    denoiser = posterior @ atoms
    expected = -(values - a * denoiser) / (1.5 * gamma_power)
    assert np.max(np.abs(actual - expected)) < 2e-5
