"""Deterministic Fourier evaluation of the forward denoiser derivative."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from experiments.exp2.score_tables import ScoreTable, interpolate_score_numpy
from experiments.exp2.theory import Model, forward_marginal_cf
from levy_experiments.errors import ArtifactError, ConfigurationError

from .config import SpectralSettings

CharacteristicFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DenoiserProfile:
    """Resolved derivative profile on the non-negative half-line."""

    model: Model
    time: float
    x: np.ndarray
    derivative: np.ndarray
    density: np.ndarray
    ct: float
    maximizer: float
    boundary_abs_derivative: float
    min_density: float
    cutoff_characteristic: float
    imaginary_residual: float

    def truncated_ct(self, radius: float) -> float:
        if not math.isfinite(radius) or not self.x[0] <= radius <= self.x[-1]:
            raise ConfigurationError("truncation radius lies outside the computed profile")
        count = int(np.searchsorted(self.x, radius, side="right"))
        return float(np.max(np.abs(self.derivative[:count])))


def forward_scales(model: Model, time: float, *, alpha: float, beta: float) -> tuple[float, float]:
    """Return ``a(t)`` and the score multiplier in the Tweedie denoiser."""

    if model not in {"stable", "vp"}:
        raise ConfigurationError(f"unknown model: {model!r}")
    if not math.isfinite(time) or time <= 0.0:
        raise ConfigurationError("the denoiser derivative requires positive finite time")
    if not math.isfinite(beta) or beta <= 0.0:
        raise ConfigurationError("beta must be finite and positive")
    if not math.isfinite(alpha) or not 1.0 < alpha < 2.0:
        raise ConfigurationError("alpha must lie in (1,2)")
    exponent = alpha if model == "stable" else 2.0
    attenuation = math.exp(-beta * time / exponent)
    innovation_power = -math.expm1(-beta * time)
    score_scale = alpha * innovation_power if model == "stable" else innovation_power
    return attenuation, score_scale


def spectral_denoiser_profile(
    model: Model,
    time: float,
    settings: SpectralSettings,
    *,
    alpha: float,
    beta: float,
    characteristic: CharacteristicFunction,
    cutoff_tolerance: float = 1.0e-10,
) -> DenoiserProfile:
    """Evaluate ``m_t'`` without finite-differencing a score table.

    The Fourier convention is ``hat f(xi)=integral f(x) exp(i x xi) dx``.
    Hence spatial differentiation has multiplier ``-i xi``.  The stable
    fractional score numerator has multiplier ``-i xi |xi|^(alpha-2)``;
    the VP numerator uses ``-i xi``.  Density, numerator, and both derivatives
    are transformed analytically before taking their quotient.
    """

    settings.validate(f"{model}_spectral")
    attenuation, score_scale = forward_scales(model, time, alpha=alpha, beta=beta)
    count = settings.fft_points
    dx = 2.0 * settings.fft_half_width / count
    points = math.floor(settings.spatial_max / dx + 1.0e-12) + 1
    x = np.arange(points, dtype=np.float64) * dx
    frequencies = 2.0 * math.pi * np.fft.fftfreq(count, d=dx)
    characteristic_values = np.asarray(characteristic(frequencies), dtype=np.float64)
    if characteristic_values.shape != frequencies.shape or np.any(
        ~np.isfinite(characteristic_values)
    ):
        raise ArtifactError("the characteristic function returned invalid values")
    cutoff = float(abs(characteristic_values[int(np.argmax(np.abs(frequencies)))]))
    if cutoff > cutoff_tolerance:
        raise ArtifactError(
            "the Fourier cutoff is unresolved: "
            f"|phi(xi_max)|={cutoff:.3e} > {cutoff_tolerance:.3e}"
        )

    imaginary_residual = 0.0

    def transform(spectrum: np.ndarray) -> np.ndarray:
        nonlocal imaginary_residual
        transformed = np.fft.fft(spectrum)[:points] / (count * dx)
        imaginary_residual = max(
            imaginary_residual,
            float(np.max(np.abs(transformed.imag))),
        )
        return np.asarray(transformed.real, dtype=np.float64)

    spatial_multiplier = -1j * frequencies
    density = transform(characteristic_values)
    density_derivative = transform(spatial_multiplier * characteristic_values)
    if model == "stable":
        score_multiplier = np.zeros(count, dtype=np.complex128)
        nonzero = frequencies != 0.0
        score_multiplier[nonzero] = (
            -1j
            * frequencies[nonzero]
            * np.abs(frequencies[nonzero]) ** (alpha - 2.0)
        )
    else:
        score_multiplier = spatial_multiplier
    score_numerator = transform(score_multiplier * characteristic_values)
    numerator_derivative = transform(
        spatial_multiplier * score_multiplier * characteristic_values
    )
    if imaginary_residual > 1.0e-9:
        raise ArtifactError(f"Fourier symmetry residual is too large: {imaginary_residual:.3e}")
    if np.any(~np.isfinite(density)) or np.any(density <= settings.density_floor):
        index = int(np.argmin(density))
        raise ArtifactError(
            "density is unresolved on the declared spatial domain: "
            f"p({x[index]:.8g})={density[index]:.3e}, floor={settings.density_floor:.3e}"
        )
    score_derivative = (
        numerator_derivative / density
        - score_numerator * density_derivative / np.square(density)
    )
    derivative = (1.0 + score_scale * score_derivative) / attenuation
    if np.any(~np.isfinite(derivative)):
        raise ArtifactError("the denoiser derivative contains a non-finite value")
    absolute = np.abs(derivative)
    maximizing_index = int(np.argmax(absolute))
    x.setflags(write=False)
    derivative.setflags(write=False)
    density.setflags(write=False)
    return DenoiserProfile(
        model=model,
        time=float(time),
        x=x,
        derivative=derivative,
        density=density,
        ct=float(absolute[maximizing_index]),
        maximizer=float(x[maximizing_index]),
        boundary_abs_derivative=float(absolute[-1]),
        min_density=float(np.min(density)),
        cutoff_characteristic=cutoff,
        imaginary_residual=imaginary_residual,
    )


def student_denoiser_profile(
    model: Model,
    time: float,
    settings: SpectralSettings,
    *,
    alpha: float,
    beta: float,
) -> DenoiserProfile:
    """Specialize the spectral evaluator to the Experiment 2 Student-t(4) law."""

    return spectral_denoiser_profile(
        model,
        time,
        settings,
        alpha=alpha,
        beta=beta,
        characteristic=lambda frequency: np.asarray(
            forward_marginal_cf(
                frequency,
                time,
                model,
                alpha=alpha,
                beta=beta,
            ),
            dtype=np.float64,
        ),
    )


def score_table_ct_proxy(
    table: ScoreTable,
    time: float,
    *,
    radius: float,
    points: int = 8193,
) -> float:
    """Differentiate the cached regular Tweedie residual for a source check.

    This is intentionally secondary: piecewise interpolation makes a stored
    score table less accurate for derivatives than the direct spectral
    calculation.  It is used only as an independent overlap diagnostic.
    """

    if not table.time_nodes[0] <= time <= table.time_nodes[-1]:
        raise ConfigurationError("source-table check time lies outside the table")
    if not math.isfinite(radius) or not 0.0 < radius <= table.space_nodes[-1]:
        raise ConfigurationError("source-table check radius lies outside the table")
    if not isinstance(points, int) or isinstance(points, bool) or points < 257:
        raise ConfigurationError("source-table derivative grid requires at least 257 points")
    alpha = float(table.metadata["alpha"])
    beta = float(table.metadata["beta"])
    attenuation, score_scale = forward_scales(
        table.model,
        time,
        alpha=alpha,
        beta=beta,
    )
    x = np.linspace(0.0, radius, points, dtype=np.float64)
    score = np.asarray(interpolate_score_numpy(x, time, table), dtype=np.float64)
    denoiser = (x + score_scale * score) / attenuation
    derivative = np.gradient(denoiser, x, edge_order=2)
    if np.any(~np.isfinite(derivative)):
        raise ArtifactError("source score table produced a non-finite derivative proxy")
    return float(np.max(np.abs(derivative)))


def truncated_values(profile: DenoiserProfile, radii: Sequence[float]) -> tuple[float, ...]:
    """Return nested-domain suprema in the requested deterministic order."""

    return tuple(profile.truncated_ct(float(radius)) for radius in radii)


__all__ = [
    "DenoiserProfile",
    "forward_scales",
    "score_table_ct_proxy",
    "spectral_denoiser_profile",
    "student_denoiser_profile",
    "truncated_values",
]
