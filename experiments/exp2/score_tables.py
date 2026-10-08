"""Deterministic float64 spectral score tables for Experiment 2.

Tables are model-specific: the stable model stores the fractional score with
Fourier multiplier ``-i xi |xi|**(alpha-2)``, whereas the VP model stores the
ordinary score with multiplier ``-i xi``.  Only non-negative spatial nodes are
stored; symmetry is imposed exactly during interpolation.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import quad

from levy_experiments.errors import ArtifactError, ConfigurationError
from levy_experiments.scores.stable_table import three_term_asymptotic_coefficients

from .theory import (
    STUDENT_VARIANCE,
    Model,
    forward_marginal_cf,
    stable_tail_score,
    vp_tail_score,
)


@dataclass(frozen=True)
class ScoreTable:
    """Portable, content-addressed table consumed by the simulation drivers."""

    model: Model
    time_nodes: np.ndarray
    space_nodes: np.ndarray
    scores: np.ndarray
    table_hash: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _BuildSpec:
    model: Model
    alpha: float
    beta: float
    time_nodes: np.ndarray
    space_points: int
    x_max: float
    fft_half_width: float
    fft_points: int
    construction_workers: int
    blend_start: float
    blend_end: float
    validation_points: int
    validation_x_max: float
    validation_rtol: float
    validation_atol: float

    def payload(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "alpha": self.alpha,
            "beta": self.beta,
            "time_nodes": self.time_nodes.tolist(),
            "space_points": self.space_points,
            "x_max": self.x_max,
            "fft_half_width": self.fft_half_width,
            "fft_points": self.fft_points,
            "blend_start": self.blend_start,
            "blend_end": self.blend_end,
            "validation_points": self.validation_points,
            "validation_x_max": self.validation_x_max,
            "validation_rtol": self.validation_rtol,
            "validation_atol": self.validation_atol,
        }


_MISSING = object()
_NO_DEFAULT = object()


def _member(container: Any, name: str, default: Any = _NO_DEFAULT) -> Any:
    if isinstance(container, Mapping):
        if name in container:
            return container[name]
    elif hasattr(container, name):
        return getattr(container, name)
    if default is _NO_DEFAULT:
        raise ConfigurationError(f"missing Experiment 2 configuration field: {name}")
    return default


def _section(config: Any, name: str) -> Any:
    return _member(config, name)


def _model_setting(
    settings: Any, model: Model, names: Sequence[str], default: Any = _NO_DEFAULT
) -> Any:
    for name in names:
        for candidate in (f"{model}_{name}", f"{name}_{model}"):
            value = _member(settings, candidate, _MISSING)
            if value is not _MISSING:
                return value
        value = _member(settings, name, _MISSING)
        if value is _MISSING:
            continue
        if isinstance(value, Mapping):
            if model in value:
                return value[model]
            continue
        if dataclasses.is_dataclass(value) or hasattr(value, model):
            nested = _member(value, model, _MISSING)
            if nested is not _MISSING:
                return nested
        return value
    if default is _NO_DEFAULT:
        joined = ", ".join(names)
        raise ConfigurationError(f"missing {model}-specific score-table setting: {joined}")
    return default


def _positive_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ConfigurationError(f"{name} must be a finite positive number")
    return result


def _positive_int(name: str, value: Any) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or int(value) <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return int(value)


def _derive_time_nodes(config: Any, settings: Any, model: Model) -> np.ndarray:
    explicit = _model_setting(settings, model, ("time_nodes", "times"), _MISSING)
    if explicit is not _MISSING:
        values = np.asarray(explicit, dtype=np.float64)
    else:
        experiment = _section(config, "experiment")
        epsilon = _positive_float("experiment.epsilon", _member(experiment, "epsilon"))
        design = _section(config, "design")
        horizon_values = [
            _positive_float("design.refinement_horizon", _member(design, "refinement_horizon"))
        ]
        for item in _member(design, "horizon_sweep", ()):
            horizon_values.append(
                _positive_float("design.horizon_sweep.horizon", _member(item, "horizon"))
            )
        horizon = max(horizon_values)
        count = _model_setting(settings, model, ("time_points", "temporal_points"), _MISSING)
        if count is _MISSING:
            steps = _member(experiment, "steps", _MISSING)
            if steps is not _MISSING:
                candidates = np.asarray(steps, dtype=np.int64).reshape(-1)
                count = int(np.max(candidates)) + 1
            else:
                count = 257
        values = np.geomspace(epsilon, horizon, _positive_int("time_points", count))
    if values.ndim != 1 or values.size < 2:
        raise ConfigurationError(
            "score_table.time_nodes must be a one-dimensional array of size >= 2"
        )
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0) or np.any(np.diff(values) <= 0.0):
        raise ConfigurationError(
            "score-table times must be finite, positive, and strictly increasing"
        )
    return values


def _resolve_build_spec(config: Any, model: str) -> _BuildSpec:
    if model not in {"stable", "vp"}:
        raise ConfigurationError(f"unknown Experiment 2 model: {model!r}")
    checked_model: Model = model  # type: ignore[assignment]
    experiment = _section(config, "experiment")
    settings = _section(config, "score_table")
    stable_settings = _section(config, "stable")
    alpha = float(_member(stable_settings, "alpha"))
    beta = _positive_float("experiment.beta", _member(experiment, "beta", 1.0))
    if checked_model == "stable" and not math.isclose(alpha, 1.5, rel_tol=0.0, abs_tol=1.0e-12):
        raise ConfigurationError("the validated Experiment 2 stable table fixes alpha=1.5")
    time_nodes = _derive_time_nodes(config, settings, checked_model)
    defaults = {
        "stable": {"x_max": 500.0, "fft_half_width": 32768.0, "fft_points": 2**22},
        "vp": {"x_max": 20.0, "fft_half_width": 512.0, "fft_points": 2**18},
    }[checked_model]
    space_points = _positive_int(
        "score_table.space_points",
        _model_setting(settings, checked_model, ("space_points", "spatial_points", "points"), 8193),
    )
    if space_points < 3:
        raise ConfigurationError("score_table.space_points must be at least 3")
    x_max = _positive_float(
        "score_table.x_max",
        _model_setting(settings, checked_model, ("x_max",), defaults["x_max"]),
    )
    fft_half_width = _positive_float(
        "score_table.fft_half_width",
        _model_setting(
            settings,
            checked_model,
            ("fft_half_width", "spectral_half_width", "tail_l"),
            defaults["fft_half_width"],
        ),
    )
    fft_points = _positive_int(
        "score_table.fft_points",
        _model_setting(
            settings, checked_model, ("fft_points", "spectral_points"), defaults["fft_points"]
        ),
    )
    if fft_points < 1024 or fft_points % 2 or fft_points & (fft_points - 1):
        raise ConfigurationError("score_table.fft_points must be an even power of two >= 1024")
    default_workers = min(8, os.cpu_count() or 1, time_nodes.size)
    construction_workers = _positive_int(
        "score_table.workers",
        _model_setting(
            settings, checked_model, ("workers", "construction_workers"), default_workers
        ),
    )
    construction_workers = min(construction_workers, time_nodes.size)
    dx = 2.0 * fft_half_width / fft_points
    if x_max >= fft_half_width - 2.0 * dx:
        raise ConfigurationError("x_max must lie strictly inside the FFT spatial domain")
    blend_start = _positive_float(
        "score_table.blend_start",
        _model_setting(settings, checked_model, ("blend_start",)),
    )
    blend_end = _positive_float(
        "score_table.blend_end",
        _model_setting(settings, checked_model, ("blend_end",)),
    )
    if not 0.0 < blend_start < blend_end <= x_max:
        raise ConfigurationError("tail blend must satisfy 0 < start < end <= x_max")
    validation_points = _positive_int(
        "score_table.validation_points",
        _model_setting(settings, checked_model, ("validation_points",), 12),
    )
    default_validation_x = min(blend_start * 0.75, 20.0 if checked_model == "stable" else 6.0)
    validation_x_max = _positive_float(
        "score_table.validation_x_max",
        _model_setting(settings, checked_model, ("validation_x_max",), default_validation_x),
    )
    if validation_x_max >= x_max:
        raise ConfigurationError("validation_x_max must be strictly below x_max")
    tier = str(_member(experiment, "tier", "final"))
    tier_rtol = {
        "smoke": 2.0e-2,
        "pilot": 3.0e-4,
        "final": 2.0e-4 if checked_model == "stable" else 5.0e-5,
    }
    default_rtol = tier_rtol.get(tier, 2.0e-4 if checked_model == "stable" else 5.0e-5)
    validation_rtol = _positive_float(
        "score_table.validation_rtol",
        _model_setting(settings, checked_model, ("validation_rtol",), default_rtol),
    )
    validation_atol = float(_model_setting(settings, checked_model, ("validation_atol",), 2.0e-7))
    if not math.isfinite(validation_atol) or validation_atol < 0.0:
        raise ConfigurationError("score_table.validation_atol must be finite and non-negative")
    return _BuildSpec(
        model=checked_model,
        alpha=alpha,
        beta=beta,
        time_nodes=time_nodes,
        space_points=space_points,
        x_max=x_max,
        fft_half_width=fft_half_width,
        fft_points=fft_points,
        construction_workers=construction_workers,
        blend_start=blend_start,
        blend_end=blend_end,
        validation_points=validation_points,
        validation_x_max=validation_x_max,
        validation_rtol=validation_rtol,
        validation_atol=validation_atol,
    )


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _content_hash(
    model: Model,
    time_nodes: np.ndarray,
    space_nodes: np.ndarray,
    scores: np.ndarray,
    metadata: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(model.encode("ascii"))
    for values in (time_nodes, space_nodes, scores):
        contiguous = np.ascontiguousarray(values, dtype="<f8")
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes())
    digest.update(_canonical_json(metadata))
    return digest.hexdigest()


def _validated_table(
    model: Model,
    time_nodes: np.ndarray,
    space_nodes: np.ndarray,
    scores: np.ndarray,
    metadata: dict[str, Any],
    table_hash: str | None = None,
) -> ScoreTable:
    times = np.asarray(time_nodes, dtype=np.float64)
    spaces = np.asarray(space_nodes, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ArtifactError("score table has invalid time nodes")
    if spaces.ndim != 1 or spaces.size < 3 or spaces[0] != 0.0 or np.any(np.diff(spaces) <= 0.0):
        raise ArtifactError("score table has invalid non-negative space nodes")
    if values.shape != (times.size, spaces.size) or np.any(~np.isfinite(values)):
        raise ArtifactError("score table has invalid score array")
    if np.any(np.abs(values[:, 0]) > 1.0e-14):
        raise ArtifactError("the score at the symmetry point must be exactly zero")
    expected = _content_hash(model, times, spaces, values, metadata)
    if table_hash is not None and table_hash != expected:
        raise ArtifactError("score table content hash mismatch")
    times.setflags(write=False)
    spaces.setflags(write=False)
    values.setflags(write=False)
    return ScoreTable(model, times, spaces, values, expected, dict(metadata))


def _spectral_row(spec: _BuildSpec, time: float, space_nodes: np.ndarray) -> np.ndarray:
    count = spec.fft_points
    dx = 2.0 * spec.fft_half_width / count
    frequencies = 2.0 * math.pi * np.fft.fftfreq(count, d=dx)
    characteristic = np.asarray(
        forward_marginal_cf(frequencies, time, spec.model, alpha=spec.alpha, beta=spec.beta),
        dtype=np.float64,
    )
    nyquist_index = int(np.argmax(np.abs(frequencies)))
    if abs(characteristic[nyquist_index]) > 1.0e-10:
        raise ConfigurationError(
            "FFT frequency cutoff is insufficient: "
            f"|phi(xi_max)|={abs(characteristic[nyquist_index]):.3e}"
        )
    density_complex = np.fft.fftshift(np.fft.fft(characteristic)) / (count * dx)
    if spec.model == "stable":
        multiplier = np.zeros(count, dtype=np.complex128)
        nonzero = frequencies != 0.0
        multiplier[nonzero] = (
            -1j * frequencies[nonzero] * np.abs(frequencies[nonzero]) ** (spec.alpha - 2.0)
        )
    else:
        multiplier = -1j * frequencies
    numerator_complex = np.fft.fftshift(np.fft.fft(multiplier * characteristic)) / (count * dx)
    imaginary_residual = max(
        float(np.max(np.abs(density_complex.imag))),
        float(np.max(np.abs(numerator_complex.imag))),
    )
    if imaginary_residual > 1.0e-9:
        raise ArtifactError(f"FFT symmetry residual is too large: {imaginary_residual:.3e}")
    source_space = (np.arange(count, dtype=np.float64) - count // 2) * dx
    positive = (source_space >= 0.0) & (source_space <= spec.x_max + 2.0 * dx)
    source_positive = source_space[positive]
    density = density_complex.real[positive]
    numerator = numerator_complex.real[positive]
    density_nodes = np.interp(space_nodes, source_positive, density)
    numerator_nodes = np.interp(space_nodes, source_positive, numerator)
    if np.any(~np.isfinite(density_nodes)) or np.any(density_nodes <= 0.0):
        index = int(np.argmin(density_nodes))
        raise ArtifactError(
            f"non-positive spectral density at x={space_nodes[index]:.8g}, t={time:.8g}"
        )
    row = numerator_nodes / density_nodes
    row[0] = 0.0
    # The FFT is authoritative in the bulk.  A C1 smoothstep transitions to
    # the explicit asymptotic extrapolation before periodic-image error can
    # dominate the algebraic stable tails (and before VP underflow in the far
    # tail).  The final stored node is therefore exactly asymptotic.
    safe_tail_space = np.where(space_nodes > 0.0, space_nodes, spec.blend_start)
    if spec.model == "stable":
        tail = np.asarray(
            stable_tail_score(safe_tail_space, time, alpha=spec.alpha, beta=spec.beta),
            dtype=np.float64,
        )
    else:
        tail = np.asarray(vp_tail_score(safe_tail_space, time, beta=spec.beta), dtype=np.float64)
    blend_coordinate = np.clip(
        (space_nodes - spec.blend_start) / (spec.blend_end - spec.blend_start),
        0.0,
        1.0,
    )
    blend_weight = blend_coordinate * blend_coordinate * (3.0 - 2.0 * blend_coordinate)
    row = (1.0 - blend_weight) * row + blend_weight * tail
    row[0] = 0.0
    row[space_nodes >= spec.blend_end] = tail[space_nodes >= spec.blend_end]
    if row[-1] != tail[-1]:
        raise ArtifactError("the last score-table node must equal the asymptotic tail exactly")
    if np.any(~np.isfinite(row)):
        raise ArtifactError(f"non-finite spectral score row at t={time:.8g}")
    return row


def direct_score_reference(
    model: Model,
    x: float,
    t: float,
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
    epsabs: float = 2.0e-11,
    epsrel: float = 2.0e-11,
) -> float:
    """Independent oscillatory-quadrature reference for an off-grid point."""

    if model not in {"stable", "vp"}:
        raise ConfigurationError(f"unknown Experiment 2 model: {model!r}")
    if not math.isfinite(x) or not math.isfinite(t) or t <= 0.0:
        raise ConfigurationError("direct validation requires finite x and positive t")
    if x == 0.0:
        return 0.0
    radius = abs(x)

    def characteristic(frequency: float) -> float:
        return float(forward_marginal_cf(frequency, t, model, alpha=alpha, beta=beta))

    density_integral, _ = quad(
        characteristic,
        0.0,
        np.inf,
        weight="cos",
        wvar=radius,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=300,
        limlst=300,
        maxp1=200,
    )
    power = alpha - 1.0 if model == "stable" else 1.0

    def weighted_characteristic(frequency: float) -> float:
        if frequency == 0.0:
            return 0.0
        return frequency**power * characteristic(frequency)

    numerator_magnitude, _ = quad(
        weighted_characteristic,
        0.0,
        np.inf,
        weight="sin",
        wvar=radius,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=300,
        limlst=300,
        maxp1=200,
    )
    if not math.isfinite(density_integral) or density_integral <= 0.0:
        raise ArtifactError(f"direct score quadrature returned invalid density at x={x}, t={t}")
    # Both 1/pi factors cancel.  The numerator multiplier produces -sin.
    positive_score = -numerator_magnitude / density_integral
    # The expected positive-x score is negative; retain the quadrature sign and
    # impose oddness without assuming that fact in unusual parameter regimes.
    return positive_score if x > 0.0 else -positive_score


def interpolate_score_numpy(x, t: float, table: ScoreTable):
    """Interpolate the regularized Tweedie residual and reconstruct the score.

    Directly interpolating the score is inaccurate near the terminal epsilon
    because its leading term scales as ``1 / (1-exp(-beta*t))``.  For either
    model the exactly equivalent residual ``R_t(x)=scale(t)*score_t(x)+x`` is
    ``a(t) E[X_0 | X_t=x]`` and is regular in time.
    """

    if not math.isfinite(t):
        raise ConfigurationError("score interpolation time must be finite")
    values = np.asarray(x, dtype=np.float64)
    radii = np.abs(values)
    signs = np.sign(values)
    valid_time = table.time_nodes[0] <= t <= table.time_nodes[-1]
    safe_time = float(np.clip(t, table.time_nodes[0], table.time_nodes[-1]))
    upper_t = int(np.searchsorted(table.time_nodes, safe_time, side="right"))
    lower_t = min(max(upper_t - 1, 0), table.time_nodes.size - 2)
    transformed_nodes = np.log1p(table.space_nodes)
    transformed = np.log1p(np.minimum(radii, table.space_nodes[-1]))
    spacing = transformed_nodes[1] - transformed_nodes[0]
    position = (transformed - transformed_nodes[0]) / spacing
    lower_x = np.clip(np.floor(position).astype(np.int64), 0, table.space_nodes.size - 2)
    fraction_x = position - lower_x
    beta = float(table.metadata["beta"])
    alpha = float(table.metadata["alpha"])
    node_innovation_power = -np.expm1(-beta * table.time_nodes)
    node_scale = alpha * node_innovation_power if table.model == "stable" else node_innovation_power
    if table.time_nodes.size >= 4:
        stencil_start = min(max(lower_t - 1, 0), table.time_nodes.size - 4)
        time_indices = np.arange(stencil_start, stencil_start + 4)
    else:
        time_indices = np.asarray([lower_t, lower_t + 1])
    log_nodes = np.log(table.time_nodes[time_indices])
    log_query = math.log(safe_time)
    time_weights = np.ones(time_indices.size, dtype=np.float64)
    for index in range(time_indices.size):
        for other in range(time_indices.size):
            if other != index:
                time_weights[index] *= (log_query - log_nodes[other]) / (
                    log_nodes[index] - log_nodes[other]
                )
    residual_rows = []
    for time_index in time_indices:
        residual_left = (
            node_scale[time_index] * table.scores[time_index, lower_x] + table.space_nodes[lower_x]
        )
        residual_right = (
            node_scale[time_index] * table.scores[time_index, lower_x + 1]
            + table.space_nodes[lower_x + 1]
        )
        residual_rows.append((1.0 - fraction_x) * residual_left + fraction_x * residual_right)
    weight_shape = (time_indices.size,) + (1,) * values.ndim
    residual = np.sum(time_weights.reshape(weight_shape) * np.stack(residual_rows), axis=0)
    innovation_power = -math.expm1(-beta * safe_time)
    query_scale = alpha * innovation_power if table.model == "stable" else innovation_power
    inside_score = signs * (residual - np.minimum(radii, table.space_nodes[-1])) / query_scale
    tail_signs = np.where(values < 0.0, -1.0, 1.0)
    safe_tail_x = np.where(
        radii > table.space_nodes[-1], values, tail_signs * table.space_nodes[-1]
    )
    if table.model == "stable":
        tail_score = stable_tail_score(
            safe_tail_x,
            safe_time,
            alpha=alpha,
            beta=beta,
        )
    else:
        tail_score = vp_tail_score(safe_tail_x, safe_time, beta=beta)
    result = np.where(radii <= table.space_nodes[-1], inside_score, tail_score)
    result = np.where(values == 0.0, 0.0, result)
    if not valid_time:
        result = np.full_like(result, np.nan)
    return float(result) if result.ndim == 0 else result


def interpolate_score_jax(x, t, table: ScoreTable, dtype):
    """On-device bilinear interpolation; output broadcasts to the shape of ``x``.

    Values outside the tabulated spatial domain use a documented model-specific
    asymptotic expansion.  Times outside the tabulated domain return NaN rather
    than being silently clipped.
    """

    import jax.numpy as jnp

    values = jnp.asarray(x, dtype=dtype)
    times = jnp.asarray(t, dtype=dtype)
    values, times = jnp.broadcast_arrays(values, times)
    radii = jnp.abs(values)
    signs = jnp.sign(values)
    time_nodes = jnp.asarray(table.time_nodes, dtype=dtype)
    transformed_nodes = jnp.log1p(jnp.asarray(table.space_nodes, dtype=dtype))
    score_values = jnp.asarray(table.scores, dtype=dtype)
    valid_time = (times >= time_nodes[0]) & (times <= time_nodes[-1])
    safe_time = jnp.clip(times, time_nodes[0], time_nodes[-1])
    lower_t = jnp.clip(
        jnp.searchsorted(time_nodes, safe_time, side="right") - 1,
        0,
        time_nodes.size - 2,
    )
    x_max = jnp.asarray(table.space_nodes[-1], dtype=dtype)
    safe_radius = jnp.minimum(radii, x_max)
    transformed = jnp.log1p(safe_radius)
    spacing = transformed_nodes[1] - transformed_nodes[0]
    position = (transformed - transformed_nodes[0]) / spacing
    lower_x = jnp.clip(
        jnp.floor(position).astype(jnp.int32), 0, transformed_nodes.size - 2
    )
    space_fraction = position - lower_x.astype(dtype)
    beta = jnp.asarray(float(table.metadata["beta"]), dtype=dtype)
    alpha_value = jnp.asarray(float(table.metadata["alpha"]), dtype=dtype)
    space_nodes = jnp.asarray(table.space_nodes, dtype=dtype)
    node_innovation_power = -jnp.expm1(-beta * time_nodes)
    if table.model == "stable":
        node_scale = alpha_value * node_innovation_power
    else:
        node_scale = node_innovation_power
    if table.time_nodes.size >= 4:
        stencil_start = jnp.clip(lower_t - 1, 0, table.time_nodes.size - 4)
        time_indices = stencil_start[..., None] + jnp.arange(4, dtype=jnp.int32)
    else:
        time_indices = jnp.stack((lower_t, lower_t + 1), axis=-1)
    log_nodes = jnp.log(time_nodes[time_indices])
    log_query = jnp.log(safe_time)
    if table.time_nodes.size >= 4:
        weight0 = (
            (log_query - log_nodes[..., 1])
            * (log_query - log_nodes[..., 2])
            * (log_query - log_nodes[..., 3])
            / (
                (log_nodes[..., 0] - log_nodes[..., 1])
                * (log_nodes[..., 0] - log_nodes[..., 2])
                * (log_nodes[..., 0] - log_nodes[..., 3])
            )
        )
        weight1 = (
            (log_query - log_nodes[..., 0])
            * (log_query - log_nodes[..., 2])
            * (log_query - log_nodes[..., 3])
            / (
                (log_nodes[..., 1] - log_nodes[..., 0])
                * (log_nodes[..., 1] - log_nodes[..., 2])
                * (log_nodes[..., 1] - log_nodes[..., 3])
            )
        )
        weight2 = (
            (log_query - log_nodes[..., 0])
            * (log_query - log_nodes[..., 1])
            * (log_query - log_nodes[..., 3])
            / (
                (log_nodes[..., 2] - log_nodes[..., 0])
                * (log_nodes[..., 2] - log_nodes[..., 1])
                * (log_nodes[..., 2] - log_nodes[..., 3])
            )
        )
        weight3 = (
            (log_query - log_nodes[..., 0])
            * (log_query - log_nodes[..., 1])
            * (log_query - log_nodes[..., 2])
            / (
                (log_nodes[..., 3] - log_nodes[..., 0])
                * (log_nodes[..., 3] - log_nodes[..., 1])
                * (log_nodes[..., 3] - log_nodes[..., 2])
            )
        )
        time_weights = jnp.stack((weight0, weight1, weight2, weight3), axis=-1)
    else:
        fraction = (log_query - log_nodes[..., 0]) / (log_nodes[..., 1] - log_nodes[..., 0])
        time_weights = jnp.stack((1.0 - fraction, fraction), axis=-1)
    left_indices = lower_x[..., None]
    right_indices = left_indices + 1
    residual_left = (
        node_scale[time_indices] * score_values[time_indices, left_indices]
        + space_nodes[left_indices]
    )
    residual_right = (
        node_scale[time_indices] * score_values[time_indices, right_indices]
        + space_nodes[right_indices]
    )
    residual_rows = (1.0 - space_fraction[..., None]) * residual_left + space_fraction[
        ..., None
    ] * residual_right
    residual = jnp.sum(time_weights * residual_rows, axis=-1)
    innovation_power = -jnp.expm1(-beta * safe_time)
    query_scale = alpha_value * innovation_power if table.model == "stable" else innovation_power
    inside_score = signs * (residual - safe_radius) / query_scale

    tail_radius = jnp.where(radii > x_max, radii, x_max)
    tail_signs = jnp.where(values < 0.0, -1.0, 1.0)
    tail_x = tail_signs * tail_radius
    if table.model == "vp":
        a2 = jnp.exp(-beta * safe_time)
        g2 = -jnp.expm1(-beta * safe_time)
        coefficient_a = 15.0 * g2 - 10.0 * a2
        coefficient_b = 70.0 * a2**2 - 280.0 * a2 * g2 + 210.0 * g2**2
        tail_score = (
            -5.0 / tail_x
            - 2.0 * coefficient_a / tail_x**3
            + (2.0 * coefficient_a**2 - 4.0 * coefficient_b) / tail_x**5
        )
    else:
        alpha = float(table.metadata["alpha"])
        innovation_power = -jnp.expm1(-beta * safe_time)
        a = jnp.exp(-beta * safe_time / alpha_value)
        coefficients = jnp.asarray(three_term_asymptotic_coefficients(alpha), dtype=dtype)
        density_tail = jnp.zeros_like(tail_radius)
        radial_derivative = jnp.zeros_like(tail_radius)
        for k in range(1, 4):
            exponent = k * alpha + 1.0
            scaled = coefficients[k - 1] * innovation_power**k
            density_tail = density_tail + scaled * tail_radius ** (-exponent)
            radial_derivative = radial_derivative - (
                exponent * scaled * tail_radius ** (-exponent - 1.0)
            )
        log_derivative = tail_signs * radial_derivative / density_tail
        residual = tail_x + a * a * STUDENT_VARIANCE * log_derivative
        tail_score = -residual / (alpha_value * innovation_power)
    result = jnp.where(radii <= x_max, inside_score, tail_score)
    result = jnp.where(values == 0.0, jnp.asarray(0.0, dtype=dtype), result)
    return jnp.where(valid_time, result, jnp.asarray(jnp.nan, dtype=dtype))


def validate_score_table_off_grid(
    table: ScoreTable,
    *,
    points: Sequence[tuple[float, float]] | None = None,
    validation_points: int = 12,
    validation_x_max: float | None = None,
    rtol: float = 2.0e-4,
    atol: float = 2.0e-7,
) -> dict[str, float | int]:
    """Compare spatially off-grid table values with direct Fourier quadrature."""

    if points is None:
        if validation_points <= 0:
            raise ConfigurationError("validation_points must be positive")
        maximum = validation_x_max or min(
            table.space_nodes[-1] * 0.2, 50.0 if table.model == "stable" else 15.0
        )
        fractions = (np.arange(validation_points, dtype=np.float64) + 0.371) / (
            validation_points + 0.742
        )
        radii = np.expm1(fractions * math.log1p(maximum))
        indices = np.rint(np.linspace(0, table.time_nodes.size - 2, validation_points)).astype(int)
        points = tuple(
            (
                float(radius),
                float(math.sqrt(table.time_nodes[index] * table.time_nodes[index + 1])),
            )
            for radius, index in zip(radii, indices, strict=True)
        )
    errors: list[float] = []
    normalized_errors: list[float] = []
    ratios: list[float] = []
    allowances: list[float] = []
    alpha = float(table.metadata["alpha"])
    beta = float(table.metadata["beta"])
    for x_value, time in points:
        estimate = float(interpolate_score_numpy(x_value, time, table))
        reference = direct_score_reference(table.model, x_value, time, alpha=alpha, beta=beta)
        error = abs(estimate - reference)
        # The prescribed mixed relative criterion remains meaningful at the
        # symmetry point, where the exact odd score vanishes.
        allowed = atol + rtol * (1.0 + abs(reference))
        errors.append(error)
        normalized_errors.append(error / (1.0 + abs(reference)))
        ratios.append(error / allowed)
        allowances.append(allowed)
    worst = int(np.argmax(ratios))
    if ratios[worst] > 1.0:
        x_value, time = points[worst]
        raise ArtifactError(
            "off-grid score validation failed at "
            f"x={x_value:.8g}, t={time:.8g}: error={errors[worst]:.3e}, "
            f"allowed={allowances[worst]:.3e}"
        )
    return {
        "points": len(points),
        "max_absolute_error": float(max(errors, default=0.0)),
        "max_normalized_error": float(max(normalized_errors, default=0.0)),
        "max_tolerance_ratio": float(max(ratios, default=0.0)),
    }


def build_score_table(config: Any, model: Model) -> ScoreTable:
    """Build and directly validate a model-specific deterministic float64 table."""

    spec = _resolve_build_spec(config, model)
    transformed = np.linspace(0.0, math.log1p(spec.x_max), spec.space_points, dtype=np.float64)
    space_nodes = np.expm1(transformed)

    def construct_row(time: np.float64) -> np.ndarray:
        return _spectral_row(spec, float(time), space_nodes)

    if spec.construction_workers == 1:
        constructed = [construct_row(time) for time in spec.time_nodes]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=spec.construction_workers
        ) as executor:
            # Executor.map preserves input ordering, hence table bytes and hash.
            constructed = list(executor.map(construct_row, spec.time_nodes))
    rows = np.stack(constructed, axis=0)
    base_metadata: dict[str, Any] = {
        "schema_version": 1,
        "model": spec.model,
        "alpha": spec.alpha,
        "beta": spec.beta,
        "target": "centered_student_t_df4",
        "student_variance": STUDENT_VARIANCE,
        "score_definition": (
            "fractional_multiplier_-i_xi_absxi^(alpha-2)"
            if spec.model == "stable"
            else "ordinary_multiplier_-i_xi"
        ),
        "fourier_convention": "hat_f(xi)=integral f(x) exp(i*x*xi) dx",
        "space_transform": "log1p_abs_x",
        "time_grid": "geometric_in_forward_time",
        "interpolation_quantity": (
            "R=x+alpha*(1-exp(-beta*t))*fractional_score=a*posterior_mean"
            if spec.model == "stable"
            else "R=x+(1-exp(-beta*t))*ordinary_score=a*posterior_mean"
        ),
        "time_interpolation": "four_point_lagrange_in_log_forward_time",
        "tail_extrapolation": (
            "three_term_stable_density_plus_leading_student_variance_posterior_correction"
            if spec.model == "stable"
            else "student_t4_gaussian_convolution_through_abs_x^-5"
        ),
        "fft_half_width": spec.fft_half_width,
        "fft_points": spec.fft_points,
        "fft_dx": 2.0 * spec.fft_half_width / spec.fft_points,
        "construction_workers": spec.construction_workers,
        "blend_start": spec.blend_start,
        "blend_end": spec.blend_end,
        "blend_weight": "cubic_smoothstep_3s^2-2s^3",
        "config_signature": hashlib.sha256(_canonical_json(spec.payload())).hexdigest(),
    }
    preliminary = _validated_table(spec.model, spec.time_nodes, space_nodes, rows, base_metadata)
    validation = validate_score_table_off_grid(
        preliminary,
        validation_points=spec.validation_points,
        validation_x_max=spec.validation_x_max,
        rtol=spec.validation_rtol,
        atol=spec.validation_atol,
    )
    metadata = dict(base_metadata)
    metadata["direct_off_grid_validation"] = {
        **validation,
        "axes": "space_and_time",
        "rtol": spec.validation_rtol,
        "atol": spec.validation_atol,
        "x_max": spec.validation_x_max,
        "error_criterion": "abs(error) <= atol + rtol*(1+abs(reference))",
    }
    return _validated_table(spec.model, spec.time_nodes, space_nodes, rows, metadata)


def score_table_path(run_dir: str | Path, config: Any, model: Model) -> Path:
    """Return the model-addressed cache path derived from resolved settings."""

    spec = _resolve_build_spec(config, model)
    signature = hashlib.sha256(_canonical_json(spec.payload())).hexdigest()[:16]
    return Path(run_dir) / "score_tables" / f"exp2_{model}_score_{signature}.npz"


def _save_score_table(table: ScoreTable, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ArtifactError(f"refusing to overwrite score table: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                model=np.asarray(table.model),
                time_nodes=table.time_nodes,
                space_nodes=table.space_nodes,
                scores=table.scores,
                table_hash=np.asarray(table.table_hash),
                metadata_json=np.asarray(_canonical_json(table.metadata).decode("utf-8")),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def load_score_table(path: str | Path) -> ScoreTable:
    """Load, structurally validate, and hash-check a saved score table."""

    source = Path(path)
    if not source.is_file():
        raise ArtifactError(f"score table not found: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            model = str(archive["model"].item())
            if model not in {"stable", "vp"}:
                raise ArtifactError(f"invalid score-table model: {model!r}")
            metadata = json.loads(str(archive["metadata_json"].item()))
            return _validated_table(
                model,  # type: ignore[arg-type]
                np.asarray(archive["time_nodes"], dtype=np.float64),
                np.asarray(archive["space_nodes"], dtype=np.float64),
                np.asarray(archive["scores"], dtype=np.float64),
                metadata,
                str(archive["table_hash"].item()),
            )
    except ArtifactError:
        raise
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"corrupt score table {source}: {exc}") from exc


def ensure_score_table(
    run_dir: str | Path, config: Any, model: Model, resume: bool = True
) -> ScoreTable:
    """Load or exclusively build a table without overwriting artifacts.

    The exclusive lock is intentionally fail-fast.  On a scheduler, score-table
    construction should be a dependency job rather than duplicated by every
    array task.  A stale lock is never removed silently.
    """

    path = score_table_path(run_dir, config, model)
    spec = _resolve_build_spec(config, model)
    expected_signature = hashlib.sha256(_canonical_json(spec.payload())).hexdigest()
    if path.exists():
        if not resume:
            raise ArtifactError(f"score table already exists and resume is disabled: {path}")
        table = load_score_table(path)
        if table.model != model or table.metadata.get("config_signature") != expected_signature:
            raise ArtifactError(f"cached score table does not match resolved configuration: {path}")
        return table
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ArtifactError(
            f"score-table build lock already exists: {lock_path}; "
            "use a scheduler dependency for table construction, or verify and "
            "remove a stale lock explicitly"
        ) from exc
    try:
        lock_identity = os.fstat(descriptor)
        lock_payload = _canonical_json(
            {
                "pid": os.getpid(),
                "model": model,
                "config_signature": expected_signature,
            }
        )
        os.write(descriptor, lock_payload)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    try:
        # A completed table may have appeared between the optimistic existence
        # check and lock acquisition (for example after a previous lock holder
        # finished).  Never replace it.
        if path.exists():
            if not resume:
                raise ArtifactError(f"score table already exists and resume is disabled: {path}")
            table = load_score_table(path)
            if table.model != model or table.metadata.get("config_signature") != expected_signature:
                raise ArtifactError(
                    f"cached score table does not match resolved configuration: {path}"
                )
            return table
        table = build_score_table(config, model)
        _save_score_table(table, path)
        return table
    finally:
        try:
            current_identity = os.stat(lock_path)
        except FileNotFoundError as exc:
            raise ArtifactError(
                f"score-table build lock disappeared unexpectedly: {lock_path}"
            ) from exc
        if (current_identity.st_dev, current_identity.st_ino) != (
            lock_identity.st_dev,
            lock_identity.st_ino,
        ):
            raise ArtifactError(
                f"score-table build lock identity changed unexpectedly: {lock_path}"
            )
        lock_path.unlink()


__all__ = [
    "ScoreTable",
    "build_score_table",
    "direct_score_reference",
    "ensure_score_table",
    "interpolate_score_jax",
    "interpolate_score_numpy",
    "load_score_table",
    "score_table_path",
    "validate_score_table_off_grid",
]
