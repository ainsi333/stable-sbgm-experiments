"""Deterministic model-and-target-specific score tables for Experiment 1."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.integrate import quad

from levy_experiments.errors import ArtifactError, ConfigurationError

from .theory import (
    Model,
    forward_marginal_cf,
    stable_density_tail_constant,
    stable_tail_score,
    student_density_tail_constant,
    vp_tail_score,
)

ValidationTier = Literal["smoke", "pilot", "final"]
_ALGORITHM_VERSION = "exp1-score-table-v1-leading-mixture-tail"


@dataclass(frozen=True)
class ScoreTable:
    """Portable content-addressed table consumed by the simulation kernels."""

    model: Model
    time_nodes: np.ndarray
    space_nodes: np.ndarray
    scores: np.ndarray
    table_hash: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _BuildSpec:
    model: Model
    nu: float
    alpha: float
    beta: float
    time_nodes: np.ndarray
    space_points: int
    x_max: float
    fft_half_width: float
    fft_points: int
    blend_start: float
    blend_end: float
    workers: int
    tier: ValidationTier

    def payload(self) -> dict[str, Any]:
        """Scientific signature; execution worker count is intentionally excluded."""

        return {
            "algorithm_version": _ALGORITHM_VERSION,
            "model": self.model,
            "nu": self.nu,
            "alpha": self.alpha,
            "beta": self.beta,
            "time_nodes": self.time_nodes.tolist(),
            "space_points": self.space_points,
            "x_max": self.x_max,
            "fft_half_width": self.fft_half_width,
            "fft_points": self.fft_points,
            "blend_start": self.blend_start,
            "blend_end": self.blend_end,
        }


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _field(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _check_model(model: str) -> Model:
    if model not in {"stable", "vp"}:
        raise ConfigurationError(f"unknown Experiment 1 score model: {model!r}")
    return model  # type: ignore[return-value]


def _check_nu(nu: float) -> float:
    value = float(nu)
    if not math.isfinite(value) or value <= 1.0:
        raise ConfigurationError("score tables require a finite Student index nu > 1")
    return value


def table_key(model: Model, nu: float) -> str:
    """Stable human-readable identifier used by tasks, paths, and manifests."""

    checked_model = _check_model(model)
    checked_nu = _check_nu(nu)
    token = format(checked_nu, ".12g").replace("-", "m").replace(".", "p")
    return f"{checked_model}_student_t_nu{token}"


def sensitivity_config(config: Any) -> Any:
    """Return the frozen independently rebuilt 2x-resolution table configuration."""

    settings = _field(config, "score_table")
    factor = int(_field(settings, "sensitivity_refinement_factor"))
    refined = dataclasses.replace(
        settings,
        time_points=factor * (int(_field(settings, "time_points")) - 1) + 1,
        space_points=factor * (int(_field(settings, "space_points")) - 1) + 1,
        stable_fft_points=factor * int(_field(settings, "stable_fft_points")),
        vp_fft_points=factor * int(_field(settings, "vp_fft_points")),
    )
    return dataclasses.replace(config, score_table=refined)


def sensitivity_score_table_path(
    run_dir: str | Path, config: Any, model: Model, nu: float
) -> Path:
    """Content-addressed path for the independent high-resolution table."""

    return score_table_path(
        Path(run_dir) / "score_sensitivity", sensitivity_config(config), model, nu
    )


def ensure_sensitivity_score_table(
    run_dir: str | Path,
    config: Any,
    model: Model,
    nu: float,
    *,
    resume: bool = True,
) -> ScoreTable:
    """Build or verify the independent high-resolution sensitivity table."""

    return ensure_score_table(
        Path(run_dir) / "score_sensitivity",
        sensitivity_config(config),
        model,
        nu,
        resume=resume,
    )


def _resolve_build_spec(config: Any, model: Model, nu: float) -> _BuildSpec:
    checked_model = _check_model(model)
    checked_nu = _check_nu(nu)
    experiment = _field(config, "experiment")
    stable = _field(config, "stable")
    target = _field(config, "target")
    settings = _field(config, "score_table")
    if experiment is None or stable is None or settings is None:
        raise ConfigurationError("score table config requires experiment, stable and score_table")
    beta = float(_field(experiment, "beta"))
    epsilon = float(_field(experiment, "epsilon"))
    horizon = float(_field(experiment, "horizon"))
    alpha = float(_field(stable, "alpha"))
    if not all(math.isfinite(value) for value in (beta, epsilon, horizon, alpha)):
        raise ConfigurationError("score-table mathematical parameters must be finite")
    if beta <= 0.0 or not 0.0 < epsilon < horizon or not 1.0 < alpha < 2.0:
        raise ConfigurationError("score-table beta/time/alpha parameters are invalid")
    if target is not None:
        allowed = _field(target, "stable_nus" if checked_model == "stable" else "vp_nus", ())
        if allowed and not any(
            math.isclose(checked_nu, float(item), rel_tol=0.0, abs_tol=1.0e-12) for item in allowed
        ):
            raise ConfigurationError(
                f"nu={checked_nu:g} is not configured for the {checked_model} model"
            )
    time_points = int(_field(settings, "time_points"))
    space_points = int(_field(settings, "space_points"))
    prefix = "stable" if checked_model == "stable" else "vp"
    fft_points = int(_field(settings, f"{prefix}_fft_points"))
    fft_half_width = float(_field(settings, f"{prefix}_tail_l"))
    x_max = float(_field(settings, f"{prefix}_x_max"))
    blend_start = float(_field(settings, f"{prefix}_blend_start"))
    blend_end = float(_field(settings, f"{prefix}_blend_end"))
    if time_points < 5 or space_points < 17 or time_points % 2 == 0 or space_points % 2 == 0:
        raise ConfigurationError("score tables require odd time>=5 and space>=17 node counts")
    if fft_points < 1024 or fft_points & (fft_points - 1):
        raise ConfigurationError("score-table FFT size must be a power of two >= 1024")
    if not 0.0 < blend_start < blend_end <= x_max < fft_half_width:
        raise ConfigurationError("score-table spatial and blend domains are invalid")
    explicit_workers = _field(settings, "workers", None)
    workers = (
        min(8, os.cpu_count() or 1, time_points)
        if explicit_workers is None
        else min(int(explicit_workers), time_points)
    )
    if workers <= 0:
        raise ConfigurationError("score-table construction workers must be positive")
    tier = str(_field(experiment, "tier", "smoke"))
    if tier not in {"smoke", "pilot", "final"}:
        tier = "smoke"
    return _BuildSpec(
        model=checked_model,
        nu=checked_nu,
        alpha=alpha,
        beta=beta,
        time_nodes=np.geomspace(epsilon, horizon, time_points, dtype=np.float64),
        space_points=space_points,
        x_max=x_max,
        fft_half_width=fft_half_width,
        fft_points=fft_points,
        blend_start=blend_start,
        blend_end=blend_end,
        workers=workers,
        tier=tier,  # type: ignore[arg-type]
    )


def _table_hash(
    model: Model,
    time_nodes: np.ndarray,
    space_nodes: np.ndarray,
    scores: np.ndarray,
    metadata: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(model.encode("ascii"))
    for array in (time_nodes, space_nodes, scores):
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    digest.update(_canonical_json(metadata))
    return digest.hexdigest()


def _validated_table(
    model: str,
    time_nodes: np.ndarray,
    space_nodes: np.ndarray,
    scores: np.ndarray,
    metadata: dict[str, Any],
    expected_hash: str | None = None,
) -> ScoreTable:
    checked_model = _check_model(model)
    times = np.asarray(time_nodes, dtype=np.float64)
    spaces = np.asarray(space_nodes, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if times.ndim != 1 or spaces.ndim != 1 or values.shape != (times.size, spaces.size):
        raise ArtifactError("score-table arrays have incompatible shapes")
    if times.size < 2 or spaces.size < 2 or times[0] <= 0.0 or spaces[0] != 0.0:
        raise ArtifactError("score-table coordinate domains are invalid")
    if np.any(np.diff(times) <= 0.0) or np.any(np.diff(spaces) <= 0.0):
        raise ArtifactError("score-table coordinates must be strictly increasing")
    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(spaces)) or np.any(~np.isfinite(values)):
        raise ArtifactError("score table contains non-finite values")
    if not np.array_equal(values[:, 0], np.zeros(times.size, dtype=np.float64)):
        raise ArtifactError("score table must be exactly zero at the symmetry origin")
    required = {
        "algorithm_version",
        "alpha",
        "beta",
        "config_signature",
        "model",
        "nu",
        "target",
    }
    if not required.issubset(metadata):
        raise ArtifactError("score-table metadata is incomplete")
    actual_hash = _table_hash(checked_model, times, spaces, values, metadata)
    if expected_hash is not None and expected_hash != actual_hash:
        raise ArtifactError("score-table content hash mismatch")
    return ScoreTable(checked_model, times, spaces, values, actual_hash, metadata)


def _spectral_values(
    spec: _BuildSpec,
    time: float,
    query: np.ndarray,
    *,
    fft_half_width: float | None = None,
    fft_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    half_width = spec.fft_half_width if fft_half_width is None else fft_half_width
    count = spec.fft_points if fft_points is None else fft_points
    dx = 2.0 * half_width / count
    frequencies = 2.0 * math.pi * np.fft.fftfreq(count, d=dx)
    characteristic = np.asarray(
        forward_marginal_cf(
            frequencies,
            time,
            spec.model,
            spec.nu,
            alpha=spec.alpha,
            beta=spec.beta,
        ),
        dtype=np.float64,
    )
    nyquist = int(np.argmax(np.abs(frequencies)))
    if abs(characteristic[nyquist]) > 1.0e-10:
        raise ArtifactError(
            "FFT frequency cutoff is insufficient: "
            f"|phi(xi_max)|={abs(characteristic[nyquist]):.3e}"
        )
    density_complex = np.fft.fftshift(np.fft.fft(characteristic)) / (count * dx)
    multiplier = np.zeros(count, dtype=np.complex128)
    nonzero = frequencies != 0.0
    if spec.model == "stable":
        multiplier[nonzero] = (
            -1j * frequencies[nonzero] * np.abs(frequencies[nonzero]) ** (spec.alpha - 2.0)
        )
    else:
        multiplier[nonzero] = -1j * frequencies[nonzero]
    numerator_complex = np.fft.fftshift(np.fft.fft(multiplier * characteristic)) / (count * dx)
    imaginary_residual = max(
        float(np.max(np.abs(density_complex.imag))),
        float(np.max(np.abs(numerator_complex.imag))),
    )
    if imaginary_residual > 2.0e-9:
        raise ArtifactError(f"FFT symmetry residual is too large: {imaginary_residual:.3e}")
    source = (np.arange(count, dtype=np.float64) - count // 2) * dx
    positive = (source >= 0.0) & (source <= float(np.max(query)) + 2.0 * dx)
    density = np.interp(query, source[positive], density_complex.real[positive])
    numerator = np.interp(query, source[positive], numerator_complex.real[positive])
    if np.any(~np.isfinite(density)) or np.any(density <= 0.0):
        index = int(np.argmin(density))
        raise ArtifactError(f"non-positive spectral density at x={query[index]:.8g}, t={time:.8g}")
    return density, numerator


def _tail_score_numpy(spec: _BuildSpec, values: np.ndarray, time: float) -> np.ndarray:
    if spec.model == "stable":
        return np.asarray(
            stable_tail_score(values, time, spec.nu, alpha=spec.alpha, beta=spec.beta),
            dtype=np.float64,
        )
    return np.asarray(vp_tail_score(values, time, spec.nu, beta=spec.beta), dtype=np.float64)


def _spectral_row(spec: _BuildSpec, time: float, space_nodes: np.ndarray) -> np.ndarray:
    density, numerator = _spectral_values(spec, time, space_nodes)
    raw = numerator / density
    raw[0] = 0.0
    safe_space = np.where(space_nodes > 0.0, space_nodes, spec.blend_start)
    tail = _tail_score_numpy(spec, safe_space, time)
    coordinate = np.clip(
        (space_nodes - spec.blend_start) / (spec.blend_end - spec.blend_start),
        0.0,
        1.0,
    )
    weight = coordinate * coordinate * (3.0 - 2.0 * coordinate)
    row = (1.0 - weight) * raw + weight * tail
    row[0] = 0.0
    row[space_nodes >= spec.blend_end] = tail[space_nodes >= spec.blend_end]
    if not np.array_equal(row[space_nodes >= spec.blend_end], tail[space_nodes >= spec.blend_end]):
        raise ArtifactError("tail nodes must equal the explicit asymptotic exactly")
    if np.any(~np.isfinite(row)):
        raise ArtifactError(f"non-finite score row at t={time:.8g}")
    return row


def _time_interpolation_indices(
    time_nodes: np.ndarray, time: float
) -> tuple[np.ndarray, np.ndarray]:
    upper = int(np.searchsorted(time_nodes, time, side="right"))
    lower = min(max(upper - 1, 0), time_nodes.size - 2)
    if time_nodes.size >= 4:
        start = min(max(lower - 1, 0), time_nodes.size - 4)
        indices = np.arange(start, start + 4)
    else:
        indices = np.asarray([lower, lower + 1])
    logs = np.log(time_nodes[indices])
    query = math.log(time)
    weights = np.ones(indices.size, dtype=np.float64)
    for index in range(indices.size):
        for other in range(indices.size):
            if other != index:
                weights[index] *= (query - logs[other]) / (logs[index] - logs[other])
    return indices, weights


def interpolate_score_numpy(x, t: float, table: ScoreTable):
    """Interpolate the exact Tweedie residual, then reconstruct the score."""

    if not math.isfinite(t):
        raise ConfigurationError("score interpolation time must be finite")
    values = np.asarray(x, dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ConfigurationError("score interpolation inputs must be finite")
    valid_time = table.time_nodes[0] <= t <= table.time_nodes[-1]
    safe_time = float(np.clip(t, table.time_nodes[0], table.time_nodes[-1]))
    radii = np.abs(values)
    signs = np.sign(values)
    transformed_nodes = np.log1p(table.space_nodes)
    capped_radii = np.minimum(radii, table.space_nodes[-1])
    transformed = np.log1p(capped_radii)
    spacing = transformed_nodes[1] - transformed_nodes[0]
    position = (transformed - transformed_nodes[0]) / spacing
    left = np.clip(np.floor(position).astype(np.int64), 0, table.space_nodes.size - 2)
    fraction = position - left
    beta = float(table.metadata["beta"])
    alpha = float(table.metadata["alpha"])
    node_power = -np.expm1(-beta * table.time_nodes)
    node_scale = alpha * node_power if table.model == "stable" else node_power
    indices, weights = _time_interpolation_indices(table.time_nodes, safe_time)
    residual_rows = []
    for time_index in indices:
        residual_left = (
            table.space_nodes[left] + node_scale[time_index] * table.scores[time_index, left]
        )
        residual_right = (
            table.space_nodes[left + 1]
            + node_scale[time_index] * table.scores[time_index, left + 1]
        )
        residual_rows.append((1.0 - fraction) * residual_left + fraction * residual_right)
    weight_shape = (indices.size,) + (1,) * values.ndim
    residual = np.sum(weights.reshape(weight_shape) * np.stack(residual_rows), axis=0)
    query_power = -math.expm1(-beta * safe_time)
    query_scale = alpha * query_power if table.model == "stable" else query_power
    inside = signs * (residual - capped_radii) / query_scale
    tail_sign = np.where(values < 0.0, -1.0, 1.0)
    tail_x = np.where(radii > table.space_nodes[-1], values, tail_sign * table.space_nodes[-1])
    spec = _spec_from_table(table)
    tail = _tail_score_numpy(spec, np.asarray(tail_x, dtype=np.float64), safe_time)
    result = np.where(radii <= table.space_nodes[-1], inside, tail)
    result = np.where(values == 0.0, 0.0, result)
    if not valid_time:
        result = np.full_like(result, np.nan)
    return float(result) if result.ndim == 0 else result


def _spec_from_table(table: ScoreTable) -> _BuildSpec:
    metadata = table.metadata
    return _BuildSpec(
        model=table.model,
        nu=float(metadata["nu"]),
        alpha=float(metadata["alpha"]),
        beta=float(metadata["beta"]),
        time_nodes=table.time_nodes,
        space_points=table.space_nodes.size,
        x_max=float(table.space_nodes[-1]),
        fft_half_width=float(metadata["fft_half_width"]),
        fft_points=int(metadata["fft_points"]),
        blend_start=float(metadata["blend_start"]),
        blend_end=float(metadata["blend_end"]),
        workers=int(metadata.get("construction_workers", 1)),
        tier=str(metadata.get("tier", "smoke")),  # type: ignore[arg-type]
    )


def interpolate_score_jax(x, t, table: ScoreTable, dtype):
    """JAX interpolation with the same residual and tail formulas as NumPy."""

    import jax.numpy as jnp

    values = jnp.asarray(x, dtype=dtype)
    times = jnp.asarray(t, dtype=dtype)
    time_nodes = jnp.asarray(table.time_nodes, dtype=dtype)
    spaces = jnp.asarray(table.space_nodes, dtype=dtype)
    scores = jnp.asarray(table.scores, dtype=dtype)
    beta = jnp.asarray(float(table.metadata["beta"]), dtype=dtype)
    alpha = jnp.asarray(float(table.metadata["alpha"]), dtype=dtype)
    nu = jnp.asarray(float(table.metadata["nu"]), dtype=dtype)
    valid_time = (times >= time_nodes[0]) & (times <= time_nodes[-1])
    safe_time = jnp.clip(times, time_nodes[0], time_nodes[-1])
    radii = jnp.abs(values)
    signs = jnp.sign(values)
    transformed_nodes = jnp.log1p(spaces)
    x_max = spaces[-1]
    capped = jnp.minimum(radii, x_max)
    transformed = jnp.log1p(capped)
    spacing = transformed_nodes[1] - transformed_nodes[0]
    position = (transformed - transformed_nodes[0]) / spacing
    left = jnp.clip(jnp.floor(position).astype(jnp.int32), 0, spaces.size - 2)
    fraction = position - left.astype(dtype)
    lower_time = jnp.clip(
        jnp.searchsorted(time_nodes, safe_time, side="right") - 1,
        0,
        time_nodes.size - 2,
    )
    if table.time_nodes.size >= 4:
        start = jnp.clip(lower_time - 1, 0, time_nodes.size - 4)
        time_indices = start + jnp.arange(4, dtype=jnp.int32)
    else:
        time_indices = lower_time + jnp.arange(2, dtype=jnp.int32)
    log_nodes = jnp.log(time_nodes[time_indices])
    log_query = jnp.log(safe_time)
    weights = []
    for index in range(time_indices.size):
        weight = jnp.asarray(1.0, dtype=dtype)
        for other in range(time_indices.size):
            if other != index:
                weight = (
                    weight * (log_query - log_nodes[other]) / (log_nodes[index] - log_nodes[other])
                )
        weights.append(weight)
    time_weights = jnp.stack(weights)
    node_power = -jnp.expm1(-beta * time_nodes)
    node_scale = jnp.where(table.model == "stable", alpha * node_power, node_power)
    residual_rows = []
    for index in range(time_indices.size):
        time_index = time_indices[index]
        residual_left = spaces[left] + node_scale[time_index] * scores[time_index, left]
        residual_right = spaces[left + 1] + node_scale[time_index] * scores[time_index, left + 1]
        residual_rows.append((1.0 - fraction) * residual_left + fraction * residual_right)
    weight_shape = (time_indices.size,) + (1,) * values.ndim
    residual = jnp.sum(time_weights.reshape(weight_shape) * jnp.stack(residual_rows), axis=0)
    query_power = -jnp.expm1(-beta * safe_time)
    query_scale = jnp.where(table.model == "stable", alpha * query_power, query_power)
    inside = signs * (residual - capped) / query_scale
    tail_sign = jnp.where(values < 0.0, -1.0, 1.0)
    tail_x = jnp.where(radii > x_max, values, tail_sign * x_max)
    tail_radius = jnp.abs(tail_x)
    if table.model == "stable":
        a = jnp.exp(-beta * safe_time / alpha)
        stable_constant = jnp.asarray(
            stable_density_tail_constant(float(table.metadata["alpha"])), dtype=dtype
        )
        student_constant = jnp.asarray(
            student_density_tail_constant(float(table.metadata["nu"])), dtype=dtype
        )
        denominator = student_constant * a**nu * tail_radius ** (
            -(nu + 1.0)
        ) + stable_constant * query_power * tail_radius ** (-(alpha + 1.0))
        numerator = -tail_sign * (stable_constant / alpha) * tail_radius ** (-alpha)
        tail = numerator / denominator
    else:
        a2 = jnp.exp(-beta * safe_time)
        g2 = query_power
        coefficient_a = -0.5 * nu * (nu + 1.0) * a2 + 0.5 * (nu + 1.0) * (nu + 2.0) * g2
        coefficient_b = (
            nu**2 * (nu + 1.0) * (nu + 3.0) * a2**2 / 8.0
            - nu * (nu + 1.0) * (nu + 3.0) * (nu + 4.0) * a2 * g2 / 4.0
            + (nu + 1.0) * (nu + 2.0) * (nu + 3.0) * (nu + 4.0) * g2**2 / 8.0
        )
        tail = (
            -(nu + 1.0) / tail_x
            - 2.0 * coefficient_a / tail_x**3
            + (2.0 * coefficient_a**2 - 4.0 * coefficient_b) / tail_x**5
        )
    result = jnp.where(radii <= x_max, inside, tail)
    result = jnp.where(values == 0.0, jnp.asarray(0.0, dtype=dtype), result)
    return jnp.where(valid_time, result, jnp.asarray(jnp.nan, dtype=dtype))


def _direct_score(spec: _BuildSpec, x: float, time: float) -> float:
    def characteristic(frequency: float) -> float:
        return float(
            forward_marginal_cf(
                frequency,
                time,
                spec.model,
                spec.nu,
                alpha=spec.alpha,
                beta=spec.beta,
            )
        )

    density = (
        quad(
            characteristic,
            0.0,
            np.inf,
            weight="cos",
            wvar=x,
            epsabs=2.0e-11,
            epsrel=2.0e-11,
            limit=500,
            limlst=500,
        )[0]
        / math.pi
    )
    power = spec.alpha - 1.0 if spec.model == "stable" else 1.0
    numerator = (
        -quad(
            lambda frequency: frequency**power * characteristic(frequency),
            0.0,
            np.inf,
            weight="sin",
            wvar=x,
            epsabs=2.0e-11,
            epsrel=2.0e-11,
            limit=500,
            limlst=500,
        )[0]
        / math.pi
    )
    if not math.isfinite(density) or density <= 0.0 or not math.isfinite(numerator):
        raise ArtifactError(f"independent quadrature failed at x={x:g}, t={time:g}")
    return numerator / density


def validate_score_table(table: ScoreTable) -> dict[str, float]:
    """Validate bulk independently and the full blend against an embedded 2x FFT."""

    spec = _spec_from_table(table)
    bulk_rtol = 8.0e-3 if spec.tier == "smoke" else 2.0e-3
    blend_rtol = 2.0e-2 if spec.tier == "smoke" else 2.0e-3
    atol = 2.0e-6
    direct_errors: list[float] = []
    direct_relative: list[float] = []
    direct_times = (
        math.sqrt(table.time_nodes[0] * table.time_nodes[1]),
        math.sqrt(
            table.time_nodes[table.time_nodes.size // 2 - 1]
            * table.time_nodes[table.time_nodes.size // 2]
        ),
        math.sqrt(table.time_nodes[-2] * table.time_nodes[-1]),
    )
    bulk_max = min(6.0, spec.blend_start * 0.5)
    direct_spaces = (0.5, min(2.0, bulk_max), bulk_max)
    for time in direct_times:
        for space in direct_spaces:
            reference = _direct_score(spec, space, time)
            estimate = float(interpolate_score_numpy(space, time, table))
            error = abs(estimate - reference)
            relative = error / max(abs(reference), atol / bulk_rtol)
            direct_errors.append(error)
            direct_relative.append(relative)
            if error > atol + bulk_rtol * abs(reference):
                raise ArtifactError(
                    "independent score validation failed at "
                    f"model={spec.model}, nu={spec.nu:g}, x={space:g}, t={time:g}: "
                    f"estimate={estimate:.12g}, reference={reference:.12g}, error={error:.3e}"
                )

    blend_errors: list[float] = []
    blend_relative: list[float] = []
    blend_spaces = np.linspace(spec.blend_start, spec.blend_end, 65, dtype=np.float64)
    validation_times = (
        float(table.time_nodes[0]),
        float(table.time_nodes[table.time_nodes.size // 2]),
        float(table.time_nodes[-1]),
    )
    for time in validation_times:
        density, numerator = _spectral_values(
            spec,
            time,
            blend_spaces,
            fft_half_width=2.0 * spec.fft_half_width,
            fft_points=2 * spec.fft_points,
        )
        reference = numerator / density
        estimate = np.asarray(interpolate_score_numpy(blend_spaces, time, table))
        errors = np.abs(estimate - reference)
        relatives = errors / np.maximum(np.abs(reference), atol / blend_rtol)
        blend_errors.extend(errors.tolist())
        blend_relative.extend(relatives.tolist())
        invalid = errors > atol + blend_rtol * np.abs(reference)
        if np.any(invalid):
            index = int(np.argmax(relatives))
            raise ArtifactError(
                "embedded 2x-FFT blend validation failed at "
                f"model={spec.model}, nu={spec.nu:g}, x={blend_spaces[index]:g}, t={time:g}: "
                f"relative={relatives[index]:.3e}, tolerance={blend_rtol:.3e}"
            )
    return {
        "independent_bulk_max_abs": float(max(direct_errors, default=0.0)),
        "independent_bulk_max_relative": float(max(direct_relative, default=0.0)),
        "embedded_fft2_blend_max_abs": float(max(blend_errors, default=0.0)),
        "embedded_fft2_blend_max_relative": float(max(blend_relative, default=0.0)),
        "bulk_rtol": bulk_rtol,
        "blend_rtol": blend_rtol,
        "atol": atol,
    }


def build_score_table(config: Any, model: Model, nu: float) -> ScoreTable:
    """Build and validate a deterministic float64 table keyed by model and nu."""

    spec = _resolve_build_spec(config, model, nu)
    transformed = np.linspace(0.0, math.log1p(spec.x_max), spec.space_points, dtype=np.float64)
    space_nodes = np.expm1(transformed)

    def construct(time: np.float64) -> np.ndarray:
        return _spectral_row(spec, float(time), space_nodes)

    if spec.workers == 1:
        rows = np.stack([construct(time) for time in spec.time_nodes])
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=spec.workers) as executor:
            rows = np.stack(list(executor.map(construct, spec.time_nodes)))
    metadata: dict[str, Any] = {
        "algorithm_version": _ALGORITHM_VERSION,
        "alpha": spec.alpha,
        "beta": spec.beta,
        "blend_end": spec.blend_end,
        "blend_start": spec.blend_start,
        "blend_weight": "cubic_smoothstep_3s^2-2s^3",
        "config_signature": hashlib.sha256(_canonical_json(spec.payload())).hexdigest(),
        "construction_workers": spec.workers,
        "fft_half_width": spec.fft_half_width,
        "fft_points": spec.fft_points,
        "interpolation_quantity": (
            "R=x+alpha*(1-exp(-beta*t))*fractional_score=a*posterior_mean"
            if spec.model == "stable"
            else "R=x+(1-exp(-beta*t))*ordinary_score=a*posterior_mean"
        ),
        "model": spec.model,
        "nu": spec.nu,
        "schema_version": 1,
        "score_definition": (
            "fractional_multiplier_-i_xi_absxi^(alpha-2)"
            if spec.model == "stable"
            else "ordinary_multiplier_-i_xi"
        ),
        "space_transform": "log1p_abs_x",
        "stable_density_tail_constant": stable_density_tail_constant(spec.alpha),
        "student_density_tail_constant": student_density_tail_constant(spec.nu),
        "tail_extrapolation": (
            "student_plus_stable_leading_density_mixture_and_universal_fractional_numerator"
            if spec.model == "stable"
            else "student_gaussian_convolution_through_abs_x^-5"
        ),
        "target": f"centered_unit_student_t_df_{spec.nu:g}",
        "tier": spec.tier,
        "time_grid": "geometric_in_forward_time",
        "time_interpolation": "four_point_lagrange_in_log_forward_time",
    }
    preliminary = _validated_table(spec.model, spec.time_nodes, space_nodes, rows, metadata)
    validation = validate_score_table(preliminary)
    final_metadata = metadata | {"validation": validation}
    return _validated_table(spec.model, spec.time_nodes, space_nodes, rows, final_metadata)


def score_table_path(run_dir: str | Path, config: Any, model: Model, nu: float) -> Path:
    spec = _resolve_build_spec(config, model, nu)
    signature = hashlib.sha256(_canonical_json(spec.payload())).hexdigest()[:16]
    return Path(run_dir) / "score_tables" / f"exp1_{table_key(model, nu)}_{signature}.npz"


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
    source = Path(path)
    if not source.is_file():
        raise ArtifactError(f"score table not found: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            return _validated_table(
                str(archive["model"].item()),
                np.asarray(archive["time_nodes"], dtype=np.float64),
                np.asarray(archive["space_nodes"], dtype=np.float64),
                np.asarray(archive["scores"], dtype=np.float64),
                json.loads(str(archive["metadata_json"].item())),
                str(archive["table_hash"].item()),
            )
    except ArtifactError:
        raise
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"corrupt score table {source}: {exc}") from exc


def ensure_score_table(
    run_dir: str | Path,
    config: Any,
    model: Model,
    nu: float,
    resume: bool = True,
) -> ScoreTable:
    """Load or exclusively build a model-and-nu table without overwriting."""

    path = score_table_path(run_dir, config, model, nu)
    spec = _resolve_build_spec(config, model, nu)
    expected_signature = hashlib.sha256(_canonical_json(spec.payload())).hexdigest()
    if path.exists():
        if not resume:
            raise ArtifactError(f"score table exists and resume is disabled: {path}")
        table = load_score_table(path)
        if (
            table.model != model
            or not math.isclose(float(table.metadata["nu"]), float(nu))
            or table.metadata.get("config_signature") != expected_signature
        ):
            raise ArtifactError(f"cached score table does not match configuration: {path}")
        return table
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(f"{path.name}.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ArtifactError(
            f"score-table build lock exists: {lock}; use a scheduler table dependency "
            "or explicitly inspect and remove a stale lock"
        ) from exc
    identity = None
    try:
        identity = os.fstat(descriptor)
        os.write(
            descriptor,
            _canonical_json(
                {
                    "pid": os.getpid(),
                    "table_key": table_key(model, nu),
                    "config_signature": expected_signature,
                }
            ),
        )
        os.fsync(descriptor)
        os.close(descriptor)
        if path.exists():
            if not resume:
                raise ArtifactError(f"score table appeared while lock was held: {path}")
            return load_score_table(path)
        table = build_score_table(config, model, nu)
        _save_score_table(table, path)
        return table
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        if identity is not None and lock.exists():
            current = lock.stat()
            if current.st_ino == identity.st_ino and current.st_dev == identity.st_dev:
                lock.unlink()


__all__ = [
    "ScoreTable",
    "build_score_table",
    "ensure_score_table",
    "ensure_sensitivity_score_table",
    "interpolate_score_jax",
    "interpolate_score_numpy",
    "load_score_table",
    "score_table_path",
    "sensitivity_config",
    "sensitivity_score_table_path",
    "table_key",
    "validate_score_table",
]
