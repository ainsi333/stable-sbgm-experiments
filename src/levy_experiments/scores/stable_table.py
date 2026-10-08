"""Validated float64 table for the one-dimensional S1.5S density."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import gamma
from scipy.stats import levy_stable

from ..config import Experiment3Config
from ..errors import ArtifactError, ConfigurationError


@dataclass(frozen=True)
class StableDensityTable:
    alpha: float
    transformed_grid: np.ndarray
    log_density: np.ndarray
    z_max: float
    tail_coefficients: np.ndarray
    table_hash: str

    @property
    def spacing(self) -> float:
        return float(self.transformed_grid[1] - self.transformed_grid[0])


def three_term_asymptotic_coefficients(alpha: float) -> np.ndarray:
    """Return the first three density-tail coefficients for alpha=1.5.

    Their positivity and truncation accuracy are specific to the experiment's
    fixed alpha.  This function intentionally refuses generic alpha values.
    """

    if not math.isclose(alpha, 1.5, rel_tol=0.0, abs_tol=1e-12):
        raise ConfigurationError("the validated three-term tail is specific to alpha=1.5")
    coefficients = []
    for k in range(1, 4):
        coefficient = (
            (-1.0) ** (k + 1)
            * gamma(k * alpha + 1.0)
            * math.sin(k * math.pi * alpha / 2.0)
            / (math.pi * math.factorial(k))
        )
        coefficients.append(coefficient)
    result = np.asarray(coefficients, dtype=np.float64)
    if np.any(result <= 0.0):
        raise ConfigurationError("validated alpha=1.5 tail coefficients must all be positive")
    return result


def three_term_asymptotic_density(values: np.ndarray, alpha: float) -> np.ndarray:
    radii = np.asarray(values, dtype=np.float64)
    if np.any(radii <= 0.0):
        raise ConfigurationError("tail density is defined here only for positive radii")
    coefficients = three_term_asymptotic_coefficients(alpha)
    result = np.zeros_like(radii)
    for k, coefficient in enumerate(coefficients, start=1):
        result += coefficient * np.power(radii, -(k * alpha + 1.0))
    return result


def _density_chunk(values: np.ndarray, alpha: float) -> np.ndarray:
    return np.asarray(levy_stable.pdf(values, alpha, 0.0, loc=0.0, scale=1.0), dtype=np.float64)


def _table_hash(
    alpha: float, transformed: np.ndarray, log_density: np.ndarray, z_max: float
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray([alpha, z_max], dtype="<f8").tobytes())
    digest.update(np.asarray(transformed, dtype="<f8").tobytes())
    digest.update(np.asarray(log_density, dtype="<f8").tobytes())
    digest.update(three_term_asymptotic_coefficients(alpha).astype("<f8").tobytes())
    return digest.hexdigest()


def interpolate_log_density_numpy(table: StableDensityTable, values: np.ndarray) -> np.ndarray:
    radii = np.abs(np.asarray(values, dtype=np.float64))
    transformed = np.log1p(radii)
    inside = radii <= table.z_max
    safe_transformed = np.where(inside, transformed, table.transformed_grid[-1])
    interpolated = np.interp(safe_transformed, table.transformed_grid, table.log_density)
    safe_tail_radius = np.where(inside, table.z_max, radii)
    tail = np.log(three_term_asymptotic_density(safe_tail_radius, table.alpha))
    return np.where(inside, interpolated, tail)


def interpolate_log_density_jax(table: StableDensityTable, values, dtype):
    """Interpolate on-device and use an explicit validated tail outside the grid."""

    import jax.numpy as jnp

    radii = jnp.abs(values)
    z_max = jnp.asarray(table.z_max, dtype=dtype)
    inside = radii <= z_max
    transformed = jnp.log1p(radii)
    end = jnp.asarray(table.transformed_grid[-1], dtype=dtype)
    safe_transformed = jnp.where(inside, transformed, end)
    spacing = jnp.asarray(table.spacing, dtype=dtype)
    position = safe_transformed / spacing
    last_interval = table.transformed_grid.size - 2
    index = jnp.minimum(jnp.floor(position).astype(jnp.int32), last_interval)
    fraction = position - index.astype(dtype)
    log_values = jnp.asarray(table.log_density, dtype=dtype)
    interpolated = (1.0 - fraction) * log_values[index] + fraction * log_values[index + 1]
    safe_radius = jnp.where(inside, z_max, radii)
    coefficients = jnp.asarray(table.tail_coefficients, dtype=dtype)
    tail_density = jnp.zeros_like(safe_radius)
    for k in range(1, 4):
        tail_density = tail_density + coefficients[k - 1] * safe_radius ** (
            -(k * table.alpha + 1.0)
        )
    tail = jnp.log(tail_density)
    return jnp.where(inside, interpolated, tail)


def build_stable_density_table(config: Experiment3Config) -> StableDensityTable:
    """Build and independently validate the score table in float64."""

    settings = config.score_table
    alpha = config.experiment.alpha
    transformed = np.linspace(0.0, math.log1p(settings.z_max), settings.points, dtype=np.float64)
    positive_grid = np.expm1(transformed)
    chunks = [chunk for chunk in np.array_split(positive_grid, settings.workers) if chunk.size]
    if settings.workers == 1:
        density_parts = [_density_chunk(chunks[0], alpha)]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=settings.workers) as executor:
            density_parts = list(executor.map(_density_chunk, chunks, [alpha] * len(chunks)))
    density = np.concatenate(density_parts)
    if (
        density.shape != positive_grid.shape
        or np.any(~np.isfinite(density))
        or np.any(density <= 0.0)
    ):
        raise ArtifactError("SciPy produced an invalid stable-density table")
    log_density = np.log(density)
    table = StableDensityTable(
        alpha=alpha,
        transformed_grid=transformed,
        log_density=log_density,
        z_max=settings.z_max,
        tail_coefficients=three_term_asymptotic_coefficients(alpha),
        table_hash=_table_hash(alpha, transformed, log_density, settings.z_max),
    )
    validate_stable_density_table(
        table, settings.validation_points, settings.validation_rtol, settings.validation_atol
    )
    return table


def validate_stable_density_table(
    table: StableDensityTable, validation_points: int, rtol: float, atol: float
) -> dict[str, float]:
    """Validate at transformed midpoints absent from the construction grid."""

    positions = (np.arange(validation_points, dtype=np.float64) + 0.5) / validation_points
    transformed = positions * math.log1p(table.z_max)
    points = np.expm1(transformed)
    reference = _density_chunk(points, table.alpha)
    estimate = np.exp(interpolate_log_density_numpy(table, points))
    errors = np.abs(estimate - reference)
    allowed = atol + rtol * reference
    if np.any(errors > allowed):
        worst = int(np.argmax(errors / allowed))
        raise ArtifactError(
            "stable table validation failed at "
            f"z={points[worst]:.8g}: error={errors[worst]:.3e}, allowed={allowed[worst]:.3e}"
        )
    seam_reference = float(_density_chunk(np.asarray([table.z_max]), table.alpha)[0])
    seam_tail = float(three_term_asymptotic_density(np.asarray([table.z_max]), table.alpha)[0])
    seam_relative = abs(seam_tail / seam_reference - 1.0)
    if seam_relative >= 1e-7:
        raise ArtifactError(f"stable table tail seam is too large: {seam_relative:.3e}")
    return {
        "max_absolute_error": float(errors.max()),
        "max_relative_error": float(np.max(errors / reference)),
        "p99_absolute_log_error": float(
            np.quantile(np.abs(np.log(estimate) - np.log(reference)), 0.99)
        ),
        "seam_relative_error": seam_relative,
    }


def save_stable_density_table(table: StableDensityTable, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ArtifactError(f"Refusing to overwrite score table: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            alpha=np.asarray(table.alpha, dtype=np.float64),
            transformed_grid=table.transformed_grid,
            log_density=table.log_density,
            z_max=np.asarray(table.z_max, dtype=np.float64),
            tail_coefficients=table.tail_coefficients,
            table_hash=np.asarray(table.table_hash),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_stable_density_table(path: Path) -> StableDensityTable:
    if not path.is_file():
        raise ArtifactError(f"Score table not found: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            table = StableDensityTable(
                alpha=float(archive["alpha"]),
                transformed_grid=np.asarray(archive["transformed_grid"], dtype=np.float64),
                log_density=np.asarray(archive["log_density"], dtype=np.float64),
                z_max=float(archive["z_max"]),
                tail_coefficients=np.asarray(archive["tail_coefficients"], dtype=np.float64),
                table_hash=str(archive["table_hash"]),
            )
    except (OSError, KeyError, ValueError) as exc:
        raise ArtifactError(f"Corrupt score table {path}: {exc}") from exc
    expected = _table_hash(table.alpha, table.transformed_grid, table.log_density, table.z_max)
    if table.table_hash != expected:
        raise ArtifactError(f"Score table content hash mismatch: {path}")
    return table


def score_table_path(run_dir: Path, config: Experiment3Config) -> Path:
    settings_payload = {
        "alpha": config.experiment.alpha,
        "z_max": config.score_table.z_max,
        "points": config.score_table.points,
        "tail_terms": 3,
        "grid": "log1p",
    }
    digest = hashlib.sha256(
        json.dumps(settings_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return run_dir / "score_tables" / f"stable_density_{digest}.npz"
