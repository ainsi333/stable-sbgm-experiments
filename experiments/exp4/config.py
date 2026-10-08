"""Strict configuration for the deterministic Experiment 4 analysis."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from levy_experiments.errors import ConfigurationError


@dataclass(frozen=True)
class SpectralSettings:
    """One model's deterministic Fourier grid."""

    fft_half_width: float
    fft_points: int
    spatial_max: float
    density_floor: float

    def validate(self, name: str) -> None:
        if not math.isfinite(self.fft_half_width) or self.fft_half_width <= 0.0:
            raise ConfigurationError(f"{name}.fft_half_width must be finite and positive")
        if (
            not isinstance(self.fft_points, int)
            or isinstance(self.fft_points, bool)
            or self.fft_points < 4096
            or self.fft_points & (self.fft_points - 1)
        ):
            raise ConfigurationError(f"{name}.fft_points must be a power of two >= 4096")
        if not math.isfinite(self.spatial_max) or not 0.0 < self.spatial_max < self.fft_half_width:
            raise ConfigurationError(
                f"{name}.spatial_max must lie strictly inside the Fourier domain"
            )
        if not math.isfinite(self.density_floor) or not 0.0 < self.density_floor < 1.0:
            raise ConfigurationError(f"{name}.density_floor must lie in (0,1)")


@dataclass(frozen=True)
class SourceSettings:
    """Frozen provenance expected from the publication-scale Experiment 2 run."""

    required: bool
    run_dir: str
    config_hash: str
    code_hash: str
    stable_main_file: str
    stable_main_hash: str
    stable_hires_file: str
    stable_hires_hash: str
    vp_main_file: str
    vp_main_hash: str
    vp_hires_file: str
    vp_hires_hash: str

    def validate(self) -> None:
        if self.required and not self.run_dir:
            raise ConfigurationError("source.run_dir is required for the final analysis")
        hashes = (
            self.config_hash,
            self.code_hash,
            self.stable_main_hash,
            self.stable_hires_hash,
            self.vp_main_hash,
            self.vp_hires_hash,
        )
        files = (
            self.stable_main_file,
            self.stable_hires_file,
            self.vp_main_file,
            self.vp_hires_file,
        )
        if self.required and (
            any(
                len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                for value in hashes
            )
            or any(not value or Path(value).name != value for value in files)
        ):
            raise ConfigurationError(
                "the final source inventory must contain filenames and SHA-256 hashes"
            )


@dataclass(frozen=True)
class ValidationSettings:
    """Predeclared numerical consistency criteria, not scientific hypotheses."""

    times: tuple[float, ...]
    domain_radii: tuple[float, ...]
    refinement_factor: int
    resolution_rtol: float
    vp_tail_reference_rtol: float
    stable_boundary_fraction: float
    source_table_rtol: float
    maximizer_inner_radius: float

    def validate(self, *, time_min: float, time_max: float, spatial_max: float) -> None:
        if (
            len(self.times) < 2
            or any(not math.isfinite(value) for value in self.times)
            or tuple(sorted(set(self.times))) != self.times
            or self.times[0] < time_min
            or self.times[-1] > time_max
        ):
            raise ConfigurationError(
                "validation.times must be sorted, unique, and inside the time range"
            )
        if (
            len(self.domain_radii) < 2
            or any(not math.isfinite(value) or value <= 0.0 for value in self.domain_radii)
            or tuple(sorted(set(self.domain_radii))) != self.domain_radii
            or not math.isclose(self.domain_radii[-1], spatial_max, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ConfigurationError(
                "validation.domain_radii must be increasing and end at spatial_max"
            )
        if self.refinement_factor != 2:
            raise ConfigurationError("validation.refinement_factor is fixed to two")
        for name, value in (
            ("resolution_rtol", self.resolution_rtol),
            ("vp_tail_reference_rtol", self.vp_tail_reference_rtol),
            ("stable_boundary_fraction", self.stable_boundary_fraction),
            ("source_table_rtol", self.source_table_rtol),
        ):
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ConfigurationError(f"validation.{name} must lie in (0,1)")
        if (
            not math.isfinite(self.maximizer_inner_radius)
            or not 0.0 < self.maximizer_inner_radius < self.domain_radii[-1]
        ):
            raise ConfigurationError("validation.maximizer_inner_radius is outside the domain")


@dataclass(frozen=True)
class Experiment4Config:
    """Resolved, hash-addressed Experiment 4 configuration."""

    tier: str
    target_df: float
    alpha: float
    beta: float
    time_min: float
    time_max: float
    time_points: int
    precision: str
    stable_spectral: SpectralSettings
    vp_spectral: SpectralSettings
    validation: ValidationSettings
    source: SourceSettings
    source_path: str
    source_hash: str

    def validate(self) -> None:
        if self.tier not in {"smoke", "final"}:
            raise ConfigurationError("experiment.tier must be smoke or final")
        if not math.isclose(self.target_df, 4.0, rel_tol=0.0, abs_tol=1e-12):
            raise ConfigurationError("Experiment 4 is defined for the Student-t(4) target")
        if not math.isclose(self.alpha, 1.5, rel_tol=0.0, abs_tol=1e-12):
            raise ConfigurationError("Experiment 4 fixes alpha=1.5")
        if not math.isfinite(self.beta) or self.beta <= 0.0:
            raise ConfigurationError("experiment.beta must be finite and positive")
        if (
            not math.isfinite(self.time_min)
            or not math.isfinite(self.time_max)
            or not 0.0 < self.time_min < self.time_max
        ):
            raise ConfigurationError("the time interval must satisfy 0 < time_min < time_max")
        if (
            not isinstance(self.time_points, int)
            or isinstance(self.time_points, bool)
            or self.time_points < 3
        ):
            raise ConfigurationError("experiment.time_points must be an integer >= 3")
        if self.precision != "float64":
            raise ConfigurationError("the validated deterministic analysis requires float64")
        self.stable_spectral.validate("stable_spectral")
        self.vp_spectral.validate("vp_spectral")
        if not math.isclose(
            self.stable_spectral.spatial_max,
            self.vp_spectral.spatial_max,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ConfigurationError("both models must use the same reported spatial domain")
        self.validation.validate(
            time_min=self.time_min,
            time_max=self.time_max,
            spatial_max=self.stable_spectral.spatial_max,
        )
        self.source.validate()

    def canonical_dict(self, *, include_paths: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("source_hash")
        if not include_paths:
            payload.pop("source_path")
            payload["source"]["run_dir"] = "<external-exp2-run>" if self.source.required else ""
        return payload

    @property
    def resolved_hash(self) -> str:
        encoded = json.dumps(
            self.canonical_dict(include_paths=False),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


_SECTIONS = {
    "experiment": {
        "name",
        "tier",
        "target_df",
        "alpha",
        "beta",
        "time_min",
        "time_max",
        "time_points",
        "precision",
    },
    "source": {
        "required",
        "run_dir",
        "config_hash",
        "code_hash",
        "stable_main_file",
        "stable_main_hash",
        "stable_hires_file",
        "stable_hires_hash",
        "vp_main_file",
        "vp_main_hash",
        "vp_hires_file",
        "vp_hires_hash",
    },
    "stable_spectral": {"fft_half_width", "fft_points", "spatial_max", "density_floor"},
    "vp_spectral": {"fft_half_width", "fft_points", "spatial_max", "density_floor"},
    "validation": {
        "times",
        "domain_radii",
        "refinement_factor",
        "resolution_rtol",
        "vp_tail_reference_rtol",
        "stable_boundary_fraction",
        "source_table_rtol",
        "maximizer_inner_radius",
    },
}


def _strict_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigurationError(f"missing TOML section [{name}]")
    missing = _SECTIONS[name] - set(value)
    extra = set(value) - _SECTIONS[name]
    if missing or extra:
        raise ConfigurationError(
            f"invalid [{name}] keys: missing={sorted(missing)}, unexpected={sorted(extra)}"
        )
    return value


def load_config(
    path: str | Path,
    *,
    source_run_override: str | Path | None = None,
) -> Experiment4Config:
    """Load a strict TOML file and resolve only its external source path."""

    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise ConfigurationError(f"Experiment 4 configuration not found: {source_path}")
    try:
        source_bytes = source_path.read_bytes()
        raw = tomllib.loads(source_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"invalid Experiment 4 TOML: {exc}") from exc
    if set(raw) != set(_SECTIONS):
        raise ConfigurationError(
            f"invalid top-level sections: expected={sorted(_SECTIONS)}, actual={sorted(raw)}"
        )
    experiment = _strict_section(raw, "experiment")
    if experiment["name"] != "experiment4":
        raise ConfigurationError("experiment.name must be experiment4")
    source_values = _strict_section(raw, "source")
    stable_values = _strict_section(raw, "stable_spectral")
    vp_values = _strict_section(raw, "vp_spectral")
    validation_values = _strict_section(raw, "validation")
    configured_run = str(source_values["run_dir"])
    if source_run_override is not None:
        resolved_source = Path(source_run_override).resolve()
    elif configured_run:
        candidate = Path(configured_run)
        resolved_source = candidate.resolve() if candidate.is_absolute() else (
            source_path.parent / candidate
        ).resolve()
    else:
        resolved_source = Path()
    try:
        config = Experiment4Config(
            tier=str(experiment["tier"]),
            target_df=float(experiment["target_df"]),
            alpha=float(experiment["alpha"]),
            beta=float(experiment["beta"]),
            time_min=float(experiment["time_min"]),
            time_max=float(experiment["time_max"]),
            time_points=int(experiment["time_points"]),
            precision=str(experiment["precision"]),
            stable_spectral=SpectralSettings(
                fft_half_width=float(stable_values["fft_half_width"]),
                fft_points=int(stable_values["fft_points"]),
                spatial_max=float(stable_values["spatial_max"]),
                density_floor=float(stable_values["density_floor"]),
            ),
            vp_spectral=SpectralSettings(
                fft_half_width=float(vp_values["fft_half_width"]),
                fft_points=int(vp_values["fft_points"]),
                spatial_max=float(vp_values["spatial_max"]),
                density_floor=float(vp_values["density_floor"]),
            ),
            validation=ValidationSettings(
                times=tuple(float(value) for value in validation_values["times"]),
                domain_radii=tuple(float(value) for value in validation_values["domain_radii"]),
                refinement_factor=int(validation_values["refinement_factor"]),
                resolution_rtol=float(validation_values["resolution_rtol"]),
                vp_tail_reference_rtol=float(validation_values["vp_tail_reference_rtol"]),
                stable_boundary_fraction=float(validation_values["stable_boundary_fraction"]),
                source_table_rtol=float(validation_values["source_table_rtol"]),
                maximizer_inner_radius=float(validation_values["maximizer_inner_radius"]),
            ),
            source=SourceSettings(
                required=bool(source_values["required"]),
                run_dir=str(resolved_source) if configured_run or source_run_override else "",
                config_hash=str(source_values["config_hash"]),
                code_hash=str(source_values["code_hash"]),
                stable_main_file=str(source_values["stable_main_file"]),
                stable_main_hash=str(source_values["stable_main_hash"]),
                stable_hires_file=str(source_values["stable_hires_file"]),
                stable_hires_hash=str(source_values["stable_hires_hash"]),
                vp_main_file=str(source_values["vp_main_file"]),
                vp_main_hash=str(source_values["vp_main_hash"]),
                vp_hires_file=str(source_values["vp_hires_file"]),
                vp_hires_hash=str(source_values["vp_hires_hash"]),
            ),
            source_path=str(source_path),
            source_hash=hashlib.sha256(source_bytes).hexdigest(),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid numerical value in {source_path}: {exc}") from exc
    config.validate()
    return config


__all__ = [
    "Experiment4Config",
    "SourceSettings",
    "SpectralSettings",
    "ValidationSettings",
    "load_config",
]
