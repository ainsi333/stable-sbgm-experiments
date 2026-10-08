"""Strict configuration and deterministic task map for Experiment 1.

Experiment 1 tests target-specific tail coverage.  Each dynamic task owns the
stationary-reference arm used by the tail theorem, an exact-``p_T`` marginal
control arm, and three independent exact checkpoint references.  Direct
SalphaS, Pareto, and Gaussian controls validate the tail estimators without a
reverse integration.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from levy_experiments.errors import ConfigurationError

Precision = Literal["float32", "float64", "mixed"]
RunTier = Literal["smoke", "pilot", "final"]
TaskKind = Literal["dynamic", "control"]
Model = Literal["stable", "vp", "sas", "pareto", "gaussian"]
Purpose = Literal["primary", "refinement", "control"]
InitializationArm = Literal[
    "stationary_tail_reference",
    "exact_p_T_marginal_control",
]
ReferenceSample = Literal["exact_p_s_a", "exact_p_s_b", "exact_p_s_c"]

FINAL_RUN_GUARD = "EXP1_FINAL_PUBLICATION_RUN"


def _strict_keys(section: str, values: dict[str, Any], allowed: set[str]) -> None:
    unexpected = set(values) - allowed
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ConfigurationError(f"Unknown key(s) in [{section}]: {names}")


def _section(raw: dict[str, Any], name: str, allowed: set[str]) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Missing TOML section [{name}]")
    _strict_keys(name, value, allowed)
    return value


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ConfigurationError(f"{name} must be finite")
    return result


def _positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigurationError(f"{name} must be a nonnegative integer")
    return value


def _float_tuple(name: str, value: Any, *, allow_empty: bool = False) -> tuple[float, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "an" if allow_empty else "a non-empty"
        raise ConfigurationError(f"{name} must be {qualifier} TOML array")
    return tuple(_finite_float(name, item) for item in value)


def _positive_int_tuple(name: str, value: Any, *, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "an" if allow_empty else "a non-empty"
        raise ConfigurationError(f"{name} must be {qualifier} TOML array")
    return tuple(_positive_int(name, item) for item in value)


def _seed_tuple(name: str, value: Any, *, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "an" if allow_empty else "a non-empty"
        raise ConfigurationError(f"{name} must be {qualifier} TOML array")
    seeds: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or not 0 <= item <= 2**31 - 1:
            raise ConfigurationError(f"{name} must contain integers in [0, 2^31-1]")
        seeds.append(item)
    return tuple(seeds)


def _string_tuple(name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ConfigurationError(f"{name} must be a non-empty string array")
    return tuple(value)


def _intervals(name: str, value: Any) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{name} must be a non-empty array of two-point arrays")
    intervals: list[tuple[float, float]] = []
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            raise ConfigurationError(f"{name}[{index}] must contain exactly two endpoints")
        intervals.append(
            (_finite_float(f"{name}[{index}]", item[0]), _finite_float(f"{name}[{index}]", item[1]))
        )
    return tuple(intervals)


@dataclass(frozen=True)
class ExperimentSettings:
    name: str
    tier: RunTier
    horizon: float
    beta: float
    epsilon: float
    primary_seeds: tuple[int, ...]
    refinement_seeds: tuple[int, ...]
    main_particles: int
    control_particles: int
    refinement_particles: int
    refinement_control_particles: int
    batch_size: int
    precision: Precision
    publication_scale: bool
    final_run_guard: str


@dataclass(frozen=True)
class TargetSettings:
    distribution: str
    location: float
    scale: float
    stable_nus: tuple[float, ...]
    vp_nus: tuple[float, ...]


@dataclass(frozen=True)
class StableSettings:
    alpha: float
    eta: float
    stationary_law: str
    score: str


@dataclass(frozen=True)
class VPSettings:
    stationary_law: str
    score: str


@dataclass(frozen=True)
class ControlSettings:
    sas_alpha: float
    pareto_tail_index: float
    pareto_minimum: float
    pareto_symmetric: bool
    gaussian_mean: float
    gaussian_std: float


@dataclass(frozen=True)
class DesignSettings:
    primary_seed_steps: tuple[int, ...]
    primary_steps: int
    refinement_seed_steps: tuple[int, ...]
    checkpoint_fractions: tuple[float, ...]
    initialization_arms: tuple[InitializationArm, ...]
    reference_samples: tuple[ReferenceSample, ...]
    direct_controls: tuple[Model, ...]


@dataclass(frozen=True)
class ScoreTableSettings:
    time_points: int
    space_points: int
    stable_fft_points: int
    stable_tail_l: float
    stable_x_max: float
    stable_blend_start: float
    stable_blend_end: float
    vp_fft_points: int
    vp_tail_l: float
    vp_x_max: float
    vp_blend_start: float
    vp_blend_end: float
    sensitivity_refinement_factor: int


@dataclass(frozen=True)
class AnalysisSettings:
    wasserstein_orders: tuple[float, ...]
    ecf_frequency_min: float
    ecf_frequency_max: float
    ecf_frequency_count: int
    ecf_weight: str
    tail_fractions: tuple[float, ...]
    primary_tail_fraction: float
    plateau_fractions: tuple[float, ...]
    survival_probabilities: tuple[float, ...]
    fit_primary: tuple[float, float]
    fit_sensitivities: tuple[tuple[float, float], ...]
    bootstrap_replicates: int
    subsample_fractions: tuple[float, ...]
    blocks: int
    mean_excess_threshold_method: str
    mean_excess_fractions: tuple[float, ...]
    minimum_exceedances_per_seed: int
    minimum_pooled_exceedances: int
    minimum_contributing_seeds: int
    chunk_size: int
    pt_ecf_familywise_alpha: float
    score_sensitivity_floor_fraction: float
    score_sensitivity_hill_tolerance: float
    score_sensitivity_constant_relative_tolerance: float
    refinement_floor_fraction: float
    refinement_hill_tolerance: float
    refinement_constant_relative_tolerance: float
    refinement_m_slope_tolerance: float
    gaussian_hill_separation_margin: float


@dataclass(frozen=True)
class RuntimeOverrides:
    precision: Precision | None
    seed: int | None


@dataclass(frozen=True)
class Experiment1Config:
    experiment: ExperimentSettings
    target: TargetSettings
    stable: StableSettings
    vp: VPSettings
    controls: ControlSettings
    design: DesignSettings
    score_table: ScoreTableSettings
    analysis: AnalysisSettings
    overrides: RuntimeOverrides
    source_path: str
    source_hash: str
    final_execution_authorized: bool = field(default=False, repr=False, compare=False)

    @property
    def resolved_hash(self) -> str:
        canonical = json.dumps(
            self.canonical_dict(include_source_path=False),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def canonical_dict(self, *, include_source_path: bool = True) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload.pop("final_execution_authorized", None)
        if not include_source_path:
            payload.pop("source_path", None)
            payload.pop("source_hash", None)
        return payload


@dataclass(frozen=True)
class Experiment1Task:
    index: int
    kind: TaskKind
    model: Model
    nu: float | None
    steps: int
    seed: int
    particles: int
    control_particles: int
    purpose: Purpose
    coupling_steps: int

    @property
    def task_id(self) -> str:
        canonical = json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))
        suffix = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
        nu = "" if self.nu is None else f"-nu{self.nu:g}".replace(".", "p")
        return (
            f"task-{self.index:04d}-{self.kind}-{self.model}{nu}-N{self.steps}"
            f"-s{self.seed}-{self.purpose}-{suffix}"
        )


_SURVIVAL_GRID = (
    0.02,
    0.015581556161088883,
    0.012139244620058345,
    0.00945741609003176,
    0.0073680629972807735,
    0.005740294369528563,
    0.00447213595499958,
    0.0034841418771425405,
    0.002714417616594907,
    0.0021147425268811283,
    0.0016475489724420657,
    0.0012835688421125163,
    0.001,
)

_EXPECTED_PROTOCOL: dict[RunTier, dict[str, Any]] = {
    "smoke": {
        "primary_seeds": (0, 1),
        "refinement_seeds": (),
        "particles": (4096, 4096, 4096, 4096),
        "primary_seed_steps": (4, 8),
        "primary_steps": 8,
        "refinement_seed_steps": (),
        "checkpoints": (0.0, 0.5, 1.0),
        "bootstrap": 200,
        "count_gates": (10, 20, 2),
        "score_table": (
            17,
            1025,
            131072,
            2048.0,
            128.0,
            64.0,
            128.0,
            65536,
            256.0,
            16.0,
            12.0,
            16.0,
            2,
        ),
    },
    "pilot": {
        "primary_seeds": (100, 101, 102, 103),
        "refinement_seeds": (),
        "particles": (65536, 16384, 65536, 16384),
        "primary_seed_steps": (20, 40, 80),
        "primary_steps": 80,
        "refinement_seed_steps": (),
        "checkpoints": (0.0, 0.25, 0.5, 0.75, 1.0),
        "bootstrap": 2000,
        "count_gates": (100, 400, 4),
        "score_table": (
            65,
            4097,
            524288,
            8192.0,
            384.0,
            256.0,
            384.0,
            262144,
            1024.0,
            16.0,
            12.0,
            16.0,
            2,
        ),
    },
    "final": {
        "primary_seeds": tuple(range(1000, 1012)),
        "refinement_seeds": (200, 201, 202, 203),
        "particles": (262144, 65536, 65536, 16384),
        "primary_seed_steps": (80,),
        "primary_steps": 80,
        "refinement_seed_steps": (40, 80, 160),
        "checkpoints": (0.0, 0.25, 0.5, 0.75, 1.0),
        "bootstrap": 10000,
        "count_gates": (200, 2400, 8),
        "score_table": (
            129,
            8193,
            524288,
            8192.0,
            384.0,
            256.0,
            384.0,
            262144,
            1024.0,
            16.0,
            12.0,
            16.0,
            2,
        ),
    },
}


def load_config(
    path: str | Path,
    *,
    precision_override: Precision | None = None,
    seed_override: int | None = None,
    allow_final: bool = False,
) -> Experiment1Config:
    """Load and strictly validate one frozen Experiment 1 TOML file."""

    source = Path(path).resolve()
    if not source.is_file():
        raise ConfigurationError(f"Configuration file not found: {source}")
    source_bytes = source.read_bytes()
    try:
        raw = tomllib.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"Invalid TOML configuration {source}: {exc}") from exc
    _strict_keys(
        "root",
        raw,
        {
            "experiment",
            "target",
            "stable",
            "vp",
            "controls",
            "design",
            "score_table",
            "analysis",
        },
    )

    exp_raw = _section(
        raw,
        "experiment",
        {
            "name",
            "tier",
            "horizon",
            "beta",
            "epsilon",
            "primary_seeds",
            "refinement_seeds",
            "main_particles",
            "control_particles",
            "refinement_particles",
            "refinement_control_particles",
            "batch_size",
            "precision",
            "publication_scale",
            "final_run_guard",
        },
    )
    tier = exp_raw.get("tier")
    if tier not in {"smoke", "pilot", "final"}:
        raise ConfigurationError("experiment.tier must be smoke, pilot, or final")
    precision = exp_raw.get("precision")
    if precision not in {"float32", "float64", "mixed"}:
        raise ConfigurationError("experiment.precision must be float32, float64, or mixed")
    if precision_override not in {None, "float32", "float64", "mixed"}:
        raise ConfigurationError("precision override must be float32, float64, or mixed")
    if seed_override is not None and (
        not isinstance(seed_override, int)
        or isinstance(seed_override, bool)
        or not 0 <= seed_override <= 2**31 - 1
    ):
        raise ConfigurationError("seed override must be an integer in [0, 2^31-1]")
    publication_scale = exp_raw.get("publication_scale")
    if not isinstance(publication_scale, bool):
        raise ConfigurationError("experiment.publication_scale must be boolean")
    requested_primary = _seed_tuple("experiment.primary_seeds", exp_raw.get("primary_seeds"))
    requested_refinement = _seed_tuple(
        "experiment.refinement_seeds", exp_raw.get("refinement_seeds"), allow_empty=True
    )
    if seed_override is None:
        primary_seeds, refinement_seeds = requested_primary, requested_refinement
    else:
        primary_seeds = (seed_override,) if seed_override in requested_primary else ()
        refinement_seeds = (seed_override,) if seed_override in requested_refinement else ()
        if not primary_seeds and not refinement_seeds:
            raise ConfigurationError("seed override is outside the frozen tier seed sets")
    experiment = ExperimentSettings(
        name=str(exp_raw.get("name")),
        tier=tier,
        horizon=_finite_float("experiment.horizon", exp_raw.get("horizon")),
        beta=_finite_float("experiment.beta", exp_raw.get("beta")),
        epsilon=_finite_float("experiment.epsilon", exp_raw.get("epsilon")),
        primary_seeds=primary_seeds,
        refinement_seeds=refinement_seeds,
        main_particles=_positive_int("experiment.main_particles", exp_raw.get("main_particles")),
        control_particles=_positive_int(
            "experiment.control_particles", exp_raw.get("control_particles")
        ),
        refinement_particles=_positive_int(
            "experiment.refinement_particles", exp_raw.get("refinement_particles")
        ),
        refinement_control_particles=_positive_int(
            "experiment.refinement_control_particles",
            exp_raw.get("refinement_control_particles"),
        ),
        batch_size=_positive_int("experiment.batch_size", exp_raw.get("batch_size")),
        precision=precision_override if precision_override is not None else precision,
        publication_scale=publication_scale,
        final_run_guard=str(exp_raw.get("final_run_guard", "")),
    )

    target_raw = _section(
        raw,
        "target",
        {"distribution", "location", "scale", "stable_nus", "vp_nus"},
    )
    target = TargetSettings(
        distribution=str(target_raw.get("distribution")),
        location=_finite_float("target.location", target_raw.get("location")),
        scale=_finite_float("target.scale", target_raw.get("scale")),
        stable_nus=_float_tuple("target.stable_nus", target_raw.get("stable_nus")),
        vp_nus=_float_tuple("target.vp_nus", target_raw.get("vp_nus")),
    )
    stable_raw = _section(raw, "stable", {"alpha", "eta", "stationary_law", "score"})
    stable = StableSettings(
        alpha=_finite_float("stable.alpha", stable_raw.get("alpha")),
        eta=_finite_float("stable.eta", stable_raw.get("eta")),
        stationary_law=str(stable_raw.get("stationary_law")),
        score=str(stable_raw.get("score")),
    )
    vp_raw = _section(raw, "vp", {"stationary_law", "score"})
    vp = VPSettings(
        stationary_law=str(vp_raw.get("stationary_law")),
        score=str(vp_raw.get("score")),
    )
    controls_raw = _section(
        raw,
        "controls",
        {
            "sas_alpha",
            "pareto_tail_index",
            "pareto_minimum",
            "pareto_symmetric",
            "gaussian_mean",
            "gaussian_std",
        },
    )
    pareto_symmetric = controls_raw.get("pareto_symmetric")
    if not isinstance(pareto_symmetric, bool):
        raise ConfigurationError("controls.pareto_symmetric must be boolean")
    controls = ControlSettings(
        sas_alpha=_finite_float("controls.sas_alpha", controls_raw.get("sas_alpha")),
        pareto_tail_index=_finite_float(
            "controls.pareto_tail_index", controls_raw.get("pareto_tail_index")
        ),
        pareto_minimum=_finite_float("controls.pareto_minimum", controls_raw.get("pareto_minimum")),
        pareto_symmetric=pareto_symmetric,
        gaussian_mean=_finite_float("controls.gaussian_mean", controls_raw.get("gaussian_mean")),
        gaussian_std=_finite_float("controls.gaussian_std", controls_raw.get("gaussian_std")),
    )

    design_raw = _section(
        raw,
        "design",
        {
            "primary_seed_steps",
            "primary_steps",
            "refinement_seed_steps",
            "checkpoint_fractions",
            "initialization_arms",
            "reference_samples",
            "direct_controls",
        },
    )
    design = DesignSettings(
        primary_seed_steps=_positive_int_tuple(
            "design.primary_seed_steps", design_raw.get("primary_seed_steps")
        ),
        primary_steps=_positive_int("design.primary_steps", design_raw.get("primary_steps")),
        refinement_seed_steps=_positive_int_tuple(
            "design.refinement_seed_steps",
            design_raw.get("refinement_seed_steps"),
            allow_empty=True,
        ),
        checkpoint_fractions=_float_tuple(
            "design.checkpoint_fractions", design_raw.get("checkpoint_fractions")
        ),
        initialization_arms=_string_tuple(
            "design.initialization_arms", design_raw.get("initialization_arms")
        ),
        reference_samples=_string_tuple(
            "design.reference_samples", design_raw.get("reference_samples")
        ),
        direct_controls=_string_tuple("design.direct_controls", design_raw.get("direct_controls")),
    )

    table_raw = _section(
        raw,
        "score_table",
        {
            "time_points",
            "space_points",
            "stable_fft_points",
            "stable_tail_l",
            "stable_x_max",
            "stable_blend_start",
            "stable_blend_end",
            "vp_fft_points",
            "vp_tail_l",
            "vp_x_max",
            "vp_blend_start",
            "vp_blend_end",
            "sensitivity_refinement_factor",
        },
    )
    score_table = ScoreTableSettings(
        time_points=_positive_int("score_table.time_points", table_raw.get("time_points")),
        space_points=_positive_int("score_table.space_points", table_raw.get("space_points")),
        stable_fft_points=_positive_int(
            "score_table.stable_fft_points", table_raw.get("stable_fft_points")
        ),
        stable_tail_l=_finite_float("score_table.stable_tail_l", table_raw.get("stable_tail_l")),
        stable_x_max=_finite_float("score_table.stable_x_max", table_raw.get("stable_x_max")),
        stable_blend_start=_finite_float(
            "score_table.stable_blend_start", table_raw.get("stable_blend_start")
        ),
        stable_blend_end=_finite_float(
            "score_table.stable_blend_end", table_raw.get("stable_blend_end")
        ),
        vp_fft_points=_positive_int("score_table.vp_fft_points", table_raw.get("vp_fft_points")),
        vp_tail_l=_finite_float("score_table.vp_tail_l", table_raw.get("vp_tail_l")),
        vp_x_max=_finite_float("score_table.vp_x_max", table_raw.get("vp_x_max")),
        vp_blend_start=_finite_float("score_table.vp_blend_start", table_raw.get("vp_blend_start")),
        vp_blend_end=_finite_float("score_table.vp_blend_end", table_raw.get("vp_blend_end")),
        sensitivity_refinement_factor=_positive_int(
            "score_table.sensitivity_refinement_factor",
            table_raw.get("sensitivity_refinement_factor"),
        ),
    )

    analysis_raw = _section(
        raw,
        "analysis",
        {
            "wasserstein_orders",
            "ecf_frequency_min",
            "ecf_frequency_max",
            "ecf_frequency_count",
            "ecf_weight",
            "tail_fractions",
            "primary_tail_fraction",
            "plateau_fractions",
            "survival_probabilities",
            "fit_primary",
            "fit_sensitivities",
            "bootstrap_replicates",
            "subsample_fractions",
            "blocks",
            "mean_excess_threshold_method",
            "mean_excess_fractions",
            "minimum_exceedances_per_seed",
            "minimum_pooled_exceedances",
            "minimum_contributing_seeds",
            "chunk_size",
            "pt_ecf_familywise_alpha",
            "score_sensitivity_floor_fraction",
            "score_sensitivity_hill_tolerance",
            "score_sensitivity_constant_relative_tolerance",
            "refinement_floor_fraction",
            "refinement_hill_tolerance",
            "refinement_constant_relative_tolerance",
            "refinement_m_slope_tolerance",
            "gaussian_hill_separation_margin",
        },
    )
    fit_primary_values = _float_tuple("analysis.fit_primary", analysis_raw.get("fit_primary"))
    if len(fit_primary_values) != 2:
        raise ConfigurationError("analysis.fit_primary must contain exactly two endpoints")
    analysis = AnalysisSettings(
        wasserstein_orders=_float_tuple(
            "analysis.wasserstein_orders", analysis_raw.get("wasserstein_orders")
        ),
        ecf_frequency_min=_finite_float(
            "analysis.ecf_frequency_min", analysis_raw.get("ecf_frequency_min")
        ),
        ecf_frequency_max=_finite_float(
            "analysis.ecf_frequency_max", analysis_raw.get("ecf_frequency_max")
        ),
        ecf_frequency_count=_positive_int(
            "analysis.ecf_frequency_count", analysis_raw.get("ecf_frequency_count")
        ),
        ecf_weight=str(analysis_raw.get("ecf_weight")),
        tail_fractions=_float_tuple("analysis.tail_fractions", analysis_raw.get("tail_fractions")),
        primary_tail_fraction=_finite_float(
            "analysis.primary_tail_fraction", analysis_raw.get("primary_tail_fraction")
        ),
        plateau_fractions=_float_tuple(
            "analysis.plateau_fractions", analysis_raw.get("plateau_fractions")
        ),
        survival_probabilities=_float_tuple(
            "analysis.survival_probabilities", analysis_raw.get("survival_probabilities")
        ),
        fit_primary=(fit_primary_values[0], fit_primary_values[1]),
        fit_sensitivities=_intervals(
            "analysis.fit_sensitivities", analysis_raw.get("fit_sensitivities")
        ),
        bootstrap_replicates=_positive_int(
            "analysis.bootstrap_replicates", analysis_raw.get("bootstrap_replicates")
        ),
        subsample_fractions=_float_tuple(
            "analysis.subsample_fractions", analysis_raw.get("subsample_fractions")
        ),
        blocks=_positive_int("analysis.blocks", analysis_raw.get("blocks")),
        mean_excess_threshold_method=str(analysis_raw.get("mean_excess_threshold_method")),
        mean_excess_fractions=_float_tuple(
            "analysis.mean_excess_fractions", analysis_raw.get("mean_excess_fractions")
        ),
        minimum_exceedances_per_seed=_positive_int(
            "analysis.minimum_exceedances_per_seed",
            analysis_raw.get("minimum_exceedances_per_seed"),
        ),
        minimum_pooled_exceedances=_positive_int(
            "analysis.minimum_pooled_exceedances",
            analysis_raw.get("minimum_pooled_exceedances"),
        ),
        minimum_contributing_seeds=_positive_int(
            "analysis.minimum_contributing_seeds",
            analysis_raw.get("minimum_contributing_seeds"),
        ),
        chunk_size=_positive_int("analysis.chunk_size", analysis_raw.get("chunk_size")),
        pt_ecf_familywise_alpha=_finite_float(
            "analysis.pt_ecf_familywise_alpha",
            analysis_raw.get("pt_ecf_familywise_alpha"),
        ),
        score_sensitivity_floor_fraction=_finite_float(
            "analysis.score_sensitivity_floor_fraction",
            analysis_raw.get("score_sensitivity_floor_fraction"),
        ),
        score_sensitivity_hill_tolerance=_finite_float(
            "analysis.score_sensitivity_hill_tolerance",
            analysis_raw.get("score_sensitivity_hill_tolerance"),
        ),
        score_sensitivity_constant_relative_tolerance=_finite_float(
            "analysis.score_sensitivity_constant_relative_tolerance",
            analysis_raw.get("score_sensitivity_constant_relative_tolerance"),
        ),
        refinement_floor_fraction=_finite_float(
            "analysis.refinement_floor_fraction", analysis_raw.get("refinement_floor_fraction")
        ),
        refinement_hill_tolerance=_finite_float(
            "analysis.refinement_hill_tolerance", analysis_raw.get("refinement_hill_tolerance")
        ),
        refinement_constant_relative_tolerance=_finite_float(
            "analysis.refinement_constant_relative_tolerance",
            analysis_raw.get("refinement_constant_relative_tolerance"),
        ),
        refinement_m_slope_tolerance=_finite_float(
            "analysis.refinement_m_slope_tolerance",
            analysis_raw.get("refinement_m_slope_tolerance"),
        ),
        gaussian_hill_separation_margin=_finite_float(
            "analysis.gaussian_hill_separation_margin",
            analysis_raw.get("gaussian_hill_separation_margin"),
        ),
    )

    config = Experiment1Config(
        experiment=experiment,
        target=target,
        stable=stable,
        vp=vp,
        controls=controls,
        design=design,
        score_table=score_table,
        analysis=analysis,
        overrides=RuntimeOverrides(precision=precision_override, seed=seed_override),
        source_path=str(source),
        source_hash=hashlib.sha256(source_bytes).hexdigest(),
        final_execution_authorized=allow_final,
    )
    validate_config(config)
    return config


def validate_config(config: Experiment1Config) -> None:
    """Enforce the frozen mathematics, resources, estimators, and task coupling."""

    exp = config.experiment
    if exp.name != "experiment1":
        raise ConfigurationError("experiment.name must be 'experiment1'")
    if not (
        math.isclose(exp.horizon, 2.0)
        and math.isclose(exp.epsilon, 0.5)
        and math.isclose(exp.beta, 1.0)
    ):
        raise ConfigurationError("Experiment 1 fixes T=2, epsilon=0.5, and beta=1")
    if exp.horizon <= exp.epsilon:
        raise ConfigurationError("experiment.horizon must exceed epsilon")
    all_seeds = exp.primary_seeds + exp.refinement_seeds
    if len(set(all_seeds)) != len(all_seeds):
        raise ConfigurationError("primary and refinement seed sets must be unique and disjoint")
    for name, count in (
        ("main_particles", exp.main_particles),
        ("control_particles", exp.control_particles),
        ("refinement_particles", exp.refinement_particles),
        ("refinement_control_particles", exp.refinement_control_particles),
    ):
        if count % exp.batch_size:
            raise ConfigurationError(f"experiment.{name} must be divisible by batch_size")

    if config.target != TargetSettings("scipy.stats.t", 0.0, 1.0, (1.2, 1.5, 3.0), (1.5,)):
        raise ConfigurationError(
            "targets must be standard Student-t with stable nu=(1.2,1.5,3) and VP nu=(1.5,)"
        )
    if config.stable != StableSettings(1.5, 0.5, "sas_unit", "fractional_exact_cf_tabulated"):
        raise ConfigurationError(
            "stable model must use alpha=1.5, eta=0.5, SalphaS(1), and exact CF score"
        )
    if config.vp != VPSettings("normal_standard", "gaussian_exact_cf_tabulated"):
        raise ConfigurationError("VP must use N(0,1) and its own exact CF score")
    if config.controls != ControlSettings(1.5, 1.5, 1.0, True, 0.0, 1.0):
        raise ConfigurationError("direct controls must be unit SalphaS/Pareto/Gaussian controls")

    design = config.design
    if design.primary_steps not in design.primary_seed_steps:
        raise ConfigurationError("design.primary_steps must belong to primary_seed_steps")
    if len(set(design.primary_seed_steps)) != len(design.primary_seed_steps) or len(
        set(design.refinement_seed_steps)
    ) != len(design.refinement_seed_steps):
        raise ConfigurationError("dynamic step lists must not contain duplicates")
    if design.initialization_arms != (
        "stationary_tail_reference",
        "exact_p_T_marginal_control",
    ):
        raise ConfigurationError("every dynamic task must contain both prescribed arms")
    if design.reference_samples != ("exact_p_s_a", "exact_p_s_b", "exact_p_s_c"):
        raise ConfigurationError("every dynamic task must contain three exact references")
    if design.direct_controls != ("sas", "pareto", "gaussian"):
        raise ConfigurationError("direct controls must be sas, pareto, gaussian in that order")
    if tuple(sorted(design.checkpoint_fractions)) != design.checkpoint_fractions or len(
        set(design.checkpoint_fractions)
    ) != len(design.checkpoint_fractions):
        raise ConfigurationError("checkpoint fractions must be distinct and increasing")
    if design.checkpoint_fractions[0] != 0.0 or design.checkpoint_fractions[-1] != 1.0:
        raise ConfigurationError("checkpoint fractions must include 0 and 1")
    groups = (
        (design.primary_seed_steps, max(design.primary_seed_steps)),
        (
            design.refinement_seed_steps,
            max(design.refinement_seed_steps) if design.refinement_seed_steps else 0,
        ),
    )
    for steps_group, coupling_steps in groups:
        for steps in steps_group:
            if coupling_steps % steps:
                raise ConfigurationError(f"N={steps} must divide coupling_steps={coupling_steps}")
            for fraction in design.checkpoint_fractions:
                index = steps * fraction
                if not math.isclose(index, round(index), rel_tol=0.0, abs_tol=1e-12):
                    raise ConfigurationError(
                        f"checkpoint fraction {fraction} is not a grid point for N={steps}"
                    )

    table = config.score_table
    if table.time_points % 2 == 0 or table.space_points % 2 == 0:
        raise ConfigurationError("score-table time_points and space_points must be odd")
    if table.stable_fft_points & (table.stable_fft_points - 1):
        raise ConfigurationError("score-table stable_fft_points must be a power of two")
    if table.vp_fft_points & (table.vp_fft_points - 1):
        raise ConfigurationError("score-table vp_fft_points must be a power of two")
    if not (0 < table.stable_blend_start < table.stable_blend_end <= table.stable_x_max):
        raise ConfigurationError("invalid stable tail-blend interval")
    if not 0 < table.vp_blend_start < table.vp_blend_end <= table.vp_x_max:
        raise ConfigurationError("invalid VP tail-blend interval")
    if table.stable_tail_l <= table.stable_x_max:
        raise ConfigurationError("score-table stable_tail_l must exceed stable_x_max")
    if table.vp_tail_l <= table.vp_x_max:
        raise ConfigurationError("score-table vp_tail_l must exceed vp_x_max")
    if table.sensitivity_refinement_factor != 2:
        raise ConfigurationError("score-table sensitivity refinement factor must be exactly 2")

    analysis = config.analysis
    if analysis.wasserstein_orders != (1.0,):
        raise ConfigurationError("Experiment 1 marginal controls use W1 only")
    if (
        analysis.ecf_frequency_min != -5.0
        or analysis.ecf_frequency_max != 5.0
        or analysis.ecf_frequency_count != 81
        or analysis.ecf_weight != "exp(-u^2/2)"
    ):
        raise ConfigurationError("ECF protocol must be 81 points on [-5,5] with exp(-u^2/2)")
    if analysis.tail_fractions != (0.003, 0.004, 0.006, 0.008, 0.012, 0.018, 0.03):
        raise ConfigurationError("tail fractions differ from the frozen protocol")
    if analysis.primary_tail_fraction != 0.006 or analysis.plateau_fractions != (
        0.004,
        0.006,
        0.008,
    ):
        raise ConfigurationError("primary/plateau tail fractions differ from the frozen protocol")
    if analysis.survival_probabilities != _SURVIVAL_GRID:
        raise ConfigurationError("survival grid differs from geomspace(0.02,0.001,13)")
    if analysis.fit_primary != (0.001, 0.01) or analysis.fit_sensitivities != (
        (0.002, 0.02),
        (0.001, 0.005),
    ):
        raise ConfigurationError("tail-slope fit windows differ from the frozen protocol")
    if (
        analysis.subsample_fractions != (0.25, 0.5, 1.0)
        or analysis.blocks != 4
        or analysis.mean_excess_threshold_method != "theoretical_discrete_rho"
        or analysis.mean_excess_fractions != (0.012, 0.006, 0.003)
        or analysis.chunk_size != 4096
    ):
        raise ConfigurationError("analysis diagnostics differ from the frozen protocol")
    if not (
        math.isclose(analysis.pt_ecf_familywise_alpha, 0.01)
        and math.isclose(analysis.score_sensitivity_floor_fraction, 0.5)
        and math.isclose(analysis.score_sensitivity_hill_tolerance, 0.05)
        and math.isclose(analysis.score_sensitivity_constant_relative_tolerance, 0.10)
        and math.isclose(analysis.refinement_floor_fraction, 1.0)
        and math.isclose(analysis.refinement_hill_tolerance, 0.10)
        and math.isclose(analysis.refinement_constant_relative_tolerance, 0.25)
        and math.isclose(analysis.refinement_m_slope_tolerance, 0.05)
        and math.isclose(analysis.gaussian_hill_separation_margin, 0.25)
    ):
        raise ConfigurationError("scientific gate thresholds differ from the frozen protocol")
    for count in (
        exp.main_particles,
        exp.control_particles,
        exp.refinement_particles,
        exp.refinement_control_particles,
    ):
        for fraction in analysis.subsample_fractions:
            retained = count * fraction
            if not math.isclose(retained, round(retained), rel_tol=0.0, abs_tol=1e-12):
                raise ConfigurationError("subsample fractions must produce integral counts")
        if count % analysis.blocks:
            raise ConfigurationError("all particle counts must be divisible by analysis.blocks")

    expected = _EXPECTED_PROTOCOL[exp.tier]
    expected_primary = expected["primary_seeds"]
    expected_refinement = expected["refinement_seeds"]
    if config.overrides.seed is None:
        seeds_match = (
            exp.primary_seeds == expected_primary and exp.refinement_seeds == expected_refinement
        )
    else:
        seed = config.overrides.seed
        seeds_match = (exp.primary_seeds == ((seed,) if seed in expected_primary else ())) and (
            exp.refinement_seeds == ((seed,) if seed in expected_refinement else ())
        )
    actual_table = (
        table.time_points,
        table.space_points,
        table.stable_fft_points,
        table.stable_tail_l,
        table.stable_x_max,
        table.stable_blend_start,
        table.stable_blend_end,
        table.vp_fft_points,
        table.vp_tail_l,
        table.vp_x_max,
        table.vp_blend_start,
        table.vp_blend_end,
        table.sensitivity_refinement_factor,
    )
    if (
        not seeds_match
        or (
            exp.main_particles,
            exp.control_particles,
            exp.refinement_particles,
            exp.refinement_control_particles,
        )
        != expected["particles"]
        or design.primary_seed_steps != expected["primary_seed_steps"]
        or design.primary_steps != expected["primary_steps"]
        or design.refinement_seed_steps != expected["refinement_seed_steps"]
        or design.checkpoint_fractions != expected["checkpoints"]
        or analysis.bootstrap_replicates != expected["bootstrap"]
        or (
            analysis.minimum_exceedances_per_seed,
            analysis.minimum_pooled_exceedances,
            analysis.minimum_contributing_seeds,
        )
        != expected["count_gates"]
        or actual_table != expected["score_table"]
    ):
        raise ConfigurationError(f"configuration does not match the frozen {exp.tier} protocol")

    if exp.tier == "final":
        if not exp.publication_scale or exp.precision != "float64":
            raise ConfigurationError("final tier must be publication-scale float64")
        if exp.final_run_guard != FINAL_RUN_GUARD:
            raise ConfigurationError("final configuration is missing its immutable guard token")
    elif exp.publication_scale or exp.final_run_guard:
        raise ConfigurationError("smoke/pilot tiers cannot enable the final-run guard")


def validate_execution(config: Experiment1Config) -> None:
    """Block final publication writes unless the caller explicitly opted in."""

    validate_config(config)
    if config.experiment.tier == "final" and not config.final_execution_authorized:
        raise ConfigurationError("final run blocked; explicitly pass allow_final=True")


def pilot_compatibility_payload(config: Experiment1Config) -> dict[str, Any]:
    """Scientific signature that a successful pilot must share with the final tier.

    Tier-scaled seeds, particle/batch budgets, integration steps, bootstrap/count
    budgets, and table resolutions are deliberately excluded.  Mathematical
    choices, estimands, checkpoint design, and fail-closed decision thresholds
    remain immutable.
    """

    exp = config.experiment
    design = config.design
    table = config.score_table
    analysis = config.analysis
    return {
        "experiment": {
            "name": exp.name,
            "horizon": exp.horizon,
            "beta": exp.beta,
            "epsilon": exp.epsilon,
            "precision": exp.precision,
        },
        "target": dataclasses.asdict(config.target),
        "stable": dataclasses.asdict(config.stable),
        "vp": dataclasses.asdict(config.vp),
        "controls": dataclasses.asdict(config.controls),
        "design": {
            "checkpoint_fractions": design.checkpoint_fractions,
            "initialization_arms": design.initialization_arms,
            "reference_samples": design.reference_samples,
            "direct_controls": design.direct_controls,
        },
        "score_table": {
            "stable_tail_l": table.stable_tail_l,
            "stable_x_max": table.stable_x_max,
            "stable_blend_start": table.stable_blend_start,
            "stable_blend_end": table.stable_blend_end,
            "vp_tail_l": table.vp_tail_l,
            "vp_x_max": table.vp_x_max,
            "vp_blend_start": table.vp_blend_start,
            "vp_blend_end": table.vp_blend_end,
            "sensitivity_refinement_factor": table.sensitivity_refinement_factor,
        },
        "analysis": {
            "wasserstein_orders": analysis.wasserstein_orders,
            "ecf_frequency_min": analysis.ecf_frequency_min,
            "ecf_frequency_max": analysis.ecf_frequency_max,
            "ecf_frequency_count": analysis.ecf_frequency_count,
            "ecf_weight": analysis.ecf_weight,
            "tail_fractions": analysis.tail_fractions,
            "primary_tail_fraction": analysis.primary_tail_fraction,
            "plateau_fractions": analysis.plateau_fractions,
            "survival_probabilities": analysis.survival_probabilities,
            "fit_primary": analysis.fit_primary,
            "fit_sensitivities": analysis.fit_sensitivities,
            "subsample_fractions": analysis.subsample_fractions,
            "blocks": analysis.blocks,
            "mean_excess_threshold_method": analysis.mean_excess_threshold_method,
            "mean_excess_fractions": analysis.mean_excess_fractions,
            "pt_ecf_familywise_alpha": analysis.pt_ecf_familywise_alpha,
            "score_sensitivity_floor_fraction": analysis.score_sensitivity_floor_fraction,
            "score_sensitivity_hill_tolerance": analysis.score_sensitivity_hill_tolerance,
            "score_sensitivity_constant_relative_tolerance": (
                analysis.score_sensitivity_constant_relative_tolerance
            ),
            "refinement_floor_fraction": analysis.refinement_floor_fraction,
            "refinement_hill_tolerance": analysis.refinement_hill_tolerance,
            "refinement_constant_relative_tolerance": (
                analysis.refinement_constant_relative_tolerance
            ),
            "refinement_m_slope_tolerance": analysis.refinement_m_slope_tolerance,
            "gaussian_hill_separation_margin": analysis.gaussian_hill_separation_margin,
        },
    }


def _dynamic_specs(config: Experiment1Config) -> tuple[tuple[Model, float], ...]:
    return tuple(("stable", nu) for nu in config.target.stable_nus) + tuple(
        ("vp", nu) for nu in config.target.vp_nus
    )


def enumerate_tasks(config: Experiment1Config) -> tuple[Experiment1Task, ...]:
    """Enumerate the immutable dynamic and direct-control task map."""

    tasks: list[Experiment1Task] = []

    def append_dynamic_group(
        seeds: tuple[int, ...],
        steps_group: tuple[int, ...],
        particles: int,
        control_particles: int,
        *,
        refinement_only: bool,
    ) -> None:
        if not seeds or not steps_group:
            return
        coupling_steps = max(steps_group)
        for model, nu in _dynamic_specs(config):
            for steps in steps_group:
                purpose: Purpose = (
                    "refinement"
                    if refinement_only or steps != config.design.primary_steps
                    else "primary"
                )
                for seed in seeds:
                    tasks.append(
                        Experiment1Task(
                            index=len(tasks),
                            kind="dynamic",
                            model=model,
                            nu=nu,
                            steps=steps,
                            seed=seed,
                            particles=particles,
                            control_particles=control_particles,
                            purpose=purpose,
                            coupling_steps=coupling_steps,
                        )
                    )

    append_dynamic_group(
        config.experiment.primary_seeds,
        config.design.primary_seed_steps,
        config.experiment.main_particles,
        config.experiment.control_particles,
        refinement_only=False,
    )
    append_dynamic_group(
        config.experiment.refinement_seeds,
        config.design.refinement_seed_steps,
        config.experiment.refinement_particles,
        config.experiment.refinement_control_particles,
        refinement_only=True,
    )
    for model in config.design.direct_controls:
        for seed in config.experiment.primary_seeds:
            tasks.append(
                Experiment1Task(
                    index=len(tasks),
                    kind="control",
                    model=model,
                    nu=None,
                    steps=0,
                    seed=seed,
                    particles=config.experiment.main_particles,
                    control_particles=0,
                    purpose="control",
                    coupling_steps=0,
                )
            )
    keys = {(task.kind, task.model, task.nu, task.steps, task.seed, task.purpose) for task in tasks}
    if len(keys) != len(tasks):
        raise ConfigurationError("internal error: Experiment 1 task map is not unique")
    return tuple(tasks)


def task_step_size(config: Experiment1Config, task: Experiment1Task) -> float:
    if task.kind != "dynamic":
        raise ConfigurationError("direct controls do not have a time step")
    return (config.experiment.horizon - config.experiment.epsilon) / task.steps


def checkpoint_step_indices(config: Experiment1Config, task: Experiment1Task) -> tuple[int, ...]:
    if task.kind != "dynamic":
        return ()
    return tuple(round(fraction * task.steps) for fraction in config.design.checkpoint_fractions)
