"""Strict, self-contained configuration for Experiment 2.

Experiment 2 measures propagation of the initialization error for the stable
and VP reverse SDEs.  A task is deliberately coarse grained: it owns both
initialization arms and both independent exact references for one
``(model, T, N, seed)`` tuple.  This prevents accidental cross-task pairing of
samples when empirical Wasserstein distances are computed.
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
Model = Literal["stable", "vp"]
Purpose = Literal["horizon_sweep", "refinement"]
InitializationArm = Literal["exact_p_T", "stationary_reference"]
ReferenceSample = Literal[
    "exact_p_epsilon_a",
    "exact_p_epsilon_b",
    "exact_p_epsilon_c",
]

FINAL_RUN_GUARD = "EXP2_FINAL_PUBLICATION_RUN"


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


def _float_tuple(name: str, value: Any) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{name} must be a non-empty TOML array")
    return tuple(_finite_float(name, item) for item in value)


def _positive_int_tuple(name: str, value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{name} must be a non-empty TOML array")
    return tuple(_positive_int(name, item) for item in value)


def _seed_tuple(name: str, value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{name} must be a non-empty TOML array")
    seeds: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or not 0 <= item <= 2**31 - 1:
            raise ConfigurationError(f"{name} must contain integers in [0, 2^31-1]")
        seeds.append(item)
    return tuple(seeds)


@dataclass(frozen=True)
class ExperimentSettings:
    name: str
    tier: RunTier
    beta: float
    epsilon: float
    seeds: tuple[int, ...]
    particles: int
    batch_size: int
    precision: Precision
    publication_scale: bool
    final_run_guard: str


@dataclass(frozen=True)
class TargetSettings:
    distribution: str
    degrees_of_freedom: float
    location: float
    scale: float


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
class HorizonSetting:
    horizon: float
    steps: int


@dataclass(frozen=True)
class DesignSettings:
    horizon_sweep: tuple[HorizonSetting, ...]
    refinement_horizon: float
    refinement_steps: tuple[int, ...]
    checkpoint_fractions: tuple[float, ...]
    initialization_arms: tuple[InitializationArm, ...]
    reference_samples: tuple[ReferenceSample, ...]


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


@dataclass(frozen=True)
class AnalysisSettings:
    wasserstein_orders: tuple[float, ...]
    ecf_frequency_min: float
    ecf_frequency_max: float
    ecf_frequency_count: int
    ecf_weight: str
    subsample_fractions: tuple[float, ...]
    chunk_size: int


@dataclass(frozen=True)
class RuntimeOverrides:
    """Audited command-line overrides applied after TOML parsing."""

    precision: Precision | None
    seed: int | None


@dataclass(frozen=True)
class Experiment2Config:
    experiment: ExperimentSettings
    target: TargetSettings
    stable: StableSettings
    vp: VPSettings
    design: DesignSettings
    score_table: ScoreTableSettings
    score_table_hires: ScoreTableSettings
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
class Experiment2Task:
    index: int
    model: Model
    horizon: float
    steps: int
    seed: int
    purposes: tuple[Purpose, ...]
    initialization_arms: tuple[InitializationArm, ...]
    reference_samples: tuple[ReferenceSample, ...]

    @property
    def task_id(self) -> str:
        canonical = json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))
        suffix = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
        horizon = f"{self.horizon:g}".replace(".", "p")
        return (
            f"task-{self.index:04d}-{self.model}-T{horizon}-N{self.steps}"
            f"-s{self.seed}-{suffix}"
        )


_EXPECTED_PROTOCOL: dict[RunTier, dict[str, Any]] = {
    "smoke": {
        "seeds": (0, 1),
        "particles": 2048,
        "batch_size": 1024,
        "horizon_sweep": ((0.25, 4), (2.0, 8)),
        "refinement_horizon": 2.0,
        "refinement_steps": (8, 16),
        "checkpoints": (0.0, 0.5, 1.0),
        "score_table": (
            17,
            513,
            131072,
            2048.0,
            192.0,
            128.0,
            192.0,
            32768,
            256.0,
            32.0,
            12.0,
            16.0,
        ),
        "score_table_hires": (
            33,
            1025,
            262144,
            2048.0,
            192.0,
            128.0,
            192.0,
            65536,
            256.0,
            32.0,
            12.0,
            16.0,
        ),
    },
    "pilot": {
        "seeds": tuple(range(200, 208)),
        "particles": 8192,
        "batch_size": 8192,
        "horizon_sweep": (
            (0.1, 4),
            (0.15, 8),
            (0.25, 16),
            (0.5, 36),
            (1.0, 76),
            (2.0, 156),
        ),
        "refinement_horizon": 2.0,
        "refinement_steps": (156, 312, 624),
        "checkpoints": (0.0, 0.25, 0.5, 0.75, 1.0),
        "score_table": (
            129,
            2049,
            262144,
            4096.0,
            256.0,
            192.0,
            256.0,
            131072,
            512.0,
            64.0,
            12.0,
            16.0,
        ),
        "score_table_hires": (
            257,
            4097,
            524288,
            4096.0,
            256.0,
            192.0,
            256.0,
            262144,
            512.0,
            64.0,
            12.0,
            16.0,
        ),
    },
    "final": {
        "seeds": tuple(range(1000, 1012)),
        "particles": 16384,
        "batch_size": 8192,
        "horizon_sweep": (
            (0.1, 4),
            (0.15, 8),
            (0.25, 16),
            (0.5, 36),
            (1.0, 76),
            (2.0, 156),
        ),
        "refinement_horizon": 2.0,
        "refinement_steps": (156, 312, 624),
        "checkpoints": (0.0, 0.25, 0.5, 0.75, 1.0),
        "score_table": (
            257,
            4097,
            524288,
            8192.0,
            384.0,
            256.0,
            384.0,
            262144,
            1024.0,
            96.0,
            12.0,
            16.0,
        ),
        "score_table_hires": (
            513,
            8193,
            1048576,
            8192.0,
            384.0,
            256.0,
            384.0,
            524288,
            1024.0,
            96.0,
            12.0,
            16.0,
        ),
    },
}


def _parse_horizon_sweep(value: Any) -> tuple[HorizonSetting, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError("design.horizon_sweep must be a non-empty array of tables")
    settings: list[HorizonSetting] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ConfigurationError(f"design.horizon_sweep[{index}] must be a table")
        _strict_keys(f"design.horizon_sweep[{index}]", item, {"horizon", "steps"})
        settings.append(
            HorizonSetting(
                horizon=_finite_float("design.horizon_sweep.horizon", item.get("horizon")),
                steps=_positive_int("design.horizon_sweep.steps", item.get("steps")),
            )
        )
    return tuple(settings)


_SCORE_TABLE_KEYS = {
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
}


def _parse_score_table_settings(raw: dict[str, Any], section: str) -> ScoreTableSettings:
    values = _section(raw, section, _SCORE_TABLE_KEYS)
    return ScoreTableSettings(
        time_points=_positive_int(f"{section}.time_points", values.get("time_points")),
        space_points=_positive_int(f"{section}.space_points", values.get("space_points")),
        stable_fft_points=_positive_int(
            f"{section}.stable_fft_points", values.get("stable_fft_points")
        ),
        stable_tail_l=_finite_float(
            f"{section}.stable_tail_l", values.get("stable_tail_l")
        ),
        stable_x_max=_finite_float(
            f"{section}.stable_x_max", values.get("stable_x_max")
        ),
        stable_blend_start=_finite_float(
            f"{section}.stable_blend_start", values.get("stable_blend_start")
        ),
        stable_blend_end=_finite_float(
            f"{section}.stable_blend_end", values.get("stable_blend_end")
        ),
        vp_fft_points=_positive_int(
            f"{section}.vp_fft_points", values.get("vp_fft_points")
        ),
        vp_tail_l=_finite_float(f"{section}.vp_tail_l", values.get("vp_tail_l")),
        vp_x_max=_finite_float(f"{section}.vp_x_max", values.get("vp_x_max")),
        vp_blend_start=_finite_float(
            f"{section}.vp_blend_start", values.get("vp_blend_start")
        ),
        vp_blend_end=_finite_float(
            f"{section}.vp_blend_end", values.get("vp_blend_end")
        ),
    )


def load_config(
    path: str | Path,
    *,
    precision_override: Precision | None = None,
    seed_override: int | None = None,
    allow_final: bool = False,
) -> Experiment2Config:
    """Load and strictly validate one Experiment 2 TOML configuration.

    ``allow_final`` records execution authorization but does not prevent a
    read-only dry run from loading and enumerating the final configuration.
    :func:`validate_execution` (also called by storage preparation) enforces
    the guard immediately before any output is created.
    """

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
            "design",
            "score_table",
            "score_table_hires",
            "analysis",
        },
    )

    exp_raw = _section(
        raw,
        "experiment",
        {
            "name",
            "tier",
            "beta",
            "epsilon",
            "seeds",
            "particles",
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
    experiment = ExperimentSettings(
        name=str(exp_raw.get("name")),
        tier=tier,
        beta=_finite_float("experiment.beta", exp_raw.get("beta")),
        epsilon=_finite_float("experiment.epsilon", exp_raw.get("epsilon")),
        seeds=(seed_override,)
        if seed_override is not None
        else _seed_tuple("experiment.seeds", exp_raw.get("seeds")),
        particles=_positive_int("experiment.particles", exp_raw.get("particles")),
        batch_size=_positive_int("experiment.batch_size", exp_raw.get("batch_size")),
        precision=precision_override if precision_override is not None else precision,
        publication_scale=publication_scale,
        final_run_guard=str(exp_raw.get("final_run_guard", "")),
    )

    target_raw = _section(
        raw,
        "target",
        {"distribution", "degrees_of_freedom", "location", "scale"},
    )
    target = TargetSettings(
        distribution=str(target_raw.get("distribution")),
        degrees_of_freedom=_finite_float(
            "target.degrees_of_freedom", target_raw.get("degrees_of_freedom")
        ),
        location=_finite_float("target.location", target_raw.get("location")),
        scale=_finite_float("target.scale", target_raw.get("scale")),
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

    design_raw = _section(
        raw,
        "design",
        {
            "horizon_sweep",
            "refinement_horizon",
            "refinement_steps",
            "checkpoint_fractions",
            "initialization_arms",
            "reference_samples",
        },
    )
    arms = design_raw.get("initialization_arms")
    references = design_raw.get("reference_samples")
    if not isinstance(arms, list) or not all(isinstance(item, str) for item in arms):
        raise ConfigurationError("design.initialization_arms must be a string array")
    if not isinstance(references, list) or not all(isinstance(item, str) for item in references):
        raise ConfigurationError("design.reference_samples must be a string array")
    design = DesignSettings(
        horizon_sweep=_parse_horizon_sweep(design_raw.get("horizon_sweep")),
        refinement_horizon=_finite_float(
            "design.refinement_horizon", design_raw.get("refinement_horizon")
        ),
        refinement_steps=_positive_int_tuple(
            "design.refinement_steps", design_raw.get("refinement_steps")
        ),
        checkpoint_fractions=_float_tuple(
            "design.checkpoint_fractions", design_raw.get("checkpoint_fractions")
        ),
        initialization_arms=tuple(arms),
        reference_samples=tuple(references),
    )

    score_table = _parse_score_table_settings(raw, "score_table")
    score_table_hires = _parse_score_table_settings(raw, "score_table_hires")

    analysis_raw = _section(
        raw,
        "analysis",
        {
            "wasserstein_orders",
            "ecf_frequency_min",
            "ecf_frequency_max",
            "ecf_frequency_count",
            "ecf_weight",
            "subsample_fractions",
            "chunk_size",
        },
    )
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
        subsample_fractions=_float_tuple(
            "analysis.subsample_fractions", analysis_raw.get("subsample_fractions")
        ),
        chunk_size=_positive_int("analysis.chunk_size", analysis_raw.get("chunk_size")),
    )

    config = Experiment2Config(
        experiment=experiment,
        target=target,
        stable=stable,
        vp=vp,
        design=design,
        score_table=score_table,
        score_table_hires=score_table_hires,
        analysis=analysis,
        overrides=RuntimeOverrides(precision=precision_override, seed=seed_override),
        source_path=str(source),
        source_hash=hashlib.sha256(source_bytes).hexdigest(),
        final_execution_authorized=allow_final,
    )
    validate_config(config)
    return config


def _score_table_tuple(table: ScoreTableSettings) -> tuple[int | float, ...]:
    return (
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
    )


def _validate_score_table_settings(table: ScoreTableSettings, *, name: str) -> None:
    if table.time_points % 2 == 0 or table.space_points % 2 == 0:
        raise ConfigurationError(f"{name} time_points and space_points must be odd")
    if table.stable_fft_points & (table.stable_fft_points - 1):
        raise ConfigurationError(f"{name} stable_fft_points must be a power of two")
    if table.vp_fft_points & (table.vp_fft_points - 1):
        raise ConfigurationError(f"{name} vp_fft_points must be a power of two")
    if not 0 < table.vp_blend_start < table.vp_blend_end <= table.vp_x_max:
        raise ConfigurationError(f"invalid {name} VP tail-blend interval")
    if not (
        0 < table.stable_blend_start < table.stable_blend_end <= table.stable_x_max
    ):
        raise ConfigurationError(f"invalid {name} stable tail-blend interval")
    if table.stable_tail_l <= table.stable_x_max:
        raise ConfigurationError(f"{name} stable_tail_l must exceed stable_x_max")
    if table.vp_tail_l <= table.vp_x_max:
        raise ConfigurationError(f"{name} vp_tail_l must exceed vp_x_max")


def validate_config(config: Experiment2Config) -> None:
    """Enforce the frozen mathematical protocol and resource tiers."""

    exp = config.experiment
    if exp.name != "experiment2":
        raise ConfigurationError("experiment.name must be 'experiment2'")
    if not math.isclose(exp.beta, 1.0) or not math.isclose(exp.epsilon, 0.05):
        raise ConfigurationError("Experiment 2 fixes beta=1 and epsilon=0.05")
    if len(set(exp.seeds)) != len(exp.seeds):
        raise ConfigurationError("experiment.seeds contains duplicates")
    if exp.particles % exp.batch_size:
        raise ConfigurationError("experiment.particles must be divisible by batch_size")

    target = config.target
    if (
        target.distribution != "scipy.stats.t"
        or not math.isclose(target.degrees_of_freedom, 4.0)
        or not math.isclose(target.location, 0.0)
        or not math.isclose(target.scale, 1.0)
    ):
        raise ConfigurationError("Experiment 2 fixes scipy.stats.t(df=4, loc=0, scale=1)")
    stable = config.stable
    if (
        not math.isclose(stable.alpha, 1.5)
        or not math.isclose(stable.eta, 0.5)
        or stable.stationary_law != "sas_unit"
        or stable.score != "fractional_tweedie_own_forward"
    ):
        raise ConfigurationError(
            "stable model must use alpha=1.5, eta=0.5, SalphaS(1), and its own forward score"
        )
    if config.vp != VPSettings("normal_standard", "gaussian_own_forward"):
        raise ConfigurationError("VP must use N(0,1) and its own Gaussian-forward score")

    design = config.design
    expected = _EXPECTED_PROTOCOL[exp.tier]
    if design.refinement_horizon != expected["refinement_horizon"]:
        raise ConfigurationError(
            "Experiment 2 refinement horizon differs from the frozen tier protocol"
        )
    if design.initialization_arms != ("exact_p_T", "stationary_reference"):
        raise ConfigurationError("every task must contain both prescribed initialization arms")
    if design.reference_samples != (
        "exact_p_epsilon_a",
        "exact_p_epsilon_b",
        "exact_p_epsilon_c",
    ):
        raise ConfigurationError("every task must contain three independent exact references")
    if tuple(sorted(design.checkpoint_fractions)) != design.checkpoint_fractions or len(
        set(design.checkpoint_fractions)
    ) != len(design.checkpoint_fractions):
        raise ConfigurationError("checkpoint fractions must be distinct and increasing")
    if design.checkpoint_fractions[0] != 0.0 or design.checkpoint_fractions[-1] != 1.0:
        raise ConfigurationError("checkpoint fractions must include 0 and 1")

    pairs = tuple((item.horizon, item.steps) for item in design.horizon_sweep)
    if len(set(pairs)) != len(pairs):
        raise ConfigurationError("design.horizon_sweep contains duplicate (T,N) pairs")
    if any(horizon <= exp.epsilon for horizon, _ in pairs):
        raise ConfigurationError("every horizon must exceed epsilon")
    if len(set(design.refinement_steps)) != len(design.refinement_steps):
        raise ConfigurationError("design.refinement_steps contains duplicates")
    finest_refinement_steps = max(design.refinement_steps)
    if any(finest_refinement_steps % steps for steps in design.refinement_steps):
        raise ConfigurationError(
            "every refinement N must divide the finest N for nested EI noise coupling"
        )
    all_points = set(pairs) | {(design.refinement_horizon, n) for n in design.refinement_steps}
    for _, steps in all_points:
        for fraction in design.checkpoint_fractions:
            index = steps * fraction
            if not math.isclose(index, round(index), rel_tol=0.0, abs_tol=1e-12):
                raise ConfigurationError(
                    f"checkpoint fraction {fraction} is not a grid point for N={steps}"
                )

    analysis = config.analysis
    if any(order < 1.0 or order >= stable.alpha for order in analysis.wasserstein_orders):
        raise ConfigurationError("stable Wasserstein orders must satisfy 1 <= p < alpha")
    if analysis.wasserstein_orders != (1.0, 1.25):
        raise ConfigurationError("wasserstein orders must be exactly [1, 1.25]")
    if (
        analysis.ecf_frequency_min != -5.0
        or analysis.ecf_frequency_max != 5.0
        or analysis.ecf_frequency_count != 81
        or analysis.ecf_weight != "exp(-u^2/2)"
    ):
        raise ConfigurationError("ECF protocol must be 81 points on [-5,5] with exp(-u^2/2)")
    if analysis.subsample_fractions != (0.25, 0.5, 1.0) or analysis.chunk_size != 4096:
        raise ConfigurationError(
            "analysis subsampling/chunk protocol differs from the frozen design"
        )
    for fraction in analysis.subsample_fractions:
        count = exp.particles * fraction
        if not math.isclose(count, round(count), rel_tol=0.0, abs_tol=1e-12):
            raise ConfigurationError("subsample fractions must produce integral particle counts")

    table = config.score_table
    hires = config.score_table_hires
    _validate_score_table_settings(table, name="score_table")
    _validate_score_table_settings(hires, name="score_table_hires")
    if hires.time_points != 2 * table.time_points - 1:
        raise ConfigurationError(
            "score_table_hires.time_points must be 2*score_table.time_points-1"
        )
    if hires.space_points != 2 * table.space_points - 1:
        raise ConfigurationError(
            "score_table_hires.space_points must be 2*score_table.space_points-1"
        )
    if hires.stable_fft_points != 2 * table.stable_fft_points:
        raise ConfigurationError(
            "score_table_hires.stable_fft_points must double the main FFT"
        )
    if hires.vp_fft_points != 2 * table.vp_fft_points:
        raise ConfigurationError("score_table_hires.vp_fft_points must double the main FFT")
    main_domains = _score_table_tuple(table)[3:7] + _score_table_tuple(table)[8:]
    hires_domains = _score_table_tuple(hires)[3:7] + _score_table_tuple(hires)[8:]
    if hires_domains != main_domains:
        raise ConfigurationError(
            "score_table_hires must keep the main domains and tail blends fixed"
        )

    actual_table = _score_table_tuple(table)
    actual_hires = _score_table_tuple(hires)
    expected_seeds = expected["seeds"]
    if config.overrides.seed is None:
        seeds_match = exp.seeds == expected_seeds
    else:
        seeds_match = (
            config.overrides.seed in expected_seeds
            and exp.seeds == (config.overrides.seed,)
        )
    if (
        not seeds_match
        or exp.particles != expected["particles"]
        or exp.batch_size != expected["batch_size"]
        or pairs != expected["horizon_sweep"]
        or design.refinement_horizon != expected["refinement_horizon"]
        or design.refinement_steps != expected["refinement_steps"]
        or design.checkpoint_fractions != expected["checkpoints"]
        or actual_table != expected["score_table"]
        or actual_hires != expected["score_table_hires"]
    ):
        raise ConfigurationError(f"configuration does not match the frozen {exp.tier} protocol")

    if exp.tier == "final":
        if not exp.publication_scale or exp.precision != "float64":
            raise ConfigurationError("final tier must be publication-scale float64")
        if exp.final_run_guard != FINAL_RUN_GUARD:
            raise ConfigurationError("final configuration is missing its immutable guard token")
    elif exp.publication_scale or exp.final_run_guard:
        raise ConfigurationError("smoke/pilot tiers cannot enable the final-run guard")


def validate_execution(config: Experiment2Config) -> None:
    """Block publication-scale writes unless the caller explicitly opted in."""

    validate_config(config)
    if config.experiment.tier == "final" and not config.final_execution_authorized:
        raise ConfigurationError("final run blocked; explicitly pass allow_final=True")


def pilot_compatibility_payload(config: Experiment2Config) -> dict[str, Any]:
    """Return scientific invariants that a pilot and final must share.

    Resource scale, tier guards, step counts and table resolutions are
    intentionally absent: the frozen protocol allows those quantities to grow
    from pilot to final.  Horizon values, mathematical models, observables and
    decision rules remain represented explicitly.
    """

    return {
        "schema_version": 1,
        "experiment": {
            "name": config.experiment.name,
            "beta": config.experiment.beta,
            "epsilon": config.experiment.epsilon,
            "precision": config.experiment.precision,
        },
        "target": dataclasses.asdict(config.target),
        "stable": dataclasses.asdict(config.stable),
        "vp": dataclasses.asdict(config.vp),
        "design": {
            "horizon_sweep": [
                {"horizon": setting.horizon}
                for setting in config.design.horizon_sweep
            ],
            "refinement_horizon": config.design.refinement_horizon,
            "checkpoint_fractions": list(config.design.checkpoint_fractions),
            "initialization_arms": list(config.design.initialization_arms),
            "reference_samples": list(config.design.reference_samples),
        },
        "analysis": dataclasses.asdict(config.analysis),
        "score_table_sensitivity_design": {
            "paired_initialization": "same_exact_p_T",
            "paired_primitive_noise": True,
            "time_points_rule": "hires=2*main-1",
            "space_points_rule": "hires=2*main-1",
            "fft_points_rule": "hires=2*main",
            "domains_and_tail_blends": "identical_between_main_and_hires",
        },
        "decision_rules": {
            "score_sensitivity": "W_p(main,hires)<=F/2",
            "wasserstein_floor": "independent_exact_a_vs_exact_b_not_subtracted",
            "refinement_scope": "fixed_T_and_epsilon_only",
            "refinement_finest_pair_tolerance": "max(F,0.1*value)",
            "refinement_fit": ">=3_controlled_points_and_D>2F",
            "horizon_fit": ">=3_horizons_and_I>2F_and_D<=max(2F,0.25I)",
            "minimum_seed_count_per_model_order": 3,
            "minimum_seed_coverage_fraction_per_model_order": 0.75,
            "triangle_violation_count": 0,
            "exact_ECF_within_simultaneous_radius_fraction": 1.0,
        },
    }


def design_points(config: Experiment2Config) -> tuple[tuple[float, int, tuple[Purpose, ...]], ...]:
    """Return unique ``(T,N,purposes)`` points in deterministic protocol order."""

    ordered: list[tuple[float, int]] = []
    purposes: dict[tuple[float, int], list[Purpose]] = {}
    for setting in config.design.horizon_sweep:
        key = (setting.horizon, setting.steps)
        if key not in purposes:
            ordered.append(key)
            purposes[key] = []
        purposes[key].append("horizon_sweep")
    for steps in config.design.refinement_steps:
        key = (config.design.refinement_horizon, steps)
        if key not in purposes:
            ordered.append(key)
            purposes[key] = []
        purposes[key].append("refinement")
    return tuple((horizon, steps, tuple(purposes[(horizon, steps)])) for horizon, steps in ordered)


def enumerate_tasks(config: Experiment2Config) -> tuple[Experiment2Task, ...]:
    """Enumerate unique tasks; each task contains both arms and both references."""

    tasks: list[Experiment2Task] = []
    index = 0
    for model in ("stable", "vp"):
        for horizon, steps, purposes in design_points(config):
            for seed in config.experiment.seeds:
                tasks.append(
                    Experiment2Task(
                        index=index,
                        model=model,
                        horizon=horizon,
                        steps=steps,
                        seed=seed,
                        purposes=purposes,
                        initialization_arms=config.design.initialization_arms,
                        reference_samples=config.design.reference_samples,
                    )
                )
                index += 1
    keys = {(task.model, task.horizon, task.steps, task.seed) for task in tasks}
    if len(keys) != len(tasks):
        raise ConfigurationError("internal error: Experiment 2 task map is not unique")
    return tuple(tasks)


def task_step_size(config: Experiment2Config, task: Experiment2Task) -> float:
    return (task.horizon - config.experiment.epsilon) / task.steps


def checkpoint_step_indices(config: Experiment2Config, task: Experiment2Task) -> tuple[int, ...]:
    return tuple(round(fraction * task.steps) for fraction in config.design.checkpoint_fractions)
