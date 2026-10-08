"""Strict TOML configuration and deterministic task enumeration for Experiment 3."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .errors import ConfigurationError

Precision = Literal["float32", "float64", "mixed"]


def _strict_keys(section: str, values: dict[str, Any], allowed: set[str]) -> None:
    unexpected = set(values) - allowed
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ConfigurationError(f"Unknown key(s) in [{section}]: {names}")


def _positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer, got {value!r}")
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigurationError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _float_tuple(name: str, value: Any) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{name} must be a non-empty TOML array")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must contain only numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise ConfigurationError(f"{name} contains a non-finite value")
    return result


def _int_tuple(name: str, value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{name} must be a non-empty TOML array")
    return tuple(_positive_int(name, item) for item in value)


def _seed_tuple(name: str, value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{name} must be a non-empty TOML array")
    result: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ConfigurationError(f"{name} must contain non-negative integers")
        result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class ExperimentSettings:
    alpha: float
    beta: float
    horizon: float
    epsilon: float
    etas: tuple[float, ...]
    seeds: tuple[int, ...]
    precision: Precision
    publication_scale: bool


@dataclass(frozen=True)
class StationarySettings:
    lags: tuple[float, ...]
    particles: int
    batch_size: int


@dataclass(frozen=True)
class NonstationarySettings:
    atoms: tuple[float, ...]
    weights: tuple[float, ...]
    checkpoints_forward: tuple[float, ...]
    steps: tuple[int, ...]
    particles: int
    batch_size: int
    hybrid_drift_eta: float
    hybrid_noise_eta: float


@dataclass(frozen=True)
class ScoreTableSettings:
    z_max: float
    points: int
    workers: int
    validation_points: int
    validation_rtol: float
    validation_atol: float


@dataclass(frozen=True)
class AnalysisSettings:
    marginal_frequency_max: float
    marginal_frequency_count: int
    joint_frequency_max: float
    joint_frequency_count: int
    joint_probe_u: float
    joint_probe_v: float
    mmd_bandwidth: float
    mmd_feature_count: int
    mmd_feature_seed: int
    sample_chunk_size: int
    refinement_tolerance_multiplier: float
    stationary_component_tolerance: float
    stationary_joint_separation_min: float
    stationary_min_particles: int
    hybrid_family_error: float
    hybrid_required_seed_fraction: float


@dataclass(frozen=True)
class Experiment3Config:
    experiment: ExperimentSettings
    stationary: StationarySettings
    nonstationary: NonstationarySettings
    score_table: ScoreTableSettings
    analysis: AnalysisSettings
    source_path: str
    source_hash: str

    @property
    def resolved_hash(self) -> str:
        payload = self.canonical_dict(include_source_path=False)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def canonical_dict(self, *, include_source_path: bool = True) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        if not include_source_path:
            payload.pop("source_path", None)
            payload.pop("source_hash", None)
        return payload


@dataclass(frozen=True)
class ExperimentTask:
    index: int
    target: Literal["stationary_sas", "three_atoms"]
    method: Literal["exact_transition", "exact_reference", "popov_ei", "hybrid_ei_invalid"]
    horizon: float
    eta: float
    noise_eta: float
    steps: int
    seed: int

    @property
    def task_id(self) -> str:
        canonical = json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))
        suffix = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
        return f"task-{self.index:04d}-{self.target}-{self.method}-s{self.seed}-{suffix}"


def _section(raw: dict[str, Any], name: str, allowed: set[str]) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Missing TOML section [{name}]")
    _strict_keys(name, value, allowed)
    return value


def load_config(
    path: str | Path,
    *,
    precision_override: Precision | None = None,
    seed_override: int | None = None,
) -> Experiment3Config:
    """Load and strictly validate an Experiment 3 TOML configuration."""

    source = Path(path).resolve()
    if not source.is_file():
        raise ConfigurationError(f"Configuration file not found: {source}")
    source_bytes = source.read_bytes()
    try:
        raw = tomllib.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"Invalid TOML configuration {source}: {exc}") from exc
    _strict_keys(
        "root", raw, {"experiment", "stationary", "nonstationary", "score_table", "analysis"}
    )

    exp = _section(
        raw,
        "experiment",
        {
            "name",
            "alpha",
            "beta",
            "horizon",
            "epsilon",
            "etas",
            "seeds",
            "precision",
            "publication_scale",
        },
    )
    if exp.get("name") != "experiment3":
        raise ConfigurationError("[experiment].name must be 'experiment3'")
    precision = precision_override or exp.get("precision")
    if precision not in {"float32", "float64", "mixed"}:
        raise ConfigurationError("precision must be float32, float64, or mixed")
    seeds = (
        (seed_override,)
        if seed_override is not None
        else _seed_tuple("experiment.seeds", exp.get("seeds"))
    )
    if any(seed < 0 or seed > 2**31 - 1 for seed in seeds):
        raise ConfigurationError("seeds must be in [0, 2^31-1]")
    if len(set(seeds)) != len(seeds):
        raise ConfigurationError("experiment.seeds contains duplicates")
    experiment = ExperimentSettings(
        alpha=float(exp.get("alpha")),
        beta=float(exp.get("beta")),
        horizon=float(exp.get("horizon")),
        epsilon=float(exp.get("epsilon")),
        etas=_float_tuple("experiment.etas", exp.get("etas")),
        seeds=seeds,
        precision=precision,
        publication_scale=bool(exp.get("publication_scale", False)),
    )

    stat = _section(raw, "stationary", {"lags", "particles", "batch_size"})
    stationary = StationarySettings(
        lags=_float_tuple("stationary.lags", stat.get("lags")),
        particles=_positive_int("stationary.particles", stat.get("particles")),
        batch_size=_positive_int("stationary.batch_size", stat.get("batch_size")),
    )

    nonstat = _section(
        raw,
        "nonstationary",
        {
            "atoms",
            "weights",
            "checkpoints_forward",
            "steps",
            "particles",
            "batch_size",
            "hybrid_drift_eta",
            "hybrid_noise_eta",
        },
    )
    nonstationary = NonstationarySettings(
        atoms=_float_tuple("nonstationary.atoms", nonstat.get("atoms")),
        weights=_float_tuple("nonstationary.weights", nonstat.get("weights")),
        checkpoints_forward=_float_tuple(
            "nonstationary.checkpoints_forward", nonstat.get("checkpoints_forward")
        ),
        steps=_int_tuple("nonstationary.steps", nonstat.get("steps")),
        particles=_positive_int("nonstationary.particles", nonstat.get("particles")),
        batch_size=_positive_int("nonstationary.batch_size", nonstat.get("batch_size")),
        hybrid_drift_eta=float(nonstat.get("hybrid_drift_eta")),
        hybrid_noise_eta=float(nonstat.get("hybrid_noise_eta")),
    )

    table = _section(
        raw,
        "score_table",
        {"z_max", "points", "workers", "validation_points", "validation_rtol", "validation_atol"},
    )
    score_table = ScoreTableSettings(
        z_max=float(table.get("z_max")),
        points=_positive_int("score_table.points", table.get("points")),
        workers=_positive_int("score_table.workers", table.get("workers")),
        validation_points=_positive_int(
            "score_table.validation_points", table.get("validation_points")
        ),
        validation_rtol=float(table.get("validation_rtol")),
        validation_atol=float(table.get("validation_atol")),
    )

    analysis_raw = _section(
        raw,
        "analysis",
        {
            "marginal_frequency_max",
            "marginal_frequency_count",
            "joint_frequency_max",
            "joint_frequency_count",
            "joint_probe_u",
            "joint_probe_v",
            "mmd_bandwidth",
            "mmd_feature_count",
            "mmd_feature_seed",
            "sample_chunk_size",
            "refinement_tolerance_multiplier",
            "stationary_component_tolerance",
            "stationary_joint_separation_min",
            "stationary_min_particles",
            "hybrid_family_error",
            "hybrid_required_seed_fraction",
        },
    )
    analysis = AnalysisSettings(
        marginal_frequency_max=float(analysis_raw.get("marginal_frequency_max")),
        marginal_frequency_count=_positive_int(
            "analysis.marginal_frequency_count", analysis_raw.get("marginal_frequency_count")
        ),
        joint_frequency_max=float(analysis_raw.get("joint_frequency_max")),
        joint_frequency_count=_positive_int(
            "analysis.joint_frequency_count", analysis_raw.get("joint_frequency_count")
        ),
        joint_probe_u=float(analysis_raw.get("joint_probe_u")),
        joint_probe_v=float(analysis_raw.get("joint_probe_v")),
        mmd_bandwidth=float(analysis_raw.get("mmd_bandwidth")),
        mmd_feature_count=_positive_int(
            "analysis.mmd_feature_count", analysis_raw.get("mmd_feature_count")
        ),
        mmd_feature_seed=_nonnegative_int(
            "analysis.mmd_feature_seed", analysis_raw.get("mmd_feature_seed")
        ),
        sample_chunk_size=_positive_int(
            "analysis.sample_chunk_size", analysis_raw.get("sample_chunk_size")
        ),
        refinement_tolerance_multiplier=float(
            analysis_raw.get("refinement_tolerance_multiplier")
        ),
        stationary_component_tolerance=float(
            analysis_raw.get("stationary_component_tolerance")
        ),
        stationary_joint_separation_min=float(
            analysis_raw.get("stationary_joint_separation_min")
        ),
        stationary_min_particles=_positive_int(
            "analysis.stationary_min_particles",
            analysis_raw.get("stationary_min_particles"),
        ),
        hybrid_family_error=float(analysis_raw.get("hybrid_family_error")),
        hybrid_required_seed_fraction=float(
            analysis_raw.get("hybrid_required_seed_fraction")
        ),
    )

    config = Experiment3Config(
        experiment=experiment,
        stationary=stationary,
        nonstationary=nonstationary,
        score_table=score_table,
        analysis=analysis,
        source_path=str(source),
        source_hash=hashlib.sha256(source_bytes).hexdigest(),
    )
    validate_config(config)
    return config


def validate_config(config: Experiment3Config) -> None:
    """Apply mathematical and computational consistency checks."""

    exp = config.experiment
    if not 1.0 < exp.alpha < 2.0:
        raise ConfigurationError("Experiment 3 requires 1 < alpha < 2")
    if not math.isclose(exp.alpha, 1.5, rel_tol=0.0, abs_tol=1e-12):
        raise ConfigurationError("The prescribed Experiment 3 protocol fixes alpha=1.5")
    if exp.beta <= 0.0 or not math.isfinite(exp.beta):
        raise ConfigurationError("beta must be finite and positive")
    if not 0.0 < exp.epsilon < exp.horizon:
        raise ConfigurationError("epsilon must satisfy 0 < epsilon < horizon")
    if any(eta <= 0.0 for eta in exp.etas) or len(set(exp.etas)) != len(exp.etas):
        raise ConfigurationError("etas must be distinct and strictly positive")
    if exp.publication_scale and exp.precision != "float64":
        raise ConfigurationError("publication-scale configurations must use float64")
    if any(lag <= 0.0 for lag in config.stationary.lags):
        raise ConfigurationError("stationary lags must be positive")
    if config.stationary.particles % config.stationary.batch_size:
        raise ConfigurationError("stationary.particles must be divisible by batch_size")

    ns = config.nonstationary
    prescribed_atoms = (-3.0, 0.5, 2.0)
    prescribed_weights = (0.25, 0.5, 0.25)
    if len(ns.atoms) != 3 or any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
        for actual, expected in zip(ns.atoms, prescribed_atoms, strict=False)
    ):
        raise ConfigurationError(
            f"the prescribed three-atom target fixes atoms={prescribed_atoms}"
        )
    if len(ns.weights) != 3 or any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
        for actual, expected in zip(ns.weights, prescribed_weights, strict=False)
    ):
        raise ConfigurationError(
            f"the prescribed three-atom target fixes weights={prescribed_weights}"
        )
    if len(ns.atoms) != len(ns.weights) or len(ns.atoms) < 2:
        raise ConfigurationError("atoms and weights must have the same length >= 2")
    if any(weight <= 0.0 for weight in ns.weights) or not math.isclose(
        sum(ns.weights), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ConfigurationError("nonstationary weights must be positive and sum to one")
    mean = sum(weight * atom for weight, atom in zip(ns.weights, ns.atoms, strict=True))
    if not math.isclose(mean, 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ConfigurationError("the prescribed three-atom target must be centered")
    if tuple(sorted(ns.checkpoints_forward, reverse=True)) != ns.checkpoints_forward:
        raise ConfigurationError(
            "checkpoints_forward must be strictly ordered from T toward epsilon"
        )
    if len(set(ns.checkpoints_forward)) != len(ns.checkpoints_forward):
        raise ConfigurationError("checkpoints_forward contains duplicates")
    if any(not exp.epsilon < time < exp.horizon for time in ns.checkpoints_forward):
        raise ConfigurationError(
            "configured checkpoints must lie strictly inside (epsilon, horizon); "
            "the terminal epsilon is recorded automatically"
        )
    if ns.particles % ns.batch_size:
        raise ConfigurationError("nonstationary.particles must be divisible by batch_size")
    if ns.hybrid_drift_eta <= 0.0 or ns.hybrid_noise_eta <= 0.0:
        raise ConfigurationError("hybrid eta values must be positive")
    if math.isclose(ns.hybrid_drift_eta, ns.hybrid_noise_eta):
        raise ConfigurationError("the invalid hybrid control must use two different eta values")
    duration = exp.horizon - exp.epsilon
    if any(steps <= 0 for steps in ns.steps):
        raise ConfigurationError("nonstationary step counts must be strictly positive")
    if tuple(sorted(ns.steps)) != ns.steps or len(set(ns.steps)) != len(ns.steps):
        raise ConfigurationError(
            "nonstationary.steps must be strictly increasing without duplicates"
        )
    finest_steps = max(ns.steps)
    if any(finest_steps % steps != 0 for steps in ns.steps):
        raise ConfigurationError(
            "every nonstationary step count must divide the finest count for nested CRN"
        )
    if len(ns.steps) > 1 and any(
        fine != 2 * coarse for coarse, fine in zip(ns.steps, ns.steps[1:], strict=False)
    ):
        raise ConfigurationError(
            "Experiment 3 refinement levels must use the prespecified factor-two nesting"
        )
    for steps in ns.steps:
        h = duration / steps
        for forward_time in ns.checkpoints_forward:
            backward_time = exp.horizon - forward_time
            grid_index = backward_time / h
            if not math.isclose(grid_index, round(grid_index), rel_tol=0.0, abs_tol=1e-10):
                raise ConfigurationError(
                    f"forward checkpoint {forward_time} is not a grid point for N={steps}"
                )

    table = config.score_table
    if table.z_max <= 1.0 or table.points < 64 or table.validation_points < 8:
        raise ConfigurationError(
            "score table requires z_max > 1, >=64 points and >=8 validation points"
        )
    if table.validation_rtol <= 0.0 or table.validation_atol < 0.0:
        raise ConfigurationError("score-table validation tolerances are invalid")
    analysis = config.analysis
    if analysis.marginal_frequency_max <= 0.0 or analysis.joint_frequency_max <= 0.0:
        raise ConfigurationError("frequency domains must be positive")
    if analysis.marginal_frequency_count % 2 == 0 or analysis.joint_frequency_count % 2 == 0:
        raise ConfigurationError("frequency counts must be odd so that zero is represented exactly")
    if analysis.mmd_bandwidth <= 0.0:
        raise ConfigurationError("MMD bandwidth must be positive")
    if not math.isclose(
        analysis.refinement_tolerance_multiplier,
        0.25,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ConfigurationError(
            "the prespecified refinement tolerance multiplier must equal 0.25"
        )
    frozen_gate_values = (
        ("stationary_component_tolerance", analysis.stationary_component_tolerance, 0.006),
        ("stationary_joint_separation_min", analysis.stationary_joint_separation_min, 0.02),
        ("hybrid_family_error", analysis.hybrid_family_error, 0.05),
        ("hybrid_required_seed_fraction", analysis.hybrid_required_seed_fraction, 0.75),
    )
    for name, actual, expected in frozen_gate_values:
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-15):
            raise ConfigurationError(f"the prespecified {name} must equal {expected:g}")
    if analysis.stationary_min_particles != 200_000:
        raise ConfigurationError("the prespecified stationary_min_particles must equal 200000")
    if not (
        math.isclose(analysis.joint_probe_u, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(analysis.joint_probe_v, 0.5, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise ConfigurationError(
            "the prescribed joint probe is (u,v)=(1,0.5); null or symmetric probes are forbidden"
        )
    from .theory import stationary_joint_cf, stationary_true_reverse_joint_cf

    low_eta, high_eta = min(exp.etas), max(exp.etas)
    probe_u = [analysis.joint_probe_u]
    probe_v = [analysis.joint_probe_v]
    eta_separations: list[float] = []
    reverse_separations: list[float] = []
    for lag in config.stationary.lags:
        low = stationary_joint_cf(
            probe_u,
            probe_v,
            alpha=exp.alpha,
            beta=exp.beta,
            eta=low_eta,
            lag=lag,
        )[0]
        high = stationary_joint_cf(
            probe_u,
            probe_v,
            alpha=exp.alpha,
            beta=exp.beta,
            eta=high_eta,
            lag=lag,
        )[0]
        true_reverse = stationary_true_reverse_joint_cf(
            probe_u,
            probe_v,
            alpha=exp.alpha,
            beta=exp.beta,
            lag=lag,
        )[0]
        eta_separations.append(float(abs(high - low)))
        reverse_separations.extend(
            (float(abs(low - true_reverse)), float(abs(high - true_reverse)))
        )
    if max(eta_separations) < analysis.stationary_joint_separation_min:
        raise ConfigurationError(
            "the stationary eta design has insufficient analytic joint-CF separation"
        )
    if max(reverse_separations) < analysis.stationary_joint_separation_min:
        raise ConfigurationError(
            "the stationary design cannot distinguish Popov pairs from the true reverse pair"
        )


def pilot_compatibility_payload(config: Experiment3Config) -> dict[str, Any]:
    """Return scientific choices that must agree between pilot and final tiers.

    Only execution budgets, tier guards, and score-table resolution controls are
    removed. The explicit target/method inventory prevents an implementation
    change from remaining invisible merely because it is not a TOML field.
    """

    return {
        "schema": "experiment3-pilot-compatibility-v1",
        "experiment": {
            "alpha": config.experiment.alpha,
            "beta": config.experiment.beta,
            "horizon": config.experiment.horizon,
            "epsilon": config.experiment.epsilon,
            "etas": list(config.experiment.etas),
            "precision": config.experiment.precision,
        },
        "stationary": {"lags": list(config.stationary.lags)},
        "nonstationary": {
            "atoms": list(config.nonstationary.atoms),
            "weights": list(config.nonstationary.weights),
            "checkpoints_forward": list(config.nonstationary.checkpoints_forward),
            "steps": list(config.nonstationary.steps),
            "hybrid_drift_eta": config.nonstationary.hybrid_drift_eta,
            "hybrid_noise_eta": config.nonstationary.hybrid_noise_eta,
        },
        "score_table": {
            "z_max": config.score_table.z_max,
            "validation_rtol": config.score_table.validation_rtol,
            "validation_atol": config.score_table.validation_atol,
        },
        "analysis": {
            field.name: getattr(config.analysis, field.name)
            for field in dataclasses.fields(config.analysis)
            if field.name != "sample_chunk_size"
        },
        "targets": ["stationary_sas", "three_atoms"],
        "methods": [
            "exact_transition",
            "exact_reference",
            "popov_ei",
            "hybrid_ei_invalid",
        ],
    }


def pilot_compatibility_hash(config: Experiment3Config) -> str:
    """Hash :func:`pilot_compatibility_payload` canonically."""

    canonical = json.dumps(
        pilot_compatibility_payload(config), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def enumerate_tasks(config: Experiment3Config) -> tuple[ExperimentTask, ...]:
    """Map array indices deterministically to independent Experiment 3 tasks."""

    tasks: list[ExperimentTask] = []
    index = 0
    for lag in config.stationary.lags:
        for eta in config.experiment.etas:
            for seed in config.experiment.seeds:
                tasks.append(
                    ExperimentTask(
                        index, "stationary_sas", "exact_transition", lag, eta, eta, 0, seed
                    )
                )
                index += 1
    duration = config.experiment.horizon - config.experiment.epsilon
    for steps in config.nonstationary.steps:
        for seed in config.experiment.seeds:
            tasks.append(
                ExperimentTask(
                    index, "three_atoms", "exact_reference", duration, 0.0, 0.0, steps, seed
                )
            )
            index += 1
        for eta in config.experiment.etas:
            for seed in config.experiment.seeds:
                tasks.append(
                    ExperimentTask(
                        index, "three_atoms", "popov_ei", duration, eta, eta, steps, seed
                    )
                )
                index += 1
        for seed in config.experiment.seeds:
            tasks.append(
                ExperimentTask(
                    index,
                    "three_atoms",
                    "hybrid_ei_invalid",
                    duration,
                    config.nonstationary.hybrid_drift_eta,
                    config.nonstationary.hybrid_noise_eta,
                    steps,
                    seed,
                )
            )
            index += 1
    return tuple(tasks)
