"""Read-only, seed-first statistical analysis for Experiment 1.

The module consumes already-simulated NPZ task artifacts and never imports a
simulator.  A task always contains ``terminal_sample``.  A forward-marginal
control may additionally contain ``control_forward_times`` and the four
matrices ``control_numerical`` and ``control_exact_a/b/c``.

Scalar NPZ entries are retained as metadata.  The canonical scalar names are
``task_kind``, ``model``, ``seed``, ``alpha``, ``eta``, ``beta``, ``T``,
``epsilon``, ``steps``, ``target`` and ``target_nu``; documented aliases are
accepted so that immutable raw artifacts do not need to be rewritten merely
for aggregation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from levy_experiments.errors import ArtifactError
from levy_experiments.metrics.characteristic import empirical_cf_1d, weighted_cf_rmse

from .metrics import (
    TailMetricError,
    discrete_light_stable_scale,
    discrete_light_vp_scale,
    gaussian_reference_thresholds,
    gpd_at_k,
    hill_grid,
    mean_excess_sufficient,
    pickands_grid,
    pooled_mean_excess,
    power_law_fit,
    radial_quantiles,
    sas_abs_tail_constant,
    sorted_radial,
    stable_reference_thresholds,
    student_abs_quantiles,
    tail_constant_at_k,
    tail_count,
)
from .theory import forward_marginal_cf

TAIL_FRACTIONS = (0.003, 0.004, 0.006, 0.008, 0.012, 0.018, 0.030)
PRIMARY_FRACTION = 0.006
PLATEAU_FRACTIONS = (0.004, 0.006, 0.008)
GPD_FRACTIONS = (0.003, 0.006, 0.012)
MEAN_EXCESS_FRACTIONS = (0.012, 0.006, 0.003)
DEFAULT_M_MASSES = tuple(float(value) for value in np.geomspace(0.02, 0.001, 8))
DEFAULT_M_PRIMARY_WINDOW = (0.001, 0.010)
DEFAULT_M_SENSITIVITY_WINDOWS = ((0.002, 0.020), (0.001, 0.005))

ALPHA_MARGIN = 0.10
XI_MARGIN = 0.05
PLATEAU_SLOPE_MARGIN = 0.03
M_SLOPE_MARGIN = 0.05
CONSTANT_RELATIVE_MARGIN = 0.25
MINIMUM_GPD_K = 8
MINIMUM_EXCEEDANCES_PER_SEED = 30
MINIMUM_POOLED_EXCEEDANCES = 200
MINIMUM_QUALIFYING_SEEDS = 3


class Exp1ArtifactError(ArtifactError):
    """Raised when an Experiment 1 task or aggregate is invalid."""


@dataclass(frozen=True)
class TaskData:
    """Validated in-memory representation of one immutable raw task."""

    path: Path
    task_kind: str
    purpose: str
    model: str
    control_distribution: str
    target: str
    target_nu: float
    alpha: float
    eta: float
    beta: float
    horizon: float
    epsilon: float
    steps: int
    seed: int
    terminal_sample: np.ndarray
    control_forward_times: np.ndarray | None
    control_numerical: np.ndarray | None
    control_exact_a: np.ndarray | None
    control_exact_b: np.ndarray | None
    control_exact_c: np.ndarray | None
    score_hires_terminal: np.ndarray | None
    score_sensitivity_forward_times: np.ndarray | None
    score_sensitivity_mean_abs: np.ndarray | None
    metadata: dict[str, Any]

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.task_kind,
            self.purpose,
            self.model,
            self.control_distribution,
            self.target,
            self.target_nu,
            self.alpha,
            self.horizon,
            self.steps,
            self.seed,
        )

    @property
    def sample_count(self) -> int:
        return int(self.terminal_sample.size)

    @property
    def is_control(self) -> bool:
        return self.task_kind in {"control", "tail_control", "dynamic_control"} or bool(
            self.control_distribution
        )

    @property
    def is_stable_tail(self) -> bool:
        return self.model == "stable" or self.control_distribution in {"pareto", "sas"}


@dataclass(frozen=True)
class AnalysisSettings:
    """Frozen downstream choices; none changes a simulated trajectory."""

    tier: str = "final"
    bootstrap_replicates: int | None = None
    subsample_fractions: tuple[float, ...] = (0.25, 0.5)
    block_count: int = 4
    m_masses: tuple[float, ...] = DEFAULT_M_MASSES
    m_primary_window: tuple[float, float] = DEFAULT_M_PRIMARY_WINDOW
    m_sensitivity_windows: tuple[tuple[float, float], ...] = DEFAULT_M_SENSITIVITY_WINDOWS
    ecf_frequency_max: float = 5.0
    ecf_frequency_count: int = 81
    ecf_chunk_size: int = 8192
    minimum_exceedances_per_seed: int = MINIMUM_EXCEEDANCES_PER_SEED
    minimum_pooled_exceedances: int = MINIMUM_POOLED_EXCEEDANCES
    minimum_contributing_seeds: int = MINIMUM_QUALIFYING_SEEDS
    pt_ecf_familywise_alpha: float = 0.01
    score_sensitivity_floor_fraction: float = 0.5
    score_sensitivity_hill_tolerance: float = 0.05
    score_sensitivity_constant_relative_tolerance: float = 0.10
    refinement_floor_fraction: float = 1.0
    refinement_hill_tolerance: float = 0.10
    refinement_constant_relative_tolerance: float = 0.25
    refinement_m_slope_tolerance: float = 0.05
    gaussian_hill_separation_margin: float = 0.25

    @property
    def resolved_bootstrap_replicates(self) -> int:
        if self.bootstrap_replicates is not None:
            return self.bootstrap_replicates
        return {"smoke": 0, "pilot": 2_000, "final": 10_000}[self.tier]

    @property
    def minimum_gpd_k(self) -> int:
        return {"smoke": 8, "pilot": 100, "final": 1_000}[self.tier]

    @property
    def m_windows(self) -> dict[str, tuple[float, float]]:
        if len(self.m_sensitivity_windows) != 2:
            raise Exp1ArtifactError("exactly two M-ratio sensitivity windows are required")
        return {
            "primary": self.m_primary_window,
            "broad": self.m_sensitivity_windows[0],
            "extreme": self.m_sensitivity_windows[1],
        }

    def validate(self) -> None:
        if self.tier not in {"smoke", "pilot", "final"}:
            raise Exp1ArtifactError(f"unknown analysis tier: {self.tier!r}")
        if self.resolved_bootstrap_replicates < 0:
            raise Exp1ArtifactError("bootstrap replicate count must be non-negative")
        if (
            tuple(sorted(set(self.subsample_fractions))) != self.subsample_fractions
            or any(not 0.0 < value < 1.0 for value in self.subsample_fractions)
            or self.block_count < 2
        ):
            raise Exp1ArtifactError("invalid nested/block subsampling design")
        masses = np.asarray(self.m_masses, dtype=np.float64)
        if (
            masses.size < 4
            or np.any(~np.isfinite(masses))
            or np.any((masses <= 0.0) | (masses >= 1.0))
            or np.unique(masses).size != masses.size
        ):
            raise Exp1ArtifactError("M-mass grid requires at least four distinct values in (0,1)")
        for name, (lower, upper) in self.m_windows.items():
            if not (0.0 < lower < upper < 1.0):
                raise Exp1ArtifactError(f"invalid M-ratio window {name}: {(lower, upper)}")
        if (
            not math.isfinite(self.ecf_frequency_max)
            or self.ecf_frequency_max <= 0.0
            or self.ecf_frequency_count < 3
            or self.ecf_frequency_count % 2 == 0
            or self.ecf_chunk_size <= 0
            or self.minimum_exceedances_per_seed <= 0
            or self.minimum_pooled_exceedances <= 0
            or self.minimum_contributing_seeds <= 0
        ):
            raise Exp1ArtifactError("invalid ECF analysis settings")
        fractions = (self.score_sensitivity_floor_fraction, self.refinement_floor_fraction)
        tolerances = (
            self.score_sensitivity_hill_tolerance,
            self.score_sensitivity_constant_relative_tolerance,
            self.refinement_hill_tolerance,
            self.refinement_constant_relative_tolerance,
            self.refinement_m_slope_tolerance,
            self.gaussian_hill_separation_margin,
        )
        if not 0.0 < self.pt_ecf_familywise_alpha < 1.0:
            raise Exp1ArtifactError("pT ECF family-wise alpha must lie in (0,1)")
        if any(not 0.0 < value <= 1.0 for value in fractions) or any(
            not math.isfinite(value) or value <= 0.0 for value in tolerances
        ):
            raise Exp1ArtifactError("scientific gate tolerances must be finite and positive")


def _decode_scalar(value: np.ndarray) -> Any:
    item = value.reshape(()).item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    if isinstance(item, np.generic):
        return item.item()
    return item


def _scalars(archive: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name in archive.files:
        value = np.asarray(archive[name])
        if value.size == 1 and value.ndim <= 1:
            metadata[name] = _decode_scalar(value)
    for key in ("metadata", "task_metadata"):
        value = metadata.get(key)
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as exc:
                raise Exp1ArtifactError(f"{key} is not valid JSON") from exc
            if not isinstance(decoded, dict):
                raise Exp1ArtifactError(f"{key} JSON must encode an object")
            metadata.update(decoded)
    return metadata


def _first(metadata: dict[str, Any], names: tuple[str, ...], default: Any) -> Any:
    for name in names:
        if name in metadata:
            return metadata[name]
    return default


def _float_metadata(
    metadata: dict[str, Any], names: tuple[str, ...], default: float = math.nan
) -> float:
    raw = _first(metadata, names, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise Exp1ArtifactError(f"metadata {names[0]} must be numeric") from exc
    if not math.isnan(value) and not math.isfinite(value):
        raise Exp1ArtifactError(f"metadata {names[0]} must be finite")
    return value


def _int_metadata(metadata: dict[str, Any], names: tuple[str, ...], default: int) -> int:
    raw = _first(metadata, names, default)
    if isinstance(raw, bool):
        raise Exp1ArtifactError(f"metadata {names[0]} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise Exp1ArtifactError(f"metadata {names[0]} must be an integer") from exc
    if value != float(raw):
        raise Exp1ArtifactError(f"metadata {names[0]} must be an integer")
    return value


def _finite_vector(archive: Any, name: str, *, minimum: int) -> np.ndarray:
    if name not in archive:
        raise Exp1ArtifactError(f"missing NPZ key: {name}")
    values = np.asarray(archive[name])
    if (
        values.ndim != 1
        or not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.complexfloating)
    ):
        raise Exp1ArtifactError(f"{name} must be a one-dimensional real array")
    values = np.asarray(values, dtype=np.float64)
    if values.size < minimum or np.any(~np.isfinite(values)):
        raise Exp1ArtifactError(f"{name} must contain at least {minimum} finite values")
    return values


def _finite_matrix(archive: Any, name: str) -> np.ndarray:
    if name not in archive:
        raise Exp1ArtifactError(f"missing NPZ key: {name}")
    values = np.asarray(archive[name])
    if (
        values.ndim != 2
        or not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.complexfloating)
    ):
        raise Exp1ArtifactError(f"{name} must be a two-dimensional real array")
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1] < 8 or np.any(~np.isfinite(values)):
        raise Exp1ArtifactError(f"{name} rows must contain at least eight finite samples")
    return values


def load_task_npz(path: str | Path) -> TaskData:
    """Load a raw task without allowing pickles or silently repairing arrays."""

    source = Path(path).resolve()
    if not source.is_file():
        raise Exp1ArtifactError(f"task NPZ not found: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            metadata = _scalars(archive)
            terminal = _finite_vector(archive, "terminal_sample", minimum=8)
            dynamic_names = {
                "control_forward_times",
                "control_numerical",
                "control_exact_a",
                "control_exact_b",
                "control_exact_c",
                "score_hires_terminal",
                "score_sensitivity_forward_times",
                "score_sensitivity_mean_abs",
            }
            supplied_dynamic = dynamic_names.intersection(archive.files)
            if supplied_dynamic and supplied_dynamic != dynamic_names:
                missing = sorted(dynamic_names - supplied_dynamic)
                raise Exp1ArtifactError(f"incomplete dynamic control arrays: missing={missing}")
            if supplied_dynamic:
                times = _finite_vector(archive, "control_forward_times", minimum=1)
                numerical = _finite_matrix(archive, "control_numerical")
                exact_a = _finite_matrix(archive, "control_exact_a")
                exact_b = _finite_matrix(archive, "control_exact_b")
                exact_c = _finite_matrix(archive, "control_exact_c")
                score_hires = _finite_vector(
                    archive, "score_hires_terminal", minimum=terminal.size
                )
                sensitivity_times = _finite_vector(
                    archive, "score_sensitivity_forward_times", minimum=1
                )
                sensitivity_mean_abs = _finite_vector(
                    archive, "score_sensitivity_mean_abs", minimum=1
                )
                if not (numerical.shape == exact_a.shape == exact_b.shape == exact_c.shape):
                    raise Exp1ArtifactError("dynamic control matrices must have identical shapes")
                if times.size != numerical.shape[0] or np.any(np.diff(times) <= 0.0):
                    raise Exp1ArtifactError(
                        "control_forward_times must be increasing and match matrix rows"
                    )
                if score_hires.shape != terminal.shape:
                    raise Exp1ArtifactError("main and high-resolution terminal samples must match")
                if (
                    sensitivity_times.shape != sensitivity_mean_abs.shape
                    or np.any(np.diff(sensitivity_times) <= 0.0)
                    or not np.array_equal(sensitivity_times, times)
                ):
                    raise Exp1ArtifactError(
                        "score-sensitivity checkpoints must match the forward-control grid"
                    )
            else:
                times = numerical = exact_a = exact_b = exact_c = None
                score_hires = sensitivity_times = sensitivity_mean_abs = None
    except (OSError, ValueError) as exc:
        if isinstance(exc, Exp1ArtifactError):
            raise
        raise Exp1ArtifactError(f"cannot read task NPZ: {source}") from exc

    task_kind = str(_first(metadata, ("task_kind", "kind"), "main")).strip().lower()
    model = str(_first(metadata, ("model", "sampler"), "")).strip().lower()
    control = str(_first(metadata, ("control_distribution", "control_kind"), "")).strip().lower()
    if not control and model.startswith("control_"):
        control, model = model.removeprefix("control_"), "control"
    aliases = {
        "salpha_s": "sas",
        "stable_control": "sas",
        "normal": "gaussian",
    }
    control = aliases.get(control, control)
    if control and task_kind == "main":
        task_kind = "control"
    if times is not None:
        task_kind = "dynamic_control" if task_kind == "main" else task_kind
    default_purpose = "control" if task_kind in {"control", "tail_control"} else "primary"
    purpose = str(_first(metadata, ("purpose",), default_purpose)).strip().lower()
    if purpose not in {"primary", "refinement", "control"}:
        raise Exp1ArtifactError(f"unknown task purpose: {purpose!r}")
    seed = _int_metadata(metadata, ("seed",), -1)
    steps = _int_metadata(metadata, ("steps", "n_steps"), 0)
    if seed < 0 or steps < 0:
        raise Exp1ArtifactError("seed and steps must be non-negative")
    alpha = _float_metadata(metadata, ("alpha", "tail_index"))
    if (model == "stable" or control in {"pareto", "sas"}) and (
        math.isnan(alpha) or not 1.0 < alpha < 2.0
    ):
        raise Exp1ArtifactError("stable/Pareto/SAS task requires alpha in (1,2)")
    return TaskData(
        path=source,
        task_kind=task_kind,
        purpose=purpose,
        model=model,
        control_distribution=control,
        target=str(_first(metadata, ("target", "target_distribution"), "")).strip().lower(),
        target_nu=_float_metadata(metadata, ("target_nu", "target_df", "degrees_of_freedom", "nu")),
        alpha=alpha,
        eta=_float_metadata(metadata, ("eta", "noise_eta")),
        beta=_float_metadata(metadata, ("beta", "beta0"), 1.0),
        horizon=_float_metadata(metadata, ("T", "horizon")),
        epsilon=_float_metadata(metadata, ("epsilon", "eps")),
        steps=steps,
        seed=seed,
        terminal_sample=terminal,
        control_forward_times=times,
        control_numerical=numerical,
        control_exact_a=exact_a,
        control_exact_b=exact_b,
        control_exact_c=exact_c,
        score_hires_terminal=score_hires,
        score_sensitivity_forward_times=sensitivity_times,
        score_sensitivity_mean_abs=sensitivity_mean_abs,
        metadata=metadata,
    )


def load_tasks(paths: Iterable[str | Path]) -> list[TaskData]:
    tasks = [load_task_npz(path) for path in paths]
    if not tasks:
        raise Exp1ArtifactError("no Experiment 1 task was supplied")
    keys = [task.key for task in tasks]
    if len(set(keys)) != len(keys):
        raise Exp1ArtifactError("duplicate Experiment 1 task identity")
    return sorted(tasks, key=lambda task: tuple(str(item) for item in task.key))


def semantic_permutation(task: TaskData, label: str) -> np.ndarray:
    """Stable task-semantic permutation; independent of paths and wall time."""

    token = "|".join(str(item) for item in ("exp1", *task.key, label)).encode()
    seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little", signed=False)
    return np.random.default_rng(seed).permutation(task.sample_count)


def sample_plans(task: TaskData, settings: AnalysisSettings) -> list[dict[str, Any]]:
    """Full sample, nested prefixes and disjoint canonical blocks."""

    permutation = semantic_permutation(task, "terminal")
    plans: list[dict[str, Any]] = [
        {
            "sample_kind": "full",
            "sample_size": task.sample_count,
            "replicate": -1,
            "indices": np.arange(task.sample_count),
        }
    ]
    seen: set[int] = set()
    for fraction in settings.subsample_fractions:
        size = math.floor(task.sample_count * fraction)
        if size >= 8 and size < task.sample_count and size not in seen:
            seen.add(size)
            plans.append(
                {
                    "sample_kind": "nested",
                    "sample_size": size,
                    "replicate": -1,
                    "indices": permutation[:size],
                }
            )
    block_size = task.sample_count // settings.block_count
    if block_size >= 8:
        for replicate in range(settings.block_count):
            start, stop = replicate * block_size, (replicate + 1) * block_size
            plans.append(
                {
                    "sample_kind": "block",
                    "sample_size": block_size,
                    "replicate": replicate,
                    "indices": permutation[start:stop],
                }
            )
    return plans


def _identity(task: TaskData, plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": str(task.path),
        "task_kind": task.task_kind,
        "purpose": task.purpose,
        "model": task.model,
        "control_distribution": task.control_distribution,
        "target": task.target,
        "target_nu": task.target_nu,
        "alpha_theory": task.alpha,
        "eta": task.eta,
        "horizon": task.horizon,
        "epsilon": task.epsilon,
        "steps": task.steps,
        "seed": task.seed,
        "sample_kind": plan["sample_kind"],
        "sample_size": plan["sample_size"],
        "replicate": plan["replicate"],
    }


def _metric_row(
    identity: dict[str, Any],
    *,
    metric: str,
    value: float,
    fraction: float = math.nan,
    k: int = 0,
    threshold: float = math.nan,
    beta: float = math.nan,
    auxiliary: float = math.nan,
    expected: float = math.nan,
    valid: bool = True,
    reason: str = "",
    controlled: bool = False,
) -> dict[str, Any]:
    return identity | {
        "metric": metric,
        "fraction": fraction,
        "k": k,
        "threshold": threshold,
        "beta": beta,
        "value": value,
        "auxiliary": auxiliary,
        "expected": expected,
        "valid": valid,
        "reason": reason,
        "controlled": controlled,
    }


def _reference_scale(task: TaskData) -> float:
    supplied = _first(
        task.metadata,
        ("discrete_rho", "reference_scale", "tail_scale", "rho_discrete"),
        math.nan,
    )
    try:
        scale = float(supplied)
    except (TypeError, ValueError):
        scale = math.nan
    if math.isfinite(scale) and scale > 0.0:
        return scale
    if task.control_distribution in {"pareto", "sas"}:
        raw = _first(task.metadata, ("scale", "pareto_scale", "xmin"), 1.0)
        scale = float(raw)
        return scale if scale > 0.0 and math.isfinite(scale) else math.nan
    is_light = task.target in {"", "light", "empirical", "gaussian_atoms", "discrete"}
    if not is_light or task.steps <= 0:
        return math.nan
    try:
        if task.model == "stable":
            return discrete_light_stable_scale(
                alpha=task.alpha,
                eta=task.eta,
                beta=task.beta,
                horizon=task.horizon,
                epsilon=task.epsilon,
                steps=task.steps,
            )
        if task.model == "vp":
            return discrete_light_vp_scale(
                beta=task.beta,
                horizon=task.horizon,
                epsilon=task.epsilon,
                steps=task.steps,
            )
    except TailMetricError:
        return math.nan
    return math.nan


def _tail_constant_reference(task: TaskData, scale: float) -> float:
    supplied = _first(
        task.metadata,
        ("tail_constant_reference", "discrete_tail_constant", "c_discrete"),
        math.nan,
    )
    try:
        expected = float(supplied)
    except (TypeError, ValueError):
        expected = math.nan
    if math.isfinite(expected) and expected > 0.0:
        return expected
    if task.control_distribution == "pareto" and math.isfinite(scale):
        return scale**task.alpha
    if (task.model == "stable" or task.control_distribution == "sas") and math.isfinite(scale):
        return sas_abs_tail_constant(task.alpha) * scale**task.alpha
    return math.nan


def _mean_excess_thresholds(task: TaskData, scale: float) -> np.ndarray | None:
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    if task.control_distribution == "pareto":
        fractions = np.asarray(MEAN_EXCESS_FRACTIONS, dtype=np.float64)
        return scale * fractions ** (-1.0 / task.alpha)
    try:
        if task.model == "stable" or task.control_distribution == "sas":
            return stable_reference_thresholds(MEAN_EXCESS_FRACTIONS, alpha=task.alpha, scale=scale)
        if task.model == "vp" or task.control_distribution == "gaussian":
            return gaussian_reference_thresholds(MEAN_EXCESS_FRACTIONS, scale=scale)
    except TailMetricError:
        return None
    return None


def _analyze_tail_plan(
    task: TaskData,
    sample: np.ndarray,
    plan: dict[str, Any],
    settings: AnalysisSettings,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    identity = _identity(task, plan)
    ordered = sorted_radial(sample)
    metrics: list[dict[str, Any]] = []
    mean_rows: list[dict[str, Any]] = []
    slope_rows: list[dict[str, Any]] = []

    expected_xi = 1.0 / task.alpha if task.is_stable_tail else math.nan
    for row in hill_grid(ordered, TAIL_FRACTIONS):
        common = {
            "fraction": row["fraction"],
            "k": row["k"],
            "threshold": row["threshold"],
            "valid": row["valid"],
            "reason": row["reason"],
        }
        metrics.append(
            _metric_row(
                identity,
                metric="hill_xi",
                value=row["xi"],
                expected=expected_xi,
                **common,
            )
        )
        metrics.append(
            _metric_row(
                identity,
                metric="hill_alpha",
                value=row["alpha"],
                expected=task.alpha if task.is_stable_tail else math.nan,
                **common,
            )
        )
    for row in pickands_grid(ordered, TAIL_FRACTIONS):
        metrics.append(
            _metric_row(
                identity,
                metric="pickands_xi",
                value=row["xi"],
                fraction=row["fraction"],
                k=row["k"],
                threshold=row["threshold"],
                expected=expected_xi,
                valid=row["valid"],
                reason=row["reason"],
            )
        )
    for fraction in GPD_FRACTIONS:
        k = tail_count(ordered.size, fraction)
        try:
            if k < settings.minimum_gpd_k:
                raise TailMetricError(
                    f"GPD k={k} is below the {settings.tier} gate {settings.minimum_gpd_k}"
                )
            xi, scale, threshold = gpd_at_k(ordered, k)
            metrics.append(
                _metric_row(
                    identity,
                    metric="gpd_xi",
                    value=xi,
                    auxiliary=scale,
                    fraction=fraction,
                    k=k,
                    threshold=threshold,
                    expected=expected_xi,
                )
            )
        except TailMetricError as exc:
            metrics.append(
                _metric_row(
                    identity,
                    metric="gpd_xi",
                    value=math.nan,
                    fraction=fraction,
                    k=k,
                    expected=expected_xi,
                    valid=False,
                    reason=str(exc),
                )
            )

    reference_scale = _reference_scale(task)
    constant_reference = _tail_constant_reference(task, reference_scale)
    if task.is_stable_tail:
        for fraction in TAIL_FRACTIONS:
            k = tail_count(ordered.size, fraction)
            try:
                constant, log_constant = tail_constant_at_k(ordered, k, task.alpha)
                metrics.append(
                    _metric_row(
                        identity,
                        metric="tail_constant",
                        value=constant,
                        auxiliary=log_constant,
                        fraction=fraction,
                        k=k,
                        threshold=float(ordered[k]),
                        expected=constant_reference,
                    )
                )
            except TailMetricError as exc:
                metrics.append(
                    _metric_row(
                        identity,
                        metric="tail_constant",
                        value=math.nan,
                        fraction=fraction,
                        k=k,
                        expected=constant_reference,
                        valid=False,
                        reason=str(exc),
                    )
                )

    thresholds = _mean_excess_thresholds(task, reference_scale)
    if thresholds is not None:
        expected_excess = 1.0 / (task.alpha - 1.0) if task.is_stable_tail else 0.0
        for fraction, row in zip(
            MEAN_EXCESS_FRACTIONS,
            mean_excess_sufficient(sample, thresholds),
            strict=True,
        ):
            mean_rows.append(
                identity
                | {
                    "threshold_fraction": fraction,
                    "threshold": row["threshold"],
                    "exceedance_sum": row["exceedance_sum"],
                    "exceedance_count": row["exceedance_count"],
                    "ratio": row["ratio"],
                    "expected": expected_excess,
                    "valid": row["valid"],
                    "reason": row["reason"],
                }
            )

    if math.isfinite(task.target_nu) and task.target_nu > 0.0:
        masses = np.asarray(settings.m_masses, dtype=np.float64)
        probabilities = 1.0 - masses
        target_quantiles = student_abs_quantiles(
            probabilities,
            task.target_nu,
            location=float(_first(task.metadata, ("target_location", "location"), 0.0)),
            scale=float(_first(task.metadata, ("target_scale",), 1.0)),
        )
        generated_quantiles = radial_quantiles(sample, probabilities)
        ratios = generated_quantiles / target_quantiles
        expected_slope = (
            1.0 / task.target_nu - 1.0 / task.alpha
            if task.model == "stable" and math.isfinite(task.alpha)
            else math.nan
        )
        for mass, probability, ratio in zip(masses, probabilities, ratios, strict=True):
            metrics.append(
                _metric_row(
                    identity,
                    metric="mass_ratio",
                    value=float(ratio),
                    fraction=float(mass),
                    beta=float(probability),
                    expected=math.nan,
                )
            )
        for name, (lower, upper) in settings.m_windows.items():
            try:
                fit = power_law_fit(masses, ratios, lower=lower, upper=upper)
                slope_rows.append(
                    identity
                    | {
                        "window": name,
                        "mass_lower": lower,
                        "mass_upper": upper,
                        "n_points": fit.n_points,
                        "slope": fit.slope,
                        "intercept": fit.intercept,
                        "r_squared": fit.r_squared,
                        "expected_slope": expected_slope,
                        "valid": True,
                        "reason": "",
                    }
                )
            except TailMetricError as exc:
                slope_rows.append(
                    identity
                    | {
                        "window": name,
                        "mass_lower": lower,
                        "mass_upper": upper,
                        "n_points": 0,
                        "slope": math.nan,
                        "intercept": math.nan,
                        "r_squared": math.nan,
                        "expected_slope": expected_slope,
                        "valid": False,
                        "reason": str(exc),
                    }
                )
    return metrics, mean_rows, slope_rows


def _sort_w1_sample(sample: np.ndarray) -> np.ndarray:
    """Return one canonical float64 sort for all checkpoint W1 pairings."""

    values = np.asarray(sample, dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise Exp1ArtifactError("marginal-control W1 samples must be finite and non-empty")
    return np.sort(values)


def _w1_from_sorted(first: np.ndarray, second: np.ndarray) -> float:
    """Exact empirical W1 for equal-size samples already sorted increasingly."""

    if first.shape != second.shape:
        raise Exp1ArtifactError("sorted marginal-control W1 samples have unequal sizes")
    return float(np.mean(np.abs(first - second)))


def _ecf_simultaneous_radius(
    sample_count: int,
    frequency_count: int,
    *,
    familywise_alpha: float,
    familywise_test_count: int,
) -> float:
    """Hoeffding/Bonferroni radius for the maximum complex ECF error."""

    if sample_count <= 0 or frequency_count <= 0 or familywise_test_count <= 0:
        raise Exp1ArtifactError("ECF simultaneous-band counts must be positive")
    local_alpha = familywise_alpha / familywise_test_count
    return 2.0 * math.sqrt(math.log(4.0 * frequency_count / local_alpha) / sample_count)


def _tail_pair_diagnostics(
    first: np.ndarray, second: np.ndarray, alpha: float
) -> tuple[float, float]:
    first_ordered, second_ordered = sorted_radial(first), sorted_radial(second)
    first_hill = hill_grid(first_ordered, (PRIMARY_FRACTION,))[0]
    second_hill = hill_grid(second_ordered, (PRIMARY_FRACTION,))[0]
    if not first_hill["valid"] or not second_hill["valid"]:
        return math.nan, math.nan
    hill_difference = abs(float(first_hill["alpha"]) - float(second_hill["alpha"]))
    k = tail_count(first_ordered.size, PRIMARY_FRACTION)
    first_constant = tail_constant_at_k(first_ordered, k, alpha)[0]
    second_constant = tail_constant_at_k(second_ordered, k, alpha)[0]
    relative_constant = abs(first_constant - second_constant) / max(
        abs(second_constant), np.finfo(np.float64).tiny
    )
    return hill_difference, relative_constant


def _analyze_dynamic_control(
    task: TaskData,
    settings: AnalysisSettings,
    *,
    familywise_test_count: int = 1,
) -> list[dict[str, Any]]:
    if task.control_forward_times is None:
        return []
    assert task.control_numerical is not None
    assert task.control_exact_a is not None
    assert task.control_exact_b is not None
    assert task.control_exact_c is not None
    assert task.score_hires_terminal is not None
    assert task.score_sensitivity_forward_times is not None
    assert task.score_sensitivity_mean_abs is not None
    identity = _identity(
        task,
        {
            "sample_kind": "full",
            "sample_size": task.control_numerical.shape[1],
            "replicate": -1,
        },
    )
    frequencies = np.linspace(
        -settings.ecf_frequency_max,
        settings.ecf_frequency_max,
        settings.ecf_frequency_count,
    )
    weights = np.exp(-0.5 * frequencies**2)
    rows: list[dict[str, Any]] = []
    for index, forward_time in enumerate(task.control_forward_times):
        numerical = task.control_numerical[index]
        exact_a = task.control_exact_a[index]
        exact_b = task.control_exact_b[index]
        exact_c = task.control_exact_c[index]
        numerical_sorted, exact_a_sorted, exact_b_sorted, exact_c_sorted = (
            _sort_w1_sample(sample) for sample in (numerical, exact_a, exact_b, exact_c)
        )
        w1 = float(
            np.median(
                [
                    _w1_from_sorted(numerical_sorted, exact)
                    for exact in (exact_a_sorted, exact_b_sorted, exact_c_sorted)
                ]
            )
        )
        w1_floor = max(
            _w1_from_sorted(first, second)
            for first, second in (
                (exact_a_sorted, exact_b_sorted),
                (exact_a_sorted, exact_c_sorted),
                (exact_b_sorted, exact_c_sorted),
            )
        )
        numerical_cf = empirical_cf_1d(numerical, frequencies, chunk_size=settings.ecf_chunk_size)
        exact_a_cf = empirical_cf_1d(exact_a, frequencies, chunk_size=settings.ecf_chunk_size)
        exact_b_cf = empirical_cf_1d(exact_b, frequencies, chunk_size=settings.ecf_chunk_size)
        exact_c_cf = empirical_cf_1d(exact_c, frequencies, chunk_size=settings.ecf_chunk_size)
        ecf = float(
            np.median(
                [
                    weighted_cf_rmse(numerical_cf, exact_cf, weights)
                    for exact_cf in (exact_a_cf, exact_b_cf, exact_c_cf)
                ]
            )
        )
        ecf_floor = max(
            weighted_cf_rmse(first, second, weights)
            for first, second in (
                (exact_a_cf, exact_b_cf),
                (exact_a_cf, exact_c_cf),
                (exact_b_cf, exact_c_cf),
            )
        )
        rows.extend(
            (
                _metric_row(
                    identity,
                    metric="control_w1",
                    value=w1,
                    threshold=float(forward_time),
                    auxiliary=w1_floor,
                    expected=0.0,
                ),
                _metric_row(
                    identity,
                    metric="control_ecf_rmse",
                    value=ecf,
                    threshold=float(forward_time),
                    auxiliary=ecf_floor,
                    expected=0.0,
                ),
            )
        )
        analytic_cf = np.asarray(
            forward_marginal_cf(
                frequencies,
                float(forward_time),
                task.model,
                task.target_nu,
                alpha=task.alpha,
                beta=task.beta,
            ),
            dtype=np.complex128,
        )
        radius = _ecf_simultaneous_radius(
            numerical.size,
            frequencies.size,
            familywise_alpha=settings.pt_ecf_familywise_alpha,
            familywise_test_count=familywise_test_count,
        )
        for arm, empirical in (
            ("numerical", numerical_cf),
            ("exact_a", exact_a_cf),
            ("exact_b", exact_b_cf),
            ("exact_c", exact_c_cf),
        ):
            value = float(np.max(np.abs(empirical - analytic_cf)))
            controlled = value <= radius
            rows.append(
                _metric_row(
                    identity,
                    metric=f"pt_ecf_sup_{arm}",
                    value=value,
                    threshold=float(forward_time),
                    auxiliary=radius,
                    expected=0.0,
                    controlled=controlled,
                    reason=(
                        ""
                        if controlled
                        else "sample ECF exits the prespecified simultaneous analytic band"
                    ),
                )
            )

    terminal_sorted = _sort_w1_sample(task.terminal_sample)
    hires_sorted = _sort_w1_sample(task.score_hires_terminal)
    score_w1 = _w1_from_sorted(terminal_sorted, hires_sorted)
    score_ecf = weighted_cf_rmse(
        empirical_cf_1d(task.terminal_sample, frequencies, chunk_size=settings.ecf_chunk_size),
        empirical_cf_1d(
            task.score_hires_terminal, frequencies, chunk_size=settings.ecf_chunk_size
        ),
        weights,
    )
    exact_terminal = (
        task.control_exact_a[0],
        task.control_exact_b[0],
        task.control_exact_c[0],
    )
    exact_sorted = tuple(_sort_w1_sample(sample) for sample in exact_terminal)
    score_w1_floor = max(
        _w1_from_sorted(first, second)
        for first, second in (
            (exact_sorted[0], exact_sorted[1]),
            (exact_sorted[0], exact_sorted[2]),
            (exact_sorted[1], exact_sorted[2]),
        )
    )
    exact_cf = tuple(
        empirical_cf_1d(sample, frequencies, chunk_size=settings.ecf_chunk_size)
        for sample in exact_terminal
    )
    score_ecf_floor = max(
        weighted_cf_rmse(first, second, weights)
        for first, second in (
            (exact_cf[0], exact_cf[1]),
            (exact_cf[0], exact_cf[2]),
            (exact_cf[1], exact_cf[2]),
        )
    )
    hill_difference, constant_relative = _tail_pair_diagnostics(
        task.terminal_sample, task.score_hires_terminal, task.alpha
    )
    for metric, value, limit in (
        (
            "score_table_w1",
            score_w1,
            settings.score_sensitivity_floor_fraction * score_w1_floor,
        ),
        (
            "score_table_ecf_rmse",
            score_ecf,
            settings.score_sensitivity_floor_fraction * score_ecf_floor,
        ),
        (
            "score_table_hill_alpha_absdiff",
            hill_difference,
            settings.score_sensitivity_hill_tolerance,
        ),
        (
            "score_table_tail_constant_reldiff",
            constant_relative,
            settings.score_sensitivity_constant_relative_tolerance,
        ),
    ):
        controlled = math.isfinite(value) and value <= limit
        rows.append(
            _metric_row(
                identity,
                metric=metric,
                value=float(value),
                auxiliary=float(limit),
                expected=0.0,
                controlled=controlled,
                reason="" if controlled else "main/hires score-table sensitivity exceeds its gate",
            )
        )
    for forward_time, value in zip(
        task.score_sensitivity_forward_times,
        task.score_sensitivity_mean_abs,
        strict=True,
    ):
        rows.append(
            _metric_row(
                identity,
                metric="score_table_trajectory_mean_abs",
                value=float(value),
                threshold=float(forward_time),
                expected=0.0,
            )
        )
    return rows


def analyze_task(
    task: TaskData,
    settings: AnalysisSettings,
    *,
    familywise_test_count: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    settings.validate()
    metric_rows: list[dict[str, Any]] = []
    mean_rows: list[dict[str, Any]] = []
    slope_rows: list[dict[str, Any]] = []
    for plan in sample_plans(task, settings):
        sample = task.terminal_sample[np.asarray(plan["indices"], dtype=np.int64)]
        metrics, mean, slopes = _analyze_tail_plan(task, sample, plan, settings)
        metric_rows.extend(metrics)
        mean_rows.extend(mean)
        slope_rows.extend(slopes)
    metric_rows.extend(
        _analyze_dynamic_control(
            task, settings, familywise_test_count=familywise_test_count
        )
    )
    return metric_rows, mean_rows, slope_rows


def analyze_tasks(
    tasks: Sequence[TaskData], settings: AnalysisSettings
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    settings.validate()
    metrics: list[dict[str, Any]] = []
    mean_rows: list[dict[str, Any]] = []
    slopes: list[dict[str, Any]] = []
    familywise_test_count = 4 * sum(
        int(task.control_forward_times.size)
        for task in tasks
        if task.purpose == "primary" and task.control_forward_times is not None
    )
    familywise_test_count = max(1, familywise_test_count)
    for task in tasks:
        task_metrics, task_mean, task_slopes = analyze_task(
            task, settings, familywise_test_count=familywise_test_count
        )
        metrics.extend(task_metrics)
        mean_rows.extend(task_mean)
        slopes.extend(task_slopes)
    sort_fields = (
        "task_kind",
        "purpose",
        "model",
        "control_distribution",
        "target",
        "target_nu",
        "alpha_theory",
        "seed",
        "sample_kind",
        "sample_size",
        "replicate",
    )
    metrics.sort(
        key=lambda row: (
            *(str(row[name]) for name in sort_fields),
            row["metric"],
            str(row["fraction"]),
            str(row["threshold"]),
        )
    )
    mean_rows.sort(
        key=lambda row: (
            *(str(row[name]) for name in sort_fields),
            str(row["threshold_fraction"]),
        )
    )
    slopes.sort(key=lambda row: (*(str(row[name]) for name in sort_fields), row["window"]))
    return metrics, mean_rows, slopes


SUMMARY_GROUP_FIELDS = (
    "task_kind",
    "model",
    "control_distribution",
    "target",
    "target_nu",
    "alpha_theory",
    "eta",
    "horizon",
    "epsilon",
    "steps",
    "sample_kind",
    "sample_size",
    "metric",
    "fraction",
    "k",
    "threshold",
    "beta",
    "expected",
    "purpose",
)


def _row_field(row: dict[str, Any], name: str) -> Any:
    if name == "purpose":
        return row.get(name, "primary")
    return row[name]


def _group_key(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    values: list[Any] = []
    metric = str(row.get("metric", ""))
    for name in fields:
        value = _row_field(row, name)
        if name == "threshold" and metric and not metric.startswith(
            ("control_", "pt_ecf_", "score_table_trajectory_")
        ):
            value = "__sample_dependent_threshold__"
        elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            value = f"__nonfinite_{name}__"
        values.append(value)
    return tuple(values)


def _bootstrap_seed(
    values: np.ndarray,
    *,
    replicates: int,
    salt: str,
    probability: float,
) -> tuple[float, float]:
    if values.size < 2 or replicates <= 0:
        return math.nan, math.nan
    seed = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    medians = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1024):
        stop = min(start + 1024, replicates)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        medians[start:stop] = np.median(values[indices], axis=1)
    tail = (1.0 - probability) / 2.0
    return float(np.quantile(medians, tail)), float(np.quantile(medians, 1.0 - tail))


def summarize_tail_metrics(
    rows: Sequence[dict[str, Any]], settings: AnalysisSettings
) -> list[dict[str, Any]]:
    """Reduce blocks within each seed, then summarize independent seeds."""

    fields = SUMMARY_GROUP_FIELDS
    per_seed: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if bool(row["valid"]) and math.isfinite(float(row["value"])):
            per_seed[(*_group_key(row, fields), row["seed"])].append(row)
    seed_rows: list[dict[str, Any]] = []
    for key, members in per_seed.items():
        value = float(np.median([float(member["value"]) for member in members]))
        representative = {name: _row_field(members[0], name) for name in fields}
        if not str(members[0]["metric"]).startswith("control_"):
            thresholds = [
                float(member["threshold"])
                for member in members
                if math.isfinite(float(member["threshold"]))
            ]
            representative["threshold"] = float(np.median(thresholds)) if thresholds else math.nan
        seed_rows.append(
            representative
            | {
                "seed": key[-1],
                "value": value,
                "n_within_seed": len(members),
                "controlled": bool(all(bool(member["controlled"]) for member in members)),
            }
        )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[_group_key(row, fields)].append(row)
    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        members = grouped[key]
        values = np.asarray([member["value"] for member in members], dtype=np.float64)
        salt = "|".join(str(value) for value in key)
        boot90 = _bootstrap_seed(
            values,
            replicates=settings.resolved_bootstrap_replicates,
            salt=f"{salt}|90",
            probability=0.90,
        )
        boot95 = _bootstrap_seed(
            values,
            replicates=settings.resolved_bootstrap_replicates,
            salt=f"{salt}|95",
            probability=0.95,
        )
        representative = {name: _row_field(members[0], name) for name in fields}
        if not str(members[0]["metric"]).startswith("control_"):
            thresholds = [
                float(member["threshold"])
                for member in members
                if math.isfinite(float(member["threshold"]))
            ]
            representative["threshold"] = float(np.median(thresholds)) if thresholds else math.nan
        summaries.append(
            representative
            | {
                "n_seeds": len(members),
                "q16": float(np.quantile(values, 0.16)),
                "median": float(np.median(values)),
                "q84": float(np.quantile(values, 0.84)),
                "boot90_low": boot90[0],
                "boot90_high": boot90[1],
                "boot95_low": boot95[0],
                "boot95_high": boot95[1],
                "controlled_fraction": float(np.mean([member["controlled"] for member in members])),
            }
        )
    return summaries


def pool_mean_excess_rows(
    rows: Sequence[dict[str, Any]],
    settings: AnalysisSettings | None = None,
) -> list[dict[str, Any]]:
    """Pool full-sample sufficient statistics without treating blocks as seeds."""

    settings = settings or AnalysisSettings()
    settings.validate()
    fields = (
        "task_kind",
        "model",
        "control_distribution",
        "target",
        "target_nu",
        "alpha_theory",
        "eta",
        "horizon",
        "epsilon",
        "steps",
        "purpose",
        "sample_kind",
        "sample_size",
        "threshold_fraction",
        "threshold",
        "expected",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row, fields)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        members = grouped[key]
        # A block collection is a stability diagnostic.  Concatenating its
        # blocks would reproduce the full sample and would double-count the
        # same seed if it were mixed with full/nested rows.
        by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for member in members:
            by_seed[int(member["seed"])].append(member)
        seed_sums: list[float] = []
        seed_counts: list[int] = []
        qualifying = 0
        seed_ratios: list[float] = []
        for seed_members in by_seed.values():
            if seed_members[0]["sample_kind"] == "block":
                ratios = [
                    float(member["ratio"])
                    for member in seed_members
                    if int(member["exceedance_count"]) >= settings.minimum_exceedances_per_seed
                    and math.isfinite(float(member["ratio"]))
                ]
                if ratios:
                    seed_ratios.append(float(np.median(ratios)))
                seed_sums.append(
                    float(sum(float(member["exceedance_sum"]) for member in seed_members))
                )
                seed_counts.append(
                    int(sum(int(member["exceedance_count"]) for member in seed_members))
                )
            else:
                member = seed_members[0]
                seed_sums.append(float(member["exceedance_sum"]))
                seed_counts.append(int(member["exceedance_count"]))
                if int(member["exceedance_count"]) >= settings.minimum_exceedances_per_seed:
                    seed_ratios.append(float(member["ratio"]))
            if seed_counts[-1] >= settings.minimum_exceedances_per_seed:
                qualifying += 1
        total = int(sum(seed_counts))
        valid = (
            total >= settings.minimum_pooled_exceedances
            and qualifying >= settings.minimum_contributing_seeds
        )
        ratio = (
            pooled_mean_excess(
                np.asarray(seed_sums), np.asarray(seed_counts), float(members[0]["threshold"])
            )
            if valid
            else math.nan
        )
        output.append(
            {name: _row_field(members[0], name) for name in fields}
            | {
                "n_seeds": len(by_seed),
                "qualifying_seeds": qualifying,
                "pooled_exceedances": total,
                "pooled_sum": float(sum(seed_sums)),
                "pooled_ratio": ratio,
                "seed_q16": float(np.quantile(seed_ratios, 0.16)) if seed_ratios else math.nan,
                "seed_median": float(np.median(seed_ratios)) if seed_ratios else math.nan,
                "seed_q84": float(np.quantile(seed_ratios, 0.84)) if seed_ratios else math.nan,
                "valid": valid,
                "reason": (
                    ""
                    if valid
                    else (
                        f"requires >={settings.minimum_pooled_exceedances} pooled, "
                        f">={settings.minimum_contributing_seeds} seeds with "
                        f">={settings.minimum_exceedances_per_seed} exceedances"
                    )
                ),
            }
        )
    return output


def summarize_m_slopes(
    rows: Sequence[dict[str, Any]], settings: AnalysisSettings
) -> list[dict[str, Any]]:
    fields = (
        "task_kind",
        "model",
        "control_distribution",
        "target",
        "target_nu",
        "alpha_theory",
        "eta",
        "horizon",
        "epsilon",
        "steps",
        "purpose",
        "sample_kind",
        "sample_size",
        "window",
        "mass_lower",
        "mass_upper",
        "expected_slope",
    )
    per_seed: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if bool(row["valid"]) and math.isfinite(float(row["slope"])):
            per_seed[(*_group_key(row, fields), row["seed"])].append(row)
    reduced: list[dict[str, Any]] = []
    for key, members in per_seed.items():
        reduced.append(
            {name: _row_field(members[0], name) for name in fields}
            | {
                "seed": key[-1],
                "slope": float(np.median([member["slope"] for member in members])),
                "r_squared": float(np.nanmedian([member["r_squared"] for member in members])),
                "n_within_seed": len(members),
            }
        )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in reduced:
        grouped[_group_key(row, fields)].append(row)
    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        members = grouped[key]
        values = np.asarray([member["slope"] for member in members], dtype=np.float64)
        salt = "M|" + "|".join(str(value) for value in key)
        boot90 = _bootstrap_seed(
            values,
            replicates=settings.resolved_bootstrap_replicates,
            salt=f"{salt}|90",
            probability=0.90,
        )
        boot95 = _bootstrap_seed(
            values,
            replicates=settings.resolved_bootstrap_replicates,
            salt=f"{salt}|95",
            probability=0.95,
        )
        summaries.append(
            {name: _row_field(members[0], name) for name in fields}
            | {
                "n_seeds": len(members),
                "q16": float(np.quantile(values, 0.16)),
                "median": float(np.median(values)),
                "q84": float(np.quantile(values, 0.84)),
                "boot90_low": boot90[0],
                "boot90_high": boot90[1],
                "boot95_low": boot95[0],
                "boot95_high": boot95[1],
                "median_r_squared": float(np.nanmedian([row["r_squared"] for row in members])),
            }
        )
    return summaries


def _configuration_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["task_kind"],
        row["model"],
        row["control_distribution"],
        row["target"],
        row["target_nu"],
        row["alpha_theory"],
        row["eta"],
        row["horizon"],
        row["epsilon"],
        row["steps"],
        row.get("purpose", "primary"),
    )


def _select_summary(
    summaries: Sequence[dict[str, Any]],
    key: tuple[Any, ...],
    *,
    metric: str,
    fraction: float,
) -> dict[str, Any] | None:
    matches = [
        row
        for row in summaries
        if _configuration_key(row) == key
        and row["sample_kind"] == "full"
        and row["metric"] == metric
        and math.isclose(float(row["fraction"]), fraction, abs_tol=1.0e-14)
    ]
    return matches[0] if len(matches) == 1 else None


def _plateau_summary(
    metric_rows: Sequence[dict[str, Any]],
    key: tuple[Any, ...],
    settings: AnalysisSettings,
) -> dict[str, float | int]:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        if (
            _configuration_key(row) == key
            and row["sample_kind"] == "full"
            and row["metric"] == "hill_xi"
            and any(
                math.isclose(float(row["fraction"]), fraction, abs_tol=1.0e-14)
                for fraction in PLATEAU_FRACTIONS
            )
            and row["valid"]
        ):
            by_seed[int(row["seed"])].append(row)
    slopes: list[float] = []
    for members in by_seed.values():
        if len(members) != len(PLATEAU_FRACTIONS):
            continue
        members.sort(key=lambda row: float(row["fraction"]))
        x = np.log2(
            np.asarray([row["fraction"] for row in members], dtype=np.float64) / PRIMARY_FRACTION
        )
        y = np.asarray([row["value"] for row in members], dtype=np.float64)
        slopes.append(float(np.polyfit(x, y, 1)[0]))
    values = np.asarray(slopes, dtype=np.float64)
    if values.size == 0:
        return {
            "n_seeds": 0,
            "median": math.nan,
            "boot90_low": math.nan,
            "boot90_high": math.nan,
            "boot95_low": math.nan,
            "boot95_high": math.nan,
        }
    salt = "plateau|" + "|".join(str(item) for item in key)
    boot90 = _bootstrap_seed(
        values,
        replicates=settings.resolved_bootstrap_replicates,
        salt=f"{salt}|90",
        probability=0.90,
    )
    boot95 = _bootstrap_seed(
        values,
        replicates=settings.resolved_bootstrap_replicates,
        salt=f"{salt}|95",
        probability=0.95,
    )
    return {
        "n_seeds": int(values.size),
        "median": float(np.median(values)),
        "boot90_low": boot90[0],
        "boot90_high": boot90[1],
        "boot95_low": boot95[0],
        "boot95_high": boot95[1],
    }


def _inside(row: dict[str, Any] | None, expected: float, margin: float, interval: str) -> bool:
    if row is None:
        return False
    low, high = float(row[f"{interval}_low"]), float(row[f"{interval}_high"])
    return (
        math.isfinite(low)
        and math.isfinite(high)
        and low >= expected - margin
        and high <= expected + margin
    )


def _outside(row: dict[str, Any] | None, expected: float, margin: float, interval: str) -> int:
    if row is None:
        return 0
    low, high = float(row[f"{interval}_low"]), float(row[f"{interval}_high"])
    if not math.isfinite(low) or not math.isfinite(high):
        return 0
    if high < expected - margin:
        return -1
    if low > expected + margin:
        return 1
    return 0


def _control_gate(
    tasks: Sequence[TaskData], summaries: Sequence[dict[str, Any]], settings: AnalysisSettings
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    required_pt_metrics = (
        "pt_ecf_sup_numerical",
        "pt_ecf_sup_exact_a",
        "pt_ecf_sup_exact_b",
        "pt_ecf_sup_exact_c",
    )
    required_sensitivity_metrics = (
        "score_table_w1",
        "score_table_ecf_rmse",
        "score_table_hill_alpha_absdiff",
        "score_table_tail_constant_reldiff",
    )
    primary_cells: dict[tuple[str, float, int], list[TaskData]] = defaultdict(list)
    for task in tasks:
        if task.purpose == "primary" and task.task_kind in {"dynamic", "dynamic_control"}:
            primary_cells[(task.model, task.target_nu, task.steps)].append(task)
    if not primary_cells:
        failures.append("missing primary dynamic cells")
    for (model, nu, steps), members in primary_cells.items():
        expected_seeds = len({task.seed for task in members})
        checkpoint_sets = {
            tuple(float(value) for value in task.control_forward_times)
            for task in members
            if task.control_forward_times is not None
        }
        if len(checkpoint_sets) != 1:
            failures.append(f"{model} nu={nu:g} N={steps}: inconsistent pT checkpoint cells")
            continue
        checkpoints = next(iter(checkpoint_sets))
        for metric in required_pt_metrics:
            for checkpoint in checkpoints:
                matches = [
                    row
                    for row in summaries
                    if row.get("purpose", "primary") == "primary"
                    and row.get("model") == model
                    and math.isclose(float(row.get("target_nu", math.nan)), nu, abs_tol=1.0e-14)
                    and int(row.get("steps", -1)) == steps
                    and row.get("sample_kind") == "full"
                    and row.get("metric") == metric
                    and math.isclose(
                        float(row.get("threshold", math.nan)), checkpoint, abs_tol=1.0e-14
                    )
                ]
                if len(matches) != 1 or int(matches[0].get("n_seeds", -1)) != expected_seeds:
                    failures.append(
                        f"{model} nu={nu:g} N={steps} t={checkpoint:g}: missing {metric} cell"
                    )
                elif float(matches[0]["controlled_fraction"]) < 1.0:
                    failures.append(
                        f"{model} nu={nu:g} N={steps} t={checkpoint:g}: {metric} exits ECF band"
                    )
        for metric in required_sensitivity_metrics:
            matches = [
                row
                for row in summaries
                if row.get("purpose", "primary") == "primary"
                and row.get("model") == model
                and math.isclose(float(row.get("target_nu", math.nan)), nu, abs_tol=1.0e-14)
                and int(row.get("steps", -1)) == steps
                and row.get("sample_kind") == "full"
                and row.get("metric") == metric
            ]
            if len(matches) != 1 or int(matches[0].get("n_seeds", -1)) != expected_seeds:
                failures.append(
                    f"{model} nu={nu:g} N={steps}: missing {metric} sensitivity cell"
                )
            elif float(matches[0]["controlled_fraction"]) < 1.0:
                failures.append(f"{model} nu={nu:g} N={steps}: {metric} sensitivity fails")
    heavy_controls = [
        task
        for task in tasks
        if task.purpose == "control" and task.control_distribution in {"pareto", "sas"}
    ]
    heavy_specs = {(task.control_distribution, task.alpha) for task in heavy_controls}
    if heavy_specs != {("pareto", 1.5), ("sas", 1.5)}:
        failures.append("missing Pareto/SAS tail control")
    for distribution, alpha in sorted(heavy_specs):
        expected_seeds = len(
            {task.seed for task in heavy_controls if task.control_distribution == distribution}
        )
        matches = [
            row
            for row in summaries
            if row.get("control_distribution") == distribution
            and row.get("sample_kind") == "full"
            and row.get("metric") == "hill_alpha"
            and math.isclose(float(row.get("fraction", math.nan)), PRIMARY_FRACTION, abs_tol=1e-14)
        ]
        if len(matches) != 1 or int(matches[0].get("n_seeds", -1)) != expected_seeds:
            failures.append(f"missing unique {distribution} Hill control cell")
        elif settings.tier in {"pilot", "final"} and not _inside(
            matches[0], alpha, ALPHA_MARGIN, "boot90"
        ):
            failures.append(f"{distribution} alpha={alpha:g} fails Hill equivalence")
    gaussian_tasks = [
        task
        for task in tasks
        if task.purpose == "control" and task.control_distribution == "gaussian"
    ]
    gaussian_rows = [
        row
        for row in summaries
        if row.get("control_distribution", "") == "gaussian"
        and row.get("sample_kind", "") == "full"
        and row["metric"] == "hill_alpha"
        and math.isclose(float(row["fraction"]), PRIMARY_FRACTION, abs_tol=1.0e-14)
    ]
    gaussian_seed_count = len({task.seed for task in gaussian_tasks})
    if (
        not gaussian_tasks
        or len(gaussian_rows) != 1
        or int(gaussian_rows[0].get("n_seeds", -1)) != gaussian_seed_count
    ):
        failures.append("missing unique Gaussian negative-control Hill summary")
    elif settings.tier in {"pilot", "final"}:
        lower = float(gaussian_rows[0]["boot95_low"])
        stable_alpha = float(gaussian_tasks[0].alpha)
        if not math.isfinite(lower) or lower <= (
            stable_alpha + settings.gaussian_hill_separation_margin
        ):
            failures.append("Gaussian negative control is not separated from a stable tail")
    return not failures, failures


def _m_slope_for_sample(
    task: TaskData, sample: np.ndarray, settings: AnalysisSettings
) -> float:
    masses = np.asarray(settings.m_masses, dtype=np.float64)
    probabilities = 1.0 - masses
    target_quantiles = student_abs_quantiles(
        probabilities,
        task.target_nu,
        location=float(_first(task.metadata, ("target_location", "location"), 0.0)),
        scale=float(_first(task.metadata, ("target_scale",), 1.0)),
    )
    ratios = radial_quantiles(sample, probabilities) / target_quantiles
    lower, upper = settings.m_primary_window
    return power_law_fit(masses, ratios, lower=lower, upper=upper).slope


def _refinement_gate(
    tasks: Sequence[TaskData], settings: AnalysisSettings
) -> tuple[bool, list[str], list[dict[str, Any]]]:
    """Fail-closed paired highest-two-N gate using common primitive randomness."""

    frequencies = np.linspace(
        -settings.ecf_frequency_max,
        settings.ecf_frequency_max,
        settings.ecf_frequency_count,
    )
    weights = np.exp(-0.5 * frequencies**2)
    groups: dict[tuple[str, float, int], list[TaskData]] = defaultdict(list)
    expected_specs = {
        (task.model, task.target_nu)
        for task in tasks
        if task.task_kind in {"dynamic", "dynamic_control"}
    }
    for task in tasks:
        if task.task_kind in {"dynamic", "dynamic_control"}:
            groups[(task.model, task.target_nu, task.seed)].append(task)
    details: list[dict[str, Any]] = []
    failures: list[str] = []
    counts: dict[tuple[str, float], int] = defaultdict(int)
    for (model, nu, seed), members in groups.items():
        by_steps = {task.steps: task for task in members}
        if len(by_steps) < 2:
            continue
        low_steps, high_steps = sorted(by_steps)[-2:]
        low, high = by_steps[low_steps], by_steps[high_steps]
        counts[(model, nu)] += 1
        if low.terminal_sample.shape != high.terminal_sample.shape:
            failures.append(f"{model} nu={nu:g} seed={seed}: unequal refinement samples")
            continue
        assert high.control_exact_a is not None
        assert high.control_exact_b is not None
        assert high.control_exact_c is not None
        exact = (high.control_exact_a[0], high.control_exact_b[0], high.control_exact_c[0])
        exact_sorted = tuple(_sort_w1_sample(sample) for sample in exact)
        w1_floor = max(
            _w1_from_sorted(first, second)
            for first, second in (
                (exact_sorted[0], exact_sorted[1]),
                (exact_sorted[0], exact_sorted[2]),
                (exact_sorted[1], exact_sorted[2]),
            )
        )
        w1 = _w1_from_sorted(
            _sort_w1_sample(low.terminal_sample), _sort_w1_sample(high.terminal_sample)
        )
        exact_cf = tuple(
            empirical_cf_1d(sample, frequencies, chunk_size=settings.ecf_chunk_size)
            for sample in exact
        )
        ecf_floor = max(
            weighted_cf_rmse(first, second, weights)
            for first, second in (
                (exact_cf[0], exact_cf[1]),
                (exact_cf[0], exact_cf[2]),
                (exact_cf[1], exact_cf[2]),
            )
        )
        ecf = weighted_cf_rmse(
            empirical_cf_1d(
                low.terminal_sample, frequencies, chunk_size=settings.ecf_chunk_size
            ),
            empirical_cf_1d(
                high.terminal_sample, frequencies, chunk_size=settings.ecf_chunk_size
            ),
            weights,
        )
        checks = {
            "w1": (w1, settings.refinement_floor_fraction * w1_floor),
            "ecf_rmse": (ecf, settings.refinement_floor_fraction * ecf_floor),
        }
        if model == "stable":
            hill, constant = _tail_pair_diagnostics(
                low.terminal_sample, high.terminal_sample, high.alpha
            )
            try:
                slope = abs(
                    _m_slope_for_sample(low, low.terminal_sample, settings)
                    - _m_slope_for_sample(high, high.terminal_sample, settings)
                )
            except TailMetricError:
                slope = math.nan
            checks.update(
                {
                    "hill_alpha_absdiff": (hill, settings.refinement_hill_tolerance),
                    "tail_constant_reldiff": (
                        constant,
                        settings.refinement_constant_relative_tolerance,
                    ),
                    "m_slope_absdiff": (slope, settings.refinement_m_slope_tolerance),
                }
            )
        for metric, (value, limit) in checks.items():
            passed = math.isfinite(value) and value <= limit
            details.append(
                {
                    "model": model,
                    "target_nu": nu,
                    "seed": seed,
                    "low_steps": low_steps,
                    "high_steps": high_steps,
                    "metric": metric,
                    "value": float(value),
                    "limit": float(limit),
                    "passed": passed,
                }
            )
            if not passed:
                failures.append(
                    f"{model} nu={nu:g} seed={seed} N={low_steps}->{high_steps} {metric} fails"
                )
    required_seeds = {"smoke": 2, "pilot": 4, "final": 4}[settings.tier]
    for model, nu in expected_specs:
        if counts[(model, nu)] < required_seeds:
            failures.append(
                f"{model} nu={nu:g}: refinement gate has "
                f"{counts[(model, nu)]}/{required_seeds} seeds"
            )
    return not failures, failures, details


def make_decisions(
    tasks: Sequence[TaskData],
    metric_rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    mean_pooled: Sequence[dict[str, Any]],
    slope_summaries: Sequence[dict[str, Any]],
    settings: AnalysisSettings,
) -> list[dict[str, Any]]:
    """Apply only the frozen compatible/incompatible/nonconclusive rules."""

    control_pass, control_failures = _control_gate(tasks, summaries, settings)
    keys = sorted(
        {
            _configuration_key(row)
            for row in summaries
            if row["sample_kind"] == "full"
            and row.get("purpose", "primary") in {"primary", "control"}
        },
        key=lambda key: tuple(str(item) for item in key),
    )
    decisions: list[dict[str, Any]] = []
    for key in keys:
        template = next(row for row in summaries if _configuration_key(row) == key)
        model = str(template["model"])
        is_control = bool(template["control_distribution"])
        alpha = float(template["alpha_theory"])
        base = {name: template[name] for name in SUMMARY_GROUP_FIELDS[:10]} | {
            "purpose": template.get("purpose", "primary")
        }
        if model == "stable" or is_control:
            hill = _select_summary(summaries, key, metric="hill_alpha", fraction=PRIMARY_FRACTION)
            pickands = _select_summary(
                summaries, key, metric="pickands_xi", fraction=PRIMARY_FRACTION
            )
            gpd = _select_summary(summaries, key, metric="gpd_xi", fraction=PRIMARY_FRACTION)
            plateau = _plateau_summary(metric_rows, key, settings)
            expected_xi = 1.0 / alpha if math.isfinite(alpha) else math.nan
            hill_compatible = _inside(hill, alpha, ALPHA_MARGIN, "boot90")
            plateau_compatible = (
                math.isfinite(float(plateau["boot90_low"]))
                and float(plateau["boot90_low"]) >= -PLATEAU_SLOPE_MARGIN
                and float(plateau["boot90_high"]) <= PLATEAU_SLOPE_MARGIN
            )
            aux_compatible = _inside(pickands, expected_xi, XI_MARGIN, "boot90") or _inside(
                gpd, expected_xi, XI_MARGIN, "boot90"
            )
            hill_side = _outside(hill, alpha, ALPHA_MARGIN, "boot95")
            aux_sides = {
                _outside(pickands, expected_xi, XI_MARGIN, "boot95"),
                _outside(gpd, expected_xi, XI_MARGIN, "boot95"),
            }
            # alpha=1/xi is decreasing, so corroborating departures have opposite signs.
            corroborated = hill_side != 0 and -hill_side in aux_sides
            if settings.tier == "smoke":
                verdict, reason = "nonconclusive", "smoke tier validates the pipeline only"
            elif not is_control and not control_pass:
                verdict = "nonconclusive"
                reason = "control gate failed: " + "; ".join(control_failures)
            elif hill_compatible and plateau_compatible and aux_compatible:
                verdict, reason = "compatible", "Hill equivalence, plateau and auxiliary agree"
            elif corroborated:
                verdict, reason = "incompatible", "Hill departure corroborated by EVT sensitivity"
            else:
                verdict, reason = "nonconclusive", "equivalence/corroboration criteria not all met"
            decisions.append(
                base
                | {
                    "claim": "stable_tail_index",
                    "expected": alpha,
                    "margin": ALPHA_MARGIN,
                    "estimate": float(hill["median"]) if hill is not None else math.nan,
                    "interval_low": float(hill["boot90_low"]) if hill is not None else math.nan,
                    "interval_high": float(hill["boot90_high"]) if hill is not None else math.nan,
                    "plateau_slope": float(plateau["median"]),
                    "control_gate": control_pass or is_control,
                    "decision": verdict,
                    "reason": reason,
                }
            )

            constant = _select_summary(
                summaries, key, metric="tail_constant", fraction=PRIMARY_FRACTION
            )
            expected_constant = float(constant["expected"]) if constant is not None else math.nan
            constant_plateau = [
                _select_summary(summaries, key, metric="tail_constant", fraction=fraction)
                for fraction in PLATEAU_FRACTIONS
            ]
            within_constant = (
                constant is not None
                and math.isfinite(expected_constant)
                and float(constant["boot90_low"])
                >= (1.0 - CONSTANT_RELATIVE_MARGIN) * expected_constant
                and float(constant["boot90_high"])
                <= (1.0 + CONSTANT_RELATIVE_MARGIN) * expected_constant
            )
            plateau_constants = bool(constant_plateau) and all(
                row is not None
                and math.isfinite(expected_constant)
                and (1.0 - CONSTANT_RELATIVE_MARGIN) * expected_constant
                <= float(row["median"])
                <= (1.0 + CONSTANT_RELATIVE_MARGIN) * expected_constant
                for row in constant_plateau
            )
            outside_constant = (
                constant is not None
                and math.isfinite(expected_constant)
                and (
                    float(constant["boot95_high"])
                    < (1.0 - CONSTANT_RELATIVE_MARGIN) * expected_constant
                    or float(constant["boot95_low"])
                    > (1.0 + CONSTANT_RELATIVE_MARGIN) * expected_constant
                )
            )
            if settings.tier == "smoke":
                c_verdict, c_reason = "nonconclusive", "smoke tier validates the pipeline only"
            elif not is_control and not control_pass:
                c_verdict, c_reason = "nonconclusive", "control gate failed"
            elif within_constant and plateau_constants and hill_compatible:
                c_verdict, c_reason = "compatible", "constant equivalence and plateau pass"
            elif outside_constant and hill_compatible and plateau_compatible:
                c_verdict, c_reason = (
                    "incompatible",
                    "resolved regular variation has wrong constant",
                )
            else:
                c_verdict, c_reason = "nonconclusive", "constant is unresolved or lacks plateau"
            decisions.append(
                base
                | {
                    "claim": "tail_constant",
                    "expected": expected_constant,
                    "margin": CONSTANT_RELATIVE_MARGIN,
                    "estimate": float(constant["median"]) if constant is not None else math.nan,
                    "interval_low": float(constant["boot90_low"])
                    if constant is not None
                    else math.nan,
                    "interval_high": float(constant["boot90_high"])
                    if constant is not None
                    else math.nan,
                    "plateau_slope": math.nan,
                    "control_gate": control_pass or is_control,
                    "decision": c_verdict,
                    "reason": c_reason,
                }
            )
        elif model == "vp":
            decisions.append(
                base
                | {
                    "claim": "vp_finite_tail_trend",
                    "expected": 0.0,
                    "margin": XI_MARGIN,
                    "estimate": math.nan,
                    "interval_low": math.nan,
                    "interval_high": math.nan,
                    "plateau_slope": math.nan,
                    "control_gate": control_pass,
                    "decision": "nonconclusive",
                    "reason": "finite VP tail diagnostics do not prove an asymptotic tail index",
                }
            )

        matched_slopes = [
            row
            for row in slope_summaries
            if _configuration_key(row) == key
            and row["sample_kind"] == "full"
            and row["window"] == "primary"
        ]
        if matched_slopes:
            primary = matched_slopes[0]
            expected_slope = float(primary["expected_slope"])
            sensitivity = [
                row
                for row in slope_summaries
                if _configuration_key(row) == key
                and row["sample_kind"] == "full"
                and row["window"] in {"broad", "extreme"}
            ]
            compatible_windows = all(
                _inside(row, expected_slope, M_SLOPE_MARGIN, "boot90") for row in sensitivity
            )
            primary_compatible = _inside(primary, expected_slope, M_SLOPE_MARGIN, "boot90")
            primary_side = _outside(primary, expected_slope, M_SLOPE_MARGIN, "boot95")
            same_side = primary_side != 0 and any(
                _outside(row, expected_slope, M_SLOPE_MARGIN, "boot95") == primary_side
                for row in sensitivity
            )
            if settings.tier == "smoke":
                verdict, reason = "nonconclusive", "smoke tier validates the pipeline only"
            elif model != "stable":
                verdict, reason = "nonconclusive", "VP slope is a finite trend diagnostic only"
            elif not control_pass:
                verdict, reason = "nonconclusive", "control gate failed"
            elif primary_compatible and len(sensitivity) == 2 and compatible_windows:
                verdict, reason = "compatible", "primary and sensitivity M slopes agree with theory"
            elif same_side:
                verdict, reason = "incompatible", "M slope departure persists across fit windows"
            else:
                verdict, reason = "nonconclusive", "M slope or window stability is unresolved"
            decisions.append(
                base
                | {
                    "claim": "mass_ratio_slope",
                    "expected": expected_slope,
                    "margin": M_SLOPE_MARGIN,
                    "estimate": float(primary["median"]),
                    "interval_low": float(primary["boot90_low"]),
                    "interval_high": float(primary["boot90_high"]),
                    "plateau_slope": math.nan,
                    "control_gate": control_pass,
                    "decision": verdict,
                    "reason": reason,
                }
            )

    for row in mean_pooled:
        if (
            row["sample_kind"] != "full"
            or row.get("purpose", "primary") != "primary"
            or not math.isclose(
                float(row["threshold_fraction"]), PRIMARY_FRACTION, abs_tol=1.0e-14
            )
        ):
            continue
        base = {name: row[name] for name in SUMMARY_GROUP_FIELDS[:10]} | {
            "purpose": row.get("purpose", "primary")
        }
        decisions.append(
            base
            | {
                "claim": "mean_excess_regime",
                "expected": row["expected"],
                "margin": math.nan,
                "estimate": row["pooled_ratio"],
                "interval_low": row["seed_q16"],
                "interval_high": row["seed_q84"],
                "plateau_slope": math.nan,
                "control_gate": control_pass,
                "decision": "nonconclusive",
                "reason": "descriptive heavy-tail stability diagnostic; band is not a CI",
            }
        )
    return sorted(
        decisions,
        key=lambda row: (
            *(str(row[name]) for name in SUMMARY_GROUP_FIELDS[:10]),
            row["claim"],
        ),
    )


def diagnostics(
    tasks: Sequence[TaskData],
    metric_rows: Sequence[dict[str, Any]],
    mean_rows: Sequence[dict[str, Any]],
    slope_rows: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
    settings: AnalysisSettings,
) -> dict[str, Any]:
    control_pass, failures = _control_gate(
        tasks, summarize_tail_metrics(metric_rows, settings), settings
    )
    refinement_pass, refinement_failures, refinement_details = _refinement_gate(tasks, settings)
    publication_failures = [*failures, *refinement_failures]
    if settings.tier == "smoke":
        publication_failures.append("smoke tier cannot authorize publication-scale execution")
    publication_pass = not publication_failures
    return {
        "experiment": 1,
        "tier": settings.tier,
        "task_count": len(tasks),
        "task_paths": sorted(str(task.path) for task in tasks),
        "models": sorted({task.model for task in tasks}),
        "controls": sorted(
            {task.control_distribution for task in tasks if task.control_distribution}
        ),
        "seeds": sorted({task.seed for task in tasks}),
        "sample_counts": sorted({task.sample_count for task in tasks}),
        "tail_fractions": list(TAIL_FRACTIONS),
        "primary_fraction": PRIMARY_FRACTION,
        "plateau_fractions": list(PLATEAU_FRACTIONS),
        "gpd_fractions": list(GPD_FRACTIONS),
        "mean_excess_fractions": list(MEAN_EXCESS_FRACTIONS),
        "m_masses": list(settings.m_masses),
        "m_windows": {name: list(window) for name, window in settings.m_windows.items()},
        "bootstrap": {
            "unit": "independent seed",
            "replicates": settings.resolved_bootstrap_replicates,
            "smoke_inference_disabled": settings.tier == "smoke",
        },
        "aggregation": "blocks reduced within seed before cross-seed summaries",
        "mean_excess_uncertainty": "descriptive subsampling only; no naive bootstrap",
        "control_gate_pass": control_pass,
        "control_gate_failures": failures,
        "refinement_gate_pass": refinement_pass,
        "refinement_gate_failures": refinement_failures,
        "refinement_gate_details": refinement_details,
        "publication_gate_pass": publication_pass,
        "publication_gate_failures": publication_failures,
        "row_counts": {
            "tail_metrics": len(metric_rows),
            "mean_excess": len(mean_rows),
            "m_slopes": len(slope_rows),
            "decisions": len(decisions),
        },
        "invalid_metric_rows": sum(not bool(row["valid"]) for row in metric_rows),
        "decision_counts": {
            name: sum(row["decision"] == name for row in decisions)
            for name in ("compatible", "incompatible", "nonconclusive")
        },
    }
