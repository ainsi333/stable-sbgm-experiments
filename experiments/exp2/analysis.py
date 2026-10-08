"""Read-only statistical analysis for Experiment 2.

Experiment 2 is fixed to a Student-t(4) target, the alpha=1.5, eta=0.5
stable sampler, and the VP baseline.  This module never simulates a trajectory;
it consumes one NPZ artifact per ``(model, T, steps, seed)``.  Horizon sweeps
are retained, while refinement fits are pre-specified at ``T=2`` only.

Required NPZ keys
-----------------
``model`` (``"stable"`` or ``"vp"``), ``steps``, ``seed``, ``T``,
``epsilon``, ``alpha``, ``eta``, ``beta``, ``forward_times`` and five
``(n_times, n_samples)`` arrays:

``numerical_exact_init``
    EI trajectories initialized from the exact forward marginal p_T.
``numerical_reference_init``
    EI trajectories initialized from the asymptotic reference law.
``exact_a``, ``exact_b``, ``exact_c``
    Three mutually independent exact-marginal samples.  A/B define the
    exact--exact finite-sample Monte Carlo baseline; C is reserved for comparisons
    with the numerical laws so that the baseline is not correlated with them.

Control keys
------------
``wasserstein_orders`` must equal ``[1, 1.25]`` when present.
``score_sensitivity_wp`` has shape ``(n_times, 2)`` and contains the W_p
distance between the primary run and a higher-accuracy score run.  A D point is
controlled when this sensitivity is at most half its matched exact--exact Monte
Carlo baseline. ``d_controlled`` stores that pre-specified boolean decision.
Both arrays are mandatory: legacy artifacts without the paired high-resolution
arm fail closed and cannot enter an aggregate.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import kv

from levy_experiments.errors import ArtifactError
from levy_experiments.metrics.characteristic import (
    empirical_cf_1d,
    hoeffding_complex_radius,
    weighted_cf_rmse,
)

ORDERS = (1.0, 1.25)
MODELS = ("stable", "vp")
REQUIRED_SAMPLE_KEYS = (
    "numerical_exact_init",
    "numerical_reference_init",
    "exact_a",
    "exact_b",
    "exact_c",
)
PRIMARY_T = 2.0
PRIMARY_EPSILON = 0.05
PRIMARY_ALPHA = 1.5
PRIMARY_BETA = 1.0
MIN_GATE_SEEDS_PER_MODEL_ORDER = 3
MIN_GATE_SEED_COVERAGE_FRACTION = 0.75


class Exp2ArtifactError(ArtifactError):
    """Raised when an Experiment 2 task artifact violates the interface."""


@dataclass(frozen=True)
class TaskData:
    """Validated, in-memory representation of one simulation task."""

    path: Path
    model: str
    steps: int
    seed: int
    horizon: float
    epsilon: float
    alpha: float
    eta: float
    beta: float
    forward_times: np.ndarray
    numerical_exact_init: np.ndarray
    numerical_reference_init: np.ndarray
    exact_a: np.ndarray
    exact_b: np.ndarray
    exact_c: np.ndarray
    score_sensitivity_wp: np.ndarray
    d_controlled: np.ndarray

    @property
    def h(self) -> float:
        return (self.horizon - self.epsilon) / self.steps

    @property
    def sample_count(self) -> int:
        return int(self.numerical_exact_init.shape[1])

    @property
    def key(self) -> tuple[str, float, int, int]:
        return self.model, self.horizon, self.steps, self.seed


@dataclass(frozen=True)
class AnalysisSettings:
    """Analysis-only choices; none changes a simulated trajectory."""

    frequency_max: float = 5.0
    frequency_count: int = 81
    ecf_chunk_size: int = 8192
    subsample_fractions: tuple[float, ...] = (0.25, 0.5)
    block_count: int = 4

    def validate(self) -> None:
        if not math.isfinite(self.frequency_max) or self.frequency_max <= 0.0:
            raise Exp2ArtifactError("frequency_max must be finite and positive")
        if self.frequency_count < 3 or self.frequency_count % 2 == 0:
            raise Exp2ArtifactError("frequency_count must be odd and at least three")
        if self.ecf_chunk_size <= 0 or self.block_count < 2:
            raise Exp2ArtifactError("ecf_chunk_size must be positive and block_count >= 2")
        if any(not 0.0 < value < 1.0 for value in self.subsample_fractions):
            raise Exp2ArtifactError("subsample fractions must lie strictly between zero and one")
        if tuple(sorted(set(self.subsample_fractions))) != self.subsample_fractions:
            raise Exp2ArtifactError("subsample fractions must be sorted and distinct")


def _scalar(archive: Any, name: str) -> Any:
    if name not in archive:
        raise Exp2ArtifactError(f"missing NPZ key: {name}")
    value = np.asarray(archive[name])
    if value.size != 1:
        raise Exp2ArtifactError(f"{name} must be scalar")
    item = value.reshape(()).item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return item


def _finite_matrix(archive: Any, name: str) -> np.ndarray:
    if name not in archive:
        raise Exp2ArtifactError(f"missing NPZ key: {name}")
    values = np.asarray(archive[name])
    if (
        values.ndim != 2
        or not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.complexfloating)
    ):
        raise Exp2ArtifactError(f"{name} must be a two-dimensional real numerical array")
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1] < 8 or np.any(~np.isfinite(values)):
        raise Exp2ArtifactError(f"{name} must contain at least eight finite samples per time")
    return values


def load_task_npz(path: str | Path) -> TaskData:
    """Load and strictly validate one Experiment 2 task NPZ."""

    source = Path(path).resolve()
    if not source.is_file():
        raise Exp2ArtifactError(f"task NPZ not found: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            model = str(_scalar(archive, "model")).strip().lower()
            steps = int(_scalar(archive, "steps"))
            seed = int(_scalar(archive, "seed"))
            horizon = float(_scalar(archive, "T"))
            epsilon = float(_scalar(archive, "epsilon"))
            alpha = float(_scalar(archive, "alpha"))
            eta = float(_scalar(archive, "eta"))
            beta = float(_scalar(archive, "beta"))
            forward_times = np.asarray(archive["forward_times"], dtype=np.float64).reshape(-1)
            samples = {name: _finite_matrix(archive, name) for name in REQUIRED_SAMPLE_KEYS}
            if "wasserstein_orders" in archive:
                supplied_orders = tuple(
                    float(value)
                    for value in np.asarray(archive["wasserstein_orders"]).reshape(-1)
                )
                if supplied_orders != ORDERS:
                    raise Exp2ArtifactError(
                        f"wasserstein_orders must be exactly {ORDERS}, got {supplied_orders}"
                    )
            if "score_sensitivity_wp" not in archive:
                raise Exp2ArtifactError("missing NPZ key: score_sensitivity_wp")
            sensitivity = np.asarray(archive["score_sensitivity_wp"], dtype=np.float64)
            if "d_controlled" not in archive:
                raise Exp2ArtifactError("missing NPZ key: d_controlled")
            raw_controlled = np.asarray(archive["d_controlled"])
            if raw_controlled.dtype != np.dtype(bool):
                raise Exp2ArtifactError("d_controlled must have boolean dtype")
            controlled = np.asarray(raw_controlled, dtype=bool)
    except (OSError, KeyError, ValueError, UnicodeError) as exc:
        if isinstance(exc, Exp2ArtifactError):
            raise
        raise Exp2ArtifactError(f"cannot load {source}: {exc}") from exc

    if model not in MODELS:
        raise Exp2ArtifactError(f"model must be one of {MODELS}, got {model!r}")
    if steps <= 0 or seed < 0:
        raise Exp2ArtifactError("steps must be positive and seed non-negative")
    fixed_values = (
        ("epsilon", epsilon, PRIMARY_EPSILON),
        ("alpha", alpha, PRIMARY_ALPHA),
        ("beta", beta, PRIMARY_BETA),
    )
    for name, value, expected in fixed_values:
        if not math.isfinite(value) or not math.isclose(value, expected, abs_tol=1e-12):
            raise Exp2ArtifactError(f"Experiment 2 fixes {name}={expected}, got {value}")
    if not math.isfinite(horizon) or horizon <= epsilon:
        raise Exp2ArtifactError(f"T must be finite and greater than epsilon, got {horizon}")
    expected_eta = 0.5 if model == "stable" else 1.0
    if not math.isclose(eta, expected_eta, abs_tol=1e-12):
        raise Exp2ArtifactError(f"{model} task requires eta={expected_eta}, got {eta}")
    if forward_times.size == 0 or np.any(~np.isfinite(forward_times)):
        raise Exp2ArtifactError("forward_times must be non-empty and finite")
    if forward_times.size > 1 and not math.isclose(
        float(forward_times[0]), horizon, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise Exp2ArtifactError("the first forward time must equal T")
    if forward_times.size > 1 and np.any(np.diff(forward_times) >= 0.0):
        raise Exp2ArtifactError("forward_times must be strictly decreasing")
    if np.any(forward_times > horizon + 1e-12) or np.any(forward_times < epsilon - 1e-12):
        raise Exp2ArtifactError("forward_times must lie in [epsilon,T]")
    if not math.isclose(float(forward_times[-1]), epsilon, abs_tol=1e-12):
        raise Exp2ArtifactError("the final forward time must equal epsilon")
    common_shape = next(iter(samples.values())).shape
    if common_shape[0] != forward_times.size or any(
        value.shape != common_shape for value in samples.values()
    ):
        raise Exp2ArtifactError("all sample arrays must share shape (len(forward_times), n)")
    for first, second in (("exact_a", "exact_b"), ("exact_a", "exact_c"), ("exact_b", "exact_c")):
        if np.array_equal(samples[first], samples[second]):
            raise Exp2ArtifactError(
                f"{first} and {second} are identical; exact references must be independent"
            )
    expected_control_shape = (forward_times.size, len(ORDERS))
    if (
        sensitivity.shape != expected_control_shape
        or np.any(~np.isfinite(sensitivity))
        or np.any(sensitivity < 0.0)
    ):
        raise Exp2ArtifactError(
            "score_sensitivity_wp must be finite, non-negative and shape "
            f"{expected_control_shape}"
        )
    if controlled.shape != expected_control_shape:
        raise Exp2ArtifactError(f"d_controlled must have shape {expected_control_shape}")
    return TaskData(
        path=source,
        model=model,
        steps=steps,
        seed=seed,
        horizon=horizon,
        epsilon=epsilon,
        alpha=alpha,
        eta=eta,
        beta=beta,
        forward_times=forward_times,
        score_sensitivity_wp=sensitivity,
        d_controlled=controlled,
        **samples,
    )


def validate_task_collection(
    tasks: Sequence[TaskData],
    *,
    checkpoint_fractions: Sequence[float] | None = None,
) -> None:
    """Reject mixed, incomplete or duplicate model/N/seed task collections."""

    if not tasks:
        raise Exp2ArtifactError("no Experiment 2 task NPZ was supplied")
    keys = [task.key for task in tasks]
    if len(set(keys)) != len(keys):
        raise Exp2ArtifactError("duplicate (model, T, steps, seed) task")
    sample_count = tasks[0].sample_count
    for task in tasks[1:]:
        if task.sample_count != sample_count:
            raise Exp2ArtifactError("all tasks must use the same sample count")
    by_design: dict[float, np.ndarray] = {}
    for task in tasks:
        previous = by_design.setdefault(task.horizon, task.forward_times)
        if not np.array_equal(task.forward_times, previous):
            raise Exp2ArtifactError("all tasks at a given T must use identical checkpoints")
    if checkpoint_fractions is not None:
        fractions = np.asarray(checkpoint_fractions, dtype=np.float64)
        if (
            fractions.ndim != 1
            or fractions.size == 0
            or np.any(~np.isfinite(fractions))
            or not math.isclose(float(fractions[0]), 0.0, abs_tol=1.0e-14)
            or not math.isclose(float(fractions[-1]), 1.0, abs_tol=1.0e-14)
        ):
            raise Exp2ArtifactError("checkpoint fractions must be finite and span [0,1]")
        for task in tasks:
            expected = task.horizon - fractions * (task.horizon - task.epsilon)
            expected[0], expected[-1] = task.horizon, task.epsilon
            if task.forward_times.shape != expected.shape or not np.allclose(
                task.forward_times, expected, rtol=0.0, atol=1.0e-12
            ):
                raise Exp2ArtifactError(
                    "checkpoint times do not match configured fractions for "
                    f"{task.key}: expected={expected.tolist()}, "
                    f"got={task.forward_times.tolist()}"
                )
    by_model: dict[str, set[tuple[float, int, int]]] = defaultdict(set)
    for task in tasks:
        by_model[task.model].add((task.horizon, task.steps, task.seed))
    if set(by_model) != set(MODELS):
        raise Exp2ArtifactError("both stable and VP task families are required")
    if by_model["stable"] != by_model["vp"]:
        raise Exp2ArtifactError("stable and VP must have identical step/seed budgets")


def student_t4_cf(frequencies: np.ndarray) -> np.ndarray:
    """Characteristic function of a standard Student-t distribution with nu=4."""

    u = np.asarray(frequencies, dtype=np.float64)
    absolute = np.abs(u)
    result = np.empty_like(absolute)
    small = absolute < 1.0e-5
    # Student-t4 has variance two, hence phi(u)=1-u^2+o(u^2).
    result[small] = 1.0 - absolute[small] ** 2
    regular = ~small
    result[regular] = 2.0 * absolute[regular] ** 2 * kv(
        2.0, 2.0 * absolute[regular]
    )
    if np.any(~np.isfinite(result)):
        raise Exp2ArtifactError("non-finite Student-t4 characteristic function")
    return result


def exact_forward_cf(
    model: str,
    forward_time: float,
    frequencies: np.ndarray,
    *,
    alpha: float = PRIMARY_ALPHA,
    beta: float = PRIMARY_BETA,
) -> np.ndarray:
    """Analytic CF of the stable- or VP-corrupted Student-t4 target."""

    if model not in MODELS or forward_time < 0.0:
        raise Exp2ArtifactError("invalid model or forward time for exact CF")
    u = np.asarray(frequencies, dtype=np.float64)
    lost_mass = -math.expm1(-beta * forward_time)
    if model == "stable":
        attenuation = math.exp(-beta * forward_time / alpha)
        noise_cf = np.exp(-lost_mass * np.abs(u) ** alpha)
    else:
        attenuation = math.exp(-beta * forward_time / 2.0)
        noise_cf = np.exp(-0.5 * lost_mass * u**2)
    return np.asarray(student_t4_cf(attenuation * u) * noise_cf, dtype=np.complex128)


def _semantic_permutation(task: TaskData, label: str) -> np.ndarray:
    token = (
        f"exp2|{task.model}|T={task.horizon:.17g}|{task.steps}|{task.seed}|{label}"
    ).encode()
    seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little", signed=False)
    return np.random.default_rng(seed).permutation(task.sample_count)


def _sample_plans(task: TaskData, settings: AnalysisSettings) -> list[dict[str, Any]]:
    n = task.sample_count
    labels = REQUIRED_SAMPLE_KEYS
    permutations = {label: _semantic_permutation(task, label) for label in labels}
    plans: list[dict[str, Any]] = [
        {
            "sample_kind": "full",
            "sample_size": n,
            "replicate": -1,
            "indices": {label: np.arange(n) for label in labels},
        }
    ]
    seen_sizes: set[int] = set()
    for fraction in settings.subsample_fractions:
        size = max(8, math.floor(fraction * n))
        if size >= n or size in seen_sizes:
            continue
        seen_sizes.add(size)
        plans.append(
            {
                "sample_kind": "nested",
                "sample_size": size,
                "replicate": -1,
                "indices": {
                    label: permutation[:size] for label, permutation in permutations.items()
                },
            }
        )
    block_size = n // settings.block_count
    if block_size >= 8:
        for block in range(settings.block_count):
            start, stop = block * block_size, (block + 1) * block_size
            plans.append(
                {
                    "sample_kind": "block",
                    "sample_size": block_size,
                    "replicate": block,
                    "indices": {
                        label: permutation[start:stop]
                        for label, permutation in permutations.items()
                    },
                }
            )
    return plans


def _base_row(task: TaskData, time: float) -> dict[str, Any]:
    return {
        "source": str(task.path),
        "model": task.model,
        "horizon": task.horizon,
        "steps": task.steps,
        "h": task.h,
        "seed": task.seed,
        "forward_time": float(time),
    }


def _empirical_wasserstein_sorted_1d(
    first: np.ndarray, second: np.ndarray, *, order: float
) -> float:
    """Evaluate one-dimensional empirical OT from pre-sorted samples."""

    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    if x.size == 0 or x.size != y.size:
        raise Exp2ArtifactError("sorted empirical OT requires equal non-empty samples")
    if order < 1.0:
        raise Exp2ArtifactError("sorted empirical OT requires order >= 1")
    differences = np.abs(x - y) ** order
    return float(np.mean(differences) ** (1.0 / order))


def _wp_rows(task: TaskData, settings: AnalysisSettings) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    plans = _sample_plans(task, settings)
    matrices = {name: getattr(task, name) for name in REQUIRED_SAMPLE_KEYS}
    for time_index, forward_time in enumerate(task.forward_times):
        for plan in plans:
            selected = {
                name: matrix[time_index, plan["indices"][name]]
                for name, matrix in matrices.items()
            }
            # Every component and both prescribed orders use the same empirical
            # quantile functions.  Sorting each selected sample once preserves
            # the exact estimator while avoiding eleven of sixteen sorts.
            sorted_samples = {name: np.sort(values) for name, values in selected.items()}
            for order_index, order in enumerate(ORDERS):
                d_value = _empirical_wasserstein_sorted_1d(
                    sorted_samples["numerical_exact_init"],
                    sorted_samples["exact_c"],
                    order=order,
                )
                i_value = _empirical_wasserstein_sorted_1d(
                    sorted_samples["numerical_reference_init"],
                    sorted_samples["numerical_exact_init"],
                    order=order,
                )
                e_value = _empirical_wasserstein_sorted_1d(
                    sorted_samples["numerical_reference_init"],
                    sorted_samples["exact_c"],
                    order=order,
                )
                floor = _empirical_wasserstein_sorted_1d(
                    sorted_samples["exact_a"], sorted_samples["exact_b"], order=order
                )
                sensitivity = float(
                    task.score_sensitivity_wp[time_index, order_index]
                )
                controlled = bool(sensitivity <= 0.5 * floor)
                explicit_control = bool(task.d_controlled[time_index, order_index])
                if plan["sample_kind"] == "full" and controlled != explicit_control:
                    raise Exp2ArtifactError(
                        "d_controlled disagrees with score_sensitivity_wp <= F/2 for "
                        f"{task.key}, t={forward_time}, p={order}"
                    )
                controlled = controlled and explicit_control
                common = _base_row(task, float(forward_time)) | {
                    "metric": "wasserstein",
                    "order": order,
                    "sample_kind": plan["sample_kind"],
                    "sample_size": plan["sample_size"],
                    "replicate": plan["replicate"],
                    "floor": floor,
                    "score_sensitivity": sensitivity,
                    "controlled": controlled,
                    "hoeffding_radius": math.nan,
                }
                for component, value in (
                    ("D", d_value),
                    ("I", i_value),
                    ("E", e_value),
                    ("F", floor),
                ):
                    rows.append(common | {"component": component, "value": value})
                slack = i_value + d_value - e_value
                tolerance = 5e-12 * max(1.0, i_value, d_value, e_value)
                if slack < -tolerance:
                    raise Exp2ArtifactError(
                        f"empirical triangle failed for {task.key}, t={forward_time}, p={order}: "
                        f"slack={slack}"
                    )
                rows.append(
                    common
                    | {
                        "metric": "triangle_slack",
                        "component": "triangle",
                        "value": slack,
                    }
                )
    return rows


_ExactReferenceEcfCache = dict[tuple[str, str, int], np.ndarray]


def _float64_content_hash(values: np.ndarray) -> str:
    """Hash shape and ordered float64 bytes for safe deterministic reuse."""

    contiguous = np.ascontiguousarray(values, dtype=np.dtype("<f8"))
    digest = hashlib.sha256()
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _ecf_rows(
    task: TaskData,
    settings: AnalysisSettings,
    *,
    exact_reference_cache: _ExactReferenceEcfCache | None = None,
) -> list[dict[str, Any]]:
    frequencies = np.linspace(
        -settings.frequency_max,
        settings.frequency_max,
        settings.frequency_count,
        dtype=np.float64,
    )
    weights = np.exp(-0.5 * frequencies**2)
    frequency_hash = _float64_content_hash(frequencies)
    radius = hoeffding_complex_radius(
        sample_count=task.sample_count,
        frequency_count=settings.frequency_count,
    )
    rows: list[dict[str, Any]] = []
    for time_index, forward_time in enumerate(task.forward_times):
        analytic = exact_forward_cf(
            task.model,
            float(forward_time),
            frequencies,
            alpha=task.alpha,
            beta=task.beta,
        )
        ecfs: dict[str, np.ndarray] = {}
        for name in REQUIRED_SAMPLE_KEYS:
            samples = getattr(task, name)[time_index]
            if exact_reference_cache is None or name not in {"exact_a", "exact_b", "exact_c"}:
                ecfs[name] = empirical_cf_1d(
                    samples, frequencies, chunk_size=settings.ecf_chunk_size
                )
                continue
            cache_key = (
                _float64_content_hash(samples),
                frequency_hash,
                settings.ecf_chunk_size,
            )
            cached = exact_reference_cache.get(cache_key)
            if cached is None:
                cached = empirical_cf_1d(
                    samples, frequencies, chunk_size=settings.ecf_chunk_size
                )
                cached.setflags(write=False)
                exact_reference_cache[cache_key] = cached
            ecfs[name] = cached
        rmse = {
            name: weighted_cf_rmse(values, analytic, weights) for name, values in ecfs.items()
        }
        max_error = {
            name: float(
                max(
                    np.max(np.abs((values - analytic).real)),
                    np.max(np.abs((values - analytic).imag)),
                )
            )
            for name, values in ecfs.items()
        }
        for metric_name, values in (("ecf_rmse", rmse), ("ecf_max_component", max_error)):
            common = _base_row(task, float(forward_time)) | {
                "metric": metric_name,
                "order": math.nan,
                "sample_kind": "full",
                "sample_size": task.sample_count,
                "replicate": -1,
                "floor": math.nan,
                "score_sensitivity": math.nan,
                "controlled": False,
                "hoeffding_radius": radius,
            }
            for component, source_name in (
                ("D", "numerical_exact_init"),
                ("E", "numerical_reference_init"),
                ("A_exact", "exact_a"),
                ("B_exact", "exact_b"),
                ("C_exact", "exact_c"),
            ):
                rows.append(common | {"component": component, "value": values[source_name]})
    return rows


def analyze_task(task: TaskData, settings: AnalysisSettings | None = None) -> list[dict[str, Any]]:
    """Compute all per-seed metrics for one already-simulated task."""

    settings = AnalysisSettings() if settings is None else settings
    settings.validate()
    return _wp_rows(task, settings) + _ecf_rows(task, settings)


def analyze_tasks(
    tasks: Sequence[TaskData], settings: AnalysisSettings | None = None
) -> list[dict[str, Any]]:
    settings = AnalysisSettings() if settings is None else settings
    settings.validate()
    validate_task_collection(tasks)
    rows: list[dict[str, Any]] = []
    exact_reference_cache: _ExactReferenceEcfCache = {}
    for task in sorted(tasks, key=lambda item: item.key):
        rows.extend(_wp_rows(task, settings))
        rows.extend(
            _ecf_rows(task, settings, exact_reference_cache=exact_reference_cache)
        )
    rows.sort(
        key=lambda row: (
            row["model"],
            row["horizon"],
            row["steps"],
            row["seed"],
            -row["forward_time"],
            row["metric"],
            str(row["order"]),
            row["sample_kind"],
            row["sample_size"],
            row["replicate"],
            row["component"],
        )
    )
    return rows


SUMMARY_GROUP_FIELDS = (
    "model",
    "horizon",
    "steps",
    "h",
    "forward_time",
    "metric",
    "component",
    "order",
    "sample_kind",
    "sample_size",
)


def _summary_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        None
        if isinstance(row[name], (float, np.floating))
        and math.isnan(float(row[name]))
        else row[name]
        for name in SUMMARY_GROUP_FIELDS
    )


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate by independent seed; blocks are first reduced within seed."""

    per_seed: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (*_summary_key(row), row["seed"])
        per_seed[key].append(row)
    seed_rows: list[dict[str, Any]] = []
    for key, members in per_seed.items():
        values = np.asarray([float(member["value"]) for member in members], dtype=np.float64)
        floors = np.asarray([float(member["floor"]) for member in members], dtype=np.float64)
        finite_sensitivity = [
            float(member["score_sensitivity"])
            for member in members
            if math.isfinite(float(member["score_sensitivity"]))
        ]
        finite_radius = [
            float(member["hoeffding_radius"])
            for member in members
            if math.isfinite(float(member["hoeffding_radius"]))
        ]
        payload = {name: members[0][name] for name in SUMMARY_GROUP_FIELDS}
        payload["seed"] = key[-1]
        payload.update(
            value=float(np.median(values)),
            floor=float(np.median(floors)),
            controlled=bool(all(bool(member["controlled"]) for member in members)),
            score_sensitivity=(
                float(np.median(finite_sensitivity)) if finite_sensitivity else math.nan
            ),
            hoeffding_radius=float(np.median(finite_radius)) if finite_radius else math.nan,
            n_within_seed=len(members),
        )
        seed_rows.append(payload)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[_summary_key(row)].append(row)
    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        members = grouped[key]
        values = np.asarray([member["value"] for member in members], dtype=np.float64)
        floors = np.asarray([member["floor"] for member in members], dtype=np.float64)
        radii = np.asarray(
            [member["hoeffding_radius"] for member in members], dtype=np.float64
        )
        payload = {name: members[0][name] for name in SUMMARY_GROUP_FIELDS}
        payload.update(
            n_seeds=len(members),
            q16=float(np.quantile(values, 0.16)),
            median=float(np.median(values)),
            q84=float(np.quantile(values, 0.84)),
            floor_median=float(np.median(floors)),
            controlled_fraction=float(np.mean([member["controlled"] for member in members])),
            hoeffding_radius=(
                float(np.nanmedian(radii)) if np.any(np.isfinite(radii)) else math.nan
            ),
        )
        summaries.append(payload)
    return summaries


def fit_refinement_per_seed(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fit log D versus log h only when the pre-specified eligibility rule holds."""

    candidates = [
        row
        for row in rows
        if row["metric"] == "wasserstein"
        and row["component"] == "D"
        and row["sample_kind"] == "full"
        and row["replicate"] == -1
        and math.isclose(float(row["horizon"]), PRIMARY_T, abs_tol=1.0e-12)
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[
            (
                row["model"],
                row["horizon"],
                row["seed"],
                row["forward_time"],
                row["order"],
            )
        ].append(row)
    fits: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        eligible = [
            row
            for row in members
            if bool(row["controlled"])
            and float(row["value"]) > 2.0 * float(row["floor"])
            and float(row["value"]) > 0.0
        ]
        by_steps = {int(row["steps"]): row for row in eligible}
        eligible = list(by_steps.values())
        if len(eligible) < 3:
            continue
        h = np.asarray([float(row["h"]) for row in eligible], dtype=np.float64)
        error = np.asarray([float(row["value"]) for row in eligible], dtype=np.float64)
        slope, intercept = np.polyfit(np.log(h), np.log(error), 1)
        predicted = intercept + slope * np.log(h)
        residual = float(np.sum((np.log(error) - predicted) ** 2))
        total = float(np.sum((np.log(error) - np.mean(np.log(error))) ** 2))
        fits.append(
            {
                "model": key[0],
                "horizon": key[1],
                "seed": key[2],
                "forward_time": key[3],
                "order": key[4],
                "n_points": len(eligible),
                "steps": ";".join(str(value) for value in sorted(by_steps)),
                "h_min": float(np.min(h)),
                "h_max": float(np.max(h)),
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": 1.0 - residual / total if total > 0.0 else math.nan,
            }
        )
    return fits


def _horizon_fit_eligibility(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    terminal_full = [
        row
        for row in rows
        if row["metric"] == "wasserstein"
        and row["sample_kind"] == "full"
        and row["replicate"] == -1
        and math.isclose(
            float(row["forward_time"]), PRIMARY_EPSILON, abs_tol=1.0e-12
        )
    ]
    d_lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in terminal_full:
        task_key = (
            row["model"],
            row["horizon"],
            row["steps"],
            row["seed"],
            row["forward_time"],
            row["order"],
        )
        if row["component"] == "D":
            if task_key in d_lookup:
                raise Exp2ArtifactError(f"duplicate D row for horizon fit key {task_key}")
            d_lookup[task_key] = row
        elif row["component"] == "I":
            groups[(row["model"], row["seed"], row["order"])].append(row)

    results: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        finest: dict[float, dict[str, Any]] = {}
        for row in members:
            horizon = float(row["horizon"])
            current = finest.get(horizon)
            if current is None or float(row["h"]) < float(current["h"]):
                finest[horizon] = row
        eligible: list[dict[str, Any]] = []
        excluded_unresolved = 0
        excluded_d_confounded = 0
        for row in finest.values():
            value = float(row["value"])
            floor = float(row["floor"])
            if value <= 2.0 * floor or value <= 0.0:
                excluded_unresolved += 1
                continue
            task_key = (
                row["model"],
                row["horizon"],
                row["steps"],
                row["seed"],
                row["forward_time"],
                row["order"],
            )
            d_row = d_lookup.get(task_key)
            d_limit = max(2.0 * floor, 0.25 * value)
            if d_row is None or float(d_row["value"]) > d_limit:
                excluded_d_confounded += 1
                continue
            eligible.append(row)
        results.append(
            {
                "key": key,
                "eligible": eligible,
                "n_candidate_horizons": len(finest),
                "n_excluded_unresolved": excluded_unresolved,
                "n_excluded_D_confounded": excluded_d_confounded,
            }
        )
    return results


def fit_horizon_initialization_per_seed(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Describe log-I decay with T, per seed, on controlled horizon points.

    The finest available discretization is selected at each horizon.  A point
    must satisfy ``I > 2F`` and ``D <= max(2F, 0.25 I)``.  At least three such
    horizons are required.  Fits are descriptive and never compared between
    stable and VP; total error E is not fitted because it mixes D and I.
    """

    fits: list[dict[str, Any]] = []
    for group in _horizon_fit_eligibility(rows):
        key = group["key"]
        eligible = group["eligible"]
        if len(eligible) < 3:
            continue
        horizons = np.asarray(
            [float(row["horizon"]) for row in eligible], dtype=np.float64
        )
        errors = np.asarray([float(row["value"]) for row in eligible], dtype=np.float64)
        step_sizes = np.asarray([float(row["h"]) for row in eligible], dtype=np.float64)
        slope, intercept = np.polyfit(horizons, np.log(errors), 1)
        predicted = intercept + slope * horizons
        residual = float(np.sum((np.log(errors) - predicted) ** 2))
        total = float(np.sum((np.log(errors) - np.mean(np.log(errors))) ** 2))
        fits.append(
            {
                "model": key[0],
                "seed": key[1],
                "order": key[2],
                "component": "I",
                "n_points": len(eligible),
                "n_candidate_horizons": group["n_candidate_horizons"],
                "n_excluded_unresolved": group["n_excluded_unresolved"],
                "n_excluded_D_confounded": group["n_excluded_D_confounded"],
                "horizons": ";".join(format(value, ".17g") for value in sorted(horizons)),
                "steps": ";".join(
                    str(int(row["steps"]))
                    for row in sorted(eligible, key=lambda item: float(item["horizon"]))
                ),
                "step_size_min": float(np.min(step_sizes)),
                "step_size_max": float(np.max(step_sizes)),
                "step_size_cv": float(np.std(step_sizes) / np.mean(step_sizes)),
                "log_error_slope": float(slope),
                "log_error_intercept": float(intercept),
                "r_squared": 1.0 - residual / total if total > 0.0 else math.nan,
                "status": "descriptive_per_seed_only",
            }
        )
    return fits


def _finest_refinement_fraction(rows: Sequence[dict[str, Any]]) -> float | None:
    candidates = [
        row
        for row in rows
        if row["metric"] == "wasserstein"
        and row["component"] in {"D", "I"}
        and row["sample_kind"] == "full"
        and row["replicate"] == -1
        and math.isclose(float(row["horizon"]), PRIMARY_T, abs_tol=1.0e-12)
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[
            (
                row["model"],
                row["horizon"],
                row["seed"],
                row["forward_time"],
                row["order"],
                row["component"],
            )
        ].append(row)
    passed: list[bool] = []
    for members in groups.values():
        ordered = sorted(members, key=lambda row: float(row["h"]))
        if len(ordered) < 2:
            continue
        fine, next_fine = ordered[0], ordered[1]
        tolerance = max(float(fine["floor"]), 0.1 * float(fine["value"]))
        control_ok = fine["component"] != "D" or (
            bool(fine["controlled"]) and bool(next_fine["controlled"])
        )
        passed.append(
            control_ok and abs(float(fine["value"]) - float(next_fine["value"])) <= tolerance
        )
    return float(np.mean(passed)) if passed else None


def _required_gate_seed_count(seed_count: int) -> int:
    return max(
        MIN_GATE_SEEDS_PER_MODEL_ORDER,
        math.ceil(MIN_GATE_SEED_COVERAGE_FRACTION * seed_count),
    )


def _publication_gate_by_model_order(
    tasks: Sequence[TaskData],
    rows: Sequence[dict[str, Any]],
    horizon_fits: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate every distributional claim separately by model and W_p order."""

    expected_seeds = sorted({int(task.seed) for task in tasks})
    required_seeds = _required_gate_seed_count(len(expected_seeds))
    details: list[dict[str, Any]] = []
    for model in MODELS:
        for order in ORDERS:
            d_group = [
                row
                for row in rows
                if row["metric"] == "wasserstein"
                and row["component"] == "D"
                and row["sample_kind"] == "full"
                and row["replicate"] == -1
                and row["model"] == model
                and math.isclose(float(row["order"]), order, abs_tol=1.0e-12)
            ]
            d_seeds = {int(row["seed"]) for row in d_group}
            d_gate_pass = (
                d_seeds == set(expected_seeds)
                and bool(d_group)
                and all(bool(row["controlled"]) for row in d_group)
            )

            horizon_fit_seeds = {
                int(fit["seed"])
                for fit in horizon_fits
                if fit.get("model") == model
                and math.isclose(float(fit.get("order", math.nan)), order, abs_tol=1.0e-12)
            }
            horizon_gate_pass = len(horizon_fit_seeds) >= required_seeds

            refinement_rows = [
                row
                for row in rows
                if row["metric"] == "wasserstein"
                and row["component"] in {"D", "I"}
                and row["sample_kind"] == "full"
                and row["replicate"] == -1
                and row["model"] == model
                and math.isclose(float(row["order"]), order, abs_tol=1.0e-12)
                and math.isclose(float(row["horizon"]), PRIMARY_T, abs_tol=1.0e-12)
            ]
            refinement_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in refinement_rows:
                refinement_by_seed[int(row["seed"])].append(row)
            evaluable_seeds: list[int] = []
            passing_seeds: list[int] = []
            for seed in expected_seeds:
                comparisons: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
                for row in refinement_by_seed.get(seed, []):
                    comparisons[(float(row["forward_time"]), str(row["component"]))].append(
                        row
                    )
                if not comparisons or {key[1] for key in comparisons} != {"D", "I"}:
                    continue
                comparison_passes: list[bool] = []
                complete = True
                for (_, component), members in comparisons.items():
                    ordered = sorted(members, key=lambda row: float(row["h"]))
                    if len(ordered) < 2:
                        complete = False
                        break
                    fine, next_fine = ordered[0], ordered[1]
                    tolerance = max(float(fine["floor"]), 0.1 * float(fine["value"]))
                    control_ok = component != "D" or (
                        bool(fine["controlled"]) and bool(next_fine["controlled"])
                    )
                    comparison_passes.append(
                        control_ok
                        and abs(float(fine["value"]) - float(next_fine["value"]))
                        <= tolerance
                    )
                if complete:
                    evaluable_seeds.append(seed)
                    if comparison_passes and all(comparison_passes):
                        passing_seeds.append(seed)
            refinement_gate_pass = len(passing_seeds) >= required_seeds

            failures: list[str] = []
            if not d_gate_pass:
                failures.append("D_score_table_control_incomplete_or_failed")
            if not horizon_gate_pass:
                failures.append(
                    f"horizon_fit_seed_coverage_{len(horizon_fit_seeds)}_below_"
                    f"{required_seeds}"
                )
            if not refinement_gate_pass:
                failures.append(
                    f"finest_pair_passing_seed_coverage_{len(passing_seeds)}_below_"
                    f"{required_seeds}"
                )
            details.append(
                {
                    "model": model,
                    "order": order,
                    "expected_seed_count": len(expected_seeds),
                    "required_seed_count": required_seeds,
                    "D_controlled_point_count": sum(
                        bool(row["controlled"]) for row in d_group
                    ),
                    "D_control_point_count": len(d_group),
                    "D_control_gate_pass": d_gate_pass,
                    "horizon_fit_seeds": sorted(horizon_fit_seeds),
                    "horizon_fit_seed_count": len(horizon_fit_seeds),
                    "horizon_fit_gate_pass": horizon_gate_pass,
                    "finest_pair_evaluable_seeds": evaluable_seeds,
                    "finest_pair_passing_seeds": passing_seeds,
                    "finest_pair_gate_pass": refinement_gate_pass,
                    "gate_pass": not failures,
                    "failures": failures,
                }
            )
    return details


def _subsample_stability_fraction(rows: Sequence[dict[str, Any]]) -> float | None:
    primary = [
        row
        for row in rows
        if row["metric"] == "wasserstein"
        and row["component"] in {"D", "I", "E"}
        and row["sample_kind"] in {"full", "nested"}
        and row["replicate"] == -1
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in primary:
        groups[
            (
                row["model"],
                row["horizon"],
                row["steps"],
                row["seed"],
                row["forward_time"],
                row["order"],
                row["component"],
            )
        ].append(row)
    passed: list[bool] = []
    for members in groups.values():
        full = [row for row in members if row["sample_kind"] == "full"]
        nested = sorted(
            (row for row in members if row["sample_kind"] == "nested"),
            key=lambda row: int(row["sample_size"]),
            reverse=True,
        )
        if len(full) != 1 or not nested:
            continue
        reference, half = full[0], nested[0]
        tolerance = max(float(reference["floor"]), 0.1 * float(reference["value"]))
        passed.append(abs(float(reference["value"]) - float(half["value"])) <= tolerance)
    return float(np.mean(passed)) if passed else None


def diagnostics(
    tasks: Sequence[TaskData],
    rows: Sequence[dict[str, Any]],
    fits: Sequence[dict[str, Any]],
    horizon_fits: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Machine-readable validity diagnostics; no threshold is silently hidden."""

    triangle = [float(row["value"]) for row in rows if row["metric"] == "triangle_slack"]
    d_rows = [
        row
        for row in rows
        if row["metric"] == "wasserstein"
        and row["component"] == "D"
        and row["sample_kind"] == "full"
        and row["replicate"] == -1
    ]
    exact_ecf = [
        row
        for row in rows
        if row["metric"] == "ecf_max_component"
        and row["component"] in {"A_exact", "B_exact", "C_exact"}
    ]
    horizon_eligibility = _horizon_fit_eligibility(rows)
    gate_by_model_order = _publication_gate_by_model_order(tasks, rows, horizon_fits)
    finest_pair_fraction = _finest_refinement_fraction(rows)
    triangle_violation_count = int(sum(value < -1e-10 for value in triangle))
    exact_ecf_fraction = (
        float(
            np.mean(
                [
                    float(row["value"]) <= float(row["hoeffding_radius"])
                    for row in exact_ecf
                ]
            )
        )
        if exact_ecf
        else None
    )
    uncontrolled_count = sum(not bool(row["controlled"]) for row in d_rows)
    publication_gate_failures: list[str] = []
    if not d_rows:
        publication_gate_failures.append("no_full_sample_D_controls")
    elif uncontrolled_count:
        publication_gate_failures.append(
            "score_table_sensitivity_exceeds_half_exact_exact_floor_for_"
            f"{uncontrolled_count}_of_{len(d_rows)}_D_points"
        )
    if triangle_violation_count:
        publication_gate_failures.append(
            f"empirical_triangle_violated_for_{triangle_violation_count}_rows"
        )
    if exact_ecf_fraction is None:
        publication_gate_failures.append("no_exact_ECF_controls")
    elif exact_ecf_fraction < 1.0:
        publication_gate_failures.append(
            "exact_ECF_outside_simultaneous_radius_fraction_"
            f"{1.0 - exact_ecf_fraction:.17g}"
        )
    for detail in gate_by_model_order:
        prefix = f"{detail['model']}_W{detail['order']:g}"
        publication_gate_failures.extend(
            f"{prefix}:{failure}" for failure in detail["failures"]
        )
    return {
        "experiment": 2,
        "target": "Student-t4",
        "models": list(MODELS),
        "orders": list(ORDERS),
        "T": PRIMARY_T,
        "epsilon": PRIMARY_EPSILON,
        "task_count": len(tasks),
        "steps": sorted({task.steps for task in tasks}),
        "seeds": sorted({task.seed for task in tasks}),
        "sample_count": tasks[0].sample_count,
        "horizons": sorted({task.horizon for task in tasks}),
        "design_points": sorted({(task.horizon, task.steps) for task in tasks}),
        "triangle_min_slack": float(min(triangle)) if triangle else None,
        "triangle_violation_count": triangle_violation_count,
        "D_controlled_fraction": (
            float(np.mean([bool(row["controlled"]) for row in d_rows]))
            if d_rows
            else 0.0
        ),
        "D_controlled_count": len(d_rows) - uncontrolled_count,
        "D_control_count": len(d_rows),
        "publication_gate_pass": not publication_gate_failures,
        "publication_gate_failures": publication_gate_failures,
        "D_resolved_above_2floor_fraction": float(
            np.mean([float(row["value"]) > 2.0 * float(row["floor"]) for row in d_rows])
        ),
        "eligible_refinement_fit_count": len(fits),
        "eligible_horizon_fit_count": len(horizon_fits),
        "publication_gate_by_model_order": gate_by_model_order,
        "horizon_fit_candidate_point_count": int(
            sum(group["n_candidate_horizons"] for group in horizon_eligibility)
        ),
        "horizon_fit_excluded_unresolved_count": int(
            sum(group["n_excluded_unresolved"] for group in horizon_eligibility)
        ),
        "horizon_fit_excluded_D_confounded_count": int(
            sum(group["n_excluded_D_confounded"] for group in horizon_eligibility)
        ),
        "horizon_fit_status": (
            "descriptive_per_seed_only_no_between_model_test"
            if all(detail["horizon_fit_gate_pass"] for detail in gate_by_model_order)
            else "nonconclusive_insufficient_per_model_order_seed_coverage"
        ),
        "horizon_claim_permitted": all(
            detail["horizon_fit_gate_pass"] for detail in gate_by_model_order
        ),
        "refinement_claim_permitted": all(
            detail["finest_pair_gate_pass"] for detail in gate_by_model_order
        ),
        "exact_ECF_within_simultaneous_radius_fraction": exact_ecf_fraction,
        "finest_pair_refinement_pass_fraction": finest_pair_fraction,
        "subsample_stability_pass_fraction": _subsample_stability_fraction(rows),
        "interpretation": {
            "D": (
                "numerical exact-pT initialization versus independent exact p_t; "
                "includes score error"
            ),
            "I": (
                "finite-h initialization proxy between numerical reference- and "
                "exact-pT-initialized laws"
            ),
            "E": "numerical reference initialization versus independent exact p_t",
            "F": (
                "independent exact-exact finite-sample Monte Carlo baseline; "
                "not a mathematical lower bound and never subtracted"
            ),
            "fits": "per seed only; requires >=3 controlled D points with D>2F",
            "score_table_gate": (
                "fail closed; every full-sample D point requires paired main-versus-hires "
                "W_p <= F/2"
            ),
            "horizon_fits": (
                "descriptive log(I)-on-T per seed; finest h at each T; requires "
                "I>2F and D<=max(2F,0.25I); no stable-versus-VP slope test"
            ),
            "horizon_gate": (
                "each (model,p) requires fits from max(3,ceil(0.75*n_seeds)) seeds; "
                "insufficient coverage is a publication-gate failure"
            ),
            "refinement_gate": (
                "each (model,p) requires all finest-pair D/I checkpoint comparisons "
                "to pass in max(3,ceil(0.75*n_seeds)) independent seeds"
            ),
        },
    }


def load_tasks(
    paths: Iterable[str | Path],
    *,
    checkpoint_fractions: Sequence[float] | None = None,
) -> list[TaskData]:
    tasks = [load_task_npz(path) for path in paths]
    validate_task_collection(tasks, checkpoint_fractions=checkpoint_fractions)
    return tasks
