"""Audited post-processing for the Experiment 2 initialization comparison.

The publication-scale simulations originally stored complete terminal samples,
but prescribed Wasserstein orders 1 and 1.25 for their primary aggregation.
This module performs a separate, explicitly post-hoc analysis at two orders in
the strict heavy-tail mixing range.  It never imports or runs a simulator.

Only tasks tagged ``horizon_sweep`` are used.  This is essential: the
Experiment 2 run also contains finer meshes at T=2 for a different refinement
study, and mixing those meshes into the horizon curve would confound horizon
and discretization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import tomllib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.exp2.config import Experiment2Task, enumerate_tasks, load_config
from experiments.exp2.storage import (
    compute_code_hash,
    load_task_artifact,
    run_layout,
    sha256_file,
)
from levy_experiments.errors import ArtifactError, ConfigurationError, ExperimentError
from levy_experiments.storage import write_or_verify_bytes

MODELS = ("stable", "vp")
MODEL_LABELS = {
    "stable": r"SP-SDE ($\alpha=1.5$, $\eta=0.5$)",
    "vp": "VP-SDE (Brownian)",
}
MODEL_COLORS = {"stable": "#0072B2", "vp": "#D55E00"}
MODEL_MARKERS = {"stable": "o", "vp": "s"}
MODEL_LINESTYLES = {"stable": "-", "vp": "--"}
REQUIRED_SAMPLE_KEYS = (
    "numerical_exact_init",
    "numerical_reference_init",
    "exact_a",
    "exact_b",
    "exact_c",
)
PER_SEED_FIELDS = (
    "model",
    "horizon",
    "steps",
    "h",
    "terminal_time",
    "sample_count",
    "seed",
    "order",
    "initialization_wp",
    "discretization_wp",
    "total_wp",
    "mc_reference_wp",
    "triangle_slack",
    "mc_resolved",
    "discretization_not_dominant",
    "source_task_id",
    "source_arrays_sha256",
)
SUMMARY_FIELDS = (
    "model",
    "horizon",
    "steps",
    "h",
    "terminal_time",
    "sample_count",
    "order",
    "n_seeds",
    "initialization_q16",
    "initialization_q25",
    "initialization_median",
    "initialization_q75",
    "initialization_q84",
    "initialization_ci_low",
    "initialization_ci_high",
    "discretization_median",
    "mc_reference_q16",
    "mc_reference_median",
    "mc_reference_q84",
    "mc_reference_ci_low",
    "mc_reference_ci_high",
    "mc_resolved_fraction",
    "discretization_not_dominant_fraction",
    "marker_resolved",
)


@dataclass(frozen=True)
class ReanalysisSettings:
    """Immutable choices for the post-hoc initialization figure."""

    orders: tuple[float, float]
    confidence_level: float
    bootstrap_replicates: int
    bootstrap_seed: int
    mc_resolution_multiplier: float
    minimum_resolved_seed_fraction: float

    def validate(self, *, alpha: float) -> None:
        if len(self.orders) != 2:
            raise ConfigurationError("exactly two Wasserstein orders are required")
        if tuple(sorted(set(self.orders))) != self.orders:
            raise ConfigurationError("Wasserstein orders must be distinct and increasing")
        if any(not math.isfinite(order) or not 1.0 < order < alpha for order in self.orders):
            raise ConfigurationError(
                f"the theorem-aligned comparison requires 1 < p < alpha={alpha:g}"
            )
        if not math.isfinite(self.confidence_level) or not 0.5 < self.confidence_level < 1.0:
            raise ConfigurationError("confidence_level must lie strictly between 0.5 and 1")
        if self.bootstrap_replicates < 1000:
            raise ConfigurationError("bootstrap_replicates must be at least 1000")
        if not 0 <= self.bootstrap_seed <= 2**31 - 1:
            raise ConfigurationError("bootstrap_seed must lie in [0, 2^31-1]")
        if (
            not math.isfinite(self.mc_resolution_multiplier)
            or self.mc_resolution_multiplier <= 1.0
        ):
            raise ConfigurationError("mc_resolution_multiplier must be greater than one")
        if (
            not math.isfinite(self.minimum_resolved_seed_fraction)
            or not 0.5 <= self.minimum_resolved_seed_fraction <= 1.0
        ):
            raise ConfigurationError(
                "minimum_resolved_seed_fraction must lie in [0.5,1]"
            )

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_reanalysis_settings(path: str | Path, *, alpha: float) -> ReanalysisSettings:
    """Load the strict, analysis-only TOML file."""

    source = Path(path).resolve()
    if not source.is_file():
        raise ConfigurationError(f"post-processing configuration not found: {source}")
    try:
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"invalid post-processing TOML {source}: {exc}") from exc
    if set(raw) != {"analysis"} or not isinstance(raw["analysis"], dict):
        raise ConfigurationError("post-processing TOML must contain only [analysis]")
    values = raw["analysis"]
    expected = {
        "orders",
        "confidence_level",
        "bootstrap_replicates",
        "bootstrap_seed",
        "mc_resolution_multiplier",
        "minimum_resolved_seed_fraction",
    }
    if set(values) != expected:
        missing = sorted(expected - set(values))
        extra = sorted(set(values) - expected)
        raise ConfigurationError(
            f"invalid [analysis] keys: missing={missing}, unexpected={extra}"
        )
    raw_orders = values["orders"]
    if not isinstance(raw_orders, list) or any(isinstance(value, bool) for value in raw_orders):
        raise ConfigurationError("analysis.orders must be a TOML array of numbers")
    try:
        settings = ReanalysisSettings(
            orders=tuple(float(value) for value in raw_orders),  # type: ignore[arg-type]
            confidence_level=float(values["confidence_level"]),
            bootstrap_replicates=int(values["bootstrap_replicates"]),
            bootstrap_seed=int(values["bootstrap_seed"]),
            mc_resolution_multiplier=float(values["mc_resolution_multiplier"]),
            minimum_resolved_seed_fraction=float(
                values["minimum_resolved_seed_fraction"]
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("non-numerical value in [analysis]") from exc
    if isinstance(values["bootstrap_replicates"], bool) or isinstance(
        values["bootstrap_seed"], bool
    ):
        raise ConfigurationError("bootstrap counts and seeds must be integers")
    if settings.bootstrap_replicates != values["bootstrap_replicates"]:
        raise ConfigurationError("bootstrap_replicates must be an integer")
    if settings.bootstrap_seed != values["bootstrap_seed"]:
        raise ConfigurationError("bootstrap_seed must be an integer")
    settings.validate(alpha=alpha)
    return settings


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"corrupt {description}: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"invalid {description}: {path}")
    return value


def _postprocessor_code_hash() -> str:
    path = Path(__file__).resolve()
    digest = hashlib.sha256()
    digest.update(path.name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_source_version(config_hash: str, source_code_hash: str) -> None:
    """Read-only analysis accepts the archived final or the installed simulator.

    Task identities, byte/content hashes and configurations are checked separately;
    this exception never authorizes resuming historical simulations with new code.
    """
    frozen = (
        "eaaf84c9384771ba1526332eca3021d0ace1c02f32d9f1e5a7f52b062f7ef737",
        "e160f3e2e59784d145a316519347656cf5217615e6e7ed6cc03712408d04fe52",
    )
    if source_code_hash != compute_code_hash() and (config_hash, source_code_hash) != frozen:
        raise ArtifactError("unsupported Experiment 2 source version for post-processing")


def _validate_source_run(
    run_dir: Path,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any], str, str]:
    """Validate the immutable run and return its config, layout and manifests."""

    run = run_dir.resolve()
    manifest_path = run / "aggregates" / "aggregate_manifest.json"
    provenance_path = run / "provenance.json"
    requested_config_path = run / "requested_config.toml"
    resolved_config_path = run / "resolved_config.json"
    manifest = _read_json(manifest_path, description="Experiment 2 aggregate manifest")
    provenance = _read_json(provenance_path, description="Experiment 2 provenance")
    resolved = _read_json(resolved_config_path, description="Experiment 2 configuration")
    if (
        manifest.get("status") != "complete"
        or manifest.get("experiment") != 2
        or manifest.get("publication_gate_pass") is not True
        or manifest.get("publication_gate_failures") != []
    ):
        raise ArtifactError("source Experiment 2 aggregate is not publication-gate complete")
    config_hash = str(manifest.get("config_hash", ""))
    source_code_hash = str(manifest.get("code_hash", ""))
    if not config_hash or not source_code_hash:
        raise ArtifactError("source aggregate has no config_hash or code_hash")
    if (
        provenance.get("config_hash") != config_hash
        or provenance.get("code_hash") != source_code_hash
        or resolved.get("resolved_hash") != config_hash
    ):
        raise ArtifactError("source run disagrees on its configuration or code hash")
    requested_hash = str(resolved.get("requested_config_source_hash", ""))
    if not requested_hash or sha256_file(requested_config_path) != requested_hash:
        raise ArtifactError("requested simulation configuration checksum mismatch")
    overrides = resolved.get("overrides")
    if not isinstance(overrides, dict) or set(overrides) != {"precision", "seed"}:
        raise ArtifactError("resolved simulation configuration has invalid overrides")
    precision_override = overrides["precision"]
    seed_override = overrides["seed"]
    if precision_override not in {None, "float32", "float64", "mixed"}:
        raise ArtifactError("resolved simulation precision override is invalid")
    if seed_override is not None and (
        not isinstance(seed_override, int) or isinstance(seed_override, bool)
    ):
        raise ArtifactError("resolved simulation seed override is invalid")
    config = load_config(
        requested_config_path,
        precision_override=precision_override,
        seed_override=seed_override,
    )
    if config.resolved_hash != config_hash:
        raise ArtifactError("reloaded simulation configuration hash mismatch")
    _validate_source_version(config_hash, source_code_hash)
    layout = run_layout(run.parent, config)
    if layout.run_dir != run:
        raise ArtifactError(
            f"run directory name does not match its configuration hash: {run}"
        )
    source = manifest.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("tasks"), list):
        raise ArtifactError("aggregate manifest has no source task inventory")
    tasks = enumerate_tasks(config)
    inventory = source["tasks"]
    if source.get("task_count") != len(tasks) or len(inventory) != len(tasks):
        raise ArtifactError("aggregate task count differs from the frozen design")
    expected_ids = {task.task_id for task in tasks}
    inventory_ids = {
        str(entry.get("task_id")) for entry in inventory if isinstance(entry, dict)
    }
    actual_ids = {
        path.name
        for path in layout.raw_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    if inventory_ids != expected_ids or actual_ids != expected_ids:
        raise ArtifactError("raw, configured and aggregate task inventories disagree")
    return (
        config,
        layout,
        manifest,
        provenance,
        sha256_file(manifest_path),
        source_code_hash,
    )


def select_horizon_tasks(tasks: Sequence[Experiment2Task]) -> tuple[Experiment2Task, ...]:
    """Select the prescribed horizon sweep and reject budget imbalance."""

    selected = tuple(task for task in tasks if "horizon_sweep" in task.purposes)
    if not selected:
        raise ArtifactError("no task is tagged horizon_sweep")
    keys_by_model: dict[str, set[tuple[float, int, int]]] = defaultdict(set)
    for task in selected:
        keys_by_model[task.model].add((task.horizon, task.steps, task.seed))
    if set(keys_by_model) != set(MODELS) or keys_by_model["stable"] != keys_by_model["vp"]:
        raise ArtifactError("stable and VP horizon sweeps do not have identical budgets")
    per_model = list(keys_by_model["stable"])
    duplicate_check = len(selected) == 2 * len(per_model)
    if not duplicate_check:
        raise ArtifactError("duplicate tasks exist in the horizon sweep")
    steps_per_horizon: dict[float, set[int]] = defaultdict(set)
    seeds_per_horizon: dict[float, set[int]] = defaultdict(set)
    for horizon, steps, seed in per_model:
        steps_per_horizon[horizon].add(steps)
        seeds_per_horizon[horizon].add(seed)
    if any(len(steps) != 1 for steps in steps_per_horizon.values()):
        raise ArtifactError("the horizon sweep has more than one N at a fixed T")
    seed_sets = list(seeds_per_horizon.values())
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ArtifactError("horizon points do not share the same independent seeds")
    return tuple(sorted(selected, key=lambda task: (task.model, task.horizon, task.seed)))


def _scalar_array(arrays: dict[str, np.ndarray], name: str) -> Any:
    if name not in arrays or np.asarray(arrays[name]).size != 1:
        raise ArtifactError(f"task array {name} must be scalar")
    value = np.asarray(arrays[name]).reshape(()).item()
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _sorted_terminal_samples(
    arrays: dict[str, np.ndarray], task: Experiment2Task, config: Any
) -> tuple[dict[str, np.ndarray], float, int]:
    """Validate task semantics and sort each terminal sample exactly once."""

    expected_scalars = {
        "model": task.model,
        "steps": task.steps,
        "seed": task.seed,
        "T": task.horizon,
        "epsilon": config.experiment.epsilon,
        "alpha": config.stable.alpha,
        "eta": config.stable.eta if task.model == "stable" else 1.0,
        "beta": config.experiment.beta,
    }
    for name, expected in expected_scalars.items():
        actual = _scalar_array(arrays, name)
        if isinstance(expected, str):
            valid = str(actual) == expected
        elif isinstance(expected, int):
            valid = int(actual) == expected
        else:
            valid = math.isclose(float(actual), float(expected), abs_tol=1.0e-12)
        if not valid:
            raise ArtifactError(
                f"task {task.task_id} has {name}={actual!r}, expected {expected!r}"
            )
    if "forward_times" not in arrays:
        raise ArtifactError(f"task {task.task_id} has no forward_times")
    times = np.asarray(arrays["forward_times"], dtype=np.float64).reshape(-1)
    fractions = np.asarray(config.design.checkpoint_fractions, dtype=np.float64)
    expected_times = task.horizon - fractions * (
        task.horizon - config.experiment.epsilon
    )
    expected_times[0], expected_times[-1] = task.horizon, config.experiment.epsilon
    if times.shape != expected_times.shape or not np.allclose(
        times, expected_times, rtol=0.0, atol=1.0e-12
    ):
        raise ArtifactError(f"task {task.task_id} checkpoint grid mismatch")
    sample_count = config.experiment.particles
    sorted_samples: dict[str, np.ndarray] = {}
    for name in REQUIRED_SAMPLE_KEYS:
        if name not in arrays:
            raise ArtifactError(f"task {task.task_id} has no {name}")
        values = np.asarray(arrays[name])
        if (
            values.shape != (times.size, sample_count)
            or not np.issubdtype(values.dtype, np.floating)
            or np.any(~np.isfinite(values))
        ):
            raise ArtifactError(
                f"task {task.task_id} {name} must be finite with shape "
                f"({times.size},{sample_count})"
            )
        sorted_samples[name] = np.sort(np.asarray(values[-1], dtype=np.float64))
    if any(
        np.array_equal(sorted_samples[first], sorted_samples[second])
        for first, second in (("exact_a", "exact_b"), ("exact_a", "exact_c"))
    ):
        raise ArtifactError(f"task {task.task_id} reuses an exact reference sample")
    return sorted_samples, float(times[-1]), sample_count


def wasserstein_from_sorted(
    first: np.ndarray, second: np.ndarray, *, order: float
) -> float:
    """Exact equal-weight one-dimensional empirical W_p for sorted samples."""

    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    if x.size == 0 or x.size != y.size:
        raise ArtifactError("sorted empirical W_p requires equal non-empty samples")
    if not math.isfinite(order) or order < 1.0:
        raise ArtifactError("empirical W_p requires finite p >= 1")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ArtifactError("empirical W_p received a non-finite sample")
    return float(np.mean(np.abs(x - y) ** order) ** (1.0 / order))


def _task_metric_rows(
    *,
    arrays: dict[str, np.ndarray],
    task: Experiment2Task,
    config: Any,
    settings: ReanalysisSettings,
    arrays_sha256: str,
) -> list[dict[str, Any]]:
    samples, terminal_time, sample_count = _sorted_terminal_samples(arrays, task, config)
    rows: list[dict[str, Any]] = []
    for order in settings.orders:
        initialization = wasserstein_from_sorted(
            samples["numerical_reference_init"],
            samples["numerical_exact_init"],
            order=order,
        )
        discretization = wasserstein_from_sorted(
            samples["numerical_exact_init"], samples["exact_c"], order=order
        )
        total = wasserstein_from_sorted(
            samples["numerical_reference_init"], samples["exact_c"], order=order
        )
        mc_reference = wasserstein_from_sorted(
            samples["exact_a"], samples["exact_b"], order=order
        )
        triangle_slack = initialization + discretization - total
        tolerance = 5.0e-12 * max(1.0, initialization, discretization, total)
        if triangle_slack < -tolerance:
            raise ArtifactError(
                f"empirical W_p triangle failed for {task.task_id}, p={order:g}"
            )
        rows.append(
            {
                "model": task.model,
                "horizon": task.horizon,
                "steps": task.steps,
                "h": (task.horizon - config.experiment.epsilon) / task.steps,
                "terminal_time": terminal_time,
                "sample_count": sample_count,
                "seed": task.seed,
                "order": order,
                "initialization_wp": initialization,
                "discretization_wp": discretization,
                "total_wp": total,
                "mc_reference_wp": mc_reference,
                "triangle_slack": max(0.0, triangle_slack),
                "mc_resolved": bool(
                    initialization
                    > settings.mc_resolution_multiplier * mc_reference
                ),
                "discretization_not_dominant": bool(
                    discretization <= 0.5 * initialization
                ),
                "source_task_id": task.task_id,
                "source_arrays_sha256": arrays_sha256,
            }
        )
    return rows


def _semantic_bootstrap_seed(base_seed: int, model: str, order: float) -> int:
    token = f"exp2-init-bootstrap|{base_seed}|{model}|{order:.17g}".encode("ascii")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def joint_seed_bootstrap_intervals(
    values: np.ndarray,
    *,
    confidence_level: float,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise percentile intervals with seed labels resampled jointly in T."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ArtifactError("joint bootstrap requires a seeds-by-horizons matrix")
    if np.any(~np.isfinite(matrix)):
        raise ArtifactError("joint bootstrap input contains a non-finite value")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, matrix.shape[0], size=(replicates, matrix.shape[0]))
    bootstrap_medians = np.median(matrix[indices, :], axis=1)
    tail = 0.5 * (1.0 - confidence_level)
    lower, upper = np.quantile(bootstrap_medians, (tail, 1.0 - tail), axis=0)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def summarize_initialization_rows(
    rows: Sequence[dict[str, Any]], settings: ReanalysisSettings
) -> list[dict[str, Any]]:
    """Summarize independent seeds and compute paired-seed bootstrap intervals."""

    if not rows:
        raise ArtifactError("no initialization rows to summarize")
    summaries: list[dict[str, Any]] = []
    for model in MODELS:
        for order in settings.orders:
            members = [
                row
                for row in rows
                if row["model"] == model and float(row["order"]) == order
            ]
            horizons = sorted({float(row["horizon"]) for row in members})
            seeds = sorted({int(row["seed"]) for row in members})
            if not horizons or not seeds:
                raise ArtifactError(f"missing rows for {model}, p={order:g}")
            by_key = {
                (float(row["horizon"]), int(row["seed"])): row for row in members
            }
            expected_keys = {(horizon, seed) for horizon in horizons for seed in seeds}
            if set(by_key) != expected_keys or len(members) != len(expected_keys):
                raise ArtifactError(
                    f"incomplete or duplicate horizon/seed grid for {model}, p={order:g}"
                )
            initialization = np.asarray(
                [
                    [by_key[(horizon, seed)]["initialization_wp"] for horizon in horizons]
                    for seed in seeds
                ],
                dtype=np.float64,
            )
            mc_reference = np.asarray(
                [
                    [by_key[(horizon, seed)]["mc_reference_wp"] for horizon in horizons]
                    for seed in seeds
                ],
                dtype=np.float64,
            )
            semantic_seed = _semantic_bootstrap_seed(
                settings.bootstrap_seed, model, order
            )
            init_low, init_high = joint_seed_bootstrap_intervals(
                initialization,
                confidence_level=settings.confidence_level,
                replicates=settings.bootstrap_replicates,
                seed=semantic_seed,
            )
            mc_low, mc_high = joint_seed_bootstrap_intervals(
                mc_reference,
                confidence_level=settings.confidence_level,
                replicates=settings.bootstrap_replicates,
                seed=semantic_seed,
            )
            for index, horizon in enumerate(horizons):
                horizon_rows = [by_key[(horizon, seed)] for seed in seeds]
                init_values = initialization[:, index]
                mc_values = mc_reference[:, index]
                first = horizon_rows[0]
                summaries.append(
                    {
                        "model": model,
                        "horizon": horizon,
                        "steps": int(first["steps"]),
                        "h": float(first["h"]),
                        "terminal_time": float(first["terminal_time"]),
                        "sample_count": int(first["sample_count"]),
                        "order": order,
                        "n_seeds": len(seeds),
                        "initialization_q16": float(np.quantile(init_values, 0.16)),
                        "initialization_q25": float(np.quantile(init_values, 0.25)),
                        "initialization_median": float(np.median(init_values)),
                        "initialization_q75": float(np.quantile(init_values, 0.75)),
                        "initialization_q84": float(np.quantile(init_values, 0.84)),
                        "initialization_ci_low": float(init_low[index]),
                        "initialization_ci_high": float(init_high[index]),
                        "discretization_median": float(
                            np.median(
                                [row["discretization_wp"] for row in horizon_rows]
                            )
                        ),
                        "mc_reference_q16": float(np.quantile(mc_values, 0.16)),
                        "mc_reference_median": float(np.median(mc_values)),
                        "mc_reference_q84": float(np.quantile(mc_values, 0.84)),
                        "mc_reference_ci_low": float(mc_low[index]),
                        "mc_reference_ci_high": float(mc_high[index]),
                        "mc_resolved_fraction": float(
                            np.mean([row["mc_resolved"] for row in horizon_rows])
                        ),
                        "discretization_not_dominant_fraction": float(
                            np.mean(
                                [
                                    row["discretization_not_dominant"]
                                    for row in horizon_rows
                                ]
                            )
                        ),
                        "marker_resolved": bool(
                            np.mean([row["mc_resolved"] for row in horizon_rows])
                            >= settings.minimum_resolved_seed_fraction
                        ),
                    }
                )
    summaries.sort(key=lambda row: (row["order"], row["model"], row["horizon"]))
    return summaries


def _log_axis_limits(rows: Sequence[dict[str, Any]]) -> tuple[float, float]:
    displayed: list[float] = []
    for row in rows:
        displayed.extend(
            (
                float(row["initialization_ci_low"]),
                float(row["initialization_ci_high"]),
                float(row["mc_reference_median"]),
            )
        )
    values = np.asarray(displayed, dtype=np.float64)
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ArtifactError("logarithmic plot requires finite positive displayed values")
    log_min, log_max = float(np.log10(values.min())), float(np.log10(values.max()))
    span = max(log_max - log_min, 0.5)
    padding = 0.10 * span
    return 10.0 ** (log_min - padding), 10.0 ** (log_max + padding)


def render_initialization_figure(
    summaries: Sequence[dict[str, Any]], settings: ReanalysisSettings
):
    """Create the two-panel comparison without opaque uncertainty ribbons."""

    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    matplotlib_cache = Path(tempfile.gettempdir()) / "levy-exp2-matplotlib-cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with plt.rc_context(
        {
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "legend.fontsize": 6.8,
            "lines.linewidth": 1.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    ):
        figure, axes = plt.subplots(
            1, 2, figsize=(7.05, 3.25), constrained_layout=True, sharex=True
        )
        all_h = {round(float(row["h"]), 14) for row in summaries}
        all_n = {int(row["sample_count"]) for row in summaries}
        all_seed_counts = {int(row["n_seeds"]) for row in summaries}
        if len(all_h) != 1 or len(all_n) != 1 or len(all_seed_counts) != 1:
            raise ArtifactError("figure requires common h, particle count and seed count")
        for axis, order in zip(axes, settings.orders, strict=True):
            panel = [row for row in summaries if float(row["order"]) == order]
            if {row["model"] for row in panel} != set(MODELS):
                raise ArtifactError(f"both models are required in the p={order:g} panel")
            for model in MODELS:
                model_rows = sorted(
                    (row for row in panel if row["model"] == model),
                    key=lambda row: float(row["horizon"]),
                )
                x = np.asarray([row["horizon"] for row in model_rows], dtype=np.float64)
                median = np.asarray(
                    [row["initialization_median"] for row in model_rows],
                    dtype=np.float64,
                )
                lower = np.asarray(
                    [row["initialization_ci_low"] for row in model_rows],
                    dtype=np.float64,
                )
                upper = np.asarray(
                    [row["initialization_ci_high"] for row in model_rows],
                    dtype=np.float64,
                )
                axis.plot(
                    x,
                    median,
                    color=MODEL_COLORS[model],
                    linestyle=MODEL_LINESTYLES[model],
                    zorder=3,
                )
                axis.errorbar(
                    x,
                    median,
                    yerr=np.vstack((median - lower, upper - median)),
                    fmt="none",
                    ecolor=MODEL_COLORS[model],
                    elinewidth=0.8,
                    capsize=2.0,
                    capthick=0.8,
                    alpha=0.78,
                    zorder=2,
                )
                for row in model_rows:
                    axis.plot(
                        float(row["horizon"]),
                        float(row["initialization_median"]),
                        marker=MODEL_MARKERS[model],
                        markersize=4.2,
                        markerfacecolor=MODEL_COLORS[model],
                        markeredgecolor=MODEL_COLORS[model],
                        markeredgewidth=1.0,
                        linestyle="none",
                        zorder=4,
                    )
                axis.plot(
                    x,
                    [row["mc_reference_median"] for row in model_rows],
                    color=MODEL_COLORS[model],
                    linestyle=":",
                    linewidth=0.95,
                    alpha=0.63,
                    zorder=1,
                )
            axis.set_yscale("log")
            axis.set_ylim(*_log_axis_limits(panel))
            axis.set_xticks([0.1, 0.25, 0.5, 1.0, 2.0])
            axis.set_xticklabels(["0.1", "0.25", "0.5", "1", "2"])
            axis.grid(alpha=0.20, which="major")
            axis.grid(alpha=0.08, which="minor", axis="y")
            axis.set_xlabel(r"forward horizon $T$")
            axis.set_ylabel(rf"terminal empirical $W_{{{order:g}}}$")
            qualifier = "near 1" if order == settings.orders[0] else r"near $\alpha=1.5$"
            axis.set_title(rf"$p={order:g}$ ({qualifier})")
        h_value = next(iter(all_h))
        n_value = next(iter(all_n))
        seed_count = next(iter(all_seed_counts))
        figure.suptitle(
            "Student-t4: finite-step initialization proxy "
            rf"($h={h_value:g}$, $n={n_value:,}$, {seed_count} seeds)",
            y=1.03,
        )
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS[model],
                linestyle=MODEL_LINESTYLES[model],
                marker=MODEL_MARKERS[model],
                markerfacecolor=MODEL_COLORS[model],
                markeredgecolor=MODEL_COLORS[model],
                markersize=4.0,
                label=MODEL_LABELS[model],
            )
            for model in MODELS
        ]
        legend_handles.extend(
            (
                Line2D(
                    [0],
                    [0],
                    color="#666666",
                    linestyle=":",
                    linewidth=1.0,
                    label="exact--exact MC reference (median)",
                ),
            )
        )
        figure.legend(
            handles=legend_handles,
            loc="outside lower center",
            frameon=False,
            ncols=3,
        )
        return figure


def _figure_bytes(figure: Any, suffix: str) -> bytes:
    descriptor, temporary_name = tempfile.mkstemp(suffix=suffix)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if suffix == ".pdf":
            figure.savefig(
                temporary,
                format="pdf",
                bbox_inches="tight",
                metadata={
                    "Creator": "levy-experiments-exp2-postprocess",
                    "Producer": "levy-experiments-exp2-postprocess",
                    "CreationDate": None,
                    "ModDate": None,
                },
            )
        elif suffix == ".png":
            figure.savefig(
                temporary,
                format="png",
                dpi=240,
                bbox_inches="tight",
                metadata={"Software": "levy-experiments-exp2-postprocess"},
            )
        else:
            raise ArtifactError(f"unsupported figure format: {suffix}")
        return temporary.read_bytes()
    finally:
        temporary.unlink(missing_ok=True)


def _csv_bytes(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(fields) - set(row):
            raise ArtifactError("post-processing CSV row is incomplete")
        values: dict[str, str] = {}
        for name in fields:
            value = row[name]
            if isinstance(value, (bool, np.bool_)):
                values[name] = "true" if value else "false"
            elif isinstance(value, (float, np.floating)):
                values[name] = format(float(value), ".17g")
            else:
                values[name] = str(value)
        writer.writerow(values)
    return buffer.getvalue().encode("utf-8")


def _caption(settings: ReanalysisSettings, summaries: Sequence[dict[str, Any]]) -> str:
    seed_count = len(
        {
            int(row["n_seeds"])
            for row in summaries
        }
    )
    if seed_count != 1:
        raise ArtifactError("summary rows disagree on their seed count")
    n_seeds = int(summaries[0]["n_seeds"])
    sample_count = int(summaries[0]["sample_count"])
    confidence = 100.0 * settings.confidence_level
    first, second = settings.orders
    return (
        "**Finite-horizon initialization comparison on a Student-t(4) target.** "
        f"Each panel reports the terminal finite-step proxy $I_p$ for p={first:g} "
        f"and p={second:g}, respectively, for the isotropic alpha-stable SP-SDE "
        "(alpha=1.5, eta=0.5) and the Brownian VP-SDE. Both orders lie strictly "
        "inside the common theorem-aligned range 1<p<alpha. Every horizon uses "
        "the pre-specified horizon-sweep mesh, giving h=0.0125 throughout. Points "
        f"are medians over {n_seeds} independent seeds with n={sample_count} particles; "
        f"capped bars are pointwise {confidence:g}% percentile bootstrap intervals "
        "for the median, obtained by jointly resampling seed labels across horizons. "
        "Thin dotted curves show the model-specific median exact--exact Monte Carlo "
        "reference F_p; F_p is a finite-sample reference, not a mathematical lower "
        "bound, and is never subtracted. The resolution diagnostic I_p>2F_p remains "
        "available in the accompanying numerical tables but has no graphical encoding. "
        "The p=1.4 panel is more sensitive to extreme observations because p approaches "
        "alpha. No W_2 comparison is reported, since the stable reference has no finite "
        "second moment.\n"
    )


def _validation_report(
    *,
    run_dir: Path,
    destination: Path,
    settings: ReanalysisSettings,
    rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    source_config_hash: str,
    source_code_hash: str,
    analysis_hash: str,
    verified_task_count: int,
    selected_task_count: int,
) -> str:
    h_values = sorted({round(float(row["h"]), 14) for row in rows})
    resolution_limited_points = sum(
        not bool(row["marker_resolved"]) for row in summaries
    )
    return f"""# Experiment 2 initialization reanalysis

Status: complete. No trajectory was simulated or modified.

- Source run: `{run_dir}`
- Destination: `{destination}`
- Source configuration hash: `{source_config_hash}`
- Source simulation code hash: `{source_code_hash}`
- Analysis specification hash: `{analysis_hash}`
- Orders: {list(settings.orders)}; strict admissible range: `1 < p < 1.5`
- Selected task purpose: `horizon_sweep` only
- Verified source raw tasks: {verified_task_count}
- Selected horizon-sweep tasks: {selected_task_count}
- Per-seed metric rows: {len(rows)}
- Horizon step sizes observed: {h_values}
- Numerical resolution diagnostic not satisfied: {resolution_limited_points}/{len(summaries)}
- This diagnostic is retained in the CSV and is not encoded by marker fill.

The plotted quantity is the finite-step empirical initialization proxy
`I_p = W_p(numerical_reference_init, numerical_exact_init)`. The exact--exact
quantity `F_p` is shown only as a finite-sample resolution reference. The CSV
also records `D_p`, the total empirical discrepancy, the triangle slack, the
full inter-seed 16--84% dispersion, and the bootstrap intervals used in the
figure.

The saved high-resolution score-table comparison exists only for the original
orders p=1 and p=1.25. It was therefore not relabelled or extrapolated to p=1.1
and p=1.4. This reanalysis makes no new score-table-sensitivity gate claim.
"""


def run_reanalysis(
    *,
    run_dir: str | Path,
    settings_path: str | Path,
    output_root: str | Path,
    resume: bool,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Verify a final run, recompute two W_p curves and write immutable outputs."""

    run = Path(run_dir).resolve()
    if not run.is_dir():
        raise ArtifactError(f"Experiment 2 run directory not found: {run}")
    (
        config,
        layout,
        source_manifest,
        _,
        source_manifest_sha256,
        source_code_hash,
    ) = _validate_source_run(run)
    settings_source = Path(settings_path).resolve()
    settings = load_reanalysis_settings(settings_source, alpha=config.stable.alpha)
    postprocessor_hash = _postprocessor_code_hash()
    analysis_spec = {
        "experiment": 2,
        "analysis": "terminal_initialization_horizon_comparison",
        "post_hoc": True,
        "horizon_selection": "task metadata purpose == horizon_sweep",
        "central_estimator": "median across independent seeds",
        "uncertainty": "joint-seed percentile bootstrap for the median",
        "score_sensitivity_gate_recomputed": False,
        "settings": settings.canonical_dict(),
        "source_config_hash": config.resolved_hash,
        "source_code_hash": source_code_hash,
        "source_aggregate_manifest_sha256": source_manifest_sha256,
        "postprocessor_code_hash": postprocessor_hash,
    }
    analysis_hash = _canonical_hash(analysis_spec)
    destination = Path(output_root).resolve() / f"exp2-initialization-{analysis_hash[:12]}"
    all_tasks = enumerate_tasks(config)
    selected_tasks = select_horizon_tasks(all_tasks)
    selected_ids = {task.task_id for task in selected_tasks}
    if dry_run:
        return {
            "status": "dry-run",
            "source_run": str(run),
            "source_config_hash": config.resolved_hash,
            "source_code_hash": source_code_hash,
            "analysis_spec_hash": analysis_hash,
            "destination": str(destination),
            "verified_task_count_when_executed": len(all_tasks),
            "selected_horizon_task_count": len(selected_tasks),
            "orders": list(settings.orders),
        }

    inventory = {
        str(entry["task_id"]): entry
        for entry in source_manifest["source"]["tasks"]
    }
    metric_rows: list[dict[str, Any]] = []
    used_tasks: list[dict[str, Any]] = []
    for task in all_tasks:
        arrays, metadata = load_task_artifact(
            layout,
            task,
            expected_config_hash=config.resolved_hash,
            expected_code_hash=source_code_hash,
        )
        entry = inventory[task.task_id]
        arrays_path = layout.raw_dir / task.task_id / "arrays.npz"
        metadata_path = layout.raw_dir / task.task_id / "metadata.json"
        arrays_sha = sha256_file(arrays_path)
        metadata_sha = sha256_file(metadata_path)
        if (
            entry.get("arrays_sha256") != arrays_sha
            or entry.get("metadata_sha256") != metadata_sha
            or metadata.get("arrays_file_sha256") != arrays_sha
        ):
            raise ArtifactError(f"aggregate inventory checksum mismatch for {task.task_id}")
        if task.task_id not in selected_ids:
            continue
        metric_rows.extend(
            _task_metric_rows(
                arrays=arrays,
                task=task,
                config=config,
                settings=settings,
                arrays_sha256=arrays_sha,
            )
        )
        used_tasks.append(
            {
                "task_id": task.task_id,
                "arrays_sha256": arrays_sha,
                "metadata_sha256": metadata_sha,
            }
        )
    metric_rows.sort(
        key=lambda row: (row["order"], row["model"], row["horizon"], row["seed"])
    )
    summaries = summarize_initialization_rows(metric_rows, settings)

    destination.mkdir(parents=True, exist_ok=True)
    old_backend = os.environ.get("MPLBACKEND")
    os.environ["MPLBACKEND"] = "Agg"
    try:
        figure = render_initialization_figure(summaries, settings)
        try:
            pdf_payload = _figure_bytes(figure, ".pdf")
            png_payload = _figure_bytes(figure, ".png")
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)
    finally:
        if old_backend is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = old_backend

    payloads = {
        "requested_analysis_config.toml": settings_source.read_bytes(),
        "resolved_analysis_spec.json": (
            json.dumps(
                analysis_spec | {"analysis_spec_hash": analysis_hash},
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
        "initialization_wp_per_seed.csv": _csv_bytes(metric_rows, PER_SEED_FIELDS),
        "initialization_wp_summary.csv": _csv_bytes(summaries, SUMMARY_FIELDS),
        "experiment2_initialization_comparison_cropped.pdf": pdf_payload,
        "experiment2_initialization_comparison_cropped.png": png_payload,
        "caption.md": _caption(settings, summaries).encode("utf-8"),
        "validation_report.md": _validation_report(
            run_dir=run,
            destination=destination,
            settings=settings,
            rows=metric_rows,
            summaries=summaries,
            source_config_hash=config.resolved_hash,
            source_code_hash=source_code_hash,
            analysis_hash=analysis_hash,
            verified_task_count=len(all_tasks),
            selected_task_count=len(selected_tasks),
        ).encode("utf-8"),
    }
    outputs: dict[str, str] = {}
    for name, payload in payloads.items():
        path = destination / name
        write_or_verify_bytes(path, payload, resume=resume)
        outputs[name] = str(path)
    manifest = {
        "status": "complete",
        "experiment": 2,
        "analysis": "terminal_initialization_horizon_comparison",
        "post_hoc": True,
        "analysis_spec_hash": analysis_hash,
        "postprocessor_code_hash": postprocessor_hash,
        "source": {
            "run_dir": str(run),
            "config_hash": config.resolved_hash,
            "code_hash": source_code_hash,
            "aggregate_manifest_sha256": source_manifest_sha256,
            "publication_gate_pass": True,
            "verified_task_count": len(all_tasks),
            "selected_horizon_task_count": len(selected_tasks),
            "used_tasks": sorted(used_tasks, key=lambda entry: entry["task_id"]),
        },
        "limitations": [
            (
                "I_p is a finite-step empirical initialization proxy, not an exact "
                "continuous-time error"
            ),
            "F_p is an exact--exact finite-sample reference, not a lower bound",
            "score-table sensitivity was not recomputed at the post-hoc orders",
            "p=1.4 is close to alpha=1.5 and is sensitive to extreme observations",
        ],
        "artifacts": {
            name: {
                "path": path,
                "sha256": hashlib.sha256(payloads[name]).hexdigest(),
                "size_bytes": len(payloads[name]),
            }
            for name, path in sorted(outputs.items())
        },
    }
    manifest_path = destination / "manifest.json"
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    write_or_verify_bytes(manifest_path, manifest_payload, resume=resume)
    outputs["manifest.json"] = str(manifest_path)
    return {
        "status": "complete",
        "analysis_spec_hash": analysis_hash,
        "source_config_hash": config.resolved_hash,
        "source_code_hash": source_code_hash,
        "destination": str(destination),
        "outputs": outputs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the Experiment 2 initialization W_p comparison from an "
            "immutable completed run; no simulation is executed."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        result = run_reanalysis(
            run_dir=arguments.run_dir,
            settings_path=arguments.config,
            output_root=arguments.output_dir,
            resume=arguments.resume,
            dry_run=arguments.dry_run,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except ExperimentError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
