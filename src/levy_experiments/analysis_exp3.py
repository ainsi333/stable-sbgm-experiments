"""Post-processing for Experiment 3, separated from JAX propagation."""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .authority import authority_manifest
from .config import (
    Experiment3Config,
    ExperimentTask,
    enumerate_tasks,
    pilot_compatibility_hash,
    pilot_compatibility_payload,
)
from .errors import ArtifactError
from .metrics import (
    empirical_cf_1d,
    empirical_cf_joint,
    empirical_wasserstein_1d,
    hoeffding_complex_radius,
    linear_mmd2,
    weighted_cf_rmse,
)
from .storage import (
    RunLayout,
    compute_code_hash,
    load_task_artifact,
    sha256_file,
    task_directory,
    write_or_verify_bytes,
)
from .theory import (
    discrete_marginal_cf,
    discrete_true_reverse_joint_cf,
    stable_marginal_cf,
    stationary_joint_cf,
    stationary_true_reverse_joint_cf,
)

METRIC_FIELDS = (
    "section",
    "method",
    "valid_method",
    "steps",
    "seed",
    "eta",
    "noise_eta",
    "lag",
    "forward_time",
    "pair_later_time",
    "pair_earlier_time",
    "metric",
    "value",
    "analytic_value",
    "floor_value",
    "hoeffding_radius",
)

REFINEMENT_FIELDS = (
    "coarse_steps",
    "fine_steps",
    "seed",
    "eta",
    "forward_time",
    "metric",
    "value",
    "threshold",
    "passed",
)

AGGREGATE_ARTIFACT_NAMES = (
    "metrics_per_seed.csv",
    "metrics_summary.csv",
    "refinement_per_seed.csv",
    "diagnostics.json",
)


@dataclass(frozen=True)
class _ExactReferenceMetrics:
    """Metrics depending only on one exact-reference sample pair."""

    forward_times: np.ndarray
    w1_floors: tuple[float, ...]
    joint_ecf_floors: tuple[float, ...]
    joint_mmd_floors: tuple[float, ...]


def _row(
    task: ExperimentTask, *, section: str, metric: str, value: float, **extra: Any
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "section": section,
        "method": task.method,
        "valid_method": task.method != "hybrid_ei_invalid",
        "steps": task.steps,
        "seed": task.seed,
        "eta": task.eta,
        "noise_eta": task.noise_eta,
        "lag": task.horizon if task.target == "stationary_sas" else "",
        "forward_time": "",
        "pair_later_time": "",
        "pair_earlier_time": "",
        "metric": metric,
        "value": value,
        "analytic_value": "",
        "floor_value": "",
        "hoeffding_radius": "",
    }
    result.update(extra)
    return result


def _frequency_grids(config: Experiment3Config):
    analysis = config.analysis
    marginal = np.linspace(
        -analysis.marginal_frequency_max,
        analysis.marginal_frequency_max,
        analysis.marginal_frequency_count,
        dtype=np.float64,
    )
    axis = np.linspace(
        -analysis.joint_frequency_max,
        analysis.joint_frequency_max,
        analysis.joint_frequency_count,
        dtype=np.float64,
    )
    mesh_u, mesh_v = np.meshgrid(axis, axis, indexing="ij")
    joint_u = mesh_u.reshape(-1)
    joint_v = mesh_v.reshape(-1)
    nonzero = (joint_u != 0.0) | (joint_v != 0.0)
    return marginal, joint_u[nonzero], joint_v[nonzero]


def _stationary_metrics(
    config: Experiment3Config,
    task: ExperimentTask,
    arrays: dict[str, np.ndarray],
    marginal_frequencies: np.ndarray,
    joint_u: np.ndarray,
    joint_v: np.ndarray,
) -> list[dict[str, Any]]:
    chunk = config.analysis.sample_chunk_size
    initial = arrays["initial"]
    terminal = arrays["terminal"]
    expected_shape = (config.stationary.particles,)
    if initial.shape != expected_shape or terminal.shape != expected_shape:
        raise ArtifactError(
            "stationary artifact particle count differs from the resolved configuration"
        )
    reference_marginal = stable_marginal_cf(marginal_frequencies, alpha=config.experiment.alpha)
    weights_marginal = np.exp(-0.5 * marginal_frequencies**2)
    initial_cf = empirical_cf_1d(initial, marginal_frequencies, chunk_size=chunk)
    terminal_cf = empirical_cf_1d(terminal, marginal_frequencies, chunk_size=chunk)
    radius = hoeffding_complex_radius(
        sample_count=initial.size,
        frequency_count=marginal_frequencies.size,
    )
    empirical_joint = empirical_cf_joint(initial, terminal, joint_u, joint_v, chunk_size=chunk)
    analytic_joint = stationary_joint_cf(
        joint_u,
        joint_v,
        alpha=config.experiment.alpha,
        beta=config.experiment.beta,
        eta=task.eta,
        lag=task.horizon,
    )
    weights_joint = np.exp(-0.5 * (joint_u**2 + joint_v**2))
    probe_u = np.asarray([config.analysis.joint_probe_u])
    probe_v = np.asarray([config.analysis.joint_probe_v])
    empirical_probe = empirical_cf_joint(initial, terminal, probe_u, probe_v, chunk_size=chunk)[0]
    analytic_probe = stationary_joint_cf(
        probe_u,
        probe_v,
        alpha=config.experiment.alpha,
        beta=config.experiment.beta,
        eta=task.eta,
        lag=task.horizon,
    )[0]
    true_probe = stationary_true_reverse_joint_cf(
        probe_u,
        probe_v,
        alpha=config.experiment.alpha,
        beta=config.experiment.beta,
        lag=task.horizon,
    )[0]
    initial_difference = initial_cf - reference_marginal
    terminal_difference = terminal_cf - reference_marginal
    joint_difference = empirical_joint - analytic_joint
    return [
        _row(
            task,
            section="stationary",
            metric="marginal_ecf_rmse_initial",
            value=weighted_cf_rmse(initial_cf, reference_marginal, weights_marginal),
            hoeffding_radius=radius,
        ),
        _row(
            task,
            section="stationary",
            metric="marginal_ecf_rmse_terminal",
            value=weighted_cf_rmse(terminal_cf, reference_marginal, weights_marginal),
            hoeffding_radius=radius,
        ),
        _row(
            task,
            section="stationary",
            metric="marginal_ecf_max_component_error_initial",
            value=float(
                max(
                    np.max(np.abs(initial_difference.real)),
                    np.max(np.abs(initial_difference.imag)),
                )
            ),
            hoeffding_radius=radius,
        ),
        _row(
            task,
            section="stationary",
            metric="marginal_ecf_max_component_error_terminal",
            value=float(
                max(
                    np.max(np.abs(terminal_difference.real)),
                    np.max(np.abs(terminal_difference.imag)),
                )
            ),
            hoeffding_radius=radius,
        ),
        _row(
            task,
            section="stationary",
            metric="joint_ecf_rmse_to_analytic_popov",
            value=weighted_cf_rmse(empirical_joint, analytic_joint, weights_joint),
        ),
        _row(
            task,
            section="stationary",
            metric="joint_ecf_max_component_error_to_analytic_popov",
            value=float(
                max(
                    np.max(np.abs(joint_difference.real)),
                    np.max(np.abs(joint_difference.imag)),
                )
            ),
        ),
        _row(
            task,
            section="stationary",
            metric="joint_probe_abs_error_to_analytic_popov",
            value=float(abs(empirical_probe - analytic_probe)),
            analytic_value=float(analytic_probe.real),
        ),
        _row(
            task,
            section="stationary",
            metric="joint_probe_real",
            value=float(empirical_probe.real),
            analytic_value=float(analytic_probe.real),
            floor_value=float(true_probe.real),
        ),
    ]


def _exact_reference_metrics(
    config: Experiment3Config,
    reference_arrays: dict[str, np.ndarray],
    joint_u: np.ndarray,
    joint_v: np.ndarray,
) -> _ExactReferenceMetrics:
    """Compute reusable exact-versus-exact floors for one ``(N, seed)`` pair."""

    reference_a = reference_arrays["reference_a"]
    reference_b = reference_arrays["reference_b"]
    times = reference_arrays["forward_times"]
    if reference_a.shape != reference_b.shape or reference_a.shape[0] != times.size:
        raise ArtifactError("exact-reference arrays have incompatible shapes")
    chunk = config.analysis.sample_chunk_size
    joint_weights = np.exp(-0.5 * (joint_u**2 + joint_v**2))
    w1_floors = tuple(
        empirical_wasserstein_1d(reference_a[index], reference_b[index])
        for index in range(times.size)
    )
    joint_ecf_floors: list[float] = []
    joint_mmd_floors: list[float] = []
    for pair_index in range(times.size - 1):
        later_time = float(times[pair_index])
        earlier_time = float(times[pair_index + 1])
        exact_joint = discrete_true_reverse_joint_cf(
            joint_u,
            joint_v,
            later_forward_time=later_time,
            earlier_forward_time=earlier_time,
            alpha=config.experiment.alpha,
            beta=config.experiment.beta,
            atoms=config.nonstationary.atoms,
            weights=config.nonstationary.weights,
        )
        reference_joint = empirical_cf_joint(
            reference_a[pair_index],
            reference_a[pair_index + 1],
            joint_u,
            joint_v,
            chunk_size=chunk,
        )
        joint_ecf_floors.append(weighted_cf_rmse(reference_joint, exact_joint, joint_weights))
        reference_pairs_a = np.column_stack(
            (reference_a[pair_index], reference_a[pair_index + 1])
        )
        reference_pairs_b = np.column_stack(
            (reference_b[pair_index], reference_b[pair_index + 1])
        )
        joint_mmd_floors.append(
            linear_mmd2(
                reference_pairs_a,
                reference_pairs_b,
                bandwidth=config.analysis.mmd_bandwidth,
                pairing_seed=config.analysis.mmd_feature_seed,
                feature_count=config.analysis.mmd_feature_count,
                chunk_size=config.analysis.sample_chunk_size,
            )
        )
    return _ExactReferenceMetrics(
        forward_times=np.asarray(times, dtype=np.float64),
        w1_floors=w1_floors,
        joint_ecf_floors=tuple(joint_ecf_floors),
        joint_mmd_floors=tuple(joint_mmd_floors),
    )


def _nonstationary_metrics(
    config: Experiment3Config,
    task: ExperimentTask,
    arrays: dict[str, np.ndarray],
    reference_arrays: dict[str, np.ndarray],
    marginal_frequencies: np.ndarray,
    joint_u: np.ndarray,
    joint_v: np.ndarray,
    reference_metrics: _ExactReferenceMetrics | None = None,
) -> list[dict[str, Any]]:
    chunk = config.analysis.sample_chunk_size
    samples = arrays["samples"]
    reference_a = reference_arrays["reference_a"]
    reference_b = reference_arrays["reference_b"]
    times = arrays["forward_times"]
    if (
        samples.shape != reference_a.shape
        or samples.shape != reference_b.shape
        or not np.array_equal(times, reference_arrays["forward_times"])
    ):
        raise ArtifactError(f"reference/sample shape mismatch for {task.task_id}")
    if reference_metrics is None:
        reference_metrics = _exact_reference_metrics(config, reference_arrays, joint_u, joint_v)
    if not np.array_equal(times, reference_metrics.forward_times):
        raise ArtifactError(f"reference-metric time mismatch for {task.task_id}")
    rows: list[dict[str, Any]] = []
    marginal_weights = np.exp(-0.5 * marginal_frequencies**2)
    radius = hoeffding_complex_radius(
        sample_count=samples.shape[1],
        frequency_count=marginal_frequencies.size,
    )
    for time_index, forward_time in enumerate(times):
        exact_cf = discrete_marginal_cf(
            marginal_frequencies,
            float(forward_time),
            alpha=config.experiment.alpha,
            beta=config.experiment.beta,
            atoms=config.nonstationary.atoms,
            weights=config.nonstationary.weights,
        )
        empirical = empirical_cf_1d(samples[time_index], marginal_frequencies, chunk_size=chunk)
        difference = empirical - exact_cf
        w1 = empirical_wasserstein_1d(samples[time_index], reference_a[time_index])
        floor = reference_metrics.w1_floors[time_index]
        rows.extend(
            (
                _row(
                    task,
                    section="nonstationary_marginal",
                    metric="marginal_ecf_rmse",
                    value=weighted_cf_rmse(empirical, exact_cf, marginal_weights),
                    forward_time=float(forward_time),
                    hoeffding_radius=radius,
                ),
                _row(
                    task,
                    section="nonstationary_marginal",
                    metric="marginal_ecf_max_component_error",
                    value=float(
                        max(np.max(np.abs(difference.real)), np.max(np.abs(difference.imag)))
                    ),
                    forward_time=float(forward_time),
                    hoeffding_radius=radius,
                ),
                _row(
                    task,
                    section="nonstationary_marginal",
                    metric="w1_to_exact_reference",
                    value=w1,
                    forward_time=float(forward_time),
                    floor_value=floor,
                ),
            )
        )
    joint_weights = np.exp(-0.5 * (joint_u**2 + joint_v**2))
    probe_u = np.asarray([config.analysis.joint_probe_u])
    probe_v = np.asarray([config.analysis.joint_probe_v])
    for pair_index in range(times.size - 1):
        later_time = float(times[pair_index])
        earlier_time = float(times[pair_index + 1])
        empirical_joint = empirical_cf_joint(
            samples[pair_index],
            samples[pair_index + 1],
            joint_u,
            joint_v,
            chunk_size=chunk,
        )
        exact_joint = discrete_true_reverse_joint_cf(
            joint_u,
            joint_v,
            later_forward_time=later_time,
            earlier_forward_time=earlier_time,
            alpha=config.experiment.alpha,
            beta=config.experiment.beta,
            atoms=config.nonstationary.atoms,
            weights=config.nonstationary.weights,
        )
        empirical_probe = empirical_cf_joint(
            samples[pair_index],
            samples[pair_index + 1],
            probe_u,
            probe_v,
            chunk_size=chunk,
        )[0]
        exact_probe = discrete_true_reverse_joint_cf(
            probe_u,
            probe_v,
            later_forward_time=later_time,
            earlier_forward_time=earlier_time,
            alpha=config.experiment.alpha,
            beta=config.experiment.beta,
            atoms=config.nonstationary.atoms,
            weights=config.nonstationary.weights,
        )[0]
        simulated_pairs = np.column_stack((samples[pair_index], samples[pair_index + 1]))
        rows.extend(
            (
                _row(
                    task,
                    section="nonstationary_joint",
                    metric="joint_ecf_rmse_to_true_reverse",
                    value=weighted_cf_rmse(empirical_joint, exact_joint, joint_weights),
                    floor_value=reference_metrics.joint_ecf_floors[pair_index],
                    pair_later_time=later_time,
                    pair_earlier_time=earlier_time,
                ),
                _row(
                    task,
                    section="nonstationary_joint",
                    metric="joint_probe_real",
                    value=float(empirical_probe.real),
                    analytic_value=float(exact_probe.real),
                    pair_later_time=later_time,
                    pair_earlier_time=earlier_time,
                ),
                _row(
                    task,
                    section="nonstationary_joint",
                    metric="rff_mmd2_to_true_reverse",
                    value=linear_mmd2(
                        simulated_pairs,
                        np.column_stack(
                            (reference_a[pair_index], reference_a[pair_index + 1])
                        ),
                        bandwidth=config.analysis.mmd_bandwidth,
                        pairing_seed=config.analysis.mmd_feature_seed,
                        feature_count=config.analysis.mmd_feature_count,
                        chunk_size=config.analysis.sample_chunk_size,
                    ),
                    floor_value=reference_metrics.joint_mmd_floors[pair_index],
                    pair_later_time=later_time,
                    pair_earlier_time=earlier_time,
                ),
            )
        )
    return rows


def refinement_rows(
    config: Experiment3Config,
    tasks: tuple[ExperimentTask, ...],
    loaded: dict[int, tuple[dict[str, np.ndarray], dict[str, Any]]],
    marginal_frequencies: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute paired finest-level ECF changes using nested common randomness."""

    levels = tuple(sorted(config.nonstationary.steps))
    if len(levels) < 2:
        return []
    coarse_steps, fine_steps = levels[-2:]
    task_map = {
        (task.steps, task.eta, task.seed): task
        for task in tasks
        if task.method == "popov_ei" and task.steps in {coarse_steps, fine_steps}
    }
    threshold = config.analysis.refinement_tolerance_multiplier / np.sqrt(
        config.nonstationary.particles
    )
    rows: list[dict[str, Any]] = []
    for eta in config.experiment.etas:
        for seed in config.experiment.seeds:
            coarse_task = task_map.get((coarse_steps, eta, seed))
            fine_task = task_map.get((fine_steps, eta, seed))
            if coarse_task is None or fine_task is None:
                raise ArtifactError(
                    "paired refinement task missing for "
                    f"eta={eta:g}, seed={seed}, N={coarse_steps}->{fine_steps}"
                )
            coarse = loaded[coarse_task.index][0]
            fine = loaded[fine_task.index][0]
            if not np.array_equal(coarse["forward_times"], fine["forward_times"]):
                raise ArtifactError("paired refinement tasks use different checkpoint times")
            if coarse["samples"].shape != fine["samples"].shape:
                raise ArtifactError("paired refinement tasks use different sample shapes")
            for time_index, forward_time in enumerate(coarse["forward_times"]):
                coarse_cf = empirical_cf_1d(
                    coarse["samples"][time_index],
                    marginal_frequencies,
                    chunk_size=config.analysis.sample_chunk_size,
                )
                fine_cf = empirical_cf_1d(
                    fine["samples"][time_index],
                    marginal_frequencies,
                    chunk_size=config.analysis.sample_chunk_size,
                )
                difference = fine_cf - coarse_cf
                value = float(
                    max(np.max(np.abs(difference.real)), np.max(np.abs(difference.imag)))
                )
                rows.append(
                    {
                        "coarse_steps": coarse_steps,
                        "fine_steps": fine_steps,
                        "seed": seed,
                        "eta": eta,
                        "forward_time": float(forward_time),
                        "metric": "paired_marginal_ecf_max_component_change",
                        "value": value,
                        "threshold": float(threshold),
                        "passed": value <= threshold,
                    }
                )
    rows.sort(key=lambda row: (float(row["eta"]), int(row["seed"]), -float(row["forward_time"])))
    return rows


def refinement_convergence_diagnostic(
    config: Experiment3Config, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Apply the frozen finest-pair convergence rule, failing closed."""

    levels = tuple(sorted(config.nonstationary.steps))
    rule = (
        "for every eta/checkpoint cell, the inter-seed median of paired maximum "
        "ECF-component changes must not exceed 0.25/sqrt(particles_per_seed)"
    )
    if len(levels) < 2:
        return {
            "cells": [],
            "coarse_steps": None,
            "fine_steps": None,
            "pass": False,
            "reason": "at least two nested refinement levels are required",
            "rule": rule,
            "status": "not_evaluable",
        }
    coarse_steps, fine_steps = levels[-2:]
    groups: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(float(row["eta"]), float(row["forward_time"]))].append(row)
    cells: list[dict[str, Any]] = []
    missing_cells: list[str] = []
    expected_seeds = set(config.experiment.seeds)
    for eta in config.experiment.etas:
        for forward_time in (
            *config.nonstationary.checkpoints_forward,
            config.experiment.epsilon,
        ):
            members = groups.get((eta, forward_time), [])
            member_seeds = {int(member["seed"]) for member in members}
            if member_seeds != expected_seeds:
                missing_cells.append(f"eta={eta:g},t={forward_time:g}")
                continue
            values = np.asarray([float(member["value"]) for member in members], dtype=np.float64)
            thresholds = {float(member["threshold"]) for member in members}
            if len(thresholds) != 1:
                raise ArtifactError("inconsistent paired-refinement thresholds")
            threshold = thresholds.pop()
            median = float(np.median(values))
            cells.append(
                {
                    "eta": eta,
                    "forward_time": forward_time,
                    "median_change": median,
                    "n_seeds": len(members),
                    "pass": median <= threshold,
                    "q16": float(np.quantile(values, 0.16)),
                    "q84": float(np.quantile(values, 0.84)),
                    "threshold": threshold,
                }
            )
    expected_cell_count = len(config.experiment.etas) * (
        len(config.nonstationary.checkpoints_forward) + 1
    )
    complete = not missing_cells and len(cells) == expected_cell_count
    passed = complete and all(bool(cell["pass"]) for cell in cells)
    return {
        "cells": cells,
        "coarse_steps": coarse_steps,
        "expected_cell_count": expected_cell_count,
        "failed_cell_count": sum(not bool(cell["pass"]) for cell in cells),
        "fine_steps": fine_steps,
        "missing_cells": missing_cells,
        "pass": passed,
        "rule": rule,
        "status": "pass" if passed else "fail",
    }


def stationary_truth_diagnostic(
    config: Experiment3Config, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Check exact stationary CFs and discriminating joint-law separation."""

    tolerance = config.analysis.stationary_component_tolerance
    component_metrics = {
        "marginal_ecf_max_component_error_initial",
        "marginal_ecf_max_component_error_terminal",
        "joint_ecf_max_component_error_to_analytic_popov",
        "joint_probe_abs_error_to_analytic_popov",
    }
    component_rows = [
        row
        for row in rows
        if row["section"] == "stationary" and row["metric"] in component_metrics
    ]
    expected_per_metric = (
        len(config.stationary.lags)
        * len(config.experiment.etas)
        * len(config.experiment.seeds)
    )
    metric_counts = {
        metric: sum(row["metric"] == metric for row in component_rows)
        for metric in sorted(component_metrics)
    }
    complete = all(count == expected_per_metric for count in metric_counts.values())
    component_pass = complete and all(float(row["value"]) <= tolerance for row in component_rows)

    probe_rows = [
        row
        for row in rows
        if row["section"] == "stationary" and row["metric"] == "joint_probe_real"
    ]
    low_eta, high_eta = min(config.experiment.etas), max(config.experiment.etas)
    separation_cells: list[dict[str, Any]] = []
    reverse_analytic_separations: list[float] = []
    for lag in config.stationary.lags:
        low = {
            int(row["seed"]): row
            for row in probe_rows
            if float(row["lag"]) == lag and float(row["eta"]) == low_eta
        }
        high = {
            int(row["seed"]): row
            for row in probe_rows
            if float(row["lag"]) == lag and float(row["eta"]) == high_eta
        }
        if set(low) != set(config.experiment.seeds) or set(high) != set(
            config.experiment.seeds
        ):
            continue
        empirical = np.asarray(
            [abs(float(high[seed]["value"]) - float(low[seed]["value"])) for seed in low],
            dtype=np.float64,
        )
        analytic = abs(
            float(next(iter(high.values()))["analytic_value"])
            - float(next(iter(low.values()))["analytic_value"])
        )
        separation_cells.append(
            {
                "analytic_separation": analytic,
                "empirical_median_separation": float(np.median(empirical)),
                "eta_high": high_eta,
                "eta_low": low_eta,
                "lag": lag,
                "pass": (
                    analytic >= config.analysis.stationary_joint_separation_min
                    and float(np.median(empirical))
                    >= config.analysis.stationary_joint_separation_min
                ),
            }
        )
    for row in probe_rows:
        reverse_analytic_separations.append(
            abs(float(row["analytic_value"]) - float(row["floor_value"]))
        )
    separation_pass = bool(separation_cells) and any(
        bool(cell["pass"]) for cell in separation_cells
    )
    reverse_separation_pass = bool(reverse_analytic_separations) and max(
        reverse_analytic_separations
    ) >= config.analysis.stationary_joint_separation_min
    particle_count_pass = (
        config.stationary.particles >= config.analysis.stationary_min_particles
    )
    passed = (
        particle_count_pass
        and component_pass
        and separation_pass
        and reverse_separation_pass
    )
    return {
        "component_metric_counts": metric_counts,
        "component_tolerance": tolerance,
        "exact_cf_component_pass": component_pass,
        "joint_eta_separation_pass": separation_pass,
        "minimum_particles": config.analysis.stationary_min_particles,
        "particle_count": config.stationary.particles,
        "particle_count_pass": particle_count_pass,
        "pass": passed,
        "reverse_pair_analytic_separation_max": (
            max(reverse_analytic_separations) if reverse_analytic_separations else None
        ),
        "reverse_pair_separation_pass": reverse_separation_pass,
        "separation_cells": separation_cells,
        "status": "pass" if passed else "fail",
    }


def hybrid_rejection_diagnostic(
    config: Experiment3Config, hybrid_errors: list[dict[str, Any]]
) -> dict[str, Any]:
    """Reject the invalid hybrid with a seed-robust simultaneous ECF rule."""

    time_count = len(config.nonstationary.checkpoints_forward) + 1
    simultaneous_radius = hoeffding_complex_radius(
        sample_count=config.nonstationary.particles,
        frequency_count=config.analysis.marginal_frequency_count * time_count,
        family_error=config.analysis.hybrid_family_error,
    )
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in hybrid_errors:
        by_seed[int(row["seed"])].append(row)
    seed_decisions: list[dict[str, Any]] = []
    complete = set(by_seed) == set(config.experiment.seeds)
    expected_times = {
        *config.nonstationary.checkpoints_forward,
        config.experiment.epsilon,
    }
    for seed in config.experiment.seeds:
        members = by_seed.get(seed, [])
        seed_complete = (
            len(members) == time_count
            and {float(member["forward_time"]) for member in members} == expected_times
        )
        maximum = max((float(row["value"]) for row in members), default=float("-inf"))
        seed_decisions.append(
            {
                "complete": seed_complete,
                "maximum_component_error": maximum,
                "rejected": seed_complete and maximum > simultaneous_radius,
                "seed": seed,
            }
        )
        complete = complete and seed_complete
    required_rejections = math.ceil(
        config.analysis.hybrid_required_seed_fraction * len(config.experiment.seeds)
    )
    rejected_count = sum(bool(item["rejected"]) for item in seed_decisions)
    false_rejection_bound = sum(
        math.comb(len(config.experiment.seeds), count)
        * config.analysis.hybrid_family_error**count
        * (1.0 - config.analysis.hybrid_family_error)
        ** (len(config.experiment.seeds) - count)
        for count in range(required_rejections, len(config.experiment.seeds) + 1)
    )
    passed = complete and rejected_count >= required_rejections
    return {
        "complete": complete,
        "family_error_per_seed": config.analysis.hybrid_family_error,
        "false_rejection_probability_upper_bound": false_rejection_bound,
        "pass": passed,
        "rejected_seed_count": rejected_count,
        "required_rejected_seed_count": required_rejections,
        "required_seed_fraction": config.analysis.hybrid_required_seed_fraction,
        "seed_decisions": seed_decisions,
        "simultaneous_radius": simultaneous_radius,
        "status": "pass" if passed else "fail",
    }


def _float_text(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")
    return str(value)


def rows_to_csv(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _float_text(row.get(field, "")) for field in fields})
    return buffer.getvalue().encode("utf-8")


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouping_fields = (
        "section",
        "method",
        "valid_method",
        "steps",
        "eta",
        "noise_eta",
        "lag",
        "forward_time",
        "pair_later_time",
        "pair_earlier_time",
        "metric",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in grouping_fields)].append(row)
    summaries: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        members = groups[key]
        values = np.asarray([float(member["value"]) for member in members], dtype=np.float64)
        summary = dict(zip(grouping_fields, key, strict=True))
        summary.update(
            {
                "n_seeds": len(members),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "q16": float(np.quantile(values, 0.16)),
                "q84": float(np.quantile(values, 0.84)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
        for field in ("analytic_value", "floor_value", "hoeffding_radius"):
            present = [float(member[field]) for member in members if member[field] != ""]
            summary[field] = float(np.median(present)) if present else ""
        summaries.append(summary)
    return summaries


def _aggregate_manifest_payload(
    config: Experiment3Config,
    layout: RunLayout,
    tasks: tuple[ExperimentTask, ...],
    *,
    code_hash: str,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    artifacts = {
        name: sha256_file(layout.aggregate_dir / name) for name in AGGREGATE_ARTIFACT_NAMES
    }
    task_artifacts = []
    for task in tasks:
        directory = task_directory(layout, task)
        task_artifacts.append(
            {
                "arrays_npz_sha256": sha256_file(directory / "arrays.npz"),
                "metadata_json_sha256": sha256_file(directory / "metadata.json"),
                "task_id": task.task_id,
            }
        )
    return {
        "aggregate_artifacts": artifacts,
        "code_hash": code_hash,
        "config_hash": config.resolved_hash,
        "pilot_compatibility_hash": pilot_compatibility_hash(config),
        "pilot_compatibility_payload": pilot_compatibility_payload(config),
        "publication_gate_failures": diagnostics["publication_gate_failures"],
        "publication_gate_pass": diagnostics["publication_gate_pass"],
        "schema_version": 1,
        "status": "complete",
        "task_artifacts": task_artifacts,
        "task_count": len(tasks),
    }


def verify_aggregate_manifest(
    config: Experiment3Config, layout: RunLayout
) -> dict[str, Any]:
    """Verify the aggregate and every raw task hash without recomputing metrics."""

    path = layout.aggregate_dir / "aggregate_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError("aggregate manifest is absent or corrupt") from exc
    expected_code_hash = compute_code_hash()
    if (
        manifest.get("status") != "complete"
        or manifest.get("config_hash") != config.resolved_hash
        or manifest.get("code_hash") != expected_code_hash
        or manifest.get("task_count") != len(enumerate_tasks(config))
        or manifest.get("pilot_compatibility_hash") != pilot_compatibility_hash(config)
        or manifest.get("pilot_compatibility_payload")
        != pilot_compatibility_payload(config)
    ):
        raise ArtifactError("aggregate manifest identity mismatch")
    artifact_hashes = manifest.get("aggregate_artifacts")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != set(
        AGGREGATE_ARTIFACT_NAMES
    ):
        raise ArtifactError("aggregate manifest artifact inventory mismatch")
    for name, expected_hash in artifact_hashes.items():
        artifact = layout.aggregate_dir / name
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise ArtifactError(f"aggregate artifact checksum mismatch: {name}")
    try:
        diagnostics = json.loads(
            (layout.aggregate_dir / "diagnostics.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError("aggregate diagnostics are corrupt") from exc
    if (
        diagnostics.get("publication_gate_pass")
        is not manifest.get("publication_gate_pass")
        or diagnostics.get("publication_gate_failures")
        != manifest.get("publication_gate_failures")
    ):
        raise ArtifactError("publication gate differs between diagnostics and manifest")
    tasks = enumerate_tasks(config)
    task_entries = manifest.get("task_artifacts")
    if not isinstance(task_entries, list) or len(task_entries) != len(tasks):
        raise ArtifactError("aggregate manifest task inventory mismatch")
    entries_by_id = {
        str(entry.get("task_id")): entry for entry in task_entries if isinstance(entry, dict)
    }
    if set(entries_by_id) != {task.task_id for task in tasks}:
        raise ArtifactError("aggregate manifest task identifiers mismatch")
    for task in tasks:
        entry = entries_by_id[task.task_id]
        directory = task_directory(layout, task)
        for filename, field in (
            ("arrays.npz", "arrays_npz_sha256"),
            ("metadata.json", "metadata_json_sha256"),
        ):
            artifact = directory / filename
            if not artifact.is_file() or sha256_file(artifact) != entry.get(field):
                raise ArtifactError(
                    f"raw task checksum mismatch: {task.task_id}/{filename}"
                )
    return manifest


def aggregate_experiment3(
    config: Experiment3Config,
    layout: RunLayout,
    *,
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Verify all tasks, calculate metrics, and write deterministic aggregates."""

    tasks = enumerate_tasks(config)
    expected_code_hash = compute_code_hash()
    loaded: dict[int, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
    missing: list[str] = []
    backend_signatures: set[tuple[str, str, str]] = set()
    for task in tasks:
        try:
            arrays, metadata = load_task_artifact(
                layout,
                task,
                expected_config_hash=config.resolved_hash,
                expected_code_hash=expected_code_hash,
            )
        except ArtifactError:
            missing.append(task.task_id)
            continue
        loaded[task.index] = (arrays, metadata)
        backend = metadata["backend"]
        backend_signatures.add(
            (str(backend["platform"]), str(backend["device_kind"]), str(metadata["precision"]))
        )
    if missing:
        preview = ", ".join(missing[:5])
        raise ArtifactError(f"Cannot aggregate: {len(missing)} missing/corrupt task(s): {preview}")
    if len(backend_signatures) != 1:
        raise ArtifactError(f"Refusing to mix backends or precisions: {sorted(backend_signatures)}")
    references: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for task in tasks:
        if task.method == "exact_reference":
            references[(task.steps, task.seed)] = loaded[task.index][0]
    marginal_frequencies, joint_u, joint_v = _frequency_grids(config)
    reference_metrics = {
        key: _exact_reference_metrics(config, arrays, joint_u, joint_v)
        for key, arrays in references.items()
    }
    rows: list[dict[str, Any]] = []
    for task in tasks:
        arrays = loaded[task.index][0]
        if task.target == "stationary_sas":
            rows.extend(
                _stationary_metrics(config, task, arrays, marginal_frequencies, joint_u, joint_v)
            )
        elif task.method not in {"exact_reference"}:
            rows.extend(
                _nonstationary_metrics(
                    config,
                    task,
                    arrays,
                    references[(task.steps, task.seed)],
                    marginal_frequencies,
                    joint_u,
                    joint_v,
                    reference_metrics[(task.steps, task.seed)],
                )
            )
    rows.sort(
        key=lambda row: tuple(
            str(row[field])
            for field in (
                "section",
                "method",
                "steps",
                "eta",
                "seed",
                "lag",
                "forward_time",
                "pair_later_time",
                "metric",
            )
        )
    )
    summaries = summarize_rows(rows)
    paired_refinement = refinement_rows(config, tasks, loaded, marginal_frequencies)
    convergence = refinement_convergence_diagnostic(config, paired_refinement)
    stationary_truth = stationary_truth_diagnostic(config, rows)
    summary_fields = tuple(summaries[0])
    layout.aggregate_dir.mkdir(parents=True, exist_ok=True)
    write_or_verify_bytes(
        layout.aggregate_dir / "metrics_per_seed.csv",
        rows_to_csv(rows, METRIC_FIELDS),
        resume=resume,
    )
    write_or_verify_bytes(
        layout.aggregate_dir / "metrics_summary.csv",
        rows_to_csv(summaries, summary_fields),
        resume=resume,
    )
    write_or_verify_bytes(
        layout.aggregate_dir / "refinement_per_seed.csv",
        rows_to_csv(paired_refinement, REFINEMENT_FIELDS),
        resume=resume,
    )
    valid_errors = [
        row
        for row in rows
        if row["metric"] == "marginal_ecf_max_component_error" and row["method"] == "popov_ei"
    ]
    finest_steps = max(config.nonstationary.steps)
    hybrid_errors = [
        row
        for row in rows
        if row["metric"] == "marginal_ecf_max_component_error"
        and row["method"] == "hybrid_ei_invalid"
        and int(row["steps"]) == finest_steps
    ]
    hybrid_terminal = [
        row
        for row in rows
        if row["metric"] == "w1_to_exact_reference"
        and row["method"] == "hybrid_ei_invalid"
        and float(row["forward_time"]) == config.experiment.epsilon
        and int(row["steps"]) == finest_steps
    ]
    valid_terminal_eta = [
        row
        for row in rows
        if row["metric"] == "w1_to_exact_reference"
        and row["method"] == "popov_ei"
        and float(row["forward_time"]) == config.experiment.epsilon
        and int(row["steps"]) == finest_steps
        and float(row["eta"]) == config.nonstationary.hybrid_drift_eta
    ]
    valid_finest_errors = [row for row in valid_errors if int(row["steps"]) == finest_steps]
    valid_finest_pass = bool(valid_finest_errors) and all(
        float(row["value"]) <= float(row["hoeffding_radius"])
        for row in valid_finest_errors
    )
    hybrid_rejection = hybrid_rejection_diagnostic(config, hybrid_errors)
    publication_gate_failures: list[str] = []
    if not bool(convergence["pass"]):
        if convergence["status"] == "not_evaluable":
            publication_gate_failures.append(str(convergence["reason"]))
        else:
            publication_gate_failures.append(
                "paired refinement failed for "
                f"{convergence['failed_cell_count']}/{convergence['expected_cell_count']} "
                f"eta/checkpoint cells at N={convergence['coarse_steps']}->"
                f"{convergence['fine_steps']}"
            )
    if not valid_finest_pass:
        publication_gate_failures.append(
            "at least one valid finest-level marginal ECF exceeds its simultaneous envelope"
        )
    if not bool(stationary_truth["pass"]):
        publication_gate_failures.append(
            "stationary analytic-CF or joint-law separation gate failed"
        )
    if not bool(hybrid_rejection["pass"]):
        publication_gate_failures.append(
            "the invalid hybrid failed the seed-robust simultaneous rejection rule"
        )
    diagnostics = {
        "authority": authority_manifest(),
        "backend_signature": list(next(iter(backend_signatures))),
        "code_hash": expected_code_hash,
        "config_hash": config.resolved_hash,
        "ecf_simultaneous_pass_fraction_valid": float(
            np.mean([float(row["value"]) <= float(row["hoeffding_radius"]) for row in valid_errors])
        ),
        "finest_valid_marginal_gate_pass": valid_finest_pass,
        "hybrid_control": {
            "excluded_from_valid_comparisons": True,
            "label": (
                "invalid hybrid: drift eta="
                f"{config.nonstationary.hybrid_drift_eta:g}, noise eta="
                f"{config.nonstationary.hybrid_noise_eta:g}"
            ),
            "simultaneous_rejection": hybrid_rejection,
            "median_terminal_w1": float(
                np.median([float(row["value"]) for row in hybrid_terminal])
            ),
            "matched_drift_valid_median_terminal_w1": float(
                np.median([float(row["value"]) for row in valid_terminal_eta])
            ),
        },
        "missing_tasks": [],
        "pilot_compatibility_hash": pilot_compatibility_hash(config),
        "pilot_compatibility_payload": pilot_compatibility_payload(config),
        "publication_gate_failures": publication_gate_failures,
        "publication_gate_pass": not publication_gate_failures,
        "refinement_convergence": convergence,
        "stationary_truth": stationary_truth,
        "task_count": len(tasks),
    }
    diagnostics_bytes = (
        json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    write_or_verify_bytes(
        layout.aggregate_dir / "diagnostics.json", diagnostics_bytes, resume=resume
    )
    manifest = _aggregate_manifest_payload(
        config,
        layout,
        tasks,
        code_hash=expected_code_hash,
        diagnostics=diagnostics,
    )
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    write_or_verify_bytes(
        layout.aggregate_dir / "aggregate_manifest.json",
        manifest_bytes,
        resume=resume,
    )
    verify_aggregate_manifest(config, layout)
    return rows, summaries, diagnostics


def load_metrics_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
