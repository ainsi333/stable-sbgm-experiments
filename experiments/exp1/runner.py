"""Audited execution, provenance, and atomic commits for Experiment 1."""

from __future__ import annotations

import math
import traceback
from typing import Any

import numpy as np

from levy_experiments.devices import RuntimeDevice, simulation_dtype
from levy_experiments.errors import ArtifactError
from levy_experiments.random import stream_manifest

from .config import Experiment1Config, Experiment1Task
from .score_tables import (
    ScoreTable,
    ensure_score_table,
    ensure_sensitivity_score_table,
    table_key,
)
from .simulation import (
    compile_control_kernel,
    compile_dynamic_kernels,
    execute_control_batches,
    execute_dynamic_batches,
    stream_labels,
)
from .storage import (
    RunLayout,
    compute_code_hash,
    config_source_is_unchanged,
    load_task_artifact,
    task_directory,
    write_failure,
    write_task_artifact,
)
from .theory import (
    continuous_stable_scale_power,
    continuous_stable_tail_constant,
    discrete_stable_scale_power,
    discrete_stable_tail_constant,
    stable_radial_tail_constant,
)


def ensure_score_tables(
    layout: RunLayout,
    config: Experiment1Config,
    *,
    resume: bool,
) -> dict[str, ScoreTable]:
    """Build or verify every target-specific table before array jobs start."""

    specifications = tuple(("stable", nu) for nu in config.target.stable_nus) + tuple(
        ("vp", nu) for nu in config.target.vp_nus
    )
    tables: dict[str, ScoreTable] = {}
    for model, nu in specifications:
        key = table_key(model, nu)
        tables[key] = ensure_score_table(
            layout.run_dir, config, model, nu, resume=resume
        )
        tables[f"{key}:hires"] = ensure_sensitivity_score_table(
            layout.run_dir, config, model, nu, resume=resume
        )
    return tables


def _tail_reference_metadata(
    config: Experiment1Config,
    task: Experiment1Task,
) -> dict[str, float]:
    if task.kind == "dynamic" and task.model == "stable" and task.nu is not None:
        power = discrete_stable_scale_power(
            task.nu,
            config.experiment.horizon,
            config.experiment.epsilon,
            task.steps,
            alpha=config.stable.alpha,
            beta=config.experiment.beta,
            eta=config.stable.eta,
        )
        continuous_power = continuous_stable_scale_power(
            task.nu,
            config.experiment.horizon,
            config.experiment.epsilon,
            alpha=config.stable.alpha,
            beta=config.experiment.beta,
            eta=config.stable.eta,
        )
        return {
            "discrete_rho": power ** (1.0 / config.stable.alpha),
            "discrete_rho_power": power,
            "tail_constant_reference": discrete_stable_tail_constant(
                task.nu,
                config.experiment.horizon,
                config.experiment.epsilon,
                task.steps,
                alpha=config.stable.alpha,
                beta=config.experiment.beta,
                eta=config.stable.eta,
            ),
            "continuous_rho": continuous_power ** (1.0 / config.stable.alpha),
            "continuous_tail_constant": continuous_stable_tail_constant(
                task.nu,
                config.experiment.horizon,
                config.experiment.epsilon,
                alpha=config.stable.alpha,
                beta=config.experiment.beta,
                eta=config.stable.eta,
            ),
        }
    if task.kind == "dynamic" and task.model == "vp":
        return {"reference_scale": 1.0}
    if task.model == "sas":
        return {
            "reference_scale": 1.0,
            "tail_constant_reference": stable_radial_tail_constant(config.controls.sas_alpha),
        }
    if task.model == "pareto":
        return {
            "reference_scale": config.controls.pareto_minimum,
            "tail_constant_reference": config.controls.pareto_minimum
            ** config.controls.pareto_tail_index,
        }
    if task.model == "gaussian":
        return {"reference_scale": config.controls.gaussian_std}
    return {}


def _scientific_scalar_arrays(
    config: Experiment1Config,
    task: Experiment1Task,
) -> dict[str, np.ndarray]:
    dynamic = task.kind == "dynamic"
    eta = config.stable.eta if task.model == "stable" else 1.0
    payload: dict[str, Any] = {
        "task_kind": task.kind,
        "model": task.model,
        "control_distribution": "" if dynamic else task.model,
        "target": "student_t" if dynamic else "direct_control",
        "target_nu": 0.0 if task.nu is None else task.nu,
        "alpha": config.stable.alpha,
        "eta": eta,
        "beta": config.experiment.beta,
        "T": config.experiment.horizon,
        "epsilon": config.experiment.epsilon,
        "steps": task.steps,
        "seed": task.seed,
        "purpose": task.purpose,
        "coupling_steps": task.coupling_steps,
        "tier": config.experiment.tier,
        "particles": task.particles,
        "control_particles": task.control_particles,
    }
    payload.update(_tail_reference_metadata(config, task))
    return {name: np.asarray(value) for name, value in payload.items()}


def _lookup_table(
    task: Experiment1Task,
    tables: dict[str, ScoreTable],
) -> ScoreTable:
    if task.nu is None or task.model not in {"stable", "vp"}:
        raise ArtifactError(f"dynamic task lacks a score-table identity: {task.task_id}")
    key = table_key(task.model, task.nu)
    if key not in tables:
        raise ArtifactError(f"score table {key} was not supplied for {task.task_id}")
    table = tables[key]
    if table.model != task.model or not math.isclose(
        float(table.metadata["nu"]), task.nu, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ArtifactError(f"wrong score table supplied for {task.task_id}")
    return table


def _lookup_sensitivity_table(
    task: Experiment1Task,
    tables: dict[str, ScoreTable],
) -> ScoreTable:
    if task.nu is None or task.model not in {"stable", "vp"}:
        raise ArtifactError(f"dynamic task lacks a score-table identity: {task.task_id}")
    key = f"{table_key(task.model, task.nu)}:hires"
    if key not in tables:
        raise ArtifactError(
            f"high-resolution score table {key} was not supplied for {task.task_id}"
        )
    return tables[key]


def run_task(
    config: Experiment1Config,
    task: Experiment1Task,
    layout: RunLayout,
    *,
    device,
    runtime: RuntimeDevice,
    tables: dict[str, ScoreTable],
    resume: bool,
) -> dict[str, Any]:
    """Compile, execute, check finiteness, and atomically commit one task."""

    code_hash = compute_code_hash()
    destination = task_directory(layout, task)
    if destination.exists():
        if not resume:
            raise ArtifactError(
                f"Task already exists; pass --resume to verify and skip: {task.task_id}"
            )
        _, metadata = load_task_artifact(
            layout,
            task,
            expected_config_hash=config.resolved_hash,
            expected_code_hash=code_hash,
        )
        return {"task_id": task.task_id, "status": "skipped", "metadata": metadata}

    try:
        dtype = simulation_dtype(config.experiment.precision)
        labels = stream_labels(task)
        score_hash: str | None = None
        if task.kind == "dynamic":
            table = _lookup_table(task, tables)
            sensitivity_table = _lookup_sensitivity_table(task, tables)
            compiled = compile_dynamic_kernels(
                config,
                task,
                table,
                sensitivity_table,
                dtype=dtype,
                device=device,
            )
            arrays, execution_seconds = execute_dynamic_batches(
                compiled,
                config,
                task,
                device=device,
            )
            compile_seconds = compiled.main_compile_seconds + compiled.control_compile_seconds
            compile_breakdown = {
                "main": compiled.main_compile_seconds,
                "marginal_control": compiled.control_compile_seconds,
            }
            update_count = (2 * task.particles + task.control_particles) * task.steps
            throughput_definition = (
                "main plus paired hires plus exact-p_T control reverse particle-steps"
            )
            work_metadata = {"work_particle_steps": update_count}
            batches = max(
                task.particles // config.experiment.batch_size,
                task.control_particles // config.experiment.batch_size,
            )
            score_hash = table.table_hash
            score_sensitivity_hash = sensitivity_table.table_hash
        else:
            compiled_control = compile_control_kernel(
                config,
                task,
                dtype=dtype,
                device=device,
            )
            arrays, execution_seconds = execute_control_batches(
                compiled_control,
                config,
                task,
                device=device,
            )
            compile_seconds = compiled_control.compile_seconds
            compile_breakdown = {"direct_control": compiled_control.compile_seconds}
            update_count = task.particles
            throughput_definition = "direct control samples per second"
            work_metadata = {"work_samples": update_count}
            batches = task.particles // config.experiment.batch_size
            score_sensitivity_hash = None
        arrays.update(_scientific_scalar_arrays(config, task))
        throughput = update_count / max(execution_seconds, np.finfo(float).tiny)
        warnings = []
        if config.experiment.tier == "smoke":
            warnings.append("smoke tier: inferential decisions are disabled")
        metadata = {
            "backend": runtime.to_dict(),
            "batch_size": config.experiment.batch_size,
            "code_hash": code_hash,
            "compile_seconds": compile_seconds,
            "compile_seconds_breakdown": compile_breakdown,
            "config_hash": config.resolved_hash,
            "config_source_hash": config.source_hash,
            "execution_seconds_excluding_compilation": execution_seconds,
            "precision": config.experiment.precision,
            "random_stream_tokens": stream_manifest(labels),
            "random_batch_indices": {label: [0, max(0, batches - 1)] for label in labels},
            "score_table_hash": score_hash,
            "score_sensitivity_table_hash": score_sensitivity_hash,
            "seed": task.seed,
            "status": "complete",
            "throughput_particle_steps_per_second": throughput,
            "throughput_definition": throughput_definition,
            "warnings": warnings,
            **work_metadata,
        }
        if not config_source_is_unchanged(config):
            raise ArtifactError("configuration source changed during task execution")
        write_task_artifact(layout, task, arrays, metadata, resume=False)
        return {"task_id": task.task_id, "status": "complete", "metadata": metadata}
    except Exception as exc:
        write_failure(
            layout,
            task,
            {
                "config_hash": config.resolved_hash,
                "code_hash": code_hash,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


__all__ = ["ensure_score_tables", "run_task"]
