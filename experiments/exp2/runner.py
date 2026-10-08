"""Audited task execution and provenance for Experiment 2."""

from __future__ import annotations

import dataclasses
import traceback
from typing import Any

import numpy as np

from levy_experiments.devices import RuntimeDevice, simulation_dtype
from levy_experiments.errors import ArtifactError
from levy_experiments.random import stream_manifest

from .config import Experiment2Config, Experiment2Task
from .score_tables import ScoreTable, ensure_score_table
from .simulation import (
    compile_task_kernel,
    execute_task_batches,
    refinement_coupling_steps,
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


def ensure_score_tables(
    layout: RunLayout, config: Experiment2Config, *, resume: bool
) -> dict[str, ScoreTable]:
    """Build or verify the two model-specific tables before array jobs start."""

    hires_config = dataclasses.replace(config, score_table=config.score_table_hires)
    tables: dict[str, ScoreTable] = {}
    for model in ("stable", "vp"):
        main = ensure_score_table(layout.run_dir, config, model, resume=resume)
        hires = ensure_score_table(layout.run_dir, hires_config, model, resume=resume)
        if main.table_hash == hires.table_hash:
            raise ArtifactError(f"{model} main and high-resolution score tables are identical")
        tables[model] = main
        tables[f"{model}_hires"] = hires
    return tables


def _scientific_scalar_arrays(
    config: Experiment2Config, task: Experiment2Task
) -> dict[str, np.ndarray]:
    eta = config.stable.eta if task.model == "stable" else 1.0
    return {
        "model": np.asarray(task.model),
        "steps": np.asarray(task.steps, dtype=np.int64),
        "seed": np.asarray(task.seed, dtype=np.int64),
        "T": np.asarray(task.horizon, dtype=np.float64),
        "epsilon": np.asarray(config.experiment.epsilon, dtype=np.float64),
        "alpha": np.asarray(config.stable.alpha, dtype=np.float64),
        "eta": np.asarray(eta, dtype=np.float64),
        "beta": np.asarray(config.experiment.beta, dtype=np.float64),
        "wasserstein_orders": np.asarray(
            config.analysis.wasserstein_orders, dtype=np.float64
        ),
    }


def run_task(
    config: Experiment2Config,
    task: Experiment2Task,
    layout: RunLayout,
    *,
    device,
    runtime: RuntimeDevice,
    table: ScoreTable,
    hires_table: ScoreTable,
    resume: bool,
) -> dict[str, Any]:
    """Compile, execute, validate, and atomically commit one task."""

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
    if table.model != task.model:
        raise ArtifactError(
            f"task {task.task_id} received the {table.model} score instead of {task.model}"
        )
    if hires_table.model != task.model:
        raise ArtifactError(
            f"task {task.task_id} received the {hires_table.model} high-resolution "
            f"score instead of {task.model}"
        )
    if hires_table.table_hash == table.table_hash:
        raise ArtifactError(
            f"task {task.task_id} requires independent main and high-resolution tables"
        )

    try:
        dtype = simulation_dtype(config.experiment.precision)
        compiled, compile_seconds = compile_task_kernel(
            config, task, table, hires_table, dtype=dtype, device=device
        )
        arrays, execution_seconds = execute_task_batches(
            compiled, config, task, device=device
        )
        arrays.update(_scientific_scalar_arrays(config, task))
        primitive_steps = refinement_coupling_steps(config, task)
        integrator_particle_steps = 3 * config.experiment.particles * task.steps
        extra_primitive_particle_steps = (
            2 * config.experiment.particles * (primitive_steps - task.steps)
        )
        work_particle_steps = integrator_particle_steps + extra_primitive_particle_steps
        throughput = work_particle_steps / max(
            execution_seconds, np.finfo(float).tiny
        )
        labels = stream_labels(task, coupling_steps=primitive_steps)
        batches = config.experiment.particles // config.experiment.batch_size
        metadata = {
            "backend": runtime.to_dict(),
            "batch_size": config.experiment.batch_size,
            "code_hash": code_hash,
            "compile_seconds": compile_seconds,
            "config_hash": config.resolved_hash,
            "config_source_hash": config.source_hash,
            "execution_seconds_excluding_compilation": execution_seconds,
            "extra_primitive_particle_steps": extra_primitive_particle_steps,
            "integrator_particle_steps": integrator_particle_steps,
            "precision": config.experiment.precision,
            "primitive_coupling_steps": primitive_steps,
            "random_stream_tokens": stream_manifest(labels),
            "random_batch_indices": {label: [0, batches - 1] for label in labels},
            "score_table_hash": table.table_hash,
            "score_table_hires_hash": hires_table.table_hash,
            "score_table_sensitivity_coupling": (
                "same_exact_pT_initialization_and_same_primitive_noise"
            ),
            "seed": task.seed,
            "status": "complete",
            "throughput_particle_steps_per_second": throughput,
            "throughput_definition": (
                "three integrated arms times particles times N plus two shared-noise "
                "streams times particles times (primitive_N-N)"
            ),
            "work_particle_steps": work_particle_steps,
            "warnings": [],
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
