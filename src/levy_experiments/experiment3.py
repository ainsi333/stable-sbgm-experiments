"""JAX kernels and task runner for Experiment 3 only."""

from __future__ import annotations

import math
import os
import time
import traceback
from itertools import pairwise
from typing import Any

import numpy as np

from .config import Experiment3Config, ExperimentTask
from .devices import RuntimeDevice, simulation_dtype
from .errors import ArtifactError, NumericalError
from .integrators import popov_remainder, stable_ei_coefficients
from .random import named_key, stream_manifest
from .scores.discrete import discrete_fractional_score
from .scores.stable_table import (
    StableDensityTable,
    build_stable_density_table,
    load_stable_density_table,
    save_stable_density_table,
    score_table_path,
    validate_stable_density_table,
)
from .stable import sample_symmetric_stable_1d
from .storage import (
    RunLayout,
    compute_code_hash,
    config_source_is_unchanged,
    load_task_artifact,
    task_directory,
    write_failure,
    write_task_artifact,
)
from .theory import stationary_transition_coefficients


def validate_task_eta_consistency(task: ExperimentTask) -> None:
    """Keep valid Popov members distinct from the intentionally invalid control."""

    matched = math.isclose(task.eta, task.noise_eta, rel_tol=0.0, abs_tol=1e-15)
    if task.method == "popov_ei" and not matched:
        raise NumericalError("a valid Popov task requires identical drift and noise eta")
    if task.method == "hybrid_ei_invalid" and matched:
        raise NumericalError("the hybrid control must keep distinct drift and noise eta")


def output_forward_times(config: Experiment3Config) -> tuple[float, ...]:
    """Return recorded forward times, including the terminal early-stop time."""

    return (*config.nonstationary.checkpoints_forward, config.experiment.epsilon)


def checkpoint_step_indices(config: Experiment3Config, steps: int) -> tuple[int, ...]:
    """Map every recorded forward time to its exact post-update grid index."""

    step = (config.experiment.horizon - config.experiment.epsilon) / steps
    return tuple(
        round((config.experiment.horizon - forward_time) / step)
        for forward_time in output_forward_times(config)
    )


def nonstationary_crn_stream_label(config: Experiment3Config) -> str:
    """Return the resolution-independent stream shared by all EI refinements."""

    return f"nonstationary/nested-ei-fine-{max(config.nonstationary.steps)}/batch"


def nested_ei_noise_weights(
    *,
    alpha: float,
    beta: float,
    duration: float,
    task_steps: int,
    finest_steps: int,
    drift_eta: float,
    noise_eta: float,
) -> tuple[float, ...]:
    """Weights of shared fine OU innovations forming one exact coarse innovation.

    If ``m = finest_steps / task_steps``, the primitive fine-cell innovations
    are ordered from oldest to newest and weighted by ``q_f d_f**(m-1-j)``.
    Stability then gives ``sum(abs(w_j)**alpha) = q_h**alpha``, where ``q_h``
    is the exact EI noise scale on the coarse interval.
    """

    if task_steps <= 0 or finest_steps <= 0 or finest_steps % task_steps:
        raise NumericalError("nested EI requires task_steps to divide finest_steps")
    ratio = finest_steps // task_steps
    fine = stable_ei_coefficients(
        alpha=alpha,
        beta=beta,
        step=duration / finest_steps,
        drift_eta=drift_eta,
        noise_eta=noise_eta,
    )
    return tuple(
        fine.noise_scale * fine.decay ** (ratio - 1 - fine_index)
        for fine_index in range(ratio)
    )


def ensure_score_table(
    layout: RunLayout, config: Experiment3Config
) -> tuple[StableDensityTable, dict[str, float]]:
    """Build a configuration-addressed table once, or validate the existing one."""

    path = score_table_path(layout.run_dir, config)
    metadata_path = path.with_suffix(".json")
    if path.exists():
        table = load_stable_density_table(path)
        if (
            not math.isclose(table.alpha, config.experiment.alpha)
            or not math.isclose(table.z_max, config.score_table.z_max)
            or table.transformed_grid.size != config.score_table.points
        ):
            raise ArtifactError(
                "cached score table is incompatible with the resolved configuration"
            )
        validation = validate_stable_density_table(
            table,
            config.score_table.validation_points,
            config.score_table.validation_rtol,
            config.score_table.validation_atol,
        )
        return table, validation
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ArtifactError(
            f"score-table build lock exists ({lock}); "
            "on Slurm, wait for the table job before the array"
        ) from exc
    try:
        os.close(descriptor)
        started = time.perf_counter()
        table = build_stable_density_table(config)
        validation = validate_stable_density_table(
            table,
            config.score_table.validation_points,
            config.score_table.validation_rtol,
            config.score_table.validation_atol,
        )
        save_stable_density_table(table, path)
        from .storage import atomic_write_json

        atomic_write_json(
            metadata_path,
            {
                "alpha": table.alpha,
                "config_hash": config.resolved_hash,
                "construction_dtype": "float64",
                "elapsed_seconds": time.perf_counter() - started,
                "grid": "uniform-log1p-positive-half-line",
                "points": int(table.transformed_grid.size),
                "table_hash": table.table_hash,
                "tail": "three-term asymptotic expansion, numerically validated for alpha=1.5",
                "validation": validation,
                "z_max": table.z_max,
            },
        )
        return table, validation
    finally:
        lock.unlink(missing_ok=True)


def _compile_kernel(kernel, example_key, device):
    import jax

    with jax.default_device(device):
        jitted = jax.jit(kernel)
        started = time.perf_counter()
        compiled = jitted.lower(example_key).compile()
        compile_seconds = time.perf_counter() - started
    return compiled, compile_seconds


def _run_batched_kernel(
    compiled,
    *,
    seed: int,
    label: str,
    batches: int,
    device,
) -> tuple[list[tuple[np.ndarray, ...]], float]:
    import jax

    outputs: list[tuple[np.ndarray, ...]] = []
    started = time.perf_counter()
    with jax.default_device(device):
        for batch_index in range(batches):
            key = named_key(seed, label, batch_index)
            result = compiled(key)
            leaves = jax.tree.leaves(result)
            if leaves:
                leaves[0].block_until_ready()
            if isinstance(result, tuple):
                outputs.append(tuple(np.asarray(item) for item in result))
            else:
                outputs.append((np.asarray(result),))
    execution_seconds = time.perf_counter() - started
    return outputs, execution_seconds


def _stationary_kernel(config: Experiment3Config, task: ExperimentTask, dtype):
    import jax
    import jax.numpy as jnp

    retained, innovation = stationary_transition_coefficients(
        alpha=config.experiment.alpha,
        beta=config.experiment.beta,
        eta=task.eta,
        lag=task.horizon,
    )
    retained_value = jnp.asarray(retained, dtype=dtype)
    innovation_value = jnp.asarray(innovation, dtype=dtype)
    batch_size = config.stationary.batch_size

    def kernel(key):
        key_initial, key_innovation = jax.random.split(key)
        initial = sample_symmetric_stable_1d(
            key_initial, config.experiment.alpha, (batch_size,), dtype
        )
        noise = sample_symmetric_stable_1d(
            key_innovation, config.experiment.alpha, (batch_size,), dtype
        )
        terminal = retained_value * initial + innovation_value * noise
        return initial, terminal

    return kernel


def _exact_reference_kernel(config: Experiment3Config, dtype):
    import jax
    import jax.numpy as jnp

    ns = config.nonstationary
    alpha = config.experiment.alpha
    beta = config.experiment.beta
    batch_size = ns.batch_size
    output_times_descending = output_forward_times(config)
    times_ascending = tuple(reversed(output_times_descending))
    atoms = jnp.asarray(ns.atoms, dtype=dtype)
    log_weights = jnp.log(jnp.asarray(ns.weights, dtype=dtype))

    def kernel(key):
        keys = jax.random.split(key, len(times_ascending) + 2)
        indices = jax.random.categorical(keys[0], log_weights, shape=(batch_size,))
        first_time = times_ascending[0]
        a = jnp.asarray(math.exp(-beta * first_time / alpha), dtype=dtype)
        gamma = jnp.asarray((-math.expm1(-beta * first_time)) ** (1.0 / alpha), dtype=dtype)
        state = a * atoms[indices] + gamma * sample_symmetric_stable_1d(
            keys[1], alpha, (batch_size,), dtype
        )
        states = [state]
        for transition_index, (earlier, later) in enumerate(pairwise(times_ascending), start=2):
            delta = later - earlier
            retained = jnp.asarray(math.exp(-beta * delta / alpha), dtype=dtype)
            innovation = jnp.asarray((-math.expm1(-beta * delta)) ** (1.0 / alpha), dtype=dtype)
            state = retained * state + innovation * sample_symmetric_stable_1d(
                keys[transition_index], alpha, (batch_size,), dtype
            )
            states.append(state)
        return jnp.stack(tuple(reversed(states)), axis=0)

    return kernel


def _nonstationary_kernel(
    config: Experiment3Config,
    task: ExperimentTask,
    table: StableDensityTable,
    dtype,
):
    import jax
    import jax.numpy as jnp

    exp = config.experiment
    ns = config.nonstationary
    alpha = exp.alpha
    beta = exp.beta
    duration = exp.horizon - exp.epsilon
    step = duration / task.steps
    finest_steps = max(ns.steps)
    refinement_ratio = finest_steps // task.steps
    coefficients = stable_ei_coefficients(
        alpha=alpha,
        beta=beta,
        step=step,
        drift_eta=task.eta,
        noise_eta=task.noise_eta,
    )
    output_times = output_forward_times(config)
    checkpoint_steps = checkpoint_step_indices(config, task.steps)
    checkpoint_indices = jnp.asarray(checkpoint_steps, dtype=jnp.int32)
    batch_size = ns.batch_size
    atoms = jnp.asarray(ns.atoms, dtype=dtype)
    weights = jnp.asarray(ns.weights, dtype=dtype)
    log_weights = jnp.log(weights)
    decay = jnp.asarray(coefficients.decay, dtype=dtype)
    remainder_multiplier = jnp.asarray(coefficients.remainder_multiplier, dtype=dtype)
    nested_noise_weights = jnp.asarray(
        nested_ei_noise_weights(
            alpha=alpha,
            beta=beta,
            duration=duration,
            task_steps=task.steps,
            finest_steps=finest_steps,
            drift_eta=task.eta,
            noise_eta=task.noise_eta,
        ),
        dtype=dtype,
    )
    horizon = jnp.asarray(exp.horizon, dtype=dtype)
    step_value = jnp.asarray(step, dtype=dtype)

    def kernel(key):
        key_atom, key_initial, key_steps = jax.random.split(key, 3)
        indices = jax.random.categorical(key_atom, log_weights, shape=(batch_size,))
        a_horizon = jnp.asarray(math.exp(-beta * exp.horizon / alpha), dtype=dtype)
        gamma_horizon = jnp.asarray(
            (-math.expm1(-beta * exp.horizon)) ** (1.0 / alpha), dtype=dtype
        )
        state = a_horizon * atoms[indices] + gamma_horizon * sample_symmetric_stable_1d(
            key_initial, alpha, (batch_size,), dtype
        )
        checkpoints = jnp.zeros((len(output_times), batch_size), dtype=dtype)

        def body(step_index, carry):
            current, recorded = carry
            forward_time = horizon - step_index.astype(dtype) * step_value
            score = discrete_fractional_score(
                current,
                forward_time,
                alpha=alpha,
                beta=beta,
                atoms=atoms,
                weights=weights,
                table=table,
                dtype=dtype,
            )
            remainder = popov_remainder(current, score, alpha=alpha, beta=beta, eta=task.eta)
            first_primitive_index = step_index * refinement_ratio

            def add_fine_innovation(fine_index, accumulated):
                primitive_index = first_primitive_index + fine_index
                primitive_key = jax.random.fold_in(key_steps, primitive_index)
                primitive = sample_symmetric_stable_1d(
                    primitive_key, alpha, (batch_size,), dtype
                )
                return accumulated + nested_noise_weights[fine_index] * primitive

            innovation = jax.lax.fori_loop(
                0,
                refinement_ratio,
                add_fine_innovation,
                jnp.zeros((batch_size,), dtype=dtype),
            )
            updated = decay * current + remainder_multiplier * remainder + innovation
            matches = checkpoint_indices == (step_index + 1)
            recorded = jnp.where(matches[:, None], updated[None, :], recorded)
            return updated, recorded

        _, recorded = jax.lax.fori_loop(0, task.steps, body, (state, checkpoints))
        return recorded

    return kernel


def _common_metadata(
    config: Experiment3Config,
    task: ExperimentTask,
    runtime: RuntimeDevice,
    *,
    compile_seconds: float,
    execution_seconds: float,
    throughput: float,
    batch_size: int,
    warnings: list[str],
    score_table_hash: str | None,
    stream_labels: tuple[str, ...],
    batches: int,
) -> dict[str, Any]:
    return {
        "backend": runtime.to_dict(),
        "batch_size": batch_size,
        "code_hash": compute_code_hash(),
        "compile_seconds": compile_seconds,
        "config_hash": config.resolved_hash,
        "config_source_hash": config.source_hash,
        "execution_seconds_excluding_compilation": execution_seconds,
        "precision": config.experiment.precision,
        "random_stream_tokens": stream_manifest(stream_labels),
        "random_batch_indices": {label: [0, batches - 1] for label in stream_labels},
        "score_table_hash": score_table_hash,
        "seed": task.seed,
        "status": "complete",
        "throughput_particle_steps_per_second": throughput,
        "warnings": warnings,
    }


def run_task(
    config: Experiment3Config,
    task: ExperimentTask,
    layout: RunLayout,
    *,
    device,
    runtime: RuntimeDevice,
    table: StableDensityTable | None,
    resume: bool,
) -> dict[str, Any]:
    """Compile, execute, validate, and atomically commit one independent task."""

    validate_task_eta_consistency(task)
    current_code_hash = compute_code_hash()
    destination = task_directory(layout, task)
    if destination.exists():
        if not resume:
            raise ArtifactError(
                f"Task already exists; pass --resume to verify and skip: {task.task_id}"
            )
        _, existing_metadata = load_task_artifact(
            layout,
            task,
            expected_config_hash=config.resolved_hash,
            expected_code_hash=current_code_hash,
        )
        return {"task_id": task.task_id, "status": "skipped", "metadata": existing_metadata}
    dtype = simulation_dtype(config.experiment.precision)
    try:
        if task.target == "stationary_sas":
            kernel = _stationary_kernel(config, task, dtype)
            batch_size = config.stationary.batch_size
            batches = config.stationary.particles // batch_size
            label = f"stationary/{task.horizon:.17g}/batch"
            example_key = named_key(task.seed, label, 0)
            compiled, compile_seconds = _compile_kernel(kernel, example_key, device)
            outputs, execution_seconds = _run_batched_kernel(
                compiled, seed=task.seed, label=label, batches=batches, device=device
            )
            initial = np.concatenate([output[0] for output in outputs])
            terminal = np.concatenate([output[1] for output in outputs])
            arrays = {
                "initial": initial,
                "lag": np.asarray(task.horizon, dtype=np.float64),
                "terminal": terminal,
            }
            particle_steps = config.stationary.particles
            metadata_work_counts = {
                "work_particle_steps": particle_steps,
                "work_particle_steps_definition": "exact_transition_pairs",
            }
            table_hash = None
            stream_labels = (label,)
        elif task.method == "exact_reference":
            kernel = _exact_reference_kernel(config, dtype)
            batch_size = config.nonstationary.batch_size
            batches = config.nonstationary.particles // batch_size
            label_a = "nonstationary_reference_a/batch"
            example_key = named_key(task.seed, label_a, 0)
            compiled, compile_seconds = _compile_kernel(kernel, example_key, device)
            outputs_a, execution_a = _run_batched_kernel(
                compiled, seed=task.seed, label=label_a, batches=batches, device=device
            )
            outputs_b, execution_b = _run_batched_kernel(
                compiled,
                seed=task.seed,
                label="nonstationary_reference_b/batch",
                batches=batches,
                device=device,
            )
            arrays = {
                "forward_times": np.asarray(
                    output_forward_times(config),
                    dtype=np.float64,
                ),
                "reference_a": np.concatenate([output[0] for output in outputs_a], axis=1),
                "reference_b": np.concatenate([output[0] for output in outputs_b], axis=1),
            }
            execution_seconds = execution_a + execution_b
            particle_steps = 2 * config.nonstationary.particles * len(arrays["forward_times"])
            metadata_work_counts = {
                "work_particle_steps": particle_steps,
                "work_particle_steps_definition": "exact_reference_checkpoint_states",
            }
            table_hash = None
            stream_labels = (label_a, "nonstationary_reference_b/batch")
        else:
            if table is None:
                raise ArtifactError("nonstationary EI tasks require the validated score table")
            kernel = _nonstationary_kernel(config, task, table, dtype)
            batch_size = config.nonstationary.batch_size
            batches = config.nonstationary.particles // batch_size
            label = nonstationary_crn_stream_label(config)
            example_key = named_key(task.seed, label, 0)
            compiled, compile_seconds = _compile_kernel(kernel, example_key, device)
            outputs, execution_seconds = _run_batched_kernel(
                compiled, seed=task.seed, label=label, batches=batches, device=device
            )
            arrays = {
                "forward_times": np.asarray(
                    output_forward_times(config),
                    dtype=np.float64,
                ),
                "samples": np.concatenate([output[0] for output in outputs], axis=1),
            }
            integrator_particle_steps = config.nonstationary.particles * task.steps
            primitive_particle_steps = config.nonstationary.particles * max(
                config.nonstationary.steps
            )
            particle_steps = integrator_particle_steps + primitive_particle_steps
            metadata_work_counts = {
                "integrator_particle_steps": integrator_particle_steps,
                "primitive_particle_steps": primitive_particle_steps,
                "work_particle_steps": particle_steps,
                "work_particle_steps_definition": (
                    "integrator_updates_plus_backend_stable_primitives"
                ),
            }
            table_hash = table.table_hash
            stream_labels = (label,)
            metadata_crn = {
                "finest_steps": max(config.nonstationary.steps),
                "primitive_innovation_scheme": "nested_exact_ou_convolution",
                "refinement_ratio": max(config.nonstationary.steps) // task.steps,
                "shared_initials_across_steps": True,
                "shared_primitive_innovations_across_steps": True,
            }
        for name, values in arrays.items():
            if np.issubdtype(values.dtype, np.number) and np.any(~np.isfinite(values)):
                raise NumericalError(
                    f"non-finite values in task={task.task_id}, seed={task.seed}, array={name}"
                )
        throughput = particle_steps / max(execution_seconds, np.finfo(float).tiny)
        warnings = (
            ["hybrid_drift_noise_mismatch: invalid control excluded from valid-method claims"]
            if task.method == "hybrid_ei_invalid"
            else []
        )
        metadata = _common_metadata(
            config,
            task,
            runtime,
            compile_seconds=compile_seconds,
            execution_seconds=execution_seconds,
            throughput=throughput,
            batch_size=batch_size,
            warnings=warnings,
            score_table_hash=table_hash,
            stream_labels=stream_labels,
            batches=batches,
        )
        metadata.update(metadata_work_counts)
        if task.target == "three_atoms" and task.method not in {"exact_reference"}:
            metadata["refinement_crn"] = metadata_crn
        if not config_source_is_unchanged(config):
            raise ArtifactError("configuration source changed during task execution")
        write_task_artifact(layout, task, arrays, metadata)
        return {"task_id": task.task_id, "status": "complete", "metadata": metadata}
    except Exception as exc:
        write_failure(
            layout,
            task,
            {
                "config_hash": config.resolved_hash,
                "code_hash": current_code_hash,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
