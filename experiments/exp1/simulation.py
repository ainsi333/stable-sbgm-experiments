"""JAX simulation kernels for target-specific tail coverage.

Dynamic tasks propagate only terminal states for the theorem arm and prescribed
checkpoints for the exact-``p_T`` control.  Refinement levels use a common fine
OU innovation stream: every coarse EI noise is the exact aggregation of the
same primitive sub-increments used by the finest configured level.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from levy_experiments.errors import ConfigurationError, NumericalError
from levy_experiments.integrators import popov_remainder, stable_ei_coefficients
from levy_experiments.random import named_key
from levy_experiments.stable import sample_symmetric_stable_1d

from .config import Experiment1Config, Experiment1Task, checkpoint_step_indices
from .score_tables import ScoreTable, interpolate_score_jax
from .theory import sample_exact_forward_marginal, sample_stationary_reference


@dataclass(frozen=True)
class VPEICoefficients:
    """Frozen-score exponential-integrator coefficients for reverse VP."""

    decay: float
    remainder_multiplier: float
    noise_scale: float


@dataclass(frozen=True)
class CompiledDynamicKernels:
    """Separately compiled theorem and marginal-control kernels."""

    main: Any
    control: Any
    main_compile_seconds: float
    control_compile_seconds: float


@dataclass(frozen=True)
class CompiledControlKernel:
    """Compiled direct tail-control sampler."""

    sample: Any
    compile_seconds: float


def vp_ei_coefficients(*, beta: float, step: float) -> VPEICoefficients:
    """Integrate the VP linear drift and Brownian convolution exactly."""

    if not math.isfinite(beta) or not math.isfinite(step) or beta <= 0.0 or step <= 0.0:
        raise ConfigurationError("VP EI coefficients require finite positive beta and h")
    rate = beta / 2.0
    return VPEICoefficients(
        decay=math.exp(-rate * step),
        remainder_multiplier=-math.expm1(-rate * step) / rate,
        noise_scale=math.sqrt(-math.expm1(-beta * step)),
    )


def output_forward_times(
    *, horizon: float, epsilon: float, checkpoint_fractions: tuple[float, ...]
) -> tuple[float, ...]:
    """Return checkpoint forward times in increasing order."""

    duration = horizon - epsilon
    descending = tuple(horizon - fraction * duration for fraction in checkpoint_fractions)
    return tuple(reversed(descending))


def stream_labels(task: Experiment1Task) -> tuple[str, ...]:
    """Semantic random streams, deliberately independent of the tested ``N``."""

    if task.kind == "control":
        return (f"exp1/control/{task.model}/sample",)
    if task.nu is None:
        raise ConfigurationError("a dynamic task requires a Student index")
    prefix = f"exp1/{task.model}/nu={task.nu:.17g}/coupling_steps={task.coupling_steps}"
    return (
        f"{prefix}/stationary_initialization",
        f"{prefix}/stationary_fine_dynamics",
        f"{prefix}/exact_p_T_initialization",
        f"{prefix}/exact_p_T_fine_dynamics",
        f"{prefix}/exact_reference_a",
        f"{prefix}/exact_reference_b",
        f"{prefix}/exact_reference_c",
    )


def _dynamic_constants(config: Experiment1Config, task: Experiment1Task):
    if task.kind != "dynamic" or task.nu is None or task.steps <= 0:
        raise ConfigurationError("dynamic kernel requires a valid dynamic task")
    if task.coupling_steps % task.steps:
        raise ConfigurationError("coupling_steps must be divisible by task steps")
    duration = config.experiment.horizon - config.experiment.epsilon
    step = duration / task.steps
    fine_step = duration / task.coupling_steps
    ratio = task.coupling_steps // task.steps
    if task.model == "stable":
        coarse = stable_ei_coefficients(
            alpha=config.stable.alpha,
            beta=config.experiment.beta,
            step=step,
            drift_eta=config.stable.eta,
        )
        fine = stable_ei_coefficients(
            alpha=config.stable.alpha,
            beta=config.experiment.beta,
            step=fine_step,
            drift_eta=config.stable.eta,
        )
    elif task.model == "vp":
        coarse = vp_ei_coefficients(beta=config.experiment.beta, step=step)
        fine = vp_ei_coefficients(beta=config.experiment.beta, step=fine_step)
    else:
        raise ConfigurationError(f"unsupported dynamic model: {task.model}")
    return duration, step, fine_step, ratio, coarse, fine


def _aggregated_noise(
    base_key,
    coarse_index,
    *,
    task: Experiment1Task,
    config: Experiment1Config,
    ratio: int,
    fine_decay,
    fine_noise_scale,
    batch_size: int,
    dtype,
):
    """Aggregate shared fine OU innovations into one exact coarse innovation."""

    import jax
    import jax.numpy as jnp

    weights = fine_noise_scale * jnp.power(
        fine_decay,
        jnp.arange(ratio - 1, -1, -1, dtype=dtype),
    )
    start = coarse_index * ratio

    def add_one(index, accumulator):
        key = jax.random.fold_in(base_key, start + index)
        if task.model == "stable":
            innovation = sample_symmetric_stable_1d(
                key,
                config.stable.alpha,
                (batch_size,),
                dtype,
            )
        else:
            innovation = jax.random.normal(key, shape=(batch_size,), dtype=dtype)
        return accumulator + weights[index] * innovation

    return jax.lax.fori_loop(
        0,
        ratio,
        add_one,
        jnp.zeros((batch_size,), dtype=dtype),
    )


def _advance_state(
    state,
    noise,
    forward_time,
    *,
    config: Experiment1Config,
    task: Experiment1Task,
    table: ScoreTable,
    dtype,
    coarse_decay,
    coarse_remainder,
):
    score = interpolate_score_jax(state, forward_time, table=table, dtype=dtype)
    if task.model == "stable":
        remainder = popov_remainder(
            state,
            score,
            alpha=config.stable.alpha,
            beta=config.experiment.beta,
            eta=config.stable.eta,
        )
    else:
        remainder = config.experiment.beta * (state + score)
    return coarse_decay * state + coarse_remainder * remainder + noise


def _update_state(
    state,
    noise_key,
    step_index,
    *,
    config: Experiment1Config,
    task: Experiment1Task,
    table: ScoreTable,
    dtype,
    step_value,
    coarse_decay,
    coarse_remainder,
    fine_decay,
    fine_noise_scale,
    ratio: int,
):
    import jax.numpy as jnp

    forward_time = (
        jnp.asarray(config.experiment.horizon, dtype=dtype) - step_index.astype(dtype) * step_value
    )
    noise = _aggregated_noise(
        noise_key,
        step_index,
        task=task,
        config=config,
        ratio=ratio,
        fine_decay=fine_decay,
        fine_noise_scale=fine_noise_scale,
        batch_size=config.experiment.batch_size,
        dtype=dtype,
    )
    return _advance_state(
        state,
        noise,
        forward_time,
        config=config,
        task=task,
        table=table,
        dtype=dtype,
        coarse_decay=coarse_decay,
        coarse_remainder=coarse_remainder,
    )


def _main_kernel(
    config: Experiment1Config,
    task: Experiment1Task,
    table: ScoreTable,
    sensitivity_table: ScoreTable,
    dtype,
):
    import jax
    import jax.numpy as jnp

    _, step, _, ratio, coarse, fine = _dynamic_constants(config, task)
    step_value = jnp.asarray(step, dtype=dtype)
    coarse_decay = jnp.asarray(coarse.decay, dtype=dtype)
    coarse_remainder = jnp.asarray(coarse.remainder_multiplier, dtype=dtype)
    fine_decay = jnp.asarray(fine.decay, dtype=dtype)
    fine_noise_scale = jnp.asarray(fine.noise_scale, dtype=dtype)
    batch_size = config.experiment.batch_size
    checkpoint_indices = checkpoint_step_indices(config, task)
    index_array = jnp.asarray(checkpoint_indices, dtype=jnp.int32)
    natural_times = tuple(
        config.experiment.horizon
        - fraction * (config.experiment.horizon - config.experiment.epsilon)
        for fraction in config.design.checkpoint_fractions
    )
    increasing_times = jnp.asarray(tuple(reversed(natural_times)), dtype=dtype)

    def kernel(keys):
        initial = sample_stationary_reference(
            keys[0],
            task.model,
            (batch_size,),
            dtype,
            alpha=config.stable.alpha,
        )

        recorded = jnp.zeros((len(checkpoint_indices),), dtype=dtype)

        def body(index, carry):
            main_state, hires_state, stored = carry
            forward_time = (
                jnp.asarray(config.experiment.horizon, dtype=dtype)
                - index.astype(dtype) * step_value
            )
            noise = _aggregated_noise(
                keys[1],
                index,
                config=config,
                task=task,
                ratio=ratio,
                fine_decay=fine_decay,
                fine_noise_scale=fine_noise_scale,
                batch_size=batch_size,
                dtype=dtype,
            )
            main_updated = _advance_state(
                main_state,
                noise,
                forward_time,
                config=config,
                task=task,
                table=table,
                dtype=dtype,
                coarse_decay=coarse_decay,
                coarse_remainder=coarse_remainder,
            )
            hires_updated = _advance_state(
                hires_state,
                noise,
                forward_time,
                config=config,
                task=task,
                table=sensitivity_table,
                dtype=dtype,
                coarse_decay=coarse_decay,
                coarse_remainder=coarse_remainder,
            )
            matches = index_array == (index + 1)
            stored = jnp.where(matches, jnp.mean(jnp.abs(main_updated - hires_updated)), stored)
            return main_updated, hires_updated, stored

        main_terminal, hires_terminal, differences_descending = jax.lax.fori_loop(
            0, task.steps, body, (initial, initial, recorded)
        )
        return (
            main_terminal,
            hires_terminal,
            increasing_times,
            differences_descending[::-1],
        )

    return kernel


def _control_kernel(
    config: Experiment1Config,
    task: Experiment1Task,
    table: ScoreTable,
    dtype,
):
    import jax
    import jax.numpy as jnp

    _, step, _, ratio, coarse, fine = _dynamic_constants(config, task)
    checkpoint_indices = checkpoint_step_indices(config, task)
    index_array = jnp.asarray(checkpoint_indices, dtype=jnp.int32)
    natural_times = tuple(
        config.experiment.horizon
        - fraction * (config.experiment.horizon - config.experiment.epsilon)
        for fraction in config.design.checkpoint_fractions
    )
    increasing_times = jnp.asarray(tuple(reversed(natural_times)), dtype=dtype)
    step_value = jnp.asarray(step, dtype=dtype)
    coarse_decay = jnp.asarray(coarse.decay, dtype=dtype)
    coarse_remainder = jnp.asarray(coarse.remainder_multiplier, dtype=dtype)
    fine_decay = jnp.asarray(fine.decay, dtype=dtype)
    fine_noise_scale = jnp.asarray(fine.noise_scale, dtype=dtype)
    batch_size = config.experiment.batch_size

    def exact_stack(base_key):
        samples = tuple(
            sample_exact_forward_marginal(
                jax.random.fold_in(base_key, index),
                task.model,
                float(task.nu),
                float(time_value),
                (batch_size,),
                dtype,
                alpha=config.stable.alpha,
                beta=config.experiment.beta,
            )
            for index, time_value in enumerate(natural_times)
        )
        return jnp.stack(tuple(reversed(samples)), axis=0)

    def kernel(keys):
        initial = sample_exact_forward_marginal(
            keys[0],
            task.model,
            float(task.nu),
            config.experiment.horizon,
            (batch_size,),
            dtype,
            alpha=config.stable.alpha,
            beta=config.experiment.beta,
        )
        recorded = jnp.zeros((len(checkpoint_indices), batch_size), dtype=dtype).at[0].set(initial)

        def body(index, carry):
            state, stored = carry
            updated = _update_state(
                state,
                keys[1],
                index,
                config=config,
                task=task,
                table=table,
                dtype=dtype,
                step_value=step_value,
                coarse_decay=coarse_decay,
                coarse_remainder=coarse_remainder,
                fine_decay=fine_decay,
                fine_noise_scale=fine_noise_scale,
                ratio=ratio,
            )
            matches = index_array == (index + 1)
            stored = jnp.where(matches[:, None], updated[None, :], stored)
            return updated, stored

        _, numerical_descending = jax.lax.fori_loop(
            0,
            task.steps,
            body,
            (initial, recorded),
        )
        return (
            increasing_times,
            numerical_descending[::-1],
            exact_stack(keys[2]),
            exact_stack(keys[3]),
            exact_stack(keys[4]),
        )

    return kernel


def compile_dynamic_kernels(
    config: Experiment1Config,
    task: Experiment1Task,
    table: ScoreTable,
    sensitivity_table: ScoreTable,
    *,
    dtype,
    device,
) -> CompiledDynamicKernels:
    """Compile theorem and exact-marginal arms, timing each separately."""

    import jax
    import jax.numpy as jnp

    labels = stream_labels(task)
    main_keys = jnp.stack((named_key(task.seed, labels[0], 0), named_key(task.seed, labels[1], 0)))
    control_keys = jnp.stack(tuple(named_key(task.seed, label, 0) for label in labels[2:]), axis=0)
    with jax.default_device(device):
        started = time.perf_counter()
        main = (
            jax.jit(_main_kernel(config, task, table, sensitivity_table, dtype))
            .lower(main_keys)
            .compile()
        )
        main_seconds = time.perf_counter() - started
        started = time.perf_counter()
        control = jax.jit(_control_kernel(config, task, table, dtype)).lower(control_keys).compile()
        control_seconds = time.perf_counter() - started
    return CompiledDynamicKernels(main, control, main_seconds, control_seconds)


def _collect_dynamic(
    compiled: CompiledDynamicKernels,
    config: Experiment1Config,
    task: Experiment1Task,
    *,
    device,
) -> tuple[dict[str, np.ndarray], float]:
    import jax
    import jax.numpy as jnp

    labels = stream_labels(task)
    main_batches = task.particles // config.experiment.batch_size
    control_batches = task.control_particles // config.experiment.batch_size
    terminal: list[np.ndarray] = []
    sensitivity_terminal: list[np.ndarray] = []
    sensitivity_differences: list[np.ndarray] = []
    sensitivity_times: np.ndarray | None = None
    control_results: list[tuple[np.ndarray, ...]] = []
    started = time.perf_counter()
    with jax.default_device(device):
        for batch_index in range(main_batches):
            keys = jnp.stack(
                (
                    named_key(task.seed, labels[0], batch_index),
                    named_key(task.seed, labels[1], batch_index),
                )
            )
            result = compiled.main(keys)
            jax.block_until_ready(result)
            main_result, hires_result, batch_times, batch_differences = (
                np.asarray(item) for item in result
            )
            if sensitivity_times is None:
                sensitivity_times = batch_times
            elif not np.array_equal(sensitivity_times, batch_times):
                raise NumericalError("score-sensitivity batches disagree on checkpoint times")
            terminal.append(main_result)
            sensitivity_terminal.append(hires_result)
            sensitivity_differences.append(batch_differences)
        for batch_index in range(control_batches):
            keys = jnp.stack(
                tuple(named_key(task.seed, label, batch_index) for label in labels[2:]),
                axis=0,
            )
            result = compiled.control(keys)
            jax.block_until_ready(result)
            control_results.append(tuple(np.asarray(item) for item in result))
    elapsed = time.perf_counter() - started
    if not control_results:
        raise NumericalError(f"dynamic task has no marginal-control batch: {task.task_id}")
    if sensitivity_times is None or not sensitivity_differences:
        raise NumericalError(f"dynamic task has no score-sensitivity batch: {task.task_id}")
    forward_times = control_results[0][0]
    if any(not np.array_equal(batch[0], forward_times) for batch in control_results[1:]):
        raise NumericalError("marginal-control batches disagree on checkpoint times")
    arrays = {
        "terminal_sample": np.concatenate(terminal),
        "score_hires_terminal": np.concatenate(sensitivity_terminal),
        "score_sensitivity_forward_times": np.asarray(sensitivity_times, dtype=np.float64),
        "score_sensitivity_mean_abs": np.mean(
            np.stack(sensitivity_differences, axis=0), axis=0, dtype=np.float64
        ),
        "control_forward_times": np.asarray(forward_times, dtype=np.float64),
        "control_numerical": np.concatenate([batch[1] for batch in control_results], axis=1),
        "control_exact_a": np.concatenate([batch[2] for batch in control_results], axis=1),
        "control_exact_b": np.concatenate([batch[3] for batch in control_results], axis=1),
        "control_exact_c": np.concatenate([batch[4] for batch in control_results], axis=1),
    }
    _require_finite(arrays, task)
    return arrays, elapsed


def _open_uniform(key, shape: tuple[int, ...], dtype):
    import jax.numpy as jnp
    from jax import random

    raw = random.uniform(key, shape=shape, minval=0.0, maxval=1.0, dtype=dtype)
    half_eps = jnp.asarray(jnp.finfo(dtype).eps / 2.0, dtype=dtype)
    return (raw + half_eps) / (jnp.asarray(1.0, dtype=dtype) + 2.0 * half_eps)


def _direct_control_kernel(config: Experiment1Config, task: Experiment1Task, dtype):
    import jax
    import jax.numpy as jnp

    batch_size = config.experiment.batch_size

    def kernel(key):
        if task.model == "sas":
            return sample_symmetric_stable_1d(
                key,
                config.controls.sas_alpha,
                (batch_size,),
                dtype,
            )
        if task.model == "gaussian":
            return jnp.asarray(config.controls.gaussian_mean, dtype=dtype) + jnp.asarray(
                config.controls.gaussian_std, dtype=dtype
            ) * jax.random.normal(key, shape=(batch_size,), dtype=dtype)
        if task.model == "pareto":
            key_radius, key_sign = jax.random.split(key)
            radial = jnp.asarray(config.controls.pareto_minimum, dtype=dtype) * jnp.power(
                _open_uniform(key_radius, (batch_size,), dtype),
                -1.0 / config.controls.pareto_tail_index,
            )
            signs = jnp.where(
                jax.random.bernoulli(key_sign, shape=(batch_size,)),
                jnp.asarray(1.0, dtype=dtype),
                jnp.asarray(-1.0, dtype=dtype),
            )
            return signs * radial if config.controls.pareto_symmetric else radial
        raise ConfigurationError(f"unsupported direct control: {task.model}")

    return kernel


def compile_control_kernel(
    config: Experiment1Config,
    task: Experiment1Task,
    *,
    dtype,
    device,
) -> CompiledControlKernel:
    import jax

    label = stream_labels(task)[0]
    example = named_key(task.seed, label, 0)
    with jax.default_device(device):
        started = time.perf_counter()
        compiled = jax.jit(_direct_control_kernel(config, task, dtype)).lower(example).compile()
        seconds = time.perf_counter() - started
    return CompiledControlKernel(compiled, seconds)


def execute_control_batches(
    compiled: CompiledControlKernel,
    config: Experiment1Config,
    task: Experiment1Task,
    *,
    device,
) -> tuple[dict[str, np.ndarray], float]:
    import jax

    label = stream_labels(task)[0]
    batches = task.particles // config.experiment.batch_size
    collected: list[np.ndarray] = []
    started = time.perf_counter()
    with jax.default_device(device):
        for batch_index in range(batches):
            result = compiled.sample(named_key(task.seed, label, batch_index))
            jax.block_until_ready(result)
            collected.append(np.asarray(result))
    elapsed = time.perf_counter() - started
    arrays = {"terminal_sample": np.concatenate(collected)}
    _require_finite(arrays, task)
    return arrays, elapsed


def execute_dynamic_batches(
    compiled: CompiledDynamicKernels,
    config: Experiment1Config,
    task: Experiment1Task,
    *,
    device,
) -> tuple[dict[str, np.ndarray], float]:
    """Execute all theorem/control batches and transfer only requested outputs."""

    return _collect_dynamic(compiled, config, task, device=device)


def _require_finite(arrays: dict[str, np.ndarray], task: Experiment1Task) -> None:
    for name, values in arrays.items():
        if np.any(~np.isfinite(values)):
            raise NumericalError(
                f"non-finite values in task={task.task_id}, seed={task.seed}, array={name}"
            )


__all__ = [
    "CompiledControlKernel",
    "CompiledDynamicKernels",
    "compile_control_kernel",
    "compile_dynamic_kernels",
    "execute_control_batches",
    "execute_dynamic_batches",
    "output_forward_times",
    "stream_labels",
    "vp_ei_coefficients",
]
