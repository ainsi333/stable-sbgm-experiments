"""Exact reference sampling and JAX propagation for Experiment 2."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from levy_experiments.errors import ConfigurationError, NumericalError
from levy_experiments.integrators import popov_remainder, stable_ei_coefficients
from levy_experiments.random import named_key
from levy_experiments.stable import sample_symmetric_stable_1d

from .theory import (
    sample_exact_forward_marginal,
    sample_stationary_reference,
)

Model = Literal["stable", "vp"]


@dataclass(frozen=True)
class VPEICoefficients:
    """Frozen-score exponential-integrator coefficients for the VP reverse SDE."""

    decay: float
    remainder_multiplier: float
    noise_scale: float


def vp_ei_coefficients(*, beta: float, step: float) -> VPEICoefficients:
    """Integrate the linear ``-beta*x/2`` part and Brownian noise exactly."""

    if not math.isfinite(beta) or not math.isfinite(step) or beta <= 0.0 or step <= 0.0:
        raise ConfigurationError("VP EI coefficients require finite positive beta and h")
    rate = beta / 2.0
    return VPEICoefficients(
        decay=math.exp(-rate * step),
        remainder_multiplier=-math.expm1(-rate * step) / rate,
        noise_scale=math.sqrt(-math.expm1(-beta * step)),
    )


def checkpoint_indices(fractions: tuple[float, ...], steps: int) -> tuple[int, ...]:
    """Map prescribed backward-time fractions to exact post-update grid indices."""

    if steps <= 0:
        raise ConfigurationError("steps must be positive")
    indices: list[int] = []
    for fraction in fractions:
        raw = fraction * steps
        index = round(raw)
        if not math.isclose(raw, index, rel_tol=0.0, abs_tol=1e-10):
            raise ConfigurationError(
                f"checkpoint fraction {fraction} is not represented by N={steps}"
            )
        indices.append(index)
    if not indices or indices[0] != 0 or indices[-1] != steps:
        raise ConfigurationError("checkpoint fractions must include 0 and 1")
    return tuple(indices)


def forward_coefficients(
    *, model: Model, time: float, alpha: float, beta: float
) -> tuple[float, float]:
    """Return ``a(t), gamma(t)`` for the stable or VP forward marginal."""

    if time < 0.0 or alpha <= 0.0 or beta <= 0.0:
        raise ConfigurationError("forward coefficients require t>=0, alpha>0 and beta>0")
    if model == "stable":
        return (
            math.exp(-beta * time / alpha),
            (-math.expm1(-beta * time)) ** (1.0 / alpha),
        )
    if model == "vp":
        return math.exp(-beta * time / 2.0), math.sqrt(-math.expm1(-beta * time))
    raise ConfigurationError(f"unsupported model: {model}")


def output_forward_times(
    *, horizon: float, epsilon: float, checkpoint_fractions: tuple[float, ...]
) -> tuple[float, ...]:
    """Return the exact forward times represented by backward-time checkpoints."""

    duration = horizon - epsilon
    values = [horizon - fraction * duration for fraction in checkpoint_fractions]
    values[0] = horizon
    values[-1] = epsilon
    return tuple(values)


def refinement_coupling_steps(config: Any, task: Any) -> int:
    """Return the finest primitive grid shared by one refinement family."""

    design = config.design
    is_refinement = math.isclose(
        float(task.horizon),
        float(design.refinement_horizon),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ) and int(task.steps) in set(design.refinement_steps)
    if not is_refinement:
        return int(task.steps)
    finest = max(int(value) for value in design.refinement_steps)
    if finest % int(task.steps):
        raise ConfigurationError(
            f"refinement N={task.steps} does not divide the finest N={finest}"
        )
    return finest


def aggregate_ei_primitives(primitives, *, decay, scale):
    """Aggregate fine innovations into the exact coarse EI increment."""

    import jax.numpy as jnp

    values = jnp.asarray(primitives)
    if values.ndim < 1 or values.shape[0] < 1:
        raise ConfigurationError("EI primitive array must have a non-empty leading axis")
    exponents = jnp.arange(values.shape[0] - 1, -1, -1, dtype=values.dtype)
    weights = jnp.asarray(scale, dtype=values.dtype) * jnp.power(
        jnp.asarray(decay, dtype=values.dtype), exponents
    )
    return jnp.tensordot(weights, values, axes=((0,), (0,)))


def stream_labels(task: Any, *, coupling_steps: int | None = None) -> tuple[str, ...]:
    """Return semantic streams, separating initialization, dynamics and references."""

    prefix = f"exp2/{task.model}/T={task.horizon:.17g}"
    primitive_steps = int(task.steps if coupling_steps is None else coupling_steps)
    dynamics = f"{prefix}/primitive_N={primitive_steps}"
    return (
        f"{prefix}/exact_initialization",
        f"{prefix}/stationary_initialization",
        f"{dynamics}/exact_initialization_dynamics",
        f"{dynamics}/stationary_initialization_dynamics",
        f"{prefix}/exact_reference_a",
        f"{prefix}/exact_reference_b",
        f"{prefix}/exact_reference_c",
    )


def _task_kernel(config: Any, task: Any, table: Any, hires_table: Any, dtype):
    """Create a fused batched kernel; the only time loop is a JAX loop."""

    import jax
    import jax.numpy as jnp

    from .score_tables import interpolate_score_jax

    experiment = config.experiment
    stable = config.stable
    design = config.design
    alpha = stable.alpha
    beta = experiment.beta
    eta = stable.eta
    epsilon = experiment.epsilon
    horizon = task.horizon
    duration = horizon - epsilon
    step = duration / task.steps
    primitive_steps = refinement_coupling_steps(config, task)
    primitive_ratio = primitive_steps // task.steps
    primitive_step = duration / primitive_steps
    fractions = design.checkpoint_fractions
    indices = checkpoint_indices(fractions, task.steps)
    index_array = jnp.asarray(indices, dtype=jnp.int32)
    forward_times = output_forward_times(
        horizon=horizon,
        epsilon=epsilon,
        checkpoint_fractions=fractions,
    )
    time_array = jnp.asarray(forward_times, dtype=dtype)
    batch_size = experiment.batch_size

    if task.model == "stable":
        coefficients = stable_ei_coefficients(
            alpha=alpha,
            beta=beta,
            step=step,
            drift_eta=eta,
        )
        primitive_coefficients = stable_ei_coefficients(
            alpha=alpha,
            beta=beta,
            step=primitive_step,
            drift_eta=eta,
        )
    else:
        coefficients = vp_ei_coefficients(beta=beta, step=step)
        primitive_coefficients = vp_ei_coefficients(beta=beta, step=primitive_step)
    decay = jnp.asarray(coefficients.decay, dtype=dtype)
    remainder_multiplier = jnp.asarray(coefficients.remainder_multiplier, dtype=dtype)
    primitive_decay = jnp.asarray(primitive_coefficients.decay, dtype=dtype)
    primitive_noise_scale = jnp.asarray(
        primitive_coefficients.noise_scale, dtype=dtype
    )
    horizon_value = jnp.asarray(horizon, dtype=dtype)
    step_value = jnp.asarray(step, dtype=dtype)

    def exact_stack(base_key):
        reference_keys = jax.random.split(base_key, len(forward_times))
        return jnp.stack(
            tuple(
                sample_exact_forward_marginal(
                    reference_keys[index],
                    task.model,
                    time_value,
                    (batch_size,),
                    dtype,
                    alpha=alpha,
                    beta=beta,
                )
                for index, time_value in enumerate(forward_times)
            ),
            axis=0,
        )

    def kernel(keys):
        exact_state = sample_exact_forward_marginal(
            keys[0],
            task.model,
            horizon,
            (batch_size,),
            dtype,
            alpha=alpha,
            beta=beta,
        )
        reference_state = sample_stationary_reference(
            keys[1], task.model, (batch_size,), dtype, alpha=alpha
        )
        exact_recorded = jnp.zeros((len(indices), batch_size), dtype=dtype).at[0].set(
            exact_state
        )
        reference_recorded = jnp.zeros((len(indices), batch_size), dtype=dtype).at[0].set(
            reference_state
        )
        hires_state = exact_state
        hires_recorded = jnp.zeros((len(indices), batch_size), dtype=dtype).at[0].set(
            hires_state
        )

        def body(step_index, carry):
            (
                current_exact,
                current_reference,
                current_hires,
                stored_exact,
                stored_reference,
                stored_hires,
            ) = carry
            forward_time = horizon_value - step_index.astype(dtype) * step_value
            exact_score = interpolate_score_jax(
                current_exact, forward_time, table=table, dtype=dtype
            )
            reference_score = interpolate_score_jax(
                current_reference, forward_time, table=table, dtype=dtype
            )
            hires_score = interpolate_score_jax(
                current_hires, forward_time, table=hires_table, dtype=dtype
            )
            if task.model == "stable":
                exact_remainder = popov_remainder(
                    current_exact, exact_score, alpha=alpha, beta=beta, eta=eta
                )
                reference_remainder = popov_remainder(
                    current_reference, reference_score, alpha=alpha, beta=beta, eta=eta
                )
                hires_remainder = popov_remainder(
                    current_hires, hires_score, alpha=alpha, beta=beta, eta=eta
                )
                exact_primitives = jnp.stack(
                    tuple(
                        sample_symmetric_stable_1d(
                            jax.random.fold_in(
                                keys[2], step_index * primitive_ratio + substep
                            ),
                            alpha,
                            (batch_size,),
                            dtype,
                        )
                        for substep in range(primitive_ratio)
                    ),
                    axis=0,
                )
                reference_primitives = jnp.stack(
                    tuple(
                        sample_symmetric_stable_1d(
                            jax.random.fold_in(
                                keys[3], step_index * primitive_ratio + substep
                            ),
                            alpha,
                            (batch_size,),
                            dtype,
                        )
                        for substep in range(primitive_ratio)
                    ),
                    axis=0,
                )
            else:
                exact_remainder = beta * (current_exact + exact_score)
                reference_remainder = beta * (current_reference + reference_score)
                hires_remainder = beta * (current_hires + hires_score)
                exact_primitives = jnp.stack(
                    tuple(
                        jax.random.normal(
                            jax.random.fold_in(
                                keys[2], step_index * primitive_ratio + substep
                            ),
                            shape=(batch_size,),
                            dtype=dtype,
                        )
                        for substep in range(primitive_ratio)
                    ),
                    axis=0,
                )
                reference_primitives = jnp.stack(
                    tuple(
                        jax.random.normal(
                            jax.random.fold_in(
                                keys[3], step_index * primitive_ratio + substep
                            ),
                            shape=(batch_size,),
                            dtype=dtype,
                        )
                        for substep in range(primitive_ratio)
                    ),
                    axis=0,
                )
            exact_noise = aggregate_ei_primitives(
                exact_primitives,
                decay=primitive_decay,
                scale=primitive_noise_scale,
            )
            reference_noise = aggregate_ei_primitives(
                reference_primitives,
                decay=primitive_decay,
                scale=primitive_noise_scale,
            )
            updated_exact = (
                decay * current_exact
                + remainder_multiplier * exact_remainder
                + exact_noise
            )
            updated_reference = (
                decay * current_reference
                + remainder_multiplier * reference_remainder
                + reference_noise
            )
            # The high-resolution arm deliberately reuses the exact-pT
            # initialization and every primitive noise increment.  Therefore
            # its difference from ``updated_exact`` isolates score tabulation.
            updated_hires = (
                decay * current_hires
                + remainder_multiplier * hires_remainder
                + exact_noise
            )
            matches = index_array == (step_index + 1)
            stored_exact = jnp.where(matches[:, None], updated_exact[None, :], stored_exact)
            stored_reference = jnp.where(
                matches[:, None], updated_reference[None, :], stored_reference
            )
            stored_hires = jnp.where(
                matches[:, None], updated_hires[None, :], stored_hires
            )
            return (
                updated_exact,
                updated_reference,
                updated_hires,
                stored_exact,
                stored_reference,
                stored_hires,
            )

        _, _, _, exact_recorded, reference_recorded, hires_recorded = jax.lax.fori_loop(
            0,
            task.steps,
            body,
            (
                exact_state,
                reference_state,
                hires_state,
                exact_recorded,
                reference_recorded,
                hires_recorded,
            ),
        )
        return (
            time_array,
            exact_recorded,
            reference_recorded,
            hires_recorded,
            exact_stack(keys[4]),
            exact_stack(keys[5]),
            exact_stack(keys[6]),
        )

    return kernel


def compile_task_kernel(
    config: Any, task: Any, table: Any, hires_table: Any, *, dtype, device
):
    """Compile once and report compilation independently from execution."""

    import jax
    import jax.numpy as jnp

    labels = stream_labels(
        task, coupling_steps=refinement_coupling_steps(config, task)
    )
    example = jnp.stack(tuple(named_key(task.seed, label, 0) for label in labels), axis=0)
    with jax.default_device(device):
        jitted = jax.jit(_task_kernel(config, task, table, hires_table, dtype))
        started = time.perf_counter()
        compiled = jitted.lower(example).compile()
        compile_seconds = time.perf_counter() - started
    return compiled, compile_seconds


def score_sensitivity_from_checkpoints(
    main: np.ndarray,
    hires: np.ndarray,
    exact_a: np.ndarray,
    exact_b: np.ndarray,
    orders: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired table sensitivity and its pre-specified ``F/2`` gate."""

    matrices = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (main, hires, exact_a, exact_b)
    )
    invalid_rank = any(values.ndim != 2 for values in matrices)
    if invalid_rank or len({values.shape for values in matrices}) != 1:
        raise NumericalError("score-sensitivity inputs must share a two-dimensional shape")
    if any(np.any(~np.isfinite(values)) for values in matrices):
        raise NumericalError("score-sensitivity inputs must be finite")
    if not orders or any(not math.isfinite(order) or order < 1.0 for order in orders):
        raise ConfigurationError("score-sensitivity Wasserstein orders must be finite and >=1")

    main_values, hires_values, floor_a, floor_b = matrices
    sensitivity = np.empty((main_values.shape[0], len(orders)), dtype=np.float64)
    controlled = np.empty_like(sensitivity, dtype=bool)
    for time_index in range(main_values.shape[0]):
        sorted_main = np.sort(main_values[time_index])
        sorted_hires = np.sort(hires_values[time_index])
        sorted_a = np.sort(floor_a[time_index])
        sorted_b = np.sort(floor_b[time_index])
        for order_index, order in enumerate(orders):
            table_delta = float(
                np.mean(np.abs(sorted_main - sorted_hires) ** order) ** (1.0 / order)
            )
            floor = float(
                np.mean(np.abs(sorted_a - sorted_b) ** order) ** (1.0 / order)
            )
            sensitivity[time_index, order_index] = table_delta
            controlled[time_index, order_index] = table_delta <= 0.5 * floor
    return sensitivity, controlled


def execute_task_batches(
    compiled,
    config: Any,
    task: Any,
    *,
    device,
) -> tuple[dict[str, np.ndarray], float]:
    """Execute complete trajectories on-device and transfer only recorded checkpoints."""

    import jax
    import jax.numpy as jnp

    labels = stream_labels(
        task, coupling_steps=refinement_coupling_steps(config, task)
    )
    batches = config.experiment.particles // config.experiment.batch_size
    collected: list[tuple[np.ndarray, ...]] = []
    started = time.perf_counter()
    with jax.default_device(device):
        for batch_index in range(batches):
            keys = jnp.stack(
                tuple(named_key(task.seed, label, batch_index) for label in labels), axis=0
            )
            result = compiled(keys)
            jax.block_until_ready(result)
            collected.append(tuple(np.asarray(item) for item in result))
    elapsed = time.perf_counter() - started
    forward_times = collected[0][0]
    if any(not np.array_equal(batch[0], forward_times) for batch in collected[1:]):
        raise NumericalError("batches disagree on checkpoint times")
    numerical_exact = np.concatenate([batch[1] for batch in collected], axis=1)
    numerical_reference = np.concatenate([batch[2] for batch in collected], axis=1)
    numerical_hires = np.concatenate([batch[3] for batch in collected], axis=1)
    exact_a = np.concatenate([batch[4] for batch in collected], axis=1)
    exact_b = np.concatenate([batch[5] for batch in collected], axis=1)
    exact_c = np.concatenate([batch[6] for batch in collected], axis=1)
    sensitivity, controlled = score_sensitivity_from_checkpoints(
        numerical_exact,
        numerical_hires,
        exact_a,
        exact_b,
        tuple(float(order) for order in config.analysis.wasserstein_orders),
    )
    arrays = {
        "forward_times": np.asarray(forward_times, dtype=np.float64),
        "numerical_exact_init": numerical_exact,
        "numerical_reference_init": numerical_reference,
        "exact_a": exact_a,
        "exact_b": exact_b,
        "exact_c": exact_c,
        "score_sensitivity_wp": sensitivity,
        "d_controlled": controlled,
    }
    for name, values in arrays.items():
        if np.any(~np.isfinite(values)):
            raise NumericalError(
                f"non-finite values in task={task.task_id}, seed={task.seed}, array={name}"
            )
    return arrays, elapsed
