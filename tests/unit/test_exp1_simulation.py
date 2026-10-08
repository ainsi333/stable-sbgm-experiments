from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.exp1.config import Experiment1Task, load_config
from experiments.exp1.runner import _lookup_table
from experiments.exp1.score_tables import ScoreTable, table_key
from experiments.exp1.simulation import (
    compile_control_kernel,
    compile_dynamic_kernels,
    execute_control_batches,
    execute_dynamic_batches,
    stream_labels,
    vp_ei_coefficients,
)
from levy_experiments.errors import ArtifactError
from levy_experiments.integrators import stable_ei_coefficients

ROOT = Path(__file__).resolve().parents[2]


def _tiny_config():
    config = load_config(ROOT / "configs" / "smoke" / "exp1.toml")
    return replace(config, experiment=replace(config.experiment, batch_size=32))


def _dummy_table(model: str, nu: float) -> ScoreTable:
    times = np.geomspace(0.5, 2.0, 9)
    spaces = np.expm1(np.linspace(0.0, np.log1p(1.0e6), 257))
    scores = np.zeros((times.size, spaces.size), dtype=np.float64)
    return ScoreTable(
        model=model,
        time_nodes=times,
        space_nodes=spaces,
        scores=scores,
        table_hash=f"dummy-{model}-{nu}",
        metadata={
            "model": model,
            "nu": nu,
            "alpha": 1.5,
            "beta": 1.0,
            "blend_start": 5.0e5,
            "blend_end": 1.0e6,
            "fft_half_width": 2.0e6,
            "fft_points": 1024,
            "tier": "smoke",
        },
    )


def _task(model: str, steps: int, *, kind: str = "dynamic") -> Experiment1Task:
    return Experiment1Task(
        index=0,
        kind=kind,
        model=model,
        nu=1.5 if kind == "dynamic" else None,
        steps=steps if kind == "dynamic" else 0,
        seed=17,
        particles=64,
        control_particles=64 if kind == "dynamic" else 0,
        purpose="refinement" if kind == "dynamic" else "control",
        coupling_steps=4 if kind == "dynamic" else 0,
    )


def test_ei_factors_use_eta_beta_h_exactly() -> None:
    stable = stable_ei_coefficients(alpha=1.5, beta=1.2, step=0.075, drift_eta=0.5)
    assert np.isclose(stable.decay, np.exp(-0.5 * 1.2 * 0.075 / 1.5))
    assert np.isclose(stable.noise_scale**1.5, 1.0 - np.exp(-0.5 * 1.2 * 0.075))
    vp = vp_ei_coefficients(beta=1.2, step=0.075)
    assert np.isclose(vp.decay, np.exp(-1.2 * 0.075 / 2.0))
    assert np.isclose(vp.noise_scale**2, 1.0 - np.exp(-1.2 * 0.075))


def test_wrong_target_score_table_is_rejected_at_point_of_use() -> None:
    task = replace(_task("stable", 2), nu=1.2)
    wrong_nu_table = _dummy_table("stable", 1.5)
    tables = {table_key("stable", 1.2): wrong_nu_table}

    with pytest.raises(ArtifactError, match="wrong score table"):
        _lookup_table(task, tables)


@pytest.mark.parametrize("model", ("stable", "vp"))
def test_cpu_dynamic_kernel_is_finite_reproducible_and_has_three_references(
    model: str, cpu_device
) -> None:
    config = _tiny_config()
    task = _task(model, 2)
    table = _dummy_table(model, 1.5)
    compiled = compile_dynamic_kernels(
        config, task, table, table, dtype=jnp.float64, device=cpu_device
    )
    first, _ = execute_dynamic_batches(compiled, config, task, device=cpu_device)
    second, _ = execute_dynamic_batches(compiled, config, task, device=cpu_device)
    assert set(first) == {
        "terminal_sample",
        "score_hires_terminal",
        "score_sensitivity_forward_times",
        "score_sensitivity_mean_abs",
        "control_forward_times",
        "control_numerical",
        "control_exact_a",
        "control_exact_b",
        "control_exact_c",
    }
    for name in first:
        assert np.array_equal(first[name], second[name])
        assert np.all(np.isfinite(first[name]))
    assert np.array_equal(first["control_forward_times"], np.asarray([0.5, 1.25, 2.0]))
    assert first["terminal_sample"].shape == (64,)
    assert np.array_equal(first["terminal_sample"], first["score_hires_terminal"])
    assert np.array_equal(first["score_sensitivity_mean_abs"], np.zeros(3))
    assert first["control_numerical"].shape == (3, 64)
    assert not np.array_equal(first["control_exact_a"], first["control_exact_b"])
    assert not np.array_equal(first["control_exact_b"], first["control_exact_c"])


def test_refinement_streams_share_initials_and_primitive_fine_noise() -> None:
    coarse = _task("stable", 2)
    fine = _task("stable", 4)
    assert stream_labels(coarse) == stream_labels(fine)


@pytest.mark.parametrize("model", ("sas", "pareto", "gaussian"))
def test_direct_controls_are_finite_reproducible_and_unclipped(model: str, cpu_device) -> None:
    config = _tiny_config()
    task = _task(model, 0, kind="control")
    compiled = compile_control_kernel(config, task, dtype=jnp.float64, device=cpu_device)
    first, _ = execute_control_batches(compiled, config, task, device=cpu_device)
    second, _ = execute_control_batches(compiled, config, task, device=cpu_device)
    assert np.array_equal(first["terminal_sample"], second["terminal_sample"])
    assert np.all(np.isfinite(first["terminal_sample"]))
    if model == "pareto":
        assert np.min(np.abs(first["terminal_sample"])) >= 1.0
        assert np.max(np.abs(first["terminal_sample"])) > 5.0


def test_gpu_control_parity_when_available(cpu_device) -> None:
    try:
        gpu = jax.devices("gpu")[0]
    except (RuntimeError, IndexError):
        pytest.skip("no JAX GPU available")
    config = _tiny_config()
    task = _task("gaussian", 0, kind="control")
    cpu_kernel = compile_control_kernel(config, task, dtype=jnp.float64, device=cpu_device)
    gpu_kernel = compile_control_kernel(config, task, dtype=jnp.float64, device=gpu)
    cpu, _ = execute_control_batches(cpu_kernel, config, task, device=cpu_device)
    gpu_values, _ = execute_control_batches(gpu_kernel, config, task, device=gpu)
    assert np.allclose(cpu["terminal_sample"], gpu_values["terminal_sample"], atol=1.0e-11)
