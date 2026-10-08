from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levy_experiments.devices import configure_runtime
from levy_experiments.random import named_key
from levy_experiments.stable import sample_symmetric_stable_1d


def test_cpu_selection_is_explicit() -> None:
    _, info = configure_runtime("cpu", "float64")
    assert info.platform == "cpu"


@pytest.mark.skipif(not any(device.platform == "gpu" for device in jax.devices()), reason="no GPU")
def test_cpu_gpu_statistical_parity_on_same_small_configuration() -> None:
    cpu = jax.devices("cpu")[0]
    gpu, info = configure_runtime("gpu", "float64")
    assert info.platform == "gpu"
    key = named_key(808, "cpu-gpu-parity")

    def kernel(random_key):
        return sample_symmetric_stable_1d(random_key, 1.5, (16_384,), jnp.float64)

    with jax.default_device(cpu):
        cpu_values = np.asarray(jax.jit(kernel)(key))
    with jax.default_device(gpu):
        gpu_values = np.asarray(jax.jit(kernel)(key))
    frequencies = np.asarray([0.5, 1.0, 2.0])
    cpu_cf = np.mean(np.exp(1j * cpu_values[:, None] * frequencies), axis=0)
    gpu_cf = np.mean(np.exp(1j * gpu_values[:, None] * frequencies), axis=0)
    assert np.max(np.abs(cpu_cf - gpu_cf)) < 0.03
