"""JAX precision and single-device selection."""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass
from typing import Literal

from .errors import ConfigurationError

DeviceRequest = Literal["cpu", "gpu", "auto"]


@dataclass(frozen=True)
class RuntimeDevice:
    requested: DeviceRequest
    platform: str
    device_kind: str
    device_id: int
    process_index: int
    cpu: str
    logical_cpu_count: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def configure_runtime(device_request: DeviceRequest, precision: str):
    """Enable the requested precision and return exactly one JAX device."""

    if precision not in {"float32", "float64", "mixed"}:
        raise ConfigurationError(f"Unsupported precision: {precision}")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax

    jax.config.update("jax_enable_x64", precision in {"float64", "mixed"})
    cpu_devices = tuple(jax.devices("cpu"))
    try:
        gpu_devices = tuple(jax.devices("gpu"))
    except RuntimeError:
        gpu_devices = ()
    if device_request == "gpu":
        if not gpu_devices:
            raise ConfigurationError("--device gpu requested, but JAX found no GPU")
        device = gpu_devices[0]
    elif device_request == "cpu":
        if not cpu_devices:
            raise ConfigurationError("JAX found no CPU device")
        device = cpu_devices[0]
    else:
        device = gpu_devices[0] if gpu_devices else cpu_devices[0]
    info = RuntimeDevice(
        requested=device_request,
        platform=device.platform,
        device_kind=getattr(device, "device_kind", str(device)),
        device_id=int(getattr(device, "id", 0)),
        process_index=int(getattr(device, "process_index", 0)),
        cpu=platform.processor() or platform.machine(),
        logical_cpu_count=os.cpu_count(),
    )
    return device, info


def simulation_dtype(precision: str):
    """Return the propagation dtype; mixed keeps references/tables on CPU in float64."""

    import jax.numpy as jnp

    return jnp.float64 if precision == "float64" else jnp.float32
