"""Shared deterministic JAX test setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import pytest

# Verify this checkout even when an older wheel is installed in the environment.
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

jax.config.update("jax_enable_x64", True)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Keep Windows test artifacts local without cross-session deletion races."""

    if config.option.basetemp is None:
        config.option.basetemp = Path(config.rootpath) / f".pytest_tmp_{os.getpid()}"


@pytest.fixture(scope="session")
def cpu_device():
    return jax.devices("cpu")[0]
