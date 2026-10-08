from __future__ import annotations

import pytest

from levy_experiments.config import ExperimentTask
from levy_experiments.errors import NumericalError
from levy_experiments.experiment3 import validate_task_eta_consistency


def _task(method: str, drift_eta: float, noise_eta: float) -> ExperimentTask:
    return ExperimentTask(0, "three_atoms", method, 2.5, drift_eta, noise_eta, 11, 7)


def test_valid_popov_task_requires_one_eta_for_drift_and_noise() -> None:
    validate_task_eta_consistency(_task("popov_ei", 0.5, 0.5))
    with pytest.raises(NumericalError, match="identical drift and noise eta"):
        validate_task_eta_consistency(_task("popov_ei", 0.5, 1.0))


def test_invalid_hybrid_control_cannot_be_mislabeled_as_matched() -> None:
    validate_task_eta_consistency(_task("hybrid_ei_invalid", 0.5, 1.0))
    with pytest.raises(NumericalError, match="must keep distinct"):
        validate_task_eta_consistency(_task("hybrid_ei_invalid", 0.5, 0.5))
