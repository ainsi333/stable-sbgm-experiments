from __future__ import annotations

import inspect
import math

import numpy as np

from experiments.exp1.simulation import _update_state as exp1_update_state
from experiments.exp2.simulation import _task_kernel as exp2_task_kernel
from levy_experiments.experiment3 import _nonstationary_kernel as exp3_nonstationary_kernel
from levy_experiments.integrators import popov_remainder, stable_ei_coefficients


def _compact_source(function) -> str:
    return " ".join(inspect.getsource(function).split())


def test_eta_removed_and_euler_noise_mutants_violate_the_ei_oracle() -> None:
    alpha, eta, beta, step = 1.5, 0.5, 1.3, 0.07
    exact = stable_ei_coefficients(
        alpha=alpha, beta=beta, step=step, drift_eta=eta
    ).noise_scale**alpha
    eta_removed_mutant = -math.expm1(-beta * step)
    euler_mutant = eta * beta * step
    tolerance = 5.0e-15 + 2.0e-13 * abs(exact)

    assert abs(eta_removed_mutant - exact) > tolerance
    assert abs(euler_mutant - exact) > tolerance


def test_reversed_score_sign_mutant_violates_the_popov_oracle() -> None:
    x = np.asarray([-2.0, 0.5, 3.0])
    score = np.asarray([0.4, -0.2, 1.1])
    alpha, eta, beta = 1.5, 0.7, 1.3
    exact = popov_remainder(x, score, alpha=alpha, beta=beta, eta=eta)
    sign_mutant = ((1.0 + eta) * beta / alpha) * (x - alpha * score)

    assert not np.allclose(exact, sign_mutant, rtol=2.0e-13, atol=5.0e-15)


def test_reverse_kernels_evaluate_scores_on_the_descending_forward_grid() -> None:
    """Kill the explicit ``epsilon + k*h`` reverse-time mutation in every kernel."""

    horizon, epsilon, steps = 2.5, 0.25, 110
    step = (horizon - epsilon) / steps
    indices = np.arange(steps + 1, dtype=np.float64)
    expected = horizon - indices * step
    ascending_mutant = epsilon + indices * step
    assert math.isclose(expected[0], horizon, rel_tol=0.0, abs_tol=5.0e-14)
    assert math.isclose(expected[-1], epsilon, rel_tol=0.0, abs_tol=5.0e-14)
    assert not np.allclose(expected, ascending_mutant, rtol=0.0, atol=5.0e-14)

    sources = (
        _compact_source(exp1_update_state),
        _compact_source(exp2_task_kernel),
        _compact_source(exp3_nonstationary_kernel),
    )
    required = "- step_index.astype(dtype) * step_value"
    for source in sources:
        assert required in source
        assert "epsilon + step_index.astype(dtype) * step_value" not in source
