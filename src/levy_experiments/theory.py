"""Closed-form characteristic functions and exact transitions for Experiment 3."""

from __future__ import annotations

import math

import numpy as np


def stationary_transition_coefficients(
    *, alpha: float, beta: float, eta: float, lag: float
) -> tuple[float, float]:
    """Return r and innovation scale for the stationary Popov transition."""

    retained = math.exp(-eta * beta * lag / alpha)
    innovation = (-math.expm1(-eta * beta * lag)) ** (1.0 / alpha)
    return retained, innovation


def stationary_joint_cf(u, v, *, alpha: float, beta: float, eta: float, lag: float):
    """CF E exp(i[u Y_0 + v Y_lag]) for the stationary Popov member."""

    retained, _ = stationary_transition_coefficients(alpha=alpha, beta=beta, eta=eta, lag=lag)
    u_values = np.asarray(u, dtype=np.float64)
    v_values = np.asarray(v, dtype=np.float64)
    return np.exp(
        -(np.abs(u_values + retained * v_values) ** alpha)
        - (1.0 - retained**alpha) * np.abs(v_values) ** alpha
    ).astype(np.complex128)


def stationary_true_reverse_joint_cf(u, v, *, alpha: float, beta: float, lag: float):
    """CF of (X_lag, X_0), the true reversed stationary forward pair."""

    retained = math.exp(-beta * lag / alpha)
    u_values = np.asarray(u, dtype=np.float64)
    v_values = np.asarray(v, dtype=np.float64)
    return np.exp(
        -(np.abs(retained * u_values + v_values) ** alpha)
        - (1.0 - retained**alpha) * np.abs(u_values) ** alpha
    ).astype(np.complex128)


def stable_marginal_cf(frequencies, *, alpha: float):
    frequencies = np.asarray(frequencies, dtype=np.float64)
    return np.exp(-(np.abs(frequencies) ** alpha)).astype(np.complex128)


def discrete_marginal_cf(
    frequencies,
    time: float,
    *,
    alpha: float,
    beta: float,
    atoms: tuple[float, ...],
    weights: tuple[float, ...],
):
    """Exact CF of a noised finite target at forward time t."""

    u = np.asarray(frequencies, dtype=np.float64)
    a = math.exp(-beta * time / alpha)
    gamma_power = -math.expm1(-beta * time)
    atom_values = np.asarray(atoms, dtype=np.float64)
    mixture_weights = np.asarray(weights, dtype=np.float64)
    mixture_cf = np.sum(
        mixture_weights[None, :] * np.exp(1j * u[..., None] * a * atom_values[None, :]),
        axis=-1,
    )
    return np.exp(-gamma_power * np.abs(u) ** alpha) * mixture_cf


def discrete_true_reverse_joint_cf(
    u,
    v,
    *,
    later_forward_time: float,
    earlier_forward_time: float,
    alpha: float,
    beta: float,
    atoms: tuple[float, ...],
    weights: tuple[float, ...],
):
    """CF of (X_later, X_earlier), ordered along backward time."""

    if later_forward_time <= earlier_forward_time:
        raise ValueError("later_forward_time must exceed earlier_forward_time")
    u_values = np.asarray(u, dtype=np.float64)
    v_values = np.asarray(v, dtype=np.float64)
    retained = math.exp(-beta * (later_forward_time - earlier_forward_time) / alpha)
    innovation_power = 1.0 - retained**alpha
    marginal = discrete_marginal_cf(
        retained * u_values + v_values,
        earlier_forward_time,
        alpha=alpha,
        beta=beta,
        atoms=atoms,
        weights=weights,
    )
    return np.exp(-innovation_power * np.abs(u_values) ** alpha) * marginal
