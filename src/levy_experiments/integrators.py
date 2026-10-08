"""Exponential-integrator coefficients for the Popov reverse family."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import ConfigurationError


@dataclass(frozen=True)
class EICoefficients:
    decay: float
    remainder_multiplier: float
    noise_scale: float
    drift_eta: float
    noise_eta: float


def stable_ei_coefficients(
    *, alpha: float, beta: float, step: float, drift_eta: float, noise_eta: float | None = None
) -> EICoefficients:
    """Return exact frozen-remainder EI coefficients.

    For a valid Popov member ``noise_eta == drift_eta``.  A distinct
    ``noise_eta`` is accepted only to construct the explicitly labelled invalid
    hybrid control.  Its OU convolution is still integrated exactly.
    """

    noise_eta = drift_eta if noise_eta is None else noise_eta
    if alpha <= 0.0 or beta <= 0.0 or step <= 0.0 or drift_eta <= 0.0 or noise_eta <= 0.0:
        raise ConfigurationError("EI coefficients require positive alpha, beta, h and eta values")
    rate = drift_eta * beta / alpha
    lost_mass = -math.expm1(-drift_eta * beta * step)
    decay = math.exp(-rate * step)
    remainder_multiplier = -math.expm1(-rate * step) / rate
    noise_power = (noise_eta / drift_eta) * lost_mass
    return EICoefficients(
        decay=decay,
        remainder_multiplier=remainder_multiplier,
        noise_scale=noise_power ** (1.0 / alpha),
        drift_eta=drift_eta,
        noise_eta=noise_eta,
    )


def stationary_terminal_scale_power(
    *, beta: float, lag: float, drift_eta: float, noise_eta: float
) -> float:
    """Return scale**alpha after one stationary stable-OU transition."""

    if min(beta, lag, drift_eta, noise_eta) <= 0.0:
        raise ConfigurationError("stationary scale arguments must be positive")
    retained = math.exp(-drift_eta * beta * lag)
    injected = (noise_eta / drift_eta) * (1.0 - retained)
    return retained + injected


def popov_remainder(x, score, *, alpha: float, beta: float, eta: float):
    """Evaluate F=(1+eta) beta (x+alpha S)/alpha on a backend array."""

    return ((1.0 + eta) * beta / alpha) * (x + alpha * score)
