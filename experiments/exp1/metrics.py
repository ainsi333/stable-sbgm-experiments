"""Pure, auditable tail statistics for Experiment 1.

All estimators operate on the radial sample ``abs(X)`` in one dimension.  The
module never clips, truncates or winsorizes observations: an extreme finite
value remains an extreme finite value throughout the analysis.  Functions
return explicit validity flags instead of silently changing ``k`` or a
threshold when a requested statistic is under-resolved.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize, stats


class TailMetricError(ValueError):
    """Raised when a tail statistic is mathematically undefined."""


@dataclass(frozen=True)
class PowerLawFit:
    """Log--log least-squares fit used for tail-ratio slopes."""

    slope: float
    intercept: float
    r_squared: float
    n_points: int


def finite_sample(sample: np.ndarray, *, minimum: int = 2) -> np.ndarray:
    """Return a one-dimensional float64 view after strict finite validation."""

    values = np.asarray(sample)
    if (
        values.ndim != 1
        or not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.complexfloating)
    ):
        raise TailMetricError("sample must be a one-dimensional real numerical array")
    values = np.asarray(values, dtype=np.float64)
    if values.size < minimum:
        raise TailMetricError(f"sample must contain at least {minimum} observations")
    if np.any(~np.isfinite(values)):
        raise TailMetricError("sample contains a non-finite observation")
    return values


def sorted_radial(sample: np.ndarray) -> np.ndarray:
    """Descending order statistics of ``abs(sample)`` without clipping."""

    values = finite_sample(sample)
    radial = np.abs(values)
    if np.any(~np.isfinite(radial)):
        raise TailMetricError("absolute value overflowed")
    return np.sort(radial)[::-1]


def tail_count(sample_count: int, fraction: float) -> int:
    """Frozen floor convention for converting a tail fraction to ``k``."""

    if sample_count < 2 or not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise TailMetricError("tail count requires n>=2 and a fraction in (0,1)")
    return math.floor(sample_count * fraction)


def _validate_ordered(ordered: np.ndarray) -> np.ndarray:
    values = finite_sample(ordered)
    if np.any(values < 0.0) or np.any(values[:-1] < values[1:]):
        raise TailMetricError("radial order statistics must be non-negative and descending")
    return values


def hill_at_k(ordered: np.ndarray, k: int) -> tuple[float, float, float]:
    """Return ``(xi, alpha, threshold)`` for Hill's top-``k`` estimator."""

    radial = _validate_ordered(ordered)
    if not 1 <= k < radial.size:
        raise TailMetricError("Hill requires 1<=k<n")
    threshold = float(radial[k])
    if threshold <= 0.0 or np.any(radial[:k] <= 0.0):
        raise TailMetricError("Hill requires strictly positive top order statistics")
    xi = float(np.mean(np.log(radial[:k]) - math.log(threshold)))
    if not math.isfinite(xi) or xi <= 0.0:
        raise TailMetricError("Hill log-spacing is non-positive or non-finite")
    alpha = 1.0 / xi
    return xi, alpha, threshold


def hill_grid(ordered: np.ndarray, fractions: tuple[float, ...]) -> list[dict[str, Any]]:
    """Evaluate Hill on a fixed grid, preserving invalid requests as rows."""

    radial = _validate_ordered(ordered)
    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        k = tail_count(radial.size, fraction)
        try:
            xi, alpha, threshold = hill_at_k(radial, k)
            rows.append(
                {
                    "fraction": fraction,
                    "k": k,
                    "threshold": threshold,
                    "xi": xi,
                    "alpha": alpha,
                    "valid": True,
                    "reason": "",
                }
            )
        except TailMetricError as exc:
            rows.append(
                {
                    "fraction": fraction,
                    "k": k,
                    "threshold": math.nan,
                    "xi": math.nan,
                    "alpha": math.nan,
                    "valid": False,
                    "reason": str(exc),
                }
            )
    return rows


def pickands_at_k(ordered: np.ndarray, k: int) -> tuple[float, float, float]:
    """Return Pickands' EVI, reciprocal index and the top-``k`` threshold."""

    radial = _validate_ordered(ordered)
    if k < 1 or 4 * k > radial.size:
        raise TailMetricError("Pickands requires k>=1 and 4k<=n")
    first = float(radial[k - 1])
    second = float(radial[2 * k - 1])
    fourth = float(radial[4 * k - 1])
    numerator = first - second
    denominator = second - fourth
    if numerator <= 0.0 or denominator <= 0.0:
        raise TailMetricError("Pickands order-statistic spacings must be positive")
    xi = math.log(numerator / denominator) / math.log(2.0)
    if not math.isfinite(xi):
        raise TailMetricError("Pickands estimate is non-finite")
    alpha = 1.0 / xi if xi > 0.0 else math.nan
    return xi, alpha, first


def pickands_grid(ordered: np.ndarray, fractions: tuple[float, ...]) -> list[dict[str, Any]]:
    """Evaluate Pickands on the same prespecified ``k/n`` grid as Hill."""

    radial = _validate_ordered(ordered)
    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        k = tail_count(radial.size, fraction)
        try:
            xi, alpha, threshold = pickands_at_k(radial, k)
            rows.append(
                {
                    "fraction": fraction,
                    "k": k,
                    "threshold": threshold,
                    "xi": xi,
                    "alpha": alpha,
                    "valid": True,
                    "reason": "",
                }
            )
        except TailMetricError as exc:
            rows.append(
                {
                    "fraction": fraction,
                    "k": k,
                    "threshold": math.nan,
                    "xi": math.nan,
                    "alpha": math.nan,
                    "valid": False,
                    "reason": str(exc),
                }
            )
    return rows


def gpd_at_k(ordered: np.ndarray, k: int) -> tuple[float, float, float]:
    """Fit a zero-location GPD to the top-``k`` threshold exceedances."""

    radial = _validate_ordered(ordered)
    if not 8 <= k < radial.size:
        raise TailMetricError("GPD fit requires 8<=k<n")
    threshold = float(radial[k])
    excess = radial[:k] - threshold
    if threshold < 0.0 or np.any(excess <= 0.0):
        raise TailMetricError("GPD threshold exceedances must be strictly positive")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            shape, location, scale = stats.genpareto.fit(excess, floc=0.0)
    except (ValueError, FloatingPointError, RuntimeWarning) as exc:
        raise TailMetricError("GPD maximum-likelihood fit failed") from exc
    shape, location, scale = float(shape), float(location), float(scale)
    if location != 0.0 or not math.isfinite(shape) or not math.isfinite(scale) or scale <= 0.0:
        raise TailMetricError("GPD fit returned invalid parameters")
    return shape, scale, threshold


def tail_constant_at_k(ordered: np.ndarray, k: int, alpha: float) -> tuple[float, float]:
    """Fixed-index estimator ``(k/n) * R_(k+1)**alpha`` and its logarithm."""

    radial = _validate_ordered(ordered)
    if not 1 <= k < radial.size or not math.isfinite(alpha) or alpha <= 0.0:
        raise TailMetricError("tail constant requires 1<=k<n and alpha>0")
    threshold = float(radial[k])
    if threshold <= 0.0:
        raise TailMetricError("tail-constant threshold must be positive")
    log_constant = math.log(k / radial.size) + alpha * math.log(threshold)
    if not math.isfinite(log_constant):
        raise TailMetricError("tail-constant logarithm is non-finite")
    try:
        constant = math.exp(log_constant)
    except OverflowError as exc:
        raise TailMetricError("tail constant overflows float64") from exc
    if not math.isfinite(constant):
        raise TailMetricError("tail constant is non-finite")
    return constant, log_constant


def mean_excess_sufficient(
    sample: np.ndarray, thresholds: np.ndarray
) -> list[dict[str, float | int | bool | str]]:
    """Per-threshold sufficient statistics for exact cross-seed pooling."""

    values = finite_sample(sample)
    radial = np.abs(values)
    requested = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if requested.size == 0 or np.any(~np.isfinite(requested)) or np.any(requested <= 0.0):
        raise TailMetricError("mean-excess thresholds must be finite and positive")
    rows: list[dict[str, float | int | bool | str]] = []
    for threshold in requested:
        exceedances = radial[radial > threshold] - threshold
        count = int(exceedances.size)
        total = float(np.sum(exceedances, dtype=np.float64)) if count else 0.0
        ratio = total / count / threshold if count else math.nan
        rows.append(
            {
                "threshold": float(threshold),
                "exceedance_sum": total,
                "exceedance_count": count,
                "ratio": ratio,
                "valid": count > 0,
                "reason": "" if count else "no threshold exceedance",
            }
        )
    return rows


def pooled_mean_excess(
    sums: np.ndarray,
    counts: np.ndarray,
    threshold: float,
) -> float:
    """Exactly reproduce the mean excess of the concatenated exceedances."""

    total_sums = np.asarray(sums, dtype=np.float64).reshape(-1)
    total_counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    if (
        total_sums.shape != total_counts.shape
        or not math.isfinite(threshold)
        or threshold <= 0.0
        or np.any(~np.isfinite(total_sums))
        or np.any(total_sums < 0.0)
        or np.any(total_counts < 0)
    ):
        raise TailMetricError("invalid pooled mean-excess sufficient statistics")
    count = int(total_counts.sum())
    if count == 0:
        raise TailMetricError("pooled mean excess has no exceedance")
    return float(total_sums.sum() / count / threshold)


def student_abs_quantiles(
    probabilities: np.ndarray,
    degrees_of_freedom: float,
    *,
    location: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Exact radial quantiles of ``abs(location + scale*T_nu)``."""

    beta = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if (
        beta.size == 0
        or np.any(~np.isfinite(beta))
        or np.any((beta <= 0.0) | (beta >= 1.0))
        or not math.isfinite(degrees_of_freedom)
        or degrees_of_freedom <= 0.0
        or not math.isfinite(location)
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        raise TailMetricError("invalid Student absolute-quantile parameters")
    if location == 0.0:
        return np.asarray(
            scale * stats.t.ppf((1.0 + beta) / 2.0, df=degrees_of_freedom),
            dtype=np.float64,
        )

    def radial_cdf(radius: float) -> float:
        upper = (radius - location) / scale
        lower = (-radius - location) / scale
        return float(
            stats.t.cdf(upper, df=degrees_of_freedom) - stats.t.cdf(lower, df=degrees_of_freedom)
        )

    result = np.empty_like(beta)
    for index, probability in enumerate(beta):
        upper = max(abs(location) + scale, scale)
        while radial_cdf(upper) < probability:
            upper *= 2.0
            if not math.isfinite(upper):
                raise TailMetricError("failed to bracket Student radial quantile")
        result[index] = optimize.brentq(
            lambda radius, target=probability: radial_cdf(radius) - target,
            0.0,
            upper,
            xtol=1.0e-13,
            rtol=1.0e-13,
        )
    return result


def radial_quantiles(sample: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Pinned NumPy linear quantiles of ``abs(sample)``."""

    values = finite_sample(sample)
    beta = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if beta.size == 0 or np.any(~np.isfinite(beta)) or np.any((beta <= 0.0) | (beta >= 1.0)):
        raise TailMetricError("quantile probabilities must lie in (0,1)")
    return np.asarray(np.quantile(np.abs(values), beta, method="linear"), dtype=np.float64)


def power_law_fit(
    one_minus_probability: np.ndarray,
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
    minimum_points: int = 4,
) -> PowerLawFit:
    """Fit ``log(value) = intercept + slope*log(1-beta)`` in a frozen window."""

    mass = np.asarray(one_minus_probability, dtype=np.float64).reshape(-1)
    observed = np.asarray(values, dtype=np.float64).reshape(-1)
    if mass.shape != observed.shape or not 0.0 < lower <= upper < 1.0:
        raise TailMetricError("invalid power-law fit arrays or window")
    valid = (
        np.isfinite(mass)
        & np.isfinite(observed)
        & (mass >= lower)
        & (mass <= upper)
        & (mass > 0.0)
        & (observed > 0.0)
    )
    if int(valid.sum()) < minimum_points:
        raise TailMetricError("too few positive points in the prespecified slope window")
    x = np.log(mass[valid])
    y = np.log(observed[valid])
    slope, intercept = np.polyfit(x, y, 1)
    prediction = slope * x + intercept
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else math.nan
    return PowerLawFit(float(slope), float(intercept), r_squared, int(valid.sum()))


def sas_abs_tail_constant(alpha: float) -> float:
    """Two-sided 1D tail constant for CF ``exp(-|u|**alpha)``."""

    if not math.isfinite(alpha) or not 0.0 < alpha < 2.0:
        raise TailMetricError("S-alpha-S power tail constant requires alpha in (0,2)")
    return 2.0 * math.gamma(alpha) * math.sin(math.pi * alpha / 2.0) / math.pi


def discrete_light_stable_scale(
    *,
    alpha: float,
    eta: float,
    beta: float,
    horizon: float,
    epsilon: float,
    steps: int,
) -> float:
    """Scale of the exact stable leading term of the implemented light-target EI."""

    if (
        not 1.0 < alpha < 2.0
        or eta <= 0.0
        or beta <= 0.0
        or not 0.0 < epsilon < horizon
        or steps <= 0
    ):
        raise TailMetricError("invalid stable EI scale parameters")
    h = (horizon - epsilon) / steps
    lam = eta * beta / alpha
    decay = math.exp(-lam * h)
    drift_weight = -math.expm1(-lam * h) / lam
    sigma_power = -math.expm1(-eta * beta * h)
    rho_power = 1.0
    for index in range(steps):
        forward_time = horizon - index * h
        gamma_power = -math.expm1(-beta * forward_time)
        force_linear = (1.0 + eta) * beta / alpha * (1.0 - 1.0 / gamma_power)
        multiplier = decay + drift_weight * force_linear
        rho_power = abs(multiplier) ** alpha * rho_power + sigma_power
    return rho_power ** (1.0 / alpha)


def discrete_light_vp_scale(*, beta: float, horizon: float, epsilon: float, steps: int) -> float:
    """Gaussian leading-term scale for the light-target VP EI recurrence."""

    if beta <= 0.0 or not 0.0 < epsilon < horizon or steps <= 0:
        raise TailMetricError("invalid VP EI scale parameters")
    h = (horizon - epsilon) / steps
    lam = beta / 2.0
    decay = math.exp(-lam * h)
    drift_weight = -math.expm1(-lam * h) / lam
    sigma_variance = -math.expm1(-beta * h)
    variance = 1.0
    for index in range(steps):
        forward_time = horizon - index * h
        gamma_variance = -math.expm1(-beta * forward_time)
        multiplier = decay + drift_weight * beta * (1.0 - 1.0 / gamma_variance)
        variance = multiplier * multiplier * variance + sigma_variance
    return math.sqrt(variance)


def stable_reference_thresholds(
    fractions: tuple[float, ...], *, alpha: float, scale: float
) -> np.ndarray:
    """Frozen thresholds from the exact scaled S-alpha-S radial law."""

    if scale <= 0.0 or not math.isfinite(scale):
        raise TailMetricError("stable reference scale must be finite and positive")
    beta = 1.0 - np.asarray(fractions, dtype=np.float64)
    quantiles = stats.levy_stable.ppf((1.0 + beta) / 2.0, alpha, 0.0, scale=scale)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if np.any(~np.isfinite(quantiles)) or np.any(quantiles <= 0.0):
        raise TailMetricError("stable reference quantile evaluation failed")
    return quantiles


def gaussian_reference_thresholds(fractions: tuple[float, ...], *, scale: float) -> np.ndarray:
    """Frozen thresholds from a centered Gaussian radial reference."""

    if scale <= 0.0 or not math.isfinite(scale):
        raise TailMetricError("Gaussian reference scale must be finite and positive")
    beta = 1.0 - np.asarray(fractions, dtype=np.float64)
    return np.asarray(scale * stats.norm.ppf((1.0 + beta) / 2.0), dtype=np.float64)
