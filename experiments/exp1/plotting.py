"""Deterministic, colorblind-safe publication figures for Experiment 1."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


COLORS = {
    1.2: "#0072B2",
    1.5: "#D55E00",
    3.0: "#009E73",
    "vp": "#CC79A7",
    "control": "#666666",
}
MARKERS = {1.2: "o", 1.5: "s", 3.0: "^", "vp": "D", "control": "x"}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def _maximum_steps(
    rows: Sequence[dict[str, Any]], model: str, nu: float, *, purpose: str | None = None
) -> int:
    candidates = [
        int(row["steps"])
        for row in rows
        if row["model"] == model and math.isclose(float(row["target_nu"]), nu, abs_tol=1.0e-12)
        and (purpose is None or row.get("purpose", "primary") == purpose)
    ]
    return max(candidates, default=0)


def _series(
    rows: Sequence[dict[str, Any]],
    *,
    model: str,
    nu: float,
    metric: str,
    steps: int,
    purpose: str = "primary",
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["task_kind"] in {"dynamic", "dynamic_control"}
        and row["model"] == model
        and math.isclose(float(row["target_nu"]), nu, abs_tol=1.0e-12)
        and row["metric"] == metric
        and row["sample_kind"] == "full"
        and int(row["steps"]) == steps
        and row.get("purpose", "primary") == purpose
    ]
    return sorted(selected, key=lambda row: float(row["fraction"]))


def _band(ax, x: np.ndarray, rows: Sequence[dict[str, Any]], color: str) -> None:
    low = np.asarray([float(row["q16"]) for row in rows])
    high = np.asarray([float(row["q84"]) for row in rows])
    if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
        ax.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0.0)


def _tier_mark(fig, tier: str) -> None:
    if tier != "final":
        fig.text(
            0.005,
            0.005,
            f"{tier.upper()} — descriptive only",
            ha="left",
            va="bottom",
            color="#777777",
            fontsize=6.5,
        )


def _save(fig, path: Path) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = path.with_suffix(".pdf")
    png = path.with_suffix(".png")
    metadata = {
        "Creator": "levy_experiments Experiment 1",
        "Producer": "Matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf, bbox_inches="tight", metadata=metadata)
    fig.savefig(png, bbox_inches="tight", dpi=220, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return pdf, png


def plot_main(
    summaries: Sequence[dict[str, Any]],
    figure_dir: Path,
    *,
    tier: str,
    primary_steps: int | None = None,
) -> tuple[Path, Path]:
    """Tail index, target-specific quantile ratio, and tail constant."""

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35), constrained_layout=True)
    for nu in (1.2, 1.5, 3.0):
        steps = primary_steps or _maximum_steps(summaries, "stable", nu, purpose="primary")
        rows = _series(
            summaries,
            model="stable",
            nu=nu,
            metric="hill_alpha",
            steps=steps,
        )
        if rows:
            x = np.asarray([float(row["fraction"]) for row in rows])
            y = np.asarray([float(row["median"]) for row in rows])
            color = COLORS[nu]
            axes[0].plot(x, y, marker=MARKERS[nu], color=color, label=rf"Stable, $\nu={nu:g}$")
            _band(axes[0], x, rows, color)
    axes[0].axhline(1.5, color="black", linestyle="--", linewidth=1.0, label=r"$\alpha=1.5$")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"upper fraction $k/n$")
    axes[0].set_ylabel(r"Hill tail-index estimate $\widehat\alpha$")
    axes[0].set_title("(a) Tail index")

    for model, nu in (("stable", 1.2), ("stable", 1.5), ("stable", 3.0), ("vp", 1.5)):
        steps = primary_steps or _maximum_steps(summaries, model, nu, purpose="primary")
        rows = _series(
            summaries,
            model=model,
            nu=nu,
            metric="mass_ratio",
            steps=steps,
        )
        if not rows:
            continue
        x = np.asarray([float(row["fraction"]) for row in rows])
        y = np.asarray([float(row["median"]) for row in rows])
        key: float | str = "vp" if model == "vp" else nu
        label = r"VP, $\nu=1.5$" if model == "vp" else rf"Stable, $\nu={nu:g}$"
        axes[1].plot(
            x,
            y,
            marker=MARKERS[key],
            color=COLORS[key],
            linestyle=":" if model == "vp" else "-",
            label=label,
        )
        _band(axes[1], x, rows, COLORS[key])
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xlabel(r"survival mass $1-\kappa$")
    axes[1].set_ylabel(r"$\widehat q_\kappa(|Y|)/q_\kappa(|T_\nu|)$")
    axes[1].set_title("(b) Target-specific coverage")

    for nu in (1.2, 1.5, 3.0):
        steps = primary_steps or _maximum_steps(summaries, "stable", nu, purpose="primary")
        rows = _series(
            summaries,
            model="stable",
            nu=nu,
            metric="tail_constant",
            steps=steps,
        )
        rows = [row for row in rows if float(row["expected"]) > 0.0]
        if rows:
            x = np.asarray([float(row["fraction"]) for row in rows])
            y = np.asarray([float(row["median"]) / float(row["expected"]) for row in rows])
            axes[2].plot(
                x,
                y,
                marker=MARKERS[nu],
                color=COLORS[nu],
                label=rf"$\nu={nu:g}$",
            )
            low = np.asarray([float(row["q16"]) / float(row["expected"]) for row in rows])
            high = np.asarray([float(row["q84"]) / float(row["expected"]) for row in rows])
            axes[2].fill_between(x, low, high, color=COLORS[nu], alpha=0.16)
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_xscale("log")
    axes[2].set_xlabel(r"upper fraction $k/n$")
    axes[2].set_ylabel(r"$\widehat c/c_{N,h}$")
    axes[2].set_title("(c) EI tail constant")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    for ax in axes:
        ax.grid(True, which="major", color="#DDDDDD", linewidth=0.5)
    _tier_mark(fig, tier)
    return _save(fig, figure_dir / "experiment1_main")


def plot_quantile_ratio(
    summaries: Sequence[dict[str, Any]],
    figure_dir: Path,
    *,
    tier: str,
    primary_steps: int | None = None,
) -> tuple[Path, Path]:
    """Render the manuscript's standalone target-coverage panel."""

    _style()
    fig, axis = plt.subplots(figsize=(2.5, 2.55), constrained_layout=True)
    for model, nu in (("stable", 1.2), ("stable", 1.5), ("stable", 3.0), ("vp", 1.5)):
        steps = primary_steps or _maximum_steps(summaries, model, nu, purpose="primary")
        rows = _series(
            summaries,
            model=model,
            nu=nu,
            metric="mass_ratio",
            steps=steps,
        )
        if not rows:
            continue
        x = np.asarray([float(row["fraction"]) for row in rows])
        y = np.asarray([float(row["median"]) for row in rows])
        key: float | str = "vp" if model == "vp" else nu
        label = r"VP, $\nu=1.5$" if model == "vp" else rf"Stable, $\nu={nu:g}$"
        axis.plot(
            x,
            y,
            marker=MARKERS[key],
            color=COLORS[key],
            linestyle=":" if model == "vp" else "-",
            label=label,
        )
        _band(axis, x, rows, COLORS[key])
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.invert_xaxis()
    axis.set_xlabel(r"survival mass $1-\kappa$")
    axis.set_ylabel(r"$\widehat q_\kappa(|Y|)/q_\kappa(|T_\nu|)$")
    axis.set_title("Target-specific coverage")
    axis.grid(True, which="major", color="#DDDDDD", linewidth=0.5)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    _tier_mark(fig, tier)
    return _save(fig, figure_dir / "experiment1_quantile_ratio")


def _pooled_series(
    rows: Sequence[dict[str, Any]], model: str, nu: float, primary_steps: int | None
) -> list[dict[str, Any]]:
    steps = primary_steps or _maximum_steps(rows, model, nu, purpose="primary")
    return sorted(
        [
            row
            for row in rows
            if row["model"] == model
            and math.isclose(float(row["target_nu"]), nu, abs_tol=1.0e-12)
            and row["sample_kind"] == "full"
            and int(row["steps"]) == steps
            and row.get("purpose", "primary") == "primary"
            and bool(row.get("valid", True))
            and math.isfinite(float(row["threshold_fraction"]))
            and float(row["threshold_fraction"]) > 0.0
            and math.isfinite(float(row["pooled_ratio"]))
        ],
        key=lambda row: float(row["threshold_fraction"]),
    )


def _primary_slope_candidate(
    rows: Sequence[dict[str, Any]], nu: float, primary_steps: int | None
) -> dict[str, Any] | None:
    """Select the prespecified high-statistics branch for a mass-ratio slope."""

    steps = primary_steps or _maximum_steps(rows, "stable", nu, purpose="primary")
    candidates = [
        row
        for row in rows
        if row["model"] == "stable"
        and row["window"] == "primary"
        and row["sample_kind"] == "full"
        and row.get("purpose", "primary") == "primary"
        and int(row["steps"]) == steps
        and math.isclose(float(row["target_nu"]), nu, abs_tol=1.0e-12)
    ]
    if len(candidates) > 1:
        raise ValueError(f"ambiguous primary slope summary for nu={nu:g}, N={steps}")
    return candidates[0] if candidates else None


def _primary_control_rows(
    rows: Sequence[dict[str, Any]], primary_steps: int | None
) -> list[dict[str, Any]]:
    """Keep only controls from the prespecified primary simulation branch."""

    selected: list[dict[str, Any]] = []
    for row in rows:
        if (
            row["metric"] not in {"control_w1", "control_ecf_rmse"}
            or row["sample_kind"] != "full"
            or float(row["auxiliary"]) <= 0.0
            or row.get("purpose", "primary") != "primary"
        ):
            continue
        steps = primary_steps or _maximum_steps(
            rows, str(row["model"]), float(row["target_nu"]), purpose="primary"
        )
        if int(row["steps"]) == steps:
            selected.append(row)
    return selected


def _refinement_hill_series(
    rows: Sequence[dict[str, Any]], nu: float
) -> list[dict[str, Any]]:
    """Return one refinement-only Hill summary per numerical step count."""

    candidates = [
        row
        for row in rows
        if row["model"] == "stable"
        and row["metric"] == "hill_alpha"
        and row["sample_kind"] == "full"
        and row.get("purpose", "primary") == "refinement"
        and math.isclose(float(row["fraction"]), 0.006, abs_tol=1.0e-12)
        and math.isclose(float(row["target_nu"]), nu, abs_tol=1.0e-12)
    ]
    candidates.sort(key=lambda row: int(row["steps"]))
    step_counts = [int(row["steps"]) for row in candidates]
    if len(step_counts) != len(set(step_counts)):
        raise ValueError(f"duplicate refinement Hill summary for nu={nu:g}")
    return candidates


def plot_diagnostics(
    summaries: Sequence[dict[str, Any]],
    pooled_mean: Sequence[dict[str, Any]],
    slope_summaries: Sequence[dict[str, Any]],
    metric_rows: Sequence[dict[str, Any]],
    figure_dir: Path,
    *,
    tier: str,
    primary_steps: int | None = None,
) -> tuple[Path, Path]:
    """Estimator stability, mean excess, marginal controls, and refinement."""

    _style()
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.0), constrained_layout=True)
    has_mean_excess = False
    for nu in (1.2, 1.5, 3.0):
        rows = _pooled_series(pooled_mean, "stable", nu, primary_steps)
        if rows:
            x = np.asarray([float(row["threshold_fraction"]) for row in rows])
            y = np.asarray([float(row["pooled_ratio"]) for row in rows])
            axes[0, 0].plot(x, y, marker=MARKERS[nu], color=COLORS[nu], label=rf"$\nu={nu:g}$")
            has_mean_excess = True
    axes[0, 0].axhline(2.0, color="black", linestyle="--", linewidth=1.0)
    if has_mean_excess:
        axes[0, 0].set_xscale("log")
    else:
        axes[0, 0].text(
            0.5,
            0.5,
            "not estimable at this tier",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )
    axes[0, 0].set_xlabel("reference survival fraction")
    axes[0, 0].set_ylabel(r"pooled $\widehat e(u)/u$")
    axes[0, 0].set_title("(a) Mean excess (descriptive)")

    positions = np.arange(3, dtype=np.float64)
    for index, nu in enumerate((1.2, 1.5, 3.0)):
        row = _primary_slope_candidate(slope_summaries, nu, primary_steps)
        if row is not None:
            axes[0, 1].errorbar(
                positions[index],
                float(row["median"]),
                yerr=np.asarray(
                    [
                        [float(row["median"]) - float(row["q16"])],
                        [float(row["q84"]) - float(row["median"])],
                    ]
                ),
                marker=MARKERS[nu],
                color=COLORS[nu],
                capsize=2,
            )
        axes[0, 1].plot(
            positions[index],
            1.0 / nu - 1.0 / 1.5,
            marker="_",
            markersize=12,
            color="black",
        )
    axes[0, 1].set_xticks(positions, ["1.2", "1.5", "3"])
    axes[0, 1].set_xlabel(r"Student index $\nu$")
    axes[0, 1].set_ylabel(r"slope of $\log \mathcal{M}_\nu$")
    axes[0, 1].set_title("(b) Quantile-ratio slope")

    control = _primary_control_rows(metric_rows, primary_steps)
    for metric, marker, linestyle in (
        ("control_w1", "o", "-"),
        ("control_ecf_rmse", "s", "--"),
    ):
        grouped: dict[tuple[str, float, float], list[float]] = {}
        for row in control:
            if row["metric"] != metric:
                continue
            key = (str(row["model"]), float(row["target_nu"]), float(row["threshold"]))
            grouped.setdefault(key, []).append(float(row["value"]) / float(row["auxiliary"]))
        by_model: dict[tuple[str, float], list[tuple[float, float]]] = {}
        for (model, nu, time_value), values in grouped.items():
            by_model.setdefault((model, nu), []).append((time_value, float(np.median(values))))
        for (model, nu), values in by_model.items():
            values.sort()
            axes[1, 0].plot(
                [item[0] for item in values],
                [item[1] for item in values],
                marker=marker,
                linestyle=linestyle,
                color=COLORS["vp" if model == "vp" else nu],
                label=f"{model}, nu={nu:g}, {metric.removeprefix('control_')}",
            )
    axes[1, 0].axhline(2.0, color="black", linestyle=":", linewidth=1.0)
    axes[1, 0].set_xlabel("forward checkpoint time")
    axes[1, 0].set_ylabel("numerical / exact-exact envelope")
    axes[1, 0].set_title("(c) Marginal-control diagnostics")
    control_handles, control_labels = axes[1, 0].get_legend_handles_labels()
    if control_handles:
        axes[1, 0].legend(
            control_handles,
            control_labels,
            fontsize=5.2,
            ncol=2,
            frameon=False,
            loc="upper right",
        )

    for nu in (1.2, 1.5, 3.0):
        candidates = _refinement_hill_series(summaries, nu)
        if candidates:
            axes[1, 1].plot(
                [int(row["steps"]) for row in candidates],
                [float(row["median"]) for row in candidates],
                marker=MARKERS[nu],
                color=COLORS[nu],
                label=rf"$\nu={nu:g}$",
            )
    axes[1, 1].axhline(1.5, color="black", linestyle="--", linewidth=1.0)
    axes[1, 1].set_xlabel("EI steps $N$")
    axes[1, 1].set_ylabel(r"refinement $\widehat\alpha$")
    axes[1, 1].set_title("(d) Discretization check")

    for ax in axes.flat:
        ax.grid(True, which="major", color="#DDDDDD", linewidth=0.5)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    if handles:
        axes[1, 1].legend(handles, labels, frameon=False)
    _tier_mark(fig, tier)
    return _save(fig, figure_dir / "experiment1_diagnostics")


def write_caption(path: Path, *, tier: str) -> Path:
    text = (
        "**Proposed caption.** Target-specific tail diagnostics at $T=2$, "
        "$\\varepsilon=0.5$, $\\alpha=1.5$, and $\\eta=0.5$. Bands show the "
        "16th-84th percentiles across independent seeds. The stable sampler uses "
        "a separately tabulated mathematical score for each Student target; the VP "
        "baseline is shown only at the index-matched target. Quantile panels use "
        "survival mass $1-\\kappa$, where $\\kappa$ is the radial quantile level. "
        "Tail-index, quantile-ratio, "
        "and discrete-EI tail-constant diagnostics are finite-sample numerical "
        "illustrations, not evidence of process equality or a proof of asymptotic "
        f"equivalence. This artifact was generated from the **{tier}** tier.\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


__all__ = ["plot_diagnostics", "plot_main", "plot_quantile_ratio", "write_caption"]
