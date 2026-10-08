"""Deterministic, publication-oriented plots for Experiment 2 aggregates.

This module is deliberately downstream-only: every curve is built from summary
rows produced by :mod:`experiments.exp2.analysis`; it cannot launch or import a
simulator.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from levy_experiments.storage import write_or_verify_bytes

from .analysis import MODELS, ORDERS, PRIMARY_EPSILON, PRIMARY_T, Exp2ArtifactError

COLORS = {
    "D": "#0072B2",
    "I": "#E69F00",
    "E": "#009E73",
    "F": "#777777",
    "A_exact": "#777777",
    "B_exact": "#999999",
    "C_exact": "#BBBBBB",
}
MARKERS = {"D": "o", "I": "s", "E": "^", "F": "."}
LINESTYLES = {"D": "-", "I": "--", "E": "-.", "F": ":"}
LABELS = {
    "D": r"$D$: numerical exact-init vs exact",
    "I": r"$I$: initialization proxy",
    "E": r"$E$: total",
    "F": r"$F$: exact--exact MC baseline",
    "A_exact": "exact A vs analytic CF",
    "B_exact": "exact B vs analytic CF",
    "C_exact": "exact C vs analytic CF",
}
MARKERS.update({"A_exact": ".", "B_exact": "x", "C_exact": "+"})
LINESTYLES.update({"A_exact": ":", "B_exact": ":", "C_exact": ":"})


def _select(rows: Iterable[dict[str, Any]], **criteria: Any) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if all(row.get(name) == value for name, value in criteria.items())
    ]


def _ordered_series(
    rows: Sequence[dict[str, Any]], *, x_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordered = sorted(rows, key=lambda row: float(row[x_name]))
    x = np.asarray([float(row[x_name]) for row in ordered], dtype=np.float64)
    lower = np.asarray([float(row["q16"]) for row in ordered], dtype=np.float64)
    median = np.asarray([float(row["median"]) for row in ordered], dtype=np.float64)
    upper = np.asarray([float(row["q84"]) for row in ordered], dtype=np.float64)
    if x.size == 0:
        raise Exp2ArtifactError("cannot render an empty summary series")
    if np.unique(x).size != x.size:
        raise Exp2ArtifactError(f"duplicate {x_name} values in plotted summary series")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(lower + median + upper)):
        raise Exp2ArtifactError("non-finite value in plotted summary series")
    if np.any(x <= 0.0) or np.any(lower <= 0.0):
        raise Exp2ArtifactError("log-scale Experiment 2 plots require strictly positive values")
    return x, lower, median, upper


def _draw_series(axis, rows: Sequence[dict[str, Any]], *, component: str, x_name: str) -> None:
    x, lower, median, upper = _ordered_series(rows, x_name=x_name)
    axis.fill_between(x, lower, upper, color=COLORS[component], alpha=0.15, linewidth=0)
    axis.plot(
        x,
        median,
        color=COLORS[component],
        linestyle=LINESTYLES[component],
        marker=MARKERS[component],
        markersize=3.5,
        label=LABELS[component],
    )


def _save_figure_bytes(figure, suffix: str) -> bytes:
    descriptor, temporary_name = tempfile.mkstemp(suffix=suffix)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if suffix == ".pdf":
            figure.savefig(
                temporary,
                format="pdf",
                bbox_inches="tight",
                metadata={
                    "Creator": "levy-experiments-exp2",
                    "Producer": "levy-experiments-exp2",
                    "CreationDate": None,
                    "ModDate": None,
                },
            )
        elif suffix == ".png":
            figure.savefig(
                temporary,
                format="png",
                dpi=220,
                bbox_inches="tight",
                metadata={"Software": "levy-experiments-exp2"},
            )
        else:
            raise Exp2ArtifactError(f"unsupported figure suffix: {suffix}")
        return temporary.read_bytes()
    finally:
        temporary.unlink(missing_ok=True)


def _setup_matplotlib(output_dir: Path):
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    cache = output_dir / ".matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "legend.fontsize": 7,
            "lines.linewidth": 1.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    return plt


def _render_wp_figure(plt, summaries: Sequence[dict[str, Any]]):
    wasserstein = _select(
        summaries,
        metric="wasserstein",
        sample_kind="full",
        horizon=PRIMARY_T,
    )
    if not wasserstein:
        raise Exp2ArtifactError("no full-sample Wasserstein summaries to plot")
    terminal_time = min(float(row["forward_time"]) for row in wasserstein)
    figure, axes = plt.subplots(2, 2, figsize=(7.05, 5.2), constrained_layout=True)
    for model_index, model in enumerate(MODELS):
        for order_index, order in enumerate(ORDERS):
            axis = axes[model_index, order_index]
            plotted_h: list[float] = []
            for component in ("D", "I", "E", "F"):
                members = _select(
                    wasserstein,
                    model=model,
                    forward_time=terminal_time,
                    order=order,
                    component=component,
                )
                if not members:
                    raise Exp2ArtifactError(
                        f"missing {model}, p={order:g}, component {component} summaries"
                    )
                plotted_h.extend(float(row["h"]) for row in members)
                _draw_series(axis, members, component=component, x_name="h")
            axis.set_xscale("log")
            tick_values = sorted(set(plotted_h))
            axis.set_xticks(tick_values)
            axis.set_xticklabels([format(value, ".3g") for value in tick_values])
            axis.set_xticks([], minor=True)
            axis.set_yscale("log")
            axis.invert_xaxis()
            axis.grid(alpha=0.22, which="both")
            axis.set_xlabel(r"step size $h=(T-\varepsilon)/N$")
            axis.set_ylabel(rf"empirical $W_{{{order:g}}}$")
            model_label = r"stable ($\alpha=1.5,\eta=0.5$)" if model == "stable" else "VP"
            axis.set_title(f"{model_label}; terminal t={terminal_time:g}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", frameon=False, ncols=4)
    figure.suptitle("Student-t4 target: error decomposition and fixed-horizon refinement", y=1.02)
    return figure


def _finest_per_horizon(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[float, dict[str, Any]] = {}
    for row in rows:
        horizon = float(row["horizon"])
        current = selected.get(horizon)
        if current is None or float(row["h"]) < float(current["h"]):
            selected[horizon] = row
    return list(selected.values())


def _render_horizon_figure(plt, summaries: Sequence[dict[str, Any]]):
    wasserstein = _select(
        summaries,
        metric="wasserstein",
        sample_kind="full",
        forward_time=PRIMARY_EPSILON,
    )
    if not wasserstein:
        raise Exp2ArtifactError("no terminal horizon-sweep summaries to plot")
    figure, axes = plt.subplots(2, 2, figsize=(7.05, 5.2), constrained_layout=True)
    for model_index, model in enumerate(MODELS):
        for order_index, order in enumerate(ORDERS):
            axis = axes[model_index, order_index]
            plotted_h: list[float] = []
            for component in ("D", "I", "E", "F"):
                members = _finest_per_horizon(
                    _select(
                        wasserstein,
                        model=model,
                        order=order,
                        component=component,
                    )
                )
                if not members:
                    raise Exp2ArtifactError(
                        f"missing horizon series for {model}, p={order:g}, {component}"
                    )
                plotted_h.extend(float(row["h"]) for row in members)
                _draw_series(axis, members, component=component, x_name="horizon")
            axis.set_yscale("log")
            axis.grid(alpha=0.22, which="both")
            axis.set_xlabel(r"forward horizon $T$")
            axis.set_ylabel(rf"terminal empirical $W_{{{order:g}}}$")
            model_label = r"stable ($\alpha=1.5,\eta=0.5$)" if model == "stable" else "VP"
            relative_h_span = (
                (max(plotted_h) - min(plotted_h)) / np.mean(plotted_h)
                if len(plotted_h) > 1
                else 0.0
            )
            axis.set_title(f"{model_label}; finest h span={100 * relative_h_span:.1f}%")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", frameon=False, ncols=4)
    figure.suptitle("Student-t4 target: initialization effect versus forward horizon", y=1.02)
    return figure


def _render_diagnostics_figure(plt, summaries: Sequence[dict[str, Any]]):
    figure, axes = plt.subplots(2, 2, figsize=(7.05, 5.35), constrained_layout=True)
    refinement = [
        row
        for row in summaries
        if math.isclose(float(row["horizon"]), PRIMARY_T, abs_tol=1.0e-12)
    ]
    all_times = [float(row["forward_time"]) for row in refinement]
    if not all_times:
        raise Exp2ArtifactError("no Experiment 2 summaries to plot")
    terminal_time = min(all_times)
    finest_steps = max(int(row["steps"]) for row in refinement)
    for model_index, model in enumerate(MODELS):
        axis = axes[0, model_index]
        plotted_h: list[float] = []
        for component in ("D", "E", "A_exact", "B_exact", "C_exact"):
            members = _select(
                refinement,
                model=model,
                metric="ecf_rmse",
                component=component,
                sample_kind="full",
                forward_time=terminal_time,
            )
            if not members:
                raise Exp2ArtifactError(f"missing terminal ECF summaries for {model}/{component}")
            plotted_h.extend(float(row["h"]) for row in members)
            _draw_series(axis, members, component=component, x_name="h")
        axis.set_xscale("log")
        tick_values = sorted(set(plotted_h))
        axis.set_xticks(tick_values)
        axis.set_xticklabels([format(value, ".3g") for value in tick_values])
        axis.set_xticks([], minor=True)
        axis.set_yscale("log")
        axis.invert_xaxis()
        axis.grid(alpha=0.22, which="both")
        axis.set_xlabel(r"step size $h$")
        axis.set_ylabel("weighted ECF RMSE")
        axis.set_title(f"{model}: exact-CF check")

        axis = axes[1, model_index]
        for component in ("D", "I", "E", "F"):
            members = [
                row
                for row in refinement
                if row["model"] == model
                and row["metric"] == "wasserstein"
                and row["component"] == component
                and row["order"] == 1.25
                and row["steps"] == finest_steps
                and row["forward_time"] == terminal_time
                and row["sample_kind"] in {"nested", "full"}
            ]
            if not members:
                raise Exp2ArtifactError(f"missing subsampling summaries for {model}/{component}")
            _draw_series(axis, members, component=component, x_name="sample_size")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(alpha=0.22, which="both")
        axis.set_xlabel("particles retained (nested subsets)")
        axis.set_ylabel(r"empirical $W_{1.25}$")
        axis.set_title(f"{model}: sample-size sensitivity at N={finest_steps}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", frameon=False, ncols=5)
    figure.suptitle("Student-t4 target: exact characteristic function and subsampling", y=1.02)
    return figure


def render_figures(
    summaries: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    resume: bool,
    tier: str | None = None,
) -> dict[str, str]:
    """Render deterministic PDF/PNG figures from aggregate rows only."""

    if tier not in {None, "smoke", "pilot", "final"}:
        raise Exp2ArtifactError(f"unknown rendering tier: {tier!r}")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plt = _setup_matplotlib(destination)
    figures = {
        "experiment2_wp": _render_wp_figure(plt, summaries),
        "experiment2_horizon": _render_horizon_figure(plt, summaries),
        "experiment2_diagnostics": _render_diagnostics_figure(plt, summaries),
    }
    tier_label = {
        "smoke": "SMOKE - pipeline validation only",
        "pilot": "PILOT",
    }.get(tier)
    if tier_label is not None:
        for figure in figures.values():
            figure.text(
                0.995,
                1.055,
                tier_label,
                ha="right",
                va="top",
                color="#777777",
                fontsize=6.5,
            )
    outputs: dict[str, str] = {}
    for stem, figure in figures.items():
        try:
            for suffix in (".pdf", ".png"):
                payload = _save_figure_bytes(figure, suffix)
                path = destination / f"{stem}{suffix}"
                write_or_verify_bytes(path, payload, resume=resume)
                outputs[f"{stem}{suffix}"] = str(path)
        finally:
            plt.close(figure)
    return outputs
