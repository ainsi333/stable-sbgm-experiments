"""Deterministic, publication-oriented rendering for Experiment 3."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .authority import authority_manifest
from .config import Experiment3Config
from .errors import ArtifactError
from .storage import RunLayout, atomic_write_json, write_or_verify_bytes

COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
MARKERS = ("o", "s", "^", "D")
LINESTYLES = ("-", "--", "-.", ":")


def _matches(row: dict[str, Any], **criteria: Any) -> bool:
    return all(row.get(name) == value for name, value in criteria.items())


def _quantiles(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return tuple(float(value) for value in np.quantile(array, (0.16, 0.5, 0.84)))


def _save_figure_bytes(figure, suffix: str) -> bytes:
    descriptor, name = tempfile.mkstemp(suffix=suffix)
    os.close(descriptor)
    path = Path(name)
    try:
        if suffix == ".pdf":
            figure.savefig(
                path,
                format="pdf",
                bbox_inches="tight",
                metadata={"Creator": "levy-experiments", "CreationDate": None, "ModDate": None},
            )
        else:
            figure.savefig(
                path,
                format="png",
                dpi=220,
                bbox_inches="tight",
                metadata={"Software": "levy-experiments"},
            )
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def render_experiment3_figure(
    config: Experiment3Config,
    layout: RunLayout,
    rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, str]:
    """Render the four-panel Experiment 3 figure from stored task results only."""

    if config.experiment.publication_scale and diagnostics.get("publication_gate_pass") is not True:
        failures = diagnostics.get(
            "publication_gate_failures", ["unknown publication-gate failure"]
        )
        raise ArtifactError(
            "publication figure rendering refused: " + "; ".join(str(item) for item in failures)
        )

    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    matplotlib_cache = layout.run_dir / ".matplotlib-cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
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
    figure, axes = plt.subplots(2, 2, figsize=(7.05, 5.55), constrained_layout=True)
    eta_values = config.experiment.etas

    axis = axes[0, 0]
    for index, lag in enumerate(config.stationary.lags):
        medians: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for eta in eta_values:
            values = [
                float(row["value"])
                for row in rows
                if _matches(
                    row,
                    section="stationary",
                    method="exact_transition",
                    metric="marginal_ecf_rmse_terminal",
                    eta=eta,
                    lag=lag,
                )
            ]
            q16, median, q84 = _quantiles(values)
            medians.append(median)
            lower.append(median - q16)
            upper.append(q84 - median)
        axis.errorbar(
            eta_values,
            medians,
            yerr=np.asarray((lower, upper)),
            color=COLORS[index],
            linestyle=LINESTYLES[index],
            marker=MARKERS[index],
            capsize=2,
            label=rf"$\Delta={lag:g}$",
        )
    axis.set_yscale("log")
    axis.set_xlabel(r"backward-noise parameter $\eta$")
    axis.set_ylabel("marginal ECF RMSE")
    axis.set_title("(a) Same stationary marginals")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, ncols=3)

    axis = axes[0, 1]
    selected_lag = max(config.stationary.lags)
    empirical_medians: list[float] = []
    empirical_lower: list[float] = []
    empirical_upper: list[float] = []
    analytic_values: list[float] = []
    true_reverse = None
    for eta in eta_values:
        members = [
            row
            for row in rows
            if _matches(
                row,
                section="stationary",
                method="exact_transition",
                metric="joint_probe_real",
                eta=eta,
                lag=selected_lag,
            )
        ]
        q16, median, q84 = _quantiles([float(row["value"]) for row in members])
        empirical_medians.append(median)
        empirical_lower.append(median - q16)
        empirical_upper.append(q84 - median)
        analytic_values.append(float(members[0]["analytic_value"]))
        true_reverse = float(members[0]["floor_value"])
    axis.plot(
        eta_values,
        analytic_values,
        color="#000000",
        linestyle="-",
        label="Popov analytic CF",
    )
    axis.errorbar(
        eta_values,
        empirical_medians,
        yerr=np.asarray((empirical_lower, empirical_upper)),
        color=COLORS[0],
        linestyle="none",
        marker=MARKERS[0],
        capsize=2,
        label="empirical CF",
    )
    axis.axhline(
        true_reverse,
        color="#666666",
        linestyle="--",
        label="true reversed pair",
    )
    axis.set_xlabel(r"backward-noise parameter $\eta$")
    axis.set_ylabel(r"$\mathrm{Re}\,\Phi(1,0.5)$")
    axis.set_title(rf"(b) Different stationary pairs ($\Delta={selected_lag:g}$)")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)

    finest_steps = max(config.nonstationary.steps)
    axis = axes[1, 0]
    forward_times = (*config.nonstationary.checkpoints_forward, config.experiment.epsilon)
    backward_times = np.asarray(
        [config.experiment.horizon - value for value in forward_times], dtype=np.float64
    )
    for index, eta in enumerate(eta_values):
        medians = []
        lower = []
        upper = []
        for forward_time in forward_times:
            values = [
                float(row["value"])
                for row in rows
                if _matches(
                    row,
                    section="nonstationary_marginal",
                    method="popov_ei",
                    metric="w1_to_exact_reference",
                    steps=finest_steps,
                    eta=eta,
                    forward_time=forward_time,
                )
            ]
            q16, median, q84 = _quantiles(values)
            medians.append(median)
            lower.append(median - q16)
            upper.append(q84 - median)
        axis.errorbar(
            backward_times,
            medians,
            yerr=np.asarray((lower, upper)),
            color=COLORS[index],
            linestyle=LINESTYLES[index],
            marker=MARKERS[index],
            capsize=2,
            label=rf"valid $\eta={eta:g}$",
        )
    floor_medians = []
    hybrid_medians = []
    for forward_time in forward_times:
        valid_members = [
            row
            for row in rows
            if _matches(
                row,
                section="nonstationary_marginal",
                method="popov_ei",
                metric="w1_to_exact_reference",
                steps=finest_steps,
                eta=eta_values[0],
                forward_time=forward_time,
            )
        ]
        floor_medians.append(float(np.median([float(row["floor_value"]) for row in valid_members])))
        hybrid_members = [
            row
            for row in rows
            if _matches(
                row,
                section="nonstationary_marginal",
                method="hybrid_ei_invalid",
                metric="w1_to_exact_reference",
                steps=finest_steps,
                forward_time=forward_time,
            )
        ]
        hybrid_medians.append(float(np.median([float(row["value"]) for row in hybrid_members])))
    axis.plot(
        backward_times,
        floor_medians,
        color="#777777",
        linestyle=":",
        marker=".",
        label="exact-exact MC baseline",
    )
    axis.plot(
        backward_times,
        hybrid_medians,
        color="#D55E00",
        linestyle="--",
        marker="X",
        label="invalid hybrid",
    )
    axis.set_yscale("log")
    axis.set_xlabel(r"backward time $\tau$")
    axis.set_ylabel(r"empirical marginal $W_1$")
    axis.set_title(rf"(c) Three-atom target ($N={finest_steps}$)")
    axis.grid(alpha=0.22)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        frameon=False,
        ncols=3,
    )

    axis = axes[1, 1]
    later_time, earlier_time = forward_times[0], forward_times[1]
    medians = []
    lower = []
    upper = []
    true_value = None
    for eta in eta_values:
        members = [
            row
            for row in rows
            if _matches(
                row,
                section="nonstationary_joint",
                method="popov_ei",
                metric="joint_probe_real",
                steps=finest_steps,
                eta=eta,
                pair_later_time=later_time,
                pair_earlier_time=earlier_time,
            )
        ]
        q16, median, q84 = _quantiles([float(row["value"]) for row in members])
        medians.append(median)
        lower.append(median - q16)
        upper.append(q84 - median)
        true_value = float(members[0]["analytic_value"])
    axis.errorbar(
        eta_values,
        medians,
        yerr=np.asarray((lower, upper)),
        color=COLORS[2],
        linestyle="-",
        marker=MARKERS[2],
        capsize=2,
        label="Popov pairs",
    )
    axis.axhline(
        true_value,
        color="#666666",
        linestyle="--",
        label="true reversed pair",
    )
    axis.set_xlabel(r"backward-noise parameter $\eta$")
    axis.set_ylabel(r"$\mathrm{Re}\,\Phi(1,0.5)$")
    axis.set_title(rf"(d) Nonstationary pair ($t={later_time:g}\to{earlier_time:g}$)")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)

    pdf_bytes = _save_figure_bytes(figure, ".pdf")
    png_bytes = _save_figure_bytes(figure, ".png")
    plt.close(figure)
    layout.figure_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = layout.figure_dir / "experiment3.pdf"
    png_path = layout.figure_dir / "experiment3.png"
    write_or_verify_bytes(pdf_path, pdf_bytes, resume=resume)
    write_or_verify_bytes(png_path, png_bytes, resume=resume)
    caption = (
        "**Experiment 3 — matching marginals do not imply matching path laws.** "
        "(a) In the stationary S1.5S case, all exact Popov transitions retain the same "
        "marginal characteristic function; bars show the 16th-84th percentile seed spread. "
        "(b) Their joint characteristic functions vary with eta and generally differ from "
        "the true time-reversed forward pair. (c) For the centered three-atom target, the "
        "valid EI samplers are compared with independent exact-marginal samples in W1; the "
        "red cross curve is the deliberately invalid drift/noise hybrid and is excluded from "
        "valid-method claims. (d) The nonstationary joint statistic likewise depends on eta. "
        "The exact-exact curve is a finite-sample Monte Carlo baseline, not a bound "
        "or a subtracted bias.\n"
    ).encode()
    write_or_verify_bytes(layout.figure_dir / "caption.md", caption, resume=resume)
    manifest = {
        "authority": authority_manifest(),
        "config_hash": config.resolved_hash,
        "pdf": {"path": pdf_path.name, "sha256": hashlib.sha256(pdf_bytes).hexdigest()},
        "png": {"path": png_path.name, "sha256": hashlib.sha256(png_bytes).hexdigest()},
        "source": "aggregates/metrics_per_seed.csv",
        "styles": {
            "colors": list(COLORS),
            "linestyles": list(LINESTYLES),
            "markers": list(MARKERS),
        },
    }
    manifest_path = layout.figure_dir / "manifest.json"
    if manifest_path.exists():
        existing = manifest_path.read_text(encoding="utf-8")
        expected = __import__("json").dumps(manifest, indent=2, sort_keys=True) + "\n"
        if existing != expected:
            raise ArtifactError("recomputed figure manifest differs from existing output")
    else:
        atomic_write_json(manifest_path, manifest)
    return {"pdf": str(pdf_path), "png": str(png_path)}
