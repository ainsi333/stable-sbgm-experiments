"""Run, validate, and plot the deterministic Experiment 4 analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import sys
import tempfile
import time
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from levy_experiments.errors import ArtifactError, ConfigurationError, ExperimentError
from levy_experiments.storage import write_or_verify_bytes

from .computation import score_table_ct_proxy, student_denoiser_profile, truncated_values
from .config import Experiment4Config, SpectralSettings, load_config
from .source import SourceTables, validate_source

MODELS = ("stable", "vp")
MODEL_LABELS = {
    "stable": r"SP-SDE ($\alpha=1.5$)",
    "vp": "VP-SDE (Brownian)",
}
MODEL_COLORS = {"stable": "#0072B2", "vp": "#D55E00"}
MODEL_MARKERS = {"stable": "o", "vp": "s"}
CURVE_FIELDS = (
    "model",
    "time",
    "ct_truncated",
    "maximizer_abs_x",
    "boundary_abs_derivative",
    "min_density",
    "cutoff_characteristic",
    "imaginary_residual",
    "brownian_reference",
)
DIAGNOSTIC_FIELDS = (
    "model",
    "time",
    "ct_main",
    "ct_refined",
    "resolution_relative_error",
    "maximizer_abs_x",
    "boundary_abs_derivative",
    "boundary_fraction",
    "brownian_reference",
    "vp_boundary_reference_relative_error",
    "domain_ct_json",
    "domain_relative_spread",
    "source_main_ct",
    "source_hires_ct",
    "source_hires_relative_error",
    "source_main_hires_relative_error",
)


def _settings(config: Experiment4Config, model: str) -> SpectralSettings:
    return config.stable_spectral if model == "stable" else config.vp_spectral


def _profile(config: Experiment4Config, model: str, time_value: float, *, refined: bool = False):
    settings = _settings(config, model)
    if refined:
        settings = replace(
            settings,
            fft_points=settings.fft_points * config.validation.refinement_factor,
        )
    return student_denoiser_profile(
        model,  # type: ignore[arg-type]
        time_value,
        settings,
        alpha=config.alpha,
        beta=config.beta,
    )


def compute_curve(config: Experiment4Config) -> tuple[list[dict[str, Any]], float]:
    """Compute the displayed curve on the declared deterministic grid."""

    start = time.perf_counter()
    times = np.linspace(
        config.time_min,
        config.time_max,
        config.time_points,
        dtype=np.float64,
    )
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for time_value in times:
            profile = _profile(config, model, float(time_value))
            reference = math.exp(config.beta * float(time_value) / 2.0)
            rows.append(
                {
                    "model": model,
                    "time": float(time_value),
                    "ct_truncated": profile.ct,
                    "maximizer_abs_x": profile.maximizer,
                    "boundary_abs_derivative": profile.boundary_abs_derivative,
                    "min_density": profile.min_density,
                    "cutoff_characteristic": profile.cutoff_characteristic,
                    "imaginary_residual": profile.imaginary_residual,
                    "brownian_reference": reference,
                }
            )
    return rows, time.perf_counter() - start


def _source_proxy(
    source: SourceTables,
    model: str,
    time_value: float,
    *,
    radius: float,
) -> tuple[float, float]:
    if not source.available:
        return float("nan"), float("nan")
    main = source.tables[f"{model}_main"]
    hires = source.tables[f"{model}_hires"]
    if not (
        main.time_nodes[0] <= time_value <= main.time_nodes[-1]
        and hires.time_nodes[0] <= time_value <= hires.time_nodes[-1]
    ):
        return float("nan"), float("nan")
    main_ct = score_table_ct_proxy(main, time_value, radius=radius, points=8193)
    hires_ct = score_table_ct_proxy(hires, time_value, radius=radius, points=16385)
    return main_ct, hires_ct


def compute_diagnostics(
    config: Experiment4Config,
    source: SourceTables,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    """Run independent resolution, domain, tail, and cached-table checks."""

    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    gates: list[dict[str, Any]] = []
    for model in MODELS:
        for time_value in config.validation.times:
            main = _profile(config, model, time_value)
            refined = _profile(config, model, time_value, refined=True)
            resolution_error = abs(refined.ct - main.ct) / refined.ct
            domain_values = truncated_values(refined, config.validation.domain_radii)
            domain_spread = (max(domain_values) - min(domain_values)) / refined.ct
            reference = math.exp(config.beta * time_value / 2.0)
            vp_tail_error = (
                abs(refined.boundary_abs_derivative - reference) / reference
                if model == "vp"
                else float("nan")
            )
            boundary_fraction = refined.boundary_abs_derivative / refined.ct
            source_main, source_hires = _source_proxy(
                source,
                model,
                time_value,
                radius=config.validation.maximizer_inner_radius,
            )
            source_hires_error = (
                abs(source_hires - refined.ct) / refined.ct
                if math.isfinite(source_hires)
                else float("nan")
            )
            source_main_hires_error = (
                abs(source_main - source_hires) / source_hires
                if math.isfinite(source_main) and math.isfinite(source_hires)
                else float("nan")
            )
            row = {
                "model": model,
                "time": time_value,
                "ct_main": main.ct,
                "ct_refined": refined.ct,
                "resolution_relative_error": resolution_error,
                "maximizer_abs_x": refined.maximizer,
                "boundary_abs_derivative": refined.boundary_abs_derivative,
                "boundary_fraction": boundary_fraction,
                "brownian_reference": reference,
                "vp_boundary_reference_relative_error": vp_tail_error,
                "domain_ct_json": json.dumps(
                    dict(zip(config.validation.domain_radii, domain_values, strict=True)),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "domain_relative_spread": domain_spread,
                "source_main_ct": source_main,
                "source_hires_ct": source_hires,
                "source_hires_relative_error": source_hires_error,
                "source_main_hires_relative_error": source_main_hires_error,
            }
            rows.append(row)
            checks = {
                "resolution": resolution_error <= config.validation.resolution_rtol,
                "domain": domain_spread <= config.validation.resolution_rtol,
                "interior_maximizer": (
                    refined.maximizer <= config.validation.maximizer_inner_radius
                ),
            }
            if model == "stable":
                if config.tier == "final" and math.isclose(
                    time_value,
                    config.validation.times[-1],
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ):
                    checks["stable_late_time_boundary"] = (
                        boundary_fraction <= config.validation.stable_boundary_fraction
                    )
            else:
                checks["vp_tail_reference"] = (
                    vp_tail_error <= config.validation.vp_tail_reference_rtol
                )
                checks["vp_lower_bound"] = (
                    refined.ct + config.validation.resolution_rtol * reference >= reference
                )
            if math.isfinite(source_hires_error):
                checks["source_hires_overlap"] = (
                    source_hires_error <= config.validation.source_table_rtol
                )
            for name, passed in checks.items():
                gate = {
                    "name": name,
                    "model": model,
                    "time": time_value,
                    "passed": bool(passed),
                }
                gates.append(gate)
                if not passed:
                    failures.append(f"{model} t={time_value:g}: {name}")
    diagnostics = {
        "status": "pass" if not failures else "fail",
        "all_gates_pass": not failures,
        "failures": failures,
        "gates": gates,
        "thresholds": asdict(config.validation),
        "interpretation": (
            "Numerical consistency gates only; they are not tests of the manuscript theorem."
        ),
    }
    if failures:
        raise ArtifactError("Experiment 4 numerical gates failed: " + "; ".join(failures))
    return rows, diagnostics, time.perf_counter() - start


def render_figure(rows: Sequence[dict[str, Any]]):
    """Render one publication-sized panel with no Monte Carlo uncertainty."""

    cache = Path(tempfile.gettempdir()) / "levy-exp4-matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    with plt.rc_context(
        {
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "legend.fontsize": 7,
            "lines.linewidth": 1.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    ):
        figure, axis = plt.subplots(figsize=(4.7, 3.15), constrained_layout=True)
        for model in MODELS:
            selected = sorted(
                (row for row in rows if row["model"] == model),
                key=lambda row: float(row["time"]),
            )
            axis.plot(
                [row["time"] for row in selected],
                [row["ct_truncated"] for row in selected],
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                markevery=max(1, len(selected) // 9),
                markersize=3.2,
                markerfacecolor="white",
                markeredgewidth=0.9,
                label=MODEL_LABELS[model],
                zorder=3,
            )
        stable_rows = sorted(
            (row for row in rows if row["model"] == "stable"),
            key=lambda row: float(row["time"]),
        )
        axis.plot(
            [row["time"] for row in stable_rows],
            [row["brownian_reference"] for row in stable_rows],
            color="#4D4D4D",
            linestyle=(0, (4, 2.5)),
            linewidth=1.25,
            label=r"$1/a_{\rm VP}(t)=e^{t/2}$",
            zorder=2,
        )
        axis.set_yscale("log")
        axis.set_xlabel(r"forward time $t$")
        axis.set_ylabel(r"truncated denoiser modulus $\widehat C_t(40)$")
        axis.set_title(r"Student-$t_4$ target")
        axis.grid(alpha=0.22, which="major")
        axis.grid(alpha=0.08, which="minor", axis="y")
        axis.legend(frameon=False, loc="best")
        return figure


def _figure_bytes(figure: Any, suffix: str) -> bytes:
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
                    "Creator": "levy-experiments-exp4",
                    "Producer": "levy-experiments-exp4",
                    "CreationDate": None,
                    "ModDate": None,
                },
            )
        elif suffix == ".png":
            figure.savefig(
                temporary,
                format="png",
                dpi=260,
                bbox_inches="tight",
                metadata={"Software": "levy-experiments-exp4"},
            )
        else:
            raise ArtifactError(f"unsupported figure format: {suffix}")
        return temporary.read_bytes()
    finally:
        temporary.unlink(missing_ok=True)


def _csv_bytes(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        values: dict[str, str] = {}
        for field in fields:
            value = row[field]
            if isinstance(value, (float, np.floating)):
                values[field] = "nan" if math.isnan(float(value)) else format(float(value), ".17g")
            elif isinstance(value, (bool, np.bool_)):
                values[field] = "true" if value else "false"
            else:
                values[field] = str(value)
        writer.writerow(values)
    return buffer.getvalue().encode("utf-8")


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Encode NumPy arrays in an NPZ with fixed ordering and timestamps."""

    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for name in sorted(arrays):
            array_buffer = io.BytesIO()
            np.lib.format.write_array(
                array_buffer,
                np.asarray(arrays[name]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, array_buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    return output.getvalue()


def _caption(config: Experiment4Config) -> str:
    return (
        "**Forward denoiser modulus on a Student-$t_4$ target.** Solid curves show "
        "$\\widehat C_t(40)=\\max_{|x|\\leq40}|m_t'(x)|$ for the isotropic "
        "$\\alpha$-stable SP-SDE ($\\alpha=1.5$) and the Brownian VP-SDE, computed "
        "deterministically by Fourier differentiation of their exact forward "
        "marginals. The dashed curve is $1/a_{\\rm VP}(t)=e^{t/2}$: for the "
        "heavy-tailed target it is the Brownian spatial limiting slope and hence a "
        "lower bound on $C_t^{\\rm VP}$, not an equality imposed on the numerical "
        "curve. The vertical axis is logarithmic. Doubling the Fourier cutoff and "
        "expanding nested spatial domains leave the displayed suprema within the "
        f"predeclared numerical tolerances (relative FFT tolerance "
        f"{config.validation.resolution_rtol:g}). The calculation uses neither "
        "Monte Carlo samples nor a learned score; $C_t$ is a forward quantity and "
        "is independent of the backward parameter $\\eta$.\n"
    )


def _code_hash() -> str:
    project = Path(__file__).resolve().parents[2]
    candidates = list((project / "experiments" / "exp4").glob("*.py"))
    candidates.extend(
        (
            project / "experiments" / "exp2" / "theory.py",
            project / "experiments" / "exp2" / "score_tables.py",
            project / "scripts" / "run_exp4_ct.py",
        )
    )
    digest = hashlib.sha256()
    for path in sorted(candidates, key=lambda item: item.relative_to(project).as_posix()):
        if not path.is_file():
            raise ArtifactError(f"Experiment 4 code-hash input is missing: {path}")
        digest.update(path.relative_to(project).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _dependencies() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "matplotlib", "levy-experiments"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _arrays(rows: Sequence[dict[str, Any]], diagnostics: Sequence[dict[str, Any]]):
    ordered = sorted(rows, key=lambda row: (str(row["model"]), float(row["time"])))
    diagnostic_ordered = sorted(
        diagnostics,
        key=lambda row: (str(row["model"]), float(row["time"])),
    )
    return {
        "model": np.asarray([row["model"] for row in ordered], dtype="U6"),
        "time": np.asarray([row["time"] for row in ordered], dtype=np.float64),
        "ct_truncated": np.asarray(
            [row["ct_truncated"] for row in ordered], dtype=np.float64
        ),
        "maximizer_abs_x": np.asarray(
            [row["maximizer_abs_x"] for row in ordered], dtype=np.float64
        ),
        "brownian_reference": np.asarray(
            [row["brownian_reference"] for row in ordered], dtype=np.float64
        ),
        "diagnostic_model": np.asarray(
            [row["model"] for row in diagnostic_ordered], dtype="U6"
        ),
        "diagnostic_time": np.asarray(
            [row["time"] for row in diagnostic_ordered], dtype=np.float64
        ),
        "diagnostic_ct_refined": np.asarray(
            [row["ct_refined"] for row in diagnostic_ordered], dtype=np.float64
        ),
        "diagnostic_resolution_relative_error": np.asarray(
            [row["resolution_relative_error"] for row in diagnostic_ordered],
            dtype=np.float64,
        ),
    }


def _report(
    config: Experiment4Config,
    source: SourceTables,
    rows: Sequence[dict[str, Any]],
    diagnostics: Sequence[dict[str, Any]],
    timings: dict[str, float],
    code_hash: str,
) -> str:
    start_values = {
        model: next(row for row in rows if row["model"] == model)["ct_truncated"]
        for model in MODELS
    }
    end_values = {
        model: [row for row in rows if row["model"] == model][-1]["ct_truncated"]
        for model in MODELS
    }
    maximum_resolution = max(float(row["resolution_relative_error"]) for row in diagnostics)
    maximum_source_error = max(
        (
            float(row["source_hires_relative_error"])
            for row in diagnostics
            if math.isfinite(float(row["source_hires_relative_error"]))
        ),
        default=float("nan"),
    )
    source_error_text = (
        f"{maximum_source_error:.6g}"
        if math.isfinite(maximum_source_error)
        else "not available"
    )
    source_text = str(source.run_dir) if source.available else "not requested (smoke mode)"
    boundary_text = (
        "The final-grid stable derivative is negligible at x=40, while the VP "
        "boundary derivative matches the analytic tail slope within the declared tolerance."
        if config.tier == "final"
        else "The smoke grid validates the location of both suprema; its reduced stable "
        "Fourier domain is not used to certify the far-tail derivative at x=40."
    )
    return f"""# Experiment 4 validation report

Status: complete; all numerical consistency gates passed.

## Scientific quantity

The target is the centred Student-t(4) distribution. In one dimension,
`C_t = sup_x |m_t'(x)|`. The reported value is the explicitly truncated proxy
`C_hat_t(40) = max_{{|x|<=40}} |m_t'(x)|`. Symmetry reduces the numerical search
to `x in [0,40]`. Both forward marginal characteristic functions are exact.

The stable curve uses `a_alpha(t)=exp(-t/alpha)` with `alpha=1.5`; the VP curve
uses `a_VP(t)=exp(-t/2)`. The dashed comparison is therefore
`1/a_VP(t)=exp(t/2)`. It is the Brownian tail slope and a lower bound on the
population VP modulus, not a claimed identity with the finite-domain curve.
The backward parameter eta does not enter this forward diagnostic.

## Numerical method and checks

- Configuration hash: `{config.resolved_hash}`
- Experiment 4 code hash: `{code_hash}`
- Source Experiment 2 run: `{source_text}`
- Main time grid: {config.time_points} points on [{config.time_min:g},{config.time_max:g}]
- Stable FFT: L={config.stable_spectral.fft_half_width:g}, N={config.stable_spectral.fft_points}
- VP FFT: L={config.vp_spectral.fft_half_width:g}, N={config.vp_spectral.fft_points}
- Validation times: {list(config.validation.times)}
- Nested radii: {list(config.validation.domain_radii)}
- Maximum observed main/refined relative difference: {maximum_resolution:.6g}
- Maximum cached high-resolution-table overlap difference: {source_error_text}
- Curve time: {timings['curve_seconds']:.3f} s
- Validation time: {timings['validation_seconds']:.3f} s
- Total computation time: {timings['total_seconds']:.3f} s

The score itself is never finite-differenced. Fourier multipliers are applied
to the density and score numerator before forming the quotient derivative.
The refined checks double N at fixed L. The nested-domain check verifies that
the maximizer is already inside radius {config.validation.maximizer_inner_radius:g}.
{boundary_text}

## Descriptive result

- Stable: C_hat changes from {float(start_values['stable']):.6g} to
  {float(end_values['stable']):.6g} over the displayed interval.
- VP: C_hat changes from {float(start_values['vp']):.6g} to
  {float(end_values['vp']):.6g} over the displayed interval.

These deterministic curves illustrate the forward-denoiser dichotomy for this
target and parameter choice. They do not prove the asymptotic theorem, compare
backward path laws, or establish a universal mixing advantage.
"""


def _validate_complete_run(run_dir: Path, config: Experiment4Config) -> bool:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"corrupt Experiment 4 manifest: {manifest_path}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "complete"
        or manifest.get("config_hash") != config.resolved_hash
        or manifest.get("code_hash") != _code_hash()
        or not isinstance(manifest.get("artifacts"), dict)
    ):
        raise ArtifactError("existing Experiment 4 run is incompatible")
    for relative, record in manifest["artifacts"].items():
        path = run_dir / relative
        if (
            not isinstance(record, dict)
            or not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
            or hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256")
        ):
            raise ArtifactError(f"existing Experiment 4 artifact is corrupt: {path}")
    return True


def run_experiment(
    config: Experiment4Config,
    *,
    output_root: str | Path,
    resume: bool,
    dry_run: bool,
    device: str,
    precision: str,
    seed: int | None,
) -> Path:
    """Execute the complete deterministic analysis and commit auditable artifacts."""

    if device not in {"cpu", "auto"}:
        raise ConfigurationError(
            "Experiment 4 is an FFT post-process implemented in NumPy; use --device cpu or auto"
        )
    if precision != "float64" or config.precision != "float64":
        raise ConfigurationError("Experiment 4 is validated only in float64")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ConfigurationError("--seed must be a non-negative integer when recorded")
    source = validate_source(config)
    run_dir = Path(output_root).resolve() / f"exp4-{config.resolved_hash[:12]}"
    if dry_run:
        print(
            json.dumps(
                {
                    "status": "dry-run",
                    "config_hash": config.resolved_hash,
                    "run_dir": str(run_dir),
                    "source": source.provenance,
                    "estimated_main_ffts": 8 * config.time_points,
                    "estimated_validation_ffts": 16 * len(config.validation.times),
                    "device": "cpu",
                    "precision": "float64",
                    "randomness": "none",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return run_dir
    if run_dir.exists():
        if not resume:
            raise ArtifactError(f"refusing to overwrite existing Experiment 4 run: {run_dir}")
        if _validate_complete_run(run_dir, config):
            print(f"verified complete Experiment 4 run: {run_dir}")
            return run_dir
        if any(run_dir.iterdir()):
            raise ArtifactError(
                "incomplete Experiment 4 directory exists; preserve it and choose a new output root"
            )

    total_start = time.perf_counter()
    rows, curve_seconds = compute_curve(config)
    diagnostic_rows, diagnostic_summary, validation_seconds = compute_diagnostics(config, source)
    total_seconds = time.perf_counter() - total_start
    timings = {
        "curve_seconds": curve_seconds,
        "validation_seconds": validation_seconds,
        "total_seconds": total_seconds,
    }
    code_hash = _code_hash()
    figure = render_figure(rows)
    try:
        pdf = _figure_bytes(figure, ".pdf")
        png = _figure_bytes(figure, ".png")
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
    requested = Path(config.source_path).read_bytes()
    resolved = config.canonical_dict(include_paths=True) | {
        "resolved_hash": config.resolved_hash,
        "requested_config_source_hash": config.source_hash,
        "source_validation": source.provenance,
    }
    provenance = {
        "experiment": 4,
        "config_hash": config.resolved_hash,
        "code_hash": code_hash,
        "requested_config_sha256": config.source_hash,
        "source": source.provenance,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu": {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "logical_count": os.cpu_count(),
        },
        "dependencies": _dependencies(),
        "backend": "numpy-cpu",
        "precision": "float64",
        "seed": seed,
        "randomness": "none",
        "command": [sys.executable, *sys.argv],
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    report = _report(config, source, rows, diagnostic_rows, timings, code_hash)
    artifacts = {
        "requested_config.toml": requested,
        "resolved_config.json": _json_bytes(resolved),
        "provenance.json": _json_bytes(provenance),
        "aggregates/ct_curve.csv": _csv_bytes(rows, CURVE_FIELDS),
        "aggregates/numerical_diagnostics.csv": _csv_bytes(
            diagnostic_rows, DIAGNOSTIC_FIELDS
        ),
        "aggregates/diagnostics.json": _json_bytes(diagnostic_summary),
        "aggregates/analysis_arrays.npz": _deterministic_npz_bytes(
            _arrays(rows, diagnostic_rows)
        ),
        "figures/experiment4_ct.pdf": pdf,
        "figures/experiment4_ct.png": png,
        "figures/caption.md": _caption(config).encode("utf-8"),
        "logs/run.log": (
            f"status=complete\ncurve_seconds={curve_seconds:.9f}\n"
            f"validation_seconds={validation_seconds:.9f}\n"
            f"total_seconds={total_seconds:.9f}\n"
        ).encode(),
        "validation_report.md": report.encode("utf-8"),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    for relative, payload in artifacts.items():
        write_or_verify_bytes(run_dir / relative, payload, resume=resume)
    manifest_artifacts = {
        relative: {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for relative, payload in artifacts.items()
    }
    manifest = {
        "experiment": 4,
        "status": "complete",
        "all_numerical_gates_pass": diagnostic_summary["all_gates_pass"],
        "config_hash": config.resolved_hash,
        "code_hash": code_hash,
        "artifacts": manifest_artifacts,
    }
    write_or_verify_bytes(run_dir / "manifest.json", _json_bytes(manifest), resume=resume)
    print(f"Experiment 4 complete: {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default="auto")
    parser.add_argument(
        "--precision",
        choices=("float32", "float64", "mixed"),
        default="float64",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config, source_run_override=args.source_run)
        run_experiment(
            config,
            output_root=args.output_dir,
            resume=args.resume,
            dry_run=args.dry_run,
            device=args.device,
            precision=args.precision,
            seed=args.seed,
        )
    except (ExperimentError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "compute_curve",
    "compute_diagnostics",
    "main",
    "render_figure",
    "run_experiment",
]
