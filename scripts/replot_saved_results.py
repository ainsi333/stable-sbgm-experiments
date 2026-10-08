"""Regenerate manuscript figures from the archived plot-ready tables only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.exp1.plotting import plot_main, plot_quantile_ratio
from experiments.exp4.run import _figure_bytes as exp4_figure_bytes
from experiments.exp4.run import render_figure as render_exp4_figure
from levy_experiments.exp2_initialization import (
    _figure_bytes as exp2_figure_bytes,
)
from levy_experiments.exp2_initialization import (
    load_reanalysis_settings,
    render_initialization_figure,
)


def _coerce(value: str) -> Any:
    stripped = value.strip()
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    if stripped == "":
        return ""
    try:
        number = float(stripped)
    except ValueError:
        return value
    return number if math.isfinite(number) or stripped.lower() == "nan" else value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing archived table: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {name: _coerce(value) for name, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _write(path: Path, payload: bytes, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force explicitly")
    path.write_bytes(payload)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_exp1(data_root: Path, output: Path) -> dict[str, str]:
    source = data_root / "exp1" / "tail_summary.csv"
    rows = _read_csv(source)
    paths = (
        *plot_main(rows, output, tier="final", primary_steps=80),
        *plot_quantile_ratio(rows, output, tier="final", primary_steps=80),
    )
    return {path.name: _hash(path) for path in paths} | {"source:tail_summary.csv": _hash(source)}


def _render_exp2(data_root: Path, output: Path, *, force: bool) -> dict[str, str]:
    source = data_root / "exp2" / "initialization_wp_summary.csv"
    settings_path = data_root / "exp2" / "requested_analysis_config.toml"
    rows = _read_csv(source)
    settings = load_reanalysis_settings(settings_path, alpha=1.5)
    figure = render_initialization_figure(rows, settings)
    try:
        pdf = output / "experiment2_initialization_comparison_cropped.pdf"
        png = output / "experiment2_initialization_comparison_cropped.png"
        _write(pdf, exp2_figure_bytes(figure, ".pdf"), force=force)
        _write(png, exp2_figure_bytes(figure, ".png"), force=force)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
    return {
        pdf.name: _hash(pdf),
        png.name: _hash(png),
        "source:initialization_wp_summary.csv": _hash(source),
        "source:requested_analysis_config.toml": _hash(settings_path),
    }


def _render_exp4(data_root: Path, output: Path, *, force: bool) -> dict[str, str]:
    source = data_root / "exp4" / "ct_curve.csv"
    rows = _read_csv(source)
    figure = render_exp4_figure(rows)
    try:
        pdf = output / "experiment4_ct.pdf"
        png = output / "experiment4_ct.png"
        _write(pdf, exp4_figure_bytes(figure, ".pdf"), force=force)
        _write(png, exp4_figure_bytes(figure, ".png"), force=force)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
    return {pdf.name: _hash(pdf), png.name: _hash(png), "source:ct_curve.csv": _hash(source)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("artifact_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reproduced_figures"))
    parser.add_argument("--experiment", choices=("all", "1", "2", "4"), default="all")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = ("1", "2", "4") if args.experiment == "all" else (args.experiment,)
    expected_names = {
        "1": (
            "experiment1_main.pdf",
            "experiment1_main.png",
            "experiment1_quantile_ratio.pdf",
            "experiment1_quantile_ratio.png",
        ),
        "2": (
            "experiment2_initialization_comparison_cropped.pdf",
            "experiment2_initialization_comparison_cropped.png",
        ),
        "4": ("experiment4_ct.pdf", "experiment4_ct.png"),
    }
    existing = [
        args.output_dir / name
        for experiment in selected
        for name in expected_names[experiment]
        if (args.output_dir / name).exists()
    ]
    manifest_path = args.output_dir / "figure_manifest.json"
    if manifest_path.exists():
        existing.append(manifest_path)
    if existing and not args.force:
        raise FileExistsError(
            "refusing to overwrite existing figures; pass --force explicitly: "
            + ", ".join(str(path) for path in existing)
        )
    manifest: dict[str, Any] = {"status": "complete", "figures": {}}
    if "1" in selected:
        manifest["figures"]["experiment1"] = _render_exp1(args.data_root, args.output_dir)
    if "2" in selected:
        manifest["figures"]["experiment2"] = _render_exp2(
            args.data_root, args.output_dir, force=args.force
        )
    if "4" in selected:
        manifest["figures"]["experiment4"] = _render_exp4(
            args.data_root, args.output_dir, force=args.force
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
