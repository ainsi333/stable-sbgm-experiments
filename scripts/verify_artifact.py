"""Verify archived scientific tables, figure receipts, and anonymity invariants."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

import matplotlib.colors

from levy_experiments.exp2_initialization import (
    load_reanalysis_settings,
    render_initialization_figure,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DATA_HASHES = {
    "artifact_data/exp1/tail_summary.csv": (
        "0569fd40f6a53271459887b7e2c44fefbb379e264903e2ab125538f929acaf70"
    ),
    "artifact_data/exp2/initialization_wp_per_seed.csv": (
        "0f9913e27d79d539398efc9300958755b1c4e7baf59f87fb21e25e24392fc60a"
    ),
    "artifact_data/exp2/initialization_wp_summary.csv": (
        "711caf2907c3e3914173dba752e22f79c4dbc8b05bd40b243161d21d8a873cb6"
    ),
    "artifact_data/exp3/metrics_per_seed.csv": (
        "2510a7c6226d29b42321af881ad26062b38e4e3a343a18824b10c4f9142dccc9"
    ),
    "artifact_data/exp4/ct_curve.csv": (
        "f055865a7e395bb1ff4c6fa3b43c70a5eea830743ac11e508f1f3b7027e80236"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checksums(root: Path) -> int:
    """Validate every packaged file, not only the five main numerical tables."""
    root = root.resolve()
    entries: set[str] = set()
    for line in (root / "SHA256SUMS").read_text("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise RuntimeError("invalid SHA256SUMS entry")
        expected, relative = match.groups()
        parts = PurePosixPath(relative)
        if (
            parts.is_absolute() or ".." in parts.parts or "\\" in relative
            or ":" in relative or relative in entries or relative == "SHA256SUMS"
        ):
            raise RuntimeError(f"unsafe or duplicate checksum path: {relative}")
        path = root / relative
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise RuntimeError(f"missing or unsafe packaged file: {relative}")
        if _sha256(path) != expected:
            raise RuntimeError(f"packaged file hash mismatch: {relative}")
        entries.add(relative)
    if not entries:
        raise RuntimeError("empty SHA256SUMS")
    # Generated runtime outputs are intentionally outside the frozen inventory.
    runtime = {
        ".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__",
        ".matplotlib-cache", "outputs", "reproduced_figures", "slurm_logs",
        "dist", "build",
    }
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(p in runtime or p.startswith(".pytest_tmp_") for p in relative.parts):
            continue
        if path.is_file() and relative.as_posix() not in entries | {"SHA256SUMS"}:
            raise RuntimeError(f"unlisted packaged file: {relative}")
    return len(entries)


def _coerce(value: str):
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return float(value)
    except ValueError:
        return value


def _verify_exp2_marker_convention() -> None:
    summary = ROOT / "artifact_data" / "exp2" / "initialization_wp_summary.csv"
    with summary.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            {name: _coerce(value) for name, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    settings = load_reanalysis_settings(
        ROOT / "artifact_data" / "exp2" / "requested_analysis_config.toml",
        alpha=1.5,
    )
    figure = render_initialization_figure(rows, settings)
    try:
        labels = {text.get_text() for legend in figure.legends for text in legend.get_texts()}
        if "MC-resolution limited" in labels:
            raise RuntimeError("abandoned Experiment 2 legend entry was reintroduced")
        point_lines = [
            line
            for axis in figure.axes
            for line in axis.lines
            if line.get_marker() not in {"None", "none", "", None}
            and line.get_linestyle() in {"None", "none", "", None}
        ]
        if not point_lines or any(
            matplotlib.colors.to_rgba(line.get_markerfacecolor())
            == matplotlib.colors.to_rgba("white")
            for line in point_lines
        ):
            raise RuntimeError("Experiment 2 contains an unfilled method marker")
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def _verify_figure_manifest() -> None:
    manifest = json.loads((ROOT / "figures" / "figure_manifest.json").read_text("utf-8"))
    for experiment, group in manifest["figures"].items():
        for name, expected in group.items():
            if name.startswith("source:"):
                path = ROOT / "artifact_data" / experiment.replace("experiment", "exp") / name[7:]
            else:
                path = ROOT / "figures" / name
            if _sha256(path) != expected:
                raise RuntimeError(f"figure hash mismatch: {path}")


def _verify_anonymity() -> None:
    forbidden = (b"c:" + b"\\users\\",)
    suffixes = {".csv", ".json", ".md", ".py", ".toml", ".txt", ".def"}
    excluded = {".git", ".matplotlib-cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        relative = path.relative_to(ROOT)
        if any(part in excluded or part.startswith(".pytest_tmp_") for part in relative.parts):
            continue
        lowered = path.read_bytes().lower()
        if any(token in lowered for token in forbidden):
            raise RuntimeError(f"personal path or identifier remains in {relative}")


def main() -> int:
    count = verify_checksums(ROOT)
    for relative, expected in EXPECTED_DATA_HASHES.items():
        path = ROOT / relative
        if _sha256(path) != expected:
            raise RuntimeError(f"scientific table hash mismatch: {relative}")
    _verify_exp2_marker_convention()
    _verify_figure_manifest()
    _verify_anonymity()
    print(f"artifact verification: PASS ({count} packaged files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
