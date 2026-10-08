"""Quantile notation regression tests using synthetic plotting fixtures only."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from experiments.exp1 import plotting


@pytest.mark.parametrize(
    ("plot", "axis_index", "output_stem"),
    [
        (plotting.plot_main, 1, "experiment1_main"),
        (plotting.plot_quantile_ratio, 0, "experiment1_quantile_ratio"),
    ],
    ids=["main-coverage-panel", "standalone-quantile-ratio"],
)
def test_quantile_figures_use_kappa_without_changing_data(
    plot: Callable[..., tuple[Path, Path]],
    axis_index: int,
    output_stem: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fractions = np.asarray([0.004, 0.008, 0.02])
    series = (("stable", 1.2), ("stable", 1.5), ("stable", 3.0), ("vp", 1.5))
    summaries: list[dict[str, Any]] = []
    for series_index, (model, nu) in enumerate(series):
        for point_index, fraction in enumerate(fractions):
            median = 1.0 + series_index + point_index / 4.0
            summaries.append(
                {
                    "task_kind": "dynamic",
                    "model": model,
                    "target_nu": nu,
                    "metric": "mass_ratio",
                    "sample_kind": "full",
                    "steps": 8,
                    "purpose": "primary",
                    "fraction": float(fraction),
                    "median": median,
                    "q16": median - 0.125,
                    "q84": median + 0.25,
                }
            )

    captured: list[Figure] = []

    def capture_save(fig: Figure, path: Path) -> tuple[Path, Path]:
        captured.append(fig)
        fig.canvas.draw()
        return path.with_suffix(".pdf"), path.with_suffix(".png")

    monkeypatch.setattr(plotting, "_save", capture_save)
    try:
        with plt.rc_context():
            outputs = plot(list(reversed(summaries)), tmp_path, tier="smoke", primary_steps=8)
        assert outputs == (
            tmp_path / f"{output_stem}.pdf",
            tmp_path / f"{output_stem}.png",
        )
        assert not any(path.exists() for path in outputs)
        assert len(captured) == 1
        axis = captured[0].axes[axis_index]
        assert axis.get_xlabel() == r"survival mass $1-\kappa$"
        assert axis.get_ylabel() == r"$\widehat q_\kappa(|Y|)/q_\kappa(|T_\nu|)$"
        assert r"\beta" not in axis.get_xlabel() + axis.get_ylabel()
        assert axis.get_xscale() == "log"
        assert axis.get_yscale() == "log"
        assert axis.xaxis_inverted()
        assert len(axis.lines) == len(series)
        assert len(axis.collections) == len(series)

        for series_index, (line, band) in enumerate(zip(axis.lines, axis.collections, strict=True)):
            medians = 1.0 + series_index + np.arange(fractions.size) / 4.0
            np.testing.assert_array_equal(line.get_xdata(), fractions)
            np.testing.assert_array_equal(line.get_ydata(), medians)
            paths = band.get_paths()
            assert len(paths) == 1
            vertices = paths[0].vertices
            np.testing.assert_array_equal(np.unique(vertices[:, 0]), fractions)
            for fraction, median in zip(fractions, medians, strict=True):
                band_values = vertices[vertices[:, 0] == fraction, 1]
                np.testing.assert_array_equal(
                    np.unique(band_values), [median - 0.125, median + 0.25]
                )
    finally:
        for fig in captured:
            plt.close(fig)
