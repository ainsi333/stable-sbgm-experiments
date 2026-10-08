"""Recompute displayed summaries from archived seed statistics, without simulation."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from replot_saved_results import _read_csv

from levy_experiments.exp2_initialization import (
    load_reanalysis_settings,
    summarize_initialization_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    data = ROOT / "artifact_data"
    settings = load_reanalysis_settings(
        data / "exp2/requested_analysis_config.toml", alpha=1.5,
    )
    summaries = summarize_initialization_rows(
        _read_csv(data / "exp2/initialization_wp_per_seed.csv"), settings,
    )
    saved = _read_csv(data / "exp2/initialization_wp_summary.csv")
    assert len(saved) == len(summaries) == 24
    for actual, expected in zip(summaries, saved, strict=True):
        for field, value in actual.items():
            if isinstance(value, str):
                assert value == expected[field], (field, value, expected[field])
            else:
                np.testing.assert_allclose(value, expected[field], rtol=2e-14, atol=1e-15)

    # Independent calculation of every tail point/band displayed in the paper.
    fields = ("model", "target_nu", "metric", "fraction", "beta", "sample_size")
    grouped = defaultdict(list)
    with (data / "exp1/tail_metrics_per_seed.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (row["purpose"] == "primary" and row["sample_kind"] == "full"
                    and row["steps"] == "80" and row["task_kind"] == "dynamic"
                    and row["valid"].lower() == "true"):
                grouped[tuple(row[key] for key in fields)].append(
                    (int(row["seed"]), float(row["value"]))
                )
    count = 0
    with (data / "exp1/tail_summary.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not (row["purpose"] == "primary" and row["sample_kind"] == "full"
                    and row["steps"] == "80" and row["task_kind"] == "dynamic"
                    and row["metric"] in {"hill_alpha", "tail_constant", "mass_ratio"}):
                continue
            if row["model"] == "vp" and row["metric"] != "mass_ratio":
                continue
            members = grouped[tuple(row[key] for key in fields)]
            assert len(members) == len({seed for seed, _ in members}) == 12
            values = [value for _, value in members]
            for field, probability in (("q16", .16), ("median", .5), ("q84", .84)):
                np.testing.assert_allclose(
                    np.quantile(values, probability), float(row[field]), rtol=2e-14, atol=1e-15,
                )
            count += 1
    assert count > 0
    print(f"saved statistics: PASS (24 initialization rows including bootstrap; {count} tail rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
